"""TorchRL env factories for NS-Gym training.

Thin wrappers that combine NS envs, PLR, and TorchRL transforms into
ready-to-use TransformedEnv instances for collection and evaluation.
"""

from __future__ import annotations

import logging
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig

from torchrl.data import OneHot, UnboundedContinuous, Unbounded
from torchrl.envs.gym_like import default_info_dict_reader
from torchrl.envs import (
    CatFrames,
    Compose,
    DoubleToFloat,
    ObservationNorm,
    ParallelEnv,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.transforms import Transform
from torchrl.envs.libs.gym import GymWrapper

from AAMAS_Comp.curriculum import PLREnv, FixedNSEnv, RandomNSEnv, sample_held_out_configs
from AAMAS_Comp.envs.wrappers import NoInfoWrapper, RunningMeanStd, SeededResetWrapper, WorkerIdentityWrapper
from AAMAS_Comp.curriculum.feedback import PLR_KEYS


def worker_seed(seed, worker_idx=0, stream=0):
    return int(np.random.SeedSequence([int(seed), int(worker_idx), int(stream)]).generate_state(1)[0])


def worker_gym_env(env, device, worker_idx):
    base = GymWrapper(WorkerIdentityWrapper(env, worker_idx), device=device)
    base.set_info_dict_reader(default_info_dict_reader(
        keys=["env_worker"], spec={"env_worker": Unbounded(shape=(), dtype=torch.int64)}))
    return base

log = logging.getLogger(__name__)


class DiscreteObsToFloat(Transform):
    """Cast a one-hot discrete observation from int64 to float32.

    TorchRL's GymWrapper converts gym.spaces.Discrete observations to
    one-hot integer tensors of shape (n,).  This transform casts them to
    float32 so they can be fed directly into an MLP.
    """

    def __init__(self):
        super().__init__(in_keys=["observation"], out_keys=["observation"])

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.float()

    def _reset(self, tensordict, tensordict_reset):
        # Base Transform._reset does NOT call _apply_transform; call it explicitly.
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        spec = observation_spec["observation"]
        observation_spec["observation"] = UnboundedContinuous(
            shape=spec.shape,
            dtype=torch.float32,
        )
        return observation_spec


def initialize_obs_norm(
    cfg: DictConfig,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> RunningMeanStd | None:
    """Bootstrap observation normalization statistics from an initial random rollout.

    Returns:
        RunningMeanStd with bootstrap stats, or None if normalize_obs is disabled.
    """
    if not cfg.env.normalize_obs:
        return None

    log.info("Bootstrapping observation normalization stats...")
    base_env = GymWrapper(NoInfoWrapper(SeededResetWrapper(gym.make(cfg.env.id),
                           worker_seed(cfg.seed, stream=10))), device=device)
    base_env.set_seed(worker_seed(cfg.seed, stream=10))
    tmp = TransformedEnv(base_env, Compose(DoubleToFloat(), StepCounter()))
    tmp.reset()
    td = tmp.rollout(max_steps=cfg.env.normalize_obs_init_steps, break_when_any_done=False)
    all_obs = td["observation"]

    obs_dim = tmp.observation_spec["observation"].shape[-1]
    obs_rms = RunningMeanStd(shape=(obs_dim,), device=device)
    obs_rms.count = 1e-4  # SB3-style epsilon init so initial std = 1
    obs_rms.update(all_obs)
    tmp.close()

    log.info(
        "Observation normalization bootstrapped (mean=%.3f, std=%.3f)",
        obs_rms.mean.mean().item(),
        obs_rms.std.mean().item(),
    )
    return obs_rms


def _make_env_transforms(
    base_env,
    obs_rms: RunningMeanStd | None,
    frame_stack: int = 1,
    max_steps: int | None = None,
) -> list:
    """Return the TorchRL transform list appropriate for this env."""
    transforms = []
    obs_spec = base_env.observation_spec["observation"]
    if isinstance(obs_spec, OneHot):
        transforms.append(DiscreteObsToFloat())
    elif obs_rms is not None:
        obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
        obs_norm.loc = torch.nn.Parameter(obs_rms.mean.float().clone(), requires_grad=False)
        obs_norm.scale = torch.nn.Parameter(obs_rms.std.float().clone(), requires_grad=False)
        transforms.append(obs_norm)
    transforms.append(DoubleToFloat())
    transforms.append(StepCounter(max_steps=max_steps))
    if frame_stack > 1:
        transforms.append(CatFrames(N=frame_stack, dim=-1, in_keys=["observation"]))
    return transforms


def make_single_env(
    cfg: DictConfig,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    dtype: torch.dtype | None = None,
    worker_idx: int = 0,
) -> TransformedEnv:
    """Create a single TorchRL TransformedEnv instance."""
    frame_stack = cfg.env.get("frame_stack", 1)
    base_env = worker_gym_env(NoInfoWrapper(SeededResetWrapper(gym.make(
                           cfg.env.id, max_episode_steps=cfg.collector.max_frames_per_traj),
                           worker_seed(cfg.seed, worker_idx, 1))), device, worker_idx)
    return TransformedEnv(base_env, Compose(*_make_env_transforms(
        base_env, obs_rms, frame_stack, cfg.collector.max_frames_per_traj)))


def make_parallel_env(
    cfg: DictConfig,
    device: torch.device,
    num_envs: int = 1,
    dtype: torch.dtype | None = None,
    obs_rms: RunningMeanStd | None = None,
) -> TransformedEnv | ParallelEnv:
    """Create vectorized parallel environments."""
    if num_envs == 1:
        return make_single_env(cfg, device, obs_rms=obs_rms, dtype=dtype)
    return ParallelEnv(
        num_workers=num_envs,
        create_env_fn=partial(make_single_env, cfg, device, obs_rms, dtype),
        serial_for_single=True,
    )


def _get_context_cfg(cfg: DictConfig) -> tuple[list[str], dict]:
    """Extract context_features and context_defaults from env config."""
    context_keys = list(cfg.env.get("context_features", []) or [])
    raw_defaults = cfg.env.get("context_defaults", {}) or {}
    context_defaults = {k: list(v) for k, v in raw_defaults.items()}
    return context_keys, context_defaults


def make_ns_plr_env(
    cfg: DictConfig,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    dtype: torch.dtype | None = None,
    stats_queue=None,
    worker_idx: int = 0,
    score_array=None,
    feedback_queue=None,
) -> TransformedEnv:
    """Create a PLREnv-backed TorchRL TransformedEnv for NS training.

    Each call produces an independent env with its own PLRBuffer — safe to
    run in separate ParallelEnv subprocesses.
    """
    plr_cfg = cfg.env.plr
    frame_stack = cfg.env.get("frame_stack", 1)
    context_keys, context_defaults = _get_context_cfg(cfg)
    sampler_override = None
    if cfg.get('experiment', {}).get('method') in ('static_plr', 'static_plr_mutations'):
        from AAMAS_Comp.envs.static_curriculum import StaticDynamicsSampler
        definition = static_training_definition(cfg)
        sampler_override = StaticDynamicsSampler(plr_cfg.sampler_key, definition['dr_samples'],
                                                worker_seed(cfg.seed, worker_idx, 2),
                                                mutation_protocol=plr_cfg.get('static_mutation_protocol',
                                                    'relative-reflect-symmetric-v2'))
    plr_env = PLREnv(
        sampler_key=plr_cfg.sampler_key,
        plr_capacity=plr_cfg.capacity,
        replay_prob=plr_cfg.replay_prob,
        mutation_prob=plr_cfg.get("mutation_prob", 0.0),
        mutation_sigma=plr_cfg.get("mutation_sigma", 0.2),
        audit_dir=plr_cfg.get('audit_dir'),
        score_temp=plr_cfg.score_temp,
        staleness_coef=plr_cfg.staleness_coef,
        seed=worker_seed(cfg.seed, worker_idx, 2),
        stats_queue=stats_queue,
        worker_idx=worker_idx,
        score_array=score_array,
        feedback_queue=feedback_queue,
        context_keys=context_keys,
        context_defaults=context_defaults,
        protect_unscored=plr_cfg.get("protect_unscored", True),
        uniform_replay=plr_cfg.get("uniform_replay", False),
        score_ema_alpha=plr_cfg.get("score_ema_alpha", 1.0),
        replay_requires_score=plr_cfg.get("replay_requires_score", True),
        selection_staleness=plr_cfg.get("selection_staleness", True),
        eviction_strategy=plr_cfg.get("eviction_strategy", None),
        frozenlake_initial_mode=cfg.env.get("initial_mode", None),
        sampler_override=sampler_override,
    )
    base_env = GymWrapper(NoInfoWrapper(SeededResetWrapper(plr_env,
                           worker_seed(cfg.seed, worker_idx, 1)), keep_keys=PLR_KEYS), device=device)
    base_env.set_info_dict_reader(default_info_dict_reader(
        keys=list(PLR_KEYS), spec={k: Unbounded(shape=(), dtype=torch.int64) for k in PLR_KEYS}))
    return TransformedEnv(base_env, Compose(*_make_env_transforms(
        base_env, obs_rms, frame_stack, cfg.collector.max_frames_per_traj)))


def make_ns_random_env(
    cfg: DictConfig,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    dtype: torch.dtype | None = None,
    worker_idx: int = 0,
) -> TransformedEnv:
    """Create a randomly-sampled NS env (no PLR) for NS baseline training.

    Samples a fresh NS config on every reset() using uniform random sampling —
    no PLR scoring or replay.  Eval uses the same held-out NS configs as PLR
    (make_ns_eval_shards) for a fair comparison.
    """
    plr_cfg = cfg.env.plr
    frame_stack = cfg.env.get("frame_stack", 1)
    context_keys, context_defaults = _get_context_cfg(cfg)
    random_env = RandomNSEnv(
        sampler_key=plr_cfg.sampler_key,
        seed=worker_seed(cfg.seed, worker_idx, 2),
        context_keys=context_keys,
        context_defaults=context_defaults,
        frozenlake_initial_mode=cfg.env.get("initial_mode", None),
    )
    base_env = worker_gym_env(NoInfoWrapper(SeededResetWrapper(random_env,
                           worker_seed(cfg.seed, worker_idx, 1))), device, worker_idx)
    return TransformedEnv(base_env, Compose(*_make_env_transforms(
        base_env, obs_rms, frame_stack, cfg.collector.max_frames_per_traj)))


def static_training_definition(cfg):
    import json
    from pathlib import Path
    manifest = json.loads(Path(cfg.experiment.manifest).read_text())
    training = manifest.get("training")
    definition = (training["tasks"] if training else manifest["tasks"])[cfg.env.plr.sampler_key]
    if training and cfg.env.id == "FrozenLake-v1":
        if training["frozenlake_initial_mode"] != cfg.env.initial_mode:
            raise ValueError("FrozenLake DR calibration and training initial mode differ")
    return definition


def make_static_dr_env(cfg, device, obs_rms=None, dtype=None, worker_idx=0):
    from AAMAS_Comp.envs.domain_randomization import DynamicsEnv
    definition = static_training_definition(cfg)
    env = DynamicsEnv(cfg.env.plr.sampler_key, definition["dr_ranges"],
                      seed=worker_seed(cfg.seed, worker_idx, 2), samples=definition["dr_samples"])
    base_env = worker_gym_env(NoInfoWrapper(SeededResetWrapper(env,
                           worker_seed(cfg.seed, worker_idx, 1))), device, worker_idx)
    return TransformedEnv(base_env, Compose(*_make_env_transforms(
        base_env, obs_rms, cfg.env.frame_stack, cfg.collector.max_frames_per_traj)))


def make_fixed_ns_eval_env(
    config,
    cfg: DictConfig,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    dtype: torch.dtype | None = None,
    rollout_seed: int = 0,
) -> TransformedEnv:
    """Create a fixed-config NS env for held-out evaluation."""
    frame_stack = cfg.env.get("frame_stack", 1)
    context_keys, context_defaults = _get_context_cfg(cfg)
    fixed_env = FixedNSEnv(config, context_keys=context_keys, context_defaults=context_defaults)
    base_env = GymWrapper(NoInfoWrapper(SeededResetWrapper(fixed_env, rollout_seed)), device=device)
    return TransformedEnv(base_env, Compose(*_make_env_transforms(
        base_env, obs_rms, frame_stack, cfg.collector.max_frames_per_traj)))


def make_ns_eval_shards(
    cfg: DictConfig,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    dtype: torch.dtype | None = None,
) -> list[list]:
    """Prepare lazy eval shards for PLR runs — no subprocesses spawned here.

    Samples num_eval_configs NS configs once using eval_seed, then groups them
    into shards of at most num_eval_episodes configs each.  Each shard is a
    list of factory callables that the caller builds into a ParallelEnv on
    demand and tears down after the rollout.

    Returns:
        List of lists, where each inner list contains factory callables.
    """
    plr_cfg = cfg.env.plr
    batch_size = cfg.training.num_eval_episodes

    held_out = sample_held_out_configs(
        sampler_key=plr_cfg.sampler_key,
        n_configs=plr_cfg.num_eval_configs,
        seed=plr_cfg.eval_seed,
        frozenlake_initial_mode=cfg.env.get("initial_mode", None),
    )

    shards: list[list] = []
    for start in range(0, len(held_out), batch_size):
        shard_configs = held_out[start : start + batch_size]
        shards.append([
            partial(make_fixed_ns_eval_env, c, cfg, device, obs_rms, dtype,
                    worker_seed(plr_cfg.eval_seed, start + i, 3))
            for i, c in enumerate(shard_configs)
        ])
    return shards
