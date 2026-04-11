"""TorchRL env factories for NS-Gym training.

Thin wrappers that combine NS envs, PLR, and TorchRL transforms into
ready-to-use TransformedEnv instances for collection and evaluation.
"""

from __future__ import annotations

import logging
from functools import partial

import gymnasium as gym
import torch
from omegaconf import DictConfig

from torchrl.data import OneHot, UnboundedContinuous
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
from AAMAS_Comp.envs.wrappers import NoInfoWrapper, RunningMeanStd

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
    base_env = GymWrapper(NoInfoWrapper(gym.make(cfg.env.id)), device=device)
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
    transforms.append(StepCounter())
    if frame_stack > 1:
        transforms.append(CatFrames(N=frame_stack, dim=-1, in_keys=["observation"]))
    return transforms


def make_single_env(
    cfg: DictConfig,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    dtype: torch.dtype | None = None,
) -> TransformedEnv:
    """Create a single TorchRL TransformedEnv instance."""
    frame_stack = cfg.env.get("frame_stack", 1)
    base_env = GymWrapper(NoInfoWrapper(gym.make(cfg.env.id)), device=device)
    return TransformedEnv(base_env, Compose(*_make_env_transforms(base_env, obs_rms, frame_stack)))


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


def make_ns_plr_env(
    cfg: DictConfig,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    dtype: torch.dtype | None = None,
    stats_queue=None,
    worker_idx: int = 0,
    score_array=None,
) -> TransformedEnv:
    """Create a PLREnv-backed TorchRL TransformedEnv for NS training.

    Each call produces an independent env with its own PLRBuffer — safe to
    run in separate ParallelEnv subprocesses.
    """
    plr_cfg = cfg.env.plr
    frame_stack = cfg.env.get("frame_stack", 1)
    plr_env = PLREnv(
        sampler_key=plr_cfg.sampler_key,
        plr_capacity=plr_cfg.capacity,
        replay_prob=plr_cfg.replay_prob,
        mutation_prob=plr_cfg.get("mutation_prob", 0.0),
        mutation_sigma=plr_cfg.get("mutation_sigma", 0.2),
        score_temp=plr_cfg.score_temp,
        staleness_coef=plr_cfg.staleness_coef,
        seed=None,  # OS entropy so each worker explores different configs
        stats_queue=stats_queue,
        worker_idx=worker_idx,
        score_array=score_array,
    )
    base_env = GymWrapper(NoInfoWrapper(plr_env), device=device)
    return TransformedEnv(base_env, Compose(*_make_env_transforms(base_env, obs_rms, frame_stack)))


def make_ns_random_env(
    cfg: DictConfig,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    dtype: torch.dtype | None = None,
) -> TransformedEnv:
    """Create a randomly-sampled NS env (no PLR) for NS baseline training.

    Samples a fresh NS config on every reset() using uniform random sampling —
    no PLR scoring or replay.  Eval uses the same held-out NS configs as PLR
    (make_ns_eval_shards) for a fair comparison.
    """
    plr_cfg = cfg.env.plr
    frame_stack = cfg.env.get("frame_stack", 1)
    random_env = RandomNSEnv(sampler_key=plr_cfg.sampler_key, seed=None)
    base_env = GymWrapper(NoInfoWrapper(random_env), device=device)
    return TransformedEnv(base_env, Compose(*_make_env_transforms(base_env, obs_rms, frame_stack)))


def make_fixed_ns_eval_env(
    config,
    cfg: DictConfig,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    dtype: torch.dtype | None = None,
) -> TransformedEnv:
    """Create a fixed-config NS env for held-out evaluation."""
    frame_stack = cfg.env.get("frame_stack", 1)
    fixed_env = FixedNSEnv(config)
    base_env = GymWrapper(NoInfoWrapper(fixed_env), device=device)
    return TransformedEnv(base_env, Compose(*_make_env_transforms(base_env, obs_rms, frame_stack)))


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
    )

    shards: list[list] = []
    for start in range(0, len(held_out), batch_size):
        shard_configs = held_out[start : start + batch_size]
        shards.append([
            partial(make_fixed_ns_eval_env, c, cfg, device, obs_rms, dtype)
            for c in shard_configs
        ])
    return shards
