"""PPO training script with Hydra config + Weights & Biases logging.

Usage
-----
    # default (Ant-v5)
    python scripts/train.py

    # override env
    python scripts/train.py env=cartpole

    # override hyper-params on the CLI
    python scripts/train.py agent.lr=1e-3 collector.total_frames=500_000

    # disable wandb
    python scripts/train.py wandb.enabled=false
"""

from __future__ import annotations

import logging
import os
import random
import time
import warnings
from pathlib import Path

# Suppress FutureWarnings from torchrl (Python 3.13 compatibility issues)
# Must be registered before any torchrl import, as the warnings fire at module load time.
warnings.filterwarnings("ignore", category=FutureWarning, module="torchrl.modules.mcts.scores")

import multiprocessing as mp
from functools import partial

import hydra
import numpy as np
import torch
import torch._dynamo
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from AAMAS_Comp.agent import compute_and_save_fisher

# EnvBase.__del__ calls set_num_threads() to restore subprocess state.  If
# this fires while dynamo is recompiling (e.g. async collector background
# thread), the guard-check assertion fails inside __del__ and Python prints
# "Exception ignored in: EnvBase.__del__".  Training is unaffected (it falls
# back to eager), but the noise is confusing.  suppress_errors makes dynamo
# fall back silently instead of propagating the AssertionError.
torch._dynamo.config.suppress_errors = True

import wandb

from AAMAS_Comp.optimizers.cbp import ContinualBackpropagation

from torchrl.collectors import MultiAsyncCollector, MultiSyncCollector
from torchrl.envs import ParallelEnv

from AAMAS_Comp.agents.ppo import PPOAgent, make_ppo_models, RPOTanhNormal
from AAMAS_Comp.envs.wrappers import RunningMeanStd, RewardNormalizer
from AAMAS_Comp.envs.torchrl_factory import (
    initialize_obs_norm,
    make_single_env,
    make_ns_plr_env,
    make_ns_random_env,
    make_ns_eval_shards,
    make_static_dr_env,
)
from AAMAS_Comp.evaluation.utils import run_eval_shards
from AAMAS_Comp.curriculum.feedback import EpisodeScores
from AAMAS_Comp.agents.episode_batches import CompleteEpisodes, estimate_advantages
from AAMAS_Comp.evaluation.training_metrics import TrainingEpisodes, WorkerCurriculumMetrics

log = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


@hydra.main(version_base=None, config_path="../config", config_name="config_ant")
def train(cfg: DictConfig) -> None:
    # ── Resolve device & seed ──────────────────────────────────────────
    device = _resolve_device(cfg.device)
    if hasattr(cfg.agent, "network_device") and cfg.agent.network_device != "auto":
        network_device = _resolve_device(cfg.agent.network_device)
    else:
        network_device = device
    # Tune PyTorch intra-op threads — small MLPs on CPU benefit from fewer threads
    # (thread-launch overhead > benefit for small matrix sizes)
    num_threads = cfg.get("num_threads", 0)
    if num_threads > 0:
        torch.set_num_threads(num_threads)
        log.info("torch.set_num_threads(%d)", num_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    log.info(f"Device: {device} | Network Device: {network_device} | Seed: {cfg.seed}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Weights & Biases ───────────────────────────────────────────────
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else [],
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            dir=cfg.wandb.dir,
            reinit=True,
        )
        wandb.define_metric("train/global_step")
        for namespace in ("train/*", "eval/*", "test/*", "plr/*", "accel/*"):
            wandb.define_metric(namespace, step_metric="train/global_step")
    log.info("wandb %s", "enabled" if cfg.wandb.enabled else "disabled")

    # ── Environment ────────────────────────────────────────────────────
    num_envs = cfg.collector.num_envs
    
    # Determine network dtype early
    dtype_str = cfg.agent.get("dtype", "float32")
    dtype_map = {
        "float32": None,  # None means use default float32
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    network_dtype = dtype_map.get(dtype_str, None)
    if network_dtype is not None:
        log.info(f"Creating networks and observations in dtype: {network_dtype}")
    
    # ── Observation normalization ────────────────────────────────────
    obs_rms = initialize_obs_norm(cfg, device, dtype=network_dtype)

    # ── Reward normalization (VecNormalize-style) ──────────────────────
    ret_normalizer = None
    if cfg.env.get("normalize_reward", False):
        ret_normalizer = RewardNormalizer(device=device)
        log.info("Reward normalization enabled")

    num_groups = cfg.collector.get("num_groups", 1)
    envs_per_group = max(1, num_envs // num_groups)
    plr_enabled = cfg.env.get("plr", {}).get("enabled", False)
    ns_baseline = cfg.env.get("ns_baseline", False)

    # Manager().Queue() uses a proxy object that works across process boundaries
    # regardless of spawn/fork start method. Plain mp.Queue() can silently
    # disconnect when pickled through ParallelEnv workers.
    if plr_enabled:
        _plr_mp_manager = mp.Manager()
        plr_stats_queue = _plr_mp_manager.Queue(maxsize=2000)
        feedback_queues = [_plr_mp_manager.Queue() for _ in range(num_envs)]
        episode_scores = EpisodeScores(require_complete=True)
        plr_score_array = None
        collector_factory = None  # replaced by per-worker factory lists below
    elif ns_baseline:
        _plr_mp_manager = None
        plr_stats_queue = None
        plr_score_array = None
        collector_factory = partial(make_ns_random_env, cfg, device, obs_rms, network_dtype)
    else:
        _plr_mp_manager = None
        plr_stats_queue = None
        plr_score_array = None
        collector_factory = partial(make_single_env, cfg, device, obs_rms, network_dtype)

    experiment = cfg.get("experiment", {})
    method = experiment.get("method", None)
    if method == "standard_dr":
        collector_factory = partial(make_static_dr_env, cfg, device, obs_rms, network_dtype)

    collector_envs = [
        ParallelEnv(
            num_workers=envs_per_group,
            create_env_fn=(
                [
                    partial(make_ns_plr_env, cfg, device, obs_rms, network_dtype,
                            plr_stats_queue,
                            worker_idx=g * envs_per_group + w,
                            feedback_queue=feedback_queues[g * envs_per_group + w])
                    for w in range(envs_per_group)
                ]
                if plr_enabled else [partial(collector_factory,
                    worker_idx=g * envs_per_group + w) for w in range(envs_per_group)]
            ),
            serial_for_single=True,
        )
        for g in range(num_groups)
    ]

    # Use first env only for shape logging
    env = collector_envs[0]
    train_mode = method or ("PLR" if plr_enabled else ("NS-baseline" if ns_baseline else "stationary"))
    log.info(
        "Env: %s | mode=%s | num_groups=%d | envs_per_group=%d | total_envs=%d | obs=%s  act=%s",
        cfg.env.id, train_mode, num_groups, envs_per_group, num_groups * envs_per_group,
        env.observation_spec["observation"].shape,
        env.action_spec.shape,
    )

    num_eval_episodes = cfg.training.get("num_eval_episodes", 5)
    if plr_enabled or ns_baseline or experiment:
        n_configs = cfg.env.plr.num_eval_configs
        eval_shard_factories = make_ns_eval_shards(cfg, device, obs_rms=obs_rms, dtype=network_dtype)
        log.info(
            "Eval: %d held-out NS configs (seed=%d), %d parallel eval envs, %d shard(s)",
            n_configs, cfg.env.plr.eval_seed, num_eval_episodes, len(eval_shard_factories),
        )
    else:
        # Single shard of num_eval_episodes base-env instances.
        eval_shard_factories = [
            [partial(make_single_env, cfg, device, obs_rms, network_dtype)
             for _ in range(num_eval_episodes)]
        ]

    # ── Build PPO modules ──────────────────────────────────────────────
    # Set RPO alpha before building models (distribution class reads it)
    rpo_alpha = cfg.agent.get("rpo_alpha", 0.0)
    RPOTanhNormal.rpo_alpha = rpo_alpha
    if rpo_alpha > 0:
        log.info("RPO enabled with alpha=%.2f", rpo_alpha)

    # Match initial weights across arms regardless of environment construction.
    torch.manual_seed(cfg.seed)
    models = make_ppo_models(env, cfg, device=device, network_device=network_device, dtype=network_dtype)
    actor = models["actor"]
    critic = models.get("critic") 
    advantage_module = models["advantage"]
    loss_module = models["loss_module"]
    optimizer = models["optimizer"]
    scheduler = models["scheduler"]
    optim_params = models["optim_params"]

    # ── Continual Backpropagation (optional) ──────────────────────────
    cbp = None
    cbp_cfg = cfg.agent.get("cbp", None)
    if cbp_cfg and cbp_cfg.get("enabled", False):
        cbp = ContinualBackpropagation(
            model=loss_module,          # wraps actor + critic weights
            optimizer=optimizer,
            reset_rate=cbp_cfg.get("reset_rate", 0.01),
            maturity_threshold=cbp_cfg.get("maturity_threshold", 50),
            utility_decay=cbp_cfg.get("utility_decay", 0.05),
            reset_init=cbp_cfg.get("reset_init", "uniform"),
            momentum=cbp_cfg.get("momentum", 0.9),
            device=network_device,
        )
        log.info("CBP enabled — reset_rate=%.3f  maturity_threshold=%d",
                 cbp_cfg.reset_rate, cbp_cfg.maturity_threshold)
    
    # ── Data collector ─────────────────────────────────────────────────
    # ObservationNorm is in the env transforms → observations in tensordict
    # are already normalized. Use plain actor for collection.
    collector_class = (MultiSyncCollector if cfg.collector.get("synchronous", False)
                       else MultiAsyncCollector)
    collector = collector_class(
        collector_envs,
        actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        num_threads=max(1, int(cfg.get("num_threads", 1))),
        num_sub_threads=1,
    )
    collector.set_seed(int(cfg.seed))

    # ── Checkpoint directory ───────────────────────────────────────────
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Context metadata (for inference-time obs reconstruction) ──────
    from omegaconf import OmegaConf as _OC
    _context_keys = list(cfg.env.get("context_features", []) or [])
    _raw_defaults = cfg.env.get("context_defaults", {}) or {}
    _raw_defaults_dict = _OC.to_container(_raw_defaults, resolve=True) if _OC.is_config(_raw_defaults) else _raw_defaults
    _context_defaults = {k: list(v) for k, v in _raw_defaults_dict.items()}
    # n_state: number of discrete states before context augmentation (None for continuous)
    import gymnasium as _gym
    _tmp_env = _gym.make(cfg.env.id)
    _n_state = _tmp_env.observation_space.n if hasattr(_tmp_env.observation_space, "n") else None
    _tmp_env.close()
    del _tmp_env
    _context_meta = {
        "context_keys": _context_keys,
        "context_defaults": _context_defaults,
        "n_state": _n_state,
    }

    # ── V-trace setup ──────────────────────────────────────────────────
    use_vtrace = cfg.agent.get("use_vtrace", False)
    if use_vtrace:
        log.info("V-trace off-policy correction enabled for async collector")

    # ── Warmup torch.compile before any env teardown can race with it ──
    # The first actor(td) call triggers dynamo tracing. If an eval env's
    # __del__ fires concurrently (resetting num_threads), dynamo raises an
    # AssertionError inside __del__ which Python suppresses — benign but noisy.
    # Forcing a compile pass here ensures tracing finishes before the first
    # eval shard is torn down.
    with torch.no_grad():
        _warmup_td = collector_envs[0].reset()
        actor(_warmup_td)
        del _warmup_td

    # ── Training loop ──────────────────────────────────────────────────
    total_frames = cfg.collector.total_frames
    pbar = tqdm(total=total_frames, desc="Training")

    # Eval scheduling: trigger eval at evenly-spaced frame thresholds.
    # eval_ratio=0.1 → eval at 0%, 10%, 20%, …, 100% of total_frames.
    eval_ratio = cfg.training.get("eval_ratio", 0.1)
    eval_every_frames = max(1, int(total_frames * eval_ratio))
    next_eval_frame = 0  # eval on first iteration

    global_step = 0
    best_eval_iqm = float("-inf")
    training_started = time.monotonic()
    completed_episodes = TrainingEpisodes(frozenlake=cfg.env.plr.sampler_key == "frozenlake")
    curriculum_metrics = WorkerCurriculumMetrics(num_envs)
    feedback_sent = 0
    last_metric_values = {}
    complete_batches = CompleteEpisodes() if cfg.training.get("complete_episodes", False) else None
    bootstrap_timeouts = cfg.agent.get("bootstrap_timeouts", True)
    if complete_batches is not None:
        if not cfg.collector.get("synchronous", False) or ret_normalizer is not None:
            raise ValueError("Complete episodes require synchronous collection and raw rewards")
        if use_vtrace:
            raise ValueError("Complete-episode diagnostics use GAE, not V-trace")

    for collect_iter, tensordict_data in enumerate(collector):
        batch_frames = tensordict_data.numel()
        global_step += batch_frames
        did_eval_this_iter = False

        # Drop actor intermediate outputs — ClipPPOLoss recomputes them from
        # "observation"; keeping them just wastes memory and TensorDict ops.
        # Covers both continuous (loc, scale) and discrete (logits) policies.
        keys_to_drop = [k for k in ("loc", "scale", "logits") if k in tensordict_data.keys()]
        if keys_to_drop:
            tensordict_data.exclude(*keys_to_drop)

        # ── Save raw reward for logging (before normalization) ─────────
        raw_mean_reward = tensordict_data["next", "reward"].mean().item()
        episode_metrics = completed_episodes.update(tensordict_data)

        # ── Normalize rewards (VecNormalize-style) ─────────────────────
        if ret_normalizer is not None:
            ret_normalizer.normalize_batch(tensordict_data)

        if complete_batches is not None:
            tensordict_data = complete_batches.update(tensordict_data)
            if tensordict_data is None:
                pbar.update(batch_frames)
                continue

        # ── PPO inner optimisation ─────────────────────────────────────
        epoch_losses: dict[str, list[float]] = {
            "loss_objective": [],
            "loss_critic": [],
            "loss_entropy": [],
            "loss_total": [],
            "kl_approx": [],
            "clip_fraction": [],
        }
        target_kl = cfg.training.get("target_kl", None)
        epochs_done = 0

        # Compute V-trace advantages once before epoch loop (expensive due
        # to actor forward pass; correction is for behavior→current mismatch,
        # not intra-epoch drift).  GAE is cheap so we keep it per-epoch.
        if use_vtrace:
            with torch.no_grad():
                if tensordict_data.ndim == 1:
                    time_steps = batch_frames // num_envs
                    td_seq = tensordict_data.reshape(num_envs, time_steps)
                else:
                    td_seq = tensordict_data
                estimate_advantages(advantage_module, td_seq, bootstrap_timeouts=bootstrap_timeouts)
                if tensordict_data.ndim == 1:
                    tensordict_data = td_seq.reshape(-1)

        # Compute advantages once before the epoch loop.
        # With only a few epochs the value net changes minimally between epochs,
        # so recomputing GAE every epoch is not worth the extra critic forward pass.
        if not use_vtrace:
            with torch.no_grad():
                estimate_advantages(advantage_module, tensordict_data, bootstrap_timeouts=bootstrap_timeouts)

        if plr_enabled:
            feedback_sent += episode_scores.update(tensordict_data, feedback_queues)
            episode_metrics.update({
                "plr/incomplete_episode_scores_total": episode_scores.incomplete_episodes,
                "plr/abandoned_episode_scores_total": episode_scores.abandoned_episodes,
            })

        data_view = tensordict_data.reshape(-1)
        n_sub = cfg.collector.frames_per_batch // cfg.training.sub_batch_size
        if complete_batches is not None:
            n_sub = (data_view.numel() + cfg.training.sub_batch_size - 1) // cfg.training.sub_batch_size

        for _epoch in range(cfg.training.num_epochs):
            perm = torch.randperm(data_view.batch_size[0], device=device)
            epoch_kl = 0.0

            for i in range(n_sub):
                idx = perm[i * cfg.training.sub_batch_size : (i + 1) * cfg.training.sub_batch_size]
                subdata = data_view[idx]

                RPOTanhNormal.rpo_enabled = True
                loss_vals = loss_module(subdata)
                RPOTanhNormal.rpo_enabled = False

                loss_total = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals.get("loss_entropy", 0.0)
                )

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    optim_params, cfg.training.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if cbp is not None:
                    cbp.step(global_step)

                epoch_losses["loss_objective"].append(loss_vals["loss_objective"].detach())
                epoch_losses["loss_critic"].append(loss_vals["loss_critic"].detach())
                epoch_losses["loss_entropy"].append(loss_vals.get("loss_entropy", torch.tensor(0.0)).detach())
                epoch_losses["loss_total"].append(loss_total.detach())
                epoch_losses["kl_approx"].append(loss_vals.get("kl_approx", torch.tensor(0.0)).detach())
                epoch_losses["clip_fraction"].append(loss_vals.get("clip_fraction", torch.tensor(0.0)).detach())
                epoch_kl += loss_vals.get("kl_approx", torch.tensor(0.0)).item()

            epochs_done += 1
            if target_kl is not None and (epoch_kl / n_sub) > 1.5 * target_kl:
                break

        # Compute policy entropy once per collect_iter (not per mini-batch).
        # Uses first mini-batch of data_view as a representative sample.
        with torch.no_grad():
            dist = actor.get_dist(data_view[:cfg.training.sub_batch_size])
            # Continuous: dist is a transformed dist with base_dist (e.g. TanhNormal)
            # Discrete:   dist is a Categorical with no base_dist
            inner = getattr(dist, "base_dist", dist)
            policy_entropy = inner.entropy().mean().item()

        scheduler.step()

        # ── Logging ────────────────────────────────────────────────────
        mean_reward = raw_mean_reward  # Use raw (non-normalized) reward
        max_step_count = tensordict_data["step_count"].max().item()
        lr_now = optimizer.param_groups[0]["lr"]

        avg_losses = {k: torch.stack(v).mean().item() for k, v in epoch_losses.items()}

        metrics = {
            "train/reward_mean": mean_reward,
            "train/reward_per_step_mean": mean_reward,
            "train/step_count_max": max_step_count,
            "train/lr": lr_now,
            "train/global_step": global_step,
            "train/policy_entropy": policy_entropy,
            "train/epochs_done": epochs_done,
            "train/wall_seconds": time.monotonic() - training_started,
            "train/frames_per_second": global_step / (time.monotonic() - training_started),
            **{f"train/{k}": v for k, v in avg_losses.items()},
            **episode_metrics,
        }
        if plr_enabled:
            metrics["plr/feedback_sent_total"] = feedback_sent
        if complete_batches is not None:
            metrics["train/complete_episode_frames_total"] = complete_batches.frames_emitted
            metrics["train/pending_episode_frames"] = complete_batches.pending_frames

        # ── Evaluation ────────────────────────────────────────────────
        if global_step >= next_eval_frame:
            did_eval_this_iter = True
            next_eval_frame += eval_every_frames

            eval_results = run_eval_shards(
                actor,
                eval_shard_factories,
                eval_rollout_steps=cfg.training.eval_rollout_steps,
                device=device,
            )
            eval_reward_mean = eval_results["eval/reward_mean"]
            eval_reward_iqm = eval_results["eval/reward_iqm"]
            best_eval_iqm = max(best_eval_iqm, eval_reward_iqm)
            metrics.update({k: v for k, v in eval_results.items() if k not in ("n_shards", "n_configs")})
            log.info(
                "[iter %d | frames %d] eval_return=%.2f  eval_iqm=%.2f  eval_steps=%.1f  (%d configs, %d shards)",
                collect_iter, global_step, eval_reward_mean, eval_reward_iqm,
                eval_results["eval/step_count"], eval_results["n_configs"], eval_results["n_shards"],
            )

        # ── PLR buffer stats (aggregated from all worker subprocesses) ───
        if plr_stats_queue is not None:
            plr_samples = []
            while True:
                try:
                    plr_samples.append(plr_stats_queue.get_nowait())
                except Exception:
                    break
            if plr_samples:
                metrics.update(curriculum_metrics.update(plr_samples))

        last_metric_values.update(metrics)
        if cfg.wandb.enabled:
            wandb.log(metrics, step=global_step)

        pbar.update(batch_frames)
        postfix = dict(reward=f"{mean_reward:.3f}")
        if did_eval_this_iter:
            postfix["eval_return"] = f"{eval_reward_mean:.2f}  iqm={eval_reward_iqm:.2f}"
        pbar.set_postfix(postfix)

        # ── Checkpointing ─────────────────────────────────────────────
        if collect_iter % cfg.training.save_interval == 0 and collect_iter > 0:
            ckpt_path = ckpt_dir / f"ppo_iter_{collect_iter}.pt"
            agent = PPOAgent(actor=actor, critic=critic, device=device,
                             obs_rms=obs_rms, context_meta=_context_meta)
            agent.save(str(ckpt_path))
            log.info("Saved checkpoint → %s", ckpt_path)

    pbar.close()
    collector.shutdown()
    for e in collector_envs:
        try:
            e.close()
        except RuntimeError:
            pass  # Already closed by collector shutdown
    # ── Final save ─────────────────────────────────────────────────────
    final_path = ckpt_dir / "ppo_final.pt"
    PPOAgent(actor=actor, critic=critic, device=device, obs_rms=obs_rms, context_meta=_context_meta).save(str(final_path))
    log.info("Training complete. Final model → %s", final_path)

    if _plr_mp_manager is not None:
        _plr_mp_manager.shutdown()

    # ── Hyperparameter search output ───────────────────────────────────────
    # Written when train.py is launched by hparam_search.py via the
    # `+hparam_output_path=<file>` Hydra override.
    hparam_output_path = cfg.get("hparam_output_path", None)
    if hparam_output_path:
        import json as _json
        Path(hparam_output_path).write_text(
            _json.dumps({"eval/reward_iqm": best_eval_iqm})
        )
        log.info("Wrote hparam metric to %s (best eval/reward_iqm=%.4f)", hparam_output_path, best_eval_iqm)

    if not experiment:
        compute_and_save_fisher(
            model_path=str(final_path),
            output_path=str(final_path),
            actor=actor,
            tensordict_data=tensordict_data,
            device=str(device),
            n_samples=cfg.get("fisher_samples", 2000),
        )

    if experiment:
        import json
        from importlib.metadata import version
        metadata = {"method": method, "task": cfg.env.plr.sampler_key,
                    "seed": int(cfg.seed), "stage": experiment.get("stage", "validation"),
                    "actual_frames": global_step,
                    "training_seconds": time.monotonic() - training_started,
                    "package_versions": {name: version(name) for name in
                                         ("torch", "torchrl", "tensordict", "gymnasium", "ns-gym", "wandb")},
                    "config": OmegaConf.to_container(cfg, resolve=True)}
        if cfg.wandb.enabled:
            metadata["wandb_run"] = {"entity": wandb.run.entity, "project": wandb.run.project,
                                     "id": wandb.run.id, "url": wandb.run.url}
        (ckpt_dir / "run.json").write_text(json.dumps(metadata, indent=2))
        (ckpt_dir / "training_metrics_summary.json").write_text(json.dumps({
            "semantics": "Last logged value of each training/validation metric, matching W&B summaries",
            "metrics": last_metric_values}, indent=2))
        if experiment.get("final_evaluation", True):
            from AAMAS_Comp.evaluation.paper import evaluate_checkpoint
            metrics = evaluate_checkpoint(str(final_path), experiment.manifest,
                                          cfg.env.plr.sampler_key, ckpt_dir, metadata)
            if experiment.get("public_evaluation", False):
                from AAMAS_Comp.evaluation.public import evaluate_public_checkpoint
                metrics.update(evaluate_public_checkpoint(
                    str(final_path), cfg.env.plr.sampler_key, ckpt_dir, metadata,
                    episodes_per_level=int(experiment.get("public_episodes", 100))))
            if cfg.wandb.enabled:
                wandb.log(metrics, step=global_step)
        if cfg.wandb.enabled:
            artifact = wandb.Artifact(f"{metadata['stage']}-{cfg.env.plr.sampler_key}-{method}-seed{cfg.seed}",
                                      type="paper-run", metadata=metadata)
            for path in ckpt_dir.iterdir():
                if path.is_file() and (path.suffix in (".json", ".jsonl", ".npz") or path == final_path):
                    artifact.add_file(str(path))
            artifact.add_file(str(experiment.manifest), name="manifest.json")
            if (ckpt_dir/'curriculum').is_dir():
                artifact.add_dir(str(ckpt_dir/'curriculum'), name='curriculum')
            wandb.log_artifact(artifact)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    train()
