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
import warnings
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb

# Suppress FutureWarnings from torchrl (Python 3.13 compatibility issues)
warnings.filterwarnings("ignore", category=FutureWarning, module="torchrl.modules.mcts.scores")

# TorchRL imports
from torchrl.collectors import MultiAsyncCollector
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    ParallelEnv,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Project imports
from AAMAS_Comp.agents.ppo import PPOAgent, make_ppo_models

log = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_single_env(cfg: DictConfig, device: torch.device, obs_norm_state: dict | None = None) -> TransformedEnv:
    """Create a single TorchRL ``TransformedEnv`` instance.
    
    Args:
        cfg: Hydra config.
        device: Device for the environment.
        obs_norm_state: Optional pre-computed observation normalization stats (loc, scale).
    """
    base_env = GymEnv(cfg.env.id, device=device)

    transforms = []
    if cfg.env.normalize_obs:
        obs_norm = ObservationNorm(in_keys=["observation"])
        if obs_norm_state is not None:
            obs_norm.loc = obs_norm_state["loc"]
            obs_norm.scale = obs_norm_state["scale"]
        transforms.append(obs_norm)
    transforms.append(DoubleToFloat())
    transforms.append(StepCounter())

    env = TransformedEnv(base_env, Compose(*transforms))
    return env


def make_parallel_env(cfg: DictConfig, device: torch.device, num_envs: int = 1) -> TransformedEnv | ParallelEnv:
    """Create vectorized parallel environments."""
    from functools import partial
    
    obs_norm_state = None
    
    # If using observation normalization, compute stats from a single env first
    if cfg.env.normalize_obs and num_envs > 1:
        log.info("Initializing observation normalization stats...")
        temp_env = make_single_env(cfg, device, obs_norm_state=None)
        temp_env.transform[0].init_stats(
            num_iter=cfg.env.normalize_obs_init_steps,
            reduce_dim=0,
            cat_dim=0,
        )
        obs_norm_state = {
            "loc": temp_env.transform[0].loc.clone(),
            "scale": temp_env.transform[0].scale.clone(),
        }
        temp_env.close()
        log.info("Observation normalization stats initialized.")
    
    if num_envs == 1:
        env = make_single_env(cfg, device, obs_norm_state=obs_norm_state)
        if cfg.env.normalize_obs:
            env.transform[0].init_stats(
                num_iter=cfg.env.normalize_obs_init_steps,
                reduce_dim=0,
                cat_dim=0,
            )
    else:
        # Use functools.partial instead of lambda for proper serialization
        env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=partial(make_single_env, cfg, device, obs_norm_state),
            serial_for_single=True,
        )
    
    return env


# ---------------------------------------------------------------------------
# Resolve device helper
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig) -> None:
    # ── Resolve device & seed ──────────────────────────────────────────
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    log.info(f"Device: {device} | Seed: {cfg.seed}")
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
    log.info("wandb %s", "enabled" if cfg.wandb.enabled else "disabled")

    # ── Environment ────────────────────────────────────────────────────
    num_envs = cfg.collector.num_envs
    env = make_parallel_env(cfg, device, num_envs=num_envs)
    log.info("Env: %s | num_envs=%d | obs=%s  act=%s", cfg.env.id, num_envs,
             env.observation_spec["observation"].shape,
             env.action_spec.shape)

    # ── Build PPO modules ──────────────────────────────────────────────
    models = make_ppo_models(env, cfg, device=device)
    actor = models["actor"]
    advantage_module = models["advantage"]
    loss_module = models["loss_module"]
    optimizer = models["optimizer"]
    scheduler = models["scheduler"]

    # ── Data collector ─────────────────────────────────────────────────
    collector = MultiAsyncCollector(
        [env],
        actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
    )

    # ── Checkpoint directory ───────────────────────────────────────────
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────
    total_frames = cfg.collector.total_frames
    pbar = tqdm(total=total_frames, desc="Training")

    global_step = 0

    for collect_iter, tensordict_data in enumerate(collector):
        batch_frames = tensordict_data.numel()
        global_step += batch_frames

        # ── PPO inner optimisation ─────────────────────────────────────
        epoch_losses: dict[str, list[float]] = {
            "loss_objective": [],
            "loss_critic": [],
            "loss_entropy": [],
            "loss_total": [],
        }

        for _epoch in range(cfg.training.num_epochs):
            # Recompute advantage each epoch (value net is being updated)
            advantage_module(tensordict_data)

            # Flatten the data and create random permutation for mini-batch sampling
            data_view = tensordict_data.reshape(-1)
            perm = torch.randperm(data_view.batch_size[0], device=device)
            
            n_sub = cfg.collector.frames_per_batch // cfg.training.sub_batch_size
            for i in range(n_sub):
                # Sample mini-batch indices
                idx = perm[i * cfg.training.sub_batch_size : (i + 1) * cfg.training.sub_batch_size]
                subdata = data_view[idx]
                loss_vals = loss_module(subdata)

                loss_total = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.training.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()

                epoch_losses["loss_objective"].append(loss_vals["loss_objective"].item())
                epoch_losses["loss_critic"].append(loss_vals["loss_critic"].item())
                epoch_losses["loss_entropy"].append(loss_vals["loss_entropy"].item())
                epoch_losses["loss_total"].append(loss_total.item())

        scheduler.step()

        # ── Logging ────────────────────────────────────────────────────
        mean_reward = tensordict_data["next", "reward"].mean().item()
        max_step_count = tensordict_data["step_count"].max().item()
        lr_now = optimizer.param_groups[0]["lr"]

        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}

        metrics = {
            "train/reward_mean": mean_reward,
            "train/step_count_max": max_step_count,
            "train/lr": lr_now,
            "train/global_step": global_step,
            **{f"train/{k}": v for k, v in avg_losses.items()},
        }

        # ── Evaluation ────────────────────────────────────────────────
        if collect_iter % cfg.training.eval_interval == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_rollout = env.rollout(
                    cfg.training.eval_rollout_steps, actor
                )
                eval_reward_sum = eval_rollout["next", "reward"].sum().item()
                eval_reward_mean = eval_rollout["next", "reward"].mean().item()
                eval_steps = eval_rollout["step_count"].max().item()
                del eval_rollout

            metrics.update({
                "eval/reward_sum": eval_reward_sum,
                "eval/reward_mean": eval_reward_mean,
                "eval/step_count": eval_steps,
            })
            log.info(
                "[iter %d | frames %d] eval_return=%.2f  eval_steps=%d",
                collect_iter, global_step, eval_reward_sum, eval_steps,
            )

        if cfg.wandb.enabled:
            wandb.log(metrics, step=global_step)

        pbar.update(batch_frames)
        pbar.set_postfix(
            reward=f"{mean_reward:.3f}",
            steps=max_step_count,
            lr=f"{lr_now:.2e}",
        )

        # ── Checkpointing ─────────────────────────────────────────────
        if collect_iter % cfg.training.save_interval == 0 and collect_iter > 0:
            ckpt_path = ckpt_dir / f"ppo_iter_{collect_iter}.pt"
            agent = PPOAgent(actor=actor, device=device)
            agent.save(str(ckpt_path))
            log.info("Saved checkpoint → %s", ckpt_path)

    pbar.close()
    collector.shutdown()

    # ── Final save ─────────────────────────────────────────────────────
    final_path = ckpt_dir / "ppo_final.pt"
    PPOAgent(actor=actor, device=device).save(str(final_path))
    log.info("Training complete. Final model → %s", final_path)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    train()
