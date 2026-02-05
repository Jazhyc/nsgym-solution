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
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb

# TorchRL imports
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
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

def make_env(cfg: DictConfig, device: torch.device) -> TransformedEnv:
    """Create a TorchRL ``TransformedEnv`` from Hydra config."""
    base_env = GymEnv(cfg.env.id, device=device)

    transforms = []
    if cfg.env.normalize_obs:
        transforms.append(ObservationNorm(in_keys=["observation"]))
    transforms.append(DoubleToFloat())
    transforms.append(StepCounter())

    env = TransformedEnv(base_env, Compose(*transforms))

    # Initialise observation normalisation stats with random rollouts
    if cfg.env.normalize_obs:
        env.transform[0].init_stats(
            num_iter=cfg.env.normalize_obs_init_steps,
            reduce_dim=0,
            cat_dim=0,
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
    env = make_env(cfg, device)
    log.info("Env: %s | obs=%s  act=%s", cfg.env.id,
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
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        split_trajs=False,
        device=device,
    )

    # ── Replay buffer ──────────────────────────────────────────────────
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.collector.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
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

            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            n_sub = cfg.collector.frames_per_batch // cfg.training.sub_batch_size
            for _ in range(n_sub):
                subdata = replay_buffer.sample(cfg.training.sub_batch_size)
                loss_vals = loss_module(subdata.to(device))

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
