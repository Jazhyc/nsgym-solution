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
# Running observation normalization
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford online running mean / variance tracker.

    Tracks the sufficient statistics (mean, var, count) so that
    ``ObservationNorm`` transforms can be kept up-to-date as training
    progresses.
    """

    def __init__(self, shape: tuple[int, ...] = (), device: torch.device | None = None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count: float = 1e-4  # small epsilon to avoid div-by-zero

    def update(self, batch: torch.Tensor) -> None:
        """Update stats with a new batch of observations ``(N, *shape)``."""
        batch = batch.reshape(-1, *self.mean.shape).float()
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, correction=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var).clamp(min=1e-6)

    def state_dict(self) -> dict:
        return {"mean": self.mean.clone(), "var": self.var.clone(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


def sync_obs_norm(obs_norm: ObservationNorm, rms: RunningMeanStd) -> None:
    """Push ``RunningMeanStd`` statistics into an ``ObservationNorm``
    that uses ``standard_normal=True``."""
    obs_norm.loc.copy_(rms.mean)
    obs_norm.scale.copy_(rms.std)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_single_env(cfg: DictConfig, device: torch.device, obs_norm_state: dict | None = None, dtype: torch.dtype | None = None) -> TransformedEnv:
    """Create a single TorchRL ``TransformedEnv`` instance.
    
    Args:
        cfg: Hydra config.
        device: Device for the environment.
        obs_norm_state: Optional dict with ``mean`` and ``std`` tensors to
            initialise the ``ObservationNorm`` (``standard_normal=True``).
        dtype: Optional dtype to cast observations to (e.g., torch.bfloat16).
    """
    base_env = GymEnv(cfg.env.id, device=device)

    transforms = []
    if cfg.env.normalize_obs:
        obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
        if obs_norm_state is not None:
            # Materialise the lazy buffers and fill them
            obs_norm.loc = torch.nn.Parameter(obs_norm_state["mean"].clone(), requires_grad=False)
            obs_norm.scale = torch.nn.Parameter(obs_norm_state["std"].clone(), requires_grad=False)
        transforms.append(obs_norm)
    transforms.append(DoubleToFloat())
    
    # Add dtype casting if specified (e.g., for bfloat16 networks)
    # Observations are float32 after DoubleToFloat, so convert from float32 to target dtype
    if dtype is not None and dtype != torch.float32:
        from torchrl.envs import DTypeCastTransform
        transforms.append(DTypeCastTransform(dtype_in=torch.float32, dtype_out=dtype, in_keys=["observation"]))
    
    transforms.append(StepCounter())

    env = TransformedEnv(base_env, Compose(*transforms))
    return env


def make_parallel_env(cfg: DictConfig, device: torch.device, num_envs: int = 1, dtype: torch.dtype | None = None, obs_norm_state: dict | None = None) -> TransformedEnv | ParallelEnv:
    """Create vectorized parallel environments."""
    from functools import partial
    
    if num_envs == 1:
        env = make_single_env(cfg, device, obs_norm_state=obs_norm_state, dtype=dtype)
    else:
        # Use functools.partial instead of lambda for proper serialization
        env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=partial(make_single_env, cfg, device, obs_norm_state, dtype),
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
    # Network device defaults to same as device if set to "auto"
    if hasattr(cfg.agent, "network_device") and cfg.agent.network_device != "auto":
        network_device = resolve_device(cfg.agent.network_device)
    else:
        network_device = device
    torch.manual_seed(cfg.seed)
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
    obs_rms: RunningMeanStd | None = None
    init_obs_norm_state: dict | None = None
    if cfg.env.normalize_obs:
        # Bootstrap stats from short random rollout in a temporary env
        log.info("Bootstrapping observation normalization stats...")
        _tmp = make_single_env(cfg, device, obs_norm_state=None, dtype=network_dtype)
        # Use init_stats to get initial loc (mean) and scale (std) via standard_normal=True
        _tmp.transform[0].init_stats(
            num_iter=cfg.env.normalize_obs_init_steps, reduce_dim=0, cat_dim=0,
        )
        obs_dim = _tmp.observation_spec["observation"].shape[-1]
        obs_rms = RunningMeanStd(shape=(obs_dim,), device=device)
        obs_rms.mean = _tmp.transform[0].loc.clone()
        obs_rms.var = _tmp.transform[0].scale.clone().pow(2)  # scale == std → var = std²
        obs_rms.count = float(cfg.env.normalize_obs_init_steps)
        init_obs_norm_state = {"mean": obs_rms.mean.clone(), "std": obs_rms.std.clone()}
        _tmp.close()
        log.info("Observation normalization bootstrapped (mean=%.3f, std=%.3f)",
                 obs_rms.mean.mean().item(), obs_rms.std.mean().item())

    env = make_parallel_env(cfg, device, num_envs=num_envs, dtype=network_dtype,
                            obs_norm_state=init_obs_norm_state)
    log.info("Env: %s | num_envs=%d | obs=%s  act=%s", cfg.env.id, num_envs,
             env.observation_spec["observation"].shape,
             env.action_spec.shape)

    # Separate eval env (single, deterministic) to avoid ParallelEnv auto-reset artefacts
    eval_env = make_single_env(cfg, device, obs_norm_state=init_obs_norm_state, dtype=network_dtype)

    # ── Build PPO modules ──────────────────────────────────────────────
    models = make_ppo_models(env, cfg, device=device, network_device=network_device, dtype=network_dtype)
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
                num_eval_episodes = 5
                ep_returns = []
                ep_lengths = []
                for _ in range(num_eval_episodes):
                    reset_td = eval_env.reset()
                    eval_rollout = eval_env.rollout(
                        cfg.training.eval_rollout_steps, actor,
                        auto_reset=False, tensordict=reset_td,
                        break_when_any_done=True,
                    )
                    ep_return = eval_rollout["next", "reward"].sum().item()
                    ep_len = eval_rollout.batch_size[0]
                    ep_returns.append(ep_return)
                    ep_lengths.append(ep_len)
                    del eval_rollout

                eval_reward_sum = sum(ep_returns) / len(ep_returns)
                eval_steps = sum(ep_lengths) / len(ep_lengths)

            metrics.update({
                "eval/reward_sum": eval_reward_sum,
                "eval/reward_mean": eval_reward_sum / max(eval_steps, 1),
                "eval/step_count": eval_steps,
            })
            log.info(
                "[iter %d | frames %d] eval_return=%.2f  eval_steps=%.1f",
                collect_iter, global_step, eval_reward_sum, eval_steps,
            )

        if cfg.wandb.enabled:
            wandb.log(metrics, step=global_step)

        pbar.update(batch_frames)
        postfix = dict(reward=f"{mean_reward:.3f}")
        if collect_iter % cfg.training.eval_interval == 0:
            postfix["eval_return"] = f"{eval_reward_sum:.2f}"
        pbar.set_postfix(postfix)

        # ── Checkpointing ─────────────────────────────────────────────
        if collect_iter % cfg.training.save_interval == 0 and collect_iter > 0:
            ckpt_path = ckpt_dir / f"ppo_iter_{collect_iter}.pt"
            agent = PPOAgent(actor=actor, device=device,
                             obs_rms=obs_rms)
            agent.save(str(ckpt_path))
            log.info("Saved checkpoint → %s", ckpt_path)

    pbar.close()
    collector.shutdown()
    try:
        env.close()
    except RuntimeError:
        pass  # Already closed by collector shutdown
    eval_env.close()

    # ── Final save ─────────────────────────────────────────────────────
    final_path = ckpt_dir / "ppo_final.pt"
    PPOAgent(actor=actor, device=device, obs_rms=obs_rms).save(str(final_path))
    log.info("Training complete. Final model → %s", final_path)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    train()
