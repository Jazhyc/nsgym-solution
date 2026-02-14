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
import gymnasium as gym
from gymnasium import Wrapper as GymnasiumWrapper

from torchrl.envs.libs.gym import GymEnv, GymWrapper


class NoInfoWrapper(GymnasiumWrapper):
    """Drop the info dict from step/reset — prevents unused keys (Ant reward
    components, position/velocity diagnostics) from being serialized over IPC
    on every environment step."""

    def step(self, action):
        obs, reward, terminated, truncated, _info = self.env.step(action)
        return obs, reward, terminated, truncated, {}

    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return obs, {}
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Project imports
from AAMAS_Comp.agents.ppo import PPOAgent, make_ppo_models, RPOTanhNormal

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
        # Use float64 for better numerical stability in running statistics
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.var = torch.ones(shape, device=device, dtype=torch.float64)
        self.count: float = 0  # actual number of samples seen

    def update(self, batch: torch.Tensor) -> None:
        """Update stats with a new batch of observations ``(N, *shape)``."""
        batch = batch.reshape(-1, *self.mean.shape)
        # Compute batch stats in float64 for numerical stability
        batch_f64 = batch.double()
        batch_mean = batch_f64.mean(dim=0)
        batch_var = batch_f64.var(dim=0, correction=0)
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
        # Clamp variance before sqrt for stability, use higher minimum for safety
        return torch.sqrt(self.var.clamp(min=1e-8)).clamp(min=1e-4)

    def state_dict(self) -> dict:
        return {"mean": self.mean.clone(), "var": self.var.clone(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


class RewardNormalizer:
    """VecNormalize-style reward scaling.
    
    Tracks running variance of rewards and divides by sqrt(var) to bring
    reward magnitudes to ~O(1).  Does NOT subtract the mean.
    
    Uses epsilon-initialized count (like SB3) so the initial std = 1.0
    (identity scaling) and statistics build up gradually.
    """
    def __init__(self, device: torch.device | None = None):
        self.reward_rms = RunningMeanStd(shape=(), device=device)
        # Initialize with epsilon count so initial std = sqrt(1.0) = 1.0
        self.reward_rms.count = 1e-4
    
    def normalize_batch(self, tensordict_data) -> None:
        """Update stats and normalize rewards in a collected batch in-place."""
        rewards = tensordict_data["next", "reward"]
        # Update running stats with raw rewards (vectorized, no Python loop)
        self.reward_rms.update(rewards)
        # Scale rewards by 1/std (no mean subtraction)
        std = self.reward_rms.std.to(rewards.dtype)
        tensordict_data["next", "reward"].copy_(rewards / std)


def initialize_obs_norm(cfg: DictConfig, device: torch.device, 
                        dtype: torch.dtype | None = None) -> RunningMeanStd | None:
    """Bootstrap observation normalization statistics from initial random rollout.
    
    Returns:
        RunningMeanStd with bootstrap stats, or None if not enabled.
    """
    if not cfg.env.normalize_obs:
        return None
    
    log.info("Bootstrapping observation normalization stats...")
    # Use a plain env (no normalization) to collect raw observations
    base_env = GymWrapper(NoInfoWrapper(gym.make(cfg.env.id)), device=device)
    tmp = TransformedEnv(base_env, Compose(DoubleToFloat(), StepCounter()))
    tmp.reset()
    td = tmp.rollout(max_steps=cfg.env.normalize_obs_init_steps, break_when_any_done=False)
    all_obs = td["observation"]
    
    obs_dim = tmp.observation_spec["observation"].shape[-1]
    obs_rms = RunningMeanStd(shape=(obs_dim,), device=device)
    obs_rms.count = 1e-4  # SB3-style epsilon init so initial std=1
    obs_rms.update(all_obs)
    tmp.close()
    
    log.info("Observation normalization bootstrapped (mean=%.3f, std=%.3f)",
             obs_rms.mean.mean().item(), obs_rms.std.mean().item())
    
    return obs_rms


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_single_env(cfg: DictConfig, device: torch.device, obs_rms: RunningMeanStd | None = None, dtype: torch.dtype | None = None) -> TransformedEnv:
    """Create a single TorchRL ``TransformedEnv`` instance.

    Args:
        cfg: Hydra config.
        device: Device for the environment.
        obs_rms: Optional RunningMeanStd with frozen stats for ObservationNorm.
        dtype: Optional dtype to cast observations to.
    """
    # NoInfoWrapper drops Ant's reward components + position/velocity diagnostics
    # from the TensorDict — they're serialized over IPC every step but never used.
    base_env = GymWrapper(NoInfoWrapper(gym.make(cfg.env.id)), device=device)

    transforms = []
    if obs_rms is not None:
        obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
        # Materialise lazy buffers with frozen stats from bootstrap
        obs_norm.loc = torch.nn.Parameter(obs_rms.mean.float().clone(), requires_grad=False)
        obs_norm.scale = torch.nn.Parameter(obs_rms.std.float().clone(), requires_grad=False)
        transforms.append(obs_norm)
    transforms.append(DoubleToFloat())
    transforms.append(StepCounter())

    env = TransformedEnv(base_env, Compose(*transforms))
    return env


def make_parallel_env(cfg: DictConfig, device: torch.device, num_envs: int = 1, dtype: torch.dtype | None = None, obs_rms: RunningMeanStd | None = None) -> TransformedEnv | ParallelEnv:
    """Create vectorized parallel environments."""
    from functools import partial
    
    if num_envs == 1:
        env = make_single_env(cfg, device, obs_rms=obs_rms, dtype=dtype)
    else:
        env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=partial(make_single_env, cfg, device, obs_rms, dtype),
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
    # Tune PyTorch intra-op threads — small MLPs on CPU benefit from fewer threads
    # (thread-launch overhead > benefit for small matrix sizes)
    num_threads = cfg.get("num_threads", 0)
    if num_threads > 0:
        torch.set_num_threads(num_threads)
        log.info("torch.set_num_threads(%d)", num_threads)
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
    obs_rms = initialize_obs_norm(cfg, device, dtype=network_dtype)

    # ── Reward normalization (VecNormalize-style) ──────────────────────
    ret_normalizer = None
    if cfg.env.get("normalize_reward", False):
        ret_normalizer = RewardNormalizer(device=device)
        log.info("Reward normalization enabled")

    num_groups = cfg.collector.get("num_groups", 1)
    envs_per_group = max(1, num_envs // num_groups)
    collector_envs = [
        make_parallel_env(cfg, device, num_envs=envs_per_group, dtype=network_dtype, obs_rms=obs_rms)
        for _ in range(num_groups)
    ]
    # Use first env only for shape logging
    env = collector_envs[0]
    log.info("Env: %s | num_groups=%d | envs_per_group=%d | total_envs=%d | obs=%s  act=%s",
             cfg.env.id, num_groups, envs_per_group, num_groups * envs_per_group,
             env.observation_spec["observation"].shape,
             env.action_spec.shape)

    eval_env = make_single_env(cfg, device, obs_rms=obs_rms, dtype=network_dtype)

    # ── Build PPO modules ──────────────────────────────────────────────
    # Set RPO alpha before building models (distribution class reads it)
    rpo_alpha = cfg.agent.get("rpo_alpha", 0.0)
    RPOTanhNormal.rpo_alpha = rpo_alpha
    if rpo_alpha > 0:
        log.info("RPO enabled with alpha=%.2f", rpo_alpha)

    models = make_ppo_models(env, cfg, device=device, network_device=network_device, dtype=network_dtype)
    actor = models["actor"]
    advantage_module = models["advantage"]
    loss_module = models["loss_module"]
    optimizer = models["optimizer"]
    scheduler = models["scheduler"]
    optim_params = models["optim_params"]
    
    # ── Data collector ─────────────────────────────────────────────────
    # ObservationNorm is in the env transforms → observations in tensordict
    # are already normalized. Use plain actor for collection.
    collector = MultiAsyncCollector(
        collector_envs,
        actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
    )

    # ── Checkpoint directory ───────────────────────────────────────────
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── V-trace setup ──────────────────────────────────────────────────
    use_vtrace = cfg.agent.get("use_vtrace", False)
    if use_vtrace:
        log.info("V-trace off-policy correction enabled for async collector")

    # ── Training loop ──────────────────────────────────────────────────
    total_frames = cfg.collector.total_frames
    pbar = tqdm(total=total_frames, desc="Training")

    global_step = 0

    for collect_iter, tensordict_data in enumerate(collector):
        batch_frames = tensordict_data.numel()
        global_step += batch_frames

        # Drop actor intermediate outputs — ClipPPOLoss recomputes them from
        # "observation"; keeping them just wastes memory and TensorDict ops.
        tensordict_data.exclude("loc", "scale")

        # ── Save raw reward for logging (before normalization) ─────────
        raw_mean_reward = tensordict_data["next", "reward"].mean().item()

        # ── Normalize rewards (VecNormalize-style) ─────────────────────
        if ret_normalizer is not None:
            ret_normalizer.normalize_batch(tensordict_data)

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
                advantage_module(td_seq)
                if tensordict_data.ndim == 1:
                    tensordict_data = td_seq.reshape(-1)

        # Compute advantages once before the epoch loop.
        # With only a few epochs the value net changes minimally between epochs,
        # so recomputing GAE every epoch is not worth the extra critic forward pass.
        if not use_vtrace:
            with torch.no_grad():
                advantage_module(tensordict_data)

        data_view = tensordict_data.reshape(-1)
        n_sub = cfg.collector.frames_per_batch // cfg.training.sub_batch_size

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
            policy_entropy = dist.base_dist.entropy().mean().item()

        scheduler.step()

        # ── Logging ────────────────────────────────────────────────────
        mean_reward = raw_mean_reward  # Use raw (non-normalized) reward
        max_step_count = tensordict_data["step_count"].max().item()
        lr_now = optimizer.param_groups[0]["lr"]

        avg_losses = {k: torch.stack(v).mean().item() for k, v in epoch_losses.items()}

        metrics = {
            "train/reward_mean": mean_reward,
            "train/step_count_max": max_step_count,
            "train/lr": lr_now,
            "train/global_step": global_step,
            "train/policy_entropy": policy_entropy,
            "train/epochs_done": epochs_done,
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
    for e in collector_envs:
        try:
            e.close()
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
