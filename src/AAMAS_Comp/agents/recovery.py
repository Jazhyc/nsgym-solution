"""Explicit learner recovery and non-finite diagnostics for interrupted runs."""

import json
from pathlib import Path

import torch


def validate_models(checkpoint):
    for key in ("actor", "critic"):
        if key not in checkpoint:
            raise ValueError(f"Recovery checkpoint is missing {key}")
        if any(not torch.isfinite(t).all() for t in checkpoint[key].state_dict().values()):
            raise ValueError(f"Recovery checkpoint has non-finite {key} weights")


def check_finite(tensors, *, stage, frames, directory, batch=None):
    bad = [name for name, value in tensors.items()
           if torch.is_tensor(value) and not torch.isfinite(value).all()]
    if not bad:
        return
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "nonfinite_failure.json").write_text(json.dumps(
        {"stage": stage, "global_step": frames, "nonfinite_keys": bad}, indent=2))
    if batch is not None:
        torch.save(batch.detach().cpu().clone(), folder / "nonfinite_batch.pt")
    raise FloatingPointError(f"Non-finite {stage} at frame {frames}: {bad}")


def learner_state(optimizer, scheduler, obs_rms, ret_normalizer, frames, iteration):
    return {
        "global_step": frames, "collect_iter": iteration,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "obs_rms": obs_rms.state_dict() if obs_rms is not None else None,
        "reward_rms": ret_normalizer.reward_rms.state_dict() if ret_normalizer is not None else None,
        # Worker simulator/RNG/buffer state is deliberately not claimed to be saved.
        "worker_state_saved": False,
    }


def restore_learner(checkpoint, actor, critic, optimizer, scheduler, completed_iterations):
    validate_models(checkpoint)
    actor.load_state_dict(checkpoint["actor"].state_dict())
    critic.load_state_dict(checkpoint["critic"].state_dict())
    state = checkpoint.get("training_state")
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    else:
        # Legacy checkpoints omit Adam moments. Keep the original LR clock,
        # rather than restarting its annealing from the initial learning rate.
        factor = scheduler.start_factor + (scheduler.end_factor - scheduler.start_factor) * min(completed_iterations, scheduler.total_iters) / scheduler.total_iters
        scheduler.last_epoch = completed_iterations
        scheduler._step_count = completed_iterations + 1
        scheduler._last_lr = [lr * factor for lr in scheduler.base_lrs]
        for group, lr in zip(optimizer.param_groups, scheduler._last_lr):
            group["lr"] = lr
