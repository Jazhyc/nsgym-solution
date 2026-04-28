#!/usr/bin/env python3
"""Generic reward plotting utilities for evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _plot_reward_traces(
    *,
    episodes: list[dict[str, Any]],
    output_path: Path,
    title: str,
    change_steps: list[int],
    y_min: float | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_len = max(len(episode.get("reward_trace", [])) for episode in episodes)
    if max_len == 0:
        raise ValueError("Episodes do not contain any reward_trace data.")

    rewards = np.full((len(episodes), max_len), np.nan, dtype=np.float32)
    for row, episode in enumerate(episodes):
        reward_trace = np.asarray(episode.get("reward_trace", []), dtype=np.float32)
        rewards[row, : len(reward_trace)] = reward_trace

    mean_reward = np.nanmean(rewards, axis=0)
    timesteps = np.arange(1, max_len + 1, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(11, 6))
    for episode in episodes:
        reward_trace = np.asarray(episode.get("reward_trace", []), dtype=np.float32)
        episode_timesteps = np.arange(1, len(reward_trace) + 1, dtype=np.int32)
        ax.plot(episode_timesteps, reward_trace, alpha=0.20, linewidth=1.0, color="C0")

    ax.plot(timesteps, mean_reward, linewidth=2.5, color="C1", label="Mean reward")

    for idx, change_step in enumerate(change_steps):
        ax.axvline(
            change_step,
            color="C3",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="Scheduled change" if idx == 0 else None,
        )

    if y_min is not None:
        finite_rewards = rewards[np.isfinite(rewards)]
        upper = float(np.max(finite_rewards)) if finite_rewards.size else y_min
        if upper <= y_min:
            upper = y_min + 1e-3
        pad = max((upper - y_min) * 0.05, 1e-4)
        ax.set_ylim(bottom=y_min, top=upper + pad)

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_reward_vs_timestep_plot(
    *,
    episodes: list[dict[str, Any]],
    output_path: str | Path,
    title: str,
    change_steps: list[int],
) -> Path:
    if not episodes:
        raise ValueError("No episodes were provided for plotting.")

    output_path = Path(output_path)
    return _plot_reward_traces(
        episodes=episodes,
        output_path=output_path,
        title=title,
        change_steps=change_steps,
    )


def save_zoomed_reward_vs_timestep_plot(
    *,
    episodes: list[dict[str, Any]],
    output_path: str | Path,
    title: str,
    change_steps: list[int],
    y_min: float,
) -> Path:
    if not episodes:
        raise ValueError("No episodes were provided for plotting.")

    output_path = Path(output_path)
    return _plot_reward_traces(
        episodes=episodes,
        output_path=output_path,
        title=title,
        change_steps=change_steps,
        y_min=y_min,
    )
