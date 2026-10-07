from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def build_summary(results_table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_episode = {}
    for seed_key, episode in results_table.items():
        rewards = episode.get("rewards", [])
        decision_times = episode.get("decision_time", [])
        env_changes = episode.get("env_change", [])
        per_episode[seed_key] = {
            "total_reward": float(np.sum(rewards)) if rewards else 0.0,
            "num_steps": len(rewards),
            "mean_decision_time": float(np.mean(decision_times)) if decision_times else 0.0,
            "std_decision_time": float(np.std(decision_times)) if decision_times else 0.0,
            "num_transition_fn_changes": int(np.sum(env_changes)) if env_changes else 0,
        }

    if not per_episode:
        aggregate = {
            "mean_total_reward": 0.0,
            "std_total_reward": 0.0,
            "iqm_total_reward": 0.0,
            "mean_episode_steps": 0.0,
            "mean_decision_time": 0.0,
            "std_decision_time": 0.0,
            "mean_transition_fn_changes": 0.0,
            "std_transition_fn_changes": 0.0,
        }
        return {"aggregate": aggregate, "per_episode": per_episode}

    all_rewards = np.array([item["total_reward"] for item in per_episode.values()], dtype=float)
    all_steps = np.array([item["num_steps"] for item in per_episode.values()], dtype=float)
    all_mean_dt = np.array([item["mean_decision_time"] for item in per_episode.values()], dtype=float)
    all_tf_changes = np.array(
        [item["num_transition_fn_changes"] for item in per_episode.values()],
        dtype=float,
    )

    q1, q3 = np.percentile(all_rewards, [25, 75])
    iqm_rewards = all_rewards[(all_rewards >= q1) & (all_rewards <= q3)]
    iqm_total_reward = float(np.mean(iqm_rewards)) if len(iqm_rewards) > 0 else float(np.mean(all_rewards))

    aggregate = {
        "mean_total_reward": float(np.mean(all_rewards)),
        "std_total_reward": float(np.std(all_rewards)),
        "iqm_total_reward": iqm_total_reward,
        "mean_episode_steps": float(np.mean(all_steps)),
        "mean_decision_time": float(np.mean(all_mean_dt)),
        "std_decision_time": float(np.std(all_mean_dt)),
        "mean_transition_fn_changes": float(np.mean(all_tf_changes)),
        "std_transition_fn_changes": float(np.std(all_tf_changes)),
    }
    return {"aggregate": aggregate, "per_episode": per_episode}


def write_result_bundle(
    *,
    save_dir: Path,
    name_prefix: str,
    results_table: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    save_dir.mkdir(parents=True, exist_ok=True)

    legacy_episodes_path = save_dir / "episodes.json"
    if legacy_episodes_path.exists():
        legacy_episodes_path.unlink()

    with (save_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=json_default)

    json_path = save_dir / f"{name_prefix}.json"
    zip_path = save_dir / f"{name_prefix}.zip"

    with json_path.open("w") as f:
        json.dump(results_table, f, default=json_default)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, json_path.name)

    json_path.unlink()

    summary_data = build_summary(results_table)
    with (save_dir / "summary.json").open("w") as f:
        json.dump(summary_data, f, indent=2, default=json_default)

    return summary_data
