#!/usr/bin/env python3
"""Evaluate the submitted Ant agent on a custom NS-Gym torso-mass schedule.

Default setup:
- Base env: Ant-v5
- Scheduler: DiscreteScheduler(event_list={500})
- Update: StepWiseUpdate(param_list=[0.7])
- Tunable parameter: torso_mass

This evaluator supports multiple torso-mass changes within a single episode by
pairing:
- `DiscreteScheduler(event_list={...})` for the change timesteps
- `StepWiseUpdate(param_list=[...])` for the ordered target masses
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from statistics import mean, pstdev
from typing import Any
import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm

from plot_reward_traces import save_reward_vs_timestep_plot


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ns_gym.schedulers import DiscreteScheduler # noqa: E402
from ns_gym.update_functions import StepWiseUpdate # noqa: E402
from ns_gym.wrappers import MujocoWrapper # noqa: E402
from result_bundle import write_result_bundle # noqa: E402
from submission import get_agent # noqa: E402


NOTIFICATION_LEVELS = {
"notify-none": (False, False),
"notify-change": (True, False),
"notify-full": (True, True),
}

RESULT_PREFIX = "ExampleNSAnt-v0"


def _parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(item) for item in values]


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return [float(item) for item in values]


def _resolve_schedule(args: argparse.Namespace) -> tuple[list[int], list[float]]:
    using_multi_steps = args.change_steps is not None or args.torso_masses is not None
    if using_multi_steps:
        if args.change_steps is None or args.torso_masses is None:
            raise ValueError("Pass both --change-steps and --torso-masses together.")
        change_steps = _parse_int_list(args.change_steps)
        torso_masses = _parse_float_list(args.torso_masses)
    else:
        change_steps = [args.change_step]
        torso_masses = [args.torso_mass]

    if len(change_steps) != len(torso_masses):
        raise ValueError("The number of change steps must match the number of torso masses.")
    if any(step < 0 for step in change_steps):
        raise ValueError("Change steps must be non-negative.")
    if any(step >= args.max_episode_steps for step in change_steps):
        raise ValueError("Each change step must be smaller than --max-episode-steps.")

    # StepWiseUpdate pops param_list in order, so sort by timestep before building
    # DiscreteScheduler(event_list=...) and the corresponding target-mass list.
    schedule = sorted(zip(change_steps, torso_masses), key=lambda item: item[0])
    sorted_steps = [step for step, _ in schedule]
    if len(set(sorted_steps)) != len(sorted_steps):
        raise ValueError("Change steps must be unique.")
    sorted_masses = [mass for _, mass in schedule]
    return sorted_steps, sorted_masses


def _format_schedule(change_steps: list[int], torso_masses: list[float]) -> str:
    return ", ".join(
        f"t={step}->{mass:g} kg" for step, mass in zip(change_steps, torso_masses)
    )


def make_ant_env(
    *,
    change_steps: list[int],
    torso_masses: list[float],
    change_notification: bool,
    delta_change_notification: bool,
    max_episode_steps: int,
) -> gym.Env:
    base_env = gym.make(
        "Ant-v5",
        max_episode_steps=max_episode_steps,
        disable_env_checker=True,
    )
    scheduler = DiscreteScheduler(event_list=set(change_steps))
    update_fn = StepWiseUpdate(scheduler=scheduler, param_list=list(torso_masses))

    return MujocoWrapper(
        base_env,
        {"torso_mass": update_fn},
        change_notification=change_notification,
        delta_change_notification=delta_change_notification,
    )


def run_episode(env, agent, seed: int) -> dict[str, Any]:
    obs, reset_info = env.reset(seed=seed)
    if hasattr(agent, "update_context"):
        agent.update_context(reset_info)

    initial_torso_mass = float(env._get_param_value("torso_mass"))
    episode_metrics = {
        "step_number": [],
        "rewards": [],
        "observations": [],
        "notification": [],
        "actions": [],
        "decision_time": [],
        "info": [],
        "env_change": [],
        "relative_time": [],
    }
    changes: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        pre_step_t = int(obs.get("relative_time", steps))
        action, decision_time = agent.validate_and_get_action(obs, env.action_space)
        obs, reward, terminated, truncated, info = env.step(action)

        if hasattr(agent, "update_context"):
            agent.update_context(info)

        steps += 1

        env_change = info.get("Ground Truth Env Change", {})
        episode_metrics["step_number"].append(pre_step_t)
        episode_metrics["rewards"].append(float(reward))
        episode_metrics["observations"].append(obs["state"])
        episode_metrics["notification"].append(obs["env_change"])
        episode_metrics["actions"].append(np.asarray(action).tolist())
        episode_metrics["decision_time"].append(float(decision_time))
        episode_metrics["info"].append(info)
        episode_metrics["env_change"].append(int(max(env_change.values())) if env_change else 0)
        episode_metrics["relative_time"].append(int(obs.get("relative_time", steps)))

        if int(env_change.get("torso_mass", 0)):
            delta_change = info.get("Ground Truth Delta Change", {})
            changes.append(
                {
                    "pre_step_t": pre_step_t,
                    "post_step_relative_time": int(obs.get("relative_time", pre_step_t + 1)),
                    "delta": float(delta_change.get("torso_mass", 0.0)),
                    "torso_mass_after": float(env._get_param_value("torso_mass")),
                }
            )

    episode_metrics["terminated"] = bool(terminated)
    episode_metrics["truncated"] = bool(truncated)
    episode_metrics["initial_torso_mass"] = initial_torso_mass
    episode_metrics["final_torso_mass"] = float(env._get_param_value("torso_mass"))
    episode_metrics["changes"] = changes
    return episode_metrics


def _episode_record(seed: int, episode_metrics: dict[str, Any]) -> dict[str, Any]:
    rewards = episode_metrics["rewards"]
    decision_times = episode_metrics["decision_time"]
    return {
        "seed": seed,
        "total_reward": float(sum(rewards)),
        "steps": len(rewards),
        "terminated": bool(episode_metrics["terminated"]),
        "truncated": bool(episode_metrics["truncated"]),
        "initial_torso_mass": float(episode_metrics["initial_torso_mass"]),
        "final_torso_mass": float(episode_metrics["final_torso_mass"]),
        "changes": episode_metrics["changes"],
        "reward_trace": rewards,
        "timestep_trace": episode_metrics["relative_time"],
        "mean_decision_time": float(mean(decision_times)) if decision_times else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
    description="Run submission.py's Ant-v5 agent on a multi-step torso-mass NS-Gym environment."
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--change-step", type=int, default=500)
    parser.add_argument("--torso-mass", type=float, default=0.7)
    parser.add_argument(
    "--change-steps",
    type=str,
    default=None,
    help="Comma-separated change steps, e.g. '250,500,750'. Overrides --change-step when provided.",
    )
    parser.add_argument(
    "--torso-masses",
    type=str,
    default=None,
    help="Comma-separated target torso masses, e.g. '0.7,1.0,0.5'. Overrides --torso-mass when provided.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument(
    "--save-dir-root",
    type=Path,
    default=REPO_ROOT / "results",
    )
    args = parser.parse_args()

    change_steps, torso_masses = _resolve_schedule(args)
    schedule_text = _format_schedule(change_steps, torso_masses)
    for notify_label, (change_notification, delta_change_notification) in NOTIFICATION_LEVELS.items():
        agent = get_agent("Ant-v5", notify_label)
        save_dir = args.save_dir_root / f"{RESULT_PREFIX}__{notify_label}"

        episodes = []
        results_table = {}
        start_time = perf_counter()
        desc = f"Ant {notify_label} [{schedule_text}]"
        pbar = tqdm(range(args.num_episodes), desc=desc, unit="episode")
        for i in pbar:
            seed = args.start_seed + i
            env = make_ant_env(
                change_steps=change_steps,
                torso_masses=torso_masses,
                change_notification=change_notification,
                delta_change_notification=delta_change_notification,
                max_episode_steps=args.max_episode_steps,
            )
            try:
                raw_episode = run_episode(env, agent, seed=seed)
            finally:
                env.close()

            results_table[str(seed)] = raw_episode
            episode = _episode_record(seed, raw_episode)
            episodes.append(episode)
            pbar.set_postfix(
                reward=f"{episode['total_reward']:.2f}",
                steps=episode["steps"],
            )
            pbar.write(
                f"Episode {i + 1}/{args.num_episodes} "
                f"(seed={episode['seed']}): reward={episode['total_reward']:.4f}, "
                f"steps={episode['steps']}"
            )

        metadata = {
            "name_prefix": save_dir.name,
            "start_seed": args.start_seed,
            "end_seed": args.start_seed + args.num_episodes - 1,
            "num_episodes": args.num_episodes,
            "change_notification": change_notification,
            "delta_change_notification": delta_change_notification,
            "total_time_seconds": perf_counter() - start_time,
            "timestamp": datetime.now().isoformat(),
            "env_id": "Ant-v5",
            "notify": notify_label,
            "change_steps": change_steps,
            "target_torso_masses": torso_masses,
            "schedule": [
                {"step": step, "torso_mass": mass}
                for step, mass in zip(change_steps, torso_masses)
            ],
        }
        if len(change_steps) == 1:
            metadata["change_step"] = change_steps[0]
            metadata["target_torso_mass"] = torso_masses[0]

        summary = write_result_bundle(
            save_dir=save_dir,
            name_prefix=save_dir.name,
            results_table=results_table,
            metadata=metadata,
        )
        plot_path = save_dir / "reward_vs_timestep.png"
        save_reward_vs_timestep_plot(
            episodes=episodes,
            output_path=plot_path,
            title=f"Ant reward vs timestep ({notify_label}, {schedule_text})",
            change_steps=change_steps,
        )

        print("\nEvaluation summary")
        print(f" Notify: {notify_label}")
        print(f" Episodes: {args.num_episodes}")
        print(f" Seeds: {args.start_seed}-{args.start_seed + args.num_episodes - 1}")
        print(f" Torso mass schedule: {schedule_text}")
        print(
            " Mean return: "
            f"{summary['aggregate']['mean_total_reward']:.4f} +/- "
            f"{summary['aggregate']['std_total_reward']:.4f}"
        )
        print(f" Mean steps: {summary['aggregate']['mean_episode_steps']:.1f}")
        print(f" Saved to: {save_dir}")
        print(f" Reward plot: {plot_path}")


if __name__ == "__main__":
    main()
