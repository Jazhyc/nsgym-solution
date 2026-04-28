#!/usr/bin/env python3
"""Evaluate the submitted CartPole agent on a multi-step NS-Gym masspole schedule."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from statistics import mean, pstdev
from typing import Any
import gymnasium as gym
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from AAMAS_Comp import base_agent  # noqa: E402
from ns_gym.schedulers import DiscreteScheduler  # noqa: E402
from ns_gym.update_functions import StepWiseUpdate  # noqa: E402
from ns_gym.wrappers import NSClassicControlWrapper  # noqa: E402
from plot_reward_traces import (  # noqa: E402
    save_reward_vs_timestep_plot,
    save_zoomed_reward_vs_timestep_plot,
)
from result_bundle import write_result_bundle  # noqa: E402
from submission import get_agent  # noqa: E402


NOTIFICATION_LEVELS = {
    "notify-none": (False, False),
    "notify-change": (True, False),
    "notify-full": (True, True),
}

RESULT_PREFIX = "ExampleNSCartPole-v0"


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
    using_multi_steps = args.change_steps is not None or args.masspoles is not None
    if using_multi_steps:
        if args.change_steps is None or args.masspoles is None:
            raise ValueError("Pass both --change-steps and --masspoles together.")
        change_steps = _parse_int_list(args.change_steps)
        masspoles = _parse_float_list(args.masspoles)
    else:
        change_steps = [args.change_step]
        masspoles = [args.masspole]

    if len(change_steps) != len(masspoles):
        raise ValueError("The number of change steps must match the number of masspole values.")
    if any(step < 0 for step in change_steps):
        raise ValueError("Change steps must be non-negative.")
    if any(step >= args.max_episode_steps for step in change_steps):
        raise ValueError("Each change step must be smaller than --max-episode-steps.")

    schedule = sorted(zip(change_steps, masspoles), key=lambda item: item[0])
    sorted_steps = [step for step, _ in schedule]
    if len(set(sorted_steps)) != len(sorted_steps):
        raise ValueError("Change steps must be unique.")
    sorted_masspoles = [masspole for _, masspole in schedule]
    if any(masspole <= 0 for masspole in sorted_masspoles):
        raise ValueError("CartPole masspole values must be greater than zero.")
    return sorted_steps, sorted_masspoles


def _format_schedule(change_steps: list[int], masspoles: list[float]) -> str:
    return ", ".join(
        f"t={step}->{masspole:g}" for step, masspole in zip(change_steps, masspoles)
    )


def make_env(
    *,
    change_notification: bool = False,
    delta_change_notification: bool = False,
    masspoles: list[float] | None = None,
    change_steps: list[int] | None = None,
    max_episode_steps: int = 500,
) -> gym.Env:
    base_env = gym.make(
        "CartPole-v1",
        max_episode_steps=max_episode_steps,
        disable_env_checker=True,
    )
    change_steps = change_steps or [1]
    masspoles = masspoles or [0.2]
    scheduler = DiscreteScheduler(set(change_steps))
    update_fn = StepWiseUpdate(scheduler, list(masspoles))

    ns_env = NSClassicControlWrapper(
        base_env,
        {"masspole": update_fn},
        change_notification=change_notification,
        delta_change_notification=delta_change_notification,
    )
    return ns_env


def run_episode(env, agent, seed: int) -> dict[str, Any]:
    obs, reset_info = env.reset(seed=seed)
    if hasattr(agent, "update_context"):
        agent.update_context(reset_info)

    wrapper = env
    initial_masspole = float(wrapper.unwrapped.masspole)
    episode_metrics = {
        "step_number": [],
        "rewards": [],
        "base_rewards": [],
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
    is_model_based_agent = isinstance(agent, base_agent.ModelBasedAgent)

    while not (terminated or truncated):
        pre_step_t = int(obs.get("relative_time", steps))
        if is_model_based_agent:
            action, decision_time = agent.validate_and_get_action(obs, env.get_planning_env())
        else:
            action, decision_time = agent.validate_and_get_action(obs, env.action_space)
        obs, base_reward, terminated, truncated, info = env.step(action)

        if hasattr(agent, "update_context"):
            agent.update_context(info)

        reward = float(base_reward)

        steps += 1

        env_change = info.get("Ground Truth Env Change", {})
        episode_metrics["step_number"].append(pre_step_t)
        episode_metrics["rewards"].append(float(reward))
        episode_metrics["base_rewards"].append(float(base_reward))
        episode_metrics["observations"].append(obs["state"])
        episode_metrics["notification"].append(obs["env_change"])
        episode_metrics["actions"].append(int(action))
        episode_metrics["decision_time"].append(float(decision_time))
        episode_metrics["info"].append(info)
        episode_metrics["env_change"].append(int(max(env_change.values())) if env_change else 0)
        episode_metrics["relative_time"].append(int(obs.get("relative_time", steps)))

        if int(env_change.get("masspole", 0)):
            delta_change = info.get("Ground Truth Delta Change", {})
            changes.append(
                {
                    "pre_step_t": pre_step_t,
                    "post_step_relative_time": int(obs.get("relative_time", pre_step_t + 1)),
                    "delta": float(delta_change.get("masspole", 0.0)),
                    "masspole_after": float(wrapper.unwrapped.masspole),
                }
            )

    episode_metrics["terminated"] = bool(terminated)
    episode_metrics["truncated"] = bool(truncated)
    episode_metrics["initial_masspole"] = initial_masspole
    episode_metrics["final_masspole"] = float(wrapper.unwrapped.masspole)
    episode_metrics["changes"] = changes
    return episode_metrics


def _episode_record(seed: int, episode_metrics: dict[str, Any]) -> dict[str, Any]:
    rewards = episode_metrics["rewards"]
    base_rewards = episode_metrics["base_rewards"]
    decision_times = episode_metrics["decision_time"]
    return {
        "seed": seed,
        "total_reward": float(sum(rewards)),
        "steps": len(rewards),
        "terminated": bool(episode_metrics["terminated"]),
        "truncated": bool(episode_metrics["truncated"]),
        "base_total_reward": float(sum(base_rewards)),
        "initial_masspole": float(episode_metrics["initial_masspole"]),
        "final_masspole": float(episode_metrics["final_masspole"]),
        "changes": episode_metrics["changes"],
        "reward_trace": rewards,
        "base_reward_trace": base_rewards,
        "timestep_trace": episode_metrics["relative_time"],
        "mean_decision_time": float(mean(decision_times)) if decision_times else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run submission.py's CartPole-v1 agent on a multi-step masspole NS-Gym environment."
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--change-step", type=int, default=1)
    parser.add_argument("--masspole", type=float, default=0.2)
    parser.add_argument(
        "--change-steps",
        type=str,
        default=None,
        help="Comma-separated change steps, e.g. '50,100,150'. Overrides --change-step when provided.",
    )
    parser.add_argument(
        "--masspoles",
        type=str,
        default=None,
        help="Comma-separated target masspole values, e.g. '0.2,0.4,0.1'. Overrides --masspole when provided.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument(
        "--save-dir-root",
        type=Path,
        default=REPO_ROOT / "results",
    )
    args = parser.parse_args()

    change_steps, masspoles = _resolve_schedule(args)
    schedule_text = _format_schedule(change_steps, masspoles)
    for notify_label, (change_notification, delta_change_notification) in NOTIFICATION_LEVELS.items():
        agent = get_agent("CartPole-v1", notify_label)
        save_dir = args.save_dir_root / f"{RESULT_PREFIX}__{notify_label}"

        episodes = []
        results_table = {}
        start_time = perf_counter()
        desc = f"CartPole {notify_label} [{schedule_text}]"
        pbar = tqdm(range(args.num_episodes), desc=desc, unit="episode")
        for i in pbar:
            seed = args.start_seed + i
            env = make_env(
                change_notification=change_notification,
                delta_change_notification=delta_change_notification,
                masspoles=masspoles,
                change_steps=change_steps,
                max_episode_steps=args.max_episode_steps,
            )
            try:
                raw_episode = run_episode(env, agent, seed=seed)
            finally:
                env.close()

            results_table[str(seed)] = raw_episode
            episode = _episode_record(seed, raw_episode)
            episodes.append(episode)
            pbar.set_postfix(reward=f"{episode['total_reward']:.2f}", steps=episode["steps"])
            pbar.write(
                f"Episode {i + 1}/{args.num_episodes} "
                f"(seed={episode['seed']}): reward={episode['total_reward']:.4f}, "
                f"steps={episode['steps']}"
            )

        base_returns = [ep["base_total_reward"] for ep in episodes]
        metadata = {
            "name_prefix": save_dir.name,
            "start_seed": args.start_seed,
            "end_seed": args.start_seed + args.num_episodes - 1,
            "num_episodes": args.num_episodes,
            "change_notification": change_notification,
            "delta_change_notification": delta_change_notification,
            "total_time_seconds": perf_counter() - start_time,
            "timestamp": datetime.now().isoformat(),
            "env_id": "CartPole-v1",
            "notify": notify_label,
            "reward_mode": "env_reward",
            "change_steps": change_steps,
            "target_masspoles": masspoles,
            "schedule": [
                {"step": step, "masspole": masspole}
                for step, masspole in zip(change_steps, masspoles)
            ],
            "mean_base_total_reward": float(mean(base_returns)) if base_returns else 0.0,
            "std_base_total_reward": float(pstdev(base_returns)) if len(base_returns) > 1 else 0.0,
        }
        if len(change_steps) == 1:
            metadata["change_step"] = change_steps[0]
            metadata["target_masspole"] = masspoles[0]

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
            title=f"CartPole reward vs timestep ({notify_label}, {schedule_text})",
            change_steps=change_steps,
        )
        zoom_plot_path = save_dir / "reward_vs_timestep_above_0p996.png"
        save_zoomed_reward_vs_timestep_plot(
            episodes=episodes,
            output_path=zoom_plot_path,
            title=f"CartPole reward vs timestep above 0.996 ({notify_label}, {schedule_text})",
            change_steps=change_steps,
            y_min=0.996,
        )

        print("\nEvaluation summary")
        print(f" Notify: {notify_label}")
        print(f" Episodes: {args.num_episodes}")
        print(f" Seeds: {args.start_seed}-{args.start_seed + args.num_episodes - 1}")
        print(f" Masspole schedule: {schedule_text}")
        print(
            " Mean return: "
            f"{summary['aggregate']['mean_total_reward']:.4f} +/- "
            f"{summary['aggregate']['std_total_reward']:.4f}"
        )
        print(f" Mean steps: {summary['aggregate']['mean_episode_steps']:.1f}")
        print(f" Saved to: {save_dir}")
        print(f" Reward plot: {plot_path}")
        print(f" Zoomed reward plot: {zoom_plot_path}")


if __name__ == "__main__":
    main()
