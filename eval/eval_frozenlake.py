#!/usr/bin/env python3
"""Evaluate the submitted FrozenLake agent on a multi-step NS-Gym slip schedule."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from AAMAS_Comp import base_agent  # noqa: E402
from ns_gym.schedulers import DiscreteScheduler  # noqa: E402
from ns_gym.update_functions import DistributionStepWiseUpdate  # noqa: E402
from ns_gym.wrappers import NSFrozenLakeWrapper  # noqa: E402
from plot_reward_traces import save_reward_vs_timestep_plot  # noqa: E402
from result_bundle import write_result_bundle  # noqa: E402
from submission import get_agent  # noqa: E402


NOTIFICATION_LEVELS = {
    "notify-none": (False, False),
    "notify-change": (True, False),
    "notify-full": (True, True),
}

RESULT_PREFIX = "ExampleNSFrozenLake-v0"


def get_ns_wrapper(env):
    curr = env
    while hasattr(curr, "env"):
        if isinstance(curr, NSFrozenLakeWrapper):
            return curr
        curr = curr.env
    return curr if isinstance(curr, NSFrozenLakeWrapper) else None


def _parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(item) for item in values]


def _parse_prob_schedule(raw: str) -> list[list[float]]:
    schedule_parts = [part.strip() for part in raw.split(";") if part.strip()]
    if not schedule_parts:
        raise ValueError("Expected at least one probability triple.")

    parsed: list[list[float]] = []
    for part in schedule_parts:
        probs = [float(item.strip()) for item in part.split(",") if item.strip()]
        if len(probs) != 3:
            raise ValueError("Each FrozenLake slip distribution must contain exactly 3 probabilities.")
        if any(prob < 0 for prob in probs):
            raise ValueError("Slip probabilities must be non-negative.")
        if not np.isclose(sum(probs), 1.0):
            raise ValueError("Each FrozenLake slip distribution must sum to 1.")
        parsed.append(probs)
    return parsed


def _resolve_schedule(args: argparse.Namespace) -> tuple[list[int], list[list[float]]]:
    using_multi_steps = args.change_steps is not None or args.slip_probs_schedule is not None
    if using_multi_steps:
        if args.change_steps is None or args.slip_probs_schedule is None:
            raise ValueError("Pass both --change-steps and --slip-probs-schedule together.")
        change_steps = _parse_int_list(args.change_steps)
        slip_prob_schedule = _parse_prob_schedule(args.slip_probs_schedule)
    else:
        change_steps = [args.change_step]
        slip_prob_schedule = [args.slip_probs]

    if len(change_steps) != len(slip_prob_schedule):
        raise ValueError("The number of change steps must match the number of slip-probability updates.")
    if any(step < 0 for step in change_steps):
        raise ValueError("Change steps must be non-negative.")
    if any(step >= args.max_episode_steps for step in change_steps):
        raise ValueError("Each change step must be smaller than --max-episode-steps.")

    schedule = sorted(zip(change_steps, slip_prob_schedule), key=lambda item: item[0])
    sorted_steps = [step for step, _ in schedule]
    if len(set(sorted_steps)) != len(sorted_steps):
        raise ValueError("Change steps must be unique.")
    sorted_probs = [probs for _, probs in schedule]
    return sorted_steps, sorted_probs


def _format_schedule(change_steps: list[int], slip_prob_schedule: list[list[float]]) -> str:
    return ", ".join(
        f"t={step}->{[round(prob, 4) for prob in probs]}"
        for step, probs in zip(change_steps, slip_prob_schedule)
    )


def make_env(
    *,
    change_notification: bool = False,
    delta_change_notification: bool = False,
    slip_prob_schedule: list[list[float]] | None = None,
    change_steps: list[int] | None = None,
    max_episode_steps: int = 100,
) -> gym.Env:
    initial_prob_dist = [1.0, 0.0, 0.0]
    change_steps = change_steps or [1]
    slip_prob_schedule = slip_prob_schedule or [[0.8, 0.1, 0.1]]

    base_env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        disable_env_checker=True,
        max_episode_steps=max_episode_steps,
    )
    scheduler = DiscreteScheduler(set(change_steps))
    update_fn = DistributionStepWiseUpdate(scheduler, [list(probs) for probs in slip_prob_schedule])

    ns_env = NSFrozenLakeWrapper(
        base_env,
        tunable_params={"P": update_fn},
        change_notification=change_notification,
        delta_change_notification=delta_change_notification,
        initial_prob_dist=initial_prob_dist,
    )
    return ns_env


def run_episode(env, agent, seed: int) -> dict[str, Any]:
    obs, reset_info = env.reset(seed=seed)
    if hasattr(agent, "update_context"):
        agent.update_context(reset_info)

    wrapper = get_ns_wrapper(env)
    initial_transition_prob = list(wrapper.transition_prob)
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

        if int(env_change.get("P", 0)):
            delta_change = info.get("Ground Truth Delta Change", {})
            changes.append(
                {
                    "pre_step_t": pre_step_t,
                    "post_step_relative_time": int(obs.get("relative_time", pre_step_t + 1)),
                    "delta": float(delta_change.get("P", 0.0)),
                    "transition_prob_after": list(wrapper.transition_prob),
                }
            )

    episode_metrics["terminated"] = bool(terminated)
    episode_metrics["truncated"] = bool(truncated)
    episode_metrics["initial_transition_prob"] = initial_transition_prob
    episode_metrics["final_transition_prob"] = list(wrapper.transition_prob)
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
        "initial_transition_prob": episode_metrics["initial_transition_prob"],
        "final_transition_prob": episode_metrics["final_transition_prob"],
        "changes": episode_metrics["changes"],
        "reward_trace": rewards,
        "base_reward_trace": base_rewards,
        "timestep_trace": episode_metrics["relative_time"],
        "mean_decision_time": float(mean(decision_times)) if decision_times else 0.0,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run submission.py's FrozenLake-v1 agent on a multi-step slip-probability NS-Gym environment."
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--change-step", type=int, default=1)
    parser.add_argument(
        "--slip-probs",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("P_INTENDED", "P_RIGHT", "P_LEFT"),
        help="Single-step slip distribution. Used when --change-steps/--slip-probs-schedule are omitted.",
    )
    parser.add_argument(
        "--change-steps",
        type=str,
        default=None,
        help="Comma-separated change steps, e.g. '1,5,10'.",
    )
    parser.add_argument(
        "--slip-probs-schedule",
        type=str,
        default=None,
        help="Semicolon-separated probability triples, e.g. '0.8,0.1,0.1;0.6,0.2,0.2'.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument(
        "--save-dir-root",
        type=Path,
        default=REPO_ROOT / "results",
    )
    args = parser.parse_args()

    change_steps, slip_prob_schedule = _resolve_schedule(args)
    schedule_text = _format_schedule(change_steps, slip_prob_schedule)
    for notify_label, (change_notification, delta_change_notification) in NOTIFICATION_LEVELS.items():
        agent = get_agent("FrozenLake-v1", notify_label)
        save_dir = args.save_dir_root / f"{RESULT_PREFIX}__{notify_label}"

        episodes = []
        results_table = {}
        start_time = perf_counter()
        desc = f"FrozenLake {notify_label} [{schedule_text}]"
        pbar = tqdm(range(args.num_episodes), desc=desc, unit="episode")
        for i in pbar:
            seed = args.start_seed + i
            env = make_env(
                change_notification=change_notification,
                delta_change_notification=delta_change_notification,
                slip_prob_schedule=slip_prob_schedule,
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
            "env_id": "FrozenLake-v1",
            "notify": notify_label,
            "reward_mode": "env_reward",
            "change_steps": change_steps,
            "target_slip_prob_schedule": slip_prob_schedule,
            "schedule": [
                {"step": step, "slip_probs": probs}
                for step, probs in zip(change_steps, slip_prob_schedule)
            ],
            "mean_base_total_reward": float(mean(base_returns)) if base_returns else 0.0,
            "std_base_total_reward": float(pstdev(base_returns)) if len(base_returns) > 1 else 0.0,
        }
        if len(change_steps) == 1:
            metadata["change_step"] = change_steps[0]
            metadata["target_slip_probs"] = slip_prob_schedule[0]

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
            title=f"FrozenLake reward vs timestep ({notify_label}, {schedule_text})",
            change_steps=change_steps,
        )

        print("\nEvaluation summary")
        print(f" Notify: {notify_label}")
        print(f" Episodes: {args.num_episodes}")
        print(f" Seeds: {args.start_seed}-{args.start_seed + args.num_episodes - 1}")
        print(f" Slip schedule: {schedule_text}")
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
