"""Profile the evaluation loop per environment.

Usage:
    python scripts/profile_evaluator.py [--episodes 5] [--steps 200]

Reports per-step timing breakdown for:
  - obs preparation (_prepare_obs + _normalise)
  - actor forward pass (_sample_action internals)
  - env.step()
  - update_context()
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import gymnasium as gym
import AAMAS_Comp  # noqa: F401

from submission import get_agent

ENVIRONMENTS = {
    "ExampleNSFrozenLake-v0": "FrozenLake-v1",
    "ExampleNSCartPole-v0": "CartPole-v1",
    "ExampleNSAnt-v0": "Ant-v5",
}

NOTIFY = ("notify-none", True, False)  # label, change_notification, delta_change_notification


def _time(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def profile_agent_step(agent, obs, env):
    """Break down one agent decision + env step into sub-timings."""
    timings = {}

    # --- obs prep + normalise ---
    raw_state = obs["state"] if isinstance(obs, dict) else obs
    _, timings["obs_prep"] = _time(agent._prepare_obs, raw_state)
    flat_obs = agent._prepare_obs(raw_state)
    _, timings["normalise"] = _time(agent._normalise, flat_obs)
    s = agent._normalise(flat_obs)

    # --- net forward: measure whichever backend _sample_action will use ---
    if getattr(agent, "_np_layers", None) is not None:
        x = s.numpy()
        t0 = time.perf_counter()
        for W, b, act in agent._np_layers:
            x = x @ W + b
            if act is not None:
                x = act(x)
        timings["raw_net_forward"] = time.perf_counter() - t0
    elif getattr(agent, "_ort_session", None) is not None:
        obs_np = s.unsqueeze(0).numpy()
        in_name = agent._ort_session.get_inputs()[0].name
        t0 = time.perf_counter()
        agent._ort_session.run(None, {in_name: obs_np})
        timings["raw_net_forward"] = time.perf_counter() - t0
    else:
        obs_t = s.unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = agent._raw_net(obs_t)
        timings["raw_net_forward"] = time.perf_counter() - t0

    # --- full _sample_action (includes raw_net + sample overhead) ---
    _, timings["sample_action"] = _time(agent._sample_action, s)
    action = agent._sample_action(s)

    # --- env.step ---
    _, timings["env_step"] = _time(env.step, action)
    next_obs, reward, done, trunc, info = env.step(action)

    # --- update_context ---
    if hasattr(agent, "update_context"):
        _, timings["update_context"] = _time(agent.update_context, info)

    timings["total_actor"] = timings["obs_prep"] + timings["normalise"] + timings["sample_action"]
    return timings, next_obs, done or trunc


def run_profile(env_id, base_env_id, notify_label, change_notification,
                delta_change_notification, num_steps, seed):
    agent = get_agent(base_env_id, notify_label)
    env = gym.make(
        env_id,
        change_notification=change_notification,
        delta_change_notification=delta_change_notification,
        disable_env_checker=True,
        order_enforce=False,
    )

    obs, _ = env.reset(seed=seed)
    accumulated = defaultdict(list)
    steps_done = 0

    while steps_done < num_steps:
        timings, obs, terminal = profile_agent_step(agent, obs, env)
        for k, v in timings.items():
            accumulated[k].append(v)
        steps_done += 1
        if terminal:
            obs, _ = env.reset()

    env.close()

    print(f"\n{'='*60}")
    print(f"  {env_id} | {notify_label}  ({steps_done} steps)")
    print(f"{'='*60}")
    keys = ["obs_prep", "normalise", "raw_net_forward", "sample_action",
            "total_actor", "env_step", "update_context"]
    for k in keys:
        if k not in accumulated:
            continue
        vals = np.array(accumulated[k]) * 1e3  # ms
        pct = (np.mean(accumulated[k]) /
               (np.mean(accumulated["total_actor"]) + np.mean(accumulated["env_step"]))) * 100
        print(f"  {k:<22}  mean={np.mean(vals):.4f}ms  p95={np.percentile(vals,95):.4f}ms  ({pct:.1f}%)")

    total = (np.mean(accumulated["total_actor"]) + np.mean(accumulated["env_step"])) * 1e3
    print(f"\n  Est. steps/sec (actor+env): {1000/total:.0f} sps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env", choices=["frozenlake", "cartpole", "ant", "all"],
                        default="all")
    args = parser.parse_args()

    targets = {
        "frozenlake": ("ExampleNSFrozenLake-v0", "FrozenLake-v1"),
        "cartpole":   ("ExampleNSCartPole-v0",   "CartPole-v1"),
        "ant":        ("ExampleNSAnt-v0",         "Ant-v5"),
    }
    if args.env != "all":
        targets = {args.env: targets[args.env]}

    for name, (env_id, base_env_id) in targets.items():
        run_profile(
            env_id=env_id,
            base_env_id=base_env_id,
            notify_label="notify-none",
            change_notification=False,
            delta_change_notification=False,
            num_steps=args.steps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
