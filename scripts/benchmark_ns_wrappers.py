"""Benchmark stock ns-gym wrappers against the fast training wrappers.

Measures mean per-step latency with random actions for:
  - CartPole-v1:   NSClassicControlWrapper vs FastNSClassicControlWrapper vs plain env
  - FrozenLake-v1: NSFrozenLakeWrapper vs FastNSFrozenLakeWrapper vs plain env

The scheduler is ContinuousScheduler (parameter update fires every step), the
worst case for wrapper overhead. Source of the wrapper latency table in the
report. See scripts/benchmark_frozenlake_wrapper.py for a per-scheduler
breakdown of the FrozenLake case.

Usage:
    python scripts/benchmark_ns_wrappers.py [--steps 2000] [--warmup 200]
        [--output results/ns_wrapper_bench.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import gymnasium as gym

from ns_gym.wrappers import NSClassicControlWrapper, NSFrozenLakeWrapper
from ns_gym.update_functions import IncrementUpdate, DistributionDecrementUpdate
from ns_gym.schedulers import ContinuousScheduler

from AAMAS_Comp.envs.fast_ns_wrapper import FastNSClassicControlWrapper
from AAMAS_Comp.envs.wrappers import FastNSFrozenLakeWrapper


def bench(env, steps: int, warmup: int) -> dict:
    env.reset()
    done = True
    times = []
    for i in range(steps + warmup):
        if done:
            env.reset()
        action = env.action_space.sample()
        t0 = time.perf_counter()
        _, _, term, trunc, _ = env.step(action)
        dt = time.perf_counter() - t0
        done = term or trunc
        if i >= warmup:
            times.append(dt)
    env.close()
    arr = np.asarray(times) * 1e6  # µs
    return {
        "mean_us": float(arr.mean()),
        "p50_us": float(np.percentile(arr, 50)),
        "p95_us": float(np.percentile(arr, 95)),
    }


def cartpole_envs():
    def stock():
        return NSClassicControlWrapper(
            gym.make("CartPole-v1", disable_env_checker=True),
            tunable_params={"masspole": IncrementUpdate(scheduler=ContinuousScheduler(), k=0.001)},
            change_notification=False,
            delta_change_notification=False,
        )

    def fast():
        return FastNSClassicControlWrapper(
            gym.make("CartPole-v1", disable_env_checker=True),
            {"masspole": IncrementUpdate(scheduler=ContinuousScheduler(), k=0.001)},
        )

    def plain():
        return gym.make("CartPole-v1", disable_env_checker=True)

    return {"stock": stock, "fast": fast, "plain": plain}


def frozenlake_envs():
    def stock():
        return NSFrozenLakeWrapper(
            gym.make("FrozenLake-v1", disable_env_checker=True),
            tunable_params={"P": DistributionDecrementUpdate(scheduler=ContinuousScheduler(), k=0.005)},
            change_notification=False,
            delta_change_notification=False,
            initial_prob_dist=[1 / 3, 1 / 3, 1 / 3],
        )

    def fast():
        return FastNSFrozenLakeWrapper(
            gym.make("FrozenLake-v1", disable_env_checker=True),
            tunable_params={"P": DistributionDecrementUpdate(scheduler=ContinuousScheduler(), k=0.005)},
            initial_prob_dist=[1 / 3, 1 / 3, 1 / 3],
        )

    def plain():
        return gym.make("FrozenLake-v1", disable_env_checker=True)

    return {"stock": stock, "fast": fast, "plain": plain}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--output", default="results/ns_wrapper_bench.json")
    args = parser.parse_args()

    out = {
        "metadata": {
            "steps": args.steps,
            "warmup": args.warmup,
            "scheduler": "ContinuousScheduler (fires every step)",
        },
        "results": {},
    }
    for name, factories in [("cartpole", cartpole_envs()), ("frozenlake", frozenlake_envs())]:
        out["results"][name] = {}
        for variant, make_env in factories.items():
            res = bench(make_env(), args.steps, args.warmup)
            out["results"][name][variant] = res
            print(f"  {name:<11} {variant:<6} mean={res['mean_us']:7.1f}µs"
                  f"  p50={res['p50_us']:7.1f}µs  p95={res['p95_us']:7.1f}µs")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
