"""
Benchmark: NSFrozenLakeWrapper vs FastNSFrozenLakeWrapper step time.

Run: source .venv/bin/activate && python scripts/benchmark_frozenlake_wrapper.py
"""

import sys
sys.path.insert(0, "src")

import time
import numpy as np
import gymnasium as gym
from ns_gym.wrappers import NSFrozenLakeWrapper
from ns_gym.update_functions import DistributionDecrementUpdate
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler

from AAMAS_Comp.envs.wrappers import FastNSFrozenLakeWrapper

STEPS = 2000
WARMUP = 200


def benchmark_env(env, steps=STEPS, warmup=WARMUP, label=""):
    env.reset()
    timings = []
    done = True
    for i in range(steps + warmup):
        if done:
            env.reset()
        action = env.action_space.sample()
        t0 = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(action)
        dt = (time.perf_counter() - t0) * 1e6
        done = terminated or truncated
        if i >= warmup:
            timings.append(dt)

    arr = np.array(timings)
    print(f"  {label:45s}  mean={arr.mean():6.1f}µs  "
          f"p50={np.percentile(arr,50):6.1f}µs  "
          f"p95={np.percentile(arr,95):6.1f}µs  "
          f"max={arr.max():7.1f}µs")
    return arr


def make_standard(sched, k=0.005):
    base = gym.make("FrozenLake-v1", disable_env_checker=True)
    fn = DistributionDecrementUpdate(scheduler=sched, k=k)
    return NSFrozenLakeWrapper(base, tunable_params={"P": fn},
                               change_notification=False,
                               delta_change_notification=False,
                               initial_prob_dist=[1/3, 1/3, 1/3])


def make_fast(sched, k=0.005):
    base = gym.make("FrozenLake-v1", disable_env_checker=True)
    fn = DistributionDecrementUpdate(scheduler=sched, k=k)
    return FastNSFrozenLakeWrapper(base, tunable_params={"P": fn},
                                   initial_prob_dist=[1/3, 1/3, 1/3])


SCHEDULERS = [
    ("ContinuousScheduler       ", ContinuousScheduler),
    ("PeriodicScheduler(period=10)", lambda: PeriodicScheduler(period=10)),
    ("PeriodicScheduler(period=50)", lambda: PeriodicScheduler(period=50)),
]

print(f"{'='*85}")
print("Standard NSFrozenLakeWrapper (ns-gym)")
print(f"{'='*85}")
for label, sched_fn in SCHEDULERS:
    env = make_standard(sched_fn())
    benchmark_env(env, label=label)
    env.close()

print(f"\n{'='*85}")
print("FastNSFrozenLakeWrapper (numpy, no dict rebuild)")
print(f"{'='*85}")
for label, sched_fn in SCHEDULERS:
    env = make_fast(sched_fn())
    benchmark_env(env, label=label)
    env.close()

print(f"\n{'='*85}")
print("Baseline: plain gymnasium FrozenLake-v1")
print(f"{'='*85}")
base = gym.make("FrozenLake-v1", disable_env_checker=True)
base.reset()
timings = []
done = True
for i in range(STEPS + WARMUP):
    if done:
        base.reset()
    action = base.action_space.sample()
    t0 = time.perf_counter()
    _, _, terminated, truncated, _ = base.step(action)
    dt = (time.perf_counter() - t0) * 1e6
    done = terminated or truncated
    if i >= WARMUP:
        timings.append(dt)
arr = np.array(timings)
print(f"  {'plain FrozenLake-v1':45s}  mean={arr.mean():6.1f}µs  "
      f"p50={np.percentile(arr,50):6.1f}µs  "
      f"p95={np.percentile(arr,95):6.1f}µs  "
      f"max={arr.max():7.1f}µs")
base.close()
