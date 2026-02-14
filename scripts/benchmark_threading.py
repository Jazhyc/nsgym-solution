"""Benchmark: subprocess (ParallelEnv) vs thread pool for Ant-v5.

Confirms whether MuJoCo's GIL release makes threading competitive.
Run before committing to a full TorchRL threading integration.

Usage:
    python scripts/benchmark_threading.py
"""

import time
from concurrent.futures import ThreadPoolExecutor

import gymnasium as gym
import numpy as np

ENV_ID = "Ant-v5"
NUM_ENVS = 12
STEPS = 2000


# ── Helpers ────────────────────────────────────────────────────────────────

def make_env():
    env = gym.make(ENV_ID)
    env.reset()
    return env


def step_env(env_action):
    env, action = env_action
    obs, r, term, trunc, info = env.step(action)
    if term or trunc:
        obs, _ = env.reset()
    return obs, r


# ── Single env baseline ────────────────────────────────────────────────────

def bench_single(steps=STEPS):
    env = make_env()
    action = env.action_space.sample()
    t = time.perf_counter()
    for _ in range(steps):
        obs, r, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset()
    elapsed = time.perf_counter() - t
    env.close()
    return steps / elapsed


# ── Thread pool ────────────────────────────────────────────────────────────

def bench_threads(num_envs=NUM_ENVS, steps=STEPS):
    import os
    # Disable MuJoCo's internal OpenMP threading per-worker to prevent
    # thread pool contention when multiple instances run in the same process.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    envs = [make_env() for _ in range(num_envs)]
    actions = [e.action_space.sample() for e in envs]

    with ThreadPoolExecutor(max_workers=num_envs) as executor:
        t = time.perf_counter()
        for _ in range(steps):
            futures = [executor.submit(step_env, (env, action))
                       for env, action in zip(envs, actions)]
            results = [f.result() for f in futures]
        elapsed = time.perf_counter() - t

    for e in envs:
        e.close()
    return (steps * num_envs) / elapsed


# ── Subprocess (gym AsyncVectorEnv) ────────────────────────────────────────

def bench_subproc(num_envs=NUM_ENVS, steps=STEPS):
    envs = gym.make_vec(ENV_ID, num_envs=num_envs, vectorization_mode="async")
    actions = envs.action_space.sample()
    envs.reset()
    t = time.perf_counter()
    for _ in range(steps):
        envs.step(actions)
    elapsed = time.perf_counter() - t
    envs.close()
    return (steps * num_envs) / elapsed


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Benchmarking {ENV_ID} with {NUM_ENVS} envs, {STEPS} steps each\n")

    single = bench_single()
    print(f"Single env:                {single:7.0f} SPS")
    print(f"Theoretical ceiling (×{NUM_ENVS}): {single * NUM_ENVS:7.0f} SPS")

    threads = bench_threads()
    print(f"Thread pool (×{NUM_ENVS}):          {threads:7.0f} SPS  "
          f"({100*threads/(single*NUM_ENVS):.0f}% of ceiling)")

    subproc = bench_subproc()
    print(f"Subprocess AsyncVectorEnv: {subproc:7.0f} SPS  "
          f"({100*subproc/(single*NUM_ENVS):.0f}% of ceiling)")

    print(f"\nThread speedup vs subprocess: {threads/subproc:.1f}×")
