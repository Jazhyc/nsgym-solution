#!/usr/bin/env python3
"""
PLR + NS-CartPole benchmark: SPS and overhead assessment.

Measures:
  1. Stationary CartPole baseline SPS (gymnasium only).
  2. NS CartPole with a fixed config (NS wrapper overhead, no PLR).
  3. NS CartPole with PLR active (full pipeline: sample config → run episode →
     score → update buffer).

A dummy value function (constant 0) is used so the TD-error score is just
|return|.  This isolates environment and PLR overhead from any neural net cost.

Usage:
    python scripts/benchmark_plr_cartpole.py
    python scripts/benchmark_plr_cartpole.py --n-episodes 500 --n-envs 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.AAMAS_Comp.envs import NS_ENV_SAMPLERS, NSEnvFactory, NS_ENV_CONFIGS
from src.AAMAS_Comp.curriculum import PLRBuffer, td_error_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_episode_stationary(env: gym.Env) -> tuple[int, float]:
    """Random policy on a stationary env. Returns (steps, episode_return)."""
    env.reset()
    steps, total_reward = 0, 0.0
    while True:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        total_reward += reward
        if terminated or truncated:
            return steps, total_reward


def run_episode_scored(env: gym.Env, gamma: float = 0.99) -> tuple[int, float, float]:
    """Random policy with GAE-style scoring.  Returns (steps, return, score)."""
    env.reset()
    rewards, dones = [], []

    while True:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        rewards.append(reward)
        dones.append(done)
        if done:
            break

    T = len(rewards)
    # Dummy value function V(s) = 0 everywhere → TD error = |return_t|
    values = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        g = rewards[t] + gamma * g * (1 - dones[t])
        returns[t] = g

    score = td_error_score(values, returns)
    return T, float(sum(rewards)), score


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_stationary(n_episodes: int) -> dict:
    env = gym.make("CartPole-v1")
    steps_list, return_list = [], []

    t0 = time.perf_counter()
    for _ in range(n_episodes):
        s, r = run_episode_stationary(env)
        steps_list.append(s)
        return_list.append(r)
    elapsed = time.perf_counter() - t0
    env.close()

    total_steps = sum(steps_list)
    return {
        "label":          "Stationary CartPole (baseline)",
        "n_episodes":     n_episodes,
        "total_steps":    total_steps,
        "elapsed_s":      elapsed,
        "sps":            total_steps / elapsed,
        "mean_length":    float(np.mean(steps_list)),
        "mean_return":    float(np.mean(return_list)),
    }


def bench_ns_fixed(n_episodes: int) -> dict:
    """NS wrapper with a fixed config (no PLR sampling overhead)."""
    config = NS_ENV_CONFIGS["cartpole_multi_param"]()
    env = NSEnvFactory.make(config)
    steps_list, return_list = [], []

    t0 = time.perf_counter()
    for _ in range(n_episodes):
        s, r = run_episode_stationary(env)
        steps_list.append(s)
        return_list.append(r)
    elapsed = time.perf_counter() - t0
    env.close()

    total_steps = sum(steps_list)
    return {
        "label":          "NS CartPole, fixed config (wrapper overhead only)",
        "n_episodes":     n_episodes,
        "total_steps":    total_steps,
        "elapsed_s":      elapsed,
        "sps":            total_steps / elapsed,
        "mean_length":    float(np.mean(steps_list)),
        "mean_return":    float(np.mean(return_list)),
    }


def bench_plr(n_episodes: int, plr_capacity: int, replay_prob: float, seed: int) -> dict:
    """Full PLR pipeline: sample config → make env → run episode → score → update."""
    sampler = NS_ENV_SAMPLERS["cartpole"](seed=seed)
    plr = PLRBuffer(
        sampler,
        capacity=plr_capacity,
        replay_prob=replay_prob,
        score_temp=0.1,
        staleness_coef=0.1,
        min_fill=0.1,
        seed=seed,
    )

    steps_list, return_list, score_list = [], [], []
    env_rebuild_times = []
    env = None

    t0 = time.perf_counter()
    prev_level_id = None
    prev_score = None

    for ep in range(n_episodes):
        # Score the PREVIOUS episode before sampling the next config
        if prev_level_id is not None:
            plr.update(prev_level_id, prev_score)

        # Sample next config; build env only when config changes
        tb = time.perf_counter()
        level_id, config = plr.sample()
        env_rebuild_times.append(time.perf_counter() - tb)

        if env is not None:
            env.close()
        env = NSEnvFactory.make(config)

        steps, ep_return, score = run_episode_scored(env)
        steps_list.append(steps)
        return_list.append(ep_return)
        score_list.append(score)

        prev_level_id = level_id
        prev_score = score

    # Final update
    if prev_level_id is not None:
        plr.update(prev_level_id, prev_score)

    elapsed = time.perf_counter() - t0
    if env is not None:
        env.close()

    total_steps = sum(steps_list)
    plr_stats = plr.stats()

    return {
        "label":             "NS CartPole + PLR (full pipeline)",
        "n_episodes":        n_episodes,
        "total_steps":       total_steps,
        "elapsed_s":         elapsed,
        "sps":               total_steps / elapsed,
        "mean_length":       float(np.mean(steps_list)),
        "mean_return":       float(np.mean(return_list)),
        "mean_score":        float(np.mean(score_list)),
        # PLR overhead
        "env_rebuild_ms_mean": float(np.mean(env_rebuild_times)) * 1000,
        "env_rebuild_ms_total": float(np.sum(env_rebuild_times)) * 1000,
        **plr_stats,
    }


# ---------------------------------------------------------------------------
# Parallel variant (multiple envs, no PLR — pure throughput ceiling)
# ---------------------------------------------------------------------------

def bench_parallel(n_episodes: int, n_envs: int) -> dict:
    """gymnasium AsyncVectorEnv to show the parallel SPS ceiling."""
    envs = gym.make_vec("CartPole-v1", num_envs=n_envs, vectorization_mode="async")
    steps_total = 0

    t0 = time.perf_counter()
    obs, _ = envs.reset()
    # Run steps, not episodes — just measure raw throughput
    n_steps = (n_episodes * 200) // n_envs  # approx same total steps
    for _ in range(n_steps):
        actions = envs.action_space.sample()
        _, _, terminated, truncated, _ = envs.step(actions)
        steps_total += n_envs
    elapsed = time.perf_counter() - t0
    envs.close()

    return {
        "label":       f"Stationary CartPole, {n_envs}× AsyncVectorEnv (throughput ceiling)",
        "n_envs":      n_envs,
        "total_steps": steps_total,
        "elapsed_s":   elapsed,
        "sps":         steps_total / elapsed,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def print_result(r: dict) -> None:
    print(f"\n  {r['label']}")
    print(f"  {'─' * 60}")
    print(f"    Episodes : {r.get('n_episodes', '—')}")
    print(f"    Steps    : {r['total_steps']:,}")
    print(f"    Time     : {r['elapsed_s']:.2f}s")
    print(f"    SPS      : {r['sps']:,.0f}")
    if "mean_length" in r:
        print(f"    Mean ep. length : {r['mean_length']:.1f}")
    if "mean_return" in r:
        print(f"    Mean return     : {r['mean_return']:.2f}")
    if "mean_score" in r:
        print(f"    Mean PLR score  : {r['mean_score']:.3f}")
    if "env_rebuild_ms_mean" in r:
        print(f"    Env rebuild     : {r['env_rebuild_ms_mean']:.3f}ms/ep  "
              f"({r['env_rebuild_ms_total']:.1f}ms total)")
    if "plr/size" in r:
        print(f"    PLR buffer size : {r['plr/size']}")
        print(f"    PLR score mean  : {r['plr/score_mean']:.3f}  "
              f"max={r['plr/score_max']:.3f}")
        print(f"    PLR staleness   : mean={r['plr/staleness_mean']:.1f}  "
              f"max={r['plr/staleness_max']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PLR + NS-CartPole SPS benchmark")
    parser.add_argument("--n-episodes",   type=int, default=300,
                        help="Episodes for each serial benchmark (default: 300)")
    parser.add_argument("--n-envs",       type=int, default=8,
                        help="Parallel envs for throughput-ceiling test (default: 8)")
    parser.add_argument("--plr-capacity", type=int, default=100)
    parser.add_argument("--replay-prob",  type=float, default=0.5)
    parser.add_argument("--seed",         type=int, default=0)
    args = parser.parse_args()

    print(f"\n{'='*68}")
    print(f"  PLR + NS-CartPole Benchmark")
    print(f"  n_episodes={args.n_episodes}  n_envs={args.n_envs}  "
          f"plr_capacity={args.plr_capacity}  replay_prob={args.replay_prob}")
    print(f"{'='*68}")
    print("\n  Note: random policy used throughout — this measures env + PLR")
    print("  overhead only, NOT learning quality.\n")

    results = []

    print("  [1/4] Stationary baseline...")
    results.append(bench_stationary(args.n_episodes))

    print("  [2/4] NS CartPole, fixed config...")
    results.append(bench_ns_fixed(args.n_episodes))

    print("  [3/4] NS CartPole + PLR...")
    results.append(bench_plr(args.n_episodes, args.plr_capacity, args.replay_prob, args.seed))

    print(f"  [4/4] Parallel throughput ceiling ({args.n_envs}× AsyncVectorEnv)...")
    results.append(bench_parallel(args.n_episodes, args.n_envs))

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*68}")
    print(f"  Results")
    print(f"{'='*68}")

    for r in results:
        print_result(r)

    # ── SPS summary table ─────────────────────────────────────────────────────
    baseline_sps = results[0]["sps"]
    print(f"\n\n  {'─'*68}")
    print(f"  {'Benchmark':<48}  {'SPS':>8}  {'vs baseline':>12}")
    print(f"  {'─'*68}")
    for r in results:
        ratio = r["sps"] / baseline_sps
        print(f"  {r['label']:<48}  {r['sps']:>8,.0f}  {ratio:>11.2f}×")
    print(f"  {'─'*68}\n")


if __name__ == "__main__":
    main()
