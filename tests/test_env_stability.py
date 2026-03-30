#!/usr/bin/env python3
"""
Stability diagnostic for NS-Gym sampled environments.

Runs a random policy on sampled NSEnvConfig objects and compares episode
statistics against a stationary baseline.

Two heuristics — both robust to a bad random policy:

  1. Median episode length ratio  (primary)
     A degenerate config collapses episode length regardless of policy quality.
     Flag when median_ns < baseline_median * threshold.

  2. Minimum episode length  (secondary, absolute)
     If ANY episode ends in <= MIN_LEN_DEGENERATE steps the env is physically
     broken (e.g. gravity flipped sign, mass zeroed out).  This is independent
     of the baseline because 1-step episodes are always wrong.

Early termination rate is intentionally NOT used: with a random policy the
baseline is already 40-60% on CartPole/Ant, making it indistinguishable from
any NS config.

Usage:
    python scripts/test_env_stability.py --env ant
    python scripts/test_env_stability.py --env cartpole --n-configs 30 --n-episodes 100
    python scripts/test_env_stability.py --env frozenlake --n-configs 20
    python scripts/test_env_stability.py --env ant --seed 42 --show-flagged-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.AAMAS_Comp.envs import NS_ENV_SAMPLERS, NSEnvFactory

# ---------------------------------------------------------------------------
# Per-environment constants
# ---------------------------------------------------------------------------

ENV_META = {
    "ant": {
        "gym_id":    "Ant-v5",
        "gym_kwargs": {},
        # Absolute minimum: a 1-step Ant episode means the physics exploded
        "min_len_degenerate": 5,
        "min_len_warn":       15,
    },
    "cartpole": {
        "gym_id":    "CartPole-v1",
        "gym_kwargs": {},
        "min_len_degenerate": 2,
        "min_len_warn":       5,
    },
    "frozenlake": {
        "gym_id":    "FrozenLake-v1",
        "gym_kwargs": {"disable_env_checker": True},
        "min_len_degenerate": 1,
        "min_len_warn":       2,
    },
}

# Fraction of baseline median below which a config is flagged (heuristic 1)
LENGTH_WARN       = 0.40   # < 40% of baseline median → WARN
LENGTH_DEGENERATE = 0.15   # < 15% of baseline median → DEGENERATE


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(env: gym.Env, n_episodes: int) -> dict:
    """Run n_episodes with a random policy; return summary statistics."""
    lengths, returns, truncated_flags = [], [], []

    for _ in range(n_episodes):
        env.reset()
        ep_return, ep_length, terminated, truncated = 0.0, 0, False, False

        while True:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            ep_return += reward
            ep_length += 1
            if terminated or truncated:
                break

        lengths.append(ep_length)
        returns.append(ep_return)
        truncated_flags.append(truncated)

    return {
        "mean_length":     float(np.mean(lengths)),
        "median_length":   float(np.median(lengths)),
        "min_length":      int(np.min(lengths)),
        "std_length":      float(np.std(lengths)),
        "truncation_rate": float(np.mean(truncated_flags)),
        "mean_return":     float(np.mean(returns)),
    }


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------

def classify(stats: dict, baseline: dict, meta: dict) -> tuple[str, str]:
    """Return (status, reason) where status is 'OK', 'WARN', or 'DEGENERATE'."""
    med_ratio = stats["median_length"] / max(baseline["median_length"], 1)
    min_len   = stats["min_length"]

    # Heuristic 1: length ratio vs baseline
    if med_ratio < LENGTH_DEGENERATE:
        return "DEGENERATE", f"median length {med_ratio:.0%} of baseline"
    if med_ratio < LENGTH_WARN:
        return "WARN", f"median length {med_ratio:.0%} of baseline"

    # Heuristic 2: absolute minimum episode length
    if min_len <= meta["min_len_degenerate"]:
        return "DEGENERATE", f"min episode length = {min_len} steps"
    if min_len <= meta["min_len_warn"]:
        return "WARN", f"min episode length = {min_len} steps"

    return "OK", ""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

STATUS_COLOUR = {"OK": "\033[32m", "WARN": "\033[33m", "DEGENERATE": "\033[31m"}
RESET = "\033[0m"

def colour(text: str, status: str) -> str:
    return f"{STATUS_COLOUR.get(status, '')}{text}{RESET}"

def fmt_config(config) -> str:
    parts = []
    for pname, pc in config.tunable_params.items():
        sched  = pc.scheduler.cls.replace("Scheduler", "")
        fn     = pc.update_fn.cls
        kw     = pc.update_fn.kwargs
        kw_str = ""
        if kw:
            key, val = next(iter(kw.items()))
            kw_str = f"({key}={val:.4g})"
        parts.append(f"{pname}: {sched}+{fn}{kw_str}")
    return "  |  ".join(parts)

def print_row(idx: str, status: str, reason: str, stats: dict, baseline: dict, config=None):
    med_ratio = stats["median_length"] / max(baseline["median_length"], 1)
    label = colour(f"[{status:^11}]", status)
    reason_str = f"  ← {reason}" if reason else ""
    print(
        f"  {idx:<4} {label}  "
        f"med={stats['median_length']:6.1f} ({med_ratio:4.0%})  "
        f"min={stats['min_length']:4d}  "
        f"trunc={stats['truncation_rate']:4.1%}  "
        f"ret={stats['mean_return']:+8.2f}"
        f"{reason_str}"
    )
    if config is not None:
        print(f"       {fmt_config(config)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NS-Gym environment stability diagnostic")
    parser.add_argument("--env",              default="ant",
                        choices=list(ENV_META), help="Environment family")
    parser.add_argument("--n-configs",        type=int, default=20,
                        help="Number of random configs to test")
    parser.add_argument("--n-episodes",       type=int, default=30,
                        help="Episodes per config (and for baseline)")
    parser.add_argument("--seed",             type=int, default=0)
    parser.add_argument("--show-flagged-only", action="store_true",
                        help="Only print WARN/DEGENERATE configs")
    args = parser.parse_args()

    meta   = ENV_META[args.env]
    gym_id = meta["gym_id"]
    gym_kw = meta["gym_kwargs"]

    print(f"\n{'='*72}")
    print(f"  NS-Gym Stability Diagnostic  |  env={args.env}  "
          f"n_configs={args.n_configs}  n_episodes={args.n_episodes}")
    print(f"{'='*72}")
    print(f"\n  Heuristics:")
    print(f"    WARN       if median < {LENGTH_WARN:.0%} of baseline  "
          f"OR min episode <= {meta['min_len_warn']} steps")
    print(f"    DEGENERATE if median < {LENGTH_DEGENERATE:.0%} of baseline  "
          f"OR min episode <= {meta['min_len_degenerate']} steps")
    print(f"\n  Note: early-termination rate is NOT used — it is already high")
    print(f"  (~40-70%) for a random policy and adds no discriminating signal.")

    # ── Baseline ──────────────────────────────────────────────────────────
    print(f"\n  Running baseline ({gym_id}, stationary, random policy)...")
    base_env = gym.make(gym_id, **gym_kw)
    baseline = run_episodes(base_env, args.n_episodes)
    base_env.close()

    print(f"\n  BASELINE  "
          f"med={baseline['median_length']:6.1f}  "
          f"min={baseline['min_length']:4d}  "
          f"trunc={baseline['truncation_rate']:4.1%}  "
          f"ret={baseline['mean_return']:+8.2f}")
    print()

    # ── Sampled configs ───────────────────────────────────────────────────
    sampler = NS_ENV_SAMPLERS[args.env](seed=args.seed)
    counts  = {"OK": 0, "WARN": 0, "DEGENERATE": 0}

    for i in range(args.n_configs):
        config = sampler.sample()
        try:
            env   = NSEnvFactory.make(config)
            stats = run_episodes(env, args.n_episodes)
            env.close()
        except Exception as e:
            print(f"  {i:<4} {colour('[BUILD ERROR]', 'DEGENERATE')}  {e}")
            print(f"       {fmt_config(config)}")
            counts["DEGENERATE"] += 1
            continue

        status, reason = classify(stats, baseline, meta)
        counts[status] += 1

        if args.show_flagged_only and status == "OK":
            continue

        print_row(str(i), status, reason, stats, baseline, config)

    # ── Summary ───────────────────────────────────────────────────────────
    total = args.n_configs
    print(f"\n  {'─'*68}")
    print(f"  Summary: {colour(str(counts['OK']), 'OK')}/{total} OK  "
          f"{colour(str(counts['WARN']), 'WARN')}/{total} WARN  "
          f"{colour(str(counts['DEGENERATE']), 'DEGENERATE')}/{total} DEGENERATE")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
