#!/usr/bin/env python3
"""
PLR + NS-CartPole SPS benchmark using TorchRL collectors.

Compares three setups with MultiAsyncCollector (same config as training):

  1. Stationary CartPole            — plain gym.make, baseline
  2. NS CartPole, fixed config      — FastNSClassicControlWrapper, no PLR
  3. NS CartPole, PLR rotating      — PLREnv rotates configs on each episode reset

PLREnv embeds the PLR buffer inside a gymnasium wrapper so TorchRL's
auto-reset calls plr.sample() naturally at every episode boundary.
Each parallel worker gets its own buffer copy — no IPC overhead.

Usage:
    python scripts/benchmark_plr_torchrl.py
    python scripts/benchmark_plr_torchrl.py --total-frames 200_000 --num-groups 2 --envs-per-group 8
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from torchrl.collectors import MultiAsyncCollector, SyncDataCollector
from torchrl.envs import Compose, DoubleToFloat, ParallelEnv, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymWrapper

from src.AAMAS_Comp.envs import NS_ENV_CONFIGS, NS_ENV_SAMPLERS, NSEnvFactory
from src.AAMAS_Comp.envs.fast_ns_wrapper import FastNSClassicControlWrapper
from src.AAMAS_Comp.curriculum import PLRBuffer


# ---------------------------------------------------------------------------
# PLREnv — PLR buffer embedded in a gymnasium wrapper
# ---------------------------------------------------------------------------

class PLREnv(gym.Env):
    """Gymnasium env that rotates NS configs via PLR on every reset().

    Design:
    - On reset(): scores the previous episode (using episode return as proxy),
      samples a new config from PLR, and rebuilds the underlying NS env.
    - Exposes raw numpy observations (no dict wrapping) via FastNSClassicControlWrapper.
    - Thread/process-safe: each instance is fully self-contained.

    Args:
        sampler_key: Key into NS_ENV_SAMPLERS ("cartpole", "ant", "frozenlake").
        plr_capacity: Max levels in the PLR buffer.
        replay_prob:  Probability of replaying vs exploring a new config.
        seed:         RNG seed for the PLR buffer and sampler.
    """

    def __init__(
        self,
        sampler_key: str = "cartpole",
        plr_capacity: int = 100,
        replay_prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        sampler = NS_ENV_SAMPLERS[sampler_key](seed=seed)
        self.plr = PLRBuffer(
            sampler,
            capacity=plr_capacity,
            replay_prob=replay_prob,
            seed=seed,
        )
        self._level_id: int | None = None
        self._episode_return: float = 0.0
        self._env: gym.Env | None = None

        # Build a throwaway env to read spaces (immediately closed)
        _, init_config = self.plr.sample()
        tmp = self._build_env(init_config)
        self.observation_space = tmp.observation_space
        self.action_space = tmp.action_space
        tmp.close()

    def _build_env(self, config) -> FastNSClassicControlWrapper:
        base = gym.make(config.env_id, **config.gym_kwargs)
        tunable = {name: pc.build() for name, pc in config.tunable_params.items()}
        return FastNSClassicControlWrapper(base, tunable)

    def reset(self, *, seed=None, options=None):
        # Score the episode that just ended
        if self._level_id is not None:
            self.plr.update(self._level_id, self._episode_return)

        # Sample next config and build its env
        self._level_id, config = self.plr.sample()
        if self._env is not None:
            self._env.close()
        self._env = self._build_env(config)

        self._episode_return = 0.0
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._episode_return += float(reward)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()
        super().close()


# ---------------------------------------------------------------------------
# Env factories (each returns a TorchRL TransformedEnv)
# ---------------------------------------------------------------------------

def _wrap_torchrl(gym_env: gym.Env) -> TransformedEnv:
    """Minimal TorchRL wrapping: GymWrapper → DoubleToFloat → StepCounter."""
    base = GymWrapper(gym_env, device="cpu")
    return TransformedEnv(base, Compose(DoubleToFloat(), StepCounter()))


def make_stationary() -> TransformedEnv:
    return _wrap_torchrl(gym.make("CartPole-v1"))


def make_ns_fixed() -> TransformedEnv:
    config = NS_ENV_CONFIGS["cartpole_multi_param"]()
    base = gym.make(config.env_id)
    tunable = {name: pc.build() for name, pc in config.tunable_params.items()}
    return _wrap_torchrl(FastNSClassicControlWrapper(base, tunable))


def make_plr(seed: int = 0) -> TransformedEnv:
    return _wrap_torchrl(PLREnv(sampler_key="cartpole", seed=seed))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _make_dummy_actor(env: TransformedEnv):
    """Random actor that just samples from action_space — no network overhead."""
    from torchrl.envs.utils import RandomPolicy
    return RandomPolicy(env.action_spec)


def run_benchmark(
    label: str,
    env_factory,
    total_frames: int,
    num_groups: int,
    envs_per_group: int,
    frames_per_batch: int,
    seed: int,
    use_sync: bool = False,
) -> dict:
    """Run a collector benchmark and return SPS stats."""
    device = torch.device("cpu")

    # Build a reference env for actor spec only
    ref_env = env_factory()
    actor = _make_dummy_actor(ref_env)
    ref_env.close()

    if use_sync:
        env = env_factory()
        collector = SyncDataCollector(
            env,
            actor,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
        )
    else:
        collector_envs = [
            ParallelEnv(
                num_workers=envs_per_group,
                create_env_fn=env_factory,
                serial_for_single=True,
            )
            for _ in range(num_groups)
        ]
        collector = MultiAsyncCollector(
            collector_envs,
            actor,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
        )

    # Warmup: one batch to let subprocesses spin up
    batch_frames_list = []
    batch_times = []

    it = iter(collector)
    warmup = next(it)
    t_start = time.perf_counter()

    for batch in it:
        t0 = time.perf_counter()
        batch_frames_list.append(batch.numel())
        batch_times.append(time.perf_counter() - t0)

    elapsed = time.perf_counter() - t_start
    total_collected = sum(batch_frames_list)

    collector.shutdown()
    gc.collect()
    time.sleep(1)  # let OS release subprocess resources before next test

    return {
        "label":          label,
        "total_frames":   total_collected,
        "elapsed_s":      elapsed,
        "sps":            total_collected / elapsed if elapsed > 0 else 0,
        "n_batches":      len(batch_frames_list),
        "use_sync":       use_sync,
        "num_groups":     1 if use_sync else num_groups,
        "envs_per_group": 1 if use_sync else envs_per_group,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def print_result(r: dict) -> None:
    collector_str = (
        f"SyncDataCollector (1 env)"
        if r["use_sync"]
        else f"MultiAsyncCollector ({r['num_groups']}g × {r['envs_per_group']}e)"
    )
    print(f"\n  {r['label']}")
    print(f"  {'─' * 62}")
    print(f"    Collector : {collector_str}")
    print(f"    Frames    : {r['total_frames']:,}")
    print(f"    Time      : {r['elapsed_s']:.2f}s")
    print(f"    SPS       : {r['sps']:,.0f}")
    print(f"    Batches   : {r['n_batches']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PLR + NS-CartPole TorchRL SPS benchmark")
    parser.add_argument("--total-frames",    type=int, default=100_000)
    parser.add_argument("--frames-per-batch", type=int, default=1_000)
    parser.add_argument("--num-groups",      type=int, default=2,
                        help="MultiAsyncCollector groups (default: 2)")
    parser.add_argument("--envs-per-group",  type=int, default=2,
                        help="ParallelEnv workers per group (default: 2). "
                             "Keep ≤3 on WSL2: the benchmark tears down and recreates "
                             "4 collector sets sequentially and OS semaphores don't release "
                             "fast enough with more workers. The ratios are stable at 2×2.")
    parser.add_argument("--seed",            type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(4)

    print(f"\n{'='*68}")
    print(f"  PLR + NS-CartPole — TorchRL Collector SPS Benchmark")
    print(f"  total_frames={args.total_frames:,}  fpb={args.frames_per_batch}  "
          f"groups={args.num_groups}  envs/group={args.envs_per_group}")
    print(f"{'='*68}")
    print(f"\n  Random actor used — measures env + collector overhead only.\n")

    results = []
    common = dict(
        total_frames=args.total_frames,
        num_groups=args.num_groups,
        envs_per_group=args.envs_per_group,
        frames_per_batch=args.frames_per_batch,
        seed=args.seed,
    )

    configs = [
        ("Stationary CartPole — Sync (1 env)",         make_stationary,                True),
        ("Stationary CartPole — MultiAsync",            make_stationary,                False),
        ("NS fixed config (fast wrapper) — MultiAsync", make_ns_fixed,                  False),
        ("NS + PLR rotating — MultiAsync",              partial(make_plr, args.seed),   False),
    ]

    for i, (label, factory, use_sync) in enumerate(configs, 1):
        print(f"  [{i}/{len(configs)}] {label}...")
        r = run_benchmark(label, factory, use_sync=use_sync, **common)
        results.append(r)
        print(f"          → {r['sps']:,.0f} SPS")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*68}")
    print(f"  Results")
    print(f"{'='*68}")
    for r in results:
        print_result(r)

    baseline_sps = results[1]["sps"]  # MultiAsync stationary as baseline
    print(f"\n\n  {'─'*68}")
    print(f"  {'Benchmark':<44}  {'SPS':>8}  {'vs async baseline':>18}")
    print(f"  {'─'*68}")
    for r in results:
        ratio = r["sps"] / baseline_sps
        print(f"  {r['label']:<44}  {r['sps']:>8,.0f}  {ratio:>17.2f}×")
    print(f"  {'─'*68}\n")


if __name__ == "__main__":
    main()
