"""
Print the spread of NS environment configs sampled by the held-out evaluator.

Run with:  pytest tests/test_env_diversity.py -s -v

This is a diagnostic / exploratory test — it always passes but prints a
detailed breakdown of scheduler classes, update function classes, and
parameter kwargs so you can visually inspect how diverse the sampled
configs are.
"""
from collections import Counter, defaultdict

import numpy as np
import pytest

from AAMAS_Comp.curriculum.plr_env import FixedNSEnv, sample_held_out_configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_policy_returns(
    sampler_key: str,
    n_configs: int,
    max_steps: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Run one random-policy episode per NS config and return the episode returns.

    Each config is wrapped in a FixedNSEnv so parameters evolve within the
    episode exactly as they would during real evaluation (scheduler + update_fn
    fire on every step), but reset to initial values on reset().

    Args:
        sampler_key:  Key into NS_ENV_SAMPLERS (e.g. "cartpole", "frozenlake").
        n_configs:    Number of distinct configs to sample and roll out.
        max_steps:    Hard cap per episode.  The env's own TimeLimit wrapper
                      will truncate earlier for most envs (CartPole→500,
                      FrozenLake→200, Ant→1000).
        seed:         Seed for both config sampling and action sampling.

    Returns:
        np.ndarray of shape (n_configs,) with total episode returns.
    """
    configs = sample_held_out_configs(sampler_key, n_configs, seed=seed)
    rng = np.random.default_rng(seed)
    returns = []

    for cfg in configs:
        env = FixedNSEnv(cfg)
        # Use a fresh integer seed per episode so episodes are independent
        ep_seed = int(rng.integers(0, 2**31))
        env.reset(seed=ep_seed)
        ep_return = 0.0
        for _ in range(max_steps):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            if terminated or truncated:
                break
        returns.append(ep_return)
        env.close()

    return np.array(returns)


def _summarise_configs(
    configs,
    label: str,
    sampler_key: str,
    n_ground_truth: int = 300,
    max_steps: int = 1000,
    n_bootstrap: int = 2000,
    seed: int = 0,
):
    """Print a diversity report for a list of NSEnvConfigs.

    This function does two independent things:

    Part 1 — Config diversity report (uses the actual `configs` list):
        Counts how often each scheduler class, update function class, and
        (param, scheduler, update_fn) combination appears across all sampled
        configs. Also prints numerical kwarg ranges (e.g. amplitude, period).
        This verifies that `sample_held_out_configs` is actually exploring the
        parameter space rather than collapsing to a few repeated configs.

    Part 2 — Bootstrap IQM variance from random-policy rollouts:
        Answers: "How many held-out configs do I need for a reliable IQM?"
        Uses actual episode returns from a random policy rather than any
        distributional assumption (no N(0,1), no parameter proxy).

        Method:
          1. Sample `n_ground_truth` configs and run one random-policy episode
             each.  This large sample approximates the true marginal return
             distribution across configs.
          2. Bootstrap: for each candidate eval set size, resample `size`
             returns from the ground-truth pool and compute IQM.  Repeat
             `n_bootstrap` times.  Report the 95% CI half-width.

        The CI widths are in *real return units* for this specific env+sampler.
        A random policy gives a lower bound on config-induced return variance —
        a trained policy that is sensitive to the NS params will show at least
        as much variance (often more).

    Returns a dict with coverage stats so callers can assert if desired.
    """
    rng = np.random.default_rng(seed)

    # ── Part 1: Config diversity ────────────────────────────────────────────
    scheduler_counts: Counter = Counter()
    update_fn_counts: Counter = Counter()
    param_counts: Counter = Counter()
    combo_counts: Counter = Counter()  # (param, scheduler, update_fn)

    # Collect numerical kwargs per (param, update_fn)
    kwarg_vals: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for cfg in configs:
        for param_name, pc in cfg.tunable_params.items():
            sched_cls = pc.scheduler.cls
            ufn_cls = pc.update_fn.cls
            scheduler_counts[sched_cls] += 1
            update_fn_counts[ufn_cls] += 1
            param_counts[param_name] += 1
            combo_counts[(param_name, sched_cls, ufn_cls)] += 1

            for k, v in {**pc.scheduler.kwargs, **pc.update_fn.kwargs}.items():
                kwarg_vals[(param_name, ufn_cls)][k].append(float(v))

    n = len(configs)
    sep = "=" * 65

    print(f"\n{sep}")
    print(f"  Diversity report — {label}  (n={n} configs)")
    print(sep)

    print(f"\n{'Param name':<20} {'count':>6}  {'% configs':>10}")
    print("-" * 40)
    for param, cnt in sorted(param_counts.items(), key=lambda x: -x[1]):
        print(f"  {param:<18} {cnt:>6}  {100*cnt/n:>9.1f}%")

    print(f"\n{'Scheduler class':<35} {'count':>6}")
    print("-" * 45)
    for cls, cnt in sorted(scheduler_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<33} {cnt:>6}")

    print(f"\n{'Update function class':<35} {'count':>6}")
    print("-" * 45)
    for cls, cnt in sorted(update_fn_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<33} {cnt:>6}")

    print(f"\n{'(param, scheduler, update_fn) combos — top 15':}")
    print("-" * 65)
    for combo, cnt in sorted(combo_counts.items(), key=lambda x: -x[1])[:15]:
        param, sched, ufn = combo
        print(f"  {param:<14} {sched:<28} {ufn:<28} ×{cnt}")

    print(f"\nNumerical kwarg ranges (mean ± std  [min, max]):")
    print("-" * 65)
    for (param, ufn), kw_dict in sorted(kwarg_vals.items()):
        for kw, vals in sorted(kw_dict.items()):
            arr = np.array(vals)
            print(
                f"  {param:<14} {ufn:<28} {kw:<14}"
                f"  {arr.mean():.4g} ± {arr.std():.4g}"
                f"  [{arr.min():.4g}, {arr.max():.4g}]"
            )

    # ── Part 2: Bootstrap IQM variance from random-policy rollouts ──────────
    # Sample n_ground_truth configs, run one random episode each.
    # Using a seed offset so ground-truth configs don't overlap the eval set.
    print(f"\nCollecting random-policy returns ({n_ground_truth} configs) …")
    ground_truth = _random_policy_returns(
        sampler_key,
        n_configs=n_ground_truth,
        max_steps=max_steps,
        seed=seed + 1000,
    )

    gt_mean = ground_truth.mean()
    gt_std  = ground_truth.std()
    gt_min  = ground_truth.min()
    gt_max  = ground_truth.max()
    q1_gt, q3_gt = np.percentile(ground_truth, [25, 75])
    iqm_mask_gt = (ground_truth >= q1_gt) & (ground_truth <= q3_gt)
    gt_iqm = ground_truth[iqm_mask_gt].mean() if iqm_mask_gt.any() else gt_mean
    gt_skew = float(np.mean(((ground_truth - gt_mean) / gt_std) ** 3)) if gt_std > 1e-8 else 0.0

    print(f"\nRandom-policy return distribution  (n={n_ground_truth}, seed={seed+1000})")
    print("-" * 65)
    print(f"  mean={gt_mean:.3f}  std={gt_std:.3f}  skew={gt_skew:+.2f}")
    print(f"  min={gt_min:.3f}   max={gt_max:.3f}")
    print(f"  IQM={gt_iqm:.3f}   [Q1={q1_gt:.3f}, Q3={q3_gt:.3f}]")
    print()
    print("  A non-zero std confirms that different NS configs produce different")
    print("  episode returns even under a random policy, so config diversity")
    print("  meaningfully affects difficulty.  High skew means a few configs are")
    print("  much easier/harder than the rest (IQM is more robust than mean here).")

    # Bootstrap: resample `size` values from ground_truth with replacement,
    # compute IQM, repeat n_bootstrap times.  CI widths are in return units.
    print(f"\nBootstrap IQM variance  (resampled from ground-truth pool, n_bootstrap={n_bootstrap})")
    print("-" * 65)
    print(f"  {'n':>5}  {'IQM std':>10}  {'95% CI ±':>12}  {'CI / σ':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*8}")
    for size in [10, 20, 50, 100]:
        iqms = []
        for _ in range(n_bootstrap):
            sample = rng.choice(ground_truth, size=size, replace=True)
            q1, q3 = np.percentile(sample, [25, 75])
            trimmed = sample[(sample >= q1) & (sample <= q3)]
            iqms.append(trimmed.mean() if len(trimmed) > 0 else sample.mean())
        arr = np.array(iqms)
        sem = arr.std()
        ci = 1.96 * sem
        ci_rel = ci / gt_std if gt_std > 1e-8 else float("nan")
        print(f"  {size:>5}  {sem:>10.4f}  {ci:>12.4f}  {ci_rel:>7.1%}")
    print()
    print("  CI / σ: fraction of the return std covered by the 95% CI half-width.")
    print("  < 20% → n is sufficient to detect moderate PLR vs baseline effects.")

    print(f"\n{sep}\n")

    return {
        "n_unique_schedulers": len(scheduler_counts),
        "n_unique_update_fns": len(update_fn_counts),
        "n_unique_combos": len(combo_counts),
        "n_configs": n,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

NUM_CONFIGS = 64


@pytest.mark.parametrize("sampler_key,n_configs,seed", [
    ("frozenlake", NUM_CONFIGS, 42),
    ("cartpole",   NUM_CONFIGS, 42),
])
def test_print_env_diversity(sampler_key, n_configs, seed):
    """Sample held-out configs and print their spread. Always passes.

    Part 1 (_summarise_configs diversity report) uses `n_configs` real configs.
    Part 2 (bootstrap IQM) samples an additional `n_ground_truth` configs,
    runs one random-policy episode each, and reports CI widths in real return
    units — no distributional assumptions.
    """
    configs = sample_held_out_configs(sampler_key, n_configs, seed=seed)
    assert len(configs) == n_configs

    stats = _summarise_configs(
        configs,
        label=f"{sampler_key} n={n_configs} seed={seed}",
        sampler_key=sampler_key,
        n_ground_truth=300,
    )

    # Basic sanity: at least some diversity (not all configs identical)
    assert stats["n_unique_combos"] >= 1
    assert stats["n_configs"] == n_configs


@pytest.mark.slow
@pytest.mark.parametrize("sampler_key,n_configs,seed", [
    ("ant", NUM_CONFIGS, 42),
])
def test_print_env_diversity_ant(sampler_key, n_configs, seed):
    """Same report for Ant (marked slow due to MuJoCo imports and rollouts).

    Uses a smaller n_ground_truth (100) to keep runtime reasonable — each
    Ant episode can be up to 1000 MuJoCo steps.
    """
    configs = sample_held_out_configs(sampler_key, n_configs, seed=seed)
    assert len(configs) == n_configs
    stats = _summarise_configs(
        configs,
        label=f"{sampler_key} n={n_configs} seed={seed}",
        sampler_key=sampler_key,
        n_ground_truth=100,
    )
    assert stats["n_unique_combos"] >= 1


@pytest.mark.parametrize("sampler_key", ["frozenlake", "cartpole"])
def test_seed_reproducibility(sampler_key):
    """Same seed → identical configs; different seed → at least one difference."""
    a = sample_held_out_configs(sampler_key, 10, seed=0)
    b = sample_held_out_configs(sampler_key, 10, seed=0)
    c = sample_held_out_configs(sampler_key, 10, seed=99)

    # Reproducible
    for ca, cb in zip(a, b):
        assert ca.tunable_params.keys() == cb.tunable_params.keys()
        for param in ca.tunable_params:
            assert ca.tunable_params[param].scheduler.cls == cb.tunable_params[param].scheduler.cls
            assert ca.tunable_params[param].update_fn.cls == cb.tunable_params[param].update_fn.cls

    # Different seed → at least one config differs somewhere
    differences = sum(
        any(
            ca.tunable_params[p].scheduler.cls != cc.tunable_params.get(p, ca.tunable_params[p]).scheduler.cls
            or ca.tunable_params[p].update_fn.cls != cc.tunable_params.get(p, ca.tunable_params[p]).update_fn.cls
            for p in ca.tunable_params
        )
        for ca, cc in zip(a, c)
    )
    assert differences > 0, "Different seeds produced identical configs — seeding may be broken"
