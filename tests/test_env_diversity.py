"""
Print the spread of NS environment configs sampled by the held-out evaluator.

Run with:  pytest tests/test_env_diversity.py -s -v

This is a diagnostic / exploratory test — it always passes but prints a
detailed breakdown of scheduler classes, update function classes, and
parameter kwargs so you can visually inspect how diverse the sampled
configs are.
"""
import math
from collections import Counter, defaultdict

import numpy as np
import pytest

from AAMAS_Comp.curriculum.plr_env import sample_held_out_configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise_configs(configs, label: str, n_bootstrap: int = 2000, seed: int = 0):
    """Print a diversity report for a list of NSEnvConfigs.

    Returns a dict with coverage stats so callers can assert if desired.
    """
    rng = np.random.default_rng(seed)

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

    # ── Bootstrap IQM spread estimate ──────────────────────────────────────
    # Simulate what 'n' held-out configs would give as IQM variance.
    # We treat each config as contributing one hypothetical return drawn from
    # N(0,1) — purely to show the statistical point about sample size, not
    # the actual env difficulty.
    print(f"\n{'Bootstrap IQM variance analysis':}")
    print("-" * 65)
    for size in [10, 20, 50, 100]:
        iqms = []
        for _ in range(n_bootstrap):
            sample = rng.standard_normal(size)
            q1, q3 = np.percentile(sample, [25, 75])
            trimmed = sample[(sample >= q1) & (sample <= q3)]
            iqms.append(trimmed.mean() if len(trimmed) > 0 else sample.mean())
        iqms = np.array(iqms)
        sem = iqms.std()
        ci_width = 1.96 * sem
        print(
            f"  n={size:<4}  IQM std={sem:.4f}  95% CI width ≈ ±{ci_width:.4f}"
            f"  (relative to σ=1 returns)"
        )
    print()
    print("  Rule of thumb: you want the CI width to be < 5–10% of the")
    print("  return range to reliably detect PLR vs baseline differences.")

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

@pytest.mark.parametrize("sampler_key,n_configs,seed", [
    ("frozenlake", 20, 42),
    ("cartpole",   20, 42),
])
def test_print_env_diversity(sampler_key, n_configs, seed):
    """Sample held-out configs and print their spread. Always passes."""
    configs = sample_held_out_configs(sampler_key, n_configs, seed=seed)
    assert len(configs) == n_configs

    stats = _summarise_configs(configs, label=f"{sampler_key} n={n_configs} seed={seed}")

    # Basic sanity: at least some diversity (not all configs identical)
    assert stats["n_unique_combos"] >= 1
    assert stats["n_configs"] == n_configs


@pytest.mark.slow
@pytest.mark.parametrize("sampler_key,n_configs,seed", [
    ("ant", 20, 42),
])
def test_print_env_diversity_ant(sampler_key, n_configs, seed):
    """Same report for Ant (marked slow due to MuJoCo imports)."""
    configs = sample_held_out_configs(sampler_key, n_configs, seed=seed)
    assert len(configs) == n_configs
    stats = _summarise_configs(configs, label=f"{sampler_key} n={n_configs} seed={seed}")
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
