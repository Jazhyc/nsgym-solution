#!/usr/bin/env python3
"""Random search hyperparameter tuning for train.py.

Reads a YAML search config, samples `n_trials` parameter combinations, runs
train.py as a subprocess for each, collects the metric, and logs a summary
table to WandB.

Usage
-----
    python scripts/hparam_search.py config/hparam/temperature.yaml

    # Override number of trials from CLI
    python scripts/hparam_search.py config/hparam/temperature.yaml --n_trials 5

The per-trial training runs log to `wandb.project` specified in the hparam
config.  A summary WandB run (with a results table) is created in the same
project at the end.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import wandb
import yaml


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def _sample_param(param_cfg: dict, rng: np.random.Generator):
    dist = param_cfg.get("distribution", "uniform")
    if dist == "uniform":
        return float(rng.uniform(param_cfg["low"], param_cfg["high"]))
    elif dist == "log_uniform":
        return float(np.exp(rng.uniform(np.log(param_cfg["low"]), np.log(param_cfg["high"]))))
    elif dist == "int_uniform":
        return int(rng.integers(param_cfg["low"], param_cfg["high"] + 1))
    elif dist == "categorical":
        choices = param_cfg["choices"]
        return choices[int(rng.integers(0, len(choices)))]
    else:
        raise ValueError(f"Unknown distribution '{dist}' in param '{param_cfg['name']}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Random search over train.py hyperparameters")
    parser.add_argument("hparam_config", help="Path to hparam YAML config")
    parser.add_argument("--n_trials", type=int, default=None,
                        help="Override n_trials from the config file")
    args = parser.parse_args()

    hparam_config_path = Path(args.hparam_config)
    with open(hparam_config_path) as f:
        hcfg = yaml.safe_load(f)

    n_trials     = args.n_trials or hcfg.get("n_trials", 10)
    metric_key   = hcfg.get("metric", "eval/reward_iqm")
    direction    = hcfg.get("direction", "maximize")
    env_name     = hcfg.get("env", "ant")
    fixed_ovr    = hcfg.get("fixed_overrides", {})
    params_cfg   = hcfg["params"]
    wandb_cfg    = hcfg.get("wandb", {})
    sampler_seed = hcfg.get("seed", 0)

    rng = np.random.default_rng(sampler_seed)

    train_script = Path(__file__).parent / "train.py"
    results: list[dict] = []

    print(f"Starting random search: {n_trials} trials | metric={metric_key} ({direction})")
    print(f"Config: {hparam_config_path}")
    print(f"Params: {[p['name'] for p in params_cfg]}\n")

    for trial_idx in range(n_trials):
        sampled = {p["name"]: _sample_param(p, rng) for p in params_cfg}
        trial_seed = int(rng.integers(0, 1_000_000))

        # Temp file for train.py to write its best metric into
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_path = Path(tmp.name)

        # Build Hydra CLI overrides
        overrides: list[str] = [f"env={env_name}"]
        for k, v in fixed_ovr.items():
            overrides.append(f"{k}={v}")
        for k, v in sampled.items():
            overrides.append(f"{k}={v}")
        overrides.append(f"seed={trial_seed}")
        overrides.append(f"+hparam_output_path={output_path}")

        # Route each trial run to the hparam wandb project
        wandb_project = wandb_cfg.get("project", "nsgym-hparam")
        wandb_group   = wandb_cfg.get("group", hparam_config_path.stem)
        overrides += [
            f"wandb.project={wandb_project}",
            f"wandb.group={wandb_group}",
            f"wandb.name=trial_{trial_idx:02d}",
        ]

        print(f"[Trial {trial_idx + 1}/{n_trials}]  seed={trial_seed}")
        for k, v in sampled.items():
            print(f"  {k} = {v:.6g}" if isinstance(v, float) else f"  {k} = {v}")

        cmd = [sys.executable, str(train_script)] + overrides
        proc = subprocess.run(cmd)

        # Collect metric written by train.py
        metric_val = None
        if proc.returncode == 0:
            try:
                data = json.loads(output_path.read_text())
                metric_val = data.get(metric_key)
            except Exception as e:
                print(f"  Warning: could not read metric — {e}")
        else:
            print(f"  Warning: train.py exited with code {proc.returncode}")
        output_path.unlink(missing_ok=True)

        results.append({
            "trial": trial_idx,
            "seed": trial_seed,
            "metric": metric_val,
            **sampled,
        })
        status = f"{metric_val:.4f}" if metric_val is not None else "FAILED"
        print(f"  {metric_key} = {status}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = [r for r in results if r["metric"] is not None]

    if valid:
        best = (max if direction == "maximize" else min)(valid, key=lambda r: r["metric"])
        print("=" * 60)
        print(f"Best trial {best['trial']:02d}:  {metric_key} = {best['metric']:.4f}")
        for p in params_cfg:
            k = p["name"]
            v = best[k]
            print(f"  {k} = {v:.6g}" if isinstance(v, float) else f"  {k} = {v}")
        print("=" * 60)
    else:
        print("No successful trials — check train.py output above.")

    # Save results JSON
    results_path = Path("hparam_results.json")
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {results_path}")

    # ── WandB summary run ────────────────────────────────────────────────────
    # One lightweight run that logs the results table and best-config summary
    # so the entire search is visible in a single WandB view.
    if wandb_cfg:
        param_names = [p["name"] for p in params_cfg]
        columns = ["trial", "seed", metric_key] + param_names
        table = wandb.Table(columns=columns)
        for r in results:
            row = [
                r["trial"],
                r["seed"],
                r["metric"] if r["metric"] is not None else float("nan"),
            ] + [r.get(k) for k in param_names]
            table.add_data(*row)

        summary_run = wandb.init(
            project=wandb_project,
            group=wandb_group,
            name="hparam_summary",
            config={
                "n_trials": n_trials,
                "metric": metric_key,
                "direction": direction,
                "env": env_name,
                "fixed_overrides": fixed_ovr,
                "params": params_cfg,
            },
            job_type="hparam_summary",
            reinit=True,
        )
        summary_run.log({"trials": table})
        if valid:
            summary_run.summary.update({
                f"best/{metric_key}": best["metric"],
                **{f"best/{k}": best[k] for k in param_names},
                "best_trial": best["trial"],
            })
        summary_run.finish()
        print(f"Summary run logged to WandB project '{wandb_project}'")


if __name__ == "__main__":
    main()
