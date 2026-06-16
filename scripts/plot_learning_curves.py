"""Generate the training-curve figure for the report (Figure 3).

Plots, per task, the held-out evaluation IQM and the PLR curriculum episode
return against environment steps. Data is read from the cached
``results/learning_curves.json``; if that file is missing it is rebuilt by
scanning the offline W&B run logs for the three shipped training runs.

Usage:
    python scripts/plot_learning_curves.py
        [--out report/figures/learning_curves.pdf]
        [--data results/learning_curves.json] [--rebuild]

The figure shows the divergence between a steadily rising held-out return and
the lower, more volatile curriculum return (PLR concentrates on high-regret
levels). FrozenLake inverts the ordering: with no context features the policy
cannot identify the slip probabilities from a single observation, so held-out
success saturates while PLR avoids the unsolvable levels.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shipped full-length training runs (env -> offline W&B run directory).
RUNS = {
    "ant":        "logs/wandb/wandb/run-20260413_041525-h7ha5ocy",
    "cartpole":   "logs/wandb/wandb/run-20260424_210613-l8kz4hcd",
    "frozenlake": "logs/wandb/wandb/run-20260424_220234-vkal8ltu",  # no-context, 4M
}
# (key, title, smoothing window, lower clip for the volatility band or None)
PANELS = [("ant", "Ant", 151, None), ("cartpole", "CartPole", 151, 0.0),
          ("frozenlake", "FrozenLake", 61, 0.0)]
EVAL_COLOR = "#1f3a93"
PLR_COLOR = "#e07b00"


def extract_from_wandb(run_dir: str, max_pts: int = 3000) -> dict:
    """Read eval IQM and PLR episode return time series from an offline run."""
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2 as pb

    def full_key(item):
        return "/".join(item.nested_key) if item.nested_key else item.key

    path = glob.glob(f"{run_dir}/run-*.wandb")[0]
    ds = DataStore()
    ds.open_for_scan(path)
    cur_step, plr, ev = None, [], []
    while True:
        rec = ds.scan_data()
        if rec is None:
            break
        r = pb.Record()
        r.ParseFromString(rec)
        if r.WhichOneof("record_type") != "history":
            continue
        d = {full_key(it): it.value_json for it in r.history.item}
        if "train/global_step" in d:
            cur_step = float(json.loads(d["train/global_step"]))
        if "plr/episode_return" in d and cur_step is not None:
            plr.append((cur_step, float(json.loads(d["plr/episode_return"]))))
        if "eval/reward_iqm" in d and cur_step is not None:
            ev.append((cur_step, float(json.loads(d["eval/reward_iqm"]))))
    a = np.array(plr)
    if len(a) > max_pts:
        a = a[np.linspace(0, len(a) - 1, max_pts).astype(int)]
    return {
        "plr_step": a[:, 0].tolist(), "plr_ret": a[:, 1].tolist(),
        "eval_step": [e[0] for e in ev], "eval_iqm": [e[1] for e in ev],
    }


def smooth(y, w):
    y = np.asarray(y, float)
    if len(y) < 5:
        return y
    if w % 2 == 0:
        w += 1
    w = max(3, min(w, (len(y) // 2) * 2 - 1))
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w) / w, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/figures/learning_curves.pdf")
    ap.add_argument("--data", default="results/learning_curves.json")
    ap.add_argument("--rebuild", action="store_true", help="re-extract from W&B logs")
    args = ap.parse_args()

    data_path = Path(args.data)
    if args.rebuild or not data_path.exists():
        data = {k: extract_from_wandb(v) for k, v in RUNS.items()}
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps(data))
        print(f"Wrote {data_path}")
    else:
        data = json.loads(data_path.read_text())

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    })
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6))
    for ax, (key, title, w, clip_low) in zip(axes, PANELS):
        r = data[key]
        ps = np.array(r["plr_step"]) / 1e6
        es = np.array(r["eval_step"]) / 1e6
        pr = np.array(r["plr_ret"], float)
        mean = smooth(pr, w)
        std = np.sqrt(np.clip(smooth(pr * pr, w) - mean ** 2, 0, None))
        lo, hi = mean - std, mean + std
        if clip_low is not None:
            lo = np.clip(lo, clip_low, None)
        ax.fill_between(ps, lo, hi, color=PLR_COLOR, alpha=0.18, lw=0, zorder=1)
        ax.plot(ps, mean, color=PLR_COLOR, lw=1.7, zorder=2, label="PLR curriculum return")
        ax.plot(es, r["eval_iqm"], color=EVAL_COLOR, lw=1.7, marker="o", ms=4, zorder=3,
                label="Held-out eval (IQM)")
        ax.set_title(title)
        ax.set_xlabel("Environment steps (M)")
        ax.margins(x=0.02)
        ax.grid(True, alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Episode return")
    axes[2].set_ylabel("Success rate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
