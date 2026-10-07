"""Return summaries with rank-based trimming, including tied/binary returns."""

import numpy as np


def trimmed_mean(values, lower=0.25, upper=0.75):
    ordered = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if not len(ordered):
        raise ValueError("Cannot summarize empty returns")
    left, right = lower * len(ordered), upper * len(ordered)
    indices = np.arange(len(ordered))
    weights = np.maximum(0, np.minimum(indices + 1, right) - np.maximum(indices, left))
    return float(np.sum(weights * ordered) / weights.sum())


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite evaluation return")
    return {"mean": float(values.mean()), "iqm": trimmed_mean(values),
            "worst_decile": trimmed_mean(values, 0, 0.1)}
