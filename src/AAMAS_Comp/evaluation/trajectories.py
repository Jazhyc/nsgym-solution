"""Lossless, compressed evaluation trajectories for later change-based analysis."""

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def trace_parameters(env, task, observed):
    """Record full physical gravity on Ant, including lateral components."""
    if task == "ant":
        return {**observed, "gravity": env.unwrapped.model.opt.gravity.copy()}
    return observed


def flatten_parameters(values):
    names, numbers = [], []
    for name, value in sorted(values.items()):
        array = np.asarray(value).reshape(-1)
        names.extend([name] if array.size == 1 else [f"{name}[{i}]" for i in range(array.size)])
        numbers.extend(array.tolist())
    return names, np.asarray(numbers, dtype=np.float64)


class EvaluationTrajectories:
    """Stream one episode at a time into a standard NPZ; never pickle objects."""

    def __init__(self, path):
        self.path = Path(path)
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        if self.path.exists():
            raise FileExistsError(self.path)
        self.archive = None
        self.count = 0

    def __enter__(self):
        self.archive = ZipFile(self.partial, "w", compression=ZIP_DEFLATED, compresslevel=1)
        return self

    def write(self, record, trajectory):
        key = f"episode_{self.count:06d}"
        record["trajectory_key"] = key
        payload = {**trajectory, "metadata": np.asarray(json.dumps(record))}
        for name, value in payload.items():
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError(f"Object arrays are not allowed in trajectories: {name}")
            buffer = BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            self.archive.writestr(f"{key}/{name}.npy", buffer.getvalue())
        self.count += 1

    def __exit__(self, kind, value, traceback):
        self.archive.close()
        if kind is None:
            self.partial.replace(self.path)
        else:
            self.partial.unlink(missing_ok=True)


TRAJECTORY_SCHEMA = {
    "version": 1,
    "file": "test_trajectories.npz",
    "index": "test_episodes.jsonl:trajectory_key",
    "precision": "Original observation/action dtypes; rewards and parameters float64; lossless compression",
    "observations": "T+1 observations returned by the environment; checkpoint stores policy normalization",
    "actions": "T actions actually supplied to the environment",
    "rewards": "T undiscounted step rewards",
    "parameter_values": "T+1 rows: reset parameters, then actual parameters after each transition",
    "parameter_names": "Column names for parameter_values; Ant gravity includes all three components",
    "parameter_changed": "T bools using the evaluator's observed change threshold",
    "terminated": "T environment terminal flags",
    "truncated": "T truncation flags, including the declared evaluation horizon",
    "timestep": "T zero-based transition indices",
    "action_mode": "greedy; frozen policy, no online learning",
    "change_observation": "Ground-truth dynamics are recorded for analysis and are not supplied to the policy",
}
