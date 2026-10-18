"""Shared, checkpoint-independent paper test suites and paired-seed aggregation."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from AAMAS_Comp.curriculum.plr_env import _build_ns_env
from AAMAS_Comp.envs.domain_randomization import (
    DynamicsEnv, HORIZONS, parameters, seed_update_functions, static_env,
)
from AAMAS_Comp.envs.ns_env_factory import NSEnvConfig, ParamConfig, SchedulerConfig, UpdateFnConfig
from AAMAS_Comp.envs.ns_env_sampler import NS_ENV_SAMPLERS, make_sampler
from AAMAS_Comp.envs.constraints import valid_cartpole_parameter
from AAMAS_Comp.evaluation.statistics import summarize
from AAMAS_Comp.evaluation.trajectories import (
    EvaluationTrajectories, TRAJECTORY_SCHEMA, flatten_parameters, trace_parameters,
)


def decode_config(value):
    value = dict(value)
    value["tunable_params"] = {
        name: ParamConfig(SchedulerConfig(**spec["scheduler"]), UpdateFnConfig(**spec["update_fn"]))
        for name, spec in value["tunable_params"].items()
    }
    return NSEnvConfig(**value)


def calibrate_ranges(task, count=256, seed=71301):
    """Policy-independent parameter trajectories, without stepping a policy.

    Positive mass/length values only; Ant rejects mass <= 1e-6, matching its
    NS wrapper. DR samples the joint time snapshots; bounds are descriptive.
    """
    from ns_gym.schedulers import MemorylessScheduler

    env = static_env(task)
    env.reset(seed=seed)
    defaults = parameters(env, task)
    env.close()
    if task == "frozenlake":
        return {}, defaults, []
    samples = {key: [value] for key, value in defaults.items()}
    snapshots = []
    sampler = NS_ENV_SAMPLERS[task](seed=seed)
    for i in range(count):
        config = sampler.sample()
        functions = {}
        for j, (key, spec) in enumerate(config.tunable_params.items()):
            fn = spec.build()
            for k, obj in enumerate((fn, fn.scheduler)):
                if hasattr(obj, "rng"):
                    obj.rng = np.random.default_rng(np.random.SeedSequence([seed, i, j, k]))
                    # Construction samples this event before the seeded RNG is
                    # installed, just as in live environment update functions.
                    if isinstance(obj, MemorylessScheduler):
                        obj.transition_time = obj.rng.geometric(p=obj.p, size=(1,))
            functions[key] = fn
        state = dict(defaults)
        snapshot_times = set(np.linspace(0, HORIZONS[task] - 1, 16, dtype=int))
        for t in range(HORIZONS[task]):
            for key, fn in functions.items():
                candidate, _, _ = fn(state[key], t)
                candidate = float(candidate)
                positive = "mass" in key or key == "length"
                valid = (valid_cartpole_parameter(key, candidate) if task == 'cartpole' else
                         np.isfinite(candidate) and (not positive or candidate > 1e-6))
                if task in ("ant", "cartpole") and not valid:
                    candidate = state[key]  # reject the update, retaining the previous valid state
                state[key] = candidate
                if valid:
                    samples[key].append(candidate)
            if t in snapshot_times and all((valid_cartpole_parameter(k, v) if task == 'cartpole' else
                                           np.isfinite(v) and (v > 1e-6 if "mass" in k or k == "length" else True))
                                           for k, v in state.items()):
                snapshots.append(dict(state))
    return {key: [float(min(values)), float(max(values))] for key, values in samples.items()}, defaults, snapshots


def make_manifest(path, n_configs=200, rollout_seeds=5, calibration_count=256):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace frozen test manifest: {path}")
    manifest = {"version": 1, "n_configs": n_configs, "rollouts": rollout_seeds,
                "test_seed": 91203, "calibration_seed": 71301,
                "calibration_count": calibration_count,
                "dr_range_rule": "fixed empirical joint distribution: 16 time snapshots per policy-independent trajectory; valid mass/length only",
                "hyperparameters": "fixed existing task configs; no tuning", "tasks": {}}
    for task in HORIZONS:
        ranges, defaults, snapshots = calibrate_ranges(task, calibration_count)
        iid_sampler = NS_ENV_SAMPLERS[task](seed=91203)
        target_env = DynamicsEnv(task, ranges, seed=91204, samples=snapshots)
        suites = {"iid": [], "abrupt": [], "gradual": [], "no_change": []}
        for i in range(n_configs):
            seeds = [int(np.random.SeedSequence([91205, i, r]).generate_state(1)[0] % (2**31))
                     for r in range(rollout_seeds)]
            suites["iid"].append({"id": i, "config": asdict(iid_sampler.sample()), "seeds": seeds})
            target = target_env.sample_values()
            for suite, duration in (("abrupt", 0), ("gradual", HORIZONS[task] // 4), ("no_change", -1)):
                suites[suite].append({"id": i, "seeds": seeds, "intervention": {
                    "initial": defaults, "target": target, "step": max(5, HORIZONS[task] // 10),
                    "duration": duration}})
        target_env.close()
        manifest["tasks"][task] = {"dr_ranges": ranges, "dr_samples": snapshots,
                                  "defaults": defaults, "suites": suites}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def calibrate_frozenlake_training(count=256, seed=71301, initial_mode="uniform"):
    """Policy-independent slip snapshots from the chosen training generator.

    This calibrates only static DR's training distribution. It never changes
    the frozen broader evaluation configurations or their intervention targets.
    Updates precede transitions, matching the live NS wrappers.
    """
    sampler = make_sampler("frozenlake", seed, initial_mode)
    times = set(np.linspace(0, HORIZONS["frozenlake"] - 1, 16, dtype=int))
    snapshots = []
    for i in range(count):
        config = sampler.sample()
        env = _build_ns_env(config)
        episode_seed = int(np.random.SeedSequence([seed, i]).generate_state(1)[0])
        try:
            env.reset(seed=episode_seed)
            seed_update_functions(env, episode_seed)
            wrapper = env
            while "_update_fn" not in wrapper.__dict__:
                wrapper = wrapper.env
            probability = np.asarray(config.wrapper_kwargs["initial_prob_dist"])
            for t in range(HORIZONS["frozenlake"]):
                probability = np.asarray(wrapper._update_fn(probability.tolist(), t)[0], dtype=float)
                if (not np.isfinite(probability).all() or (probability < 0).any()
                        or not np.isclose(probability.sum(), 1)):
                    raise ValueError("Invalid FrozenLake training calibration probability")
                if t in times:
                    snapshots.append({"P": probability.tolist()})
        finally:
            env.close()
    if not snapshots:
        raise ValueError("Static DR needs nonempty training snapshots")
    return {"dr_ranges": {}, "defaults": {"P": [1 / 3] * 3}, "dr_samples": snapshots}


def training_distribution(count=256, seed=71311, frozenlake_initial_mode="uniform"):
    """Separate training calibration from already-observed evaluation cases."""
    tasks = {}
    for task in HORIZONS:
        if task == "frozenlake":
            tasks[task] = calibrate_frozenlake_training(count, seed, frozenlake_initial_mode)
        else:
            ranges, defaults, snapshots = calibrate_ranges(task, count, seed)
            tasks[task] = {"dr_ranges": ranges, "defaults": defaults, "dr_samples": snapshots}
    return {"calibration_seed": seed, "calibration_count": count, "snapshots_per_trajectory": 16,
            "frozenlake_initial_mode": frozenlake_initial_mode,
            "rule": "uniform valid joint time snapshots from policy-independent training trajectories",
            "tasks": tasks}


def episode(actor, checkpoint, env, task, seed, intervention=None, trajectory=None,
            action_mode="greedy"):
    from tensordict import TensorDict
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    seed_update_functions(env, seed)
    rewards = []
    changes = 0
    before = parameters(env, task)
    if task == 'cartpole' and not all(valid_cartpole_parameter(k, v) for k, v in before.items()):
        raise ValueError('Invalid CartPole physics at evaluation reset')
    if trajectory is not None:
        names, initial = flatten_parameters(trace_parameters(env, task, before))
        trajectory.update({"observations": [np.asarray(obs).copy()], "actions": [],
                           "parameter_names": names, "parameter_values": [initial],
                           "terminated": [], "truncated": [], "parameter_changed": []})
    horizon = HORIZONS[task]
    mean = checkpoint.get("obs_rms", {}).get("mean")
    std = checkpoint.get("obs_rms", {}).get("std")
    if action_mode not in ("greedy", "stochastic"):
        raise ValueError(action_mode)
    exploration = ExplorationType.RANDOM if action_mode == "stochastic" else ExplorationType.DETERMINISTIC
    with torch.no_grad(), set_exploration_type(exploration):
        for t in range(horizon):
            state = torch.as_tensor(obs, dtype=torch.float32)
            if state.ndim == 0:  # stationary FrozenLake's discrete observation
                state = torch.nn.functional.one_hot(state.long(), 16).float()
            if mean is not None:
                state = (state - mean.float()) / std.float()
            action = actor(TensorDict({"observation": state}, batch_size=[]))["action"].numpy()
            if env.action_space.shape == ():
                action = int(action.item()) if action.size == 1 else int(action.argmax())
            if not np.isfinite(action).all():
                raise ValueError(f"Non-finite action: {task}, seed {seed}, step {t}")
            if trajectory is not None:
                trajectory["actions"].append(np.asarray(action).copy())
            obs, reward, terminated, truncated, info = env.step(action)
            after = parameters(env, task)
            if task == 'cartpole' and not all(valid_cartpole_parameter(k, v) for k, v in after.items()):
                raise ValueError('Invalid CartPole physics during evaluation')
            changed = any(not np.allclose(before[k], after[k], rtol=0, atol=1e-12) for k in before)
            changes += int(changed)
            before = after
            if not np.isfinite(reward) or not np.isfinite(obs).all():
                raise ValueError(f"Non-finite rollout: {task}, seed {seed}, step {t}")
            rewards.append(float(reward))
            if trajectory is not None:
                trajectory["observations"].append(np.asarray(obs).copy())
                names, values = flatten_parameters(trace_parameters(env, task, after))
                if names != trajectory["parameter_names"]:
                    raise ValueError("Physical parameter columns changed during an episode")
                trajectory["parameter_values"].append(values)
                trajectory["terminated"].append(bool(terminated))
                trajectory["truncated"].append(bool(truncated or t + 1 == horizon))
                trajectory["parameter_changed"].append(bool(changed))
            if terminated or truncated:
                break
    result = {"return": sum(rewards), "steps": len(rewards), "parameter_changes": changes,
              "terminated": bool(terminated), "truncated": bool(truncated or len(rewards) == horizon)}
    if task == 'cartpole':
        current, rejections = env, 0
        while current is not None:
            rejections += current.__dict__.get('rejected_updates', 0)
            current = current.__dict__.get('env', current.__dict__.get('_env'))
        result.update(physics_invalid_steps=0, physics_constraint_rejections=int(rejections))
    if trajectory is not None:
        trajectory["rewards"] = np.asarray(rewards, dtype=np.float64)
        trajectory["timestep"] = np.arange(len(rewards), dtype=np.int64)
    if task == "frozenlake":
        result["success"] = float(sum(rewards) > 0)
    if intervention:
        start = intervention["step"]
        result["reached_change"] = len(rewards) > start
        result["post_change_return"] = sum(rewards[start:])
        window = min(25, start)
        result["pre_change_reward_per_step"] = float(np.mean(rewards[max(0, start-window):start])) if len(rewards) >= start else None
        result["post_change_reward_per_step"] = float(np.mean(rewards[start:start+window])) if len(rewards) > start else None
    return result


def evaluate_checkpoint(checkpoint_path, manifest_path, task, output_dir, metadata=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = checkpoint["actor"].to("cpu").eval()
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    definition = manifest["tasks"][task]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, metrics = [], {}
    with EvaluationTrajectories(output_dir / "test_trajectories.npz") as trajectories, \
            (output_dir / "test_episodes.jsonl").open("w") as output:
        for suite, configs in definition["suites"].items():
            suite_records = []
            for config in configs:
                if suite == "iid":
                    env = _build_ns_env(decode_config(config["config"]))
                else:
                    env = DynamicsEnv(task, definition["dr_ranges"], intervention=config["intervention"])
                try:
                    for seed in config["seeds"]:
                        # Rebuild iid update functions for every rollout seed.
                        if suite == "iid":
                            env.close()
                            env = _build_ns_env(decode_config(config["config"]))
                        trajectory = {}
                        record = {"suite": suite, "config_id": config["id"], "rollout_seed": seed,
                                  **episode(actor, checkpoint, env, task, seed,
                                            config.get("intervention"), trajectory,
                                            action_mode=manifest.get("action_mode", "greedy"))}
                        trajectories.write(record, trajectory)
                        output.write(json.dumps(record) + "\n")
                        suite_records.append(record)
                        records.append(record)
                finally:
                    env.close()
            for name, value in summarize([r["return"] for r in suite_records]).items():
                metrics[f"test/{suite}/{name}"] = value
            metrics[f"test/{suite}/changed_fraction"] = float(np.mean([r["parameter_changes"] > 0 for r in suite_records]))
            metrics[f"test/{suite}/episode_length_mean"] = float(np.mean([r["steps"] for r in suite_records]))
            metrics[f"test/{suite}/timeout_rate"] = float(np.mean([not r["terminated"] for r in suite_records]))
            metrics[f"test/{suite}/parameter_changes_mean"] = float(np.mean([r["parameter_changes"] for r in suite_records]))
            if task == 'cartpole':
                metrics[f'test/{suite}/physics_invalid_steps'] = sum(r['physics_invalid_steps'] for r in suite_records)
                metrics[f'test/{suite}/physics_constraint_rejections_mean'] = float(np.mean([
                    r['physics_constraint_rejections'] for r in suite_records]))
            if suite != "iid":
                metrics[f"test/{suite}/reached_change_fraction"] = float(np.mean([r["reached_change"] for r in suite_records]))
                metrics[f"test/{suite}/post_change_return"] = float(np.mean([r["post_change_return"] for r in suite_records]))
            if task == "frozenlake":
                metrics[f"test/{suite}/success_rate"] = float(np.mean([r["success"] for r in suite_records]))
                metrics[f"test/{suite}/hole_rate"] = float(np.mean([r["terminated"] and not r["success"] for r in suite_records]))
    # Matched counterfactual return differences include early terminations.
    controls = {(r["config_id"], r["rollout_seed"]): r for r in records if r["suite"] == "no_change"}
    for suite in ("abrupt", "gradual"):
        differences = [r["return"] - controls[r["config_id"], r["rollout_seed"]]["return"]
                       for r in records if r["suite"] == suite]
        metrics[f"test/{suite}/paired_return_difference"] = float(np.mean(differences))
    summary = {"metadata": metadata or {}, "action_mode": manifest.get("action_mode", "greedy"),
               "horizon": HORIZONS[task], "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "metrics": metrics}
    schema = {**TRAJECTORY_SCHEMA, "task": task,
              "action_mode": manifest.get("action_mode", "greedy"),
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
              "n_episodes": len(records)}
    (output_dir / "test_trajectory_metadata.json").write_text(json.dumps(schema, indent=2))
    (output_dir / "test_summary.json").write_text(json.dumps(summary, indent=2))
    return metrics


def aggregate(root, output, bootstrap=2000, replacements=None):
    """Seed-level CIs and paired method-minus-full differences; never pool seeds."""
    grouped = {}
    replacements = replacements or {}
    manifest_hashes = {}
    paths = [p for p in Path(root).glob("*/*/seed*/test_summary.json")
             if p.relative_to(root).parts[0] not in replacements]
    for task, replacement_root in replacements.items():
        paths.extend((Path(replacement_root) / task).glob("*/seed*/test_summary.json"))
    for path in paths:
        summary = json.loads(path.read_text())
        meta = summary["metadata"]
        key = (meta["task"], meta["method"])
        seeds = grouped.setdefault(key, {})
        if meta["seed"] in seeds:
            raise ValueError(f"Duplicate task/method/seed: {key}, {meta['seed']}")
        seeds[meta["seed"]] = summary["metrics"]
        public_summary = path.with_name("public_summary.json")
        if public_summary.exists():
            seeds[meta["seed"]].update(json.loads(public_summary.read_text())["metrics"])
        manifest_hashes.setdefault(meta["task"], set()).add(summary["manifest_sha256"])
    if not grouped or any(len(hashes) != 1 for hashes in manifest_hashes.values()):
        raise ValueError("Need one shared manifest within each task")
    rng = np.random.default_rng(91206)
    results = []
    for (task, method), seeds in sorted(grouped.items()):
        for metric in next(iter(seeds.values())):
            values = np.array([seeds[s][metric] for s in sorted(seeds)])
            draws = values[rng.integers(0, len(values), size=(bootstrap, len(values)))].mean(axis=1)
            record = {"task": task, "method": method, "metric": metric, "n_seeds": len(values),
                      "mean": float(values.mean()), "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
                      "manifest_sha256": next(iter(manifest_hashes[task])),
                      "source_root": str(Path(replacements.get(task, root)).resolve())}
            full = grouped.get((task, "full"), {})
            paired = sorted(set(seeds) & set(full))
            if paired:
                delta = np.array([seeds[s][metric] - full[s][metric] for s in paired])
                draws = delta[rng.integers(0, len(delta), size=(bootstrap, len(delta)))].mean(axis=1)
                record["difference_vs_full"] = {"n_pairs": len(paired), "mean": float(delta.mean()),
                                                "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}
            results.append(record)
    Path(output).write_text(json.dumps(results, indent=2))
    return results
