"""Prepare, run, evaluate and aggregate the fixed-budget DR/PLR comparison.

This shared launcher adds the missing task/method/seed matrix and snapshot
provenance to the existing Hydra trainer. Scientific test logic lives in
AAMAS_Comp.evaluation.paper; Slurm only allocates resources.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
METHODS = ("stationary", "standard_dr", "random_ns", "plr", "full")
TASKS = ("cartpole", "frozenlake", "ant")
SEEDS = (0, 1, 2, 3, 4)


def matrix(stage):
    seeds = SEEDS if stage == "final" else (0,)
    return [(task, method, seed) for task in TASKS for method in METHODS for seed in seeds]


def provenance():
    paths = sorted(p for folder in ("src", "config", "scripts")
                   for p in (REPO / folder).rglob("*")
                   if p.is_file() and p.suffix in (".py", ".yaml", ".sbatch"))
    paths += [REPO / "pyproject.toml", REPO / "uv.lock"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                            capture_output=True, check=False)
    commit = result.stdout.strip() if result.returncode == 0 else None
    return {"source_sha256": digest.hexdigest(), "git_commit": commit,
            "note": "source digest includes uncommitted implementation changes"}


def run(args):
    task, method, seed = matrix(args.stage)[args.index]
    output = Path(args.root).resolve() / args.stage / task / method / f"seed{seed}"
    if (output / "ppo_final.pt").exists():
        raise FileExistsError(f"Refusing to overwrite an existing trained run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest).resolve()
    definition = json.loads(manifest.read_text())
    if provenance()["source_sha256"] != definition["provenance"]["source_sha256"]:
        raise RuntimeError("Source differs from the frozen manifest; prepare a new release")
    plr = method in ("plr", "full")
    overrides = [f"seed={seed}", f"env.plr.enabled={str(plr).lower()}",
                 f"env.ns_baseline={str(method == 'random_ns').lower()}",
                 f"env.plr.mutation_prob={0.25 if method == 'full' else 0.0}",
                 f"training.checkpoint_dir={output}", f"hydra.run.dir={output / 'hydra'}",
                 f"wandb.project={args.project}", f"wandb.group={args.group}-{args.stage}",
                 f"wandb.name={task}-{method}-seed{seed}",
                 f"wandb.tags=[paper,{args.stage},{task},{method}]", f"wandb.mode={args.wandb_mode}",
                 f"wandb.enabled={str(args.wandb_mode != 'disabled').lower()}",
                 f"wandb.dir={output}", f"+experiment.method={method}",
                 f"+experiment.stage={args.stage}",
                 f"+experiment.manifest={manifest}", "+experiment.final_evaluation=true",
                 "env.plr.eval_seed=72301"]
    if args.stage == "pilot":
        overrides += ["collector.total_frames=25600", "training.eval_ratio=1.0",
                      "env.plr.num_eval_configs=4", "training.save_interval=100000"]
    command = [sys.executable, str(REPO / "scripts/train.py"), "--config-name", f"config_{task}", *overrides]
    (output / "command.json").write_text(json.dumps(command, indent=2))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(config_name=f"config_{task}", overrides=overrides)
        (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    print(json.dumps({"task": task, "method": method, "seed": seed,
                      "command": command, "source": definition["provenance"]}), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=REPO, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--configs", type=int, default=200)
    prepare.add_argument("--rollouts", type=int, default=5)
    prepare.add_argument("--calibration-count", type=int, default=256)
    runner = sub.add_parser("run")
    runner.add_argument("--index", required=True, type=int)
    runner.add_argument("--stage", choices=("pilot", "final"), required=True)
    runner.add_argument("--root", required=True)
    runner.add_argument("--manifest", required=True)
    runner.add_argument("--project", default="nsgym-paper")
    runner.add_argument("--group", default="dr-plr-20261007")
    runner.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    agg = sub.add_parser("aggregate")
    agg.add_argument("--root", required=True)
    agg.add_argument("--output", required=True)
    agg.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        from AAMAS_Comp.evaluation.paper import make_manifest
        definition = make_manifest(args.manifest, args.configs, args.rollouts, args.calibration_count)
        definition["provenance"] = provenance()
        definition["matrix"] = matrix("final")
        Path(args.manifest).write_text(json.dumps(definition, indent=2))
        pilot = json.loads(json.dumps(definition))
        pilot["n_configs"], pilot["rollouts"] = 2, 1
        for task in pilot["tasks"].values():
            for suite, configs in task["suites"].items():
                task["suites"][suite] = configs[:2]
                for config in task["suites"][suite]:
                    config["seeds"] = config["seeds"][:1]
        pilot_path = Path(args.manifest).with_name("pilot_manifest.json")
        if pilot_path.exists():
            raise FileExistsError(pilot_path)
        pilot_path.write_text(json.dumps(pilot, indent=2))
        print(f"Frozen manifest: {args.manifest}; pilot: {pilot_path}")
    elif args.command == "run":
        run(args)
    else:
        from AAMAS_Comp.evaluation.paper import aggregate
        expected = [Path(args.root) / task / method / f"seed{seed}" / "test_summary.json"
                    for task, method, seed in matrix("final")]
        missing = [str(p) for p in expected if not p.exists()]
        if missing:
            raise RuntimeError(f"Campaign incomplete: {len(missing)} of 75 runs missing")
        results = aggregate(args.root, args.output)
        if args.wandb:
            import wandb
            with wandb.init(project="nsgym-paper", group="dr-plr-20261007-final", job_type="aggregate"):
                columns = ["task", "method", "metric", "n_seeds", "mean", "ci95", "difference_vs_full"]
                wandb.log({"comparison": wandb.Table(columns=columns,
                    data=[[r.get(c) for c in columns] for r in results])})
                artifact = wandb.Artifact("dr-plr-comparison", type="paper-results")
                artifact.add_file(args.output)
                wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
