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
METHODS = ("stationary", "standard_dr", "random_ns", "plr", "full", "uniform_replay", "uniform_mutations")
MAIN_METHODS = METHODS[:5]
STATIC_METHODS = ("static_plr", "static_plr_mutations")
PROTOCOL = "plr-v2-async-fixed-start-v1"
METHOD_LABELS = {
    "stationary": "Stationary training",
    "standard_dr": "Static domain randomization",
    "random_ns": "Temporal domain randomization",
    "plr": "PLR",
    "full": "PLR with mutations",
    "uniform_replay": "Uniform replay ablation",
    "uniform_mutations": "Uniform replay with mutations ablation",
    "static_plr": "Static PLR",
    "static_plr_mutations": "Static PLR with mutations",
}
RUN_SLUGS = {"standard_dr": "static-dr", "random_ns": "temporal-dr", "full": "plr-mutations",
             "static_plr": "static-plr", "static_plr_mutations": "static-plr-mutations"}
TASKS = ("cartpole", "frozenlake", "ant")
SEEDS = (0, 1, 2, 3, 4)


def matrix(stage, methods=None):
    seeds = SEEDS if stage == "final" else (0,)
    return [(task, method, seed) for seed in seeds for task in TASKS for method in (methods or MAIN_METHODS)]


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
    manifest = Path(args.manifest).resolve()
    definition = json.loads(manifest.read_text())
    if definition.get("protocol") != PROTOCOL:
        raise ValueError("Prepare a new v2 asynchronous manifest; legacy protocols must use their frozen source")
    task, method, seed = definition["matrix"][args.index]
    output = Path(args.root).resolve() / args.stage / task / method / f"seed{seed}"
    if (output / "ppo_final.pt").exists():
        raise FileExistsError(f"Refusing to overwrite an existing trained run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if provenance()["source_sha256"] != definition["provenance"]["source_sha256"]:
        raise RuntimeError("Source differs from the frozen manifest; prepare a new release")
    plr = method in ("plr", "full", "uniform_replay", "uniform_mutations", *STATIC_METHODS)
    overrides = [f"seed={seed}", f"env.plr.enabled={str(plr).lower()}",
                 "+collector.synchronous=false",
                 f"env.ns_baseline={str(method == 'random_ns').lower()}",
                 f"env.plr.mutation_prob={0.25 if method in ('full', 'uniform_mutations', 'static_plr_mutations') else 0.0}",
                 f"training.checkpoint_dir={output}", f"hydra.run.dir={output / 'hydra'}",
                 f"wandb.project={args.project}", "wandb.entity=ml_exp", f"wandb.group={args.group}-{args.stage}",
                 f"wandb.name={task}-{RUN_SLUGS.get(method, method)}-seed{seed}",
                 f"wandb.tags=[paper,{args.stage},{task},{method}]", f"wandb.mode={args.wandb_mode}",
                 f"wandb.enabled={str(args.wandb_mode != 'disabled').lower()}",
                 f"wandb.dir={output}", f"+experiment.method={method}",
                 f"+experiment.method_label={json.dumps(METHOD_LABELS[method])}",
                 f"+experiment.comparison_role={'ablation' if method.startswith('uniform_') or definition.get('ablation_protocol') else 'main'}",
                 f"+experiment.stage={args.stage}",
                 f"+experiment.protocol={PROTOCOL}",
                 f"+experiment.manifest={manifest}", "+experiment.final_evaluation=true",
                 "env.plr.eval_seed=72301", "+experiment.public_evaluation=true",
                 f"+experiment.public_episodes={2 if args.stage == 'pilot' else 100}"]
    if task == "frozenlake":
        overrides.append(f"env.initial_mode={definition['training']['frozenlake_initial_mode']}")
    if definition.get('physics_protocol'):
        overrides.append(f"+experiment.physics_protocol={definition['physics_protocol']}")
    if definition.get('ablation_protocol'):
        overrides.append(f"+experiment.ablation_protocol={definition['ablation_protocol']}")
    if method in STATIC_METHODS:
        mutation_protocol = definition.get('static_mutation_protocol', 'range-clip-v1')
        revised = mutation_protocol == 'relative-reflect-symmetric-v2'
        overrides += ["+experiment.training_dynamics=static",
                      f"+env.plr.static_mutation_protocol={mutation_protocol}",
                      f"+experiment.static_mutation={mutation_protocol}",
                      f"+experiment.static_mutation_scale={'parent_magnitude' if revised else 'snapshot_marginal_range'}"]
        import copy
        from AAMAS_Comp.envs.static_curriculum import StaticDynamicsSampler
        samples = definition['training']['tasks'][task]['dr_samples']
        sampler = StaticDynamicsSampler(task, samples, mutation_protocol=mutation_protocol)
        protocol = copy.deepcopy(definition.get('extension', {}))
        protocol.update({'dynamics': 'fixed within every episode', 'snapshot_count': len(samples),
                         'snapshot_sha256': hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
                         'bounds': {key: [lo.tolist(), hi.tolist()] for key, (lo, hi) in sampler.bounds.items()},
                         'mutation_protocol': mutation_protocol,
                         'reflection': revised, 'symmetric_frozenlake': revised,
                         'reference_scales': sampler.reference_scales,
                         'relative_scale_floor_fraction': .01 if revised else None,
                         'mutation_sigma': 0.2, 'mutation_probability': 0.25 if method=='static_plr_mutations' else 0.0})
        (output / 'static_protocol.json').write_text(json.dumps(protocol, indent=2))
    if args.stage == "pilot":
        budgets = (dict(cartpole=256000, frozenlake=51200, ant=1024000)
                   if method in STATIC_METHODS else dict(cartpole=128000, frozenlake=51200, ant=512000))
        overrides += [f"collector.total_frames={budgets[task]}", "training.eval_ratio=1.0",
                      "env.plr.num_eval_configs=4", "training.save_interval=100000"]
    if plr:
        overrides += [f'+env.plr.audit_dir={output}/curriculum',
                      '+experiment.curriculum_audit=lineage-v1']
        overrides.append(f"+env.plr.protect_unscored={args.protect_unscored}")
        overrides.append(f"+env.plr.uniform_replay={str(method.startswith('uniform_')).lower()}")
    command = [sys.executable, str(REPO / "scripts/train.py"), "--config-name", f"config_{task}", *overrides]
    (output / "command.json").write_text(json.dumps(command, indent=2))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(config_name=f"config_{task}", overrides=overrides)
        if cfg.collector.synchronous or cfg.collector.num_groups != 2 or cfg.collector.num_envs != 24:
            raise ValueError("The approved protocol requires two asynchronous groups of 12")
        if plr and (cfg.env.plr.score_ema_alpha != 1 or not cfg.env.plr.replay_requires_score
                    or not cfg.env.plr.selection_staleness or not cfg.env.plr.protect_unscored):
            raise ValueError("The approved corrected PLR settings are not enabled")
        (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    print(json.dumps({"task": task, "method": method, "seed": seed,
                      "command": command, "source": definition["provenance"]}), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if not args.dry_run:
        subprocess.run(command, cwd=REPO, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--evaluation-manifest", required=True,
                         help="Preserve numerical test cases from this already-frozen manifest")
    prepare.add_argument("--include-uniform-ablations", action="store_true")
    prepare.add_argument("--frozenlake-initial-mode", choices=("uniform", "random"), default="uniform")
    prepare.add_argument("--calibration-count", type=int, default=256)
    prepare.add_argument("--calibration-seed", type=int, default=71311,
                         help="Independent of the preserved evaluation calibration seed 71301")
    runner = sub.add_parser("run")
    runner.add_argument("--index", required=True, type=int)
    runner.add_argument("--stage", choices=("pilot", "final"), required=True)
    runner.add_argument("--root", required=True)
    runner.add_argument("--manifest", required=True)
    runner.add_argument("--project", default="nsgym-paper")
    runner.add_argument("--group", default="plr-v2-20261017")
    runner.add_argument("--dry-run", action="store_true", help="Resolve/save config and command without starting training")
    runner.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    runner.add_argument("--protect-unscored", choices=("true", "false"), default="true",
                        help="Disable only for the delayed-feedback control experiment")
    agg = sub.add_parser("aggregate")
    agg.add_argument("--root", required=True)
    agg.add_argument("--manifest", required=True)
    agg.add_argument("--output", required=True)
    agg.add_argument("--wandb", action="store_true")
    agg.add_argument("--group", default="dr-plr-20261007-final")
    agg.add_argument("--artifact-name", default="dr-plr-comparison")
    agg.add_argument("--replace-task", choices=TASKS)
    agg.add_argument("--replacement-root")
    args = parser.parse_args()
    if args.command == "prepare":
        from AAMAS_Comp.evaluation.paper import training_distribution
        if Path(args.manifest).exists() or Path(args.manifest).with_name("pilot_manifest.json").exists():
            raise FileExistsError("Refusing to replace a frozen campaign manifest")
        source = Path(args.evaluation_manifest).resolve()
        raw = source.read_bytes()
        definition = json.loads(raw)
        methods = METHODS if args.include_uniform_ablations else MAIN_METHODS
        definition["methods"] = methods
        definition["training"] = training_distribution(args.calibration_count, seed=args.calibration_seed,
            frozenlake_initial_mode=args.frozenlake_initial_mode)
        definition["evaluation_source"] = {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest()}
        definition["evaluation_cases_sha256"] = hashlib.sha256(
            json.dumps(definition["tasks"], sort_keys=True).encode()).hexdigest()
        definition["action_mode"] = "stochastic"
        definition["protocol"] = PROTOCOL
        definition["public_protocol"] = {"action_mode": "stochastic", "episodes_per_notification": 100, "horizons": {"cartpole": 500, "frozenlake": 100, "ant": 1000}}
        definition["provenance"] = provenance()
        definition["matrix"] = matrix("final", methods)
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(definition, indent=2))
        pilot = json.loads(json.dumps(definition))
        pilot["n_configs"], pilot["rollouts"] = 2, 1
        pilot["matrix"] = matrix("pilot", methods)
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
        if bool(args.replace_task) != bool(args.replacement_root):
            parser.error("--replace-task and --replacement-root must be supplied together")
        replacements = {args.replace_task: args.replacement_root} if args.replace_task else {}
        definition = json.loads(Path(args.manifest).read_text())
        expected = [Path(replacements.get(task, args.root)) / task / method / f"seed{seed}" / "test_summary.json"
                    for task, method, seed in definition["matrix"]]
        missing = [str(p) for p in expected if not p.exists()]
        if missing:
            raise RuntimeError(f"Campaign incomplete: {len(missing)} of {len(expected)} runs missing")
        results = aggregate(args.root, args.output, replacements=replacements)
        if args.wandb:
            import wandb
            with wandb.init(entity="ml_exp", project="nsgym-paper", group=args.group, job_type="aggregate"):
                columns = ["task", "method", "metric", "n_seeds", "mean", "ci95", "difference_vs_full"]
                wandb.log({"comparison": wandb.Table(columns=columns,
                    data=[[r.get(c) for c in columns] for r in results])})
                artifact = wandb.Artifact(args.artifact_name, type="paper-results")
                artifact.add_file(args.output)
                wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
