"""Reduced-worker Ant validation only; never starts an Ant full-budget run."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    root, release = args.root.resolve(), args.release.resolve()
    if root.exists():
        raise FileExistsError('Use a new validation directory')
    root.mkdir(parents=True)
    sys.path.insert(0, str(release/'src'))
    sys.path.insert(0, str(release/'scripts'))
    campaign = load_module('ant_validation_campaign', release/'scripts/paper_campaign.py')
    parent = json.loads(args.parent.read_text())
    definition = deepcopy(parent)
    definition.update(matrix=[['ant', 'static_plr_mutations', 0]],
                      methods=['static_plr_mutations'],
                      static_mutation_protocol='relative-reflect-symmetric-v2',
                      ablation_protocol='ant-static-mutation-v2-validation-only',
                      provenance=campaign.provenance(), n_configs=4, rollouts=1)
    definition['tasks'] = {'ant': definition['tasks']['ant']}
    definition['training']['tasks'] = {'ant': definition['training']['tasks']['ant']}
    for suite, rows in definition['tasks']['ant']['suites'].items():
        definition['tasks']['ant']['suites'][suite] = rows[:4]
        for row in definition['tasks']['ant']['suites'][suite]:
            row['seeds'] = row['seeds'][:1]
    definition['evaluation_cases_sha256'] = hashlib.sha256(json.dumps(
        definition['tasks'], sort_keys=True).encode()).hexdigest()
    definition['validation_only'] = {'environments': 2, 'asynchronous_groups': 2,
                                   'full_training_environments_unchanged': 24,
                                   'full_budget_run': False, 'pilot_frames': 1024000}
    definition['extension'] = {'description': 'Ant reduced-environment validation only',
                               'mutation_rule': 'Relative scalar noise and reflection',
                               'full_budget_run': False}
    definition['execution'] = {'backend': 'local_cpu', 'multiprocessing_start_method': 'fork',
                               'environments': 2, 'asynchronous_groups': 2}
    manifest = root/'pilot_manifest.json'
    manifest.write_text(json.dumps(definition, indent=2))
    campaign.run(SimpleNamespace(stage='pilot', index=0, root=root, manifest=manifest,
                                 project='nsgym-paper', group=root.name, wandb_mode='online',
                                 protect_unscored='true', dry_run=True))
    folder = root/'pilot/ant/static_plr_mutations/seed0'
    command = json.loads((folder/'command.json').read_text())
    command += ['collector.num_envs=2', '+experiment.validation_only=true',
                '+experiment.validation_env_count=2']
    # Record the actual validation configuration, rather than leaving the
    # preceding 24-env dry-run config beside a two-worker checkpoint.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(release/'config'), version_base=None):
        cfg = compose(config_name='config_ant', overrides=command[4:])
        assert cfg.collector.num_envs == 2 and cfg.collector.num_groups == 2
        assert not cfg.collector.synchronous and cfg.collector.total_frames == 1024000
        (folder/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (folder/'command.json').write_text(json.dumps(command, indent=2))
    subprocess.run(command, cwd=release, check=True)
    meta = json.loads((folder/'run.json').read_text())
    metrics = json.loads((folder/'training_metrics_summary.json').read_text())['metrics']
    assert 1024000 <= meta['actual_frames'] < 1027000
    assert meta['config']['collector']['num_envs'] == 2
    assert metrics['plr/workers_reporting'] == 2 and metrics['plr/worker_coverage'] == 1
    assert metrics['plr/applied_scores_observed_total'] > 0
    assert metrics['plr/unscored_evictions_observed_total'] == 0
    assert metrics['plr/unscored_priority_mass'] == 0
    assert metrics['accel/mutation_count_observed_total'] >= 10
    assert metrics['plr/mutated_replay_count_observed_total'] > 0
    summary_module = load_module('ant_curriculum_summary',
                                 release/'scripts/diagnostics/mutation_audit/summarize_curriculum.py')
    summary = summary_module.summarize(folder/'curriculum')
    assert summary['by_origin']['mutated']['completed'] > 0
    from AAMAS_Comp.envs.static_curriculum import StaticDynamicsConfig, build_static_level
    from AAMAS_Comp.envs.domain_randomization import parameters
    # Validate every generated level, then step real fresh/mutated/replayed
    # examples to verify actual MuJoCo physics, not just configuration fields.
    samples = {}
    bounds = json.loads((folder/'static_protocol.json').read_text())['bounds']
    for path in (folder/'curriculum').glob('*.jsonl'):
        levels = {}
        for line in path.open():
            event = json.loads(line)
            if event['event'] == 'level':
                values = event['config']['values']
                assert np.isfinite(list(values.values())).all()
                assert values['torso_mass'] > 1e-6
                for key, (lo, hi) in bounds.items():
                    assert lo <= values[key] <= hi
                levels[event['uid']] = event['config']
                samples.setdefault(event['origin'], event['config'])
            elif event['event'] == 'selection' and event['replay']:
                samples.setdefault('replay', levels[event['uid']])
    assert set(samples) == {'fresh', 'mutated', 'replay'}
    physics_checks = []
    for label, config in samples.items():
        env = build_static_level(StaticDynamicsConfig(**config))
        try:
            env.reset(seed=84711)
            before = parameters(env, 'ant')
            for step in range(50):
                _, reward, done, truncated, _ = env.step(np.zeros(env.action_space.shape))
                after = parameters(env, 'ant')
                assert before == after and before == config['values']
                assert np.isfinite(reward)
                if done or truncated:
                    break
            physics_checks.append({'origin': label, 'steps': step+1, 'values': before})
        finally:
            env.close()
    import torch
    checkpoint = torch.load(folder/'ppo_final.pt', map_location='cpu', weights_only=False)
    assert all(torch.isfinite(v).all() for v in checkpoint['actor'].state_dict().values())
    for name in ('test_summary.json', 'public_summary.json'):
        s = json.loads((folder/name).read_text())
        assert s['action_mode'] == 'stochastic' and s['horizon'] == 1000
    import wandb
    info = meta['wandb_run']
    run = wandb.Api(timeout=60).run(f"{info['entity']}/{info['project']}/{info['id']}")
    assert run.state == 'finished'
    artifacts = [a for a in run.logged_artifacts() if a.type == 'paper-run']
    assert any(any(k.startswith('curriculum/') for k in a.manifest.entries) for a in artifacts)
    result = dict(passed=True, validation_only=True, environments=2, asynchronous_groups=2,
                  full_protocol_environments=24, actual_frames=meta['actual_frames'],
                  source_sha256=definition['provenance']['source_sha256'],
                  mutations=metrics['accel/mutation_count_observed_total'],
                  mutant_replays=metrics['plr/mutated_replay_count_observed_total'],
                  physics_checks=physics_checks, curriculum=summary, wandb=info,
                  raw_logs_uploaded=True, full_budget_ant_runs_started=0)
    (root/'validation_result.json').write_text(json.dumps(result, indent=2))
    print('PASS ANT REDUCED-ENV VALIDATION', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
