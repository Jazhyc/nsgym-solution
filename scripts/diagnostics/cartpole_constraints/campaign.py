"""Matched seven-method CartPole retraining with native physical constraints."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'scripts'))
import paper_campaign as campaign

PHYSICS = 'native-cartpole-constraints-v1'
METHODS = (*campaign.MAIN_METHODS, *campaign.STATIC_METHODS)
PYTHON = '/scratch/s4325621/venvs/nsgym-paper/bin/python'
TESTS = ('test_cartpole_constraints test_static_plr test_complete_episode_training '
         'test_paper_protocol test_plr_collector_feedback test_training_metrics '
         'test_evaluation_trajectories test_frozenlake_diagnostics test_interface '
         'test_corrected_protocol test_plr_sampling_integrity test_async_prelaunch_protocol').split()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


def corrected_definition(original, count=256):
    import numpy as np
    from AAMAS_Comp.evaluation.paper import calibrate_ranges
    from AAMAS_Comp.envs.constraints import valid_cartpole_parameter
    definition = deepcopy(original)
    train_seed = original['training']['calibration_seed']
    test_seed = original['calibration_seed']
    train_ranges, defaults, train_samples = calibrate_ranges('cartpole', count, train_seed)
    test_ranges, test_defaults, test_samples = calibrate_ranges('cartpole', count, test_seed)
    assert defaults == test_defaults
    for samples in (train_samples, test_samples):
        assert len(samples) == count*16
        assert all(valid_cartpole_parameter(k, v) for row in samples for k, v in row.items())
    task = deepcopy(original['tasks']['cartpole'])
    task.update(dr_ranges=test_ranges, dr_samples=test_samples, defaults=defaults)
    rng = np.random.default_rng(91204)
    suites = task['suites']
    assert all(len(suites[s]) == len(suites['no_change']) for s in ('abrupt', 'gradual'))
    for index in range(len(suites['no_change'])):
        target = deepcopy(test_samples[int(rng.integers(len(test_samples)))])
        for suite in ('abrupt', 'gradual', 'no_change'):
            intervention = suites[suite][index]['intervention']
            intervention.update(initial=deepcopy(defaults), target=deepcopy(target))
    # Same IID configurations, episode seeds, horizons, intervention times and
    # public example. Recalibrated controlled targets define a new benchmark.
    assert suites['iid'] == original['tasks']['cartpole']['suites']['iid']
    definition['tasks'] = {'cartpole': task}
    definition['training']['tasks'] = {'cartpole': {
        'dr_ranges': train_ranges, 'defaults': defaults, 'dr_samples': train_samples}}
    definition['training']['calibration_count'] = count
    definition.update(physics_protocol=PHYSICS, methods=list(METHODS),
        matrix=[('cartpole', method, seed) for seed in range(5) for method in METHODS],
        action_mode='stochastic', calibration_count=count,
        dr_range_rule='Joint time snapshots with native CartPole validity; invalid updates rejected before advancing state')
    definition['evaluation_cases_sha256'] = digest(definition['tasks'])
    definition['physics_correction'] = {
        'rules': 'finite parameters; masspole/masscart/length > 0; gravity >= 0; reject invalid proposals',
        'training_snapshot_count': len(train_samples), 'training_snapshot_sha256': digest(train_samples),
        'test_snapshot_count': len(test_samples), 'test_snapshot_sha256': digest(test_samples),
        'training_calibration_seed': train_seed, 'test_calibration_seed': test_seed,
        'iid_configs_preserved': True, 'controlled_targets_recalibrated': True,
        'legacy_results_are_separate': True}
    return definition


def prepare(root, parent, static_parent):
    from omegaconf import OmegaConf
    if (root/'manifest.json').exists():
        raise FileExistsError('Refusing to overwrite a frozen campaign')
    original = json.loads((parent/'manifest.json').read_text())
    assert original['protocol'] == campaign.PROTOCOL
    definition = corrected_definition(original)
    definition['evaluation_source'] = {'path': str(parent/'manifest.json'),
        'sha256': hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest()}
    definition['provenance'] = campaign.provenance()
    save(root/'manifest.json', definition)
    pilot = deepcopy(definition)
    pilot.update(n_configs=4, rollouts=1, matrix=[('cartpole', method, 0) for method in METHODS])
    for suite, configs in pilot['tasks']['cartpole']['suites'].items():
        pilot['tasks']['cartpole']['suites'][suite] = configs[:4]
        for config in pilot['tasks']['cartpole']['suites'][suite]:
            config['seeds'] = config['seeds'][:1]
    pilot['evaluation_cases_sha256'] = digest(pilot['tasks'])
    save(root/'pilot_manifest.json', pilot)
    for i, (task, method, seed) in enumerate(definition['matrix']):
        campaign.run(SimpleNamespace(stage='final', index=i, root=root, manifest=root/'manifest.json',
            project='nsgym-paper', group=root.name, wandb_mode='online', protect_unscored='true', dry_run=True))
        path = root/'final'/task/method/f'seed{seed}'/'resolved_config.yaml'
        reference_root = static_parent if method in campaign.STATIC_METHODS else parent
        reference = reference_root/'final'/task/method/f'seed{seed}'/'resolved_config.yaml'
        cfg, old = [OmegaConf.to_container(OmegaConf.load(p), resolve=True) for p in (path, reference)]
        for key in ('agent', 'collector', 'env'):
            assert cfg[key] == old[key], (method, key)
        for key, value in old['training'].items():
            if key != 'checkpoint_dir':
                assert cfg['training'][key] == value, (method, key)
        assert cfg['collector']['total_frames'] == 10000000
        assert cfg['experiment']['physics_protocol'] == PHYSICS
    save(root/'launch_plan.json', {'full_runs': 35, 'methods': list(METHODS),
        'source_sha256': campaign.provenance()['source_sha256'],
        'physics_correction': definition['physics_correction'], 'ppo_configs_match_legacy': True})
    print('PASS: all 35 configs matched; valid shared training/test pools; original IID cases preserved')


def submit_job(root, name, command, dependency=None, array=None, gate=False, minutes=480):
    prefix = (f'export PYTHONPATH={root}/release/src:{root}/release; '
              'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl; '
              f'export WANDB_CACHE_DIR=/scratch/s4325621/wandb_cache WANDB_DATA_DIR={root}/wandb_data WANDB_INIT_TIMEOUT=120; '
              f'cd {root}/release; ')
    args = ['sbatch', '--parsable', f'--job-name={name}', '--partition=regularshort',
            f'--time={minutes}', '--cpus-per-task='+('4' if gate else '32'),
            '--mem='+('8G' if gate else '32G'),
            f'--output={root}/slurm/%x-%A_%a.out', f'--error={root}/slurm/%x-%A_%a.err']
    if dependency:
        args += ['--dependency=afterok:'+dependency, '--kill-on-invalid-dep=yes']
    if array:
        args += ['--array='+array]
    return subprocess.check_output(args+['--wrap', prefix+command], text=True).strip().split(';')[0]


def pilot(root, index):
    # Reuse the standard runtime checks and verify the recorded actual physics.
    from diagnostics.cartpole_constraints.audit_trajectories import audit
    subprocess.run([PYTHON, str(REPO/'scripts/diagnostics/prelaunch/pilot.py'), str(root), str(index)], check=True)
    definition = json.loads((root/'pilot_manifest.json').read_text())
    task, method, seed = definition['matrix'][index]
    folder = root/'pilot'/task/method/f'seed{seed}'
    result = audit(folder/'test_trajectories.npz')
    assert all(s['native_constraint_violation_episodes'] == 0 for s in result['suites'].values())
    metadata = json.loads((folder/'run.json').read_text())
    assert metadata['config']['experiment']['physics_protocol'] == PHYSICS
    save(folder/'physics_verified.json', result)
    print('PASS: corrected physics checked in every saved pilot state')


def launch(root):
    definition = json.loads((root/'manifest.json').read_text())
    assert definition['physics_protocol'] == PHYSICS
    assert definition['provenance']['source_sha256'] == campaign.provenance()['source_sha256']
    checked = []
    for method in METHODS:
        folder = root/'pilot/cartpole'/method/'seed0'
        checked.append(json.loads((folder/'pilot_verified.json').read_text()))
        physics = json.loads((folder/'physics_verified.json').read_text())
        assert all(x['native_constraint_violation_episodes'] == 0 for x in physics['suites'].values())
    path = root/'full_jobs.json'
    if path.exists():
        raise FileExistsError('Full launch already attempted')
    save(root/'readiness.json', {'ready': True, 'physics_protocol': PHYSICS, 'pilots': checked,
        'source_sha256': campaign.provenance()['source_sha256']})
    jobs = {'waves': [], 'physics_protocol': PHYSICS, 'source_sha256': campaign.provenance()['source_sha256']}
    save(path, jobs)
    previous = None
    for seed in range(5):
        indices = [i for i, (_, _, s) in enumerate(definition['matrix']) if s == seed]
        command = (f'{PYTHON} scripts/paper_campaign.py run --root {root} --manifest {root}/manifest.json '
                   f'--stage final --group {root.name} --index $SLURM_ARRAY_TASK_ID')
        previous = submit_job(root, f'cp_native_seed{seed}', command, previous,
            ','.join(map(str, indices))+('%7' if seed==0 else '%4'))
        jobs['waves'].append({'seed': seed, 'job': previous, 'indices': indices})
        save(path, jobs)
        if seed == 0:
            jobs['preliminary'] = submit_job(root, 'cp_native_seed0_results',
                f'{PYTHON} scripts/diagnostics/cartpole_constraints/campaign.py preliminary {root}',
                previous, gate=True, minutes=30)
            save(path, jobs)
    command = (f'{PYTHON} scripts/paper_campaign.py aggregate --root {root}/final '
               f'--manifest {root}/manifest.json --output {root}/comparison.json --wandb '
               f'--group {root.name}-final --artifact-name cartpole-native-constraints-comparison')
    jobs['aggregate'] = submit_job(root, 'cp_native_aggregate', command, previous, gate=True, minutes=30)
    save(path, jobs)
    print('LAUNCHED', json.dumps(jobs))


def preliminary(root):
    import wandb
    rows = []
    for method in METHODS:
        folder = root/'final/cartpole'/method/'seed0'
        public = json.loads((folder/'public_summary.json').read_text())
        broad = json.loads((folder/'test_summary.json').read_text())
        assert public['action_mode'] == broad['action_mode'] == 'stochastic'
        m = broad['metrics']
        assert all(m[f'test/{s}/physics_invalid_steps'] == 0 for s in ('iid', 'abrupt', 'gradual', 'no_change'))
        rows.append({'method': method, 'training_seed': 0, 'public': public['metrics']['public/pooled/mean'],
                     **{s: m[f'test/{s}/mean'] for s in ('iid', 'abrupt', 'gradual', 'no_change')}})
    save(root/'seed0_results.json', {'physics_protocol': PHYSICS, 'n_training_seeds': 1, 'results': rows})
    with wandb.init(entity='ml_exp', project='nsgym-paper', group=root.name+'-seed0-results', job_type='aggregate'):
        columns = list(rows[0])
        wandb.log({'seed0_comparison': wandb.Table(columns=columns, data=[[r[k] for k in columns] for r in rows])})
        artifact = wandb.Artifact('cartpole-native-seed0-results', type='paper-results')
        artifact.add_file(str(root/'seed0_results.json'))
        wandb.log_artifact(artifact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['submit', 'prepare', 'pilot', 'launch', 'preliminary'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--static-parent', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'submit':
        assert args.parent and args.static_parent
        (root/'slurm').mkdir()
        tests = ' '.join('tests/'+t+'.py' for t in TESTS)
        jobs = {'tests': submit_job(root, 'cp_native_tests', f'{PYTHON} -m pytest -q {tests}', gate=True, minutes=30)}
        save(root/'jobs.json', jobs)
        command = (f'{PYTHON} scripts/diagnostics/cartpole_constraints/campaign.py prepare {root} '
                   f'--parent {args.parent.resolve()} --static-parent {args.static_parent.resolve()}')
        jobs['prepare'] = submit_job(root, 'cp_native_plan', command, jobs['tests'], gate=True, minutes=30)
        save(root/'jobs.json', jobs)
        command = f'{PYTHON} scripts/diagnostics/cartpole_constraints/campaign.py pilot {root} --index $SLURM_ARRAY_TASK_ID'
        jobs['pilots'] = submit_job(root, 'cp_native_pilots', command, jobs['prepare'], '0-6%3', minutes=30)
        save(root/'jobs.json', jobs)
        command = f'{PYTHON} scripts/diagnostics/cartpole_constraints/campaign.py launch {root}'
        jobs['verify_launch'] = submit_job(root, 'cp_native_verify_launch', command, jobs['pilots'], gate=True, minutes=30)
        save(root/'jobs.json', jobs)
        print(json.dumps(jobs))
    elif args.command == 'prepare':
        prepare(root, args.parent.resolve(), args.static_parent.resolve())
    elif args.command == 'pilot':
        pilot(root, args.index)
    else:
        globals()[args.command](root)


if __name__ == '__main__':
    main()
