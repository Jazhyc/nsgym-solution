"""Add 30 static-curriculum runs to an existing frozen 75-run campaign.

The parent release/results are never modified. Full jobs are submitted only
after cluster tests, all resolved configs and six exercised pilots pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'scripts'))
import paper_campaign as campaign

PYTHON = '/scratch/s4325621/venvs/nsgym-paper/bin/python'
TESTS = ('test_static_plr test_complete_episode_training test_paper_protocol '
         'test_plr_collector_feedback test_training_metrics test_evaluation_trajectories '
         'test_frozenlake_diagnostics test_interface test_corrected_protocol '
         'test_plr_sampling_integrity test_async_prelaunch_protocol').split()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


def submit(root, name, command, dependency=None, array=None, ant=False, gate=False):
    prefix = (f'export PYTHONPATH={root}/release/src:{root}/release; '
              'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl; '
              f'export WANDB_CACHE_DIR=/scratch/s4325621/wandb_cache WANDB_DATA_DIR={root}/wandb_data WANDB_INIT_TIMEOUT=120; '
              f'cd {root}/release; ')
    args = ['sbatch', '--parsable', f'--job-name={name}',
            '--partition='+('regularlong' if ant else 'regularshort'),
            '--time='+('5-00:00:00' if ant else '08:00:00'),
            '--cpus-per-task='+('4' if gate else '32'),
            '--mem='+('8G' if gate else '96G' if ant or name=='static_pilots' else '32G'),
            f'--output={root}/slurm/%x-%A_%a.out', f'--error={root}/slurm/%x-%A_%a.err']
    if array:
        args += ['--array='+array]
    if dependency:
        args += ['--dependency=afterok:'+dependency, '--kill-on-invalid-dep=yes']
    return subprocess.check_output(args+['--wrap', prefix+command], text=True).strip().split(';')[0]


def prepare(root, parent):
    from omegaconf import OmegaConf
    definition = json.loads((parent/'manifest.json').read_text())
    assert definition['protocol'] == campaign.PROTOCOL
    assert not (root/'manifest.json').exists()
    extension = {'parent_root': str(parent),
                 'parent_source_sha256': definition['provenance']['source_sha256'],
                 'parent_manifest_sha256': hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest(),
                 'training_snapshot_hash': digest(definition['training']),
                 'description': 'Static PLR and static PLR+mutations; exact static-DR fresh joint prior; no schedules',
                 'mutation': 'Gaussian physical values, sigma .2 times empirical marginal width; clipped scalars, bounded-simplex P'}
    definition.update({'methods': list(campaign.STATIC_METHODS), 'matrix': campaign.matrix('final', campaign.STATIC_METHODS),
                       'extension': extension, 'provenance': campaign.provenance()})
    save(root/'manifest.json', definition)
    pilot = json.loads((parent/'pilot_manifest.json').read_text())
    assert pilot['training'] == definition['training']
    pilot.update({'methods': list(campaign.STATIC_METHODS), 'matrix': campaign.matrix('pilot', campaign.STATIC_METHODS),
                  'extension': extension, 'provenance': campaign.provenance()})
    save(root/'pilot_manifest.json', pilot)
    assert digest(definition['tasks']) == definition['evaluation_cases_sha256']
    for i, (task, method, seed) in enumerate(definition['matrix']):
        campaign.run(SimpleNamespace(stage='final', index=i, root=root, manifest=root/'manifest.json',
            project='nsgym-paper', group=root.name, wandb_mode='online', protect_unscored='true', dry_run=True))
        folder = root/'final'/task/method/f'seed{seed}'
        cfg = OmegaConf.to_container(OmegaConf.load(folder/'resolved_config.yaml'), resolve=True)
        reference = OmegaConf.to_container(OmegaConf.load(
            parent/'final'/task/'standard_dr'/f'seed{seed}'/'resolved_config.yaml'), resolve=True)
        for key in ('agent', 'collector'):
            assert cfg[key] == reference[key], (task, method, key)
        for key, value in reference['training'].items():
            if key != 'checkpoint_dir':
                assert cfg['training'][key] == value, (task, key)
        for key in ('normalize_obs', 'normalize_reward', 'frame_stack', 'frame_skip'):
            assert cfg['env'][key] == reference['env'][key]
        for key in ('capacity', 'replay_prob', 'score_temp', 'staleness_coef', 'mutation_sigma', 'num_eval_configs'):
            assert cfg['env']['plr'][key] == reference['env']['plr'][key]
        assert cfg['env']['plr']['enabled'] and cfg['env']['plr']['mutation_prob'] == (.25 if method.endswith('mutations') else 0)
    save(root/'launch_plan.json', {'n_runs': 30, 'source_sha256': campaign.provenance()['source_sha256'],
                                 'extension': extension, 'matched_parent_configs': True})
    # This command executes on a compute node, so config writes also verify the
    # actual output filesystem. Never use login-only /projects for active runs.
    print('PASS: 30 configs match parent static DR; exact training/evaluation manifests preserved')


def launch(root, parent):
    definition = json.loads((root/'manifest.json').read_text())
    plan = json.loads((root/'launch_plan.json').read_text())
    assert plan['n_runs'] == 30 and plan['source_sha256'] == campaign.provenance()['source_sha256']
    assert definition['provenance']['source_sha256'] == plan['source_sha256']
    original = json.loads((parent/'manifest.json').read_text())
    assert definition['training'] == original['training'] and definition['tasks'] == original['tasks']
    verified = []
    for task, method, seed in campaign.matrix('pilot', campaign.STATIC_METHODS):
        verified.append(json.loads((root/'pilot'/task/method/f'seed{seed}'/'pilot_verified.json').read_text()))
    assert len(verified) == 6 and all(p['replays'] >= 20 for p in verified)
    assert all(p['mutations'] >= 10 for p in verified if p['method'].endswith('mutations'))
    save(root/'readiness.json', {'ready': True, 'pilots': verified, 'source_sha256': plan['source_sha256'],
                               'parent_training_distribution_matched': True, 'full_runs': 30})
    path = root/'full_jobs.json'
    if path.exists():
        raise FileExistsError('Inspect existing full submission; do not duplicate jobs')
    jobs = {'waves': [], 'source_sha256': plan['source_sha256'], 'parent_root': str(parent)}
    save(path, jobs)
    previous = {}
    for seed in range(5):
        for task in campaign.TASKS:
            indices = [i for i, (t, m, s) in enumerate(definition['matrix']) if t == task and s == seed]
            command = (f'{PYTHON} scripts/paper_campaign.py run --root {root} --manifest {root}/manifest.json '
                       f'--stage final --group {root.name} --index $SLURM_ARRAY_TASK_ID')
            job = submit(root, f'static_{task}_seed{seed}', command, previous.get(task),
                         ','.join(map(str, indices))+'%2', ant=task=='ant')
            jobs['waves'].append({'task': task, 'seed': seed, 'job': job, 'indices': indices})
            previous[task] = job
            save(path, jobs)
    parent_jobs = json.loads((parent/'full_jobs.json').read_text())
    command = f'{PYTHON} scripts/diagnostics/static_plr/campaign.py aggregate {root} --parent {parent}'
    jobs['aggregate'] = submit(root, 'static_combined_results', command,
        ':'.join([*previous.values(), parent_jobs['aggregate']]), gate=True)
    save(path, jobs)
    print('LAUNCHED', json.dumps(jobs))


def aggregate(root, parent):
    import numpy as np
    import wandb
    a, b = [json.loads((p/'manifest.json').read_text()) for p in (root, parent)]
    assert a['training'] == b['training'] and a['tasks'] == b['tasks']
    data = {}
    for source, definition in ((parent, b), (root, a)):
        for task, method, seed in definition['matrix']:
            folder = source/'final'/task/method/f'seed{seed}'
            broad = json.loads((folder/'test_summary.json').read_text())
            public = json.loads((folder/'public_summary.json').read_text())
            assert broad['action_mode'] == public['action_mode'] == 'stochastic'
            assert broad['manifest_sha256'] == hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
            data.setdefault((task, method), {})[seed] = {**broad['metrics'], **public['metrics']}
    rng = np.random.default_rng(91206)
    records = []
    for (task, method), seeds in sorted(data.items()):
        assert set(seeds) == set(range(5))
        for metric in seeds[0]:
            values = np.array([seeds[s][metric] for s in range(5)])
            draws = values[rng.integers(0, 5, (2000, 5))].mean(1)
            baseline = data[(task, 'standard_dr')]
            delta = values - np.array([baseline[s][metric] for s in range(5)])
            differences = delta[rng.integers(0, 5, (2000, 5))].mean(1)
            records.append({'task': task, 'method': method, 'metric': metric, 'n_seeds': 5,
                            'mean': float(values.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist(),
                            'difference_vs_static_dr': float(delta.mean()),
                            'difference_ci95': np.quantile(differences, [.025, .975]).tolist()})
    save(root/'combined_comparison.json', records)
    with wandb.init(entity='ml_exp', project='nsgym-paper', group=root.name+'-combined', job_type='aggregate'):
        columns = ['task', 'method', 'metric', 'n_seeds', 'mean', 'ci95', 'difference_vs_static_dr', 'difference_ci95']
        wandb.log({'comparison': wandb.Table(columns=columns, data=[[r[k] for k in columns] for r in records])})
        artifact = wandb.Artifact('static-temporal-plr-comparison', type='paper-results')
        artifact.add_file(str(root/'combined_comparison.json'))
        artifact.add_file(str(root/'manifest.json'), name='static_manifest.json')
        artifact.add_file(str(parent/'manifest.json'), name='temporal_manifest.json')
        wandb.log_artifact(artifact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['submit', 'prepare', 'launch', 'aggregate'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--parent', required=True, type=Path)
    args = parser.parse_args()
    root, parent = args.root.resolve(), args.parent.resolve()
    if args.command == 'submit':
        (root/'slurm').mkdir()
        jobs = {}
        tests = ' '.join('tests/'+t+'.py' for t in TESTS)
        jobs['tests'] = submit(root, 'static_tests', f'{PYTHON} -m pytest -q {tests}', gate=True)
        save(root/'jobs.json', jobs)
        command = f'{PYTHON} scripts/diagnostics/static_plr/campaign.py prepare {root} --parent {parent}'
        jobs['prepare'] = submit(root, 'static_plan', command, jobs['tests'], gate=True)
        save(root/'jobs.json', jobs)
        command = f'{PYTHON} scripts/diagnostics/prelaunch/pilot.py {root} $SLURM_ARRAY_TASK_ID'
        jobs['pilots'] = submit(root, 'static_pilots', command, jobs['prepare'], '0-5%3')
        save(root/'jobs.json', jobs)
        command = f'{PYTHON} scripts/diagnostics/static_plr/campaign.py launch {root} --parent {parent}'
        jobs['verify_launch'] = submit(root, 'static_verify_launch', command, jobs['pilots'], gate=True)
        save(root/'jobs.json', jobs)
        print(json.dumps(jobs))
    else:
        globals()[args.command](root, parent)


if __name__ == '__main__':
    main()
