"""Frozen 100M Ant comparison with bounded temporal and matched static physics."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
METHODS = ['stationary', 'random_ns', 'plr', 'full', 'standard_dr', 'static_plr', 'static_plr_mutations']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def source_hashes():
    paths = [*sorted((REPO / 'src').rglob('*.py')), *sorted((REPO / 'config').rglob('*.yaml')),
             REPO / 'scripts/train.py', Path(__file__).resolve()]
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(root):
    from AAMAS_Comp.envs.ant_curriculum import calibrate_ant_v2, ANT_BOUNDS
    from AAMAS_Comp.envs.ns_env_sampler import make_sampler
    if (root / 'manifest.json').exists():
        raise FileExistsError('Never overwrite a frozen campaign')
    training = calibrate_ant_v2(256, 71311)
    evaluation = calibrate_ant_v2(256, 71301)
    sampler = make_sampler('ant_v2', seed=91203)
    rng = np.random.default_rng(91204)
    suites = {key: [] for key in ('iid', 'abrupt', 'gradual', 'no_change')}
    for i in range(200):
        seeds = [int(np.random.SeedSequence([91205, i, r]).generate_state(1)[0] % (2**31)) for r in range(5)]
        suites['iid'].append({'id': i, 'seeds': seeds, 'config': asdict(sampler.sample())})
        target = evaluation['dr_samples'][int(rng.integers(len(evaluation['dr_samples'])))]
        for suite, duration in [('abrupt', 0), ('gradual', 250), ('no_change', -1)]:
            suites[suite].append({'id': i, 'seeds': seeds, 'intervention': {
                'initial': evaluation['defaults'], 'target': target, 'step': 100, 'duration': duration}})
    evaluation['suites'] = suites
    manifest = {'version': 2, 'protocol': 'ant-bounded-v2-100m', 'native_ant_reward': True,
                'created_at': datetime.now(timezone.utc).isoformat(), 'action_mode': 'stochastic',
                'n_configs': 200, 'rollouts': 5, 'test_seed': 91203,
                'training': {'calibration_seed': 71311, 'calibration_count': 256,
                             'tasks': {'ant': training}}, 'tasks': {'ant': evaluation},
                'bounds': ANT_BOUNDS, 'source_hashes': source_hashes(),
                'matrix': [{'index': i, 'method': method, 'seed': seed}
                           for i, (seed, method) in enumerate((s, m) for s in range(5) for m in METHODS)]}
    manifest['training_pool_sha256'] = hashlib.sha256(json.dumps(training['dr_samples'], sort_keys=True).encode()).hexdigest()
    manifest['evaluation_cases_sha256'] = hashlib.sha256(json.dumps(evaluation, sort_keys=True).encode()).hexdigest()
    save(root / 'manifest.json', manifest)
    print(json.dumps({'prepared': str(root), 'runs': 35, 'training_snapshots': len(training['dr_samples']),
                      'training_pool_sha256': manifest['training_pool_sha256']}), flush=True)


def command(root, row, *, frames=100000000, local=False, online=True):
    method, seed = row['method'], row['seed']
    folder = root / ('local' if local else 'final') / 'ant' / method / f'seed{seed}'
    plr = method in ('plr', 'full', 'static_plr', 'static_plr_mutations')
    overrides = [f'seed={seed}', f'env.plr.enabled={str(plr).lower()}',
                 f'env.ns_baseline={str(method == "random_ns").lower()}',
                 f'env.plr.mutation_prob={0.25 if method in ("full", "static_plr_mutations") else 0}',
                 f'collector.total_frames={frames}', f'training.checkpoint_dir={folder}',
                 f'hydra.run.dir={folder / "hydra"}', '+training.local_metrics=true',
                 'env.plr.eval_seed=72301', 'wandb.project=nsgym-paper', 'wandb.entity=ml_exp',
                 f'wandb.group={root.name}', f'wandb.name=ant-{method}-seed{seed}-100m',
                 f'wandb.tags=[ant,bounded-v2,100m,{method}]',
                 f'wandb.enabled={str(online).lower()}', 'wandb.mode=online', f'wandb.dir={folder}',
                 f'+experiment.method={method}', '+experiment.stage=final',
                 f'+experiment.manifest={root / "manifest.json"}',
                 f'+experiment.final_evaluation={str(not local).lower()}',
                 '+experiment.public_evaluation=true', '+experiment.public_episodes=100',
                 '+experiment.public_environment=ExampleNSAnt-v2', '+experiment.notification_modes=[none]']
    if plr:
        overrides += [f'+env.plr.audit_dir={folder / "curriculum"}']
    if local:
        overrides += ['collector.num_envs=2', 'collector.num_groups=1',
                      'collector.frames_per_batch=1024', 'num_threads=1',
                      'env.plr.num_eval_configs=4', 'training.num_eval_episodes=2',
                      'training.eval_ratio=0.5', 'training.save_interval=32']
    return [sys.executable, str(REPO / 'scripts/train.py'), '--config-name=config_ant_v2', *overrides], folder


def run(root, index, *, dry=False, frames=100000000, local=False, online=True):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['source_hashes'] != source_hashes():
        raise RuntimeError('Source differs from the frozen campaign')
    row = manifest['matrix'][index]
    cmd, folder = command(root, row, frames=frames, local=local, online=online)
    if (folder / 'ppo_final.pt').exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(REPO / 'config'), version_base=None):
        cfg = compose(config_name='config_ant_v2', overrides=cmd[3:])
    assert cfg.collector.synchronous and cfg.training.minibatch_kl_guard and cfg.training.nonnegative_kl_guard
    assert cfg.env.plr.sampler_key == 'ant_v2' and not cfg.agent.use_layer_norm
    from AAMAS_Comp.envs.torchrl_factory import static_training_definition, task_key
    assert task_key(cfg) == 'ant'
    if row['method'] in ('standard_dr', 'static_plr', 'static_plr_mutations'):
        definition = static_training_definition(cfg)
        assert len(definition['dr_samples']) == 4096
    (folder / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    save(folder / 'command.json', cmd)
    if dry:
        return
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO / 'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MUJOCO_GL='egl', WANDB_INIT_TIMEOUT='120',
               MPLCONFIGDIR=str(folder / 'cache/matplotlib'),
               WANDB_CACHE_DIR=str(folder / 'cache/wandb'), WANDB_DATA_DIR=str(folder / 'wandb_data'))
    if online and not env.get('WANDB_API_KEY'):
        import netrc
        auth = netrc.netrc(root / 'operations/wandb.netrc').authenticators('api.wandb.ai')
        if auth is None:
            raise RuntimeError('W&B credentials unavailable')
        env['WANDB_API_KEY'] = auth[2]
    status = {**row, 'job_id': env.get('SLURM_JOB_ID'), 'started_at': datetime.now(timezone.utc).isoformat()}
    save(folder / 'execution_status.json', {**status, 'state': 'running'})
    result = subprocess.run(cmd, cwd=REPO, env=env)
    save(folder / 'execution_status.json', {**status, 'state': 'finished' if result.returncode == 0 else 'failed',
                                          'returncode': result.returncode})
    raise SystemExit(result.returncode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'check', 'run'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--frames', type=int, default=100000000)
    p.add_argument('--local', action='store_true')
    p.add_argument('--offline', action='store_true')
    a = p.parse_args(); root = a.root.resolve()
    if a.action == 'prepare':
        prepare(root)
    elif a.action == 'check':
        for i in range(35):
            run(root, i, dry=True)
        print('PASS: all 35 resolved configurations', flush=True)
    else:
        run(root, a.index, frames=a.frames, local=a.local, online=not a.offline)


if __name__ == '__main__':
    main()
