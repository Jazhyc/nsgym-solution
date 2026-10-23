"""Three-seed, 500M Ant extension using the frozen 100M dynamics/evaluation pool."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import netrc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import ant_campaign_v2 as base


def hashes():
    return {**base.source_hashes(), 'scripts/ant_campaign_500m.py':
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def prepare(root, previous):
    if (root / 'manifest.json').exists():
        raise FileExistsError(root)
    manifest = json.loads((previous / 'manifest.json').read_text())
    manifest.update(protocol='ant-bounded-v2-500m-3seeds',
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_hashes=hashes(), previous_campaign=str(previous),
                    total_frames=500000000, lr_decay_frames=100000000,
                    resume_semantics='Learner restored; worker/RNG/replay state restarted')
    rows = []
    for seed in range(3):
        for method in base.METHODS:
            folder = previous / 'final/ant' / method / f'seed{seed}'
            checkpoint = folder / 'ppo_final.pt'
            if not checkpoint.exists():
                periodic = sorted(folder.glob('ppo_iter_*.pt'),
                                  key=lambda p: int(p.stem.split('_')[-1]))
                checkpoint = periodic[-1] if periodic else None
            row = dict(index=len(rows), method=method, seed=seed,
                       resume_checkpoint=str(checkpoint) if checkpoint else None)
            if checkpoint:
                row['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            rows.append(row)
    manifest['matrix'] = rows
    base.save(root / 'manifest.json', manifest)
    print(json.dumps({'prepared': str(root), 'runs': len(rows),
                      'resumed': sum(bool(r['resume_checkpoint']) for r in rows),
                      'training_pool_sha256': manifest['training_pool_sha256']}), flush=True)


def command(root, row):
    cmd, folder = base.command(root, row, frames=500000000)
    cmd = [s.replace('-100m', '-500m').replace('bounded-v2,100m,', 'bounded-v2,500m,')
           if s.startswith(('wandb.name=', 'wandb.tags=')) else s for s in cmd]
    cmd += ['training.eval_ratio=0.02', '+training.lr_decay_frames=100000000',
            '+experiment.log_artifact=false']
    if row['resume_checkpoint']:
        cmd += [f'+training.resume_checkpoint={row["resume_checkpoint"]}']
    return cmd, folder


def stop_group(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run(root, index, dry=False):
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from AAMAS_Comp.agents.recovery import validate_models
    from AAMAS_Comp.envs.torchrl_factory import static_training_definition, task_key
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['source_hashes'] != hashes():
        raise RuntimeError('Source differs from frozen extension')
    row = manifest['matrix'][index]
    cmd, folder = command(root, row)
    if (folder / 'ppo_final.pt').exists():
        raise FileExistsError(folder)
    restored_frames = 0
    if row['resume_checkpoint']:
        p = Path(row['resume_checkpoint'])
        if hashlib.sha256(p.read_bytes()).hexdigest() != row['checkpoint_sha256']:
            raise RuntimeError('Recovery checkpoint changed')
        torch.set_num_threads(1)
        checkpoint = torch.load(p, map_location='cpu', weights_only=False)
        validate_models(checkpoint)
        state = checkpoint['training_state']
        assert state['optimizer']['state'] and state['reward_rms'] is not None
        restored_frames = int(state['global_step'])
        assert 0 < restored_frames < 500000000
        del checkpoint
    with initialize_config_dir(config_dir=str(base.REPO / 'config'), version_base=None):
        cfg = compose(config_name='config_ant_v2', overrides=cmd[3:])
    assert cfg.collector.total_frames == 500000000 and cfg.training.lr_decay_frames == 100000000
    assert cfg.collector.synchronous and cfg.training.minibatch_kl_guard and cfg.training.nonnegative_kl_guard
    assert task_key(cfg) == 'ant' and not cfg.experiment.log_artifact
    assert cfg.wandb.enabled and cfg.wandb.mode == 'online'
    if row['method'] in ('standard_dr', 'static_plr', 'static_plr_mutations'):
        assert len(static_training_definition(cfg)['dr_samples']) == 4096
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    base.save(folder / 'command.json', cmd)
    if dry:
        print(json.dumps({**row, 'restored_frames': restored_frames, 'validated': True}), flush=True)
        return
    env = os.environ.copy()
    env.update(PYTHONPATH=str(base.REPO / 'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MUJOCO_GL='egl', WANDB_INIT_TIMEOUT='120',
               MPLCONFIGDIR=str(folder / 'cache/matplotlib'),
               WANDB_CACHE_DIR=str(folder / 'cache/wandb'), WANDB_DATA_DIR=str(folder / 'wandb_data'))
    if not env.get('WANDB_API_KEY'):
        auth = netrc.netrc(root / 'operations/wandb.netrc').authenticators('api.wandb.ai')
        if auth is None:
            raise RuntimeError('W&B credentials unavailable')
        env['WANDB_API_KEY'] = auth[2]
    status = {**row, 'restored_frames': restored_frames, 'job_id': env.get('SLURM_JOB_ID'),
              'started_at': datetime.now(timezone.utc).isoformat()}
    base.save(folder / 'execution_status.json', {**status, 'state': 'running'})
    proc = subprocess.Popen(cmd, cwd=base.REPO, env=env, start_new_session=True)
    complete_since = failure_since = None
    try:
        while proc.poll() is None:
            complete = all((folder / name).exists() for name in
                           ('ppo_final.pt', 'run.json', 'test_summary.json', 'public_summary.json'))
            failed = (folder / 'nonfinite_failure.json').exists()
            now = time.monotonic()
            if complete and complete_since is None:
                complete_since = now
            if failed and failure_since is None:
                failure_since = now
            if failure_since is not None and now - failure_since > 10:
                stop_group(proc)
                base.save(folder / 'execution_status.json', {**status, 'state': 'failed',
                          'reason': 'nonfinite_failure; worker cleanup bounded'})
                return 1
            if complete_since is not None and now - complete_since > 300:
                stop_group(proc)
                base.save(folder / 'execution_status.json', {**status,
                          'state': 'completed_local_wandb_pending',
                          'reason': 'Models and evaluations saved; W&B shutdown exceeded 300s'})
                return 0
            time.sleep(5)
        code = proc.returncode
        base.save(folder / 'execution_status.json', {**status,
                  'state': 'finished' if code == 0 else 'failed', 'returncode': code})
        return code
    finally:
        stop_group(proc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'check', 'run'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path)
    p.add_argument('--index', type=int, default=0)
    a = p.parse_args(); root = a.root.resolve()
    if a.action == 'prepare':
        prepare(root, a.previous.resolve())
    elif a.action == 'check':
        for row in json.loads((root / 'manifest.json').read_text())['matrix']:
            run(root, row['index'], dry=True)
        print('PASS: all 21 configurations and available recovery checkpoints', flush=True)
    else:
        raise SystemExit(run(root, a.index))


if __name__ == '__main__':
    main()
