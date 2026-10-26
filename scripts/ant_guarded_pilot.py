"""Frozen, fresh-seed Ant pilots for the post-update KL guard.

This runner never alters earlier campaigns or cancels cluster jobs. Pilots
share the bounded-v2 training pool and a fixed subset of its held-out suites.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import netrc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import ant_campaign_v2 as base


def hashes():
    return {**base.source_hashes(), 'scripts/ant_guarded_pilot.py':
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def prepare(root, previous, *, frames=10000000, full_evaluation=False):
    if (root/'manifest.json').exists():
        raise FileExistsError('Never overwrite a frozen pilot')
    original = json.loads((previous/'manifest.json').read_text())
    manifest = json.loads(json.dumps(original))
    if not full_evaluation:
        for suite in manifest['tasks']['ant']['suites'].values():
            del suite[32:]
            for case in suite:
                case['seeds'] = case['seeds'][:2]
    stage = 'final' if full_evaluation else 'pilot'
    manifest.update(protocol=('ant-bounded-v2-poststep-fresh-500m' if full_evaluation
                              else 'ant-bounded-v2-poststep-pilot-10m'),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    previous_campaign=str(previous), source_hashes=hashes(),
                    total_frames=frames, lr_decay_frames=100000000, campaign_stage=stage,
                    resume_semantics='All methods start fresh; no checkpoint continuation',
                    evaluation_subset=('Full frozen 200 cases per suite, 5 rollout seeds' if full_evaluation
                                       else 'First 32 frozen cases per suite, first 2 rollout seeds'),
                    evaluation_cases_sha256=hashlib.sha256(json.dumps(
                        manifest['tasks']['ant'], sort_keys=True).encode()).hexdigest(),
                    matrix=[{'index': seed*7+i, 'seed': seed, 'method': method}
                            for seed in range(3) for i, method in enumerate(base.METHODS)])
    assert manifest['training'] == original['training']
    base.save(root/'manifest.json', manifest)
    print(json.dumps({'root': str(root), 'runs': 21,
                      'training_pool_sha256': manifest['training_pool_sha256']}))


def command(root, row, *, local=False, frames=None):
    manifest = json.loads((root/'manifest.json').read_text())
    final = manifest.get('campaign_stage', 'pilot') == 'final'
    frames = frames or (65536 if local else manifest['total_frames'])
    cmd, folder = base.command(root, row, frames=frames, local=local, online=not local)
    label = 'guarded-500m' if final else 'guarded-pilot'
    cmd = [arg.replace('-100m', '-'+label).replace('bounded-v2,100m,', 'bounded-v2,'+label+',')
           if arg.startswith(('wandb.name=', 'wandb.tags=')) else arg for arg in cmd]
    if not final:
        cmd = [arg.replace('+experiment.stage=final', '+experiment.stage=pilot') for arg in cmd]
    cmd += ['+training.lr_decay_frames=100000000', 'training.post_step_kl_guard=true',
            '+training.max_zero_update_batches=100', '+experiment.log_artifact=false']
    if not local:
        cmd += (['training.eval_ratio=0.02', 'training.save_interval=3256'] if final else
                ['training.eval_ratio=0.1', 'training.save_interval=651'])
    return cmd, folder


def validate_config(root, row, *, local=False, frames=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from AAMAS_Comp.envs.torchrl_factory import static_training_definition
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['source_hashes'] != hashes():
        raise RuntimeError('Source differs from frozen pilot')
    cmd, folder = command(root, row, local=local, frames=frames)
    with initialize_config_dir(config_dir=str(base.REPO/'config'), version_base=None):
        cfg = compose(config_name='config_ant_v2', overrides=cmd[3:])
    assert cfg.training.post_step_kl_guard and cfg.collector.synchronous
    assert cfg.training.minibatch_kl_guard and cfg.training.nonnegative_kl_guard
    assert cfg.training.lr_decay_frames == manifest['lr_decay_frames']
    assert not cfg.training.get('resume_checkpoint')
    assert cfg.env.plr.sampler_key == 'ant_v2'
    assert cfg.experiment.public_environment == 'ExampleNSAnt-v2'
    assert list(cfg.experiment.notification_modes) == ['none']
    if row['method'] in ('standard_dr', 'static_plr', 'static_plr_mutations'):
        definition = static_training_definition(cfg)
        digest = hashlib.sha256(json.dumps(definition['dr_samples'], sort_keys=True).encode()).hexdigest()
        assert digest == manifest['training_pool_sha256']
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    base.save(folder/'command.json', cmd)
    return cmd, folder


def verify(folder):
    """Verify updates occurred and no accepted batch exceeded its KL limit."""
    import torch
    from AAMAS_Comp.agents.recovery import validate_models
    checkpoint = torch.load(folder/'ppo_final.pt', map_location='cpu', weights_only=False)
    validate_models(checkpoint)
    count = steps = maximum_zero = 0
    maximum_kl = maximum_saturation = maximum_location = 0.
    last = None
    with (folder/'training_metrics.jsonl').open() as stream:
        for line in stream:
            last = json.loads(line)
            assert all(math.isfinite(v) for v in last.values() if isinstance(v, (int, float)))
            count += 1
            steps += last['train/optimizer_steps']
            maximum_zero = max(maximum_zero, last['train/zero_update_batches'])
            maximum_kl = max(maximum_kl, last['train/post_step_kl_max'])
            maximum_location = max(maximum_location, last.get('train/gaussian_loc_abs_max', 0.))
            maximum_saturation = max(maximum_saturation, last.get('train/action_saturation_fraction', 0.))
    assert count and steps > 0 and maximum_kl <= .03 + 1e-12
    assert last['train/global_step'] == checkpoint['training_state']['global_step']
    assert maximum_zero < 100
    result = {'frames': last['train/global_step'], 'batches': count, 'optimizer_steps': steps,
              'maximum_accepted_batch_kl': maximum_kl, 'maximum_zero_update_streak': maximum_zero,
              'maximum_gaussian_location': maximum_location,
              'maximum_action_saturation_fraction': maximum_saturation,
              'backtracks': last['train/post_step_kl_backtracks_total'],
              'rejected_updates': last['train/post_step_kl_rejections_total'], 'passed': True}
    if 'plr/feedback_sent_total' in last:
        assert last['plr/feedback_sent_total'] > 0
        assert last.get('plr/incomplete_episode_scores_total', 0) == 0
        assert last.get('plr/abandoned_episode_scores_total', 0) == 0
        result['feedback_sent'] = last['plr/feedback_sent_total']
    base.save(folder/'pilot_validation.json', result)
    return result


def stop_group(proc):
    # The parent may have exited while its collector workers remain alive.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run(root, row, *, local=False, frames=None):
    cmd, folder = validate_config(root, row, local=local, frames=frames)
    if (folder/'ppo_final.pt').exists() or (folder/'training_metrics.jsonl').exists():
        raise FileExistsError('Refuse to overwrite an existing run')
    env = os.environ.copy()
    env.update(PYTHONPATH=str(base.REPO/'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MUJOCO_GL='egl', WANDB_INIT_TIMEOUT='120',
               MPLCONFIGDIR=str(folder/'cache/matplotlib'),
               WANDB_CACHE_DIR=str(folder/'cache/wandb'), WANDB_DATA_DIR=str(folder/'wandb_data'))
    if not local and not env.get('WANDB_API_KEY'):
        auth = netrc.netrc(root/'operations/wandb.netrc').authenticators('api.wandb.ai')
        if auth is None:
            raise RuntimeError('W&B credentials unavailable')
        env['WANDB_API_KEY'] = auth[2]
    status = {**row, 'job_id': env.get('SLURM_JOB_ID'), 'local': local,
              'started_at': datetime.now(timezone.utc).isoformat()}
    base.save(folder/'execution_status.json', {**status, 'state': 'running'})
    complete_since = failure_since = None
    stream = (folder/'console.log').open('w')
    proc = subprocess.Popen(cmd, cwd=base.REPO, env=env, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while proc.poll() is None:
            now = time.monotonic()
            failures = [name for name in ('nonfinite_failure.json', 'training_failure.json')
                        if (folder/name).exists()]
            completed = all((folder/name).exists() for name in
                            (('ppo_final.pt', 'run.json') if local else
                             ('ppo_final.pt', 'run.json', 'test_summary.json', 'public_summary.json')))
            if failures and failure_since is None:
                failure_since = now
            if completed and complete_since is None:
                complete_since = now
            if failure_since is not None and now-failure_since > 10:
                stop_group(proc)
                break
            if complete_since is not None and now-complete_since > 300:
                stop_group(proc)
                result = verify(folder)
                base.save(folder/'execution_status.json', {**status,
                          'state': 'completed_local_wandb_pending', 'validation': result})
                return 0
            time.sleep(2)
        if proc.returncode != 0:
            base.save(folder/'execution_status.json', {**status, 'state': 'failed',
                                                      'returncode': proc.returncode})
            return 1
        result = verify(folder)
        base.save(folder/'execution_status.json', {**status, 'state': 'finished', 'validation': result})
        print(json.dumps({**row, **result}), flush=True)
        return 0
    except BaseException as exc:
        base.save(folder/'execution_status.json', {**status, 'state': 'failed',
                                                  'reason': type(exc).__name__+': '+str(exc)})
        raise
    finally:
        stop_group(proc)
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check', 'run', 'verify'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--full-evaluation', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        if args.full_evaluation and args.frames != 500000000:
            raise ValueError('The full evaluation campaign requires the agreed 500M budget')
        prepare(root, args.previous.resolve(), frames=args.frames or 10000000,
                full_evaluation=args.full_evaluation)
        return
    manifest = json.loads((root/'manifest.json').read_text())
    if args.action == 'check':
        for row in manifest['matrix']:
            validate_config(root, row, local=args.local, frames=args.frames)
        print('PASS: all 21 configurations')
        return
    row = manifest['matrix'][args.index]
    if args.action == 'verify':
        _, folder = command(root, row, local=args.local, frames=args.frames)
        print(json.dumps(verify(folder)))
    else:
        raise SystemExit(run(root, row, local=args.local, frames=args.frames))


if __name__ == '__main__':
    main()
