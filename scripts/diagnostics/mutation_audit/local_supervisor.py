"""Persistent, memory-gated local execution of a frozen mutation ablation.

This orchestrator lives outside the frozen release. It never changes PPO,
collector settings, scientific manifests, or existing completed outputs.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

GIB = 1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def admit_second(available, peak_pss, cores_busy, *, reserve=4., minimum_new=6.):
    """Require startup headroom plus a 35% margin on measured process memory."""
    required = max(minimum_new, 1.35*peak_pss) + reserve
    return available >= required and cores_busy < 10., required


def outputs_complete(folder, task):
    required = ('ppo_final.pt', 'run.json', 'training_metrics_summary.json',
                'test_summary.json', 'test_episodes.jsonl', 'test_trajectories.npz',
                'test_trajectory_metadata.json', 'public_summary.json', 'public_episodes.jsonl')
    if not all((folder/name).exists() for name in required):
        return False
    try:
        run = json.loads((folder/'run.json').read_text())
        broad = json.loads((folder/'test_summary.json').read_text())
        public = json.loads((folder/'public_summary.json').read_text())
        budget = {'cartpole': 10000000, 'frozenlake': 4000000}[task]
        if run['actual_frames'] < budget:
            return False
        if broad['action_mode'] != 'stochastic' or public['action_mode'] != 'stochastic':
            return False
        if sum(1 for _ in (folder/'test_episodes.jsonl').open()) != 4000:
            return False
        if sum(1 for _ in (folder/'public_episodes.jsonl').open()) != 300:
            return False
        if task == 'cartpole' and any(broad['metrics'][f'test/{s}/physics_invalid_steps']
                                     for s in ('iid', 'abrupt', 'gradual', 'no_change')):
            return False
        return True
    except (ValueError, KeyError, OSError):
        return False


def tree_resources(pid):
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0., 0., 0., 0
    rss = pss = cpu = 0.
    alive = 0
    for p in processes:
        try:
            memory = p.memory_full_info()
            rss += memory.rss
            # PSS shares library pages fairly; summing RSS overcounts torch.
            pss += getattr(memory, 'pss', memory.rss)
            times = p.cpu_times()
            cpu += times.user+times.system
            alive += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss/GIB, pss/GIB, cpu, alive


def terminate_tree(process):
    """Only used after all scientific outputs have been verified complete."""
    try:
        parent = psutil.Process(process.pid)
        targets = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return
    for p in targets:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=10)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=20)


def supervisor(root, python):
    root = root.resolve()
    lock = (root/'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['ablation_protocol'] == 'static-mutation-relative-v2'
    assert manifest['static_mutation_protocol'] == 'relative-reflect-symmetric-v2'
    assert len(manifest['matrix']) == 10
    assert all(t in ('cartpole', 'frozenlake') and m == 'static_plr_mutations'
               for t, m, _ in manifest['matrix'])
    release = root/'release'
    env = os.environ.copy()
    env.update(PYTHONPATH=str(release/'src')+os.pathsep+str(release),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               WANDB_INIT_TIMEOUT='120', WANDB_CACHE_DIR=str(root/'cache/wandb'),
               WANDB_CONFIG_DIR=str(root/'cache/wandb-config'),
               WANDB_DATA_DIR=str(root/'wandb_data'), MPLCONFIGDIR=str(root/'cache/matplotlib'))
    if (root/'runtime/sitecustomize.py').exists():
        env['PYTHONPATH'] = str(root/'runtime')+os.pathsep+env['PYTHONPATH']
        runtime = json.loads((root/'runtime/execution_policy.json').read_text())
        env.update(NSGYM_LOCAL_START_METHOD=runtime['start_method'], NSGYM_FROZEN_RELEASE=str(release))
    (root/'logs').mkdir(exist_ok=True)
    previous = root/'status.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        if any(r.get('state') == 'running' and psutil.pid_exists(r['pid']) for r in old['runs']):
            raise RuntimeError('Recorded children still alive; inspect instead of duplicating experiments')
    state = {'started_at': now(), 'supervisor_pid': os.getpid(), 'hostname': os.uname().nodename,
             'source_sha256': manifest['provenance']['source_sha256'],
             'orchestrator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'wandb_group': root.name+'-final', 'runs': [], 'max_parallel': 2}
    for index, (task, method, seed) in enumerate(manifest['matrix']):
        folder = root/'final'/task/method/f'seed{seed}'
        complete = outputs_complete(folder, task)
        if not complete and (folder/'ppo_final.pt').exists():
            raise RuntimeError(f'Existing checkpoint with incomplete evaluation: {folder}; do not overwrite')
        state['runs'].append(dict(index=index, task=task, method=method, seed=seed,
                                  folder=str(folder), state='already_complete' if complete else 'queued'))
    active = {}
    last_tick = time.monotonic()
    last_swap_out = psutil.swap_memory().sout
    while True:
        tick = time.monotonic()
        swap = psutil.swap_memory()
        swap_rate = max(0, swap.sout-last_swap_out)/max(tick-last_tick, .001)/(1024**2)
        control_path = root/'control.json'
        control = json.loads(control_path.read_text()) if control_path.exists() else {}
        limit = min(2, max(1, int(control.get('max_parallel', 2))))
        available = psutil.virtual_memory().available/GIB
        cpu_busy = 0.
        for index in list(active):
            job = active[index]
            row = state['runs'][index]
            process = job['process']
            rss, pss, cpu, count = tree_resources(process.pid)
            cores = max(0., cpu-job.get('cpu_seconds', cpu))/max(tick-last_tick, .001)
            cpu_busy += cores
            job['cpu_seconds'] = cpu
            row.update(rss_gib=round(rss, 2), pss_gib=round(pss, 2),
                       peak_pss_gib=max(row.get('peak_pss_gib', 0), round(pss, 2)),
                       cores_busy=round(cores, 2), processes=count,
                       elapsed_seconds=round(tick-job['started']))
            code = process.poll()
            complete = outputs_complete(Path(row['folder']), row['task'])
            if complete and job.get('compute_complete_at') is None:
                job['compute_complete_at'] = tick
                row['compute_complete_at'] = now()
            if code is None and complete and tick-job['compute_complete_at'] > 300:
                # W&B finalization has hung repeatedly on Habrok. Keep all raw
                # run data; mark upload pending rather than pretending success.
                save(Path(row['folder'])/'upload_recovery_status.json', {
                    'compute_complete': True, 'wandb_upload_complete': False,
                    'reason': 'Five-minute W&B finalization allowance exceeded', 'timestamp': now()})
                terminate_tree(process)
                code = process.returncode
                row['upload_timeout'] = True
            if code is not None:
                job['log'].close()
                row.update(exit_code=code, finished_at=now(), compute_complete=complete,
                           state=('completed' if code == 0 else 'compute_complete_upload_pending')
                           if complete else 'failed')
                del active[index]
                print(now(), row['task'], row['seed'], row['state'], flush=True)
        queued = [r for r in state['runs'] if r['state'] == 'queued']
        allowed = not control.get('pause_new_runs', False) and len(active) < limit and available >= 6.
        reason = 'ready' if allowed else 'paused, concurrency limit, or low available memory'
        if allowed and active:
            # Observe one initialized training process before deciding whether
            # another copy of 24 torch workers fits in this machine.
            initialized = all(tick-j['started'] >= 90 and state['runs'][i].get('pss_gib', 0) > 1
                              for i, j in active.items())
            peak = max(state['runs'][i]['peak_pss_gib'] for i in active)
            allowed, needed = admit_second(available, peak, cpu_busy)
            allowed = allowed and initialized and swap_rate < .1
            reason = f'need >= {needed:.1f} GiB available, startup observation, CPU and swap headroom'
        if allowed and queued:
            row = queued[0]
            index = row['index']
            log = (root/'logs'/f'{index:02d}-{row["task"]}-seed{row["seed"]}.log').open('x')
            command = [str(python), str(release/'scripts/paper_campaign.py'), 'run',
                       '--root', str(root), '--manifest', str(root/'manifest.json'),
                       '--stage', 'final', '--group', root.name, '--index', str(index)]
            process = subprocess.Popen(command, cwd=release, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            active[index] = dict(process=process, log=log, started=time.monotonic())
            row.update(state='running', pid=process.pid, started_at=now(), peak_pss_gib=0.)
            print(now(), 'START', row['task'], 'seed', row['seed'], 'pid', process.pid, flush=True)
        state.update(updated_at=now(), available_memory_gib=round(available, 2),
                     swap_used_gib=round(psutil.swap_memory().used/GIB, 2),
                     swap_out_mib_per_second=round(swap_rate, 3),
                     active_runs=len(active), admission_note=reason, max_parallel=limit)
        save(root/'status.json', state)
        last_tick = tick
        last_swap_out = swap.sout
        if not active and not any(r['state'] == 'queued' for r in state['runs']):
            if all(r['state'] in ('completed', 'already_complete', 'compute_complete_upload_pending')
                   for r in state['runs']):
                with (root/'logs/aggregation.log').open('w') as log:
                    aggregated = subprocess.run([str(python), str(release/'scripts/paper_campaign.py'),
                        'aggregate', '--root', str(root/'final'), '--manifest', str(root/'manifest.json'),
                        '--output', str(root/'comparison.json')], cwd=release, env=env,
                        stdout=log, stderr=subprocess.STDOUT, timeout=300)
                state['aggregation_exit_code'] = aggregated.returncode
            state['finished_at'] = now()
            save(root/'status.json', state)
            print(now(), 'ALL RUNS TERMINAL', flush=True)
            break
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--python', required=True, type=Path)
    args = parser.parse_args()
    supervisor(args.root, args.python)
