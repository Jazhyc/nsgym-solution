"""Run a matched, local-only Ant pipeline check; never submits cluster jobs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=65536)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [root / 'scripts/train.py', *sorted((root / 'src').rglob('*.py')),
               *sorted((root / 'config').rglob('*.yaml'))]
    (output / 'source_hashes.json').write_text(json.dumps({
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources}, indent=2))

    def run(item):
        method, seed = item
        folder = output / method / f'seed{seed}'
        folder.mkdir(parents=True)
        cmd = [sys.executable, 'scripts/train.py', '--config-name=config_ant_v2',
               f'seed={seed}', 'wandb.enabled=false', 'num_threads=1',
               'collector.num_envs=2', 'collector.num_groups=1',
               'collector.frames_per_batch=1024', f'collector.total_frames={args.frames}',
               'env.plr.num_eval_configs=4', 'env.plr.eval_seed=72301',
               'training.num_eval_episodes=2', 'training.eval_ratio=0.5',
               'training.save_interval=100000', '+training.local_metrics=true',
               f'training.checkpoint_dir={folder}', f'hydra.run.dir={folder / "hydra"}',
               f'+experiment.method={method}', '+experiment.final_evaluation=false',
               '+experiment.stage=local_validation',
               f'env.plr.enabled={str(method in ("plr", "full")).lower()}',
               f'env.ns_baseline={str(method == "random_ns").lower()}',
               f'env.plr.mutation_prob={0.25 if method == "full" else 0}',
               f'+env.plr.audit_dir={folder / "curriculum"}']
        (folder / 'command.json').write_text(json.dumps(cmd, indent=2))
        with (folder / 'console.log').open('w') as log:
            result = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        record = {'method': method, 'seed': seed, 'returncode': result.returncode,
                  'frames_requested': args.frames}
        summary = folder / 'training_metrics_summary.json'
        if summary.exists():
            record['metrics'] = json.loads(summary.read_text())['metrics']
        print(json.dumps(record), flush=True)
        return record

    items = [(method, seed) for seed in args.seeds
             for method in ('stationary', 'random_ns', 'plr', 'full')]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(run, items))
    (output / 'summary.json').write_text(json.dumps(records, indent=2))
    if any(row['returncode'] for row in records):
        raise SystemExit('Local pilot failed; inspect console.log before proceeding.')


if __name__ == '__main__':
    main()
