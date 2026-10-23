"""Align Ant campaign evals to equal frame timesteps and average across seeds.

Eval points are written every ``eval_ratio * total_frames`` frames, but resumed
runs start at an offset (e.g. 100,002,816 frames).  This maps every eval to the
nearest nominal timestep, then averages seed-level eval metrics per method.
It never pools seeds with different methods.

The three static variants share one frozen 4096-config training pool:
  standard_dr             - i.i.d. draws from that pool (the "standard" control)
  static_plr              - prioritized selection from that pool
  static_plr_mutations    - prioritized selection + mutation within that pool
Temporal variants (stationary, random_ns, plr, full) sample non-stationary
tasks online and are kept separate.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ['stationary', 'random_ns', 'plr', 'full',
           'standard_dr', 'static_plr', 'static_plr_mutations']
STATIC = {'standard_dr', 'static_plr', 'static_plr_mutations'}
TOTAL_FRAMES = 500_000_000
EVAL_FRAMES = 10_000_000  # eval_ratio=0.02 of 500M
EVAL_RE = re.compile(r'\[iter \d+ \| frames (\d+)\] eval_return=(-?[\d.]+)\s+eval_iqm=(-?[\d.]+)')


def load_runs(manifest_path, slurm_dir, status_root):
    manifest = json.loads(Path(manifest_path).read_text())
    runs = []
    for row in manifest['matrix']:
        status_path = Path(status_root) / row['method'] / f"seed{row['seed']}" / 'execution_status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        runs.append({**row, 'restored_frames': int(status.get('restored_frames', 0)),
                     'state': status.get('state', 'not_started')})
    return runs


def parse_evals(slurm_dir, runs):
    by_index = {row['index']: row for row in runs}
    evals = defaultdict(list)
    for path in sorted(Path(slurm_dir).glob('*.out')):
        match = re.search(r'_(\d+)\.out$', path.name)
        if not match or int(match.group(1)) not in by_index:
            continue
        row = by_index[int(match.group(1))]
        key = (row['method'], row['seed'])
        for line in path.open(errors='ignore'):
            hit = EVAL_RE.search(line)
            if hit:
                frames, ret, iqm = int(hit.group(1)), float(hit.group(2)), float(hit.group(3))
                evals[key].append((frames, ret, iqm))
    return evals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--slurm-dir', type=Path, required=True)
    parser.add_argument('--status-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--axis', choices=('total', 'new'), default='total',
                        help='total frames, or frames since resume/start')
    parser.add_argument('--metric', choices=('eval_iqm', 'eval_return'), default='eval_iqm')
    args = parser.parse_args()

    runs = load_runs(args.manifest, args.slurm_dir, args.status_root)
    evals = parse_evals(args.slurm_dir, runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Per-run points on the nominal grid.
    run_rows = []
    aligned = defaultdict(lambda: defaultdict(list))  # method -> timestep -> [(seed, value)]
    for row in runs:
        key = (row['method'], row['seed'])
        offset = row['restored_frames'] if args.axis == 'new' else 0
        for frames, ret, iqm in evals.get(key, []):
            value = iqm if args.metric == 'eval_iqm' else ret
            step = int(round((frames - offset) / EVAL_FRAMES)) * int(EVAL_FRAMES / 1e6)
            if step < 0:
                continue
            run_rows.append({'method': row['method'], 'seed': row['seed'], 'timestep_M': step,
                             'frames': frames, 'eval_return': ret, 'eval_iqm': iqm})
            aligned[row['method']][step].append((row['seed'], value))

    with (args.output_dir / f'{args.metric}_{args.axis}_runs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['method', 'seed', 'timestep_M', 'frames',
                                                    'eval_return', 'eval_iqm'])
        writer.writeheader()
        writer.writerows(sorted(run_rows, key=lambda r: (r['method'], r['seed'], r['timestep_M'])))

    # Mean across seeds at each matched timestep; n is recorded explicitly.
    method_rows = []
    for method in METHODS:
        for step in sorted(aligned.get(method, {})):
            values = np.array([value for _, value in aligned[method][step]], dtype=float)
            method_rows.append({'method': method, 'kind': 'static' if method in STATIC else 'temporal',
                                'timestep_M': step, 'n_seeds': len(values),
                                'mean': float(values.mean()),
                                'min': float(values.min()), 'max': float(values.max()),
                                'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                                'seeds': ';'.join(str(seed) for seed, _ in aligned[method][step])})
    with (args.output_dir / f'{args.metric}_{args.axis}_methods.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_rows[0]))
        writer.writeheader()
        writer.writerows(method_rows)

    print(f'wrote {len(run_rows)} run-eval rows and {len(method_rows)} method rows to {args.output_dir}')
    # Compact table: only timesteps present for every method with >=1 seed.
    common = sorted(set.intersection(*(set(aligned[m]) for m in METHODS if aligned[m])))
    header = 'timestep_M  ' + ''.join(f'{m[:13]:>16}' for m in METHODS)
    print(header)
    for step in common:
        cells = []
        for method in METHODS:
            vals = aligned[method][step]
            cells.append(f'{np.mean([v for _, v in vals]):8.0f} (n={len(vals)})')
        print(f'{step:9.0f}  ' + ''.join(f'{c:>16}' for c in cells))


if __name__ == '__main__':
    main()
