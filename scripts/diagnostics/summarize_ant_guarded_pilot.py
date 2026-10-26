"""Summarize every planned Ant pilot seed without excluding poor outcomes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from AAMAS_Comp.evaluation.statistics import trimmed_mean


def summarize(root, local=False):
    manifest = json.loads((root/'manifest.json').read_text())
    rows = []
    for planned in manifest['matrix']:
        folder = root/('local' if local else 'final')/'ant'/planned['method']/f"seed{planned['seed']}"
        row = {**planned, 'state': 'not_started', 'metrics': {}, 'validation': {}}
        if (folder/'execution_status.json').exists():
            row['state'] = json.loads((folder/'execution_status.json').read_text())['state']
        if (folder/'pilot_validation.json').exists():
            row['validation'] = json.loads((folder/'pilot_validation.json').read_text())
        for name in ['test_summary.json', 'public_summary.json']:
            if (folder/name).exists():
                row['metrics'].update(json.loads((folder/name).read_text())['metrics'])
        rows.append(row)
    aggregates = {}
    by_method = defaultdict(list)
    for row in rows:
        by_method[row['method']].append(row)
    for method, method_rows in by_method.items():
        metric_names = sorted({k for row in method_rows for k in row['metrics']
                               if k.endswith('/mean')})
        aggregates[method] = {}
        for metric in metric_names:
            values = {str(row['seed']): row['metrics'][metric]
                      for row in method_rows if metric in row['metrics']}
            aggregates[method][metric] = {
                'seeds': values, 'n': len(values), 'all_planned_seeds': len(values) == 3,
                'mean': statistics.mean(values.values()),
                'sd': statistics.stdev(values.values()) if len(values)>1 else None,
                'seed_iqm': trimmed_mean(list(values.values()))}
    final = manifest.get('campaign_stage', 'pilot') == 'final'
    report = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'campaign': str(root), 'protocol': manifest['protocol'],
              'evaluation_subset': manifest['evaluation_subset'],
              'state_counts': dict(Counter(row['state'] for row in rows)),
              'planned_runs': len(rows), 'runs': rows, 'aggregates': aggregates,
              'note': 'All observed seeds included, including collapsed policies. '
                      'Incomplete seed sets are explicitly marked; campaigns are kept separate.'}
    destination = root/('local_summary' if local else ('campaign_summary' if final else 'pilot_summary'))
    destination.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Ant post-update guard '+('full campaign' if final else 'pilot'), '', f"Checked: {report['checked_at']}", '',
             f"States: {report['state_counts']}", '', report['note'], '',
             'Scores are mean return ± sample standard deviation across training seeds. '
             'Episode-level returns are first averaged within each training seed.', '',
             '| Method | IID | Abrupt | Gradual | Public mass decay |',
             '|---|---:|---:|---:|---:|']
    metrics = ['test/iid/mean', 'test/abrupt/mean', 'test/gradual/mean', 'public/none/mean']
    for method, values in aggregates.items():
        cells = []
        for metric in metrics:
            result = values.get(metric)
            if result is None:
                cells.append('pending')
                continue
            sd = f" ± {result['sd']:.1f}" if result['sd'] is not None else ''
            cells.append(f"{result['mean']:.1f}{sd} (n={result['n']})")
        lines.append('| '+method+' | '+' | '.join(cells)+' |')
    lines += ['', '## Training checks', '',
              '| Method | Seed | State | Frames | Optimizer steps | Maximum accepted KL |',
              '|---|---:|---|---:|---:|---:|']
    for row in rows:
        v = row['validation']
        lines.append(f"| {row['method']} | {row['seed']} | {row['state']} | "
                     f"{v.get('frames', '')} | {v.get('optimizer_steps', '')} | "
                     f"{v.get('maximum_accepted_batch_kl', '')} |")
    destination.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    return report


def plot(root, local=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    manifest = json.loads((root/'manifest.json').read_text())
    methods = list(dict.fromkeys(row['method'] for row in manifest['matrix']))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    any_data = False
    for method, ax in zip(methods, axes.flat):
        grouped = defaultdict(list)
        for seed in range(3):
            path = root/('local' if local else 'final')/'ant'/method/f'seed{seed}'/'training_metrics.jsonl'
            if not path.exists():
                continue
            curve = []
            with path.open() as stream:
                for line in stream:
                    if '"eval/reward_mean"' in line:
                        row = json.loads(line)
                        curve.append((row['train/global_step'], row['eval/reward_mean']))
            if curve:
                any_data = True
                xs, ys = zip(*curve)
                ax.plot(np.asarray(xs)/1e6, ys, alpha=.4, lw=1, label=f'seed {seed}')
                for x, y in curve:
                    grouped[x].append(y)
        common = sorted((x, statistics.mean(ys)) for x, ys in grouped.items() if len(ys)==3)
        if common:
            xs, ys = zip(*common)
            ax.plot(np.asarray(xs)/1e6, ys, color='black', lw=2, label='3-seed mean')
        ax.set_title(method)
        ax.set_xlabel('Training transitions (millions)')
        ax.set_ylabel('Deterministic monitor return')
        ax.grid(alpha=.2)
        if grouped:
            ax.legend(fontsize=7)
    axes.flat[-1].axis('off')
    if any_data:
        fig.savefig(root/('local_learning_curves.png' if local else 'pilot_learning_curves.png'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    report = summarize(args.root.resolve(), args.local)
    if args.plot:
        plot(args.root.resolve(), args.local)
    print(json.dumps({'state_counts': report['state_counts'], 'planned_runs': report['planned_runs']}))
