"""Verify completed local Ant pilots from saved metrics, models, and audits."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    hashes = json.loads((output / 'source_hashes.json').read_text())
    for path, digest in hashes.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
    results = json.loads((output / 'summary.json').read_text())
    checks = []
    torch.set_num_threads(1)
    for row in results:
        assert row['returncode'] == 0, row
        folder = output / row['method'] / f"seed{row['seed']}"
        history = [json.loads(line) for line in (folder / 'training_metrics.jsonl').read_text().splitlines()]
        assert history[-1]['train/global_step'] >= row['frames_requested']
        assert all(math.isfinite(value) for h in history for value in h.values()
                   if isinstance(value, (int, float)))
        assert not (folder / 'nonfinite_failure.json').exists()
        checkpoint = torch.load(folder / 'ppo_final.pt', weights_only=False, map_location='cpu')
        assert all(torch.isfinite(p).all() for key in ('actor', 'critic')
                   for p in checkpoint[key].parameters())
        initial_kl = max(h['train/initial_policy_kl'] for h in history)
        assert initial_kl < 1e-6, (row['method'], 'behavior policy mismatch', initial_kl)
        check = {'method': row['method'], 'seed': row['seed'],
                 'frames': history[-1]['train/global_step'], 'finite': True,
                 'initial_policy_kl_max': initial_kl,
                 'guard_stops': history[-1]['train/minibatch_kl_guard_stops_total'],
                 'monitoring_return': row['metrics']['eval/reward_mean']}
        if row['method'] in ('plr', 'full'):
            events = [json.loads(line) for path in (folder / 'curriculum').glob('*.jsonl')
                      for line in path.read_text().splitlines()]
            levels = [e for e in events if e['event'] == 'level']
            assert levels
            assert all(e['config']['wrapper_kwargs']['ant_physics_protocol'] == 'bounded-v2'
                       for e in levels)
            scores = [e for e in events if e['event'] == 'score']
            assert scores and all(math.isfinite(e['score']) and e['score'] >= 0 for e in scores)
            assert history[-1]['plr/incomplete_episode_scores_total'] == 0
            assert history[-1]['plr/abandoned_episode_scores_total'] == 0
            replays = sum(e['event'] == 'selection' and e['replay'] for e in events)
            mutants = sum(e['origin'] == 'mutated' for e in levels)
            assert replays > 0, 'Pilot did not reach replay; extend local budget'
            if row['method'] == 'full':
                assert mutants > 0, 'Pilot did not exercise mutations'
            else:
                assert mutants == 0
            check.update(replay_selections=replays, mutant_levels=mutants,
                         applied_scores=sum(e['applied'] for e in scores),
                         discarded_scores=sum(not e['applied'] for e in scores))
        checks.append(check)
    aggregates = {}
    for method in sorted({r['method'] for r in checks}):
        selected = [r for r in checks if r['method'] == method]
        returns = [r['monitoring_return'] for r in selected]
        aggregates[method] = {'training_seeds': [r['seed'] for r in selected],
                              'mean_monitoring_return': statistics.mean(returns),
                              'sample_std_across_training_seeds': statistics.stdev(returns)
                              if len(returns) > 1 else None}
    report = {'purpose': 'Local pipeline and early stability validation, not performance superiority',
              'runs': checks, 'across_training_seeds': aggregates}
    (output / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
