"""Summarize durable worker audit logs; complete episodes and origin stay separate."""
import argparse
import json
from pathlib import Path


def summarize(folder):
    rows = {origin: dict(selections=0, replays=0, completed=0, partial=0,
                         return_sum=0., steps_sum=0, maximum_mutation_depth=0)
            for origin in ('fresh', 'mutated')}
    replay_expected = 0.
    files = sorted(folder.glob('*.jsonl'))
    incomplete_tail_lines = 0
    for path in files:
        with path.open() as stream:
            for line in stream:
                if not line.endswith('\n'):
                    # Live readers may catch a write in progress. Completed
                    # malformed JSON records still raise rather than disappear.
                    incomplete_tail_lines += 1
                    continue
                event = json.loads(line)
                if event['event'] not in ('selection', 'outcome'):
                    continue
                row = rows[event['origin']]
                row['maximum_mutation_depth'] = max(row['maximum_mutation_depth'], event['mutation_depth'])
                if event['event'] == 'selection':
                    row['selections'] += 1
                    row['replays'] += int(event['replay'])
                    if event['replay']:
                        replay_expected += event['conditional_replay_mutated_mass']
                elif event['complete']:
                    row['completed'] += 1
                    row['return_sum'] += event['episode_return']
                    row['steps_sum'] += event['steps']
                else:
                    row['partial'] += 1
    for row in rows.values():
        row['return_mean'] = row['return_sum']/row['completed'] if row['completed'] else None
        row['length_mean'] = row['steps_sum']/row['completed'] if row['completed'] else None
    replays = sum(r['replays'] for r in rows.values())
    return {'files': len(files), 'incomplete_tail_lines': incomplete_tail_lines, 'by_origin': rows,
            'mutated_replay_share': rows['mutated']['replays']/replays if replays else None,
            'expected_mutated_replay_share': replay_expected/replays if replays else None,
            'note': 'Counts are from audit events, including collector work not delivered to the learner. '
                    'Returns exclude partial episodes; expected replay share uses actual pre-draw conditional probabilities.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.folder), indent=2)
    if args.output:
        args.output.write_text(result)
    print(result)
