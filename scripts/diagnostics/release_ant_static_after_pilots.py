"""Release the entire static final cohort only after all nine pilot seeds pass."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def assess(pilot, final, minimum_return=1000.):
    previous = json.loads((pilot/'manifest.json').read_text())
    manifest = json.loads((final/'manifest.json').read_text())
    problems, evidence, indices = [], [], []
    if previous['training'] != manifest['training']:
        problems.append('Training physics pools differ')
    for key, value in previous['source_hashes'].items():
        if key != 'scripts/ant_guarded_pilot.py' and manifest['source_hashes'].get(key) != value:
            problems.append('Training source differs: '+key)
    for row in manifest['matrix']:
        if row['method'] not in ('standard_dr', 'static_plr', 'static_plr_mutations'):
            continue
        indices.append(row['index'])
        folder = pilot/'final/ant'/row['method']/f"seed{row['seed']}"
        required = ['pilot_validation.json', 'public_summary.json', 'test_summary.json']
        if not all((folder/name).exists() for name in required):
            problems.append(f"{row['method']} seed {row['seed']}: incomplete pilot")
            continue
        validation = json.loads((folder/'pilot_validation.json').read_text())
        public = json.loads((folder/'public_summary.json').read_text())['metrics']['public/none/mean']
        iid = json.loads((folder/'test_summary.json').read_text())['metrics']['test/iid/mean']
        healthy = (validation['passed'] and validation['frames'] >= 10000000
                   and validation['maximum_accepted_batch_kl'] <= .03+1e-12
                   and validation['optimizer_steps'] > 0
                   and validation['maximum_zero_update_streak'] < 100)
        # This checks basic locomotion beyond the approximate survival-only
        # return; it never selects a winner or drops individual training seeds.
        if not healthy or min(public, iid) <= minimum_return:
            problems.append(f"{row['method']} seed {row['seed']}: numerical/locomotion review required")
        evidence.append({**row, 'validation': validation, 'public_mean': public, 'iid_mean': iid})
    return {'checked_at': datetime.now(timezone.utc).isoformat(),
            'passed': not problems, 'minimum_pilot_return': minimum_return,
            'problems': problems, 'evidence': evidence, 'release_indices': indices}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--final', type=Path, required=True)
    parser.add_argument('--minimum-return', type=float, default=1000.)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    result = assess(args.pilot, args.final, args.minimum_return)
    if not args.check_only:
        (args.final/'operations/static_release_gate.json').write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']:
        print(json.dumps({'state': 'held_for_review', 'problems': result['problems']}))
        raise SystemExit(2)
    if not args.check_only:
        path = args.final/'submitted_jobs.json'
        submitted = json.loads(path.read_text())
        for index in submitted.get('static_held', []):
            assert index in result['release_indices']
            subprocess.run(['scontrol', 'release', submitted['array']+'_'+str(index)], check=True)
        submitted.update(state='all_released', static_released=result['release_indices'],
                         static_held=[], all_released_at=datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(submitted, indent=2)+'\n')
    print(json.dumps({'state': 'passed' if args.check_only else 'all_released',
                      'static_indices': result['release_indices']}))
