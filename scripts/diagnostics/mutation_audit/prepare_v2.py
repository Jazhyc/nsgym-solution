"""Prepare an explicitly separate static-mutation ablation. Never submits jobs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'scripts'))
import paper_campaign as campaign


def prepare(output, parents):
    if output.exists() and (not output.is_dir() or any(p.name != 'release' for p in output.iterdir())):
        raise FileExistsError('Refusing to overwrite an ablation directory')
    output.mkdir(parents=True, exist_ok=True)
    cp = json.loads(parents['cartpole'].read_text())
    if cp.get('physics_protocol') != 'native-cartpole-constraints-v1':
        raise ValueError('CartPole ablation requires corrected native physics')
    definition = deepcopy(cp)
    sources = {}
    for task, path in parents.items():
        parent = json.loads(path.read_text())
        if parent['protocol'] != campaign.PROTOCOL:
            raise ValueError('Parent must use corrected asynchronous PLR')
        definition['tasks'][task] = deepcopy(parent['tasks'][task])
        definition['training']['tasks'][task] = deepcopy(parent['training']['tasks'][task])
        sources[task] = {'manifest': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    definition['tasks'] = {task: definition['tasks'][task] for task in parents}
    definition['training']['tasks'] = {task: definition['training']['tasks'][task] for task in parents}
    definition.update(static_mutation_protocol='relative-reflect-symmetric-v2',
                      ablation_protocol='static-mutation-relative-v2',
                      methods=['static_plr_mutations'],
                      matrix=[(task, 'static_plr_mutations', seed)
                              for seed in range(5) for task in parents],
                      provenance=campaign.provenance(), evaluation_sources=sources,
                      evaluation_cases_sha256=hashlib.sha256(json.dumps(
                          definition['tasks'], sort_keys=True).encode()).hexdigest())
    definition['extension'] = {
        'description': 'Separate mutation ablation; retain all v1 results and reuse existing static DR/PLR controls',
        'parent_manifests': sources,
        'mutation_rule': 'sigma .2 relative scalar step with 1%-median floor; reflect at bounds; symmetric scalar FrozenLake slip',
        'hyperparameter_selection': 'Mechanistic scale correction; sigma .2 retained, no test-result tuning',
        'launch_status': 'prepared_only; no jobs submitted'}
    manifest = output/'manifest.json'
    manifest.write_text(json.dumps(definition, indent=2))
    for index in range(len(definition['matrix'])):
        campaign.run(SimpleNamespace(stage='final', index=index, root=output, manifest=manifest,
                                     project='nsgym-paper', group=output.name, wandb_mode='online',
                                     protect_unscored='true', dry_run=True))
    pilot = deepcopy(definition)
    pilot['matrix'] = [row for row in definition['matrix'] if row[2] == 0]
    for task in pilot['tasks'].values():
        for suite, rows in task['suites'].items():
            task['suites'][suite] = rows[:4]
            for row in task['suites'][suite]:
                row['seeds'] = row['seeds'][:1]
    pilot['evaluation_cases_sha256'] = hashlib.sha256(json.dumps(pilot['tasks'], sort_keys=True).encode()).hexdigest()
    pilot_manifest = output/'pilot_manifest.json'
    pilot_manifest.write_text(json.dumps(pilot, indent=2))
    for index in range(len(pilot['matrix'])):
        campaign.run(SimpleNamespace(stage='pilot', index=index, root=output, manifest=pilot_manifest,
                                     project='nsgym-paper', group=output.name, wandb_mode='online',
                                     protect_unscored='true', dry_run=True))
    (output/'readiness.json').write_text(json.dumps({
        'prepared_runs': len(definition['matrix']), 'prepared_pilots': len(pilot['matrix']),
        'source_sha256': definition['provenance']['source_sha256'],
        'cluster_pilots_verified': False, 'full_jobs_submitted': False,
        'note': 'No Ant runs. Existing experiment allocations are unchanged.'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cartpole-manifest', required=True, type=Path)
    parser.add_argument('--frozenlake-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.output.resolve(), {'cartpole': args.cartpole_manifest.resolve(),
                                    'frozenlake': args.frozenlake_manifest.resolve()})
