"""Final static cohorts must never silently omit failed or missing pilot seeds."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('ant_static_gate',
    Path(__file__).resolve().parents[1]/'scripts/diagnostics/release_ant_static_after_pilots.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def campaigns(tmp_path):
    pilot, final = tmp_path/'pilot', tmp_path/'final'
    pilot.mkdir(); final.mkdir()
    matrix = [{'index': seed*7+i, 'seed': seed, 'method': method}
              for seed in range(3) for i, method in enumerate(
                  ['stationary', 'random_ns', 'plr', 'full', 'standard_dr', 'static_plr', 'static_plr_mutations'])]
    manifest = {'matrix': matrix, 'training': {'pool': [1, 2]},
                'source_hashes': {'scripts/train.py': 'same-training-code'}}
    for root in [pilot, final]:
        (root/'manifest.json').write_text(json.dumps(manifest))
    for row in matrix:
        if row['index'] % 7 < 4:
            continue
        folder = pilot/'final/ant'/row['method']/f"seed{row['seed']}"
        folder.mkdir(parents=True)
        validation = {'passed': True, 'frames': 10002432, 'maximum_accepted_batch_kl': .029,
                      'optimizer_steps': 10000, 'maximum_zero_update_streak': 0}
        (folder/'pilot_validation.json').write_text(json.dumps(validation))
        for filename, metric in [('public_summary.json', 'public/none/mean'),
                                 ('test_summary.json', 'test/iid/mean')]:
            (folder/filename).write_text(json.dumps({'metrics': {metric: 2000.}}))
    return pilot, final


def test_all_nine_planned_static_seeds_are_required(campaigns):
    pilot, final = campaigns
    result = gate.assess(pilot, final)
    assert result['passed'] and len(result['evidence']) == 9
    assert result['release_indices'] == [4, 5, 6, 11, 12, 13, 18, 19, 20]


def test_one_collapsed_seed_blocks_the_entire_static_cohort(campaigns):
    pilot, final = campaigns
    p = pilot/'final/ant/static_plr_mutations/seed1/public_summary.json'
    p.write_text(json.dumps({'metrics': {'public/none/mean': -31.}}))
    result = gate.assess(pilot, final)
    assert not result['passed'] and len(result['evidence']) == 9
    assert any('static_plr_mutations seed 1' in reason for reason in result['problems'])


def test_missing_pilot_cannot_be_treated_as_a_pass(campaigns):
    pilot, final = campaigns
    (pilot/'final/ant/static_plr/seed2/pilot_validation.json').unlink()
    assert not gate.assess(pilot, final)['passed']


def test_training_source_drift_blocks_automatic_release(campaigns):
    pilot, final = campaigns
    p = final/'manifest.json';manifest = json.loads(p.read_text())
    manifest['source_hashes']['scripts/train.py'] = 'untested-code'
    p.write_text(json.dumps(manifest))
    assert not gate.assess(pilot, final)['passed']
