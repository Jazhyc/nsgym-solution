"""Regression coverage for outlier scaling, mutation lineage, and audit integrity."""
from copy import deepcopy
import json
from queue import Queue

import numpy as np
import pytest

from AAMAS_Comp.envs.static_curriculum import StaticDynamicsConfig, StaticDynamicsSampler
from AAMAS_Comp.curriculum.plr import PLRBuffer
from AAMAS_Comp.curriculum.plr_env import PLREnv
from AAMAS_Comp.envs.ns_env_sampler import make_sampler
from AAMAS_Comp.evaluation.training_metrics import WorkerCurriculumMetrics


def pool():
    return [dict(masspole=v, masscart=1., length=.5, gravity=9.8) for v in [.00001, .1, .1, .1, 45.]]


def test_typical_mutation_scale_is_independent_of_rare_calibration_maximum():
    small = StaticDynamicsSampler('cartpole', pool())
    wide_pool = pool()
    wide_pool[-1]['masspole'] = 45000.
    wide = StaticDynamicsSampler('cartpole', wide_pool)
    parent = StaticDynamicsConfig('cartpole', pool()[1])
    for seed in range(100):
        a = small.mutate(parent, np.random.default_rng(seed))
        b = wide.mutate(parent, np.random.default_rng(seed))
        assert a.values['masspole'] == pytest.approx(b.values['masspole'])
        assert .02 < a.values['masspole'] < .2


def test_reflection_does_not_pile_up_at_scalar_boundaries():
    sampler = StaticDynamicsSampler('cartpole', pool())
    parent = StaticDynamicsConfig('cartpole', pool()[0])
    for seed in range(100):
        value = sampler.mutate(parent, np.random.default_rng(seed)).values['masspole']
        assert sampler.bounds['masspole'][0] < value < sampler.bounds['masspole'][1]


def test_frozenlake_mutations_preserve_slip_symmetry_across_generations():
    sampler = StaticDynamicsSampler('frozenlake', [{'P': [0., .5, .5]}, {'P': [1., 0., 0.]}])
    parent = StaticDynamicsConfig('frozenlake', {'P': [0., .5, .5]})
    for seed in range(500):
        parent = sampler.mutate(parent, np.random.default_rng(seed))
        p = parent.values['P']
        assert p[1] == p[2] and sum(p) == pytest.approx(1.)
        assert 0 < p[0] < 1
    with pytest.raises(ValueError, match='symmetric parents'):
        sampler.mutate(StaticDynamicsConfig('frozenlake', {'P': [.5, .4, .1]}), np.random.default_rng(0))


def test_explicit_legacy_protocol_still_reproduces_range_clipping():
    sampler = StaticDynamicsSampler('cartpole', pool(), mutation_protocol='range-clip-v1')
    parent = StaticDynamicsConfig('cartpole', pool()[1])
    rng = np.random.default_rng(1)
    assert any(sampler.mutate(parent, rng).values['masspole'] == .00001 for _ in range(100))


def test_lineage_survives_parent_eviction_and_replay_retains_origin():
    b = PLRBuffer(make_sampler('cartpole', seed=1), capacity=1, replay_prob=0,
                  mutation_prob=1, seed=1)
    slot, _ = b.sample()
    original_uid = b._entries[slot].uid
    b.update(slot, 1.)
    slot, _ = b.sample()
    child = b._entries[slot]
    assert child.origin == 'mutated' and child.parent_uid == original_uid
    assert child.uid != original_uid and child.mutation_depth == 1
    b.update(slot, 1.)
    b.replay_prob = 1.
    slot, _ = b.sample()
    assert b.last_was_replay and not b.last_was_mutation
    assert b._entries[slot].origin == 'mutated'
    assert b.stats()['plr/mutated_priority_mass'] == pytest.approx(1.)


def test_audit_does_not_change_sampling_and_records_outcomes_and_delayed_scores(tmp_path):
    envs = [PLREnv('cartpole', plr_capacity=2, replay_prob=.5, mutation_prob=1,
                   feedback_queue=Queue(), seed=5, audit_dir=tmp_path if audit else None)
            for audit in (False, True)]
    try:
        for episode in range(12):
            configs = []
            for env in envs:
                if env._level_id is not None:
                    env._feedback_queue.put((env._level_tokens[env._level_id], episode-1, 1.))
                env.reset(seed=episode)
                configs.append(deepcopy(env.plr._entries[env._level_id]))
                for _ in range(500):
                    _, _, terminated, truncated, _ = env.step(0)
                    if terminated or truncated:
                        break
            assert configs[0] == configs[1]
        # A partial reset must not count as a completed episode.
        envs[1].reset(seed=20)
        envs[1].step(0)
        envs[1].reset(seed=21)
    finally:
        for env in envs:
            env.close()
    events = [json.loads(line) for p in tmp_path.glob('*.jsonl') for line in p.read_text().splitlines()]
    levels = {e['uid']: e for e in events if e['event'] == 'level'}
    assert any(e['origin'] == 'mutated' and e['parent_uid'] in levels for e in levels.values())
    assert any(e['event'] == 'score' and e['applied'] for e in events)
    outcomes = [e for e in events if e['event'] == 'outcome']
    assert any(not e['complete'] for e in outcomes)
    assert any(e['complete'] for e in outcomes)
    for e in events:
        if e['event'] in ('outcome', 'selection'):
            assert e['uid'] in levels and e['origin'] == levels[e['uid']]['origin']
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1]/'scripts/diagnostics/mutation_audit/summarize_curriculum.py'
    spec = importlib.util.spec_from_file_location('audit_summary_test', script)
    summary_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary_module)
    summary = summary_module.summarize(tmp_path)
    assert summary['files'] == 1
    assert sum(r['completed'] for r in summary['by_origin'].values()) == sum(e['complete'] for e in outcomes)
    assert sum(r['partial'] for r in summary['by_origin'].values()) == sum(not e['complete'] for e in outcomes)


def test_origin_metrics_weight_workers_by_completed_episodes():
    metrics = WorkerCurriculumMetrics(2).update([
        {'plr/worker_id': 0, 'plr/replay_count': 8, 'plr/mutated_replay_count': 6,
         'plr/mutated_completed_count': 2, 'plr/mutated_return_sum': 100., 'plr/mutated_steps_sum': 120},
        {'plr/worker_id': 1, 'plr/replay_count': 2, 'plr/mutated_replay_count': 1,
         'plr/mutated_completed_count': 8, 'plr/mutated_return_sum': 200., 'plr/mutated_steps_sum': 300}])
    assert metrics['plr/mutated_return_mean_observed'] == 30.
    assert metrics['plr/mutated_episode_length_mean_observed'] == 42.
    assert metrics['plr/mutated_replay_share_observed'] == .7


def test_audit_replay_probability_is_the_actual_distribution_before_staleness_update():
    b = PLRBuffer(make_sampler('cartpole', seed=1), capacity=3, replay_prob=0, seed=1)
    for _ in range(3):
        slot, _ = b.sample()
        b.update(slot, float(slot+1))
    b.replay_prob = 1.
    before = b.sampling_distribution().copy()
    slot, _ = b.sample()
    assert b.last_replay_probability == pytest.approx(before[slot])


def test_campaign_v2_declares_protocol_scale_and_durable_logging(tmp_path):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    import yaml
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('v2_campaign_test', repo/'scripts/paper_campaign.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'protocol': campaign.PROTOCOL, 'provenance': campaign.provenance(),
        'matrix': [['cartpole', 'static_plr_mutations', 0]],
        'static_mutation_protocol': 'relative-reflect-symmetric-v2',
        'ablation_protocol': 'static-mutation-relative-v2',
        'training': {'tasks': {'cartpole': {'dr_samples': pool()}}}}))
    campaign.run(SimpleNamespace(stage='final', index=0, root=tmp_path/'output', manifest=manifest,
                                project='nsgym-paper', group='v2', wandb_mode='disabled',
                                protect_unscored='true', dry_run=True))
    folder = tmp_path/'output/final/cartpole/static_plr_mutations/seed0'
    cfg = yaml.safe_load((folder/'resolved_config.yaml').read_text())
    assert cfg['env']['plr']['static_mutation_protocol'] == 'relative-reflect-symmetric-v2'
    assert cfg['env']['plr']['audit_dir'] == str(folder/'curriculum')
    assert cfg['experiment']['comparison_role'] == 'ablation'
    assert cfg['collector']['total_frames'] == 10000000
    protocol = json.loads((folder/'static_protocol.json').read_text())
    assert protocol['mutation_protocol'] == 'relative-reflect-symmetric-v2'
    assert protocol['reference_scales']['masspole'] == .1
