"""Fast temporal dynamics and static calibration must obey native constraints."""
import warnings
from copy import deepcopy
import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from ns_gym.wrappers import NSClassicControlWrapper

from AAMAS_Comp.envs.domain_randomization import seed_update_functions, set_parameters
from AAMAS_Comp.envs.fast_ns_wrapper import FastNSClassicControlWrapper
from AAMAS_Comp.envs.ns_env_factory import NSEnvConfig, ParamConfig, SchedulerConfig, UpdateFnConfig
from AAMAS_Comp.envs.static_curriculum import StaticDynamicsSampler
from AAMAS_Comp.evaluation.paper import calibrate_ranges


def config():
    def param(cls, **kwargs):
        return ParamConfig(SchedulerConfig('ContinuousScheduler'), UpdateFnConfig(cls, kwargs))
    return NSEnvConfig('CartPole-v1', {
        'masspole': param('DecrementUpdate', k=.002),
        'masscart': param('RandomWalk', sigma=.3),
        'length': param('DecrementUpdate', k=.005),
        'gravity': param('RandomWalk', sigma=2.),
    })


def test_fast_cartpole_matches_native_through_rejected_updates():
    c = config()
    environments = [cls(gym.make('CartPole-v1'), {k: pc.build() for k, pc in c.tunable_params.items()})
                    for cls in (FastNSClassicControlWrapper, NSClassicControlWrapper)]
    try:
        for e in environments:
            e.reset(seed=3)
            seed_update_functions(e, 13)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for _ in range(200):
                observations = []
                for e in environments:
                    # Keep the episode alive to probe every proposed update,
                    # rather than ending before a constraint boundary is reached.
                    e.unwrapped.state = np.zeros(4)
                    obs, _, _, _, _ = e.step(0)
                    observations.append(obs['state'] if isinstance(obs, dict) else obs)
                for key in c.tunable_params:
                    assert getattr(environments[0].unwrapped, key) == getattr(environments[1].unwrapped, key)
                np.testing.assert_allclose(*observations, rtol=1e-12, atol=1e-12)
        assert environments[0].rejected_updates > 0
        environments[0].reset(seed=3)
        assert environments[0].rejected_updates == 0
    finally:
        for e in environments:
            e.close()


def test_cartpole_calibration_rejects_updates_before_advancing_state(monkeypatch):
    import AAMAS_Comp.evaluation.paper as paper
    class Sampler:
        def sample(self):
            c = config()
            # Deterministic values cross each constraint during this trajectory.
            c.tunable_params['masscart'].update_fn = UpdateFnConfig('DecrementUpdate', {'k': .02})
            c.tunable_params['gravity'].update_fn = UpdateFnConfig('DecrementUpdate', {'k': .2})
            return c
    monkeypatch.setitem(paper.NS_ENV_SAMPLERS, 'cartpole', lambda seed: Sampler())
    ranges, _, snapshots = calibrate_ranges('cartpole', count=1, seed=3)
    assert len(snapshots) == 16
    for key in ('masspole', 'masscart', 'length'):
        assert ranges[key][0] > 0 and snapshots[-1][key] > 0
    assert ranges['gravity'][0] >= 0
    # Once rejected, a decrement cannot move deeper into negative state.
    assert snapshots[-1] == snapshots[-2]


@pytest.mark.parametrize('key,value', [('masspole', -.1), ('length', 0), ('gravity', -1)])
def test_static_cartpole_rejects_invalid_physics_before_modifying_simulator(key, value):
    values = {'masspole': .1, 'masscart': 1., 'length': .5, 'gravity': 9.8}
    values[key] = value
    with pytest.raises(ValueError):
        StaticDynamicsSampler('cartpole', [values])
    env = gym.make('CartPole-v1')
    try:
        before = env.unwrapped.masspole
        with pytest.raises(ValueError):
            set_parameters(env, 'cartpole', values)
        assert env.unwrapped.masspole == before
    finally:
        env.close()


def test_evaluator_fails_before_policy_action_on_invalid_initial_physics():
    from AAMAS_Comp.evaluation.paper import episode
    env = gym.make('CartPole-v1')
    env.unwrapped.masspole = -.1
    try:
        with pytest.raises(ValueError, match='Invalid CartPole physics at evaluation reset'):
            episode(None, {}, env, 'cartpole', 0)
    finally:
        env.close()


def test_corrected_campaign_preserves_cases_budgets_and_rebuilds_both_valid_pools(monkeypatch):
    import AAMAS_Comp.evaluation.paper as paper
    path = Path(__file__).resolve().parents[1]/'scripts/diagnostics/cartpole_constraints/campaign.py'
    spec = importlib.util.spec_from_file_location('cartpole_native_campaign', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    defaults = dict(masspole=.1, masscart=1., length=.5, gravity=9.8)
    def calibration(task, count, seed):
        assert task == 'cartpole'
        values = {**defaults, 'gravity': float(seed)}
        return {k: [v, v] for k, v in values.items()}, defaults, [values.copy() for _ in range(count*16)]
    monkeypatch.setattr(paper, 'calibrate_ranges', calibration)
    suites = {'iid': [{'id': 0, 'config': {'test': 'preserved'}, 'seeds': [10, 11]}]}
    for label, duration in [('abrupt', 0), ('gradual', 125), ('no_change', -1)]:
        suites[label] = [{'id': 0, 'seeds': [10, 11], 'intervention': {
            'initial': defaults, 'target': {**defaults, 'gravity': -2}, 'step': 50, 'duration': duration}}]
    original = {'training': {'calibration_seed': 71311, 'tasks': {'cartpole': {'old': True}}},
                'calibration_seed': 71301, 'tasks': {'cartpole': {'suites': suites}}}
    old = deepcopy(original)
    definition = driver.corrected_definition(original, count=2)
    assert original == old
    assert len(definition['matrix']) == 35 and len(set(m for _, m, _ in definition['matrix'])) == 7
    assert definition['tasks']['cartpole']['suites']['iid'] == suites['iid']
    assert len(definition['training']['tasks']['cartpole']['dr_samples']) == 32
    for label in ('abrupt', 'gradual', 'no_change'):
        case = definition['tasks']['cartpole']['suites'][label][0]
        assert case['seeds'] == [10, 11] and case['intervention']['step'] == 50
        assert case['intervention']['duration'] == suites[label][0]['intervention']['duration']
        assert case['intervention']['target']['gravity'] == 71301
    assert definition['training']['tasks']['cartpole']['dr_samples'][0]['gravity'] == 71311
    assert definition['physics_protocol'] == driver.PHYSICS
