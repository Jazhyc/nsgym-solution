"""Temporal Ant regression tests against actual NS-Gym update semantics."""

from copy import deepcopy
import math

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from pathlib import Path

from AAMAS_Comp.envs.ant_curriculum import ANT_BOUNDS, ANT_TORSO_MASS, BoundedAntWrapper
from AAMAS_Comp.envs.ns_env_sampler import make_sampler, ANT_PARAM_SPECS
from AAMAS_Comp.envs.ns_env_factory import NSEnvConfig, NSEnvFactory, ParamConfig, SchedulerConfig, UpdateFnConfig


def test_continuous_trend_and_decay_are_calibrated_for_cumulative_time():
    before = deepcopy(ANT_PARAM_SPECS)
    sampler = make_sampler('ant_v2', seed=0)
    for name, start, expected in [('gravity', -9.81, -9.81 + 9e-6 * 499500),
                                  ('torso_mass', ANT_TORSO_MASS, ANT_TORSO_MASS * math.exp(-2e-6 * 499500))]:
        spec = next(s for s in sampler.param_specs if s.param_name == name)
        fn_name = 'DeterministicTrend' if name == 'gravity' else 'ExponentialDecay'
        kwargs = {k: v[1] for k, v in spec.update_fn_overrides[fn_name].items()}
        fn = ParamConfig(SchedulerConfig('ContinuousScheduler'), UpdateFnConfig(fn_name, kwargs)).build()
        value = start
        for t in range(1000):
            value, _, _ = fn(value, t)
            assert ANT_BOUNDS[name][0] <= value <= ANT_BOUNDS[name][1]
        assert value == pytest.approx(expected)
    assert ANT_PARAM_SPECS == before  # Historical protocol remains identical.


@pytest.mark.parametrize('mutation', [False, True])
def test_seeded_fresh_and_mutated_levels_keep_protocol(mutation):
    a, b = make_sampler('ant_v2', seed=8), make_sampler('ant_v2', seed=8)
    ra, rb = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(100):
        ca, cb = a.sample(), b.sample()
        if mutation:
            parent = deepcopy(ca)
            ca, cb = a.mutate(ca, ra), b.mutate(cb, rb)
            assert parent.wrapper_kwargs == ca.wrapper_kwargs
        assert ca == cb
        assert ca.wrapper_kwargs['ant_physics_protocol'] == 'bounded-v2'


def test_invalid_physics_is_rejected_and_notification_describes_applied_change():
    cfg = NSEnvConfig('Ant-v5', {
        'gravity': ParamConfig(SchedulerConfig('ContinuousScheduler'), UpdateFnConfig('IncrementUpdate', {'k': 100})),
        'torso_mass': ParamConfig(SchedulerConfig('ContinuousScheduler'), UpdateFnConfig('DecrementUpdate', {'k': 1})),
    }, change_notification=True, delta_change_notification=True,
        wrapper_kwargs={'ant_physics_protocol': 'bounded-v2'},
        gym_kwargs={'terminate_when_unhealthy': False})
    env = NSEnvFactory.make(cfg)
    try:
        assert isinstance(env, BoundedAntWrapper)
        env.reset(seed=5)
        for _ in range(1000):
            obs, reward, _, _, info = env.step(np.zeros(8))
            assert np.isfinite(reward)
            assert env.unwrapped.model.opt.gravity[-1] == pytest.approx(-9.81)
            assert env.unwrapped.model.body('torso').mass[0] == pytest.approx(ANT_TORSO_MASS)
            assert not any(obs['env_change'].values())
            assert not any(obs['delta_change'].values())
        assert info['ant/rejected_updates'] == 2000
        env.reset(seed=5)
        assert env.rejected_updates == 0
    finally:
        env.close()


def test_new_training_config_is_explicit_and_synchronous():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'config'), version_base=None):
        cfg = compose(config_name='config_ant_v2')
    assert cfg.env.plr.sampler_key == 'ant_v2'
    assert cfg.collector.synchronous and cfg.training.minibatch_kl_guard
    assert cfg.collector.frames_per_batch % cfg.collector.num_envs == 0
    assert cfg.collector.frames_per_batch % cfg.training.sub_batch_size == 0
    assert not cfg.agent.use_layer_norm
