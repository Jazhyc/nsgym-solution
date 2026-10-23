"""Bounded calibration, static routing, and independent intervention rewards."""
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from AAMAS_Comp.envs.ant_curriculum import ANT_BOUNDS, calibrate_ant_v2, validate_ant_samples
from AAMAS_Comp.envs.domain_randomization import DynamicsEnv, parameters
from AAMAS_Comp.envs.static_curriculum import StaticDynamicsSampler, build_static_level
from AAMAS_Comp.envs.torchrl_factory import static_training_definition, task_key


def test_calibration_is_reproducible_and_all_snapshots_and_mutants_are_bounded():
    a, b = calibrate_ant_v2(4, 71311), calibrate_ant_v2(4, 71311)
    assert a == b and len(a['dr_samples']) == 64
    sampler = StaticDynamicsSampler('ant', a['dr_samples'], seed=42)
    rng = np.random.default_rng(9)
    parent = sampler.sample()
    for _ in range(1000):
        parent = sampler.mutate(parent, rng)
        validate_ant_samples([parent.values])


def test_static_factory_maps_versioned_sampler_to_ant_and_rejects_old_pool(tmp_path):
    path = tmp_path / 'manifest.json'
    good = {'gravity': -9.81, 'torso_mass': .32724923474893675}
    path.write_text(json.dumps({'training': {'tasks': {'ant': {'dr_samples': [good]}}}}))
    cfg = OmegaConf.create({'env': {'id': 'Ant-v5', 'plr': {'sampler_key': 'ant_v2'}},
                            'experiment': {'manifest': str(path)}})
    assert task_key(cfg) == 'ant'
    assert static_training_definition(cfg)['dr_samples'] == [good]
    path.write_text(json.dumps({'training': {'tasks': {'ant': {'dr_samples': [dict(good, gravity=20)]}}}}))
    with pytest.raises(ValueError, match='recalibrate'):
        static_training_definition(cfg)


def test_mutated_static_level_keeps_exact_physics_throughout_episode():
    pool = calibrate_ant_v2(4, 71311)
    sampler = StaticDynamicsSampler('ant', pool['dr_samples'], seed=7)
    config = sampler.mutate(sampler.sample(), np.random.default_rng(3))
    env = build_static_level(config)
    try:
        env.reset(seed=6)
        for _ in range(1000):
            _, _, _, _, info = env.step(np.zeros(8))
            assert parameters(env, 'ant') == config.values
            assert not info['verified_change']
    finally:
        env.close()


@pytest.mark.parametrize('duration', [0, 25])
def test_corrected_intervention_reward_uses_pre_intervention_position(duration):
    env = DynamicsEnv('ant', {}, intervention={
        'initial': {'gravity': -9.81, 'torso_mass': .32724923474893675},
        'target': {'gravity': -6., 'torso_mass': .2}, 'step': 10, 'duration': duration},
        native_ant_reward=True)
    try:
        env.reset(seed=42)
        ant = env.unwrapped
        for _ in range(60):
            before = ant.data.body(ant._main_body).xpos[:2].copy()
            action = np.zeros(8)
            _, reward, _, _, info = env.step(action)
            velocity = (ant.data.body(ant._main_body).xpos[:2] - before) / ant.dt
            expected = (ant._forward_reward_weight * velocity[0] + ant.healthy_reward
                        - ant.control_cost(action) - ant.contact_cost)
            assert reward == pytest.approx(expected, abs=1e-10)
            values = parameters(env, 'ant')
            validate_ant_samples([values])
    finally:
        env.close()
