"""Independent MuJoCo integration and reward-accounting regression checks."""

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import AAMAS_Comp


@pytest.mark.parametrize('notify,delta', [(False, False), (True, False), (True, True)])
def test_ant_v2_matches_independent_physics_and_reward(notify, delta):
    wrapped = gym.make('ExampleNSAnt-v2', terminate_when_unhealthy=False,
                       change_notification=notify, delta_change_notification=delta)
    native = gym.make('Ant-v5', terminate_when_unhealthy=False)
    try:
        wrapped.reset(seed=42)
        native.reset(seed=42)
        ant = native.unwrapped
        model, data = ant.model, ant.data
        body = model.body('torso').id
        initial_mass = float(model.body_mass[body])
        rng = np.random.default_rng(7)
        for t in range(1000):
            # Cache the native Ant baseline BEFORE the physical intervention.
            before = data.body(ant._main_body).xpos[:2].copy()
            changed = t in range(100, 901, 100)
            if changed:
                spec = mujoco.mjtState.mjSTATE_INTEGRATION
                state = np.empty(mujoco.mj_stateSize(model, spec))
                mujoco.mj_getState(model, data, state, spec)
                model.body_mass[body] *= .9
                mujoco.mj_setConst(model, data)
                mujoco.mj_setState(model, data, state, spec)
                mujoco.mj_forward(model, data)
            action = rng.uniform(-.3, .3, 8)
            obs, reward, done, truncated, info = wrapped.step(action)
            expected_obs, native_reward, expected_done, expected_truncated, _ = native.step(action)
            velocity = (data.body(ant._main_body).xpos[:2] - before) / ant.dt
            # Independent explicit native reward formula, including all costs.
            expected_reward = (ant._forward_reward_weight * velocity[0] + ant.healthy_reward
                               - ant.control_cost(action) - ant.contact_cost)
            np.testing.assert_allclose(obs['state'], expected_obs, atol=1e-10, rtol=0)
            assert reward == pytest.approx(expected_reward, abs=1e-10)
            if not changed:
                assert reward == pytest.approx(native_reward, abs=1e-10)
            assert info['reward_forward'] == pytest.approx(ant._forward_reward_weight * velocity[0])
            assert info['x_velocity'] == pytest.approx(velocity[0])
            assert (done, truncated) == (expected_done, expected_truncated)
            assert obs['env_change']['torso_mass'] == int(changed and notify)
            assert wrapped.unwrapped.model.body_mass[body] == pytest.approx(model.body_mass[body])
        wrapped.reset(seed=42)
        assert wrapped.unwrapped.model.body_mass[body] == initial_mass
    finally:
        wrapped.close()
        native.close()
