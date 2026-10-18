import numpy as np
import pytest

from AAMAS_Comp.envs.domain_randomization import static_env
from AAMAS_Comp.evaluation.frozenlake_diagnostics import exact_returns


@pytest.mark.parametrize('greedy', [True, False])
def test_exact_return_stops_at_goal_and_discounts_only_elapsed_steps(greedy):
    env = static_env('frozenlake')
    try:
        wrapper = env.env
        # Six deterministic actions: 0 -> 4 -> 8 -> 9 -> 13 -> 14 -> 15.
        actions = np.zeros(16, dtype=int)
        actions[[0, 4, 9]] = 1
        actions[[8, 13, 14]] = 2
        policies = np.eye(4)[actions][None, :, :]
        timeline = np.tile([1., 0., 0.], (1, 10, 1))
        result = exact_returns(policies, timeline, wrapper._next_states,
                               wrapper._rewards, wrapper._dones, greedy=greedy)
        assert result[0, 0] == pytest.approx([1., .99 ** 5])
        result = exact_returns(policies, timeline[:, :5], wrapper._next_states,
                               wrapper._rewards, wrapper._dones, greedy=greedy)
        assert result[0, 0] == pytest.approx([0., 0.])
    finally:
        env.close()


def test_exact_return_keeps_dynamics_trajectories_separate():
    env = static_env('frozenlake')
    try:
        wrapper = env.env
        policies = np.ones((1, 16, 4)) / 4
        timelines = np.tile(np.array([[1., 0., 0.], [.1, .6, .3]])[:, None, :], (1, 20, 1))
        result = exact_returns(policies, timelines, wrapper._next_states,
                               wrapper._rewards, wrapper._dones)
        separate = np.concatenate([exact_returns(policies, t[None], wrapper._next_states,
                                  wrapper._rewards, wrapper._dones) for t in timelines], axis=1)
        assert result == pytest.approx(separate)
    finally:
        env.close()
