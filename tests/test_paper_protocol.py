"""Scientific protocol checks: feedback identity, randomness and actual dynamics."""

from queue import Queue

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from AAMAS_Comp.curriculum.feedback import EpisodeScores
from AAMAS_Comp.curriculum.plr_env import FixedNSEnv, PLREnv
from AAMAS_Comp.envs.domain_randomization import DynamicsEnv, parameters
from AAMAS_Comp.envs.ns_env_factory import NSEnvConfig, ParamConfig, SchedulerConfig, UpdateFnConfig
from AAMAS_Comp.envs.wrappers import SeededResetWrapper
from AAMAS_Comp.evaluation.statistics import trimmed_mean


def score_batch(worker, token, episode, advantages, done):
    n = len(advantages)
    return TensorDict({"plr_worker": torch.full((n,), worker),
                       "plr_level": torch.full((n,), token),
                       "plr_episode": torch.full((n,), episode),
                       "advantage": torch.tensor(advantages).reshape(n, 1),
                       "next": {"done": torch.tensor(done).reshape(n, 1)}}, batch_size=[n])


def test_episode_feedback_crosses_batches_and_keeps_groups_separate():
    scores, queues = EpisodeScores(), [Queue(), Queue()]
    scores.update(score_batch(0, 2, 1, [1., -3.], [False, False]), queues)
    scores.update(score_batch(1, 2, 1, [8.], [True]), queues)
    scores.update(score_batch(0, 2, 1, [5.], [True]), queues)
    assert queues[0].get_nowait() == (2, 1, 3.)
    assert queues[1].get_nowait() == (2, 1, 8.)
    assert not scores.pending


def test_delayed_score_cannot_update_replacement_level():
    queue = Queue()
    env = PLREnv(plr_capacity=1, replay_prob=0, feedback_queue=queue, seed=3)
    try:
        _, info = env.reset(seed=4)
        token = int(info["plr_level"])
        env.reset(seed=5)  # replaces the previous slot
        queue.put((token, 1, 999.))
        env.reset(seed=6)
        assert env._discarded_scores == 1
        assert all(e.score != 999 for e in env.plr._entries)
    finally:
        env.close()


@pytest.mark.parametrize("task,ranges", [
    ("cartpole", {"masspole": [.1, .2], "length": [.4, .7]}),
    ("ant", {"torso_mass": [.3, .5], "gravity": [-12, -8]}),
    ("frozenlake", {}),
])
def test_standard_dr_changes_at_reset_and_stays_constant(task, ranges):
    env = DynamicsEnv(task, ranges, seed=3)
    try:
        env.reset(seed=4)
        before = parameters(env, task)
        env.step(env.action_space.sample())
        after = parameters(env, task)
        assert all(np.allclose(before[k], after[k]) for k in before)
        env.reset(seed=5)
        new = parameters(env, task)
        assert any(not np.allclose(before[k], new[k]) for k in before)
    finally:
        env.close()


@pytest.mark.parametrize("task,initial,target", [
    ("cartpole", {"masspole": .1, "masscart": 1., "length": .5, "gravity": 9.8},
     {"masspole": .2, "masscart": 1., "length": .8, "gravity": 12.}),
    ("ant", {"torso_mass": .327, "gravity": -9.81}, {"torso_mass": .5, "gravity": -12.}),
    ("frozenlake", {"P": [1/3]*3}, {"P": [.9, .05, .05]}),
])
def test_controlled_shift_reaches_requested_simulator_values(task, initial, target):
    env = DynamicsEnv(task, {}, intervention={"initial": initial, "target": target, "step": 1, "duration": 0})
    try:
        env.reset(seed=3)
        action = env.action_space.sample()
        env.step(action)
        assert all(np.allclose(parameters(env, task)[k], initial[k]) for k in initial)
        *_, info = env.step(action)
        assert info["verified_change"]
        assert all(np.allclose(parameters(env, task)[k], target[k]) for k in target)
    finally:
        env.close()


def test_stochastic_functions_and_fixed_reset_reproduce():
    config = NSEnvConfig("CartPole-v1", {"gravity": ParamConfig(
        SchedulerConfig("RandomScheduler", {"probability": .5}),
        UpdateFnConfig("RandomWalk", {"sigma": .01}))})
    env = SeededResetWrapper(FixedNSEnv(config), 3)
    try:
        traces = []
        for _ in range(2):
            env.reset(seed=4)
            values = []
            for i in range(10):
                env.step(i % 2)
                values.append(parameters(env, "cartpole")["gravity"])
            traces.append(values)
        assert traces[0] == traces[1]
        assert len(set(traces[0])) > 1
    finally:
        env.close()


def test_iqm_trims_by_rank_even_with_binary_ties():
    assert trimmed_mean([0, 0, 0, 1, 1, 1, 1, 1]) == .75
    assert trimmed_mean([1, 1, 1]) == 1
