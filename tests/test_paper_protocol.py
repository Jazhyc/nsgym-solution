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
from AAMAS_Comp.envs.wrappers import FastNSFrozenLakeWrapper, SeededResetWrapper
from AAMAS_Comp.evaluation.statistics import trimmed_mean


def score_batch(worker, token, episode, advantages, done, steps=None):
    n = len(advantages)
    return TensorDict({"plr_worker": torch.full((n,), worker),
                       "plr_level": torch.full((n,), token),
                       "plr_episode": torch.full((n,), episode),
                       "step_count": torch.tensor(list(range(n)) if steps is None else steps).reshape(n, 1),
                       "advantage": torch.tensor(advantages).reshape(n, 1),
                       "next": {"done": torch.tensor(done).reshape(n, 1)}}, batch_size=[n])


def test_episode_feedback_crosses_batches_and_keeps_groups_separate():
    scores, queues = EpisodeScores(), [Queue(), Queue()]
    scores.update(score_batch(0, 2, 1, [1., -3.], [False, False]), queues)
    scores.update(score_batch(1, 2, 1, [8.], [True]), queues)
    scores.update(score_batch(0, 2, 1, [5.], [True], steps=[2]), queues)
    assert queues[0].get_nowait() == (2, 1, 3.)
    assert queues[1].get_nowait() == (2, 1, 8.)
    assert not scores.pending


@pytest.mark.parametrize("first,last", [([], [2, 3]), ([0], [2, 3]), ([0, 1], [1, 2])])
def test_episode_feedback_rejects_missing_or_duplicate_steps(first, last):
    scores, queue = EpisodeScores(), Queue()
    if first:
        scores.update(score_batch(0, 1, 1, [1.] * len(first), [False] * len(first), first), [queue])
    scores.update(score_batch(0, 1, 1, [2.] * len(last), [False] * (len(last)-1) + [True], last), [queue])
    assert queue.empty()
    assert scores.incomplete_episodes == 1
    assert not scores.pending and not scores.pending_steps


def test_episode_feedback_discards_abandoned_fragment_and_recovers():
    scores, queue = EpisodeScores(), Queue()
    scores.update(score_batch(0, 4, 1, [999.], [False]), [queue])
    scores.update(score_batch(0, 2, 2, [1., 3.], [False, True]), [queue])
    assert queue.get_nowait() == (2, 2, 2.)
    assert scores.abandoned_episodes == 1
    assert not scores.pending and not scores.pending_steps


def test_delayed_score_cannot_update_replacement_level():
    queue = Queue()
    env = PLREnv(plr_capacity=1, replay_prob=0, feedback_queue=queue, seed=3)
    try:
        _, info = env.reset(seed=4)
        token = int(info["plr_level"])
        queue.put((token, 1, 1.))  # first score makes this level eligible for eviction
        env.reset(seed=5)  # replaces the previous slot
        queue.put((token, 1, 999.))
        env.reset(seed=6)
        assert env._discarded_scores == 1
        assert all(e.score != 999 for e in env.plr._entries)
    finally:
        env.close()


def test_new_level_survives_until_first_async_feedback():
    queue = Queue()
    env = PLREnv(plr_capacity=2, replay_prob=0, feedback_queue=queue, seed=3)
    try:
        _, first = env.reset(seed=4)
        queue.put((int(first["plr_level"]), 1, 1.))
        _, second = env.reset(seed=5)
        second_token = int(second["plr_level"])
        env.reset(seed=6)  # exploration must replace a scored slot
        assert second_token in env._level_tokens.values()
        assert env.plr.last_was_replay is False
        queue.put((second_token, 2, 999.))
        env.reset(seed=7)
        assert env._applied_scores == 2
        assert env._discarded_scores == 0
    finally:
        env.close()


def test_full_unscored_buffer_waits_without_reassigning_feedback():
    queue = Queue()
    env = PLREnv(plr_capacity=1, replay_prob=0, mutation_prob=1,
                 feedback_queue=queue, seed=3)
    try:
        _, first = env.reset(seed=4)
        token = int(first["plr_level"])
        _, repeated = env.reset(seed=5)
        assert int(repeated["plr_level"]) == token
        assert env.plr.last_was_replay
        assert env.plr.last_was_forced_replay
        assert not env.plr.last_was_mutation
        queue.put((token, 1, 1.))
        _, replacement = env.reset(seed=6)
        assert int(replacement["plr_level"]) != token
        assert env.plr.last_was_mutation
        assert not env.plr.last_was_forced_replay
        assert env._applied_scores == 1
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


def test_memoryless_scheduler_initial_event_is_reseeded_on_reset():
    from AAMAS_Comp.envs.domain_randomization import seed_update_functions

    config = NSEnvConfig("FrozenLake-v1", {"P": ParamConfig(
        SchedulerConfig("MemorylessScheduler", {"p": .1}),
        UpdateFnConfig("DistributionDecrementUpdate", {"k": .01}))})
    traces = []
    for _ in range(3):
        env = FixedNSEnv(config)
        try:
            for _ in range(2):
                env.reset(seed=123)
                seed_update_functions(env, 123)
                fn = env._env.env._update_fn
                p = [.8, .1, .1]
                trace = []
                for t in range(100):
                    p = fn(list(p), t)[0]
                    trace.append(tuple(p))
                traces.append(trace)
        finally:
            env.close()
    assert all(trace == traces[0] for trace in traces)
    assert len(set(traces[0])) > 1


def test_iqm_trims_by_rank_even_with_binary_ties():
    assert trimmed_mean([0, 0, 0, 1, 1, 1, 1, 1]) == .75
    assert trimmed_mean([1, 1, 1]) == 1


@pytest.mark.parametrize("slippery", [True, False])
@pytest.mark.parametrize("probabilities", [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.7, .2, .1]])
def test_fast_frozenlake_matches_ns_gym_transition_kernel(slippery, probabilities):
    import gymnasium as gym
    from ns_gym.wrappers import NSFrozenLakeWrapper

    def update():
        return ParamConfig(SchedulerConfig("ContinuousScheduler"),
                           UpdateFnConfig("UniformDrift", {"rate": 0.0})).build()

    fast = FastNSFrozenLakeWrapper(gym.make("FrozenLake-v1", is_slippery=slippery),
                                   {"P": update()}, initial_prob_dist=probabilities)
    reference = NSFrozenLakeWrapper(gym.make("FrozenLake-v1", is_slippery=slippery),
                                    {"P": update()}, initial_prob_dist=probabilities)
    try:
        for state in range(16):
            for action in range(4):
                expected = {}
                for p, ns, reward, done in reference.P[state][action]:
                    key = (ns, reward, done)
                    expected[key] = expected.get(key, 0.0) + p
                actual = {}
                for i, p in enumerate(probabilities):
                    key = (int(fast._next_states[state, action, i]),
                           float(fast._rewards[state, action, i]),
                           bool(fast._dones[state, action, i]))
                    actual[key] = actual.get(key, 0.0) + p
                assert actual == pytest.approx(expected)
    finally:
        fast.close()
        reference.close()


def test_fast_frozenlake_seed_reproduces_stock_simulator_trajectories():
    import gymnasium as gym
    from ns_gym.wrappers import NSFrozenLakeWrapper
    from AAMAS_Comp.envs.domain_randomization import seed_update_functions

    def update():
        return ParamConfig(SchedulerConfig("ContinuousScheduler"),
                           UpdateFnConfig("UniformDrift", {"rate": .01})).build()

    fast = FastNSFrozenLakeWrapper(gym.make("FrozenLake-v1"), {"P": update()},
                                   initial_prob_dist=[.7, .2, .1])

    def trace(env, seed):
        env.reset(seed=seed)
        seed_update_functions(env, seed)
        result = []
        for action in [0, 1, 2, 3] * 30:
            obs, reward, done, truncated, _ = env.step(action)
            result.append((obs["state"], float(reward), done))
            if done or truncated:
                break
        return result

    try:
        for seed in range(10):
            reference = NSFrozenLakeWrapper(gym.make("FrozenLake-v1"), {"P": update()},
                                            initial_prob_dist=[.7, .2, .1])
            try:
                expected = trace(reference, seed)
                assert trace(fast, seed) == expected
                assert trace(fast, seed) == expected
            finally:
                reference.close()
    finally:
        fast.close()


def test_stationary_frozenlake_uses_declared_training_horizon():
    from omegaconf import OmegaConf
    from AAMAS_Comp.envs.torchrl_factory import make_single_env

    cfg = OmegaConf.create({"seed": 3, "env": {"id": "FrozenLake-v1"},
                           "collector": {"max_frames_per_traj": 200}})
    env = make_single_env(cfg, torch.device("cpu"))
    try:
        assert env.base_env._env.get_wrapper_attr("_max_episode_steps") == 200
    finally:
        env.close()


def test_aggregate_replaces_entire_task_and_preserves_other_tasks(tmp_path):
    import json
    from AAMAS_Comp.evaluation.paper import aggregate

    def summary(root, task, method, seed, value, manifest):
        path = root / task / method / f"seed{seed}" / "test_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metadata": {"task": task, "method": method, "seed": seed},
                                    "manifest_sha256": manifest, "metrics": {"test/iid/mean": value}}))

    original, corrected = tmp_path / "original", tmp_path / "corrected"
    summary(original, "cartpole", "stationary", 0, 500, "original")
    summary(original, "frozenlake", "stationary", 0, .1, "original")
    summary(corrected, "frozenlake", "stationary", 0, .8, "corrected")
    rows = aggregate(original, tmp_path / "results.json", replacements={"frozenlake": corrected})
    assert {r["task"]: r["mean"] for r in rows} == {"cartpole": 500, "frozenlake": .8}
    summary(corrected, "frozenlake", "full", 1, .7, "wrong-manifest")
    with pytest.raises(ValueError, match="within each task"):
        aggregate(original, tmp_path / "results.json", replacements={"frozenlake": corrected})


def test_dr_calibration_reseeds_memoryless_constructor_event(monkeypatch):
    from AAMAS_Comp.evaluation import paper

    config = NSEnvConfig("CartPole-v1", {"gravity": ParamConfig(
        SchedulerConfig("MemorylessScheduler", {"p": .1}),
        UpdateFnConfig("IncrementUpdate", {"k": .01}))})

    class Sampler:
        def __init__(self, seed):
            pass

        def sample(self):
            return config

    monkeypatch.setitem(paper.NS_ENV_SAMPLERS, "cartpole", Sampler)
    first = paper.calibrate_ranges("cartpole", count=4, seed=123)
    assert first == paper.calibrate_ranges("cartpole", count=4, seed=123)
    assert first != paper.calibrate_ranges("cartpole", count=4, seed=124)
