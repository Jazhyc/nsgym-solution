import pytest
import torch
from tensordict import TensorDict

from AAMAS_Comp.evaluation.training_metrics import TrainingEpisodes, WorkerCurriculumMetrics


def batch(ids, rewards, done, terminal):
    n = len(ids)
    return TensorDict({"collector": {"traj_ids": torch.tensor(ids)}, "next": {
        "reward": torch.tensor(rewards).reshape(n, 1),
        "done": torch.tensor(done).reshape(n, 1),
        "terminated": torch.tensor(terminal).reshape(n, 1)}}, batch_size=[n])


def test_completed_episode_metrics_cross_batches_and_distinguish_timeout_hole():
    logger = TrainingEpisodes(frozenlake=True)
    partial = logger.update(batch([5, 9, 9], [0., 0., 0.], [False]*3, [False]*3))
    assert partial == {"train/episodes_completed_total": 0, "train/episodes_completed_batch": 0}
    metrics = logger.update(batch([5, 9, 9, 11], [1., 0., 0., 0.],
                                  [True, False, True, False], [True, False, False, False]))
    assert metrics["train/episode_return_mean"] == .5
    assert metrics["train/episode_length_mean"] == 3
    assert metrics["train/success_rate"] == .5
    assert metrics["train/hole_rate"] == 0
    assert metrics["train/timeout_rate"] == .5
    hole = logger.update(batch([11], [0.], [True], [True]))
    assert hole["train/hole_rate"] == 1
    assert hole["train/timeout_rate"] == 0
    assert hole["train/episodes_completed_total"] == 3
    assert not logger.pending


def test_worker_stats_use_latest_snapshot_and_weight_episode_counts():
    logger = WorkerCurriculumMetrics(n_workers=2)
    first = logger.update([
        {"plr/worker_id": 0, "plr/applied_scores": 1, "plr/discarded_scores": 1,
         "plr/episode_count": 10, "plr/replay_count": 1},
        {"plr/worker_id": 0, "plr/applied_scores": 3, "plr/discarded_scores": 1,
         "plr/episode_count": 20, "plr/replay_count": 2}])
    assert first["plr/applied_scores_observed_total"] == 3
    assert first["plr/workers_reporting"] == 1
    assert first["plr/feedback_discard_fraction_observed"] == .25
    metrics = logger.update([{"plr/worker_id": 1, "plr/applied_scores": 5,
        "plr/discarded_scores": 0, "plr/episode_count": 5, "plr/replay_count": 4}])
    assert metrics["plr/applied_scores_observed_total"] == 8
    assert metrics["plr/feedback_discard_fraction_observed"] == pytest.approx(1/9)
    assert metrics["plr/replay_fraction"] == pytest.approx(6/25)
    assert metrics["plr/worker_coverage"] == 1


def test_stacked_groups_initial_trajectory_ids_do_not_merge_episodes():
    logger = TrainingEpisodes(frozenlake=True)
    data = batch([0, 0, 0, 0], [0., 1., 0., 0.],
                 [False, True, False, True], [False, True, False, False]).reshape(2, 1, 2)
    metrics = logger.update(data)
    assert metrics['train/episodes_completed_total'] == 2
    assert metrics['train/success_rate'] == .5
    assert metrics['train/timeout_rate'] == .5
    assert metrics['train/episode_length_mean'] == 2


def test_unscored_eviction_counter_detects_old_bug_and_stays_zero_with_fix():
    from AAMAS_Comp.curriculum.plr import PLRBuffer
    from AAMAS_Comp.envs.ns_env_sampler import NS_ENV_SAMPLERS
    for protect in (True, False):
        buffer = PLRBuffer(NS_ENV_SAMPLERS["frozenlake"](seed=1), capacity=1,
                           replay_prob=0, protect_unscored=protect, seed=1)
        buffer.sample()
        buffer.sample()
        assert buffer.stats()["plr/unscored_evictions"] == (0 if protect else 1)
        assert buffer.stats()["plr/levels_first_scored"] == 0
        slot, _ = buffer.sample()
        buffer.update(slot, 0.)  # a genuinely zero score is still evaluated
        assert buffer.stats()["plr/levels_first_scored"] == 1
