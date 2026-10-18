import pytest
import torch
from tensordict import TensorDict
from torchrl.objectives.value import GAE

from AAMAS_Comp.agents.episode_batches import CompleteEpisodes, estimate_advantages


def batch(ids, steps, rewards, done, terminated=None):
    shape = torch.tensor(ids).shape
    return TensorDict({
        'collector': TensorDict({'traj_ids': torch.tensor(ids)}, batch_size=shape),
        'step_count': torch.tensor(steps).unsqueeze(-1),
        'state_value': torch.full((*shape, 1), .4),
        'next': TensorDict({
            'reward': torch.tensor(rewards, dtype=torch.float64).unsqueeze(-1),
            'done': torch.tensor(done).unsqueeze(-1),
            'terminated': torch.tensor(terminated if terminated is not None else done).unsqueeze(-1),
            'state_value': torch.full((*shape, 1), .8),
        }, batch_size=shape),
    }, batch_size=shape)


def test_episode_carry_preserves_all_frames_and_isolates_workers():
    episodes = CompleteEpisodes()
    first = batch([[10, 10], [20, 21]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[False, False], [True, False]])
    ready = episodes.update(first)
    assert ready['collector', 'traj_ids'].tolist() == [20]
    first['next', 'reward'].fill_(9)  # collector storage reuse must not alter pending samples
    second = batch([[10, 11], [21, 21]], [[2, 0], [1, 2]], [[1, 1], [0, 1]], [[True, True], [False, True]])
    ready = episodes.update(second)
    assert ready['collector', 'traj_ids'].tolist() == [10, 10, 10, 11, 21, 21, 21]
    assert ready['next', 'reward'].flatten().tolist() == [0, 0, 1, 1, 0, 0, 1]
    assert episodes.frames_emitted == 8
    assert episodes.pending_frames == 0


def test_episode_carry_accepts_stacked_collector_groups():
    episodes = CompleteEpisodes()
    data = batch([[[1, 1]], [[2, 2]]], [[[0, 1]], [[0, 1]]], [[[0, 1]], [[0, 0]]], [[[False, True]], [[False, False]]])
    assert episodes.update(data).numel() == 2
    assert episodes.pending_frames == 2


def test_episode_carry_scopes_repeated_ids_to_environment_rows():
    episodes = CompleteEpisodes()
    data = batch([[[0, 0]], [[0, 0]]], [[[0, 1]], [[0, 1]]],
                 [[[0, 1]], [[0, 0]]], [[[False, True]], [[False, False]]])
    ready = episodes.update(data)
    assert ready.numel() == 2
    assert ready['next', 'reward'].sum() == 1
    assert episodes.pending_frames == 2


def test_episode_carry_rejects_missing_start_or_gap():
    episodes = CompleteEpisodes()
    with pytest.raises(ValueError, match='Incomplete trajectory'):
        episodes.update(batch([1], [1], [0], [False]))
    assert episodes.update(batch([2], [0], [0], [False])) is None
    with pytest.raises(ValueError, match='Incomplete trajectory'):
        episodes.update(batch([2], [2], [1], [True]))


def test_full_episode_lambda_one_targets_match_returns_at_timeout():
    data = batch([1, 1, 2, 2], [0, 1, 0, 1], [0, 1, 0, 0],
                 [False, True, False, True], [False, True, False, False])
    # Each nonterminal next state is the following step's current state.
    data['next', 'state_value'] = torch.tensor([[.4], [.8], [.4], [.8]])
    estimator = GAE(gamma=1, lmbda=1, value_network=None, average_gae=False)
    estimate_advantages(estimator, data, bootstrap_timeouts=False)
    assert torch.allclose(data['value_target'].flatten(), data['value_target'].new_tensor([1., 1., 0., 0.]))
    assert not data['next', 'terminated'][-1]  # recorded environment outcomes unchanged


def test_ordinary_nonterminal_bootstrapping_is_retained():
    data = batch([1], [0], [0], [False])
    estimate_advantages(GAE(gamma=1, lmbda=1, value_network=None, average_gae=False), data, bootstrap_timeouts=False)
    assert torch.allclose(data['value_target'], torch.full_like(data['value_target'], .8))
