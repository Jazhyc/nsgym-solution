"""Exercise realized MuJoCo dynamics, rather than just the update function."""

import warnings
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from ns_gym.wrappers.mujoco_env import ConstraintViolationWarning

import AAMAS_Comp  # register evaluation environments
from AAMAS_Comp.evaluation.public import evaluate_public_checkpoint


@pytest.mark.parametrize('notify,delta', [(True, True), (True, False), (False, False)])
def test_corrected_ant_realizes_mass_changes_and_resets(notify, delta):
    env = gym.make('ExampleNSAnt-v1', change_notification=notify,
                   delta_change_notification=delta, terminate_when_unhealthy=False)
    try:
        env.reset(seed=42)
        model = env.unwrapped.model
        torso = model.body('torso').id
        initial = float(model.body_mass[torso])
        subtree = float(model.body_subtreemass[torso])
        trace = []
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConstraintViolationWarning)
            for time in range(1000):
                obs, reward, terminated, truncated, info = env.step(np.zeros(8))
                events = min(time // 100, 9)
                expected = initial * 0.9**events
                changed = time in range(100, 901, 100)
                assert model.body_mass[torso] == pytest.approx(expected)
                # Derived MuJoCo constants must reflect the actual mass change.
                assert model.body_subtreemass[torso] == pytest.approx(subtree - initial + expected)
                assert info['Ground Truth Env Change']['torso_mass'] == int(changed)
                assert obs['env_change']['torso_mass'] == int(changed and notify)
                assert bool(obs['delta_change']['torso_mass']) == (changed and delta)
                assert np.isfinite(reward) and np.isfinite(obs['state']).all()
                assert not terminated
                assert truncated == (time == 999)
                trace.append((expected, obs['state'].copy(), reward))
        assert model.body_mass[torso] > 1e-6
        env.reset(seed=42)
        assert model.body_mass[torso] == initial
        assert model.body_subtreemass[torso] == pytest.approx(subtree)
        for expected, state, reward in trace[:201]:
            obs, repeated_reward, _, _, _ = env.step(np.zeros(8))
            assert model.body_mass[torso] == pytest.approx(expected)
            np.testing.assert_allclose(obs['state'], state, rtol=0, atol=1e-12)
            assert repeated_reward == pytest.approx(reward, abs=1e-12)
    finally:
        env.close()


def test_ant_environment_override_is_task_specific(tmp_path):
    with pytest.raises(ValueError, match='Unsupported evaluation environment'):
        evaluate_public_checkpoint('unused.pt', 'cartpole', tmp_path,
                                   environment_id='ExampleNSAnt-v1')
    assert not (tmp_path / 'public_episodes.jsonl').exists()


def test_corrected_ant_checkpoint_evaluation(tmp_path):
    checkpoint = Path(__file__).resolve().parents[1] / 'models/ppo_ant/ppo_final.pt'
    if not checkpoint.exists():
        pytest.skip('historical Ant checkpoint unavailable')
    evaluate_public_checkpoint(checkpoint, 'ant', tmp_path, episodes_per_level=1,
                               environment_id='ExampleNSAnt-v1')
    summary = json.loads((tmp_path / 'public_summary.json').read_text())
    assert summary['environment'] == 'ExampleNSAnt-v1'
    rows = [json.loads(line) for line in (tmp_path / 'public_episodes.jsonl').read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]['trajectory']['actions'] == rows[1]['trajectory']['actions'] == rows[2]['trajectory']['actions']
    assert rows[0]['return'] == rows[1]['return'] == rows[2]['return']
    assert sum(rows[0]['trajectory']['env_change']) > 0
    single = tmp_path / 'none_only'
    single.mkdir()
    evaluate_public_checkpoint(checkpoint, 'ant', single, episodes_per_level=1,
                               environment_id='ExampleNSAnt-v1', notification_modes=('none',))
    single_summary = json.loads((single / 'public_summary.json').read_text())
    single_rows = [json.loads(line) for line in (single / 'public_episodes.jsonl').read_text().splitlines()]
    assert single_summary['notification_modes'] == ['none']
    assert len(single_rows) == 1
    assert single_rows[0]['trajectory']['actions'] == rows[2]['trajectory']['actions']
    assert single_rows[0]['return'] == rows[2]['return']
    assert 'public/full/mean' not in single_summary['metrics']
