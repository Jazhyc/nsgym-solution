"""Reproducible stochastic submission evaluation and uniform replay control."""
import json
import numpy as np
from AAMAS_Comp.curriculum.plr import PLRBuffer, _LevelEntry
from AAMAS_Comp.envs.ns_env_sampler import NS_ENV_SAMPLERS
from AAMAS_Comp.evaluation.public import evaluate_public_checkpoint


def test_uniform_replay_ignores_scores_and_staleness():
    sampler = NS_ENV_SAMPLERS['frozenlake'](seed=0)
    buffer = PLRBuffer(sampler, uniform_replay=True)
    buffer._entries = [_LevelEntry(config=sampler.sample(), n_episodes=1) for _ in range(3)]
    for i, entry in enumerate(buffer._entries):
        entry.score = 100. ** i
        entry.staleness = 20 * i
    np.testing.assert_allclose(buffer._compute_distribution(), [1/3] * 3)
    buffer.uniform_replay = False
    assert not np.allclose(buffer._compute_distribution(), [1/3] * 3)


def test_public_submission_reproducible_and_notification_agnostic(tmp_path):
    from pathlib import Path
    checkpoint = Path(__file__).resolve().parents[1] / 'models/ppo_frozenlake/ppo_final_no_notify.pt'
    if not checkpoint.exists():
        import pytest
        pytest.skip('historical checkpoint unavailable')
    results = []
    for i in range(2):
        folder = tmp_path / str(i); folder.mkdir()
        evaluate_public_checkpoint(checkpoint, 'frozenlake', folder, episodes_per_level=3)
        rows = [json.loads(line) for line in (folder / 'public_episodes.jsonl').read_text().splitlines()]
        results.append([(r['return'], r['steps'], r['trajectory']['actions']) for r in rows])
        assert results[-1][:3] == results[-1][3:6] == results[-1][6:9]
        assert all(r['steps'] <= 100 for r in rows)
    assert results[0] == results[1]
