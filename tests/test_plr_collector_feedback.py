"""Verify PLR identity and scoring through actual TorchRL collection and GAE."""

from pathlib import Path
from queue import Queue

from hydra import compose, initialize_config_dir
import numpy as np
import pytest
import torch
from torchrl.collectors import Collector

from AAMAS_Comp.agents.ppo import make_ppo_models
from AAMAS_Comp.curriculum.feedback import EpisodeScores, PLR_KEYS
from AAMAS_Comp.curriculum.plr_env import PLREnv
from AAMAS_Comp.envs.torchrl_factory import make_ns_plr_env
from AAMAS_Comp.evaluation.training_metrics import TrainingEpisodes


@pytest.mark.parametrize("protect", [False, True])
def test_collector_keeps_episode_identity_and_delivers_gae_scores(protect):
    torch.set_num_threads(1)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"),
                               version_base=None):
        cfg = compose(config_name="config_frozenlake", overrides=[
            "env.plr.capacity=3", "collector.max_frames_per_traj=8",
            f"+env.plr.protect_unscored={str(protect).lower()}"])
    feedback = Queue()
    env = make_ns_plr_env(cfg, torch.device("cpu"), feedback_queue=feedback)
    current = env.base_env._env
    while not isinstance(current, PLREnv):
        current = current.env
    plr_env = current
    torch.manual_seed(7)
    models = make_ppo_models(env, cfg, device=torch.device("cpu"),
                             network_device=torch.device("cpu"))
    scores = EpisodeScores()
    episodes_logger = TrainingEpisodes(frozenlake=True)
    completed_rewards = completed_steps = raw_rewards = 0
    collected = []
    collector = Collector(env, models["actor"], frames_per_batch=64, total_frames=512)
    try:
        for batch in collector:
            raw_rewards += batch["next", "reward"].sum().item()
            metrics = episodes_logger.update(batch)
            count = metrics["train/episodes_completed_batch"]
            if count:
                completed_rewards += count * metrics["train/episode_return_mean"]
                completed_steps += count * metrics["train/episode_length_mean"]
                assert metrics["train/success_rate"] + metrics["train/hole_rate"] + metrics["train/timeout_rate"] == pytest.approx(1)
            for key in PLR_KEYS:
                # Each transition must keep the identity of the level on
                # which its action was taken, including terminal transitions.
                assert torch.equal(batch[key], batch["next", key])
            assert (batch["plr_level"] > 0).all()
            assert (batch["plr_worker"] == 0).all()
            with torch.no_grad():
                models["advantage"](batch)
            assert torch.isfinite(batch["advantage"]).all()
            collected.extend(zip(batch["plr_episode"].tolist(),
                                 batch["next", "done"].reshape(-1).tolist()))
            scores.update(batch, [feedback])
        assert plr_env._applied_scores > 0
        assert completed_steps + sum(steps for _, steps in episodes_logger.pending.values()) == 512
        assert completed_rewards + sum(reward for reward, _ in episodes_logger.pending.values()) == pytest.approx(raw_rewards)
        episodes = np.asarray([row[0] for row in collected])
        assert (np.diff(episodes) >= 0).all()
        for episode in np.unique(episodes)[:-1]:
            rows = [row for row in collected if row[0] == episode]
            assert rows[-1][1] is True
            assert not any(row[1] for row in rows[:-1])
        assert plr_env._forced_replay_count > 0 if protect else plr_env._forced_replay_count == 0
    finally:
        collector.shutdown()
