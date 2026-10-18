import json

import numpy as np
import pytest
import torch

from AAMAS_Comp.envs.domain_randomization import DynamicsEnv
from AAMAS_Comp.evaluation.paper import episode, evaluate_checkpoint
from AAMAS_Comp.evaluation.trajectories import EvaluationTrajectories


class ConstantAction(torch.nn.Module):
    def __init__(self, task="frozenlake"):
        super().__init__()
        self.task = task

    def forward(self, td):
        td["action"] = torch.zeros(8) if self.task == "ant" else torch.tensor(0)
        return td


@pytest.mark.parametrize("task,initial,target", [
    ("frozenlake", {"P": [1., 0., 0.]}, {"P": [.8, .1, .1]}),
    ("cartpole", {"masspole": .1, "masscart": 1., "length": .5, "gravity": 9.8},
     {"masspole": .2, "masscart": 1., "length": .7, "gravity": 12.}),
    ("ant", {"torso_mass": .327, "gravity": -9.81}, {"torso_mass": .5, "gravity": -8.}),
])
def test_recording_is_noninvasive_and_retains_actual_change(task, initial, target):
    env = DynamicsEnv(task, {}, intervention={"initial": initial, "target": target,
                                            "step": 2, "duration": 0})
    try:
        actor = ConstantAction(task)
        without = episode(actor, {}, env, task, seed=3, intervention=env.intervention)
        trace = {}
        recorded = episode(actor, {}, env, task, seed=3, intervention=env.intervention, trajectory=trace)
        assert without == recorded
        steps = recorded["steps"]
        assert len(trace["observations"]) == steps + 1
        assert len(trace["parameter_values"]) == steps + 1
        assert len(trace["actions"]) == steps
        assert sum(trace["rewards"].tolist()) == recorded["return"]
        assert sum(trace["parameter_changed"]) == recorded["parameter_changes"]
        assert trace["parameter_changed"][2]
        column = trace["parameter_names"].index("P[0]" if task == "frozenlake" else
                                               "gravity[2]" if task == "ant" else "gravity")
        assert trace["parameter_values"][2][column] == pytest.approx(initial["P"][0] if task == "frozenlake" else initial["gravity"])
        assert trace["parameter_values"][3][column] == pytest.approx(target["P"][0] if task == "frozenlake" else target["gravity"])
    finally:
        env.close()


def test_archive_is_lossless_standard_npz_and_failed_write_is_not_published(tmp_path):
    path = tmp_path / "trajectories.npz"
    values = np.asarray([1 / 3, np.nextafter(1., 2.)], dtype=np.float64)
    record = {"suite": "iid", "config_id": 0, "rollout_seed": 3}
    with EvaluationTrajectories(path) as writer:
        writer.write(record, {"parameter_values": values})
    with np.load(path, allow_pickle=False) as archive:
        key = record["trajectory_key"]
        assert np.array_equal(archive[f"{key}/parameter_values"], values)
        assert json.loads(archive[f"{key}/metadata"].item()) == record
    bad = tmp_path / "failed.npz"
    with pytest.raises(ValueError):
        with EvaluationTrajectories(bad) as writer:
            writer.write({}, {"objects": np.asarray([{}], dtype=object)})
    assert not bad.exists()
    assert not bad.with_suffix(".npz.partial").exists()


@pytest.mark.parametrize('action_mode', ['greedy', 'stochastic'])
def test_checkpoint_evaluation_artifact_reconstructs_all_episode_results(tmp_path, action_mode):
    config = {"env_id": "FrozenLake-v1", "tunable_params": {"P": {
        "scheduler": {"cls": "ContinuousScheduler", "kwargs": {}},
        "update_fn": {"cls": "UniformDrift", "kwargs": {"rate": .04}}}},
        "gym_kwargs": {}, "wrapper_kwargs": {"initial_prob_dist": [1., 0., 0.]}}
    suites = {"iid": [{"id": 0, "config": config, "seeds": [3, 4]}]}
    for name, duration in (("abrupt", 0), ("gradual", 5), ("no_change", -1)):
        suites[name] = [{"id": 0, "seeds": [3, 4], "intervention": {
            "initial": {"P": [1/3]*3}, "target": {"P": [.9, .05, .05]},
            "step": 2, "duration": duration}}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"action_mode": action_mode,
                                   "tasks": {"frozenlake": {"dr_ranges": {}, "suites": suites}}}))
    checkpoint = tmp_path / "policy.pt"
    torch.save({"actor": ConstantAction()}, checkpoint)
    output = tmp_path / "evaluation"
    metrics = evaluate_checkpoint(checkpoint, manifest, "frozenlake", output)
    schema = json.loads((output / 'test_trajectory_metadata.json').read_text())
    summary = json.loads((output / 'test_summary.json').read_text())
    assert schema['action_mode'] == summary['action_mode'] == action_mode
    records = [json.loads(row) for row in (output / "test_episodes.jsonl").read_text().splitlines()]
    assert len(records) == 8
    with np.load(output / "test_trajectories.npz", allow_pickle=False) as archive:
        for record in records:
            key = record["trajectory_key"]
            rewards = archive[f"{key}/rewards"]
            assert rewards.sum() == record["return"]
            assert len(rewards) == record["steps"]
            assert archive[f"{key}/parameter_changed"].sum() == record["parameter_changes"]
            assert archive[f"{key}/terminated"][-1] == record["terminated"]
            assert archive[f"{key}/truncated"][-1] == record["truncated"]
        for suite in suites:
            rows = [r for r in records if r["suite"] == suite]
            assert metrics[f"test/{suite}/mean"] == np.mean([r["return"] for r in rows])
            assert metrics[f"test/{suite}/hole_rate"] + metrics[f"test/{suite}/timeout_rate"] + metrics[f"test/{suite}/success_rate"] == pytest.approx(1)
