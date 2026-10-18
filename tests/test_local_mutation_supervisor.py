"""Guard local memory admission and never terminate incomplete computation."""
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/diagnostics/mutation_audit/local_supervisor.py'
spec = importlib.util.spec_from_file_location('local_supervisor_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_second_run_requires_memory_and_cpu_headroom():
    assert not module.admit_second(9., 8., 3.)[0]
    assert not module.admit_second(18., 8., 12.)[0]
    assert module.admit_second(18., 8., 6.)[0]


def test_upload_timeout_gate_requires_all_episodes_and_valid_physics(tmp_path):
    for name in ('ppo_final.pt', 'test_trajectories.npz', 'training_metrics_summary.json',
                 'test_trajectory_metadata.json'):
        (tmp_path/name).touch()
    (tmp_path/'run.json').write_text(json.dumps({'actual_frames': 10002360}))
    metrics = {f'test/{s}/physics_invalid_steps': 0 for s in ('iid', 'abrupt', 'gradual', 'no_change')}
    (tmp_path/'test_summary.json').write_text(json.dumps({'action_mode': 'stochastic', 'metrics': metrics}))
    (tmp_path/'public_summary.json').write_text(json.dumps({'action_mode': 'stochastic'}))
    (tmp_path/'test_episodes.jsonl').write_text('{}\n'*4000)
    (tmp_path/'public_episodes.jsonl').write_text('{}\n'*299)
    assert not module.outputs_complete(tmp_path, 'cartpole')
    (tmp_path/'public_episodes.jsonl').write_text('{}\n'*300)
    assert module.outputs_complete(tmp_path, 'cartpole')
    metrics['test/iid/physics_invalid_steps'] = 1
    (tmp_path/'test_summary.json').write_text(json.dumps({'action_mode': 'stochastic', 'metrics': metrics}))
    assert not module.outputs_complete(tmp_path, 'cartpole')
