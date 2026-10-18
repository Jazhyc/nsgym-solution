"""Static curricula use DR's joint prior, fixed physics, and real PLR feedback."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from AAMAS_Comp.curriculum.plr_env import PLREnv
from AAMAS_Comp.envs.domain_randomization import DynamicsEnv, parameters
from AAMAS_Comp.envs.static_curriculum import StaticDynamicsSampler, build_static_level
from AAMAS_Comp.envs.torchrl_factory import make_ns_plr_env

REPO = Path(__file__).resolve().parents[1]
SAMPLES = {
    'cartpole': [{'masspole': .1, 'masscart': 1., 'length': .5, 'gravity': 9.8},
                 {'masspole': .3, 'masscart': 1.5, 'length': .8, 'gravity': 5.}],
    'ant': [{'torso_mass': 3., 'gravity': -9.81}, {'torso_mass': 4., 'gravity': -6.}],
    'frozenlake': [{'P': [.8, .1, .1]}, {'P': [.1, .45, .45]}, {'P': [1/3]*3}],
}


@pytest.mark.parametrize('task', SAMPLES)
def test_fresh_static_levels_draw_exact_dr_joint_prior(task):
    sampler = StaticDynamicsSampler(task, SAMPLES[task], 8)
    dr = DynamicsEnv(task, {}, seed=8, samples=SAMPLES[task])
    try:
        for _ in range(100):
            assert sampler.sample().values == dr.sample_values()
    finally:
        dr.close()


@pytest.mark.parametrize('task', SAMPLES)
def test_static_mutations_are_seeded_valid_bounded_and_do_not_change_parent(task):
    sampler = StaticDynamicsSampler(task, SAMPLES[task], 8)
    parent = sampler.sample()
    original = deepcopy(parent.values)
    assert sampler.mutate(parent, np.random.default_rng(1), 0).values == original
    child = sampler.mutate(parent, np.random.default_rng(1))
    assert child == sampler.mutate(parent, np.random.default_rng(1))
    assert parent.values == original
    assert any(sampler.mutate(parent, np.random.default_rng(i)).values != original for i in range(20))
    for i in range(100):
        child = sampler.mutate(child, np.random.default_rng(i), .8)
        for key, (lo, hi) in sampler.bounds.items():
            values = np.asarray(child.values[key])
            assert np.isfinite(values).all() and (values >= lo-1e-12).all() and (values <= hi+1e-12).all()
        if task == 'frozenlake':
            assert np.isclose(sum(child.values['P']), 1) and min(child.values['P']) >= 0


@pytest.mark.parametrize('task', SAMPLES)
def test_static_plr_replay_and_mutation_levels_keep_physics_fixed(task):
    queue = Queue()
    env = PLREnv(task, plr_capacity=10, replay_prob=.5, mutation_prob=1,
                 seed=8, feedback_queue=queue,
                 sampler_override=StaticDynamicsSampler(task, SAMPLES[task], 8))
    try:
        for episode in range(30):
            if env._level_id is not None:
                queue.put((env._level_tokens[env._level_id], episode, 1.))
            _, info = env.reset(seed=episode)
            level = env.plr._entries[env._level_id].config
            before = parameters(env._env, task)
            assert before == level.values
            assert info['plr_level'] == env._level_tokens[env._level_id]
            for _ in range(10):
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                assert parameters(env._env, task) == before
                if terminated or truncated:
                    break
        assert env._replay_count > 0 and env._mutation_count > 0 and env._applied_scores > 0
    finally:
        env.close()


@pytest.mark.parametrize('task', SAMPLES)
def test_actual_torchrl_factory_selects_static_levels_and_preserves_plr_ids(task, tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'training': {'frozenlake_initial_mode': 'uniform',
        'tasks': {task: {'dr_samples': SAMPLES[task], 'dr_ranges': {}}}}}))
    with initialize_config_dir(config_dir=str(REPO/'config'), version_base=None):
        cfg = compose(config_name=f'config_{task}', overrides=[
            '+experiment.method=static_plr', f'+experiment.manifest={manifest}'])
    env = make_ns_plr_env(cfg, torch.device('cpu'), worker_idx=12, feedback_queue=Queue())
    try:
        td = env.reset()
        assert td['plr_worker'].item() == 12
        current = env.base_env._env
        while not isinstance(current, PLREnv):
            current = current.env
        assert isinstance(current.plr.sampler, StaticDynamicsSampler)
        before = parameters(current._env, task)
        for _ in range(3):
            td = env.rand_step(td)['next']
            assert parameters(current._env, task) == before
            if td['done'].any():
                break
    finally:
        env.close()


def test_static_method_resolves_full_budget_and_mutation_settings(tmp_path):
    spec = importlib.util.spec_from_file_location('static_campaign', REPO/'scripts/paper_campaign.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    assert len(campaign.matrix('final', campaign.STATIC_METHODS)) == 30
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'protocol': campaign.PROTOCOL, 'provenance': campaign.provenance(),
        'matrix': campaign.matrix('final', campaign.STATIC_METHODS),
        'training': {'frozenlake_initial_mode': 'uniform', 'tasks': {
            t: {'dr_samples': s} for t, s in SAMPLES.items()}}}))
    args = SimpleNamespace(stage='final', index=1, root=tmp_path/'plan', manifest=manifest,
        project='nsgym-paper', group='test', wandb_mode='disabled', protect_unscored='true', dry_run=True)
    campaign.run(args)
    import yaml
    folder = args.root/'final/cartpole/static_plr_mutations/seed0'
    cfg = yaml.safe_load((folder/'resolved_config.yaml').read_text())
    assert cfg['env']['plr']['enabled'] and cfg['env']['plr']['mutation_prob'] == .25
    assert cfg['collector']['total_frames'] == 10000000 and not cfg['collector']['synchronous']
    assert cfg['experiment']['training_dynamics'] == 'static'
    protocol = json.loads((folder/'static_protocol.json').read_text())
    assert protocol['snapshot_count'] == len(SAMPLES['cartpole'])


def test_static_sampler_rejects_invalid_physics():
    with pytest.raises(ValueError):
        StaticDynamicsSampler('frozenlake', [{'P': [.5, .5, .5]}])
    with pytest.raises(ValueError):
        StaticDynamicsSampler('ant', [{'gravity': -9.81, 'torso_mass': -1}])
