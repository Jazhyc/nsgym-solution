"""Fixed-start training, separate DR calibration, and async metric integrity."""
from dataclasses import asdict
from pathlib import Path
import importlib.util
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

from AAMAS_Comp.envs.ns_env_sampler import make_sampler, NS_ENV_SAMPLERS
from AAMAS_Comp.curriculum.plr_env import PLREnv, RandomNSEnv, sample_held_out_configs
from AAMAS_Comp.envs.domain_randomization import parameters
from AAMAS_Comp.evaluation.paper import calibrate_frozenlake_training
from AAMAS_Comp.envs.torchrl_factory import make_static_dr_env
from AAMAS_Comp.evaluation.training_metrics import TrainingEpisodes

REPO=Path(__file__).resolve().parents[1]


def test_fixed_initial_mode_preserves_schedules_and_mutation_draws():
    fixed=make_sampler('frozenlake',8,'uniform')
    random=make_sampler('frozenlake',8,'random')
    legacy=NS_ENV_SAMPLERS['frozenlake'](seed=8)
    for i in range(30):
        a,b=fixed.sample(),random.sample()
        assert asdict(b)==asdict(legacy.sample())
        assert a.tunable_params==b.tunable_params
        assert a.wrapper_kwargs['initial_prob_dist']==[1/3]*3
        assert b.wrapper_kwargs['initial_prob_dist']!=[1/3]*3
        x=fixed.mutate(a,np.random.default_rng(i))
        y=random.mutate(b,np.random.default_rng(i))
        assert x.tunable_params==y.tunable_params
        assert x.wrapper_kwargs['initial_prob_dist']==[1/3]*3
        assert y.wrapper_kwargs['initial_prob_dist']!=[1/3]*3


@pytest.mark.parametrize('kind',[PLREnv,RandomNSEnv])
def test_real_training_environments_reset_to_fixed_uniform(kind):
    env=kind(sampler_key='frozenlake',seed=4,frozenlake_initial_mode='uniform')
    try:
        for seed in range(10):
            env.reset(seed=seed)
            np.testing.assert_allclose(parameters(env._env,'frozenlake')['P'],[1/3]*3)
    finally:env.close()


def test_validation_initial_condition_is_explicit_and_catalog_stays_random():
    fixed=sample_held_out_configs('frozenlake',8,72301,'uniform')
    random=sample_held_out_configs('frozenlake',8,72301,'random')
    assert all(c.wrapper_kwargs['initial_prob_dist']==[1/3]*3 for c in fixed)
    assert all(a.tunable_params==b.tunable_params for a,b in zip(fixed,random))
    assert any(c.wrapper_kwargs['initial_prob_dist']!=[1/3]*3 for c in random)


def test_frozenlake_training_calibration_is_seeded_nonconstant_and_valid():
    a=calibrate_frozenlake_training(count=6,seed=71301)
    assert a==calibrate_frozenlake_training(count=6,seed=71301)
    assert a!=calibrate_frozenlake_training(count=6,seed=71302)
    p=np.array([r['P'] for r in a['dr_samples']])
    assert len(p)==6*16 and np.isfinite(p).all() and (p>=0).all()
    np.testing.assert_allclose(p.sum(1),1)
    assert np.max(np.ptp(p,axis=0))>.1


def test_static_dr_uses_training_snapshots_not_test_distribution(tmp_path):
    training=[{'P':[.9,.05,.05]},{'P':[.2,.3,.5]}]
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps({'training':{'frozenlake_initial_mode':'uniform','tasks':{
        'frozenlake':{'dr_ranges':{},'dr_samples':training}}},'tasks':{
        'frozenlake':{'dr_ranges':{},'dr_samples':[{'P':[.7,.1,.2]}]}}}))
    with initialize_config_dir(config_dir=str(REPO/'config'),version_base=None):
        cfg=compose(config_name='config_frozenlake',overrides=[f'+experiment.manifest={path}'])
    env=make_static_dr_env(cfg,torch.device('cpu'),worker_idx=12)
    seen=set()
    try:
        for _ in range(15):
            td=env.reset();p=parameters(env.base_env._env,'frozenlake')['P'];seen.add(tuple(p))
            assert p in [r['P'] for r in training]
            assert td['env_worker'].item()==12
            for _ in range(3):
                td=env.rand_step(td)['next']
                np.testing.assert_allclose(parameters(env.base_env._env,'frozenlake')['P'],p)
        assert len(seen)==2
    finally:env.close()


def metric_batch(worker,trajectory,steps,rewards,done):
    n=len(steps)
    return TensorDict({'env_worker':torch.full((n,),worker),
       'collector':{'traj_ids':torch.full((n,),trajectory)},'step_count':torch.tensor(steps).reshape(n,1),
       'next':{'reward':torch.tensor(rewards).reshape(n,1),'done':torch.tensor(done).reshape(n,1),
               'terminated':torch.tensor(done).reshape(n,1)}},[n])


def test_async_metric_logger_rejects_lost_batches_and_recovers():
    log=TrainingEpisodes(frozenlake=True)
    log.update(metric_batch(0,0,[0],[0.],[False]))
    result=log.update(metric_batch(0,0,[2,3],[0.,1.],[False,True]))
    assert result['train/episodes_completed_total']==0
    assert result['train/incomplete_episode_metrics_total']==1
    log.update(metric_batch(0,1,[0],[0.],[False]))
    result=log.update(metric_batch(0,2,[0,1],[0.,1.],[False,True]))
    assert result['train/abandoned_episode_metrics_total']==1
    assert result['train/episodes_completed_total']==1 and result['train/success_rate']==1
    assert not log.pending and not log.pending_steps


def test_async_metric_groups_do_not_share_trajectory_namespaces():
    log=TrainingEpisodes(frozenlake=True)
    log.update(metric_batch(0,0,[0],[0.],[False]))
    log.update(metric_batch(12,0,[0],[0.],[False]))
    assert len(log.pending)==2
    a=log.update(metric_batch(0,0,[1],[1.],[True]))
    b=log.update(metric_batch(12,0,[1],[0.],[True]))
    assert a['train/success_rate']==1 and b['train/success_rate']==0
    assert b['train/episodes_completed_total']==2 and not log.pending


def campaign_module():
    spec=importlib.util.spec_from_file_location('campaign_protocol',REPO/'scripts/paper_campaign.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_main_matrix_and_resolved_config_are_reviewable_without_training(tmp_path,monkeypatch):
    campaign=campaign_module()
    assert len(campaign.matrix('final'))==75
    assert len(campaign.matrix('final',campaign.METHODS))==105
    manifest=tmp_path/'manifest.json'
    definition={'protocol':campaign.PROTOCOL,'provenance':campaign.provenance(),
                'matrix':campaign.matrix('final'),'training':{'frozenlake_initial_mode':'uniform'}}
    manifest.write_text(json.dumps(definition))
    # Index 9 is FrozenLake/full in the five-method seed-first matrix.
    args=SimpleNamespace(stage='final',index=9,root=tmp_path/'plan',manifest=manifest,
            project='nsgym-paper',group='preflight',wandb_mode='disabled',protect_unscored='true',dry_run=True)
    campaign.run(args)
    import yaml
    cfg=yaml.safe_load((tmp_path/'plan/final/frozenlake/full/seed0/resolved_config.yaml').read_text())
    assert cfg['collector']['synchronous'] is False
    assert cfg['collector']['num_envs']==24 and cfg['collector']['num_groups']==2
    assert cfg['env']['initial_mode']=='uniform' and cfg['env']['plr']['score_ema_alpha']==1
    assert cfg['collector']['total_frames']==4000000


def test_launch_script_defaults_to_review_and_refuses_unready_launch(tmp_path):
    import shutil,subprocess,sys
    release=tmp_path/'release/scripts';release.mkdir(parents=True)
    shutil.copy(REPO/'scripts/paper_campaign.py',release/'paper_campaign.py')
    (tmp_path/'manifest.json').write_text(json.dumps({'matrix':[['cartpole','stationary',0]]}))
    script=REPO/'scripts/diagnostics/prelaunch/launch_full.py'
    review=subprocess.run([sys.executable,str(script),str(tmp_path)],capture_output=True,text=True)
    assert review.returncode==0 and '"full_training_submitted": false' in review.stdout
    blocked=subprocess.run([sys.executable,str(script),str(tmp_path),'--launch'],capture_output=True,text=True)
    assert blocked.returncode!=0 and 'Prelaunch validation is not complete' in blocked.stderr
    assert not (tmp_path/'full_jobs.json').exists()
