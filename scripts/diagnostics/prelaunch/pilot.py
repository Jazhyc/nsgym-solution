"""Execute and verify one short pilot from the proposed asynchronous protocol."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import wandb
root=Path(sys.argv[1]).resolve();index=int(sys.argv[2])
sys.path.insert(0,str(root/'release/scripts'))
import paper_campaign as campaign
manifest=root/'pilot_manifest.json';definition=json.loads(manifest.read_text())
task,method,seed=definition['matrix'][index]
args=SimpleNamespace(stage='pilot',index=index,root=root,manifest=manifest,project='nsgym-paper',
 group=root.name,wandb_mode='online',protect_unscored='true',dry_run=False)
campaign.run(args)
p=root/'pilot'/task/method/f'seed{seed}'
meta=json.loads((p/'run.json').read_text());cfg=meta['config']
assert cfg['collector']['synchronous'] is False and cfg['collector']['num_envs']==24 and cfg['collector']['num_groups']==2
static=method in campaign.STATIC_METHODS
budget=({'cartpole':256000,'frozenlake':51200,'ant':1024000} if static else
        {'cartpole':128000,'frozenlake':51200,'ant':512000})[task]
assert budget<=meta['actual_frames']<budget+3000
metrics=json.loads((p/'training_metrics_summary.json').read_text())['metrics']
if method in ('plr','full',*campaign.STATIC_METHODS):
 assert metrics['plr/feedback_sent_total']>0 and metrics['plr/applied_scores_observed_total']>0
 assert metrics['plr/unscored_evictions_observed_total']==0
 assert metrics['plr/unscored_priority_mass']==0 and metrics['plr/worker_coverage']==1
 assert metrics['plr/replay_count_observed_total']>=20
 assert metrics['accel/mutation_count_observed_total'] >= (10 if method in ('full','static_plr_mutations') else 0)
 if method in ('plr','static_plr'):assert metrics['accel/mutation_count_observed_total']==0
 # Async can discard whole batches. Rejections are expected integrity events,
 # not scores that were erroneously applied. Never require zero rejections.
 assert metrics['plr/incomplete_episode_scores_total']>=0
 assert metrics['plr/abandoned_episode_scores_total']>=0
 if task=='frozenlake':assert cfg['env']['initial_mode']=='uniform'
if static:
 import hashlib
 protocol=json.loads((p/'static_protocol.json').read_text())
 assert cfg['experiment']['training_dynamics']=='static'
 samples=definition['training']['tasks'][task]['dr_samples']
 assert protocol['snapshot_sha256']==hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest()
public=json.loads((p/'public_summary.json').read_text())
broad=json.loads((p/'test_summary.json').read_text())
assert public['action_mode']==broad['action_mode']=='stochastic'
assert json.loads((p/'test_trajectory_metadata.json').read_text())['action_mode']=='stochastic'
assert public['horizon']=={'cartpole':500,'frozenlake':100,'ant':1000}[task]
assert broad['horizon']=={'cartpole':500,'frozenlake':200,'ant':1000}[task]
info=meta['wandb_run'];run=wandb.Api(timeout=60).run(f"{info['entity']}/{info['project']}/{info['id']}")
assert run.state=='finished'
assert any(a.type=='paper-run' for a in run.logged_artifacts())
checkpoint=torch.load(p/'ppo_final.pt',map_location='cpu',weights_only=False)
assert all(torch.isfinite(t).all() for t in checkpoint['actor'].state_dict().values())
(p/'pilot_verified.json').write_text(json.dumps({'task':task,'method':method,'frames':meta['actual_frames'],
 'protocol':campaign.PROTOCOL,'wandb':info,
 'incomplete_scores_rejected':metrics.get('plr/incomplete_episode_scores_total',0),
 'abandoned_scores_rejected':metrics.get('plr/abandoned_episode_scores_total',0),
 'replays':metrics.get('plr/replay_count_observed_total',0),
 'mutations':metrics.get('accel/mutation_count_observed_total',0)},indent=2))
print('PASS',task,method,'approved async protocol, feedback, evaluation and W&B')
