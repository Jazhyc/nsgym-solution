"""Resolve every proposed full run without executing any training command."""
import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import yaml

root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'release/scripts'))
import paper_campaign as campaign
manifest=root/'manifest.json';definition=json.loads(manifest.read_text())
reference_path=root/'release/docs/pre_dr_configuration_reference.json'
reference=json.loads(reference_path.read_text())
assert len(definition['matrix'])==75
assert hashlib.sha256(json.dumps(definition['tasks'],sort_keys=True).encode()).hexdigest()==definition['evaluation_cases_sha256']
source=json.loads(Path(definition['evaluation_source']['path']).read_text())
assert definition['tasks']==source['tasks'], 'Numerical test cases must not change'
planned=[]
with (root/'plan.log').open('w') as log,contextlib.redirect_stdout(log):
 for index,(task,method,seed) in enumerate(definition['matrix']):
  args=SimpleNamespace(stage='final',index=index,root=root,manifest=manifest,
      project='nsgym-paper',group=root.name,wandb_mode='online',protect_unscored='true',dry_run=True)
  campaign.run(args)
  folder=root/'final'/task/method/f'seed{seed}'
  cfg=yaml.safe_load((folder/'resolved_config.yaml').read_text());old=reference['tasks'][task]
  for key,value in old['agent'].items():assert cfg['agent'][key]==value,(task,'agent',key)
  for key in ('total_frames','frames_per_batch','max_frames_per_traj','num_envs','num_groups'):
   assert cfg['collector'][key]==old['collector'][key],(task,'collector',key)
  for key in ('num_epochs','sub_batch_size','max_grad_norm','target_kl','eval_ratio','eval_rollout_steps','num_eval_episodes'):
   assert cfg['training'][key]==old['training'][key],(task,'training',key)
  for key in ('normalize_obs','normalize_reward','frame_stack','frame_skip'):
   assert cfg['env'][key]==old['env'][key],(task,'env',key)
  for key in ('capacity','replay_prob','score_temp','staleness_coef','mutation_sigma','num_eval_configs'):
   assert cfg['env']['plr'][key]==old['env']['plr'][key],(task,'plr',key)
  assert cfg['collector']['synchronous'] is False
  if task=='frozenlake':assert cfg['env']['initial_mode']=='uniform'
  planned.append({'index':index,'task':task,'method':method,'seed':seed,'resolved_config':str(folder/'resolved_config.yaml')})
summary={'protocol':campaign.PROTOCOL,'source_sha256':campaign.provenance()['source_sha256'],
 'reference_commit':reference['commit'],'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
 'full_training_submitted':False,'n_runs':len(planned),'runs':planned}
(root/'launch_plan.json').write_text(json.dumps(summary,indent=2))
print('PASS: all 75 full configurations resolved, pre-DR PPO values match, no training executed')
