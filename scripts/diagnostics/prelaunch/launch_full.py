"""Review the prepared full launch; submission requires explicit --launch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path)
parser.add_argument('--launch',action='store_true',help='Submit full training only after user authorization')
args=parser.parse_args();root=args.root.resolve()
sys.path.insert(0,str(root/'release/scripts'));import paper_campaign as campaign
manifest=json.loads((root/'manifest.json').read_text())
ready_path=root/'readiness.json';ready=json.loads(ready_path.read_text()) if ready_path.exists() else {}
review={'protocol':campaign.PROTOCOL,'main_runs':len(manifest['matrix']),
 'ready':bool(ready.get('ready')),'collector':'two asynchronous groups of 12',
 'seeds':[0,1,2,3,4],'full_training_submitted':False,
 'resources':{'cartpole':'32 CPUs / 32GB / 8h','frozenlake':'32 CPUs / 32GB / 8h','ant':'32 CPUs / 96GB / 5 days'}}
if not args.launch:
 print(json.dumps(review,indent=2));print('Review only: pass --launch only after explicit authorization.');sys.exit(0)
if not ready.get('ready'):raise RuntimeError('Prelaunch validation is not complete')
assert ready['source_sha256']==campaign.provenance()['source_sha256']
assert ready['manifest_sha256']==hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
path=root/'full_jobs.json'
if path.exists():raise FileExistsError('Full launch already attempted; inspect the recorded jobs before resuming')
python='/scratch/s4325621/venvs/nsgym-paper/bin/python'
prefix=f'export PYTHONPATH={root}/release/src:{root}/release; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl; export WANDB_CACHE_DIR=/scratch/s4325621/wandb_cache WANDB_DATA_DIR={root}/wandb_data WANDB_INIT_TIMEOUT=120; cd {root}/release; '
previous={};jobs={'waves':[],'source_sha256':ready['source_sha256']}
path.write_text(json.dumps(jobs,indent=2))
for seed in range(5):
 for task in ('cartpole','frozenlake','ant'):
  indices=[i for i,(t,m,s) in enumerate(manifest['matrix']) if t==task and s==seed]
  ant=task=='ant'
  command=f'{python} scripts/paper_campaign.py run --root {root} --manifest {root}/manifest.json --stage final --group {root.name} --index $SLURM_ARRAY_TASK_ID'
  sbatch=['sbatch','--parsable',f'--job-name=v2_{task}_seed{seed}',
   '--partition='+('regularlong' if ant else 'regularmedium'),'--time='+('5-00:00:00' if ant else '08:00:00'),
   '--cpus-per-task=32','--mem='+('96G' if ant else '32G'),
   '--array='+','.join(map(str,indices))+('%2' if ant else '%4'),
   f'--output={root}/slurm/%x-%A_%a.out',f'--error={root}/slurm/%x-%A_%a.err']
  if task in previous:sbatch+=['--dependency=afterok:'+previous[task],'--kill-on-invalid-dep=yes']
  job=subprocess.check_output(sbatch+['--wrap',prefix+command],text=True).strip().split(';')[0]
  previous[task]=job;jobs['waves'].append({'task':task,'seed':seed,'job':job,'indices':indices})
  path.write_text(json.dumps(jobs,indent=2))
command=f'{python} scripts/paper_campaign.py aggregate --root {root}/final --manifest {root}/manifest.json --output {root}/comparison.json --wandb --group {root.name}-final --artifact-name plr-v2-async-fixed-comparison'
jobs['aggregate']=subprocess.check_output(['sbatch','--parsable','--job-name=v2_aggregate',
 '--partition=regularshort','--time=04:00:00','--cpus-per-task=4','--mem=8G',
 '--dependency=afterok:'+':'.join(previous.values()),'--kill-on-invalid-dep=yes',
 f'--output={root}/slurm/%x-%A_%a.out',f'--error={root}/slurm/%x-%A_%a.err',
 '--wrap',prefix+command],text=True).strip().split(';')[0]
path.write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs,indent=2))
