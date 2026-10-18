"""Submit tests, preparation and short pilots only; never submit full training."""
import argparse
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path)
parser.add_argument('--evaluation-manifest',required=True,type=Path)
args=parser.parse_args();root=args.root.resolve();(root/'slurm').mkdir()
python='/scratch/s4325621/venvs/nsgym-paper/bin/python'
prefix=f'export PYTHONPATH={root}/release/src:{root}/release; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl; export WANDB_CACHE_DIR=/scratch/s4325621/wandb_cache WANDB_DATA_DIR={root}/wandb_data WANDB_INIT_TIMEOUT=120; cd {root}/release; '
jobs={'full_training_submitted':False}
def submit(name,command,dependency=None,array=None):
 sbatch=['sbatch','--parsable',f'--job-name={name}','--partition=regularshort','--time=04:00:00',
  '--cpus-per-task='+('32' if array else '4'),'--mem='+('96G' if array else '8G'),
  f'--output={root}/slurm/%x-%A_%a.out',f'--error={root}/slurm/%x-%A_%a.err']
 if dependency:sbatch+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
 if array:sbatch+=['--array='+array]
 return subprocess.check_output(sbatch+['--wrap',prefix+command],text=True).strip().split(';')[0]
def save(): (root/'jobs.json').write_text(json.dumps(jobs,indent=2))
tests='tests/test_complete_episode_training.py tests/test_paper_protocol.py tests/test_plr_collector_feedback.py tests/test_training_metrics.py tests/test_evaluation_trajectories.py tests/test_frozenlake_diagnostics.py tests/test_interface.py tests/test_corrected_protocol.py tests/test_plr_sampling_integrity.py tests/test_async_prelaunch_protocol.py'
jobs['tests']=submit('async_prelaunch_tests',f'{python} -m pytest -q {tests}');save()
jobs['prepare']=submit('async_prelaunch_prepare',f'{python} scripts/paper_campaign.py prepare --manifest {root}/manifest.json --evaluation-manifest {args.evaluation_manifest}',jobs['tests']);save()
jobs['plan']=submit('async_prelaunch_plan',f'{python} scripts/diagnostics/prelaunch/plan.py {root}',jobs['prepare']);save()
jobs['pilots']=submit('async_prelaunch_pilots',f'{python} scripts/diagnostics/prelaunch/pilot.py {root} $SLURM_ARRAY_TASK_ID',jobs['plan'],'0-14%3');save()
jobs['verification']=submit('async_prelaunch_verify',f'{python} scripts/diagnostics/prelaunch/verify.py {root}',jobs['pilots']);save()
print(json.dumps(jobs,indent=2))
