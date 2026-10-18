"""Verify existing output/artifact digests and finish a timed-out W&B run."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import argparse
import wandb
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('folder',type=Path)
parser.add_argument('--minimum-frames',type=int,default=10000000)
parser.add_argument('--original-job',default='32904036_6')
args=parser.parse_args()
folder=args.folder.resolve()
meta=json.loads((folder/'run.json').read_text());info=meta['wandb_run']
assert meta['actual_frames']>=args.minimum_frames
api=wandb.Api(timeout=60)
run=api.run(f"{info['entity']}/{info['project']}/{info['id']}")
artifacts=[a for a in run.logged_artifacts() if a.type=='paper-run']
assert artifacts, 'Missing committed run artifact'
artifact=artifacts[-1]
checked=[]
for path in folder.iterdir():
 if not path.is_file() or not (path.suffix in ('.json','.jsonl','.npz') or path.name=='ppo_final.pt'):continue
 if path.name=='finalization_recovery.json':continue
 entry=artifact.manifest.entries.get(path.name)
 assert entry is not None, f'Missing artifact file {path.name}'
 md5=hashlib.md5()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):md5.update(chunk)
 digest=base64.b64encode(md5.digest()).decode()
 assert entry.digest==digest, (path.name,entry.digest,digest)
 checked.append(path.name)
assert {'ppo_final.pt','public_summary.json','test_summary.json','test_trajectories.npz'}<=set(checked)
with wandb.init(entity=info['entity'],project=info['project'],id=info['id'],resume='must',dir=str(folder)) as resumed:
 resumed.summary['recovery/outputs_and_artifact_verified']=True
 resumed.summary['recovery/original_job']=args.original_job
(folder/'finalization_recovery.json').write_text(json.dumps({'verified_files':checked,'artifact':artifact.name,'training_rerun':False,'wandb':info},indent=2))
print('Recovered completed training/evaluation; verified all existing artifact file digests')
