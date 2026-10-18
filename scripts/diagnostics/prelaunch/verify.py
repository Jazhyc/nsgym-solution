"""Write readiness only after every short pilot and full dry-run passes."""
import hashlib
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'release/scripts'));import paper_campaign as campaign
plan=json.loads((root/'launch_plan.json').read_text())
assert plan['n_runs']==75 and not plan['full_training_submitted']
assert plan['source_sha256']==campaign.provenance()['source_sha256']
manifest=json.loads((root/'pilot_manifest.json').read_text())
verified=[]
for task,method,seed in manifest['matrix']:
 p=root/'pilot'/task/method/f'seed{seed}'/'pilot_verified.json'
 verified.append(json.loads(p.read_text()))
assert len(verified)==15
ready={'ready':True,'protocol':campaign.PROTOCOL,'source_sha256':plan['source_sha256'],
 'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
 'main_run_count':75,'full_training_submitted':False,'pilots':verified}
(root/'readiness.json').write_text(json.dumps(ready,indent=2))
print('READY: 15 approved async pilots and 75 resolved full configurations; full training NOT submitted')
