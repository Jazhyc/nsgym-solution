"""Exercise the real asynchronous pilot and verify mutation audit artifacts."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
index = int(sys.argv[2])
release = root/'release'
subprocess.run([sys.executable, str(release/'scripts/diagnostics/prelaunch/pilot.py'),
                str(root), str(index)], check=True)
definition = json.loads((root/'pilot_manifest.json').read_text())
task, method, seed = definition['matrix'][index]
folder = root/'pilot'/task/method/f'seed{seed}'
spec = importlib.util.spec_from_file_location('curriculum_summary',
    release/'scripts/diagnostics/mutation_audit/summarize_curriculum.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
summary = module.summarize(folder/'curriculum')
assert summary['incomplete_tail_lines'] == 0
assert summary['by_origin']['mutated']['completed'] > 0
assert summary['by_origin']['mutated']['replays'] > 0
cfg = json.loads((folder/'run.json').read_text())['config']
assert cfg['env']['plr']['static_mutation_protocol'] == 'relative-reflect-symmetric-v2'
from AAMAS_Comp.envs.constraints import valid_cartpole_parameter
for path in (folder/'curriculum').glob('*.jsonl'):
    for line in path.open():
        event = json.loads(line)
        if event['event'] != 'level':
            continue
        values = event['config']['values']
        if task == 'cartpole':
            assert all(valid_cartpole_parameter(k, v) for k, v in values.items())
        else:
            assert abs(values['P'][1]-values['P'][2]) < 1e-12
            assert abs(sum(values['P'])-1) < 1e-12
import wandb
info = json.loads((folder/'run.json').read_text())['wandb_run']
run = wandb.Api(timeout=60).run(f"{info['entity']}/{info['project']}/{info['id']}")
artifacts = [a for a in run.logged_artifacts() if a.type == 'paper-run']
assert any(any(name.startswith('curriculum/') for name in a.manifest.entries) for a in artifacts)
(folder/'mutation_v2_verified.json').write_text(json.dumps({
    'task': task, 'mutation_protocol': 'relative-reflect-symmetric-v2',
    'wandb': info, 'curriculum': summary, 'raw_logs_uploaded': True}, indent=2))
print('PASS: v2 physics, actual async feedback, mutation-origin metrics and uploaded lineage logs', flush=True)
