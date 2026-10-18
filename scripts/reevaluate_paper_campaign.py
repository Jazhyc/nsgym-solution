"""Reevaluate existing paper checkpoints after an evaluation seed correction.

Preserves trained weights, training configuration and old results. Outputs
new evaluation records with explicit source/checkpoint/manifest provenance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from paper_campaign import matrix, provenance
from AAMAS_Comp.evaluation.paper import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--frozenlake-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--group', default='dr-plr-20261009-seeded-reevaluation')
    parser.add_argument('--reason', default='Record complete evaluation trajectories with corrected scheduler seeding')
    args = parser.parse_args()
    torch.set_num_threads(1)
    definition = json.loads(args.manifest.read_text())
    source = provenance()
    if source['source_sha256'] != definition['provenance']['source_sha256']:
        raise RuntimeError('Source differs from reevaluation manifest')
    task, method, seed = matrix('final')[args.index]
    source_root = args.frozenlake_root if task == 'frozenlake' else args.original_root
    folder = source_root / task / method / f'seed{seed}'
    output = args.output_root / task / method / f'seed{seed}'
    if (output / 'test_summary.json').exists():
        raise FileExistsError(f'Refusing to overwrite reevaluated results: {output}')
    checkpoint = folder / 'ppo_final.pt'
    metadata = json.loads((folder / 'run.json').read_text())
    metadata['reevaluation'] = {
        'reason': args.reason,
        'source': source, 'original_run': str(folder),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'training_unchanged': True}
    output.mkdir(parents=True, exist_ok=True)
    for name in ('ppo_final.pt', 'resolved_config.yaml', 'command.json'):
        if (folder / name).exists():
            shutil.copy2(folder / name, output / name)
    (output / 'run.json').write_text(json.dumps(metadata, indent=2))
    metrics = evaluate_checkpoint(checkpoint, args.manifest, task, output, metadata)
    import wandb
    with wandb.init(entity='ml_exp', project='nsgym-paper',
                    group=args.group,
                    name=f'{task}-{method}-seed{seed}', job_type='reevaluation',
                    tags=['paper', 'reevaluation', task, method],
                    config={'task': task, 'method': method, 'seed': seed, **metadata['reevaluation']},
                    dir=str(output)):
        wandb.log(metrics)
        artifact = wandb.Artifact(f'seeded-evaluation-{task}-{method}-seed{seed}',
                                  type='paper-reevaluation', metadata=metadata['reevaluation'])
        for name in ('test_summary.json', 'test_episodes.jsonl', 'test_trajectories.npz',
                     'test_trajectory_metadata.json', 'run.json'):
            artifact.add_file(str(output / name))
        artifact.add_file(str(args.manifest), name='manifest.json')
        wandb.log_artifact(artifact)
    print(json.dumps({'task': task, 'method': method, 'seed': seed,
                      'iid': metrics['test/iid/mean']}), flush=True)


if __name__ == '__main__':
    main()
