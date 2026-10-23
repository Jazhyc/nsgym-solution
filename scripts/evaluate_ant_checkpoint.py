"""Evaluate an existing Ant checkpoint on corrected ExampleNSAnt-v2 dynamics.

Creates a new output directory; never overwrites historical results or weights.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from AAMAS_Comp.evaluation.public import evaluate_public_checkpoint
from AAMAS_Comp.examples.environments import nsAnt_corrected
from AAMAS_Comp.examples.environments import nsAnt_native_reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes-per-level', type=int, default=100)
    parser.add_argument('--environment', choices=['ExampleNSAnt-v1', 'ExampleNSAnt-v2'],
                        default='ExampleNSAnt-v2')
    args = parser.parse_args()
    if args.episodes_per_level < 1:
        parser.error('--episodes-per-level must be positive')
    checkpoint = args.checkpoint.resolve(strict=True)
    metadata = {
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'environment': args.environment,
        'environment_source_sha256': hashlib.sha256(
            Path((nsAnt_native_reward if args.environment == 'ExampleNSAnt-v2'
                  else nsAnt_corrected).__file__).read_bytes()).hexdigest(),
        'training_unchanged': True,
        'notification_modes': ['none'],
        'rollout_seed_start': 42,
        'mass_update': {'factor': 0.9, 'wrapper_times': list(range(100, 901, 100))},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'evaluation_provenance.json').write_text(json.dumps(metadata, indent=2))
    torch.set_num_threads(1)
    metrics = evaluate_public_checkpoint(
        checkpoint, 'ant', args.output, metadata,
        episodes_per_level=args.episodes_per_level, environment_id=args.environment,
        notification_modes=('none',),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
