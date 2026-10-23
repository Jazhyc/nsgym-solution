"""Check local episode-wise pilots against the frozen bounded cluster pool."""
import argparse
import json
import math
from pathlib import Path
import torch
from AAMAS_Comp.envs.ant_curriculum import validate_ant_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    validate_ant_samples(manifest['training']['tasks']['ant']['dr_samples'])
    results = []
    torch.set_num_threads(1)
    for method in ('standard_dr', 'static_plr', 'static_plr_mutations'):
        folder = root / 'local/ant' / method / 'seed0'
        metadata = json.loads((folder / 'run.json').read_text())
        history = [json.loads(line) for line in (folder / 'training_metrics.jsonl').read_text().splitlines()]
        assert metadata['task'] == 'ant' and metadata['actual_frames'] == 65536
        assert all(math.isfinite(v) for row in history for v in row.values() if isinstance(v, (int, float)))
        assert not (folder / 'nonfinite_failure.json').exists()
        checkpoint = torch.load(folder / 'ppo_final.pt', map_location='cpu', weights_only=False)
        assert all(torch.isfinite(p).all() for key in ('actor', 'critic') for p in checkpoint[key].parameters())
        assert checkpoint['training_state']['optimizer']['state']
        assert checkpoint['training_state']['reward_rms']
        assert max(h['train/initial_policy_kl'] for h in history) < 1e-6
        checks = dict(method=method, frames=metadata['actual_frames'], finite=True, learner_state_saved=True)
        if method != 'standard_dr':
            events = [json.loads(line) for path in (folder / 'curriculum').glob('*.jsonl')
                      for line in path.read_text().splitlines()]
            levels = [e for e in events if e['event'] == 'level']
            assert levels and all(set(e['config']) == {'task', 'values'} for e in levels)
            validate_ant_samples([e['config']['values'] for e in levels])
            scores = [e for e in events if e['event'] == 'score']
            assert scores and all(e['applied'] and math.isfinite(e['score']) for e in scores)
            replays = sum(e['event'] == 'selection' and e['replay'] for e in events)
            mutants = sum(e['origin'] == 'mutated' for e in levels)
            assert replays > 0
            assert (mutants > 0) == (method == 'static_plr_mutations')
            assert history[-1]['plr/incomplete_episode_scores_total'] == 0
            assert history[-1]['plr/abandoned_episode_scores_total'] == 0
            checks.update(replays=replays, mutants=mutants, applied_scores=len(scores), discarded_scores=0)
        results.append(checks)
    report = {'passed': True, 'training_pool_sha256': manifest['training_pool_sha256'], 'runs': results}
    (root / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
