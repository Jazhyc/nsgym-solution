"""Audit saved trajectories for native CartPole constraint violations."""
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def audit(path):
    stats = {}
    with ZipFile(path) as archive:
        episodes = sorted({name.split('/')[0] for name in archive.namelist()})
        for episode in episodes:
            def array(name):
                return np.load(BytesIO(archive.read(f'{episode}/{name}.npy')), allow_pickle=False)
            meta = json.loads(str(array('metadata')))
            names = array('parameter_names').tolist()
            values = array('parameter_values')
            record = stats.setdefault(meta['suite'], {
                'episodes': 0, 'nonpositive_mass_or_length_episodes': 0,
                'negative_gravity_episodes': 0, 'native_constraint_violation_episodes': 0,
                'invalid_states': 0, 'recorded_states': 0, 'minima': {}})
            bad = ~np.isfinite(values).all(axis=1)
            nonpositive = np.zeros(len(values), dtype=bool)
            for key in ('masspole', 'masscart', 'length'):
                nonpositive |= values[:, names.index(key)] <= 0
            negative_gravity = values[:, names.index('gravity')] < 0
            bad |= nonpositive | negative_gravity
            record['episodes'] += 1
            record['nonpositive_mass_or_length_episodes'] += int(nonpositive.any())
            record['negative_gravity_episodes'] += int(negative_gravity.any())
            record['native_constraint_violation_episodes'] += int(bad.any())
            record['invalid_states'] += int(bad.sum())
            record['recorded_states'] += len(values)
            for i, name in enumerate(names):
                record['minima'][name] = min(record['minima'].get(name, float('inf')),
                                             float(values[:, i].min()))
    return {'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'rules': 'masspole/masscart/length > 0; gravity >= 0; finite values',
            'suites': stats}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectories', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.trajectories)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
