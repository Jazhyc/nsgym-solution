"""Fixed physics levels drawn from the same joint snapshot pool as static DR."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from AAMAS_Comp.envs.domain_randomization import DynamicsEnv, TASK_IDS
from AAMAS_Comp.envs.constraints import valid_cartpole_parameter


@dataclass
class StaticDynamicsConfig:
    task: str
    values: dict

    @property
    def env_id(self):
        return TASK_IDS[self.task]


def build_static_level(config):
    # One possible snapshot means every reset reapplies exactly this level.
    # DynamicsEnv without an intervention never changes physics during step().
    return DynamicsEnv(config.task, {}, samples=[config.values])


class StaticDynamicsSampler:
    """Uniform fresh joint draws; Gaussian mutations of physical values only.

    V2 scalar noise is sigma times the parent's absolute magnitude, with a
    floor of 1% of the pool's median nonzero magnitude. Proposals reflect into
    the calibrated bounds. FrozenLake mutates only straight-motion probability,
    reflecting within its feasible bounds and retaining equal lateral slips.
    Legacy range-scaled clipping remains available explicitly for reproduction.
    Fresh draws, parent configs and the saved joint pool are never modified.
    """

    def __init__(self, task, samples, seed=None, mutation_protocol='relative-reflect-symmetric-v2'):
        if task not in TASK_IDS or not samples:
            raise ValueError("Static PLR requires a known task and nonempty training snapshots")
        self.task = task
        if mutation_protocol not in ('range-clip-v1', 'relative-reflect-symmetric-v2'):
            raise ValueError('Unknown static mutation protocol')
        self.mutation_protocol = mutation_protocol
        self.samples = deepcopy(samples)
        self.rng = np.random.default_rng(seed)
        self.bounds = {}
        self.reference_scales = {}
        keys = set(self.samples[0])
        expected = {'frozenlake': {'P'}, 'ant': {'gravity', 'torso_mass'},
                    'cartpole': {'masspole', 'masscart', 'length', 'gravity'}}[task]
        if keys != expected or any(set(row) != keys for row in self.samples):
            raise ValueError("Static snapshot parameter keys do not match task")
        for key in sorted(keys):
            values = np.asarray([row[key] for row in self.samples], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite static physics")
            if task == 'cartpole' and not all(valid_cartpole_parameter(key, value) for value in values):
                raise ValueError("Invalid CartPole static physics; recalibrate with native constraints")
            if key == 'P':
                if values.shape != (len(samples), 3) or np.any(values < 0) or not np.allclose(values.sum(1), 1):
                    raise ValueError("Invalid static transition probabilities")
            elif values.ndim != 1 or (key != 'gravity' and np.any(values <= (0 if task == 'cartpole' else 1e-6))):
                raise ValueError("Invalid static scalar physics")
            self.bounds[key] = (values.min(axis=0), values.max(axis=0))
            if key != 'P':
                nonzero = np.abs(values[values != 0])
                self.reference_scales[key] = float(np.median(nonzero)) if len(nonzero) else 0.

    @staticmethod
    def _reflect(value, lower, upper):
        """Fold a proposal into its interval without a clipping probability atom."""
        if upper == lower:
            return float(lower)
        width = upper-lower
        offset = (value-lower) % (2*width)
        return float(lower + (offset if offset <= width else 2*width-offset))

    def sample(self):
        return StaticDynamicsConfig(self.task, deepcopy(
            self.samples[int(self.rng.integers(len(self.samples)))]))

    def mutate(self, parent, rng, sigma=0.2):
        if parent.task != self.task or not np.isfinite(sigma) or sigma < 0:
            raise ValueError("Invalid static mutation")
        values = {}
        for key, (lower, upper) in self.bounds.items():
            original = np.asarray(parent.values[key], dtype=float)
            if sigma == 0:
                values[key] = deepcopy(parent.values[key])
                continue
            if self.mutation_protocol == 'relative-reflect-symmetric-v2':
                if key == 'P':
                    if not np.isclose(original[1], original[2], rtol=0, atol=1e-12):
                        raise ValueError('Symmetric FrozenLake mutation requires symmetric parents')
                    lo = max(lower[0], 1-2*upper[1], 1-2*upper[2])
                    hi = min(upper[0], 1-2*lower[1], 1-2*lower[2])
                    if lo > hi:
                        raise ValueError('No feasible symmetric FrozenLake mutation interval')
                    straight = self._reflect(float(original[0])+rng.normal()*sigma*(hi-lo), lo, hi)
                    values[key] = [straight, (1-straight)/2, (1-straight)/2]
                else:
                    # Local physical magnitude sets the step. A small median-
                    # based floor permits movement from zero/near-zero values;
                    # rare pool maxima never set the step for typical parents.
                    scale = max(abs(float(original)), .01*self.reference_scales[key])
                    values[key] = self._reflect(float(original)+rng.normal()*sigma*scale,
                                                float(lower), float(upper))
                continue
            proposed = original + rng.normal(size=original.shape) * sigma * (upper - lower)
            if key == 'P':
                # Euclidean projection onto a bounded probability simplex.
                left, right = float(np.min(proposed - upper)), float(np.max(proposed - lower))
                for _ in range(80):
                    middle = (left + right) / 2
                    if np.clip(proposed - middle, lower, upper).sum() > 1:
                        left = middle
                    else:
                        right = middle
                proposed = np.clip(proposed - (left + right) / 2, lower, upper)
            else:
                proposed = np.clip(proposed, lower, upper)
            values[key] = proposed.tolist()
        return StaticDynamicsConfig(self.task, values)
