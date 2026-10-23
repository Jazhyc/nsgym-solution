"""Versioned Ant training physics; historical samplers remain reproducible."""

import numpy as np

from AAMAS_Comp.examples.environments.nsAnt_native_reward import NativeRewardAntWrapper


# Engineering limits for the new training protocol, not fitted to policy scores.
# Gravity: half to twice Earth gravity, always downward. Torso: 1/4 to 4x
# the native Ant-v5 mass. Both fresh levels and mutation offspring use these.
ANT_TORSO_MASS = 0.32724923474893675
ANT_BOUNDS = {"gravity": (-19.62, -4.905),
              "torso_mass": (ANT_TORSO_MASS / 4, ANT_TORSO_MASS * 4)}


class BoundedAntWrapper(NativeRewardAntWrapper):
    """Reject out-of-domain proposals before applying physics or notifications.

    Rejection holds the previous valid value. This also avoids silently
    projecting different schedules onto a boundary. Counts expose saturation.
    """

    def reset(self, **kwargs):
        self.rejected_updates = 0
        return super().reset(**kwargs)

    def _constraint_checker(self, new_vals):
        invalid = {}
        for key, value in new_vals.items():
            scalar = float(value[-1]) if key == "gravity" else float(value)
            lo, hi = ANT_BOUNDS[key]
            invalid[key] = not (np.isfinite(scalar) and lo <= scalar <= hi)
        self.rejected_updates += sum(invalid.values())
        return invalid

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["ant/rejected_updates"] = self.rejected_updates
        return obs, reward, terminated, truncated, info


def make_ant_v2_sampler(seed=None):
    from copy import deepcopy
    from .ns_env_sampler import ANT_PARAM_SPECS, NSEnvSampler

    specs = deepcopy(ANT_PARAM_SPECS)
    gravity, mass = specs
    # NS-Gym adds slope*t and multiplies exp(-rate*t) at EACH firing.
    # sum(t, t=0..999)=499500, not 1000. Bound full-horizon effects.
    gravity.update_fn_overrides["DeterministicTrend"] = {"slope": (1e-8, 9e-6)}
    gravity.update_fn_overrides["LinearInterpolation"]["T"] = (100, 1000)
    mass.update_fn_overrides["ExponentialDecay"] = {"decay_rate": (1e-7, 2e-6)}
    return NSEnvSampler("Ant-v5", specs, seed=seed,
                        wrapper_kwargs={"ant_physics_protocol": "bounded-v2"})


def validate_ant_samples(samples):
    if not samples:
        raise ValueError('Empty bounded Ant training pool')
    for row in samples:
        if set(row) != set(ANT_BOUNDS) or any(
                not np.isfinite(row[k]) or not lo <= row[k] <= hi
                for k, (lo, hi) in ANT_BOUNDS.items()):
            raise ValueError('Static Ant pool violates bounded-v2 physics; recalibrate')


def calibrate_ant_v2(count=256, seed=71311):
    """Joint time snapshots from the exact bounded temporal update process.

    No policy or termination-dependent selection: sixteen snapshots per full
    1,000-step parameter trajectory, with the same rejection rule as the live
    wrapper. Training and test callers must use independent seeds.
    """
    from ns_gym.schedulers import MemorylessScheduler
    sampler = make_ant_v2_sampler(seed)
    defaults = {'gravity': -9.81, 'torso_mass': ANT_TORSO_MASS}
    times = set(np.linspace(0, 999, 16, dtype=int))
    snapshots = []
    for i in range(count):
        functions = {k: pc.build() for k, pc in sampler.sample().tunable_params.items()}
        for j, fn in enumerate(functions.values()):
            for k, obj in enumerate((fn, fn.scheduler)):
                if hasattr(obj, 'rng'):
                    obj.rng = np.random.default_rng(np.random.SeedSequence([seed, i, j, k]))
                    if isinstance(obj, MemorylessScheduler):
                        obj.transition_time = obj.rng.geometric(p=obj.p, size=(1,))
        values = dict(defaults)
        for t in range(1000):
            for key, fn in functions.items():
                candidate, _, _ = fn(values[key], t)
                lo, hi = ANT_BOUNDS[key]
                if np.isfinite(candidate) and lo <= candidate <= hi:
                    values[key] = float(candidate)
            if t in times:
                snapshots.append(dict(values))
    validate_ant_samples(snapshots)
    return {'defaults': defaults, 'dr_samples': snapshots,
            'dr_ranges': {k: [min(row[k] for row in snapshots), max(row[k] for row in snapshots)]
                          for k in ANT_BOUNDS}}
