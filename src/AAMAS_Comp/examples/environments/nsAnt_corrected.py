"""Versioned Ant evaluation with effective, episode-local torso mass decay.

The legacy v0 environment remains available for reproducing historical scores.
Unlike ExponentialDecay, GeometricProgression applies a factor per event rather
than multiplying by exp(-rate * absolute_time).
"""

import gymnasium as gym
from ns_gym.schedulers import PeriodicScheduler
from ns_gym.update_functions import GeometricProgression
from ns_gym.wrappers import MujocoWrapper


def make_env(change_notification=False, delta_change_notification=False, **kwargs):
    """Reduce torso mass by 10% at wrapper times 100, 200, ..., 900.

    Updates occur before actions 101, 201, ..., 901. The first 100 actions use
    default mass; after nine changes it is 0.9**9 (~39%) of default. Normal Ant
    termination and the 1,000-action limit apply. Reset restores default mass.
    """
    # The shared evaluator passes this legacy registration option as a kwarg.
    # Registration above already disables OrderEnforcing for the NS interface.
    kwargs.pop("order_enforce", None)
    env = gym.make("Ant-v5", max_episode_steps=1000, **kwargs)
    update = GeometricProgression(
        scheduler=PeriodicScheduler(period=100, start=100, end=900), r=0.9
    )
    return MujocoWrapper(
        env,
        {"torso_mass": update},
        change_notification=change_notification,
        delta_change_notification=delta_change_notification,
    )
