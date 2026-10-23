"""Ant v2: effective mass decay and Gymnasium-compatible forward reward."""

import gymnasium as gym
from ns_gym.schedulers import PeriodicScheduler
from ns_gym.update_functions import GeometricProgression
from ns_gym.wrappers import MujocoWrapper


class NativeRewardAntWrapper(MujocoWrapper):
    """Preserve Ant's pre-step position baseline across NS-Gym's mj_forward.

    MuJoCo's cached body positions at the end of a step differ from freshly
    forwarded positions. MujocoWrapper refreshes them before Ant.step(), which
    otherwise changes the displacement used in the forward reward even when
    no dynamics update occurs. Use the same before/after positions as native
    Ant, while retaining NS-Gym's physical updates and notifications.
    """

    def step(self, action):
        ant = self.unwrapped
        before = ant.data.body(ant._main_body).xpos[:2].copy()
        obs, _, terminated, truncated, info = super().step(action)
        after = ant.data.body(ant._main_body).xpos[:2].copy()
        velocity = (after - before) / ant.dt
        reward, components = ant._get_rew(velocity[0], action)
        info.update(components)
        info.update(x_velocity=float(velocity[0]), y_velocity=float(velocity[1]))
        return obs, float(reward), terminated, truncated, info


def make_env(change_notification=False, delta_change_notification=False, **kwargs):
    kwargs.pop('order_enforce', None)
    env = gym.make('Ant-v5', max_episode_steps=1000, **kwargs)
    update = GeometricProgression(
        PeriodicScheduler(period=100, start=100, end=900), r=0.9)
    return NativeRewardAntWrapper(
        env, {'torso_mass': update}, change_notification=change_notification,
        delta_change_notification=delta_change_notification)
