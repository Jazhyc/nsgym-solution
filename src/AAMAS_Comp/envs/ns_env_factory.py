"""
NSEnvFactory — dynamically build non-stationary NS-Gym environments from config.

Each NSEnvConfig specifies:
  - env_id          : gymnasium environment id
  - tunable_params  : {param_name: ParamConfig(scheduler, update_fn)}
  - wrapper options : change_notification, delta_change_notification

Pre-built named configs live in NS_ENV_CONFIGS and cover Ant, CartPole, and
FrozenLake.  The intent is that PLR can sample from this catalogue, mutate
configs, and call NSEnvFactory.make() to instantiate environments on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import gymnasium as gym

import ns_gym.schedulers as sched_module
import ns_gym.update_functions as uf_module
from ns_gym.wrappers import MujocoWrapper, NSClassicControlWrapper, NSFrozenLakeWrapper


# ---------------------------------------------------------------------------
# Env-family routing
# ---------------------------------------------------------------------------

_MUJOCO_IDS = {"Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Humanoid-v5"}
_CLASSIC_IDS = {"CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"}
_FROZEN_IDS  = {"FrozenLake-v1", "FrozenLake8x8-v1"}


def _env_family(env_id: str) -> str:
    if env_id in _MUJOCO_IDS:
        return "mujoco"
    if env_id in _CLASSIC_IDS:
        return "classic"
    if env_id in _FROZEN_IDS:
        return "frozen"
    raise ValueError(f"Unknown env family for id '{env_id}'. "
                     "Add it to the routing sets in ns_env_factory.py.")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    """Scheduler to use and its constructor kwargs."""
    cls: str                         # name from ns_gym.schedulers
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self):
        cls = getattr(sched_module, self.cls)
        return cls(**self.kwargs)


@dataclass
class UpdateFnConfig:
    """Update function and its constructor kwargs (scheduler injected at build time)."""
    cls: str                         # name from ns_gym.update_functions
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, scheduler):
        cls = getattr(uf_module, self.cls)
        return cls(scheduler=scheduler, **self.kwargs)


@dataclass
class ParamConfig:
    """Full spec for a single tunable parameter."""
    scheduler: SchedulerConfig
    update_fn: UpdateFnConfig

    def build(self):
        s = self.scheduler.build()
        return self.update_fn.build(s)


@dataclass
class NSEnvConfig:
    """Complete spec for a non-stationary environment."""
    env_id: str
    tunable_params: dict[str, ParamConfig]
    change_notification: bool = False
    delta_change_notification: bool = False
    # Extra kwargs forwarded to gym.make() (e.g. disable_env_checker=True)
    gym_kwargs: dict[str, Any] = field(default_factory=dict)
    # Extra kwargs forwarded to NS wrapper (e.g. initial_prob_dist for FrozenLake)
    wrapper_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class NSEnvFactory:
    """Build gymnasium-compatible NS-Gym environments from NSEnvConfig objects."""

    @staticmethod
    def make(config: NSEnvConfig) -> gym.Env:
        base_env = gym.make(config.env_id, **config.gym_kwargs)
        tunable = {name: pc.build() for name, pc in config.tunable_params.items()}
        family  = _env_family(config.env_id)

        common = dict(
            change_notification=config.change_notification,
            delta_change_notification=config.delta_change_notification,
            **config.wrapper_kwargs,
        )

        if family == "mujoco":
            return MujocoWrapper(base_env, tunable, **common)
        if family == "classic":
            return NSClassicControlWrapper(base_env, tunable, **common)
        if family == "frozen":
            return NSFrozenLakeWrapper(base_env, tunable_params=tunable, **common)

        raise RuntimeError(f"Unhandled family '{family}'")  # unreachable


# ---------------------------------------------------------------------------
# Pre-built named configurations
# ---------------------------------------------------------------------------
# Access via: NS_ENV_CONFIGS["ant_gravity_decay"]
# Each value is a *callable* that returns a fresh NSEnvConfig so the mutable
# dataclass objects (dicts) are not shared between calls.

def _ant_gravity_decay() -> NSEnvConfig:
    """
    Ant-v5: gravity magnitude decreases steadily (env becomes 'floatier').

    MuJoCo gravity is a 3D vector [0, 0, -9.81].  IncrementUpdate(k=+0.1)
    with a ContinuousScheduler adds +0.1 to every component each step, so
    the z-axis weakens from -9.81 toward 0 over ~98 steps.
    """
    return NSEnvConfig(
        env_id="Ant-v5",
        tunable_params={
            "gravity": ParamConfig(
                scheduler=SchedulerConfig("ContinuousScheduler"),
                update_fn=UpdateFnConfig("IncrementUpdate", {"k": 0.1}),
            )
        },
    )


def _ant_torso_mass_decay() -> NSEnvConfig:
    """Ant-v5: torso mass decays exponentially (90% every 500 steps)."""
    return NSEnvConfig(
        env_id="Ant-v5",
        tunable_params={
            "torso_mass": ParamConfig(
                scheduler=SchedulerConfig("PeriodicScheduler", {"period": 500}),
                update_fn=UpdateFnConfig("ExponentialDecay", {"decay_rate": 0.9}),
            )
        },
    )


def _ant_floor_friction_random_walk() -> NSEnvConfig:
    """
    Ant-v5: floor friction via random walk (placeholder — floor_friction is not
    in AntEnv's TUNABLE_PARAMS registry; falls back to torso_mass random walk).
    """
    return NSEnvConfig(
        env_id="Ant-v5",
        tunable_params={
            "torso_mass": ParamConfig(
                scheduler=SchedulerConfig("PeriodicScheduler", {"period": 200}),
                update_fn=UpdateFnConfig("RandomWalk"),
            )
        },
    )


def _ant_multi_param() -> NSEnvConfig:
    """
    Ant-v5: gravity weakens continuously + torso mass oscillates periodically.
    Tests adaptation to simultaneous, asynchronous parameter changes.
    """
    return NSEnvConfig(
        env_id="Ant-v5",
        tunable_params={
            "gravity": ParamConfig(
                scheduler=SchedulerConfig("ContinuousScheduler"),
                update_fn=UpdateFnConfig("IncrementUpdate", {"k": 0.05}),
            ),
            "torso_mass": ParamConfig(
                scheduler=SchedulerConfig("PeriodicScheduler", {"period": 300}),
                update_fn=UpdateFnConfig("OscillatingUpdate"),
            ),
        },
    )


def _cartpole_mass_increment() -> NSEnvConfig:
    """CartPole-v1: pole mass increases steadily (harder to balance)."""
    return NSEnvConfig(
        env_id="CartPole-v1",
        tunable_params={
            "masspole": ParamConfig(
                scheduler=SchedulerConfig("ContinuousScheduler"),
                update_fn=UpdateFnConfig("IncrementUpdate", {"k": 0.01}),
            )
        },
    )


def _cartpole_gravity_random_walk() -> NSEnvConfig:
    """CartPole-v1: gravity drifts via random walk every 3 steps."""
    return NSEnvConfig(
        env_id="CartPole-v1",
        tunable_params={
            "gravity": ParamConfig(
                scheduler=SchedulerConfig("PeriodicScheduler", {"period": 3}),
                update_fn=UpdateFnConfig("RandomWalk"),
            )
        },
    )


def _cartpole_multi_param() -> NSEnvConfig:
    """CartPole-v1: pole mass increases + gravity random walk (original ns example)."""
    return NSEnvConfig(
        env_id="CartPole-v1",
        tunable_params={
            "masspole": ParamConfig(
                scheduler=SchedulerConfig("ContinuousScheduler"),
                update_fn=UpdateFnConfig("IncrementUpdate", {"k": 0.1}),
            ),
            "gravity": ParamConfig(
                scheduler=SchedulerConfig("PeriodicScheduler", {"period": 3}),
                update_fn=UpdateFnConfig("RandomWalk"),
            ),
        },
    )


def _frozenlake_transition_decay() -> NSEnvConfig:
    """FrozenLake-v1: transition probabilities shift over time (original ns example)."""
    return NSEnvConfig(
        env_id="FrozenLake-v1",
        gym_kwargs={"disable_env_checker": True},
        wrapper_kwargs={"initial_prob_dist": [1, 0, 0]},
        tunable_params={
            "P": ParamConfig(
                scheduler=SchedulerConfig("ContinuousScheduler"),
                update_fn=UpdateFnConfig("DistributionDecrementUpdate", {"k": 0.025}),
            )
        },
    )


NS_ENV_CONFIGS: dict[str, callable] = {
    # Ant
    "ant_gravity_decay":           _ant_gravity_decay,
    "ant_torso_mass_decay":        _ant_torso_mass_decay,
    "ant_floor_friction_rw":       _ant_floor_friction_random_walk,
    "ant_multi_param":             _ant_multi_param,
    # CartPole
    "cartpole_mass_increment":     _cartpole_mass_increment,
    "cartpole_gravity_rw":         _cartpole_gravity_random_walk,
    "cartpole_multi_param":        _cartpole_multi_param,
    # FrozenLake
    "frozenlake_transition_decay": _frozenlake_transition_decay,
}
