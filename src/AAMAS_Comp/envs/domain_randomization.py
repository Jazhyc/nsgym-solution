"""Reset-only dynamics randomization and explicit, verifiable test interventions."""

from copy import deepcopy

import gymnasium as gym
import numpy as np

from AAMAS_Comp.envs.ns_env_factory import NSEnvConfig, ParamConfig, SchedulerConfig, UpdateFnConfig
from AAMAS_Comp.envs.constraints import valid_cartpole_parameter

TASK_IDS = {"ant": "Ant-v5", "cartpole": "CartPole-v1", "frozenlake": "FrozenLake-v1"}
HORIZONS = {"ant": 1000, "cartpole": 500, "frozenlake": 200}


def seed_update_functions(env, seed):
    from ns_gym.schedulers import MemorylessScheduler

    rng = np.random.default_rng(seed)
    current = env
    while current is not None:
        functions = list(current.__dict__.get("tunable_params", {}).values())
        if "_update_fn" in current.__dict__:
            functions.append(current._update_fn)
        for fn in functions:
            for obj in (fn, fn.scheduler):
                if hasattr(obj, "rng"):
                    obj.rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                    # This scheduler draws its first event in __init__. Merely
                    # replacing its RNG leaves that event seeded from OS entropy.
                    if isinstance(obj, MemorylessScheduler):
                        obj.transition_time = obj.rng.geometric(p=obj.p, size=(1,))
        current = current.__dict__.get("env", current.__dict__.get("_env"))


def parameters(env, task):
    current = env
    while current.__dict__.get("env", current.__dict__.get("_env")) is not None:
        current = current.__dict__.get("env", current.__dict__.get("_env"))
    raw = current.unwrapped
    if task == "ant":
        return {"gravity": float(raw.model.opt.gravity[2]),
                "torso_mass": float(raw.model.body_mass[raw.model.body("torso").id])}
    if task == "cartpole":
        return {key: float(getattr(raw, key)) for key in ("masspole", "masscart", "length", "gravity")}
    current = env
    while current is not None:
        if "_prob" in current.__dict__:
            return {"P": current._prob.tolist()}
        current = current.__dict__.get("env", current.__dict__.get("_env"))
    raise ValueError("FrozenLake parameter inspection requires the fast NS wrapper")


def set_parameters(env, task, values):
    raw = env.unwrapped
    if task == "ant":
        import mujoco
        if "gravity" in values:
            raw.model.opt.gravity[2] = values["gravity"]
        if "torso_mass" in values:
            if values["torso_mass"] <= 1e-6:
                raise ValueError("Torso mass must be positive")
            raw.model.body_mass[raw.model.body("torso").id] = values["torso_mass"]
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(raw.model, spec))
        mujoco.mj_getState(raw.model, raw.data, state, spec)
        mujoco.mj_setConst(raw.model, raw.data)
        mujoco.mj_setState(raw.model, raw.data, state, spec)
        mujoco.mj_forward(raw.model, raw.data)
    elif task == "cartpole":
        if any(not valid_cartpole_parameter(key, value) for key, value in values.items()):
            raise ValueError("Invalid CartPole physical parameters")
        for key, value in values.items():
            setattr(raw, key, float(value))
        raw.total_mass = raw.masspole + raw.masscart
        raw.polemass_length = raw.length * raw.masspole
    else:
        prob = np.asarray(values["P"], dtype=float)
        if np.any(prob < 0) or not np.isclose(prob.sum(), 1):
            raise ValueError("Invalid transition distribution")
        current = env
        while "_prob" not in current.__dict__:
            current = current.env
        current._prob = prob.copy()


def static_env(task):
    if task != "frozenlake":
        return gym.make(TASK_IDS[task])
    from AAMAS_Comp.curriculum.plr_env import _build_ns_env
    config = NSEnvConfig(TASK_IDS[task], {"P": ParamConfig(
        SchedulerConfig("ContinuousScheduler"), UpdateFnConfig("UniformDrift", {"rate": 0.0}))},
        wrapper_kwargs={"initial_prob_dist": [1 / 3] * 3})
    return _build_ns_env(config)


class DynamicsEnv(gym.Wrapper):
    """Randomize at reset, or apply a fixed abrupt/gradual test intervention."""

    def __init__(self, task, ranges, seed=0, intervention=None, samples=None, native_ant_reward=False):
        super().__init__(static_env(task))
        self.task = task
        self.ranges = deepcopy(ranges)
        self.rng = np.random.default_rng(seed)
        self.intervention = intervention
        self.samples = samples
        self.native_ant_reward = native_ant_reward
        self.t = 0

    def sample_values(self):
        if self.samples:
            return deepcopy(self.samples[int(self.rng.integers(len(self.samples)))])
        if self.task == "frozenlake":
            straight = self.rng.uniform()
            left = self.rng.uniform(0, 1 - straight)
            return {"P": [float(straight), float(left), float(1 - straight - left)]}
        return {key: float(self.rng.uniform(*bounds)) for key, bounds in self.ranges.items()}

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.t = 0
        values = self.sample_values() if self.intervention is None else self.intervention["initial"]
        set_parameters(self.env, self.task, values)
        if self.task == "ant":
            obs = self.unwrapped._get_obs()
        return obs, info

    def step(self, action):
        before_xy = (self.unwrapped.data.body(self.unwrapped._main_body).xpos[:2].copy()
                     if self.task == 'ant' and self.native_ant_reward else None)
        changed = False
        if self.intervention is not None:
            start = self.intervention["step"]
            duration = self.intervention["duration"]
            if self.t >= start and duration >= 0:
                fraction = 1.0 if duration == 0 else min((self.t - start + 1) / duration, 1.0)
                values = {key: (np.asarray(value) + fraction * (
                    np.asarray(self.intervention["target"][key]) - np.asarray(value))).tolist()
                    for key, value in self.intervention["initial"].items()}
                before = parameters(self.env, self.task)
                set_parameters(self.env, self.task, values)
                after = parameters(self.env, self.task)
                changed = any(not np.allclose(before[k], after[k], rtol=0, atol=1e-12) for k in after)
                if not all(np.allclose(after[k], values[k]) for k in values):
                    raise RuntimeError("Simulator did not apply the requested intervention")
        obs, reward, terminated, truncated, info = self.env.step(action)
        if before_xy is not None:
            ant = self.unwrapped
            velocity = (ant.data.body(ant._main_body).xpos[:2] - before_xy) / ant.dt
            reward, components = ant._get_rew(velocity[0], action)
            info.update(components, x_velocity=float(velocity[0]), y_velocity=float(velocity[1]))
        self.t += 1
        return obs, reward, terminated, truncated, {**info, "verified_change": changed}
