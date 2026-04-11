"""
PLREnv — gymnasium wrapper that embeds a PLRBuffer.

On every reset():
  1. Scores the previous episode using mean |advantage| written by the main
     process after GAE (correct learning-potential signal).  Falls back to
     episode return if the score array hasn't been populated yet (first batch).
  2. Calls plr.update() to update the buffer.
  3. Samples the next config from the buffer.
  4. Rebuilds the underlying NS env from that config.

Env-family dispatch:
  - CartPole-family: FastNSClassicControlWrapper (low overhead, raw numpy obs).
  - MuJoCo / FrozenLake: NSEnvFactory.make() (stock NS wrappers).

Each instance is fully self-contained — no shared state between workers.
TorchRL's ParallelEnv can safely spawn one per subprocess.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym

from AAMAS_Comp.envs.ns_env_factory import NSEnvConfig, NSEnvFactory
from AAMAS_Comp.envs.fast_ns_wrapper import FastNSClassicControlWrapper
from AAMAS_Comp.envs.ns_env_sampler import NS_ENV_SAMPLERS
from AAMAS_Comp.curriculum.plr import PLRBuffer

# Env families that benefit from FastNSClassicControlWrapper
_CLASSIC_CONTROL_IDS = {"CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"}
# MuJoCo env IDs whose NS wrapper returns a composite dict obs keyed by tunable param names
_MUJOCO_IDS = {"Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Humanoid-v5"}


class _MujocoRawObsAdapter(gym.Wrapper):
    """Strip the composite dict obs from MujocoWrapper and return raw physics obs.

    MujocoWrapper returns a composite observation whose keys are the names of the
    tunable parameters (e.g. {"gravity": ..., "torso_mass": ...}).  When PLR
    rotates between configs with different tunable params the obs spec changes,
    which TorchRL cannot handle at runtime.

    This adapter bypasses the composite obs entirely:
      - It still calls MujocoWrapper.step() so NS param updates happen correctly.
      - It then reads the raw physics observation directly from the unwrapped env
        via env.unwrapped._get_obs(), which is stable across all configs.

    The resulting observation space matches the base gymnasium MuJoCo env.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = env.unwrapped.observation_space

    def reset(self, *, seed=None, options=None):
        self.env.reset(seed=seed, options=options)  # NS wrapper initialises params
        obs = self.env.unwrapped._get_obs()
        return obs, {}

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)  # NS params updated
        obs = self.env.unwrapped._get_obs()
        return obs, reward, terminated, truncated, info


def _build_ns_env(config: NSEnvConfig) -> gym.Env:
    """Build the appropriate NS wrapper for a given config.

    - Classic control: FastNSClassicControlWrapper (raw array obs, low overhead).
    - MuJoCo: MujocoWrapper via NSEnvFactory + _MujocoRawObsAdapter (strips
      the composite dict obs so the obs space is stable across PLR configs).
    - Other (e.g. FrozenLake): NSEnvFactory.make() as-is.
    """
    if config.env_id in _CLASSIC_CONTROL_IDS:
        base = gym.make(config.env_id, **config.gym_kwargs)
        tunable = {name: pc.build() for name, pc in config.tunable_params.items()}
        return FastNSClassicControlWrapper(base, tunable)
    if config.env_id in _MUJOCO_IDS:
        ns_env = NSEnvFactory.make(config)
        return _MujocoRawObsAdapter(ns_env)
    return NSEnvFactory.make(config)


class PLREnv(gym.Env):
    """Gymnasium env that rotates NS configs via PLR on every reset().

    Each subprocess worker owns an independent PLRBuffer — no IPC needed.
    Episode return is used as the PLR score proxy (TD-error scoring requires
    critic values which are not available inside the env).

    Args:
        sampler_key:    Key into NS_ENV_SAMPLERS ("cartpole", "ant", "frozenlake").
        plr_capacity:   Max levels in the PLR buffer.
        replay_prob:    Probability of replaying vs exploring a new config.
        mutation_prob:  Of the explore steps, fraction that use ACCEL mutation
                        (perturbing a high-scoring config) rather than random
                        sampling. 0.0 = pure PLR (default).
        mutation_sigma: Noise scale for mutation (fraction of each kwarg's range
                        width). Default: 0.2.
        score_temp:     Softmax temperature for score-based sampling.
        staleness_coef: Weight for staleness component in sampling distribution.
        seed:           RNG seed for the PLR buffer (None → OS entropy).
    """

    def __init__(
        self,
        sampler_key: str = "cartpole",
        plr_capacity: int = 100,
        replay_prob: float = 0.5,
        mutation_prob: float = 0.0,
        mutation_sigma: float = 0.2,
        score_temp: float = 0.1,
        staleness_coef: float = 0.1,
        seed: int | None = None,
        stats_queue=None,
        worker_idx: int = 0,
        score_array=None,
    ) -> None:
        super().__init__()
        sampler = NS_ENV_SAMPLERS[sampler_key](seed=seed)
        self.plr = PLRBuffer(
            sampler,
            capacity=plr_capacity,
            replay_prob=replay_prob,
            mutation_prob=mutation_prob,
            mutation_sigma=mutation_sigma,
            score_temp=score_temp,
            staleness_coef=staleness_coef,
            seed=seed,
        )
        self._level_id: int | None = None
        self._episode_return: float = 0.0
        self._env: gym.Env | None = None
        self._stats_queue = stats_queue
        self._worker_idx: int = worker_idx
        self._score_array = score_array
        self._episode_count: int = 0
        self._replay_count: int = 0
        self._mutation_count: int = 0

        # Build a throwaway env to read spaces, then discard it
        _, init_config = self.plr.sample()
        tmp = _build_ns_env(init_config)
        self.observation_space = tmp.observation_space
        self.action_space = tmp.action_space
        tmp.close()

    def reset(self, *, seed=None, options=None):
        # Score the episode that just ended (skip before first episode).
        # Prefer mean |advantage| written by the main process after GAE (correct
        # learning-potential signal).  Fall back to episode return on the first
        # batch before the score array has been populated (value == 0.0).
        if self._level_id is not None:
            if self._score_array is not None:
                td_score = float(self._score_array[self._worker_idx])
                score = td_score if td_score > 0.0 else self._episode_return
            else:
                score = self._episode_return
            self.plr.update(self._level_id, score)

        # Sample next config and rebuild env
        self._level_id, config = self.plr.sample()

        # Track replay/mutation/explore ratios and push stats to main process
        if self.plr.last_was_replay:
            self._replay_count += 1
        if self.plr.last_was_mutation:
            self._mutation_count += 1
        self._episode_count += 1
        if self._stats_queue is not None:
            stats = self.plr.stats()
            stats["plr/replay_fraction"] = self._replay_count / self._episode_count
            stats["accel/mutation_fraction"] = self._mutation_count / self._episode_count
            stats["plr/episode_return"] = self._episode_return
            try:
                self._stats_queue.put_nowait(stats)
            except Exception:
                pass  # queue full — drop silently

        if self._env is not None:
            self._env.close()
        self._env = _build_ns_env(config)

        self._episode_return = 0.0
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._episode_return += float(reward)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()
        super().close()


class RandomNSEnv(gym.Env):
    """NS env that randomly samples a new config on every reset() — no PLR.

    Used for the NS baseline: trains on uniformly-random NS configs without
    curriculum, enabling a fair comparison against PLR on the same held-out
    NS evaluation configs.

    Args:
        sampler_key: Key into NS_ENV_SAMPLERS ("cartpole", "ant", "frozenlake").
        seed:        RNG seed (None → OS entropy so each worker differs).
    """

    def __init__(self, sampler_key: str = "cartpole", seed: int | None = None) -> None:
        super().__init__()
        self._sampler = NS_ENV_SAMPLERS[sampler_key](seed=seed)
        self._env: gym.Env | None = None

        # Build a throwaway env to read spaces, then discard it
        config = self._sampler.sample()
        tmp = _build_ns_env(config)
        self.observation_space = tmp.observation_space
        self.action_space = tmp.action_space
        tmp.close()

    def reset(self, *, seed=None, options=None):
        config = self._sampler.sample()
        if self._env is not None:
            self._env.close()
        self._env = _build_ns_env(config)
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        if self._env is not None:
            self._env.close()
        super().close()


class FixedNSEnv(gym.Env):
    """Fixed-config NS env for held-out evaluation.

    Uses the same NS config every episode — parameters evolve within each
    episode (as defined by the scheduler/update_fn) but reset to initial
    values on each reset(). This gives a stable eval signal over training.

    Args:
        config: Pre-sampled NSEnvConfig to use for every episode.
    """

    def __init__(self, config: NSEnvConfig) -> None:
        super().__init__()
        self._env = _build_ns_env(config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()
        super().close()


def sample_held_out_configs(
    sampler_key: str,
    n_configs: int,
    seed: int,
) -> list[NSEnvConfig]:
    """Sample a fixed set of NS configs for held-out evaluation.

    Uses a dedicated seed so eval configs are independent of training configs.

    Args:
        sampler_key: Key into NS_ENV_SAMPLERS.
        n_configs:   Number of distinct configs to sample.
        seed:        Deterministic seed for reproducible eval sets.

    Returns:
        List of NSEnvConfig objects, fixed for the lifetime of training.
    """
    sampler = NS_ENV_SAMPLERS[sampler_key](seed=seed)
    return [sampler.sample() for _ in range(n_configs)]
