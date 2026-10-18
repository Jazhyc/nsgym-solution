"""
PLREnv — gymnasium wrapper that embeds a PLRBuffer.

On every reset():
  1. Applies delayed, episode-specific mean |advantage| feedback from the
     learner. Replaced level generations reject stale feedback.
  2. Calls plr.update() to update the buffer.
  3. Samples the next config from the buffer.
  4. Rebuilds the underlying NS env from that config.

Env-family dispatch:
  - CartPole-family: FastNSClassicControlWrapper (low overhead, raw numpy obs).
  - MuJoCo / FrozenLake: NSEnvFactory.make() (stock NS wrappers).

Each instance owns its buffer and feedback queue. TorchRL's ParallelEnv can
safely spawn one per subprocess; no score slots are shared between groups.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
from queue import Empty
from typing import Any
from uuid import uuid4

import gymnasium as gym
import numpy as np

from AAMAS_Comp.envs.ns_env_factory import NSEnvConfig, NSEnvFactory
from AAMAS_Comp.envs.fast_ns_wrapper import FastNSClassicControlWrapper
from AAMAS_Comp.envs.ns_env_sampler import NS_ENV_SAMPLERS, make_sampler
from AAMAS_Comp.envs.wrappers import ContextFlatWrapper
from AAMAS_Comp.curriculum.plr import PLRBuffer

# Env families that benefit from FastNSClassicControlWrapper
_CLASSIC_CONTROL_IDS = {"CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"}
# MuJoCo env IDs whose NS wrapper returns a composite dict obs keyed by tunable param names
_MUJOCO_IDS = {"Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Humanoid-v5"}
# NSWrapper envs that return dict obs {"state": int, ...} — need ContextFlatWrapper
_DICT_OBS_IDS = {"FrozenLake-v1"}


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


def _build_ns_env(
    config: NSEnvConfig,
    context_keys: list[str] | None = None,
    context_defaults: dict | None = None,
) -> gym.Env:
    """Build the appropriate NS wrapper for a given config.

    - Classic control: FastNSClassicControlWrapper (raw array obs, low overhead).
    - MuJoCo: MujocoWrapper via NSEnvFactory + _MujocoRawObsAdapter (strips
      the composite dict obs so the obs space is stable across PLR configs).
    - Dict-obs envs (FrozenLake): NSEnvFactory + ContextFlatWrapper (converts
      the NSWrapper dict obs to a flat float32 array and optionally appends
      context features read from the info dict).
    """
    from AAMAS_Comp.envs.static_curriculum import StaticDynamicsConfig, build_static_level
    if isinstance(config, StaticDynamicsConfig):
        return build_static_level(config)
    if config.env_id in _CLASSIC_CONTROL_IDS:
        base = gym.make(config.env_id, **config.gym_kwargs)
        tunable = {name: pc.build() for name, pc in config.tunable_params.items()}
        return FastNSClassicControlWrapper(base, tunable)
    if config.env_id in _MUJOCO_IDS:
        ns_env = NSEnvFactory.make(config)
        return _MujocoRawObsAdapter(ns_env)
    ns_env = NSEnvFactory.make(config)
    if config.env_id in _DICT_OBS_IDS:
        return ContextFlatWrapper(ns_env, context_keys or [], context_defaults or {})
    return ns_env


class PLREnv(gym.Env):
    """Gymnasium env that rotates NS configs via PLR on every reset().

    Each worker owns a buffer. The training path receives scored episode IDs
    through its feedback queue; legacy callers without feedback use return.

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
        feedback_queue=None,
        context_keys: list[str] | None = None,
        context_defaults: dict | None = None,
        protect_unscored: bool = True,
        uniform_replay: bool = False,
        score_ema_alpha: float | None = 1.0,
        replay_requires_score: bool = True,
        selection_staleness: bool = True,
        eviction_strategy: str | None = None,
        frozenlake_initial_mode: str | None = None,
        sampler_override=None,
        audit_dir=None,
    ) -> None:
        super().__init__()
        self._context_keys = list(context_keys or [])
        self._context_defaults = dict(context_defaults or {})
        sampler = (sampler_override if sampler_override is not None else
                   make_sampler(sampler_key, seed, frozenlake_initial_mode))
        self.plr = PLRBuffer(
            sampler,
            capacity=plr_capacity,
            replay_prob=replay_prob,
            mutation_prob=mutation_prob,
            mutation_sigma=mutation_sigma,
            score_temp=score_temp,
            staleness_coef=staleness_coef,
            seed=seed,
            protect_unscored=protect_unscored,
            uniform_replay=uniform_replay,
            score_ema_alpha=score_ema_alpha,
            replay_requires_score=replay_requires_score,
            selection_staleness=selection_staleness,
            eviction_strategy=eviction_strategy,
        )
        self._level_id: int | None = None
        self._episode_return: float = 0.0
        self._env: gym.Env | None = None
        self._stats_queue = stats_queue
        self._worker_idx: int = worker_idx
        self._score_array = score_array
        self._feedback_queue = feedback_queue
        self._level_tokens = {}
        self._token_counter = 0
        self._applied_scores = 0
        self._discarded_scores = 0
        self._episode_count: int = 0
        self._replay_count: int = 0
        self._mutation_count: int = 0
        self._forced_replay_count: int = 0
        self._audit_dir = Path(audit_dir) if audit_dir else None
        self._audit_file = None
        self._current_selection = None
        self._episode_steps = 0
        self._episode_terminated = self._episode_truncated = False
        self._outcome_recorded = True
        self._origin_counts = {origin: dict(selection_count=0, replay_count=0,
                                           completed_count=0, return_sum=0., steps_sum=0,
                                           partial_count=0)
                               for origin in ('fresh', 'mutated')}

        # Build a throwaway env to read spaces, then discard it
        init_config = sampler.sample()
        tmp = _build_ns_env(init_config, self._context_keys, self._context_defaults)
        self.observation_space = tmp.observation_space
        self.action_space = tmp.action_space
        tmp.close()

    def reset(self, *, seed=None, options=None):
        self._record_outcome()
        # Async feedback may arrive after later episodes have started. Apply it
        # only to the same level generation, never to a replacement buffer slot.
        if self._feedback_queue is not None:
            while True:
                try:
                    token, episode, score = self._feedback_queue.get_nowait()
                except Empty:
                    break
                slot = next((i for i, t in self._level_tokens.items() if t == token), None)
                if slot is None:
                    self._discarded_scores += 1
                    self._audit('score', token=token, episode=episode, score=score, applied=False)
                else:
                    self.plr.update(slot, score)
                    self._applied_scores += 1
                    self._audit('score', token=token, episode=episode, score=score, applied=True,
                                uid=self.plr._entries[slot].uid)
        elif self._level_id is not None:
            if self._score_array is not None:
                td_score = float(self._score_array[self._worker_idx])
                score = td_score if td_score > 0.0 else self._episode_return
            else:
                score = self._episode_return
            self.plr.update(self._level_id, score)

        # Sample next config and rebuild env
        self._level_id, config = self.plr.sample()
        if not self.plr.last_was_replay:
            self._token_counter += 1
            self._level_tokens[self._level_id] = self._token_counter
        entry = self.plr._entries[self._level_id]
        origin = entry.origin
        self._origin_counts[origin]['selection_count'] += 1
        self._origin_counts[origin]['replay_count'] += int(self.plr.last_was_replay)

        # Track replay/mutation/explore ratios and push stats to main process
        if self.plr.last_was_replay:
            self._replay_count += 1
        if self.plr.last_was_mutation:
            self._mutation_count += 1
        if self.plr.last_was_forced_replay:
            self._forced_replay_count += 1
        self._episode_count += 1
        self._current_selection = dict(uid=entry.uid, token=self._level_tokens[self._level_id],
                                       origin=origin, episode=self._episode_count,
                                       replay=self.plr.last_was_replay,
                                       mutation_depth=entry.mutation_depth)
        if not self.plr.last_was_replay:
            self._audit('level', **self._current_selection, parent_uid=entry.parent_uid,
                        config=asdict(config))
        self._audit('selection', **self._current_selection, forced_replay=self.plr.last_was_forced_replay,
                    conditional_replay_probability=self.plr.last_replay_probability,
                    conditional_replay_mutated_mass=self.plr.last_replay_mutated_mass,
                    parent_probability=self.plr.last_parent_probability, score=entry.score,
                    seed=seed)
        if self._stats_queue is not None:
            stats = self.plr.stats()
            stats["plr/worker_id"] = self._worker_idx
            stats["plr/episode_count"] = self._episode_count
            stats["plr/replay_count"] = self._replay_count
            stats["accel/mutation_count"] = self._mutation_count
            stats["plr/forced_replay_count"] = self._forced_replay_count
            stats["plr/replay_fraction"] = self._replay_count / self._episode_count
            stats["accel/mutation_fraction"] = self._mutation_count / self._episode_count
            stats["plr/forced_replay_fraction"] = self._forced_replay_count / self._episode_count
            stats["plr/unscored_levels"] = sum(e.n_episodes == 0 for e in self.plr._entries)
            stats["plr/episode_return"] = self._episode_return
            stats["plr/applied_scores"] = self._applied_scores
            stats["plr/discarded_scores"] = self._discarded_scores
            for label, counts in self._origin_counts.items():
                stats.update({f'plr/{label}_{key}': value for key, value in counts.items()})
            try:
                self._stats_queue.put_nowait(stats)
            except Exception:
                pass  # queue full — drop silently

        if self._env is not None:
            self._env.close()
        self._env = _build_ns_env(config, self._context_keys, self._context_defaults)

        self._episode_return = 0.0
        self._episode_steps = 0
        self._episode_terminated = self._episode_truncated = False
        self._outcome_recorded = False
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, {**info, **self._metadata()}

    def _audit(self, event, **data):
        if self._audit_dir is None:
            return
        if self._audit_file is None:
            self._audit_dir.mkdir(parents=True, exist_ok=True)
            path = self._audit_dir/f'worker{self._worker_idx}_pid{os.getpid()}_{uuid4().hex[:8]}.jsonl'
            self._audit_file = path.open('x', buffering=1)
        self._audit_file.write(json.dumps(dict(schema_version=1, event=event,
                                               worker=self._worker_idx, **data))+'\n')

    def _record_outcome(self):
        if self._outcome_recorded or self._current_selection is None:
            return
        complete = self._episode_terminated or self._episode_truncated
        counts = self._origin_counts[self._current_selection['origin']]
        if complete:
            counts['completed_count'] += 1
            counts['return_sum'] += self._episode_return
            counts['steps_sum'] += self._episode_steps
        else:
            counts['partial_count'] += 1
        self._audit('outcome', **self._current_selection, complete=complete,
                    episode_return=self._episode_return, steps=self._episode_steps,
                    terminated=self._episode_terminated, truncated=self._episode_truncated)
        self._outcome_recorded = True

    def _metadata(self):
        return {
            "plr_worker": np.asarray(self._worker_idx, dtype=np.int64),
            "plr_level": np.asarray(self._level_tokens[self._level_id], dtype=np.int64),
            "plr_episode": np.asarray(self._episode_count, dtype=np.int64),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._episode_return += float(reward)
        self._episode_steps += 1
        self._episode_terminated, self._episode_truncated = bool(terminated), bool(truncated)
        if terminated or truncated:
            self._record_outcome()
        return obs, reward, terminated, truncated, {**info, **self._metadata()}

    def close(self):
        self._record_outcome()
        if self._env is not None:
            self._env.close()
        if self._audit_file is not None:
            self._audit_file.close()
            self._audit_file = None
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

    def __init__(
        self,
        sampler_key: str = "cartpole",
        seed: int | None = None,
        context_keys: list[str] | None = None,
        context_defaults: dict | None = None,
        frozenlake_initial_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._context_keys = list(context_keys or [])
        self._context_defaults = dict(context_defaults or {})
        self._sampler = make_sampler(sampler_key, seed, frozenlake_initial_mode)
        self._env: gym.Env | None = None

        # Build a throwaway env to read spaces, then discard it
        config = self._sampler.sample()
        tmp = _build_ns_env(config, self._context_keys, self._context_defaults)
        self.observation_space = tmp.observation_space
        self.action_space = tmp.action_space
        tmp.close()

    def reset(self, *, seed=None, options=None):
        config = self._sampler.sample()
        if self._env is not None:
            self._env.close()
        self._env = _build_ns_env(config, self._context_keys, self._context_defaults)
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

    def __init__(
        self,
        config: NSEnvConfig,
        context_keys: list[str] | None = None,
        context_defaults: dict | None = None,
    ) -> None:
        super().__init__()
        self._config = deepcopy(config)
        self._context_keys = list(context_keys or [])
        self._context_defaults = dict(context_defaults or {})
        self._env = _build_ns_env(config, context_keys or [], context_defaults or {})
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        # Rebuild schedulers/update functions as well as resetting physical values.
        self._env.close()
        self._env = _build_ns_env(self._config, self._context_keys, self._context_defaults)
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
    frozenlake_initial_mode: str | None = None,
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
    sampler = make_sampler(sampler_key, seed, frozenlake_initial_mode)
    return [sampler.sample() for _ in range(n_configs)]
