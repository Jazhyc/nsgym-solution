"""
FastNSClassicControlWrapper — low-overhead NS wrapper for CartPole-family envs.

Drops the per-step costs that the stock NSClassicControlWrapper pays regardless
of configuration:

  1. _constraint_checker: string class-name dispatch + warning infrastructure.
     Our sampler already keeps params in safe ranges; we skip this.
  2. NSWrapper.step(): builds 4 dicts + nested obs dict on every step even when
     change_notification=False. We bypass it and call gym.Wrapper.step directly.
  3. _dependency_resolver: class-name dispatch + equality checks before setattr.
     We inline the CartPole deps unconditionally.
  4. Info augmentation ("Ground Truth Env Change" etc.) — skipped.

The returned observation is the raw numpy array from gymnasium (not a dict),
matching the base CartPole observation space. This makes TorchRL GymWrapper
work without any extra transforms.

All NS parameter updates still fire every step — the update functions and
schedulers are called identically to the original wrapper.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Union

import gymnasium as gym
import numpy as np

import ns_gym.base as base


class FastNSClassicControlWrapper(gym.Wrapper):
    """Minimal-overhead NS wrapper for Classic Control envs (CartPole, etc.).

    Args:
        env:            Base gymnasium environment (unwrapped).
        tunable_params: Dict mapping param name → built UpdateFn object.
                        Build via `{name: ParamConfig.build() for ...}`.
        has_masscart:   Set False for envs that don't have masscart/length
                        (i.e., not CartPoleEnv). Disables dep resolution.
    """

    def __init__(
        self,
        env: gym.Env,
        tunable_params: dict[str, Any],
        has_cartpole_deps: bool = True,
    ) -> None:
        super().__init__(env)
        self.tunable_params = tunable_params
        self._has_cartpole_deps = has_cartpole_deps
        self.t = 0

        # Store initial param values for reset
        self._initial_params = {
            k: deepcopy(getattr(env.unwrapped, k)) for k in tunable_params
        }

        # Expose raw (non-dict) observation space so TorchRL/GymWrapper works
        self.observation_space = env.unwrapped.observation_space

    # ── Core step (hot path) ──────────────────────────────────────────────────

    def step(self, action: Union[int, float]):
        # 1. Apply all parameter updates directly — no constraint checking
        unwrapped = self.env.unwrapped
        for p, fn in self.tunable_params.items():
            new_val, _, _ = fn(getattr(unwrapped, p), self.t)
            setattr(unwrapped, p, new_val)

        # 2. Inline CartPole dependency resolution (no class-name dispatch)
        if self._has_cartpole_deps:
            unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
            unwrapped.polemass_length = unwrapped.length * unwrapped.masspole

        # 3. Call gymnasium step directly — no dict wrapping, no info augment
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += 1
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.t = 0
        # Restore initial parameter values (matches NSClassicControlWrapper behaviour)
        unwrapped = self.env.unwrapped
        for k, v in self._initial_params.items():
            setattr(unwrapped, k, deepcopy(v))
        if self._has_cartpole_deps:
            unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
            unwrapped.polemass_length = unwrapped.length * unwrapped.masspole
        return obs, info
