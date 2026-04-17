"""Gymnasium wrappers and normalization utilities for NS-Gym training.

No TorchRL dependency — safe to import from submission.py or any context.
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
from gymnasium import Wrapper as GymnasiumWrapper


class ContextFlatWrapper(GymnasiumWrapper):
    """Converts NSWrapper dict obs ``{"state": int, ...}`` to a flat float32 array.

    NSFrozenLakeWrapper returns a composite dict observation that is
    incompatible with TorchRL MLP networks.  This wrapper:

      1. One-hot encodes the discrete ``obs["state"]`` integer.
      2. Optionally appends context scalars extracted from the info dict
         (e.g. ``info["transition_prob"]``).

    The resulting observation space is ``Box(n_state + n_context,)``.

    At episode reset the context is re-initialised from ``context_defaults``.
    After every step the cache is updated from ``info`` if the key is present.

    Args:
        env: A gymnasium env whose observation space is ``Dict`` with a
            ``"state"`` key that is ``Discrete(n)``.
        context_keys: Ordered list of ``info`` keys to append (e.g.
            ``["transition_prob"]``).  Empty list = one-hot only.
        context_defaults: Mapping from key → default list value used when
            the key is absent from ``info`` (e.g. at episode start).
    """

    def __init__(
        self,
        env: gym.Env,
        context_keys: list[str] | None = None,
        context_defaults: dict | None = None,
    ) -> None:
        super().__init__(env)
        self.context_keys: list[str] = list(context_keys or [])
        self.context_defaults: dict = dict(context_defaults or {})

        self._n_state: int = env.observation_space["state"].n
        self._context_sizes: dict[str, int] = {
            k: len(v) for k, v in self.context_defaults.items()
        }
        n_context = sum(self._context_sizes.get(k, 1) for k in self.context_keys)

        self._last_context: dict[str, np.ndarray] = {}
        self._reset_context()

        total_dim = self._n_state + n_context
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

    def _reset_context(self) -> None:
        for k in self.context_keys:
            default = self.context_defaults.get(k, [0.0])
            self._last_context[k] = np.array(default, dtype=np.float32)

    def _update_context(self, info: dict) -> None:
        for k in self.context_keys:
            if k in info:
                self._last_context[k] = np.array(info[k], dtype=np.float32)

    def _build_flat_obs(self, state_int: int) -> np.ndarray:
        one_hot = np.zeros(self._n_state, dtype=np.float32)
        one_hot[int(state_int)] = 1.0
        if not self.context_keys:
            return one_hot
        ctx = np.concatenate([self._last_context[k] for k in self.context_keys])
        return np.concatenate([one_hot, ctx])

    def reset(self, *, seed=None, options=None):
        self._reset_context()
        obs, info = self.env.reset(seed=seed, options=options)
        self._update_context(info)
        return self._build_flat_obs(obs["state"]), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_context(info)
        return self._build_flat_obs(obs["state"]), reward, terminated, truncated, info


class NoInfoWrapper(GymnasiumWrapper):
    """Drop the info dict from step/reset — prevents unused keys (Ant reward
    components, position/velocity diagnostics) from being serialized over IPC
    on every environment step."""

    def step(self, action):
        obs, reward, terminated, truncated, _info = self.env.step(action)
        return obs, reward, terminated, truncated, {}

    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return obs, {}


class RunningMeanStd:
    """Welford online running mean / variance tracker.

    Tracks the sufficient statistics (mean, var, count) so that
    ObservationNorm transforms can be initialised from a random rollout
    and frozen during training.
    """

    def __init__(self, shape: tuple[int, ...] = (), device: torch.device | None = None):
        # float64 for better numerical stability in running statistics
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.var = torch.ones(shape, device=device, dtype=torch.float64)
        self.count: float = 0

    def update(self, batch: torch.Tensor) -> None:
        """Update stats with a new batch of observations (N, *shape)."""
        batch = batch.reshape(-1, *self.mean.shape)
        batch_f64 = batch.double()
        batch_mean = batch_f64.mean(dim=0)
        batch_var = batch_f64.var(dim=0, correction=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var.clamp(min=1e-8)).clamp(min=1e-4)

    def state_dict(self) -> dict:
        return {"mean": self.mean.clone(), "var": self.var.clone(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


class RewardNormalizer:
    """VecNormalize-style reward scaling.

    Tracks running variance of rewards and divides by sqrt(var) to bring
    reward magnitudes to ~O(1).  Does NOT subtract the mean.

    Uses epsilon-initialized count (like SB3) so the initial std = 1.0
    (identity scaling) and statistics build up gradually.
    """

    def __init__(self, device: torch.device | None = None):
        self.reward_rms = RunningMeanStd(shape=(), device=device)
        self.reward_rms.count = 1e-4  # epsilon init → initial std = 1.0

    def normalize_batch(self, tensordict_data) -> None:
        """Update stats and normalize rewards in a collected batch in-place."""
        rewards = tensordict_data["next", "reward"]
        self.reward_rms.update(rewards)
        std = self.reward_rms.std.to(rewards.dtype)
        tensordict_data["next", "reward"].copy_(rewards / std)
