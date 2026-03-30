"""
Prioritized Level Replay (PLR) for NS-Gym environments.

Based on: Jiang et al. (2021) "Prioritized Level Replay"
https://arxiv.org/abs/2010.03934

The key idea: maintain a replay buffer of NSEnvConfig "levels" scored by
learning potential.  Replay high-scoring levels with probability `replay_prob`;
otherwise explore by sampling a fresh config from the NSEnvSampler.

Score semantics
---------------
Higher score = more learning potential = higher replay priority.  The standard
choice is mean absolute TD error (GAE regret):

    score = mean(|V_target(s) - V(s)|)

This is policy-agnostic (no sign), bounded away from zero on hard levels, and
naturally decays to zero once the critic has learned the level.

Staleness
---------
Levels that haven't been visited in a long time get a staleness bonus to
prevent the buffer from fixating on a small subset.  The final sampling
distribution mixes the score distribution and the staleness distribution:

    P(level) = (1 - staleness_coef) * P_score + staleness_coef * P_staleness

Usage
-----
    from AAMAS_Comp.envs import NS_ENV_SAMPLERS, NSEnvFactory
    from AAMAS_Comp.curriculum.plr import PLRBuffer, td_error_score

    sampler = NS_ENV_SAMPLERS["ant"](seed=0)
    plr = PLRBuffer(sampler, capacity=500, replay_prob=0.5)

    # Episode loop (simplified):
    level_id, config = plr.sample()
    env = NSEnvFactory.make(config)

    obs, _ = env.reset()
    values, returns = [], []
    done = False
    while not done:
        action, value = actor(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        values.append(value)
        # ... accumulate returns via GAE ...
        done = terminated or truncated

    score = td_error_score(values_array, returns_array)
    plr.update(level_id, score)

TorchRL integration
-------------------
After `advantage_module(tensordict_data)`, the tensordict contains:
  - "state_value"   : V(s) predictions, shape (T,)
  - "advantage"     : GAE advantages, shape (T,) [= V_target - V(s) approx]

The convenience function `score_from_tensordict` extracts these automatically.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional

from AAMAS_Comp.envs.ns_env_factory import NSEnvConfig
from AAMAS_Comp.envs.ns_env_sampler import NSEnvSampler


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def td_error_score(values: np.ndarray, returns: np.ndarray) -> float:
    """Mean absolute TD error.  Higher = critic still has room to learn this level.

    Args:
        values:  V(s) predictions for each timestep, shape (T,).
        returns: GAE return targets (V_target) for each timestep, shape (T,).

    Returns:
        Scalar score ≥ 0.
    """
    return float(np.mean(np.abs(np.asarray(returns) - np.asarray(values))))


def score_from_tensordict(td) -> float:
    """Extract a TD-error score from a TorchRL tensordict after GAE.

    Expects keys "state_value" and "advantage" to be present (set by
    GAEModule / VTrace before the PPO epoch loop).

    The advantage is approximately (V_target - V(s)), so:
        |TD error| ≈ |advantage| + |V(s) - V(s)| = |advantage|

    This is a slightly looser bound but avoids recomputing V_target explicitly.

    Args:
        td: A TorchRL TensorDict with "state_value" and "advantage" keys.

    Returns:
        Scalar float score.
    """
    if "advantage" not in td.keys():
        raise KeyError("'advantage' key missing — run advantage_module first")
    advantage = td["advantage"].detach().float()
    return float(advantage.abs().mean().item())


# ---------------------------------------------------------------------------
# Buffer internals
# ---------------------------------------------------------------------------

@dataclass
class _LevelEntry:
    config: NSEnvConfig
    score: float = 0.0       # current learning-potential estimate
    staleness: int = 0       # steps since this level was last visited
    n_episodes: int = 0      # total episodes played on this level


# ---------------------------------------------------------------------------
# PLR Buffer
# ---------------------------------------------------------------------------

class PLRBuffer:
    """Prioritized Level Replay buffer for NSEnvConfig objects.

    Maintains a catalog of up to `capacity` sampled NSEnvConfig levels, each
    with an associated learning-potential score.  Balances:
      - Exploration: sampling a fresh random config from the NSEnvSampler.
      - Exploitation: replaying a high-score level from the buffer.

    Args:
        sampler:        NSEnvSampler providing new random configs.
        capacity:       Maximum number of levels held simultaneously.
        replay_prob:    Probability of replaying an existing level vs exploring
                        a new one (once `min_fill` fraction of capacity is
                        populated).  Default: 0.5.
        score_temp:     Temperature for rank-based sampling: P ∝ (1/rank)^(1/temp).
                        temp=1 → standard 1/rank (paper default).
                        temp→0 → greedy (always rank 1).
                        temp→∞ → uniform.
                        Default: 0.1.
        staleness_coef: Weight in [0, 1] for the staleness component of the
                        sampling distribution.  0 = pure score-based.
                        Default: 0.1.
        min_fill:       Fraction of `capacity` to fill with exploration-only
                        episodes before activating the replay/explore mix.
                        Default: 0.1 (10 % of capacity).
        score_ema_alpha: EMA coefficient for updating scores.  If None, uses
                        1/(n_episodes + 1) (harmonic average).  If a float in
                        (0, 1), uses fixed EMA: score = α * new + (1-α) * old.
                        Default: None (harmonic average).
        seed:           RNG seed for reproducibility.
    """

    def __init__(
        self,
        sampler: NSEnvSampler,
        capacity: int = 500,
        replay_prob: float = 0.5,
        score_temp: float = 0.1,
        staleness_coef: float = 0.1,
        min_fill: float = 0.1,
        score_ema_alpha: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.sampler = sampler
        self.capacity = capacity
        self.replay_prob = replay_prob
        self.score_temp = max(score_temp, 1e-8)
        self.staleness_coef = staleness_coef
        self.min_fill_count = max(1, int(capacity * min_fill))
        self.score_ema_alpha = score_ema_alpha
        self.rng = np.random.default_rng(seed)

        self._entries: list[_LevelEntry] = []
        self.last_was_replay: bool = False  # set by sample(); True = replay, False = explore

    # ── Public API ────────────────────────────────────────────────────────────

    def sample(self) -> tuple[int, NSEnvConfig]:
        """Sample a level to run next.

        Returns:
            (level_id, config) where `level_id` is a stable buffer index.
            Pass `level_id` back to `update()` after the episode completes.
        """
        if self._should_explore():
            return self._explore()
        return self._replay()

    def update(self, level_id: int, score: float) -> None:
        """Record the learning-potential score for a completed episode.

        Args:
            level_id: Index returned by `sample()`.
            score:    Learning potential score (higher = replay more).
                      Use `td_error_score` or `score_from_tensordict`.
        """
        if level_id >= len(self._entries):
            raise IndexError(
                f"level_id {level_id} out of range ({len(self._entries)} entries). "
                "Did you call sample() first?"
            )
        entry = self._entries[level_id]

        # Update score: harmonic average or fixed EMA
        if self.score_ema_alpha is not None:
            α = self.score_ema_alpha
            entry.score = α * score + (1.0 - α) * entry.score
        else:
            # Harmonic average: new data counts less as n_episodes grows.
            # First episode: entry.score = score exactly.
            α = 1.0 / (entry.n_episodes + 1)
            entry.score = (1.0 - α) * entry.score + α * score

        entry.n_episodes += 1
        entry.staleness = 0  # reset staleness for this level

        # Increment staleness for all OTHER levels
        for i, e in enumerate(self._entries):
            if i != level_id:
                e.staleness += 1

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def is_warm(self) -> bool:
        """True once the buffer has enough levels to start replaying."""
        return self.size >= self.min_fill_count

    def stats(self) -> dict:
        """Summary statistics for logging (e.g. to WandB)."""
        if not self._entries:
            return {"plr/size": 0}
        scores = np.array([e.score for e in self._entries])
        staleness = np.array([e.staleness for e in self._entries])
        n_episodes = np.array([e.n_episodes for e in self._entries])
        return {
            "plr/size":            self.size,
            "plr/score_mean":      float(scores.mean()),
            "plr/score_max":       float(scores.max()),
            "plr/score_std":       float(scores.std()),
            "plr/staleness_mean":  float(staleness.mean()),
            "plr/staleness_max":   int(staleness.max()),
            "plr/episodes_mean":   float(n_episodes.mean()),
        }

    def top_k(self, k: int = 5) -> list[tuple[float, NSEnvConfig]]:
        """Return the top-k levels by score as [(score, config), ...]."""
        if not self._entries:
            return []
        sorted_entries = sorted(self._entries, key=lambda e: e.score, reverse=True)
        return [(e.score, e.config) for e in sorted_entries[:k]]

    def sampling_distribution(self) -> np.ndarray:
        """Current normalized sampling probability for each buffer entry."""
        return self._compute_distribution()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _should_explore(self) -> bool:
        # Always explore until min_fill is reached
        if self.size < self.min_fill_count:
            return True
        # If buffer is full, never explore (all new configs evict old ones)
        if self.size >= self.capacity:
            return self.rng.random() > self.replay_prob
        # Mixed phase
        return self.rng.random() > self.replay_prob

    def _explore(self) -> tuple[int, NSEnvConfig]:
        self.last_was_replay = False
        config = self.sampler.sample()
        if self.size >= self.capacity:
            # Evict the lowest-scoring entry
            idx = int(np.argmin([e.score for e in self._entries]))
            self._entries[idx] = _LevelEntry(config=config)
        else:
            idx = len(self._entries)
            self._entries.append(_LevelEntry(config=config))
        return idx, config

    def _replay(self) -> tuple[int, NSEnvConfig]:
        self.last_was_replay = True
        dist = self._compute_distribution()
        idx = int(self.rng.choice(len(self._entries), p=dist))
        return idx, self._entries[idx].config

    def _compute_distribution(self) -> np.ndarray:
        n = len(self._entries)
        if n == 0:
            return np.array([])

        scores = np.array([e.score for e in self._entries])
        staleness = np.array([e.staleness for e in self._entries], dtype=float)

        # Rank-based score distribution (Jiang et al. 2021).
        # rank 1 = highest score. Weight = (1/rank)^(1/temp), then normalise.
        # Scale-invariant: only ordering matters, not absolute score values.
        ranks = np.empty(n, dtype=float)
        ranks[np.argsort(scores)[::-1]] = np.arange(1, n + 1)
        score_dist = (1.0 / ranks) ** (1.0 / self.score_temp)
        score_dist /= score_dist.sum()

        # Staleness distribution: proportional to staleness count
        if staleness.sum() > 0:
            staleness_dist = staleness / staleness.sum()
        else:
            staleness_dist = np.ones(n) / n  # uniform if all freshly visited

        # Combine and renormalize
        dist = (1.0 - self.staleness_coef) * score_dist + self.staleness_coef * staleness_dist
        dist /= dist.sum()  # guard against floating-point drift
        return dist


# ---------------------------------------------------------------------------
# Episode runner compatible with PLR (gymnasium-based, no TorchRL)
# ---------------------------------------------------------------------------

def run_plr_episode(
    env,
    actor_fn,
    value_fn,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[float, float]:
    """Run one episode and return (episode_return, td_error_score).

    This is a reference implementation for environments used outside TorchRL.
    The actor and value functions are plain callables:

        action           = actor_fn(obs)       # obs: np.ndarray
        value: float     = value_fn(obs)

    GAE is used to compute the return targets for scoring.

    Returns:
        (episode_return, score) where score = mean |advantage|
    """
    obs, _ = env.reset()

    observations, actions, rewards, values, dones = [], [], [], [], []

    done = False
    while not done:
        action = actor_fn(obs)
        value = value_fn(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        dones.append(done)

        obs = next_obs

    # Bootstrap final value (0 if terminal, else V(s_T))
    last_value = 0.0 if dones[-1] else value_fn(obs)

    # Compute GAE returns
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = np.array(values) + advantages
    score = td_error_score(np.array(values), returns)
    episode_return = float(sum(rewards))

    return episode_return, score
