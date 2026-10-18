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

This uses the current policy's value error without its sign. It is not a
success or difficulty score: an unsuccessful policy with a correct near-zero
value prediction can also receive a near-zero score.

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

    With GAE targets, advantage = V_target - V(s). This is the absolute GAE
    target error, rather than the absolute one-step TD residual.

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
    origin: str = 'fresh'
    uid: int = 0
    parent_uid: int | None = None
    mutation_depth: int = 0


# ---------------------------------------------------------------------------
# PLR Buffer
# ---------------------------------------------------------------------------

class PLRBuffer:
    """Prioritized Level Replay buffer for NSEnvConfig objects.

    Maintains a catalog of up to `capacity` sampled NSEnvConfig levels, each
    with an associated learning-potential score.  Balances:
      - Exploration:  sampling a fresh random config from the NSEnvSampler.
      - Mutation:     perturbing a high-scoring config (ACCEL extension).
      - Exploitation: replaying a high-score level from the buffer.

    When `mutation_prob > 0`, this becomes an ACCEL buffer: a fraction of the
    explore steps are replaced by mutations of high-scoring buffer entries.
    Set `mutation_prob=0.0` (default) for standard PLR behaviour.

    Args:
        sampler:        NSEnvSampler providing new random configs (and mutations).
        capacity:       Maximum number of levels held simultaneously.
        replay_prob:    Probability of replaying an existing level vs exploring
                        a new one (once `min_fill` fraction of capacity is
                        populated).  Default: 0.5.
        mutation_prob:  Of the explore steps, fraction that are ACCEL mutations
                        (perturbing a high-scoring config) rather than random
                        new samples.  0.0 = pure PLR.  Default: 0.0.
        mutation_sigma: Noise scale for mutation as a fraction of each kwarg's
                        range width.  0.2 → std = 20% of (hi - lo).  Default: 0.2.
        score_temp:     Temperature for rank-based sampling: P ∝ (1/rank)^(1/temp).
                        temp=1 → 1/rank weighting.
                        temp→0 → greedy (always rank 1).
                        temp→∞ → uniform.
                        Default: 0.1.
        staleness_coef: Weight in [0, 1] for the staleness component of the
                        sampling distribution.  0 = pure score-based.
                        Default: 0.1.
        min_fill:       Fraction of `capacity` to fill with exploration-only
                        episodes before activating the replay/explore mix.
                        Default: 0.1 (10 % of capacity).
        score_ema_alpha: EMA coefficient for updating scores. If None, uses
                        1/(n_episodes + 1) (lifetime arithmetic mean). If a float
                        in (0, 1], uses fixed EMA: score = α * new + (1-α) * old.
                        Alpha=1 replaces the score with the latest observation.
                        Default: 1 (latest completed-episode score).
        seed:           RNG seed for reproducibility.
    """

    def __init__(
        self,
        sampler: NSEnvSampler,
        capacity: int = 500,
        replay_prob: float = 0.5,
        mutation_prob: float = 0.0,
        mutation_sigma: float = 0.2,
        score_temp: float = 0.1,
        staleness_coef: float = 0.1,
        min_fill: float = 0.1,
        score_ema_alpha: Optional[float] = 1.0,
        seed: Optional[int] = None,
        protect_unscored: bool = True,
        uniform_replay: bool = False,
        replay_requires_score: bool = True,
        selection_staleness: bool = True,
        eviction_strategy: Optional[str] = None,
    ) -> None:
        self.sampler = sampler
        self.capacity = capacity
        self.replay_prob = replay_prob
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.score_temp = max(score_temp, 1e-8)
        self.staleness_coef = staleness_coef
        self.min_fill_count = max(1, int(capacity * min_fill))
        self.score_ema_alpha = score_ema_alpha
        self.rng = np.random.default_rng(seed)
        self.protect_unscored = protect_unscored
        self.uniform_replay = uniform_replay
        self.replay_requires_score = replay_requires_score
        self.selection_staleness = selection_staleness
        self.eviction_strategy = eviction_strategy or ("random" if uniform_replay else "score")
        if self.eviction_strategy not in ("random", "score"):
            raise ValueError("eviction_strategy must be 'random' or 'score'")
        if score_ema_alpha is not None and not 0 < score_ema_alpha <= 1:
            raise ValueError("score_ema_alpha must be in (0, 1], or None for legacy lifetime averaging")
        if capacity < 1 or not 0 <= replay_prob <= 1 or not 0 <= staleness_coef <= 1:
            raise ValueError("Invalid capacity, replay probability or staleness coefficient")

        self._entries: list[_LevelEntry] = []
        self.last_was_replay: bool = False    # True = replay, False = explore/mutate
        self.last_was_mutation: bool = False  # True = ACCEL mutation, False = random explore
        self.last_was_forced_replay: bool = False
        self._levels_created = 0
        self._levels_first_scored = 0
        self._unscored_evictions = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def sample(self) -> tuple[int, NSEnvConfig]:
        """Sample a level to run next.

        Returns:
            (level_id, config) where `level_id` is a stable buffer index.
            Pass `level_id` back to `update()` after the episode completes.
        """
        self.last_was_forced_replay = False
        self.last_replay_probability = None
        self.last_replay_mutated_mass = None
        self.last_parent_probability = None
        if self._should_explore():
            result = self._explore()
        else:
            result = self._replay()
        if self.selection_staleness:
            selected = result[0]
            for i, entry in enumerate(self._entries):
                entry.staleness = 0 if i == selected else entry.staleness + 1
        return result

    def update(self, level_id: int, score: float) -> None:
        """Record the learning-potential score for a completed episode.

        Args:
            level_id: Index returned by `sample()`.
            score:    Learning potential score (higher = replay more).
                      Use `td_error_score` or `score_from_tensordict`.
        """
        if not 0 <= level_id < len(self._entries):
            raise IndexError(
                f"level_id {level_id} out of range ({len(self._entries)} entries). "
                "Did you call sample() first?"
            )
        entry = self._entries[level_id]
        if not np.isfinite(score) or score < 0:
            raise ValueError("PLR scores must be finite and nonnegative")
        if entry.n_episodes == 0:
            self._levels_first_scored += 1

        # Update score: lifetime arithmetic mean or fixed EMA
        if entry.n_episodes == 0:
            entry.score = float(score)
        elif self.score_ema_alpha is not None:
            α = self.score_ema_alpha
            entry.score = α * score + (1.0 - α) * entry.score
        else:
            # Lifetime mean: new data counts less as n_episodes grows.
            # First episode: entry.score = score exactly.
            α = 1.0 / (entry.n_episodes + 1)
            entry.score = (1.0 - α) * entry.score + α * score

        entry.n_episodes += 1
        if not self.selection_staleness:
            # Explicit legacy diagnostic mode. Delayed feedback should not
            # otherwise make an old selection appear to have happened now.
            entry.staleness = 0
            for i, e in enumerate(self._entries):
                if i != level_id:
                    e.staleness += 1

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def is_warm(self) -> bool:
        """True once enough eligible levels have complete learner feedback."""
        count = sum(e.n_episodes > 0 for e in self._entries) if self.replay_requires_score else self.size
        return count >= self.min_fill_count

    def stats(self) -> dict:
        """Summary statistics for logging (e.g. to WandB)."""
        if not self._entries:
            return {"plr/size": 0}
        scores = np.array([e.score for e in self._entries])
        staleness = np.array([e.staleness for e in self._entries])
        n_episodes = np.array([e.n_episodes for e in self._entries])
        distribution = self.sampling_distribution()
        squared_mass = float(np.square(distribution).sum())
        return {
            "plr/size":            self.size,
            "plr/score_mean":      float(scores.mean()),
            "plr/score_max":       float(scores.max()),
            "plr/score_std":       float(scores.std()),
            "plr/staleness_mean":  float(staleness.mean()),
            "plr/staleness_max":   int(staleness.max()),
            "plr/episodes_mean":   float(n_episodes.mean()),
            "plr/levels_created": self._levels_created,
            "plr/levels_first_scored": self._levels_first_scored,
            "plr/unscored_evictions": self._unscored_evictions,
            "plr/scored_levels": int((n_episodes > 0).sum()),
            "plr/unscored_priority_mass": float(distribution[n_episodes == 0].sum()),
            "plr/replay_max_probability": float(distribution.max()),
            "plr/replay_effective_levels": 1.0 / squared_mass if squared_mass else 0.0,
            "plr/mutated_level_fraction": float(np.mean([e.origin == 'mutated' for e in self._entries])),
            "plr/mutated_priority_mass": float(sum(p for p, e in zip(distribution, self._entries)
                                                   if e.origin == 'mutated')),
            "plr/fresh_priority_mass": float(sum(p for p, e in zip(distribution, self._entries)
                                                 if e.origin == 'fresh')),
        }

    def top_k(self, k: int = 5) -> list[tuple[float, NSEnvConfig]]:
        """Return the top-k levels by score as [(score, config), ...]."""
        if not self._entries:
            return []
        sorted_entries = sorted(self._entries, key=lambda e: e.score, reverse=True)
        return [(e.score, e.config) for e in sorted_entries[:k]]

    def sampling_distribution(self) -> np.ndarray:
        """Eligible replay probabilities; all zeros until any score is ready."""
        return self._compute_distribution()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _should_explore(self) -> bool:
        # Always explore until min_fill is reached
        if not self.is_warm:
            return True
        # At capacity, exploration replaces an eligible old entry.
        if self.size >= self.capacity:
            return self.rng.random() > self.replay_prob
        # Mixed phase
        return self.rng.random() > self.replay_prob

    def _explore(self) -> tuple[int, NSEnvConfig]:
        eviction_candidates = [i for i, entry in enumerate(self._entries)
                               if not self.protect_unscored or entry.n_episodes > 0]
        if self.size >= self.capacity and (not eviction_candidates or
                (self.protect_unscored and self.replay_requires_score and not self.is_warm)):
            # Async workers can collect many episodes before the learner returns
            # their first scores. Keep these levels until they are evaluated;
            # zero here means unknown, not low learning potential.
            self.last_was_forced_replay = True
            # No slot can be replaced and no score is available yet. Uniform
            # re-evaluation keeps collectors moving until feedback arrives;
            # it is explicitly counted, not arbitrary ranked prioritization.
            self.last_was_replay = True
            self.last_was_mutation = False
            pending = [i for i, entry in enumerate(self._entries) if entry.n_episodes == 0]
            candidates = pending if pending else np.arange(self.size)
            idx = int(self.rng.choice(candidates))
            self.last_replay_probability = 1./len(candidates)
            self.last_replay_mutated_mass = float(np.mean([self._entries[i].origin == 'mutated'
                                                           for i in candidates]))
            return idx, self._entries[idx].config
        self.last_was_replay = False
        parent_uid, mutation_depth = None, 0
        # ACCEL: mutate a high-scoring config instead of random exploration
        if (
            self.mutation_prob > 0.0
            and self.is_warm
            and self.rng.random() < self.mutation_prob
        ):
            dist = self._compute_distribution()
            parent_idx = int(self.rng.choice(len(self._entries), p=dist))
            self.last_parent_probability = float(dist[parent_idx])
            parent = self._entries[parent_idx]
            parent_uid, mutation_depth = parent.uid, parent.mutation_depth + 1
            config = self.sampler.mutate(
                self._entries[parent_idx].config, self.rng, self.mutation_sigma
            )
            self.last_was_mutation = True
        else:
            config = self.sampler.sample()
            self.last_was_mutation = False

        if self.size >= self.capacity:
            # An unscored level must survive until its first learner feedback.
            if self.eviction_strategy == "random":
                idx = int(self.rng.choice(eviction_candidates))
            else:
                idx = min(eviction_candidates, key=lambda i: self._entries[i].score)
            self._unscored_evictions += int(self._entries[idx].n_episodes == 0)
            self._entries[idx] = _LevelEntry(config=config, origin='mutated' if self.last_was_mutation else 'fresh',
                                             uid=self._levels_created+1, parent_uid=parent_uid,
                                             mutation_depth=mutation_depth)
        else:
            idx = len(self._entries)
            self._entries.append(_LevelEntry(config=config, origin='mutated' if self.last_was_mutation else 'fresh',
                                             uid=self._levels_created+1, parent_uid=parent_uid,
                                             mutation_depth=mutation_depth))
        self._levels_created += 1
        return idx, config

    def _replay(self) -> tuple[int, NSEnvConfig]:
        self.last_was_replay = True
        self.last_was_mutation = False
        dist = self._compute_distribution()
        idx = int(self.rng.choice(len(self._entries), p=dist))
        self.last_replay_probability = float(dist[idx])
        self.last_replay_mutated_mass = float(sum(p for p, e in zip(dist, self._entries)
                                                  if e.origin == 'mutated'))
        return idx, self._entries[idx].config

    def _compute_distribution(self) -> np.ndarray:
        n = len(self._entries)
        if n == 0:
            return np.array([])
        eligible = np.array([not self.replay_requires_score or e.n_episodes > 0 for e in self._entries])
        indices = np.flatnonzero(eligible)
        result = np.zeros(n)
        if not len(indices):
            return result
        if self.uniform_replay:
            result[indices] = 1.0 / len(indices)
            return result

        scores = np.array([self._entries[i].score for i in indices])
        staleness = np.array([self._entries[i].staleness for i in indices], dtype=float)

        # Rank-based score distribution (Jiang et al. 2021).
        # rank 1 = highest score. Weight = (1/rank)^(1/temp), then normalise.
        # Scale-invariant: only ordering matters, not absolute score values.
        # Equal evidence must not confer a priority based on buffer position.
        ranks = np.empty(len(indices), dtype=float)
        order = np.argsort(-scores, kind="stable")
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and scores[order[end]] == scores[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        log_weights = -np.log(ranks) / self.score_temp
        score_dist = np.exp(log_weights - log_weights.max())
        score_dist /= score_dist.sum()

        # Staleness distribution: proportional to staleness count
        if staleness.sum() > 0:
            staleness_dist = staleness / staleness.sum()
        else:
            staleness_dist = np.ones(len(indices)) / len(indices)

        # Combine and renormalize
        dist = (1.0 - self.staleness_coef) * score_dist + self.staleness_coef * staleness_dist
        dist /= dist.sum()  # guard against floating-point drift
        result[indices] = dist
        return result


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
