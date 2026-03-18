"""
NSEnvSampler — continuously sample random NSEnvConfig objects from a parameter space.

Both the scheduler type, update function type, AND their numerical kwargs are
sampled uniformly from the eligible sets defined in each ParamSpec.

Global kwarg ranges live in SCHEDULER_SPACE and UPDATE_FN_SPACE.  Per-parameter
overrides narrow these where the physics demands it (e.g. OrnsteinUhlenbeck.mu
for gravity vs. torso_mass).

Requires ns-gym >= 1.0.10.  Changes from 1.0.8:
  - DecrementUpdate is now exported and works correctly.
  - RandomWalkWithDrift signature changed to (scheduler, alpha, mu, sigma) and
    now returns a scalar (the (1,) array bug is fixed).
  - New update fns: OrnsteinUhlenbeck, BoundedRandomWalk, LinearInterpolation,
    SigmoidTransition, CyclicUpdate, PolynomialTrend, UniformDrift (distribution).
  - New schedulers: BurstScheduler, DecayingProbabilityScheduler, WindowScheduler.

Excluded classes (not easily parameterised as float ranges):
  - WindowScheduler, DiscreteScheduler, CustomScheduler  — need explicit lists/fns
  - CyclicUpdate, PolynomialTrend                        — need explicit lists
  - SigmoidTransition                                     — start/end are env-specific
  - TargetReversion                                       — target dist is a list

Usage:
    sampler = NS_ENV_SAMPLERS["ant"](seed=42)
    config  = sampler.sample()    # → NSEnvConfig with random scheduler + update fn
    env     = sampler.make()      # → ready gym.Env
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym

from .ns_env_factory import NSEnvConfig, NSEnvFactory, ParamConfig, SchedulerConfig, UpdateFnConfig


# ---------------------------------------------------------------------------
# Global kwarg spaces — {cls_name: {kwarg_name: (min, max)}}
# Python ints in bounds → sampled as int (e.g. period, on_duration, T).
# ---------------------------------------------------------------------------

SCHEDULER_SPACE: dict[str, dict[str, tuple]] = {
    "ContinuousScheduler": {},
    "PeriodicScheduler":              {"period":               (20, 1000)},
    # kwarg is "probability", not "p"
    "RandomScheduler":                {"probability":          (0.01, 0.5)},
    "MemorylessScheduler":            {"p":                    (0.001, 0.1)},
    "BurstScheduler":                 {"on_duration":          (10, 200),
                                       "off_duration":         (100, 2000)},
    "DecayingProbabilityScheduler":   {"initial_probability":  (0.1, 0.9),
                                       "decay_rate":           (1e-5, 1e-3)},
}

# Ranges are calibrated so the total parameter change over a 1000-step episode
# is bounded even with ContinuousScheduler (worst case: fires every step).
#
#   IncrementUpdate/Decrement: k × 1000 ≤ 5 units  → k ≤ 0.005
#   DeterministicTrend:  slope × 1000 ≤ 0.5 units  → slope ≤ 5e-4
#   RandomWalk:     σ × √1000 ≤ 1.6 units std dev  → σ ≤ 0.05
#   RandomWalkWithDrift: same + drift ≤ 5 units     → alpha ≤ 0.005
#   GeometricProgression:   r^1000 ∈ [0.49, 2.04]  → r ∈ (0.9993, 1.0007)
#   OscillatingUpdate:  bounded by ≈ ±2δ always     → no change needed
#   Bounded/OU/Linear:  inherently range-safe        → no change needed
#   NoUpdate:           stationary baseline for PLR  → always valid
SCALAR_UPDATE_FN_SPACE: dict[str, dict[str, tuple]] = {
    "NoUpdate":            {},
    "IncrementUpdate":     {"k":          (0.0001, 0.005)},
    "DecrementUpdate":     {"k":          (0.0001, 0.005)},
    "ExponentialDecay":    {"decay_rate": (1e-6, 1e-4)},
    "DeterministicTrend":  {"slope":      (1e-6, 5e-4)},
    "RandomWalk":          {"sigma":      (0.001, 0.05)},
    # 1.0.10: signature is (scheduler, alpha, mu, sigma); all required; returns scalar
    "RandomWalkWithDrift": {"alpha":      (-0.005, 0.005),
                            "mu":         (-0.001, 0.001),
                            "sigma":      (0.001, 0.05)},
    "OscillatingUpdate":   {"delta":      (0.1, 2.0)},
    # r^1000 ∈ [0.49, 2.04]: meaningful decay/growth without blowup
    "GeometricProgression": {"r":         (0.9993, 1.0007)},
    # New in 1.0.10 ─────────────────────────────────────────────────────
    # OU process: reverts toward mu; per-param override of mu recommended
    "OrnsteinUhlenbeck":   {"theta":      (0.001, 0.3),
                            "mu":         (-5.0, 5.0),
                            "sigma":      (0.01, 0.5)},
    # Bounded RW: lo/hi ranges must not overlap — use per-param overrides
    "BoundedRandomWalk":   {"mu":         (-0.1, 0.1),
                            "sigma":      (0.01, 0.5),
                            "lo":         (-10.0, 0.0),
                            "hi":         (0.0, 10.0)},
    # Smooth linear ramp from start_val to end_val over T steps
    "LinearInterpolation": {"start_val":  (-10.0, 10.0),
                            "end_val":    (-10.0, 10.0),
                            "T":          (10000, 500000)},
}

DISTRIBUTION_UPDATE_FN_SPACE: dict[str, dict[str, tuple]] = {
    "DistributionDecrementUpdate": {"k":   (0.005, 0.05)},
    "DistributionIncrementUpdate": {"k":   (0.005, 0.05)},
    "UniformDrift":                {"rate": (0.001, 0.1)},
}

UPDATE_FN_SPACE: dict[str, dict[str, tuple]] = {
    **SCALAR_UPDATE_FN_SPACE,
    **DISTRIBUTION_UPDATE_FN_SPACE,
}


# ---------------------------------------------------------------------------
# ParamSpec
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """Continuous + structural parameter space for one NS-Gym tunable parameter.

    Args:
        param_name:          NS-Gym parameter key (e.g. "gravity", "torso_mass").
        schedulers:          Eligible scheduler class names (sampled uniformly).
        update_fns:          Eligible update function class names (sampled uniformly).
        scheduler_overrides: {cls: {kwarg: (min, max)}} — merged over global defaults.
        update_fn_overrides: {cls: {kwarg: (min, max)}} — merged over global defaults.
        required:            Always include this parameter (if False, include_prob applies).
        include_prob:        Probability of inclusion when required=False.
    """
    param_name: str
    schedulers: list[str]
    update_fns: list[str]
    scheduler_overrides: dict[str, dict[str, tuple]] = field(default_factory=dict)
    update_fn_overrides: dict[str, dict[str, tuple]] = field(default_factory=dict)
    required: bool = True
    include_prob: float = 0.7


# ---------------------------------------------------------------------------
# NSEnvSampler
# ---------------------------------------------------------------------------

class NSEnvSampler:
    """Randomly samples NSEnvConfig objects from a structured parameter space.

    Both the scheduler type and update function type are sampled uniformly from
    the eligible sets defined in each ParamSpec.  Kwarg values are then sampled
    uniformly within the merged global + override ranges.

    Args:
        env_id:         Gymnasium environment id.
        param_specs:    List of ParamSpec objects.
        gym_kwargs:     Forwarded to gym.make().
        wrapper_kwargs: Forwarded to the NS-Gym wrapper.
        seed:           Optional RNG seed.
    """

    def __init__(
        self,
        env_id: str,
        param_specs: list[ParamSpec],
        gym_kwargs: dict[str, Any] | None = None,
        wrapper_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
    ):
        self.env_id = env_id
        self.param_specs = param_specs
        self.gym_kwargs = gym_kwargs or {}
        self.wrapper_kwargs = wrapper_kwargs or {}
        self.rng = np.random.default_rng(seed)

    def _sample_kwargs(self, ranges: dict) -> dict:
        kwargs = {}
        for key, (lo, hi) in ranges.items():
            if isinstance(lo, int) and isinstance(hi, int):
                kwargs[key] = int(self.rng.integers(lo, hi + 1))
            else:
                kwargs[key] = float(self.rng.uniform(lo, hi))
        return kwargs

    def _sample_param_config(self, spec: ParamSpec) -> ParamConfig:
        sched_cls = str(self.rng.choice(spec.schedulers))
        sched_ranges = {
            **SCHEDULER_SPACE.get(sched_cls, {}),
            **spec.scheduler_overrides.get(sched_cls, {}),
        }

        ufn_cls = str(self.rng.choice(spec.update_fns))
        ufn_ranges = {
            **UPDATE_FN_SPACE.get(ufn_cls, {}),
            **spec.update_fn_overrides.get(ufn_cls, {}),
        }

        return ParamConfig(
            scheduler=SchedulerConfig(sched_cls, self._sample_kwargs(sched_ranges)),
            update_fn=UpdateFnConfig(ufn_cls, self._sample_kwargs(ufn_ranges)),
        )

    def sample(self) -> NSEnvConfig:
        """Sample a random NSEnvConfig. Guaranteed to include at least one param."""
        tunable_params: dict[str, ParamConfig] = {}

        for spec in self.param_specs:
            if not spec.required and self.rng.random() > spec.include_prob:
                continue
            tunable_params[spec.param_name] = self._sample_param_config(spec)

        if not tunable_params:
            return self.sample()

        return NSEnvConfig(
            env_id=self.env_id,
            tunable_params=tunable_params,
            gym_kwargs=self.gym_kwargs,
            wrapper_kwargs=self.wrapper_kwargs,
        )

    def make(self, config: NSEnvConfig | None = None) -> gym.Env:
        """Instantiate an environment from a config (or sample a fresh one)."""
        return NSEnvFactory.make(config or self.sample())


# ---------------------------------------------------------------------------
# Pre-built ParamSpec lists
# ---------------------------------------------------------------------------

_ALL_SCHEDULERS = [
    "ContinuousScheduler", "PeriodicScheduler", "RandomScheduler",
    "MemorylessScheduler", "BurstScheduler", "DecayingProbabilityScheduler",
]

# --- Ant-v5 ----------------------------------------------------------------
# AntEnv exposes: "gravity" (3D vector, z-component updated as scalar by the
# wrapper) and "torso_mass".  No other params in v1.0.10.

ANT_PARAM_SPECS: list[ParamSpec] = [
    ParamSpec(
        param_name="gravity",
        schedulers=_ALL_SCHEDULERS,
        update_fns=[
            "NoUpdate",
            "IncrementUpdate", "DecrementUpdate", "RandomWalk",
            "RandomWalkWithDrift", "DeterministicTrend", "OscillatingUpdate",
            "OrnsteinUhlenbeck", "BoundedRandomWalk", "LinearInterpolation",
        ],
        update_fn_overrides={
            # Gravity z-component is ~-9.81; equilibrium should stay negative
            "OrnsteinUhlenbeck":  {"theta": (0.001, 0.1),
                                   "mu":    (-15.0, -3.0),
                                   "sigma": (0.01, 0.5)},
            # lo < hi guaranteed: lo ∈ (-20,-12), hi ∈ (-5,-1)
            "BoundedRandomWalk":  {"mu":    (0.0, 0.0),
                                   "sigma": (0.1, 1.0),
                                   "lo":    (-20.0, -12.0),
                                   "hi":    (-5.0, -1.0)},
            # Ramp gravity from near-default to weaker or stronger
            "LinearInterpolation": {"start_val": (-12.0, -8.0),
                                    "end_val":   (-18.0, -2.0),
                                    "T":         (10000, 500000)},
        },
        required=True,
    ),
    ParamSpec(
        param_name="torso_mass",
        schedulers=_ALL_SCHEDULERS,
        update_fns=[
            "NoUpdate",
            "IncrementUpdate", "DecrementUpdate", "ExponentialDecay",
            "GeometricProgression", "RandomWalk",
            "OrnsteinUhlenbeck", "BoundedRandomWalk", "LinearInterpolation",
        ],
        update_fn_overrides={
            # Mass must stay positive; keep decrements small (already tighter than global)
            "DecrementUpdate":    {"k":     (0.00001, 0.001)},
            "ExponentialDecay":   {"decay_rate": (1e-6, 5e-5)},
            # OU equilibrium around biologically plausible torso mass
            "OrnsteinUhlenbeck":  {"theta": (0.001, 0.1),
                                   "mu":    (0.1, 2.0),
                                   "sigma": (0.001, 0.1)},
            # lo < hi guaranteed: lo ∈ (0.01, 0.1), hi ∈ (0.5, 5.0)
            "BoundedRandomWalk":  {"mu":    (0.0, 0.0),
                                   "sigma": (0.01, 0.1),
                                   "lo":    (0.01, 0.1),
                                   "hi":    (0.5, 5.0)},
            "LinearInterpolation": {"start_val": (0.2, 1.0),
                                    "end_val":   (0.05, 3.0),
                                    "T":         (10000, 500000)},
        },
        required=False,
        include_prob=0.6,
    ),
]

# --- CartPole-v1 ------------------------------------------------------------

CARTPOLE_PARAM_SPECS: list[ParamSpec] = [
    ParamSpec(
        param_name="masspole",
        schedulers=_ALL_SCHEDULERS,
        update_fns=[
            "NoUpdate",
            "IncrementUpdate", "DecrementUpdate", "ExponentialDecay",
            "GeometricProgression", "RandomWalk",
            "OrnsteinUhlenbeck", "BoundedRandomWalk",
        ],
        update_fn_overrides={
            # Pole mass starts at 0.1 kg; keep mass positive
            "DecrementUpdate":   {"k":     (0.00001, 0.001)},
            "ExponentialDecay":  {"decay_rate": (1e-6, 1e-4)},
            "OrnsteinUhlenbeck": {"theta": (0.001, 0.1),
                                  "mu":    (0.05, 1.0),
                                  "sigma": (0.001, 0.05)},
            # lo < hi guaranteed: lo ∈ (0.01, 0.05), hi ∈ (0.3, 2.0)
            "BoundedRandomWalk": {"mu":    (0.0, 0.0),
                                  "sigma": (0.005, 0.05),
                                  "lo":    (0.01, 0.05),
                                  "hi":    (0.3, 2.0)},
        },
        required=True,
    ),
    ParamSpec(
        param_name="gravity",
        schedulers=_ALL_SCHEDULERS,
        update_fns=[
            "NoUpdate",
            "IncrementUpdate", "DecrementUpdate", "RandomWalk",
            "RandomWalkWithDrift", "DeterministicTrend", "OscillatingUpdate",
            "OrnsteinUhlenbeck", "BoundedRandomWalk",
        ],
        update_fn_overrides={
            "OrnsteinUhlenbeck": {"theta": (0.001, 0.1),
                                  "mu":    (5.0, 15.0),
                                  "sigma": (0.01, 0.5)},
            # CartPole gravity is positive (default 9.8); lo < hi guaranteed
            "BoundedRandomWalk": {"mu":    (0.0, 0.0),
                                  "sigma": (0.05, 0.5),
                                  "lo":    (1.0, 5.0),
                                  "hi":    (15.0, 25.0)},
        },
        required=False,
        include_prob=0.6,
    ),
]

# --- FrozenLake-v1 ----------------------------------------------------------

FROZENLAKE_PARAM_SPECS: list[ParamSpec] = [
    ParamSpec(
        param_name="P",
        schedulers=_ALL_SCHEDULERS,
        update_fns=[
            "DistributionDecrementUpdate",
            "DistributionIncrementUpdate",
            "UniformDrift",
        ],
        required=True,
    ),
]


# ---------------------------------------------------------------------------
# Named sampler catalogue
# ---------------------------------------------------------------------------

NS_ENV_SAMPLERS: dict[str, callable] = {
    "ant": lambda seed=None: NSEnvSampler(
        env_id="Ant-v5",
        param_specs=ANT_PARAM_SPECS,
        seed=seed,
    ),
    "cartpole": lambda seed=None: NSEnvSampler(
        env_id="CartPole-v1",
        param_specs=CARTPOLE_PARAM_SPECS,
        seed=seed,
    ),
    "frozenlake": lambda seed=None: NSEnvSampler(
        env_id="FrozenLake-v1",
        param_specs=FROZENLAKE_PARAM_SPECS,
        gym_kwargs={"disable_env_checker": True},
        wrapper_kwargs={"initial_prob_dist": [1, 0, 0]},
        seed=seed,
    ),
}
