"""Exact policy expectations for policy-independent FrozenLake dynamics."""

import numpy as np


def exact_returns(policies, timelines, next_states, rewards, terminated,
                  gammas=(1.0, .99), greedy=False):
    """Return [policy, dynamics trajectory, discount] expectations.

    Timelines have shape [trajectory, time, outcome], in the wrapper's
    [action, action+1, action-1] order. The reward on the first step is
    undiscounted. Recursion stops at holes/goals and at the declared horizon.
    Each timeline retains its own dynamics throughout the episode; averaging
    transition probabilities first would describe a different environment.
    """
    policies, timelines = np.asarray(policies), np.asarray(timelines)
    gammas = np.asarray(gammas)
    n_policies, n_states, _ = policies.shape
    n_trajectories, horizon, _ = timelines.shape
    values = np.zeros((n_policies, n_trajectories, n_states, len(gammas)))
    if greedy:
        states = np.arange(n_states)[None, :]
        actions = policies.argmax(-1)
        next_states = next_states[states, actions]
        rewards = rewards[states, actions]
        terminated = terminated[states, actions]
    for t in range(horizon - 1, -1, -1):
        p = timelines[:, t, :]
        if greedy:
            continuation = values[np.arange(n_policies)[:, None, None, None],
                                  np.arange(n_trajectories)[None, :, None, None],
                                  next_states[:, None, :, :], :]
            q = rewards[:, None, :, :, None] + gammas * continuation * ~terminated[:, None, :, :, None]
            values = (p[None, :, None, :, None] * q).sum(3)
        else:
            continuation = values[:, :, next_states, :]
            q = (p[None, :, None, None, :, None] * (
                rewards[None, None, :, :, :, None] + gammas * continuation *
                ~terminated[None, None, :, :, :, None])).sum(4)
            values = (q * policies[:, None, :, :, None]).sum(3)
    return values[:, :, 0, :]
