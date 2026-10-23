"""Numerically stable policy-change diagnostics for PPO minibatch guards."""

import torch


def nonnegative_sample_kl(current_log_prob, behavior_log_prob):
    """Estimate KL(behavior || current) without cancellation across samples.

    For log ratio d, exp(d)-1-d is nonnegative and has the same expectation
    as -d under the behavior distribution. expm1 is accurate near zero.
    Both inputs must contain joint-action log probabilities with equal shape.
    """
    if current_log_prob.shape != behavior_log_prob.shape:
        raise ValueError('Policy log probability shapes differ')
    delta = current_log_prob.double() - behavior_log_prob.double()
    # A float32 importance ratio can overflow before the guard rejects it.
    # Saturation only affects astronomically large divergences, which must
    # already stop PPO; the diagnostic itself stays finite for logging.
    return (torch.expm1(delta.clamp_max(700)) - delta).clamp_min(0).mean()
