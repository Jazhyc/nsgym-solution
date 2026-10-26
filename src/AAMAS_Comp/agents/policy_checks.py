"""Numerically stable policy-change diagnostics for PPO minibatch guards."""

from copy import deepcopy
from dataclasses import dataclass

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


@dataclass(frozen=True)
class GuardedStep:
    accepted: bool
    attempts: int
    lr_scale: float
    kl: float


def guarded_optimizer_step(optimizer, parameters, measure_kl, max_kl, *, max_backtracks=12):
    """Commit an update only if its measured policy divergence is acceptable.

    Retry the same clipped gradients with half the step size after a rejected
    candidate. Parameters AND optimizer moments/step counters roll back before
    every retry. Keep the scheduled learning rate unchanged: the scale applies
    only to this update. On failure or an exception, restore the original state.
    The caller chooses the observations/actions on which divergence is measured.
    """
    if max_kl <= 0 or max_backtracks < 0:
        raise ValueError('Invalid post-update KL guard limits')
    parameters = list(parameters)
    saved_parameters = [p.detach().clone() for p in parameters]
    saved_optimizer = deepcopy(optimizer.state_dict())
    learning_rates = [group['lr'] for group in optimizer.param_groups]

    def restore():
        with torch.no_grad():
            for parameter, saved in zip(parameters, saved_parameters):
                parameter.copy_(saved)
        optimizer.load_state_dict(deepcopy(saved_optimizer))

    kl = float('inf')
    try:
        for attempt in range(max_backtracks + 1):
            if attempt:
                restore()
            scale = 0.5 ** attempt
            for group, lr in zip(optimizer.param_groups, learning_rates):
                group['lr'] = lr * scale
            optimizer.step()
            with torch.no_grad():
                kl = float(measure_kl())
            if 0 <= kl <= max_kl:
                for group, lr in zip(optimizer.param_groups, learning_rates):
                    group['lr'] = lr
                return GuardedStep(True, attempt + 1, scale, kl)
    except BaseException:
        restore()
        raise
    restore()
    return GuardedStep(False, max_backtracks + 1, 0., kl)
