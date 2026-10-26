import pytest
import torch

from AAMAS_Comp.agents.policy_checks import nonnegative_sample_kl, guarded_optimizer_step


def test_signed_estimate_cancellation_does_not_hide_large_ratios():
    old = torch.zeros(2)
    new = torch.tensor([-2., 2.])
    assert (old - new).mean() == 0
    assert nonnegative_sample_kl(new, old) > 2


def test_identical_policies_have_zero_kl_and_tiny_changes_are_finite():
    old = torch.zeros(4, dtype=torch.float64)
    assert nonnegative_sample_kl(old, old) == 0
    assert 0 < nonnegative_sample_kl(old + 1e-8, old) < 1e-15


def test_exact_discrete_expectation_matches_kl():
    # Uniform old policy: each action appears once, so the mean is exact.
    old = torch.full((4,), .25, dtype=torch.float64).log()
    new = torch.tensor([.1, .2, .3, .4], dtype=torch.float64).log()
    assert nonnegative_sample_kl(new, old) == pytest.approx((old - new).mean().item())


def test_mismatched_shapes_are_rejected_instead_of_broadcasting():
    with pytest.raises(ValueError, match='shapes'):
        nonnegative_sample_kl(torch.zeros(2), torch.zeros(2, 1))


def test_guard_rejects_ratios_that_overflow_the_ppo_objective():
    old = torch.tensor([-1000., -10000.])
    current = torch.zeros(2)
    assert torch.isinf((current - old).exp()).all()
    kl = nonnegative_sample_kl(current, old)
    assert torch.isfinite(kl) and kl > 1.5 * .02


def test_post_update_guard_backtracks_an_oversized_adam_step():
    # A narrow Gaussian makes even a small Adam mean update exceed target KL.
    mean = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([mean], lr=.1)
    mean.grad = torch.tensor(-1.)
    result = guarded_optimizer_step(optimizer, [mean],
                                    lambda: .5 * (mean / .01).square(), .03)
    assert result.accepted and result.attempts > 1 and result.lr_scale < 1
    assert float(.5 * (mean.detach()/.01).square()) <= .03
    assert optimizer.param_groups[0]['lr'] == .1
    assert optimizer.state[mean]['step'] == 1  # retries must not accumulate Adam steps


def test_rejected_adam_update_restores_parameters_and_existing_moments():
    from copy import deepcopy
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([parameter], lr=.1)
    parameter.grad = torch.tensor(1.)
    optimizer.step()
    old_parameter = parameter.detach().clone()
    old_state = deepcopy(optimizer.state_dict())
    parameter.grad = torch.tensor(-2.)
    result = guarded_optimizer_step(optimizer, [parameter], lambda: float('inf'),
                                    .03, max_backtracks=2)
    assert not result.accepted and result.attempts == 3
    assert torch.equal(parameter, old_parameter)
    assert optimizer.state_dict()['param_groups'] == old_state['param_groups']
    for key, value in old_state['state'][0].items():
        assert torch.equal(optimizer.state[parameter][key], value)


def test_post_update_check_exception_restores_candidate():
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([parameter], lr=.1)
    parameter.grad = torch.tensor(1.)

    def invalid_measurement():
        raise RuntimeError('failed diagnostic')

    with pytest.raises(RuntimeError, match='diagnostic'):
        guarded_optimizer_step(optimizer, [parameter], invalid_measurement, .03)
    assert parameter.item() == 1.
    assert not optimizer.state and optimizer.param_groups[0]['lr'] == .1
