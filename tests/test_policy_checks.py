import pytest
import torch

from AAMAS_Comp.agents.policy_checks import nonnegative_sample_kl


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
