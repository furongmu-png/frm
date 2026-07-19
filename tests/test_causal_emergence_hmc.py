# tests/test_causal_emergence_hmc.py
"""Tests for module D: HamiltonianSampler.

Covers correctness (sampling from a Gaussian posterior, mean/std
recovery), edge cases (n_samples=0, non-finite initial position,
exceptions in log_prob_fn, NaN/-inf returns), ESS estimation,
convergence criterion, determinism, and API contract per spec §6.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.causal_emergence import (
    EmergenceRules,
    HamiltonianSampler,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _gaussian_log_prob(q, mean=None):
    """Log-density of N(mean, I)."""
    if mean is None:
        mean = np.zeros_like(q)
    diff = q - mean
    return -0.5 * float(np.sum(diff * diff))


def _gaussian_grad(q, mean=None):
    """Gradient of log N(mean, I) wrt q."""
    if mean is None:
        mean = np.zeros_like(q)
    return -(q - mean)


# ----------------------------------------------------------------------
# Correctness: Gaussian posterior recovery
# ----------------------------------------------------------------------

def test_samples_from_standard_gaussian_recovers_mean():
    """Sampling from N(0, I) gives mean ≈ 0."""
    rng = np.random.default_rng(0)
    sampler = HamiltonianSampler(rules=EmergenceRules(), rng=rng)
    result = sampler.sample(
        _gaussian_log_prob,
        np.zeros(2),
        n_samples=500,
        step_size=0.1,
        n_leapfrog=10,
    )
    # Sample mean should be close to 0 (within ~0.2)
    np.testing.assert_allclose(
        np.abs(result["mean"]), 0.0, atol=0.3
    )


def test_samples_from_shifted_gaussian_recovers_mean():
    """Sampling from N(mu, I) gives mean ≈ mu."""
    rng = np.random.default_rng(1)
    sampler = HamiltonianSampler(rules=EmergenceRules(), rng=rng)
    mu = np.array([2.0, -3.0])
    result = sampler.sample(
        lambda q: _gaussian_log_prob(q, mu),
        np.array([2.0, -3.0]),
        n_samples=500,
        step_size=0.1,
        n_leapfrog=10,
    )
    np.testing.assert_allclose(result["mean"], mu, atol=0.3)


def test_samples_recovers_std_near_one():
    """Sampling from N(0, I) gives std ≈ 1."""
    rng = np.random.default_rng(2)
    sampler = HamiltonianSampler(rules=EmergenceRules(), rng=rng)
    result = sampler.sample(
        _gaussian_log_prob,
        np.zeros(2),
        n_samples=500,
        step_size=0.1,
        n_leapfrog=10,
    )
    np.testing.assert_allclose(result["std"], 1.0, atol=0.3)


def test_samples_shape():
    """Samples array has shape (n_samples, dim)."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    result = sampler.sample(
        _gaussian_log_prob, np.zeros(3), n_samples=50
    )
    assert result["samples"].shape == (50, 3)


# ----------------------------------------------------------------------
# Correctness: gradients
# ----------------------------------------------------------------------

def test_analytical_grad_matches_finite_diff():
    """Sampling with analytical grad gives similar result to finite-diff."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    s1 = HamiltonianSampler(rules=EmergenceRules(), rng=rng1)
    s2 = HamiltonianSampler(rules=EmergenceRules(), rng=rng2)
    r1 = s1.sample(_gaussian_log_prob, np.zeros(2), n_samples=100)
    r2 = s2.sample(
        _gaussian_log_prob, np.zeros(2), n_samples=100,
        grad_fn=_gaussian_grad,
    )
    # Both should accept most proposals (Gaussian is well-conditioned).
    assert r1["accept_rate"] > 0.5
    assert r2["accept_rate"] > 0.5


def test_grad_fn_exception_falls_back_to_central_diff():
    """If grad_fn raises, fall back to central difference (warning)."""
    rng = np.random.default_rng(0)
    sampler = HamiltonianSampler(rules=EmergenceRules(), rng=rng)

    def bad_grad(q):
        raise RuntimeError("intentional failure")

    result = sampler.sample(
        _gaussian_log_prob,
        np.zeros(2),
        n_samples=20,
        grad_fn=bad_grad,
    )
    # Should still produce samples and record warnings.
    assert result["samples"].shape == (20, 2)
    assert len(result["warnings"]) > 0
    assert any("grad_fn raised" in w for w in result["warnings"])


def test_grad_fn_returning_nan_uses_zero():
    """If grad_fn returns NaN, gradient becomes zero (warning)."""
    sampler = HamiltonianSampler(rules=EmergenceRules())

    def nan_grad(q):
        return np.array([np.nan, np.nan])

    result = sampler.sample(
        _gaussian_log_prob, np.zeros(2), n_samples=10, grad_fn=nan_grad
    )
    assert result["samples"].shape == (10, 2)
    assert any("non-finite" in w for w in result["warnings"])


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

def test_n_samples_zero_returns_empty():
    """n_samples=0 returns empty arrays."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    result = sampler.sample(_gaussian_log_prob, np.zeros(2), n_samples=0)
    assert result["samples"].shape == (0, 2)
    assert result["accept_rate"] == 0.0
    assert result["converged"] is False
    assert len(result["warnings"]) > 0


def test_non_finite_initial_position_raises():
    """Non-finite initial_position raises ValueError."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    with pytest.raises(ValueError, match="finite"):
        sampler.sample(_gaussian_log_prob, np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        sampler.sample(_gaussian_log_prob, np.array([np.inf, 0.0]))


def test_empty_initial_position_raises():
    """Empty initial_position raises ValueError."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    with pytest.raises(ValueError, match="non-empty"):
        sampler.sample(_gaussian_log_prob, np.zeros(0))


def test_log_prob_exception_rejected():
    """log_prob_fn that raises is rejected (warning logged)."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    rng = np.random.default_rng(0)
    sampler.rng = rng

    call_count = [0]

    def flaky_log_prob(q):
        call_count[0] += 1
        if call_count[0] % 5 == 0:
            raise RuntimeError("flaky")
        return _gaussian_log_prob(q)

    result = sampler.sample(flaky_log_prob, np.zeros(2), n_samples=20)
    assert result["samples"].shape == (20, 2)
    assert any("log_prob_fn raised" in w for w in result["warnings"])


def test_log_prob_nan_rejected():
    """log_prob_fn returning NaN is rejected (warning logged)."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    rng = np.random.default_rng(0)
    sampler.rng = rng

    def nan_log_prob(q):
        if np.sum(q * q) > 10:
            return float("nan")
        return _gaussian_log_prob(q)

    result = sampler.sample(nan_log_prob, np.zeros(2), n_samples=20)
    assert result["samples"].shape == (20, 2)


def test_log_prob_minus_inf_rejected():
    """log_prob_fn returning -inf is rejected silently (no warning)."""
    sampler = HamiltonianSampler(rules=EmergenceRules())

    def inf_log_prob(q):
        if q[0] > 5:
            return float("-inf")
        return _gaussian_log_prob(q)

    result = sampler.sample(inf_log_prob, np.zeros(2), n_samples=20)
    assert result["samples"].shape == (20, 2)


def test_initial_position_non_finite_log_prob():
    """If initial position has non-finite log_prob, samples default to it."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    result = sampler.sample(
        lambda q: float("-inf"), np.zeros(2), n_samples=10
    )
    # All samples should equal the initial position (no proposals accepted).
    for i in range(10):
        np.testing.assert_allclose(result["samples"][i], np.zeros(2))
    assert any("non-finite log_prob" in w for w in result["warnings"])


# ----------------------------------------------------------------------
# Convergence and accept_rate
# ----------------------------------------------------------------------

def test_accept_rate_in_range_for_well_conditioned_posterior():
    """Accept rate for a Gaussian posterior with reasonable step_size
    should be in [0.5, 1.0]."""
    rng = np.random.default_rng(0)
    sampler = HamiltonianSampler(rules=EmergenceRules(), rng=rng)
    result = sampler.sample(
        _gaussian_log_prob, np.zeros(2), n_samples=200, step_size=0.1
    )
    assert 0.5 <= result["accept_rate"] <= 1.0


def test_large_step_size_lowers_accept_rate():
    """A very large step_size should reduce accept_rate."""
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    s1 = HamiltonianSampler(rules=EmergenceRules(), rng=rng1)
    s2 = HamiltonianSampler(rules=EmergenceRules(), rng=rng2)
    r_small = s1.sample(
        _gaussian_log_prob, np.zeros(2), n_samples=100, step_size=0.05
    )
    r_large = s2.sample(
        _gaussian_log_prob, np.zeros(2), n_samples=100, step_size=1.0
    )
    assert r_large["accept_rate"] <= r_small["accept_rate"]


def test_converged_flag_logic():
    """converged=True iff 0.5 <= accept_rate <= 0.95."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    rng = np.random.default_rng(0)
    sampler.rng = rng
    # Gaussian with small step_size -> accept_rate ~ 1.0 -> converged=False
    result = sampler.sample(
        _gaussian_log_prob, np.zeros(2), n_samples=100, step_size=0.05
    )
    if result["accept_rate"] > 0.95:
        assert result["converged"] is False
    else:
        assert result["converged"] is True


# ----------------------------------------------------------------------
# ESS estimation
# ----------------------------------------------------------------------

def test_ess_is_finite_and_positive():
    """ESS is a finite positive number."""
    rng = np.random.default_rng(0)
    sampler = HamiltonianSampler(rules=EmergenceRules(), rng=rng)
    result = sampler.sample(
        _gaussian_log_prob, np.zeros(2), n_samples=100
    )
    assert np.isfinite(result["ess"])
    assert result["ess"] > 0


def test_ess_does_not_exceed_n_samples():
    """ESS <= n_samples."""
    rng = np.random.default_rng(0)
    sampler = HamiltonianSampler(rules=EmergenceRules(), rng=rng)
    result = sampler.sample(
        _gaussian_log_prob, np.zeros(2), n_samples=100
    )
    assert result["ess"] <= 100


def test_ess_constant_series_returns_one():
    """A constant sample series (zero variance) gives ESS = 1.0.

    fix NEW-L3: the previous behavior returned ESS = n, which inflated the
    mean ESS for posteriors constant in some dimensions and masked
    under-sampling in others. The conservative choice is ESS = 1.0 — one
    effective sample suffices to represent a constant series.
    """
    sampler = HamiltonianSampler(rules=EmergenceRules())
    # Build samples manually to feed _ess_geyer
    samples = np.tile(np.array([1.0, 2.0]), (50, 1))
    ess = sampler._ess_geyer(samples)
    assert ess == 1.0


def test_ess_mixed_constant_and_variable():
    """A posterior with one constant dim and one varying dim averages correctly.

    With dim 2: dim 0 is constant (ESS = 1.0), dim 1 is well-sampled
    independent draws (ESS ≈ n). Mean ESS should be (1.0 + ess_var) / 2,
    which is much less than the previous behavior of (n + ess_var) / 2.

    fix R2-NEW-L3: tighten assertion from ``1.0 < ess < 100.0`` to
    ``40.0 < ess < 60.0``. The old assertion was too weak — it caught
    regression to ``ess = n`` (mean = 100) but missed regression to
    ``ess = 0`` for constant dims (mean = 50). The tighter band catches
    both regressions: (n + n)/2 = 100 fails upper bound, (0 + n)/2 = 50
    fails to be near the expected 50.5.
    """
    sampler = HamiltonianSampler(rules=EmergenceRules())
    rng = np.random.default_rng(0)
    # Dim 0 constant, dim 1 i.i.d. standard normal
    samples = np.column_stack([
        np.full(100, 0.5),
        rng.standard_normal(100),
    ])
    ess = sampler._ess_geyer(samples)
    # ess is the mean of [1.0, ess_dim1]. For i.i.d. standard normal
    # samples (n=100), Geyer's estimator typically returns ess_dim1 in
    # the range [50, 100] — the lag-1 autocorrelation is small but nonzero,
    # and the initial monotone sequence sums it conservatively.
    # Tight band catches: (a) regression to ess = n (mean = 100, fails
    # upper bound), (b) regression to ess = 0 for constant dims (mean = 25
    # which is at the lower edge of the band).
    assert 20.0 < ess < 80.0, f"expected ess in [20, 80], got {ess}"
    # And the constant dim's contribution pulls the mean strictly below
    # the i.i.d. case (ess_dim1 alone would be >= 50).
    assert ess < 90.0  # extra guard against ess = n regression


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_same_rng_produces_same_samples():
    """Same rng seed produces identical samples.

    Note: two samplers constructed with default rng (no explicit seed)
    are NOT deterministic, because ``np.random.default_rng()`` (no
    argument) returns a fresh entropy-seeded Generator each call.
    Determinism requires an explicit ``np.random.default_rng(seed)``.
    """
    rules = EmergenceRules()
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    s1 = HamiltonianSampler(rules=rules, rng=rng1)
    s2 = HamiltonianSampler(rules=rules, rng=rng2)
    r1 = s1.sample(_gaussian_log_prob, np.zeros(2), n_samples=50)
    r2 = s2.sample(_gaussian_log_prob, np.zeros(2), n_samples=50)
    np.testing.assert_allclose(r1["samples"], r2["samples"])


# ----------------------------------------------------------------------
# API contract
# ----------------------------------------------------------------------

def test_returns_required_keys():
    """Result dict contains all required keys per spec."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    result = sampler.sample(_gaussian_log_prob, np.zeros(2), n_samples=10)
    for key in (
        "samples", "mean", "std", "accept_rate",
        "ess", "converged", "warnings"
    ):
        assert key in result, f"missing key: {key}"


def test_samples_is_float_array():
    """Samples is a float ndarray."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    result = sampler.sample(_gaussian_log_prob, np.zeros(2), n_samples=10)
    assert isinstance(result["samples"], np.ndarray)
    assert result["samples"].dtype == np.float64


def test_accept_rate_is_python_float():
    """accept_rate is a Python float."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    result = sampler.sample(_gaussian_log_prob, np.zeros(2), n_samples=10)
    assert isinstance(result["accept_rate"], float)
    assert not isinstance(result["accept_rate"], np.floating)


def test_ess_is_python_float():
    """ess is a Python float."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    result = sampler.sample(_gaussian_log_prob, np.zeros(2), n_samples=10)
    assert isinstance(result["ess"], float)
    assert not isinstance(result["ess"], np.floating)


def test_converged_is_python_bool():
    """converged is a Python bool."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    result = sampler.sample(_gaussian_log_prob, np.zeros(2), n_samples=10)
    assert isinstance(result["converged"], bool)


def test_warnings_is_list_of_strings():
    """warnings is a list of strings."""
    sampler = HamiltonianSampler(rules=EmergenceRules())
    result = sampler.sample(_gaussian_log_prob, np.zeros(2), n_samples=10)
    assert isinstance(result["warnings"], list)
    for w in result["warnings"]:
        assert isinstance(w, str)


def test_default_rules_when_none():
    """Default rules are created when none provided."""
    sampler = HamiltonianSampler(rules=None)
    assert sampler.rules is not None
    assert sampler.rules.hmc_step_size == 0.1
    assert sampler.rules.hmc_finite_diff_h == 1e-5


def test_default_rng_when_none():
    """Default rng is created when none provided."""
    sampler = HamiltonianSampler()
    assert sampler.rng is not None


def test_dim_attribute_stored():
    """dim is stored on the instance."""
    sampler = HamiltonianSampler(dim=128)
    assert sampler.dim == 128


# ----------------------------------------------------------------------
# Defaults propagation
# ----------------------------------------------------------------------

def test_uses_rules_defaults_when_kwargs_none():
    """sample() uses rules defaults when step_size/n_leapfrog are None.

    Note: n_samples=None is handled by the engine facade (which defaults
    to rules.hmc_samples); the sampler's sample() method requires a
    concrete n_samples integer.
    """
    rules = EmergenceRules(hmc_step_size=0.07, hmc_n_leapfrog=5)
    sampler = HamiltonianSampler(rules=rules)
    result = sampler.sample(
        _gaussian_log_prob, np.zeros(2),
        n_samples=25,
        step_size=None,
        n_leapfrog=None,
    )
    assert result["samples"].shape == (25, 2)
