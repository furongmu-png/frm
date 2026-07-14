# tests/test_capabilities_analytics_advanced.py
"""Tests for the advanced zero-data Data Analytics capability module.

The advanced analytics capabilities (causal inference, Bayesian online updating,
change-point detection) compose the existing active-inference engine and the
statistical rule library with closed-form, learning-free operators. No external
statistical libraries (no statsmodels / pyro) and no learned weights are
required.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.capabilities.analytics_advanced import (
    BayesianEstimator,
    CausalInference,
    ChangePointDetector,
)


@pytest.fixture(autouse=True)
def _deterministic_rng() -> None:
    """Pin the global RNG so the active-inference engine's random transition /
    emission matrices are reproducible across test runs."""
    np.random.seed(42)


# ---------------------------------------------------------------------------
# CausalInference
# ---------------------------------------------------------------------------


def test_causal_inference_keys_and_types():
    """infer_cause returns the three required keys with correct types."""
    ci = CausalInference(dim=64)
    rng = np.random.default_rng(7)
    cause = rng.standard_normal(40)
    # Effect is a delayed copy of the cause (a clear causal signal).
    effect = np.concatenate([np.zeros(3), cause[:-3]]) + 0.01 * rng.standard_normal(40)
    out = ci.infer_cause(cause, effect, max_lag=5)
    assert set(out.keys()) == {"best_lag", "causal_strength", "p_value_approx"}
    assert isinstance(out["best_lag"], int)
    assert isinstance(out["causal_strength"], float)
    assert isinstance(out["p_value_approx"], float)
    assert 1 <= out["best_lag"] <= 5
    assert -1.0 <= out["causal_strength"] <= 1.0
    assert 0.0 <= out["p_value_approx"] <= 1.0


def test_causal_inference_best_lag_in_range():
    """best_lag stays within 1..max_lag for any reasonable input."""
    ci = CausalInference(dim=64)
    rng = np.random.default_rng(11)
    cause = rng.standard_normal(30)
    effect = rng.standard_normal(30)
    out = ci.infer_cause(cause, effect, max_lag=3)
    assert 1 <= out["best_lag"] <= 3


def test_causal_inference_detects_strong_signal():
    """A clearly lagged copy of the cause yields a high |causal_strength|."""
    ci = CausalInference(dim=64)
    rng = np.random.default_rng(123)
    cause = rng.standard_normal(50)
    # Effect = cause shifted by 2 lags (with tiny noise so correlation is strong).
    effect = np.concatenate([np.zeros(2), cause[:-2]])
    out = ci.infer_cause(cause, effect, max_lag=5)
    assert out["best_lag"] == 2
    assert abs(out["causal_strength"]) > 0.5
    # Strong correlation with a large sample -> small p-value.
    assert out["p_value_approx"] < 0.05


def test_causal_inference_short_series_returns_safe_default():
    """Very short inputs return best_lag=0 / strength=0 / p_value=1."""
    ci = CausalInference(dim=64)
    out = ci.infer_cause(np.array([1.0, 2.0]), np.array([2.0, 1.0]))
    assert out["best_lag"] == 0
    assert out["causal_strength"] == 0.0
    assert out["p_value_approx"] == 1.0


# ---------------------------------------------------------------------------
# BayesianEstimator
# ---------------------------------------------------------------------------


def test_bayesian_posterior_mean_after_normal_updates():
    """Posterior mean shrinks toward the observations after several updates."""
    be = BayesianEstimator(dim=64)
    be.reset(prior_mean=0.0, prior_var=1.0)
    for obs in (1.0, 1.1, 0.9, 1.0, 1.05):
        be.update(obs)
    mean = be.posterior_mean()
    # All observations are around 1.0, so the posterior mean should be close
    # to 1.0 (well above the prior mean of 0.0).
    assert 0.5 < mean < 1.5
    # Posterior variance should have shrunk below the prior variance of 1.0.
    assert be.posterior_var() < 1.0


def test_bayesian_predictive_returns_two_tuple():
    """predictive() returns a (mean, std) tuple of floats."""
    be = BayesianEstimator(dim=64)
    be.reset(prior_mean=0.0, prior_var=1.0)
    be.update(0.5)
    pred = be.predictive()
    assert isinstance(pred, tuple)
    assert len(pred) == 2
    mean, std = pred
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std > 0.0


def test_bayesian_reset_restores_prior():
    """reset(prior_mean, prior_var) restores the posterior to the prior."""
    be = BayesianEstimator(dim=64)
    be.update(5.0)
    be.update(5.0)
    be.reset(prior_mean=0.0, prior_var=1.0)
    assert abs(be.posterior_mean() - 0.0) < 1e-9
    assert abs(be.posterior_var() - 1.0) < 1e-9


def test_bayesian_beta_binomial_path():
    """Tuple observations are routed to the Beta-Binomial conjugate update."""
    be = BayesianEstimator(dim=64)
    be.reset(prior_mean=0.0, prior_var=1.0)
    # Observe 9 successes out of 10 trials, then 8 out of 10.
    be.update((9, 10))
    be.update((8, 10))
    # Only Beta observations have been seen -> posterior mean is the Beta mean.
    a, b = be.beta_a, be.beta_b
    expected_mean = a / (a + b)
    assert abs(be.posterior_mean() - expected_mean) < 1e-9
    # Beta(1+9+8, 1+1+2) = Beta(18, 4) -> mean = 18/22 ~= 0.818
    assert 0.7 < be.posterior_mean() < 0.9


# ---------------------------------------------------------------------------
# ChangePointDetector
# ---------------------------------------------------------------------------


def test_change_point_detector_finds_clear_shift():
    """A clear mid-series mean shift is detected as a change point."""
    cpd = ChangePointDetector(dim=64)
    series = np.concatenate([np.zeros(20), 10.0 * np.ones(20)])
    cps = cpd.detect(series)
    assert cps.dtype == int
    assert cps.shape[0] >= 1
    # At least one detected change point should be near the true shift at 20.
    assert any(abs(int(cp) - 20) <= 5 for cp in cps.tolist())


def test_change_point_detector_count_matches_detect():
    """detect_count(series) equals len(detect(series))."""
    cpd = ChangePointDetector(dim=64)
    series = np.concatenate([np.zeros(20), 10.0 * np.ones(20)])
    n = cpd.detect_count(series)
    assert n == cpd.detect(series).shape[0]
    assert n >= 1


def test_change_point_detector_constant_series_returns_zero():
    """A perfectly constant series has no change points."""
    cpd = ChangePointDetector(dim=64)
    cps = cpd.detect(np.ones(30))
    assert cps.shape[0] == 0


def test_change_point_detector_short_series_returns_empty():
    """Series shorter than the minimum length return an empty array."""
    cpd = ChangePointDetector(dim=64)
    cps = cpd.detect(np.array([1.0, 2.0, 3.0]))
    assert cps.shape[0] == 0
    assert cps.dtype == int


def test_change_point_detector_returns_int_array():
    """detect(series) returns an np.ndarray of dtype int (np.intp)."""
    cpd = ChangePointDetector(dim=64)
    rng = np.random.default_rng(5)
    series = np.concatenate([rng.standard_normal(25), 5.0 + rng.standard_normal(25)])
    cps = cpd.detect(series)
    assert isinstance(cps, np.ndarray)
    assert cps.dtype.kind == "i"
