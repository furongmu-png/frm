# tests/test_capabilities_analytics.py
"""Tests for the zero-data Data Analytics capability module.

The analytics capabilities (time-series forecasting, anomaly detection,
pattern mining, trend analysis) compose the already-implemented core
cognitive modules with a small statistical rule library (``AnalyticsRules``)
used as prior knowledge. No external training data is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.capabilities.analytics import (
    AnomalyDetector,
    PatternMiner,
    TimeSeriesForecaster,
    TrendAnalyzer,
)


@pytest.fixture(autouse=True)
def _deterministic_rng() -> None:
    """Pin the global RNG so tests involving random core-module matrices
    (active inference transitions, quantum annealers, ...) are reproducible."""
    np.random.seed(42)


def test_forecaster_length() -> None:
    """forecast(series, horizon=5) returns a length-5 array of finite floats."""
    forecaster = TimeSeriesForecaster(dim=64)
    series = np.arange(20.0)
    out = forecaster.forecast(series, horizon=5)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5,)
    assert out.dtype.kind == "f"
    assert np.all(np.isfinite(out)), "forecast values must all be finite"


def test_forecaster_trend() -> None:
    """Forecast of an upward-trending series continues upward:
    its mean exceeds the historical series mean."""
    forecaster = TimeSeriesForecaster(dim=64)
    series = np.arange(20.0)  # mean = 9.5
    out = forecaster.forecast(series, horizon=5)
    assert float(np.mean(out)) > float(np.mean(series))


def test_anomaly_detector_length_and_bool() -> None:
    """detect(series) returns a boolean array with the same length as series."""
    detector = AnomalyDetector(dim=64)
    series = np.sin(np.linspace(0.0, 4.0 * np.pi, 25))
    out = detector.detect(series)
    assert isinstance(out, np.ndarray)
    assert out.dtype == bool
    assert out.shape == (25,)


def test_anomaly_detector_finds_outlier() -> None:
    """A series with one large spike flags at least one anomaly."""
    detector = AnomalyDetector(dim=64)
    series = np.array([1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0])
    out = detector.detect(series)
    assert out.dtype == bool
    assert out.shape == (9,)
    assert bool(np.any(out)), "expected at least one detected anomaly"


def test_pattern_miner_keys() -> None:
    """mine(series) returns the four required pattern keys."""
    miner = PatternMiner(dim=64)
    series = np.sin(np.linspace(0.0, 4.0 * np.pi, 40))
    result = miner.mine(series)
    assert set(result.keys()) == {
        "self_similarity",
        "topology",
        "automaton_rule",
        "periodicity",
    }


def test_trend_analyzer_keys() -> None:
    """analyze(series) returns the five required keys with correct types."""
    analyzer = TrendAnalyzer(dim=64)
    series = np.sin(np.linspace(0.0, 4.0 * np.pi, 30))
    result = analyzer.analyze(series)
    assert set(result.keys()) == {
        "trend_slope",
        "regime",
        "curvature",
        "geodesic_deviation",
        "isomorphism_score",
    }
    assert isinstance(result["regime"], str)
    for key in ("trend_slope", "curvature", "geodesic_deviation", "isomorphism_score"):
        assert isinstance(result[key], float), f"{key} must be a float"
        assert np.isfinite(result[key]), f"{key} must be finite"


def test_trend_analyzer_regime() -> None:
    """Regime classification follows the sign of the trend slope."""
    analyzer = TrendAnalyzer(dim=64)
    assert analyzer.analyze(np.arange(50.0))["regime"] == "up"
    assert analyzer.analyze(-np.arange(50.0))["regime"] == "down"
    assert analyzer.analyze(np.zeros(50))["regime"] == "flat"
