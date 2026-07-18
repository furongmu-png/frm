"""Tests for the Time capability domain (advanced module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    AnomalyTimingDetector,
    CyclePhaseTracker,
    ForecastabilityScorer,
    TimeRules,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


def _sine_wave(n: int = 200, period: int = 20) -> np.ndarray:
    t = np.arange(n, dtype=float)
    return np.sin(2.0 * np.pi * t / period)


def _noisy_sine(n: int = 200, period: int = 20, noise: float = 0.1) -> np.ndarray:
    return _sine_wave(n, period) + noise * np.random.randn(n)


# --------------------------------------------------------------------------- #
# AnomalyTimingDetector
# --------------------------------------------------------------------------- #

class TestAnomalyTimingDetector:
    def test_detect_regular_no_anomalies(self):
        atd = AnomalyTimingDetector(dim=16)
        ts = np.arange(20, dtype=float)  # perfectly regular
        result = atd.detect(ts)
        assert int(np.sum(result["anomalies"])) == 0

    def test_detect_with_spike(self):
        atd = AnomalyTimingDetector(dim=16)
        ts = np.array([0.0, 1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = atd.detect(ts)
        # The gap to 100.0 should be flagged.
        assert int(np.sum(result["anomalies"])) >= 1

    def test_detect_threshold_returned(self):
        atd = AnomalyTimingDetector(dim=16)
        result = atd.detect(np.arange(10, dtype=float))
        assert result["threshold"] == 2.0

    def test_detect_short_series_returns_empty(self):
        atd = AnomalyTimingDetector(dim=16)
        result = atd.detect(np.array([1.0, 2.0]))
        assert result["anomalies"].size == 1
        assert result["scores"].size == 1

    def test_detect_very_short_returns_empty(self):
        atd = AnomalyTimingDetector(dim=16)
        result = atd.detect(np.array([1.0]))
        assert result["anomalies"].size == 0

    def test_detect_scores_nonnegative(self):
        atd = AnomalyTimingDetector(dim=16)
        ts = np.sort(np.random.uniform(0, 100, 50))
        result = atd.detect(ts)
        assert np.all(result["scores"] >= 0.0)

    def test_detect_custom_threshold(self):
        rules = TimeRules(event_threshold_std=1.0)
        atd = AnomalyTimingDetector(dim=16, rules=rules)
        result = atd.detect(np.arange(10, dtype=float))
        assert result["threshold"] == 1.0


# --------------------------------------------------------------------------- #
# CyclePhaseTracker
# --------------------------------------------------------------------------- #

class TestCyclePhaseTracker:
    def test_track_with_explicit_period(self):
        cpt = CyclePhaseTracker(dim=16)
        result = cpt.track(_sine_wave(n=40, period=10), period=10)
        assert result["period"] == 10
        assert result["phases"].shape == (40,)

    def test_track_phases_in_unit_interval(self):
        cpt = CyclePhaseTracker(dim=16)
        result = cpt.track(_sine_wave(n=100, period=20), period=20)
        assert np.all(result["phases"] >= 0.0)
        assert np.all(result["phases"] < 1.0)

    def test_track_phase_coherence_in_unit_interval(self):
        cpt = CyclePhaseTracker(dim=16)
        result = cpt.track(_sine_wave(n=100, period=20), period=20)
        assert 0.0 <= result["phase_coherence"] <= 1.0

    def test_track_auto_detect_period(self):
        cpt = CyclePhaseTracker(dim=16)
        # No explicit period: should auto-detect via SeasonalityDetector.
        result = cpt.track(_sine_wave(n=400, period=20))
        assert result["period"] > 0

    def test_track_short_series_returns_zeros(self):
        cpt = CyclePhaseTracker(dim=16)
        result = cpt.track(np.array([1.0, 2.0]))
        assert result["period"] == 0
        assert result["phase_coherence"] == 0.0

    def test_track_preserves_ca_state(self):
        """CA state should be restored after tracking (R9-010)."""
        cpt = CyclePhaseTracker(dim=16)
        ca = cpt.biological.automata
        original_state = ca.state.copy()
        original_rule = ca.rule
        cpt.track(_sine_wave(n=100, period=20), period=20)
        np.testing.assert_array_equal(ca.state, original_state)
        assert ca.rule == original_rule

    def test_track_invalid_period_falls_back(self):
        cpt = CyclePhaseTracker(dim=16)
        # period=-1 -> falls back to default (n // 2).
        result = cpt.track(_sine_wave(n=40, period=10), period=-1)
        # Should detect a period via fallback (likely n // 2 = 20 or auto-detect).
        assert result["period"] > 0


# --------------------------------------------------------------------------- #
# ForecastabilityScorer
# --------------------------------------------------------------------------- #

class TestForecastabilityScorer:
    def test_score_returns_dict_with_all_fields(self):
        fs = ForecastabilityScorer(dim=16)
        result = fs.score(_noisy_sine())
        assert "forecastability" in result
        assert "entropy" in result
        assert "stationarity" in result
        assert "autocorr_strength" in result

    def test_score_values_in_unit_interval(self):
        fs = ForecastabilityScorer(dim=16)
        result = fs.score(_noisy_sine())
        assert 0.0 <= result["forecastability"] <= 1.0
        assert 0.0 <= result["entropy"] <= 1.0
        assert 0.0 <= result["stationarity"] <= 1.0
        assert 0.0 <= result["autocorr_strength"] <= 1.0

    def test_score_pure_sine_high_forecastability(self):
        fs = ForecastabilityScorer(dim=16)
        # Pure sine: low entropy, high autocorr -> high forecastability.
        result = fs.score(_sine_wave(n=400, period=20))
        assert result["forecastability"] > 0.5

    def test_score_random_noise_low_forecastability(self):
        fs = ForecastabilityScorer(dim=16)
        # Pure random noise: high entropy, low autocorr.
        result = fs.score(np.random.randn(200))
        assert result["forecastability"] < 0.6

    def test_score_short_series_returns_zero(self):
        fs = ForecastabilityScorer(dim=16)
        result = fs.score(np.array([1.0, 2.0]))
        assert result["forecastability"] == 0.0
        assert result["entropy"] == 1.0

    def test_score_constant_series_high_forecastability(self):
        fs = ForecastabilityScorer(dim=16)
        # Constant: no entropy, perfectly stationary.
        result = fs.score(np.ones(100))
        # Constant series has zero entropy and max stationarity.
        assert result["entropy"] <= 0.05
        assert result["stationarity"] >= 0.99

    def test_score_custom_min_samples(self):
        rules = TimeRules(forecast_min_samples=20)
        fs = ForecastabilityScorer(dim=16, rules=rules)
        # Below 20 samples -> returns zeros.
        result = fs.score(np.arange(15, dtype=float))
        assert result["forecastability"] == 0.0

    def test_score_nonstationary_series(self):
        fs = ForecastabilityScorer(dim=16)
        # Series with a clear trend break: first half mean 0, second half mean 100.
        series = np.concatenate([np.zeros(50), 100.0 + np.zeros(50)])
        result = fs.score(series)
        # Stationarity should be low because the means differ.
        assert result["stationarity"] < 0.5
