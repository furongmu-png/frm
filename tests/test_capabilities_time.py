"""Tests for the Time capability domain (base module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    EventTimestampAnalyzer,
    FrequencyAnalyzer,
    SeasonalityDetector,
    TimeRules,
    TimeSeriesEncoder,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


def _sine_wave(n: int = 200, period: int = 20) -> np.ndarray:
    """Pure sine wave with given period."""
    t = np.arange(n, dtype=float)
    return np.sin(2.0 * np.pi * t / period)


def _noisy_sine(n: int = 200, period: int = 20, noise: float = 0.1) -> np.ndarray:
    """Sine wave + Gaussian noise."""
    return _sine_wave(n, period) + noise * np.random.randn(n)


# --------------------------------------------------------------------------- #
# TimeRules
# --------------------------------------------------------------------------- #

def test_time_rules_defaults():
    rules = TimeRules()
    assert rules.sample_rate == 1.0
    assert rules.fft_window_size == 64
    assert rules.fft_hop_size == 32
    assert rules.seasonality_max_lag == 128
    assert rules.seasonality_threshold == 0.3
    assert rules.event_threshold_std == 2.0
    assert rules.forecast_min_samples == 8


def test_time_rules_post_init_populates_dict():
    rules = TimeRules()
    assert rules.rules["fft_window_size"] == 64
    assert rules.rules["seasonality_threshold"] == 0.3


def test_time_rules_custom_values():
    rules = TimeRules(sample_rate=2.0, fft_window_size=128)
    assert rules.sample_rate == 2.0
    assert rules.fft_window_size == 128


# --------------------------------------------------------------------------- #
# TimeSeriesEncoder
# --------------------------------------------------------------------------- #

class TestTimeSeriesEncoder:
    def test_encode_shape_and_norm(self):
        enc = TimeSeriesEncoder(dim=32)
        v = enc.encode(_noisy_sine())
        assert v.shape == (32,)
        norm = float(np.linalg.norm(v))
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_encode_empty_returns_zeros(self):
        enc = TimeSeriesEncoder(dim=16)
        v = enc.encode(np.array([]))
        assert v.shape == (16,)
        np.testing.assert_allclose(v, 0.0)

    def test_encode_constant_series_no_nan(self):
        enc = TimeSeriesEncoder(dim=16)
        v = enc.encode(np.ones(100))
        assert np.all(np.isfinite(v))

    def test_encode_deterministic(self):
        enc = TimeSeriesEncoder(dim=16)
        v1 = enc.encode(_noisy_sine())
        np.random.seed(42)
        v2 = enc.encode(_noisy_sine())
        np.testing.assert_allclose(v1, v2)

    def test_encode_short_series_padded(self):
        # Shorter than fft_window_size: should pad with zeros internally.
        enc = TimeSeriesEncoder(dim=16)
        v = enc.encode(np.array([1.0, 2.0, 3.0]))
        assert v.shape == (16,)
        assert np.all(np.isfinite(v))


# --------------------------------------------------------------------------- #
# SeasonalityDetector
# --------------------------------------------------------------------------- #

class TestSeasonalityDetector:
    def test_detect_pure_sine_period(self):
        sd = SeasonalityDetector(dim=16)
        result = sd.detect(_sine_wave(n=400, period=20))
        # The dominant period should be 20 (or a multiple thereof).
        assert 20 in result["periods"]
        assert result["dominant_period"] == 20

    def test_detect_constant_series_no_periods(self):
        sd = SeasonalityDetector(dim=16)
        result = sd.detect(np.ones(200))
        assert result["periods"] == []
        assert result["dominant_period"] == 0

    def test_detect_short_series_returns_empty(self):
        sd = SeasonalityDetector(dim=16)
        result = sd.detect(np.array([1.0, 2.0, 3.0]))
        assert result["periods"] == []
        assert result["dominant_period"] == 0

    def test_detect_returns_strengths_sorted_descending(self):
        sd = SeasonalityDetector(dim=16)
        result = sd.detect(_sine_wave(n=400, period=20))
        strengths = result["strengths"]
        if len(strengths) >= 2:
            assert strengths[0] >= strengths[1]

    def test_detect_threshold_filtering(self):
        rules = TimeRules(seasonality_threshold=0.95)
        sd = SeasonalityDetector(dim=16, rules=rules)
        result = sd.detect(_noisy_sine())
        # Very high threshold: few (if any) peaks should pass.
        for s in result["strengths"]:
            assert s > 0.95

    def test_detect_max_lag_capped(self):
        # If series is short, max_lag is capped at n // 2.
        rules = TimeRules(seasonality_max_lag=1000)
        sd = SeasonalityDetector(dim=16, rules=rules)
        result = sd.detect(_sine_wave(n=50, period=20))
        # Should still detect the period (cap at n // 2 = 25 > 20).
        assert result["dominant_period"] in (20, 0)


# --------------------------------------------------------------------------- #
# FrequencyAnalyzer
# --------------------------------------------------------------------------- #

class TestFrequencyAnalyzer:
    def test_analyze_pure_sine_dominant_freq(self):
        fa = FrequencyAnalyzer(dim=16)
        rules = TimeRules(sample_rate=1.0)
        fa = FrequencyAnalyzer(dim=16, rules=rules)
        # 200 samples at sample_rate=1.0, period=20 -> freq = 1/20 = 0.05.
        result = fa.analyze(_sine_wave(n=200, period=20))
        assert abs(result["dominant_freq"] - 0.05) < 0.01

    def test_analyze_returns_power_and_freq_arrays(self):
        fa = FrequencyAnalyzer(dim=16)
        result = fa.analyze(_noisy_sine())
        assert result["frequencies"].size == result["power"].size
        assert result["frequencies"].size > 0

    def test_analyze_spectral_entropy_in_unit_interval(self):
        fa = FrequencyAnalyzer(dim=16)
        result = fa.analyze(_noisy_sine())
        assert 0.0 <= result["spectral_entropy"] <= 1.0

    def test_analyze_pure_sine_low_entropy(self):
        fa = FrequencyAnalyzer(dim=16)
        result = fa.analyze(_sine_wave(n=400, period=20))
        # A pure sine has very concentrated spectrum -> low entropy.
        assert result["spectral_entropy"] < 0.3

    def test_analyze_short_series_returns_zeros(self):
        fa = FrequencyAnalyzer(dim=16)
        result = fa.analyze(np.array([1.0]))
        assert result["dominant_freq"] == 0.0
        assert result["power"].size == 0

    def test_analyze_constant_series_zero_dominant(self):
        fa = FrequencyAnalyzer(dim=16)
        # All-ones: centered -> all-zero spectrum.
        result = fa.analyze(np.ones(100))
        assert result["dominant_freq"] == 0.0

    def test_analyze_custom_sample_rate(self):
        rules = TimeRules(sample_rate=10.0)
        fa = FrequencyAnalyzer(dim=16, rules=rules)
        # 200 samples at sample_rate=10.0, period=20 -> freq = 10/20 = 0.5.
        result = fa.analyze(_sine_wave(n=200, period=20))
        assert abs(result["dominant_freq"] - 0.5) < 0.05


# --------------------------------------------------------------------------- #
# EventTimestampAnalyzer
# --------------------------------------------------------------------------- #

class TestEventTimestampAnalyzer:
    def test_analyze_basic(self):
        eta = EventTimestampAnalyzer(dim=16)
        ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = eta.analyze(ts)
        assert result["total_events"] == 5
        assert result["rate"] == 1.25  # 5 events / 4 time units
        np.testing.assert_allclose(result["inter_arrival"], [1.0, 1.0, 1.0, 1.0])

    def test_analyze_regular_events_zero_burstiness(self):
        eta = EventTimestampAnalyzer(dim=16)
        ts = np.arange(10, dtype=float)
        result = eta.analyze(ts)
        # Perfectly regular: std=0, mean=1, burstiness = -1.
        np.testing.assert_allclose(result["burstiness"], -1.0, atol=1e-9)

    def test_analyze_bursty_events_positive_burstiness(self):
        eta = EventTimestampAnalyzer(dim=16)
        # Most events clustered at start, then a long gap.
        ts = np.array([0.0, 0.1, 0.2, 0.3, 10.0])
        result = eta.analyze(ts)
        assert result["burstiness"] > 0.0

    def test_analyze_burstiness_in_range(self):
        eta = EventTimestampAnalyzer(dim=16)
        ts = np.sort(np.random.uniform(0, 100, 50))
        result = eta.analyze(ts)
        assert -1.0 <= result["burstiness"] <= 1.0

    def test_analyze_single_event_returns_zeros(self):
        eta = EventTimestampAnalyzer(dim=16)
        result = eta.analyze(np.array([5.0]))
        assert result["total_events"] == 1
        assert result["rate"] == 0.0
        assert result["inter_arrival"].size == 0

    def test_analyze_unsorted_timestamps_sorted_internally(self):
        eta = EventTimestampAnalyzer(dim=16)
        ts = np.array([4.0, 1.0, 3.0, 0.0, 2.0])
        result = eta.analyze(ts)
        np.testing.assert_allclose(result["inter_arrival"], [1.0, 1.0, 1.0, 1.0])

    def test_analyze_total_events_count(self):
        eta = EventTimestampAnalyzer(dim=16)
        ts = np.linspace(0, 10, 11)
        result = eta.analyze(ts)
        assert result["total_events"] == 11
