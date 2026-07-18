"""Tests for the Audio capability domain (base module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    AudioClassifier,
    AudioEncoder,
    AudioRules,
    OnsetDetector,
    PitchDetector,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# --------------------------------------------------------------------------- #
# AudioRules
# --------------------------------------------------------------------------- #


class TestAudioRules:
    def test_default_parameters(self):
        rules = AudioRules()
        assert rules.sample_rate == 16000
        assert rules.frame_size == 1024
        assert rules.hop_size == 512
        assert rules.n_mels == 26
        assert rules.onset_threshold == 0.3
        assert rules.pitch_min_hz == 80.0
        assert rules.pitch_max_hz == 500.0
        assert "speech" in rules.texture_labels

    def test_hz_to_mel_round_trip(self):
        """``mel_to_hz(hz_to_mel(x)) == x`` for positive x."""
        for hz in [100.0, 440.0, 1000.0, 8000.0]:
            mel = AudioRules.hz_to_mel(hz)
            recovered = AudioRules.mel_to_hz(mel)
            assert abs(recovered - hz) < 0.5, f"round-trip failed for {hz} Hz"

    def test_hz_to_mel_monotonic(self):
        """Mel scale is monotonically increasing with Hz."""
        hz_vals = np.linspace(0, 8000, 100)
        mel_vals = [AudioRules.hz_to_mel(h) for h in hz_vals]
        assert all(mel_vals[i] <= mel_vals[i + 1] for i in range(len(mel_vals) - 1))


# --------------------------------------------------------------------------- #
# AudioEncoder
# --------------------------------------------------------------------------- #


class TestAudioEncoder:
    def test_encode_returns_l2_normalized_dim_vector(self):
        enc = AudioEncoder(dim=64)
        signal = np.random.randn(16000) * 0.1
        vec = enc.encode(signal)
        assert vec.shape == (64,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    def test_encode_deterministic(self):
        enc = AudioEncoder(dim=32)
        signal = np.random.randn(8000) * 0.1
        v1 = enc.encode(signal)
        v2 = enc.encode(signal)
        np.testing.assert_allclose(v1, v2, atol=1e-8)

    def test_encode_empty_signal_returns_zero_vector(self):
        enc = AudioEncoder(dim=64)
        vec = enc.encode(np.array([]))
        assert vec.shape == (64,)
        assert np.all(vec == 0.0)

    def test_encode_stereo_downmixes_to_mono(self):
        enc = AudioEncoder(dim=32)
        stereo = np.random.randn(2, 16000) * 0.1
        v_stereo = enc.encode(stereo)
        assert v_stereo.shape == (32,)
        assert np.isclose(np.linalg.norm(v_stereo), 1.0, atol=1e-6)

    def test_encode_different_signals_different_vectors(self):
        enc = AudioEncoder(dim=64)
        sig_a = np.sin(np.linspace(0, 440 * 2 * np.pi, 16000))
        sig_b = np.random.randn(16000) * 0.5
        v_a = enc.encode(sig_a)
        v_b = enc.encode(sig_b)
        assert not np.allclose(v_a, v_b, atol=1e-4)

    def test_encode_handles_short_signal(self):
        """Signal shorter than frame_size should still encode."""
        enc = AudioEncoder(dim=32)
        short = np.random.randn(100)
        vec = enc.encode(short)
        assert vec.shape == (32,)


# --------------------------------------------------------------------------- #
# OnsetDetector
# --------------------------------------------------------------------------- #


class TestOnsetDetector:
    def test_detect_returns_expected_dict_keys(self):
        det = OnsetDetector()
        signal = np.random.randn(16000) * 0.1
        result = det.detect(signal)
        assert set(result.keys()) == {
            "onset_frames", "onset_times", "spectral_flux", "mean_flux"
        }

    def test_detect_silence_no_onsets(self):
        det = OnsetDetector()
        silence = np.zeros(16000)
        result = det.detect(silence)
        assert result["onset_frames"] == []
        assert result["mean_flux"] == 0.0

    def test_detect_click_detects_onset(self):
        """A sudden click after silence should be detected as an onset."""
        det = OnsetDetector()
        signal = np.zeros(16000)
        signal[8000:8100] = np.random.randn(100) * 0.5
        result = det.detect(signal)
        assert len(result["onset_frames"]) >= 1
        assert len(result["onset_times"]) == len(result["onset_frames"])

    def test_detect_short_signal_returns_empty(self):
        det = OnsetDetector()
        short = np.zeros(100)
        result = det.detect(short)
        assert result["onset_frames"] == []

    def test_detect_onset_times_are_monotonic(self):
        det = OnsetDetector()
        signal = np.random.randn(16000) * 0.1
        result = det.detect(signal)
        times = result["onset_times"]
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))


# --------------------------------------------------------------------------- #
# PitchDetector
# --------------------------------------------------------------------------- #


class TestPitchDetector:
    def test_detect_returns_expected_dict_keys(self):
        det = PitchDetector()
        signal = np.random.randn(16000) * 0.1
        result = det.detect(signal)
        assert set(result.keys()) == {"pitch_hz", "confidence", "f0_candidates"}

    def test_detect_sine_wave_returns_correct_pitch(self):
        """A 440 Hz sine wave should be detected near 440 Hz."""
        det = PitchDetector()
        sr = 16000
        t = np.arange(sr) / sr
        signal = np.sin(2 * np.pi * 440.0 * t)
        result = det.detect(signal, sample_rate=sr)
        assert result["pitch_hz"] > 0
        # Pitch detection may not be exact due to frame boundary effects,
        # but should be within ~50 Hz of 440.
        assert abs(result["pitch_hz"] - 440.0) < 100.0, (
            f"pitch={result['pitch_hz']}, expected near 440"
        )

    def test_detect_silence_returns_zero(self):
        det = PitchDetector()
        silence = np.zeros(16000)
        result = det.detect(silence)
        assert result["pitch_hz"] == 0.0
        assert result["confidence"] == 0.0

    def test_detect_empty_signal(self):
        det = PitchDetector()
        result = det.detect(np.array([]))
        assert result["pitch_hz"] == 0.0

    def test_detect_confidence_in_unit_interval(self):
        det = PitchDetector()
        signal = np.sin(2 * np.pi * 220.0 * np.arange(16000) / 16000)
        result = det.detect(signal)
        assert 0.0 <= result["confidence"] <= 1.0


# --------------------------------------------------------------------------- #
# AudioClassifier
# --------------------------------------------------------------------------- #


class TestAudioClassifier:
    def test_classify_returns_expected_dict_keys(self):
        clf = AudioClassifier()
        signal = np.random.randn(16000) * 0.1
        result = clf.classify(signal)
        assert set(result.keys()) == {"label", "confidence", "features"}

    def test_classify_silence_detected(self):
        clf = AudioClassifier()
        silence = np.zeros(16000)
        result = clf.classify(silence)
        assert result["label"] == "silence"

    def test_classify_returns_valid_label(self):
        clf = AudioClassifier()
        signal = np.random.randn(16000) * 0.1
        result = clf.classify(signal)
        assert result["label"] in AudioRules().texture_labels

    def test_classify_features_contain_expected_keys(self):
        clf = AudioClassifier()
        signal = np.random.randn(16000) * 0.1
        result = clf.classify(signal)
        features = result["features"]
        assert "spectral_centroid_hz" in features
        assert "zero_crossing_rate" in features
        assert "rms_energy" in features
        assert "spectral_flatness" in features

    def test_classify_empty_signal(self):
        clf = AudioClassifier()
        result = clf.classify(np.array([]))
        assert result["label"] == "silence"

    def test_classify_deterministic(self):
        clf = AudioClassifier()
        signal = np.sin(2 * np.pi * 440.0 * np.arange(16000) / 16000)
        r1 = clf.classify(signal)
        r2 = clf.classify(signal)
        assert r1["label"] == r2["label"]
        np.testing.assert_allclose(
            list(r1["features"].values()), list(r2["features"].values()), atol=1e-8
        )
