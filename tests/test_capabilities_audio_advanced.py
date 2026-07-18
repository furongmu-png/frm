"""Tests for the Advanced Audio capability domain."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    MusicAnalyzer,
    SpeakerRecognizer,
    SpeechSegmenter,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# --------------------------------------------------------------------------- #
# SpeechSegmenter
# --------------------------------------------------------------------------- #


class TestSpeechSegmenter:
    def test_segment_returns_expected_dict_keys(self):
        seg = SpeechSegmenter()
        signal = np.random.randn(16000) * 0.1
        result = seg.segment(signal)
        assert set(result.keys()) == {"segments", "total_duration", "speech_ratio"}

    def test_segment_silence_all_silence(self):
        seg = SpeechSegmenter()
        silence = np.zeros(16000)
        result = seg.segment(silence)
        assert result["speech_ratio"] == 0.0
        assert all(s["label"] == "silence" for s in result["segments"])

    def test_segment_speech_detected(self):
        """A signal with clearly loud sections should detect some speech."""
        seg = SpeechSegmenter()
        signal = np.zeros(16000)
        # Insert two loud speech-like sections.
        signal[2000:4000] = np.random.randn(2000) * 0.5
        signal[10000:12000] = np.random.randn(2000) * 0.5
        result = seg.segment(signal)
        assert result["speech_ratio"] > 0.0
        # At least one segment should be labeled "speech".
        assert any(s["label"] == "speech" for s in result["segments"])

    def test_segment_total_duration(self):
        seg = SpeechSegmenter()
        signal = np.random.randn(16000) * 0.1
        result = seg.segment(signal, sample_rate=16000)
        assert abs(result["total_duration"] - 1.0) < 0.01

    def test_segment_short_signal(self):
        seg = SpeechSegmenter()
        short = np.zeros(100)
        result = seg.segment(short)
        assert result["speech_ratio"] == 0.0

    def test_segment_speech_ratio_in_unit_interval(self):
        seg = SpeechSegmenter()
        signal = np.random.randn(16000) * 0.1
        result = seg.segment(signal)
        assert 0.0 <= result["speech_ratio"] <= 1.0

    def test_segment_ca_state_restored(self):
        """The biological CA state must be restored after segment() (R9-010)."""
        from zero_data_model.biological import BiologicalSubstrate

        bio = BiologicalSubstrate(dim=64)
        ca = bio.automata
        saved_state = ca.state.copy()
        saved_rule = ca.rule
        seg = SpeechSegmenter(dim=64, biological=bio)
        signal = np.random.randn(16000) * 0.1
        seg.segment(signal)
        np.testing.assert_array_equal(ca.state, saved_state)
        assert ca.rule == saved_rule


# --------------------------------------------------------------------------- #
# MusicAnalyzer
# --------------------------------------------------------------------------- #


class TestMusicAnalyzer:
    def test_analyze_returns_expected_dict_keys(self):
        ma = MusicAnalyzer()
        signal = np.random.randn(16000) * 0.1
        result = ma.analyze(signal)
        assert set(result.keys()) == {
            "tempo_bpm", "beat_frames", "beat_times", "onset_envelope"
        }

    def test_analyze_silence_returns_zero_tempo(self):
        ma = MusicAnalyzer()
        silence = np.zeros(16000)
        result = ma.analyze(silence)
        assert result["tempo_bpm"] == 0.0

    def test_analyze_beat_times_monotonic(self):
        ma = MusicAnalyzer()
        signal = np.random.randn(32000) * 0.1
        result = ma.analyze(signal)
        times = result["beat_times"]
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    def test_analyze_short_signal(self):
        ma = MusicAnalyzer()
        short = np.zeros(100)
        result = ma.analyze(short)
        assert result["tempo_bpm"] == 0.0
        assert result["beat_frames"] == []

    def test_analyze_tempo_in_reasonable_range(self):
        """Tempo should be in 60-180 BPM range if detected."""
        ma = MusicAnalyzer()
        signal = np.random.randn(32000) * 0.2
        result = ma.analyze(signal)
        if result["tempo_bpm"] > 0:
            assert 60.0 <= result["tempo_bpm"] <= 180.0


# --------------------------------------------------------------------------- #
# SpeakerRecognizer
# --------------------------------------------------------------------------- #


class TestSpeakerRecognizer:
    def test_extract_features_returns_l2_normalized_dim_vector(self):
        sr = SpeakerRecognizer(dim=64)
        signal = np.random.randn(16000) * 0.1
        vec = sr.extract_features(signal)
        assert vec.shape == (64,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    def test_extract_features_deterministic(self):
        sr = SpeakerRecognizer(dim=32)
        signal = np.random.randn(16000) * 0.1
        v1 = sr.extract_features(signal)
        v2 = sr.extract_features(signal)
        np.testing.assert_allclose(v1, v2, atol=1e-8)

    def test_extract_features_empty_signal(self):
        sr = SpeakerRecognizer(dim=64)
        vec = sr.extract_features(np.array([]))
        assert vec.shape == (64,)
        assert np.all(vec == 0.0)

    def test_compare_same_signal_returns_high_similarity(self):
        sr = SpeakerRecognizer(dim=64)
        signal = np.random.randn(16000) * 0.1
        v = sr.extract_features(signal)
        sim = sr.compare(v, v)
        assert sim > 0.99

    def test_compare_different_signals_returns_lower_similarity(self):
        sr = SpeakerRecognizer(dim=64)
        sig_a = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 0.5
        sig_b = np.random.randn(16000) * 0.5
        v_a = sr.extract_features(sig_a)
        v_b = sr.extract_features(sig_b)
        sim_same = sr.compare(v_a, v_a)
        sim_diff = sr.compare(v_a, v_b)
        assert sim_diff < sim_same

    def test_compare_empty_features(self):
        sr = SpeakerRecognizer(dim=64)
        empty = np.zeros(64)
        assert sr.compare(empty, empty) == 0.0

    def test_compare_returns_in_unit_interval(self):
        """Cosine similarity is in [-1, 1]."""
        sr = SpeakerRecognizer(dim=64)
        sig_a = np.random.randn(16000) * 0.1
        sig_b = np.random.randn(16000) * 0.2
        v_a = sr.extract_features(sig_a)
        v_b = sr.extract_features(sig_b)
        sim = sr.compare(v_a, v_b)
        assert -1.0 <= sim <= 1.0
