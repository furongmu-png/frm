# tests/test_advanced_capabilities.py
"""Model-facade tests for the 9 advanced ZeroDataModel methods that had no
direct test coverage.

The advanced capability classes themselves (MultiLingualEncoder,
SyntacticAnalyzer, SentenceEncoder, PointCloudEncoder, VideoFrameAnalyzer,
DepthEstimator, CausalInference, BayesianEstimator, ChangePointDetector) are
exercised via their respective unit-test files. This file pins the
ZeroDataModel facade methods that delegate to them, ensuring each method
returns the right type and shape with finite values.
"""

from __future__ import annotations

import numpy as np

from zero_data_model.model import ZeroDataModel


def _model() -> ZeroDataModel:
    """A small, fast model for the test suite (dim=16)."""
    return ZeroDataModel(dim=16)


# --------------------------------------------------------------------------- #
# Advanced NLP facade methods
# --------------------------------------------------------------------------- #


def test_encode_multilingual_returns_finite_dim_vector():
    """model.encode_multilingual returns a finite ndarray of shape (dim,)."""
    model = _model()
    vec = model.encode_multilingual("hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (16,)
    assert np.all(np.isfinite(vec))


def test_analyze_syntax_returns_expected_keys():
    """model.analyze_syntax returns a dict with the five syntactic keys."""
    model = _model()
    out = model.analyze_syntax("The cat sat. It ran.")
    assert isinstance(out, dict)
    assert set(out.keys()) == {
        "sentence_count",
        "avg_word_length",
        "punctuation_density",
        "pos_guesses",
        "dependency_hint",
    }
    assert isinstance(out["sentence_count"], int)
    assert out["sentence_count"] == 2


def test_encode_sentences_returns_2d_array():
    """model.encode_sentences returns an (n_sentences, dim) ndarray."""
    model = _model()
    arr = model.encode_sentences("First sentence. Second one.")
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape[0] == 2
    assert arr.shape[1] == 16
    assert np.all(np.isfinite(arr))


# --------------------------------------------------------------------------- #
# Advanced Vision facade methods
# --------------------------------------------------------------------------- #


def test_encode_point_cloud_returns_finite_dim_vector():
    """model.encode_point_cloud returns a finite ndarray of shape (dim,)."""
    model = _model()
    points = np.random.randn(50, 3)
    vec = model.encode_point_cloud(points)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (16,)
    assert np.all(np.isfinite(vec))


def test_analyze_video_returns_motion_and_keyframes():
    """model.analyze_video returns a dict with motion/keyframes/encoding keys."""
    model = _model()
    frames = [np.random.randn(8, 8) for _ in range(5)]
    out = model.analyze_video(frames)
    assert isinstance(out, dict)
    assert set(out.keys()) == {
        "motion_series",
        "keyframes",
        "temporal_encoding",
        "mean_motion",
    }
    assert isinstance(out["motion_series"], np.ndarray)
    assert isinstance(out["keyframes"], list)
    assert isinstance(out["temporal_encoding"], np.ndarray)
    assert out["temporal_encoding"].shape == (16,)


def test_estimate_depth_returns_depth_map_and_stats():
    """model.estimate_depth returns a dict with depth_map and depth_stats."""
    model = _model()
    out = model.estimate_depth(np.random.randn(16, 16))
    assert isinstance(out, dict)
    assert set(out.keys()) == {"depth_map", "depth_stats"}
    assert out["depth_map"].shape == (16, 16)
    stats = out["depth_stats"]
    assert set(stats.keys()) == {"mean", "std", "min", "max"}
    for v in stats.values():
        assert isinstance(v, float)
        assert np.isfinite(v)


# --------------------------------------------------------------------------- #
# Advanced Analytics facade methods
# --------------------------------------------------------------------------- #


def test_infer_cause_returns_dict_with_required_keys():
    """model.infer_cause returns a dict with best_lag/causal_strength/p_value."""
    model = _model()
    cause = np.random.randn(50)
    effect = np.random.randn(50)
    out = model.infer_cause(cause, effect)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"best_lag", "causal_strength", "p_value_approx"}
    assert isinstance(out["best_lag"], int)
    assert isinstance(out["causal_strength"], float)
    assert isinstance(out["p_value_approx"], float)
    assert -1.0 <= out["causal_strength"] <= 1.0
    assert 0.0 <= out["p_value_approx"] <= 1.0


def test_bayesian_update_and_predictive_returns_tuple():
    """model.bayesian_update + bayesian_predictive return a (mean, std) tuple."""
    model = _model()
    model.bayesian_update((5, 10))
    pred = model.bayesian_predictive()
    assert isinstance(pred, tuple)
    assert len(pred) == 2
    mean, std = pred
    assert isinstance(mean, float)
    assert isinstance(std, float)
    # Beta-Binomial posterior mean after (5,10) -> Beta(6, 6) -> mean = 0.5.
    assert 0.0 <= mean <= 1.0
    assert std >= 0.0


def test_detect_change_points_returns_int_ndarray():
    """model.detect_change_points returns an int ndarray of indices."""
    model = _model()
    series = np.array([1, 1, 1, 5, 5, 5, 1, 1, 1], dtype=float)
    cps = model.detect_change_points(series)
    assert isinstance(cps, np.ndarray)
    assert cps.dtype.kind == "i"
    # Each reported index must be in-bounds.
    assert all(0 <= int(i) < len(series) for i in cps.tolist())
