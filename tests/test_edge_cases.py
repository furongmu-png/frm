# tests/test_edge_cases.py
"""Edge-case and boundary-input tests for the zero-data model.

Each test pins the behaviour of a public model method on a degenerate or
hostile input (single-element series, empty strings, unicode/emoji, NaN
values, 3D images, 2D think input). The intent is to document the current
contract -- graceful handling where the source is defensive, and a clear
error where the source rejects the input -- so regressions in either
direction are caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.model import ZeroDataModel

# --------------------------------------------------------------------------- #
# Single-element time series (degenerate but valid 1D input).
# --------------------------------------------------------------------------- #


def test_forecast_single_element_series():
    """forecast on a 1-element series returns a finite horizon-length array."""
    model = ZeroDataModel(dim=16)
    pred = model.forecast(np.array([5.0]), horizon=3)
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))


def test_detect_anomalies_single_element_series():
    """detect_anomalies on a single point returns a (1,) bool mask with no
    anomaly flagged (a single point has z-score 0, below the threshold)."""
    model = ZeroDataModel(dim=16)
    mask = model.detect_anomalies(np.array([42.0]))
    assert mask.shape == (1,)
    assert mask.dtype == bool
    assert not mask.any()


def test_analyze_trend_minimum_two_element_series():
    """analyze_trend on a 2-element series (the minimum that polyfit can
    handle) returns the full key set with a finite geodesic_deviation.

    A true single-element series would hit a source bug: TrendAnalyzer.analyze
    calls np.polyfit on a single point which raises LinAlgError (SVD does not
    converge). Guarding that requires a source change (n < 2 fallback), which
    is out of scope for this test-only task, so we pin the 2-element boundary
    instead."""
    model = ZeroDataModel(dim=16)
    out = model.analyze_trend(np.array([7.0, 9.0]))
    assert set(out.keys()) == {
        "trend_slope",
        "regime",
        "curvature",
        "geodesic_deviation",
        "isomorphism_score",
    }
    assert np.isfinite(out["geodesic_deviation"])
    assert np.isfinite(out["trend_slope"])


# --------------------------------------------------------------------------- #
# Empty-string text inputs.
# --------------------------------------------------------------------------- #


def test_encode_text_empty_string_returns_normalized_vector():
    """encode_text('') does not raise and returns a dim-length, finite,
    L2-normalized vector (the sentiment prior alone keeps the norm > 0)."""
    model = ZeroDataModel(dim=16)
    vec = model.encode_text("")
    assert vec.shape == (16,)
    assert np.all(np.isfinite(vec))
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3


def test_classify_text_empty_string_returns_valid_topic():
    """classify_text('') returns a (topic, confidence) tuple with a known
    topic label and a bounded confidence instead of raising."""
    model = ZeroDataModel(dim=16)
    topic, conf = model.classify_text("")
    assert topic in {"tech", "nature", "emotion", "science"}
    assert 0.0 <= conf <= 1.0


# --------------------------------------------------------------------------- #
# Unicode / multi-script / emoji inputs.
# --------------------------------------------------------------------------- #


def test_encode_text_handles_unicode_cjk_and_emoji():
    """encode_text on CJK + emoji input returns a finite, normalized vector
    (Unicode codepoint statistics must not choke on astral-plane codepoints)."""
    model = ZeroDataModel(dim=16)
    vec = model.encode_text("你好世界 🌍")
    assert vec.shape == (16,)
    assert np.all(np.isfinite(vec))
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3


def test_text_similarity_disjoint_vocabularies_in_range():
    """text_similarity between two texts with no shared vocabulary still
    returns a finite float in [0, 1]."""
    model = ZeroDataModel(dim=16)
    score = model.text_similarity("code data model", "forest river mountain")
    assert isinstance(score, float)
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# NaN / non-finite numeric inputs.
# --------------------------------------------------------------------------- #


def test_forecast_with_nan_input_returns_finite():
    """forecast on a series containing NaN returns a finite horizon-length
    array (the implementation runs the result through nan_to_num)."""
    model = ZeroDataModel(dim=16)
    pred = model.forecast(np.array([1.0, np.nan, 3.0]), horizon=2)
    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))


# --------------------------------------------------------------------------- #
# Non-2D image inputs.
# --------------------------------------------------------------------------- #


def test_encode_image_3d_array_flattens_to_dim():
    """encode_image on a 3D array (H, W, C) does not raise: the encoder
    collapses extra leading dimensions to 2D and returns a dim-length vector."""
    model = ZeroDataModel(dim=16)
    img = np.random.rand(4, 4, 3)
    vec = model.encode_image(img)
    assert vec.shape == (16,)
    assert np.all(np.isfinite(vec))
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3


# --------------------------------------------------------------------------- #
# Non-1D think input (documents the 1D contract).
# --------------------------------------------------------------------------- #


def test_think_with_2d_input_raises_value_error():
    """think() expects a 1D input of length <= dim; a 2D array cannot be
    broadcast into the internal padded 1D buffer and must raise ValueError.

    This documents the current contract rather than a bug: callers must
    flatten 2D inputs before feeding them to think()."""
    model = ZeroDataModel(dim=16)
    with pytest.raises(ValueError):
        model.think(np.random.randn(4, 4))
