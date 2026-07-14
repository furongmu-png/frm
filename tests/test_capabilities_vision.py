# tests/test_capabilities_vision.py
"""Tests for the zero-data Computer Vision capability module."""

from __future__ import annotations

import numpy as np

from zero_data_model.capabilities.rules import VisionRules
from zero_data_model.capabilities.vision import (
    FeatureExtractor,
    ImageEncoder,
    PatternRecognizer,
    ShapeAnalyzer,
)


def _draw_circle(size: int = 16) -> np.ndarray:
    """Synthesize a filled circle on a square grid (rule-based, no learning)."""
    img = np.zeros((size, size), dtype=float)
    cy, cx = size // 2, size // 2
    yy, xx = np.indices((size, size))
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= (size // 3) ** 2] = 1.0
    return img


def test_image_encoder_shape():
    enc = ImageEncoder(dim=64)
    img = np.random.rand(16, 16)
    vec = enc.encode(img)
    assert vec.shape == (64,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6


def test_image_encoder_deterministic():
    enc = ImageEncoder(dim=64)
    img = np.random.rand(12, 12)
    v1 = enc.encode(img)
    v2 = enc.encode(img)
    np.testing.assert_allclose(v1, v2, atol=1e-8)


def test_feature_extractor_keys():
    fe = FeatureExtractor(dim=64)
    img = np.random.rand(16, 16)
    feats = fe.extract(img)
    assert set(feats.keys()) == {"edges", "texture", "morphology", "stats"}
    for key, val in feats.items():
        assert isinstance(val, np.ndarray), f"{key} must be an ndarray"
        assert val.ndim == 1, f"{key} must be 1D"


def test_pattern_recognizer_returns_shape():
    rec = PatternRecognizer(dim=64)
    img = _draw_circle(16)
    shape_name, conf = rec.recognize(img)
    assert shape_name in VisionRules().shapes
    assert 0.0 <= conf <= 1.0


def test_shape_analyzer_keys():
    sa = ShapeAnalyzer(dim=64)
    img = np.random.rand(16, 16)
    result = sa.analyze(img)
    assert set(result.keys()) == {"aspect_ratio", "symmetry", "complexity", "beti0"}
    for key, val in result.items():
        assert isinstance(val, float), f"{key} must be a float"


def test_shape_analyzer_symmetry_symmetric_input():
    sa = ShapeAnalyzer(dim=64)
    img = np.ones((10, 10))
    result = sa.analyze(img)
    assert result["symmetry"] > 0.9
