# tests/test_capabilities_vision_advanced.py
"""Tests for the advanced zero-data Computer Vision capability module.

The advanced CV capabilities (point-cloud encoding, video frame analysis,
monocular depth estimation) compose the existing core cognitive modules
(math universe, biological substrate, vision rules) with geometric and
statistical priors. No external CV libraries (no OpenCV / PCL) and no learned
weights are required.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.capabilities.vision_advanced import (
    PointCloudEncoder,
    VideoFrameAnalyzer,
    DepthEstimator,
)


@pytest.fixture(autouse=True)
def _deterministic_rng() -> None:
    """Pin the global RNG so the random core-module matrices (CA, math
    universe fractal transforms, ...) used during encoding are reproducible."""
    np.random.seed(42)


def test_point_cloud_encoder_shape_and_norm():
    """encode(Nx3 points) returns a dim-length, L2-normalized vector."""
    enc = PointCloudEncoder(dim=64)
    pts = np.random.rand(50, 3)
    vec = enc.encode(pts)
    assert vec.shape == (64,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6


def test_point_cloud_encoder_deterministic():
    """Encoding the same point cloud twice yields identical vectors."""
    enc = PointCloudEncoder(dim=64)
    pts = np.random.rand(30, 3)
    v1 = enc.encode(pts)
    v2 = enc.encode(pts)
    np.testing.assert_allclose(v1, v2, atol=1e-8)


def test_point_cloud_encoder_handles_small_input():
    """A 1-point cloud still produces a dim-length finite vector."""
    enc = PointCloudEncoder(dim=64)
    vec = enc.encode(np.array([[0.5, 0.5, 0.5]]))
    assert vec.shape == (64,)
    assert np.all(np.isfinite(vec))


def test_video_frame_analyzer_keys():
    """analyze(frames) returns the four required keys with correct types."""
    vfa = VideoFrameAnalyzer(dim=64)
    frames = [np.random.rand(8, 8) for _ in range(6)]
    out = vfa.analyze(frames)
    assert set(out.keys()) == {
        "motion_series",
        "keyframes",
        "temporal_encoding",
        "mean_motion",
    }
    assert isinstance(out["motion_series"], np.ndarray)
    assert isinstance(out["keyframes"], list)
    assert isinstance(out["temporal_encoding"], np.ndarray)
    assert isinstance(out["mean_motion"], float)


def test_video_frame_analyzer_motion_length():
    """motion_series has length len(frames) - 1 (one diff per pair)."""
    vfa = VideoFrameAnalyzer(dim=64)
    frames = [np.random.rand(8, 8) for _ in range(5)]
    out = vfa.analyze(frames)
    assert out["motion_series"].shape == (4,)
    assert out["temporal_encoding"].shape == (64,)


def test_video_frame_analyzer_keyframes_nonempty_with_shift():
    """A clear mid-sequence shift produces at least one detected keyframe."""
    vfa = VideoFrameAnalyzer(dim=64)
    # First three frames are near-zero; last three are a large constant.
    frames = [np.zeros((8, 8))] * 3 + [np.ones((8, 8)) * 10.0] * 3
    out = vfa.analyze(frames)
    assert len(out["keyframes"]) >= 1
    # The first big motion happens between frame 2 (index 2 -> 3), so the
    # first keyframe should be at frame index 3.
    assert out["keyframes"][0] == 3


def test_depth_estimator_returns_depth_map_with_stats():
    """estimate(image) returns a depth_map (same shape) and depth_stats dict."""
    de = DepthEstimator(dim=64)
    img = np.random.rand(16, 16)
    out = de.estimate(img)
    assert set(out.keys()) == {"depth_map", "depth_stats"}
    assert out["depth_map"].shape == img.shape
    assert set(out["depth_stats"].keys()) == {"mean", "std", "min", "max"}
    for k, v in out["depth_stats"].items():
        assert isinstance(v, float), f"{k} must be a float"
        assert np.isfinite(v), f"{k} must be finite"
    # Depth map should be in [0, 1] after normalization.
    assert float(out["depth_map"].min()) >= 0.0
    assert float(out["depth_map"].max()) <= 1.0


def test_depth_estimator_handles_small_image():
    """A small 2x2 image still produces a finite depth map without raising."""
    de = DepthEstimator(dim=64)
    img = np.array([[0.1, 0.5], [0.9, 0.2]])
    out = de.estimate(img)
    assert out["depth_map"].shape == (2, 2)
    assert np.all(np.isfinite(out["depth_map"]))


def test_point_cloud_encoder_empty_input():
    """An empty point cloud returns a zero vector instead of raising."""
    enc = PointCloudEncoder(dim=64)
    vec = enc.encode(np.zeros((0, 3)))
    assert vec.shape == (64,)
    assert np.all(vec == 0.0)
