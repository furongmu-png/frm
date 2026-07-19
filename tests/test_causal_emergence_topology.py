# tests/test_causal_emergence_topology.py
"""Tests for module A: PersistentHomologyPerceiver.

Covers correctness (known-answer tests on circle, disc, two clusters),
edge cases (empty, single point, NaN, large input), determinism, and
API contract per spec §3 and §10.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.causal_emergence import (
    EmergenceRules,
    PersistentHomologyPerceiver,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# ----------------------------------------------------------------------
# Correctness: known-answer tests
# ----------------------------------------------------------------------

def _circle_points(n: int = 16) -> np.ndarray:
    """Sample n points uniformly on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _disc_points(n: int = 16, rng=None) -> np.ndarray:
    """Sample n points uniformly in the unit disc."""
    rng = rng or np.random.default_rng(0)
    r = np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def test_circle_betti_numbers():
    """Unit circle sampled at 16 points -> betti = [1, 1, 0]."""
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules())
    points = _circle_points(16)
    result = perceiver.perceive(points)
    assert result["betti_numbers"][0] == 1, "circle has 1 connected component"
    assert result["betti_numbers"][1] >= 1, "circle has at least 1 loop"
    assert result["betti_numbers"][2] == 0, "circle has no voids"


def test_disc_betti_numbers():
    """Solid disc -> betti = [1, 0, 0] (connected, no loops, no voids)."""
    rng = np.random.default_rng(7)
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules(), rng=rng)
    points = _disc_points(16, rng=rng)
    result = perceiver.perceive(points)
    assert result["betti_numbers"][0] == 1
    assert result["betti_numbers"][1] == 0
    assert result["betti_numbers"][2] == 0


def test_two_clusters_betti_zero():
    """Two separated clusters -> betti_0 = 2."""
    rng = np.random.default_rng(11)
    cluster_a = rng.standard_normal((8, 2)) * 0.1
    cluster_b = rng.standard_normal((8, 2)) * 0.1 + np.array([10.0, 10.0])
    points = np.vstack([cluster_a, cluster_b])
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules(), rng=rng)
    result = perceiver.perceive(points)
    assert result["betti_numbers"][0] == 2, f"expected 2 components, got {result['betti_numbers']}"


def test_euler_characteristic_circle():
    """Circle: chi = betti_0 - betti_1 + betti_2 = 1 - 1 + 0 = 0."""
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules())
    points = _circle_points(16)
    result = perceiver.perceive(points)
    b = result["betti_numbers"]
    expected_euler = b[0] - b[1] + (b[2] if len(b) > 2 else 0)
    assert result["euler_characteristic"] == expected_euler


def test_persistence_diagram_format():
    """Persistence diagram is a list of (dim, birth, death) tuples."""
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules())
    result = perceiver.perceive(_circle_points(16))
    diagram = result["persistence_diagram"]
    assert isinstance(diagram, list)
    for entry in diagram:
        assert len(entry) == 3
        d, b, death = entry
        assert isinstance(d, int)
        assert isinstance(b, float)
        assert death == float("inf") or isinstance(death, float)


def test_persistence_entropy_in_range():
    """Persistence entropy is in [0, 1]."""
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules())
    for _ in range(5):
        rng = np.random.default_rng(_)
        points = rng.standard_normal((16, 2))
        result = perceiver.perceive(points)
        assert 0.0 <= result["persistence_entropy"] <= 1.0


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

def test_empty_input():
    """Empty input returns all-zero invariants."""
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules())
    result = perceiver.perceive(np.array([]))
    assert result["betti_numbers"] == [0, 0, 0]
    assert result["euler_characteristic"] == 0
    assert result["n_points"] == 0
    assert result["max_eps"] == 0.0


def test_single_point():
    """Single point -> betti = [1, 0, 0]."""
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules())
    result = perceiver.perceive(np.array([1.0, 2.0, 3.0]))
    assert result["betti_numbers"] == [1, 0, 0]
    assert result["n_points"] == 1
    assert result["max_eps"] == 0.0


def test_nan_input_guards():
    """NaN/Inf in input is sanitized (no exception, finite output)."""
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules())
    points = np.array([[0.0, 0.0], [1.0, np.nan], [np.inf, 2.0], [-np.inf, 3.0]])
    result = perceiver.perceive(points)
    assert np.isfinite(result["persistence_entropy"])
    assert np.isfinite(result["max_eps"])
    assert isinstance(result["betti_numbers"], list)


def test_large_input_subsampled():
    """Input > max_points is subsampled to max_points."""
    rng = np.random.default_rng(99)
    perceiver = PersistentHomologyPerceiver(
        rules=EmergenceRules(), rng=rng
    )
    points = rng.standard_normal((100, 4))
    result = perceiver.perceive(points)
    assert result["n_points"] == 16  # default topology_max_points


def test_max_points_configurable():
    """topology_max_points rules field controls subsampling cap."""
    rng = np.random.default_rng(0)
    rules = EmergenceRules(topology_max_points=8)
    perceiver = PersistentHomologyPerceiver(rules=rules, rng=rng)
    points = rng.standard_normal((50, 3))
    result = perceiver.perceive(points)
    assert result["n_points"] == 8


# ----------------------------------------------------------------------
# API contract
# ----------------------------------------------------------------------

def test_return_dict_keys():
    """Return dict has all expected keys."""
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules())
    result = perceiver.perceive(_circle_points(16))
    expected_keys = {
        "betti_numbers",
        "persistence_diagram",
        "persistence_entropy",
        "euler_characteristic",
        "n_points",
        "max_eps",
    }
    assert set(result.keys()) == expected_keys


def test_max_dim_parameter():
    """max_dim parameter controls the number of Betti numbers returned."""
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules())
    result = perceiver.perceive(_circle_points(16), max_dim=1)
    assert len(result["betti_numbers"]) == 2  # [betti_0, betti_1]
    result2 = perceiver.perceive(_circle_points(16), max_dim=2)
    assert len(result2["betti_numbers"]) == 3


def test_max_dim_parameter_overrides_rules():
    """Method max_dim parameter takes precedence over rules (fix L1)."""
    rules = EmergenceRules(topology_max_dim=1)
    perceiver = PersistentHomologyPerceiver(rules=rules)
    # Pass max_dim=2 explicitly; should return 3 betti numbers
    result = perceiver.perceive(_circle_points(16), max_dim=2)
    assert len(result["betti_numbers"]) == 3


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_deterministic_with_seed():
    """Same rng seed produces identical output."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    p1 = PersistentHomologyPerceiver(rules=EmergenceRules(), rng=rng1)
    p2 = PersistentHomologyPerceiver(rules=EmergenceRules(), rng=rng2)
    points = np.random.default_rng(0).standard_normal((50, 3))
    r1 = p1.perceive(points)
    r2 = p2.perceive(points)
    assert r1["betti_numbers"] == r2["betti_numbers"]
    assert r1["n_points"] == r2["n_points"]


def test_dtype_normalization():
    """Non-float dtype is normalized to float internally."""
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules())
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=int)
    result = perceiver.perceive(points)
    assert isinstance(result["betti_numbers"], list)


def test_non_contiguous_input():
    """Non-contiguous arrays are handled via np.ascontiguousarray."""
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules())
    np.arange(32, dtype=float).reshape(16, 2)[:, ::2]  # non-contiguous
    # Make a 2D non-contiguous slice
    big = np.random.default_rng(0).standard_normal((20, 4))
    non_contig = big[::2, 1:3]
    result = perceiver.perceive(non_contig)
    assert "betti_numbers" in result


# ----------------------------------------------------------------------
# Performance (spec §10.4)
# ----------------------------------------------------------------------

def test_performance_under_2_seconds():
    """16-point cloud computes in < 2 seconds (spec §10.4)."""
    import time
    rng = np.random.default_rng(0)
    perceiver = PersistentHomologyPerceiver(rules=EmergenceRules(), rng=rng)
    points = rng.standard_normal((16, 64))
    start = time.perf_counter()
    perceiver.perceive(points)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"topology took {elapsed:.2f}s, expected < 2s"
