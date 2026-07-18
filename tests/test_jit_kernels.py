# tests/test_jit_kernels.py
"""Tests for the numba-JIT kernels in ``hardware.kernels``.

Each test compares the JIT path against a pure-numpy reference (or the
original module method) on a fixed input and asserts numerical equality
within ``1e-6``. A timing-based test asserts the CellularAutomata JIT path
delivers at least a 2x speedup over the original Python per-cell loop.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.stats

from zero_data_model.biological import CellularAutomata
from zero_data_model.hardware import kernels

# ---------------------------------------------------------------------------
# Kernel equivalence vs. pure-numpy reference
# ---------------------------------------------------------------------------


def test_predictive_layer_forward_tanh_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16)
    W = rng.standard_normal((16, 16))
    b = rng.standard_normal(16)
    expected = np.tanh(x @ W + b)
    got = kernels._predictive_layer_forward(
        np.ascontiguousarray(x, dtype=float),
        np.ascontiguousarray(W, dtype=float),
        np.ascontiguousarray(b, dtype=float),
        "tanh",
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_predictive_layer_forward_relu_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(16)
    W = rng.standard_normal((16, 16))
    b = rng.standard_normal(16)
    # Original code path: np.clip(z, 0, None) is relu.
    expected = np.clip(x @ W + b, 0, None)
    got = kernels._predictive_layer_forward(
        np.ascontiguousarray(x, dtype=float),
        np.ascontiguousarray(W, dtype=float),
        np.ascontiguousarray(b, dtype=float),
        "relu",
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_cosine_similarity_matches_numpy():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(32)
    b = rng.standard_normal(32)
    expected = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    got = float(kernels._cosine_similarity(
        np.ascontiguousarray(a, dtype=float),
        np.ascontiguousarray(b, dtype=float),
    ))
    assert abs(expected - got) < 1e-6


def test_topos_classify_matches_numpy():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(16)
    classifier = rng.standard_normal((16, 16))
    expected = 1.0 / (1.0 + np.exp(-(x @ classifier)))
    got = kernels._topos_classify(
        np.ascontiguousarray(x, dtype=float),
        np.ascontiguousarray(classifier, dtype=float),
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_cellular_automata_step_matches_python_loop():
    rng = np.random.default_rng(4)
    size = 64
    rule = 30
    state = rng.integers(0, 2, size)

    # Reference: the original per-cell Python implementation.
    expected = np.zeros(size, dtype=int)
    for i in range(size):
        left = state[(i - 1) % size]
        center = state[i]
        right = state[(i + 1) % size]
        index = (left << 2) | (center << 1) | right
        expected[i] = (rule >> index) & 1

    got = kernels._cellular_automata_step(
        np.ascontiguousarray(state),
        int(rule),
        int(size),
    )
    np.testing.assert_array_equal(got, expected)


def test_morphogenetic_laplacian_matches_np_roll():
    rng = np.random.default_rng(5)
    grid = rng.standard_normal((8, 8))
    expected = (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0)
        + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)
        - 4 * grid
    )
    got = kernels._morphogenetic_laplacian(np.ascontiguousarray(grid, dtype=float))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_kl_divergence_matches_numpy():
    rng = np.random.default_rng(6)
    p = np.abs(rng.standard_normal(16)) + 1e-8
    q = np.abs(rng.standard_normal(16)) + 1e-8
    p_norm = p / p.sum()
    q_norm = q / q.sum()
    expected = float(np.sum(p_norm * np.log(p_norm / q_norm)))
    got = float(kernels._kl_divergence(
        np.ascontiguousarray(p, dtype=float),
        np.ascontiguousarray(q, dtype=float),
    ))
    assert abs(expected - got) < 1e-6


def test_betti_numbers_matches_python_loop():
    rng = np.random.default_rng(7)
    data = np.sort(rng.standard_normal(16))
    max_radius = 1.0
    n_points = len(data)

    # Round-9 audit R9-001: reference uses the documented contract —
    # two sorted points are in the same component iff the gap between
    # them is ``<= max_radius`` (NOT ``max_radius / n_points``).
    expected_b0 = 1
    for i in range(1, n_points):
        gap = data[i] - data[i - 1]
        if gap > max_radius:
            expected_b0 += 1
    expected_b1 = max(0, n_points - expected_b0)

    got_b0, got_b1 = kernels._betti_numbers(
        np.ascontiguousarray(data, dtype=float),
        float(max_radius),
    )
    assert (got_b0, got_b1) == (expected_b0, expected_b1)


def test_fractal_generate_matches_python_loop():
    rng = np.random.default_rng(8)
    x = rng.standard_normal(16)
    scales = rng.standard_normal((4, 16, 16))
    offsets = rng.standard_normal((4, 16))
    n_iterations = 5

    expected = x.copy()
    for it in range(n_iterations):
        t_idx = it % 4
        expected = scales[t_idx] @ expected + offsets[t_idx]
        expected = np.tanh(expected)

    got = kernels._fractal_generate(
        np.ascontiguousarray(x, dtype=float),
        np.ascontiguousarray(scales, dtype=float),
        np.ascontiguousarray(offsets, dtype=float),
        int(n_iterations),
        4,
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_quantum_classical_forward_matches_numpy():
    rng = np.random.default_rng(9)
    x = rng.standard_normal(16)
    W = rng.standard_normal((16, 16))
    expected = np.tanh(x @ W)
    got = kernels._quantum_classical_forward(
        np.ascontiguousarray(x, dtype=float),
        np.ascontiguousarray(W, dtype=float),
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_skewness_matches_scipy():
    """_skewness must match scipy.stats.skew default (bias=True) to < 1e-10."""
    rng = np.random.default_rng(10)
    for n in (3, 5, 16, 64, 100, 500):
        data = rng.standard_normal(n)
        expected = float(scipy.stats.skew(data))
        got = float(kernels._skewness(np.ascontiguousarray(data, dtype=float)))
        assert abs(expected - got) < 1e-10, (
            f"n={n}: scipy={expected!r}, kernel={got!r}, diff={abs(expected - got)!r}"
        )


def test_skewness_constant_array_no_nan():
    """A constant array has zero variance -> skewness must be 0.0, never NaN.

    This also covers the pre-existing bug where scipy.stats.skew returned NaN
    on constant input inside TopologicalAnalyzer.topological_features.
    """
    got = float(kernels._skewness(np.ascontiguousarray(np.ones(10), dtype=float)))
    assert got == 0.0
    assert np.isfinite(got)


def test_skewness_small_array():
    """Arrays with n < 3 must return 0.0 (skewness undefined)."""
    assert float(kernels._skewness(np.ascontiguousarray(np.array([1.0, 2.0]), dtype=float))) == 0.0
    assert float(kernels._skewness(np.ascontiguousarray(np.array([5.0]), dtype=float))) == 0.0
    assert float(kernels._skewness(np.ascontiguousarray(np.array([], dtype=float)))) == 0.0


# ---------------------------------------------------------------------------
# Performance regression: CellularAutomata.evolve must be >= 2x faster.
# ---------------------------------------------------------------------------


def _cellular_automata_evolve_pure_python(state, rule, size, n_steps):
    """Pure-python per-cell reference, mirroring the pre-JIT implementation."""
    history = [state.copy()]
    cur = state
    for _ in range(n_steps):
        new_state = np.zeros(size, dtype=int)
        for i in range(size):
            left = cur[(i - 1) % size]
            center = cur[i]
            right = cur[(i + 1) % size]
            index = (left << 2) | (center << 1) | right
            new_state[i] = (rule >> index) & 1
        cur = new_state
        history.append(cur.copy())
    return np.array(history)


def test_cellular_automata_jit_is_at_least_2x_faster():
    """The JIT path should be >= 2x faster than the pure-Python per-cell loop."""
    if not kernels.HAS_NUMBA:
        import pytest
        pytest.skip("numba not installed; JIT speedup not measurable")

    rng = np.random.default_rng(123)
    size = 256
    n_steps = 50
    rule = 30
    state0 = rng.integers(0, 2, size)

    # Warm up the JIT cache so the first compile time isn't measured.
    warm_ca = CellularAutomata(size=size, rule=rule)
    warm_ca.state = state0.copy()
    warm_ca.evolve(n_steps=2)

    # Time the JIT path (uses the wired-in kernel via CellularAutomata.step).
    ca = CellularAutomata(size=size, rule=rule)
    ca.state = state0.copy()
    t0 = time.perf_counter()
    jit_history = ca.evolve(n_steps=n_steps)
    t_jit = time.perf_counter() - t0

    # Time the pure-python reference.
    t0 = time.perf_counter()
    ref_history = _cellular_automata_evolve_pure_python(state0, rule, size, n_steps)
    t_ref = time.perf_counter() - t0

    # Same result.
    np.testing.assert_array_equal(jit_history, ref_history)

    speedup = t_ref / t_jit if t_jit > 0 else 0
    assert speedup >= 2.0, (
        f"JIT CellularAutomata.evolve not 2x faster: ref={t_ref*1000:.2f}ms, "
        f"jit={t_jit*1000:.2f}ms, speedup={speedup:.2f}x"
    )


# ---------------------------------------------------------------------------
# Integration: model.think() still produces finite output of the right shape.
# ---------------------------------------------------------------------------


def test_model_think_output_finite_and_correct_shape_after_jit():
    from zero_data_model.model import ZeroDataModel

    rng = np.random.default_rng(99)
    model = ZeroDataModel(dim=32)
    # Warm up the JIT caches on the small dim. Use non-zero random input so
    # the existing scipy.stats.skew path on constant arrays doesn't NaN-out
    # (pre-existing behaviour unrelated to JIT integration).
    model.think(rng.standard_normal(32))

    out = model.think(rng.standard_normal(32))
    assert out.data.shape == (32,)
    assert np.all(np.isfinite(out.data)), "think() output contained non-finite values"
    assert out.metadata["cycle"] >= 1


def test_jit_kernels_module_imports_with_numba_present():
    """The kernels module should expose the documented JIT functions."""
    assert hasattr(kernels, "_predictive_layer_forward")
    assert hasattr(kernels, "_cosine_similarity")
    assert hasattr(kernels, "_topos_classify")
    assert hasattr(kernels, "_cellular_automata_step")
    assert hasattr(kernels, "_morphogenetic_laplacian")
    assert hasattr(kernels, "_kl_divergence")
    assert hasattr(kernels, "_betti_numbers")
    assert hasattr(kernels, "_fractal_generate")
    assert hasattr(kernels, "_quantum_classical_forward")
    # In this environment numba is installed, so the JIT path should be active.
    assert kernels.HAS_NUMBA is True


def test_core_modules_advertise_jit_path():
    """The 6 core modules should expose a JIT flag indicating the kernel path."""
    from zero_data_model import (
        biological,
        category_engine,
        consciousness_core,
        math_universe,
        quantum_hybrid,
    )

    # Each module sets a module-level _HAS_JIT (or _HAS_KERNELS_JIT) flag.
    assert consciousness_core._HAS_JIT is True
    assert category_engine._HAS_JIT is True
    assert biological._HAS_JIT is True
    assert math_universe._HAS_JIT is True
    assert quantum_hybrid._HAS_KERNELS_JIT is True
