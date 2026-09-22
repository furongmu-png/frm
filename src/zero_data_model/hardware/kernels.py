"""Numba-JIT numeric kernels for the zero-data cognitive model.

Each kernel is a ``@njit(cache=True, nogil=True)`` function that takes only numpy arrays
and primitive types, returning numpy arrays or scalars. When numba is not
installed, pure-numpy reference implementations are exposed under the same
names so callers can ``try: from .kernels import _foo`` and degrade
gracefully.

Kernel index
------------
- ``_predictive_layer_forward(x, W, b, activation)`` — forward pass with tanh/relu.
- ``_cosine_similarity(a, b)`` — cosine similarity for find_isomorphism.
- ``_topos_classify(x, classifier)`` — sigmoid(x @ classifier).
- ``_cellular_automata_step(state, rule, size)`` — Wolfram rule over all cells in one JIT pass.
- ``_morphogenetic_laplacian(grid)`` — 5-point Laplacian (no temporaries).
- ``_kl_divergence(p_abs, q_abs)`` — KL divergence on already-|x|+eps inputs.
- ``_betti_numbers(sorted_vals, max_radius)`` — gap-detection persistent homology loop.
- ``_fractal_generate(x, scales, offsets, n_iterations)`` — fused matmul+tanh iterations.
- ``_quantum_classical_forward(x, W)`` — tanh(x @ W) for QuantumClassicalHybrid.
- ``_skewness(data)`` — Fisher-Pearson biased sample skewness (replaces scipy.stats.skew).

A module-level ``HAS_NUMBA`` flag is exported so callers can advertise which
path is active (mirrors the existing ``hardware/quantum.py`` pattern).
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """Pure-numpy fallback decorator: returns the function unchanged."""

        def _decorator(fn):
            return fn

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _decorator


# ---------------------------------------------------------------------------
# PredictiveLayer forward pass
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _predictive_layer_forward(x, W, b, activation):
    """Forward pass: ``z = x @ W + b`` then tanh or relu (clip to >=0)."""
    z = x @ W + b
    if activation == "tanh":
        return np.tanh(z)
    return np.maximum(z, 0.0)


# ---------------------------------------------------------------------------
# Cosine similarity (find_isomorphism)
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _cosine_similarity(a, b):
    """Cosine similarity ``dot(a,b) / (|a||b| + 1e-8)`` — single JIT pass."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    n = len(a)
    for i in range(n):
        ai = a[i]
        bi = b[i]
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi
    return dot / (np.sqrt(norm_a) * np.sqrt(norm_b) + 1e-8)


# ---------------------------------------------------------------------------
# Topos classifier (sigmoid)
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _topos_classify(x, classifier):
    """Element-wise sigmoid of ``x @ classifier``.

    M5+ 修复：数值稳定的 sigmoid。原代码 np.exp(-z[i]) 对极大
    负 z 溢出为 inf 并触发 RuntimeWarning。改用分段公式：
      z >= 0: 1 / (1 + exp(-z))   — exp(-z) ∈ (0, 1]，无溢出
      z <  0: exp(z) / (1 + exp(z)) — exp(z) ∈ (0, 1)，无溢出
    """
    z = x @ classifier
    out = np.empty_like(z)
    n = len(z)
    for i in range(n):
        zi = z[i]
        if zi >= 0.0:
            out[i] = 1.0 / (1.0 + np.exp(-zi))
        else:
            ez = np.exp(zi)
            out[i] = ez / (1.0 + ez)
    return out


# ---------------------------------------------------------------------------
# Cellular automaton — Wolfram rule over all cells in one JIT pass.
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _cellular_automata_step(state, rule, size):
    """Apply a Wolfram elementary CA rule to every cell, returning the new state.

    Boundary is periodic (matches the original ``% size`` indexing).

    F8 修复：位运算 (<<, |, >>) 要求整数操作数。若 state 是
    float64 数组（如从物理沙盒传入），numba 会抛 TypeError。
    在函数入口强制转换为 int64 以保证位运算安全。
    """
    state = state.astype(np.int64)
    new_state = np.zeros(size, dtype=np.int64)
    for i in range(size):
        left = state[(i - 1) % size]
        center = state[i]
        right = state[(i + 1) % size]
        index = (left << 2) | (center << 1) | right
        new_state[i] = (rule >> index) & 1
    return new_state


# ---------------------------------------------------------------------------
# Morphogenetic 5-point Laplacian (fused — no np.roll temporaries).
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _morphogenetic_laplacian(grid):
    """5-point periodic Laplacian on a 2D grid (matches np.roll boundary wrap)."""
    n = grid.shape[0]
    m = grid.shape[1]
    laplacian = np.empty_like(grid)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        for j in range(m):
            jp1 = (j + 1) % m
            jm1 = (j - 1) % m
            laplacian[i, j] = (
                grid[ip1, j]
                + grid[im1, j]
                + grid[i, jp1]
                + grid[i, jm1]
                - 4.0 * grid[i, j]
            )
    return laplacian


# ---------------------------------------------------------------------------
# KL divergence (assumes inputs are already |x| + 1e-8).
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _kl_divergence(p_abs, q_abs):
    """KL(p || q) for already-absolute, eps-perturbed inputs.

    The caller is responsible for ``p_abs = np.abs(p[:dim]) + 1e-8`` (and
    likewise for ``q_abs``) — matching the original InformationGeometry.kl.
    """
    n = len(p_abs)
    p_sum = 0.0
    q_sum = 0.0
    for i in range(n):
        p_sum += p_abs[i]
        q_sum += q_abs[i]
    total = 0.0
    for i in range(n):
        p_norm = p_abs[i] / p_sum
        q_norm = q_abs[i] / q_sum
        total += p_norm * np.log(p_norm / q_norm)
    return total


# ---------------------------------------------------------------------------
# Betti numbers via gap detection.
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _betti_numbers(sorted_vals, max_radius):
    """Compute (betti_0, betti_1) from sorted values + gap threshold.

    Round-9 audit R9-001: the threshold is now the raw ``max_radius``
    (matching ``connected_components_1d``'s documented contract: two
    sorted points are in the same component iff the gap between them is
    ``<= max_radius``). The previous ``max_radius / n_points`` shrank as
    the point count grew, making the count meaningless for large inputs.

    Round-10 audit R10-A-008: ``betti_1`` is now always 0. A 1D point
    cloud's Vietoris-Rips complex is at most a 1-dimensional simplicial
    complex (vertices + edges); it has no 2-simplices, so its first
    homology group ``H_1`` is trivially zero (no 1-loops). The previous
    ``betti_1 = n_points - betti_0`` formula computed an unrelated
    quantity (it would return 9 for a 10-point cloud with 1 component),
    violating the function's documented contract. The only current
    caller (``connected_components_1d``) discards ``betti_1`` via
    ``betti_0, _ = _betti_numbers(...)``, so the change is behaviorally
    a no-op for active callers but honors the contract for future use.
    """
    n_points = len(sorted_vals)
    betti_0 = 1
    for i in range(1, n_points):
        gap = sorted_vals[i] - sorted_vals[i - 1]
        if gap > max_radius:
            betti_0 += 1
    # 1D point clouds have no 1-loops: the first Betti number is 0.
    return betti_0, 0


# ---------------------------------------------------------------------------
# Fractal generator — fused matmul + tanh over n_iterations.
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _fractal_generate(x, scales, offsets, n_iterations, n_transforms):
    """Iteratively apply ``x = tanh(scales[t % n_t] @ x + offsets[t % n_t])``."""
    out = x.copy()
    for it in range(n_iterations):
        t_idx = it % n_transforms
        # matmul of (dim, dim) @ (dim,) -> (dim,) plus bias.
        new_out = scales[t_idx] @ out + offsets[t_idx]
        for i in range(len(out)):
            out[i] = np.tanh(new_out[i])
    return out


# ---------------------------------------------------------------------------
# Quantum-classical hybrid forward (tanh(x @ W)).
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _quantum_classical_forward(x, W):
    """``tanh(x @ W)`` for the QuantumClassicalHybrid classical path."""
    return np.tanh(x @ W)


# ---------------------------------------------------------------------------
# Sample skewness (replaces scipy.stats.skew in TopologicalAnalyzer).
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _skewness(data):
    """Fisher-Pearson biased sample skewness, matching ``scipy.stats.skew(data)``.

    Computes ``g1 = (m3 / n) / (m2 / n) ** 1.5`` where ``m2`` and ``m3`` are the
    second and third central moments — i.e. scipy's default (``bias=True``)
    estimator, to within ~1e-15. Returns ``0.0`` for ``n < 3`` or for a constant
    input (zero variance), which also fixes the NaN scipy produces in that case.

    When numba is unavailable the ``njit`` decorator degrades to a no-op, so this
    same body serves as the pure-numpy fallback.
    """
    n = len(data)
    if n < 3:
        return 0.0
    mean = 0.0
    for i in range(n):
        mean += data[i]
    mean /= n
    m2 = 0.0  # sum of squared deviations
    m3 = 0.0  # sum of cubed deviations
    for i in range(n):
        diff = data[i] - mean
        m2 += diff * diff
        m3 += diff * diff * diff
    if m2 == 0.0:
        # Constant array: zero variance -> skewness undefined; return 0.0
        # (avoids the NaN scipy.stats.skew yields here).
        return 0.0
    # Biased sample skewness g1 (scipy.stats.skew default, bias=True).
    return (m3 / n) / ((m2 / n) ** 1.5)
