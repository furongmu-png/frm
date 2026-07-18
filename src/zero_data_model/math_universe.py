# src/zero_data_model/math_universe.py
"""Mathematical Universe Layer — Information Geometry, Topological Data Analysis, Fractals."""

from __future__ import annotations

import numpy as np

from .base import CognitiveModule, Prediction, Signal

# JIT kernels -- graceful fallback to pure numpy if numba missing.
try:
    from .hardware.kernels import (
        _betti_numbers,
        _fractal_generate,
        _kl_divergence,
        _skewness,
    )
    _HAS_JIT = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_JIT = False


class InformationGeometry:
    """Fisher information metric and geodesics on probability simplices."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def fisher_metric(self, distribution: np.ndarray) -> np.ndarray:
        p = np.abs(distribution[: self.dim]) + 1e-8
        p = p / np.sum(p)
        return 1.0 / p

    def geodesic(self, p: np.ndarray, q: np.ndarray, t: float = 0.5) -> np.ndarray:
        p_abs = np.abs(p[: self.dim]) + 1e-8
        q_abs = np.abs(q[: self.dim]) + 1e-8
        p_sqrt = np.sqrt(p_abs / np.sum(p_abs))
        q_sqrt = np.sqrt(q_abs / np.sum(q_abs))
        interp = (1 - t) * p_sqrt + t * q_sqrt
        result = interp ** 2
        return result / (np.sum(result) + 1e-8)

    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        # Pad both inputs to ``self.dim`` with zeros so callers passing arrays
        # of differing lengths (e.g. TrendAnalyzer's odd-length halves) no
        # longer crash numpy or silently mismatch lengths in the JIT kernel.
        p_arr = np.zeros(self.dim)
        q_arr = np.zeros(self.dim)
        p_arr[: min(len(p), self.dim)] = np.abs(p[: self.dim])
        q_arr[: min(len(q), self.dim)] = np.abs(q[: self.dim])
        p_abs = p_arr + 1e-8
        q_abs = q_arr + 1e-8
        if _HAS_JIT:
            return float(_kl_divergence(
                np.ascontiguousarray(p_abs, dtype=float),
                np.ascontiguousarray(q_abs, dtype=float),
            ))
        p_norm = p_abs / np.sum(p_abs)
        q_norm = q_abs / np.sum(q_abs)
        return float(np.sum(p_norm * np.log(p_norm / q_norm)))


class TopologicalAnalyzer:
    """Simplified persistent homology — computes topological features.

    C-batch fix: the previous ``compute_betti_numbers`` did NOT compute Betti
    numbers -- it counted *gaps* in a sorted 1D value stream, which is at best
    a heuristic for ``betti_0`` (number of connected components of a 1D
    point cloud at radius ``max_radius``), and the ``betti_1 = n - betti_0``
    formula is topologically wrong (1D point clouds have no 1-loops, so
    ``betti_1`` is always 0).

    The new API exposes the honest meaning:
    - ``connected_components_1d`` -- the count the old code actually computed
    - ``vietoris_rips_betti`` -- a true Vietoris-Rips Betti computation for
      multi-dimensional point clouds (rows = points), capped at 50 points
      so triangle enumeration stays tractable.
    ``compute_betti_numbers`` is kept as a deprecated alias returning
    ``{0: cc_1d, 1: 0}`` for 1D inputs.
    """

    # Cap on the number of points accepted by ``vietoris_rips_betti`` to keep
    # the O(n^3) triangle enumeration tractable.
    _VR_POINT_CAP = 50

    def __init__(self, dim: int = 64):
        self.dim = dim

    def connected_components_1d(
        self, data: np.ndarray, max_radius: float = 1.0
    ) -> int:
        """Number of connected components in a 1D point cloud at ``max_radius``.

        Two sorted points are in the same component iff the cumulative gap
        between them is ``<= max_radius``. This is the count the previous
        ``compute_betti_numbers`` actually returned (under the name betti_0).
        """
        d = np.asarray(data, dtype=float).flatten()[: self.dim]
        if d.size == 0:
            return 0
        sorted_vals = np.sort(d)
        n_points = len(sorted_vals)
        # Reuse the JIT kernel for the gap loop when available.
        if _HAS_JIT:
            betti_0, _ = _betti_numbers(
                np.ascontiguousarray(sorted_vals, dtype=float),
                float(max_radius),
            )
            return int(betti_0)
        betti_0 = 1
        for i in range(1, n_points):
            gap = sorted_vals[i] - sorted_vals[i - 1]
            # Round-9 audit R9-001: the documented contract is "two sorted
            # points are in the same component iff the cumulative gap
            # between them is ``<= max_radius``" (see docstring above).
            # The previous ``max_radius / n_points`` threshold shrank as the
            # point count grew, so for a normalised dim-64 vector (typical
            # gaps ~ 1/64) the threshold was also ~1/64 and roughly half of
            # all adjacent gaps exceeded it -- the returned count had no
            # relation to "components at radius max_radius". Use the raw
            # ``max_radius`` so the count matches the documented semantics
            # (and matches ``vietoris_rips_betti``'s 1-skeleton rule).
            if gap > max_radius:
                betti_0 += 1
        return int(betti_0)

    def vietoris_rips_betti(
        self, points: np.ndarray, max_radius: float = 1.0
    ) -> dict[int, int]:
        """True Vietoris-Rips Betti numbers (betti_0, betti_1) for a point cloud.

        ``points`` has shape ``(n_points, n_features)``: each row is a point
        in some Euclidean space. We build the VR 1-skeleton (edges where
        pairwise distance ``<= max_radius``), enumerate 2-simplices
        (triangles/cliques of size 3), and compute homology over GF(2):

            betti_0 = # connected components
            betti_1 = # edges - # vertices + betti_0 - rank(∂_2)

        where ``∂_2`` is the triangle boundary operator. The point cloud is
        capped at ``_VR_POINT_CAP`` to keep the ``O(n^3)`` triangle enumeration
        tractable. For degenerate inputs (1 or 0 points) returns ``{0: n, 1: 0}``.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            # Promote a 1D point cloud to (n, 1) so the pairwise distance and
            # VR construction are still meaningful (1D has betti_1 = 0).
            pts = pts.reshape(-1, 1)
        n = pts.shape[0]
        if n == 0:
            return {0: 0, 1: 0}
        if n > self._VR_POINT_CAP:
            # Subsample uniformly to the cap so the count stays tractable.
            idx = np.linspace(0, n - 1, self._VR_POINT_CAP).astype(int)
            pts = pts[idx]
            n = pts.shape[0]
        # Pairwise distance matrix.
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        # 1-skeleton adjacency (exclude self-loops).
        adj = (dist <= max_radius) & ~np.eye(n, dtype=bool)
        # betti_0 via union-find on the 1-skeleton.
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    union(i, j)
        roots = {find(i) for i in range(n)}
        betti_0 = len(roots)
        # Edges and triangles in the VR complex.
        edges: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((i, j))
        e_count = len(edges)
        # Triangles (3-cliques): all pairwise edges present.
        triangles: list[tuple[int, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j]:
                    continue
                for k in range(j + 1, n):
                    if adj[i, k] and adj[j, k]:
                        triangles.append((i, j, k))
        t_count = len(triangles)
        # rank(∂_2) over GF(2): build the (e_count x t_count) boundary
        # matrix and compute its rank via Gaussian elimination mod 2.
        if t_count == 0 or e_count == 0:
            rank_d2 = 0
        else:
            edge_index = {e: i for i, e in enumerate(edges)}
            d2 = np.zeros((e_count, t_count), dtype=np.int8)
            for t_idx, (a, b, c) in enumerate(triangles):
                d2[edge_index[(min(a, b), max(a, b))], t_idx] = 1
                d2[edge_index[(min(a, c), max(a, c))], t_idx] = 1
                d2[edge_index[(min(b, c), max(b, c))], t_idx] = 1
            # Gaussian elimination mod 2 to find rank.
            rank_d2 = 0
            mat = d2.copy()
            col = 0
            for row in range(e_count):
                if col >= t_count:
                    break
                # Find a pivot at or below ``row`` in column ``col``.
                pivot = -1
                for r in range(row, e_count):
                    if mat[r, col] == 1:
                        pivot = r
                        break
                if pivot == -1:
                    col += 1
                    continue
                if pivot != row:
                    mat[[row, pivot]] = mat[[pivot, row]]
                # Eliminate below and above.
                for r in range(e_count):
                    if r != row and mat[r, col] == 1:
                        mat[r] = (mat[r] ^ mat[row]).astype(np.int8)
                rank_d2 += 1
                col += 1
        # H_1 = ker(∂_1) / im(∂_2), so dim H_1 = (e - rank(∂_1)) - rank(∂_2)
        # where rank(∂_1) = n - betti_0 (boundary rank-nullity theorem).
        betti_1 = e_count - (n - betti_0) - rank_d2
        if betti_1 < 0:
            betti_1 = 0
        return {0: int(betti_0), 1: int(betti_1)}

    def compute_betti_numbers(self, data: np.ndarray, max_radius: float = 1.0) -> dict[int, int]:
        """Deprecated: use ``connected_components_1d`` or ``vietoris_rips_betti``.

        Returns ``{0: cc_1d, 1: 0}`` for 1D input. The previous ``betti_1 =
        n - betti_0`` formula was topologically incorrect (1D point clouds
        have no 1-loops); this is now fixed.
        """
        # Round-3 audit: emit DeprecationWarning so callers are alerted.
        import warnings

        warnings.warn(
            "compute_betti_numbers is deprecated; use connected_components_1d"
            " or vietoris_rips_betti",
            DeprecationWarning,
            stacklevel=2,
        )
        return {0: self.connected_components_1d(data, max_radius), 1: 0}

    def topological_features(self, data: np.ndarray) -> np.ndarray:
        betti = self.compute_betti_numbers(data)
        features = np.zeros(self.dim)
        features[0] = betti[0]
        features[1] = betti.get(1, 0)
        d = data.flatten()[: self.dim]
        features[2] = np.mean(d)
        features[3] = np.std(d)
        if _HAS_JIT:
            features[4] = float(_skewness(np.ascontiguousarray(d, dtype=float)))
        else:
            # Pure-numpy fallback: biased sample skewness (matches scipy.stats.skew default).
            n = len(d)
            if n < 3:
                features[4] = 0.0
            else:
                diff = d - np.mean(d)
                m2 = np.mean(diff ** 2)
                if m2 == 0.0:
                    features[4] = 0.0
                else:
                    features[4] = float(np.mean(diff ** 3) / (m2 ** 1.5))
        return features


class FractalGenerator:
    """Fractal compression and generation."""

    def __init__(self, dim: int = 64, rng: np.random.Generator | None = None):
        self.dim = dim
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        self.transforms: list[tuple[np.ndarray, np.ndarray]] = []
        self._init_transforms()

    def _init_transforms(self) -> None:
        for _ in range(4):
            # Round-3 audit CRIT-1: per-module Generator
            scale = self._rng.standard_normal((self.dim, self.dim)) * 0.1
            offset = self._rng.standard_normal(self.dim) * 0.1
            self.transforms.append((scale, offset))

    def generate(self, initial: np.ndarray, n_iterations: int = 10) -> np.ndarray:
        # Round-6 audit NEW5-5: clamp n_iterations to avoid a hostile caller
        # hanging the process. Each iteration is O(dim^2) (a matmul + tanh),
        # so 10_000 iterations at dim=4096 is ~672 GFLOP — already generous.
        n_iterations = int(min(max(n_iterations, 0), 10_000))
        x = initial[: self.dim].copy()
        if len(x) < self.dim:
            x = np.pad(x, (0, self.dim - len(x)))
        if _HAS_JIT:
            scales = np.stack([t[0] for t in self.transforms])
            offsets = np.stack([t[1] for t in self.transforms])
            return _fractal_generate(
                np.ascontiguousarray(x, dtype=float),
                np.ascontiguousarray(scales, dtype=float),
                np.ascontiguousarray(offsets, dtype=float),
                int(n_iterations),
                int(len(self.transforms)),
            )
        for _ in range(n_iterations):
            transform = self.transforms[_ % len(self.transforms)]
            x = transform[0] @ x + transform[1]
            x = np.tanh(x)
        return x

    def compress(self, data: np.ndarray) -> dict:
        d = data.flatten()[: self.dim]
        if len(d) < self.dim:
            d = np.pad(d, (0, self.dim - len(d)))
        # ``np.corrcoef`` requires equal-length inputs; for odd ``dim`` the
        # naive split produces lengths that differ by one. Use the first
        # ``m = dim // 2`` samples of each half so both sides have length m.
        m = self.dim // 2
        corr = float(np.corrcoef(d[:m], d[m : 2 * m])[0, 1]) if m >= 2 else 0.0
        # Round-3 audit: ``np.corrcoef`` returns NaN when either half has
        # zero variance (constant input). Guard to prevent NaN propagation
        # into ``mine_patterns`` / ``analyze_trend`` outputs.
        if not np.isfinite(corr):
            corr = 0.0
        return {
            "mean": float(np.mean(d)),
            "std": float(np.std(d)),
            "self_similarity": corr,
        }


class MathematicalUniverse(CognitiveModule):
    """
    Mathematical Universe Layer.
    - Information geometry for probability analysis
    - Topological data analysis for shape features
    - Fractal generation and compression
    """

    def __init__(self, dim: int = 64, rng: np.random.Generator | None = None):
        self.dim = dim
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        self.info_geometry = InformationGeometry(dim)
        self.topology = TopologicalAnalyzer(dim)
        self.fractal = FractalGenerator(dim, rng=self._rng)
        # Cache of the last ``process`` output so ``predict`` can reuse it
        # instead of re-running ``fractal.generate`` (Fix 12).
        self._last_process_output: np.ndarray | None = None

    def process(self, signal: Signal) -> Signal:
        topo_features = self.topology.topological_features(signal.data)
        fractal_output = self.fractal.generate(signal.data)
        combined = 0.5 * topo_features + 0.5 * fractal_output
        # Cache the combined output for predict() to reuse (Fix 12).
        self._last_process_output = combined
        return Signal(data=combined, metadata={"topological": True, "fractal": True})

    def predict(self, signal: Signal) -> Prediction:
        # Reuse the cached process output when available so we do not call
        # ``fractal.generate`` twice per think() cycle (Fix 12). Fall back to
        # a fresh fractal pass when predict() is called standalone.
        if self._last_process_output is not None:
            value = self._last_process_output
        else:
            value = self.fractal.generate(signal.data, n_iterations=5)
        return Prediction(value=value, uncertainty=float(np.var(value)))

    def update(self, prediction_error: float) -> None:
        if not np.isfinite(prediction_error):
            return
        prediction_error = float(np.clip(prediction_error, -1e6, 1e6))
        for i, (scale, offset) in enumerate(self.fractal.transforms):
            # Round-3 audit CRIT-1: per-module Generator
            noise = self._rng.standard_normal(scale.shape) * prediction_error * 0.001
            self.fractal.transforms[i] = (scale + noise, offset)
