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
        p_abs = np.abs(p[: self.dim]) + 1e-8
        q_abs = np.abs(q[: self.dim]) + 1e-8
        if _HAS_JIT:
            return float(_kl_divergence(
                np.ascontiguousarray(p_abs, dtype=float),
                np.ascontiguousarray(q_abs, dtype=float),
            ))
        p_norm = p_abs / np.sum(p_abs)
        q_norm = q_abs / np.sum(q_abs)
        return float(np.sum(p_norm * np.log(p_norm / q_norm)))


class TopologicalAnalyzer:
    """Simplified persistent homology — computes topological features."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def compute_betti_numbers(self, data: np.ndarray, max_radius: float = 1.0) -> dict[int, int]:
        d = data.flatten()[: self.dim]
        sorted_vals = np.sort(d)
        n_points = len(sorted_vals)
        if _HAS_JIT:
            betti_0, betti_1 = _betti_numbers(
                np.ascontiguousarray(sorted_vals, dtype=float),
                float(max_radius),
            )
            return {0: int(betti_0), 1: int(betti_1)}
        betti_0 = 1
        for i in range(1, n_points):
            gap = sorted_vals[i] - sorted_vals[i - 1]
            if gap > max_radius / n_points:
                betti_0 += 1
        betti_1 = max(0, n_points - betti_0)
        return {0: betti_0, 1: betti_1}

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

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.transforms: list[tuple[np.ndarray, np.ndarray]] = []
        self._init_transforms()

    def _init_transforms(self) -> None:
        for _ in range(4):
            scale = np.random.randn(self.dim, self.dim) * 0.1
            offset = np.random.randn(self.dim) * 0.1
            self.transforms.append((scale, offset))

    def generate(self, initial: np.ndarray, n_iterations: int = 10) -> np.ndarray:
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
        return {
            "mean": float(np.mean(d)),
            "std": float(np.std(d)),
            "self_similarity": (
                float(np.corrcoef(d[: self.dim // 2], d[self.dim // 2:])[0, 1])
                if self.dim >= 2
                else 0.0
            ),
        }


class MathematicalUniverse(CognitiveModule):
    """
    Mathematical Universe Layer.
    - Information geometry for probability analysis
    - Topological data analysis for shape features
    - Fractal generation and compression
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.info_geometry = InformationGeometry(dim)
        self.topology = TopologicalAnalyzer(dim)
        self.fractal = FractalGenerator(dim)

    def process(self, signal: Signal) -> Signal:
        topo_features = self.topology.topological_features(signal.data)
        fractal_output = self.fractal.generate(signal.data)
        combined = 0.5 * topo_features + 0.5 * fractal_output
        return Signal(data=combined, metadata={"topological": True, "fractal": True})

    def predict(self, signal: Signal) -> Prediction:
        fractal_pred = self.fractal.generate(signal.data, n_iterations=5)
        return Prediction(value=fractal_pred, uncertainty=float(np.var(fractal_pred)))

    def update(self, prediction_error: float) -> None:
        for i, (scale, offset) in enumerate(self.fractal.transforms):
            noise = np.random.randn(*scale.shape) * prediction_error * 0.001
            self.fractal.transforms[i] = (scale + noise, offset)
