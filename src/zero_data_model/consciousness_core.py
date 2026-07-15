# src/zero_data_model/consciousness_core.py
"""Consciousness-inspired cognitive core using Global Workspace Theory and Predictive Processing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .base import CognitiveModule, Prediction, Signal

# JIT kernels -- graceful fallback to pure numpy if numba missing.
try:
    from .hardware.kernels import _predictive_layer_forward
    _HAS_JIT = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_JIT = False


@dataclass
class PredictiveLayer:
    """One layer in the predictive processing hierarchy."""
    weights: np.ndarray
    bias: np.ndarray
    activation: str = "tanh"

    def predict(self, x: np.ndarray) -> np.ndarray:
        if _HAS_JIT:
            return _predictive_layer_forward(
                np.ascontiguousarray(x, dtype=float),
                np.ascontiguousarray(self.weights, dtype=float),
                np.ascontiguousarray(self.bias, dtype=float),
                self.activation,
            )
        z = x @ self.weights + self.bias
        if self.activation == "tanh":
            return np.tanh(z)
        return np.clip(z, 0, None)  # relu

    def prediction_error(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        return float(np.mean((actual - predicted) ** 2))


class SelfModel:
    """System's model of itself — enables metacognition."""

    def __init__(self, dim: int = 64):
        self.state = np.zeros(dim)
        self.confidence = 0.5
        # ``deque(maxlen=100)`` so old entries are evicted automatically and
        # we no longer pay O(n) for ``pop(0)`` (Fix 23). The dead
        # ``if self.history else self.state`` branch is removed: history is
        # always non-empty after the append above.
        self.history: deque = deque(maxlen=100)
        # Dedicated recent-only deque so update() avoids ``list(history)[-10:]``
        # (a full-list materialization every frame, P-CRIT-02).
        self._recent: deque = deque(maxlen=10)

    def update(self, signal: np.ndarray) -> None:
        self.history.append(signal.copy())
        self._recent.append(signal.copy())
        # ``self.history`` is always non-empty here (we just appended).
        recent = list(self._recent)
        self.state = 0.9 * self.state + 0.1 * np.mean(recent, axis=0)
        self.confidence = min(1.0, len(self.history) / 50.0)

    def reflect(self) -> Signal:
        return Signal(
            data=self.state.copy(),
            metadata={"self_confidence": self.confidence, "history_len": len(self.history)},
        )


class GlobalWorkspace:
    """Global Workspace Theory implementation — information broadcast center."""

    def __init__(self, dim: int = 64, capacity: int = 16):
        self.dim = dim
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.attention_weights = np.ones(dim) / dim

    def broadcast(self, signal: Signal) -> Signal:
        attended = signal.data * self.attention_weights[: len(signal.data)]
        attended = attended / (np.linalg.norm(attended) + 1e-8)
        self.buffer.append(Signal(data=attended, metadata=signal.metadata))
        # ``deque(maxlen=capacity)`` evicts the oldest entry automatically, so
        # the manual ``pop(0)`` (O(n) shift) is gone (P-HIGH-03).
        integrated = np.mean([s.data for s in self.buffer], axis=0)
        return Signal(data=integrated, metadata={"source": "global_workspace"})

    def update_attention(self, relevance: np.ndarray) -> None:
        r = relevance[: self.dim]
        self.attention_weights = r / (np.sum(r) + 1e-8)


class ConsciousnessCore(CognitiveModule):
    """
    Consciousness-inspired cognitive core.
    - Predictive processing hierarchy
    - Global workspace for information integration
    - Self-model for metacognition
    """

    def __init__(self, dim: int = 64, n_layers: int = 3):
        self.dim = dim
        self.layers = [
            PredictiveLayer(
                weights=np.random.randn(dim, dim) * 0.1,
                bias=np.zeros(dim),
            )
            for _ in range(n_layers)
        ]
        self.workspace = GlobalWorkspace(dim)
        self.self_model = SelfModel(dim)
        # Cache of the last ``process`` output so ``predict`` can reuse it
        # instead of re-running the predictive hierarchy (Fix 12).
        self._last_process_output: np.ndarray | None = None

    def process(self, signal: Signal) -> Signal:
        x = signal.data[: self.dim]
        if len(x) < self.dim:
            x = np.pad(x, (0, self.dim - len(x)))

        for layer in self.layers:
            prediction = layer.predict(x)
            error = layer.prediction_error(x, prediction)
            x = prediction + np.random.randn(self.dim) * error * 0.01

        self.self_model.update(x)
        # Cache the post-hierarchy state for predict() to reuse (Fix 12).
        self._last_process_output = x
        # C-batch fix: ``update_attention`` was previously never called from
        # ``process``, so the attention weights stayed uniform (1/dim) and
        # ``broadcast``'s ``signal * attention_weights`` was a uniform scaling
        # that did not actually attend to anything. Now we update the
        # attention weights from the per-dimension relevance of the
        # post-hierarchy state BEFORE broadcasting, so the workspace
        # meaningfully emphasises salient dimensions.
        relevance = np.abs(x)
        self.workspace.update_attention(relevance)
        result = self.workspace.broadcast(Signal(data=x, metadata=signal.metadata))
        return result

    def predict(self, signal: Signal) -> Prediction:
        # Reuse the cached process output when available so we do not re-run
        # the predictive hierarchy twice per think() cycle (Fix 12). Fall back
        # to a single-layer forward pass when predict() is called standalone.
        if self._last_process_output is not None:
            predicted = self._last_process_output
        else:
            x = signal.data[: self.dim]
            if len(x) < self.dim:
                x = np.pad(x, (0, self.dim - len(x)))
            predicted = self.layers[0].predict(x)
        uncertainty = float(np.var(predicted))
        return Prediction(value=predicted, uncertainty=uncertainty)

    def update(self, prediction_error: float) -> None:
        if not np.isfinite(prediction_error):
            return
        prediction_error = float(np.clip(prediction_error, -1e6, 1e6))
        for layer in self.layers:
            noise = np.random.randn(*layer.weights.shape) * prediction_error * 0.001
            layer.weights += noise

    def reflect(self) -> Signal:
        """Metacognition — the system thinks about its own state."""
        return self.self_model.reflect()
