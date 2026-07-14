# src/zero_data_model/consciousness_core.py
"""Consciousness-inspired cognitive core using Global Workspace Theory and Predictive Processing."""

from __future__ import annotations

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
        self.history: list[np.ndarray] = []

    def update(self, signal: np.ndarray) -> None:
        self.history.append(signal.copy())
        if len(self.history) > 100:
            self.history.pop(0)
        self.state = 0.9 * self.state + 0.1 * np.mean(
            list(self.history[-10:]), axis=0
        ) if self.history else self.state
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
        self.buffer: list[Signal] = []
        self.attention_weights = np.ones(dim) / dim

    def broadcast(self, signal: Signal) -> Signal:
        attended = signal.data * self.attention_weights[: len(signal.data)]
        attended = attended / (np.linalg.norm(attended) + 1e-8)
        self.buffer.append(Signal(data=attended, metadata=signal.metadata))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
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

    def process(self, signal: Signal) -> Signal:
        x = signal.data[: self.dim]
        if len(x) < self.dim:
            x = np.pad(x, (0, self.dim - len(x)))

        for layer in self.layers:
            prediction = layer.predict(x)
            error = layer.prediction_error(x, prediction)
            x = prediction + np.random.randn(self.dim) * error * 0.01

        self.self_model.update(x)
        result = self.workspace.broadcast(Signal(data=x, metadata=signal.metadata))
        return result

    def predict(self, signal: Signal) -> Prediction:
        x = signal.data[: self.dim]
        if len(x) < self.dim:
            x = np.pad(x, (0, self.dim - len(x)))
        predicted = self.layers[0].predict(x)
        uncertainty = float(np.var(predicted))
        return Prediction(value=predicted, uncertainty=uncertainty)

    def update(self, prediction_error: float) -> None:
        for layer in self.layers:
            noise = np.random.randn(*layer.weights.shape) * prediction_error * 0.001
            layer.weights += noise

    def reflect(self) -> Signal:
        """Metacognition — the system thinks about its own state."""
        return self.self_model.reflect()
