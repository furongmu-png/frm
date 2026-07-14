from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import CognitiveModule, Prediction, Signal

@dataclass
class PredictiveLayer:
    """One layer in the predictive processing hierarchy."""
    weights: np.ndarray
    bias: np.ndarray
    activation: str = "tanh"

    def predict(self, x: np.ndarray) -> np.ndarray: ...

    def prediction_error(self, actual: np.ndarray, predicted: np.ndarray) -> float: ...


class SelfModel:
    """System's model of itself — enables metacognition."""

    state: np.ndarray
    confidence: float
    history: list[np.ndarray]

    def __init__(self, dim: int = 64) -> None: ...

    def update(self, signal: np.ndarray) -> None: ...

    def reflect(self) -> Signal: ...


class GlobalWorkspace:
    """Global Workspace Theory implementation — information broadcast center."""

    dim: int
    capacity: int
    buffer: list[Signal]
    attention_weights: np.ndarray

    def __init__(self, dim: int = 64, capacity: int = 16) -> None: ...

    def broadcast(self, signal: Signal) -> Signal: ...

    def update_attention(self, relevance: np.ndarray) -> None: ...


class ConsciousnessCore(CognitiveModule):
    """
    Consciousness-inspired cognitive core.
    - Predictive processing hierarchy
    - Global workspace for information integration
    - Self-model for metacognition
    """

    dim: int
    layers: list[PredictiveLayer]
    workspace: GlobalWorkspace
    self_model: SelfModel

    def __init__(self, dim: int = 64, n_layers: int = 3) -> None: ...

    def process(self, signal: Signal) -> Signal: ...

    def predict(self, signal: Signal) -> Prediction: ...

    def update(self, prediction_error: float) -> None: ...

    def reflect(self) -> Signal:
        """Metacognition — the system thinks about its own state."""
        ...
