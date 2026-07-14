# src/zero_data_model/base.py
"""Base classes for the zero-data model system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Signal:
    """A signal passing through the system."""
    data: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class Prediction:
    """A prediction with associated uncertainty."""
    value: np.ndarray
    uncertainty: float
    prediction_error: float = 0.0


class CognitiveModule(ABC):
    """Base class for all cognitive modules."""

    @abstractmethod
    def process(self, signal: Signal) -> Signal:
        ...

    @abstractmethod
    def predict(self, signal: Signal) -> Prediction:
        ...

    @abstractmethod
    def update(self, prediction_error: float) -> None:
        ...


class KnowledgeStore(ABC):
    """Base class for knowledge storage."""

    @abstractmethod
    def store(self, key: str, value: np.ndarray) -> None:
        ...

    @abstractmethod
    def retrieve(self, key: str) -> np.ndarray | None:
        ...

    @abstractmethod
    def generate(self, query: Signal) -> Signal:
        """Self-generate knowledge without external input."""
        ...
