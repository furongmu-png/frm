from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Layer:
    """单个预测层的数据载体。"""

    name: str
    dim: int
    belief: np.ndarray
    transition: np.ndarray
    error_history: deque[float]
    rng: np.random.Generator


class LayeredPredictor:
    """多时间尺度层级预测编码器。"""

    BELIEF_DECAY: float
    LR: float
    SPECTRAL_MARGIN: float

    dim_l0: int
    dim_l1: int
    dim_l2: int
    l1_interval: int
    l2_interval: int
    seed: int
    l0: Layer
    l1: Layer
    l2: Layer
    _lock: threading.RLock

    def __init__(
        self,
        dim_l0: int = 64,
        dim_l1: int = 32,
        dim_l2: int = 16,
        seed: int = 42,
        l1_interval: int = 10,
        l2_interval: int = 100,
    ) -> None: ...

    def update(self, observation: np.ndarray, step: int) -> dict[str, Any]: ...

    def get_context(self) -> np.ndarray: ...

    def predict_rhythm(self) -> float: ...


__all__: list[str] = ["Layer", "LayeredPredictor"]
