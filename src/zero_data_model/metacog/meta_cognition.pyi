from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np


class MetaCognition:
    """二阶信念：对自身不确定性的估计与元认知触发。"""

    _dim: int
    _window: int
    _uncertainty_threshold: float
    _rng: np.random.Generator
    _lock: threading.RLock
    _error_history: deque[float]
    _update_magnitudes: deque[float]
    _confidence_history: deque[float]
    _uncertainty_estimate: np.ndarray
    _mode: str
    _step_count: int
    _ema_alpha: float

    def __init__(
        self,
        dim: int = 64,
        window: int = 50,
        uncertainty_threshold: float = 0.5,
        seed: int = 42,
    ) -> None: ...

    def update(
        self,
        prediction_error: float,
        param_update_norm: float,
        similar_episode_error: float = 0.0,
    ) -> dict[str, Any]: ...

    def should_seek_info(self) -> bool: ...

    def get_confidence(self) -> float: ...

    def get_uncertainty_vector(self) -> np.ndarray: ...

    def record_milestone(self, event_type: str, step: int) -> dict[str, Any]: ...

    @property
    def mode(self) -> str: ...

    @property
    def confidence(self) -> float: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = ["MetaCognition"]
