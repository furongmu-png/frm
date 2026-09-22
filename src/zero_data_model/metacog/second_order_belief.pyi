from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np

class SecondOrderBelief:
    """二阶信念：对自身隐状态不确定性的高斯建模。"""

    belief_mean: np.ndarray
    uncertainty_var: np.ndarray
    _dim: int
    _ema_alpha: float
    _memory_w: float
    _param_w: float
    _error_w: float
    _rng: np.random.Generator
    _lock: threading.RLock
    _predicted_uncertainty: float
    _confidence_history: deque[float]
    _error_ema: float
    _step_count: int
    _last_decomposition: dict[str, float]

    def __init__(
        self,
        dim: int = 64,
        ema_alpha: float = 0.1,
        memory_weight: float = 0.2,
        param_weight: float = 0.3,
        error_weight: float = 0.5,
        seed: int = 42,
    ) -> None: ...

    def update(
        self,
        belief_state: np.ndarray,
        prediction_error: float,
        param_update_norm: float = 0.0,
        memory_similarity: float = 1.0,
    ) -> dict[str, Any]: ...

    @property
    def predicted_uncertainty(self) -> float: ...

    def get_confidence(self) -> float: ...

    def get_uncertainty_vector(self) -> np.ndarray: ...

    def get_belief_mean(self) -> np.ndarray: ...

    def get_confidence_history(self) -> list[float]: ...

    def get_decomposition(self) -> dict[str, float]: ...

    def sample_belief(self, n_samples: int = 1) -> np.ndarray: ...

    @property
    def confidence(self) -> float: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = ["SecondOrderBelief"]
