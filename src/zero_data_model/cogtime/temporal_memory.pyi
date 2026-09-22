from __future__ import annotations

import threading
from typing import Any

import numpy as np


class TemporalMemory:
    """固定随机权重 RNN 时序记忆（回声状态网络风格）。"""

    input_dim: int
    hidden_dim: int
    output_dim: int
    seed: int
    W_in: np.ndarray
    W_hh: np.ndarray
    W_out: np.ndarray
    hidden: np.ndarray
    _last_valid_context: np.ndarray
    _lock: threading.RLock

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int | None = None,
        seed: int = 42,
    ) -> None: ...

    def forward(self, x: np.ndarray) -> np.ndarray: ...

    def update(
        self, x: np.ndarray, target: np.ndarray, lr: float = 0.01
    ) -> float: ...

    def get_context(self) -> np.ndarray: ...

    def reset(self) -> None: ...

    @property
    def spectral_radius(self) -> float: ...


__all__: list[str] = ["TemporalMemory"]
