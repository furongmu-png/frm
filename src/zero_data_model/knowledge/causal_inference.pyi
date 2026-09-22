from __future__ import annotations

import threading
from typing import Any

import numpy as np


class CausalInference:
    """基于转移矩阵的简化因果推断。"""

    _lock: threading.RLock
    transition_matrix: np.ndarray | None

    def __init__(self, transition_matrix: np.ndarray | None = None) -> None: ...

    def set_transition(self, matrix: np.ndarray) -> None: ...

    def do_calculus(self, intervention: dict[int, float]) -> dict[str, Any]: ...

    def counterfactual(
        self, observed: np.ndarray, intervention: dict[int, float]
    ) -> np.ndarray: ...

    def identify_confounders(self, var_a: int, var_b: int) -> list[int]: ...


__all__: list[str] = ["CausalInference"]
