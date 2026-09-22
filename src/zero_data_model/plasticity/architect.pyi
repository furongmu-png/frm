from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np


class ArchitectureOptimizer:
    """动态模块生长/剪枝优化器。"""

    dim: int
    seed: int
    eval_interval: int
    split_threshold_steps: int
    split_error_ratio: float
    prune_threshold_steps: int
    prune_error_ratio: float
    rng: np.random.Generator
    _error_history: dict[str, deque[float]]
    _dormant_modules: dict[str, Any]
    _high_streak: dict[str, int]
    _low_streak: dict[str, int]
    _streak_start_cycle: dict[str, int]
    _step_count: int
    _total_splits: int
    _total_prunes: int
    _active_count: int
    _lock: threading.RLock

    def __init__(
        self,
        dim: int = 64,
        seed: int = 42,
        eval_interval: int = 100,
        split_threshold_steps: int = 200,
        split_error_ratio: float = 0.3,
        prune_threshold_steps: int = 500,
        prune_error_ratio: float = 0.01,
    ) -> None: ...

    def record_errors(
        self, module_names: list[str], errors: list[float]
    ) -> None: ...

    def evaluate(self, modules: list[Any], cycle: int) -> dict[str, Any]: ...

    def reactivate(self, modules: list[Any], name: str) -> bool: ...

    def dormant_names(self) -> list[str]: ...

    @property
    def stats(self) -> dict[str, Any]: ...
