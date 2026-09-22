from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np

class MetaTrigger:
    """元认知触发机制：不确定度超阈值时启动信息寻求。"""

    _threshold_sigma: float
    _min_history: int
    _lr_scale_factor: float
    _rng: np.random.Generator
    _lock: threading.RLock
    _uncertainty_history: deque[float]
    _triggered: bool
    _learning_rate_scale: float
    _current_threshold: float
    _step_count: int
    _pending_info_requests: list[str]
    _pending_questions: list[str]
    _trigger_events: deque[dict[str, Any]]

    def __init__(
        self,
        threshold_sigma: float = 2.0,
        history_window: int = 100,
        min_history: int = 5,
        lr_scale_factor: float = 0.5,
        seed: int = 42,
    ) -> None: ...

    def update(self, uncertainty: float, topic: str = "") -> dict[str, Any]: ...

    def _trigger(self, uncertainty: float, topic: str) -> None: ...

    def _untrigger(self) -> None: ...

    def generate_question(self, topic: str) -> str: ...

    def should_seek_info(self) -> bool: ...

    def get_pending_info_requests(self) -> list[str]: ...

    def get_pending_questions(self) -> list[str]: ...

    def get_trigger_events(self) -> list[dict[str, Any]]: ...

    @property
    def triggered(self) -> bool: ...

    @property
    def learning_rate_scale(self) -> float: ...

    @property
    def current_threshold(self) -> float: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = ["MetaTrigger"]
