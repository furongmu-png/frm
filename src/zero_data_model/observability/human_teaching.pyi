from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

class TeachingMode(Enum):
    NONE = auto()
    DEMONSTRATION = auto()
    CORRECTION = auto()


class FeedbackType(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()
    DIRECTION_HINT = auto()


@dataclass
class FeedbackRecord:
    step: int
    feedback_type: FeedbackType
    prediction: Any = None
    actual: Any = None
    error_signal: float = 0.0
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]: ...


class HumanFeedback:
    decay_rate: float
    positive_strength: float
    negative_strength: float
    max_feedback: int
    _current_weight: float
    _mode: TeachingMode
    _total_feedback: int
    _history: deque[FeedbackRecord]
    _demonstrations: deque[dict[str, Any]]
    _corrections: deque[dict[str, Any]]
    _lock: threading.RLock

    def __init__(
        self,
        decay_rate: float = 0.95,
        positive_strength: float = 0.1,
        negative_strength: float = 0.5,
        max_feedback: int = 10000,
        history_limit: int = 5000,
    ) -> None: ...

    @property
    def mode(self) -> TeachingMode: ...

    def set_mode(self, mode: TeachingMode) -> None: ...

    @property
    def is_active(self) -> bool: ...

    @property
    def current_weight(self) -> float: ...

    def give_feedback(
        self,
        step: int,
        feedback_type: FeedbackType,
        prediction: Any = None,
        actual: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackRecord: ...

    def demonstrate(
        self,
        step: int,
        observation: Any,
        action: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def get_demonstrations(self, n: int = 20) -> list[dict[str, Any]]: ...

    def correct(
        self,
        step: int,
        prediction: Any,
        correct_label: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def get_corrections(self, n: int = 20) -> list[dict[str, Any]]: ...

    @property
    def total_feedback(self) -> int: ...

    @property
    def n_demonstrations(self) -> int: ...

    @property
    def n_corrections(self) -> int: ...

    def get_history(self, n: int = 20) -> list[dict[str, Any]]: ...

    def reset_weight(self) -> None: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = [
    "FeedbackRecord",
    "FeedbackType",
    "HumanFeedback",
    "TeachingMode",
]
