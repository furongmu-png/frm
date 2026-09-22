from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ExperimentRecord:
    step: int
    hypothesis: str = ""
    intervention: str = ""
    observation: str = ""
    bayes_factor: float = 1.0
    conclusion: str = "inconclusive"
    info_gain: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]: ...


class ExperimentLogger:
    records: deque[ExperimentRecord]
    milestones: deque[dict[str, Any]]
    _lock: threading.RLock
    _push_callback: Callable[[dict[str, Any]], None] | None
    _milestone_counters: dict[str, int]

    def __init__(
        self,
        max_records: int = 1000,
        push_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None: ...

    def log(
        self,
        step: int,
        hypothesis: str = "",
        intervention: str = "",
        observation: str = "",
        bayes_factor: float = 1.0,
        conclusion: str = "inconclusive",
        info_gain: float = 0.0,
    ) -> ExperimentRecord: ...

    def log_from_dict(self, data: dict[str, Any]) -> ExperimentRecord: ...

    def _check_milestone(self, record: ExperimentRecord) -> None: ...

    def get_records(self, limit: int = 50) -> list[dict[str, Any]]: ...

    def get_milestones(self) -> list[dict[str, Any]]: ...

    def get_pending_milestones(self) -> list[dict[str, Any]]: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = ["ExperimentLogger", "ExperimentRecord"]
