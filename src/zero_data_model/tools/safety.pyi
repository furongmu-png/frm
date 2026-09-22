from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

@dataclass
class SafetyViolation:
    tool_name: str
    reason: str
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]: ...


class SafetyChecker:
    min_interval: float
    _last_call_time: dict[str, float]
    _violations: deque[SafetyViolation]
    _audit_log: deque[dict[str, Any]]
    _lock: threading.RLock

    def __init__(
        self, min_interval: float = 0.1, max_violation_log: int = 1000
    ) -> None: ...

    def check(
        self, tool_name: str, params: dict[str, Any]
    ) -> SafetyViolation | None: ...

    def record_call(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: dict[str, Any],
        trigger_reason: str = "",
    ) -> None: ...

    @property
    def violations(self) -> list[dict[str, Any]]: ...

    @property
    def audit_log(self) -> list[dict[str, Any]]: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = ["SafetyChecker", "SafetyViolation"]
