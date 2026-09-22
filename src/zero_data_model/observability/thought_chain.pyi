from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ThoughtRecord:
    """单步思维记录：一个认知周期内的模块状态变化。"""

    step: int
    timestamp: float = 0.0
    belief_before: list[float] = field(default_factory=list)
    belief_after: list[float] = field(default_factory=list)
    prediction_errors: dict[str, float] = field(default_factory=dict)
    update_magnitude: float = 0.0
    confidence: float = 1.0
    meta_triggered: bool = False
    tool_called: str = ""
    free_energy: float = 0.0
    is_anomaly: bool = False
    anomaly_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]: ...


class ThoughtChain:
    """思维链：记录和回放模型的认知过程。"""

    max_records: int
    anomaly_pe_threshold: float
    anomaly_confidence_threshold: float
    _records: deque[ThoughtRecord]
    _lock: threading.RLock

    def __init__(
        self,
        max_records: int = 5000,
        anomaly_pe_threshold: float = 2.0,
        anomaly_confidence_threshold: float = 0.3,
    ) -> None: ...

    def record(
        self,
        step: int,
        belief_before: list[float] | None = None,
        belief_after: list[float] | None = None,
        prediction_errors: dict[str, float] | None = None,
        update_magnitude: float = 0.0,
        confidence: float = 1.0,
        meta_triggered: bool = False,
        tool_called: str = "",
        free_energy: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ThoughtRecord: ...

    def get_recent(self, n: int = 20) -> list[dict[str, Any]]: ...

    def get_anomalies(self, n: int = 20) -> list[dict[str, Any]]: ...

    def get_at_step(self, step: int) -> dict[str, Any] | None: ...

    def get_range(
        self, start_step: int, end_step: int
    ) -> list[dict[str, Any]]: ...

    def replay(self) -> list[dict[str, Any]]: ...

    def clear(self) -> None: ...

    @property
    def length(self) -> int: ...

    @property
    def n_anomalies(self) -> int: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = ["ThoughtRecord", "ThoughtChain"]
