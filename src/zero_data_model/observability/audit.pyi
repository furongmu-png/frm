from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

class AuditEventType(Enum):
    ACTION_SELECTION = auto()
    TOOL_CALL = auto()
    METACOGNITION_TRIGGER = auto()
    HUMAN_INTERVENTION = auto()
    COMMUNICATION = auto()
    CULTURE_TRANSFER = auto()
    VALUE_VIOLATION = auto()
    EMERGENCY_STOP = auto()
    OTHER = auto()


@dataclass
class AuditEntry:
    timestamp: float
    event_type: AuditEventType
    module: str
    context: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    entry_id: int = 0

    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class ValueDimension:
    name: str
    description: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]: ...


class ValueVector:
    _dimensions: dict[str, ValueDimension]
    _lock: threading.RLock

    def __init__(self, dimensions: list[ValueDimension] | None = None) -> None: ...

    def add_dimension(
        self, name: str, description: str, weight: float = 1.0
    ) -> ValueDimension: ...

    def remove_dimension(self, name: str) -> bool: ...

    def get_dimension(self, name: str) -> ValueDimension | None: ...

    def set_weight(self, name: str, weight: float) -> bool: ...

    def compute_value_penalty(
        self, action_violations: dict[str, float]
    ) -> float: ...

    def list_dimensions(self) -> list[ValueDimension]: ...

    @property
    def dimension_names(self) -> list[str]: ...

    @property
    def stats(self) -> dict[str, Any]: ...

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValueVector: ...


class AuditLogger:
    _entries: deque[AuditEntry]
    _value_vector: ValueVector | None
    _next_id: int
    _lock: threading.RLock

    def __init__(
        self,
        max_entries: int = 10000,
        value_vector: ValueVector | None = None,
    ) -> None: ...

    def log(
        self,
        event_type: AuditEventType,
        module: str,
        context: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry: ...

    def log_value_violation(
        self,
        module: str,
        action_violations: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> AuditEntry | None: ...

    def query(
        self,
        event_type: AuditEventType | None = None,
        module: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_recent(self, n: int = 20) -> list[dict[str, Any]]: ...

    def get_by_event_type(
        self, event_type: AuditEventType, n: int = 20
    ) -> list[dict[str, Any]]: ...

    def export_json(self) -> str: ...

    def export_csv(self) -> str: ...

    @property
    def n_entries(self) -> int: ...

    @property
    def value_vector(self) -> ValueVector | None: ...

    def set_value_vector(self, value_vector: ValueVector) -> None: ...

    @property
    def stats(self) -> dict[str, Any]: ...

    def clear(self) -> None: ...


__all__: list[str] = [
    "AuditEntry",
    "AuditEventType",
    "AuditLogger",
    "ValueDimension",
    "ValueVector",
]
