from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .safety import SafetyChecker
from .tool_registry import ToolRegistry, ToolResult

@dataclass
class ToolDecision:
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    expected_info_gain: float = 0.0
    expected_free_energy: float = 0.0
    confidence: float = 1.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]: ...


class ToolPolicy:
    registry: ToolRegistry
    safety_checker: SafetyChecker | None
    confidence_threshold: float
    info_gain_weight: float
    _rng: np.random.Generator
    _lock: threading.RLock
    _call_history: list[dict[str, Any]]

    def __init__(
        self,
        registry: ToolRegistry,
        safety_checker: SafetyChecker | None = None,
        confidence_threshold: float = 0.6,
        info_gain_weight: float = 0.5,
        seed: int = 42,
    ) -> None: ...

    def decide(
        self,
        context: np.ndarray,
        prediction_error: float,
        confidence: float,
        tool_hint: str = "",
    ) -> ToolDecision | None: ...

    def execute(
        self, decision: ToolDecision, trigger_reason: str = ""
    ) -> ToolResult: ...

    @property
    def call_history(self) -> list[dict[str, Any]]: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = ["ToolDecision", "ToolPolicy"]
