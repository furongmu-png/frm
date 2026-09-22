from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

class _WorldModel(Protocol):
    def __call__(self, context: dict[str, Any], action: Any) -> dict[str, Any]: ...


@dataclass
class CounterfactualResult:
    factual_action: Any
    alternative_action: Any
    factual_free_energy: float
    counterfactual_free_energy: float
    factual_prediction_error: float
    counterfactual_prediction_error: float
    factual_outcome: str
    counterfactual_outcome: str
    free_energy_delta: float
    prediction_error_change_pct: float
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]: ...


class CounterfactualExplainer:
    _world_model: _WorldModel | None
    _history: deque[CounterfactualResult]
    _lock: threading.RLock

    def __init__(
        self,
        world_model: _WorldModel | None = None,
        history_limit: int = 1000,
    ) -> None: ...

    def set_world_model(self, world_model: _WorldModel) -> None: ...

    def explain(
        self,
        factual_state: dict[str, Any],
        alternative_action: Any,
    ) -> CounterfactualResult: ...

    def get_history(self, n: int = 20) -> list[dict[str, Any]]: ...

    def clear_history(self) -> None: ...

    @property
    def n_explanations(self) -> int: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = ["CounterfactualExplainer", "CounterfactualResult"]
