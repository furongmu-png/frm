from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LogicRule:
    """一条模糊逻辑规则。"""

    name: str
    antecedents: list[str]
    consequent: str
    weight: float = 1.0
    description: str = ""


class LogicLayer:
    """模糊逻辑约束层。"""

    rules: list[LogicRule]
    predicate_values: dict[str, float]
    _lock: threading.RLock

    def __init__(self) -> None: ...

    def add_rule(
        self,
        name: str,
        antecedents: list[str],
        consequent: str,
        weight: float = 1.0,
        description: str = "",
    ) -> None: ...

    def set_predicate(self, name: str, value: float) -> None: ...

    def t_norm(self, a: float, b: float) -> float: ...

    def t_conorm(self, a: float, b: float) -> float: ...

    def evaluate_rule(self, rule: LogicRule) -> tuple[float, float]: ...

    def check_all(self) -> dict[str, Any]: ...

    def get_penalty_signal(self) -> np.ndarray: ...


__all__: list[str] = ["LogicRule", "LogicLayer"]
