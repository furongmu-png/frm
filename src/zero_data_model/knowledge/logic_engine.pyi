from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass
class LogicRuleV2:
    """一条字典式模糊逻辑规则。"""

    antecedents: list[str]
    consequent: str
    weight: float = 1.0
    name: str = ""
    description: str = ""
    def __post_init__(self) -> None: ...


class LogicEngine:
    """可微模糊推理与矛盾检测引擎。"""

    rules: list[LogicRuleV2]
    predicate_embeddings: dict[str, np.ndarray]
    _predicate_dim: int
    _lr: float
    _contradiction_gain: float
    _rng: np.random.Generator
    _lock: threading.RLock
    _rule_stats: list[dict[str, int]]
    _last_contradiction: float

    def __init__(
        self,
        predicate_dim: int = 16,
        lr: float = 0.05,
        contradiction_gain: float = 1.0,
        seed: int = 42,
    ) -> None: ...

    def add_rule(self, rule: dict[str, Any]) -> None: ...

    def _ensure_predicate(self, name: str) -> np.ndarray: ...

    @staticmethod
    def t_norm(a: float, b: float) -> float: ...

    @staticmethod
    def implication(a: float, b: float) -> float: ...

    def check_consistency(self, belief_state: dict[str, float]) -> float: ...

    def resolve_contradiction(
        self,
        belief_state: dict[str, float],
        error: np.ndarray,
    ) -> np.ndarray: ...

    def _hebbian_adjust(self, rule_idx: int, effective: bool) -> None: ...

    def get_violations(
        self, belief_state: dict[str, float]
    ) -> list[dict[str, Any]]: ...

    def get_predicate_embedding(self, name: str) -> np.ndarray: ...

    @property
    def last_contradiction(self) -> float: ...

    @property
    def stats(self) -> dict[str, Any]: ...

    @staticmethod
    def _clamp_truth(value: float) -> float: ...


__all__: list[str] = ["LogicRuleV2", "LogicEngine"]
