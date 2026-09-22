from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_UNCERTAINTY: float


@dataclass
class CandidateExperiment:
    """一个候选干预实验。"""

    name: str
    intervention: dict[str, Any]
    predicted_info_gain: float = 0.0
    executed: bool = False
    result_fe_before: float = 0.0
    result_fe_after: float = 0.0
    actual_gain: float = 0.0


class BayesianExperimentPlanner:
    """主动实验设计：选择最大化预期信息增益的干预。"""

    _eval_interval: int
    _rng: np.random.Generator
    _lock: threading.RLock
    candidates: deque[CandidateExperiment]
    _history: deque[dict[str, Any]]
    _step_count: int
    _param_uncertainty: float

    def __init__(
        self,
        eval_interval: int = 1000,
        seed: int = 42,
        max_candidates: int = 256,
        max_history: int = 1024,
    ) -> None: ...

    def register_candidate(
        self, name: str, intervention: dict[str, Any]
    ) -> None: ...

    def default_candidates(self, dim: int = 64) -> None: ...

    def estimate_info_gain(
        self, candidate: CandidateExperiment, current_uncertainty: float
    ) -> float: ...

    def select_best(
        self, current_uncertainty: float
    ) -> CandidateExperiment | None: ...

    def record_result(
        self,
        candidate: CandidateExperiment,
        fe_before: float,
        fe_after: float,
    ) -> None: ...

    def evaluate(
        self, current_uncertainty: float, step: int
    ) -> dict[str, Any] | None: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = ["CandidateExperiment", "BayesianExperimentPlanner"]
