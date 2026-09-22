from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass
class Hypothesis:
    """一个因果假设。"""

    statement: str
    cause_var: int
    effect_var: int
    direction: str
    prior_odds: float = 1.0
    bayes_factor: float = 1.0
    posterior_odds: float = 1.0
    tested: bool = False
    supported: bool = False


class HypothesisTester:
    """从因果图生成假设并用实验数据检验。"""

    _edge_threshold: float
    _rng: np.random.Generator
    _lock: threading.RLock
    hypotheses: deque[Hypothesis]

    def __init__(
        self,
        edge_threshold: float = 0.05,
        seed: int = 42,
        max_hypotheses: int = 256,
    ) -> None: ...

    def generate_from_causal_graph(
        self, transition_matrix: np.ndarray, max_hyps: int = 10
    ) -> list[Hypothesis]: ...

    def design_experiment(self, hyp: Hypothesis) -> dict[str, Any]: ...

    def update_with_data(
        self, hyp: Hypothesis, observations: np.ndarray
    ) -> None: ...

    def generate_paired_hypotheses(
        self, transition_matrix: np.ndarray, max_pairs: int = 10
    ) -> list[dict[str, Any]]: ...

    def update_causal_graph(
        self,
        transition_matrix: np.ndarray,
        cause_var: int,
        effect_var: int,
        bayes_factor: float,
    ) -> np.ndarray: ...

    def get_supported(self) -> list[Hypothesis]: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = ["Hypothesis", "HypothesisTester"]
