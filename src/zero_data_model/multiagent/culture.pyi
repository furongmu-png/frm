from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass
class Generation:
    """单代际的快照：知识规模、平均自由能与学习步数。"""

    gen_id: int
    knowledge_graph_size: int
    mean_free_energy: float
    n_steps: int
    weights_snapshot: dict[str, Any] | None = None


class CulturePropagation:
    """跨代际文化演化跟踪器。"""

    _lock: threading.RLock
    _rng: np.random.Generator
    generations: deque[Generation]
    _current_gen: int

    def __init__(
        self,
        max_generations: int = 256,
        seed: int = 42,
    ) -> None: ...

    def record_generation(
        self,
        knowledge_graph_size: int,
        mean_free_energy: float,
        n_steps: int,
        weights: dict[str, Any] | None = None,
    ) -> Generation: ...

    def inherit_weights(
        self, parent_weights: dict[str, Any], mutation_rate: float = 0.01
    ) -> dict[str, Any]: ...

    def compute_cultural_acceleration(self) -> float: ...

    def get_knowledge_curve(self) -> list[tuple[int, int]]: ...

    @property
    def stats(self) -> dict[str, Any]: ...

    # Phase 3 扩展
    def save_snapshot(self) -> dict[str, Any]: ...

    def load_snapshot(self, snapshot: dict[str, Any]) -> None: ...

    def get_task_completion_curve(self) -> list[tuple[int, int]]: ...

    def get_cultural_acceleration_curve(self) -> list[tuple[int, float]]: ...


__all__: list[str] = ["Generation", "CulturePropagation"]
