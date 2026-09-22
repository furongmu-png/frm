from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    """单条情节记忆。"""

    state: np.ndarray
    action: np.ndarray | None
    next_state: np.ndarray | None
    free_energy: float
    step: int
    semantic_vec: np.ndarray | None = None


@dataclass
class EpisodeEdge:
    """情节记忆图中的有向边。"""

    action: np.ndarray | None
    count: int
    avg_free_energy: float
    transitions: int


class EpisodicGraph:
    """情节记忆图：以状态为节点、转移为边的结构化经验记忆。"""

    similarity_threshold: float
    max_nodes: int
    nodes: dict[int, Episode]
    edges: dict[tuple[int, int], EpisodeEdge]
    _next_id: int
    _adj: dict[int, set[int]]
    _lock: threading.RLock

    def __init__(
        self, similarity_threshold: float = 0.95, max_nodes: int = 5000
    ) -> None: ...

    def insert(
        self,
        state: np.ndarray,
        action: np.ndarray | None,
        next_state: np.ndarray | None,
        free_energy: float,
        step: int,
        semantic_vec: np.ndarray | None = None,
    ) -> tuple[int, int | None]: ...

    def plan(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        horizon: int = 20,
    ) -> list[int] | None: ...

    def query(
        self, semantic_vec: np.ndarray, k: int = 5
    ) -> list[tuple[int, float]]: ...

    def get_recent(self, n: int = 10) -> list[Episode]: ...

    @property
    def node_count(self) -> int: ...

    @property
    def edge_count(self) -> int: ...


__all__: list[str] = ["Episode", "EpisodeEdge", "EpisodicGraph"]
