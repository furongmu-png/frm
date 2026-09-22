from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

import numpy as np

class VirtualObservation:
    """推理产生的虚拟观测，可注入模型驱动信念更新。"""

    __slots__ = ("subject", "relation", "obj", "confidence", "source")

    subject: str
    relation: str
    obj: str
    confidence: float
    source: str

    def __init__(
        self,
        subject: str,
        relation: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "inference",
    ) -> None: ...

    def to_dict(self) -> dict[str, Any]: ...

    def __repr__(self) -> str: ...


class ReasoningGraph:
    """知识图谱增强推理引擎。"""

    triples: dict[tuple[str, str, str], float]
    exceptions: list[dict[str, Any]]
    _rng: np.random.Generator
    _lock: threading.RLock
    _adj: defaultdict[str, defaultdict[str, set[str]]]
    _radj: defaultdict[str, defaultdict[str, set[str]]]
    _virtual_obs: list[VirtualObservation]

    def __init__(self, seed: int = 42) -> None: ...

    def add_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
        weight: float = 1.0,
    ) -> None: ...

    def add_facts(self, triples: list[tuple[str, str, str]]) -> int: ...

    def has_fact(self, subject: str, relation: str, obj: str) -> bool: ...

    def transitive_inference(
        self,
        relation: str = "is_a",
        max_depth: int = 10,
    ) -> list[VirtualObservation]: ...

    def analogical_reasoning(
        self,
        entity: str,
        top_k: int = 3,
        relation: str | None = None,
    ) -> list[tuple[str, float]]: ...

    def _get_neighborhood(
        self, entity: str, relation: str | None
    ) -> set[tuple[str, str]]: ...

    def _get_all_entities(self, relation: str | None) -> set[str]: ...

    @staticmethod
    def _jaccard(a: set[Any], b: set[Any]) -> float: ...

    def default_reasoning(
        self,
        general_subject: str,
        general_relation: str,
        general_obj: str,
        exception_subject: str,
        exception_obj: str,
    ) -> dict[str, Any] | None: ...

    def get_virtual_observations(self) -> list[VirtualObservation]: ...

    def peek_virtual_observations(self) -> list[VirtualObservation]: ...

    def get_neighbors(self, entity: str, relation: str) -> list[str]: ...

    def get_ancestors(
        self, entity: str, relation: str = "is_a", max_depth: int = 10
    ) -> list[str]: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = ["VirtualObservation", "ReasoningGraph"]
