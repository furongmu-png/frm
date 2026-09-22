from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np


class SemanticIndex:
    """暴力式语义检索索引（FIFO 有界）。"""

    dim: int
    max_size: int
    entries: deque[tuple[np.ndarray, int, dict[str, Any]]]
    _lock: threading.RLock

    def __init__(self, dim: int = 64, max_size: int = 10000) -> None: ...

    def add(
        self,
        semantic_vec: np.ndarray,
        node_id: int,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def query(
        self, vec: np.ndarray, k: int = 5
    ) -> list[tuple[int, float, dict[str, Any]]]: ...

    def query_by_tag(
        self, tag: str, k: int = 5
    ) -> list[tuple[int, dict[str, Any]]]: ...

    @property
    def size(self) -> int: ...


__all__: list[str] = ["SemanticIndex"]
