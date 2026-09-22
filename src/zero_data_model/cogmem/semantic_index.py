# src/zero_data_model/cogmem/semantic_index.py
"""语义检索索引（SemanticIndex）。

在情节记忆之上提供轻量级语义检索：以 ``(semantic_vec, node_id, metadata)``
三元组线性存储，支持暴力余弦相似度 top-k 查询、按标签过滤以及 FIFO
容量限制。无外部依赖（仅 ``numpy``），可独立于 ``EpisodicGraph`` 使用。
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class SemanticIndex:
    """暴力式语义检索索引（FIFO 有界）。

    Attributes
    ----------
    dim:
        语义向量的期望维度（仅作记录，不做硬性校验）。``dim`` 是公共 API 契约
        的一部分（``model.py`` 通过 ``SemanticIndex(dim=dim)`` 构造），虽本类
        内部不强制使用，但保留以便未来扩展维度校验，且不破坏既有调用方。
    max_size:
        最大条目数；超出时按 FIFO 淘汰最旧条目。
    entries:
        ``(semantic_vec, node_id, metadata)`` 三元组的 deque（``maxlen=max_size``）。
    """

    def __init__(self, dim: int = 64, max_size: int = 10000) -> None:  # pylint: disable=unused-argument
        """初始化语义索引。

        Parameters
        ----------
        dim:
            语义向量的期望维度。保留为公共 API 契约（``model.py`` 传入），
            本类内部目前不强制校验，故标记 ``unused-argument``。
        max_size:
            最大条目数，超出时淘汰最旧条目。
        """
        self.dim: int = int(dim)
        self.max_size: int = int(max_size)
        # 条目表：(semantic_vec, node_id, metadata)
        # 使用 deque(maxlen=...) 实现 O(1) FIFO 淘汰，替代原本 list.pop(0) 的 O(n)。
        self.entries: deque[tuple[np.ndarray, int, dict]] = deque(maxlen=self.max_size)
        # 可重入锁：保护 entries 的所有读写。think() 会并发调用各模块，
        # 因此 add/query/query_by_tag/size 均需在 self._lock 下访问。
        self._lock: threading.RLock = threading.RLock()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度。

        - 任一为零向量或包含 NaN/inf 时返回 0.0，避免 NaN 传播。
        """
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        # 防御 NaN/inf：任一向量含非有限值则视为不可比较
        if not np.isfinite(aa).all() or not np.isfinite(bb).all():
            return 0.0
        na = float(np.linalg.norm(aa))
        nb = float(np.linalg.norm(bb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(aa, bb) / (na * nb))

    def add(
        self,
        semantic_vec: np.ndarray,
        node_id: int,
        metadata: dict | None = None,
    ) -> None:
        """追加一条语义条目，并在超出 ``max_size`` 时按 FIFO 淘汰。

        ``semantic_vec`` 与 ``metadata`` 均做拷贝存储，避免调用方后续修改
        泄漏到索引内部。
        """
        # 拷贝向量与 metadata，防止调用方后续修改导致别名问题
        vec_copy = np.asarray(semantic_vec, dtype=float).copy()
        meta_copy = dict(metadata) if metadata is not None else {}
        with self._lock:
            # deque(maxlen=...) 自动在超出容量时从左端 O(1) 弹出最旧条目
            self.entries.append((vec_copy, int(node_id), meta_copy))

    def query(
        self, vec: np.ndarray, k: int = 5
    ) -> list[tuple[int, float, dict]]:
        """暴力计算与 ``vec`` 的余弦相似度，返回 top-k 条目。

        复杂度：O(n) 线性扫描所有条目（n = size）；当前规模下可接受。

        Returns
        -------
        list[tuple[int, float, dict]]
            ``(node_id, similarity, metadata)`` 列表，按相似度降序排列。
        """
        with self._lock:
            if not self.entries:
                return []
            q = np.asarray(vec, dtype=float)
            scored: list[tuple[int, float, dict]] = []
            for sem, nid, meta in self.entries:
                sim = self._cosine(q, sem)
                scored.append((nid, sim, meta))
            scored.sort(key=lambda x: x[1], reverse=True)
            # 注意使用 [:k] 而非 [:k+1]：k=0 时返回 []。
            return scored[:k]

    @staticmethod
    def _has_tag(meta: dict, tag: str) -> bool:
        """判断 metadata 是否包含指定标签。

        约定：``metadata["tags"]`` 为标签集合（list/tuple/set），或
        ``metadata["tag"]`` 为单个标签字符串。
        """
        tags = meta.get("tags")
        if isinstance(tags, (list, tuple, set, frozenset)) and tag in tags:
            return True
        return meta.get("tag") == tag

    def query_by_tag(self, tag: str, k: int = 5) -> list[tuple[int, dict]]:
        """过滤包含指定标签的条目，最多返回 k 条。

        Returns
        -------
        list[tuple[int, dict]]
            ``(node_id, metadata)`` 列表（按插入顺序）。
        """
        # k<=0 时直接返回空列表，避免原实现因 ``len(results) >= k`` 在
        # k=0 时第一次匹配即满足而错误返回 1 条的 off-by-one。
        if k <= 0:
            return []
        results: list[tuple[int, dict]] = []
        with self._lock:
            for _sem, nid, meta in self.entries:
                if self._has_tag(meta, tag):
                    results.append((nid, meta))
                    if len(results) >= k:
                        break
        return results

    @property
    def size(self) -> int:
        """当前条目数。"""
        with self._lock:
            return len(self.entries)


__all__: list[str] = ["SemanticIndex"]
