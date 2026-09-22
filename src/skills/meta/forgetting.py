"""遗忘管理：在 Hopfield 风格记忆模块中引入指数衰减。

未被检索的记忆权重随时间衰减；衰减到阈值以下的记忆被移除，
防止记忆过载，为新知识腾出空间。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """单个记忆条目。"""
    item_id: str
    vector: np.ndarray
    age: int = 0  # 距上次检索的步数
    weight: float = 1.0  # 当前权重 [0, 1]
    n_retrievals: int = 0
    created_at: int = 0
    last_retrieved_at: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryDecay(SkillBase):
    """遗忘管理模块。

    - ``store(item_id, vector, metadata)``：存储记忆。
    - ``retrieve(query, k=1)``：检索最近的 k 个记忆，命中项 age 归 0。
    - ``decay()``：所有记忆的 age += 1，weight *= decay_factor。
    - ``prune()``：移除 weight < prune_threshold 的记忆。

    与 ZeroDataModel 集成：
    - ``process``：在 ``think()`` 的元认知阶段调用 ``decay`` + ``prune``。
    """

    name = "forgetting"
    dimension = "meta"

    def __init__(
        self,
        *,
        decay_factor: float = 0.95,
        prune_threshold: float = 0.1,
        max_capacity: int = 1000,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        if not 0.0 < decay_factor <= 1.0:
            raise ValueError("decay_factor must be in (0, 1]")
        if not 0.0 <= prune_threshold <= 1.0:
            raise ValueError("prune_threshold must be in [0, 1]")
        self.decay_factor = decay_factor
        self.prune_threshold = prune_threshold
        self.max_capacity = max_capacity
        self._memories: dict[str, MemoryItem] = {}
        self._step: int = 0
        self._n_evicted: int = 0
        self._n_retrievals: int = 0
        self._rng = np.random.default_rng(13)

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #

    def store(
        self,
        item_id: str,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """存储或更新记忆条目。"""
        vec = np.asarray(vector, dtype=np.float64).flatten()
        if item_id in self._memories:
            # 更新向量，重置 age 和 weight
            item = self._memories[item_id]
            item.vector = vec
            item.age = 0
            item.weight = 1.0
            item.last_retrieved_at = self._step
            if metadata is not None:
                item.metadata = dict(metadata)
            return item
        # 容量满时按 weight 最低淘汰一个
        if len(self._memories) >= self.max_capacity:
            self._evict_one()
        item = MemoryItem(
            item_id=item_id,
            vector=vec,
            age=0,
            weight=1.0,
            created_at=self._step,
            last_retrieved_at=self._step,
            metadata=metadata or {},
        )
        self._memories[item_id] = item
        return item

    def retrieve(
        self, query: np.ndarray, k: int = 1
    ) -> list[MemoryItem]:
        """按余弦相似度检索 top-k。

        命中项 age 归 0、n_retrievals += 1。
        """
        if not self._memories:
            return []
        q = np.asarray(query, dtype=np.float64).flatten()
        q_norm = np.linalg.norm(q) + 1e-9
        items = list(self._memories.values())
        sims = np.zeros(len(items), dtype=np.float64)
        for i, item in enumerate(items):
            v = item.vector
            v_norm = np.linalg.norm(v) + 1e-9
            sims[i] = float(np.dot(q, v) / (q_norm * v_norm))
        # 取 top-k
        k = min(k, len(items))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        result = []
        for idx in top_idx:
            item = items[idx]
            item.age = 0
            item.n_retrievals += 1
            item.last_retrieved_at = self._step
            self._n_retrievals += 1
            result.append(item)
        return result

    def decay(self) -> int:
        """对所有记忆执行一步衰减，返回当前活跃数量。"""
        self._step += 1
        for item in self._memories.values():
            item.age += 1
            item.weight *= self.decay_factor
        return len(self._memories)

    def prune(self) -> int:
        """移除 weight < prune_threshold 的记忆，返回移除数量。"""
        to_remove = [
            item_id
            for item_id, item in self._memories.items()
            if item.weight < self.prune_threshold
        ]
        for item_id in to_remove:
            del self._memories[item_id]
        self._n_evicted += len(to_remove)
        return len(to_remove)

    def step_and_prune(self) -> dict[str, Any]:
        """执行一次衰减 + 清理（在 ``think()`` 中调用）。"""
        active = self.decay()
        pruned = self.prune()
        return {
            "active_memories": active - pruned,
            "pruned": pruned,
            "step": self._step,
        }

    def _evict_one(self) -> None:
        """容量满时按 weight 最低淘汰一个。"""
        if not self._memories:
            return
        # 找 weight 最小且 age 最大的项
        victim_id = min(
            self._memories.keys(),
            key=lambda k: (self._memories[k].weight, -self._memories[k].age),
        )
        del self._memories[victim_id]
        self._n_evicted += 1

    # ------------------------------------------------------------------ #
    # 快照与统计
    # ------------------------------------------------------------------ #

    def usage_rate(self) -> float:
        if not self._memories:
            return 0.0
        n_used = sum(1 for it in self._memories.values() if it.n_retrievals > 0)
        return n_used / len(self._memories)

    def stats(self) -> dict[str, Any]:
        weights = [it.weight for it in self._memories.values()]
        ages = [it.age for it in self._memories.values()]
        return {
            "n_memories": len(self._memories),
            "capacity": self.max_capacity,
            "utilization": (
                len(self._memories) / self.max_capacity if self.max_capacity else 0.0
            ),
            "usage_rate": self.usage_rate(),
            "n_evicted": self._n_evicted,
            "n_retrievals": self._n_retrievals,
            "mean_weight": float(np.mean(weights)) if weights else 0.0,
            "mean_age": float(np.mean(ages)) if ages else 0.0,
            "step": self._step,
        }

    # ------------------------------------------------------------------ #
    # SkillBase 接口
    # ------------------------------------------------------------------ #

    def process(self, ctx: SkillContext) -> SkillResult:
        result = self.step_and_prune()
        return SkillResult(
            name=self.name,
            data={
                **result,
                "stats": self.stats(),
            },
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "ready": True,
            "data": {
                "n_memories": len(self._memories),
                "capacity": self.max_capacity,
                "usage_rate": self.usage_rate(),
                "n_evicted": self._n_evicted,
                "n_retrievals": self._n_retrievals,
                "step": self._step,
            },
        }
