# src/zero_data_model/cogmem/episodic_graph.py
"""情节记忆图（EpisodicGraph）。

将 Agent 的经验组织成结构化图：节点表示离散化的状态（``Episode``），
有向边表示状态转移（``EpisodeEdge``），边权为该转移的平均自由能。
支持基于余弦相似度的“查或建”节点插入、基于 Dijkstra 的最小自由能
规划、语义向量检索以及最近经验查询。

本模块为独立、自包含、可插拔组件，仅依赖 ``numpy``；不修改
``model.py``，也不引入其它认知模块。
"""

from __future__ import annotations

import heapq
import logging
import threading
from dataclasses import dataclass

import numpy as np

_logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """单条情节记忆。

    Attributes
    ----------
    state:
        该情节对应的环境状态向量。
    action:
        在该状态下采取的动作向量（可为 ``None``，例如终态）。
    next_state:
        执行动作后到达的下一状态（可为 ``None``）。
    free_energy:
        该转移的自由能（预测误差的代理量）。
    step:
        经验时间步，用于排序与“最近”查询。
    semantic_vec:
        可选的语义嵌入向量，用于语义检索。
    """

    state: np.ndarray
    action: np.ndarray | None
    next_state: np.ndarray | None
    free_energy: float
    step: int
    semantic_vec: np.ndarray | None = None


@dataclass
class EpisodeEdge:
    """情节记忆图中的有向边。

    Attributes
    ----------
    action:
        触发该转移的动作向量（保留最新一次）。
    count:
        该边被观测到的累计次数（用于增量平均）。
    avg_free_energy:
        该转移的平均自由能（Dijkstra 边权）。
    transitions:
        转移计数字段（与 ``count`` 同步递增，便于扩展区分语义）。
    """

    action: np.ndarray | None
    count: int
    avg_free_energy: float
    transitions: int


class EpisodicGraph:
    """情节记忆图：以状态为节点、转移为边的结构化经验记忆。

    通过余弦相似度对状态去重（``similarity_threshold``），并在节点数
    超过 ``max_nodes`` 时按 ``step`` 淘汰最旧节点，保证内存有界。
    """

    def __init__(self, similarity_threshold: float = 0.95, max_nodes: int = 5000) -> None:
        """初始化情节记忆图。

        Parameters
        ----------
        similarity_threshold:
            状态去重所用的余弦相似度阈值，严格大于该值视为同一节点。
        max_nodes:
            节点数上限；超出时按 ``step`` 最小者淘汰。
        """
        self.similarity_threshold: float = float(similarity_threshold)
        self.max_nodes: int = int(max_nodes)
        # 节点表：node_id -> Episode
        self.nodes: dict[int, Episode] = {}
        # 边表：(src_id, dst_id) -> EpisodeEdge
        self.edges: dict[tuple[int, int], EpisodeEdge] = {}
        # 下一个可用节点 id
        self._next_id: int = 0
        # 邻接表：src_id -> {dst_id, ...}
        self._adj: dict[int, set[int]] = {}
        # 可重入锁：保护所有可变状态（_nodes/_edges/_adj/_next_id 等）。
        # think() 会并发调用各模块，因此所有公共方法（含只读方法）
        # 都需在 self._lock 下访问共享状态以获得一致快照。
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # 内部工具（私有方法假设调用方已持有 self._lock；RLock 可重入，
    # 因此即便直接调用也是安全的）
    # ------------------------------------------------------------------
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

    def _find_similar(self, state: np.ndarray) -> int | None:
        """在现有节点中查找与 ``state`` 余弦相似度最高的节点。

        若最高相似度严格大于 ``similarity_threshold`` 则返回该节点 id，
        否则返回 ``None``。无节点时直接返回 ``None``。
        """
        if not self.nodes:
            return None
        vec = np.asarray(state, dtype=float)
        best_id: int | None = None
        best_sim: float = -1.0
        for nid, ep in self.nodes.items():
            sim = self._cosine(vec, np.asarray(ep.state, dtype=float))
            if sim > best_sim:
                best_sim = sim
                best_id = nid
        if best_id is not None and best_sim > self.similarity_threshold:
            return best_id
        return None

    def _evict_oldest(self) -> None:
        """淘汰 ``step`` 最小的节点及其关联边。"""
        if not self.nodes:
            return
        oldest = min(self.nodes, key=lambda nid: self.nodes[nid].step)
        # 删除涉及该节点的所有边
        for key in [k for k in self.edges if oldest in k]:
            del self.edges[key]
        # 清理邻接表
        self._adj.pop(oldest, None)
        for dsts in self._adj.values():
            dsts.discard(oldest)
        del self.nodes[oldest]

    def _create_node(
        self,
        state: np.ndarray,
        action: np.ndarray | None,
        next_state: np.ndarray | None,
        free_energy: float,
        step: int,
        semantic_vec: np.ndarray | None,
    ) -> int:
        """创建新节点，并在达到上限时先淘汰最旧节点。"""
        while len(self.nodes) >= self.max_nodes and self.nodes:
            self._evict_oldest()
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Episode(
            state=state,
            action=action,
            next_state=next_state,
            free_energy=float(free_energy),
            step=int(step),
            semantic_vec=semantic_vec,
        )
        self._adj.setdefault(nid, set())
        return nid

    def _add_edge(
        self,
        src: int,
        dst: int,
        action: np.ndarray | None,
        free_energy: float,
    ) -> None:
        """新增或更新 ``(src, dst)`` 边，维护计数与平均自由能。"""
        key = (src, dst)
        if key in self.edges:
            edge = self.edges[key]
            edge.count += 1
            edge.transitions += 1
            # 增量平均自由能
            edge.avg_free_energy = (
                edge.avg_free_energy * (edge.count - 1) + float(free_energy)
            ) / edge.count
            edge.action = action  # 保留最新动作
        else:
            self.edges[key] = EpisodeEdge(
                action=action,
                count=1,
                avg_free_energy=float(free_energy),
                transitions=1,
            )
        self._adj.setdefault(src, set()).add(dst)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def insert(
        self,
        state: np.ndarray,
        action: np.ndarray | None,
        next_state: np.ndarray | None,
        free_energy: float,
        step: int,
        semantic_vec: np.ndarray | None = None,
    ) -> tuple[int, int | None]:
        """插入一条经验：为 ``state`` 与 ``next_state`` 查或建节点并连边。

        Returns
        -------
        tuple[int, int | None]
            ``(state_node_id, next_state_node_id)``；当 ``next_state`` 为
            ``None`` 时第二个元素为 ``None``。
        """
        with self._lock:
            # 查或建 state 节点
            node_id = self._find_similar(state)
            if node_id is None:
                node_id = self._create_node(
                    state, action, next_state, free_energy, step, semantic_vec
                )
            elif semantic_vec is not None:
                # 找到已存在节点时，更新其语义向量（此前仅在创建时设置，
                # 导致旧节点的 semantic_vec 永不更新）。采用 running average
                # 以平滑噪声，新向量与旧向量维度一致时才更新。
                ep = self.nodes[node_id]
                new_vec = np.asarray(semantic_vec, dtype=float)
                if (
                    ep.semantic_vec is None
                    or ep.semantic_vec.shape != new_vec.shape
                ):
                    ep.semantic_vec = new_vec.copy()
                else:
                    old = np.asarray(ep.semantic_vec, dtype=float)
                    ep.semantic_vec = (old + new_vec) * 0.5
            # 查或建 next_state 节点
            next_node_id: int | None = None
            if next_state is not None:
                next_node_id = self._find_similar(next_state)
                if next_node_id is None:
                    next_node_id = self._create_node(
                        next_state, None, None, 0.0, step, None
                    )
                # 建立有向边
                self._add_edge(node_id, next_node_id, action, free_energy)
            return (node_id, next_node_id)

    def plan(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        horizon: int = 20,
    ) -> list[int] | None:
        """基于 Dijkstra 的最小总自由能规划。

        在图中寻找从 ``start_state`` 最近节点到 ``goal_state`` 最近节点、
        边权为 ``avg_free_energy`` 的最短路径。路径跳数超过 ``horizon``
        或不可达时返回 ``None``。

        注意：Dijkstra 要求边权非负。此处对 ``avg_free_energy`` 做下限为 0
        的截断（``max(weight, 0.0)``），同时跳过非有限（NaN/inf）边权——
        非有限边权被视为不可达，从而既不破坏 Dijkstra 的非负性前提，也避免
        NaN 污染堆序。捕获常见异常时返回安全默认值 ``[]``，其它异常向上抛。

        Returns
        -------
        list[int] | None
            构成路径的节点 id 序列；不可达/找不到端点时为 ``None``。
            异常时返回 ``[]``（与 ``None`` 区分：``None`` 表示明确不可达，
            ``[]`` 表示内部出错）。
        """
        try:
            with self._lock:
                start_id = self._find_similar(start_state)
                goal_id = self._find_similar(goal_state)
                if start_id is None or goal_id is None:
                    return None
                if start_id == goal_id:
                    return [start_id]

                # Dijkstra：最小化总平均自由能
                dist: dict[int, float] = {start_id: 0.0}
                prev: dict[int, int] = {}
                visited: set[int] = set()
                heap: list[tuple[float, int]] = [(0.0, start_id)]
                while heap:
                    d, u = heapq.heappop(heap)
                    if u in visited:
                        continue
                    visited.add(u)
                    if u == goal_id:
                        break
                    for v in self._adj.get(u, set()):
                        edge = self.edges.get((u, v))
                        if edge is None:
                            continue
                        w = float(edge.avg_free_energy)
                        # 非有限（NaN/inf）边权跳过——视为不可达
                        if not np.isfinite(w):
                            continue
                        # Dijkstra 要求非负：截断到 0
                        if w < 0.0:
                            w = 0.0
                        nd = d + w
                        if v not in dist or nd < dist[v]:
                            dist[v] = nd
                            prev[v] = u
                            heapq.heappush(heap, (nd, v))

                if goal_id not in dist:
                    return None

                # 回溯路径
                path: list[int] = [goal_id]
                cur: int = goal_id
                while cur != start_id:
                    if cur not in prev:
                        return None
                    cur = prev[cur]
                    path.append(cur)
                path.reverse()

                # 路径跳数超过视野则视为不可达
                if len(path) - 1 > horizon:
                    return None
                return path
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            # 不静默吞异常：记录后返回安全默认（空列表）
            _logger.warning("EpisodicGraph.plan failed: %r", exc)
            return []

    def query(self, semantic_vec: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """按语义向量的余弦相似度返回 top-k 节点。

        仅考虑设置了 ``semantic_vec`` 的节点；防御性处理空图与零向量。

        复杂度：O(n) 线性扫描所有节点（n = node_count）；当前规模下可接受，
        若将来 n 显著增长再考虑改用近似最近邻结构（超出本次范围）。

        Returns
        -------
        list[tuple[int, float]]
            ``(node_id, similarity)`` 列表，按相似度降序排列。
        """
        try:
            with self._lock:
                if not self.nodes:
                    return []
                vec = np.asarray(semantic_vec, dtype=float)
                scored: list[tuple[int, float]] = []
                for nid, ep in self.nodes.items():
                    if ep.semantic_vec is None:
                        continue
                    sim = self._cosine(vec, np.asarray(ep.semantic_vec, dtype=float))
                    scored.append((nid, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:k]
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            _logger.warning("EpisodicGraph.query failed: %r", exc)
            return []

    def get_recent(self, n: int = 10) -> list[Episode]:
        """返回 ``step`` 最大的 n 条情节（最近经验）。

        复杂度：O(n log n) 排序（n = node_count）；当前规模下可接受，
        若需频繁调用可维护有序索引（超出本次范围）。
        """
        with self._lock:
            sorted_eps = sorted(
                self.nodes.values(), key=lambda e: e.step, reverse=True
            )
            return sorted_eps[:n]

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def node_count(self) -> int:
        """当前节点数。"""
        with self._lock:
            return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """当前边数。"""
        with self._lock:
            return len(self.edges)


__all__: list[str] = ["Episode", "EpisodeEdge", "EpisodicGraph"]
