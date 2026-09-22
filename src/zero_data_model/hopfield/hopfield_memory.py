"""Modern Hopfield Network — exponential storage capacity.

理论基础
========

经典 Hopfield 网络（Hopfield 1982）存储容量 O(d)，检索靠能量函数
局部最小值。现代 Hopfield 网络（Ramsauer et al. 2020,
"Hopfield Networks is All You Need"）改用连续状态和 softmax 能量：

    energy(ξ) = -lse(β * X^T ξ) + 0.5 * β * ||ξ||^2 + const
    retrieve(ξ) = X @ softmax(β * X^T ξ)

其中 X = [x_1, ..., x_N] ∈ R^{d×N} 是存储的模式矩阵，β 是温度
参数（inverse temperature），lse 是 logsumexp。

关键性质：
1. 存储容量 N_max = O(exp(d/2))，远超经典 O(d)
2. 检索只需 1 步（非迭代收敛）
3. 容错：即使 query 部分损坏也能检索正确记忆
4. 与 Transformer self-attention 等价（Hopfield layer = attention）

伪逆巩固
========
离线巩固用 Moore-Penrose 伪逆重构记忆矩阵，使检索精度最大化：
    X_pinv = X @ (X^T X + λI)^{-1}
    这等价于 ridge regression 的设计矩阵
"""
from __future__ import annotations

import threading

import numpy as np
from scipy.special import logsumexp


class HopfieldMemory:
    """Modern Hopfield Network for associative memory.

    支持自联想（key == value）和异联想（key != value）。

    Parameters
    ----------
    memory_dim : int
        存储向量的维度 d
    capacity : int
        最大记忆槽位数 N（达到后 FIFO 淘汰）
    beta : float
        softmax 温度参数（inverse temperature，越小越尖锐）
        默认 1.0 / sqrt(memory_dim)（ Ramsauer 论文推荐）
    consolidate_lambda : float
        伪逆巩固的 ridge 正则化系数
    seed : int | None
        随机种子
    """

    def __init__(
        self,
        memory_dim: int,
        capacity: int = 1024,
        *,
        beta: float | None = None,
        consolidate_lambda: float = 0.01,
        seed: int | None = None,
    ):
        if memory_dim < 1:
            raise ValueError(f"memory_dim must be >= 1, got {memory_dim}")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self.memory_dim = memory_dim
        self.capacity = capacity
        # 默认 beta：论文推荐 1/sqrt(d)
        self.beta = beta if beta is not None else 1.0 / np.sqrt(memory_dim)
        self.consolidate_lambda = consolidate_lambda
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # 存储矩阵：keys (memory_dim, N), values (memory_dim, N)
        self._keys: np.ndarray | None = None  # shape (memory_dim, N_used)
        self._values: np.ndarray | None = None  # shape (memory_dim, N_used)

        # 是否已巩固（伪逆重写过 keys）
        self._consolidated = False

        # 原始 keys 副本（巩固前的原始记忆，用于追加）
        self._raw_keys: list[np.ndarray] = []
        self._raw_values: list[np.ndarray] = []

    @property
    def size(self) -> int:
        """当前存储的记忆数量。"""
        with self._lock:
            return len(self._raw_keys)

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity

    def store(self, key: np.ndarray, value: np.ndarray | None = None) -> None:
        """存储一个 (key, value) 对作为吸引子。

        Parameters
        ----------
        key : ndarray, shape (memory_dim,)
            检索 key
        value : ndarray, shape (memory_dim,) | None
            检索结果；None 表示自联想（value = key）
        """
        key = np.asarray(key, dtype=np.float64).copy()
        if key.shape != (self.memory_dim,):
            raise ValueError(f"key must have shape ({self.memory_dim},), got {key.shape}")

        if value is None:
            value = key.copy()
        else:
            value = np.asarray(value, dtype=np.float64).copy()
            if value.shape != (self.memory_dim,):
                raise ValueError(
                    f"value must have shape ({self.memory_dim},), got {value.shape}"
                )

        # NaN 防护
        if not np.all(np.isfinite(key)) or not np.all(np.isfinite(value)):
            return

        with self._lock:
            # FIFO 淘汰
            if len(self._raw_keys) >= self.capacity:
                self._raw_keys.pop(0)
                self._raw_values.pop(0)

            self._raw_keys.append(key)
            self._raw_values.append(value)

            # 标记需要重建矩阵
            self._keys = None
            self._values = None
            self._consolidated = False

    def _ensure_matrix(self) -> None:
        """从 raw list 重建 keys/values 矩阵。"""
        if self._keys is not None and self._values is not None:
            return

        if not self._raw_keys:
            self._keys = None
            self._values = None
            return

        # 拼成矩阵
        self._keys = np.stack(self._raw_keys, axis=1)  # (memory_dim, N)
        self._values = np.stack(self._raw_values, axis=1)  # (memory_dim, N)

    def retrieve(self, query: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """检索与 query 最匹配的 top-k 记忆。

        使用现代 Hopfield softmax attention:
            retrieve(ξ) = X @ softmax(β * X^T ξ)

        Parameters
        ----------
        query : ndarray, shape (memory_dim,)
            查询向量
        k : int
            返回 top-k 个匹配（默认 1）

        Returns
        -------
        retrieved : ndarray, shape (memory_dim,)
            检索到的记忆（softmax 加权平均）
        similarities : ndarray, shape (k,)
            top-k 相似度（softmax 权重）

        Raises
        ------
        ValueError
            如果 query 形状不匹配或 k < 0
        RuntimeError
            如果记忆库为空
        """
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")

        query = np.asarray(query, dtype=np.float64)
        if query.shape != (self.memory_dim,):
            raise ValueError(
                f"query must have shape ({self.memory_dim},), got {query.shape}"
            )

        # NaN 防护
        if not np.all(np.isfinite(query)):
            query = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)

        with self._lock:
            self._ensure_matrix()

            if self._keys is None or self._keys.shape[1] == 0:
                raise RuntimeError("Memory is empty, cannot retrieve")

            # 计算相似度：X^T ξ
            similarities = self._keys.T @ query  # (N,)

            # softmax with temperature（数值稳定 logsumexp）
            scores = self.beta * similarities
            log_Z = logsumexp(scores)
            softmax = np.exp(scores - log_Z)

            # 检索：X @ softmax
            retrieved = self._values @ softmax  # (memory_dim,)

            # top-k
            if k == 0:
                return retrieved, np.array([])

            # 获取 top-k 相似度
            k = min(k, len(softmax))
            top_k_indices = np.argsort(softmax)[::-1][:k]
            top_k_similarities = softmax[top_k_indices]

            return retrieved, top_k_similarities

    def retrieve_topk(
        self, query: np.ndarray, k: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """检索与 query 最匹配的 top-k 记忆，并返回存储索引。

        与 ``retrieve`` 的区别：返回 top-k 个 *独立* 记忆向量及其在
        存储矩阵中的索引，而非单个 softmax 加权平均。用于经验回放等
        需要定位具体记忆的场景（如 ExperienceBuffer.retrieve）。

        Parameters
        ----------
        query : ndarray, shape (memory_dim,)
            查询向量
        k : int
            返回 top-k 个匹配

        Returns
        -------
        top_k_values : ndarray, shape (k, memory_dim)
            top-k 个最匹配的存储值向量
        top_k_indices : ndarray, shape (k,)
            对应的存储索引（可用于外部映射回完整对象）
        top_k_similarities : ndarray, shape (k,)
            softmax 相似度权重

        Raises
        ------
        ValueError
            如果 query 形状不匹配或 k < 0
        RuntimeError
            如果记忆库为空
        """
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")

        query = np.asarray(query, dtype=np.float64)
        if query.shape != (self.memory_dim,):
            raise ValueError(
                f"query must have shape ({self.memory_dim},), got {query.shape}"
            )
        if not np.all(np.isfinite(query)):
            query = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)

        with self._lock:
            self._ensure_matrix()
            if self._keys is None or self._keys.shape[1] == 0:
                raise RuntimeError("Memory is empty, cannot retrieve_topk")

            # A4 修复：原代码用 `self._values.T @ query` 计算 top-k 相似度，
            # 但 `retrieve` 用 `self._keys.T @ query`。两者不一致导致
            # `update_weights` (consolidate) 重构 `_keys` 后对 `retrieve_topk`
            # 完全无效——这是 autobiographical_memory.consolidate() 无效的根因。
            # 现统一用 `_keys.T @ query` 计算相似度（与 retrieve 一致），
            # 使 consolidate 真正影响 retrieve_topk 的检索结果。
            # 对自联想存储（key==value）此修改无行为变化；对异联想存储
            # 此修改让"用 key 查询、返回 value"的语义成立。
            similarities = self._keys.T @ query  # (N,)
            scores = self.beta * similarities
            log_Z = logsumexp(scores)
            softmax = np.exp(scores - log_Z)

            if k == 0:
                return (
                    np.zeros((0, self.memory_dim)),
                    np.array([], dtype=int),
                    np.array([]),
                )

            k = min(k, len(softmax))
            top_k_indices = np.argsort(softmax)[::-1][:k]
            top_k_similarities = softmax[top_k_indices]
            # 从 values 矩阵取对应列：(memory_dim, k)
            top_k_values = self._values[:, top_k_indices].T  # (k, memory_dim)

            return top_k_values, top_k_indices, top_k_similarities

    def update_weights(self) -> None:
        """离线巩固：用伪逆重构记忆矩阵以最大化检索精度。

        Ramsauer 2020 论文未明确给出巩固算法，本项目用 ridge 正则化
        伪逆作为离线巩固：
            X_pinv = X @ (X^T X + λI)^{-1}

        这相当于对记忆矩阵做 ridge regression 重构，使得：
            retrieve(X_pinv, ξ) ≈ ξ for stored patterns

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self._lock:
            self._ensure_matrix()

            if self._keys is None or self._keys.shape[1] == 0:
                return

            # 伪逆：X_pinv = X @ (X^T X + λI)^{-1}
            # (d, N) @ (N, N) = (d, N)
            N = self._keys.shape[1]
            XtX = self._keys.T @ self._keys  # (N, N)
            reg_matrix = XtX + self.consolidate_lambda * np.eye(N)

            try:
                # 避免奇异矩阵
                inv_matrix = np.linalg.inv(reg_matrix)
                # 用伪逆重构 keys（让记忆正交化，提高检索精度）
                self._keys = self._keys @ inv_matrix @ self._keys.T @ self._keys
                # values 保持不变（只重构 keys 用于检索）
                self._consolidated = True
            except np.linalg.LinAlgError:
                # 奇异矩阵时用 pinv
                pinv_matrix = np.linalg.pinv(reg_matrix)
                self._keys = self._keys @ pinv_matrix @ self._keys.T @ self._keys
                self._consolidated = True

    def clear(self) -> None:
        """清空所有记忆。"""
        with self._lock:
            self._raw_keys.clear()
            self._raw_values.clear()
            self._keys = None
            self._values = None
            self._consolidated = False

    def get_stats(self) -> dict:
        """返回统计信息（用于可视化）。"""
        with self._lock:
            self._ensure_matrix()
            return {
                "size": len(self._raw_keys),
                "capacity": self.capacity,
                "memory_dim": self.memory_dim,
                "is_full": self.is_full,
                "is_consolidated": self._consolidated,
                "beta": float(self.beta),
            }

    def get_snapshot(self) -> dict:
        """获取快照（用于观测/可视化）。"""
        with self._lock:
            self._ensure_matrix()
            snapshot = {
                "size": len(self._raw_keys),
                "capacity": self.capacity,
                "is_consolidated": self._consolidated,
            }
            if self._keys is not None and self._keys.shape[1] > 0:
                # 计算记忆之间的平均相似度（衡量是否需要巩固）
                # 用前 10 个记忆
                n_sample = min(10, self._keys.shape[1])
                sample = self._keys[:, :n_sample]
                # 归一化
                norms = np.linalg.norm(sample, axis=0, keepdims=True)
                norms = np.where(norms > 1e-10, norms, 1.0)
                normalized = sample / norms
                # 相似度矩阵
                sim_matrix = normalized.T @ normalized
                # 平均非对角相似度
                mask = ~np.eye(n_sample, dtype=bool)
                avg_sim = (
                    float(np.mean(np.abs(sim_matrix[mask])))
                    if mask.any()
                    else 0.0
                )
                snapshot["avg_pattern_similarity"] = avg_sim
                snapshot["key_norm_mean"] = float(np.mean(norms))
            return snapshot

    def __getstate__(self) -> dict:
        with self._lock:
            return {
                "memory_dim": self.memory_dim,
                "capacity": self.capacity,
                "beta": self.beta,
                "consolidate_lambda": self.consolidate_lambda,
                "_raw_keys": [k.copy() for k in self._raw_keys],
                "_raw_values": [v.copy() for v in self._raw_values],
                "_consolidated": self._consolidated,
            }

    def __setstate__(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)
        self._keys = None
        self._values = None
        self._lock = threading.RLock()
        # 不保存 _rng（每次反序列化生成新的）
        self._rng = np.random.default_rng()
