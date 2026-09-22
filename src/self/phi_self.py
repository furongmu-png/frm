"""PhiSelf — self-relevant integrated information Φ_self.

理论基础
========
整合信息理论 (IIT, Tononi 2008) 用 Φ 衡量系统的意识水平：Φ = 系统作为
整体产生的信息减去各部分独立产生的信息之和。Φ 越高，系统越"有意识"。

本模块计算"自我 Φ"——仅考虑与自我图式相关模块（GWT、元认知、本体感觉）
的整合信息。这与全局 Φ 的区别在于：全局 Φ 衡量整个系统的整合性，而
自我 Φ 聚焦于"自我相关"的子网络。

当自我 Φ 高时，系统具有高度整合的自我表征（高自我意识）；当 Φ 降低时
（如"睡眠"期间），自我意识变弱。冲突情境下自我 Φ 应短暂波动。

实现复用 PhiCalculator 的高斯互信息估计方法，但仅对自我相关模块子集计算。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class PhiSelfState:
    """自我 Φ 计算结果。"""
    phi_self: float                    # 自我 Φ 值
    is_self_aware: bool                # 是否自我意识活跃
    self_module_count: int             # 参与计算的自我模块数
    partition: Optional[tuple[list[int], list[int]]] = None  # 最小划分
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# 主类
# ------------------------------------------------------------------ #
class PhiSelf:
    """自我相关整合信息 Φ_self 计算器。

    仅考虑与自我图式相关的模块子集，计算它们的整合信息。

    Parameters
    ----------
    n_self_modules : int
        自我相关模块数（默认 4：GWT、元认知、本体感觉、自我图式）
    history_window : int
        历史窗口大小
    sleep_threshold : float
        自我意识阈值（Φ_self 低于此值视为"睡眠"）
    """

    def __init__(
        self,
        n_self_modules: int = 4,
        history_window: int = 50,
        sleep_threshold: float = 0.1,
    ):
        if n_self_modules < 2:
            raise ValueError(
                f"n_self_modules must be >= 2 (need >=2 to compute MI), "
                f"got {n_self_modules}"
            )
        if history_window < 2:
            raise ValueError(
                f"history_window must be >= 2, got {history_window}"
            )

        self.n_self_modules = int(n_self_modules)
        self.history_window = int(history_window)
        self.sleep_threshold = float(sleep_threshold)
        self._lock = threading.RLock()

        # 每个自我模块的状态历史（标量范数序列）
        self._histories: list[list[float]] = [
            [] for _ in range(self.n_self_modules)
        ]
        self._phi_history: list[float] = []
        self._last_phi = 0.0
        self._last_partition: Optional[tuple[list[int], list[int]]] = None
        self._step_count = 0

    # ---------------------------------------------------------------- #
    # 记录
    # ---------------------------------------------------------------- #
    def record(self, self_module_states: list[np.ndarray]) -> None:
        """记录一步的各自我模块隐状态。

        Parameters
        ----------
        self_module_states : list of ndarray
            各自我模块的隐状态（每个会被压缩为范数标量）
        """
        states = list(self_module_states)
        # 对齐到 n_self_modules
        while len(states) < self.n_self_modules:
            states.append(np.zeros(1))
        states = states[: self.n_self_modules]

        with self._lock:
            self._step_count += 1
            for i, s in enumerate(states):
                arr = np.asarray(s, dtype=np.float64).flatten()
                # 压缩为范数标量（与 PhiCalculator 一致的降维策略）
                norm = float(np.linalg.norm(arr)) if arr.size > 0 else 0.0
                if not np.isfinite(norm):
                    norm = 0.0
                self._histories[i].append(norm)
                # 裁剪历史
                if len(self._histories[i]) > self.history_window:
                    self._histories[i] = self._histories[i][
                        -self.history_window :
                    ]

    # ---------------------------------------------------------------- #
    # 计算 Φ_self
    # ---------------------------------------------------------------- #
    def compute_phi(self) -> float:
        """计算自我 Φ。

        算法（与 PhiCalculator 一致的高斯近似）：
          Φ = whole_information - min_partition_information
          其中 information 用高斯互信息估计。
        """
        with self._lock:
            # 需要足够的历史
            min_len = min(len(h) for h in self._histories)
            if min_len < 2:
                self._last_phi = 0.0
                return 0.0

            # 构建状态矩阵 (min_len, n_self_modules)
            matrix = np.zeros((min_len, self.n_self_modules))
            for i in range(self.n_self_modules):
                h = self._histories[i][-min_len:]
                matrix[:, i] = h

            # 整体信息 = trace(MI) + 非对角元素和
            mi_matrix = self._mutual_information_matrix(matrix)
            whole_info = float(
                np.trace(mi_matrix)
                + np.sum(mi_matrix[np.triu_indices(self.n_self_modules, k=1)])
            )

            # 贪心搜索最小划分
            n = self.n_self_modules
            min_loss = float("inf")
            best_partition: Optional[tuple[list[int], list[int]]] = None

            # P1 修复：原代码 `for mask in range(1, (1<<n) - 1)` 枚举了
            # 每个 (A, B) 划分两次（mask 和 ~mask），重复计算。
            # 改为只枚举 mask < ~mask 的部分（即 mask 的最高有效位为 0），
            # 把计算量减半。对于 n=4：原 14 次 → 修复后 7 次。
            # 不变量：mask 的最高位（bit n-1）必须为 0，否则该 mask 是
            # 另一个 mask 的补集，已经在更小的 mask 中计算过。
            half_mask_upper = 1 << (n - 1)  # 用于判断最高位
            for mask in range(1, half_mask_upper):
                part_a = [i for i in range(n) if mask & (1 << i)]
                part_b = [i for i in range(n) if not (mask & (1 << i))]
                if not part_a or not part_b:
                    continue
                # 划分信息 = 两部分对角元素之和
                part_info = float(
                    np.sum(mi_matrix[np.ix_(part_a, part_a)])
                    + np.sum(mi_matrix[np.ix_(part_b, part_b)])
                )
                loss = whole_info - part_info
                if loss < min_loss:
                    min_loss = loss
                    best_partition = (part_a, part_b)

            # Φ = max(0, min_loss)
            phi = max(0.0, min_loss)
            # 裁剪到合理范围
            phi = min(phi, 5.0)

            self._last_phi = phi
            self._last_partition = best_partition
            self._phi_history.append(phi)
            if len(self._phi_history) > 1000:
                self._phi_history = self._phi_history[-1000:]

            return phi

    # ---------------------------------------------------------------- #
    # 高斯互信息矩阵
    # ---------------------------------------------------------------- #
    def _mutual_information_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """计算 n×n 高斯互信息矩阵。

        MI(X,Y) = 0.5 * log((var(X)*var(Y)) / (var(X)*var(Y) - cov(X,Y)^2))
        """
        n = matrix.shape[1]
        mi = np.zeros((n, n))
        # 计算每列的方差和协方差
        cov = np.cov(matrix, rowvar=False)
        if cov.ndim == 0:  # 单变量情况
            cov = np.array([[float(cov)]])
        variances = np.diag(cov)
        for i in range(n):
            for j in range(n):
                if i == j:
                    mi[i, j] = 0.0
                    continue
                vi = max(variances[i], 1e-10)
                vj = max(variances[j], 1e-10)
                c = cov[i, j]
                denom = vi * vj - c * c
                if denom <= 1e-10:
                    mi[i, j] = 0.0
                else:
                    val = 0.5 * np.log((vi * vj) / denom)
                    mi[i, j] = float(np.clip(val, 0.0, 5.0))
        return mi

    # ---------------------------------------------------------------- #
    # 查询
    # ---------------------------------------------------------------- #
    def get_state(self) -> PhiSelfState:
        """返回当前自我 Φ 状态。"""
        with self._lock:
            return PhiSelfState(
                phi_self=self._last_phi,
                is_self_aware=self._last_phi > self.sleep_threshold,
                self_module_count=self.n_self_modules,
                partition=(
                    (list(self._last_partition[0]), list(self._last_partition[1]))
                    if self._last_partition is not None
                    else None
                ),
                metadata={
                    "step": self._step_count,
                    "history_len": min(len(h) for h in self._histories),
                    "phi_history_len": len(self._phi_history),
                },
            )

    def is_self_aware(self) -> bool:
        """当前是否自我意识活跃。"""
        # P2 修复：原代码无锁读取 _last_phi，与 compute_phi() 的写入
        # 竞争。虽然在 CPython 下 float 读取是原子的，但加锁保证
        # 语义一致性（与 RLock 配合，可重入，无死锁风险）。
        with self._lock:
            return self._last_phi > self.sleep_threshold

    @property
    def last_phi(self) -> float:
        # P2 修复：加锁读取，避免与 compute_phi() 写入竞争。
        with self._lock:
            return self._last_phi

    @property
    def phi_history(self) -> list[float]:
        # P2 修复：加锁读取并返回副本，避免外部修改内部列表。
        with self._lock:
            return list(self._phi_history)

    def reset(self) -> None:
        with self._lock:
            self._histories = [[] for _ in range(self.n_self_modules)]
            self._phi_history.clear()
            self._last_phi = 0.0
            self._last_partition = None
            self._step_count = 0

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "phi_self": self._last_phi,
                "is_self_aware": self._last_phi > self.sleep_threshold,
                "sleep_threshold": self.sleep_threshold,
                "step": self._step_count,
                "history_len": min(len(h) for h in self._histories),
                "recent_phi": (
                    self._phi_history[-20:] if self._phi_history else []
                ),
            }
