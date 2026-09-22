"""GWT Φ Calculator — 简化版整合信息论意识指标。

将各模块隐状态视为因果系统，计算互信息矩阵，搜索最小信息划分。
Φ 近似为划分前后的互信息损失。使用贪心搜索（仅孤立单个模块）。
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PhiCalculator:
    """简化版整合信息 Φ 计算。

    Parameters
    ----------
    n_modules : int
        参与系统的模块数量。
    history_window : int, default 50
        保留的近期状态序列长度（用于估计互信息）。
    sleep_threshold : float, default 0.1
        Φ 低于此值视为"低意识"（睡眠/离线巩固）状态。
    """

    def __init__(
        self,
        n_modules: int,
        history_window: int = 50,
        sleep_threshold: float = 0.1,
    ) -> None:
        if n_modules <= 0:
            raise ValueError(f"n_modules must be positive, got {n_modules}")
        if history_window <= 1:
            raise ValueError(f"history_window must be > 1, got {history_window}")

        self.n_modules = int(n_modules)
        self.history_window = int(history_window)
        self.sleep_threshold = float(sleep_threshold)

        #: 每个模块的状态历史：module_idx → list[np.ndarray]
        self._module_histories: list[list[np.ndarray]] = [[] for _ in range(n_modules)]
        self._phi_history: list[float] = []
        self._max_history = 500
        self._last_phi: float = 0.0
        self._last_partition: tuple[set[int], set[int]] | None = None

    # ------------------------------------------------------------------ #
    # 状态采集
    # ------------------------------------------------------------------ #
    def record(self, module_states: list[np.ndarray]) -> None:
        """记录一步的各模块状态。

        Parameters
        ----------
        module_states : list[np.ndarray]
            长度需等于 n_modules，每个元素是该模块的隐状态向量。
        """
        if len(module_states) != self.n_modules:
            # 适配：填充或截断
            if len(module_states) < self.n_modules:
                module_states = list(module_states) + [
                    np.zeros(1) for _ in range(self.n_modules - len(module_states))
                ]
            else:
                module_states = module_states[: self.n_modules]

        for i, state in enumerate(module_states):
            vec = np.asarray(state, dtype=np.float64).flatten()
            self._module_histories[i].append(vec)
            if len(self._module_histories[i]) > self.history_window:
                self._module_histories[i] = self._module_histories[i][-self.history_window :]

    # ------------------------------------------------------------------ #
    # Φ 计算
    # ------------------------------------------------------------------ #
    def compute_phi(self) -> float:
        """计算当前 Φ 值。

        步骤:
          1. 计算系统整体的互信息（所有模块联合）
          2. 贪心搜索：尝试孤立每个单独模块，计算划分前后的信息损失
          3. Φ = min over partitions of (整体互信息 - 划分后互信息)

        Returns
        -------
        phi : float
            整合信息量。0 表示模块独立无整合，越高意识越强。
        """
        # 需要足够历史
        min_len = min(len(h) for h in self._module_histories)
        if min_len < 2:
            self._last_phi = 0.0
            self._phi_history.append(0.0)
            if len(self._phi_history) > self._max_history:
                self._phi_history = self._phi_history[-self._max_history :]
            return 0.0

        try:
            # 系统整体互信息：各模块状态拼接后的自相关信息量
            whole_mi = self._system_information()

            # 贪心搜索：孤立每个单独模块
            min_loss = float("inf")
            best_partition: tuple[set[int], set[int]] | None = None
            for isolated in range(self.n_modules):
                rest = set(range(self.n_modules)) - {isolated}
                if not rest:
                    continue
                partitioned_mi = self._partitioned_information({isolated}, rest)
                loss = whole_mi - partitioned_mi
                if loss < min_loss:
                    min_loss = loss
                    best_partition = ({isolated}, rest)

            phi = max(0.0, min_loss) if min_loss != float("inf") else 0.0
            self._last_phi = float(phi)
            self._last_partition = best_partition
        except (ValueError, FloatingPointError):
            self._last_phi = 0.0

        self._phi_history.append(self._last_phi)
        if len(self._phi_history) > self._max_history:
            self._phi_history = self._phi_history[-self._max_history :]
        return self._last_phi

    def _system_information(self) -> float:
        """系统整体的"信息量"：用各模块状态序列的总互信息近似。

        互信息矩阵的迹（各模块自信息）+ 非对角互信息之和。
        """
        mi_matrix = self._mutual_information_matrix()
        # 整体信息 = 迹 + 非对角元素（整合部分）
        return float(np.trace(mi_matrix) + np.sum(mi_matrix - np.diag(np.diag(mi_matrix))))

    def _partitioned_information(self, part_a: set[int], part_b: set[int]) -> float:
        """划分后的信息量：两部分各自信息之和（不含跨部分互信息）。"""
        mi_matrix = self._mutual_information_matrix()
        info_a = sum(mi_matrix[i, i] for i in part_a)
        info_b = sum(mi_matrix[j, j] for j in part_b)
        return float(info_a + info_b)

    def _mutual_information_matrix(self) -> np.ndarray:
        """计算 n_modules × n_modules 的互信息矩阵。

        用高斯互信息近似：MI(X,Y) = 0.5 * log((var(X)*var(Y)) / (var(X)*var(Y) - cov(X,Y)^2))
        """
        # 取每个模块状态的范数序列（1D 标量近似，降维避免高维 MI 估计）
        # 截断到最短长度，确保各序列等长（防止 xi * xj 形状不匹配）
        min_len = min((len(h) for h in self._module_histories), default=0)
        if min_len < 2:
            # 不足以估计互信息
            return np.zeros((self.n_modules, self.n_modules))

        series = []
        for i in range(self.n_modules):
            hist = self._module_histories[i]
            if not hist:
                series.append(np.zeros(min_len))
            else:
                norms = np.array([np.linalg.norm(s) for s in hist[:min_len]])
                series.append(norms)

        n = self.n_modules
        mi = np.zeros((n, n))
        eps = 1e-8
        for i in range(n):
            for j in range(i, n):
                xi = series[i] - np.mean(series[i])
                xj = series[j] - np.mean(series[j])
                vi = np.var(series[i]) + eps
                vj = np.var(series[j]) + eps
                cov = np.mean(xi * xj)
                # 高斯互信息
                det = vi * vj - cov * cov
                if det <= eps:
                    mi[i, j] = 0.0
                else:
                    mi[i, j] = 0.5 * np.log((vi * vj) / det)
                mi[j, i] = mi[i, j]
        # 数值稳定：裁剪到 [0, 5]
        return np.clip(mi, 0.0, 5.0)

    # ------------------------------------------------------------------ #
    # 意识状态判断
    # ------------------------------------------------------------------ #
    def is_conscious(self) -> bool:
        """Φ 是否高于睡眠阈值（清醒状态）。"""
        return self._last_phi > self.sleep_threshold

    @property
    def last_phi(self) -> float:
        return self._last_phi

    @property
    def phi_history(self) -> list[float]:
        return list(self._phi_history)

    @property
    def last_partition(self) -> tuple[set[int], set[int]] | None:
        return self._last_partition

    def snapshot(self) -> dict:
        return {
            "n_modules": self.n_modules,
            "phi": round(self._last_phi, 6),
            "is_conscious": self.is_conscious(),
            "sleep_threshold": self.sleep_threshold,
            "phi_history": [round(p, 6) for p in self._phi_history[-20:]],
            "min_history_len": min(
                (len(h) for h in self._module_histories), default=0
            ),
            "last_partition": (
                [sorted(list(self._last_partition[0])), sorted(list(self._last_partition[1]))]
                if self._last_partition
                else None
            ),
        }

    def configure(self, **kwargs: Any) -> None:
        if "sleep_threshold" in kwargs:
            self.sleep_threshold = float(kwargs["sleep_threshold"])
        if "history_window" in kwargs:
            v = int(kwargs["history_window"])
            if v > 1:
                self.history_window = v
