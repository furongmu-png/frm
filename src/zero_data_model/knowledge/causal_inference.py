# src/zero_data_model/knowledge/causal_inference.py
"""因果推断层（CausalInference）。

在给定（线性）转移矩阵 ``state @ transition`` 之上提供简化的 do-算子
干预、反事实估计与混杂因子识别。本实现为可插拔的启发式版本，仅依赖
``numpy``，独立、自包含；不修改 ``model.py``。

注意：``do_calculus`` 接口未显式传入“当前状态”，因此默认采用全 1 向量
作为基线状态进行前向传播；如需自定义基线，可在子类中覆盖或扩展。

API 契约：所有数组入参在进入公共方法时均经 ``np.asarray(...).ravel()``
规整为一维；输出（如 ``counterfactual`` 返回值、``do_calculus`` 中的
``predicted_state``）始终为一维 ``shape=(n,)`` 数组。
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np


class CausalInference:
    """基于转移矩阵的简化因果推断。

    Attributes
    ----------
    transition_matrix:
        线性转移矩阵 ``T``，满足 ``next_state = state @ T``；可为 ``None``，
        随后通过 :meth:`set_transition` 设置。
    """

    def __init__(self, transition_matrix: np.ndarray | None = None) -> None:
        """初始化因果推断层。

        Parameters
        ----------
        transition_matrix:
            转移矩阵（可为 ``None``，稍后设置）。
        """
        # 可重入锁：保护 transition_matrix 的读写。
        self._lock = threading.RLock()
        self.transition_matrix: np.ndarray | None = transition_matrix

    def set_transition(self, matrix: np.ndarray) -> None:
        """设置/替换转移矩阵。

        校验：矩阵必须为二维方阵，否则抛 :class:`ValueError`。
        """
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"transition_matrix 必须为二维方阵，收到 shape {arr.shape}"
            )
        with self._lock:
            self.transition_matrix = arr

    def do_calculus(self, intervention: dict[int, float]) -> dict[str, Any]:
        """do-算子：在强制干预下预测下一状态。

        以全 1 向量为基线状态，先计算自然动力学下的基线前向
        ``baseline = state @ T``；再在输入状态上将被干预变量替换为指定值，
        前向得到 ``predicted = (intervened_state) @ T``。``changed_vars``
        为 ``predicted`` 与 ``baseline`` 差异超过 ``1e-4`` 的变量索引。

        Parameters
        ----------
        intervention:
            ``{var_index: fixed_value}`` 形式的干预字典。

        Returns
        -------
        dict
            ``{"predicted_state": np.ndarray, "intervened_vars": list[int],
            "changed_vars": list[int]}``。``predicted_state`` 始终为一维。
        """
        with self._lock:
            if self.transition_matrix is None:
                # 未设置转移矩阵，返回空预测
                return {
                    "predicted_state": np.array([], dtype=float),
                    "intervened_vars": list(intervention.keys()),
                    "changed_vars": [],
                }
            tm = np.asarray(self.transition_matrix, dtype=float)
            if tm.ndim != 2 or tm.shape[0] == 0:
                return {
                    "predicted_state": np.array([], dtype=float),
                    "intervened_vars": list(intervention.keys()),
                    "changed_vars": [],
                }
            n = tm.shape[0]
            # 接口未传入当前状态，默认使用全 1 向量作为基线状态
            state = np.ones(n, dtype=float)
            # 自然动力学基线
            baseline = state @ tm
            # do-算子：在输入状态上强制干预变量取指定值
            modified_state = state.copy()
            for var, val in intervention.items():
                if 0 <= var < n:
                    modified_state[var] = float(val)
            predicted = modified_state @ tm
            # 与基线差异超过 1e-4 的变量视为“被改变”
            changed = [
                int(i)
                for i in range(predicted.shape[0])
                if abs(float(predicted[i]) - float(baseline[i])) > 1e-4
            ]
            return {
                # predicted 已为一维 (n,)，显式 ravel 以契约化输出形状。
                "predicted_state": np.asarray(predicted, dtype=float).ravel(),
                "intervened_vars": list(intervention.keys()),
                "changed_vars": changed,
            }

    def counterfactual(
        self, observed: np.ndarray, intervention: dict[int, float]
    ) -> np.ndarray:
        """反事实估计（简化版）。

        完整的三步法（abduction/action/prediction）在此简化为：直接在
        观测状态上施加干预（action 步），不做噪声推断与前向传播。

        Parameters
        ----------
        observed:
            实际观测到的状态向量（多维输入会被 ravel 为一维）。
        intervention:
            ``{var_index: fixed_value}`` 形式的干预字典。

        Returns
        -------
        np.ndarray
            施加干预后的反事实状态向量，始终为一维 ``shape=(n,)``。
        """
        with self._lock:
            if self.transition_matrix is None:
                raise TypeError(
                    "counterfactual requires non-None transition_matrix, "
                    "observed_state, intervention"
                )
            if observed is None or intervention is None:
                raise TypeError(
                    "counterfactual requires non-None transition_matrix, "
                    "observed_state, intervention"
                )
            # 规整为一维，避免 2-D (n,1)/(1,n) 输入导致索引返回数组而非标量。
            result = np.asarray(observed, dtype=float).ravel().copy()
            n = result.shape[0]
            for var, val in intervention.items():
                if 0 <= var < n:
                    result[var] = float(val)
            return result

    def identify_confounders(self, var_a: int, var_b: int) -> list[int]:
        """启发式识别两个目标变量的潜在混杂因子。

        在线性模型 ``next_state[j] = sum_k state[k] * T[k, j]`` 下，
        ``T[k, j]`` 表示变量 ``k`` 对变量 ``j`` 的影响。若某变量 ``k``
        对 ``var_a`` 与 ``var_b`` 的影响均非零，则视为潜在混杂因子。

        Parameters
        ----------
        var_a, var_b:
            两个目标变量的列索引。

        Returns
        -------
        list[int]
            潜在混杂因子的行索引列表（不含 ``var_a``/``var_b`` 自身）。
        """
        with self._lock:
            if self.transition_matrix is None:
                return []
            tm = np.asarray(self.transition_matrix, dtype=float)
            if tm.ndim != 2:
                return []
            n_rows, n_cols = tm.shape
            # 索引越界则无法判断
            if not (0 <= var_a < n_cols and 0 <= var_b < n_cols):
                return []
            confounders: list[int] = []
            for k in range(n_rows):
                if k in (var_a, var_b):
                    continue
                influence_a = abs(float(tm[k, var_a]))
                influence_b = abs(float(tm[k, var_b]))
                if influence_a > 0.0 and influence_b > 0.0:
                    confounders.append(int(k))
            return confounders


__all__: list[str] = ["CausalInference"]
