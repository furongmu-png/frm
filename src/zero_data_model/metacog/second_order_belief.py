# src/zero_data_model/metacog/second_order_belief.py
"""二阶信念系统（SecondOrderBelief）。

第二阶段 §2.1：构建关于自身隐状态不确定性的概率分布，使模型
"知道自己不知道什么"。

与现有 ``MetaCognition`` 的区别：
- **显式高斯建模**：均值 = 信念向量，方差 = 不确定度向量。
- **预测不确定度输出**：``predicted_uncertainty`` 标量，可在
  ``think()`` 中作为预测的一部分输出。
- **Hopfield 相似度集成**：检索不到相似记忆 → 不确定度上升。

不确定度来源（三项加权）：
1. 近期预测误差的指数移动平均（EMA）
2. 模型参数更新的幅度（更新大 → 不确定）
3. Hopfield 记忆检索的相似度（检索不到 → 不确定）

仅依赖 ``numpy``，独立、自包含、可插拔。
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np


class SecondOrderBelief:
    """二阶信念：对自身隐状态不确定性的高斯建模。

    维护概率分布 ``N(belief_mean, diag(uncertainty_var))``，
    其中 ``belief_mean`` 是当前信念向量，``uncertainty_var`` 是逐维
    不确定度（方差）。

    Parameters
    ----------
    dim:
        隐状态维度。
    ema_alpha:
        EMA 平滑系数（越小越平滑）。
    memory_weight:
        Hopfield 相似度项的权重。
    param_weight:
        参数更新幅度项的权重。
    error_weight:
        预测误差项的权重。
    seed:
        随机种子。

    Attributes
    ----------
    belief_mean:
        当前信念均值向量（shape ``(dim,)``）。
    uncertainty_var:
        当前不确定度方差向量（shape ``(dim,)``）。
    """

    def __init__(
        self,
        dim: int = 64,
        ema_alpha: float = 0.1,
        memory_weight: float = 0.2,
        param_weight: float = 0.3,
        error_weight: float = 0.5,
        seed: int = 42,
    ) -> None:
        """初始化二阶信念系统。"""
        if dim <= 0:
            raise ValueError(f"dim 必须为正整数，收到 {dim}")
        self._dim = int(dim)
        self._ema_alpha = max(0.001, min(1.0, float(ema_alpha)))
        self._memory_w = float(memory_weight)
        self._param_w = float(param_weight)
        self._error_w = float(error_weight)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        # 高斯参数：均值 = 信念，方差 = 不确定度。
        self.belief_mean: np.ndarray = np.zeros(self._dim)
        self.uncertainty_var: np.ndarray = np.ones(self._dim) * 0.5
        # 预测不确定度（标量，下一步的预期不确定度）。
        self._predicted_uncertainty: float = 0.5
        # 置信度历史（供可视化绘制曲线）。
        self._confidence_history: deque[float] = deque(maxlen=500)
        # EMA 预测误差。
        self._error_ema: float = 0.5
        self._step_count: int = 0
        # 最近一次不确定度来源分解。
        self._last_decomposition: dict[str, float] = {
            "error_term": 0.0,
            "param_term": 0.0,
            "memory_term": 0.0,
        }

    # ------------------------------------------------------------------ #
    # 核心更新
    # ------------------------------------------------------------------ #
    def update(
        self,
        belief_state: np.ndarray,
        prediction_error: float,
        param_update_norm: float = 0.0,
        memory_similarity: float = 1.0,
    ) -> dict[str, Any]:
        """每步更新二阶信念。

        Parameters
        ----------
        belief_state:
            当前信念向量（shape ``(dim,)``）。若维度不匹配会自动截断/padding。
        prediction_error:
            本步预测误差（标量，来自自由能历史）。
        param_update_norm:
            本步模型参数更新的 Frobenius 范数（越大越不稳定）。
        memory_similarity:
            Hopfield 记忆检索的最相似度（0=无匹配 → 不确定，1=完美匹配）。

        Returns
        -------
        dict
            ``{"predicted_uncertainty": float, "confidence": float,
            "mean_uncertainty": float, "decomposition": dict}``。
        """
        with self._lock:
            # 更新信念均值。
            bs = np.asarray(belief_state, dtype=np.float64).flatten()
            if bs.shape[0] >= self._dim:
                self.belief_mean = bs[: self._dim].copy()
            else:
                padded = np.zeros(self._dim)
                padded[: bs.shape[0]] = bs
                self.belief_mean = padded

            # NaN 防护。
            err = float(prediction_error) if np.isfinite(prediction_error) else 1e6
            mag = float(param_update_norm) if np.isfinite(param_update_norm) else 0.0
            sim = float(memory_similarity) if np.isfinite(memory_similarity) else 0.0
            sim = max(0.0, min(1.0, sim))

            # EMA 预测误差。
            self._error_ema = (
                self._error_ema * (1.0 - self._ema_alpha)
                + err * self._ema_alpha
            )

            # 三项不确定度来源：
            # 1. 误差项：EMA 预测误差（归一化到 [0,1)）。
            error_term = self._error_ema / (self._error_ema + 1.0)
            # 2. 参数项：更新幅度归一化。
            param_term = mag / (mag + 1.0)
            # 3. 记忆项：1 - 相似度（无匹配 → 高不确定）。
            memory_term = 1.0 - sim

            # 加权组合。
            total_w = self._error_w + self._param_w + self._memory_w
            uncertainty = (
                error_term * self._error_w
                + param_term * self._param_w
                + memory_term * self._memory_w
            ) / max(total_w, 1e-12)
            uncertainty = max(0.0, min(1.0, uncertainty))

            # EMA 平滑逐维不确定度方差。
            self.uncertainty_var = (
                self.uncertainty_var * (1.0 - self._ema_alpha)
                + np.ones(self._dim) * uncertainty * self._ema_alpha
            )

            # 预测不确定度：当前不确定度的 EMA（预测下一步）。
            self._predicted_uncertainty = float(
                self._predicted_uncertainty * (1.0 - self._ema_alpha)
                + uncertainty * self._ema_alpha
            )
            self._predicted_uncertainty = max(
                0.0, min(1.0, self._predicted_uncertainty)
            )

            self._last_decomposition = {
                "error_term": float(error_term),
                "param_term": float(param_term),
                "memory_term": float(memory_term),
            }

            confidence = self.get_confidence()
            self._confidence_history.append(confidence)
            self._step_count += 1

            return {
                "predicted_uncertainty": self._predicted_uncertainty,
                "confidence": float(confidence),
                "mean_uncertainty": float(self.uncertainty_var.mean()),
                "decomposition": dict(self._last_decomposition),
            }

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #
    @property
    def predicted_uncertainty(self) -> float:
        """预测不确定度（标量，``[0, 1]``）。"""
        with self._lock:
            return self._predicted_uncertainty

    def get_confidence(self) -> float:
        """当前置信度（``1 - 平均不确定度``），范围 ``[0, 1]``。"""
        with self._lock:
            return float(
                max(0.0, min(1.0, 1.0 - self.uncertainty_var.mean()))
            )

    def get_uncertainty_vector(self) -> np.ndarray:
        """返回逐维不确定度方差向量的副本。"""
        with self._lock:
            return self.uncertainty_var.copy()

    def get_belief_mean(self) -> np.ndarray:
        """返回信念均值向量的副本。"""
        with self._lock:
            return self.belief_mean.copy()

    def get_confidence_history(self) -> list[float]:
        """返回置信度历史（供可视化）。"""
        with self._lock:
            return list(self._confidence_history)

    def get_decomposition(self) -> dict[str, float]:
        """返回最近一次不确定度来源分解。"""
        with self._lock:
            return dict(self._last_decomposition)

    # ------------------------------------------------------------------ #
    # 高斯采样
    # ------------------------------------------------------------------ #
    def sample_belief(self, n_samples: int = 1) -> np.ndarray:
        """从二阶信念高斯分布中采样。

        Parameters
        ----------
        n_samples:
            采样数量。

        Returns
        -------
        np.ndarray
            采样结果，shape ``(n_samples, dim)``。
        """
        with self._lock:
            std = np.sqrt(self.uncertainty_var)
            result: np.ndarray = (
                self.belief_mean[np.newaxis, :]
                + std[np.newaxis, :] * self._rng.standard_normal(
                    (n_samples, self._dim)
                )
            )
            return result

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def confidence(self) -> float:
        """当前置信度。"""
        return self.get_confidence()

    @property
    def stats(self) -> dict[str, Any]:
        """统计信息。"""
        with self._lock:
            return {
                "dim": self._dim,
                "step_count": self._step_count,
                "predicted_uncertainty": self._predicted_uncertainty,
                "confidence": self.get_confidence(),
                "mean_uncertainty": float(self.uncertainty_var.mean()),
                "belief_norm": float(np.linalg.norm(self.belief_mean)),
                "decomposition": dict(self._last_decomposition),
            }


__all__: list[str] = ["SecondOrderBelief"]
