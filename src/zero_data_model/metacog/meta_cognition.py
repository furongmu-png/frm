# src/zero_data_model/metacog/meta_cognition.py
"""递归自我建模与元认知（第五阶段，任务 5.1/5.2）。

本模块维护模型对自身隐状态不确定性的估计（二阶信念），并据此
驱动探索/利用/安全三种认知模式的切换。不确定性来源：

1. 近期预测误差的移动标准差（感知不确定性）
2. 模型参数的更新速率（参数不确定性，更新大表示不收敛）
3. 当前情景与记忆中最相似情景的误差（情景不确定性）

当不确定度超过阈值时，``should_seek_info()`` 返回 True，驱动
下一阶段的主动信息寻求或切换至安全模式。
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np


# ------------------------------------------------------------------ #
# 元认知模块
# ------------------------------------------------------------------ #
class MetaCognition:
    """二阶信念：对自身不确定性的估计与元认知触发。

    Parameters
    ----------
    dim : int
        隐状态维度，用于维护逐维不确定性向量。
    window : int
        移动统计窗口（近期多少步用于计算移动标准差）。
    uncertainty_threshold : float
        触发探索模式的不确定度阈值（>threshold→explore）。
    seed : int
        随机种子（保持可复现），用于初始化本地 RNG ``self._rng``。
    """

    def __init__(
        self,
        dim: int = 64,
        window: int = 50,
        uncertainty_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        self._dim = int(dim)
        self._window = int(window)
        self._uncertainty_threshold = float(uncertainty_threshold)
        # 用种子初始化本地 RNG（取代仅存储 _seed）；当前模块暂无随机调用，
        # 但保留 RNG 以便后续扩展时保持可复现。
        self._rng = np.random.default_rng(seed)

        # 可重入锁：保护 deques / ndarrays 的读写。
        self._lock = threading.RLock()

        # 近期预测误差与参数更新范数的历史（移动窗口）。
        self._error_history: deque[float] = deque(maxlen=self._window)
        self._update_magnitudes: deque[float] = deque(maxlen=self._window)
        # 置信度历史（更长，供故事模式绘制曲线）。
        self._confidence_history: deque[float] = deque(maxlen=200)

        # 逐维不确定性估计，初值 0.5（完全不确定）。
        self._uncertainty_estimate: np.ndarray = np.ones(self._dim) * 0.5
        self._mode: str = "explore"  # explore / exploit / safe / balanced
        self._step_count: int = 0
        # EMA 平滑系数。
        self._ema_alpha: float = 0.1

    # ------------------------------------------------------------------ #
    # 核心更新
    # ------------------------------------------------------------------ #
    def update(
        self,
        prediction_error: float,
        param_update_norm: float,
        similar_episode_error: float = 0.0,
    ) -> dict[str, Any]:
        """每步更新元认知状态。

        Parameters
        ----------
        prediction_error : float
            本步预测误差（来自 active_inference.free_energy_history[-1]）。
        param_update_norm : float
            本步模型参数更新的 Frobenius 范数（越大表示越不稳定）。
        similar_episode_error : float
            与记忆中最相似情景的误差（来自 episodic_graph 查询）。

        Returns
        -------
        dict
            ``{"mean_uncertainty": float, "mode": str, "confidence": float}``。
            注意：``mean_uncertainty`` 键名为历史兼容保留（前端
            ConfidenceDashboard.tsx 依赖该键），其实际存储的是一个加权
            复合不确定度（含移动标准差、参数项、情景项），并非纯粹的均值。
        """
        with self._lock:
            # 记录历史。
            err = float(prediction_error) if np.isfinite(prediction_error) else 1e6
            mag = float(param_update_norm) if np.isfinite(param_update_norm) else 0.0
            sim_err = (
                float(similar_episode_error)
                if np.isfinite(similar_episode_error)
                else 0.0
            )
            self._error_history.append(err)
            self._update_magnitudes.append(mag)
            self._step_count += 1

            # 计算不确定性：三项加权。
            moving_std = self._safe_std(list(self._error_history))
            # 防御除零：mag == -1.0 会使 (mag + 1.0) == 0。
            divisor = mag + 1.0
            param_term = 0.0 if abs(divisor) < 1e-12 else mag / divisor  # 归一化到 [0,1)
            uncertainty = (
                moving_std * 0.5
                + param_term * 0.3
                + sim_err * 0.2
            )
            # EMA 平滑不确定性估计。
            self._uncertainty_estimate = (
                self._uncertainty_estimate * (1.0 - self._ema_alpha)
                + np.ones(self._dim) * uncertainty * self._ema_alpha
            )
            # 决定模式。
            self._mode = self._decide_mode()
            confidence = self.get_confidence()
            self._confidence_history.append(confidence)
            return {
                # 键名保留以兼容前端；值实为加权复合不确定度（含 std 项）。
                "mean_uncertainty": float(uncertainty),
                "mode": self._mode,
                "confidence": float(confidence),
            }

    def _decide_mode(self) -> str:
        """根据当前不确定度决定认知模式。"""
        mean_unc = float(self._uncertainty_estimate.mean())
        if mean_unc > self._uncertainty_threshold * 1.5:
            return "safe"      # 高度不确定 → 安全模式（减少探索）
        if mean_unc > self._uncertainty_threshold:
            return "explore"   # 不确定 → 主动信息寻求
        if mean_unc < self._uncertainty_threshold * 0.5:
            return "exploit"   # 低不确定 → 利用已有知识
        return "balanced"

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #
    def should_seek_info(self) -> bool:
        """是否应触发主动信息寻求。"""
        with self._lock:
            return self._mode == "explore"

    def get_confidence(self) -> float:
        """当前置信度（1 - 平均不确定性），范围 [0,1]。"""
        with self._lock:
            return float(
                max(0.0, min(1.0, 1.0 - self._uncertainty_estimate.mean()))
            )

    def get_uncertainty_vector(self) -> np.ndarray:
        """返回逐维不确定性估计的副本。"""
        with self._lock:
            return self._uncertainty_estimate.copy()

    def record_milestone(self, event_type: str, step: int) -> dict[str, Any]:
        """生成一个元认知里程碑事件（供故事模式）。"""
        with self._lock:
            return {
                "type": "meta_cognition",
                "event": event_type,
                "step": int(step),
                "uncertainty": float(self._uncertainty_estimate.mean()),
                "mode": self._mode,
                "confidence": self.get_confidence(),
            }

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def confidence(self) -> float:
        return self.get_confidence()

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            errs = list(self._error_history)
            mags = list(self._update_magnitudes)
            return {
                "mean_error": float(np.mean(errs)) if errs else 0.0,
                "mean_update_norm": float(np.mean(mags)) if mags else 0.0,
                "n_steps": self._step_count,
                "mode": self._mode,
                "confidence": self.get_confidence(),
            }

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_std(values: list[float]) -> float:
        """安全的总体标准差（至少 2 个点，``ddof=0``）。

        采用总体标准差（``np.std`` 默认 ``ddof=0``）而非样本标准差，
        因为本模块将窗口内全部误差视作总体而非抽样。
        """
        if len(values) < 2:
            return 0.0
        arr = np.asarray(values, dtype=float)
        # ddof=0 → 总体标准差，与文档一致。
        return float(np.std(arr))
