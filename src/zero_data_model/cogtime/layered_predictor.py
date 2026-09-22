# src/zero_data_model/cogtime/layered_predictor.py
"""多层预测编码器：多时间尺度预测。

本模块为 ZeroDataModel 提供可选的"时间感知"钩子。通过三个不同时间尺度的
预测层（L0 快速、L1 中速、L2 慢速）构建层级预测编码：

- L0 每步更新，捕捉即时动态；
- L1 每 ``l1_interval`` 步更新，以 L0 的近期平均误差作为观测，捕捉中期结构；
- L2 每 ``l2_interval`` 步更新，以 L1 的近期平均误差作为观测，捕捉长程节律。

每层维护一个 ``belief``（信念状态）、一个 ``transition``（转移矩阵）和一条
预测误差历史。三层信念可拼接为上下文向量供下游模块作为额外输入使用。

设计要点：
- 完全独立、自包含、可插拔；不修改 model.py。
- 三个层各自拥有独立的随机子生成器，互不干扰。
- 所有公共方法均做防御性窄异常捕获（``ValueError``/``FloatingPointError``/
  ``OverflowError``），失败时返回安全默认值（零向量/0.0）。
- ``transition`` 在构造时做谱半径归一化（目标谱半径 ≈ 1/1.2 ≈ 0.833），
  配合 ``BELIEF_DECAY`` 抑制信念发散。
- 通过 ``self._lock``（``RLock``）保护 ``update``/``get_context``/
  ``predict_rhythm`` 的并发访问。
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


@dataclass
class Layer:
    """单个预测层的数据载体。

    Attributes
    ----------
    name:
        层名（如 "L0"）。
    dim:
        该层信念向量维度。
    belief:
        当前信念状态向量。
    transition:
        转移矩阵（单位阵 + 小噪声，并做谱半径归一化），用于一步预测。
    error_history:
        预测误差历史（deque，maxlen=200）。
    rng:
        该层专属的随机数生成器。
    """

    name: str
    dim: int
    belief: np.ndarray
    transition: np.ndarray
    error_history: deque
    rng: np.random.Generator


class LayeredPredictor:
    """多时间尺度层级预测编码器。"""

    # 信念衰减系数与梯度学习率
    BELIEF_DECAY: float = 0.99
    LR: float = 0.05
    # 谱半径归一化安全裕度：目标谱半径 ≈ 1/margin ≈ 0.833，防止发散
    SPECTRAL_MARGIN: float = 1.2

    def __init__(
        self,
        dim_l0: int = 64,
        dim_l1: int = 32,
        dim_l2: int = 16,
        seed: int = 42,
        l1_interval: int = 10,
        l2_interval: int = 100,
    ) -> None:
        self.dim_l0 = dim_l0
        self.dim_l1 = dim_l1
        self.dim_l2 = dim_l2
        self.l1_interval = max(1, l1_interval)
        self.l2_interval = max(1, l2_interval)
        self.seed = seed

        # 用基础 rng 派生 3 个独立子生成器，避免层间随机性耦合
        base_rng = np.random.default_rng(seed)
        child_seeds = base_rng.integers(0, 2**31, size=3)
        rngs = [np.random.default_rng(int(s)) for s in child_seeds]

        self.l0 = self._make_layer("L0", dim_l0, rngs[0])
        self.l1 = self._make_layer("L1", dim_l1, rngs[1])
        self.l2 = self._make_layer("L2", dim_l2, rngs[2])

        # 可重入锁：保护 l0/l1/l2 的 belief/transition/error_history 的并发访问。
        # think() 会并发调用各模块，update/get_context/predict_rhythm 均需
        # 在 self._lock 下访问可变状态。
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def update(self, observation: np.ndarray, step: int) -> dict[str, Any]:
        """更新三层预测编码。

        L0 每步更新；当 ``step`` 为 ``l1_interval`` 倍数时更新 L1；为
        ``l2_interval`` 倍数时更新 L2。

        注意：``step % interval == 0`` 在 ``step=0`` 时为真，这是**有意为之**
        的初始化脉冲——首次更新（step=0）会让 L1/L2 立即基于初始 L0 误差完成
        一次初始化，避免前若干步 L1/L2 完全空白。

        Returns
        -------
        dict
            ``{"l0_error", "l1_error", "l2_error", "l2_belief_norm"}``，
            其中 ``l1_error``/``l2_error`` 在未更新该层时为 None。
            ``l2_belief_norm`` 非有限时钳为 0.0。
        """
        with self._lock:
            try:
                obs_l0 = self._fit(observation, self.dim_l0)
                l0_error = self._update_layer(self.l0, obs_l0)

                l1_error: float | None = None
                # step=0 时 % interval == 0 为真：这是有意的初始化脉冲（见上）
                if step % self.l1_interval == 0:
                    scalar = self._recent_mean(self.l0.error_history)
                    obs_l1 = np.full(self.dim_l1, scalar, dtype=float)
                    l1_error = self._update_layer(self.l1, obs_l1)

                l2_error: float | None = None
                if step % self.l2_interval == 0:
                    scalar = self._recent_mean(self.l1.error_history)
                    obs_l2 = np.full(self.dim_l2, scalar, dtype=float)
                    l2_error = self._update_layer(self.l2, obs_l2)

                l2_belief_norm = float(np.linalg.norm(self.l2.belief))
                # 防御 NaN/inf：非有限时钳为 0.0，避免向下游传播
                if not np.isfinite(l2_belief_norm):
                    l2_belief_norm = 0.0
                return {
                    "l0_error": float(l0_error),
                    "l1_error": float(l1_error) if l1_error is not None else None,
                    "l2_error": float(l2_error) if l2_error is not None else None,
                    "l2_belief_norm": l2_belief_norm,
                }
            except (ValueError, FloatingPointError, OverflowError) as exc:
                _logger.warning("LayeredPredictor.update failed: %r", exc)
                return {
                    "l0_error": 0.0,
                    "l1_error": None,
                    "l2_error": None,
                    "l2_belief_norm": 0.0,
                }

    def get_context(self) -> np.ndarray:
        """返回拼接后的层级上下文向量 ``[L0, L1, L2]``。

        长度为 ``dim_l0 + dim_l1 + dim_l2``，可直接拼接为下游模块的额外输入。
        返回前用 ``np.nan_to_num`` 钳制 NaN/inf，避免坏信念污染下游。
        """
        with self._lock:
            try:
                total = self.dim_l0 + self.dim_l1 + self.dim_l2
                ctx = np.zeros(total, dtype=float)
                ctx[: self.dim_l0] = self.l0.belief
                ctx[self.dim_l0 : self.dim_l0 + self.dim_l1] = self.l1.belief
                ctx[self.dim_l0 + self.dim_l1 :] = self.l2.belief
                # 钳制 NaN/inf：posinf/neginf 也置 0.0（与 NaN 同处理）
                return np.nan_to_num(
                    ctx, nan=0.0, posinf=0.0, neginf=0.0
                )
            except (ValueError, FloatingPointError, OverflowError) as exc:
                _logger.warning("LayeredPredictor.get_context failed: %r", exc)
                return np.zeros(
                    self.dim_l0 + self.dim_l1 + self.dim_l2, dtype=float
                )

    def predict_rhythm(self) -> float:
        """节律置信度启发式。

        若 L2 的误差呈下降趋势，则返回 ``1.0 / (1.0 + L2 近期误差)``（落在
        [0, 1]）；否则返回 0.0（缺乏稳定节律）。返回前钳制 NaN/inf 为 0.0。
        """
        with self._lock:
            try:
                if not self._is_decreasing(self.l2.error_history):
                    return 0.0
                recent = self._recent_mean(self.l2.error_history)
                result = float(1.0 / (1.0 + recent))
                # 防御 NaN/inf：非有限时钳为 0.0（标量等价于 nan_to_num）
                if not np.isfinite(result):
                    return 0.0
                return result
            except (ValueError, FloatingPointError, OverflowError) as exc:
                _logger.warning("LayeredPredictor.predict_rhythm failed: %r", exc)
                return 0.0

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @classmethod
    def _make_layer(
        cls, name: str, dim: int, rng: np.random.Generator
    ) -> Layer:
        """构造一层：belief 置零，transition 为单位阵 + 小噪声并做谱半径归一化。

        谱半径归一化（目标 ≈ 1/SPECTRAL_MARGIN ≈ 0.833）防止 ``transition``
        谱半径 > 1 导致信念发散；若计算出的谱半径为 0/NaN/非有限，则跳过
        归一化（保留原矩阵，即"identity scaling"）。
        """
        belief = np.zeros(dim, dtype=float)
        transition = np.eye(dim, dtype=float) + rng.normal(
            0.0, 0.01, (dim, dim)
        )
        transition = cls._normalize_spectral_radius(transition)
        return Layer(
            name=name,
            dim=dim,
            belief=belief,
            transition=transition,
            error_history=deque(maxlen=200),
            rng=rng,
        )

    @classmethod
    def _normalize_spectral_radius(
        cls, W: np.ndarray
    ) -> np.ndarray:
        """对转移矩阵做谱半径归一化：除以 (最大特征值模 * SPECTRAL_MARGIN)。

        归一化后谱半径约为 ``1 / SPECTRAL_MARGIN``（≈ 0.833），保证循环动力学
        稳定。最大特征值模过小、为 0 或非有限（NaN/inf）时**跳过归一化**，
        返回原矩阵（即 identity scaling，缩放因子 = 1.0）。
        """
        try:
            eigvals = np.linalg.eigvals(W)
            max_abs = float(np.max(np.abs(eigvals)))
            # 谱半径为 0 / NaN / inf 或过小时跳过归一化（identity scaling）
            if max_abs < 1e-12 or not np.isfinite(max_abs):
                return W
            return W / (max_abs * cls.SPECTRAL_MARGIN)
        except (ValueError, FloatingPointError, OverflowError):
            return W

    def _update_layer(self, layer: Layer, observation: np.ndarray) -> float:
        """单层预测编码更新：预测 -> 误差 -> 梯度更新信念 -> 衰减。

        ``pred = belief @ transition``，``error = obs - pred``，
        ``belief += lr * (error @ transition.T)``，最后乘以 ``BELIEF_DECAY``。
        返回该步的平均绝对误差。``transition`` 已在构造时做谱半径归一化，
        配合 ``BELIEF_DECAY`` 抑制发散；若更新后 belief 含 NaN/inf，重置为零
        以防污染后续步骤。
        """
        pred = layer.belief @ layer.transition
        error = observation - pred
        # 简单梯度更新
        new_belief = layer.belief + self.LR * (error @ layer.transition.T)
        new_belief = new_belief * self.BELIEF_DECAY
        # 防御 NaN/inf：若新信念含非有限值，重置为零避免污染
        if not np.isfinite(new_belief).all():
            new_belief = np.zeros_like(new_belief)
        layer.belief = new_belief
        err_mag = float(np.mean(np.abs(error))) if error.size else 0.0
        # 防御 NaN/inf 误差量级
        if not np.isfinite(err_mag):
            err_mag = 0.0
        layer.error_history.append(err_mag)
        return err_mag

    @staticmethod
    def _fit(arr: Any, dim: int) -> np.ndarray:
        """将输入适配到指定维度（过长截断、过短补零）。"""
        a = np.asarray(arr, dtype=float).flatten()
        if a.size < dim:
            a = np.pad(a, (0, dim - a.size))
        elif a.size > dim:
            a = a[:dim]
        return a.astype(float)

    @staticmethod
    def _recent_mean(history: deque) -> float:
        """计算误差历史近期均值；为空时返回 0.0。"""
        if history is None or len(history) == 0:
            return 0.0
        try:
            return float(np.mean(list(history)))
        except (ValueError, FloatingPointError, OverflowError):
            return 0.0

    @staticmethod
    def _is_decreasing(history: deque) -> bool:
        """判断误差是否呈下降趋势（后半均值 < 前半均值）。"""
        h = list(history)
        if len(h) < 2:
            return False
        mid = len(h) // 2
        # len(h) >= 2 ⇒ mid >= 1，故 mid > 0 恒成立，无需 else 分支
        try:
            first_half = float(np.mean(h[:mid]))
            second_half = float(np.mean(h[mid:]))
            return second_half < first_half
        except (ValueError, FloatingPointError, OverflowError):
            return False
