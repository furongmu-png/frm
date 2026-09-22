"""ProprioceptiveEncoder — encodes body-internal state into latent space.

理论基础
========
本体感觉 (proprioception) 是身体对自身姿势、关节角度、肌肉张力的内部感知。
在认知科学中，本体感觉是"身体自我" (bodily self) 的基础：没有本体感觉，
智能体无法区分"自我"与"环境"。

本模块将关节角度、速度等内部状态编码为隐空间向量，作为额外模态输入到
跨模态桥接模块，与视觉、触觉共同形成多模态信念。

本体感觉预测误差用于更新"身体图式" (body schema)——模型对自己身体状态
的内部表征。当预测的身体姿态与实际不一致时，触发身体图式校准。

自由能映射：本体感觉预测误差 = 身体图式自洽性的自由能分量。
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
class ProprioceptiveState:
    """本体感觉编码状态。"""
    encoded: np.ndarray            # 编码后的隐空间向量 (dim,)
    raw: np.ndarray                # 原始本体感觉 (joint_angles, joint_velocities)
    prediction_error: float        # 预测误差（身体图式校准信号）
    body_schema_confidence: float  # 身体图式置信度
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# 主类
# ------------------------------------------------------------------ #
class ProprioceptiveEncoder:
    """本体感觉编码器。

    将关节角度/速度编码为隐空间向量，并维护一个"身体图式"
    预测模型：根据上一时刻状态预测当前状态。

    Parameters
    ----------
    input_dim : int
        原始本体感觉维度（关节角度 + 速度，通常 8）
    output_dim : int
        隐空间维度（与 ZeroDataModel.dim 一致）
    lr : float
        身体图式学习率
    seed : int | None
    """

    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 64,
        lr: float = 0.01,
        seed: Optional[int] = 42,
    ):
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.lr = float(lr)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # 编码矩阵：input_dim → output_dim（固定随机投影，Johnson-Lindenstrauss）
        self._W_encode = self._rng.standard_normal(
            (output_dim, input_dim)
        ) * np.sqrt(1.0 / input_dim)

        # 身体图式：预测下一时刻状态的转移矩阵（input_dim × input_dim）
        # 初始化为单位矩阵（假设身体状态短期不变）
        self._W_schema = np.eye(input_dim) * 0.95  # 略小于 1 → 稳定衰减

        # 上一时刻原始状态（用于预测）
        self._prev_state = np.zeros(input_dim, dtype=np.float64)

        # 累计统计
        self._step_count = 0
        self._avg_prediction_error = 0.0
        self._error_history: list[float] = []
        self._schema_confidence = 1.0

    # ---------------------------------------------------------------- #
    # 编码
    # ---------------------------------------------------------------- #
    def encode(self, raw: np.ndarray) -> np.ndarray:
        """将原始本体感觉编码为隐空间向量。

        Parameters
        ----------
        raw : ndarray, shape (input_dim,)
            原始本体感觉 [joint_angles(4), joint_velocities(4)]

        Returns
        -------
        encoded : ndarray, shape (output_dim,)
        """
        raw = np.asarray(raw, dtype=np.float64).flatten()
        if raw.shape != (self.input_dim,):
            r = np.zeros(self.input_dim)
            n = min(len(raw), self.input_dim)
            r[:n] = raw[:n]
            raw = r

        # NaN 防护
        if not np.all(np.isfinite(raw)):
            raw = np.zeros(self.input_dim)

        with self._lock:
            encoded = self._W_encode @ raw
            # tanh 压缩到 [-1, 1]
            encoded = np.tanh(encoded)
            if not np.all(np.isfinite(encoded)):
                encoded = np.zeros(self.output_dim)
            return encoded

    # ---------------------------------------------------------------- #
    # step：编码 + 预测 + 更新身体图式
    # ---------------------------------------------------------------- #
    def step(self, raw: np.ndarray) -> ProprioceptiveState:
        """编码本体感觉并更新身体图式。

        Parameters
        ----------
        raw : ndarray, shape (input_dim,)
            当前原始本体感觉

        Returns
        -------
        ProprioceptiveState
        """
        raw = np.asarray(raw, dtype=np.float64).flatten()
        if raw.shape != (self.input_dim,):
            r = np.zeros(self.input_dim)
            n = min(len(raw), self.input_dim)
            r[:n] = raw[:n]
            raw = r

        if not np.all(np.isfinite(raw)):
            raw = np.zeros(self.input_dim)

        with self._lock:
            # 编码
            encoded = self.encode(raw)

            # 身体图式预测：根据上一时刻状态预测当前状态
            predicted = self._W_schema @ self._prev_state

            # 预测误差
            error_vec = raw - predicted
            prediction_error = float(np.linalg.norm(error_vec))

            # Hebbian 更新身体图式：ΔW = lr * error * prev^T
            delta_W = self.lr * np.outer(error_vec, self._prev_state)
            # 梯度裁剪
            norm = np.linalg.norm(delta_W)
            if norm > 0.5:
                delta_W = delta_W * (0.5 / norm)
            self._W_schema += delta_W

            # 确保 schema 稳定（谱半径 < 1）
            try:
                eigvals = np.linalg.eigvals(self._W_schema)
                sr = float(np.max(np.abs(eigvals)))
                if sr > 0.99:
                    self._W_schema *= (0.99 / sr)
            except np.linalg.LinAlgError:
                pass

            # 更新状态
            self._prev_state = raw.copy()
            self._step_count += 1
            n = self._step_count
            self._avg_prediction_error = (
                (self._avg_prediction_error * (n - 1) + prediction_error) / n
            )
            self._error_history.append(prediction_error)
            if len(self._error_history) > 1000:
                self._error_history = self._error_history[-1000:]

            # 身体图式置信度：预测误差越低 → 置信度越高
            self._schema_confidence = float(
                np.exp(-prediction_error)
            )

            return ProprioceptiveState(
                encoded=encoded,
                raw=raw.copy(),
                prediction_error=prediction_error,
                body_schema_confidence=self._schema_confidence,
                metadata={
                    "step": self._step_count,
                    "avg_error": self._avg_prediction_error,
                },
            )

    # ---------------------------------------------------------------- #
    # 身体图式查询
    # ---------------------------------------------------------------- #
    def get_body_schema_snapshot(self) -> dict[str, Any]:
        """返回身体图式当前状态（用于可视化"身体自我视图"）。"""
        with self._lock:
            try:
                eigvals = np.linalg.eigvals(self._W_schema)
                sr = float(np.max(np.abs(eigvals)))
            except np.linalg.LinAlgError:
                sr = 1.0
            return {
                "step": self._step_count,
                "avg_prediction_error": self._avg_prediction_error,
                "schema_confidence": self._schema_confidence,
                "schema_spectral_radius": sr,
                "prev_state": self._prev_state.tolist(),
                "recent_error": (
                    float(np.mean(self._error_history[-20:]))
                    if self._error_history else 0.0
                ),
            }

    def get_predicted_next_state(self, current: np.ndarray) -> np.ndarray:
        """用身体图式预测下一时刻的本体感觉。"""
        current = np.asarray(current, dtype=np.float64).flatten()
        if current.shape != (self.input_dim,):
            c = np.zeros(self.input_dim)
            n = min(len(current), self.input_dim)
            c[:n] = current[:n]
            current = c
        with self._lock:
            return self._W_schema @ current

    def reset(self) -> None:
        with self._lock:
            self._prev_state = np.zeros(self.input_dim, dtype=np.float64)
            self._step_count = 0
            self._avg_prediction_error = 0.0
            self._error_history.clear()
            self._schema_confidence = 1.0
            self._W_schema = np.eye(self.input_dim) * 0.95
