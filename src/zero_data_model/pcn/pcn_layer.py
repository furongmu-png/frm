"""Predictive Coding layer — local Hebbian learning via prediction errors.

理论基础
========

预测编码（Predictive Coding, Rao & Ballard 1999）认为大脑皮层是
层级化的预测机器：每层向下一层发送预测，下一层返回预测误差。
Friston 2005 进一步证明这就是自由能原理的神经实现。

每层有两个核心计算：
1. 预测（自顶向下）：用 W_gen 从本层状态生成对下一层的预测
    x_pred_lower = W_gen @ x_self
2. 误差（自底向上）：本层实际状态与上层预测之差
    error = x_self - x_pred_from_higher

更新规则（无全局损失，局部 Hebbian）：
    Δx = lr * (W_rec @ error_lower - error_self)
    ΔW_gen = lr * error * x^T
    ΔW_rec = lr * x * error^T

关键性质：
- 层间只传递 error 和 prediction（无全局 loss）
- 误差随深度递减（高层捕捉更抽象的模式）
- 与 Active Inference 兼容：自由能 = -log p(o) ≈ ||error_lowest||^2 / 2
"""
from __future__ import annotations

import threading

import numpy as np


class PCNLayer:
    """Single Predictive Coding layer.

    Parameters
    ----------
    dim : int
        本层状态维度
    lower_dim : int | None
        下一层（更低层）状态维度；None 表示底层（无下层）
    higher_dim : int | None
        上一层（更高层）状态维度；None 表示顶层（无上层）
    lr : float
        状态和权重的学习率
    seed : int | None
        随机种子
    """

    def __init__(
        self,
        dim: int,
        *,
        lower_dim: int | None = None,
        higher_dim: int | None = None,
        lr: float = 0.01,
        seed: int | None = None,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")

        self.dim = dim
        self.lower_dim = lower_dim
        self.higher_dim = higher_dim
        self.lr = lr

        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # 隐状态
        self._state = np.zeros(dim, dtype=np.float64)
        self._last_valid_state = self._state.copy()

        # W_gen: 生成对下层的预测 (lower_dim, dim)
        # y_lower_pred = W_gen @ x_self
        if lower_dim is not None:
            self._W_gen = (
                self._rng.standard_normal((lower_dim, dim))
                * np.sqrt(2.0 / (dim + lower_dim))
            )
        else:
            self._W_gen = None

        # W_rec: 从下层误差更新本层状态 (dim, lower_dim)
        # Δx_self = lr * W_rec @ error_lower
        if lower_dim is not None:
            self._W_rec = (
                self._rng.standard_normal((dim, lower_dim))
                * np.sqrt(2.0 / (dim + lower_dim))
            )
        else:
            self._W_rec = None

        # 缓存上一次的预测误差（用于诊断和权重更新）
        self._last_error: np.ndarray | None = None
        self._last_prediction: np.ndarray | None = None

    @property
    def state(self) -> np.ndarray:
        with self._lock:
            return self._state.copy()

    @property
    def last_error(self) -> np.ndarray | None:
        with self._lock:
            return self._last_error.copy() if self._last_error is not None else None

    @property
    def last_error_norm(self) -> float:
        with self._lock:
            if self._last_error is None:
                return 0.0
            return float(np.linalg.norm(self._last_error))

    def predict(self) -> np.ndarray | None:
        """生成对下层的预测：x_pred_lower = W_gen @ x_self.

        Returns
        -------
        prediction : ndarray, shape (lower_dim,) | None
            对下层的预测；None 表示无下层
        """
        with self._lock:
            if self._W_gen is None:
                return None
            self._last_prediction = self._W_gen @ self._state
            return self._last_prediction.copy()

    def update(
        self,
        error_from_lower: np.ndarray | None = None,
        prediction_from_higher: np.ndarray | None = None,
    ) -> np.ndarray:
        """基于预测误差更新本层状态和权重。

        状态更新（Rao & Ballard 1999）：
            Δx = lr * (W_rec @ error_lower - error_self)
            其中 error_self = x_self - prediction_from_higher（如果有上层）

        权重更新（Hebbian）：
            ΔW_gen = lr * error_lower * x_self^T   (如果 W_gen 存在)
            ΔW_rec = lr * x_self * error_lower^T   (如果 W_rec 存在)

        Parameters
        ----------
        error_from_lower : ndarray, shape (lower_dim,) | None
            下层传上来的误差（下层 state - 本层对下层的预测）
            None 表示无下层或下层未提供误差
        prediction_from_higher : ndarray, shape (dim,) | None
            上层对本层的预测
            None 表示无上层

        Returns
        -------
        error_self : ndarray, shape (dim,)
            本层的预测误差（传给上层）：x_self - prediction_from_higher
        """
        with self._lock:
            # 计算本层误差（传给上层）
            if prediction_from_higher is not None:
                prediction_from_higher = np.asarray(
                    prediction_from_higher, dtype=np.float64
                )
                if prediction_from_higher.shape != (self.dim,):
                    raise ValueError(
                        f"prediction_from_higher must have shape "
                        f"({self.dim},), got {prediction_from_higher.shape}"
                    )
                if not np.all(np.isfinite(prediction_from_higher)):
                    prediction_from_higher = self._last_valid_state.copy()
                error_self = self._state - prediction_from_higher
            else:
                error_self = np.zeros(self.dim)

            self._last_error = error_self.copy()

            # 状态更新
            delta_x = -self.lr * error_self  # 朝向上层预测
            if error_from_lower is not None and self._W_rec is not None:
                error_from_lower = np.asarray(error_from_lower, dtype=np.float64)
                if error_from_lower.shape != (self.lower_dim,):
                    raise ValueError(
                        f"error_from_lower must have shape "
                        f"({self.lower_dim},), got {error_from_lower.shape}"
                    )
                if np.all(np.isfinite(error_from_lower)):
                    delta_x = delta_x + self.lr * (self._W_rec @ error_from_lower)

            # NaN 防护
            new_state = self._state + delta_x
            if np.all(np.isfinite(new_state)):
                self._last_valid_state = new_state.copy()
                self._state = new_state
            # 否则保持旧 state

            # 权重更新（Hebbian）
            if (
                error_from_lower is not None
                and self._W_gen is not None
                and np.all(np.isfinite(error_from_lower))
            ):
                delta_W_gen = self.lr * np.outer(error_from_lower, self._state)
                # 梯度裁剪
                norm = np.linalg.norm(delta_W_gen)
                if norm > 1.0:
                    delta_W_gen *= 1.0 / norm
                self._W_gen += delta_W_gen

                delta_W_rec = self.lr * np.outer(self._state, error_from_lower)
                norm = np.linalg.norm(delta_W_rec)
                if norm > 1.0:
                    delta_W_rec *= 1.0 / norm
                self._W_rec += delta_W_rec

            return error_self.copy()

    def set_state(self, new_state: np.ndarray) -> None:
        """直接设置状态（用于底层接收观测）。"""
        new_state = np.asarray(new_state, dtype=np.float64)
        if new_state.shape != (self.dim,):
            raise ValueError(
                f"new_state must have shape ({self.dim},), got {new_state.shape}"
            )
        with self._lock:
            if np.all(np.isfinite(new_state)):
                self._state = new_state.copy()
                self._last_valid_state = new_state.copy()

    def reset(self) -> None:
        with self._lock:
            self._state = np.zeros(self.dim, dtype=np.float64)
            self._last_valid_state = self._state.copy()
            self._last_error = None
            self._last_prediction = None

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                "state_norm": float(np.linalg.norm(self._state)),
                "state_first_5": self._state[: min(5, self.dim)].tolist(),
                "error_norm": self.last_error_norm,
                "has_lower": self.lower_dim is not None,
                "has_higher": self.higher_dim is not None,
            }

    def __getstate__(self) -> dict:
        with self._lock:
            return {
                "dim": self.dim,
                "lower_dim": self.lower_dim,
                "higher_dim": self.higher_dim,
                "lr": self.lr,
                "_state": self._state.copy(),
                "_last_valid_state": self._last_valid_state.copy(),
                "_W_gen": None if self._W_gen is None else self._W_gen.copy(),
                "_W_rec": None if self._W_rec is None else self._W_rec.copy(),
                "_last_error": (
                    None if self._last_error is None else self._last_error.copy()
                ),
                "_last_prediction": (
                    None
                    if self._last_prediction is None
                    else self._last_prediction.copy()
                ),
            }

    def __setstate__(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)
        self._lock = threading.RLock()
