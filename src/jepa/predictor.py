"""JEPA Online Predictor — predicts the target encoder's output from latent state.

从当前观测的在线隐状态（S4 / PCN L0）预测目标编码器的输出。
使用 2 层 MLP（固定随机初始化 + 误差驱动微调），训练信号为预测误差。
"""

from __future__ import annotations

import numpy as np


class OnlinePredictor:
    """在线预测器：从在线隐状态预测目标隐表示。

    结构: latent_dim → hidden → target_dim 的两层 MLP，tanh 非线性。
    权重固定随机初始化，通过预测误差驱动 Hebbian 微调（无反向传播）。

    Parameters
    ----------
    latent_dim : int
        在线隐状态维度（来自 S4 或 PCN L0）。
    target_dim : int
        目标编码器输出维度（与 TargetEncoder.latent_dim 一致）。
    hidden_dim : int, default 128
        隐层维度。
    lr : float, default 0.001
        误差驱动更新的学习率。
    seed : int, default 42
    """

    def __init__(
        self,
        latent_dim: int,
        target_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        if latent_dim <= 0 or target_dim <= 0:
            raise ValueError("latent_dim and target_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")

        self.latent_dim = int(latent_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)

        rng = np.random.default_rng(seed)
        # Xavier 初始化
        s1 = np.sqrt(2.0 / (latent_dim + hidden_dim))
        s2 = np.sqrt(2.0 / (hidden_dim + target_dim))
        self._W1 = rng.standard_normal((hidden_dim, latent_dim)) * s1
        self._b1 = np.zeros(hidden_dim)
        self._W2 = rng.standard_normal((target_dim, hidden_dim)) * s2
        self._b2 = np.zeros(target_dim)

        self._last_hidden: np.ndarray | None = None
        self._last_prediction: np.ndarray | None = None
        self._last_error_norm: float = 0.0
        self._update_count: int = 0

    # ------------------------------------------------------------------ #
    # 前向预测
    # ------------------------------------------------------------------ #
    def predict(self, latent_state: np.ndarray) -> np.ndarray:
        """从在线隐状态预测目标隐表示。

        latent_state → W1 → tanh → W2 → L2 normalize → 预测
        """
        x = np.asarray(latent_state, dtype=np.float64).flatten()
        if x.size != self.latent_dim:
            if x.size < self.latent_dim:
                x = np.pad(x, (0, self.latent_dim - x.size))
            else:
                x = x[: self.latent_dim]

        h = self._W1 @ x + self._b1
        h = np.tanh(h)
        y = self._W2 @ h + self._b2
        # L2 归一化与目标编码器对齐
        norm = np.linalg.norm(y)
        if norm > 1e-8:
            y = y / norm

        self._last_hidden = h
        self._last_prediction = y
        return y

    # ------------------------------------------------------------------ #
    # 误差驱动更新（Hebbian）
    # ------------------------------------------------------------------ #
    def update(self, latent_state: np.ndarray, target: np.ndarray) -> float:
        """计算预测误差并驱动权重更新。

        训练信号: error = predictor(online_state) - target_encoder(obs)
        使用 Hebbian 风格更新：ΔW ∝ -lr * error * input

        Returns
        -------
        error_norm : float
            预测误差的 L2 范数，作为自由能信号。
        """
        pred = self.predict(latent_state)
        tgt = np.asarray(target, dtype=np.float64).flatten()
        if tgt.size != self.target_dim:
            if tgt.size < self.target_dim:
                tgt = np.pad(tgt, (0, self.target_dim - tgt.size))
            else:
                tgt = tgt[: self.target_dim]

        error = pred - tgt
        error_norm = float(np.linalg.norm(error))
        self._last_error_norm = error_norm

        # Hebbian 更新（梯度下降近似）
        x = np.asarray(latent_state, dtype=np.float64).flatten()
        if x.size != self.latent_dim:
            x = (
                np.pad(x, (0, self.latent_dim - x.size))
                if x.size < self.latent_dim
                else x[: self.latent_dim]
            )
        h = self._last_hidden if self._last_hidden is not None else np.zeros(self.hidden_dim)

        # ΔW2 = -lr * outer(error, h)
        grad_W2 = np.outer(error, h)
        grad_norm = np.linalg.norm(grad_W2)
        if grad_norm > 1.0:  # 梯度裁剪
            grad_W2 = grad_W2 / grad_norm
        self._W2 -= self.lr * grad_W2
        self._b2 -= self.lr * error

        # 反向传播到隐层（简化：用 tanh 导数）
        dh = (self._W2.T @ error) * (1.0 - h * h)
        grad_W1 = np.outer(dh, x)
        grad_norm1 = np.linalg.norm(grad_W1)
        if grad_norm1 > 1.0:
            grad_W1 = grad_W1 / grad_norm1
        self._W1 -= self.lr * grad_W1
        self._b1 -= self.lr * dh

        self._update_count += 1
        return error_norm

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    @property
    def last_prediction(self) -> np.ndarray | None:
        return self._last_prediction

    @property
    def last_error_norm(self) -> float:
        return self._last_error_norm

    def get_online_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """返回在线权重 (W, b)，供 TargetEncoder EMA 追踪。

        注意：这里返回的是 W2（输出层），因为它与 target_dim 对齐，
        可作为目标编码器权重的代理。
        """
        return self._W2.copy(), self._b2.copy()

    def snapshot(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "target_dim": self.target_dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "last_error_norm": round(self._last_error_norm, 6),
            "update_count": self._update_count,
            "w1_norm": float(np.linalg.norm(self._W1)),
            "w2_norm": float(np.linalg.norm(self._W2)),
            "last_prediction_3d": (
                [round(float(x), 6) for x in self._last_prediction[:3]]
                if self._last_prediction is not None
                else [0.0, 0.0, 0.0]
            ),
        }
