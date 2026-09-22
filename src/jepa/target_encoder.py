"""JEPA Target Encoder — EMA-updated latent representation of observations.

将预测从像素空间迁移到隐空间。目标编码器使用指数移动平均（EMA）
更新权重，不参与梯度/误差驱动更新，保持稳定作为预测目标。

参考: LeCun et al., "A Path Towards Autonomous Machine Intelligence" (2022).
"""

from __future__ import annotations

import numpy as np


class TargetEncoder:
    """EMA 目标编码器。

    输入原始观测（物理帧 / 文本块 / 其他模态），输出固定维度隐空间表示。
    权重通过 EMA 从在线编码器追踪而来，自身不直接学习。

    Parameters
    ----------
    input_dim : int
        原始观测展平后的维度。
    latent_dim : int, default 64
        输出隐空间维度。
    ema_tau : float, default 0.996
        EMA 衰减系数: θ_target = τ * θ_online + (1-τ) * θ_target。
        越大越保守（接近 1）。τ 初始 0.996，逐步衰减到 ema_tau_min。
    ema_tau_min : float, default 0.9
        τ 的下限，防止 EMA 过慢导致目标滞后。
    ema_decay_step : int, default 1000
        每多少步将 τ 向 ema_tau_min 衰减一次（×0.999）。
    seed : int, default 42
        固定随机初始化权重。
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        ema_tau: float = 0.996,
        ema_tau_min: float = 0.9,
        ema_decay_step: int = 1000,
        seed: int = 42,
    ) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if not 0.0 < ema_tau <= 1.0:
            raise ValueError(f"ema_tau must be in (0, 1], got {ema_tau}")
        if not 0.0 < ema_tau_min <= ema_tau:
            raise ValueError(
                f"ema_tau_min must be in (0, ema_tau], got {ema_tau_min} > {ema_tau}"
            )

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.ema_tau = float(ema_tau)
        self.ema_tau_min = float(ema_tau_min)
        self.ema_decay_step = int(ema_decay_step)

        rng = np.random.default_rng(seed)
        # 固定随机初始化的非线性投影：input_dim → latent_dim。
        # Xavier 初始化保持方差稳定。
        scale = np.sqrt(2.0 / (input_dim + latent_dim))
        self._W = rng.standard_normal((latent_dim, input_dim)) * scale
        self._b = np.zeros(latent_dim)
        # 在线权重快照（用于 EMA 追踪），初始与目标相同。
        self._W_online = self._W.copy()
        self._b_online = self._b.copy()

        self._step = 0
        self._last_target: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # 编码
    # ------------------------------------------------------------------ #
    def encode(self, observation: np.ndarray) -> np.ndarray:
        """对原始观测计算目标隐表示。

        使用 tanh 非线性投影，输出 L2 归一化的 latent_dim 维向量。
        不更新任何权重（目标编码器自身）。
        """
        obs = np.asarray(observation, dtype=np.float64).flatten()
        if obs.size != self.input_dim:
            # 零填充或截断以匹配维度
            if obs.size < self.input_dim:
                obs = np.pad(obs, (0, self.input_dim - obs.size))
            else:
                obs = obs[: self.input_dim]
        h = self._W @ obs + self._b
        z = np.tanh(h)
        # L2 归一化，使预测误差尺度稳定
        norm = np.linalg.norm(z)
        if norm > 1e-8:
            z = z / norm
        self._last_target = z
        return z

    # ------------------------------------------------------------------ #
    # EMA 更新
    # ------------------------------------------------------------------ #
    def update_from_online(
        self, online_W: np.ndarray, online_b: np.ndarray | None = None
    ) -> None:
        """用在线编码器权重通过 EMA 更新目标编码器。

        θ_target = τ * θ_online + (1-τ) * θ_target

        Parameters
        ----------
        online_W : np.ndarray
            在线编码器的权重矩阵，形状与 self._W 相同。
        online_b : np.ndarray | None
            在线编码器的偏置，形状与 self._b 相同。None 则只更新 W。
        """
        W = np.asarray(online_W, dtype=np.float64)
        if W.shape != self._W.shape:
            raise ValueError(
                f"online_W shape {W.shape} != target shape {self._W.shape}"
            )
        tau = self.ema_tau
        self._W = tau * self._W + (1.0 - tau) * W
        if online_b is not None:
            b = np.asarray(online_b, dtype=np.float64)
            if b.shape != self._b.shape:
                raise ValueError(
                    f"online_b shape {b.shape} != target shape {self._b.shape}"
                )
            self._b = tau * self._b + (1.0 - tau) * b

        # 缓存在线权重供 snapshot
        self._W_online = W.copy()
        if online_b is not None:
            self._b_online = b.copy()

        self._step += 1
        if (
            self.ema_decay_step > 0
            and self._step % self.ema_decay_step == 0
            and self.ema_tau > self.ema_tau_min
        ):
            self.ema_tau = max(self.ema_tau_min, self.ema_tau * 0.999)

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    @property
    def weights(self) -> np.ndarray:
        """目标编码器当前权重（只读视图）。"""
        return self._W

    @property
    def last_target(self) -> np.ndarray | None:
        """最近一次 encode() 的输出，供 snapshot 与可视化。"""
        return self._last_target

    def snapshot(self) -> dict:
        """返回 JSON 安全的快照字典。"""
        return {
            "latent_dim": self.latent_dim,
            "input_dim": self.input_dim,
            "ema_tau": round(self.ema_tau, 6),
            "step": self._step,
            "target_norm": float(np.linalg.norm(self._W)),
            "online_norm": float(np.linalg.norm(self._W_online)),
            # EMA 追踪度：目标与在线权重的余弦相似度（越接近 1 越同步）
            "tracking": float(self._cosine(self._W, self._W_online)),
            "last_target_3d": (
                [round(float(x), 6) for x in self._last_target[:3]]
                if self._last_target is not None
                else [0.0, 0.0, 0.0]
            ),
        }

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a.flatten(), b.flatten()) / (na * nb))
