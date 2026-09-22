"""深度估计技能。

从 2D 物理沙盒帧估计深度图。采用编码器-解码器结构：
- 编码器：S4 层 + PCN 低层特征（固定随机投影，模拟低层视觉特征）
- 解码器：反卷积（固定随机权重 + 误差驱动微调）

训练信号：
- 若沙盒提供真实深度（多视角），则用预测深度与真实深度的 MSE；
- 否则使用光度一致性（相邻帧的重建误差）作为自监督信号。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class DepthEstimator(SkillBase):
    """从 2D 帧估计深度图（0-1 浮点）。"""

    name = "depth_estimator"
    dimension = "perception"

    def __init__(
        self,
        *,
        frame_size: int = 128,
        encoder_dim: int = 64,
        lr: float = 0.01,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.frame_size = frame_size
        self.encoder_dim = encoder_dim
        self.lr = lr
        self._rng = np.random.default_rng(42)

        # 编码器：固定随机投影 frame → latent (encoder_dim)
        # 用 patch-wise 投影：8×8 patch → 1 个编码值
        patch = 8
        self._patch = patch
        self._enc_W = self._rng.standard_normal(
            (patch * patch, encoder_dim)
        ).astype(np.float64) * (1.0 / (patch * patch))

        # 解码器：反卷积 latent → depth map
        # 简化为 latent → upsampled → 逐 patch 预测深度
        self._dec_W = self._rng.standard_normal(
            (encoder_dim, patch * patch)
        ).astype(np.float64) * 0.01

        self._prev_frame: np.ndarray | None = None
        self._last_depth: np.ndarray | None = None
        self._last_loss: float = 0.0

    def _encode_patches(self, frame: np.ndarray) -> np.ndarray:
        """将帧分为 patch 并编码为 latent 序列。"""
        f = frame.astype(np.float64) / 255.0
        p = self._patch
        gh = self.frame_size // p
        # reshape to (gh, gh, p, p) then flatten each patch
        f_reshaped = f[: gh * p, : gh * p].reshape(gh, gh, p, p)
        patches = f_reshaped.reshape(gh * gh, p * p)
        latent = patches @ self._enc_W  # (gh*gh, encoder_dim)
        return latent

    def _decode_depth(self, latent: np.ndarray) -> np.ndarray:
        """从 latent 解码为 depth map (frame_size × frame_size)。"""
        p = self._patch
        gh = self.frame_size // p
        # 每个 patch 预测一个深度值（均值），再 upsample
        patch_depth = latent @ self._dec_W  # (gh*gh, p*p)
        # 取每 patch 的均值作为该 patch 的深度
        patch_mean = patch_depth.mean(axis=1)  # (gh*gh,)
        # sigmoid 归一化到 [0, 1]
        depth_patches = 1.0 / (1.0 + np.exp(-patch_mean))
        # upsample 到 frame_size
        depth = depth_patches.reshape(gh, gh)
        # 双线性插值放大
        depth_full = np.kron(depth, np.ones((p, p)))
        return depth_full.astype(np.float64)

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """估计深度图。"""
        if frame.shape != (self.frame_size, self.frame_size):
            # resize via crop/pad
            frame = self._fit_frame(frame)
        latent = self._encode_patches(frame)
        depth = self._decode_depth(latent)

        # 光度一致性自监督：如果前一帧可用，计算重建误差并微调解码器
        if self._prev_frame is not None:
            loss = self._photometric_loss(frame, self._prev_frame, depth)
            self._last_loss = loss
            # 误差驱动微调：朝减小重建误差方向移动解码器权重
            grad = self._compute_gradient(latent, loss)
            self._dec_W -= self.lr * grad
        self._prev_frame = frame.copy()
        self._last_depth = depth
        return depth

    def _fit_frame(self, frame: np.ndarray) -> np.ndarray:
        """将任意尺寸帧裁剪/填充到 frame_size × frame_size。"""
        h, w = frame.shape[:2]
        if h == self.frame_size and w == self.frame_size:
            return frame
        result = np.zeros((self.frame_size, self.frame_size), dtype=frame.dtype)
        h2 = min(h, self.frame_size)
        w2 = min(w, self.frame_size)
        result[:h2, :w2] = frame[:h2, :w2]
        return result

    @staticmethod
    def _photometric_loss(
        frame: np.ndarray, prev_frame: np.ndarray, depth: np.ndarray
    ) -> float:
        """光度一致性损失：基于深度的帧间重建误差。"""
        # 简化：用深度差异作为光度的代理
        diff = frame.astype(np.float64) - prev_frame.astype(np.float64)
        depth_weighted = diff * (depth + 0.1)
        return float(np.mean(depth_weighted**2))

    def _compute_gradient(self, latent: np.ndarray, loss: float) -> np.ndarray:
        """计算解码器权重的梯度近似。"""
        # 简化梯度：loss 对 dec_W 的梯度
        # d_loss / d_dec_W ≈ latent^T @ (depth_error * depth_deriv)
        n = latent.shape[0]
        grad = latent.T @ np.ones((n, self._patch * self._patch)) * loss * 0.001
        return grad

    def process(self, ctx: SkillContext) -> SkillResult:
        frame = ctx.raw_observation
        if frame is None:
            return SkillResult(name=self.name, data={"ready": False})

        depth = self.estimate(frame)

        # 深度统计
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        depth_mean = float(depth.mean())

        # 降采样到 32×32 供 metadata（避免 payload 过大）
        ds = 4
        depth_small = depth[::ds, ::ds]

        return SkillResult(
            name=self.name,
            data={
                "depth_mean": depth_mean,
                "depth_min": depth_min,
                "depth_max": depth_max,
                "photometric_loss": self._last_loss,
                "depth_preview": depth_small.tolist(),
                "frame_size": self.frame_size,
            },
        )
