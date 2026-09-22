"""触觉感知技能。

在物理沙盒中模拟压力阵列（物体接触时返回接触区域的压力值矩阵），
并将压力矩阵编码为隐空间向量，与物理帧和文本共享跨模态桥接。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class TactileSensor:
    """模拟 8×8 压力阵列触觉传感器。

    在物理沙盒的 128×128 帧中检测物体接触区域，
    将接触区域的像素密度映射为压力值矩阵。
    """

    def __init__(
        self,
        grid_size: int = 8,
        frame_size: int = 128,
        contact_threshold: int = 200,
    ) -> None:
        self.grid_size = grid_size
        self.frame_size = frame_size
        self.contact_threshold = contact_threshold
        self.cell_size = frame_size // grid_size

    def sense(self, frame: np.ndarray) -> np.ndarray:
        """从物理帧提取 8×8 压力矩阵。

        每个 grid cell 的压力值 = 该区域中超过接触阈值的像素的归一化密度。
        """
        if frame.shape != (self.frame_size, self.frame_size):
            frame = self._fit(frame)
        f = frame.astype(np.float64)
        pressures = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        cs = self.cell_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = f[i * cs : (i + 1) * cs, j * cs : (j + 1) * cs]
                contact = cell[cell >= self.contact_threshold]
                if len(contact) > 0:
                    # 归一化到 [0, 1]
                    pressures[i, j] = float(np.mean(contact)) / 255.0
        return pressures

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        result = np.zeros((self.frame_size, self.frame_size), dtype=frame.dtype)
        h, w = frame.shape[:2]
        h2, w2 = min(h, self.frame_size), min(w, self.frame_size)
        result[:h2, :w2] = frame[:h2, :w2]
        return result

    def has_contact(self, pressures: np.ndarray) -> bool:
        return float(pressures.max()) > 0.0

    def contact_center(self, pressures: np.ndarray) -> tuple[float, float] | None:
        """返回接触中心坐标（归一化 0-1），无接触返回 None。"""
        if not self.has_contact(pressures):
            return None
        total = pressures.sum()
        if total == 0:
            return None
        ys, xs = np.indices(pressures.shape)
        cx = float(np.sum(xs * pressures) / total / self.grid_size)
        cy = float(np.sum(ys * pressures) / total / self.grid_size)
        return cx, cy


class TactileEncoder(SkillBase):
    """将压力矩阵编码为 32 维隐空间向量。

    使用固定随机投影（与视觉/音频编码器一致的风格），
    保持跨模态桥接的维度对齐。
    """

    name = "tactile_sensor"
    dimension = "perception"

    def __init__(
        self,
        *,
        grid_size: int = 8,
        latent_dim: int = 32,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self._sensor = TactileSensor(grid_size=grid_size)
        self.latent_dim = latent_dim
        self._rng = np.random.default_rng(123)
        grid_flat = grid_size * grid_size
        self._proj = self._rng.standard_normal((grid_flat, latent_dim)).astype(
            np.float64
        ) * (1.0 / np.sqrt(grid_flat))

        self._last_pressures: np.ndarray | None = None
        self._last_latent: np.ndarray | None = None

    def encode(self, pressures: np.ndarray) -> np.ndarray:
        """将压力矩阵编码为 latent 向量。"""
        flat = pressures.flatten()
        latent = flat @ self._proj
        # L2 归一化
        norm = np.linalg.norm(latent)
        if norm > 1e-10:
            latent = latent / norm
        return latent

    def describe_texture(self, pressures: np.ndarray) -> str:
        """根据压力分布描述表面纹理。"""
        if not self._sensor.has_contact(pressures):
            return "无接触"
        pmax = float(pressures.max())
        pmean = float(pressures.mean())
        pstd = float(pressures.std())
        if pstd < 0.05:
            return "表面均匀（平滑）"
        if pmax > 0.8:
            return "硬表面（高压集中）"
        if pmean > 0.3:
            return "软表面（压力分散）"
        return "粗糙表面（压力不均）"

    def process(self, ctx: SkillContext) -> SkillResult:
        frame = ctx.raw_observation
        if frame is None:
            return SkillResult(name=self.name, data={"ready": False})

        pressures = self._sensor.sense(frame)
        latent = self.encode(pressures)
        center = self._sensor.contact_center(pressures)
        texture = self.describe_texture(pressures)

        self._last_pressures = pressures
        self._last_latent = latent

        return SkillResult(
            name=self.name,
            data={
                "pressures": pressures.tolist(),
                "latent": latent.tolist(),
                "has_contact": self._sensor.has_contact(pressures),
                "contact_center": list(center) if center else None,
                "max_pressure": float(pressures.max()),
                "texture": texture,
            },
        )
