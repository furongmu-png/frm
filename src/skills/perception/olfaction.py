"""嗅觉感知技能。

在物理沙盒中模拟气味扩散（二维高斯浓度场），
将智能体当前位置的浓度值编码为向量，作为导航线索。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class OlfactionEncoder(SkillBase):
    """嗅觉编码器：模拟气味扩散并编码为导航信号。

    气味源在 2D 场中按高斯扩散，智能体在网格中移动时
    感知当前位置的浓度值。浓度梯度驱动导航行为。
    """

    name = "olfaction"
    dimension = "perception"

    def __init__(
        self,
        *,
        frame_size: int = 128,
        latent_dim: int = 16,
        n_sources: int = 1,
        diffusion_sigma: float = 30.0,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.frame_size = frame_size
        self.latent_dim = latent_dim
        self.n_sources = n_sources
        self.diffusion_sigma = diffusion_sigma
        self._rng = np.random.default_rng(789)

        # 浓度场（懒初始化）
        self._concentration_field: np.ndarray | None = None
        self._sources: list[tuple[float, float]] | None = None

        # 固定随机投影：浓度采样 → latent
        n_sensors = 8  # 8 个方向的浓度采样
        self._proj = self._rng.standard_normal((n_sensors, latent_dim)).astype(
            np.float64
        ) * (1.0 / np.sqrt(n_sensors))

        self._last_concentration: float = 0.0
        self._last_gradient: tuple[float, float] | None = None

    def init_field(
        self,
        sources: list[tuple[float, float]] | None = None,
    ) -> None:
        """初始化气味浓度场。

        Parameters
        ----------
        sources : list of (x, y)
            气味源坐标（像素坐标）。若为 None 则随机生成。
        """
        if sources is None:
            sources = [
                (
                    float(self._rng.uniform(0, self.frame_size)),
                    float(self._rng.uniform(0, self.frame_size)),
                )
                for _ in range(self.n_sources)
            ]
        self._sources = sources

        # 生成高斯浓度场
        x = np.arange(self.frame_size, dtype=np.float64)
        y = np.arange(self.frame_size, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)

        field = np.zeros((self.frame_size, self.frame_size), dtype=np.float64)
        for sx, sy in sources:
            field += np.exp(
                -((xx - sx) ** 2 + (yy - sy) ** 2)
                / (2 * self.diffusion_sigma**2)
            )
        # 归一化到 [0, 1]
        mx = field.max()
        if mx > 0:
            field /= mx
        self._concentration_field = field

    def get_concentration(self, pos: tuple[float, float]) -> float:
        """获取指定位置的气味浓度。"""
        if self._concentration_field is None:
            self.init_field()
        assert self._concentration_field is not None
        x = int(np.clip(pos[0], 0, self.frame_size - 1))
        y = int(np.clip(pos[1], 0, self.frame_size - 1))
        return float(self._concentration_field[x, y])

    def get_concentration_field(self) -> np.ndarray:
        """返回整个浓度场（供前端叠加显示）。"""
        if self._concentration_field is None:
            self.init_field()
        assert self._concentration_field is not None
        return self._concentration_field.copy()

    def sense_directional(
        self, pos: tuple[float, float], radius: int = 20
    ) -> np.ndarray:
        """在 8 个方向采样浓度（用于导航）。"""
        if self._concentration_field is None:
            self.init_field()
        assert self._concentration_field is not None

        angles = np.linspace(0, 2 * np.pi, 9)[:-1]  # 8 方向
        concentrations = np.zeros(8, dtype=np.float64)
        px, py = pos
        for i, angle in enumerate(angles):
            dx = int(np.clip(px + radius * np.cos(angle), 0, self.frame_size - 1))
            dy = int(np.clip(py + radius * np.sin(angle), 0, self.frame_size - 1))
            concentrations[i] = self._concentration_field[dx, dy]
        return concentrations

    def compute_gradient(self, pos: tuple[float, float]) -> tuple[float, float]:
        """计算气味浓度梯度（指向气味源的方向）。"""
        if self._concentration_field is None:
            self.init_field()
        assert self._concentration_field is not None

        x = int(np.clip(pos[0], 1, self.frame_size - 2))
        y = int(np.clip(pos[1], 1, self.frame_size - 2))
        gx = self._concentration_field[x + 1, y] - self._concentration_field[x - 1, y]
        gy = self._concentration_field[x, y + 1] - self._concentration_field[x, y - 1]
        return float(gx), float(gy)

    def encode(self, pos: tuple[float, float]) -> np.ndarray:
        """编码当前位置的嗅觉信息为 latent 向量。"""
        directional = self.sense_directional(pos)
        latent = directional @ self._proj
        norm = np.linalg.norm(latent)
        if norm > 1e-10:
            latent = latent / norm
        return latent

    def _infer_agent_pos(self, ctx: SkillContext) -> tuple[float, float]:
        """从信念向量推断智能体位置（将 belief 前两维映射到帧坐标）。"""
        if ctx.belief is not None and len(ctx.belief) >= 2:
            b = ctx.belief[:2]
            # 将 [-1, 1] 范围映射到 [0, frame_size]
            x = float((b[0] + 1) / 2 * self.frame_size)
            y = float((b[1] + 1) / 2 * self.frame_size)
            return x, y
        return float(self.frame_size / 2), float(self.frame_size / 2)

    def process(self, ctx: SkillContext) -> SkillResult:
        pos = self._infer_agent_pos(ctx)
        if self._concentration_field is None:
            self.init_field()

        concentration = self.get_concentration(pos)
        gradient = self.compute_gradient(pos)
        directional = self.sense_directional(pos)
        latent = self.encode(pos)

        # 导航奖励信号：浓度越高，奖励越大
        nav_reward = concentration

        # 梯度方向（归一化）
        gmag = float(np.sqrt(gradient[0] ** 2 + gradient[1] ** 2))
        if gmag > 1e-10:
            grad_dir = (gradient[0] / gmag, gradient[1] / gmag)
        else:
            grad_dir = (0.0, 0.0)

        self._last_concentration = concentration
        self._last_gradient = gradient

        return SkillResult(
            name=self.name,
            data={
                "concentration": concentration,
                "gradient": list(gradient),
                "gradient_direction": list(grad_dir),
                "nav_reward": nav_reward,
                "latent": latent.tolist(),
                "directional": directional.tolist(),
                "agent_pos": list(pos),
                "field_preview": self._downsample_field().tolist(),
            },
        )

    def _downsample_field(self, factor: int = 8) -> np.ndarray:
        """降采样浓度场供 metadata。"""
        assert self._concentration_field is not None
        return self._concentration_field[::factor, ::factor]
