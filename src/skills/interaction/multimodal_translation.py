"""多模态翻译技能。

强化跨模态桥接：
- 物理→文本：给定一帧，生成描述性句子。
- 文本→物理：给定指令，生成动作序列在沙盒中执行。
使用跨模态预测误差驱动对齐。
"""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)

#: 方向动作映射
ACTION_MAP = {
    "left": 0,
    "right": 1,
    "up": 2,
    "stop": 3,
    "no": 3,
    "左": 0,
    "右": 1,
    "上": 2,
    "停": 3,
    "stop": 3,
}

#: 物体描述模板
OBJECT_DESCRIPTIONS = [
    "a bright object",
    "a dark region",
    "a moving body",
    "a stationary shape",
    "multiple objects",
    "an empty space",
]


class MultiModalTranslator(SkillBase):
    """多模态翻译模块。

    - 物理→文本：从帧中提取特征，生成描述
    - 文本→物理：解析指令，生成动作序列
    - 使用跨模态预测误差驱动对齐
    """

    name = "multimodal_translation"
    dimension = "interaction"

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        frame_size: int = 128,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self._latent_dim = latent_dim
        self.frame_size = frame_size
        self._rng = np.random.default_rng(444)

        # 物理→文本投影
        self._phys_to_text_proj = self._rng.standard_normal(
            (16, latent_dim)
        ).astype(np.float64) * (1.0 / 4.0)

        # 文本→物理投影
        self._text_to_phys_proj = self._rng.standard_normal(
            (256, latent_dim)
        ).astype(np.float64) * (1.0 / 16.0)

        # 跨模态对齐矩阵（学习更新）
        self._align_W = np.eye(latent_dim, dtype=np.float64) * 0.1
        self._last_alignment_error: float = 0.0

    def _encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """编码物理帧为 16 维特征向量。"""
        if frame is None:
            return np.zeros(16)
        f = np.asarray(frame, dtype=np.float64)
        if f.ndim == 2:
            # 提取统计特征
            features = np.array(
                [
                    f.mean(),
                    f.std(),
                    f.max(),
                    f.min(),
                    float(np.sum(f > 128)),  # 亮区像素数
                    float(np.sum(f > 200)),  # 高亮像素数
                    f.mean(axis=0).mean(),  # 列均值
                    f.mean(axis=1).mean(),  # 行均值
                    float(np.argmax(f.mean(axis=0))),  # 最亮列位置
                    float(np.argmax(f.mean(axis=1))),  # 最亮行位置
                    float(np.std(f.mean(axis=0))),  # 列均值方差
                    float(np.std(f.mean(axis=1))),  # 行均值方差
                    float(np.percentile(f, 25)),
                    float(np.percentile(f, 75)),
                    float(np.sum(f > 128) / f.size),  # 亮区比例
                    float(f.size - np.sum(f > 50)),  # 暗区像素数
                ]
            )
        else:
            features = np.zeros(16)
        return features

    def _encode_text(self, text: str) -> np.ndarray:
        """编码文本为 latent。"""
        hist = np.zeros(256, dtype=np.float64)
        for byte in text.encode("utf-8", errors="replace"):
            hist[byte] += 1
        total = hist.sum()
        if total > 0:
            hist /= total
        latent = hist @ self._text_to_phys_proj
        norm = np.linalg.norm(latent)
        if norm > 1e-10:
            latent = latent / norm
        return latent

    def frame_to_text(self, frame: np.ndarray) -> str:
        """物理→文本翻译：从帧生成描述性句子。"""
        features = self._encode_frame(frame)

        # 基于特征生成描述
        brightness = features[0]
        n_bright = int(features[4])
        n_objects = max(1, int(features[5] / 100))

        # 选择描述模板
        if brightness < 10:
            obj_desc = "a dark scene"
        elif n_bright > 5000:
            obj_desc = OBJECT_DESCRIPTIONS[0]
        elif n_bright > 2000:
            obj_desc = OBJECT_DESCRIPTIONS[3]
        elif n_bright > 500:
            obj_desc = OBJECT_DESCRIPTIONS[2]
        else:
            obj_desc = OBJECT_DESCRIPTIONS[5]

        # 位置描述
        col_pos = features[8] / self.frame_size
        row_pos = features[9] / self.frame_size

        if col_pos < 0.33:
            horiz = "left"
        elif col_pos > 0.67:
            horiz = "right"
        else:
            horiz = "center"

        if row_pos < 0.33:
            vert = "top"
        elif row_pos > 0.67:
            vert = "bottom"
        else:
            vert = "middle"

        return f"I see {obj_desc} in the {vert}-{horiz} area."

    def text_to_actions(self, text: str, max_steps: int = 10) -> list[int]:
        """文本→物理翻译：从指令生成动作序列。

        支持的指令格式：
        - "move left" / "go right" / "up" / "stop"
        - "push to left" / "将红球推到右上角"
        - "go left 3 times"
        """
        lower = text.lower()
        actions: list[int] = []

        # 检测重复次数
        repeat_match = re.search(r"(\d+)\s*times?", lower)
        repeat = int(repeat_match.group(1)) if repeat_match else 1
        repeat = min(repeat, max_steps)

        # 检测方向
        detected_action: int | None = None
        for keyword, action in ACTION_MAP.items():
            if keyword in lower:
                detected_action = action
                break

        if detected_action is not None:
            for _ in range(repeat):
                actions.append(detected_action)
        else:
            # 没有明确方向：解析组合指令
            if "right" in lower and "up" in lower:
                actions.extend([1, 2] * repeat)
            elif "left" in lower and "up" in lower:
                actions.extend([0, 2] * repeat)
            elif "right" in lower and "down" in lower:
                actions.extend([1, 3] * repeat)
            else:
                # 默认：探索动作
                actions.append(3)

        return actions[:max_steps]

    def compute_alignment_error(
        self, frame: np.ndarray, text: str
    ) -> float:
        """计算跨模态对齐误差。"""
        phys_features = self._encode_frame(frame)
        phys_latent = phys_features @ self._phys_to_text_proj
        text_latent = self._encode_text(text)

        # 对齐：通过 align_W 投影物理 latent 到文本空间
        projected = phys_latent @ self._align_W
        error = float(np.linalg.norm(projected - text_latent))
        self._last_alignment_error = error
        return error

    def update_alignment(
        self, frame: np.ndarray, text: str, lr: float = 0.01
    ) -> float:
        """更新对齐矩阵以降低跨模态误差。

        使用归一化的物理特征与误差向量计算梯度，避免大数值帧统计
        （如 argmax 位置）造成的数值爆炸。
        """
        phys_features = self._encode_frame(frame)
        phys_latent = phys_features @ self._phys_to_text_proj
        # 归一化 phys_latent 到单位范数，确保梯度幅度可控
        phys_norm = float(np.linalg.norm(phys_latent)) + 1e-9
        phys_latent_n = phys_latent / phys_norm
        text_latent = self._encode_text(text)

        projected = phys_latent_n @ self._align_W
        error_vec = projected - text_latent

        # 梯度下降更新（梯度幅度受限于 ||phys_latent_n|| * ||error_vec||）
        grad = np.outer(phys_latent_n, error_vec)
        # 额外的梯度裁剪，防止单次更新过大
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1.0:
            grad = grad / grad_norm
        self._align_W -= lr * grad
        self._last_alignment_error = float(np.linalg.norm(error_vec))
        return self._last_alignment_error

    def process(self, ctx: SkillContext) -> SkillResult:
        frame = ctx.raw_observation
        text = None

        if isinstance(frame, dict):
            text = frame.get("text")
            frame = frame.get("frame")

        # 物理→文本翻译
        if frame is not None:
            frame = np.asarray(frame, dtype=np.float64)
            description = self.frame_to_text(frame)
            phys_to_text = description
        else:
            phys_to_text = None

        # 文本→物理翻译
        if text is not None:
            actions = self.text_to_actions(text)
            text_to_phys = actions
        else:
            text_to_phys = []

        # 跨模态对齐误差
        alignment_error = 0.0
        if frame is not None and text is not None:
            alignment_error = self.compute_alignment_error(frame, text)

        return SkillResult(
            name=self.name,
            data={
                "frame_to_text": phys_to_text,
                "text_to_actions": text_to_phys,
                "alignment_error": alignment_error,
                "n_actions": len(text_to_phys),
            },
        )
