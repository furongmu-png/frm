"""示教学习技能。

记录人类演示的动作序列，将演示作为高权重经验，
优先存入 Hopfield 记忆并在巩固阶段多次回放。
通过模仿学习：在类似状态下，动作选择偏向于记忆中相似演示的动作（软约束）。
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class Demonstration:
    """单次演示记录。"""

    def __init__(
        self,
        states: list[np.ndarray],
        actions: list[int],
        reward: float = 1.0,
        label: str = "",
    ) -> None:
        self.states = states
        self.actions = actions
        self.reward = reward
        self.label = label
        self.length = len(states)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "length": self.length,
            "reward": self.reward,
            "actions": self.actions,
            "first_state": self.states[0].tolist()[:8] if self.states else [],
        }


class DemonstrationLearner(SkillBase):
    """示教学习模块。

    - 记录人类演示的动作序列
    - 将演示作为高权重经验存入记忆
    - 在巩固阶段多次回放
    - 模仿学习：相似状态下偏向演示动作（软约束）
    """

    name = "learning_from_demo"
    dimension = "interaction"

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        max_demos: int = 20,
        replay_multiplier: int = 3,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self._latent_dim = latent_dim
        self.max_demos = max_demos
        self.replay_multiplier = replay_multiplier
        self._rng = np.random.default_rng(222)

        # 演示存储
        self._demos: deque[Demonstration] = deque(maxlen=max_demos)

        # 模仿记忆：state latent → action
        self._imitation_memory: list[tuple[np.ndarray, int, float]] = []
        # (state_latent, action, weight)

        # 简单状态编码器
        self._proj = self._rng.standard_normal(
            (64, latent_dim)
        ).astype(np.float64) * (1.0 / 8.0)

        self._replay_count: int = 0
        self._last_imitation_strength: float = 0.0

    def _encode_state(self, state: np.ndarray) -> np.ndarray:
        """将状态编码为 latent。"""
        s = np.asarray(state, dtype=np.float64).flatten()[:64]
        if s.size < 64:
            s = np.pad(s, (0, 64 - s.size))
        latent = s @ self._proj
        norm = np.linalg.norm(latent)
        if norm > 1e-10:
            latent = latent / norm
        return latent

    def record_demonstration(
        self,
        states: list[np.ndarray],
        actions: list[int],
        reward: float = 1.0,
        label: str = "",
    ) -> None:
        """记录一次演示。"""
        demo = Demonstration(states, actions, reward, label)
        self._demos.append(demo)

        # 将演示存入模仿记忆（高权重）
        weight = reward * 2.0  # 演示权重是普通经验的 2 倍
        for state, action in zip(states, actions, strict=False):
            latent = self._encode_state(state)
            self._imitation_memory.append((latent, action, weight))

    def consolidate(self) -> dict[str, Any]:
        """巩固阶段：多次回放演示。"""
        if not self._demos:
            return {"replayed": 0, "n_demos": 0}

        total_replays = 0
        for demo in self._demos:
            for _ in range(self.replay_multiplier):
                for state, action in zip(demo.states, demo.actions, strict=False):
                    latent = self._encode_state(state)
                    self._imitation_memory.append(
                        (latent, action, demo.reward * 1.5)
                    )
                    total_replays += 1

        self._replay_count += total_replays
        return {
            "replayed": total_replays,
            "n_demos": len(self._demos),
            "total_replays": self._replay_count,
        }

    def imitate(self, current_state: np.ndarray) -> tuple[int, float] | None:
        """模仿学习：返回最相似演示状态对应的动作和相似度。

        Returns
        -------
        (action, similarity) or None if no demos available.
        """
        if not self._imitation_memory:
            return None

        query_latent = self._encode_state(current_state)

        # 找最相似的记忆
        best_action = -1
        best_score = -1.0
        for latent, action, weight in self._imitation_memory:
            sim = float(np.dot(query_latent, latent)) * weight
            if sim > best_score:
                best_score = sim
                best_action = action

        if best_action < 0:
            return None

        # 归一化相似度到 [0, 1]
        similarity = 1.0 / (1.0 + np.exp(-best_score))
        self._last_imitation_strength = similarity
        return best_action, similarity

    def get_imitation_bias(
        self, current_state: np.ndarray, n_actions: int = 4
    ) -> np.ndarray:
        """获取模仿偏好向量（软约束）。

        返回一个 n_actions 维向量，值越大表示越偏向该动作。
        """
        result = self.imitate(current_state)
        if result is None:
            return np.ones(n_actions) / n_actions  # 均匀分布

        action, similarity = result
        bias = np.ones(n_actions) * (1 - similarity) / n_actions
        if 0 <= action < n_actions:
            bias[action] += similarity
        return bias / bias.sum()  # 归一化

    def process(self, ctx: SkillContext) -> SkillResult:
        state = ctx.belief if ctx.belief is not None else np.zeros(64)

        # 获取模仿建议
        imitation = self.imitate(state)
        bias = self.get_imitation_bias(state, n_actions=4)

        return SkillResult(
            name=self.name,
            data={
                "n_demos": len(self._demos),
                "n_memories": len(self._imitation_memory),
                "total_replays": self._replay_count,
                "imitation_available": imitation is not None,
                "imitation_action": imitation[0] if imitation else -1,
                "imitation_strength": imitation[1] if imitation else 0.0,
                "imitation_bias": bias.tolist(),
                "last_imitation_strength": self._last_imitation_strength,
                "demo_labels": [d.label for d in self._demos][-5:],
            },
        )
