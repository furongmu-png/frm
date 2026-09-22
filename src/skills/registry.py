"""技能注册中心：管理所有技能的启用/禁用、参数和统一调用。

设计原则：
- 注册中心本身不导入具体技能类（避免循环依赖）；技能在各维度
  ``__init__`` 中自注册，或由 ``HierarchicalZeroDataModel._init_skills``
  显式注册。
- ``process_all`` 按维度顺序调用所有已启用技能，返回合并后的 metadata dict。
- 线程安全（RLock），与 Phase 2/3 一致。
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from .base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)

#: 技能维度调用顺序（感知 → 认知 → 交互 → 专业 → 元）
_DIMENSION_ORDER = ("perception", "cognition", "interaction", "expertise", "meta")


class SkillRegistry:
    """线程安全的技能注册中心。"""

    def __init__(self) -> None:
        self._skills: dict[str, SkillBase] = {}
        self._lock = threading.RLock()

    def register(self, skill: SkillBase) -> None:
        """注册一个技能实例。"""
        with self._lock:
            self._skills[skill.name] = skill
            logger.debug("registered skill: %s", skill.name)

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name in self._skills:
                del self._skills[name]
                return True
            return False

    def get(self, name: str) -> SkillBase | None:
        with self._lock:
            return self._skills.get(name)

    def enable(self, name: str) -> bool:
        with self._lock:
            skill = self._skills.get(name)
            if skill is not None:
                skill.enabled = True
                return True
            return False

    def disable(self, name: str) -> bool:
        with self._lock:
            skill = self._skills.get(name)
            if skill is not None:
                skill.enabled = False
                return True
            return False

    def list_skills(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": s.name,
                    "dimension": s.dimension,
                    "enabled": s.enabled,
                    "ready": s._last_result is not None,
                }
                for s in self._skills.values()
            ]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._skills)

    @property
    def enabled_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._skills.values() if s.enabled)

    def process_all(self, ctx: SkillContext) -> dict[str, Any]:
        """按维度顺序调用所有技能，返回合并 metadata。

        Returns
        -------
        dict
            ``{skill_name: SkillResult.to_dict(), ...}``，
            直接赋给 ``signal.metadata["skills"]``。
        """
        results: dict[str, Any] = {}
        with self._lock:
            skills = list(self._skills.values())

        # 按维度排序
        dim_rank = {d: i for i, d in enumerate(_DIMENSION_ORDER)}
        ordered = sorted(skills, key=lambda s: (dim_rank.get(s.dimension, 99), s.name))

        for skill in ordered:
            result = skill.safe_process(ctx)
            results[skill.name] = result.to_dict()
        return results

    def snapshot(self) -> dict[str, Any]:
        """返回所有技能的快照（供前端面板）。"""
        with self._lock:
            return {
                "count": len(self._skills),
                "enabled_count": self.enabled_count,
                "skills": [s.snapshot() for s in self._skills.values()],
            }

    def clear(self) -> None:
        with self._lock:
            self._skills.clear()


# ------------------------------------------------------------------ #
# 模块级单例
# ------------------------------------------------------------------ #
_registry: SkillRegistry | None = None


def get_registry() -> SkillRegistry:
    """返回全局 :class:`SkillRegistry` 单例。"""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry
