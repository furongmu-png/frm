"""技能层：可插拔的认知/感知/交互/专业/元技能模块。

所有技能均遵循与 Phase 2/3 一致的契约：
- 特性开关驱动（``enable_skills``）；关闭时零成本。
- 只读核心：技能观察已计算的状态，不回调核心模块的 process/predict/update。
- try/except 隔离：单个技能失败只 warn，不崩溃 ``think()``。
- 仅写 ``signal.metadata["skills"]``，且仅在至少一个技能启用时附加该键。

技能分五个维度：
- ``perception``   感知扩展（深度、触觉、音频场景、嗅觉）
- ``cognition``    认知深化（心理理论、叙事、情感、类比）
- ``interaction``  交互协作（对话、示教、机器教学、多模态翻译）
- ``expertise``    专业技能（程序合成、定理证明、游戏、异常检测）
- ``meta``         元技能（学习如何学习、课程、遗忘、能耗感知）
"""
from __future__ import annotations

from .base import SkillBase, SkillContext, SkillResult
from .registry import SkillRegistry, get_registry

__all__ = [
    "SkillBase",
    "SkillContext",
    "SkillResult",
    "SkillRegistry",
    "get_registry",
]
