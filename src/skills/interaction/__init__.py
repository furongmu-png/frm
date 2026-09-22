"""交互与协作技能：对话、示教学习、机器教学、多模态翻译。"""
from __future__ import annotations

from .dialogue import DialogueAgent
from .learning_from_demo import DemonstrationLearner
from .machine_teaching import TeachingModule
from .multimodal_translation import MultiModalTranslator

__all__ = [
    "DialogueAgent",
    "DemonstrationLearner",
    "TeachingModule",
    "MultiModalTranslator",
]
