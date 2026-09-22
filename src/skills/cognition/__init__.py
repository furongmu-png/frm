"""认知深化技能：心理理论、叙事理解、情感、类比推理。"""
from __future__ import annotations

from .theory_of_mind import TheoryOfMindModule
from .narrative import NarrativeModule
from .affect import AffectModule
from .analogy import AnalogyEngine

__all__ = [
    "TheoryOfMindModule",
    "NarrativeModule",
    "AffectModule",
    "AnalogyEngine",
]
