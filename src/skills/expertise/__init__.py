"""专业技能：程序合成、定理证明、游戏策略、异常检测。"""
from __future__ import annotations

from .program_synthesis import ProgramSynthesizer
from .theorem_proving import TheoremProver
from .game_playing import GamePlayer
from .anomaly_detection import AnomalyDetector

__all__ = [
    "ProgramSynthesizer",
    "TheoremProver",
    "GamePlayer",
    "AnomalyDetector",
]
