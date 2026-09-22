"""Integrated World-Self Model (IWSM) — Stage-2 cognitive upgrade.

Provides:
  - SelfSchema             : predicts own attention/action/affect
  - AutobiographicalMemory : self-episode store with narrative retrieval
  - PhiSelf                : self-relevant integrated information Φ_self
  - CounterfactualSelf     : "what if I had acted differently" explanations

All components follow the FEP style: thread-safe, Hebbian, numpy-only.
"""
from .self_schema import SelfSchema, SelfPrediction, SelfSchemaStats
from .autobiographical_memory import (
    AutobiographicalMemory,
    SelfEpisode,
    NarrativeResult,
)
from .phi_self import PhiSelf, PhiSelfState
from .counterfactual_self import CounterfactualSelf, CounterfactualResult

__all__ = [
    "SelfSchema",
    "SelfPrediction",
    "SelfSchemaStats",
    "AutobiographicalMemory",
    "SelfEpisode",
    "NarrativeResult",
    "PhiSelf",
    "PhiSelfState",
    "CounterfactualSelf",
    "CounterfactualResult",
]
