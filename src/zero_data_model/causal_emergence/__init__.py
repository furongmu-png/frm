# src/zero_data_model/causal_emergence/__init__.py
"""Causal emergence engine package.

Composes five modules (persistent homology, causal discovery, differential
generator, Hamiltonian sampler, chaotic memory) into a single recursive
loop. See ``docs/superpowers/specs/2026-07-19-causal-emergence-engine-design.md``
for the full design specification.

Phase 1 (foundation): modules A (topology) and B (causal_discovery).
Phase 2 (action): module C (differential) — damped least-action generator.
Phase 3 (cognition): modules D (hmc) and E (chaotic_memory).
Phase 4 (closure): the full ``emergence_cycle`` orchestration and the
``compute_emergence_score`` heuristic.
"""

from __future__ import annotations

from .causal_discovery import CausalInferenceEngine
from .chaotic_memory import ChaoticAssociativeMemory
from .differential import DifferentialGenerator
from .engine import CausalEmergenceEngine
from .hmc import HamiltonianSampler
from .rules import EmergenceRules
from .topology import PersistentHomologyPerceiver

__all__ = [
    "EmergenceRules",
    "PersistentHomologyPerceiver",
    "CausalInferenceEngine",
    "DifferentialGenerator",
    "HamiltonianSampler",
    "ChaoticAssociativeMemory",
    "CausalEmergenceEngine",
]
