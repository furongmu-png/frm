# src/zero_data_model/causal_emergence/__init__.py
"""Causal emergence engine package.

Composes five modules (persistent homology, causal discovery, differential
generator, Hamiltonian sampler, chaotic memory) into a single recursive
loop. See ``docs/superpowers/specs/2026-07-19-causal-emergence-engine-design.md``
for the full design specification.

Phase 1 (foundation): modules A (topology) and B (causal_discovery) are
implemented. Phases 2-4 will add modules C-E and the full emergence cycle.
"""

from __future__ import annotations

from .causal_discovery import CausalInferenceEngine
from .engine import CausalEmergenceEngine
from .rules import EmergenceRules
from .topology import PersistentHomologyPerceiver

__all__ = [
    "EmergenceRules",
    "PersistentHomologyPerceiver",
    "CausalInferenceEngine",
    "CausalEmergenceEngine",
]
