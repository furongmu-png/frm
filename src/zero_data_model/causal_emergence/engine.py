# src/zero_data_model/causal_emergence/engine.py
"""Causal emergence engine — orchestrator (phases 1-3 implemented).

Phase 1 (foundation): wires up modules A (topology) and B
(causal_discovery) and exposes their facade methods.
Phase 2 (action): adds module C (differential) — damped least-action
trajectory generator with the ``generate_trajectory`` facade.
Phase 3 (cognition): adds modules D (hmc) and E (chaotic_memory) with
the ``sample_posterior`` and ``recall_memory`` facades.
Phase 4 will add the full ``emergence_cycle``.

See ``docs/superpowers/specs/2026-07-19-causal-emergence-engine-design.md``
for the full design specification.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .causal_discovery import CausalInferenceEngine
from .chaotic_memory import ChaoticAssociativeMemory
from .differential import DifferentialGenerator
from .hmc import HamiltonianSampler
from .rules import EmergenceRules
from .topology import PersistentHomologyPerceiver


class CausalEmergenceEngine:
    """Causal emergence engine orchestrator.

    Composes five modules into a recursive perception-causal-counterfactual
    -posterior-memory loop. Phases 1-3 wire up modules A-E; phase 4 will
    add the full ``emergence_cycle``.
    """

    def __init__(
        self,
        dim: int = 64,
        active_inference: Any | None = None,
        math_universe: Any | None = None,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.dim = dim
        self.rules = rules or EmergenceRules()
        self.active_inference = active_inference
        self.math_universe = math_universe
        self.rng = rng or np.random.default_rng()

        # Module A: topology
        self.topology = PersistentHomologyPerceiver(
            dim=dim,
            math_universe=math_universe,
            rules=self.rules,
            rng=self.rng,
        )
        # Module B: causal discovery
        self.causal = CausalInferenceEngine(
            dim=dim,
            active_inference=active_inference,
            rules=self.rules,
            rng=self.rng,
        )
        # Module C: differential generator (phase 2)
        self.differential = DifferentialGenerator(
            dim=dim,
            rules=self.rules,
            rng=self.rng,
        )
        # Module D: HMC sampler (phase 3)
        self.hmc = HamiltonianSampler(
            dim=dim,
            rules=self.rules,
            rng=self.rng,
        )
        # Module E: chaotic associative memory (phase 3)
        self.memory = ChaoticAssociativeMemory(
            dim=dim,
            rules=self.rules,
            rng=self.rng,
        )

    # ------------------------------------------------------------------
    # Facade methods (phase 1)
    # ------------------------------------------------------------------
    def perceive_topology(self, data: np.ndarray, max_dim: int | None = None) -> dict:
        """Module A facade: perceive topological invariants of ``data``."""
        return self.topology.perceive(data, max_dim=max_dim)

    def discover_causal_dynamics(
        self,
        data: np.ndarray,
        var_names: list[str] | None = None,
        method: str | None = None,
    ) -> dict:
        """Module B facade: discover causal DAG from observational data."""
        return self.causal.discover(data, var_names=var_names, method=method)

    def intervene(
        self,
        adjacency: np.ndarray,
        data: np.ndarray,
        intervention_var: int,
        intervention_value: float,
    ) -> dict:
        """Module B facade: estimate intervention effect (do-calculus)."""
        return self.causal.intervene(
            adjacency, data, intervention_var, intervention_value
        )

    def counterfactual(
        self,
        adjacency: np.ndarray,
        observed: np.ndarray,
        intervention_var: int,
        intervention_value: float,
    ) -> dict:
        """Module B facade: counterfactual query."""
        return self.causal.counterfactual(
            adjacency, observed, intervention_var, intervention_value
        )

    # ------------------------------------------------------------------
    # Facade methods (phase 2)
    # ------------------------------------------------------------------
    def generate_trajectory(
        self,
        start_state: np.ndarray,
        end_state: np.ndarray,
        n_steps: int = 32,
        constraints: dict | None = None,
    ) -> dict:
        """Module C facade: damped least-action trajectory generator.

        Solves a discretized Euler-Lagrange boundary value problem with
        damping (spec §5). Returns ``trajectory`` of shape
        ``(n_steps + 1, dim)`` connecting ``start_state`` to ``end_state``,
        plus per-step ``lagrangian``, total ``action``, ``converged`` flag
        and iteration count.
        """
        return self.differential.generate(
            start_state=start_state,
            end_state=end_state,
            n_steps=n_steps,
            constraints=constraints,
        )

    # ------------------------------------------------------------------
    # Facade methods (phase 3)
    # ------------------------------------------------------------------
    def sample_posterior(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        initial_position: np.ndarray,
        n_samples: int | None = None,
        step_size: float | None = None,
        n_leapfrog: int | None = None,
        grad_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> dict:
        """Module D facade: HMC posterior sampling.

        Single-chain ESS estimates have large uncertainty; recommend
        multi-chain runs with Gelman-Rubin R-hat for production use.

        Returns dict with ``samples``, ``mean``, ``std``,
        ``accept_rate``, ``ess``, ``converged``, ``warnings``.
        """
        if n_samples is None:
            n_samples = self.rules.hmc_samples
        return self.hmc.sample(
            log_prob_fn=log_prob_fn,
            initial_position=initial_position,
            n_samples=n_samples,
            step_size=step_size,
            n_leapfrog=n_leapfrog,
            grad_fn=grad_fn,
        )

    def recall_memory(
        self,
        query: np.ndarray,
        n_steps: int = 100,
    ) -> dict:
        """Module E facade: chaotic associative memory recall.

        Returns dict with ``label``, ``similarity``, ``emerged``,
        ``trajectory``, ``converged``, ``divergence``. Empty memory
        returns ``label=None, emerged=True``.
        """
        return self.memory.recall(query, n_steps=n_steps)

    # ------------------------------------------------------------------
    # Phase 4 placeholder
    # ------------------------------------------------------------------
    def emergence_cycle(self, observation: np.ndarray) -> dict:
        """Full recursive loop (phase 4). Not yet implemented."""
        raise NotImplementedError("emergence_cycle (phase 4) not yet implemented")
