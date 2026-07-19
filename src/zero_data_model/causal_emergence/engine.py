# src/zero_data_model/causal_emergence/engine.py
"""Causal emergence engine — orchestrator (phases 1-4 implemented).

Phase 1 (foundation): wires up modules A (topology) and B
(causal_discovery) and exposes their facade methods.
Phase 2 (action): adds module C (differential) — damped least-action
trajectory generator with the ``generate_trajectory`` facade.
Phase 3 (cognition): adds modules D (hmc) and E (chaotic_memory) with
the ``sample_posterior`` and ``recall_memory`` facades.
Phase 4 (closure): adds the full ``emergence_cycle`` orchestration and
the ``compute_emergence_score`` heuristic.

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
    -posterior-memory loop. Phases 1-3 wire up modules A-E; phase 4 adds
    the full ``emergence_cycle`` orchestration and emergence-score heuristic.
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
    # Phase 4: full recursive emergence cycle
    # ------------------------------------------------------------------
    def emergence_cycle(self, observation: np.ndarray) -> dict:
        """Run the full recursive emergence loop on one observation.

        Observation shape (fix C3):
        - 2D ``(n_samples, n_features)`` with ``n_samples >= 2`` and
          ``n_features >= 2``. 1D input or insufficient data returns
          ``{'emergence_score': 0.0, 'reason': 'insufficient_data'}``.

        Steps:
        1. perception = perceive_topology(observation)
        2. causal_graph = discover_causal_dynamics(observation)
        3. counterfactual = generate_trajectory(start=mean, end=mean+delta)
           where ``delta = emergence_cycle_perturbation *
           rng.standard_normal(n_features)`` (fix M7: random direction)
        4. posterior = sample_posterior(Gaussian centered at mean)
        5. memory_response = recall_memory(mean)
        6. emergence_score = compute_emergence_score(...)

        Failure degradation (fix H6): if any module raises, that
        contribution is set to a zero-value placeholder and the failure
        is logged to ``warnings``. The emergence_score is still computed
        but will be lower.

        Returns dict with ``perception``, ``causal_graph``,
        ``counterfactual``, ``posterior``, ``memory``,
        ``emergence_score``, ``warnings``.
        """
        warnings: list[str] = []

        # Validate input shape (fix C3)
        if observation is None:
            return {"emergence_score": 0.0, "reason": "insufficient_data"}
        arr = np.ascontiguousarray(observation, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return {"emergence_score": 0.0, "reason": "insufficient_data"}

        n_samples, n_features = arr.shape
        mean = arr.mean(axis=0)

        # Step 1: perception (module A)
        perception: dict = {}
        try:
            perception = self.perceive_topology(arr)
        except Exception as exc:  # noqa: BLE001 — H6 degradation
            warnings.append(f"perception failed: {type(exc).__name__}: {exc}")
            perception = {
                "betti_numbers": [0, 0, 0],
                "persistence_entropy": 0.0,
                "euler_characteristic": 0,
                "n_points": int(n_samples),
                "max_eps": 0.0,
                "persistence_diagram": [],
            }

        # Step 2: causal discovery (module B)
        causal_graph: dict = {}
        try:
            causal_graph = self.discover_causal_dynamics(arr)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"causal_discovery failed: {type(exc).__name__}: {exc}")
            causal_graph = {
                "adjacency": np.zeros((n_features, n_features)),
                "edges": [],
                "n_edges": 0,
                "is_acyclic": True,
                "var_names": [f"x{i}" for i in range(n_features)],
                "method": "none",
            }

        # Step 3: counterfactual trajectory (module C, fix M7)
        delta = (
            self.rules.emergence_cycle_perturbation
            * self.rng.standard_normal(n_features)
        )
        counterfactual: dict = {}
        try:
            counterfactual = self.generate_trajectory(
                start_state=mean,
                end_state=mean + delta,
                n_steps=16,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"differential failed: {type(exc).__name__}: {exc}")
            counterfactual = {
                "trajectory": np.tile(mean, (17, 1)),
                "lagrangian": np.zeros(16),
                "action": 0.0,
                "converged": False,
                "iterations": 0,
            }

        # Step 4: posterior sampling (module D)
        posterior: dict = {}
        try:
            posterior = self.sample_posterior(
                log_prob_fn=lambda q: -0.5 * float(np.sum((q - mean) ** 2)),
                initial_position=mean,
                n_samples=self.rules.hmc_samples,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"posterior failed: {type(exc).__name__}: {exc}")
            posterior = {
                "samples": np.zeros((0, n_features)),
                "mean": np.zeros(n_features),
                "std": np.ones(n_features),  # large std -> low score term
                "accept_rate": 0.0,
                "ess": 0.0,
                "converged": False,
                "warnings": [str(exc)],
            }

        # Step 5: memory recall (module E)
        memory_response: dict = {}
        try:
            memory_response = self.recall_memory(mean, n_steps=50)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"memory failed: {type(exc).__name__}: {exc}")
            memory_response = {
                "label": None,
                "similarity": 0.0,
                "emerged": False,
                "trajectory": np.zeros((50, 3)),
                "converged": False,
                "divergence": 0.0,
            }

        # Step 6: emergence score (fix H4, H5)
        reference_action = (
            float(np.sum(mean * mean)) * 16 / 2.0  # n_steps=16 matches step 3
        )
        emergence_score = self.compute_emergence_score(
            perception=perception,
            causal_graph=causal_graph,
            counterfactual=counterfactual,
            posterior=posterior,
            memory_response=memory_response,
            reference_action=reference_action,
            dim=n_features,
        )

        return {
            "perception": perception,
            "causal_graph": causal_graph,
            "counterfactual": counterfactual,
            "posterior": posterior,
            "memory": memory_response,
            "emergence_score": float(emergence_score),
            "warnings": warnings,
        }

    def compute_emergence_score(
        self,
        perception: dict,
        causal_graph: dict,
        counterfactual: dict,
        posterior: dict,
        memory_response: dict,
        reference_action: float | None = None,
        dim: int | None = None,
    ) -> float:
        """Heuristic emergence score in [0, 1] (fix H4, H5).

        Composition (weights sum to 1.0, all terms clipped to [0, 1]):
        - 0.30 * persistence_entropy           (topological complexity)
        - 0.20 * (n_edges / max_edges)         (causal graph density)
        - 0.20 * memory.emerged                (chaotic walk indicator)
        - 0.15 * (1 - mean(posterior.std) / dim)  (tight posterior, fix H5)
        - 0.15 * (action / reference_action)   (counterfactual cost, fix H4)
        """
        # Term 1: persistence entropy (already in [0, 1])
        persistence_entropy = float(perception.get("persistence_entropy", 0.0))
        term1 = float(np.clip(persistence_entropy, 0.0, 1.0))

        # Term 2: causal graph density = n_edges / max_edges
        n_edges = int(causal_graph.get("n_edges", 0))
        # Need adjacency to count n_vars
        adj = causal_graph.get("adjacency", np.zeros((0, 0)))
        if isinstance(adj, np.ndarray) and adj.ndim == 2 and adj.shape[0] > 1:
            n_vars = adj.shape[0]
            max_edges = n_vars * (n_vars - 1) / 2.0
        else:
            max_edges = 1.0
        term2 = float(np.clip(n_edges / max_edges, 0.0, 1.0)) if max_edges > 0 else 0.0

        # Term 3: memory emergence (boolean -> 0.0 or 1.0)
        emerged = bool(memory_response.get("emerged", False))
        term3 = 1.0 if emerged else 0.0

        # Term 4: posterior tightness (fix H5: use mean(posterior.std))
        posterior_std = posterior.get("std", np.array([]))
        if isinstance(posterior_std, np.ndarray) and posterior_std.size > 0:
            mean_std = float(np.mean(posterior_std))
        else:
            mean_std = 0.0
        # dim defaults to posterior mean's size if not provided
        if dim is None or dim <= 0:
            posterior_mean = posterior.get("mean", np.array([]))
            dim = max(posterior_mean.size, 1) if isinstance(posterior_mean, np.ndarray) else 1
        # Tight posterior -> small std -> term near 1.0
        term4 = float(np.clip(1.0 - mean_std / dim, 0.0, 1.0))

        # Term 5: counterfactual action ratio (fix H4)
        action = float(counterfactual.get("action", 0.0))
        if reference_action is None or reference_action <= 0:
            # Fallback: use action itself as the reference (term saturates)
            reference_action = abs(action) if abs(action) > 1e-12 else 1.0
        term5 = float(np.clip(action / reference_action, 0.0, 1.0))

        # Weighted combination (weights sum to 1.0)
        score = (
            0.30 * term1
            + 0.20 * term2
            + 0.20 * term3
            + 0.15 * term4
            + 0.15 * term5
        )
        return float(np.clip(score, 0.0, 1.0))
