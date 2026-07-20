# src/zero_data_model/causal_emergence/rules.py
"""Configuration rules for the causal emergence engine.

These are minimal inductive biases (algorithm parameters, thresholds,
defaults) — not learned from external data. They follow the same pattern
as the existing ``capabilities/rules.py``: a ``DomainRules`` subclass
decorated with ``@dataclass``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..capabilities.rules import DomainRules


@dataclass
class EmergenceRules(DomainRules):
    """Priors for the causal emergence engine.

    See spec §9 for the full field-by-field rationale and the v2 review
    notes (docs/superpowers/reviews/2026-07-19-causal-emergence-engine-
    spec-v2-verification.md) for the fix history.
    """

    # Module A: persistent homology (fix C1: max_points lowered to 16).
    #
    # Phase D topology optimisation: ``topology_max_dim`` default lowered
    # from 2 to 1. The boundary-matrix column reduction is
    # O(n_cols^3 / 64) where n_cols = sum_d C(n_points, d+1) up to
    # max_dim+1. For max_dim=2 this is O(n^9) in practice: n=16 finishes
    # in ~0.3s, n=24 in ~9s, n=32 in ~100s, n=40 in ~10min. Defaulting to
    # max_dim=1 (computes betti_0 connectivity + betti_1 loops, no voids)
    # keeps the default path fast (O(n^6)). Users who explicitly opt into
    # max_dim >= 2 hit ``topology_max_points_high_dim`` (hard cap on
    # n_points for high-dim computation) and ``topology_max_simplices``
    # (hard cap on total simplex count) — both raise ``ValueError`` when
    # exceeded, with a helpful message pointing at the rule fields to
    # override. This converts the silent ~100s hang into a loud,
    # actionable error.
    topology_max_points: int = 16
    topology_max_dim: int = 1
    # Hard cap on n_points when max_dim >= 2. With max_dim=2 the column
    # reduction is O(n^9); n=16 is ~0.3s, n=20 is ~2s, n=24 is ~9s. 16 is
    # the largest n that stays under 1s on commodity hardware. Override
    # ONLY if you understand the O(n^9) cost.
    topology_max_points_high_dim: int = 16
    # Hard cap on total simplex count (sum over dims 0..max_dim+1 of
    # C(n_points, d+1)). The column reduction is O(n_cols^3 / 64) word
    # ops; 5000 columns = ~2e9 ops ~= 30s on commodity hardware. This is
    # the ultimate guard against the combinatorial explosion regardless
    # of which dim/points combination triggers it.
    topology_max_simplices: int = 5000
    topology_eps_steps: int = 50

    # Module B: causal discovery
    causal_method: str = "pc"              # 'pc' | 'lingam' | 'correlation'
    causal_significance: float = 0.05      # Fisher z test p-value threshold
    causal_max_cond_set: int = 3           # PC conditioning set size cap
    causal_max_vars_lingam: int = 8        # LiNGAM variable count cap

    # Module C: differential (fix H1: damped least-action)
    differential_max_iter: int = 100       # Gauss-Seidel iteration cap
    differential_tol: float = 1e-6         # convergence tolerance
    differential_dt: float = 0.01          # time step
    differential_lambda: float = 1.0       # attraction strength (new)
    differential_gamma: float = 0.5        # damping coefficient (new)
    # Phase 5 — constraints avoidance (spec §5.5):
    # obstacle_margin defines the safety radius around each obstacle center;
    # points within this radius are projected to the boundary.
    differential_obstacle_margin: float = 1e-3
    # penalty coefficient for soft obstacle cost (added to action).
    differential_obstacle_penalty: float = 1e6

    # Module D: HMC
    hmc_step_size: float = 0.1
    hmc_n_leapfrog: int = 10
    hmc_samples: int = 100
    hmc_target_accept: float = 0.65        # Beskos et al. optimal
    hmc_finite_diff_h: float = 1e-5        # central diff step (fix M3)

    # Module E: chaotic memory (fix C4, H7, M6)
    chaotic_memory_capacity: int = 32
    chaotic_lorenz_sigma: float = 10.0
    chaotic_lorenz_rho: float = 28.0
    chaotic_lorenz_beta: float = 8.0 / 3.0
    chaotic_alpha: float = 0.1              # attraction strength (fix H7)
    chaotic_sigma_q: float = 1.0            # Gaussian kernel bandwidth (new)
    chaotic_perturbation: float = 0.01      # emergence detection perturbation
    chaotic_divergence_threshold: float = 10.0  # emergence threshold (fix M6)
    chaotic_dt: float = 0.01                # Lorenz RK4 timestep (fix NEW-M4)
    # fix R2-NEW-M2: settled-tolerance was hardcoded 5.0 calibrated to
    # default dt=0.01. Now rule-ified; default preserves old behavior, and
    # scales naturally if user changes chaotic_dt (e.g., chaotic_dt=0.001
    # + chaotic_settled_tolerance=5.0 -> tighter absolute basin check,
    # which is the desired behavior for finer integration).
    chaotic_settled_tolerance: float = 5.0

    # Engine
    emergence_cycle_perturbation: float = 0.1  # counterfactual delta scale
    # fix M8: no `seed` field — determinism controlled solely by `rng` arg

    def __post_init__(self):
        super().__init__()
        self.rules = {
            "topology_max_points": self.topology_max_points,
            "topology_max_dim": self.topology_max_dim,
            "topology_max_points_high_dim": self.topology_max_points_high_dim,
            "topology_max_simplices": self.topology_max_simplices,
            "topology_eps_steps": self.topology_eps_steps,
            "causal_method": self.causal_method,
            "causal_significance": self.causal_significance,
            "causal_max_cond_set": self.causal_max_cond_set,
            "causal_max_vars_lingam": self.causal_max_vars_lingam,
            "differential_max_iter": self.differential_max_iter,
            "differential_tol": self.differential_tol,
            "differential_dt": self.differential_dt,
            "differential_lambda": self.differential_lambda,
            "differential_gamma": self.differential_gamma,
            "differential_obstacle_margin": self.differential_obstacle_margin,
            "differential_obstacle_penalty": self.differential_obstacle_penalty,
            "hmc_step_size": self.hmc_step_size,
            "hmc_n_leapfrog": self.hmc_n_leapfrog,
            "hmc_samples": self.hmc_samples,
            "hmc_target_accept": self.hmc_target_accept,
            "hmc_finite_diff_h": self.hmc_finite_diff_h,
            "chaotic_memory_capacity": self.chaotic_memory_capacity,
            "chaotic_lorenz_sigma": self.chaotic_lorenz_sigma,
            "chaotic_lorenz_rho": self.chaotic_lorenz_rho,
            "chaotic_lorenz_beta": self.chaotic_lorenz_beta,
            "chaotic_alpha": self.chaotic_alpha,
            "chaotic_sigma_q": self.chaotic_sigma_q,
            "chaotic_perturbation": self.chaotic_perturbation,
            "chaotic_divergence_threshold": self.chaotic_divergence_threshold,
            "chaotic_dt": self.chaotic_dt,
            "chaotic_settled_tolerance": self.chaotic_settled_tolerance,
            "emergence_cycle_perturbation": self.emergence_cycle_perturbation,
        }
