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

    # Module A: persistent homology (fix C1: max_points lowered to 16)
    topology_max_points: int = 16
    topology_max_dim: int = 2
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

    # Engine
    emergence_cycle_perturbation: float = 0.1  # counterfactual delta scale
    # fix M8: no `seed` field — determinism controlled solely by `rng` arg

    def __post_init__(self):
        super().__init__()
        self.rules = {
            "topology_max_points": self.topology_max_points,
            "topology_max_dim": self.topology_max_dim,
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
            "emergence_cycle_perturbation": self.emergence_cycle_perturbation,
        }
