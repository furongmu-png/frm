# src/zero_data_model/capabilities/causal_advanced.py
"""Advanced causal/decision capabilities for the zero-data cognitive model.

Composes the core cognitive modules (active inference for free-energy
scoring) with ``CausalRules``. Provides pure-Python, rule-based POMDP
approximation, causal graph discovery, and intervention analysis -- no
external RL libraries (stable-baselines3 / pomdp-solve) are required.
"""

from __future__ import annotations

import numpy as np

from .rules import CausalRules


class POMDPApproximator:
    """Belief-state MDP approximation via point-based value iteration.

    Given transition, observation, and reward tensors, runs value iteration
    to convergence (or ``max_tree_depth`` iterations). Returns the optimal
    policy and value function.
    """

    GAMMA = 0.9  # discount factor

    def __init__(self, dim: int = 64, active_inference=None, rules: CausalRules | None = None):
        self.dim = dim
        self.rules = rules or CausalRules()
        # active_inference is accepted for API symmetry but not used in the
        # core solver; it is retained so future extensions can use it for
        # belief-state inference.
        self.active_inference = active_inference

    def solve(
        self,
        transitions: np.ndarray,
        observations: np.ndarray,
        rewards: np.ndarray,
    ) -> dict:
        """Solve a (PO)MDP via value iteration.

        ``transitions`` is ``(S, A, S)``, ``observations`` is ``(S, O)``,
        ``rewards`` is ``(S,)`` or ``(S, A)``.
        Returns ``{'policy', 'value', 'iterations', 'converged'}``.
        """
        T = np.asarray(transitions, dtype=float)
        if T.ndim != 3:
            return {
                "policy": np.zeros(0, dtype=int),
                "value": np.zeros(0),
                "iterations": 0,
                "converged": False,
            }
        n_states, n_actions, n_states2 = T.shape
        if n_states != n_states2:
            return {
                "policy": np.zeros(0, dtype=int),
                "value": np.zeros(0),
                "iterations": 0,
                "converged": False,
            }
        # rewards: (S,) broadcast to (S, A); or (S, A) used directly.
        R = np.asarray(rewards, dtype=float)
        if R.ndim == 1:
            R = np.tile(R.reshape(-1, 1), (1, n_actions))
        elif R.shape != (n_states, n_actions):
            return {
                "policy": np.zeros(n_states, dtype=int),
                "value": np.zeros(n_states),
                "iterations": 0,
                "converged": False,
            }
        # Observations accepted for API completeness (not used in this
        # MDP-level approximation; POMDP belief update would use them).
        _ = np.asarray(observations, dtype=float)
        # Value iteration.
        V = np.zeros(n_states, dtype=float)
        gamma = self.GAMMA
        converged = False
        iterations = 0
        # Bound iterations by a geometric estimate: log(tol / V_max) /
        # log(gamma). For typical gamma=0.9 and tol=1e-6 this is ~130
        # iterations; cap at 1000 to avoid pathological loops.
        max_iter = 1000
        for it in range(max_iter):
            iterations = it + 1
            # Q[s, a] = R[s, a] + gamma * sum_s' T[s, a, s'] * V[s']
            Q = R + gamma * np.einsum("sas,s->sa", T, V)
            new_V = Q.max(axis=1)
            diff = float(np.max(np.abs(new_V - V)))
            V = new_V
            if diff < 1e-6:
                converged = True
                break
        # Final policy: argmax_a Q.
        Q_final = R + gamma * np.einsum("sas,s->sa", T, V)
        policy = np.argmax(Q_final, axis=1).astype(int)
        return {
            "policy": policy,
            "value": V,
            "iterations": int(iterations),
            "converged": bool(converged),
        }


class CausalGraphBuilder:
    """Build a DAG from observational data (simplified PC algorithm).

    Computes the correlation matrix and adds an edge ``i -> j`` (for
    ``i < j``, as an orientation heuristic) whenever ``|corr[i, j]|``
    exceeds ``causal_significance``. Returns the adjacency matrix and
    edge list.
    """

    def __init__(self, dim: int = 64, rules: CausalRules | None = None):
        self.dim = dim
        self.rules = rules or CausalRules()

    def discover(
        self,
        data: np.ndarray,
        var_names: list[str] | None = None,
    ) -> dict:
        """Discover a causal graph from ``data``.

        ``data`` is ``(n_samples, n_vars)``. Returns
        ``{'adjacency', 'edges', 'n_edges', 'var_names'}``.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] < 2:
            return {
                "adjacency": np.zeros((0, 0)),
                "edges": [],
                "n_edges": 0,
                "var_names": var_names or [],
            }
        n_vars = data.shape[1]
        # Correlation matrix (suppress warnings on constant columns).
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(data, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        if corr.shape != (n_vars, n_vars):
            corr = np.zeros((n_vars, n_vars))
        threshold = float(self.rules.causal_significance)
        adjacency = np.zeros((n_vars, n_vars), dtype=float)
        edges: list[tuple[int, int]] = []
        names = var_names if var_names is not None else [str(i) for i in range(n_vars)]
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                if abs(float(corr[i, j])) > threshold and i < j:
                    adjacency[i, j] = 1.0
                    edges.append((i, j))
        return {
            "adjacency": adjacency,
            "edges": edges,
            "n_edges": int(len(edges)),
            "var_names": names,
        }


class InterventionAnalyzer:
    """do-calculus approximation: estimate intervention effects.

    Estimates the effect of ``do(X = v)`` on the population mean of all
    variables. Pre-intervention mean is the column mean of ``data``; post-
    intervention mean sets ``data[:, intervention_var] = v`` and recomputes.
    The surprisal of the change is scored by
    ``active_inference.compute_free_energy``.
    """

    def __init__(self, dim: int = 64, active_inference=None, rules: CausalRules | None = None):
        self.dim = dim
        self.rules = rules or CausalRules()
        if active_inference is None:
            from ..active_inference import ActiveInferenceEngine

            active_inference = ActiveInferenceEngine(state_dim=dim, obs_dim=dim)
        self.active_inference = active_inference

    def intervene(
        self,
        data: np.ndarray,
        intervention_var: int,
        intervention_value: float,
    ) -> dict:
        """Estimate the effect of ``do(X[intervention_var] = value)``.

        Returns ``{'pre_intervention_mean', 'post_intervention_mean',
        'effect', 'surprisal'}``.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.size == 0:
            return {
                "pre_intervention_mean": np.zeros(0),
                "post_intervention_mean": np.zeros(0),
                "effect": np.zeros(0),
                "surprisal": 0.0,
            }
        n_vars = data.shape[1]
        if not (0 <= intervention_var < n_vars):
            return {
                "pre_intervention_mean": np.zeros(n_vars),
                "post_intervention_mean": np.zeros(n_vars),
                "effect": np.zeros(n_vars),
                "surprisal": 0.0,
            }
        pre = np.mean(data, axis=0)
        # Post-intervention: replace the intervention column with the
        # constant value, then recompute means. Other variables' means
        # stay the same unless they correlate with the intervened var;
        # here we report only the marginal shift (column mean unchanged
        # for non-intervened vars).
        post_data = data.copy()
        post_data[:, intervention_var] = float(intervention_value)
        post = np.mean(post_data, axis=0)
        effect = post - pre
        # Surprisal of the effect vector.
        surprisal = float(self.active_inference.compute_free_energy(effect))
        if not np.isfinite(surprisal):
            surprisal = 0.0
        return {
            "pre_intervention_mean": pre,
            "post_intervention_mean": post,
            "effect": effect,
            "surprisal": float(surprisal),
        }
