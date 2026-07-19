# src/zero_data_model/causal_emergence/chaotic_memory.py
"""Module E: ChaoticAssociativeMemory.

Stores patterns as weak attractor basins of a modified Lorenz system.
Recall converges to the nearest stored pattern; queries on the basin
boundary produce a chaotic walk registered as "emergence" / "insight".

Algorithm (spec §7, fixes C4 / H7 / M6):
- Encoding: pattern (dim,) -> (target_y, target_z) by averaging the
  first and second halves of the pattern. Stored alongside the original
  pattern and label as ``(pattern, label, target_y, target_z)``.
- Modified Lorenz system with weak attraction toward stored targets:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y - alpha * sum_i w_i * (y - target_y_i)
    dz/dt = x * y - beta * z  - alpha * sum_i w_i * (z - target_z_i)
  where ``w_i = exp(-||query - pattern_i||^2 / (2 * sigma_q^2))`` is a
  Gaussian-kernel affinity weight.
- Recall: RK4 integrate the system for ``n_steps`` steps starting from
  ``(0, query_y, query_z)`` where ``(query_y, query_z)`` is the encoded
  query. After integration, return the nearest stored pattern (by L2
  distance to the original query).
- Emergence detection (fix M6): perturb the query by Gaussian noise,
  re-integrate, and measure trajectory divergence. If
  ``divergence > chaotic_divergence_threshold`` (default 10.0), flag
  ``emerged = True`` (the system amplified the initial perturbation,
  indicating the query is near a basin boundary).
- Capacity (fix C4): bounded by ``rules.chaotic_memory_capacity``
  (default 32). FIFO eviction when exceeded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rules import EmergenceRules


class ChaoticAssociativeMemory:
    """Lorenz-basin associative memory with chaotic emergence detection."""

    def __init__(
        self,
        dim: int = 64,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.dim = dim
        self.rules = rules or EmergenceRules()
        self.rng = rng or np.random.default_rng()
        self._patterns: list[np.ndarray] = []
        self._labels: list[Any] = []
        self._targets: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store(self, pattern: np.ndarray, label: int | str) -> dict:
        """Store ``pattern`` as a new attractor basin.

        Returns dict with ``label``, ``n_stored``, ``capacity``,
        ``evicted`` (list of evicted labels, FIFO when capacity exceeded).
        """
        p = self._sanitize(pattern)
        if p.shape[0] != self.dim and self.dim != 64:
            # Allow caller's pattern to override stored dim on first store
            # if the default 64 doesn't match; otherwise require match.
            pass
        target_y, target_z = self._encode(p)
        self._patterns.append(p)
        self._labels.append(label)
        self._targets.append((target_y, target_z))

        evicted: list[Any] = []
        cap = self.rules.chaotic_memory_capacity
        while len(self._patterns) > cap:
            evicted.append(self._labels.pop(0))
            self._patterns.pop(0)
            self._targets.pop(0)

        return {
            "label": label,
            "n_stored": len(self._patterns),
            "capacity": cap,
            "evicted": evicted,
        }

    def recall(self, query: np.ndarray, n_steps: int = 100) -> dict:
        """Recall the nearest stored pattern to ``query``.

        Returns dict with ``label``, ``similarity``, ``emerged``,
        ``trajectory``, ``converged``, ``divergence``.

        Empty memory -> ``label=None, similarity=0.0, emerged=True,
        trajectory=zeros, converged=False, divergence=0.0``.
        Query with NaN -> ``label=None, emerged=False, converged=False``.
        """
        # Empty memory case
        if not self._patterns:
            return {
                "label": None,
                "similarity": 0.0,
                "emerged": True,
                "trajectory": np.zeros((n_steps, 3)),
                "converged": False,
                "divergence": 0.0,
            }

        # NaN guard on raw query (before _sanitize replaces NaN with 0).
        raw = np.ascontiguousarray(query, dtype=float).flatten()
        if raw.size == 0 or not np.all(np.isfinite(raw)):
            return {
                "label": None,
                "similarity": 0.0,
                "emerged": False,
                "trajectory": np.zeros((n_steps, 3)),
                "converged": False,
                "divergence": 0.0,
            }

        q = self._sanitize(query)

        # Encode query -> Lorenz initial state (x0, y0, z0)
        qy, qz = self._encode(q)
        state0 = np.array([0.0, qy, qz])

        # Integrate Lorenz trajectory
        dt = 0.01
        trajectory = self._integrate_lorenz(state0, q, n_steps, dt)

        # Find nearest stored pattern (by L2 distance to original query)
        distances = [float(np.linalg.norm(q - p)) for p in self._patterns]
        nearest_idx = int(np.argmin(distances))
        nearest_dist = distances[nearest_idx]
        nearest_label = self._labels[nearest_idx]
        nearest_pattern = self._patterns[nearest_idx]

        # Similarity in [0, 1]: 1 / (1 + dist^2)
        similarity = 1.0 / (1.0 + nearest_dist * nearest_dist)

        # Emergence detection (fix M6): perturb query, re-integrate,
        # measure divergence.
        perturbation = self.rules.chaotic_perturbation
        noise = self.rng.standard_normal(q.shape[0]) * perturbation
        q_pert = q + noise
        # Re-encode perturbed query
        qy_p, qz_p = self._encode(q_pert)
        state_pert = np.array([0.0, qy_p, qz_p])
        trajectory_pert = self._integrate_lorenz(state_pert, q_pert, n_steps, dt)

        # Divergence = ||final - final_pert|| / (||q - q_pert|| + eps)
        delta_init = float(np.linalg.norm(q - q_pert))
        delta_final = float(
            np.linalg.norm(trajectory[-1] - trajectory_pert[-1])
        )
        divergence = delta_final / (delta_init + 1e-12)

        emerged = divergence > self.rules.chaotic_divergence_threshold

        # Convergence: did the trajectory settle near the nearest stored
        # pattern's target (y, z)? Use the trajectory's final state.
        target_y, target_z = self._targets[nearest_idx]
        final_state = trajectory[-1]
        target_state = np.array([0.0, target_y, target_z])
        # Settled if final state's (y, z) is within 5.0 of target (Lorenz
        # state space is large).
        settled = float(
            np.linalg.norm(final_state[1:] - target_state[1:])
        ) < 5.0

        # NaN guard on outputs
        trajectory = np.nan_to_num(trajectory, nan=0.0, posinf=0.0, neginf=0.0)
        similarity = float(np.nan_to_num(similarity, nan=0.0))
        divergence = float(np.nan_to_num(divergence, nan=0.0))

        return {
            "label": nearest_label,
            "similarity": similarity,
            "emerged": bool(emerged),
            "trajectory": trajectory,
            "converged": bool(settled),
            "divergence": divergence,
            "nearest_pattern": nearest_pattern,
        }

    def clear(self) -> None:
        """Remove all stored patterns."""
        self._patterns.clear()
        self._labels.clear()
        self._targets.clear()

    @property
    def size(self) -> int:
        """Number of stored patterns."""
        return len(self._patterns)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sanitize(self, x: Any) -> np.ndarray:
        arr = np.ascontiguousarray(x, dtype=float).flatten()
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _encode(self, pattern: np.ndarray) -> tuple[float, float]:
        """Encode a dim-D pattern as (target_y, target_z).

        First half mean -> target_y; second half mean -> target_z.
        Pattern length is split at dim//2 (fix spec §7.4: odd dim is OK).
        """
        d = pattern.shape[0]
        half = d // 2
        if half == 0:
            target_y = float(pattern.mean()) if d > 0 else 0.0
            target_z = target_y
        else:
            target_y = float(pattern[:half].mean())
            target_z = float(pattern[half:].mean()) if d > half else target_y
        return target_y, target_z

    def _integrate_lorenz(
        self,
        state0: np.ndarray,
        query: np.ndarray,
        n_steps: int,
        dt: float,
    ) -> np.ndarray:
        """RK4 integration of the modified Lorenz system.

        The modified system has weak attraction toward stored targets,
        weighted by Gaussian affinity to the query.
        """
        sigma = self.rules.chaotic_lorenz_sigma
        rho = self.rules.chaotic_lorenz_rho
        beta = self.rules.chaotic_lorenz_beta
        alpha = self.rules.chaotic_alpha
        sigma_q = self.rules.chaotic_sigma_q

        # Precompute affinity weights w_i for the query (constant over
        # the integration, since query is fixed).
        patterns = self._patterns
        targets = self._targets
        if patterns:
            # Stack patterns (n_stored, dim)
            P = np.stack(patterns)
            t_y = np.array([t[0] for t in targets])
            t_z = np.array([t[1] for t in targets])
            # Affinity weights (Gaussian kernel)
            diffs = P - query[None, :]
            sq_dists = np.sum(diffs * diffs, axis=1)
            w = np.exp(-sq_dists / (2.0 * sigma_q * sigma_q))
            # Avoid all-zero weights (e.g., when patterns are far away);
            # otherwise normalize so alpha * sum_i w_i stays bounded.
            w_sum = w.sum()
            w = np.ones_like(w) / w.size if w_sum < 1e-12 else w / w_sum
            # Attraction pull: alpha * sum_i w_i * (state - target_i)
            # For y: alpha * sum_i w_i * (y - target_y_i)
            # For z: alpha * sum_i w_i * (z - target_z_i)
            # The system is:
            #   dy/dt = x*(rho - z) - y - alpha * sum_i w_i * (y - t_y_i)
            #   dz/dt = x*y - beta*z  - alpha * sum_i w_i * (z - t_z_i)
        else:
            w = np.zeros(0)
            t_y = np.zeros(0)
            t_z = np.zeros(0)

        def f(state: np.ndarray) -> np.ndarray:
            x, y, z = state
            dx = sigma * (y - x)
            if patterns:
                attract_y = float(np.sum(w * (y - t_y)))
                attract_z = float(np.sum(w * (z - t_z)))
            else:
                attract_y = 0.0
                attract_z = 0.0
            dy = x * (rho - z) - y - alpha * attract_y
            dz = x * y - beta * z - alpha * attract_z
            return np.array([dx, dy, dz])

        traj = np.zeros((n_steps, 3))
        state = state0.copy()
        for k in range(n_steps):
            traj[k] = state
            # RK4
            k1 = f(state)
            k2 = f(state + 0.5 * dt * k1)
            k3 = f(state + 0.5 * dt * k2)
            k4 = f(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # Guard against divergence (state can blow up)
            if not np.all(np.isfinite(state)):
                state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        return traj
