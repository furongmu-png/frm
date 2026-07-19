# src/zero_data_model/causal_emergence/differential.py
"""Module C: DifferentialGenerator.

Generates continuous, physically-realistic trajectories by solving a
discretized Euler-Lagrange boundary value problem with damping.

Fix H1: the original spec's Lagrangian ``L = 0.5*||q_dot||^2 -
0.5*||q - target||^2`` is a harmonic oscillator whose solution
oscillates rather than monotonically converging. This implementation
uses a damped oscillator instead: "second time derivative + first
time derivative (damping) + elastic restoring force = 0". When
``lambda = gamma = 0`` the trajectory degenerates to a linear
interpolation (shortest path).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rules import EmergenceRules


class DifferentialGenerator:
    """Damped least-action trajectory generator.

    Constructor follows the capability-domain pattern: plain Python class,
    accepts ``dim``, ``rules``, ``rng``.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.dim = dim
        self.rules = rules or EmergenceRules()
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        start_state: np.ndarray,
        end_state: np.ndarray,
        n_steps: int = 32,
        constraints: dict | None = None,  # placeholder (fix L2)
    ) -> dict:
        """Solve the damped least-action boundary value problem.

        Parameters
        ----------
        start_state, end_state
            Boundary states. Must have the same shape. Broadcast to 1D.
        n_steps
            Number of interior steps. ``trajectory`` will have
            ``n_steps + 1`` rows.
        constraints
            Placeholder for future obstacle avoidance (fix L2). Not
            implemented in this phase.

        Returns
        -------
        dict with ``trajectory``, ``lagrangian``, ``action``,
        ``converged``, ``iterations``.
        """
        # Sanitize inputs
        start = self._sanitize(start_state)
        end = self._sanitize(end_state)
        if start.shape != end.shape:
            raise ValueError(
                f"start_state shape {start.shape} != end_state shape {end.shape}"
            )
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            raise ValueError("start_state and end_state must be finite")

        dim = start.shape[0]

        # Edge case: n_steps == 0 -> single-point trajectory
        if n_steps <= 0:
            return {
                "trajectory": start.reshape(1, -1).copy(),
                "lagrangian": np.zeros(0),
                "action": 0.0,
                "converged": True,
                "iterations": 0,
            }

        # Edge case: start == end -> constant trajectory
        if np.allclose(start, end):
            traj = np.tile(start, (n_steps + 1, 1))
            return {
                "trajectory": traj,
                "lagrangian": np.zeros(n_steps),
                "action": 0.0,
                "converged": True,
                "iterations": 0,
            }

        # Initialize with linear interpolation
        q = np.zeros((n_steps + 1, dim))
        for k in range(n_steps + 1):
            t = k / n_steps
            q[k] = (1 - t) * start + t * end

        # Gauss-Seidel relaxation of the damped oscillator:
        # (q[k+1] - 2*q[k] + q[k-1]) / dt^2
        #   + gamma * (q[k+1] - q[k-1]) / (2*dt)
        #   + lambda * (q[k] - target) = 0
        #
        # Solve for q[k] (interior point) at each iteration.
        # Boundary points q[0] = start, q[n_steps] = end are fixed.
        dt = self.rules.differential_dt
        lam = self.rules.differential_lambda
        gamma = self.rules.differential_gamma
        tol = self.rules.differential_tol
        max_iter = self.rules.differential_max_iter
        target = end  # fix H2: target = end_state

        # Coefficients in the discretized equation (per interior k):
        # a * q[k-1] + b * q[k] + c * q[k+1] = -lam * target
        # where:
        #   a = 1/dt^2 - gamma/(2*dt)
        #   b = -2/dt^2 - lam
        #   c = 1/dt^2 + gamma/(2*dt)
        a = 1.0 / (dt * dt) - gamma / (2.0 * dt)
        b = -2.0 / (dt * dt) - lam
        c = 1.0 / (dt * dt) + gamma / (2.0 * dt)
        rhs = -lam * target

        iterations = 0
        converged = False
        for it in range(max_iter):
            iterations = it + 1
            max_delta = 0.0
            for k in range(1, n_steps):
                # Solve for q[k]: b * q[k] = rhs - a * q[k-1] - c * q[k+1]
                new_q = (rhs - a * q[k - 1] - c * q[k + 1]) / b
                delta = np.max(np.abs(new_q - q[k]))
                if delta > max_delta:
                    max_delta = delta
                q[k] = new_q
            if max_delta < tol:
                converged = True
                break

        # Compute per-step Lagrangian and total action.
        # L[k] = 0.5 * ||q_dot[k]||^2 - 0.5 * lambda * ||q[k] - target||^2
        # q_dot[k] = (q[k+1] - q[k]) / dt  (forward difference)
        lagrangian = np.zeros(n_steps)
        for k in range(n_steps):
            q_dot = (q[k + 1] - q[k]) / dt
            kinetic = 0.5 * float(np.sum(q_dot * q_dot))
            potential = 0.5 * lam * float(np.sum((q[k] - target) ** 2))
            lagrangian[k] = kinetic - potential
        action = float(np.sum(lagrangian) * dt)

        # NaN guard on outputs
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        lagrangian = np.nan_to_num(lagrangian, nan=0.0, posinf=0.0, neginf=0.0)
        action = float(np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0))

        return {
            "trajectory": q,
            "lagrangian": lagrangian,
            "action": action,
            "converged": converged,
            "iterations": int(iterations),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize(x: Any) -> np.ndarray:
        """Coerce input to a 1D float ndarray, NaN/Inf guarded."""
        arr = np.ascontiguousarray(x, dtype=float).flatten()
        return arr
