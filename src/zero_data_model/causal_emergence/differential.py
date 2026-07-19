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
        constraints: dict | None = None,  # phase 5: obstacle avoidance
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
            Optional obstacle-avoidance specification (phase 5, spec §5.5).
            Recognized keys:

            - ``"obstacles"``: ``list[list[float]]`` or ``np.ndarray`` of
              shape ``(K, dim)`` — each row is an obstacle center in state
              space. Interior trajectory points within
              ``constraints["margin"]`` (or ``rules.differential_obstacle_margin``
              if absent) of any obstacle are projected to the safety
              boundary after each Gauss-Seidel sweep.
            - ``"margin"``: ``float | None`` — per-call override of
              ``rules.differential_obstacle_margin``.
            - ``"type"``: ``"soft"`` (default) — project interior points
              to obstacle boundary; hard projection is currently
              equivalent and accepted for forward compatibility.

            Empty dict / ``None`` preserves the original no-obstacle
            behavior (fix L2 backward compatibility).

        Returns
        -------
        dict with ``trajectory``, ``lagrangian``, ``action``,
        ``converged``, ``iterations``, and (when obstacles active)
        ``obstacle_violations``: int count of points that were projected.
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

        # Parse constraints (phase 5: obstacle avoidance).
        obstacles, obstacle_margin = self._parse_constraints(
            constraints, dim
        )

        # Edge case: n_steps == 0 -> single-point trajectory
        if n_steps <= 0:
            result = {
                "trajectory": start.reshape(1, -1).copy(),
                "lagrangian": np.zeros(0),
                "action": 0.0,
                "converged": True,
                "iterations": 0,
            }
            if obstacles is not None:
                result["obstacle_violations"] = 0
            return result

        # Edge case: start == end -> constant trajectory
        if np.allclose(start, end):
            traj = np.tile(start, (n_steps + 1, 1))
            result = {
                "trajectory": traj,
                "lagrangian": np.zeros(n_steps),
                "action": 0.0,
                "converged": True,
                "iterations": 0,
            }
            if obstacles is not None:
                # Boundary points are not projected; count any interior
                # point that lies within margin of an obstacle.
                violations = 0
                if obstacles.shape[0] > 0:
                    for k in range(1, n_steps):
                        for obs in obstacles:
                            d = float(np.linalg.norm(traj[k] - obs))
                            if d < obstacle_margin:
                                violations += 1
                result["obstacle_violations"] = int(violations)
            return result

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
        obstacle_violations = 0
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
            # Phase 5: project interior points out of obstacle margins
            # after each Gauss-Seidel sweep. Boundary points are fixed.
            if obstacles is not None and obstacles.shape[0] > 0:
                for k in range(1, n_steps):
                    for obs in obstacles:
                        diff = q[k] - obs
                        d = float(np.linalg.norm(diff))
                        if d < obstacle_margin:
                            # Project to the safety boundary along
                            # the radial direction. Use a tiny epsilon
                            # to avoid division-by-zero when q[k] == obs.
                            scale = obstacle_margin / (d + 1e-12)
                            q[k] = obs + diff * scale
                            obstacle_violations += 1
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

        # Phase 5: add soft obstacle penalty to action so callers can
        # detect that avoidance was active. The penalty is proportional
        # to the number of projected points, scaled by
        # rules.differential_obstacle_penalty * dt (so it has the same
        # units as action = sum(L * dt)).
        if obstacles is not None and obstacle_violations > 0:
            penalty = (
                self.rules.differential_obstacle_penalty
                * obstacle_violations
                * dt
            )
            action += penalty

        # NaN guard on outputs
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        lagrangian = np.nan_to_num(lagrangian, nan=0.0, posinf=0.0, neginf=0.0)
        action = float(np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0))

        result = {
            "trajectory": q,
            "lagrangian": lagrangian,
            "action": action,
            "converged": converged,
            "iterations": int(iterations),
        }
        if obstacles is not None:
            result["obstacle_violations"] = int(obstacle_violations)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _parse_constraints(
        self, constraints: dict | None, dim: int
    ) -> tuple[np.ndarray | None, float]:
        """Parse the ``constraints`` dict into ``(obstacles, margin)``.

        Returns ``(None, margin)`` when no obstacles are specified, which
        signals the caller to skip the projection step entirely. Empty
        dict and ``None`` are both accepted (fix L2 backward compat).

        Raises ``ValueError`` on malformed input so callers get a clear
        contract instead of a silent no-op or opaque downstream error.
        """
        if constraints is None or not constraints:
            return None, self.rules.differential_obstacle_margin

        if not isinstance(constraints, dict):
            raise ValueError(
                f"constraints must be a dict or None, got {type(constraints).__name__}"
            )

        # Recognized keys; unknown keys are ignored (forward compat).
        obstacles_raw = constraints.get("obstacles")
        margin = constraints.get(
            "margin", self.rules.differential_obstacle_margin
        )
        ctype = constraints.get("type", "soft")

        if ctype not in ("soft", "hard"):
            raise ValueError(
                f"constraints['type'] must be 'soft' or 'hard', got {ctype!r}"
            )
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError(
                f"constraints['margin'] must be positive finite, got {margin}"
            )

        if obstacles_raw is None:
            # No obstacles -> empty dict / {"margin": ...} still
            # produces no projection.
            return None, float(margin)

        obstacles = np.ascontiguousarray(obstacles_raw, dtype=float)
        if obstacles.ndim != 2 or obstacles.shape[1] != dim:
            raise ValueError(
                f"constraints['obstacles'] must have shape (K, {dim}), "
                f"got {obstacles.shape}"
            )
        if not np.all(np.isfinite(obstacles)):
            raise ValueError("constraints['obstacles'] must be finite")

        return obstacles, float(margin)

    @staticmethod
    def _sanitize(x: Any) -> np.ndarray:
        """Coerce input to a 1D float ndarray, NaN/Inf guarded."""
        arr = np.ascontiguousarray(x, dtype=float).flatten()
        return arr
