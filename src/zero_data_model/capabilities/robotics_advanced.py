# src/zero_data_model/capabilities/robotics_advanced.py
"""Advanced robotics capabilities for the zero-data cognitive model.

Composes the core cognitive modules (active inference for cost evaluation,
math universe for fractal/topological trajectory features, biological
substrate for temporal CA smoothing) with a small rule library
(``RoboticsRules``). All operators are pure numpy — no PyBullet / MuJoCo /
ROS dependencies.
"""

from __future__ import annotations

import numpy as np

from ..active_inference import ActiveInferenceEngine
from ..math_universe import MathematicalUniverse
from .rules import RoboticsRules


class TrajectoryOptimizer:
    """Optimize a trajectory by minimizing jerk (3rd time-derivative).

    Given a ``(n_steps, n_dof)`` trajectory, this class performs gradient
    descent on the jerk cost (sum of squared 3rd differences) while keeping
    the endpoints fixed. The result is a smoother trajectory with the same
    start and end configurations.
    """

    def __init__(
        self,
        dim: int = 64,
        math_universe: MathematicalUniverse | None = None,
        rules: RoboticsRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or RoboticsRules()
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)

    def optimize(self, trajectory: np.ndarray, n_iter: int = 10) -> dict:
        """Smooth ``trajectory`` by gradient-descent on jerk.

        Uses backtracking line search: each iteration halves the step size
        until the new cost is strictly less than the previous cost (or the
        step size falls below 1e-9). This guarantees monotonic cost
        decrease and avoids the overshoot that plagues a fixed step size
        (the jerk kernel amplifies updates by ~38x).

        Returns ``{'optimized', 'jerk', 'improvement'}`` where ``jerk`` is
        the final jerk cost and ``improvement`` is the relative decrease
        from the input trajectory's jerk cost.
        """
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim == 1:
            traj = traj.reshape(-1, 1)
        n_steps, n_dof = traj.shape
        if n_steps < 4:
            # Not enough points to compute a 3rd difference.
            return {
                "optimized": traj.copy(),
                "jerk": 0.0,
                "improvement": 0.0,
            }
        # Save the start/end so we can re-pin them each iteration.
        start = traj[0].copy()
        end = traj[-1].copy()
        initial_jerk = self._jerk_cost(traj)
        current = traj.copy()
        prev_cost = initial_jerk
        # Initial step size; the line search will halve this as needed.
        base_step = 0.01
        min_step = 1e-9
        for _ in range(max(1, n_iter)):
            grad = self._jerk_gradient(current)
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm < 1e-12:
                break
            step = base_step
            accepted = False
            # Backtracking line search (up to 30 halvings).
            for _back in range(30):
                candidate = current - step * grad
                candidate[0] = start
                candidate[-1] = end
                candidate = np.nan_to_num(
                    candidate, nan=0.0, posinf=0.0, neginf=0.0
                )
                new_cost = self._jerk_cost(candidate)
                if np.isfinite(new_cost) and new_cost < prev_cost:
                    current = candidate
                    prev_cost = new_cost
                    accepted = True
                    break
                step *= 0.5
                if step < min_step:
                    break
            if not accepted:
                break
        current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
        final_jerk = self._jerk_cost(current)
        improvement = 0.0
        if initial_jerk > 1e-12:
            improvement = float((initial_jerk - final_jerk) / initial_jerk)
        return {
            "optimized": current,
            "jerk": float(final_jerk),
            "improvement": float(improvement),
        }

    @staticmethod
    def _jerk_cost(traj: np.ndarray) -> float:
        """Sum of squared 3rd differences along the time axis."""
        if traj.shape[0] < 4:
            return 0.0
        third_diff = traj[3:] - 3 * traj[2:-1] + 3 * traj[1:-2] - traj[:-3]
        return float(np.sum(third_diff * third_diff))

    @staticmethod
    def _jerk_gradient(traj: np.ndarray) -> np.ndarray:
        """Gradient of ``_jerk_cost`` w.r.t. each point in ``traj``.

        For the jerk cost ``J = sum_t D_t^2`` with
        ``D_t = -x_t + 3 x_{t+1} - 3 x_{t+2} + x_{t+3}`` and ``t`` ranging
        over the valid window ``[0, n-4]``, the gradient is

            ``dJ/dx_i = 2 * sum_{m=0..3} D_{i-m} * c[m]``

        where ``c = [-1, 3, -3, 1]`` (the kernel of the 3rd difference) and
        out-of-range ``D`` contribute zero. Each ``D_t`` depends on a valid
        window only, so this formula correctly handles the finite-signal
        boundary (the autocorrelation shortcut over-counts because it
        assumes an infinite signal).
        """
        n_steps, n_dof = traj.shape
        grad = np.zeros_like(traj)
        if n_steps < 4:
            return grad
        c = np.array([-1.0, 3.0, -3.0, 1.0])
        n_valid = n_steps - 3
        # D[t] = sum_k c[k] * traj[t + k] for t in [0, n_valid - 1].
        d = c[0] * traj[:n_valid] + c[1] * traj[1 : n_valid + 1]
        d = d + c[2] * traj[2 : n_valid + 2] + c[3] * traj[3 : n_valid + 3]
        # grad[i] += 2 * c[m] * D[i - m] for m in [0..3] (valid when i - m
        # is in [0, n_valid - 1]); this is a shifted accumulate.
        for m in range(4):
            grad[m : m + n_valid] += 2.0 * c[m] * d
        # Endpoints are pinned (gradient zeroed so they are not updated).
        grad[0] = 0.0
        grad[-1] = 0.0
        return grad


class CollisionChecker:
    """Check collisions between a point/body and circular/ball obstacles.

    Obstacles are represented as ``(center, radius)`` pairs. The checker
    supports single-point checks (with an optional body radius) and full
    trajectory checks.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: RoboticsRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or RoboticsRules()

    def check(
        self,
        obstacles: list,
        position: np.ndarray,
        radius: float = 0.1,
    ) -> dict:
        """Check if a body at ``position`` with body ``radius`` collides.

        Each obstacle is ``(center, obs_radius)`` where ``center`` is a
        numpy array (2D or 3D). Returns ``{'collision', 'nearest_obstacle',
        'distance'}`` where ``distance`` is the clearance (signed: negative
        means penetration).
        """
        pos = np.asarray(position, dtype=float).flatten()
        collision = False
        nearest_idx = -1
        nearest_clearance = float("inf")
        for i, obs in enumerate(obstacles):
            if not (isinstance(obs, (list, tuple)) and len(obs) == 2):
                continue
            center, obs_r = obs
            center = np.asarray(center, dtype=float).flatten()
            # Match dimensions.
            n = min(pos.size, center.size)
            diff = pos[:n] - center[:n]
            dist = float(np.linalg.norm(diff))
            clearance = dist - (radius + float(obs_r))
            if clearance < 0.0:
                collision = True
            if clearance < nearest_clearance:
                nearest_clearance = clearance
                nearest_idx = i
        # Apply safety margin: a clearance within the margin is "near-miss".
        if not collision and nearest_clearance < self.rules.safety_margin:
            collision = True
        if nearest_idx < 0:
            nearest_clearance = float("inf")
        distance = (
            float(nearest_clearance)
            if np.isfinite(nearest_clearance)
            else float("inf")
        )
        return {
            "collision": bool(collision),
            "nearest_obstacle": int(nearest_idx),
            "distance": distance,
        }

    def check_path(
        self,
        obstacles: list,
        path: np.ndarray,
        radius: float = 0.1,
    ) -> dict:
        """Check collision along a path.

        Returns ``{'collision', 'first_collision_step', 'nearest_obstacle',
        'min_clearance'}``.
        """
        p = np.asarray(path, dtype=float)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        n_steps = p.shape[0]
        collision = False
        first_step = -1
        min_clearance = float("inf")
        nearest_obs = -1
        for step in range(n_steps):
            result = self.check(obstacles, p[step], radius=radius)
            if result["distance"] < min_clearance:
                min_clearance = result["distance"]
                nearest_obs = result["nearest_obstacle"]
            if result["collision"] and not collision:
                collision = True
                first_step = step
        min_clr = (
            float(min_clearance)
            if np.isfinite(min_clearance)
            else float("inf")
        )
        return {
            "collision": bool(collision),
            "first_collision_step": int(first_step),
            "nearest_obstacle": int(nearest_obs),
            "min_clearance": min_clr,
        }


class MPCController:
    """Model Predictive Controller with rule-based action search.

    A simple MPC that, at each control step, evaluates a set of candidate
    actions over a prediction horizon and picks the action with the lowest
    predicted cost. The cost blends:
    - Target tracking error (L2 distance to target at each predicted step)
    - Collision penalty (penalize predicted states near obstacles)
    - Free-energy surprisal (uses ``ActiveInferenceEngine.compute_free_energy``
      as a penalty for surprising state sequences)

    The dynamics model is a simple integrator: ``x_{t+1} = x_t + dt * u``.
    """

    def __init__(
        self,
        dim: int = 64,
        active_inference: ActiveInferenceEngine | None = None,
        rules: RoboticsRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or RoboticsRules()
        self.active_inference = active_inference or ActiveInferenceEngine(
            state_dim=dim, obs_dim=dim, action_dim=dim // 2
        )
        # Discrete action candidates (8 directions + zero action).
        self.action_candidates = self._build_action_candidates()

    def _build_action_candidates(self) -> np.ndarray:
        """Build 9 candidate actions: 8 cardinal/diagonal + zero."""
        actions = []
        max_v = self.rules.max_velocity
        for theta in np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False):
            actions.append([max_v * np.cos(theta), max_v * np.sin(theta)])
        actions.append([0.0, 0.0])
        return np.array(actions, dtype=float)

    def control(
        self,
        current_state: np.ndarray,
        target_state: np.ndarray,
        obstacles: list | None = None,
    ) -> dict:
        """Pick the best action by evaluating candidate actions over horizon.

        Returns ``{'action', 'predicted_trajectory', 'cost'}``.
        """
        current = np.asarray(current_state, dtype=float).flatten()
        target = np.asarray(target_state, dtype=float).flatten()
        # Match dims to the smaller of the two.
        n = min(current.size, target.size)
        current = current[:n]
        target = target[:n]
        horizon = max(1, self.rules.mpc_horizon)
        dt = self.rules.dt
        best_action = np.zeros(2, dtype=float)
        best_cost = float("inf")
        best_traj = None
        # Track the trajectory of the best action for return.
        for action in self.action_candidates:
            # Match action dim to state dim.
            a = action[:n] if n < 2 else action
            if n == 1:
                a = np.array([action[0]])
            traj = np.zeros((horizon + 1, n), dtype=float)
            traj[0] = current
            for t in range(horizon):
                traj[t + 1] = traj[t] + dt * a
            # Cost components.
            tracking = float(np.sum((traj - target) ** 2))
            collision_pen = 0.0
            if obstacles:
                checker = CollisionChecker(dim=self.dim, rules=self.rules)
                for t in range(1, horizon + 1):
                    result = checker.check(obstacles, traj[t])
                    if result["collision"]:
                        collision_pen += 100.0
                    elif result["distance"] < self.rules.safety_margin:
                        collision_pen += (self.rules.safety_margin - result["distance"]) * 10.0
            # Free-energy surprisal: feed the trajectory deviation as observation.
            deviation = traj - target
            fe = 0.0
            for t in range(1, horizon + 1):
                try:
                    fe += float(self.active_inference.compute_free_energy(deviation[t]))
                except Exception:  # pragma: no cover - defensive
                    fe += 0.0
            # Normalize free energy by horizon.
            fe = fe / max(1, horizon)
            # Total cost: tracking + collision + 0.1 * free energy.
            total = tracking + collision_pen + 0.1 * fe
            if not np.isfinite(total):
                total = 1e12
            if total < best_cost:
                best_cost = total
                best_action = a.copy()
                best_traj = traj.copy()
        if best_traj is None:
            best_traj = np.zeros((horizon + 1, n), dtype=float)
            best_traj[0] = current
        return {
            "action": best_action,
            "predicted_trajectory": best_traj,
            "cost": float(best_cost),
        }
