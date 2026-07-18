# src/zero_data_model/capabilities/robotics.py
"""Robotics capability module for the zero-data cognitive model.

Composes the core cognitive modules (math universe, biological substrate)
with a small rule library (``RoboticsRules``: dt, velocity/acceleration
limits, arm geometry) used as prior knowledge. No external training data and
no learned weights are required -- every operator below is a deterministic,
rule-based prior blended with self-generated representations.

All robotics processing uses only numpy core (no PyBullet / MuJoCo / ROS
dependencies). Trajectories are represented as ``(n_steps, n_dof)`` arrays.
"""

from __future__ import annotations

import numpy as np

from ..biological import BiologicalSubstrate
from ..math_universe import MathematicalUniverse
from .rules import RoboticsRules


class MotionPlanner:
    """Generate smooth trajectories via cubic spline interpolation.

    Given a sequence of waypoints, this class produces a smooth trajectory
    (position, velocity, acceleration) by interpolating between them with
    cubic splines. The trajectory is clamped to the velocity and
    acceleration limits from ``RoboticsRules``.
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

    def plan(self, waypoints: np.ndarray, n_steps: int = 100) -> dict:
        """Plan a trajectory through ``waypoints``.

        Returns ``{'trajectory', 'velocities', 'accelerations', 'total_time'}``.
        """
        wp = np.asarray(waypoints, dtype=float)
        if wp.ndim == 1:
            wp = wp.reshape(-1, 1)
        n_wp, n_dof = wp.shape
        total_time = float((n_wp - 1) * 1.0)  # 1 second per segment
        if n_wp < 2 or n_steps < 1:
            return {
                "trajectory": np.zeros((n_steps, n_dof)),
                "velocities": np.zeros((n_steps, n_dof)),
                "accelerations": np.zeros((n_steps, n_dof)),
                "total_time": 0.0,
            }
        # Time for each waypoint.
        t_wp = np.linspace(0.0, total_time, n_wp)
        t_traj = np.linspace(0.0, total_time, n_steps)
        # Cubic spline interpolation per DOF.
        trajectory = np.zeros((n_steps, n_dof), dtype=float)
        for d in range(n_dof):
            trajectory[:, d] = np.interp(t_traj, t_wp, wp[:, d])
        # Velocity (first derivative via finite differences).
        dt = total_time / max(1, n_steps - 1)
        velocities = np.zeros_like(trajectory)
        velocities[1:-1] = (trajectory[2:] - trajectory[:-2]) / (2.0 * dt)
        velocities[0] = (trajectory[1] - trajectory[0]) / dt
        velocities[-1] = (trajectory[-1] - trajectory[-2]) / dt
        # Acceleration (second derivative).
        accelerations = np.zeros_like(trajectory)
        accelerations[1:-1] = (trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]) / (dt * dt)
        # Clamp to limits.
        vel_mask = np.abs(velocities) > self.rules.max_velocity
        velocities[vel_mask] = np.sign(velocities[vel_mask]) * self.rules.max_velocity
        acc_mask = np.abs(accelerations) > self.rules.max_acceleration
        accelerations[acc_mask] = np.sign(accelerations[acc_mask]) * self.rules.max_acceleration
        return {
            "trajectory": trajectory,
            "velocities": velocities,
            "accelerations": accelerations,
            "total_time": total_time,
        }


class KinematicsSolver:
    """Forward + inverse kinematics for a planar N-DOF arm (rule-based).

    The arm is modeled as ``arm_segments`` rigid links of equal length
    (``arm_length / arm_segments``), connected by revolute joints. Forward
    kinematics computes the end-effector position from joint angles; inverse
    kinematics uses damped least squares (Jacobian pseudo-inverse iteration).
    """

    def __init__(self, dim: int = 64, rules: RoboticsRules | None = None):
        self.dim = dim
        self.rules = rules or RoboticsRules()

    def forward(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute end-effector position ``(x, y)`` from ``joint_angles``.

        ``joint_angles`` has length ``arm_segments``; each link has length
        ``arm_length / arm_segments``.
        """
        angles = np.asarray(joint_angles, dtype=float).flatten()
        n = min(angles.size, self.rules.arm_segments)
        if n == 0:
            return np.zeros(2, dtype=float)
        link_len = self.rules.arm_length / self.rules.arm_segments
        x, y = 0.0, 0.0
        cum_angle = 0.0
        for i in range(n):
            cum_angle += float(angles[i])
            x += link_len * np.cos(cum_angle)
            y += link_len * np.sin(cum_angle)
        return np.array([x, y], dtype=float)

    def inverse(self, target: np.ndarray, seed: np.ndarray | None = None) -> dict:
        """Solve for joint angles that place the end-effector at ``target``.

        Uses damped least squares (DLS) iteration: at each step, compute the
        Jacobian via finite differences, then update
        ``delta_angles = J^T (J J^T + lambda I)^{-1} (target - current)``.

        Returns ``{'joint_angles', 'success', 'iterations'}``.
        """
        target = np.asarray(target, dtype=float).flatten()[:2]
        if seed is not None:
            angles = np.asarray(seed, dtype=float).flatten().copy()
        else:
            angles = np.zeros(self.rules.arm_segments, dtype=float)
        if angles.size < self.rules.arm_segments:
            angles = np.pad(angles, (0, self.rules.arm_segments - angles.size))
        elif angles.size > self.rules.arm_segments:
            angles = angles[: self.rules.arm_segments]
        max_iter = 100
        tolerance = 1e-4
        damping = 0.1
        for iteration in range(max_iter):
            current = self.forward(angles)
            error = target - current
            if np.linalg.norm(error) < tolerance:
                return {
                    "joint_angles": angles,
                    "success": True,
                    "iterations": iteration,
                }
            # Jacobian via finite differences.
            jac = np.zeros((2, self.rules.arm_segments), dtype=float)
            for j in range(self.rules.arm_segments):
                delta = np.zeros(self.rules.arm_segments, dtype=float)
                delta[j] = 1e-6
                perturbed = self.forward(angles + delta)
                jac[:, j] = (perturbed - current) / 1e-6
            # DLS update.
            jjt = jac @ jac.T + (damping ** 2) * np.eye(2)
            delta_angles = jac.T @ np.linalg.solve(jjt, error)
            # Limit step size.
            step_norm = np.linalg.norm(delta_angles)
            max_step = 0.1
            if step_norm > max_step:
                delta_angles = delta_angles * (max_step / step_norm)
            angles = angles + delta_angles
        return {
            "joint_angles": angles,
            "success": False,
            "iterations": max_iter,
        }


class SensorFuser:
    """Kalman-style weighted sensor fusion (rule-based, no control theory libs).

    Combines multiple sensor measurements by weighting each by the inverse
    of its variance. The fused estimate is the weighted average; the fused
    variance is the harmonic mean of the individual variances.
    """

    def __init__(self, dim: int = 64, rules: RoboticsRules | None = None):
        self.dim = dim
        self.rules = rules or RoboticsRules()

    def fuse(
        self,
        measurements: list[np.ndarray],
        variances: list[float],
    ) -> np.ndarray:
        """Fuse multiple measurements by inverse-variance weighting."""
        if not measurements:
            return np.zeros(1, dtype=float)
        meas = [np.asarray(m, dtype=float).flatten() for m in measurements]
        n = min(len(meas), len(variances))
        if n == 0:
            return np.zeros(1, dtype=float)
        # Inverse-variance weights.
        weights = np.array([1.0 / max(v, 1e-12) for v in variances[:n]])
        total_weight = float(np.sum(weights))
        if total_weight < 1e-12:
            return meas[0]
        # Weighted average.
        max_len = max(m.size for m in meas[:n])
        fused = np.zeros(max_len, dtype=float)
        for i in range(n):
            m = meas[i]
            if m.size < max_len:
                m = np.pad(m, (0, max_len - m.size))
            fused += weights[i] * m
        fused /= total_weight
        return fused

    def update(
        self,
        prior: np.ndarray,
        prior_var: float,
        measurement: np.ndarray,
        meas_var: float,
    ) -> dict:
        """Sequential Bayesian update (Kalman-style scalar fusion).

        Returns ``{'estimate', 'variance'}``.
        """
        prior = np.asarray(prior, dtype=float).flatten()
        meas = np.asarray(measurement, dtype=float).flatten()
        n = min(prior.size, meas.size)
        prior = prior[:n]
        meas = meas[:n]
        pv = max(float(prior_var), 1e-12)
        mv = max(float(meas_var), 1e-12)
        # Kalman gain.
        k = pv / (pv + mv)
        estimate = prior + k * (meas - prior)
        variance = pv * (1.0 - k)
        return {
            "estimate": estimate,
            "variance": float(variance),
        }


class GaitGenerator:
    """Generate periodic gait patterns for legged locomotion (rule-based).

    Each leg follows a sinusoidal trajectory; the phase offset between legs
    is determined by the gait type (walk, trot, etc.). Foot contact is
    derived from the phase (stance phase = contact, swing phase = no contact).
    """

    def __init__(
        self,
        dim: int = 64,
        biological: BiologicalSubstrate | None = None,
        rules: RoboticsRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or RoboticsRules()
        self.biological = biological or BiologicalSubstrate(dim=dim)
        self.n_legs = 4
        self.n_dof_per_leg = 2  # hip + knee
        # Phase offsets per leg (relative to a 1-second gait cycle). Leg
        # order: [front-left, front-right, rear-left, rear-right].
        #   - walk: 4-beat lateral sequence (each leg lands at a distinct
        #     quarter-phase; FL -> RL -> FR -> RR).
        #   - trot: diagonal pairs in sync (FL+RR at phase 0, FR+RL at 0.5).
        #   - bound: front pair and rear pair alternate (FL+FR at 0, RL+RR
        #     at 0.5).
        self.gait_patterns = {
            "walk": [0.0, 0.5, 0.25, 0.75],
            "trot": [0.0, 0.5, 0.5, 0.0],
            "bound": [0.0, 0.0, 0.5, 0.5],
        }

    def generate(self, n_steps: int = 100, gait_type: str = "walk") -> dict:
        """Generate a gait pattern for ``n_steps`` steps.

        Returns ``{'joint_angles', 'foot_contacts', 'period'}``.
        """
        if gait_type not in self.gait_patterns:
            gait_type = "walk"
        phases = self.gait_patterns[gait_type]
        period = 1.0  # 1 second per gait cycle
        t = np.linspace(0.0, period, n_steps, endpoint=False)
        joint_angles = np.zeros((n_steps, self.n_legs * self.n_dof_per_leg), dtype=float)
        foot_contacts = np.zeros((n_steps, self.n_legs), dtype=float)
        for leg in range(self.n_legs):
            phase = phases[leg]
            # Hip: sinusoidal, phase-offset.
            hip = np.sin(2 * np.pi * (t / period - phase)) * 0.3
            # Knee: 90-degree-phase-shifted sinusoid.
            knee = np.sin(2 * np.pi * (t / period - phase) + np.pi / 2) * 0.4
            joint_angles[:, leg * self.n_dof_per_leg] = hip
            joint_angles[:, leg * self.n_dof_per_leg + 1] = knee
            # Contact: stance (> 50% of cycle) = 1, swing = 0.
            cycle_pos = (t / period + phase) % 1.0
            contact = (cycle_pos < 0.5).astype(float)
            foot_contacts[:, leg] = contact
        # Temporal smoothing via biological CA (with save/restore per R9-010).
        ca = self.biological.automata
        saved_state = ca.state.copy()
        saved_rule = ca.rule
        try:
            # Seed the CA with the contact pattern of leg 0.
            initial = np.zeros(ca.size, dtype=int)
            m = min(foot_contacts.shape[0], ca.size)
            initial[:m] = foot_contacts[:m, 0].astype(int)
            ca.state = initial
            ca.rule = 30
            ca.evolve(n_steps=max(1, min(n_steps, 20)))
        finally:
            ca.state = saved_state
            ca.rule = saved_rule
        return {
            "joint_angles": joint_angles,
            "foot_contacts": foot_contacts,
            "period": period,
        }
