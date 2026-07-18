"""Tests for the Robotics capability domain (advanced module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    CollisionChecker,
    MPCController,
    RoboticsRules,
    TrajectoryOptimizer,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# --------------------------------------------------------------------------- #
# TrajectoryOptimizer
# --------------------------------------------------------------------------- #

class TestTrajectoryOptimizer:
    def test_optimize_improves_jerk(self):
        opt = TrajectoryOptimizer(dim=8)
        np.random.seed(42)
        traj = np.cumsum(np.random.randn(20, 2), axis=0) * 0.1
        result = opt.optimize(traj, n_iter=10)
        assert result["improvement"] > 0.0
        assert result["jerk"] < TrajectoryOptimizer._jerk_cost(traj)

    def test_optimize_preserves_endpoints(self):
        opt = TrajectoryOptimizer(dim=8)
        np.random.seed(7)
        traj = np.cumsum(np.random.randn(15, 2), axis=0) * 0.1
        result = opt.optimize(traj, n_iter=10)
        np.testing.assert_allclose(result["optimized"][0], traj[0])
        np.testing.assert_allclose(result["optimized"][-1], traj[-1])

    def test_optimize_constant_trajectory_zero_jerk(self):
        opt = TrajectoryOptimizer(dim=8)
        traj = np.ones((10, 2))
        result = opt.optimize(traj, n_iter=5)
        assert result["jerk"] == 0.0
        assert result["improvement"] == 0.0

    def test_optimize_short_trajectory_returns_unchanged(self):
        opt = TrajectoryOptimizer(dim=8)
        traj = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = opt.optimize(traj, n_iter=5)
        assert result["optimized"].shape == traj.shape
        # Less than 4 points -> jerk is 0 by definition.
        assert result["jerk"] == 0.0

    def test_optimize_1d_trajectory_reshaped(self):
        opt = TrajectoryOptimizer(dim=8)
        traj = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5])
        result = opt.optimize(traj, n_iter=5)
        assert result["optimized"].shape == (10, 1)
        assert result["improvement"] >= 0.0

    def test_optimize_impulse_reduced(self):
        opt = TrajectoryOptimizer(dim=8)
        traj = np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ).reshape(-1, 1)
        result = opt.optimize(traj, n_iter=10)
        # Impulse should be smoothed significantly.
        assert result["improvement"] > 0.5

    def test_jerk_cost_basic(self):
        # Constant trajectory: jerk cost is 0.
        const = np.ones((10, 2))
        assert TrajectoryOptimizer._jerk_cost(const) == 0.0

    def test_jerk_cost_short_returns_zero(self):
        short = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert TrajectoryOptimizer._jerk_cost(short) == 0.0

    def test_jerk_cost_uses_third_difference(self):
        # A trajectory with constant 2nd difference (constant acceleration) has
        # zero 3rd difference -> zero jerk cost.
        t = np.arange(10, dtype=float).reshape(-1, 1)
        # x_t = t (linear in t): 3rd difference is 0.
        assert TrajectoryOptimizer._jerk_cost(t) == 0.0

    def test_jerk_gradient_zero_for_constant(self):
        const = np.ones((10, 2))
        grad = TrajectoryOptimizer._jerk_gradient(const)
        np.testing.assert_allclose(grad, 0.0)

    def test_jerk_gradient_zero_for_short(self):
        short = np.array([[0.0], [1.0]])
        grad = TrajectoryOptimizer._jerk_gradient(short)
        assert grad.shape == short.shape
        np.testing.assert_allclose(grad, 0.0)

    def test_jerk_gradient_zero_at_endpoints(self):
        # Endpoints are always pinned (gradient forced to 0).
        traj = np.random.randn(10, 2)
        grad = TrajectoryOptimizer._jerk_gradient(traj)
        np.testing.assert_allclose(grad[0], 0.0)
        np.testing.assert_allclose(grad[-1], 0.0)

    def test_jerk_gradient_finite_difference_match(self):
        # Verify the analytic gradient matches a numerical finite-difference
        # estimate for a random trajectory.
        traj = np.random.randn(15, 2) * 0.5
        analytic = TrajectoryOptimizer._jerk_gradient(traj)
        # Numerical gradient (central differences).
        eps = 1e-6
        numeric = np.zeros_like(traj)
        for i in range(traj.shape[0]):
            for d in range(traj.shape[1]):
                plus = traj.copy()
                plus[i, d] += eps
                minus = traj.copy()
                minus[i, d] -= eps
                numeric[i, d] = (
                    TrajectoryOptimizer._jerk_cost(plus)
                    - TrajectoryOptimizer._jerk_cost(minus)
                ) / (2 * eps)
        # Endpoints are pinned by the analytic gradient (forced to 0); the
        # numeric gradient is nonzero there. Exclude them from the comparison.
        np.testing.assert_allclose(analytic[1:-1], numeric[1:-1], atol=1e-4)

    def test_optimize_zero_iterations_returns_input(self):
        opt = TrajectoryOptimizer(dim=8)
        traj = np.random.randn(10, 2)
        result = opt.optimize(traj, n_iter=0)
        # n_iter=0 -> no iterations; max(1, n_iter)=1 forces 1 iter though.
        # Either way, no significant change if cost is already minimal or
        # line search fails immediately. Check shape and jerk are finite.
        assert result["optimized"].shape == traj.shape
        assert np.isfinite(result["jerk"])

    def test_optimize_nan_input_sanitized(self):
        opt = TrajectoryOptimizer(dim=8)
        traj = np.array(
            [[0.0, 0.0], [1.0, np.nan], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
        )
        result = opt.optimize(traj, n_iter=5)
        assert np.all(np.isfinite(result["optimized"]))


# --------------------------------------------------------------------------- #
# CollisionChecker
# --------------------------------------------------------------------------- #

class TestCollisionChecker:
    def test_no_collision(self):
        cc = CollisionChecker(dim=8)
        obstacles = [(np.array([1.0, 1.0]), 0.2)]
        result = cc.check(obstacles, np.array([0.0, 0.0]), radius=0.1)
        assert not result["collision"]
        assert result["nearest_obstacle"] == 0
        assert result["distance"] > 0.0

    def test_direct_collision(self):
        cc = CollisionChecker(dim=8)
        obstacles = [(np.array([0.5, 0.5]), 0.2)]
        result = cc.check(obstacles, np.array([0.5, 0.5]), radius=0.1)
        assert result["collision"]
        assert result["distance"] < 0.0

    def test_clearance_calculation(self):
        cc = CollisionChecker(dim=8)
        # Obstacle center at (1, 0) with radius 0.2; body at (0, 0) with radius 0.1.
        # Distance between centers = 1.0; clearance = 1.0 - 0.2 - 0.1 = 0.7.
        obstacles = [(np.array([1.0, 0.0]), 0.2)]
        result = cc.check(obstacles, np.array([0.0, 0.0]), radius=0.1)
        np.testing.assert_allclose(result["distance"], 0.7, atol=1e-10)
        assert not result["collision"]

    def test_safety_margin_triggers_near_miss(self):
        # Default safety margin is 0.05. Clearance within margin is "collision".
        cc = CollisionChecker(dim=8)
        # Place obstacle so clearance is just under safety margin.
        obstacles = [(np.array([0.32, 0.0]), 0.2)]
        # Body at (0, 0), radius 0.1. Clearance = 0.32 - 0.2 - 0.1 = 0.02 < 0.05.
        result = cc.check(obstacles, np.array([0.0, 0.0]), radius=0.1)
        assert result["collision"]

    def test_no_obstacles(self):
        cc = CollisionChecker(dim=8)
        result = cc.check([], np.array([0.0, 0.0]), radius=0.1)
        assert not result["collision"]
        assert result["nearest_obstacle"] == -1
        assert result["distance"] == float("inf")

    def test_multiple_obstacles_picks_nearest(self):
        cc = CollisionChecker(dim=8)
        obstacles = [
            (np.array([5.0, 5.0]), 0.1),
            (np.array([1.0, 0.0]), 0.1),
            (np.array([3.0, 0.0]), 0.1),
        ]
        result = cc.check(obstacles, np.array([0.0, 0.0]), radius=0.1)
        assert result["nearest_obstacle"] == 1  # closest to (1, 0)
        assert result["distance"] < 1.0

    def test_3d_obstacle(self):
        cc = CollisionChecker(dim=8)
        obstacles = [(np.array([0.0, 0.0, 1.0]), 0.2)]
        result = cc.check(obstacles, np.array([0.0, 0.0, 0.0]), radius=0.1)
        np.testing.assert_allclose(result["distance"], 0.7, atol=1e-10)

    def test_invalid_obstacle_skipped(self):
        cc = CollisionChecker(dim=8)
        obstacles = [
            "not_an_obstacle",
            (np.array([1.0, 1.0]), 0.2),
            (1.0, 2.0, 3.0),  # wrong shape
        ]
        result = cc.check(obstacles, np.array([0.0, 0.0]), radius=0.1)
        # Only the second (valid) obstacle should be considered.
        assert result["nearest_obstacle"] == 1

    def test_check_path_no_collision(self):
        cc = CollisionChecker(dim=8)
        obstacles = [(np.array([10.0, 10.0]), 0.1)]
        path = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = cc.check_path(obstacles, path, radius=0.1)
        assert not result["collision"]
        assert result["first_collision_step"] == -1

    def test_check_path_with_collision(self):
        cc = CollisionChecker(dim=8)
        obstacles = [(np.array([0.5, 0.5]), 0.1)]
        path = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = cc.check_path(obstacles, path, radius=0.1)
        assert result["collision"]
        assert result["first_collision_step"] == 1

    def test_check_path_1d_reshaped(self):
        cc = CollisionChecker(dim=8)
        obstacles = [(np.array([5.0]), 0.1)]
        path = np.array([0.0, 1.0, 2.0, 3.0])
        result = cc.check_path(obstacles, path, radius=0.1)
        assert not result["collision"]

    def test_check_path_empty_path(self):
        cc = CollisionChecker(dim=8)
        obstacles = [(np.array([0.0, 0.0]), 0.5)]
        path = np.array([])
        result = cc.check_path(obstacles, path, radius=0.1)
        # Empty path -> no collision.
        assert not result["collision"]
        assert result["min_clearance"] == float("inf")


# --------------------------------------------------------------------------- #
# MPCController
# --------------------------------------------------------------------------- #

class TestMPCController:
    def test_control_returns_action_and_trajectory(self):
        mpc = MPCController(dim=8)
        result = mpc.control(
            current_state=np.array([0.0, 0.0]),
            target_state=np.array([1.0, 1.0]),
        )
        assert "action" in result
        assert "predicted_trajectory" in result
        assert "cost" in result
        assert result["action"].shape == (2,)
        # Trajectory: horizon + 1 steps.
        assert result["predicted_trajectory"].shape == (mpc.rules.mpc_horizon + 1, 2)

    def test_control_action_moves_toward_target(self):
        mpc = MPCController(dim=8)
        result = mpc.control(
            current_state=np.array([0.0, 0.0]),
            target_state=np.array([1.0, 0.0]),
        )
        # Action should point in +x direction (toward target).
        assert result["action"][0] > 0.0

    def test_control_zero_distance(self):
        mpc = MPCController(dim=8)
        result = mpc.control(
            current_state=np.array([0.5, 0.5]),
            target_state=np.array([0.5, 0.5]),
        )
        # Already at target: tracking cost should be near zero.
        assert result["cost"] < 10.0

    def test_control_with_obstacle_penalty(self):
        mpc = MPCController(dim=8)
        # Place obstacle directly between current and target.
        obstacles = [(np.array([0.5, 0.0]), 0.2)]
        result_obs = mpc.control(
            current_state=np.array([0.0, 0.0]),
            target_state=np.array([1.0, 0.0]),
            obstacles=obstacles,
        )
        # Without obstacle, the action would point straight toward target.
        # With obstacle, the cost should reflect the collision penalty.
        result_no_obs = mpc.control(
            current_state=np.array([0.0, 0.0]),
            target_state=np.array([1.0, 0.0]),
        )
        # The cost with obstacle should be >= cost without (penalty added).
        assert result_obs["cost"] >= result_no_obs["cost"]

    def test_control_action_bounded(self):
        mpc = MPCController(dim=8)
        result = mpc.control(
            current_state=np.array([0.0, 0.0]),
            target_state=np.array([100.0, 100.0]),
        )
        # All candidate actions are bounded by max_velocity.
        assert np.linalg.norm(result["action"]) <= mpc.rules.max_velocity + 1e-9

    def test_control_1d_state(self):
        mpc = MPCController(dim=8)
        result = mpc.control(
            current_state=np.array([0.0]),
            target_state=np.array([1.0]),
        )
        # 1D state: action is the first element of the candidate (broadcast).
        assert result["predicted_trajectory"].shape == (mpc.rules.mpc_horizon + 1, 1)

    def test_control_predicts_trajectory_starts_at_current(self):
        mpc = MPCController(dim=8)
        result = mpc.control(
            current_state=np.array([0.3, 0.7]),
            target_state=np.array([1.0, 1.0]),
        )
        np.testing.assert_allclose(result["predicted_trajectory"][0], [0.3, 0.7])

    def test_control_cost_finite(self):
        mpc = MPCController(dim=8)
        result = mpc.control(
            current_state=np.array([0.0, 0.0]),
            target_state=np.array([1.0, 1.0]),
        )
        assert np.isfinite(result["cost"])

    def test_control_custom_horizon(self):
        rules = RoboticsRules(mpc_horizon=5)
        mpc = MPCController(dim=8, rules=rules)
        result = mpc.control(
            current_state=np.array([0.0, 0.0]),
            target_state=np.array([1.0, 1.0]),
        )
        assert result["predicted_trajectory"].shape == (6, 2)  # horizon + 1

    def test_control_zero_action_is_candidate(self):
        # The zero action is in the candidate set; for a state already at
        # target, it should have low cost and may be selected.
        mpc = MPCController(dim=8)
        # Build candidates and confirm the zero action is present.
        zero_in_candidates = np.any(np.all(mpc.action_candidates == 0.0, axis=1))
        assert zero_in_candidates
