"""Tests for the Robotics capability domain (base module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    GaitGenerator,
    KinematicsSolver,
    MotionPlanner,
    RoboticsRules,
    SensorFuser,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# --------------------------------------------------------------------------- #
# RoboticsRules
# --------------------------------------------------------------------------- #

def test_robotics_rules_defaults():
    rules = RoboticsRules()
    assert rules.dt == 0.01
    assert rules.max_velocity == 1.0
    assert rules.max_acceleration == 5.0
    assert rules.arm_segments == 3
    assert rules.arm_length == 1.0
    assert rules.safety_margin == 0.05
    assert rules.mpc_horizon == 10


def test_robotics_rules_post_init_populates_dict():
    rules = RoboticsRules()
    assert rules.rules["dt"] == 0.01
    assert rules.rules["arm_segments"] == 3
    assert rules.rules["mpc_horizon"] == 10


def test_robotics_rules_custom_values():
    rules = RoboticsRules(dt=0.05, max_velocity=2.0, arm_segments=5)
    assert rules.dt == 0.05
    assert rules.max_velocity == 2.0
    assert rules.arm_segments == 5


# --------------------------------------------------------------------------- #
# MotionPlanner
# --------------------------------------------------------------------------- #

class TestMotionPlanner:
    def test_basic_trajectory(self):
        planner = MotionPlanner(dim=8)
        wp = np.array([[0.0, 0.0], [1.0, 1.0]])
        r = planner.plan(wp, n_steps=20)
        assert r["trajectory"].shape == (20, 2)
        assert r["velocities"].shape == (20, 2)
        assert r["accelerations"].shape == (20, 2)
        assert r["total_time"] == 1.0

    def test_waypoints_preserved_at_ends(self):
        planner = MotionPlanner(dim=8)
        wp = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
        r = planner.plan(wp, n_steps=11)
        np.testing.assert_allclose(r["trajectory"][0], wp[0])
        np.testing.assert_allclose(r["trajectory"][-1], wp[-1])

    def test_intermediate_waypoint_hit(self):
        planner = MotionPlanner(dim=8)
        wp = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
        r = planner.plan(wp, n_steps=11)
        # The middle waypoint should be reached at the middle of the trajectory.
        mid_idx = 5  # 11 steps total -> middle at index 5
        np.testing.assert_allclose(r["trajectory"][mid_idx], wp[1], atol=1e-6)

    def test_velocity_clamped(self):
        # Use waypoints far apart to force high velocity, then check clamp.
        rules = RoboticsRules(max_velocity=0.5)
        planner = MotionPlanner(dim=8, rules=rules)
        wp = np.array([[0.0], [100.0]])  # 100 units in 1 second
        r = planner.plan(wp, n_steps=20)
        assert np.all(np.abs(r["velocities"]) <= 0.5 + 1e-9)

    def test_acceleration_clamped(self):
        rules = RoboticsRules(max_acceleration=1.0)
        planner = MotionPlanner(dim=8, rules=rules)
        wp = np.array([[0.0], [10.0], [0.0]])
        r = planner.plan(wp, n_steps=20)
        assert np.all(np.abs(r["accelerations"]) <= 1.0 + 1e-9)

    def test_single_waypoint_returns_zeros(self):
        planner = MotionPlanner(dim=8)
        wp = np.array([[1.0, 2.0]])
        r = planner.plan(wp, n_steps=10)
        assert np.all(r["trajectory"] == 0.0)
        assert r["total_time"] == 0.0

    def test_zero_steps_returns_zeros(self):
        planner = MotionPlanner(dim=8)
        wp = np.array([[0.0, 0.0], [1.0, 1.0]])
        r = planner.plan(wp, n_steps=0)
        assert r["trajectory"].shape == (0, 2)

    def test_1d_waypoints_reshaped(self):
        planner = MotionPlanner(dim=8)
        wp = np.array([0.0, 1.0, 0.0])  # 1D, 3 waypoints
        r = planner.plan(wp, n_steps=10)
        assert r["trajectory"].shape == (10, 1)


# --------------------------------------------------------------------------- #
# KinematicsSolver
# --------------------------------------------------------------------------- #

class TestKinematicsSolver:
    def test_forward_zero_angles(self):
        ks = KinematicsSolver(dim=8)
        # All zero angles => arm lies along x-axis.
        ee = ks.forward(np.array([0.0, 0.0, 0.0]))
        # 3 segments, each of length 1/3, total reach = 1.0 along x.
        np.testing.assert_allclose(ee, [1.0, 0.0])

    def test_forward_90deg_first_joint(self):
        ks = KinematicsSolver(dim=8)
        ee = ks.forward(np.array([np.pi / 2, 0.0, 0.0]))
        # First joint at 90deg, then 2 more along x in local frame => (0, 1/3 + 0) ... actually
        # end effector: cos(pi/2) * 1/3 + cos(pi/2) * 1/3 + cos(pi/2) * 1/3 = 0
        # sin(pi/2) * 1/3 + sin(pi/2) * 1/3 + sin(pi/2) * 1/3 = 1.0
        np.testing.assert_allclose(ee, [0.0, 1.0], atol=1e-10)

    def test_forward_returns_2d(self):
        ks = KinematicsSolver(dim=8)
        ee = ks.forward(np.array([0.1, 0.2, 0.3]))
        assert ee.shape == (2,)

    def test_forward_empty_returns_zero(self):
        ks = KinematicsSolver(dim=8)
        ee = ks.forward(np.array([]))
        np.testing.assert_allclose(ee, [0.0, 0.0])

    def test_inverse_reaches_target(self):
        ks = KinematicsSolver(dim=8)
        target = np.array([0.5, 0.3])
        result = ks.inverse(target, seed=np.array([0.1, 0.1, 0.1]))
        assert result["success"]
        ee = ks.forward(result["joint_angles"])
        np.testing.assert_allclose(ee, target, atol=1e-3)

    def test_inverse_target_at_origin(self):
        ks = KinematicsSolver(dim=8)
        result = ks.inverse(np.array([0.0, 0.0]))
        # Even if not converged, should return reasonable angles.
        assert "joint_angles" in result
        assert "iterations" in result

    def test_inverse_seed_padded(self):
        ks = KinematicsSolver(dim=8)
        # Seed shorter than arm_segments should be zero-padded.
        result = ks.inverse(np.array([0.3, 0.3]), seed=np.array([0.1]))
        assert result["joint_angles"].shape == (3,)

    def test_inverse_seed_truncated(self):
        ks = KinematicsSolver(dim=8)
        # Seed longer than arm_segments should be truncated.
        result = ks.inverse(
            np.array([0.3, 0.3]),
            seed=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        )
        assert result["joint_angles"].shape == (3,)

    def test_inverse_no_seed_starts_from_zero(self):
        ks = KinematicsSolver(dim=8)
        result = ks.inverse(np.array([0.5, 0.0]))
        # Should solve from a zero seed (target reachable along x-axis).
        assert result["success"]

    def test_inverse_max_reach(self):
        ks = KinematicsSolver(dim=8)
        # Target at the maximum reach (1.0 along x).
        result = ks.inverse(np.array([1.0, 0.0]))
        assert result["success"]
        ee = ks.forward(result["joint_angles"])
        np.testing.assert_allclose(ee, [1.0, 0.0], atol=1e-3)


# --------------------------------------------------------------------------- #
# SensorFuser
# --------------------------------------------------------------------------- #

class TestSensorFuser:
    def test_fuse_two_measurements(self):
        sf = SensorFuser(dim=8)
        fused = sf.fuse(
            [np.array([1.0, 2.0]), np.array([1.5, 2.5])],
            [0.1, 0.2],
        )
        # Inverse-variance weights: w1 = 10, w2 = 5, total = 15.
        # fused = (10 * [1, 2] + 5 * [1.5, 2.5]) / 15 = [17.5/15, 32.5/15]
        np.testing.assert_allclose(fused, [17.5 / 15.0, 32.5 / 15.0])

    def test_fuse_empty_returns_zeros(self):
        sf = SensorFuser(dim=8)
        fused = sf.fuse([], [])
        np.testing.assert_allclose(fused, [0.0])

    def test_fuse_single_measurement(self):
        sf = SensorFuser(dim=8)
        fused = sf.fuse([np.array([1.0, 2.0, 3.0])], [0.5])
        np.testing.assert_allclose(fused, [1.0, 2.0, 3.0])

    def test_fuse_low_variance_dominates(self):
        sf = SensorFuser(dim=8)
        # Sensor 1 with much lower variance should dominate.
        fused = sf.fuse(
            [np.array([0.0]), np.array([10.0])],
            [0.001, 100.0],
        )
        assert abs(fused[0] - 0.0) < 0.5

    def test_fuse_unequal_lengths_padded(self):
        sf = SensorFuser(dim=8)
        fused = sf.fuse(
            [np.array([1.0]), np.array([1.0, 2.0])],
            [0.1, 0.1],
        )
        assert fused.shape == (2,)

    def test_update_equal_variance(self):
        sf = SensorFuser(dim=8)
        result = sf.update(
            prior=np.array([1.0]),
            prior_var=0.5,
            measurement=np.array([2.0]),
            meas_var=0.5,
        )
        # Equal variance: K = 0.5, estimate = 1.5, var = 0.25.
        np.testing.assert_allclose(result["estimate"], [1.5])
        np.testing.assert_allclose(result["variance"], 0.25)

    def test_update_low_meas_var_pulls_toward_measurement(self):
        sf = SensorFuser(dim=8)
        result = sf.update(
            prior=np.array([0.0]),
            prior_var=10.0,
            measurement=np.array([5.0]),
            meas_var=0.1,
        )
        # K = 10 / 10.1 ≈ 0.99, estimate ≈ 4.95
        assert result["estimate"][0] > 4.5

    def test_update_zero_variance_clamped(self):
        sf = SensorFuser(dim=8)
        # Variance clamped to 1e-12 to avoid division by zero.
        result = sf.update(
            prior=np.array([1.0]),
            prior_var=0.0,
            measurement=np.array([2.0]),
            meas_var=0.0,
        )
        # Both variances clamped to 1e-12, K = 0.5, estimate = 1.5.
        np.testing.assert_allclose(result["estimate"], [1.5])


# --------------------------------------------------------------------------- #
# GaitGenerator
# --------------------------------------------------------------------------- #

class TestGaitGenerator:
    def test_generate_walk_shape(self):
        gg = GaitGenerator(dim=8)
        result = gg.generate(n_steps=50, gait_type="walk")
        # 4 legs * 2 DOF per leg = 8 DOF.
        assert result["joint_angles"].shape == (50, 8)
        assert result["foot_contacts"].shape == (50, 4)
        assert result["period"] == 1.0

    def test_generate_trot_pattern(self):
        gg = GaitGenerator(dim=8)
        result = gg.generate(n_steps=100, gait_type="trot")
        # In trot, diagonal legs are in sync (legs 0, 3 same phase; 1, 2 same).
        contacts = result["foot_contacts"]
        # At t=0, leg 0 and leg 3 should have phase 0 (contact), legs 1 and 2
        # have phase 0.5 (swing).
        assert contacts[0, 0] == 1.0  # leg 0 in stance
        assert contacts[0, 3] == 1.0  # leg 3 in stance

    def test_generate_bound_pattern(self):
        gg = GaitGenerator(dim=8)
        result = gg.generate(n_steps=50, gait_type="bound")
        # Bound: front legs (0, 1) and back legs (2, 3) alternate.
        assert result["joint_angles"].shape == (50, 8)

    def test_generate_invalid_gait_falls_back_to_walk(self):
        gg = GaitGenerator(dim=8)
        result = gg.generate(n_steps=20, gait_type="nonexistent")
        # Should not raise; falls back to walk.
        assert result["joint_angles"].shape == (20, 8)

    def test_generate_joint_angles_bounded(self):
        gg = GaitGenerator(dim=8)
        result = gg.generate(n_steps=100, gait_type="walk")
        # Hip is sin * 0.3, knee is sin * 0.4 -> all in [-0.4, 0.4].
        assert np.all(np.abs(result["joint_angles"]) <= 0.5)

    def test_generate_foot_contacts_binary(self):
        gg = GaitGenerator(dim=8)
        result = gg.generate(n_steps=50, gait_type="trot")
        contacts = result["foot_contacts"]
        # All contact values should be 0.0 or 1.0.
        assert np.all(np.isin(contacts, [0.0, 1.0]))

    def test_generate_preserves_ca_state(self):
        """CA state should be restored after gait generation (R9-010)."""
        gg = GaitGenerator(dim=8)
        ca = gg.biological.automata
        original_state = ca.state.copy()
        original_rule = ca.rule
        gg.generate(n_steps=50, gait_type="walk")
        np.testing.assert_array_equal(ca.state, original_state)
        assert ca.rule == original_rule

    def test_generate_one_step(self):
        gg = GaitGenerator(dim=8)
        result = gg.generate(n_steps=1, gait_type="walk")
        assert result["joint_angles"].shape == (1, 8)
        assert result["foot_contacts"].shape == (1, 4)
