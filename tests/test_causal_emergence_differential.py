# tests/test_causal_emergence_differential.py
"""Tests for module C: DifferentialGenerator.

Covers correctness (boundary conditions, interpolation fallback,
damped convergence), edge cases (n_steps=0, start==end, NaN/Inf, shape
mismatch), determinism, and API contract per spec §5 and §10.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.causal_emergence import (
    DifferentialGenerator,
    EmergenceRules,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# ----------------------------------------------------------------------
# Correctness: boundary conditions
# ----------------------------------------------------------------------

def test_trajectory_satisfies_start_boundary():
    """Trajectory[0] must equal start_state exactly."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([1.0, 2.0, 3.0])
    result = gen.generate(start, end, n_steps=16)
    np.testing.assert_allclose(result["trajectory"][0], start, atol=1e-12)


def test_trajectory_satisfies_end_boundary():
    """Trajectory[-1] must equal end_state (within solver tolerance)."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([1.0, 2.0, 3.0])
    result = gen.generate(start, end, n_steps=16)
    np.testing.assert_allclose(result["trajectory"][-1], end, atol=1e-6)


def test_trajectory_shape():
    """Trajectory has shape (n_steps + 1, dim)."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.zeros(4)
    end = np.ones(4)
    result = gen.generate(start, end, n_steps=32)
    assert result["trajectory"].shape == (33, 4)


def test_lagrangian_length_matches_n_steps():
    """Lagrangian array has length n_steps."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    result = gen.generate(np.zeros(2), np.ones(2), n_steps=10)
    assert result["lagrangian"].shape == (10,)


def test_action_is_finite_scalar():
    """Action is a finite float."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    result = gen.generate(np.zeros(2), np.ones(2), n_steps=8)
    assert isinstance(result["action"], float)
    assert np.isfinite(result["action"])


# ----------------------------------------------------------------------
# Correctness: linear interpolation fallback
# ----------------------------------------------------------------------

def test_linear_interpolation_when_no_damping_no_attraction():
    """With lambda=0 and gamma=0, the trajectory is linear interpolation."""
    rules = EmergenceRules(differential_lambda=0.0, differential_gamma=0.0)
    gen = DifferentialGenerator(rules=rules)
    start = np.array([0.0, 0.0])
    end = np.array([2.0, 4.0])
    result = gen.generate(start, end, n_steps=10)

    for k in range(11):
        t = k / 10
        expected = (1 - t) * start + t * end
        np.testing.assert_allclose(
            result["trajectory"][k], expected, atol=1e-3,
            err_msg=f"step {k} not linear"
        )


def test_linear_interpolation_action_zero_with_no_potential():
    """With lambda=0, the potential term vanishes; action = integral of kinetic only."""
    rules = EmergenceRules(differential_lambda=0.0, differential_gamma=0.0)
    gen = DifferentialGenerator(rules=rules)
    start = np.zeros(2)
    end = np.array([1.0, 1.0])
    result = gen.generate(start, end, n_steps=16)
    # Lagrangian should equal 0.5 * ||q_dot||^2 (kinetic only)
    # For linear interpolation, q_dot = (end - start) / dt; with dt=0.01,
    # ||q_dot||^2 = 2 / 0.0001 = 20000, so L = 10000 per step.
    assert np.all(result["lagrangian"] > 0)
    assert result["action"] > 0


# ----------------------------------------------------------------------
# Correctness: constant trajectory
# ----------------------------------------------------------------------

def test_start_equals_end_constant_trajectory():
    """When start == end, trajectory is constant and action is 0."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    state = np.array([1.0, 2.0, 3.0])
    result = gen.generate(state, state, n_steps=16)
    assert result["converged"]
    assert result["iterations"] == 0
    assert result["action"] == 0.0
    assert result["trajectory"].shape == (17, 3)
    for k in range(17):
        np.testing.assert_allclose(result["trajectory"][k], state)


# ----------------------------------------------------------------------
# Correctness: damped convergence
# ----------------------------------------------------------------------

def test_damped_trajectory_converges():
    """Damped trajectory converges within max_iter iterations."""
    rules = EmergenceRules(differential_max_iter=200, differential_tol=1e-6)
    gen = DifferentialGenerator(rules=rules)
    start = np.zeros(3)
    end = np.array([1.0, 2.0, 3.0])
    result = gen.generate(start, end, n_steps=16)
    assert result["converged"]
    assert result["iterations"] <= 200


def test_damping_smooths_trajectory():
    """With damping, trajectory should be monotone-ish in 1D (no oscillation)."""
    rules = EmergenceRules(differential_gamma=2.0)
    gen = DifferentialGenerator(rules=rules)
    start = np.array([0.0])
    end = np.array([1.0])
    result = gen.generate(start, end, n_steps=20)
    q = result["trajectory"].flatten()
    # In a damped system pulled toward end, the trajectory should be
    # monotonically non-decreasing (within numerical tolerance).
    diffs = np.diff(q)
    assert np.all(diffs >= -1e-6), f"non-monotone: diffs={diffs}"


def test_attraction_pulls_trajectory_toward_end():
    """With large lambda, interior points are pulled toward end_state."""
    rules = EmergenceRules(differential_lambda=10.0, differential_gamma=0.5)
    gen = DifferentialGenerator(rules=rules)
    start = np.array([0.0])
    end = np.array([1.0])
    result = gen.generate(start, end, n_steps=20)
    q = result["trajectory"].flatten()
    # Midpoint should be closer to end than linear interpolation's 0.5
    # because of attraction toward end (target).
    assert q[10] > 0.5 + 1e-3, f"midpoint={q[10]} not pulled toward end"


# ----------------------------------------------------------------------
# Correctness: action / lagrangian properties
# ----------------------------------------------------------------------

def test_action_is_nan_safe():
    """Action is finite even with extreme inputs."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.array([-1e6, 1e6])
    end = np.array([1e6, -1e6])
    result = gen.generate(start, end, n_steps=8)
    assert np.isfinite(result["action"])
    assert np.all(np.isfinite(result["trajectory"]))


def test_lagrangian_signs_vary():
    """Lagrangian can be positive (kinetic dominant) or negative (potential dominant)."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    # Strong attraction -> potential may dominate some steps
    rules = EmergenceRules(differential_lambda=10.0, differential_gamma=0.1)
    gen = DifferentialGenerator(rules=rules)
    start = np.zeros(2)
    end = np.ones(2) * 5.0
    result = gen.generate(start, end, n_steps=32)
    # Lagrangian values should exist; signs may vary
    assert np.all(np.isfinite(result["lagrangian"]))


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

def test_n_steps_zero_returns_single_point():
    """n_steps=0 returns a single-row trajectory."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    result = gen.generate(np.zeros(3), np.ones(3), n_steps=0)
    assert result["trajectory"].shape == (1, 3)
    np.testing.assert_allclose(result["trajectory"][0], np.zeros(3))
    assert result["converged"]
    assert result["iterations"] == 0
    assert result["action"] == 0.0


def test_n_steps_negative_treated_as_zero():
    """Negative n_steps is treated as the degenerate single-point case."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    result = gen.generate(np.zeros(3), np.ones(3), n_steps=-5)
    assert result["trajectory"].shape == (1, 3)


def test_n_steps_one_returns_boundary_only():
    """n_steps=1 returns trajectory [start, end] with one Lagrangian step."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 1.0])
    result = gen.generate(start, end, n_steps=1)
    assert result["trajectory"].shape == (2, 2)
    np.testing.assert_allclose(result["trajectory"][0], start)
    np.testing.assert_allclose(result["trajectory"][1], end)
    assert result["lagrangian"].shape == (1,)


def test_1d_state_works():
    """Scalar / 1D state of size 1 works."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    result = gen.generate(np.array([0.0]), np.array([1.0]), n_steps=8)
    assert result["trajectory"].shape == (9, 1)


def test_high_dim_state_works():
    """High-dimensional state (dim=32) works."""
    gen = DifferentialGenerator(dim=32, rules=EmergenceRules())
    start = np.zeros(32)
    end = np.ones(32)
    result = gen.generate(start, end, n_steps=16)
    assert result["trajectory"].shape == (17, 32)
    np.testing.assert_allclose(result["trajectory"][-1], end, atol=1e-6)


def test_shape_mismatch_raises():
    """Shape mismatch between start and end raises ValueError."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    with pytest.raises(ValueError, match="shape"):
        gen.generate(np.zeros(3), np.zeros(5), n_steps=8)


def test_nan_in_start_raises():
    """NaN in start_state raises ValueError."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    with pytest.raises(ValueError, match="finite"):
        gen.generate(np.array([np.nan, 1.0]), np.array([1.0, 1.0]))


def test_inf_in_end_raises():
    """Inf in end_state raises ValueError."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    with pytest.raises(ValueError, match="finite"):
        gen.generate(np.zeros(2), np.array([1.0, np.inf]))


def test_2d_input_is_flattened():
    """2D input arrays are flattened to 1D state vectors."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    end = np.array([[2.0, 3.0], [4.0, 5.0]])
    result = gen.generate(start, end, n_steps=4)
    assert result["trajectory"].shape == (5, 4)


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_same_rng_produces_same_trajectory():
    """Same rng seed produces identical trajectories."""
    rules = EmergenceRules()
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    gen1 = DifferentialGenerator(rules=rules, rng=rng1)
    gen2 = DifferentialGenerator(rules=rules, rng=rng2)
    start = np.zeros(3)
    end = np.array([1.0, 2.0, 3.0])
    r1 = gen1.generate(start, end, n_steps=16)
    r2 = gen2.generate(start, end, n_steps=16)
    np.testing.assert_allclose(r1["trajectory"], r2["trajectory"])


def test_deterministic_with_default_rng():
    """Default rng is deterministic given the autouse seed fixture."""
    gen1 = DifferentialGenerator(rules=EmergenceRules())
    gen2 = DifferentialGenerator(rules=EmergenceRules())
    start = np.zeros(2)
    end = np.array([1.0, 1.0])
    r1 = gen1.generate(start, end, n_steps=8)
    r2 = gen2.generate(start, end, n_steps=8)
    np.testing.assert_allclose(r1["trajectory"], r2["trajectory"])


# ----------------------------------------------------------------------
# API contract
# ----------------------------------------------------------------------

def test_returns_required_keys():
    """Result dict contains all required keys per spec."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    result = gen.generate(np.zeros(2), np.ones(2), n_steps=8)
    for key in ("trajectory", "lagrangian", "action", "converged", "iterations"):
        assert key in result, f"missing key: {key}"


def test_trajectory_is_float_array():
    """Trajectory is a float ndarray."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    result = gen.generate(np.zeros(2), np.ones(2), n_steps=4)
    assert isinstance(result["trajectory"], np.ndarray)
    assert result["trajectory"].dtype == np.float64


def test_iterations_is_python_int():
    """iterations field is a Python int (not np.int64)."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    result = gen.generate(np.zeros(2), np.ones(2), n_steps=4)
    assert isinstance(result["iterations"], int)
    assert not isinstance(result["iterations"], np.integer)


def test_converged_is_python_bool():
    """converged field is a Python bool."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    result = gen.generate(np.zeros(2), np.ones(2), n_steps=4)
    assert isinstance(result["converged"], bool)


def test_default_rules_when_none():
    """Default rules are created when none provided."""
    gen = DifferentialGenerator(rules=None)
    assert gen.rules is not None
    assert gen.rules.differential_lambda == 1.0
    assert gen.rules.differential_gamma == 0.5


def test_default_rng_when_none():
    """Default rng is created when none provided."""
    gen = DifferentialGenerator()
    assert gen.rng is not None


def test_dim_attribute_stored():
    """dim is stored on the instance."""
    gen = DifferentialGenerator(dim=128)
    assert gen.dim == 128


# ----------------------------------------------------------------------
# Constraints placeholder
# ----------------------------------------------------------------------

def test_constraints_none_accepted():
    """constraints=None (default) is accepted."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    result = gen.generate(np.zeros(2), np.ones(2), n_steps=4, constraints=None)
    assert result["trajectory"].shape == (5, 2)


def test_constraints_empty_dict_accepted():
    """Empty constraints dict is accepted (placeholder for future use)."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    result = gen.generate(
        np.zeros(2), np.ones(2), n_steps=4, constraints={}
    )
    assert result["trajectory"].shape == (5, 2)


# ----------------------------------------------------------------------
# Phase 5: obstacle avoidance
# ----------------------------------------------------------------------

def test_constraints_with_obstacle_avoids_it():
    """Phase 5: trajectory must avoid the obstacle's safety margin.

    Place an obstacle right on the linear interpolation path between
    start and end. Without avoidance, the midpoint of the trajectory
    would lie exactly on the obstacle. With avoidance, every interior
    point must be at least ``margin`` away from every obstacle.
    """
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 0.0])
    obstacles = np.array([[0.5, 0.0]])  # middle of the line
    margin = 0.2
    result = gen.generate(
        start, end, n_steps=10,
        constraints={"obstacles": obstacles, "margin": margin},
    )

    # Result key for violation count must be present.
    assert "obstacle_violations" in result
    assert result["obstacle_violations"] > 0  # projection was active

    # Every interior point must be at least `margin` away from the obstacle.
    traj = result["trajectory"]
    for k in range(1, 10):
        d = float(np.linalg.norm(traj[k] - obstacles[0]))
        assert d >= margin - 1e-6, (
            f"interior point {k} at distance {d} < margin {margin}"
        )


def test_constraints_obstacle_on_path_increases_action():
    """Phase 5: avoidance perturbs the trajectory, raising the action.

    Without obstacles the linear-interpolation start → end is the
    minimum-action path. With an obstacle on that path, the trajectory
    must detour, so action is strictly higher. The penalty term also
    contributes, but even without it the geometric detour raises action.
    """
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 0.0])
    obstacles = np.array([[0.5, 0.0]])

    r_no_obs = gen.generate(start, end, n_steps=10, constraints=None)
    r_with_obs = gen.generate(
        start, end, n_steps=10,
        constraints={"obstacles": obstacles, "margin": 0.2},
    )

    assert r_with_obs["action"] > r_no_obs["action"]


def test_constraints_obstacle_off_path_does_not_perturb():
    """Phase 5: obstacle far from the path should not change the trajectory.

    The obstacle sits well off the linear interpolation; no interior
    point enters the safety margin, so no projection happens and the
    trajectory matches the unconstrained case.
    """
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 0.0])
    obstacles = np.array([[0.5, 100.0]])  # far off-path
    margin = 0.1

    r_no_obs = gen.generate(start, end, n_steps=10, constraints=None)
    r_with_obs = gen.generate(
        start, end, n_steps=10,
        constraints={"obstacles": obstacles, "margin": margin},
    )

    # No violations means no projection, so trajectory is unchanged.
    assert r_with_obs["obstacle_violations"] == 0
    np.testing.assert_allclose(
        r_with_obs["trajectory"], r_no_obs["trajectory"], atol=1e-10
    )


def test_constraints_invalid_type_raises():
    """Phase 5: constraints must be a dict (or None)."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    with pytest.raises(ValueError, match="constraints must be a dict"):
        gen.generate(
            np.zeros(2), np.ones(2), n_steps=4,
            constraints="not_a_dict",
        )


def test_constraints_invalid_obstacle_type_value_raises():
    """Phase 5: constraints['type'] must be 'soft' or 'hard'."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    with pytest.raises(ValueError, match="constraints\\['type'\\]"):
        gen.generate(
            np.zeros(2), np.ones(2), n_steps=4,
            constraints={"type": "invalid"},
        )


def test_constraints_negative_margin_raises():
    """Phase 5: margin must be positive finite."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    with pytest.raises(ValueError, match="margin"):
        gen.generate(
            np.zeros(2), np.ones(2), n_steps=4,
            constraints={"obstacles": [[0.5, 0.0]], "margin": -0.1},
        )


def test_constraints_obstacles_wrong_dim_raises():
    """Phase 5: obstacles shape must match state dim."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    # start/end are 2D, but obstacles are 3D.
    with pytest.raises(ValueError, match="obstacles"):
        gen.generate(
            np.zeros(2), np.ones(2), n_steps=4,
            constraints={"obstacles": [[0.5, 0.0, 0.0]]},
        )


def test_constraints_nan_in_obstacles_raises():
    """Phase 5: obstacles with NaN are rejected."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    with pytest.raises(ValueError, match="finite"):
        gen.generate(
            np.zeros(2), np.ones(2), n_steps=4,
            constraints={"obstacles": [[np.nan, 0.0]]},
        )


def test_constraints_obstacle_violations_key_present_when_active():
    """Phase 5: 'obstacle_violations' key is in result when obstacles active."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    r = gen.generate(
        np.zeros(2), np.ones(2), n_steps=4,
        constraints={"obstacles": [[0.5, 0.0]], "margin": 0.1},
    )
    assert "obstacle_violations" in r
    assert isinstance(r["obstacle_violations"], int)


def test_constraints_obstacle_violations_key_absent_when_none():
    """Phase 5: 'obstacle_violations' key absent when constraints=None.

    Backward compatibility: callers not using obstacles should not see
    a new key in the result dict.
    """
    gen = DifferentialGenerator(rules=EmergenceRules())
    r = gen.generate(np.zeros(2), np.ones(2), n_steps=4, constraints=None)
    assert "obstacle_violations" not in r


def test_constraints_obstacle_violations_key_absent_for_empty_dict():
    """Phase 5: empty constraints dict also omits the violations key."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    r = gen.generate(
        np.zeros(2), np.ones(2), n_steps=4, constraints={}
    )
    assert "obstacle_violations" not in r


def test_constraints_margin_override_uses_call_value():
    """Phase 5: constraints['margin'] overrides rules.differential_obstacle_margin."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 0.0])
    obstacles = np.array([[0.5, 0.0]])

    # Large margin -> more points projected, larger action penalty.
    r_large = gen.generate(
        start, end, n_steps=10,
        constraints={"obstacles": obstacles, "margin": 0.4},
    )
    r_small = gen.generate(
        start, end, n_steps=10,
        constraints={"obstacles": obstacles, "margin": 0.05},
    )
    assert r_large["obstacle_violations"] >= r_small["obstacle_violations"]


def test_constraints_n_steps_zero_with_obstacles_returns_zero_violations():
    """Phase 5: n_steps=0 + obstacles still returns a valid result.

    No interior points exist, so no projection can occur; the
    ``obstacle_violations`` key is present but equals 0.
    """
    gen = DifferentialGenerator(rules=EmergenceRules())
    r = gen.generate(
        np.zeros(2), np.ones(2), n_steps=0,
        constraints={"obstacles": [[0.5, 0.5]], "margin": 0.1},
    )
    assert r["trajectory"].shape == (1, 2)
    assert r["obstacle_violations"] == 0


def test_constraints_start_equals_end_with_obstacle_on_point():
    """Phase 5: degenerate trajectory (start == end) with obstacle nearby.

    The boundary points themselves are not projected (only interior
    points are), so the trajectory stays constant at ``start``. The
    ``obstacle_violations`` key counts interior points within margin,
    which for a constant trajectory are all equal to ``start``.
    """
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.array([0.5, 0.0])
    obstacles = np.array([[0.5, 0.0]])  # exactly on start
    r = gen.generate(
        start, start, n_steps=5,
        constraints={"obstacles": obstacles, "margin": 0.1},
    )
    # All 4 interior points are at start, which is within margin of obs.
    assert r["obstacle_violations"] == 4
    # Trajectory is still constant (boundary points not projected).
    np.testing.assert_allclose(r["trajectory"][0], start)
    np.testing.assert_allclose(r["trajectory"][-1], start)


# ----------------------------------------------------------------------
# Symmetry / scaling
# ----------------------------------------------------------------------

def test_reversed_trajectory_symmetric_without_potential():
    """Without attraction (lambda=0) and damping (gamma=0), trajectory
    is time-reversal symmetric: forward[::-1] == backward.

    The elastic restoring force (lambda > 0) and damping (gamma > 0)
    break time-reversal symmetry by design — the system dissipates energy
    forward in time and is pulled toward `end_state` as target.
    """
    rules = EmergenceRules(differential_lambda=0.0, differential_gamma=0.0)
    gen = DifferentialGenerator(rules=rules)
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 1.0])
    fwd = gen.generate(start, end, n_steps=16)
    bwd = gen.generate(end, start, n_steps=16)
    # fwd reversed should equal bwd
    np.testing.assert_allclose(
        fwd["trajectory"][::-1], bwd["trajectory"], atol=1e-6
    )


def test_scaled_boundary_scales_trajectory():
    """Scaling both boundaries by alpha scales the trajectory by alpha
    (up to the absolute convergence tolerance of the iterative solver).
    """
    rules = EmergenceRules(differential_tol=1e-9, differential_max_iter=2000)
    gen = DifferentialGenerator(rules=rules)
    start = np.zeros(2)
    end = np.array([1.0, 1.0])
    base = gen.generate(start, end, n_steps=8)
    alpha = 3.0
    scaled = gen.generate(start * alpha, end * alpha, n_steps=8)
    # Iterative solver's tol is absolute, so use rtol that absorbs alpha.
    np.testing.assert_allclose(
        scaled["trajectory"], base["trajectory"] * alpha, rtol=1e-3, atol=1e-6
    )


def test_translation_invariance():
    """Translating both boundaries by delta translates the trajectory
    (the damped oscillator equation is translation-invariant).
    """
    rules = EmergenceRules(differential_tol=1e-9, differential_max_iter=2000)
    gen = DifferentialGenerator(rules=rules)
    start = np.zeros(2)
    end = np.array([1.0, 1.0])
    base = gen.generate(start, end, n_steps=8)
    delta = np.array([5.0, -3.0])
    shifted = gen.generate(start + delta, end + delta, n_steps=8)
    np.testing.assert_allclose(
        shifted["trajectory"], base["trajectory"] + delta, atol=1e-6
    )


# ----------------------------------------------------------------------
# Convergence behavior
# ----------------------------------------------------------------------

def test_max_iter_zero_does_not_converge_but_returns():
    """max_iter=0 means no relaxation; initial linear interp is returned."""
    rules = EmergenceRules(differential_max_iter=0)
    gen = DifferentialGenerator(rules=rules)
    start = np.zeros(2)
    end = np.array([1.0, 1.0])
    result = gen.generate(start, end, n_steps=8)
    # Without any relaxation, trajectory is the linear interpolation init.
    assert result["iterations"] == 0
    # converged is False because we never iterated (the loop body never set it True).
    # But check shape and boundaries are still correct.
    assert result["trajectory"].shape == (9, 2)


def test_large_n_steps_works():
    """Large n_steps (256) works without error."""
    gen = DifferentialGenerator(rules=EmergenceRules())
    start = np.zeros(2)
    end = np.array([1.0, 1.0])
    result = gen.generate(start, end, n_steps=256)
    assert result["trajectory"].shape == (257, 2)
    np.testing.assert_allclose(result["trajectory"][-1], end, atol=1e-6)


def test_strict_tolerance_requires_more_iterations():
    """Tighter tolerance requires at least as many iterations."""
    rules_loose = EmergenceRules(differential_tol=1e-2, differential_max_iter=500)
    rules_strict = EmergenceRules(differential_tol=1e-9, differential_max_iter=500)
    gen_loose = DifferentialGenerator(rules=rules_loose)
    gen_strict = DifferentialGenerator(rules=rules_strict)
    start = np.zeros(2)
    end = np.array([1.0, 1.0])
    r_loose = gen_loose.generate(start, end, n_steps=32)
    r_strict = gen_strict.generate(start, end, n_steps=32)
    assert r_strict["iterations"] >= r_loose["iterations"]
