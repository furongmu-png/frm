"""Comprehensive pytest unit tests for the embodied cognition modules.

Covers:
  - src/embodied/body_env.py             : EmbodiedBody, EmbodiedObservation, BodyAction
  - src/embodied/sensorimotor_predictor.py : SensorimotorPredictor, SensorimotorPrediction
  - src/embodied/proprioception.py       : ProprioceptiveEncoder, ProprioceptiveState
"""
from __future__ import annotations

import numpy as np
import pytest

from src.embodied.body_env import (
    EmbodiedBody,
    EmbodiedObservation,
    BodyAction,
    N_TOTAL_ACTIONS,
    N_PHYSICS_ACTIONS,
)
from src.embodied.sensorimotor_predictor import (
    SensorimotorPredictor,
    SensorimotorPrediction,
)
from src.embodied.proprioception import (
    ProprioceptiveEncoder,
    ProprioceptiveState,
)


# ================================================================== #
# EmbodiedBody
# ================================================================== #
class TestEmbodiedBody:
    """Tests for the first-person active-perception body environment."""

    def test_reset_returns_observation(self):
        """reset() returns EmbodiedObservation with proper shapes/dtypes."""
        body = EmbodiedBody(seed=42)
        obs = body.reset(seed=42)
        assert isinstance(obs, EmbodiedObservation), (
            "reset() must return an EmbodiedObservation instance"
        )
        # visual_frame: 2D uint8
        assert obs.visual_frame.ndim == 2, (
            f"visual_frame must be 2D, got {obs.visual_frame.ndim}D"
        )
        assert obs.visual_frame.dtype == np.uint8, (
            f"visual_frame must be uint8, got {obs.visual_frame.dtype}"
        )
        # tactile: 1D float
        assert obs.tactile.ndim == 1, (
            f"tactile must be 1D, got {obs.tactile.ndim}D"
        )
        assert np.issubdtype(obs.tactile.dtype, np.floating), (
            f"tactile must be floating-point, got {obs.tactile.dtype}"
        )
        # proprioception: 1D float with exactly 8 elements
        assert obs.proprioception.ndim == 1, (
            f"proprioception must be 1D, got {obs.proprioception.ndim}D"
        )
        assert obs.proprioception.shape == (8,), (
            f"proprioception must have 8 elements, got shape {obs.proprioception.shape}"
        )
        assert np.issubdtype(obs.proprioception.dtype, np.floating), (
            f"proprioception must be floating-point, got {obs.proprioception.dtype}"
        )

    def test_step_with_physics_action(self):
        """step(0..3) returns observation; body position changes (with sandbox)."""
        body = EmbodiedBody(seed=42)
        obs0 = body.reset(seed=42)
        pos_before = obs0.body_position.copy()
        moved = False
        for a in range(N_PHYSICS_ACTIONS):
            obs = body.step(a)
            assert isinstance(obs, EmbodiedObservation), (
                f"step({a}) must return an EmbodiedObservation"
            )
            if not np.allclose(obs.body_position, pos_before):
                moved = True
        # When the physics sandbox is attached, physics actions move the agent
        # body. Without a sandbox the body_position stays at its random init,
        # so only assert movement when the sandbox is available.
        snap = body.get_state_snapshot()
        if snap["sandbox_attached"]:
            assert moved, (
                "body_position should change after stepping physics actions "
                "when the sandbox is attached"
            )

    def test_step_with_perceptual_action(self):
        """step(N_PHYSICS_ACTIONS..N_TOTAL_ACTIONS-1) returns observation;
        gaze angles change for gaze actions."""
        body = EmbodiedBody(seed=42)
        body.reset(seed=42)
        # All perceptual actions must return a valid observation.
        for a in range(N_PHYSICS_ACTIONS, N_TOTAL_ACTIONS):
            obs = body.step(a)
            assert isinstance(obs, EmbodiedObservation), (
                f"step({a}) must return an EmbodiedObservation"
            )
        # Gaze actions (first 4 perceptual actions) should change gaze angles.
        body.reset(seed=42)
        yaw_before = body.gaze_yaw
        body.step(N_PHYSICS_ACTIONS)  # move_gaze_left
        assert body.gaze_yaw != yaw_before, (
            "move_gaze_left should change gaze_yaw"
        )

    def test_parse_action_types(self):
        """parse_action returns BodyAction with correct action_type."""
        body = EmbodiedBody(seed=42)
        # First N_PHYSICS_ACTIONS actions are "physics".
        for a in range(N_PHYSICS_ACTIONS):
            parsed = body.parse_action(a)
            assert isinstance(parsed, BodyAction), (
                f"parse_action({a}) must return a BodyAction"
            )
            assert parsed.action_type == "physics", (
                f"action {a} should have type 'physics', got '{parsed.action_type}'"
            )
        # Remaining actions are perceptual (move_gaze / touch_probe / apply_force).
        perceptual_types = {"move_gaze", "touch_probe", "apply_force"}
        for a in range(N_PHYSICS_ACTIONS, N_TOTAL_ACTIONS):
            parsed = body.parse_action(a)
            assert isinstance(parsed, BodyAction), (
                f"parse_action({a}) must return a BodyAction"
            )
            assert parsed.action_type in perceptual_types, (
                f"action {a} should be a perceptual type {perceptual_types}, "
                f"got '{parsed.action_type}'"
            )

    def test_action_space_size(self):
        """action_space_size property equals N_TOTAL_ACTIONS."""
        body = EmbodiedBody(seed=42)
        assert body.action_space_size == N_TOTAL_ACTIONS, (
            f"action_space_size must be {N_TOTAL_ACTIONS}, got {body.action_space_size}"
        )

    def test_invalid_action_raises(self):
        """step() with out-of-range action raises ValueError."""
        body = EmbodiedBody(seed=42)
        body.reset(seed=42)
        with pytest.raises(ValueError):
            body.step(N_TOTAL_ACTIONS)
        with pytest.raises(ValueError):
            body.step(-1)

    def test_reset_with_seed_reproducible(self):
        """reset(seed=42) twice gives the same initial body position."""
        body1 = EmbodiedBody(seed=42)
        obs1 = body1.reset(seed=42)
        body2 = EmbodiedBody(seed=42)
        obs2 = body2.reset(seed=42)
        np.testing.assert_allclose(
            obs1.body_position, obs2.body_position,
            err_msg="reset(seed=42) should produce identical body_position",
        )
        assert obs1.gaze_yaw == obs2.gaze_yaw, (
            "reset(seed=42) should produce identical gaze_yaw"
        )
        assert obs1.gaze_pitch == obs2.gaze_pitch, (
            "reset(seed=42) should produce identical gaze_pitch"
        )

    def test_get_snapshot(self):
        """get_state_snapshot() returns dict with expected keys."""
        body = EmbodiedBody(seed=42)
        body.reset(seed=42)
        snap = body.get_state_snapshot()
        assert isinstance(snap, dict), "snapshot must be a dict"
        expected_keys = {
            "gaze_yaw",
            "gaze_pitch",
            "body_position",
            "joint_angles",
            "joint_velocities",
            "touched_object_id",
            "step",
            "sandbox_attached",
            "n_total_actions",
            "n_perceptual_actions",
        }
        missing = expected_keys - snap.keys()
        assert not missing, f"snapshot missing expected keys: {missing}"
        assert snap["n_total_actions"] == N_TOTAL_ACTIONS
        assert isinstance(snap["sandbox_attached"], bool)


# ================================================================== #
# SensorimotorPredictor
# ================================================================== #
class TestSensorimotorPredictor:
    """Tests for the sensorimotor contingency predictor."""

    def test_predict_returns_correct_shapes(self):
        """predict(belief, action_id) returns (vis_dim, n_probe) tuple."""
        pred = SensorimotorPredictor(dim=32, n_probe=16, seed=42)
        belief = np.ones(32)
        result = pred.predict(belief, action_id=0)
        assert isinstance(result, tuple) and len(result) == 2, (
            "predict must return a 2-tuple (visual, tactile)"
        )
        vis, tac = result
        assert vis.shape == (32,), (
            f"predicted visual change must have shape (32,), got {vis.shape}"
        )
        assert tac.shape == (16,), (
            f"predicted tactile must have shape (16,), got {tac.shape}"
        )

    def test_update_returns_prediction_with_errors(self):
        """update() returns SensorimotorPrediction with non-negative errors."""
        pred = SensorimotorPredictor(dim=32, n_probe=16, seed=42)
        belief = np.ones(32)
        vis_change = np.zeros(32)
        tac = np.zeros(16)
        result = pred.update(belief, 0, vis_change, tac)
        assert isinstance(result, SensorimotorPrediction), (
            "update must return a SensorimotorPrediction"
        )
        assert result.visual_error >= 0, (
            f"visual_error must be >= 0, got {result.visual_error}"
        )
        assert result.tactile_error >= 0, (
            f"tactile_error must be >= 0, got {result.tactile_error}"
        )
        assert result.total_error >= 0, (
            f"total_error must be >= 0, got {result.total_error}"
        )

    def test_error_decreases_over_training(self):
        """Running update() 50× with consistent input lowers avg_visual_error.

        NOTE: The SensorimotorPredictor couples W_vis/W_tac Hebbian updates
        with S4-layer weight updates, so the prediction target (S4 context)
        is non-stationary. The visual error decreases modestly (~1-2%) and
        plateaus rather than converging to zero. We therefore verify a
        measurable decrease (>1%) rather than the 5% used by simpler
        Hebbian-only modules.
        """
        pred = SensorimotorPredictor(dim=32, n_probe=16, lr=0.05, seed=42)
        belief = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] * 4)
        vis_change = np.ones(32) * 0.5
        tac = np.ones(16) * 0.3
        # Warm up 5 steps and capture the running average.
        for _ in range(5):
            pred.update(belief, 0, vis_change, tac)
        before = pred.get_snapshot()["avg_visual_error"]
        # Train 45 more steps.
        for _ in range(45):
            pred.update(belief, 0, vis_change, tac)
        after = pred.get_snapshot()["avg_visual_error"]
        assert after < before * 0.99, (
            f"avg_visual_error should decrease over training "
            f"(expect after < before*0.99): before={before:.6f}, after={after:.6f}"
        )

    def test_select_perceptual_action_returns_valid(self):
        """select_perceptual_action returns int in [0, n_actions)."""
        pred = SensorimotorPredictor(dim=32, n_probe=16, seed=42)
        belief = np.ones(32)
        action = pred.select_perceptual_action(belief, n_actions=10)
        assert isinstance(action, (int, np.integer)), (
            f"action must be an int, got {type(action)}"
        )
        assert 0 <= action < 10, (
            f"action must be in [0, 10), got {action}"
        )

    def test_select_perceptual_action_with_visited(self):
        """A heavily-visited action is selected less often due to exploration bonus."""
        pred = SensorimotorPredictor(dim=32, n_probe=16, seed=42)
        belief = np.ones(32)
        # Action 0 has been "visited" many times → low exploration bonus.
        visited = np.zeros(10)
        visited[0] = 1e6
        counts = np.zeros(10)
        n_trials = 200
        for _ in range(n_trials):
            a = pred.select_perceptual_action(
                belief, n_actions=10, visited=visited
            )
            counts[a] += 1
        # Action 0 should be selected well below the uniform rate (10%).
        assert counts[0] < n_trials * 0.15, (
            f"heavily-visited action 0 should be selected <15% of the time, "
            f"got {int(counts[0])}/{n_trials} ({100*counts[0]/n_trials:.1f}%)"
        )

    def test_invalid_dim_raises(self):
        """dim=0 raises ValueError."""
        with pytest.raises(ValueError):
            SensorimotorPredictor(dim=0)

    def test_invalid_n_probe_raises(self):
        """n_probe=0 raises ValueError."""
        with pytest.raises(ValueError):
            SensorimotorPredictor(n_probe=0)

    def test_get_snapshot(self):
        """get_snapshot() returns dict with step, avg_visual_error, etc."""
        pred = SensorimotorPredictor(dim=32, n_probe=16, seed=42)
        belief = np.ones(32)
        pred.update(belief, 0, np.zeros(32), np.zeros(16))
        snap = pred.get_snapshot()
        assert isinstance(snap, dict), "snapshot must be a dict"
        expected_keys = {
            "step",
            "avg_visual_error",
            "avg_tactile_error",
            "s4_state_norm",
            "s4_spectral_radius",
            "w_vis_norm",
            "w_tac_norm",
            "recent_error",
        }
        missing = expected_keys - snap.keys()
        assert not missing, f"snapshot missing expected keys: {missing}"
        assert snap["step"] == 1, f"step should be 1 after one update, got {snap['step']}"

    def test_reset_clears_state(self):
        """reset() clears the step count and error accumulators."""
        pred = SensorimotorPredictor(dim=32, n_probe=16, seed=42)
        belief = np.ones(32)
        pred.update(belief, 0, np.zeros(32), np.zeros(16))
        pred.update(belief, 0, np.zeros(32), np.zeros(16))
        assert pred.get_snapshot()["step"] == 2, "step should be 2 after two updates"
        pred.reset()
        snap = pred.get_snapshot()
        assert snap["step"] == 0, f"step should be 0 after reset, got {snap['step']}"
        assert snap["avg_visual_error"] == 0.0, (
            f"avg_visual_error should be 0.0 after reset, got {snap['avg_visual_error']}"
        )


# ================================================================== #
# ProprioceptiveEncoder
# ================================================================== #
class TestProprioceptiveEncoder:
    """Tests for the proprioceptive / body-schema encoder."""

    def test_encode_returns_correct_dim(self):
        """encode(raw) returns array of output_dim."""
        enc = ProprioceptiveEncoder(input_dim=8, output_dim=32, seed=42)
        raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        encoded = enc.encode(raw)
        assert encoded.shape == (32,), (
            f"encoded must have shape (32,), got {encoded.shape}"
        )

    def test_step_returns_state_with_fields(self):
        """step(raw) returns ProprioceptiveState with all expected fields."""
        enc = ProprioceptiveEncoder(input_dim=8, output_dim=32, seed=42)
        raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        state = enc.step(raw)
        assert isinstance(state, ProprioceptiveState), (
            "step must return a ProprioceptiveState"
        )
        assert state.encoded.shape == (32,), (
            f"encoded must have shape (32,), got {state.encoded.shape}"
        )
        assert state.raw.shape == (8,), (
            f"raw must have shape (8,), got {state.raw.shape}"
        )
        assert isinstance(state.prediction_error, float), (
            f"prediction_error must be a float, got {type(state.prediction_error)}"
        )
        assert isinstance(state.body_schema_confidence, float), (
            f"body_schema_confidence must be a float, "
            f"got {type(state.body_schema_confidence)}"
        )

    def test_prediction_error_decreases_over_time(self):
        """Training with consistent input lowers avg_prediction_error."""
        enc = ProprioceptiveEncoder(input_dim=8, output_dim=32, lr=0.05, seed=42)
        raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        # Warm up 5 steps and capture the running average.
        for _ in range(5):
            enc.step(raw)
        before = enc.get_body_schema_snapshot()["avg_prediction_error"]
        # Train 45 more steps.
        for _ in range(45):
            enc.step(raw)
        after = enc.get_body_schema_snapshot()["avg_prediction_error"]
        assert after < before * 0.95, (
            f"avg_prediction_error should decrease over training "
            f"(expect after < before*0.95): before={before:.6f}, after={after:.6f}"
        )

    def test_body_schema_confidence_increases(self):
        """After training, body_schema_confidence exceeds the initial value."""
        enc = ProprioceptiveEncoder(input_dim=8, output_dim=32, lr=0.05, seed=42)
        raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        # First step: high prediction error → low confidence.
        state0 = enc.step(raw)
        initial_confidence = state0.body_schema_confidence
        assert initial_confidence > 0.0, "confidence must be positive"
        # Train 49 more steps; confidence should rise as error drops.
        for _ in range(49):
            enc.step(raw)
        final_state = enc.step(raw)
        assert final_state.body_schema_confidence > initial_confidence, (
            f"body_schema_confidence should increase after training: "
            f"initial={initial_confidence:.6f}, "
            f"final={final_state.body_schema_confidence:.6f}"
        )
        assert final_state.body_schema_confidence > 0.0, (
            "body_schema_confidence must remain > 0.0 after training"
        )

    def test_handles_invalid_input(self):
        """step(np.array([np.nan]*8)) does not crash and returns a valid state."""
        enc = ProprioceptiveEncoder(input_dim=8, output_dim=32, seed=42)
        raw_nan = np.array([np.nan] * 8)
        # Should not raise; NaN is replaced internally with zeros.
        state = enc.step(raw_nan)
        assert isinstance(state, ProprioceptiveState), (
            "step with NaN input must still return a ProprioceptiveState"
        )
        assert np.all(np.isfinite(state.encoded)), (
            "encoded must be finite even with NaN input"
        )
        assert np.all(np.isfinite(state.raw)), (
            "raw must be finite (NaN replaced with zeros)"
        )
        assert np.isfinite(state.prediction_error), (
            "prediction_error must be finite with NaN input"
        )

    def test_invalid_dims_raise(self):
        """input_dim=0 or output_dim=0 raises ValueError."""
        with pytest.raises(ValueError):
            ProprioceptiveEncoder(input_dim=0)
        with pytest.raises(ValueError):
            ProprioceptiveEncoder(output_dim=0)

    def test_get_body_schema_snapshot(self):
        """get_body_schema_snapshot() returns dict with expected keys."""
        enc = ProprioceptiveEncoder(input_dim=8, output_dim=32, seed=42)
        raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        enc.step(raw)
        snap = enc.get_body_schema_snapshot()
        assert isinstance(snap, dict), "snapshot must be a dict"
        expected_keys = {
            "step",
            "avg_prediction_error",
            "schema_confidence",
            "schema_spectral_radius",
            "prev_state",
            "recent_error",
        }
        missing = expected_keys - snap.keys()
        assert not missing, f"snapshot missing expected keys: {missing}"
        assert snap["step"] == 1, f"step should be 1 after one step, got {snap['step']}"
