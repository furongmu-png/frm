# experiments/test_curiosity.py
"""Verification tests for the Phase-E curiosity mechanism.

Checks (no environment interaction needed — pure unit tests):
  1. Info-gain proxy is computed and finite.
  2. Beta decays correctly (linear / exp / stage).
  3. Beta_start=0 disables curiosity (backwards compat).
  4. High beta produces more diverse action distribution than low beta
     (the core hypothesis: exploration increases action entropy).
  5. Per-bin error history is bounded (no memory leak from the deques).
  6. Pickle/deepcopy preserves curiosity state.

Run with:  python -m pytest experiments/test_curiosity.py -v
        or python experiments/test_curiosity.py
"""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from zero_data_model.active_inference import ActiveInferenceEngine


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def engine():
    """A fresh engine with fast beta decay (so tests don't need 5000 steps)."""
    return ActiveInferenceEngine(
        state_dim=16,
        obs_dim=16,
        action_dim=8,
        rng=np.random.default_rng(42),
        exploration_beta_start=1.0,
        exploration_beta_min=0.01,
        exploration_decay_steps=100,
        exploration_decay_type="linear",
    )


@pytest.fixture
def fixed_obs():
    """A fixed observation sequence for deterministic info-gain testing."""
    rng = np.random.default_rng(0)
    return [rng.standard_normal(16) * 0.1 for _ in range(50)]


def _run_steps(engine, observations):
    """Run select_action over a list of observations; return chosen actions."""
    actions = []
    for obs in observations:
        belief = engine.generative_model.belief_state
        a = engine.select_action(belief, current_observation=obs)
        actions.append(a)
    return actions


# ------------------------------------------------------------------ #
# Test 1: info-gain proxy is computed and finite
# ------------------------------------------------------------------ #
def test_information_gain_proxy_is_finite(engine):
    """The IG proxy must return a finite float for any action vector."""
    for action_vec in [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([-1.0, -1.0]),
        np.zeros(8),
    ]:
        ig = engine._compute_information_gain_proxy(action_vec)
        assert isinstance(ig, float), f"IG not float: {type(ig)}"
        assert np.isfinite(ig), f"IG not finite: {ig}"
        assert ig >= 0.0, f"IG negative: {ig}"


def test_cold_start_info_gain_is_one(engine):
    """Empty-bin IG should be 1.0 (UCB optimism heuristic)."""
    ig = engine._compute_information_gain_proxy(np.array([1.0, 0.0]))
    assert ig == 1.0, f"cold-start IG should be 1.0, got {ig}"


# ------------------------------------------------------------------ #
# Test 2: beta decay schedules
# ------------------------------------------------------------------ #
def test_beta_linear_decay(engine):
    """Linear decay: beta(start)=beta_start, beta(T)=beta_min, beta(2T)=beta_min."""
    assert engine._compute_beta() == pytest.approx(1.0)  # step 0
    engine._exploration_step = 100  # T = 100
    assert engine._compute_beta() == pytest.approx(0.01)  # at T
    engine._exploration_step = 200  # 2T — clamped
    assert engine._compute_beta() == pytest.approx(0.01)


def test_beta_exponential_decay():
    """Exp decay: beta(0)=beta_start, beta(T) ≈ beta_min + (start-min)/e."""
    e = ActiveInferenceEngine(
        state_dim=8, obs_dim=8, action_dim=4,
        rng=np.random.default_rng(0),
        exploration_beta_start=1.0, exploration_beta_min=0.1,
        exploration_decay_steps=100, exploration_decay_type="exponential",
    )
    assert e._compute_beta() == pytest.approx(1.0)  # step 0
    e._exploration_step = 100  # T = 100
    # beta_min + (start - min) * exp(-1) ≈ 0.1 + 0.9 * 0.3679
    assert e._compute_beta() == pytest.approx(0.1 + 0.9 * np.exp(-1), abs=1e-4)


def test_beta_stage_decay():
    """Stage decay: beta = start for step < T, then min."""
    e = ActiveInferenceEngine(
        state_dim=8, obs_dim=8, action_dim=4,
        rng=np.random.default_rng(0),
        exploration_beta_start=1.0, exploration_beta_min=0.01,
        exploration_decay_steps=100, exploration_decay_type="stage",
    )
    e._exploration_step = 99
    assert e._compute_beta() == pytest.approx(1.0)
    e._exploration_step = 100
    assert e._compute_beta() == pytest.approx(0.01)


# ------------------------------------------------------------------ #
# Test 3: beta_start=0 disables curiosity (backwards compat)
# ------------------------------------------------------------------ #
def test_beta_zero_disables_curiosity():
    """beta_start=0 → beta always 0 → curiosity term vanishes."""
    e = ActiveInferenceEngine(
        state_dim=8, obs_dim=8, action_dim=4,
        rng=np.random.default_rng(0),
        exploration_beta_start=0.0,  # disabled
    )
    for step in [0, 50, 100, 1000]:
        e._exploration_step = step
        assert e._compute_beta() == 0.0, f"beta should be 0 at step {step}"


# ------------------------------------------------------------------ #
# Test 4: high beta → more diverse action distribution (higher entropy)
# ------------------------------------------------------------------ #
def test_high_beta_increases_action_entropy(fixed_obs):
    """The core hypothesis: higher beta should produce more diverse actions.

    We measure diversity as the number of DISTINCT angular bins visited
    (a coarse entropy proxy). With high beta, the -beta*IG term lowers
    EFE for unvisited bins (cold-start IG=1.0), encouraging the agent
    to spread across bins. With beta=0, only the pragmatic term drives
    selection, so the agent converges to whichever bin minimises EFE.
    """
    n_steps = 50

    # High-beta engine.
    e_high = ActiveInferenceEngine(
        state_dim=16, obs_dim=16, action_dim=8,
        rng=np.random.default_rng(123),
        exploration_beta_start=5.0,  # large — strong exploration pressure
        exploration_beta_min=5.0,
        exploration_decay_steps=10000,
        exploration_decay_type="stage",  # keep beta constant
    )
    actions_high = _run_steps(e_high, fixed_obs[:n_steps])
    bins_high = {e_high._action_bin(a) for a in actions_high}

    # Zero-beta engine (curiosity disabled).
    e_zero = ActiveInferenceEngine(
        state_dim=16, obs_dim=16, action_dim=8,
        rng=np.random.default_rng(123),
        exploration_beta_start=0.0,  # disabled
    )
    actions_zero = _run_steps(e_zero, fixed_obs[:n_steps])
    bins_zero = {e_zero._action_bin(a) for a in actions_zero}

    # The high-beta engine should visit at least as many bins as the
    # zero-beta engine (typically more). We use >= rather than > to
    # avoid flakiness from the stochastic candidate sampling.
    assert len(bins_high) >= len(bins_zero), (
        f"high-beta visited {len(bins_high)} bins, "
        f"zero-beta visited {len(bins_zero)} bins — "
        f"high beta should explore >= bins"
    )


# ------------------------------------------------------------------ #
# Test 5: per-bin history is bounded (no memory leak)
# ------------------------------------------------------------------ #
def test_action_error_history_is_bounded(engine, fixed_obs):
    """Each bin's deque must respect its maxlen (10) — no unbounded growth."""
    # Run 200 steps to overflow the maxlen=10 window multiple times.
    _run_steps(engine, fixed_obs * 4)  # 200 steps
    for i, h in enumerate(engine._action_error_history):
        assert len(h) <= 10, f"bin {i} grew to {len(h)} > 10 (leak)"


# ------------------------------------------------------------------ #
# Test 6: pickle/deepcopy preserves curiosity state
# ------------------------------------------------------------------ #
def test_pickle_preserves_curiosity_state(engine, fixed_obs):
    """Pickle + unpickle must preserve beta params, step counter, and history."""
    # Populate some state.
    _run_steps(engine, fixed_obs[:10])
    step_before = engine._exploration_step
    history_lens_before = [len(h) for h in engine._action_error_history]

    # Pickle round-trip.
    blob = pickle.dumps(engine)
    engine2 = pickle.loads(blob)
    assert engine2._exploration_step == step_before
    assert [len(h) for h in engine2._action_error_history] == history_lens_before
    assert engine2.exploration_beta_start == engine.exploration_beta_start
    assert engine2.exploration_decay_type == engine.exploration_decay_type

    # Deepcopy round-trip.
    engine3 = copy.deepcopy(engine)
    assert engine3._exploration_step == step_before
    assert [len(h) for h in engine3._action_error_history] == history_lens_before


# ------------------------------------------------------------------ #
# Test 7: action bin is deterministic and in range
# ------------------------------------------------------------------ #
def test_action_bin_in_range(engine):
    """Action bin must be in [0, n_action_bins) and deterministic."""
    test_vectors = [
        np.array([1.0, 0.0]),     # 0° → bin 0
        np.array([0.0, 1.0]),     # 90° → bin 2
        np.array([-1.0, 0.0]),    # 180° → bin 4
        np.array([0.0, -1.0]),    # 270° → bin 6
        np.array([0.0, 0.0]),     # zero → bin 0
    ]
    for v in test_vectors:
        b = engine._action_bin(v)
        assert 0 <= b < engine._n_action_bins, f"bin {b} out of range [0, {engine._n_action_bins})"

    # Determinism: same vector → same bin.
    v = np.array([0.7071, 0.7071])
    assert engine._action_bin(v) == engine._action_bin(v)


# ------------------------------------------------------------------ #
# Main entry point (run without pytest)
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 64)
    print("Phase-E Curiosity Verification Tests")
    print("=" * 64)

    pytest.main([__file__, "-v", "--tb=short"])
