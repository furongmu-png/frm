"""Tests for the Causal/Decision capability domain (advanced module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    CausalGraphBuilder,
    CausalRules,
    InterventionAnalyzer,
    POMDPApproximator,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# --------------------------------------------------------------------------- #
# POMDPApproximator
# --------------------------------------------------------------------------- #


def test_pomdp_deterministic_transitions():
    """A 2-state deterministic MDP converges to a stable value function."""
    pomdp = POMDPApproximator()
    # Action 0: stay in current state; action 1: switch state.
    T = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    obs = np.eye(2)
    R = np.array([1.0, 0.0])  # state 0 has reward 1, state 1 has reward 0.
    result = pomdp.solve(T, obs, R)
    assert result["converged"] is True
    # State 0 (high reward) should have higher value than state 1.
    assert result["value"][0] > result["value"][1]
    # Policy: action 0 (stay) in both states is optimal.
    assert result["policy"][0] == 0
    assert result["policy"][1] == 0  # stay in state 0 too


def test_pomdp_invalid_transitions_shape():
    """Non-3D transition tensors return an empty solution."""
    pomdp = POMDPApproximator()
    result = pomdp.solve(np.array([1, 2, 3]), np.eye(2), np.array([1.0, 0.0]))
    assert result["converged"] is False
    assert result["iterations"] == 0
    assert result["policy"].size == 0


def test_pomdp_mismatched_shape():
    """Non-square transitions (S != S') are rejected."""
    pomdp = POMDPApproximator()
    # T shape: (2, 2, 3) -- invalid.
    T = np.zeros((2, 2, 3))
    result = pomdp.solve(T, np.eye(2), np.array([1.0, 0.0]))
    assert result["converged"] is False
    assert result["iterations"] == 0


def test_pomdp_reward_per_state_action():
    """Per-(state, action) rewards override the per-state reward vector."""
    pomdp = POMDPApproximator()
    T = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    obs = np.eye(2)
    # Per-(state, action) reward: state 0 action 0 = 10.
    R = np.array([[10.0, 0.0], [0.0, 0.0]])
    result = pomdp.solve(T, obs, R)
    assert result["converged"] is True
    # State 0 should have a high value because action 0 gives reward 10
    # and stays in state 0.
    assert result["value"][0] > 5.0
    assert result["policy"][0] == 0


def test_pomdp_returns_dict_structure():
    pomdp = POMDPApproximator()
    T = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    result = pomdp.solve(T, np.eye(2), np.array([1.0, 0.0]))
    assert set(result.keys()) == {"policy", "value", "iterations", "converged"}


def test_pomdp_gamma_discount():
    """Value is bounded by 1/(1-gamma) for unit rewards."""
    pomdp = POMDPApproximator()
    # Always stay in state 0 with reward 1.
    T = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    result = pomdp.solve(T, np.eye(2), np.array([1.0, 0.0]))
    # V[0] = 1 + gamma * 1 + gamma^2 * 1 + ... = 1 / (1 - gamma).
    assert result["value"][0] == pytest.approx(1.0 / (1.0 - POMDPApproximator.GAMMA), rel=1e-3)


# --------------------------------------------------------------------------- #
# CausalGraphBuilder
# --------------------------------------------------------------------------- #


def test_causal_graph_strong_correlation():
    """Strongly correlated variables produce an edge."""
    cg = CausalGraphBuilder()
    rng = np.random.RandomState(0)
    data = rng.randn(100, 3)
    # Make column 1 strongly correlated with column 0.
    data[:, 1] = data[:, 0] * 0.95 + 0.05 * rng.randn(100)
    result = cg.discover(data)
    assert result["n_edges"] >= 1
    # Edge should be (0, 1) since 0 < 1.
    assert (0, 1) in result["edges"]


def test_causal_graph_uncorrelated_no_edges():
    """Uncorrelated data produces no edges."""
    cg = CausalGraphBuilder(rules=CausalRules(causal_significance=0.9))
    rng = np.random.RandomState(0)
    data = rng.randn(100, 4)
    result = cg.discover(data)
    assert result["n_edges"] == 0
    assert result["edges"] == []


def test_causal_graph_invalid_shape():
    """1D data returns an empty graph."""
    cg = CausalGraphBuilder()
    result = cg.discover(np.array([1, 2, 3, 4]))
    assert result["n_edges"] == 0
    assert result["adjacency"].shape == (0, 0)


def test_causal_graph_too_few_samples():
    """Data with fewer than 2 samples returns an empty graph."""
    cg = CausalGraphBuilder()
    result = cg.discover(np.array([[1.0, 2.0]]))
    assert result["n_edges"] == 0


def test_causal_graph_var_names():
    """Custom variable names are reflected in the output."""
    cg = CausalGraphBuilder()
    rng = np.random.RandomState(0)
    data = rng.randn(50, 2)
    result = cg.discover(data, var_names=["temperature", "ice_cream_sales"])
    assert result["var_names"] == ["temperature", "ice_cream_sales"]


def test_causal_graph_default_var_names():
    """Without var_names, indices are used."""
    cg = CausalGraphBuilder()
    rng = np.random.RandomState(0)
    data = rng.randn(50, 3)
    result = cg.discover(data)
    assert result["var_names"] == ["0", "1", "2"]


def test_causal_graph_returns_dict_structure():
    cg = CausalGraphBuilder()
    rng = np.random.RandomState(0)
    data = rng.randn(20, 2)
    result = cg.discover(data)
    assert set(result.keys()) == {"adjacency", "edges", "n_edges", "var_names"}


def test_causal_graph_constant_column():
    """Constant columns (zero variance) do not crash the builder."""
    cg = CausalGraphBuilder()
    data = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    result = cg.discover(data)
    assert isinstance(result["n_edges"], int)


# --------------------------------------------------------------------------- #
# InterventionAnalyzer
# --------------------------------------------------------------------------- #


def test_intervention_basic():
    """Intervening on a column shifts only that column's mean."""
    ia = InterventionAnalyzer()
    rng = np.random.RandomState(0)
    data = rng.randn(100, 3)
    pre = np.mean(data, axis=0)
    result = ia.intervene(data, intervention_var=0, intervention_value=5.0)
    np.testing.assert_allclose(result["pre_intervention_mean"], pre)
    # Post mean for col 0 = 5.0; other cols unchanged.
    assert result["post_intervention_mean"][0] == pytest.approx(5.0)
    np.testing.assert_allclose(
        result["post_intervention_mean"][1:], pre[1:]
    )
    # Effect: col 0 = 5 - pre[0]; others = 0.
    assert result["effect"][0] == pytest.approx(5.0 - pre[0])
    assert result["effect"][1] == pytest.approx(0.0)
    assert result["effect"][2] == pytest.approx(0.0)
    assert result["surprisal"] >= 0.0


def test_intervention_empty_data():
    ia = InterventionAnalyzer()
    result = ia.intervene(np.array([]), intervention_var=0, intervention_value=1.0)
    assert result["effect"].size == 0
    assert result["surprisal"] == 0.0


def test_intervention_out_of_range_var():
    """Out-of-range intervention_var returns zeros."""
    ia = InterventionAnalyzer()
    rng = np.random.RandomState(0)
    data = rng.randn(10, 2)
    result = ia.intervene(data, intervention_var=99, intervention_value=5.0)
    assert np.all(result["effect"] == 0.0)


def test_intervention_1d_data_rejected():
    """1D data is rejected (must be 2D)."""
    ia = InterventionAnalyzer()
    result = ia.intervene(np.array([1.0, 2.0, 3.0]), 0, 5.0)
    assert result["effect"].size == 0


def test_intervention_returns_dict_structure():
    ia = InterventionAnalyzer()
    rng = np.random.RandomState(0)
    data = rng.randn(20, 2)
    result = ia.intervene(data, 0, 5.0)
    assert set(result.keys()) == {
        "pre_intervention_mean",
        "post_intervention_mean",
        "effect",
        "surprisal",
    }


def test_intervention_preserves_original_data():
    """The original data array is not mutated by the intervention."""
    ia = InterventionAnalyzer()
    rng = np.random.RandomState(0)
    data = rng.randn(10, 3)
    data_copy = data.copy()
    _ = ia.intervene(data, 0, 5.0)
    np.testing.assert_array_equal(data, data_copy)
