"""Tests for the Causal/Decision capability domain (base module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    CausalRules,
    CounterfactualReasoner,
    DecisionTreeBuilder,
    GameTheoryAnalyzer,
    MultiArmedBandit,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# --------------------------------------------------------------------------- #
# CausalRules
# --------------------------------------------------------------------------- #


def test_causal_rules_defaults():
    rules = CausalRules()
    assert rules.max_tree_depth == 5
    assert rules.min_samples_split == 2
    assert rules.nash_max_iter == 100
    assert rules.bandit_epsilon == 0.1
    assert rules.pomdp_horizon == 10
    assert rules.causal_significance == 0.05


def test_causal_rules_post_init_populates_dict():
    rules = CausalRules()
    assert rules.rules["max_tree_depth"] == 5
    assert rules.rules["bandit_epsilon"] == 0.1
    assert rules.rules["causal_significance"] == 0.05


def test_causal_rules_custom_values():
    rules = CausalRules(max_tree_depth=10, bandit_epsilon=0.2)
    assert rules.max_tree_depth == 10
    assert rules.bandit_epsilon == 0.2


# --------------------------------------------------------------------------- #
# DecisionTreeBuilder
# --------------------------------------------------------------------------- #


def test_decision_tree_perfect_split():
    """A feature that perfectly separates the labels is used at the root."""
    dt = DecisionTreeBuilder()
    # Feature 0 perfectly separates labels.
    X = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    y = np.array([1, 1, 0, 0])
    result = dt.fit(X, y)
    assert 0 in result["features_used"]
    assert result["n_leaves"] >= 2
    assert result["depth"] >= 1


def test_decision_tree_pure_node():
    """All-same-label data produces a single leaf."""
    dt = DecisionTreeBuilder()
    X = np.array([[1, 0], [2, 1], [3, 2]])
    y = np.array([1, 1, 1])
    result = dt.fit(X, y)
    assert result["n_leaves"] == 1
    assert result["depth"] == 0
    assert result["features_used"] == []
    assert result["tree"]["label"] == 1


def test_decision_tree_empty_input():
    dt = DecisionTreeBuilder()
    result = dt.fit(np.array([]), np.array([]))
    assert result["depth"] == 0
    assert result["n_leaves"] == 1
    assert result["features_used"] == []


def test_decision_tree_max_depth():
    """Tree depth is capped at max_tree_depth."""
    dt = DecisionTreeBuilder(rules=CausalRules(max_tree_depth=1))
    X = np.array([[i, i * 2] for i in range(8)])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    result = dt.fit(X, y)
    assert result["depth"] <= 1


def test_decision_tree_min_samples_split():
    """Below min_samples_split, the node becomes a leaf."""
    dt = DecisionTreeBuilder(rules=CausalRules(min_samples_split=10))
    X = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    y = np.array([1, 1, 0, 0])
    result = dt.fit(X, y)
    # With 4 samples < 10, the root becomes a leaf.
    assert result["depth"] == 0
    assert result["n_leaves"] == 1
    assert result["features_used"] == []


def test_decision_tree_1d_features_reshaped():
    """1D feature arrays are automatically reshaped to (n, 1)."""
    dt = DecisionTreeBuilder()
    X = np.array([1, 2, 3, 4])
    y = np.array([0, 0, 1, 1])
    result = dt.fit(X, y)
    assert isinstance(result["tree"], dict)


def test_decision_tree_entropy_static():
    """_entropy returns 0 for pure labels and >0 for mixed labels."""
    assert DecisionTreeBuilder._entropy(np.array([1, 1, 1])) == 0.0
    assert DecisionTreeBuilder._entropy(np.array([0, 1, 0, 1])) == 1.0
    assert DecisionTreeBuilder._entropy(np.array([])) == 0.0


def test_decision_tree_mismatched_lengths():
    """Mismatched feature/label lengths produce an empty tree."""
    dt = DecisionTreeBuilder()
    result = dt.fit(np.array([[1, 2], [3, 4]]), np.array([1]))
    assert result["depth"] == 0
    assert result["n_leaves"] == 1
    assert result["features_used"] == []


def test_decision_tree_returns_dict_structure():
    dt = DecisionTreeBuilder()
    X = np.array([[1], [0]])
    y = np.array([1, 0])
    result = dt.fit(X, y)
    assert set(result.keys()) == {"tree", "depth", "n_leaves", "features_used"}


# --------------------------------------------------------------------------- #
# GameTheoryAnalyzer
# --------------------------------------------------------------------------- #


def test_game_prisoners_dilemma():
    """Prisoner's Dilemma: (confess, confess) is the unique Nash."""
    gt = GameTheoryAnalyzer()
    # Row=confess(0)/silent(1), Col=confess(0)/silent(1).
    # Payoffs (years in prison, negated so larger is better):
    a = np.array([[-1, -3], [0, -2]])  # row player
    b = np.array([[-1, 0], [-3, -2]])  # col player
    result = gt.analyze(a, b)
    assert (0, 0) in result["nash_equilibria"]
    assert result["is_zero_sum"] is False


def test_game_zero_sum_default():
    """When payoff_b is None, the game is treated as zero-sum."""
    gt = GameTheoryAnalyzer()
    a = np.array([[3, 0], [5, 1]])
    result = gt.analyze(a, payoff_b=None)
    assert result["is_zero_sum"] is True
    # In zero-sum, value_b = -value_a.
    assert result["value_b"] == -result["value_a"]


def test_game_matching_pennies():
    """Matching pennies has no pure-strategy Nash equilibrium."""
    gt = GameTheoryAnalyzer()
    # Row player wins on match (1) or mismatch (-1).
    a = np.array([[1, -1], [-1, 1]])
    result = gt.analyze(a, payoff_b=None)
    assert result["nash_equilibria"] == []


def test_game_invalid_shape():
    """1D payoff matrices produce empty equilibria."""
    gt = GameTheoryAnalyzer()
    result = gt.analyze(np.array([1, 2, 3]))
    assert result["nash_equilibria"] == []
    assert result["is_zero_sum"] is False


def test_game_mismatched_payoff_shapes():
    """Mismatched A and B shapes return empty equilibria."""
    gt = GameTheoryAnalyzer()
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    result = gt.analyze(a, b)
    assert result["nash_equilibria"] == []


def test_game_coordination_game():
    """Coordination game: two Nash equilibria (both choose same action)."""
    gt = GameTheoryAnalyzer()
    a = np.array([[2, 0], [0, 1]])
    b = np.array([[1, 0], [0, 2]])
    result = gt.analyze(a, b)
    # Both (0,0) and (1,1) should be Nash equilibria.
    assert (0, 0) in result["nash_equilibria"]
    assert (1, 1) in result["nash_equilibria"]


# --------------------------------------------------------------------------- #
# CounterfactualReasoner
# --------------------------------------------------------------------------- #


def test_counterfactual_basic():
    """Replacing one element changes the mean by the expected amount."""
    cr = CounterfactualReasoner()
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    # Factual = mean = 2.5
    # Counterfactual: replace index 0 with 100 -> mean = (100+2+3+4)/4 = 27.25
    result = cr.counterfactual(observed, {"index": 0, "value": 100.0})
    assert result["factual"] == pytest.approx(2.5)
    assert result["counterfactual"] == pytest.approx(27.25)
    assert result["effect"] == pytest.approx(24.75)
    assert result["surprisal"] >= 0.0


def test_counterfactual_empty_observed():
    cr = CounterfactualReasoner()
    result = cr.counterfactual(np.array([]), {"index": 0, "value": 1.0})
    assert result["factual"] == 0.0
    assert result["counterfactual"] == 0.0
    assert result["effect"] == 0.0
    assert result["surprisal"] == 0.0


def test_counterfactual_with_model_fn():
    """A custom model_fn replaces the default mean."""
    cr = CounterfactualReasoner()
    observed = np.array([1.0, 2.0, 3.0])
    # model_fn returns the max.
    result = cr.counterfactual(
        observed,
        {"index": 0, "value": 10.0},
        model_fn=lambda x: float(np.max(x)),
    )
    assert result["factual"] == pytest.approx(3.0)
    assert result["counterfactual"] == pytest.approx(10.0)
    assert result["effect"] == pytest.approx(7.0)


def test_counterfactual_out_of_range_index():
    """Out-of-range indices leave the observed array unchanged."""
    cr = CounterfactualReasoner()
    observed = np.array([1.0, 2.0, 3.0])
    result = cr.counterfactual(observed, {"index": 99, "value": 100.0})
    assert result["factual"] == pytest.approx(2.0)
    assert result["counterfactual"] == pytest.approx(2.0)
    assert result["effect"] == pytest.approx(0.0)


def test_counterfactual_returns_dict_structure():
    cr = CounterfactualReasoner()
    result = cr.counterfactual(np.array([1.0]), {"index": 0, "value": 2.0})
    assert set(result.keys()) == {"factual", "counterfactual", "effect", "surprisal"}


# --------------------------------------------------------------------------- #
# MultiArmedBandit
# --------------------------------------------------------------------------- #


def test_bandit_exploit_best_arm():
    """With epsilon=0, the bandit always picks the best arm."""
    bandit = MultiArmedBandit(rules=CausalRules(bandit_epsilon=0.0))
    history = [[1.0, 0.9], [0.1, 0.2], [0.5, 0.5]]
    result = bandit.select(history)
    assert result["method"] == "exploit"
    # Arm 0 has mean 0.95, the highest.
    assert result["arm"] == 0


def test_bandit_empty_history():
    bandit = MultiArmedBandit()
    result = bandit.select([])
    assert result["arm"] == 0
    assert result["method"] == "none"
    assert result["expected_values"].size == 0


def test_bandit_ucb_for_unpulled_arm():
    """An unpulled arm has infinite UCB1 bound."""
    bandit = MultiArmedBandit(rules=CausalRules(bandit_epsilon=0.0))
    history = [[1.0, 0.9], []]  # arm 1 never pulled.
    result = bandit.select(history)
    # Even though arm 1 has mean 0, its UCB is infinite (count=0).
    # But with epsilon=0, we exploit (argmax of means) -> arm 0.
    assert result["arm"] == 0
    # UCB1 for arm 1 should be inf.
    assert np.isinf(result["confidence_bounds"][1])


def test_bandit_expected_values_correct():
    """Per-arm means are computed correctly."""
    bandit = MultiArmedBandit(rules=CausalRules(bandit_epsilon=0.0))
    history = [[1.0, 3.0], [2.0]]
    result = bandit.select(history)
    assert result["expected_values"][0] == pytest.approx(2.0)
    assert result["expected_values"][1] == pytest.approx(2.0)


def test_bandit_returns_dict_structure():
    bandit = MultiArmedBandit(rules=CausalRules(bandit_epsilon=0.0))
    result = bandit.select([[1.0]])
    assert set(result.keys()) == {"arm", "method", "expected_values", "confidence_bounds"}


def test_bandit_ucb_finite_for_pulled_arm():
    """A pulled arm has a finite UCB1 bound."""
    bandit = MultiArmedBandit(rules=CausalRules(bandit_epsilon=0.0))
    history = [[1.0, 0.5], [0.3, 0.7]]
    result = bandit.select(history)
    # All arms pulled -> all UCBs finite.
    assert np.all(np.isfinite(result["confidence_bounds"]))
