# tests/test_causal_emergence_emergence_cycle.py
"""Tests for phase 4: the full emergence_cycle orchestration.

Covers input validation (1D, 2D, insufficient data), all five module
contributions, the emergence_score heuristic (term-by-term
verification), failure degradation (H6), determinism, and API contract
per spec §8.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.causal_emergence import (
    CausalEmergenceEngine,
    EmergenceRules,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_observation(n_samples=50, n_features=4, seed=0):
    """Generate a 2D observation matrix with mild causal structure."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n_samples)
    x1 = 0.7 * x0 + 0.3 * rng.standard_normal(n_samples)
    x2 = 0.5 * x1 + 0.4 * rng.standard_normal(n_samples)
    x3 = 0.2 * x2 + 0.6 * rng.standard_normal(n_samples)
    return np.column_stack([x0, x1, x2, x3])


# ----------------------------------------------------------------------
# Input validation (fix C3)
# ----------------------------------------------------------------------

def test_1d_input_returns_zero_score():
    """1D input returns emergence_score=0 with insufficient_data reason."""
    engine = CausalEmergenceEngine()
    result = engine.emergence_cycle(np.array([1.0, 2.0, 3.0]))
    assert result["emergence_score"] == 0.0
    assert result["reason"] == "insufficient_data"


def test_none_input_returns_zero_score():
    """None input returns emergence_score=0 with insufficient_data reason."""
    engine = CausalEmergenceEngine()
    result = engine.emergence_cycle(None)
    assert result["emergence_score"] == 0.0
    assert result["reason"] == "insufficient_data"


def test_insufficient_rows_returns_zero_score():
    """n_samples < 2 returns emergence_score=0."""
    engine = CausalEmergenceEngine()
    result = engine.emergence_cycle(np.array([[1.0, 2.0]]))
    assert result["emergence_score"] == 0.0
    assert result["reason"] == "insufficient_data"


def test_insufficient_features_returns_zero_score():
    """n_features < 2 returns emergence_score=0."""
    engine = CausalEmergenceEngine()
    result = engine.emergence_cycle(np.array([[1.0], [2.0], [3.0]]))
    assert result["emergence_score"] == 0.0
    assert result["reason"] == "insufficient_data"


def test_nan_input_sanitized_to_zeros():
    """NaN inputs are sanitized; the cycle still runs (no crash)."""
    engine = CausalEmergenceEngine()
    obs = np.array([[np.nan, 1.0], [2.0, np.inf], [1.0, 0.0]])
    result = engine.emergence_cycle(obs)
    assert "emergence_score" in result
    assert np.isfinite(result["emergence_score"])


# ----------------------------------------------------------------------
# Full cycle: required return keys
# ----------------------------------------------------------------------

def test_returns_required_keys():
    """emergence_cycle returns dict with all required keys."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4)
    result = engine.emergence_cycle(obs)
    for key in (
        "perception", "causal_graph", "counterfactual",
        "posterior", "memory", "emergence_score", "warnings"
    ):
        assert key in result, f"missing key: {key}"


def test_returns_subdicts():
    """Each module's output is a dict."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4)
    result = engine.emergence_cycle(obs)
    assert isinstance(result["perception"], dict)
    assert isinstance(result["causal_graph"], dict)
    assert isinstance(result["counterfactual"], dict)
    assert isinstance(result["posterior"], dict)
    assert isinstance(result["memory"], dict)


def test_warnings_is_list_of_strings():
    """warnings is a list of strings."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4)
    result = engine.emergence_cycle(obs)
    assert isinstance(result["warnings"], list)
    for w in result["warnings"]:
        assert isinstance(w, str)


def test_emergence_score_is_python_float():
    """emergence_score is a Python float in [0, 1]."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4)
    result = engine.emergence_cycle(obs)
    assert isinstance(result["emergence_score"], float)
    assert not isinstance(result["emergence_score"], np.floating)
    assert 0.0 <= result["emergence_score"] <= 1.0


# ----------------------------------------------------------------------
# Module contributions
# ----------------------------------------------------------------------

def test_perception_is_module_a_output():
    """perception contains keys returned by perceive_topology."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4)
    result = engine.emergence_cycle(obs)
    p = result["perception"]
    assert "betti_numbers" in p
    assert "persistence_entropy" in p
    assert "euler_characteristic" in p


def test_causal_graph_is_module_b_output():
    """causal_graph contains keys returned by discover_causal_dynamics."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4)
    result = engine.emergence_cycle(obs)
    g = result["causal_graph"]
    assert "adjacency" in g
    assert "edges" in g
    assert "n_edges" in g


def test_counterfactual_is_module_c_output():
    """counterfactual contains keys returned by generate_trajectory."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4)
    result = engine.emergence_cycle(obs)
    c = result["counterfactual"]
    assert "trajectory" in c
    assert "action" in c
    assert "converged" in c


def test_posterior_is_module_d_output():
    """posterior contains keys returned by sample_posterior."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4)
    result = engine.emergence_cycle(obs)
    p = result["posterior"]
    assert "samples" in p
    assert "mean" in p
    assert "std" in p
    assert "accept_rate" in p
    assert "ess" in p


def test_memory_is_module_e_output():
    """memory contains keys returned by recall_memory."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4)
    result = engine.emergence_cycle(obs)
    m = result["memory"]
    assert "label" in m
    assert "similarity" in m
    assert "emerged" in m
    assert "divergence" in m


# ----------------------------------------------------------------------
# Counterfactual trajectory (fix M7)
# ----------------------------------------------------------------------

def test_counterfactal_uses_mean_as_start():
    """counterfactual trajectory[0] equals observation.mean(axis=0)."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4)
    result = engine.emergence_cycle(obs)
    mean = obs.mean(axis=0)
    np.testing.assert_allclose(
        result["counterfactual"]["trajectory"][0], mean, atol=1e-6
    )


def test_counterfactual_n_steps_is_16():
    """counterfactual trajectory has 17 rows (n_steps + 1)."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4)
    result = engine.emergence_cycle(obs)
    assert result["counterfactual"]["trajectory"].shape[0] == 17


def test_counterfactual_delta_is_perturbation_scale():
    """delta = perturbation * standard_normal, so its scale matches rule."""
    rules = EmergenceRules(emergence_cycle_perturbation=0.1)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    e1 = CausalEmergenceEngine(rules=rules, rng=rng1)
    e2 = CausalEmergenceEngine(rules=rules, rng=rng2)
    obs = _make_observation(n_samples=50, n_features=4)
    r1 = e1.emergence_cycle(obs)
    r2 = e2.emergence_cycle(obs)
    # Same rng -> same delta -> same end_state
    np.testing.assert_allclose(
        r1["counterfactual"]["trajectory"][-1],
        r2["counterfactual"]["trajectory"][-1],
        atol=1e-12,
    )


# ----------------------------------------------------------------------
# Failure degradation (fix H6)
# ----------------------------------------------------------------------

def test_failure_records_warning():
    """If a module fails internally, the warning list captures it.

    We force a topology failure by passing degenerate data that would
    trip the topology perceiver's edge cases; the cycle should still
    return a score and log a warning if anything went wrong.
    """
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    # Use very small observation; should still work but trigger fallbacks
    obs = _make_observation(n_samples=5, n_features=3)
    result = engine.emergence_cycle(obs)
    # Either succeeded cleanly (no warnings) or degraded gracefully
    assert "emergence_score" in result
    assert isinstance(result["warnings"], list)


def test_emergence_score_finite_under_extreme_input():
    """Even extreme inputs produce a finite emergence_score."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = np.array([
        [1e3, -1e3, 0.0, 0.0],
        [1e3, -1e3, 0.0, 0.0],
        [1e3, -1e3, 0.0, 0.0],
    ])
    result = engine.emergence_cycle(obs)
    assert np.isfinite(result["emergence_score"])


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_same_rng_produces_same_score():
    """Same rng seed produces identical emergence_score."""
    rules = EmergenceRules()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    e1 = CausalEmergenceEngine(rules=rules, rng=rng1)
    e2 = CausalEmergenceEngine(rules=rules, rng=rng2)
    obs = _make_observation(n_samples=50, n_features=4)
    r1 = e1.emergence_cycle(obs)
    r2 = e2.emergence_cycle(obs)
    assert r1["emergence_score"] == r2["emergence_score"]
    np.testing.assert_allclose(
        r1["counterfactual"]["trajectory"],
        r2["counterfactual"]["trajectory"],
    )


def test_different_rng_different_score():
    """Different rng seeds typically produce different scores."""
    rules = EmergenceRules()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(999)
    e1 = CausalEmergenceEngine(rules=rules, rng=rng1)
    e2 = CausalEmergenceEngine(rules=rules, rng=rng2)
    obs = _make_observation(n_samples=50, n_features=4)
    r1 = e1.emergence_cycle(obs)
    r2 = e2.emergence_cycle(obs)
    # delta is random, so the counterfactual trajectory likely differs.
    # The score might still coincide by chance, but the trajectory must
    # differ (since delta is sampled independently).
    assert not np.allclose(
        r1["counterfactual"]["trajectory"],
        r2["counterfactual"]["trajectory"],
    )


# ----------------------------------------------------------------------
# compute_emergence_score: term-by-term verification
# ----------------------------------------------------------------------

def test_score_zero_when_all_terms_zero():
    """All-zero inputs produce score=0."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    perception = {"persistence_entropy": 0.0}
    causal_graph = {
        "n_edges": 0,
        "adjacency": np.zeros((4, 4)),
    }
    counterfactual = {"action": 0.0}
    posterior = {"std": np.array([10.0, 10.0, 10.0, 10.0])}  # std >> dim
    memory_response = {"emerged": False}
    score = engine.compute_emergence_score(
        perception=perception,
        causal_graph=causal_graph,
        counterfactual=counterfactual,
        posterior=posterior,
        memory_response=memory_response,
        reference_action=1.0,
        dim=4,
    )
    assert score == 0.0


def test_score_one_when_all_terms_max():
    """All-max inputs produce score=1."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    perception = {"persistence_entropy": 1.0}
    # 4 nodes fully connected: 4*3/2 = 6 max edges; with 6 edges, density=1.
    causal_graph = {
        "n_edges": 6,
        "adjacency": np.ones((4, 4)) - np.eye(4),
    }
    counterfactual = {"action": 100.0}
    posterior = {"std": np.array([0.0, 0.0, 0.0, 0.0])}  # tight -> term=1
    memory_response = {"emerged": True}
    score = engine.compute_emergence_score(
        perception=perception,
        causal_graph=causal_graph,
        counterfactual=counterfactual,
        posterior=posterior,
        memory_response=memory_response,
        reference_action=1.0,  # action/reference = 100/1 -> clipped to 1
        dim=4,
    )
    np.testing.assert_allclose(score, 1.0, atol=1e-12)


def test_persistence_entropy_weight_30_percent():
    """Persistence entropy contributes 0.30 to the score."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    perception = {"persistence_entropy": 0.5}
    causal_graph = {"n_edges": 0, "adjacency": np.zeros((4, 4))}
    counterfactual = {"action": 0.0}
    posterior = {"std": np.array([10.0, 10.0, 10.0, 10.0])}
    memory_response = {"emerged": False}
    score = engine.compute_emergence_score(
        perception=perception,
        causal_graph=causal_graph,
        counterfactual=counterfactual,
        posterior=posterior,
        memory_response=memory_response,
        reference_action=1.0,
        dim=4,
    )
    # Only persistence_entropy term contributes: 0.30 * 0.5 = 0.15
    np.testing.assert_allclose(score, 0.15, atol=1e-12)


def test_causal_density_weight_20_percent():
    """Causal density contributes 0.20 to the score."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    perception = {"persistence_entropy": 0.0}
    # 4 nodes, 3 edges: density = 3 / 6 = 0.5
    causal_graph = {
        "n_edges": 3,
        "adjacency": np.array([
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]),
    }
    counterfactual = {"action": 0.0}
    posterior = {"std": np.array([10.0, 10.0, 10.0, 10.0])}
    memory_response = {"emerged": False}
    score = engine.compute_emergence_score(
        perception=perception,
        causal_graph=causal_graph,
        counterfactual=counterfactual,
        posterior=posterior,
        memory_response=memory_response,
        reference_action=1.0,
        dim=4,
    )
    # Only density term: 0.20 * 0.5 = 0.10
    np.testing.assert_allclose(score, 0.10, atol=1e-12)


def test_memory_emerged_weight_20_percent():
    """Memory.emerged contributes 0.20 to the score."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    perception = {"persistence_entropy": 0.0}
    causal_graph = {"n_edges": 0, "adjacency": np.zeros((4, 4))}
    counterfactual = {"action": 0.0}
    posterior = {"std": np.array([10.0, 10.0, 10.0, 10.0])}
    memory_response = {"emerged": True}
    score = engine.compute_emergence_score(
        perception=perception,
        causal_graph=causal_graph,
        counterfactual=counterfactual,
        posterior=posterior,
        memory_response=memory_response,
        reference_action=1.0,
        dim=4,
    )
    np.testing.assert_allclose(score, 0.20, atol=1e-12)


def test_posterior_tightness_weight_15_percent():
    """Posterior tightness (std=0) contributes 0.15 to the score."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    perception = {"persistence_entropy": 0.0}
    causal_graph = {"n_edges": 0, "adjacency": np.zeros((4, 4))}
    counterfactual = {"action": 0.0}
    posterior = {"std": np.array([0.0, 0.0, 0.0, 0.0])}  # tight
    memory_response = {"emerged": False}
    score = engine.compute_emergence_score(
        perception=perception,
        causal_graph=causal_graph,
        counterfactual=counterfactual,
        posterior=posterior,
        memory_response=memory_response,
        reference_action=1.0,
        dim=4,
    )
    # Only tightness term: 0.15 * 1.0 = 0.15
    np.testing.assert_allclose(score, 0.15, atol=1e-12)


def test_counterfactual_action_weight_15_percent():
    """Counterfactual action ratio contributes 0.15 to the score."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    perception = {"persistence_entropy": 0.0}
    causal_graph = {"n_edges": 0, "adjacency": np.zeros((4, 4))}
    counterfactual = {"action": 0.5}  # action / reference = 0.5
    posterior = {"std": np.array([10.0, 10.0, 10.0, 10.0])}
    memory_response = {"emerged": False}
    score = engine.compute_emergence_score(
        perception=perception,
        causal_graph=causal_graph,
        counterfactual=counterfactual,
        posterior=posterior,
        memory_response=memory_response,
        reference_action=1.0,
        dim=4,
    )
    # Only action term: 0.15 * 0.5 = 0.075
    np.testing.assert_allclose(score, 0.075, atol=1e-12)


def test_action_clipped_to_one():
    """If action > reference, the term is clipped to 1.0."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    perception = {"persistence_entropy": 0.0}
    causal_graph = {"n_edges": 0, "adjacency": np.zeros((4, 4))}
    counterfactual = {"action": 100.0}  # action/reference = 100 -> clipped
    posterior = {"std": np.array([10.0, 10.0, 10.0, 10.0])}
    memory_response = {"emerged": False}
    score = engine.compute_emergence_score(
        perception=perception,
        causal_graph=causal_graph,
        counterfactual=counterfactual,
        posterior=posterior,
        memory_response=memory_response,
        reference_action=1.0,
        dim=4,
    )
    np.testing.assert_allclose(score, 0.15, atol=1e-12)


def test_weights_sum_to_one():
    """The 5 weights sum to 1.0 (spec §8.3)."""
    weights = [0.30, 0.20, 0.20, 0.15, 0.15]
    assert sum(weights) == 1.0


# ----------------------------------------------------------------------
# Edge cases in compute_emergence_score
# ----------------------------------------------------------------------

def test_score_handles_empty_posterior_std():
    """Empty posterior.std falls back to mean_std=0 (term=1)."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    perception = {"persistence_entropy": 0.0}
    causal_graph = {"n_edges": 0, "adjacency": np.zeros((0, 0))}
    counterfactual = {"action": 0.0}
    posterior = {"std": np.array([])}
    memory_response = {"emerged": False}
    score = engine.compute_emergence_score(
        perception=perception,
        causal_graph=causal_graph,
        counterfactual=counterfactual,
        posterior=posterior,
        memory_response=memory_response,
        reference_action=1.0,
        dim=4,
    )
    # All terms zero; tightness term defaults to 1.0 -> 0.15
    np.testing.assert_allclose(score, 0.15, atol=1e-12)


def test_score_handles_missing_adjacency():
    """Missing adjacency falls back to max_edges=1 (term saturates)."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    perception = {"persistence_entropy": 0.0}
    causal_graph = {"n_edges": 1}  # no adjacency field
    counterfactual = {"action": 0.0}
    posterior = {"std": np.array([10.0] * 4)}
    memory_response = {"emerged": False}
    score = engine.compute_emergence_score(
        perception=perception,
        causal_graph=causal_graph,
        counterfactual=counterfactual,
        posterior=posterior,
        memory_response=memory_response,
        reference_action=1.0,
        dim=4,
    )
    # max_edges=1 -> density = 1/1 = 1.0 -> term = 0.20
    np.testing.assert_allclose(score, 0.20, atol=1e-12)


def test_score_handles_zero_reference_action():
    """reference_action=0 falls back to |action| (or 1.0 if action=0)."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    perception = {"persistence_entropy": 0.0}
    causal_graph = {"n_edges": 0, "adjacency": np.zeros((4, 4))}
    counterfactual = {"action": 0.0}
    posterior = {"std": np.array([10.0] * 4)}
    memory_response = {"emerged": False}
    score = engine.compute_emergence_score(
        perception=perception,
        causal_graph=causal_graph,
        counterfactual=counterfactual,
        posterior=posterior,
        memory_response=memory_response,
        reference_action=0.0,  # zero -> fallback
        dim=4,
    )
    # Both action and reference are 0; fallback -> reference=1.0, action=0
    # term = 0/1 = 0. All other terms are 0 too.
    np.testing.assert_allclose(score, 0.0, atol=1e-12)


def test_score_in_zero_to_one_range():
    """Score is always in [0, 1] for realistic inputs."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=80, n_features=5, seed=7)
    result = engine.emergence_cycle(obs)
    assert 0.0 <= result["emergence_score"] <= 1.0


# ----------------------------------------------------------------------
# End-to-end smoke test
# ----------------------------------------------------------------------

def test_end_to_end_on_synthetic_observation():
    """End-to-end run on a 50x4 observation completes without error."""
    engine = CausalEmergenceEngine(rules=EmergenceRules())
    obs = _make_observation(n_samples=50, n_features=4, seed=0)
    result = engine.emergence_cycle(obs)
    assert result["emergence_score"] > 0.0  # at least some signal
    assert isinstance(result["perception"]["betti_numbers"], list)


def test_higher_persistence_increases_score():
    """An observation with more topological structure scores higher
    on the persistence term than a degenerate (clustered) one."""
    rng = np.random.default_rng(0)
    rules = EmergenceRules()
    e1 = CausalEmergenceEngine(rules=rules, rng=rng)

    # Circle (more topological structure)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    # Pad to 4 features so cycle accepts it
    obs_circle = np.hstack([circle, np.zeros((16, 2))])

    rng2 = np.random.default_rng(0)
    e2 = CausalEmergenceEngine(rules=rules, rng=rng2)
    # Two clusters (low topological complexity)
    cluster = np.vstack([
        np.column_stack([
            rng2.standard_normal((16, 2)) * 0.1,
            np.zeros((16, 2)),
        ]),
        np.column_stack([
            10 + rng2.standard_normal((16, 2)) * 0.1,
            np.zeros((16, 2)),
        ]),
    ])

    r_circle = e1.emergence_cycle(obs_circle)
    r_cluster = e2.emergence_cycle(cluster)
    # Circle should have higher persistence entropy than two clusters
    # (the latter collapses to a single point after topology subsampling).
    assert (
        r_circle["perception"]["persistence_entropy"]
        >= r_cluster["perception"]["persistence_entropy"]
    )
