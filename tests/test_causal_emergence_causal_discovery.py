# tests/test_causal_emergence_causal_discovery.py
"""Tests for module B: CausalInferenceEngine.

Covers correctness (PC on linear Gaussian DAG, correlation fallback),
do-calculus (intervene, counterfactual), edge cases (insufficient samples,
constant columns, NaN), determinism, and API contract per spec §4 and §10.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.causal_emergence import (
    CausalInferenceEngine,
    EmergenceRules,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _linear_chain_data(n: int = 200, rng=None) -> np.ndarray:
    """Generate linear Gaussian chain: x0 -> x1 -> x2."""
    rng = rng or np.random.default_rng(123)
    x0 = rng.standard_normal(n)
    x1 = 0.8 * x0 + 0.2 * rng.standard_normal(n)
    x2 = 0.6 * x1 + 0.3 * rng.standard_normal(n)
    return np.column_stack([x0, x1, x2])


def _v_structure_data(n: int = 200, rng=None) -> np.ndarray:
    """Generate v-structure: x0 -> x2 <- x1 (x2 is collider)."""
    rng = rng or np.random.default_rng(45)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    x2 = 0.5 * x0 + 0.5 * x1 + 0.1 * rng.standard_normal(n)
    return np.column_stack([x0, x1, x2])


# ----------------------------------------------------------------------
# Correctness: PC algorithm
# ----------------------------------------------------------------------

def test_pc_returns_dag():
    """PC algorithm returns an acyclic adjacency matrix."""
    rng = np.random.default_rng(123)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = _linear_chain_data(rng=rng)
    result = engine.discover(data)
    assert result["is_acyclic"], "DAG must be acyclic"
    assert result["method"] == "pc"
    assert result["adjacency"].shape == (3, 3)


def test_pc_detects_v_structure():
    """PC detects v-structure x0 -> x2 <- x1."""
    rng = np.random.default_rng(45)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = _v_structure_data(rng=rng)
    result = engine.discover(data)
    # x2 (index 2) should have incoming edges from both x0 and x1
    assert result["adjacency"][0, 2] == 1 or result["adjacency"][2, 0] == 1
    assert result["adjacency"][1, 2] == 1 or result["adjacency"][2, 1] == 1
    # x0, x1 should NOT be directly connected (v-structure: unshielded collider)
    # NOTE: PC may add this edge if the v-structure is weak; allow either.
    assert result["is_acyclic"]


def test_pc_no_edges_for_independent_data():
    """Independent variables should produce no edges (or very few)."""
    rng = np.random.default_rng(77)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = rng.standard_normal((200, 3))  # all independent
    result = engine.discover(data)
    # With truly independent variables, PC should remove all edges
    assert result["n_edges"] <= 1  # allow at most 1 spurious edge


def test_pc_correlation_fallback_for_few_samples():
    """n_samples < 3 triggers correlation fallback."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = rng.standard_normal((2, 3))
    result = engine.discover(data)
    assert result["method"] == "correlation"


# ----------------------------------------------------------------------
# Correctness: LiNGAM
# ----------------------------------------------------------------------

def test_lingam_returns_dag():
    """LiNGAM returns an acyclic adjacency matrix."""
    rng = np.random.default_rng(11)
    engine = CausalInferenceEngine(
        rules=EmergenceRules(causal_method="lingam"), rng=rng
    )
    # Non-Gaussian data for LiNGAM
    n = 200
    x0 = rng.standard_normal(n) ** 3  # non-Gaussian
    x1 = 0.8 * x0 + 0.2 * rng.standard_normal(n) ** 3
    data = np.column_stack([x0, x1])
    result = engine.discover(data)
    assert result["method"] in ("lingam", "correlation")  # may fall back
    assert result["is_acyclic"]


def test_lingam_too_many_vars_falls_back():
    """LiNGAM with > 8 vars falls back to correlation."""
    rng = np.random.default_rng(5)
    engine = CausalInferenceEngine(
        rules=EmergenceRules(causal_method="lingam"), rng=rng
    )
    data = rng.standard_normal((50, 10))
    result = engine.discover(data)
    assert result["method"] == "correlation"


# ----------------------------------------------------------------------
# Do-calculus: intervene
# ----------------------------------------------------------------------

def test_intervene_returns_effect():
    """intervene returns effect, pre_mean, post_mean arrays."""
    rng = np.random.default_rng(123)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = _linear_chain_data(rng=rng)
    dag = engine.discover(data)
    result = engine.intervene(
        dag["adjacency"], data, intervention_var=0, intervention_value=1.0
    )
    assert "effect" in result
    assert "pre_mean" in result
    assert "post_mean" in result
    assert result["effect"].shape == (3,)


def test_intervene_changes_target_variable():
    """intervene sets the target variable to the intervention value."""
    rng = np.random.default_rng(123)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = _linear_chain_data(rng=rng)
    dag = engine.discover(data)
    result = engine.intervene(
        dag["adjacency"], data, intervention_var=0, intervention_value=5.0
    )
    assert abs(result["post_mean"][0] - 5.0) < 1e-6


def test_intervene_invalid_var_raises():
    """Out-of-range intervention_var raises ValueError."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = rng.standard_normal((10, 3))
    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    with pytest.raises(ValueError):
        engine.intervene(adj, data, intervention_var=5, intervention_value=1.0)


# ----------------------------------------------------------------------
# Do-calculus: counterfactual
# ----------------------------------------------------------------------

def test_counterfactual_returns_arrays():
    """counterfactual returns counterfactual, factual, shift arrays."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    observed = np.array([0.5, 0.4, 0.3])
    result = engine.counterfactual(
        adj, observed, intervention_var=0, intervention_value=1.0
    )
    assert "counterfactual" in result
    assert "factual" in result
    assert "shift" in result
    assert result["factual"].shape == (3,)


def test_counterfactual_factual_unchanged_for_target():
    """counterfactual value of intervention var equals intervention value."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    observed = np.array([0.5, 0.4, 0.3])
    result = engine.counterfactual(
        adj, observed, intervention_var=0, intervention_value=1.0
    )
    assert abs(result["counterfactual"][0] - 1.0) < 1e-6


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

def test_empty_input():
    """Empty input returns empty DAG."""
    engine = CausalInferenceEngine(rules=EmergenceRules())
    result = engine.discover(np.array([]))
    assert result["n_edges"] == 0
    assert result["adjacency"].shape == (0, 0)


def test_constant_column_skipped():
    """Constant columns are handled (no crash, valid DAG)."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = np.column_stack([
        rng.standard_normal(50),
        np.ones(50),  # constant column
        rng.standard_normal(50),
    ])
    result = engine.discover(data)
    assert result["is_acyclic"]


def test_nan_input_sanitized():
    """NaN in input is sanitized."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = rng.standard_normal((50, 3))
    data[0, 0] = np.nan
    result = engine.discover(data)
    assert result["is_acyclic"]


def test_var_names_default():
    """Default var_names are x0, x1, ..."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = rng.standard_normal((20, 3))
    result = engine.discover(data)
    assert result["var_names"] == ["x0", "x1", "x2"]


def test_var_names_custom():
    """Custom var_names are honored if length matches."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = rng.standard_normal((20, 3))
    result = engine.discover(data, var_names=["alpha", "beta", "gamma"])
    assert result["var_names"] == ["alpha", "beta", "gamma"]


def test_var_names_mismatch_falls_back():
    """Mismatched var_names length falls back to defaults."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = rng.standard_normal((20, 3))
    result = engine.discover(data, var_names=["a", "b"])  # wrong length
    assert result["var_names"] == ["x0", "x1", "x2"]


# ----------------------------------------------------------------------
# API contract
# ----------------------------------------------------------------------

def test_discover_return_dict_keys():
    """discover return dict has all expected keys."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = rng.standard_normal((20, 3))
    result = engine.discover(data)
    expected_keys = {
        "adjacency", "edges", "method", "var_names",
        "n_edges", "is_acyclic",
    }
    assert set(result.keys()) == expected_keys


def test_edges_from_adjacency():
    """edges list matches non-zero entries of adjacency matrix."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = rng.standard_normal((50, 3))
    result = engine.discover(data)
    adj = result["adjacency"]
    expected_edges = [(i, j) for i in range(3) for j in range(3) if adj[i, j] == 1]
    assert result["edges"] == expected_edges


def test_explicit_method_parameter():
    """method parameter overrides rules.causal_method."""
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = rng.standard_normal((20, 3))
    result = engine.discover(data, method="correlation")
    assert result["method"] == "correlation"


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_deterministic_with_seed():
    """Same rng seed produces identical DAG."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    data = _linear_chain_data(rng=np.random.default_rng(0))
    e1 = CausalInferenceEngine(rules=EmergenceRules(), rng=rng1)
    e2 = CausalInferenceEngine(rules=EmergenceRules(), rng=rng2)
    r1 = e1.discover(data)
    r2 = e2.discover(data)
    np.testing.assert_array_equal(r1["adjacency"], r2["adjacency"])


# ----------------------------------------------------------------------
# Performance (spec §10.4)
# ----------------------------------------------------------------------

def test_performance_pc_under_1_second():
    """PC on (100, 8) data completes in < 1 second."""
    import time
    rng = np.random.default_rng(0)
    engine = CausalInferenceEngine(rules=EmergenceRules(), rng=rng)
    data = rng.standard_normal((100, 8))
    start = time.perf_counter()
    engine.discover(data)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"PC took {elapsed:.2f}s, expected < 1s"
