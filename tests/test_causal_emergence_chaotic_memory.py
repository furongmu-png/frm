# tests/test_causal_emergence_chaotic_memory.py
"""Tests for module E: ChaoticAssociativeMemory.

Covers correctness (storage, recall of nearest pattern, FIFO eviction),
edge cases (empty memory, NaN query, odd dim, large capacity overflow),
determinism, Lorenz trajectory shape, emergence detection, and API
contract per spec §7.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.causal_emergence import (
    ChaoticAssociativeMemory,
    EmergenceRules,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_pattern(rng, dim=8):
    """Generate a random pattern in [-1, 1]^dim."""
    return rng.uniform(-1.0, 1.0, dim)


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------

def test_store_single_pattern():
    """Storing one pattern records it."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=8, rules=EmergenceRules(), rng=rng)
    p = _make_pattern(rng)
    result = mem.store(p, label="A")
    assert result["label"] == "A"
    assert result["n_stored"] == 1
    assert result["capacity"] == 32
    assert result["evicted"] == []


def test_store_multiple_patterns():
    """Storing multiple patterns increments count."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=8, rules=EmergenceRules(), rng=rng)
    for i in range(5):
        mem.store(_make_pattern(rng), label=f"p{i}")
    assert mem.size == 5


def test_store_returns_n_stored():
    """n_stored reflects current count after store."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    r1 = mem.store(_make_pattern(rng), label="a")
    r2 = mem.store(_make_pattern(rng), label="b")
    r3 = mem.store(_make_pattern(rng), label="c")
    assert r1["n_stored"] == 1
    assert r2["n_stored"] == 2
    assert r3["n_stored"] == 3


def test_store_fifo_eviction():
    """When capacity exceeded, oldest pattern is evicted (FIFO)."""
    rules = EmergenceRules(chaotic_memory_capacity=3)
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=rules, rng=rng)
    mem.store(_make_pattern(rng), label="a")
    mem.store(_make_pattern(rng), label="b")
    mem.store(_make_pattern(rng), label="c")
    r = mem.store(_make_pattern(rng), label="d")
    assert r["evicted"] == ["a"]
    assert r["n_stored"] == 3
    assert mem.size == 3


def test_store_capacity_zero_no_storage():
    """Capacity=0 means no patterns can be stored (each store evicts)."""
    rules = EmergenceRules(chaotic_memory_capacity=0)
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=rules, rng=rng)
    r = mem.store(_make_pattern(rng), label="x")
    assert r["n_stored"] == 0
    assert r["evicted"] == ["x"]


def test_store_label_int_or_string():
    """Labels can be integers or strings."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng), label=42)
    mem.store(_make_pattern(rng), label="hello")
    assert mem.size == 2


# ----------------------------------------------------------------------
# Recall: empty memory
# ----------------------------------------------------------------------

def test_recall_empty_memory_returns_none_label():
    """Recall from empty memory returns label=None."""
    mem = ChaoticAssociativeMemory(rules=EmergenceRules())
    result = mem.recall(np.zeros(8))
    assert result["label"] is None
    assert result["similarity"] == 0.0
    assert result["emerged"] is True
    assert result["converged"] is False
    assert result["divergence"] == 0.0


def test_recall_empty_memory_trajectory_shape():
    """Empty memory recall returns (n_steps, 3) zeros trajectory."""
    mem = ChaoticAssociativeMemory(rules=EmergenceRules())
    result = mem.recall(np.zeros(4), n_steps=50)
    assert result["trajectory"].shape == (50, 3)
    assert np.all(result["trajectory"] == 0.0)


# ----------------------------------------------------------------------
# Recall: nearest pattern
# ----------------------------------------------------------------------

def test_recall_returns_nearest_pattern_label():
    """Recall returns the label of the nearest stored pattern."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    p1 = np.array([1.0, 0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0, 1.0])
    mem.store(p1, label="near")
    mem.store(p2, label="far")

    # Query closer to p1
    q = np.array([0.9, 0.0, 0.0, 0.0])
    result = mem.recall(q, n_steps=10)
    assert result["label"] == "near"


def test_recall_similarity_in_range():
    """Similarity is in [0, 1]."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(np.array([1.0, 0.0, 0.0, 0.0]), label="A")
    result = mem.recall(np.array([0.5, 0.0, 0.0, 0.0]), n_steps=10)
    assert 0.0 <= result["similarity"] <= 1.0


def test_recall_exact_match_similarity_one():
    """Exact match gives similarity = 1.0."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    p = np.array([0.5, 0.5, 0.5, 0.5])
    mem.store(p, label="exact")
    result = mem.recall(p, n_steps=10)
    # When query == pattern, similarity = 1 / (1 + 0) = 1.0
    np.testing.assert_allclose(result["similarity"], 1.0)


def test_recall_trajectory_shape():
    """Recall trajectory has shape (n_steps, 3) — Lorenz state."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=8, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=8), label="A")
    result = mem.recall(_make_pattern(rng, dim=8), n_steps=100)
    assert result["trajectory"].shape == (100, 3)


def test_recall_trajectory_is_float_array():
    """Trajectory is a float ndarray."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    result = mem.recall(_make_pattern(rng, dim=4), n_steps=20)
    assert isinstance(result["trajectory"], np.ndarray)
    assert result["trajectory"].dtype == np.float64


# ----------------------------------------------------------------------
# Recall: NaN / edge cases
# ----------------------------------------------------------------------

def test_recall_nan_query_returns_none():
    """NaN query returns label=None, emerged=False, converged=False."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    result = mem.recall(np.array([np.nan, 0.0, 0.0, 0.0]), n_steps=10)
    assert result["label"] is None
    assert result["emerged"] is False
    assert result["converged"] is False


def test_recall_inf_query_rejected():
    """Inf query is rejected (would cause Lorenz integration to diverge).

    Spec §7.4 explicitly mentions NaN; Inf is treated the same way
    because ``np.isfinite`` catches both, and either would cause the
    RK4 integration to diverge.
    """
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(np.array([0.0, 0.0, 0.0, 0.0]), label="A")
    result = mem.recall(np.array([np.inf, 0.0, 0.0, 0.0]), n_steps=10)
    assert result["label"] is None
    assert result["emerged"] is False
    assert result["converged"] is False


def test_recall_odd_dim_pattern():
    """Odd dim pattern splits as [0:dim//2] and [dim//2:] (lengths differ by 1)."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=5, rules=EmergenceRules(), rng=rng)
    p = _make_pattern(rng, dim=5)
    mem.store(p, label="odd")
    result = mem.recall(p, n_steps=10)
    assert result["label"] == "odd"


def test_recall_2d_pattern_flattened():
    """2D pattern arrays are flattened to 1D state vectors."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    p = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    mem.store(p, label="2d")
    result = mem.recall(p, n_steps=10)
    assert result["label"] == "2d"


# ----------------------------------------------------------------------
# Emergence detection
# ----------------------------------------------------------------------

def test_recall_divergence_is_finite():
    """Divergence is a finite non-negative float."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    result = mem.recall(_make_pattern(rng, dim=4), n_steps=50)
    assert np.isfinite(result["divergence"])
    assert result["divergence"] >= 0.0


def test_emerged_flag_is_python_bool():
    """emerged is a Python bool."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    result = mem.recall(_make_pattern(rng, dim=4), n_steps=20)
    assert isinstance(result["emerged"], bool)


def test_high_threshold_never_emerges():
    """Very high divergence threshold -> emerged is always False."""
    rules = EmergenceRules(chaotic_divergence_threshold=1e9)
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=rules, rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    result = mem.recall(_make_pattern(rng, dim=4), n_steps=50)
    assert result["emerged"] is False


def test_zero_threshold_always_emerges():
    """Zero threshold -> any nonzero divergence triggers emergence."""
    rules = EmergenceRules(chaotic_divergence_threshold=0.0)
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=rules, rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    # Use a query slightly different from the stored pattern so initial
    # perturbation produces a nonzero delta_final.
    q = _make_pattern(rng, dim=4)
    result = mem.recall(q, n_steps=50)
    # If divergence is > 0, emerged must be True (threshold=0).
    if result["divergence"] > 0:
        assert result["emerged"] is True


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_same_rng_produces_same_recall():
    """Same rng seed produces identical recall results."""
    rules = EmergenceRules()
    p = np.array([1.0, 0.5, -0.5, 0.0])
    q = np.array([0.8, 0.4, -0.3, 0.1])

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    m1 = ChaoticAssociativeMemory(dim=4, rules=rules, rng=rng1)
    m2 = ChaoticAssociativeMemory(dim=4, rules=rules, rng=rng2)
    m1.store(p, label="A")
    m2.store(p, label="A")
    r1 = m1.recall(q, n_steps=50)
    r2 = m2.recall(q, n_steps=50)
    np.testing.assert_allclose(r1["trajectory"], r2["trajectory"])
    assert r1["divergence"] == r2["divergence"]
    assert r1["emerged"] == r2["emerged"]


# ----------------------------------------------------------------------
# Clear
# ----------------------------------------------------------------------

def test_clear_empties_memory():
    """clear() removes all stored patterns."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    mem.store(_make_pattern(rng, dim=4), label="B")
    assert mem.size == 2
    mem.clear()
    assert mem.size == 0


def test_recall_after_clear_returns_empty():
    """After clear, recall returns label=None."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    mem.clear()
    result = mem.recall(np.zeros(4), n_steps=10)
    assert result["label"] is None


# ----------------------------------------------------------------------
# API contract
# ----------------------------------------------------------------------

def test_store_returns_required_keys():
    """store() result contains required keys."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    result = mem.store(_make_pattern(rng, dim=4), label="A")
    for key in ("label", "n_stored", "capacity", "evicted"):
        assert key in result, f"missing key: {key}"


def test_recall_returns_required_keys():
    """recall() result contains required keys."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    result = mem.recall(_make_pattern(rng, dim=4), n_steps=10)
    for key in (
        "label", "similarity", "emerged",
        "trajectory", "converged", "divergence"
    ):
        assert key in result, f"missing key: {key}"


def test_similarity_is_python_float():
    """similarity is a Python float."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    result = mem.recall(_make_pattern(rng, dim=4), n_steps=10)
    assert isinstance(result["similarity"], float)
    assert not isinstance(result["similarity"], np.floating)


def test_divergence_is_python_float():
    """divergence is a Python float."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    result = mem.recall(_make_pattern(rng, dim=4), n_steps=10)
    assert isinstance(result["divergence"], float)
    assert not isinstance(result["divergence"], np.floating)


def test_converged_is_python_bool():
    """converged is a Python bool."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=4, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=4), label="A")
    result = mem.recall(_make_pattern(rng, dim=4), n_steps=10)
    assert isinstance(result["converged"], bool)


def test_default_rules_when_none():
    """Default rules are created when none provided."""
    mem = ChaoticAssociativeMemory(rules=None)
    assert mem.rules is not None
    assert mem.rules.chaotic_memory_capacity == 32
    assert mem.rules.chaotic_alpha == 0.1


def test_default_rng_when_none():
    """Default rng is created when none provided."""
    mem = ChaoticAssociativeMemory()
    assert mem.rng is not None


def test_dim_attribute_stored():
    """dim is stored on the instance."""
    mem = ChaoticAssociativeMemory(dim=16)
    assert mem.dim == 16


def test_size_attribute_starts_zero():
    """size starts at 0."""
    mem = ChaoticAssociativeMemory(rules=EmergenceRules())
    assert mem.size == 0


# ----------------------------------------------------------------------
# Lorenz trajectory properties
# ----------------------------------------------------------------------

def test_lorenz_trajectory_does_not_blow_up():
    """Lorenz trajectory values stay bounded (not NaN/Inf)."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=8, rules=EmergenceRules(), rng=rng)
    mem.store(_make_pattern(rng, dim=8), label="A")
    result = mem.recall(_make_pattern(rng, dim=8), n_steps=100)
    assert np.all(np.isfinite(result["trajectory"]))


def test_lorenz_initial_state_encodes_query():
    """Lorenz trajectory's first state encodes the query."""
    rng = np.random.default_rng(0)
    mem = ChaoticAssociativeMemory(dim=8, rules=EmergenceRules(), rng=rng)
    p = np.zeros(8)
    p[:4] = 0.5  # first half mean = 0.5 -> target_y
    p[4:] = 1.5   # second half mean = 1.5 -> target_z
    mem.store(p, label="A")
    result = mem.recall(p, n_steps=10)
    # Initial state is (0, target_y, target_z) = (0, 0.5, 1.5)
    np.testing.assert_allclose(result["trajectory"][0], [0.0, 0.5, 1.5])
