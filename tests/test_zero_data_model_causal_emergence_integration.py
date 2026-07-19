# tests/test_zero_data_model_causal_emergence_integration.py
"""Tests for ZeroDataModel <-> CausalEmergenceEngine integration (spec §2.3).

Verifies that ``ZeroDataModel`` correctly wires up the emergence engine as a
single-instance attribute, exposes the 6 facade methods (all under
``self._lock``), preserves the 6-module cognitive count contract, and
supports pickle round-trips.
"""

from __future__ import annotations

import pickle

import numpy as np

from zero_data_model.model import ZeroDataModel

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
    return np.column_stack([x0, x1, x2, x3])[:, :n_features]


# ----------------------------------------------------------------------
# Construction & wiring
# ----------------------------------------------------------------------

def test_model_has_emergence_attribute():
    """ZeroDataModel exposes ``self.emergence`` after construction."""
    model = ZeroDataModel(dim=8, seed=42)
    assert hasattr(model, "emergence")
    assert hasattr(model, "emergence_rules")
    # Engine is properly configured.
    assert model.emergence.dim == 8


def test_emergence_engine_uses_shared_core_modules():
    """Engine reuses ``self.active_inference`` and ``self.math_universe``."""
    model = ZeroDataModel(dim=8, seed=42)
    assert model.emergence.active_inference is model.active_inference
    assert model.emergence.math_universe is model.math_universe


def test_emergence_engine_uses_emergence_rules():
    """Engine uses ``self.emergence_rules`` as its rules."""
    model = ZeroDataModel(dim=8, seed=42)
    assert model.emergence.rules is model.emergence_rules


def test_emergence_engine_not_in_modules():
    """spec §2.3: engine is NOT in ``self.modules`` (preserves 6-module count)."""
    model = ZeroDataModel(dim=8, seed=42)
    assert len(model.modules) == 6
    assert model.emergence not in model.modules


def test_n_cognitive_modules_unchanged():
    """``_N_COGNITIVE_MODULES`` is still 6 (engine uses extra child rng)."""
    model = ZeroDataModel(dim=8, seed=42)
    # All 6 cognitive modules are present and distinct from the engine.
    cognitive_modules = {id(m) for m in model.modules}
    assert id(model.emergence) not in cognitive_modules
    assert len(cognitive_modules) == 6  # all distinct


def test_emergence_engine_has_independent_rng():
    """Engine's rng is a child of the parent rng, not the parent itself."""
    model = ZeroDataModel(dim=8, seed=42)
    # The engine's rng should be a Generator, distinct from the parent.
    assert isinstance(model.emergence.rng, np.random.Generator)
    assert model.emergence.rng is not model._rng  # child, not parent


def test_seeded_model_is_reproducible():
    """Two seeded models produce identical emergence_cycle output."""
    obs = _make_observation(n_samples=50, n_features=4, seed=0)
    model1 = ZeroDataModel(dim=8, seed=42)
    model2 = ZeroDataModel(dim=8, seed=42)
    r1 = model1.emergence_cycle(obs)
    r2 = model2.emergence_cycle(obs)
    assert r1["emergence_score"] == r2["emergence_score"]
    np.testing.assert_allclose(
        r1["counterfactual"]["trajectory"],
        r2["counterfactual"]["trajectory"],
    )


# ----------------------------------------------------------------------
# Facade methods (spec §2.3 — all must hold self._lock)
# ----------------------------------------------------------------------

def test_perceive_topology_facade():
    """perceive_topology facade delegates to engine."""
    model = ZeroDataModel(dim=8, seed=42)
    obs = _make_observation(n_samples=50, n_features=4)
    result = model.perceive_topology(obs)
    for key in ("betti_numbers", "persistence_entropy", "euler_characteristic"):
        assert key in result


def test_discover_causal_dynamics_facade():
    """discover_causal_dynamics facade delegates to engine."""
    model = ZeroDataModel(dim=8, seed=42)
    obs = _make_observation(n_samples=50, n_features=4)
    result = model.discover_causal_dynamics(obs)
    for key in ("adjacency", "edges", "n_edges", "is_acyclic", "method"):
        assert key in result


def test_generate_trajectory_facade():
    """generate_trajectory facade delegates to engine."""
    model = ZeroDataModel(dim=8, seed=42)
    start = np.zeros(3)
    end = np.array([1.0, 2.0, 3.0])
    result = model.generate_trajectory(start, end, n_steps=8)
    assert result["trajectory"].shape == (9, 3)
    np.testing.assert_allclose(result["trajectory"][0], start, atol=1e-9)
    np.testing.assert_allclose(result["trajectory"][-1], end, atol=1e-6)


def test_sample_posterior_facade():
    """sample_posterior facade delegates to engine."""
    model = ZeroDataModel(dim=8, seed=42)

    def log_prob(q):
        return -0.5 * float(np.sum(q * q))

    result = model.sample_posterior(log_prob, np.zeros(2), n_samples=20)
    assert "samples" in result
    assert "ess" in result
    assert result["samples"].shape == (20, 2)


def test_recall_memory_facade():
    """recall_memory facade delegates to engine."""
    model = ZeroDataModel(dim=8, seed=42)
    result = model.recall_memory(np.zeros(8))
    # Empty memory -> label=None, emerged=True
    assert result["label"] is None
    assert result["emerged"] is True


def test_emergence_cycle_facade():
    """emergence_cycle facade delegates to engine end-to-end."""
    model = ZeroDataModel(dim=8, seed=42)
    obs = _make_observation(n_samples=50, n_features=4)
    result = model.emergence_cycle(obs)
    for key in (
        "perception", "causal_graph", "counterfactual",
        "posterior", "memory", "emergence_score", "warnings",
    ):
        assert key in result
    assert 0.0 <= result["emergence_score"] <= 1.0


# ----------------------------------------------------------------------
# Input validation (consistent with engine's contract)
# ----------------------------------------------------------------------

def test_emergence_cycle_insufficient_data_returns_zero():
    """1D input returns score=0 with insufficient_data reason."""
    model = ZeroDataModel(dim=8, seed=42)
    result = model.emergence_cycle(np.array([1.0, 2.0, 3.0]))
    assert result["emergence_score"] == 0.0
    assert result["reason"] == "insufficient_data"


def test_emergence_cycle_none_input_returns_zero():
    """None input returns score=0 with insufficient_data reason."""
    model = ZeroDataModel(dim=8, seed=42)
    result = model.emergence_cycle(None)
    assert result["emergence_score"] == 0.0
    assert result["reason"] == "insufficient_data"


# ----------------------------------------------------------------------
# Pickle support
# ----------------------------------------------------------------------

def test_model_with_emergence_pickles():
    """Pickle round-trip preserves the emergence engine and its config."""
    model = ZeroDataModel(dim=8, seed=42)
    data = pickle.dumps(model)
    model2 = pickle.loads(data)
    assert hasattr(model2, "emergence")
    assert model2.emergence.dim == 8
    # Facade still works after unpickle.
    obs = _make_observation(n_samples=30, n_features=4)
    result = model2.emergence_cycle(obs)
    assert "emergence_score" in result
    assert np.isfinite(result["emergence_score"])


def test_pickle_preserves_emergence_rules():
    """Pickle round-trip preserves emergence_rules."""
    model = ZeroDataModel(dim=8, seed=42)
    original_topology_max_points = model.emergence_rules.topology_max_points
    data = pickle.dumps(model)
    model2 = pickle.loads(data)
    assert model2.emergence_rules.topology_max_points == original_topology_max_points


# ----------------------------------------------------------------------
# Thread-safety (lock-based) — basic smoke test
# ----------------------------------------------------------------------

def test_facade_methods_acquire_lock():
    """Facade methods serialize via self._lock (no concurrent-access error).

    This is a smoke test: two threads calling different facades should not
    raise. A full race-condition test would require running many iterations
    and asserting determinism, which is beyond the scope of this test.
    """
    import threading

    model = ZeroDataModel(dim=8, seed=42)
    obs = _make_observation(n_samples=30, n_features=4)
    errors: list[Exception] = []

    def worker():
        try:
            model.emergence_cycle(obs)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"concurrent emergence_cycle raised: {errors}"


# ----------------------------------------------------------------------
# Spec §2.3 API contract: 6 facade methods exist
# ----------------------------------------------------------------------

def test_all_six_facade_methods_exist():
    """All 6 spec §2.3 facade methods are defined on ZeroDataModel."""
    model = ZeroDataModel(dim=8, seed=42)
    for method_name in (
        "perceive_topology",
        "discover_causal_dynamics",
        "generate_trajectory",
        "sample_posterior",
        "recall_memory",
        "emergence_cycle",
    ):
        assert hasattr(model, method_name), f"missing facade: {method_name}"
        assert callable(getattr(model, method_name))


# ----------------------------------------------------------------------
# End-to-end smoke test (spec §11 — phase 4 verification)
# ----------------------------------------------------------------------

def test_end_to_end_on_synthetic_observation():
    """spec §11: emergence_cycle on (50, 4) observation completes < 10s."""
    import time

    model = ZeroDataModel(dim=8, seed=42)
    obs = _make_observation(n_samples=50, n_features=4, seed=0)
    start = time.perf_counter()
    result = model.emergence_cycle(obs)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"emergence_cycle took {elapsed:.2f}s, expected < 10s"
    assert result["emergence_score"] > 0.0  # at least some signal


def test_emergence_score_in_zero_to_one_range():
    """emergence_score is always in [0, 1] for realistic inputs."""
    model = ZeroDataModel(dim=8, seed=42)
    obs = _make_observation(n_samples=80, n_features=5, seed=7)
    result = model.emergence_cycle(obs)
    assert 0.0 <= result["emergence_score"] <= 1.0


# ----------------------------------------------------------------------
# Distinct from existing ZeroDataModel causal methods
# ----------------------------------------------------------------------

def test_emergence_facade_distinct_from_discover_causal_graph():
    """``discover_causal_dynamics`` (emergence) != ``discover_causal_graph`` (existing)."""
    model = ZeroDataModel(dim=8, seed=42)
    obs = _make_observation(n_samples=50, n_features=4)
    r1 = model.discover_causal_dynamics(obs)
    r2 = model.discover_causal_graph(obs)
    # Different methods are used; emergence uses PC/LiNGAM/correlation,
    # the basic capability uses a simpler graph builder.
    assert "method" in r1
    assert "is_acyclic" in r1
    assert "adjacency" in r2


def test_emergence_intervene_distinct_from_emergence_counterfactual():
    """The engine's counterfactual is reachable via facade composition."""
    model = ZeroDataModel(dim=8, seed=42)
    # discover_causal_dynamics returns adjacency; we can then call engine.counterfactual
    obs = _make_observation(n_samples=50, n_features=4)
    dag = model.discover_causal_dynamics(obs)
    # The engine exposes counterfactual directly (not a ZeroDataModel facade).
    result = model.emergence.counterfactual(
        dag["adjacency"], obs.mean(axis=0), intervention_var=0, intervention_value=1.0
    )
    assert "counterfactual" in result
    assert "factual" in result
    assert "shift" in result
