# tests/test_causal_emergence_engine.py
"""Tests for the CausalEmergenceEngine orchestrator (phase 1: foundation).

Phase 1 covers module A (topology) and module B (causal_discovery) facades.
Phases 2-4 (modules C-E and emergence_cycle) will be tested in later files.
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


def test_engine_constructs_with_defaults():
    """Engine constructs without arguments (uses defaults)."""
    engine = CausalEmergenceEngine()
    assert engine.topology is not None
    assert engine.causal is not None
    assert engine.differential is None  # phase 2
    assert engine.hmc is None  # phase 3
    assert engine.memory is None  # phase 3


def test_engine_perceive_topology_facade():
    """perceive_topology facade delegates to module A."""
    rng = np.random.default_rng(42)
    engine = CausalEmergenceEngine(rules=EmergenceRules(), rng=rng)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    result = engine.perceive_topology(points)
    assert result["betti_numbers"][0] == 1
    assert result["betti_numbers"][1] >= 1  # circle has a loop


def test_engine_discover_causal_dynamics_facade():
    """discover_causal_dynamics facade delegates to module B."""
    rng = np.random.default_rng(123)
    engine = CausalEmergenceEngine(rules=EmergenceRules(), rng=rng)
    n = 200
    x0 = rng.standard_normal(n)
    x1 = 0.8 * x0 + 0.2 * rng.standard_normal(n)
    x2 = 0.6 * x1 + 0.3 * rng.standard_normal(n)
    data = np.column_stack([x0, x1, x2])
    result = engine.discover_causal_dynamics(data)
    assert result["is_acyclic"]
    assert result["method"] == "pc"


def test_engine_intervene_facade():
    """intervene facade delegates to module B do-calculus."""
    rng = np.random.default_rng(123)
    engine = CausalEmergenceEngine(rules=EmergenceRules(), rng=rng)
    n = 100
    x0 = rng.standard_normal(n)
    x1 = 0.8 * x0 + 0.2 * rng.standard_normal(n)
    data = np.column_stack([x0, x1])
    dag = engine.discover_causal_dynamics(data)
    result = engine.intervene(
        dag["adjacency"], data, intervention_var=0, intervention_value=1.0
    )
    assert "effect" in result
    assert "post_mean" in result


def test_engine_counterfactual_facade():
    """counterfactual facade delegates to module B."""
    rng = np.random.default_rng(0)
    engine = CausalEmergenceEngine(rules=EmergenceRules(), rng=rng)
    adj = np.array([[0, 1], [0, 0]])
    observed = np.array([0.5, 0.4])
    result = engine.counterfactual(
        adj, observed, intervention_var=0, intervention_value=1.0
    )
    assert "counterfactual" in result
    assert "factual" in result
    assert "shift" in result


def test_engine_phase2_facade_raises():
    """Phase 2-4 facades raise NotImplementedError in phase 1."""
    engine = CausalEmergenceEngine()
    with pytest.raises(NotImplementedError):
        engine.generate_trajectory({})
    with pytest.raises(NotImplementedError):
        engine.sample_posterior(lambda x: 0.0, np.zeros(4))
    with pytest.raises(NotImplementedError):
        engine.recall_memory(np.zeros(4))
    with pytest.raises(NotImplementedError):
        engine.emergence_cycle(np.zeros((10, 4)))


def test_engine_accepts_core_modules():
    """Engine accepts optional core modules without using them in phase 1."""
    engine = CausalEmergenceEngine(
        dim=32,
        active_inference=None,
        math_universe=None,
        rules=EmergenceRules(),
    )
    assert engine.dim == 32
    assert engine.active_inference is None
    assert engine.math_universe is None


def test_engine_uses_shared_rng():
    """Engine passes its rng to submodules."""
    rng = np.random.default_rng(42)
    engine = CausalEmergenceEngine(rules=EmergenceRules(), rng=rng)
    assert engine.topology.rng is rng
    assert engine.causal.rng is rng


def test_engine_rules_propagated():
    """Engine's rules propagate to submodules."""
    rules = EmergenceRules(topology_max_points=8)
    engine = CausalEmergenceEngine(rules=rules)
    assert engine.topology.rules is rules
    assert engine.causal.rules is rules


def test_engine_default_rules_when_none():
    """Engine creates default rules when none provided."""
    engine = CausalEmergenceEngine(rules=None)
    assert engine.rules is not None
    assert engine.rules.topology_max_points == 16
    assert engine.rules.causal_method == "pc"


def test_engine_default_rng_when_none():
    """Engine creates default rng when none provided."""
    engine = CausalEmergenceEngine()
    assert engine.rng is not None
    assert engine.topology.rng is engine.rng
