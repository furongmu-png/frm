# tests/test_phase7_causal_integration.py
"""Phase 7 — Causal integration tests across facade / CLI / API / MCP.

Mirrors the Phase 7 Audio / Graph / Robotics / Time / Code / Reasoning
integration test pattern. Verifies that the 7 Causal facade methods are
reachable from all four entry-point layers (facade / CLI / API / MCP)
and produce consistent results.

The Causal facade is unique among Phase 7 domains in that all 7 query
methods are stateless (no ``add_*`` mutators that affect later queries —
``counterfactual`` and ``intervene`` use the active_inference
free-energy score, but only as a read-only scoring function, not a
state mutation). So each tool can delegate directly to
``self.model.<method>()`` without a fresh-model indirection.

Total: ~40 tests, all defensive against missing optional deps
(fastapi/httpx) and CLI subprocess failures.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

# Web stack is optional; skip API tests when missing.
try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    _HAS_WEB = True
except ImportError:
    _HAS_WEB = False

from zero_data_model import api as api_module
from zero_data_model.mcp_server import ZeroDataMCPServer
from zero_data_model.model import ZeroDataModel
from zero_data_model.persistence import set_persistence_root


# --------------------------------------------------------------------- #
# Shared fixtures — deterministic test data.
# --------------------------------------------------------------------- #

# Decision tree: 1 feature, perfect split at value 1.5.
DECISION_TREE_CONFIG = {
    "features": [[1.0], [1.0], [2.0], [2.0]],
    "labels": ["a", "a", "b", "b"],
}

# 2-player normal-form game (Stag Hunt variant).
# payoff_a = [[4, 1], [3, 2]]; payoff_b omitted => zero-sum.
GAME_CONFIG_ZERO_SUM = {
    "payoff_a": [[3.0, 0.0], [5.0, 1.0]],
}
# 2-player general-sum game (Stag Hunt).
GAME_CONFIG_GENERAL = {
    "payoff_a": [[4.0, 1.0], [3.0, 2.0]],
    "payoff_b": [[4.0, 3.0], [1.0, 2.0]],
}

# Counterfactual: replace index 1 with 10.0.
COUNTERFACTUAL_CONFIG = {
    "observed": [1.0, 2.0, 3.0, 4.0],
    "index": 1,
    "value": 10.0,
}

# Bandit: 3 arms, arm 0 pulled 3 times, arm 1 once, arm 2 never.
BANDIT_CONFIG = {
    "rewards_history": [[1.0, 0.5, 0.8], [0.2], []],
}

# POMDP: 2-state, 2-action MDP. State 0 is the "good" state.
POMDP_CONFIG = {
    "transitions": [[[0.9, 0.1], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]],
    "observations": [[1.0, 0.0], [0.0, 1.0]],
    "rewards": [1.0, 0.0],
}

# Causal graph discovery: 3 perfectly-correlated columns.
DISCOVER_GRAPH_CONFIG = {
    "data": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0],
             [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
    "var_names": ["x", "y", "z"],
}

# Intervention: do(X[0] = 100) on a 3x2 dataset.
INTERVENE_CONFIG = {
    "data": [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    "intervention_var": 0,
    "intervention_value": 100.0,
}


@pytest.fixture()
def mcp_server():
    return ZeroDataMCPServer()


@pytest.fixture()
def client(tmp_path):
    if not _HAS_WEB:
        pytest.skip("fastapi or httpx not installed")
    from fastapi.testclient import TestClient

    from zero_data_model.api import create_app

    set_persistence_root(str(tmp_path / "api_persistence"))
    api_module.set_model(ZeroDataModel(dim=8, seed=42))
    if api_module.limiter is not None:
        api_module.limiter.enabled = False
    app = create_app()
    with TestClient(app, base_url="http://localhost") as c:
        yield c
    api_module._model = None


# --------------------------------------------------------------------- #
# 1. Facade — direct method calls
# --------------------------------------------------------------------- #


class TestFacadeCausal:
    def test_fit_decision_tree_returns_expected_structure(self):
        m = ZeroDataModel(dim=8, seed=42)
        features = np.asarray(DECISION_TREE_CONFIG["features"])
        labels = np.asarray(DECISION_TREE_CONFIG["labels"])
        r = m.fit_decision_tree(features, labels)
        assert {"tree", "depth", "n_leaves", "features_used"} <= set(r.keys())
        assert r["depth"] == 1
        assert r["n_leaves"] == 2
        assert r["features_used"] == [0]

    def test_analyze_game_zero_sum_finds_equilibrium(self):
        m = ZeroDataModel(dim=8, seed=42)
        payoff_a = np.asarray(GAME_CONFIG_ZERO_SUM["payoff_a"])
        r = m.analyze_game(payoff_a)
        assert {"nash_equilibria", "value_a", "value_b", "is_zero_sum"} <= set(r.keys())
        assert r["is_zero_sum"] is True
        assert (0, 0) in r["nash_equilibria"]
        assert r["value_a"] == pytest.approx(3.0)
        assert r["value_b"] == pytest.approx(-3.0)

    def test_analyze_game_general_sum_finds_equilibrium(self):
        m = ZeroDataModel(dim=8, seed=42)
        a = np.asarray(GAME_CONFIG_GENERAL["payoff_a"])
        b = np.asarray(GAME_CONFIG_GENERAL["payoff_b"])
        r = m.analyze_game(a, b)
        assert r["is_zero_sum"] is False
        assert (0, 0) in r["nash_equilibria"]
        assert r["value_a"] == pytest.approx(4.0)
        assert r["value_b"] == pytest.approx(4.0)

    def test_counterfactual_returns_expected_keys(self):
        m = ZeroDataModel(dim=8, seed=42)
        observed = np.asarray(COUNTERFACTUAL_CONFIG["observed"])
        intervention = {
            "index": COUNTERFACTUAL_CONFIG["index"],
            "value": COUNTERFACTUAL_CONFIG["value"],
        }
        r = m.counterfactual(observed, intervention)
        assert {"factual", "counterfactual", "effect", "surprisal"} <= set(r.keys())
        assert r["factual"] == pytest.approx(2.5)
        assert r["counterfactual"] == pytest.approx(4.5)
        assert r["effect"] == pytest.approx(2.0)

    def test_select_bandit_arm_returns_expected_keys(self):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.select_bandit_arm(BANDIT_CONFIG["rewards_history"])
        assert {"arm", "method", "expected_values", "confidence_bounds"} <= set(r.keys())
        # Deterministic parts.
        means = np.asarray(r["expected_values"])
        assert means.shape == (3,)
        assert means[0] == pytest.approx(np.mean([1.0, 0.5, 0.8]))
        assert means[1] == pytest.approx(0.2)
        assert means[2] == pytest.approx(0.0)
        # arm is a valid index, method is one of the two strategies.
        assert r["arm"] in (0, 1, 2)
        assert r["method"] in ("explore", "exploit")

    def test_solve_pomdp_returns_expected_keys(self):
        m = ZeroDataModel(dim=8, seed=42)
        T = np.asarray(POMDP_CONFIG["transitions"])
        O = np.asarray(POMDP_CONFIG["observations"])
        R = np.asarray(POMDP_CONFIG["rewards"])
        r = m.solve_pomdp(T, O, R)
        assert {"policy", "value", "iterations", "converged"} <= set(r.keys())
        policy = np.asarray(r["policy"])
        assert policy.shape == (2,)
        value = np.asarray(r["value"])
        assert value.shape == (2,)
        assert r["iterations"] >= 1

    def test_discover_causal_graph_returns_expected_keys(self):
        m = ZeroDataModel(dim=8, seed=42)
        data = np.asarray(DISCOVER_GRAPH_CONFIG["data"])
        r = m.discover_causal_graph(data, var_names=DISCOVER_GRAPH_CONFIG["var_names"])
        assert {"adjacency", "edges", "n_edges", "var_names"} <= set(r.keys())
        assert r["var_names"] == ["x", "y", "z"]
        # All 3 columns are perfectly correlated => 3 edges (0->1, 0->2, 1->2).
        assert r["n_edges"] == 3

    def test_intervene_returns_expected_keys(self):
        m = ZeroDataModel(dim=8, seed=42)
        data = np.asarray(INTERVENE_CONFIG["data"])
        r = m.intervene(data, 0, 100.0)
        assert {"pre_intervention_mean", "post_intervention_mean",
                "effect", "surprisal"} <= set(r.keys())
        np.testing.assert_allclose(
            np.asarray(r["pre_intervention_mean"]), [2.0, 20.0], atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(r["post_intervention_mean"]), [100.0, 20.0], atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(r["effect"]), [98.0, 0.0], atol=1e-10
        )


# --------------------------------------------------------------------- #
# 2. CLI — subprocess invocations
# --------------------------------------------------------------------- #


class TestCLICausal:
    _env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONPATH": "src"}

    def _run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "zero_data_model"] + list(args),
            capture_output=True,
            text=True,
            timeout=30,
            env=self._env,
        )

    def _write_json(self, tmp_path, name, obj):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return str(p)

    def test_cli_decision_tree(self, tmp_path):
        f = self._write_json(tmp_path, "dt.json", DECISION_TREE_CONFIG)
        r = self._run_cli("causal", "decision-tree", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["depth"] == 1
            assert payload["n_leaves"] == 2
            assert payload["features_used"] == [0]

    def test_cli_game_zero_sum(self, tmp_path):
        f = self._write_json(tmp_path, "g.json", GAME_CONFIG_ZERO_SUM)
        r = self._run_cli("causal", "game", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["is_zero_sum"] is True
            assert [0, 0] in payload["nash_equilibria"]

    def test_cli_game_general(self, tmp_path):
        f = self._write_json(tmp_path, "g.json", GAME_CONFIG_GENERAL)
        r = self._run_cli("causal", "game", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["is_zero_sum"] is False
            assert [0, 0] in payload["nash_equilibria"]

    def test_cli_counterfactual(self, tmp_path):
        f = self._write_json(tmp_path, "cf.json", COUNTERFACTUAL_CONFIG)
        r = self._run_cli("causal", "counterfactual", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["factual"] == pytest.approx(2.5)
            assert payload["counterfactual"] == pytest.approx(4.5)
            assert payload["effect"] == pytest.approx(2.0)

    def test_cli_bandit(self, tmp_path):
        f = self._write_json(tmp_path, "b.json", BANDIT_CONFIG)
        r = self._run_cli("causal", "bandit", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "expected_values" in payload
            assert "confidence_bounds" in payload
            assert payload["arm"] in (0, 1, 2)

    def test_cli_pomdp(self, tmp_path):
        f = self._write_json(tmp_path, "p.json", POMDP_CONFIG)
        r = self._run_cli("causal", "pomdp", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "policy" in payload
            assert "value" in payload
            assert payload["iterations"] >= 1

    def test_cli_discover_graph(self, tmp_path):
        f = self._write_json(tmp_path, "dg.json", DISCOVER_GRAPH_CONFIG)
        r = self._run_cli("causal", "discover-graph", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["n_edges"] == 3
            assert payload["var_names"] == ["x", "y", "z"]

    def test_cli_intervene(self, tmp_path):
        f = self._write_json(tmp_path, "i.json", INTERVENE_CONFIG)
        r = self._run_cli("causal", "intervene", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "effect" in payload
            assert "surprisal" in payload


# --------------------------------------------------------------------- #
# 3. API — FastAPI endpoints
# --------------------------------------------------------------------- #


class TestAPICausal:
    def test_api_decision_tree(self, client):
        r = client.post("/causal/decision-tree", json=DECISION_TREE_CONFIG)
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["depth"] == 1
        assert payload["n_leaves"] == 2
        assert payload["features_used"] == [0]

    def test_api_game_zero_sum(self, client):
        r = client.post("/causal/game", json=GAME_CONFIG_ZERO_SUM)
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["is_zero_sum"] is True
        assert [0, 0] in payload["nash_equilibria"]
        assert payload["value_a"] == pytest.approx(3.0)

    def test_api_game_general(self, client):
        r = client.post("/causal/game", json=GAME_CONFIG_GENERAL)
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["is_zero_sum"] is False
        assert [0, 0] in payload["nash_equilibria"]

    def test_api_counterfactual(self, client):
        r = client.post("/causal/counterfactual", json=COUNTERFACTUAL_CONFIG)
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["factual"] == pytest.approx(2.5)
        assert payload["counterfactual"] == pytest.approx(4.5)
        assert payload["effect"] == pytest.approx(2.0)

    def test_api_bandit(self, client):
        r = client.post("/causal/bandit", json=BANDIT_CONFIG)
        assert r.status_code == 200, r.text
        payload = r.json()
        assert "expected_values" in payload
        assert "confidence_bounds" in payload
        assert payload["arm"] in (0, 1, 2)

    def test_api_pomdp(self, client):
        r = client.post("/causal/pomdp", json=POMDP_CONFIG)
        assert r.status_code == 200, r.text
        payload = r.json()
        assert "policy" in payload
        assert "value" in payload
        assert payload["iterations"] >= 1

    def test_api_discover_graph(self, client):
        r = client.post("/causal/discover-graph", json=DISCOVER_GRAPH_CONFIG)
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_edges"] == 3
        assert payload["var_names"] == ["x", "y", "z"]

    def test_api_intervene(self, client):
        r = client.post("/causal/intervene", json=INTERVENE_CONFIG)
        assert r.status_code == 200, r.text
        payload = r.json()
        assert "effect" in payload
        assert "surprisal" in payload
        assert payload["effect"][0] == pytest.approx(98.0)
        assert payload["effect"][1] == pytest.approx(0.0)

    # --- negative paths (Pydantic rejection -> 422 / 400) ---
    def test_api_decision_tree_mismatched_lengths_rejected(self, client):
        r = client.post(
            "/causal/decision-tree",
            json={"features": [[1.0], [2.0]], "labels": ["a"]},
        )
        assert r.status_code in (400, 422)

    def test_api_game_empty_payoff_rejected(self, client):
        r = client.post("/causal/game", json={"payoff_a": []})
        assert r.status_code in (400, 422)

    def test_api_counterfactual_empty_observed_rejected(self, client):
        r = client.post(
            "/causal/counterfactual",
            json={"observed": [], "index": 0, "value": 1.0},
        )
        assert r.status_code in (400, 422)

    def test_api_intervene_negative_var_rejected(self, client):
        r = client.post(
            "/causal/intervene",
            json={
                "data": [[1.0, 2.0], [3.0, 4.0]],
                "intervention_var": -1,
                "intervention_value": 5.0,
            },
        )
        assert r.status_code in (400, 422)

    def test_api_discover_graph_single_row_rejected(self, client):
        # Pydantic enforces min_length=2 (correlation needs >= 2 rows).
        r = client.post(
            "/causal/discover-graph",
            json={"data": [[1.0, 2.0]]},
        )
        assert r.status_code in (400, 422)


# --------------------------------------------------------------------- #
# 4. MCP — ZeroDataMCPServer.call_tool
# --------------------------------------------------------------------- #


class TestMCPCausal:
    def test_mcp_decision_tree(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_fit_decision_tree",
            features=DECISION_TREE_CONFIG["features"],
            labels=DECISION_TREE_CONFIG["labels"],
        )
        assert r["depth"] == 1
        assert r["n_leaves"] == 2
        assert r["features_used"] == [0]

    def test_mcp_game_zero_sum(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_analyze_game",
            payoff_a=GAME_CONFIG_ZERO_SUM["payoff_a"],
        )
        assert r["is_zero_sum"] is True
        assert [0, 0] in r["nash_equilibria"]
        assert r["value_a"] == pytest.approx(3.0)

    def test_mcp_game_general(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_analyze_game",
            payoff_a=GAME_CONFIG_GENERAL["payoff_a"],
            payoff_b=GAME_CONFIG_GENERAL["payoff_b"],
        )
        assert r["is_zero_sum"] is False
        assert [0, 0] in r["nash_equilibria"]

    def test_mcp_counterfactual(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_counterfactual",
            observed=COUNTERFACTUAL_CONFIG["observed"],
            index=COUNTERFACTUAL_CONFIG["index"],
            value=COUNTERFACTUAL_CONFIG["value"],
        )
        assert r["factual"] == pytest.approx(2.5)
        assert r["counterfactual"] == pytest.approx(4.5)
        assert r["effect"] == pytest.approx(2.0)

    def test_mcp_bandit(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_select_bandit_arm",
            rewards_history=BANDIT_CONFIG["rewards_history"],
        )
        assert "expected_values" in r
        assert "confidence_bounds" in r
        # Deterministic parts.
        assert r["expected_values"][0] == pytest.approx(np.mean([1.0, 0.5, 0.8]))
        assert r["expected_values"][1] == pytest.approx(0.2)
        assert r["expected_values"][2] == pytest.approx(0.0)
        assert r["arm"] in (0, 1, 2)

    def test_mcp_pomdp(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_solve_pomdp",
            transitions=POMDP_CONFIG["transitions"],
            observations=POMDP_CONFIG["observations"],
            rewards=POMDP_CONFIG["rewards"],
        )
        assert "policy" in r
        assert "value" in r
        assert r["iterations"] >= 1

    def test_mcp_discover_graph(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_discover_graph",
            data=DISCOVER_GRAPH_CONFIG["data"],
            var_names=DISCOVER_GRAPH_CONFIG["var_names"],
        )
        assert r["n_edges"] == 3
        assert r["var_names"] == ["x", "y", "z"]

    def test_mcp_intervene(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_intervene",
            data=INTERVENE_CONFIG["data"],
            intervention_var=0,
            intervention_value=100.0,
        )
        assert "effect" in r
        assert "surprisal" in r
        assert r["effect"][0] == pytest.approx(98.0)
        assert r["effect"][1] == pytest.approx(0.0)

    # --- negative paths ---
    def test_mcp_decision_tree_mismatched_lengths(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_fit_decision_tree",
            features=[[1.0], [2.0]],
            labels=["a"],
        )
        assert "error" in r

    def test_mcp_counterfactual_empty_observed(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_counterfactual",
            observed=[],
            index=0,
            value=1.0,
        )
        assert "error" in r

    def test_mcp_discover_graph_1d_data(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_discover_graph",
            data=[1.0, 2.0, 3.0],
        )
        assert "error" in r

    def test_mcp_discover_graph_single_row(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_discover_graph",
            data=[[1.0, 2.0]],
        )
        assert "error" in r

    def test_mcp_intervene_non_2d_data(self, mcp_server):
        r = mcp_server.call_tool(
            "causal_intervene",
            data=[1.0, 2.0, 3.0],
            intervention_var=0,
            intervention_value=5.0,
        )
        assert "error" in r

    def test_mcp_causal_tools_registered(self, mcp_server):
        tools = mcp_server.list_tools()
        for name in (
            "causal_fit_decision_tree",
            "causal_analyze_game",
            "causal_counterfactual",
            "causal_select_bandit_arm",
            "causal_solve_pomdp",
            "causal_discover_graph",
            "causal_intervene",
        ):
            assert name in tools, f"missing tool: {name}"


# --------------------------------------------------------------------- #
# 5. Cross-layer consistency
# --------------------------------------------------------------------- #


class TestCrossLayerConsistency:
    """Causal facade methods produce identical results across facade / MCP.

    All 7 Causal facade methods are stateless (no ``add_*`` mutators).
    The 6 deterministic methods (decision-tree, game, counterfactual,
    pomdp, discover-graph, intervene) produce identical numerical
    results regardless of model dim. The 7th method (select_bandit_arm)
    is non-deterministic (uses an unseeded RNG internally), so only its
    deterministic parts (expected_values, confidence_bounds) are
    compared.

    Note: ``counterfactual`` and ``intervene`` include a ``surprisal``
    field scored by ``active_inference.compute_free_energy``, which
    depends on the engine's ``belief_state`` (model-state dependent).
    The algebraic fields (factual / counterfactual / effect / means)
    are pure and must match exactly; surprisal may differ, so it is
    excluded from cross-layer comparison.
    """

    def test_facade_and_mcp_decision_tree_agree(self, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.fit_decision_tree(
            np.asarray(DECISION_TREE_CONFIG["features"]),
            np.asarray(DECISION_TREE_CONFIG["labels"]),
        )
        mcp = mcp_server.call_tool(
            "causal_fit_decision_tree",
            features=DECISION_TREE_CONFIG["features"],
            labels=DECISION_TREE_CONFIG["labels"],
        )
        assert facade["depth"] == mcp["depth"]
        assert facade["n_leaves"] == mcp["n_leaves"]
        assert facade["features_used"] == mcp["features_used"]

    def test_facade_and_mcp_game_zero_sum_agree(self, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.analyze_game(
            np.asarray(GAME_CONFIG_ZERO_SUM["payoff_a"])
        )
        mcp = mcp_server.call_tool(
            "causal_analyze_game",
            payoff_a=GAME_CONFIG_ZERO_SUM["payoff_a"],
        )
        # Facade returns tuples, MCP _to_py converts them to lists —
        # compare via list().
        assert [list(e) for e in facade["nash_equilibria"]] == mcp["nash_equilibria"]
        assert facade["value_a"] == pytest.approx(mcp["value_a"])
        assert facade["value_b"] == pytest.approx(mcp["value_b"])
        assert facade["is_zero_sum"] == mcp["is_zero_sum"]

    def test_facade_and_mcp_game_general_sum_agree(self, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.analyze_game(
            np.asarray(GAME_CONFIG_GENERAL["payoff_a"]),
            np.asarray(GAME_CONFIG_GENERAL["payoff_b"]),
        )
        mcp = mcp_server.call_tool(
            "causal_analyze_game",
            payoff_a=GAME_CONFIG_GENERAL["payoff_a"],
            payoff_b=GAME_CONFIG_GENERAL["payoff_b"],
        )
        assert [list(e) for e in facade["nash_equilibria"]] == mcp["nash_equilibria"]
        assert facade["value_a"] == pytest.approx(mcp["value_a"])
        assert facade["value_b"] == pytest.approx(mcp["value_b"])

    def test_facade_and_mcp_counterfactual_algebra_agree(self, mcp_server):
        # surprisal depends on active_inference.belief_state (model-state
        # dependent), so only the algebraic fields are compared.
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.counterfactual(
            np.asarray(COUNTERFACTUAL_CONFIG["observed"]),
            {"index": COUNTERFACTUAL_CONFIG["index"],
             "value": COUNTERFACTUAL_CONFIG["value"]},
        )
        mcp = mcp_server.call_tool(
            "causal_counterfactual",
            observed=COUNTERFACTUAL_CONFIG["observed"],
            index=COUNTERFACTUAL_CONFIG["index"],
            value=COUNTERFACTUAL_CONFIG["value"],
        )
        assert facade["factual"] == pytest.approx(mcp["factual"])
        assert facade["counterfactual"] == pytest.approx(mcp["counterfactual"])
        assert facade["effect"] == pytest.approx(mcp["effect"])

    def test_facade_and_mcp_bandit_expected_values_agree(self, mcp_server):
        # select_bandit_arm uses an unseeded RNG, so only the
        # deterministic expected_values / confidence_bounds are compared.
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.select_bandit_arm(BANDIT_CONFIG["rewards_history"])
        mcp = mcp_server.call_tool(
            "causal_select_bandit_arm",
            rewards_history=BANDIT_CONFIG["rewards_history"],
        )
        np.testing.assert_allclose(
            np.asarray(facade["expected_values"]),
            np.asarray(mcp["expected_values"]),
            atol=1e-12,
        )
        # confidence_bounds[2] is +Inf for unpulled arms; only compare
        # finite entries (arms 0 and 1).
        np.testing.assert_allclose(
            np.asarray(facade["confidence_bounds"][:2]),
            np.asarray(mcp["confidence_bounds"][:2]),
            atol=1e-12,
        )

    def test_facade_and_mcp_pomdp_agree(self, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.solve_pomdp(
            np.asarray(POMDP_CONFIG["transitions"]),
            np.asarray(POMDP_CONFIG["observations"]),
            np.asarray(POMDP_CONFIG["rewards"]),
        )
        mcp = mcp_server.call_tool(
            "causal_solve_pomdp",
            transitions=POMDP_CONFIG["transitions"],
            observations=POMDP_CONFIG["observations"],
            rewards=POMDP_CONFIG["rewards"],
        )
        np.testing.assert_array_equal(
            np.asarray(facade["policy"]),
            np.asarray(mcp["policy"]),
        )
        np.testing.assert_allclose(
            np.asarray(facade["value"]),
            np.asarray(mcp["value"]),
            atol=1e-10,
        )
        assert facade["iterations"] == mcp["iterations"]
        assert facade["converged"] == mcp["converged"]

    def test_facade_and_mcp_discover_graph_agree(self, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.discover_causal_graph(
            np.asarray(DISCOVER_GRAPH_CONFIG["data"]),
            var_names=DISCOVER_GRAPH_CONFIG["var_names"],
        )
        mcp = mcp_server.call_tool(
            "causal_discover_graph",
            data=DISCOVER_GRAPH_CONFIG["data"],
            var_names=DISCOVER_GRAPH_CONFIG["var_names"],
        )
        # Facade returns tuples, MCP _to_py converts them to lists.
        assert [list(e) for e in facade["edges"]] == mcp["edges"]
        assert facade["n_edges"] == mcp["n_edges"]
        np.testing.assert_allclose(
            np.asarray(facade["adjacency"]),
            np.asarray(mcp["adjacency"]),
            atol=1e-12,
        )

    def test_facade_and_mcp_intervene_algebra_agree(self, mcp_server):
        # surprisal depends on active_inference.belief_state (model-state
        # dependent), so only the algebraic fields are compared.
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.intervene(
            np.asarray(INTERVENE_CONFIG["data"]),
            INTERVENE_CONFIG["intervention_var"],
            INTERVENE_CONFIG["intervention_value"],
        )
        mcp = mcp_server.call_tool(
            "causal_intervene",
            data=INTERVENE_CONFIG["data"],
            intervention_var=INTERVENE_CONFIG["intervention_var"],
            intervention_value=INTERVENE_CONFIG["intervention_value"],
        )
        np.testing.assert_allclose(
            np.asarray(facade["pre_intervention_mean"]),
            np.asarray(mcp["pre_intervention_mean"]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(facade["post_intervention_mean"]),
            np.asarray(mcp["post_intervention_mean"]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(facade["effect"]),
            np.asarray(mcp["effect"]),
            atol=1e-12,
        )
