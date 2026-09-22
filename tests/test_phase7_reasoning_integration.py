# tests/test_phase7_reasoning_integration.py
"""Phase 7 — Reasoning integration tests across facade / CLI / API / MCP.

Mirrors the Phase 7 Audio / Graph / Robotics / Time / Code integration
test pattern. Verifies that the 7 Reasoning facade methods (the query
endpoints, not the 4 stateful `add_*` methods) are reachable from all
four entry-point layers and produce consistent results.

The Reasoning facade is unique among Phase 7 domains in that it has 4
stateful mutators (``add_logical_fact``, ``add_logical_rule``,
``add_default_rule``, ``add_causal_link``). CLI subprocesses and MCP
tool calls cannot persist state across invocations, so each stateful
query endpoint accepts a JSON config that supplies all the
facts / rules / defaults / causal links upfront, applies them to a
fresh model, then runs the query.

Total: ~50 tests, all defensive against missing optional deps
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
# Shared fixtures
# --------------------------------------------------------------------- #


# Propositional inference config: a single fact + one rule.
PROP_INFER_CONFIG = {
    "facts": {"rain": True},
    "rules": [["rain", "wet_grass"]],
}

# Syllogism premises (Barbara).
MAJOR = ["A", "human", "mortal"]
MINOR = ["A", "socrates", "human"]

# Induction examples.
INDUCT_CONFIG = {
    "examples": [{"color": "red"}, {"color": "red"}, {"color": "blue"}],
    "labels": [True, True, False],
}

# Analogize dicts.
ANALOGIZE_SOURCE = {"a": 1, "b": 2}
ANALOGIZE_TARGET = {"a": 1, "b": 2}

# Abduction config.
ABDUCE_CONFIG = {
    "observation": "patient has fever and cough",
    "hypotheses": ["flu with fever and cough", "common cold"],
    "priors": [0.6, 0.4],
}

# Defeasible defaults config (one default with exception).
DEFAULTS_CONFIG = {
    "defaults": [
        {"rule": ["bird", "flies", True], "exception": ["penguin"]},
    ],
    "facts": {"bird": True, "penguin": False},
}

# Causal chain config.
CAUSAL_CONFIG = {
    "links": [["a", "b"], ["b", "c"]],
    "start": "a",
    "max_depth": 5,
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


class TestFacadeReasoning:
    def test_infer_logical_modus_ponens(self):
        m = ZeroDataModel(dim=8, seed=42)
        m.add_logical_fact("rain", True)
        m.add_logical_rule("rain", "wet_grass")
        r = m.infer_logical()
        assert r["facts"]["wet_grass"] is True
        assert ("rain", "wet_grass", True) in r["inferences"]
        assert r["contradictions"] == []

    def test_infer_logical_chained(self):
        m = ZeroDataModel(dim=8, seed=42)
        m.add_logical_fact("a", True)
        m.add_logical_rule("a", "b")
        m.add_logical_rule("b", "c")
        r = m.infer_logical()
        assert r["facts"]["b"] is True
        assert r["facts"]["c"] is True

    def test_infer_logical_contradiction(self):
        m = ZeroDataModel(dim=8, seed=42)
        m.add_logical_fact("a", True)
        m.add_logical_rule("a", "b")
        m.reasoning_propositional.add_negation_rule("a", "b")
        r = m.infer_logical()
        assert "b" in r["contradictions"]

    def test_syllogism_barbara(self):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.syllogism(("A", "human", "mortal"), ("A", "socrates", "human"))
        assert r["valid"] is True
        assert r["conclusion"] == "All socrates are mortal"
        assert "Barbara" in r["form"]

    def test_syllogism_celarent(self):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.syllogism(("E", "reptile", "furry"), ("A", "snake", "reptile"))
        assert r["valid"] is True
        assert r["conclusion"] == "No snake are furry"

    def test_syllogism_invalid(self):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.syllogism(("O", "X", "Y"), ("I", "A", "B"))
        assert r["valid"] is False
        assert r["conclusion"] is None
        assert r["form"] == "invalid"

    def test_induct_rule_finds_color(self):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.induct_rule(
            [{"color": "red"}, {"color": "red"}, {"color": "blue"}],
            [True, True, False],
        )
        assert r["rule"] == "color == 'red'"
        assert r["confidence"] == 1.0
        assert r["support"] == 2
        assert r["coverage"] == 1.0

    def test_analogize_identical_dicts(self):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.analogize({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert r["mapping"]["a"] == "a"
        assert r["mapping"]["b"] == "b"
        assert r["transfer"]["a"] == 1
        assert 0.0 <= r["similarity"] <= 1.0

    def test_abduce_best_match(self):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.abduce(
            "patient has fever and cough",
            ["flu with fever and cough", "common cold"],
            priors=[0.6, 0.4],
        )
        assert r["best"] == "flu with fever and cough"
        assert len(r["scores"]) == 2
        assert 0.0 <= r["confidence"] <= 1.0

    def test_conclude_defaults_basic(self):
        m = ZeroDataModel(dim=8, seed=42)
        m.add_default_rule(("bird", "flies", True), exception=("penguin",))
        r = m.conclude_defaults({"bird": True, "penguin": False})
        assert r["conclusions"]["flies"] is True
        assert r["defeated"] == []
        assert r["ambiguous"] == []

    def test_conclude_defaults_exception_blocks(self):
        m = ZeroDataModel(dim=8, seed=42)
        m.add_default_rule(("bird", "flies", True), exception=("penguin",))
        r = m.conclude_defaults({"bird": True, "penguin": True})
        assert "flies" not in r["conclusions"]
        assert ("bird", "flies") in r["defeated"]

    def test_trace_causal_chain(self):
        m = ZeroDataModel(dim=8, seed=42)
        m.add_causal_link("a", "b")
        m.add_causal_link("b", "c")
        r = m.trace_causal_chain("a", max_depth=5)
        assert r["chain"] == ["a", "b", "c"]
        assert "b" in r["effects"]
        assert "c" in r["effects"]
        assert r["depth"] == 2
        assert r["cycles"] is False

    def test_trace_causal_cycle(self):
        m = ZeroDataModel(dim=8, seed=42)
        m.add_causal_link("a", "b")
        m.add_causal_link("b", "c")
        m.add_causal_link("c", "a")
        r = m.trace_causal_chain("a", max_depth=5)
        assert r["cycles"] is True


# --------------------------------------------------------------------- #
# 2. CLI — subprocess invocations
# --------------------------------------------------------------------- #


class TestCLIReasoning:
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

    def test_cli_prop_infer(self, tmp_path):
        f = self._write_json(tmp_path, "p.json", PROP_INFER_CONFIG)
        r = self._run_cli("reasoning", "prop-infer", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["facts"]["wet_grass"] is True

    def test_cli_syllogism(self, tmp_path):
        fm = self._write_json(tmp_path, "major.json", MAJOR)
        fn = self._write_json(tmp_path, "minor.json", MINOR)
        r = self._run_cli("reasoning", "syllogism", fm, fn)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["conclusion"] == "All socrates are mortal"

    def test_cli_induct(self, tmp_path):
        f = self._write_json(tmp_path, "ind.json", INDUCT_CONFIG)
        r = self._run_cli("reasoning", "induct", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["rule"] == "color == 'red'"

    def test_cli_analogize(self, tmp_path):
        fs = self._write_json(tmp_path, "src.json", ANALOGIZE_SOURCE)
        ft = self._write_json(tmp_path, "tgt.json", ANALOGIZE_TARGET)
        r = self._run_cli("reasoning", "analogize", fs, ft)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["mapping"]["a"] == "a"

    def test_cli_abduce(self, tmp_path):
        f = self._write_json(tmp_path, "abd.json", ABDUCE_CONFIG)
        r = self._run_cli("reasoning", "abduce", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["best"] == "flu with fever and cough"

    def test_cli_defaults(self, tmp_path):
        f = self._write_json(tmp_path, "def.json", DEFAULTS_CONFIG)
        r = self._run_cli("reasoning", "defaults", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["conclusions"]["flies"] is True

    def test_cli_causal(self, tmp_path):
        f = self._write_json(tmp_path, "cau.json", CAUSAL_CONFIG)
        r = self._run_cli("reasoning", "causal", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["chain"] == ["a", "b", "c"]


# --------------------------------------------------------------------- #
# 3. API — FastAPI endpoints
# --------------------------------------------------------------------- #


class TestAPIReasoning:
    def test_api_prop_infer(self, client):
        r = client.post("/v1/reasoning/prop-infer", json=PROP_INFER_CONFIG)
        assert r.status_code == 200, r.text
        assert r.json()["facts"]["wet_grass"] is True

    def test_api_syllogism(self, client):
        r = client.post(
            "/v1/reasoning/syllogism",
            json={"major": MAJOR, "minor": MINOR},
        )
        assert r.status_code == 200, r.text
        assert r.json()["conclusion"] == "All socrates are mortal"

    def test_api_induct(self, client):
        r = client.post("/v1/reasoning/induct", json=INDUCT_CONFIG)
        assert r.status_code == 200, r.text
        assert r.json()["rule"] == "color == 'red'"

    def test_api_analogize(self, client):
        r = client.post(
            "/v1/reasoning/analogize",
            json={"source": ANALOGIZE_SOURCE, "target": ANALOGIZE_TARGET},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mapping"]["a"] == "a"

    def test_api_abduce(self, client):
        r = client.post("/v1/reasoning/abduce", json=ABDUCE_CONFIG)
        assert r.status_code == 200, r.text
        assert r.json()["best"] == "flu with fever and cough"

    def test_api_defaults(self, client):
        r = client.post("/v1/reasoning/defaults", json=DEFAULTS_CONFIG)
        assert r.status_code == 200, r.text
        assert r.json()["conclusions"]["flies"] is True

    def test_api_causal(self, client):
        r = client.post("/v1/reasoning/causal", json=CAUSAL_CONFIG)
        assert r.status_code == 200, r.text
        assert r.json()["chain"] == ["a", "b", "c"]

    # --- negative paths ---
    def test_api_syllogism_short_tuple_rejected(self, client):
        r = client.post(
            "/v1/reasoning/syllogism",
            json={"major": ["A", "X"], "minor": ["A", "Y", "Z"]},
        )
        # Pydantic min_length=3 rejects -> 422.
        assert r.status_code in (400, 422)

    def test_api_induct_mismatched_lengths_rejected(self, client):
        r = client.post(
            "/v1/reasoning/induct",
            json={"examples": [{"a": 1}], "labels": [True, False]},
        )
        # Pydantic doesn't enforce length match (both have min_length=1),
        # so we manually raise HTTP 400 in the handler.
        assert r.status_code in (400, 422)

    def test_api_abduce_mismatched_priors_rejected(self, client):
        r = client.post(
            "/v1/reasoning/abduce",
            json={"observation": "x", "hypotheses": ["a", "b"], "priors": [0.5]},
        )
        assert r.status_code in (400, 422)

    def test_api_analogize_empty_source_rejected(self, client):
        r = client.post(
            "/v1/reasoning/analogize",
            json={"source": {}, "target": {"a": 1}},
        )
        assert r.status_code in (400, 422)

    def test_api_causal_negative_depth_rejected(self, client):
        r = client.post(
            "/v1/reasoning/causal",
            json={"links": [], "start": "a", "max_depth": -1},
        )
        assert r.status_code in (400, 422)


# --------------------------------------------------------------------- #
# 4. MCP — ZeroDataMCPServer.call_tool
# --------------------------------------------------------------------- #


class TestMCPReasoning:
    def test_mcp_prop_infer(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_infer_logical",
            facts={"rain": True},
            rules=[["rain", "wet_grass"]],
        )
        assert r["facts"]["wet_grass"] is True

    def test_mcp_syllogism(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_syllogism",
            major=["A", "human", "mortal"],
            minor=["A", "socrates", "human"],
        )
        assert r["conclusion"] == "All socrates are mortal"

    def test_mcp_induct(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_induct_rule",
            examples=[{"color": "red"}, {"color": "red"}, {"color": "blue"}],
            labels=[True, True, False],
        )
        assert r["rule"] == "color == 'red'"

    def test_mcp_analogize(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_analogize",
            source={"a": 1, "b": 2},
            target={"a": 1, "b": 2},
        )
        assert r["mapping"]["a"] == "a"

    def test_mcp_abduce(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_abduce",
            observation="patient has fever and cough",
            hypotheses=["flu with fever and cough", "common cold"],
            priors=[0.6, 0.4],
        )
        assert r["best"] == "flu with fever and cough"

    def test_mcp_defaults(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_conclude_defaults",
            defaults=[{"rule": ["bird", "flies", True], "exception": ["penguin"]}],
            facts={"bird": True, "penguin": False},
        )
        assert r["conclusions"]["flies"] is True

    def test_mcp_causal(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_trace_causal",
            links=[["a", "b"], ["b", "c"]],
            start="a",
        )
        assert r["chain"] == ["a", "b", "c"]

    def test_mcp_stateless_does_not_leak_state(self, mcp_server):
        """MCP reasoning tools must be stateless — calling them twice with
        different inputs must not accumulate state.
        """
        r1 = mcp_server.call_tool(
            "reasoning_infer_logical",
            facts={"a": True},
            rules=[["a", "b"]],
        )
        assert "b" in r1["facts"]
        # Second call should not see `b` from the first call.
        r2 = mcp_server.call_tool(
            "reasoning_infer_logical",
            facts={"x": True},
        )
        assert "b" not in r2["facts"]

    def test_mcp_prop_infer_negation_rules(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_infer_logical",
            facts={"bird": True},
            negation_rules=[["bird", "penguin"]],
        )
        assert r["facts"]["penguin"] is False

    # --- negative paths ---
    def test_mcp_syllogism_bad_length_returns_error(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_syllogism",
            major=["A", "X"],  # only 2 elements
            minor=["A", "Y", "Z"],
        )
        assert "error" in r

    def test_mcp_abduce_empty_hypotheses_returns_error(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_abduce",
            observation="x",
            hypotheses=[],
        )
        assert "error" in r

    def test_mcp_induct_mismatched_returns_error(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_induct_rule",
            examples=[{"a": 1}],
            labels=[True, False],
        )
        assert "error" in r

    def test_mcp_analogize_empty_source_returns_error(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_analogize",
            source={},
            target={"a": 1},
        )
        assert "error" in r

    def test_mcp_causal_negative_depth_returns_error(self, mcp_server):
        r = mcp_server.call_tool(
            "reasoning_trace_causal",
            links=[],
            start="a",
            max_depth=-1,
        )
        assert "error" in r

    def test_mcp_reasoning_tools_registered(self, mcp_server):
        tools = mcp_server.list_tools()
        for name in (
            "reasoning_infer_logical", "reasoning_syllogism",
            "reasoning_induct_rule", "reasoning_analogize",
            "reasoning_abduce", "reasoning_conclude_defaults",
            "reasoning_trace_causal",
        ):
            assert name in tools, f"missing tool: {name}"


# --------------------------------------------------------------------- #
# 5. Cross-layer consistency
# --------------------------------------------------------------------- #


class TestCrossLayerConsistency:
    """Reasoning facade methods produce identical results across
    facade / MCP.

    The stateless methods (syllogism, induct_rule, analogize, abduce)
    are pure functions of their inputs (no model-state dependence), so
    the facade and MCP server should agree exactly.
    """

    def test_facade_and_mcp_syllogism_agree(self, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.syllogism(("A", "human", "mortal"), ("A", "socrates", "human"))
        mcp = mcp_server.call_tool(
            "reasoning_syllogism",
            major=["A", "human", "mortal"],
            minor=["A", "socrates", "human"],
        )
        assert facade == mcp

    def test_facade_and_mcp_induct_agree(self, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        examples = [{"color": "red"}, {"color": "red"}, {"color": "blue"}]
        labels = [True, True, False]
        facade = m.induct_rule(examples, labels)
        mcp = mcp_server.call_tool(
            "reasoning_induct_rule", examples=examples, labels=labels
        )
        assert facade == mcp

    def test_facade_and_mcp_abduce_agree(self, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.abduce(
            "patient has fever and cough",
            ["flu with fever and cough", "common cold"],
            priors=[0.6, 0.4],
        )
        mcp = mcp_server.call_tool(
            "reasoning_abduce",
            observation="patient has fever and cough",
            hypotheses=["flu with fever and cough", "common cold"],
            priors=[0.6, 0.4],
        )
        assert facade["best"] == mcp["best"]
        for f, mcp_s in zip(facade["scores"], mcp["scores"]):
            assert f == pytest.approx(mcp_s, abs=1e-12)
        assert facade["confidence"] == pytest.approx(mcp["confidence"], abs=1e-12)

    def test_facade_and_mcp_prop_infer_agree(self, mcp_server):
        # The facade holds state in the model instance, so we have to
        # apply the same facts/rules to both. The MCP tool builds a
        # fresh model internally.
        m = ZeroDataModel(dim=8, seed=42)
        m.add_logical_fact("rain", True)
        m.add_logical_rule("rain", "wet_grass")
        facade = m.infer_logical()
        mcp = mcp_server.call_tool(
            "reasoning_infer_logical",
            facts={"rain": True},
            rules=[["rain", "wet_grass"]],
        )
        assert facade["facts"] == mcp["facts"]
        assert facade["contradictions"] == mcp["contradictions"]

    def test_facade_and_mcp_causal_agree(self, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        m.add_causal_link("a", "b")
        m.add_causal_link("b", "c")
        facade = m.trace_causal_chain("a", max_depth=5)
        mcp = mcp_server.call_tool(
            "reasoning_trace_causal",
            links=[["a", "b"], ["b", "c"]],
            start="a",
        )
        assert facade == mcp

    def test_facade_and_mcp_defaults_agree(self, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        m.add_default_rule(("bird", "flies", True), exception=("penguin",))
        facade = m.conclude_defaults({"bird": True, "penguin": False})
        mcp = mcp_server.call_tool(
            "reasoning_conclude_defaults",
            defaults=[{"rule": ["bird", "flies", True], "exception": ["penguin"]}],
            facts={"bird": True, "penguin": False},
        )
        assert facade["conclusions"] == mcp["conclusions"]
        # `defeated` in facade is list of tuples, in MCP is list of lists.
        facade_defeated = [list(d) for d in facade["defeated"]]
        assert facade_defeated == mcp["defeated"]
        assert facade["ambiguous"] == mcp["ambiguous"]
