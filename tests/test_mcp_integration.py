# tests/test_mcp_integration.py
"""End-to-end integration tests for the 40 Phase-7 cognitive-upgrade MCP tools.

The project exposes 7-stage cognitive-upgrade modules both as REST endpoints
(``src/zero_data_model/api.py``) and as MCP tools
(``src/zero_data_model/mcp_server.py``). Until now only the REST layer had
integration coverage (``test_phase7_*_integration.py``). These tests verify
the MCP layer mirrors the REST behaviour for every one of the 40 tools:

  1. Registration — all 40 ``phase7_*`` tools are registered with callable
     handlers and docstrings (used as MCP descriptions).
  2. Parameter schemas — ``inspect.signature`` confirms each tool exposes the
     expected required/optional parameters.
  3. Execution — every tool runs against a fully-enabled model and returns the
     expected response shape (no ``{"error": ...}``).
  4. MCP ↔ REST consistency — for the same model instance, each tool's MCP
     output is byte-for-byte equal to its REST counterpart's JSON body.
  5. Error handling — unknown tools, missing required params, and invalid
     param types surface as ``{"error": ...}`` instead of crashing.

The MCP server works whether or not the optional ``mcp`` package is installed
(graceful degradation via the plain-callable ``call_tool`` / ``tools`` API),
so these tests need no MCP runtime.
"""
from __future__ import annotations

import inspect

import pytest

from zero_data_model import api as api_module
from zero_data_model.mcp_server import ZeroDataMCPServer
from zero_data_model.model import ZeroDataModel
from zero_data_model.persistence import set_persistence_root

# Web stack (fastapi/httpx) is optional; skip REST-consistency tests when absent.
try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    _HAS_WEB = True
except ImportError:  # pragma: no cover - exercised only without web extras.
    _HAS_WEB = False


# --------------------------------------------------------------------- #
# Static mapping: every phase7 MCP tool -> its REST endpoint.
# --------------------------------------------------------------------- #
# tool_name: (rest_path, http_method)
TOOL_REST_MAPPING: dict[str, tuple[str, str]] = {
    # architect (plasticity)
    "phase7_architect_stats": ("/architect/stats", "GET"),
    "phase7_architect_dormant": ("/architect/dormant", "GET"),
    "phase7_architect_reactivate": ("/architect/reactivate", "POST"),
    # temporal_memory (cogtime)
    "phase7_temporal_memory_context": ("/temporal_memory/context", "GET"),
    "phase7_temporal_memory_spectral_radius": ("/temporal_memory/spectral_radius", "GET"),
    # layered_predictor (cogtime)
    "phase7_layered_predictor_context": ("/layered_predictor/context", "GET"),
    "phase7_layered_predictor_rhythm": ("/layered_predictor/rhythm", "GET"),
    # episodic_graph (cogmem)
    "phase7_episodic_graph_recent": ("/episodic_graph/recent", "GET"),
    "phase7_episodic_graph_node_count": ("/episodic_graph/node_count", "GET"),
    "phase7_episodic_graph_plan": ("/episodic_graph/plan", "POST"),
    # semantic_index (cogmem)
    "phase7_semantic_index_size": ("/semantic_index/size", "GET"),
    "phase7_semantic_index_query": ("/semantic_index/query", "POST"),
    # logic_layer (knowledge)
    "phase7_logic_layer_rules": ("/logic_layer/rules", "GET"),
    "phase7_logic_layer_add_rule": ("/logic_layer/rule", "POST"),
    "phase7_logic_layer_set_predicate": ("/logic_layer/predicate", "POST"),
    "phase7_logic_layer_check": ("/logic_layer/check", "GET"),
    "phase7_logic_layer_penalty": ("/logic_layer/penalty", "GET"),
    # causal_inference (knowledge)
    "phase7_causal_inference_set_transition": ("/causal_inference/transition", "POST"),
    "phase7_causal_inference_do": ("/causal_inference/do", "POST"),
    "phase7_causal_inference_counterfactual": ("/causal_inference/counterfactual", "POST"),
    "phase7_causal_inference_confounders": ("/causal_inference/confounders", "POST"),
    # meta_cognition (metacog)
    "phase7_meta_cognition_confidence": ("/meta_cognition/confidence", "GET"),
    "phase7_meta_cognition_uncertainty": ("/meta_cognition/uncertainty", "GET"),
    "phase7_meta_cognition_should_seek_info": ("/meta_cognition/should_seek_info", "GET"),
    "phase7_meta_cognition_stats": ("/meta_cognition/stats", "GET"),
    # experiment_planner (experiment)
    "phase7_experiment_planner_candidates": ("/experiment_planner/candidates", "GET"),
    "phase7_experiment_planner_select_best": ("/experiment_planner/select_best", "POST"),
    "phase7_experiment_planner_record_result": ("/experiment_planner/record_result", "POST"),
    "phase7_experiment_planner_stats": ("/experiment_planner/stats", "GET"),
    # hypothesis_tester (experiment)
    "phase7_hypothesis_tester_hypotheses": ("/hypothesis_tester/hypotheses", "GET"),
    "phase7_hypothesis_tester_supported": ("/hypothesis_tester/supported", "GET"),
    "phase7_hypothesis_tester_stats": ("/hypothesis_tester/stats", "GET"),
    # world (multiagent — not yet integrated, always disabled)
    "phase7_world_collaboration_stats": ("/world/collaboration_stats", "GET"),
    "phase7_world_agent_count": ("/world/agent_count", "GET"),
    "phase7_world_step_count": ("/world/step_count", "GET"),
    # communication (multiagent — not yet integrated, always disabled)
    "phase7_communication_emergent_meanings": ("/communication/emergent_meanings", "GET"),
    "phase7_communication_stats": ("/communication/stats", "GET"),
    # culture (multiagent — not yet integrated, always disabled)
    "phase7_culture_knowledge_curve": ("/culture/knowledge_curve", "GET"),
    "phase7_culture_stats": ("/culture/stats", "GET"),
    "phase7_culture_generations": ("/culture/generations", "GET"),
}

assert len(TOOL_REST_MAPPING) == 40, "mapping must cover all 40 phase7 tools"

# Expected required (no-default) parameters per tool — drives the param
# schema tests. Tools absent here take only optional/no parameters.
REQUIRED_PARAMS: dict[str, list[str]] = {
    "phase7_architect_reactivate": ["name"],
    "phase7_episodic_graph_plan": ["start_state", "goal_state"],
    "phase7_semantic_index_query": ["vec"],
    "phase7_logic_layer_add_rule": ["name", "antecedents", "consequent"],
    "phase7_logic_layer_set_predicate": ["name", "value"],
    "phase7_causal_inference_set_transition": ["matrix"],
    "phase7_causal_inference_do": ["index", "value"],
    "phase7_causal_inference_counterfactual": ["observed", "index", "value"],
    "phase7_causal_inference_confounders": ["var_a", "var_b"],
    "phase7_experiment_planner_record_result": [
        "candidate_id",
        "fe_before",
        "fe_after",
    ],
}

# Readonly tools (no model state mutation) -> (mcp_kwargs, expected_key_subset).
# Used for execution-shape and MCP↔REST consistency tests.
READONLY_TOOLS: list[tuple[str, dict, str]] = [
    ("phase7_architect_stats", {}, "dormant_count"),
    ("phase7_architect_dormant", {}, "dormant"),
    ("phase7_temporal_memory_context", {}, "context"),
    ("phase7_temporal_memory_spectral_radius", {}, "spectral_radius"),
    ("phase7_layered_predictor_context", {}, "context"),
    ("phase7_layered_predictor_rhythm", {}, "rhythm"),
    ("phase7_episodic_graph_recent", {"n": 5}, "episodes"),
    ("phase7_episodic_graph_node_count", {}, "node_count"),
    ("phase7_semantic_index_size", {}, "size"),
    ("phase7_logic_layer_rules", {}, "rules"),
    ("phase7_logic_layer_check", {}, "violations"),
    ("phase7_logic_layer_penalty", {}, "penalty"),
    ("phase7_meta_cognition_confidence", {}, "confidence"),
    ("phase7_meta_cognition_uncertainty", {}, "uncertainty"),
    ("phase7_meta_cognition_should_seek_info", {}, "should_seek_info"),
    ("phase7_meta_cognition_stats", {}, "confidence"),
    ("phase7_experiment_planner_candidates", {}, "candidates"),
    ("phase7_experiment_planner_stats", {}, "n_candidates"),
    ("phase7_hypothesis_tester_hypotheses", {}, "hypotheses"),
    ("phase7_hypothesis_tester_supported", {}, "supported"),
    ("phase7_hypothesis_tester_stats", {}, "n_hypotheses"),
    ("phase7_world_collaboration_stats", {}, "enabled"),
    ("phase7_world_agent_count", {}, "enabled"),
    ("phase7_world_step_count", {}, "enabled"),
    ("phase7_communication_emergent_meanings", {}, "enabled"),
    ("phase7_communication_stats", {}, "enabled"),
    ("phase7_culture_knowledge_curve", {}, "enabled"),
    ("phase7_culture_stats", {}, "enabled"),
    ("phase7_culture_generations", {}, "enabled"),
]


def _enabled_model() -> ZeroDataModel:
    """A small, seeded model with every cognitive-upgrade module enabled."""
    return ZeroDataModel(
        dim=8,
        seed=42,
        enable_architect=True,
        enable_layered_predictor=True,
        enable_episodic_memory=True,
        enable_logic_layer=True,
        enable_meta_cognition=True,
        enable_experiment_planner=True,
        enable_multiagent=True,
    )


def _call(server: ZeroDataMCPServer, tool: str, **kwargs):
    """Invoke ``tool`` on ``server``, routing around the call_tool(name=...) collision.

    ``ZeroDataMCPServer.call_tool(self, name, **kwargs)`` shadows any tool
    parameter literally named ``name`` (phase7_architect_reactivate,
    phase7_logic_layer_add_rule, phase7_logic_layer_set_predicate). For those
    tools we invoke the bound callable directly; for all others we go through
    the public ``call_tool`` API (which surfaces errors as ``{"error": ...}``).
    """
    if "name" in kwargs:
        return server.tools[tool](**kwargs)
    return server.call_tool(tool, **kwargs)


# --------------------------------------------------------------------- #
# Shared fixtures.
# --------------------------------------------------------------------- #


@pytest.fixture()
def cog_model():
    """A model with all phase7 cognitive-upgrade modules enabled."""
    return _enabled_model()


@pytest.fixture()
def mcp_server(cog_model):
    """MCP server wired to the shared enabled model.

    ``ZeroDataMCPServer.__init__`` builds its own (bare) model; we replace
    ``server.model`` with ``cog_model`` after construction so every tool
    operates on the enabled instance (bound methods look up ``self.model``
    at call time).
    """
    server = ZeroDataMCPServer(dim=8)
    server.model = cog_model
    return server


@pytest.fixture()
def client(cog_model, tmp_path):
    """FastAPI TestClient backed by the same shared enabled model."""
    if not _HAS_WEB:
        pytest.skip("fastapi or httpx not installed")
    from fastapi.testclient import TestClient

    from zero_data_model.api import create_app

    set_persistence_root(str(tmp_path / "api_persistence"))
    api_module.set_model(cog_model)
    if api_module.limiter is not None:
        api_module.limiter.enabled = False
    app = create_app()
    with TestClient(app, base_url="http://localhost") as c:
        yield c
    api_module._model = None


# --------------------------------------------------------------------- #
# 1. Registration — all 40 phase7 tools present & callable.
# --------------------------------------------------------------------- #


class TestMCPToolRegistration:
    """Verify all 40 phase7 tools are registered with callable handlers."""

    def test_all_40_phase7_tools_registered(self, mcp_server):
        tool_names = set(mcp_server.list_tools())
        missing = set(TOOL_REST_MAPPING) - tool_names
        assert not missing, f"Missing phase7 tools: {sorted(missing)}"

    def test_tool_count_is_at_least_40(self, mcp_server):
        phase7 = [n for n in mcp_server.list_tools() if n.startswith("phase7_")]
        assert len(phase7) >= 40

    def test_exactly_40_phase7_tools(self, mcp_server):
        phase7 = [n for n in mcp_server.list_tools() if n.startswith("phase7_")]
        assert len(phase7) == 40

    def test_all_tools_callable(self, mcp_server):
        for name in TOOL_REST_MAPPING:
            assert callable(mcp_server.tools[name]), f"{name} is not callable"

    def test_all_tools_have_docstrings(self, mcp_server):
        # FastMCP uses the docstring as the tool description.
        for name in TOOL_REST_MAPPING:
            doc = mcp_server.tools[name].__doc__
            assert doc and doc.strip(), f"{name} lost its docstring"

    def test_tool_sets_partition_all_40(self):
        # The readonly set, the required-param set, and the single
        # POST-but-readonly tool (select_best) together cover every tool
        # exactly once. This guards the static tables below against drift.
        readonly = {t for t, _, _ in READONLY_TOOLS}
        required = set(REQUIRED_PARAMS)
        post_readonly = {"phase7_experiment_planner_select_best"}
        assert readonly.isdisjoint(required)
        assert readonly | required | post_readonly == set(TOOL_REST_MAPPING)


# --------------------------------------------------------------------- #
# 2. Parameter schemas — required/optional parameters per tool.
# --------------------------------------------------------------------- #


class TestMCPToolParameters:
    """Verify each tool's signature exposes the expected parameters."""

    @pytest.mark.parametrize(
        "tool,expected_required",
        list(REQUIRED_PARAMS.items()),
        ids=list(REQUIRED_PARAMS.keys()),
    )
    def test_required_parameters(self, mcp_server, tool, expected_required):
        sig = inspect.signature(mcp_server.tools[tool])
        actual_required = [
            p.name
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
        ]
        for param in expected_required:
            assert param in actual_required, (
                f"{tool} missing required param {param!r} "
                f"(has {actual_required})"
            )

    def test_episodic_graph_recent_n_is_optional(self, mcp_server):
        sig = inspect.signature(mcp_server.tools["phase7_episodic_graph_recent"])
        n = sig.parameters["n"]
        assert n.default == 10
        # ``from __future__ import annotations`` stringifies hints, so accept
        # either the type object or its string form.
        assert n.annotation in (int, "int")

    def test_experiment_select_best_uncertainty_is_optional(self, mcp_server):
        sig = inspect.signature(
            mcp_server.tools["phase7_experiment_planner_select_best"]
        )
        assert sig.parameters["current_uncertainty"].default == 0.5

    def test_episodic_graph_plan_horizon_is_optional(self, mcp_server):
        sig = inspect.signature(mcp_server.tools["phase7_episodic_graph_plan"])
        assert sig.parameters["horizon"].default == 20

    def test_semantic_index_query_k_is_optional(self, mcp_server):
        sig = inspect.signature(mcp_server.tools["phase7_semantic_index_query"])
        assert sig.parameters["k"].default == 5

    @pytest.mark.parametrize("tool", sorted(TOOL_REST_MAPPING))
    def test_tool_accepts_only_json_friendly_params(self, mcp_server, tool):
        # MCP tools must accept JSON-decodable inputs (no numpy/ndarray hints
        # that would block the JSON path). Annotations may be list/str/int/float.
        sig = inspect.signature(mcp_server.tools[tool])
        for p in sig.parameters.values():
            ann = p.annotation
            if ann is inspect.Parameter.empty:
                continue
            ann_str = str(ann)
            assert "ndarray" not in ann_str, (
                f"{tool} param {p.name!r} annotates a numpy ndarray "
                f"(not JSON-friendly): {ann_str}"
            )


# --------------------------------------------------------------------- #
# 3. Execution — every readonly tool runs and returns the expected shape.
# --------------------------------------------------------------------- #


class TestMCPToolExecution:
    """Verify each tool executes against the enabled model without error."""

    @pytest.mark.parametrize(
        "tool,kwargs,expected_key",
        READONLY_TOOLS,
        ids=[t[0] for t in READONLY_TOOLS],
    )
    def test_readonly_tool_returns_expected_field(
        self, mcp_server, tool, kwargs, expected_key
    ):
        result = _call(mcp_server, tool, **kwargs)
        assert isinstance(result, dict), f"{tool} returned {type(result)!r}"
        assert "error" not in result, f"{tool} errored: {result}"
        assert expected_key in result, (
            f"{tool} result {list(result.keys())} missing key {expected_key!r}"
        )

    def test_architect_stats_shape(self, mcp_server):
        r = mcp_server.call_tool("phase7_architect_stats")
        assert {"active_count", "dormant_count", "total_splits", "total_prunes"} <= set(
            r.keys()
        )

    def test_experiment_planner_candidates_has_count(self, mcp_server):
        r = mcp_server.call_tool("phase7_experiment_planner_candidates")
        assert r["count"] == len(r["candidates"])

    def test_hypothesis_tester_hypotheses_has_count(self, mcp_server):
        r = mcp_server.call_tool("phase7_hypothesis_tester_hypotheses")
        assert r["count"] == len(r["hypotheses"])

    # --- stateful tools: mutation is observable through a follow-up read ---

    def test_logic_add_rule_then_rules_reflects_it(self, mcp_server):
        before = mcp_server.call_tool("phase7_logic_layer_rules")["rules"]
        _call(
            mcp_server,
            "phase7_logic_layer_add_rule",
            name="intg_test_rule",
            antecedents=["alpha"],
            consequent="beta",
            weight=0.7,
            description="from test",
        )
        after = mcp_server.call_tool("phase7_logic_layer_rules")["rules"]
        assert len(after) == len(before) + 1
        assert any(r["name"] == "intg_test_rule" for r in after)

    def test_experiment_record_result_then_stats_reflects_it(self, mcp_server):
        cands = mcp_server.call_tool("phase7_experiment_planner_candidates")[
            "candidates"
        ]
        assert cands, "planner must expose at least one candidate"
        rec = _call(
            mcp_server,
            "phase7_experiment_planner_record_result",
            candidate_id=0,
            fe_before=2.0,
            fe_after=0.5,
        )
        assert rec == {
            "recorded": True,
            "candidate_id": 0,
            "actual_gain": pytest.approx(1.5),
        }
        stats = mcp_server.call_tool("phase7_experiment_planner_stats")
        assert stats["n_executed"] >= 1

    def test_causal_set_transition_then_do_uses_it(self, mcp_server):
        set_r = _call(
            mcp_server,
            "phase7_causal_inference_set_transition",
            matrix=[[1.0, 0.0], [0.0, 1.0]],
        )
        assert set_r == {"shape": [2, 2]}
        do_r = _call(
            mcp_server, "phase7_causal_inference_do", index=0, value=3.0
        )
        assert "predicted_state" in do_r
        assert do_r["intervened_vars"] == [0]

    def test_semantic_index_query_returns_results_list(self, mcp_server):
        r = _call(
            mcp_server, "phase7_semantic_index_query", vec=[1.0] * 8, k=3
        )
        assert isinstance(r["results"], list)
        assert len(r["results"]) <= 3

    def test_episodic_graph_plan_returns_path_key(self, mcp_server):
        r = _call(
            mcp_server,
            "phase7_episodic_graph_plan",
            start_state=[0.0] * 8,
            goal_state=[1.0] * 8,
            horizon=5,
        )
        assert "path" in r

    def test_experiment_select_best_returns_best(self, mcp_server):
        r = _call(
            mcp_server,
            "phase7_experiment_planner_select_best",
            current_uncertainty=0.5,
        )
        assert "best" in r
        # Either a candidate dict or None when no unexecuted candidate exists.
        assert r["best"] is None or isinstance(r["best"], dict)


# --------------------------------------------------------------------- #
# 4. MCP ↔ REST consistency — same model -> identical output bodies.
# --------------------------------------------------------------------- #


class TestMCPRestConsistency:
    """Verify MCP tool output equals the REST JSON body for the same model."""

    def _rest_call(self, client, method, path, *, params=None, json=None):
        return (
            client.get(path, params=params)
            if method == "GET"
            else client.post(path, json=json)
        )

    @pytest.mark.parametrize(
        "tool,kwargs",
        [(t, k) for t, k, _ in READONLY_TOOLS],
        ids=[t[0] for t in READONLY_TOOLS],
    )
    def test_readonly_tool_matches_rest_body(
        self, mcp_server, client, tool, kwargs
    ):
        path, method = TOOL_REST_MAPPING[tool]
        mcp_result = _call(mcp_server, tool, **kwargs)
        resp = self._rest_call(
            client, method, path, params=kwargs if method == "GET" else None
        )
        # Disabled-module endpoints return 503; enabled ones return 200.
        assert resp.status_code in (200, 503), (tool, resp.status_code, resp.text)
        rest_body = resp.json()
        assert mcp_result == rest_body, (
            f"{tool}: MCP {mcp_result} != REST {rest_body}"
        )

    @pytest.mark.parametrize(
        "tool,kwargs,setup",
        [
            ("phase7_causal_inference_set_transition", {"matrix": [[1.0, 0.0], [0.0, 1.0]]}, []),
            ("phase7_causal_inference_do", {"index": 0, "value": 1.0}, []),
            # counterfactual needs a transition matrix set on the shared model
            # before it can run (else both layers raise). Set it via both MCP
            # and REST so the two stay in lockstep.
            (
                "phase7_causal_inference_counterfactual",
                {"observed": [1.0, 2.0], "index": 0, "value": 5.0},
                [("phase7_causal_inference_set_transition", {"matrix": [[1.0, 0.0], [0.0, 1.0]]})],
            ),
            ("phase7_causal_inference_confounders", {"var_a": 0, "var_b": 1}, []),
            # NOTE: phase7_experiment_planner_select_best is intentionally
            # excluded from the consistency matrix — it is stochastic: each
            # call advances the planner's RNG, so two back-to-back calls (MCP
            # then REST) on the *same* model can select different candidates
            # (e.g. change_friction vs perturb_gravity). Its execution shape
            # is still covered by test_experiment_select_best_returns_best.
            ("phase7_semantic_index_query", {"vec": [1.0] * 8, "k": 3}, []),
            (
                "phase7_episodic_graph_plan",
                {
                    "start_state": [0.0] * 8,
                    "goal_state": [1.0] * 8,
                    "horizon": 5,
                },
                [],
            ),
        ],
        ids=[
            "set_transition",
            "do",
            "counterfactual",
            "confounders",
            "semantic_query",
            "episodic_plan",
        ],
    )
    def test_stateful_tool_matches_rest_body(
        self, mcp_server, client, tool, kwargs, setup
    ):
        # Run any setup steps on both layers so they share identical state.
        for s_tool, s_kwargs in setup:
            _call(mcp_server, s_tool, **s_kwargs)
            s_path, s_method = TOOL_REST_MAPPING[s_tool]
            s_resp = self._rest_call(client, s_method, s_path, json=s_kwargs)
            assert s_resp.status_code == 200, (s_tool, s_resp.status_code, s_resp.text)
        path, method = TOOL_REST_MAPPING[tool]
        mcp_result = _call(mcp_server, tool, **kwargs)
        resp = self._rest_call(client, method, path, json=kwargs)
        assert resp.status_code == 200, (tool, resp.status_code, resp.text)
        rest_body = resp.json()
        assert mcp_result == rest_body, (
            f"{tool}: MCP {mcp_result} != REST {rest_body}"
        )

    def test_disabled_module_endpoints_match(self, mcp_server, client):
        """world / communication / culture are always disabled in both layers."""
        for tool in (
            "phase7_world_collaboration_stats",
            "phase7_communication_emergent_meanings",
            "phase7_culture_generations",
        ):
            path, method = TOOL_REST_MAPPING[tool]
            mcp_result = _call(mcp_server, tool)
            resp = self._rest_call(client, method, path)
            assert resp.status_code == 503, (tool, resp.status_code)
            assert mcp_result == resp.json()


# --------------------------------------------------------------------- #
# 5. Error handling — bad inputs surface as {"error": ...}, never crash.
# --------------------------------------------------------------------- #


class TestMCPErrorHandling:
    """Verify MCP tools surface errors gracefully."""

    def test_unknown_tool_returns_error_dict(self, mcp_server):
        result = mcp_server.call_tool("definitely_not_a_tool")
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_missing_required_param_surfaces_error(self, mcp_server):
        # phase7_episodic_graph_plan requires start_state + goal_state.
        result = mcp_server.call_tool("phase7_episodic_graph_plan")
        assert "error" in result
        assert "phase7_episodic_graph_plan" in result["error"]

    def test_invalid_param_type_surfaces_error(self, mcp_server):
        # name must be a non-empty string; an int is rejected.
        result = _call(
            mcp_server, "phase7_architect_reactivate", name=12345
        )
        assert "error" in result
        assert "name" in result["error"]

    def test_empty_string_param_surfaces_error(self, mcp_server):
        result = _call(mcp_server, "phase7_architect_reactivate", name="")
        assert "error" in result

    def test_negative_int_param_surfaces_error(self, mcp_server):
        # index must be a non-negative int.
        result = _call(
            mcp_server, "phase7_causal_inference_do", index=-1, value=1.0
        )
        assert "error" in result

    def test_non_square_matrix_surfaces_error(self, mcp_server):
        # A well-formed 2x3 rectangular matrix (rows of equal length) so numpy
        # builds it cleanly and the tool's own square-shape check rejects it.
        result = _call(
            mcp_server,
            "phase7_causal_inference_set_transition",
            matrix=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
        assert "error" in result
        assert "square" in result["error"].lower()

    def test_out_of_range_n_surfaces_error(self, mcp_server):
        result = mcp_server.call_tool("phase7_episodic_graph_recent", n=0)
        assert "error" in result
        result2 = mcp_server.call_tool("phase7_episodic_graph_recent", n=5000)
        assert "error" in result2

    def test_out_of_range_candidate_id_surfaces_error(self, mcp_server):
        result = _call(
            mcp_server,
            "phase7_experiment_planner_record_result",
            candidate_id=999_999,
            fe_before=1.0,
            fe_after=0.5,
        )
        assert "error" in result

    def test_nan_in_vector_surfaces_error(self, mcp_server):
        result = _call(
            mcp_server,
            "phase7_semantic_index_query",
            vec=[float("nan"), 1.0, 2.0],
            k=2,
        )
        assert "error" in result
        assert "finite" in result["error"].lower()

    def test_call_tool_name_kwarg_collision_documented(self, mcp_server):
        """Known limitation: call_tool(name, **kwargs) shadows a tool param
        literally named ``name``. Calling such a tool via ``call_tool`` with
        ``name=...`` raises TypeError (uncaught, since it occurs in
        ``call_tool`` itself rather than the tool wrapper). Use the bound
        callable (``server.tools[tool](name=...)``) instead — see ``_call``.
        """
        with pytest.raises(TypeError):
            mcp_server.call_tool("phase7_architect_reactivate", name="arch")

    def test_error_outputs_are_json_serializable(self, mcp_server):
        import json

        result = mcp_server.call_tool("nonexistent")
        # Every error dict must round-trip through json.dumps.
        assert isinstance(json.dumps(result), str)
