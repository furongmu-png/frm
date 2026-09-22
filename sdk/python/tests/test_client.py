"""Tests for the ZeroDataModel Python SDK client.

The suite uses :class:`httpx.MockTransport` so the client is exercised end to
end (real httpx serialization, header building, error mapping) without a live
server. A recording handler captures every request for assertions.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from zero_data_model_client import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    ZeroDataModelClient,
    ZeroDataModelError,
)


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #


class RecordingTransport(httpx.MockTransport):
    """A mock transport that records requests and dispatches to a handler.

    The handler receives the :class:`httpx.Request` and returns an
    :class:`httpx.Response`.
    """

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]):
        super().__init__(handler)
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return super().handle_request(request)


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = "test-key",
    base_url: str = "http://localhost:8000",
) -> tuple[ZeroDataModelClient, RecordingTransport]:
    """Build a client backed by a recording mock transport."""
    transport = RecordingTransport(handler)
    client = ZeroDataModelClient(
        base_url=base_url, api_key=api_key, _transport=transport
    )
    return client, transport


def json_response(
    payload: Any, status_code: int = 200, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        request=httpx.Request("GET", "http://localhost:8000/"),
    )


# --------------------------------------------------------------------------- #
# Core endpoints
# --------------------------------------------------------------------------- #


def test_health_returns_payload_and_uses_get():
    client, transport = make_client(lambda r: json_response({"status": "ok"}))
    result = client.health()
    assert result == {"status": "ok"}
    assert transport.requests[0].method == "GET"
    assert transport.requests[0].url.path == "/health"
    client.close()


def test_ready_returns_payload():
    client, _ = make_client(lambda r: json_response({"status": "ok", "model_ready": True}))
    assert client.ready() == {"status": "ok", "model_ready": True}
    client.close()


def test_think_sends_cycles_in_body():
    client, transport = make_client(
        lambda r: json_response({"cycle": 1, "free_energy": 0.42})
    )
    result = client.think(cycles=3, input=[1.0, 2.0])
    assert result["free_energy"] == 0.42
    req = transport.requests[0]
    assert req.method == "POST"
    assert req.url.path == "/think"
    body = json.loads(req.content)
    assert body == {"cycles": 3, "input": [1.0, 2.0]}
    client.close()


def test_get_knowledge_graph():
    client, transport = make_client(
        lambda r: json_response({"nodes": [{"id": 1}], "edges": []})
    )
    result = client.get_knowledge_graph()
    assert result["nodes"] == [{"id": 1}]
    assert transport.requests[0].url.path == "/knowledge-graph"
    client.close()


def test_get_story_milestones():
    client, transport = make_client(lambda r: json_response({"milestones": []}))
    client.get_story_milestones()
    assert transport.requests[0].url.path == "/story-milestones"
    client.close()


# --------------------------------------------------------------------------- #
# Phase-7 endpoints (verify method + path + body wiring)
# --------------------------------------------------------------------------- #


def test_architect_endpoints():
    client, transport = make_client(lambda r: json_response({"ok": True}))
    assert client.architect_stats() == {"ok": True}
    client.architect_dormant()
    client.architect_action("activate", "memory")
    paths = [req.url.path for req in transport.requests]
    assert paths == [
        "/architect/stats",
        "/architect/dormant",
        "/architect/action",
    ]
    assert transport.requests[2].method == "POST"
    assert json.loads(transport.requests[2].content) == {
        "action_type": "activate",
        "module_name": "memory",
    }
    client.close()


def test_temporal_memory_endpoints():
    client, transport = make_client(lambda r: json_response({"ok": True}))
    client.temporal_memory_state()
    client.temporal_memory_reset()
    assert [r.url.path for r in transport.requests] == [
        "/temporal_memory/state",
        "/temporal_memory/reset",
    ]
    assert transport.requests[1].method == "POST"
    client.close()


def test_layered_predictor_update_sends_observation():
    client, transport = make_client(lambda r: json_response({"ok": True}))
    client.layered_predictor_beliefs()
    client.layered_predictor_update([0.1, 0.2, 0.3])
    assert transport.requests[1].url.path == "/layered_predictor/update"
    assert json.loads(transport.requests[1].content) == {
        "observation": [0.1, 0.2, 0.3]
    }
    client.close()


def test_episodic_graph_plan_and_insert():
    client, transport = make_client(lambda r: json_response({"ok": True}))
    client.episodic_graph_stats()
    client.episodic_graph_plan(start_id=1, goal_id=9)
    client.episodic_graph_insert(node="x", weight=0.5)
    assert json.loads(transport.requests[1].content) == {
        "start_id": 1,
        "goal_id": 9,
    }
    assert json.loads(transport.requests[2].content) == {
        "node": "x",
        "weight": 0.5,
    }
    client.close()


def test_semantic_index_search():
    client, transport = make_client(lambda r: json_response({"ok": True}))
    client.semantic_index_stats()
    client.semantic_index_search([1.0, 2.0], k=3)
    assert json.loads(transport.requests[1].content) == {
        "vector": [1.0, 2.0],
        "k": 3,
    }
    client.close()


def test_logic_layer_endpoints():
    client, transport = make_client(lambda r: json_response({"ok": True}))
    client.logic_rules()
    client.logic_add_rule(name="r1", expr="x > 0")
    client.logic_remove_rule("r1")
    client.logic_evaluate()
    client.logic_violations()
    assert [r.method + " " + r.url.path for r in transport.requests] == [
        "GET /logic_layer/rules",
        "POST /logic_layer/add_rule",
        "DELETE /logic_layer/rules/r1",
        "POST /logic_layer/evaluate",
        "GET /logic_layer/violations",
    ]
    assert json.loads(transport.requests[1].content) == {"name": "r1", "expr": "x > 0"}
    client.close()


def test_causal_endpoints():
    client, transport = make_client(lambda r: json_response({"ok": True}))
    client.causal_graph()
    client.causal_intervene(var="x", value=1)
    client.causal_observe(data=[1, 2])
    client.causal_counterfactual(world="w")
    assert [r.url.path for r in transport.requests] == [
        "/causal_inference/graph",
        "/causal_inference/intervene",
        "/causal_inference/observe",
        "/causal_inference/counterfactual",
    ]
    client.close()


def test_metacognition_endpoints():
    client, transport = make_client(lambda r: json_response({"ok": True}))
    client.metacog_state()
    client.metacog_confidence()
    client.metacog_uncertainty()
    client.metacog_report()
    assert [r.url.path for r in transport.requests] == [
        "/meta_cognition/state",
        "/meta_cognition/confidence",
        "/meta_cognition/uncertainty",
        "/meta_cognition/report",
    ]
    client.close()


def test_experiment_planner_endpoints():
    client, transport = make_client(lambda r: json_response({"ok": True}))
    client.experiment_candidates()
    client.experiment_select()
    client.experiment_result("c1", 0.87)
    client.experiment_history()
    assert json.loads(transport.requests[2].content) == {
        "candidate_id": "c1",
        "outcome": 0.87,
    }
    client.close()


def test_hypothesis_endpoints():
    client, transport = make_client(lambda r: json_response({"ok": True}))
    client.hypothesis_list()
    client.hypothesis_test("h1", threshold=0.05)
    client.hypothesis_supported()
    assert json.loads(transport.requests[1].content) == {
        "hypothesis_id": "h1",
        "threshold": 0.05,
    }
    client.close()


def test_multi_agent_endpoints():
    client, transport = make_client(lambda r: json_response({"ok": True}))
    client.world_stats()
    client.world_step()
    client.world_agents()
    client.communication_channels()
    client.communication_send(channel="ch1", message="hi")
    client.culture_generations()
    client.culture_history()
    client.culture_propagate(meme="m1")
    assert [r.url.path for r in transport.requests] == [
        "/world/stats",
        "/world/step",
        "/world/agents",
        "/communication/channels",
        "/communication/send",
        "/culture/generations",
        "/culture/history",
        "/culture/propagate",
    ]
    client.close()


# --------------------------------------------------------------------------- #
# API key handling
# --------------------------------------------------------------------------- #


def test_api_key_header_sent_when_provided():
    client, transport = make_client(lambda r: json_response({"ok": True}), api_key="secret")
    client.health()
    assert transport.requests[0].headers["X-API-Key"] == "secret"
    client.close()


def test_api_key_header_omitted_when_absent():
    client, transport = make_client(lambda r: json_response({"ok": True}), api_key=None)
    client.health()
    assert "X-API-Key" not in transport.requests[0].headers
    client.close()


def test_base_url_trailing_slash_stripped():
    # Use a handler that echoes back the host so we can confirm routing.
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return json_response({"ok": True})

    client, _ = make_client(handler, base_url="http://localhost:8000/")
    client.health()
    assert seen[0] == "http://localhost:8000/health"
    client.close()


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "status, exc_type",
    [
        (401, AuthenticationError),
        (403, AuthenticationError),
        (404, NotFoundError),
        (422, ValidationError),
        (500, ServerError),
        (503, ServerError),
    ],
)
def test_error_status_maps_to_exception(status: int, exc_type: type[Exception]):
    client, _ = make_client(
        lambda r: json_response({"detail": "boom"}, status_code=status)
    )
    with pytest.raises(exc_type) as excinfo:
        client.health()
    assert "boom" in str(excinfo.value)
    client.close()


def test_rate_limit_error_carries_retry_after():
    client, _ = make_client(
        lambda r: json_response(
            {"detail": "slow down"}, status_code=429, headers={"Retry-After": "30"}
        )
    )
    with pytest.raises(RateLimitError) as excinfo:
        client.health()
    assert excinfo.value.retry_after == 30
    client.close()


def test_rate_limit_default_retry_after_when_header_missing():
    client, _ = make_client(
        lambda r: json_response({"detail": "slow down"}, status_code=429)
    )
    with pytest.raises(RateLimitError) as excinfo:
        client.health()
    assert excinfo.value.retry_after == 60
    client.close()


def test_unknown_4xx_raises_base_error():
    client, _ = make_client(
        lambda r: json_response({"detail": "teapot"}, status_code=418)
    )
    with pytest.raises(ZeroDataModelError) as excinfo:
        client.health()
    assert "418" in str(excinfo.value)
    client.close()


def test_request_error_wrapped():
    def boom_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, _ = make_client(boom_handler)
    with pytest.raises(ZeroDataModelError) as excinfo:
        client.health()
    assert "Request failed" in str(excinfo.value)
    client.close()


def test_all_errors_are_subclasses_of_base():
    for exc in (
        AuthenticationError,
        NotFoundError,
        RateLimitError,
        ValidationError,
        ServerError,
    ):
        assert issubclass(exc, ZeroDataModelError)


# --------------------------------------------------------------------------- #
# Context manager / lifecycle
# --------------------------------------------------------------------------- #


def test_context_manager_closes_client():
    with ZeroDataModelClient(api_key="k") as client:
        assert isinstance(client, ZeroDataModelClient)
    # Calling close() again should be idempotent and not raise.
    client.close()


def test_close_is_idempotent():
    client, _ = make_client(lambda r: json_response({"ok": True}))
    client.close()
    client.close()  # must not raise


# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #


def test_package_exports():
    import zero_data_model_client as zdm

    assert hasattr(zdm, "ZeroDataModelClient")
    for name in (
        "ZeroDataModelError",
        "AuthenticationError",
        "RateLimitError",
        "NotFoundError",
        "ValidationError",
        "ServerError",
    ):
        assert name in zdm.__all__
    assert zdm.__version__ == "1.0.0"
