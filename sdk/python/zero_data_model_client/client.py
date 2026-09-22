"""ZeroDataModel Python SDK client."""
from __future__ import annotations

from typing import Any

import httpx

from ._config import ClientConfig
from ._http import create_client
from .exceptions import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    ZeroDataModelError,
)


class ZeroDataModelClient:
    """Synchronous client for the ZeroDataModel REST API.

    Usage:
        from zero_data_model_client import ZeroDataModelClient
        client = ZeroDataModelClient(base_url="http://localhost:8000/v1", api_key="...")
        result = client.think(cycles=1)
        print(result["free_energy"])

    The client is safe to use as a context manager, which guarantees the
    underlying HTTP connection pool is closed::

        with ZeroDataModelClient(api_key="...") as client:
            client.health()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ):
        # ``_transport`` is pulled out of kwargs so it is never forwarded to
        # ClientConfig; it is used by the test-suite to inject a mock transport.
        transport = kwargs.pop("_transport", None)
        self._config = ClientConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
        )
        self._http = create_client(
            self._config,
            self._build_headers(),
            transport=transport,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._config.api_key:
            h["X-API-Key"] = self._config.api_key
        return h

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Execute an HTTP request and translate errors into SDK exceptions."""
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.RequestError as e:
            raise ZeroDataModelError(f"Request failed: {e}") from e

        if resp.status_code >= 400:
            self._handle_error(resp)

        return resp.json()

    def _handle_error(self, resp: httpx.Response) -> None:
        detail = ""
        try:
            body = resp.json()
            detail = body.get("detail", str(body))
        except Exception:
            detail = resp.text

        if resp.status_code in (401, 403):
            raise AuthenticationError(detail)
        elif resp.status_code == 404:
            raise NotFoundError(detail)
        elif resp.status_code == 422:
            raise ValidationError(detail)
        elif resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "60")
            raise RateLimitError(detail, retry_after=int(retry_after))
        elif resp.status_code >= 500:
            raise ServerError(detail)
        else:
            raise ZeroDataModelError(f"HTTP {resp.status_code}: {detail}")

    # ------------------------------------------------------------------ #
    # Core endpoints
    # ------------------------------------------------------------------ #

    def health(self) -> dict[str, Any]:
        """Check API health (liveness)."""
        return self._request("GET", "/health")

    def ready(self) -> dict[str, Any]:
        """Check API readiness."""
        return self._request("GET", "/ready")

    def think(self, cycles: int = 1, **kwargs: Any) -> dict[str, Any]:
        """Execute think cycle(s)."""
        body = {"cycles": cycles, **kwargs}
        return self._request("POST", "/think", json=body)

    def get_knowledge_graph(self) -> dict[str, Any]:
        """Get knowledge graph."""
        return self._request("GET", "/knowledge-graph")

    def get_story_milestones(self) -> dict[str, Any]:
        """Get story milestones."""
        return self._request("GET", "/story-milestones")

    # ------------------------------------------------------------------ #
    # Phase 7: Architecture
    # ------------------------------------------------------------------ #

    def architect_stats(self) -> dict[str, Any]:
        return self._request("GET", "/architect/stats")

    def architect_dormant(self) -> dict[str, Any]:
        return self._request("GET", "/architect/dormant")

    def architect_action(self, action_type: str, module_name: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/architect/action",
            json={"action_type": action_type, "module_name": module_name},
        )

    # ------------------------------------------------------------------ #
    # Phase 7: Temporal Memory
    # ------------------------------------------------------------------ #

    def temporal_memory_state(self) -> dict[str, Any]:
        return self._request("GET", "/temporal_memory/state")

    def temporal_memory_reset(self) -> dict[str, Any]:
        return self._request("POST", "/temporal_memory/reset")

    # ------------------------------------------------------------------ #
    # Phase 7: Layered Predictor
    # ------------------------------------------------------------------ #

    def layered_predictor_beliefs(self) -> dict[str, Any]:
        return self._request("GET", "/layered_predictor/beliefs")

    def layered_predictor_update(self, observation: list[float]) -> dict[str, Any]:
        return self._request(
            "POST", "/layered_predictor/update", json={"observation": observation}
        )

    # ------------------------------------------------------------------ #
    # Phase 7: Episodic Graph
    # ------------------------------------------------------------------ #

    def episodic_graph_stats(self) -> dict[str, Any]:
        return self._request("GET", "/episodic_graph/stats")

    def episodic_graph_plan(self, start_id: int, goal_id: int) -> dict[str, Any]:
        return self._request(
            "POST",
            "/episodic_graph/plan",
            json={"start_id": start_id, "goal_id": goal_id},
        )

    def episodic_graph_insert(self, **kwargs: Any) -> dict[str, Any]:
        return self._request("POST", "/episodic_graph/insert", json=kwargs)

    # ------------------------------------------------------------------ #
    # Phase 7: Semantic Index
    # ------------------------------------------------------------------ #

    def semantic_index_search(self, vector: list[float], k: int = 5) -> dict[str, Any]:
        return self._request(
            "POST", "/semantic_index/search", json={"vector": vector, "k": k}
        )

    def semantic_index_stats(self) -> dict[str, Any]:
        return self._request("GET", "/semantic_index/stats")

    # ------------------------------------------------------------------ #
    # Phase 7: Logic Layer
    # ------------------------------------------------------------------ #

    def logic_rules(self) -> dict[str, Any]:
        return self._request("GET", "/logic_layer/rules")

    def logic_add_rule(self, **kwargs: Any) -> dict[str, Any]:
        return self._request("POST", "/logic_layer/add_rule", json=kwargs)

    def logic_remove_rule(self, rule_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/logic_layer/rules/{rule_id}")

    def logic_evaluate(self) -> dict[str, Any]:
        return self._request("POST", "/logic_layer/evaluate")

    def logic_violations(self) -> dict[str, Any]:
        return self._request("GET", "/logic_layer/violations")

    # ------------------------------------------------------------------ #
    # Phase 7: Causal Inference
    # ------------------------------------------------------------------ #

    def causal_graph(self) -> dict[str, Any]:
        return self._request("GET", "/causal_inference/graph")

    def causal_intervene(self, **kwargs: Any) -> dict[str, Any]:
        return self._request("POST", "/causal_inference/intervene", json=kwargs)

    def causal_observe(self, **kwargs: Any) -> dict[str, Any]:
        return self._request("POST", "/causal_inference/observe", json=kwargs)

    def causal_counterfactual(self, **kwargs: Any) -> dict[str, Any]:
        return self._request("POST", "/causal_inference/counterfactual", json=kwargs)

    # ------------------------------------------------------------------ #
    # Phase 7: Meta-Cognition
    # ------------------------------------------------------------------ #

    def metacog_state(self) -> dict[str, Any]:
        return self._request("GET", "/meta_cognition/state")

    def metacog_confidence(self) -> dict[str, Any]:
        return self._request("GET", "/meta_cognition/confidence")

    def metacog_uncertainty(self) -> dict[str, Any]:
        return self._request("GET", "/meta_cognition/uncertainty")

    def metacog_report(self) -> dict[str, Any]:
        return self._request("GET", "/meta_cognition/report")

    # ------------------------------------------------------------------ #
    # Phase 7: Experiment Planner
    # ------------------------------------------------------------------ #

    def experiment_candidates(self) -> dict[str, Any]:
        return self._request("GET", "/experiment_planner/candidates")

    def experiment_select(self) -> dict[str, Any]:
        return self._request("POST", "/experiment_planner/select")

    def experiment_result(self, candidate_id: str, outcome: float) -> dict[str, Any]:
        return self._request(
            "POST",
            "/experiment_planner/result",
            json={"candidate_id": candidate_id, "outcome": outcome},
        )

    def experiment_history(self) -> dict[str, Any]:
        return self._request("GET", "/experiment_planner/history")

    # ------------------------------------------------------------------ #
    # Phase 7: Hypothesis Tester
    # ------------------------------------------------------------------ #

    def hypothesis_list(self) -> dict[str, Any]:
        return self._request("GET", "/hypothesis_tester/list")

    def hypothesis_test(self, hypothesis_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._request(
            "POST",
            "/hypothesis_tester/test",
            json={"hypothesis_id": hypothesis_id, **kwargs},
        )

    def hypothesis_supported(self) -> dict[str, Any]:
        return self._request("GET", "/hypothesis_tester/supported")

    # ------------------------------------------------------------------ #
    # Phase 7: Multi-Agent (world / communication / culture)
    # ------------------------------------------------------------------ #

    def world_stats(self) -> dict[str, Any]:
        return self._request("GET", "/world/stats")

    def world_step(self) -> dict[str, Any]:
        return self._request("POST", "/world/step")

    def world_agents(self) -> dict[str, Any]:
        return self._request("GET", "/world/agents")

    def communication_channels(self) -> dict[str, Any]:
        return self._request("GET", "/communication/channels")

    def communication_send(self, **kwargs: Any) -> dict[str, Any]:
        return self._request("POST", "/communication/send", json=kwargs)

    def culture_generations(self) -> dict[str, Any]:
        return self._request("GET", "/culture/generations")

    def culture_history(self) -> dict[str, Any]:
        return self._request("GET", "/culture/history")

    def culture_propagate(self, **kwargs: Any) -> dict[str, Any]:
        return self._request("POST", "/culture/propagate", json=kwargs)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "ZeroDataModelClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
