# tests/test_api_emergence.py
"""Tests for the /emergence/* FastAPI endpoints.

These tests cover the six causal-emergence endpoints added to ``api.py``:
``/emergence/perceive``, ``/emergence/causal``, ``/emergence/trajectory``,
``/emergence/sample``, ``/emergence/recall`` and ``/emergence/cycle``.

The web stack (fastapi + httpx) is an optional dependency, so the whole
module is skipped when either is missing.
"""

from __future__ import annotations

import pytest

# Skip the entire module if fastapi is not installed (the web extra is optional).
pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by fastapi.testclient.TestClient

from fastapi.testclient import TestClient  # noqa: E402

from zero_data_model import api as api_module  # noqa: E402
from zero_data_model.api import create_app  # noqa: E402
from zero_data_model.model import ZeroDataModel  # noqa: E402
from zero_data_model.persistence import set_persistence_root  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    """A TestClient backed by a fresh small model for test isolation.

    Mirrors the fixture in ``tests/test_api.py``: persistence is sandboxed
    under a per-test tmp_path, the singleton is replaced with a cheap
    ``dim=16`` model, and slowapi rate limiting is disabled so successive
    calls across tests don't trip the per-route caps.
    """
    set_persistence_root(str(tmp_path / "api_persistence"))
    api_module.set_model(ZeroDataModel(dim=16))
    prev_limiter_enabled = None
    if api_module.limiter is not None:
        prev_limiter_enabled = api_module.limiter.enabled
        api_module.limiter.enabled = False
    app = create_app()
    # TrustedHostMiddleware rejects non-allow-listed Host headers; point the
    # TestClient at ``localhost`` (in the default allow-list) so the check
    # passes without each test having to set the env var.
    with TestClient(app, base_url="http://localhost") as c:
        yield c
    # Reset the singleton so other tests get a fresh lazy-init dim=32 model.
    api_module._model = None
    if api_module.limiter is not None and prev_limiter_enabled is not None:
        api_module.limiter.enabled = prev_limiter_enabled


# 10x3 observation shared across tests (n_samples=10, n_features=3).
SAMPLE_2D = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5],
    [1.6, 1.7, 1.8],
    [1.9, 2.0, 2.1],
    [2.2, 2.3, 2.4],
    [2.5, 2.6, 2.7],
    [2.8, 2.9, 3.0],
]


# --------------------------------------------------------------------------- #
# /emergence/perceive
# --------------------------------------------------------------------------- #
def test_emergence_perceive_success(client):
    """POST /emergence/perceive returns topological invariants."""
    r = client.post("/emergence/perceive", json={"data": SAMPLE_2D})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["betti_numbers"], list)
    assert all(isinstance(x, int) for x in body["betti_numbers"])
    assert isinstance(body["persistence_entropy"], float)
    assert isinstance(body["euler_characteristic"], int)
    assert body["n_points"] == 10
    assert isinstance(body["max_eps"], float)
    # persistence_diagram omitted by default.
    assert body.get("persistence_diagram") is None


def test_emergence_perceive_1d_data_returns_400(client):
    """POST /emergence/perceive with a 1D-shaped data (n_features<2) -> 400."""
    r = client.post("/emergence/perceive", json={"data": [[0.1], [0.2], [0.3]]})
    assert r.status_code == 400


def test_emergence_perceive_full_diagram_query_param(client):
    """?full_diagram=true includes the persistence_diagram in the response."""
    r = client.post(
        "/emergence/perceive",
        json={"data": SAMPLE_2D},
        params={"full_diagram": "true"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["persistence_diagram"] is not None
    assert isinstance(body["persistence_diagram"], list)


# --------------------------------------------------------------------------- #
# /emergence/causal
# --------------------------------------------------------------------------- #
def test_emergence_causal_success(client):
    """POST /emergence/causal returns a DAG with adjacency + edges."""
    r = client.post("/emergence/causal", json={"data": SAMPLE_2D})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["adjacency"], list)
    assert all(isinstance(row, list) for row in body["adjacency"])
    assert all(
        isinstance(x, int) for row in body["adjacency"] for x in row
    )
    assert isinstance(body["edges"], list)
    assert all(isinstance(edge, list) for edge in body["edges"])
    assert isinstance(body["n_edges"], int)
    assert body["n_edges"] == len(body["edges"])
    assert isinstance(body["is_acyclic"], bool)
    assert isinstance(body["method"], str)
    assert body["method"] in {"pc", "lingam", "correlation"}
    assert isinstance(body["var_names"], list)
    assert len(body["var_names"]) == 3


def test_emergence_causal_invalid_method_returns_422(client):
    """POST /emergence/causal with an unknown method is rejected with 422."""
    r = client.post(
        "/emergence/causal",
        json={"data": SAMPLE_2D, "method": "bogus"},
    )
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# /emergence/trajectory
# --------------------------------------------------------------------------- #
def test_emergence_trajectory_success(client):
    """POST /emergence/trajectory returns trajectory + lagrangian + action."""
    r = client.post(
        "/emergence/trajectory",
        json={
            "start_state": [0.0, 0.0, 0.0],
            "end_state": [1.0, 1.0, 1.0],
            "n_steps": 8,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["trajectory"], list)
    assert len(body["trajectory"]) == 9  # n_steps + 1
    assert all(isinstance(row, list) for row in body["trajectory"])
    assert all(
        isinstance(x, float) for row in body["trajectory"] for x in row
    )
    assert isinstance(body["lagrangian"], list)
    assert len(body["lagrangian"]) == 8
    assert isinstance(body["action"], float)
    assert isinstance(body["converged"], bool)
    assert isinstance(body["iterations"], int)
    # No obstacles -> obstacle_violations is None.
    assert body.get("obstacle_violations") is None


def test_emergence_trajectory_mismatched_shapes_returns_400(client):
    """POST /emergence/trajectory with mismatched start/end shapes -> 400."""
    r = client.post(
        "/emergence/trajectory",
        json={"start_state": [0.0, 0.0], "end_state": [1.0, 1.0, 1.0]},
    )
    assert r.status_code == 400


def test_emergence_trajectory_with_obstacles(client):
    """POST /emergence/trajectory with obstacles returns obstacle_violations."""
    r = client.post(
        "/emergence/trajectory",
        json={
            "start_state": [0.0, 0.0, 0.0],
            "end_state": [1.0, 1.0, 1.0],
            "n_steps": 8,
            "obstacles": [[0.5, 0.5, 0.5]],
            "margin": 0.3,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["obstacle_violations"], int)
    assert body["obstacle_violations"] >= 0


# --------------------------------------------------------------------------- #
# /emergence/sample
# --------------------------------------------------------------------------- #
def test_emergence_sample_success(client):
    """POST /emergence/sample returns posterior mean/std + diagnostics."""
    r = client.post(
        "/emergence/sample",
        json={"mean": [0.0, 0.0, 0.0], "std": 1.0, "n_samples": 50},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["mean"], list)
    assert len(body["mean"]) == 3
    assert all(isinstance(x, float) for x in body["mean"])
    assert isinstance(body["std"], list)
    assert len(body["std"]) == 3
    assert isinstance(body["accept_rate"], float)
    assert 0.0 <= body["accept_rate"] <= 1.0
    assert isinstance(body["ess"], float)
    assert isinstance(body["converged"], bool)


def test_emergence_sample_zero_n_samples_returns_422(client):
    """POST /emergence/sample with n_samples=0 is rejected with 422 (ge=1)."""
    r = client.post(
        "/emergence/sample",
        json={"mean": [0.0, 0.0], "std": 1.0, "n_samples": 0},
    )
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# /emergence/recall
# --------------------------------------------------------------------------- #
def test_emergence_recall_success(client):
    """POST /emergence/recall returns memory recall fields.

    The model starts with an empty memory, so ``label`` is None and
    ``emerged`` is True (the spec defines empty-memory recall as emerged).
    """
    r = client.post(
        "/emergence/recall",
        json={"query": [0.1, 0.2, 0.3], "n_steps": 20},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] is None
    assert isinstance(body["similarity"], float)
    assert isinstance(body["emerged"], bool)
    assert isinstance(body["trajectory"], list)
    assert len(body["trajectory"]) == 21  # n_steps + 1
    assert all(isinstance(row, list) for row in body["trajectory"])
    assert all(
        isinstance(x, float) for row in body["trajectory"] for x in row
    )
    assert isinstance(body["converged"], bool)
    assert isinstance(body["divergence"], float)


def test_emergence_recall_empty_query_returns_422(client):
    """POST /emergence/recall with an empty query is rejected with 422."""
    r = client.post("/emergence/recall", json={"query": []})
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# /emergence/cycle
# --------------------------------------------------------------------------- #
def test_emergence_cycle_success(client):
    """POST /emergence/cycle returns the full emergence loop output."""
    r = client.post("/emergence/cycle", json={"observation": SAMPLE_2D})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "emergence_score" in body
    assert isinstance(body["emergence_score"], float)
    assert 0.0 <= body["emergence_score"] <= 1.0
    for key in (
        "perception",
        "causal_graph",
        "counterfactual",
        "posterior",
        "memory",
        "warnings",
    ):
        assert key in body, f"missing key: {key}"
    assert isinstance(body["warnings"], list)


def test_emergence_cycle_1d_observation_returns_400(client):
    """POST /emergence/cycle with a 1D observation (n_features<2) -> 400."""
    r = client.post(
        "/emergence/cycle",
        json={"observation": [[0.1], [0.2], [0.3]]},
    )
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Auth (CWE-306) — /emergence/* must honour ZDM_API_KEY
# --------------------------------------------------------------------------- #
def test_emergence_api_key_enforced_when_configured(client, monkeypatch):
    """When ZDM_API_KEY is set, /emergence/* without the header is 401."""
    monkeypatch.setenv("ZDM_API_KEY", "secret-key-123")
    r = client.post("/emergence/perceive", json={"data": SAMPLE_2D})
    assert r.status_code == 401
    # With the correct header, the request succeeds.
    r = client.post(
        "/emergence/perceive",
        json={"data": SAMPLE_2D},
        headers={"X-API-Key": "secret-key-123"},
    )
    assert r.status_code == 200
