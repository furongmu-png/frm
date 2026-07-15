# tests/test_api.py
"""Tests for the FastAPI REST API exposing the ZeroDataModel."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from zero_data_model import api as api_module
from zero_data_model.api import create_app
from zero_data_model.model import ZeroDataModel
from zero_data_model.persistence import set_persistence_root


@pytest.fixture()
def client(tmp_path):
    """A TestClient backed by a fresh small model for test isolation.

    Persistence is sandboxed under a per-test tmp_path so /save and /load
    can be exercised without touching the real /app/data root. Rate
    limiting is disabled so multiple tests sharing a TestClient IP don't
    trip the per-route 10/minute caps.
    """
    # Sandbox persistence for the duration of this test.
    set_persistence_root(str(tmp_path / "api_persistence"))
    # Replace the lazy singleton with a freshly-built dim=16 model so the
    # tests don't pay the dim=32 startup cost and stay independent.
    api_module.set_model(ZeroDataModel(dim=16))
    # Disable slowapi rate limiting for the duration of this test so
    # successive /think calls across tests don't exhaust the 10/minute
    # budget (all TestClient requests share the same client IP).
    prev_limiter_enabled = None
    if api_module.limiter is not None:
        prev_limiter_enabled = api_module.limiter.enabled
        api_module.limiter.enabled = False
    app = create_app()
    # TrustedHostMiddleware (S-MED-13) rejects non-allow-listed Host headers.
    # The TestClient defaults to ``http://testserver``; point it at
    # ``localhost`` (which is in the default ZDM_ALLOWED_HOSTS allow-list) so
    # the host-header check passes without each test having to set the env var.
    with TestClient(app, base_url="http://localhost") as c:
        yield c
    # Reset the singleton so other tests get a fresh lazy-init dim=32 model.
    api_module._model = None
    if api_module.limiter is not None and prev_limiter_enabled is not None:
        api_module.limiter.enabled = prev_limiter_enabled


def test_get_health_returns_hardware_info(client):
    """GET / returns status ok + hardware_info with backend names."""
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "hardware_info" in body
    hw = body["hardware_info"]
    assert "array_backend" in hw
    assert "quantum_backend" in hw
    assert isinstance(hw.get("gpu"), bool)


def test_health_and_ready_endpoints(client):
    """GET /health is cheap liveness; GET /ready probes the model."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_request_id_header_echoed(client):
    """Responses carry an X-Request-ID (minted or echoed)."""
    r = client.get("/health", headers={"X-Request-ID": "test-123"})
    assert r.headers.get("X-Request-ID") == "test-123"

    r = client.get("/health")
    assert r.headers.get("X-Request-ID")  # auto-generated, non-empty


def test_post_classify_returns_topic_and_confidence(client):
    """POST /classify returns a topic string + confidence in [0, 1]."""
    r = client.post("/classify", json={"text": "the algorithm computes the network"})
    assert r.status_code == 200
    body = r.json()
    assert body["topic"] in {"tech", "nature", "emotion", "science"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_post_similarity_returns_float(client):
    """POST /similarity returns a similarity float in [0, 1]."""
    r = client.post(
        "/similarity",
        json={"a": "code data model", "b": "code data model"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "similarity" in body
    assert isinstance(body["similarity"], float)
    assert 0.0 <= body["similarity"] <= 1.0
    # Identical inputs should be highly similar.
    assert body["similarity"] > 0.9


def test_post_forecast_returns_list_of_floats(client):
    """POST /forecast returns a list of length == horizon."""
    horizon = 5
    r = client.post(
        "/forecast",
        json={"series": list(range(20)), "horizon": horizon},
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["forecast"], list)
    assert len(body["forecast"]) == horizon
    assert all(isinstance(x, float) for x in body["forecast"])
    assert all(isinstance(x, (int, float)) for x in body["forecast"])


def test_post_think_with_no_input_runs_self_generation(client):
    """POST /think with no body runs self-generation and returns a cycle."""
    r = client.post("/think")
    assert r.status_code == 200
    body = r.json()
    assert body["cycle"] >= 1
    assert isinstance(body["output"], list)
    assert len(body["output"]) > 0
    assert all(isinstance(x, float) for x in body["output"])
    assert isinstance(body["confidence"], float)


def test_post_think_with_input(client):
    """POST /think with explicit input returns output of the same dim."""
    r = client.post("/think", json={"input": [0.1] * 16})
    assert r.status_code == 200
    body = r.json()
    assert body["cycle"] >= 1
    assert len(body["output"]) == 16


def test_post_generate_returns_text_of_requested_length(client):
    """POST /generate returns a string of the requested length."""
    r = client.post("/generate", json={"seed": "hello", "length": 16})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["text"], str)
    assert len(body["text"]) == 16


def test_post_anomalies_returns_bool_mask(client):
    """POST /anomalies returns a boolean mask matching the series length."""
    series = [1, 1, 1, 1, 100, 1, 1, 1, 1]
    r = client.post("/anomalies", json={"series": series})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["anomalies"], list)
    assert len(body["anomalies"]) == len(series)
    assert all(isinstance(x, bool) for x in body["anomalies"])
    assert any(body["anomalies"])


def test_post_trend_returns_all_fields(client):
    """POST /trend returns the full trend analyte dict."""
    r = client.post("/trend", json={"series": list(range(30))})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {
        "trend_slope",
        "regime",
        "curvature",
        "geodesic_deviation",
        "isomorphism_score",
    }
    assert body["regime"] in {"up", "down", "flat"}


def test_post_recognize_returns_shape_and_confidence(client):
    """POST /recognize returns a shape name and a confidence in [0, 1]."""
    image = [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]]
    r = client.post("/recognize", json={"image": image})
    assert r.status_code == 200
    body = r.json()
    assert body["shape"] in {"circle", "square", "triangle", "line", "blob"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_post_save_and_load_roundtrip(client):
    """POST /save writes a snapshot; POST /load replaces the live model.

    The snapshot name is a *relative* name (per the path-traversal fix);
    the persistence layer joins it to its sandboxed root.
    """
    snap = "snap_via_api"

    r = client.post("/save", json={"name": snap})
    assert r.status_code == 201
    assert r.json()["saved"] is True
    assert r.json()["name"] == snap
    # 201 Created must advertise the canonical resource URI (Q-LOW-13).
    assert r.headers["Location"] == f"/load/{snap}"

    # Run a couple more think() cycles so the in-memory model diverges.
    client.post("/think")
    client.post("/think")

    r = client.post("/load", json={"name": snap})
    assert r.status_code == 200
    assert r.json()["loaded"] is True

    # After load, the cycle counter should reflect the saved snapshot's cycle
    # count (the first think() after fresh model construction).
    r = client.post("/think")
    assert r.status_code == 200
    assert r.json()["cycle"] >= 1


def test_post_save_rejects_traversal_name(client):
    """POST /save with a traversal name returns 422 (regex) or 400 (sandbox)."""
    # The pydantic regex rejects ``..`` outright -> 422.
    r = client.post("/save", json={"name": "../escape"})
    assert r.status_code == 422


def test_post_save_rejects_absolute_name(client):
    """POST /save with an absolute path is rejected.

    The regex permits leading ``/`` (so the request passes pydantic), but
    the persistence-layer sandbox rejects absolute paths with ValueError,
    which the API surfaces as a 400.
    """
    r = client.post("/save", json={"name": "/etc/passwd"})
    assert r.status_code == 400


def test_api_key_enforced_when_configured(client, monkeypatch):
    """When ZDM_API_KEY is set, requests without it are 401."""
    monkeypatch.setenv("ZDM_API_KEY", "secret-key-123")
    r = client.get("/")
    assert r.status_code == 401
    r = client.get("/", headers={"X-API-Key": "secret-key-123"})
    assert r.status_code == 200


def test_health_ready_bypass_api_key(client, monkeypatch):
    """GET /health and /ready must NOT require an API key."""
    monkeypatch.setenv("ZDM_API_KEY", "secret-key-123")
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


# --------------------------------------------------------------------------- #
# 4xx error path tests (input validation + auth)
# --------------------------------------------------------------------------- #


def test_post_forecast_empty_series_returns_400(client):
    """POST /forecast with an empty series is rejected with 400."""
    r = client.post("/forecast", json={"series": [], "horizon": 3})
    assert r.status_code == 400


def test_post_anomalies_empty_series_returns_400(client):
    """POST /anomalies with an empty series is rejected with 400."""
    r = client.post("/anomalies", json={"series": []})
    assert r.status_code == 400


def test_post_trend_empty_series_returns_400(client):
    """POST /trend with an empty series is rejected with 400."""
    r = client.post("/trend", json={"series": []})
    assert r.status_code == 400


def test_post_recognize_empty_image_returns_4xx(client):
    """POST /recognize with an empty image is rejected.

    An empty list ``[]`` is caught by the pydantic ``_validate_rows`` validator
    and rejected with 422; a list with one empty row ``[[]]`` passes pydantic
    validation and is then rejected by the endpoint's
    ``if not req.image[0]`` check with 400. Both paths must reject the request.
    """
    # Empty list -> pydantic validator raises -> 422.
    r = client.post("/recognize", json={"image": []})
    assert r.status_code in {400, 422}
    # One empty row -> endpoint check fires -> 400.
    r = client.post("/recognize", json={"image": [[]]})
    assert r.status_code == 400


def test_post_recognize_non_2d_image_returns_4xx(client):
    """POST /recognize with a 1D image (single flat list of scalars) is rejected.

    A flat list of scalars (``[1.0, 2.0, 3.0, 4.0]``) cannot be coerced by
    pydantic into ``list[list[float]]`` and so is rejected with 422. The
    endpoint's own ``image.ndim != 2`` 400 path is exercised in
    ``test_post_recognize_empty_image_returns_4xx`` above.
    """
    r = client.post("/recognize", json={"image": [1.0, 2.0, 3.0, 4.0]})
    assert r.status_code in {400, 422}


def test_post_save_with_traversal_name_returns_4xx(client):
    """POST /save with a traversal name '../../etc/passwd' is rejected.

    The pydantic regex ``^[a-zA-Z0-9_\\-/]+$`` does not allow '.', so the
    request fails FastAPI's request-body validation with 422 before the
    persistence sandbox ever sees it. This is the same defence-in-depth
    pattern exercised by ``test_post_save_rejects_traversal_name`` above.
    """
    r = client.post("/save", json={"name": "../../etc/passwd"})
    assert r.status_code in {400, 422}


def test_post_load_missing_returns_404(client):
    """POST /load with a name that does not exist on disk returns 404."""
    r = client.post("/load", json={"name": "nonexistent_snapshot"})
    assert r.status_code == 404


def test_post_classify_missing_text_returns_422(client):
    """POST /classify with an empty body fails pydantic validation (422)."""
    # ClassifyRequest.text is Field(min_length=1); an empty body is rejected
    # because the required field is missing.
    r = client.post("/classify", json={})
    assert r.status_code == 422


def test_post_classify_empty_text_returns_422(client):
    """POST /classify with text='' is rejected by Field(min_length=1)."""
    r = client.post("/classify", json={"text": ""})
    assert r.status_code == 422


def test_post_forecast_oversized_horizon_returns_422(client):
    """POST /forecast with horizon > 1000 is rejected by Field(le=1000)."""
    r = client.post(
        "/forecast",
        json={"series": [1.0, 2.0, 3.0], "horizon": 1_000_000},
    )
    assert r.status_code == 422


def test_post_forecast_oversized_series_returns_422(client):
    """POST /forecast with a series > 10000 elements is rejected by
    Field(max_length=10000)."""
    big_series = [0.0] * 100_001
    r = client.post("/forecast", json={"series": big_series, "horizon": 1})
    assert r.status_code == 422


def test_post_generate_oversized_length_returns_422(client):
    """POST /generate with length > 256 is rejected by Field(le=256)."""
    r = client.post("/generate", json={"seed": "hello", "length": 1_000_000})
    assert r.status_code == 422


def test_health_endpoint_returns_ok_status(client):
    """GET /health returns 200 with status='ok'."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_ready_endpoint_returns_ok_when_model_constructed(client):
    """GET /ready returns 200 with model_ready=True once the model is built."""
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_api_key_required_when_set(tmp_path, monkeypatch):
    """When ZDM_API_KEY is set, requests without X-API-Key are 401 and
    requests with the correct header are 200.

    The API key is read from os.environ at request time inside
    ``verify_api_key``, so we can set/unset it via monkeypatch and rebuild the
    app to pick up the new auth state without touching other tests.
    """
    from zero_data_model.api import create_app as _create_app

    set_persistence_root(str(tmp_path / "api_persistence"))
    api_module.set_model(ZeroDataModel(dim=16))
    if api_module.limiter is not None:
        api_module.limiter.enabled = False

    monkeypatch.setenv("ZDM_API_KEY", "test-secret-xyz")
    app = _create_app()
    # Use ``localhost`` so TrustedHostMiddleware (S-MED-13) accepts the
    # request; the default ``testserver`` host is not in the allow-list.
    with TestClient(app, base_url="http://localhost") as c:
        # Without the header -> 401.
        r = c.get("/")
        assert r.status_code == 401
        # With the correct header -> 200.
        r = c.get("/", headers={"X-API-Key": "test-secret-xyz"})
        assert r.status_code == 200

    api_module._model = None
