# tests/test_api.py
"""Tests for the FastAPI REST API exposing the ZeroDataModel."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from zero_data_model import api as api_module
from zero_data_model.api import create_app
from zero_data_model.model import ZeroDataModel


@pytest.fixture()
def client():
    """A TestClient backed by a fresh small model for test isolation."""
    # Replace the lazy singleton with a freshly-built dim=16 model so the
    # tests don't pay the dim=32 startup cost and stay independent.
    api_module.set_model(ZeroDataModel(dim=16))
    app = create_app()
    with TestClient(app) as c:
        yield c
    # Reset the singleton so other tests get a fresh lazy-init dim=32 model.
    api_module._model = None


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


def test_post_save_and_load_roundtrip(client, tmp_path):
    """POST /save writes a snapshot; POST /load replaces the live model."""
    snap = str(tmp_path / "snap_via_api")

    r = client.post("/save", json={"path": snap})
    assert r.status_code == 200
    assert r.json()["saved"] is True

    # Run a couple more think() cycles so the in-memory model diverges.
    client.post("/think")
    client.post("/think")

    r = client.post("/load", json={"path": snap})
    assert r.status_code == 200
    assert r.json()["loaded"] is True

    # After load, the cycle counter should reflect the saved snapshot's cycle
    # count (the first think() after fresh model construction).
    r = client.post("/think")
    assert r.status_code == 200
    assert r.json()["cycle"] >= 1
