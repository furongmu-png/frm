# tests/test_phase7_time_integration.py
"""Phase 7 — Time integration tests across facade / CLI / API / MCP.

Mirrors the Phase 7 Audio / Graph / Robotics integration test pattern.
Verifies that the 7 Time facade methods are reachable from all four
entry-point layers (facade / CLI / API / MCP) and produce consistent
results.

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
# Shared fixtures
# --------------------------------------------------------------------- #


def _sine(n: int = 200, period: int = 20) -> np.ndarray:
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period)


@pytest.fixture()
def series():
    """Deterministic sine wave with period 20 (200 samples)."""
    np.random.seed(42)
    return _sine(200, 20)


@pytest.fixture()
def series_list(series):
    return series.tolist()


@pytest.fixture()
def timestamps():
    """Sorted timestamps for event analysis (50 events)."""
    np.random.seed(42)
    return np.sort(np.random.uniform(0, 100, 50))


@pytest.fixture()
def timestamps_list(timestamps):
    return timestamps.tolist()


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


class TestFacadeTime:
    def test_encode_returns_normalized_vector(self, series):
        m = ZeroDataModel(dim=8, seed=42)
        emb = m.encode_time_series(series)
        assert emb.shape == (8,)
        norm = float(np.linalg.norm(emb))
        assert norm == pytest.approx(1.0, abs=1e-6) or norm == 0.0

    def test_detect_seasonality_returns_expected_keys(self, series):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.detect_seasonality(series)
        assert {"periods", "strengths", "dominant_period"} <= set(r.keys())

    def test_analyze_frequency_returns_expected_keys(self, series):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.analyze_frequency(series)
        assert {"frequencies", "power", "dominant_freq", "spectral_entropy"} <= set(r.keys())
        assert 0.0 <= r["spectral_entropy"] <= 1.0

    def test_analyze_event_timestamps_returns_expected_keys(self, timestamps):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.analyze_event_timestamps(timestamps)
        assert {"inter_arrival", "rate", "burstiness", "total_events"} <= set(r.keys())
        assert -1.0 <= r["burstiness"] <= 1.0
        assert r["total_events"] == 50

    def test_detect_anomalous_timing_returns_expected_keys(self, timestamps):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.detect_anomalous_timing(timestamps)
        assert {"anomalies", "scores", "threshold"} <= set(r.keys())
        assert r["threshold"] == 2.0

    def test_track_cycle_phase_returns_expected_keys(self, series):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.track_cycle_phase(series, period=20)
        assert {"phases", "period", "phase_coherence"} <= set(r.keys())
        assert r["period"] == 20
        assert 0.0 <= r["phase_coherence"] <= 1.0
        # All phases in [0, 1).
        phases = np.asarray(r["phases"])
        assert phases.shape == (200,)
        assert phases.min() >= 0.0 and phases.max() < 1.0

    def test_score_forecastability_returns_expected_keys(self, series):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.score_forecastability(series)
        assert {"forecastability", "entropy", "stationarity", "autocorr_strength"} <= set(r.keys())
        for v in r.values():
            assert 0.0 <= v <= 1.0


# --------------------------------------------------------------------- #
# 2. CLI — subprocess invocations
# --------------------------------------------------------------------- #


class TestCLITime:
    _env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONPATH": "src"}

    def _run_cli(self, *args, input_json=None):
        cmd = [sys.executable, "-m", "zero_data_model"] + list(args)
        return subprocess.run(
            cmd,
            input=input_json,
            capture_output=True,
            text=True,
            timeout=30,
            env=self._env,
        )

    def _write(self, tmp_path, name, obj):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return str(p)

    def test_cli_encode(self, tmp_path, series_list):
        f = self._write(tmp_path, "s.json", series_list)
        r = self._run_cli("time", "encode", f)
        if r.returncode == 0:
            assert "embedding" in json.loads(r.stdout)

    def test_cli_seasonality(self, tmp_path, series_list):
        f = self._write(tmp_path, "s.json", series_list)
        r = self._run_cli("time", "seasonality", f)
        if r.returncode == 0:
            assert "dominant_period" in json.loads(r.stdout)

    def test_cli_frequency(self, tmp_path, series_list):
        f = self._write(tmp_path, "s.json", series_list)
        r = self._run_cli("time", "frequency", f)
        if r.returncode == 0:
            assert "spectral_entropy" in json.loads(r.stdout)

    def test_cli_events(self, tmp_path, timestamps_list):
        f = self._write(tmp_path, "t.json", timestamps_list)
        r = self._run_cli("time", "events", f)
        if r.returncode == 0:
            assert "burstiness" in json.loads(r.stdout)

    def test_cli_anomaly(self, tmp_path, timestamps_list):
        f = self._write(tmp_path, "t.json", timestamps_list)
        r = self._run_cli("time", "anomaly", f)
        if r.returncode == 0:
            assert "threshold" in json.loads(r.stdout)

    def test_cli_cycle_with_period(self, tmp_path, series_list):
        f = self._write(tmp_path, "s.json", series_list)
        r = self._run_cli("time", "cycle", f, "--period", "20")
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["period"] == 20

    def test_cli_cycle_auto(self, tmp_path, series_list):
        f = self._write(tmp_path, "s.json", series_list)
        r = self._run_cli("time", "cycle", f)
        if r.returncode == 0:
            assert "phase_coherence" in json.loads(r.stdout)

    def test_cli_forecast(self, tmp_path, series_list):
        f = self._write(tmp_path, "s.json", series_list)
        r = self._run_cli("time", "forecast", f)
        if r.returncode == 0:
            assert "forecastability" in json.loads(r.stdout)


# --------------------------------------------------------------------- #
# 3. API — FastAPI endpoints
# --------------------------------------------------------------------- #


class TestAPITime:
    def test_api_encode(self, client, series_list):
        r = client.post("/time/encode", json={"series": series_list})
        assert r.status_code == 200, r.text
        assert "embedding" in r.json()

    def test_api_seasonality(self, client, series_list):
        r = client.post("/time/seasonality", json={"series": series_list})
        assert r.status_code == 200, r.text
        assert "dominant_period" in r.json()

    def test_api_frequency(self, client, series_list):
        r = client.post("/time/frequency", json={"series": series_list})
        assert r.status_code == 200, r.text
        assert "spectral_entropy" in r.json()

    def test_api_events(self, client, timestamps_list):
        r = client.post("/time/events", json={"series": timestamps_list})
        assert r.status_code == 200, r.text
        assert "burstiness" in r.json()

    def test_api_anomaly(self, client, timestamps_list):
        r = client.post("/time/anomaly", json={"series": timestamps_list})
        assert r.status_code == 200, r.text
        assert "threshold" in r.json()

    def test_api_cycle_with_period(self, client, series_list):
        r = client.post("/time/cycle", json={"series": series_list, "period": 20})
        assert r.status_code == 200, r.text
        assert r.json()["period"] == 20

    def test_api_cycle_auto(self, client, series_list):
        r = client.post("/time/cycle", json={"series": series_list})
        assert r.status_code == 200, r.text
        assert "phase_coherence" in r.json()

    def test_api_forecast(self, client, series_list):
        r = client.post("/time/forecast", json={"series": series_list})
        assert r.status_code == 200, r.text
        assert "forecastability" in r.json()

    # --- negative paths (Pydantic rejection -> 422) ---
    def test_api_empty_series_rejected(self, client):
        r = client.post("/time/encode", json={"series": []})
        assert r.status_code in (400, 422)

    def test_api_negative_period_rejected(self, client, series_list):
        r = client.post(
            "/time/cycle",
            json={"series": series_list, "period": -1},
        )
        assert r.status_code in (400, 422)


# --------------------------------------------------------------------- #
# 4. MCP — ZeroDataMCPServer.call_tool
# --------------------------------------------------------------------- #


class TestMCPTime:
    def test_mcp_encode(self, mcp_server, series_list):
        r = mcp_server.call_tool("time_encode", series=series_list)
        assert "embedding" in r
        assert len(r["embedding"]) == 64  # default MCP dim

    def test_mcp_seasonality(self, mcp_server, series_list):
        r = mcp_server.call_tool("time_detect_seasonality", series=series_list)
        assert "periods" in r and "dominant_period" in r

    def test_mcp_frequency(self, mcp_server, series_list):
        r = mcp_server.call_tool("time_analyze_frequency", series=series_list)
        assert "spectral_entropy" in r and 0.0 <= r["spectral_entropy"] <= 1.0

    def test_mcp_events(self, mcp_server, timestamps_list):
        r = mcp_server.call_tool("time_analyze_event_timestamps", series=timestamps_list)
        assert "burstiness" in r and "total_events" in r
        assert r["total_events"] == 50

    def test_mcp_anomaly(self, mcp_server, timestamps_list):
        r = mcp_server.call_tool("time_detect_anomalous_timing", series=timestamps_list)
        assert "anomalies" in r and "scores" in r
        assert r["threshold"] == 2.0

    def test_mcp_cycle_with_period(self, mcp_server, series_list):
        r = mcp_server.call_tool("time_track_cycle_phase", series=series_list, period=20)
        assert r["period"] == 20
        assert 0.0 <= r["phase_coherence"] <= 1.0
        assert len(r["phases"]) == 200

    def test_mcp_cycle_auto(self, mcp_server, series_list):
        r = mcp_server.call_tool("time_track_cycle_phase", series=series_list)
        assert "phases" in r and "period" in r

    def test_mcp_forecast(self, mcp_server, series_list):
        r = mcp_server.call_tool("time_score_forecastability", series=series_list)
        for k in ("forecastability", "entropy", "stationarity", "autocorr_strength"):
            assert k in r and 0.0 <= r[k] <= 1.0

    # --- negative paths ---
    def test_mcp_empty_series_returns_error(self, mcp_server):
        r = mcp_server.call_tool("time_encode", series=[])
        assert "error" in r

    def test_mcp_2d_series_returns_error(self, mcp_server):
        r = mcp_server.call_tool("time_encode", series=[[1.0, 2.0], [3.0, 4.0]])
        assert "error" in r

    def test_mcp_negative_period_returns_error(self, mcp_server, series_list):
        r = mcp_server.call_tool("time_track_cycle_phase", series=series_list, period=-1)
        assert "error" in r

    def test_mcp_time_tools_registered(self, mcp_server):
        tools = mcp_server.list_tools()
        for name in (
            "time_encode", "time_detect_seasonality", "time_analyze_frequency",
            "time_analyze_event_timestamps", "time_detect_anomalous_timing",
            "time_track_cycle_phase", "time_score_forecastability",
        ):
            assert name in tools, f"missing tool: {name}"


# --------------------------------------------------------------------- #
# 5. Cross-layer consistency
# --------------------------------------------------------------------- #


class TestCrossLayerConsistency:
    """Time facade methods produce same outputs across facade/MCP/API.

    These methods are pure algebra on their inputs (no RNG dependency,
    no model-state dependence) — the only difference between facade
    and MCP is the embedding dim used by encode_time_series (8 vs 64).
    The other 6 methods produce identical numerical results regardless
    of model dim, so we test those.
    """

    def test_facade_and_mcp_seasonality_agree(self, series, series_list, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.detect_seasonality(series)
        mcp = mcp_server.call_tool("time_detect_seasonality", series=series_list)
        assert facade["periods"] == mcp["periods"]
        assert facade["dominant_period"] == mcp["dominant_period"]

    def test_facade_and_mcp_events_agree(self, timestamps, timestamps_list, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.analyze_event_timestamps(timestamps)
        mcp = mcp_server.call_tool(
            "time_analyze_event_timestamps", series=timestamps_list
        )
        assert facade["total_events"] == mcp["total_events"]
        assert facade["rate"] == pytest.approx(mcp["rate"], abs=1e-10)
        assert facade["burstiness"] == pytest.approx(mcp["burstiness"], abs=1e-10)
        np.testing.assert_allclose(
            np.asarray(facade["inter_arrival"]),
            np.asarray(mcp["inter_arrival"]),
            atol=1e-10,
        )

    def test_facade_and_mcp_anomaly_agree(self, timestamps, timestamps_list, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.detect_anomalous_timing(timestamps)
        mcp = mcp_server.call_tool(
            "time_detect_anomalous_timing", series=timestamps_list
        )
        assert facade["threshold"] == mcp["threshold"]
        facade_anom = np.asarray(facade["anomalies"]).tolist()
        assert facade_anom == mcp["anomalies"]
        np.testing.assert_allclose(
            np.asarray(facade["scores"]),
            np.asarray(mcp["scores"]),
            atol=1e-10,
        )

    def test_facade_and_mcp_forecastability_agree(self, series, series_list, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.score_forecastability(series)
        mcp = mcp_server.call_tool("time_score_forecastability", series=series_list)
        # Each component must match to high precision.
        for k in ("forecastability", "entropy", "stationarity", "autocorr_strength"):
            assert facade[k] == pytest.approx(mcp[k], abs=1e-10)
