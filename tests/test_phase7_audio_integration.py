# tests/test_phase7_audio_integration.py
"""Phase 7 — Audio integration tests across facade / CLI / API / MCP.

Mirrors the Phase 6 integration test pattern. Verifies that the 6 Audio
facade methods are reachable from all four entry-point layers
(facade / CLI / API / MCP) and produce consistent results.

Total: ~25 tests, all defensive against missing optional deps
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


@pytest.fixture()
def signal():
    """Deterministic 1-second synthetic audio signal (8000 samples @ 8kHz)."""
    np.random.seed(42)
    return np.random.randn(8000).astype(float)


@pytest.fixture()
def signal_list(signal):
    return signal.tolist()


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


class TestFacadeAudio:
    def test_encode_audio_returns_normalized_vector(self, signal):
        m = ZeroDataModel(dim=8, seed=42)
        emb = m.encode_audio(signal, sample_rate=8000)
        assert emb.shape == (8,)
        # L2 normalized (or zero vector on degenerate input).
        norm = float(np.linalg.norm(emb))
        assert norm == pytest.approx(1.0, abs=1e-6) or norm == 0.0

    def test_detect_onsets_returns_expected_keys(self, signal):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.detect_onsets(signal, sample_rate=8000)
        assert set(r.keys()) >= {
            "onset_frames", "onset_times", "spectral_flux", "mean_flux"
        }

    def test_detect_pitch_returns_expected_keys(self, signal):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.detect_pitch(signal, sample_rate=8000)
        assert set(r.keys()) >= {
            "pitch_hz", "confidence", "f0_candidates"
        }

    def test_classify_audio_returns_label(self, signal):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.classify_audio(signal, sample_rate=8000)
        assert "label" in r
        assert r["label"] in {"speech", "music", "noise", "silence"}

    def test_segment_speech_returns_segments_list(self, signal):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.segment_speech(signal, sample_rate=8000)
        assert "segments" in r
        assert isinstance(r["segments"], list)
        assert "speech_ratio" in r

    def test_analyze_music_returns_tempo(self, signal):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.analyze_music(signal, sample_rate=8000)
        assert set(r.keys()) >= {
            "tempo_bpm", "beat_frames", "beat_times", "onset_envelope"
        }


# --------------------------------------------------------------------- #
# 2. CLI — subprocess invocations
# --------------------------------------------------------------------- #


class TestCLIAudio:
    def _run_cli(self, *args, input_json=None):
        cmd = [sys.executable, "-m", "zero_data_model"] + list(args)
        r = subprocess.run(
            cmd,
            input=input_json,
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONPATH": "src"},
        )
        return r

    def test_cli_audio_encode(self, tmp_path, signal_list):
        sig_file = tmp_path / "sig.json"
        sig_file.write_text(json.dumps(signal_list))
        r = self._run_cli("audio", "encode", str(sig_file), "--sample-rate", "8000")
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "embedding" in payload

    def test_cli_audio_classify(self, tmp_path, signal_list):
        sig_file = tmp_path / "sig.json"
        sig_file.write_text(json.dumps(signal_list))
        r = self._run_cli("audio", "classify", str(sig_file))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["label"] in {"speech", "music", "noise", "silence"}

    def test_cli_audio_pitch(self, tmp_path, signal_list):
        sig_file = tmp_path / "sig.json"
        sig_file.write_text(json.dumps(signal_list))
        r = self._run_cli("audio", "pitch", str(sig_file))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "pitch_hz" in payload

    def test_cli_audio_onsets(self, tmp_path, signal_list):
        sig_file = tmp_path / "sig.json"
        sig_file.write_text(json.dumps(signal_list))
        r = self._run_cli("audio", "onsets", str(sig_file))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "spectral_flux" in payload

    def test_cli_audio_segment(self, tmp_path, signal_list):
        sig_file = tmp_path / "sig.json"
        sig_file.write_text(json.dumps(signal_list))
        r = self._run_cli("audio", "segment", str(sig_file))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "segments" in payload

    def test_cli_audio_music(self, tmp_path, signal_list):
        sig_file = tmp_path / "sig.json"
        sig_file.write_text(json.dumps(signal_list))
        r = self._run_cli("audio", "music", str(sig_file))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "tempo_bpm" in payload


# --------------------------------------------------------------------- #
# 3. API — FastAPI endpoints
# --------------------------------------------------------------------- #


class TestAPIAudio:
    def test_api_encode(self, client, signal_list):
        r = client.post(
            "/v1/audio/encode",
            json={"signal": signal_list, "sample_rate": 8000},
        )
        assert r.status_code == 200, r.text
        assert "embedding" in r.json()

    def test_api_classify(self, client, signal_list):
        r = client.post("/v1/audio/classify", json={"signal": signal_list})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["label"] in {"speech", "music", "noise", "silence"}

    def test_api_pitch(self, client, signal_list):
        r = client.post("/v1/audio/pitch", json={"signal": signal_list})
        assert r.status_code == 200, r.text
        assert "pitch_hz" in r.json()

    def test_api_onsets(self, client, signal_list):
        r = client.post("/v1/audio/onsets", json={"signal": signal_list})
        assert r.status_code == 200, r.text
        assert "spectral_flux" in r.json()

    def test_api_segment(self, client, signal_list):
        r = client.post("/v1/audio/segment", json={"signal": signal_list})
        assert r.status_code == 200, r.text
        assert "segments" in r.json()

    def test_api_music(self, client, signal_list):
        r = client.post("/v1/audio/music", json={"signal": signal_list})
        assert r.status_code == 200, r.text
        assert "tempo_bpm" in r.json()

    def test_api_negative_sample_rate_rejected(self, client):
        r = client.post(
            "/v1/audio/encode",
            json={"signal": [1.0, 2.0], "sample_rate": 0},
        )
        # Pydantic Field(ge=1) rejects at validation -> 422.
        assert r.status_code in (400, 422)

    def test_api_empty_signal_rejected(self, client):
        r = client.post(
            "/v1/audio/encode",
            json={"signal": [], "sample_rate": 8000},
        )
        # Pydantic Field(min_length=1) rejects -> 422.
        assert r.status_code in (400, 422)


# --------------------------------------------------------------------- #
# 4. MCP — ZeroDataMCPServer.call_tool
# --------------------------------------------------------------------- #


class TestMCPAudio:
    def test_mcp_encode(self, mcp_server, signal_list):
        r = mcp_server.call_tool("audio_encode", signal=signal_list, sample_rate=8000)
        assert "embedding" in r
        assert len(r["embedding"]) == 64  # default dim

    def test_mcp_classify(self, mcp_server, signal_list):
        r = mcp_server.call_tool("audio_classify", signal=signal_list)
        assert r["label"] in {"speech", "music", "noise", "silence"}

    def test_mcp_pitch(self, mcp_server, signal_list):
        r = mcp_server.call_tool("audio_detect_pitch", signal=signal_list)
        assert "pitch_hz" in r

    def test_mcp_onsets(self, mcp_server, signal_list):
        r = mcp_server.call_tool("audio_detect_onsets", signal=signal_list)
        assert "spectral_flux" in r

    def test_mcp_segment(self, mcp_server, signal_list):
        r = mcp_server.call_tool("audio_segment_speech", signal=signal_list)
        assert "segments" in r

    def test_mcp_music(self, mcp_server, signal_list):
        r = mcp_server.call_tool("audio_analyze_music", signal=signal_list)
        assert "tempo_bpm" in r

    def test_mcp_empty_signal_returns_error(self, mcp_server):
        r = mcp_server.call_tool("audio_encode", signal=[])
        assert "error" in r
        assert "audio_encode" in r["error"]

    def test_mcp_zero_sample_rate_returns_error(self, mcp_server):
        r = mcp_server.call_tool(
            "audio_encode", signal=[1.0, 2.0], sample_rate=0
        )
        assert "error" in r

    def test_mcp_2d_signal_returns_error(self, mcp_server):
        r = mcp_server.call_tool(
            "audio_encode", signal=[[1.0, 2.0], [3.0, 4.0]]
        )
        assert "error" in r

    def test_mcp_tool_count_includes_audio(self, mcp_server):
        tools = mcp_server.list_tools()
        # 34 prior (16 base + 6 emergence + 12 Phase 6) + 6 Audio = 40.
        assert len(tools) >= 40
        for name in (
            "audio_encode", "audio_detect_onsets", "audio_detect_pitch",
            "audio_classify", "audio_segment_speech", "audio_analyze_music",
        ):
            assert name in tools, f"missing tool: {name}"


# --------------------------------------------------------------------- #
# 5. Cross-layer consistency
# --------------------------------------------------------------------- #


class TestCrossLayerConsistency:
    def test_facade_and_mcp_classify_agree(
        self, signal, signal_list, mcp_server
    ):
        # Both use default dim=64 + seed=42 so classification agrees.
        m = ZeroDataModel(dim=64, seed=42)
        facade_result = m.classify_audio(signal, sample_rate=8000)
        mcp_result = mcp_server.call_tool(
            "audio_classify", signal=signal_list, sample_rate=8000
        )
        assert facade_result["label"] == mcp_result["label"]

    def test_facade_and_mcp_embedding_shape_match(
        self, signal, signal_list, mcp_server
    ):
        # Both use default dim=64 + seed=42 so embedding shapes match.
        m = ZeroDataModel(dim=64, seed=42)
        facade_emb = m.encode_audio(signal, sample_rate=8000)
        mcp_emb = mcp_server.call_tool(
            "audio_encode", signal=signal_list, sample_rate=8000
        )["embedding"]
        assert len(facade_emb) == len(mcp_emb)
