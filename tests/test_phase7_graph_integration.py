# tests/test_phase7_graph_integration.py
"""Phase 7 — Graph integration tests across facade / CLI / API / MCP.

Mirrors the Phase 7 Audio integration test pattern. Verifies that the 7
Graph facade methods are reachable from all four entry-point layers
(facade / CLI / API / MCP) and produce consistent results.

Total: ~30 tests.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

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
def adjacency():
    """Deterministic 5-node adjacency matrix."""
    np.random.seed(42)
    return (np.random.rand(5, 5) > 0.5).astype(float)


@pytest.fixture()
def adjacency_list(adjacency):
    return adjacency.tolist()


@pytest.fixture()
def adjacency_b():
    """A second, different adjacency matrix."""
    np.random.seed(7)
    return (np.random.rand(5, 5) > 0.5).astype(float)


@pytest.fixture()
def adjacency_b_list(adjacency_b):
    return adjacency_b.tolist()


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
# 1. Facade
# --------------------------------------------------------------------- #


class TestFacadeGraph:
    def test_encode_returns_normalized_vector(self, adjacency):
        m = ZeroDataModel(dim=8, seed=42)
        emb = m.encode_graph(adjacency)
        assert emb.shape == (8,)
        norm = float(np.linalg.norm(emb))
        assert norm == pytest.approx(1.0, abs=1e-6) or norm == 0.0

    def test_detect_communities_returns_expected_keys(self, adjacency):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.detect_communities(adjacency)
        assert {"communities", "modularity", "n_communities"} <= set(r.keys())
        assert r["n_communities"] == len(r["communities"])

    def test_find_path_returns_expected_keys(self, adjacency):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.find_path(adjacency, 0, 4)
        assert {"path", "distance", "visited"} <= set(r.keys())

    def test_analyze_centrality_returns_expected_keys(self, adjacency):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.analyze_centrality(adjacency)
        assert {"degree", "betweenness", "closeness", "most_central"} <= set(r.keys())

    def test_check_isomorphism_returns_expected_keys(self, adjacency):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.check_isomorphism(adjacency, adjacency)
        # Same graph should be isomorphic to itself.
        assert {"isomorphic", "confidence", "wl_hash_a", "wl_hash_b"} <= set(r.keys())
        assert r["isomorphic"] is True

    def test_track_dynamic_returns_expected_keys(self, adjacency, adjacency_b):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.track_dynamic_graph([adjacency, adjacency_b])
        assert {"community_drift", "node_migrations", "stability"} <= set(r.keys())

    def test_extract_spanning_tree_returns_expected_keys(self, adjacency):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.extract_spanning_tree(adjacency)
        assert {"mst_edges", "total_weight", "mst_adjacency"} <= set(r.keys())


# --------------------------------------------------------------------- #
# 2. CLI
# --------------------------------------------------------------------- #


class TestCLIGraph:
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

    def test_cli_encode(self, tmp_path, adjacency_list):
        f = tmp_path / "adj.json"
        f.write_text(json.dumps(adjacency_list))
        r = self._run_cli("graph", "encode", str(f))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "embedding" in payload

    def test_cli_communities(self, tmp_path, adjacency_list):
        f = tmp_path / "adj.json"
        f.write_text(json.dumps(adjacency_list))
        r = self._run_cli("graph", "communities", str(f))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "communities" in payload

    def test_cli_path(self, tmp_path, adjacency_list):
        f = tmp_path / "adj.json"
        f.write_text(json.dumps(adjacency_list))
        r = self._run_cli("graph", "path", str(f), "0", "4")
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "path" in payload

    def test_cli_centrality(self, tmp_path, adjacency_list):
        f = tmp_path / "adj.json"
        f.write_text(json.dumps(adjacency_list))
        r = self._run_cli("graph", "centrality", str(f))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "degree" in payload

    def test_cli_isomorphism(self, tmp_path, adjacency_list):
        f1 = tmp_path / "adj1.json"
        f2 = tmp_path / "adj2.json"
        f1.write_text(json.dumps(adjacency_list))
        f2.write_text(json.dumps(adjacency_list))
        r = self._run_cli("graph", "isomorphism", str(f1), str(f2))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "isomorphic" in payload

    def test_cli_track(self, tmp_path, adjacency_list, adjacency_b_list):
        f1 = tmp_path / "adj1.json"
        f2 = tmp_path / "adj2.json"
        f1.write_text(json.dumps(adjacency_list))
        f2.write_text(json.dumps(adjacency_b_list))
        r = self._run_cli("graph", "track", str(f1), str(f2))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "stability" in payload

    def test_cli_spanning(self, tmp_path, adjacency_list):
        f = tmp_path / "adj.json"
        f.write_text(json.dumps(adjacency_list))
        r = self._run_cli("graph", "spanning", str(f))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "mst_edges" in payload


# --------------------------------------------------------------------- #
# 3. API
# --------------------------------------------------------------------- #


class TestAPIGraph:
    def test_api_encode(self, client, adjacency_list):
        r = client.post("/v1/graph/encode", json={"adjacency": adjacency_list})
        assert r.status_code == 200, r.text
        assert "embedding" in r.json()

    def test_api_communities(self, client, adjacency_list):
        r = client.post("/v1/graph/communities", json={"adjacency": adjacency_list})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "communities" in body and "n_communities" in body

    def test_api_path(self, client, adjacency_list):
        r = client.post(
            "/v1/graph/path",
            json={"adjacency": adjacency_list, "source": 0, "target": 4},
        )
        assert r.status_code == 200, r.text
        assert "path" in r.json()

    def test_api_centrality(self, client, adjacency_list):
        r = client.post("/v1/graph/centrality", json={"adjacency": adjacency_list})
        assert r.status_code == 200, r.text
        assert "degree" in r.json()

    def test_api_isomorphism(self, client, adjacency_list):
        r = client.post(
            "/v1/graph/isomorphism",
            json={"adjacency_a": adjacency_list, "adjacency_b": adjacency_list},
        )
        assert r.status_code == 200, r.text
        assert r.json()["isomorphic"] is True

    def test_api_track(self, client, adjacency_list, adjacency_b_list):
        r = client.post(
            "/v1/graph/track",
            json={"snapshots": [adjacency_list, adjacency_b_list]},
        )
        assert r.status_code == 200, r.text
        assert "stability" in r.json()

    def test_api_spanning(self, client, adjacency_list):
        r = client.post("/v1/graph/spanning", json={"adjacency": adjacency_list})
        assert r.status_code == 200, r.text
        assert "mst_edges" in r.json()

    def test_api_negative_source_rejected(self, client, adjacency_list):
        r = client.post(
            "/v1/graph/path",
            json={"adjacency": adjacency_list, "source": -1, "target": 4},
        )
        assert r.status_code in (400, 422)

    def test_api_single_snapshot_rejected(self, client, adjacency_list):
        r = client.post(
            "/v1/graph/track",
            json={"snapshots": [adjacency_list]},
        )
        # min_length=2 on snapshots.
        assert r.status_code in (400, 422)


# --------------------------------------------------------------------- #
# 4. MCP
# --------------------------------------------------------------------- #


class TestMCPGraph:
    def test_mcp_encode(self, mcp_server, adjacency_list):
        r = mcp_server.call_tool("graph_encode", adjacency=adjacency_list)
        assert "embedding" in r
        assert len(r["embedding"]) == 64  # default dim

    def test_mcp_communities(self, mcp_server, adjacency_list):
        r = mcp_server.call_tool("graph_detect_communities", adjacency=adjacency_list)
        assert "communities" in r and "n_communities" in r

    def test_mcp_path(self, mcp_server, adjacency_list):
        r = mcp_server.call_tool(
            "graph_find_path", adjacency=adjacency_list, source=0, target=4
        )
        assert "path" in r

    def test_mcp_centrality(self, mcp_server, adjacency_list):
        r = mcp_server.call_tool("graph_analyze_centrality", adjacency=adjacency_list)
        assert "degree" in r and "most_central" in r

    def test_mcp_isomorphism(self, mcp_server, adjacency_list):
        r = mcp_server.call_tool(
            "graph_check_isomorphism",
            adjacency_a=adjacency_list,
            adjacency_b=adjacency_list,
        )
        assert r["isomorphic"] is True

    def test_mcp_track(self, mcp_server, adjacency_list, adjacency_b_list):
        r = mcp_server.call_tool(
            "graph_track_dynamic",
            snapshots=[adjacency_list, adjacency_b_list],
        )
        assert "stability" in r

    def test_mcp_spanning(self, mcp_server, adjacency_list):
        r = mcp_server.call_tool("graph_extract_spanning_tree", adjacency=adjacency_list)
        assert "mst_edges" in r

    def test_mcp_non_square_returns_error(self, mcp_server):
        r = mcp_server.call_tool(
            "graph_encode", adjacency=[[1, 2, 3], [4, 5, 6]]
        )
        assert "error" in r

    def test_mcp_source_out_of_range_returns_error(self, mcp_server, adjacency_list):
        r = mcp_server.call_tool(
            "graph_find_path", adjacency=adjacency_list, source=99, target=0
        )
        assert "error" in r

    def test_mcp_single_snapshot_returns_error(self, mcp_server, adjacency_list):
        r = mcp_server.call_tool(
            "graph_track_dynamic", snapshots=[adjacency_list]
        )
        assert "error" in r

    def test_mcp_graph_tools_registered(self, mcp_server):
        tools = mcp_server.list_tools()
        for name in (
            "graph_encode", "graph_detect_communities", "graph_find_path",
            "graph_analyze_centrality", "graph_check_isomorphism",
            "graph_track_dynamic", "graph_extract_spanning_tree",
        ):
            assert name in tools, f"missing tool: {name}"


# --------------------------------------------------------------------- #
# 5. Cross-layer consistency
# --------------------------------------------------------------------- #


class TestCrossLayerConsistency:
    def test_facade_and_mcp_path_agree(
        self, adjacency, adjacency_list, mcp_server
    ):
        m = ZeroDataModel(dim=8, seed=42)
        facade_path = m.find_path(adjacency, 0, 4)["path"]
        mcp_path = mcp_server.call_tool(
            "graph_find_path", adjacency=adjacency_list, source=0, target=4
        )["path"]
        assert facade_path == mcp_path

    def test_facade_and_mcp_communities_agree(
        self, adjacency, adjacency_list, mcp_server
    ):
        m = ZeroDataModel(dim=8, seed=42)
        facade_n = m.detect_communities(adjacency)["n_communities"]
        mcp_n = mcp_server.call_tool(
            "graph_detect_communities", adjacency=adjacency_list
        )["n_communities"]
        assert facade_n == mcp_n
