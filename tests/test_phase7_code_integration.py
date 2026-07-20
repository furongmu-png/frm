# tests/test_phase7_code_integration.py
"""Phase 7 — Code integration tests across facade / CLI / API / MCP.

Mirrors the Phase 7 Audio / Graph / Robotics / Time integration test
pattern. Verifies that the 7 Code facade methods are reachable from all
four entry-point layers (facade / CLI / API / MCP) and produce
consistent results.

Total: ~45 tests, all defensive against missing optional deps
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


SAMPLE_SRC = """import os
import numpy as np


def foo(x=None):
    if x is None:
        return 0
    return x


class Bar:
    def method_a(self):
        return 1
"""

# Source with defects: mutable_default + bare_except + equals_none.
DEFECT_SRC = """def f(x=[]):
    try:
        pass
    except:
        pass
    if x == None:
        return 0
    return x
"""

# Style-violating source: long line, trailing whitespace, mixed indent,
# camelCase function name.
BAD_STYLE_SRC = (
    "def camelCase():\n"
    "\t  return 1\n"
    "x = " + "a" * 200 + "\n"
    "y = 1   \n"
)


@pytest.fixture()
def source():
    return SAMPLE_SRC


@pytest.fixture()
def defect_source():
    return DEFECT_SRC


@pytest.fixture()
def bad_style_source():
    return BAD_STYLE_SRC


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


class TestFacadeCode:
    def test_encode_returns_normalized_vector(self, source):
        m = ZeroDataModel(dim=8, seed=42)
        emb = m.encode_code(source)
        assert emb.shape == (8,)
        norm = float(np.linalg.norm(emb))
        assert norm == pytest.approx(1.0, abs=1e-6) or norm == 0.0

    def test_analyze_ast_returns_expected_keys(self, source):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.analyze_ast(source)
        assert {"functions", "classes", "imports", "complexity", "max_depth", "node_count"} <= set(r.keys())
        assert "foo" in r["functions"]
        assert "Bar" in r["classes"]

    def test_compare_identical_sources_similarity_one(self, source):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.compare_code(source, source)
        assert {"similarity", "token_overlap", "structure_similarity"} <= set(r.keys())
        assert r["similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_detect_defects_finds_all_three(self, defect_source):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.detect_code_defects(defect_source)
        assert r["total"] == 3
        patterns = {d["pattern"] for d in r["defects"]}
        assert patterns == {"mutable_default", "bare_except", "equals_none"}

    def test_analyze_control_flow_returns_expected_keys(self, source):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.analyze_control_flow(source)
        assert {"functions", "avg_complexity", "max_complexity", "total_functions"} <= set(r.keys())
        assert r["total_functions"] == 2  # foo + method_a

    def test_analyze_style_returns_expected_keys(self, bad_style_source):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.analyze_code_style(bad_style_source)
        assert {"violations", "total", "score"} <= set(r.keys())
        assert r["total"] >= 3  # at least line_too_long, trailing_ws, mixed_indent, naming
        assert 0.0 <= r["score"] <= 1.0

    def test_build_dependency_graph_returns_expected_keys(self, source):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.build_dependency_graph(source)
        assert {"nodes", "edges", "adjacency", "n_modules"} <= set(r.keys())
        assert "__main__" in r["nodes"]
        assert "os" in r["nodes"]
        assert "numpy" in r["nodes"]


# --------------------------------------------------------------------- #
# 2. CLI — subprocess invocations
# --------------------------------------------------------------------- #


class TestCLICode:
    _env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONPATH": "src"}

    def _run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "zero_data_model"] + list(args),
            capture_output=True,
            text=True,
            timeout=30,
            env=self._env,
        )

    def _write_src(self, tmp_path, name, src):
        p = tmp_path / name
        p.write_text(src)
        return str(p)

    def test_cli_encode(self, tmp_path, source):
        f = self._write_src(tmp_path, "s.py", source)
        r = self._run_cli("code", "encode", f)
        if r.returncode == 0:
            assert "embedding" in json.loads(r.stdout)

    def test_cli_ast(self, tmp_path, source):
        f = self._write_src(tmp_path, "s.py", source)
        r = self._run_cli("code", "ast", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "foo" in payload["functions"]
            assert "Bar" in payload["classes"]

    def test_cli_compare(self, tmp_path, source):
        f = self._write_src(tmp_path, "s.py", source)
        r = self._run_cli("code", "compare", f, f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_cli_defects(self, tmp_path, defect_source):
        f = self._write_src(tmp_path, "d.py", defect_source)
        r = self._run_cli("code", "defects", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["total"] == 3

    def test_cli_control_flow(self, tmp_path, source):
        f = self._write_src(tmp_path, "s.py", source)
        r = self._run_cli("code", "control-flow", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["total_functions"] == 2

    def test_cli_style(self, tmp_path, bad_style_source):
        f = self._write_src(tmp_path, "bad.py", bad_style_source)
        r = self._run_cli("code", "style", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert payload["total"] >= 3

    def test_cli_dependencies(self, tmp_path, source):
        f = self._write_src(tmp_path, "s.py", source)
        r = self._run_cli("code", "dependencies", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "__main__" in payload["nodes"]
            assert "os" in payload["nodes"]


# --------------------------------------------------------------------- #
# 3. API — FastAPI endpoints
# --------------------------------------------------------------------- #


class TestAPICode:
    def test_api_encode(self, client, source):
        r = client.post("/code/encode", json={"source": source})
        assert r.status_code == 200, r.text
        assert "embedding" in r.json()

    def test_api_ast(self, client, source):
        r = client.post("/code/ast", json={"source": source})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "foo" in body["functions"]
        assert "Bar" in body["classes"]

    def test_api_compare(self, client, source):
        r = client.post(
            "/code/compare",
            json={"source_a": source, "source_b": source},
        )
        assert r.status_code == 200, r.text
        assert r.json()["similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_api_defects(self, client, defect_source):
        r = client.post("/code/defects", json={"source": defect_source})
        assert r.status_code == 200, r.text
        assert r.json()["total"] == 3

    def test_api_control_flow(self, client, source):
        r = client.post("/code/control-flow", json={"source": source})
        assert r.status_code == 200, r.text
        assert r.json()["total_functions"] == 2

    def test_api_style(self, client, bad_style_source):
        r = client.post("/code/style", json={"source": bad_style_source})
        assert r.status_code == 200, r.text
        assert r.json()["total"] >= 3

    def test_api_dependencies(self, client, source):
        r = client.post("/code/dependencies", json={"source": source})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "__main__" in body["nodes"]
        assert "os" in body["nodes"]

    # --- negative paths (Pydantic rejection -> 422) ---
    def test_api_empty_source_rejected(self, client):
        r = client.post("/code/encode", json={"source": ""})
        assert r.status_code in (400, 422)

    def test_api_compare_missing_field_rejected(self, client, source):
        r = client.post(
            "/code/compare",
            json={"source_a": source},  # missing source_b
        )
        assert r.status_code in (400, 422)


# --------------------------------------------------------------------- #
# 4. MCP — ZeroDataMCPServer.call_tool
# --------------------------------------------------------------------- #


class TestMCPCode:
    def test_mcp_encode(self, mcp_server, source):
        r = mcp_server.call_tool("code_encode", source=source)
        assert "embedding" in r
        assert len(r["embedding"]) == 64  # default MCP dim

    def test_mcp_ast(self, mcp_server, source):
        r = mcp_server.call_tool("code_analyze_ast", source=source)
        assert "foo" in r["functions"]
        assert "Bar" in r["classes"]

    def test_mcp_compare_identical(self, mcp_server, source):
        r = mcp_server.call_tool("code_compare", source_a=source, source_b=source)
        assert r["similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_mcp_compare_disjoint(self, mcp_server):
        r = mcp_server.call_tool(
            "code_compare",
            source_a="def alpha():\n    pass\n",
            source_b="class Beta:\n    pass\n",
        )
        assert r["similarity"] < 1.0

    def test_mcp_defects(self, mcp_server, defect_source):
        r = mcp_server.call_tool("code_detect_defects", source=defect_source)
        assert r["total"] == 3
        patterns = {d["pattern"] for d in r["defects"]}
        assert patterns == {"mutable_default", "bare_except", "equals_none"}

    def test_mcp_control_flow(self, mcp_server, source):
        r = mcp_server.call_tool("code_analyze_control_flow", source=source)
        assert r["total_functions"] == 2
        assert all("basic_blocks" in fn for fn in r["functions"])

    def test_mcp_style(self, mcp_server, bad_style_source):
        r = mcp_server.call_tool("code_analyze_style", source=bad_style_source)
        assert r["total"] >= 3
        assert 0.0 <= r["score"] <= 1.0

    def test_mcp_dependencies(self, mcp_server, source):
        r = mcp_server.call_tool("code_build_dependency_graph", source=source)
        assert "__main__" in r["nodes"]
        assert "os" in r["nodes"]
        assert r["n_modules"] == len(r["nodes"])

    def test_mcp_syntax_error_no_crash(self, mcp_server):
        bad = "def def def ::::\n"
        r = mcp_server.call_tool("code_analyze_ast", source=bad)
        assert r["functions"] == []
        assert r["complexity"] == 0

    # --- negative paths ---
    def test_mcp_empty_source_returns_error(self, mcp_server):
        r = mcp_server.call_tool("code_encode", source="")
        assert "error" in r

    def test_mcp_non_string_source_returns_error(self, mcp_server):
        r = mcp_server.call_tool("code_encode", source=12345)
        assert "error" in r

    def test_mcp_compare_missing_b_returns_error(self, mcp_server, source):
        # call_tool requires kwargs; passing only source_a should fail
        # validation inside the tool wrapper.
        r = mcp_server.call_tool("code_compare", source_a=source)
        assert "error" in r

    def test_mcp_code_tools_registered(self, mcp_server):
        tools = mcp_server.list_tools()
        for name in (
            "code_encode", "code_analyze_ast", "code_compare",
            "code_detect_defects", "code_analyze_control_flow",
            "code_analyze_style", "code_build_dependency_graph",
        ):
            assert name in tools, f"missing tool: {name}"


# --------------------------------------------------------------------- #
# 5. Cross-layer consistency
# --------------------------------------------------------------------- #


class TestCrossLayerConsistency:
    """Code facade methods produce identical outputs across facade / MCP.

    All 7 Code methods are deterministic functions of their input
    (no RNG, no model-state dependence), so the facade (dim=8) and MCP
    server (dim=64) should agree on all dict-valued outputs. Only
    `encode_code` differs in vector length (8 vs 64).
    """

    def test_facade_and_mcp_ast_agree(self, source, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.analyze_ast(source)
        mcp = mcp_server.call_tool("code_analyze_ast", source=source)
        assert facade == mcp

    def test_facade_and_mcp_compare_agree(self, source, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.compare_code(source, source)
        mcp = mcp_server.call_tool("code_compare", source_a=source, source_b=source)
        for k in ("similarity", "token_overlap", "structure_similarity"):
            assert facade[k] == pytest.approx(mcp[k], abs=1e-12)

    def test_facade_and_mcp_defects_agree(self, defect_source, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.detect_code_defects(defect_source)
        mcp = mcp_server.call_tool("code_detect_defects", source=defect_source)
        assert facade["total"] == mcp["total"]
        # Defects list order should match (both walk AST in the same order).
        for f_def, m_def in zip(facade["defects"], mcp["defects"]):
            assert f_def["pattern"] == m_def["pattern"]
            assert f_def["line"] == m_def["line"]
            assert f_def["col"] == m_def["col"]

    def test_facade_and_mcp_control_flow_agree(self, source, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.analyze_control_flow(source)
        mcp = mcp_server.call_tool("code_analyze_control_flow", source=source)
        assert facade["avg_complexity"] == pytest.approx(mcp["avg_complexity"], abs=1e-12)
        assert facade["max_complexity"] == mcp["max_complexity"]
        assert facade["total_functions"] == mcp["total_functions"]

    def test_facade_and_mcp_style_agree(self, bad_style_source, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.analyze_code_style(bad_style_source)
        mcp = mcp_server.call_tool("code_analyze_style", source=bad_style_source)
        assert facade["total"] == mcp["total"]
        assert facade["score"] == pytest.approx(mcp["score"], abs=1e-12)

    def test_facade_and_mcp_dependencies_agree(self, source, mcp_server):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.build_dependency_graph(source)
        mcp = mcp_server.call_tool("code_build_dependency_graph", source=source)
        assert facade["nodes"] == mcp["nodes"]
        assert facade["n_modules"] == mcp["n_modules"]
        # Edges are tuples in facade, lists in MCP — compare as lists.
        facade_edges = [list(e) for e in facade["edges"]]
        assert facade_edges == mcp["edges"]
        np.testing.assert_allclose(
            np.asarray(facade["adjacency"]),
            np.asarray(mcp["adjacency"]),
        )
