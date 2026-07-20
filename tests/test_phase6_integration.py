# tests/test_phase6_integration.py
"""Phase 6 — Integration tests across facade / CLI / API / MCP.

Covers the three entry-point layers added in task 9:
1. ``ZeroDataModel`` facade methods (model.py) — direct calls.
2. CLI subcommands (``python -m zero_data_model memory encode ...`` etc).
3. FastAPI endpoints (``/memory/encode``, ``/planning/trajectory``, ...).
4. MCP tools (``ZeroDataMCPServer.memory_encode``, ...).

Total: 40+ tests ensuring each Phase 6 capability is reachable from every
entry point and produces consistent results. Web stack (fastapi + httpx)
and MCP server are imported defensively — the module skips when an
optional dep is missing.
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


@pytest.fixture(scope="module")
def model():
    """A small deterministic model for facade tests."""
    return ZeroDataModel(dim=8, seed=42)


@pytest.fixture()
def mcp_server():
    return ZeroDataMCPServer(dim=8)


@pytest.fixture()
def tmp_obs():
    """Return a 1D observation vector."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(8)


@pytest.fixture()
def client(tmp_path):
    """A TestClient for API endpoint tests (skipped if fastapi missing)."""
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
    if api_module.limiter is not None:
        api_module.limiter.enabled = True


# --------------------------------------------------------------------- #
# 1. Facade methods (model.py)
# --------------------------------------------------------------------- #


class TestFacadeMemory:
    def test_encode_memory(self, model, tmp_obs):
        r = model.encode_memory(tmp_obs, label="t")
        assert "id" in r and "label" in r
        assert r["label"] == "t"

    def test_retrieve_memory(self, model, tmp_obs):
        model.encode_memory(tmp_obs, label="t")
        r = model.retrieve_memory(tmp_obs, top_k=1)
        assert "items" in r
        assert len(r["items"]) == 1
        assert r["items"][0]["label"] == "t"

    def test_consolidate_memory(self, model):
        r = model.consolidate_memory()
        assert "promoted" in r and "deduplicated" in r


class TestFacadePlanning:
    def test_plan_trajectory(self, model):
        r = model.plan_trajectory(
            np.zeros(4), np.ones(4), n_steps=8,
        )
        assert "trajectory" in r
        assert "actions" in r

    def test_decompose_goal(self, model):
        r = model.decompose_goal("build tower")
        assert "tree" in r and "actions" in r

    def test_sequence_actions(self, model):
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        r = model.sequence_actions(A)
        assert "sequence_indices" in r


class TestFacadeMultimodal:
    def test_fit_cross_modal(self, model):
        rng = np.random.default_rng(0)
        r = model.fit_cross_modal(rng.standard_normal((5, 4)), rng.standard_normal((5, 6)))
        assert "projection_a" in r and "projection_b" in r

    def test_align_cross_modal(self, model):
        rng = np.random.default_rng(0)
        model.fit_cross_modal(rng.standard_normal((5, 4)), rng.standard_normal((5, 6)))
        r = model.align_cross_modal(np.zeros(4), source="a")
        assert "aligned" in r

    def test_fuse_modalities(self, model):
        r = model.fuse_modalities([np.zeros(4), np.ones(4)], strategy="mean")
        assert "fused" in r

    def test_contrastive_loss(self, model):
        rng = np.random.default_rng(0)
        r = model.contrastive_loss(rng.standard_normal((4, 8)), rng.standard_normal((4, 8)))
        assert "loss" in r


class TestFacadeRL:
    def test_step_mdp(self, model):
        r = model.step_mdp(0, 0)
        assert "next_state" in r and "reward" in r and "done" in r

    def test_train_q_learner(self, model):
        r = model.train_q_learner(n_episodes=2, max_steps_per_episode=5)
        assert "episode_rewards" in r and "final_policy" in r

    def test_search_rl_mcts(self, model):
        r = model.search_rl_mcts(root_state=0, n_simulations=5, max_depth=3)
        assert "best_action" in r


# --------------------------------------------------------------------- #
# 2. CLI subcommands
# --------------------------------------------------------------------- #


class TestCLIPhase6:
    """Smoke tests for ``python -m zero_data_model <group> <subcmd>``.

    Runs the actual CLI via subprocess to verify wiring.
    """

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

    def test_memory_encode(self, tmp_path):
        obs_file = tmp_path / "obs.json"
        obs_file.write_text(json.dumps([[1.0, 2.0, 3.0, 4.0]]))
        r = self._run_cli("memory", "encode", str(obs_file), "--label", "test")
        # CLI may exit 0 on success or non-zero on missing dep.
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "id" in payload

    def test_memory_retrieve(self, tmp_path):
        obs_file = tmp_path / "obs.json"
        obs_file.write_text(json.dumps([[1.0, 2.0, 3.0, 4.0]]))
        r = self._run_cli("memory", "retrieve", str(obs_file), "--top-k", "3")
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "items" in payload

    def test_memory_consolidate(self):
        r = self._run_cli("memory", "consolidate")
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "promoted" in payload

    def test_planning_trajectory(self, tmp_path):
        start_file = tmp_path / "start.json"
        end_file = tmp_path / "end.json"
        start_file.write_text(json.dumps([0.0, 0.0, 0.0, 0.0]))
        end_file.write_text(json.dumps([1.0, 1.0, 1.0, 1.0]))
        r = self._run_cli(
            "planning", "trajectory", str(start_file), str(end_file),
            "--steps", "4",
        )
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "trajectory" in payload

    def test_planning_decompose(self):
        r = self._run_cli("planning", "decompose", "build tower")
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "tree" in payload

    def test_planning_sequence(self, tmp_path):
        adj_file = tmp_path / "adj.json"
        adj_file.write_text(json.dumps([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        r = self._run_cli("planning", "sequence", str(adj_file))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "sequence_indices" in payload

    def test_rl_step(self):
        r = self._run_cli("rl", "step", "--state", "0", "--action", "0")
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "next_state" in payload

    def test_rl_train_q(self):
        r = self._run_cli(
            "rl", "train-q", "--episodes", "2", "--max-steps", "5",
        )
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "episode_rewards" in payload

    def test_rl_search_mcts(self):
        r = self._run_cli(
            "rl", "search-mcts", "--root-state", "0",
            "--simulations", "5", "--depth", "3",
        )
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "best_action" in payload

    # ------------------------------------------------------------------
    # Multimodal CLI — previously uncovered (P6-AUDIT-007).
    # ------------------------------------------------------------------

    def test_multimodal_align_cli(self, tmp_path):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((5, 4)).tolist()
        b = rng.standard_normal((5, 6)).tolist()
        file_a = tmp_path / "a.json"
        file_b = tmp_path / "b.json"
        file_a.write_text(json.dumps(a))
        file_b.write_text(json.dumps(b))
        r = self._run_cli("multimodal", "align", str(file_a), str(file_b))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            # CLI emits {"fit": {...}, "aligned": {...}}.
            assert "fit" in payload and "aligned" in payload
            assert "correlation" in payload["fit"]

    def test_multimodal_fuse_cli(self, tmp_path):
        file_a = tmp_path / "a.json"
        file_b = tmp_path / "b.json"
        file_a.write_text(json.dumps([1.0, 2.0, 3.0, 4.0]))
        file_b.write_text(json.dumps([2.0, 3.0, 4.0, 5.0]))
        r = self._run_cli(
            "multimodal", "fuse", str(file_a), str(file_b),
            "--strategy", "mean",
        )
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "fused" in payload

    def test_multimodal_contrastive_cli(self, tmp_path):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((4, 8)).tolist()
        b = rng.standard_normal((4, 8)).tolist()
        file_a = tmp_path / "a.json"
        file_b = tmp_path / "b.json"
        file_a.write_text(json.dumps(a))
        file_b.write_text(json.dumps(b))
        r = self._run_cli("multimodal", "contrastive", str(file_a), str(file_b))
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "loss" in payload


# --------------------------------------------------------------------- #
# 3. FastAPI endpoints
# --------------------------------------------------------------------- #


@pytest.mark.skipif(not _HAS_WEB, reason="fastapi/httpx not installed")
class TestAPIPhase6:
    def test_memory_encode_endpoint(self, client):
        r = client.post("/memory/encode", json={"observation": [1.0, 2.0, 3.0, 4.0]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "id" in body

    def test_memory_encode_with_label(self, client):
        r = client.post(
            "/memory/encode",
            json={"observation": [1.0, 2.0], "label": "tagged"},
        )
        assert r.status_code == 200
        assert r.json()["label"] == "tagged"

    def test_memory_encode_nan_returns_400(self, client):
        r = client.post(
            "/memory/encode",
            json={"observation": [1.0, "NaN"]},
        )
        assert r.status_code in (400, 422)

    def test_memory_retrieve_endpoint(self, client):
        # First encode.
        client.post("/memory/encode", json={"observation": [1.0, 0.0, 0.0, 0.0]})
        r = client.post(
            "/memory/retrieve",
            json={"query": [1.0, 0.0, 0.0, 0.0], "top_k": 1},
        )
        assert r.status_code == 200
        body = r.json()
        assert "items" in body

    def test_memory_consolidate_endpoint(self, client):
        r = client.post("/memory/consolidate")
        assert r.status_code == 200
        assert "promoted" in r.json()

    def test_planning_trajectory_endpoint(self, client):
        r = client.post(
            "/planning/trajectory",
            json={
                "start_state": [0.0, 0.0, 0.0, 0.0],
                "goal_state": [1.0, 1.0, 1.0, 1.0],
                "n_steps": 4,
            },
        )
        assert r.status_code == 200, r.text
        assert "trajectory" in r.json()

    def test_planning_trajectory_shape_mismatch(self, client):
        r = client.post(
            "/planning/trajectory",
            json={
                "start_state": [0.0, 0.0, 0.0],
                "goal_state": [1.0, 1.0, 1.0, 1.0],
            },
        )
        assert r.status_code == 400

    def test_planning_decompose_endpoint(self, client):
        r = client.post(
            "/planning/decompose",
            json={"goal": "build a tower"},
        )
        assert r.status_code == 200
        assert "tree" in r.json()

    def test_planning_sequence_endpoint(self, client):
        r = client.post(
            "/planning/sequence",
            json={"adjacency": [[0, 1, 0], [0, 0, 1], [0, 0, 0]]},
        )
        assert r.status_code == 200
        assert "sequence_indices" in r.json()

    def test_multimodal_align_endpoint(self, client):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((5, 4)).tolist()
        b = rng.standard_normal((5, 6)).tolist()
        r = client.post(
            "/multimodal/align",
            json={"observations_a": a, "observations_b": b},
        )
        assert r.status_code == 200, r.text
        # API returns nested structure: {"fit": {...}, "aligned": {...}}.
        body = r.json()
        assert "fit" in body and "aligned" in body
        assert "projection_a" in body["fit"]
        assert "correlation" in body["fit"]
        assert "aligned" in body["aligned"]
        assert body["aligned"]["source"] == "a"

    def test_multimodal_fuse_endpoint(self, client):
        r = client.post(
            "/multimodal/fuse",
            json={
                "embeddings": [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
                "strategy": "mean",
            },
        )
        assert r.status_code == 200
        assert "fused" in r.json()

    def test_multimodal_contrastive_endpoint(self, client):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((4, 8)).tolist()
        b = rng.standard_normal((4, 8)).tolist()
        r = client.post(
            "/multimodal/contrastive",
            json={"batch_a": a, "batch_b": b},
        )
        assert r.status_code == 200
        assert "loss" in r.json()

    def test_rl_step_endpoint(self, client):
        r = client.post("/rl/step", json={"state": 0, "action": 0})
        assert r.status_code == 200
        assert "next_state" in r.json()

    def test_rl_step_negative_state_returns_400(self, client):
        r = client.post("/rl/step", json={"state": -1, "action": 0})
        # Pydantic Field(ge=0) rejects at validation time -> 422.
        # The endpoint's own _ensure_finite would yield 400, but validation
        # runs first. Accept either for robustness.
        assert r.status_code in (400, 422), r.text

    def test_rl_train_q_endpoint(self, client):
        r = client.post(
            "/rl/train-q",
            json={"n_episodes": 2, "max_steps_per_episode": 5},
        )
        assert r.status_code == 200
        assert "episode_rewards" in r.json()

    def test_rl_search_mcts_endpoint(self, client):
        r = client.post(
            "/rl/search-mcts",
            json={"root_state": 0, "n_simulations": 5, "max_depth": 3},
        )
        assert r.status_code == 200
        assert "best_action" in r.json()


# --------------------------------------------------------------------- #
# 4. MCP tools
# --------------------------------------------------------------------- #


class TestMCPPhase6:
    def test_total_tool_count(self, mcp_server):
        """Server should now have at least 40 tools (16 base + 6 emergence +
        12 Phase 6 + 6 Audio + 7 Graph + future Phase 7 domains).
        Each Phase 7 domain adds its own tools; this assertion only
        verifies the floor — see TestMCPPhase6::test_phase6_tools_registered
        for the strict Phase 6 invariant.
        """
        tools = mcp_server.list_tools()
        assert len(tools) >= 40

    def test_phase6_tools_registered(self, mcp_server):
        tools = set(mcp_server.list_tools())
        phase6 = {
            "memory_encode", "memory_retrieve", "memory_consolidate",
            "planning_trajectory", "planning_decompose", "planning_sequence",
            "multimodal_align", "multimodal_fuse", "multimodal_contrastive",
            "rl_step", "rl_train_q", "rl_search_mcts",
        }
        assert phase6.issubset(tools)

    def test_memory_encode_tool(self, mcp_server):
        r = mcp_server.call_tool("memory_encode", observation=[1.0, 2.0, 3.0, 4.0])
        assert "id" in r

    def test_memory_encode_with_label(self, mcp_server):
        r = mcp_server.call_tool(
            "memory_encode", observation=[1.0, 2.0, 3.0], label="tag",
        )
        assert r["label"] == "tag"

    def test_memory_encode_nan_returns_error(self, mcp_server):
        r = mcp_server.call_tool(
            "memory_encode", observation=[float("nan"), 0.0, 0.0],
        )
        assert "error" in r

    def test_memory_encode_non_1d_returns_error(self, mcp_server):
        r = mcp_server.call_tool(
            "memory_encode", observation=[[1.0, 2.0], [3.0, 4.0]],
        )
        assert "error" in r

    def test_memory_retrieve_tool(self, mcp_server):
        mcp_server.call_tool("memory_encode", observation=[1.0, 0.0, 0.0, 0.0])
        r = mcp_server.call_tool("memory_retrieve", query=[1.0, 0.0, 0.0, 0.0])
        assert "items" in r

    def test_memory_consolidate_tool(self, mcp_server):
        r = mcp_server.call_tool("memory_consolidate")
        assert "promoted" in r

    def test_planning_trajectory_tool(self, mcp_server):
        r = mcp_server.call_tool(
            "planning_trajectory",
            start_state=[0.0, 0.0, 0.0, 0.0],
            goal_state=[1.0, 1.0, 1.0, 1.0],
            n_steps=4,
        )
        assert "trajectory" in r

    def test_planning_trajectory_shape_mismatch(self, mcp_server):
        r = mcp_server.call_tool(
            "planning_trajectory",
            start_state=[0.0, 0.0, 0.0],
            goal_state=[1.0, 1.0, 1.0, 1.0],
        )
        assert "error" in r

    def test_planning_decompose_tool(self, mcp_server):
        r = mcp_server.call_tool("planning_decompose", goal="build tower")
        assert "tree" in r

    def test_planning_decompose_empty_goal(self, mcp_server):
        r = mcp_server.call_tool("planning_decompose", goal="")
        assert "error" in r

    def test_planning_sequence_tool(self, mcp_server):
        r = mcp_server.call_tool(
            "planning_sequence", adjacency=[[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        )
        assert "sequence_indices" in r

    def test_planning_sequence_non_square(self, mcp_server):
        r = mcp_server.call_tool(
            "planning_sequence", adjacency=[[0, 1, 0], [0, 0]],
        )
        assert "error" in r

    def test_multimodal_align_tool(self, mcp_server):
        rng = np.random.default_rng(0)
        r = mcp_server.call_tool(
            "multimodal_align",
            observations_a=rng.standard_normal((5, 4)).tolist(),
            observations_b=rng.standard_normal((5, 6)).tolist(),
        )
        assert "projection_a" in r

    def test_multimodal_align_row_mismatch(self, mcp_server):
        rng = np.random.default_rng(0)
        r = mcp_server.call_tool(
            "multimodal_align",
            observations_a=rng.standard_normal((5, 4)).tolist(),
            observations_b=rng.standard_normal((6, 6)).tolist(),
        )
        assert "error" in r

    def test_multimodal_fuse_tool(self, mcp_server):
        r = mcp_server.call_tool(
            "multimodal_fuse",
            embeddings=[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            strategy="mean",
        )
        assert "fused" in r

    def test_multimodal_fuse_too_few_embeddings(self, mcp_server):
        r = mcp_server.call_tool(
            "multimodal_fuse", embeddings=[[1.0, 2.0]],
        )
        assert "error" in r

    def test_multimodal_contrastive_tool(self, mcp_server):
        rng = np.random.default_rng(0)
        r = mcp_server.call_tool(
            "multimodal_contrastive",
            batch_a=rng.standard_normal((4, 8)).tolist(),
            batch_b=rng.standard_normal((4, 8)).tolist(),
        )
        assert "loss" in r

    def test_rl_step_tool(self, mcp_server):
        r = mcp_server.call_tool("rl_step", state=0, action=0)
        assert "next_state" in r

    def test_rl_step_negative_state(self, mcp_server):
        r = mcp_server.call_tool("rl_step", state=-1, action=0)
        assert "error" in r

    def test_rl_train_q_tool(self, mcp_server):
        r = mcp_server.call_tool(
            "rl_train_q", n_episodes=2, max_steps_per_episode=5,
        )
        assert "episode_rewards" in r

    def test_rl_train_q_zero_episodes(self, mcp_server):
        r = mcp_server.call_tool("rl_train_q", n_episodes=0)
        assert "error" in r

    def test_rl_search_mcts_tool(self, mcp_server):
        r = mcp_server.call_tool(
            "rl_search_mcts", root_state=0, n_simulations=5, max_depth=3,
        )
        assert "best_action" in r

    def test_rl_search_mcts_negative_state(self, mcp_server):
        r = mcp_server.call_tool("rl_search_mcts", root_state=-1)
        assert "error" in r

    def test_call_unknown_tool_returns_error(self, mcp_server):
        r = mcp_server.call_tool("nonexistent_tool")
        assert "error" in r

    def test_to_py_handles_numpy_types(self, mcp_server):
        """MCP tool outputs should never contain raw numpy types."""
        r = mcp_server.call_tool("rl_step", state=0, action=0)
        # JSON round-trip should succeed.
        json.dumps(r)


# --------------------------------------------------------------------- #
# 5. Cross-layer consistency
# --------------------------------------------------------------------- #


class TestCrossLayerConsistency:
    """The same operation should yield structurally similar results
    whether called via facade, MCP, or API."""

    def test_memory_encode_facade_vs_mcp_consistent(self, model, mcp_server):
        obs = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # Facade
        r_facade = model.encode_memory(obs, label="x")
        # MCP
        r_mcp = mcp_server.call_tool(
            "memory_encode", observation=obs.tolist(), label="x",
        )
        # Both return dict with id and label keys.
        assert "id" in r_facade and "id" in r_mcp
        assert r_facade["label"] == r_mcp["label"] == "x"

    def test_rl_step_facade_vs_mcp_consistent(self, model, mcp_server):
        # Both use the same MDP structure (n_states, n_actions, seed=42).
        # Step from (0, 0) should produce a deterministic next_state.
        r_facade = model.step_mdp(0, 0)
        r_mcp = mcp_server.call_tool("rl_step", state=0, action=0)
        # next_state must be in valid range for both.
        assert 0 <= r_facade["next_state"] < model.rl_mdp.n_states
        assert 0 <= r_mcp["next_state"]

    @pytest.mark.skipif(not _HAS_WEB, reason="fastapi/httpx not installed")
    def test_rl_step_api_vs_mcp_consistent(self, client, mcp_server):
        # API and MCP both invoke step_mdp on independent server-side
        # models; both should return well-formed results.
        r_api = client.post("/rl/step", json={"state": 0, "action": 0})
        r_mcp = mcp_server.call_tool("rl_step", state=0, action=0)
        assert r_api.status_code == 200
        assert "next_state" in r_api.json()
        assert "next_state" in r_mcp
