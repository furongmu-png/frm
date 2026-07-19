# tests/test_mcp_emergence.py
"""Tests for the causal-emergence MCP tools on ``ZeroDataMCPServer``.

Covers all six emergence tools (perceive_topology, discover_causal_dynamics,
generate_trajectory, sample_posterior, recall_memory, emergence_cycle)
through the plain-callable ``call_tool`` API, which works whether or not
the optional ``mcp`` package is installed (graceful degradation).
"""
from __future__ import annotations

import json

import numpy as np

from zero_data_model.mcp_server import ZeroDataMCPServer


def _make_server() -> ZeroDataMCPServer:
    """Fresh server with a small dim to keep tests fast."""
    return ZeroDataMCPServer(dim=8)


def _circle_points(n: int = 16) -> list[list[float]]:
    """n points uniformly on the unit circle (has β0=1, β1=1)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)]).tolist()


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def test_emergence_tools_registered():
    """All six emergence tools must appear in the tool registry."""
    server = _make_server()
    for name in (
        "perceive_topology",
        "discover_causal_dynamics",
        "generate_trajectory",
        "sample_posterior",
        "recall_memory",
        "emergence_cycle",
    ):
        assert name in server.tools, f"{name} not registered"
        assert callable(server.tools[name])
    # Backward compat: the 16 legacy tools must still be present.
    assert len(server.list_tools()) >= 22


def test_emergence_tools_docstrings_preserved():
    """Each registered tool keeps its docstring (used as MCP description)."""
    server = _make_server()
    for name in (
        "perceive_topology",
        "discover_causal_dynamics",
        "generate_trajectory",
        "sample_posterior",
        "recall_memory",
        "emergence_cycle",
    ):
        assert server.tools[name].__doc__, f"{name} lost its docstring"


# ----------------------------------------------------------------------
# perceive_topology
# ----------------------------------------------------------------------


def test_perceive_topology_success():
    """Four-cluster corners give β0=4 (4 components, no loops)."""
    server = _make_server()
    result = server.call_tool(
        "perceive_topology",
        data=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    assert isinstance(result, dict)
    assert "error" not in result
    assert set(result.keys()) == {
        "betti_numbers",
        "persistence_entropy",
        "euler_characteristic",
        "n_points",
        "max_eps",
    }
    assert isinstance(result["betti_numbers"], list)
    assert len(result["betti_numbers"]) == 3
    assert all(isinstance(b, int) for b in result["betti_numbers"])
    assert result["betti_numbers"][0] == 4  # four separate components
    assert isinstance(result["persistence_entropy"], float)
    assert isinstance(result["euler_characteristic"], int)
    assert isinstance(result["n_points"], int)
    assert result["n_points"] == 4
    assert isinstance(result["max_eps"], float)


def test_perceive_topology_circle_has_loop():
    """A unit circle must have β0=1 and β1>=1 (one loop)."""
    server = _make_server()
    result = server.call_tool("perceive_topology", data=_circle_points(16))
    assert "error" not in result
    assert result["betti_numbers"][0] == 1
    assert result["betti_numbers"][1] >= 1
    assert result["betti_numbers"][2] == 0


def test_perceive_topology_invalid_1d_returns_error():
    """1D input must surface an error dict, not raise."""
    server = _make_server()
    result = server.call_tool("perceive_topology", data=[1.0, 2.0, 3.0])
    assert "error" in result
    assert "perceive_topology" in result["error"]


def test_perceive_topology_invalid_small_returns_error():
    """A 1x1 array is too small — must surface an error dict."""
    server = _make_server()
    result = server.call_tool("perceive_topology", data=[[1.0]])
    assert "error" in result


# ----------------------------------------------------------------------
# discover_causal_dynamics
# ----------------------------------------------------------------------


def test_discover_causal_dynamics_success():
    """A simple correlated dataset yields a square adjacency matrix."""
    server = _make_server()
    # x0 -> x1 (perfect correlation), x2 independent.
    data = [
        [float(i), 2.0 * float(i) + 1.0, float(i) % 2]
        for i in range(10)
    ]
    result = server.call_tool(
        "discover_causal_dynamics",
        data=data,
        var_names=["a", "b", "c"],
    )
    assert isinstance(result, dict)
    assert "error" not in result
    assert set(result.keys()) == {
        "adjacency",
        "edges",
        "n_edges",
        "is_acyclic",
        "method",
        "var_names",
    }
    adj = result["adjacency"]
    assert isinstance(adj, list)
    assert len(adj) == 3
    assert all(isinstance(row, list) and len(row) == 3 for row in adj)
    assert all(isinstance(v, int) for row in adj for v in row)
    assert isinstance(result["n_edges"], int)
    assert isinstance(result["is_acyclic"], bool)
    assert result["var_names"] == ["a", "b", "c"]
    assert isinstance(result["method"], str)


def test_discover_causal_dynamics_invalid_returns_error():
    """1D input must surface an error dict."""
    server = _make_server()
    result = server.call_tool("discover_causal_dynamics", data=[1.0, 2.0])
    assert "error" in result
    assert "discover_causal_dynamics" in result["error"]


# ----------------------------------------------------------------------
# generate_trajectory
# ----------------------------------------------------------------------


def test_generate_trajectory_success_shape():
    """Trajectory must have ``n_steps + 1`` rows and the right dim."""
    server = _make_server()
    result = server.call_tool(
        "generate_trajectory",
        start_state=[0.0, 0.0, 0.0],
        end_state=[1.0, 2.0, 3.0],
        n_steps=8,
    )
    assert isinstance(result, dict)
    assert "error" not in result
    assert set(result.keys()) == {
        "trajectory",
        "lagrangian",
        "action",
        "converged",
        "iterations",
    }
    traj = result["trajectory"]
    assert isinstance(traj, list)
    assert len(traj) == 9  # n_steps + 1
    assert all(isinstance(row, list) and len(row) == 3 for row in traj)
    # Boundary conditions: first row = start, last row = end.
    assert traj[0] == [0.0, 0.0, 0.0]
    assert traj[-1] == [1.0, 2.0, 3.0]
    assert isinstance(result["lagrangian"], list)
    assert len(result["lagrangian"]) == 8
    assert isinstance(result["action"], float)
    assert isinstance(result["converged"], bool)
    assert isinstance(result["iterations"], int)


def test_generate_trajectory_with_obstacles_has_violations_key():
    """When obstacles are provided, ``obstacle_violations`` is included."""
    server = _make_server()
    result = server.call_tool(
        "generate_trajectory",
        start_state=[0.0, 0.0],
        end_state=[1.0, 1.0],
        n_steps=8,
        obstacles=[[0.5, 0.5]],
        margin=0.1,
    )
    assert "error" not in result
    assert "obstacle_violations" in result
    assert isinstance(result["obstacle_violations"], int)


def test_generate_trajectory_dim_mismatch_returns_error():
    """Mismatched start/end dims must surface an error dict."""
    server = _make_server()
    result = server.call_tool(
        "generate_trajectory",
        start_state=[0.0, 0.0],
        end_state=[1.0, 2.0, 3.0],
        n_steps=8,
    )
    assert "error" in result
    assert "generate_trajectory" in result["error"]


# ----------------------------------------------------------------------
# sample_posterior
# ----------------------------------------------------------------------


def test_sample_posterior_success():
    """Gaussian posterior sampling returns samples and stats."""
    server = _make_server()
    result = server.call_tool(
        "sample_posterior",
        mean=[0.0, 0.5, 1.0],
        std=1.0,
        n_samples=20,
    )
    assert isinstance(result, dict)
    assert "error" not in result
    assert set(result.keys()) == {
        "samples",
        "mean",
        "std",
        "accept_rate",
        "ess",
        "converged",
    }
    samples = result["samples"]
    assert isinstance(samples, list)
    assert len(samples) == 20
    assert all(isinstance(row, list) and len(row) == 3 for row in samples)
    assert isinstance(result["mean"], list)
    assert len(result["mean"]) == 3
    assert isinstance(result["std"], list)
    assert len(result["std"]) == 3
    assert isinstance(result["accept_rate"], float)
    assert 0.0 <= result["accept_rate"] <= 1.0
    assert isinstance(result["ess"], float)
    assert isinstance(result["converged"], bool)


def test_sample_posterior_n_samples_invalid_returns_error():
    """n_samples < 1 must surface an error dict."""
    server = _make_server()
    result = server.call_tool(
        "sample_posterior", mean=[0.0, 0.0], std=1.0, n_samples=0
    )
    assert "error" in result
    assert "sample_posterior" in result["error"]


def test_sample_posterior_std_invalid_returns_error():
    """std <= 0 must surface an error dict."""
    server = _make_server()
    result = server.call_tool(
        "sample_posterior", mean=[0.0, 0.0], std=0.0, n_samples=10
    )
    assert "error" in result


# ----------------------------------------------------------------------
# recall_memory
# ----------------------------------------------------------------------


def test_recall_memory_empty_returns_none_label():
    """Empty memory must return label=None, emerged=True (spec §7.3)."""
    server = _make_server()
    result = server.call_tool(
        "recall_memory", query=[0.1, 0.2, 0.3, 0.4], n_steps=20
    )
    assert isinstance(result, dict)
    assert "error" not in result
    assert set(result.keys()) == {
        "label",
        "similarity",
        "emerged",
        "trajectory",
        "converged",
        "divergence",
        "nearest_pattern",
    }
    assert result["label"] is None
    assert result["emerged"] is True
    assert isinstance(result["similarity"], float)
    assert isinstance(result["divergence"], float)
    assert isinstance(result["converged"], bool)
    assert result["nearest_pattern"] is None
    # Trajectory shape (n_steps + 1, 3).
    assert isinstance(result["trajectory"], list)
    assert len(result["trajectory"]) == 21
    assert all(isinstance(row, list) and len(row) == 3 for row in result["trajectory"])


def test_recall_memory_invalid_query_returns_error():
    """2D query must surface an error dict."""
    server = _make_server()
    result = server.call_tool(
        "recall_memory", query=[[0.1, 0.2], [0.3, 0.4]], n_steps=20
    )
    assert "error" in result
    assert "recall_memory" in result["error"]


# ----------------------------------------------------------------------
# emergence_cycle
# ----------------------------------------------------------------------


def test_emergence_cycle_success():
    """A 4x2 observation must produce the full emergence-cycle output."""
    server = _make_server()
    observation = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
    ]
    result = server.call_tool("emergence_cycle", observation=observation)
    assert isinstance(result, dict)
    assert "error" not in result
    assert set(result.keys()) == {
        "perception",
        "causal_graph",
        "counterfactual",
        "posterior",
        "memory",
        "emergence_score",
        "warnings",
    }
    assert isinstance(result["emergence_score"], float)
    assert 0.0 <= result["emergence_score"] <= 1.0
    assert isinstance(result["warnings"], list)
    # Sub-dicts are non-empty (engine populates them).
    assert isinstance(result["perception"], dict)
    assert isinstance(result["causal_graph"], dict)
    assert isinstance(result["counterfactual"], dict)
    assert isinstance(result["posterior"], dict)
    assert isinstance(result["memory"], dict)


def test_emergence_cycle_invalid_returns_error():
    """1D observation must surface an error dict."""
    server = _make_server()
    result = server.call_tool(
        "emergence_cycle", observation=[1.0, 2.0, 3.0, 4.0]
    )
    assert "error" in result
    assert "emergence_cycle" in result["error"]


# ----------------------------------------------------------------------
# JSON serialization (no numpy leakage)
# ----------------------------------------------------------------------


def test_emergence_tool_outputs_are_json_serializable():
    """Every emergence tool result must round-trip through ``json.dumps``."""
    server = _make_server()
    cases = [
        ("perceive_topology", {"data": _circle_points(8)}),
        ("discover_causal_dynamics", {"data": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]}),
        (
            "generate_trajectory",
            {"start_state": [0.0, 0.0], "end_state": [1.0, 1.0], "n_steps": 5},
        ),
        ("sample_posterior", {"mean": [0.0, 0.0], "std": 1.0, "n_samples": 10}),
        ("recall_memory", {"query": [0.1, 0.2], "n_steps": 10}),
        ("emergence_cycle", {"observation": [[1.0, 2.0], [2.0, 3.0]]}),
    ]
    for name, kwargs in cases:
        result = server.call_tool(name, **kwargs)
        assert isinstance(result, dict), f"{name} did not return a dict"
        assert "error" not in result, f"{name} unexpectedly failed: {result}"
        # Must not raise: confirms no numpy scalars/arrays leaked through.
        json.dumps(result)


def test_emergence_tool_failure_surfaces_error_dict():
    """A tool whose underlying model raises must surface an error dict."""
    server = _make_server()
    original = server.model.perceive_topology

    def _boom(_data, max_dim=None):
        raise RuntimeError("forced failure")

    server.model.perceive_topology = _boom  # type: ignore[method-assign]
    try:
        result = server.call_tool(
            "perceive_topology", data=[[0.0, 0.0], [1.0, 1.0]]
        )
    finally:
        server.model.perceive_topology = original  # type: ignore[method-assign]
    assert "error" in result
    assert "forced failure" in result["error"]
