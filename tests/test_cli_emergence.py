# tests/test_cli_emergence.py
"""Tests for the ``emergence`` CLI subcommand group (spec §2.3).

Covers the four ``emergence`` subcommands (perceive / causal / trajectory /
cycle) via direct ``main()`` invocation with the ``capsys`` fixture, plus a
subprocess smoke test that exercises the real ``python -m zero_data_model``
entry point. Regression tests verify that ``--version`` and the default
demo behavior are preserved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from zero_data_model import __version__
from zero_data_model.__main__ import main

# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------

def _make_observation(n_samples: int = 50, n_features: int = 4, seed: int = 0):
    """Generate a 2D observation matrix with mild causal structure."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n_samples)
    x1 = 0.7 * x0 + 0.3 * rng.standard_normal(n_samples)
    x2 = 0.5 * x1 + 0.4 * rng.standard_normal(n_samples)
    x3 = 0.2 * x2 + 0.6 * rng.standard_normal(n_samples)
    return np.column_stack([x0, x1, x2, x3])[:, :n_features]


def _save_npz(path: Path, arr: np.ndarray) -> Path:
    """Save an array to a .npz archive under key ``data``."""
    np.savez(path, data=arr)
    return path


def _save_csv(path: Path, arr: np.ndarray) -> Path:
    """Save a 2D array to a comma-separated CSV file."""
    np.savetxt(path, arr, delimiter=",")
    return path


def _save_npy(path: Path, arr: np.ndarray) -> Path:
    """Save an array to a single-array .npy file."""
    np.save(path, arr)
    return path


# ----------------------------------------------------------------------
# Regression: --version and default demo still work
# ----------------------------------------------------------------------

def test_version_flag_still_works(capsys):
    """``--version`` prints the package version and exits 0."""
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert __version__ in out
    # The demo banner should NOT be printed when --version is set.
    assert "Mini-demo" not in out


def test_version_subprocess_matches_main(capsys):
    """``python -m zero_data_model --version`` matches the in-process version."""
    proc = subprocess.run(
        [sys.executable, "-m", "zero_data_model", "--version"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0
    assert __version__ in proc.stdout


# ----------------------------------------------------------------------
# emergence perceive
# ----------------------------------------------------------------------

def test_emergence_perceive_success(tmp_path, capsys):
    """``perceive`` on a valid .npz file emits parseable JSON with all keys."""
    obs = _make_observation(n_samples=30, n_features=4, seed=1)
    path = _save_npz(tmp_path / "obs.npz", obs)

    code = main(["emergence", "perceive", str(path)])
    out = capsys.readouterr().out

    assert code == 0
    result = json.loads(out)
    for key in (
        "betti_numbers",
        "persistence_entropy",
        "euler_characteristic",
        "n_points",
        "max_eps",
    ):
        assert key in result, f"missing key: {key}"
    assert isinstance(result["betti_numbers"], list)
    assert result["n_points"] >= 1
    # persistence_diagram is omitted unless --full-diagram is set.
    assert "persistence_diagram" not in result


def test_emergence_perceive_full_diagram_flag(tmp_path, capsys):
    """``--full-diagram`` includes the persistence_diagram field."""
    obs = _make_observation(n_samples=30, n_features=4, seed=1)
    path = _save_npz(tmp_path / "obs.npz", obs)

    code = main(["emergence", "perceive", str(path), "--full-diagram"])
    out = capsys.readouterr().out

    assert code == 0
    result = json.loads(out)
    assert "persistence_diagram" in result


def test_emergence_perceive_missing_file(capsys):
    """``perceive`` on a nonexistent file exits 1 with a stderr message."""
    code = main(["emergence", "perceive", "/nonexistent/path/obs.npz"])
    err = capsys.readouterr().err

    assert code == 1
    assert err  # non-empty stderr
    assert "not found" in err.lower() or "no such file" in err.lower()


# ----------------------------------------------------------------------
# emergence causal
# ----------------------------------------------------------------------

def test_emergence_causal_success(tmp_path, capsys):
    """``causal`` on a valid .csv file emits parseable JSON with all keys."""
    obs = _make_observation(n_samples=80, n_features=4, seed=2)
    path = _save_csv(tmp_path / "obs.csv", obs)

    code = main(["emergence", "causal", str(path)])
    out = capsys.readouterr().out

    assert code == 0
    result = json.loads(out)
    for key in ("adjacency", "edges", "n_edges", "is_acyclic", "method"):
        assert key in result, f"missing key: {key}"
    assert isinstance(result["edges"], list)
    assert isinstance(result["n_edges"], int)
    assert isinstance(result["is_acyclic"], bool)
    assert isinstance(result["method"], str)


def test_emergence_causal_missing_file(capsys):
    """``causal`` on a nonexistent file exits 1 with a stderr message."""
    code = main(["emergence", "causal", "/nonexistent/path/obs.csv"])
    err = capsys.readouterr().err

    assert code == 1
    assert err


# ----------------------------------------------------------------------
# emergence trajectory
# ----------------------------------------------------------------------

def test_emergence_trajectory_success(capsys):
    """``trajectory`` with inline JSON start/end emits the full path."""
    code = main([
        "emergence", "trajectory",
        "[0.0, 0.0, 0.0]",
        "[1.0, 2.0, 3.0]",
        "--steps", "8",
    ])
    out = capsys.readouterr().out

    assert code == 0
    result = json.loads(out)
    for key in (
        "trajectory",
        "action",
        "converged",
        "iterations",
        "obstacle_violations",
    ):
        assert key in result, f"missing key: {key}"

    traj = result["trajectory"]
    assert len(traj) == 9  # n_steps + 1
    assert len(traj[0]) == 3
    assert len(traj[-1]) == 3
    # Endpoints should match the requested boundary states.
    assert traj[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert traj[-1] == pytest.approx([1.0, 2.0, 3.0], abs=1e-6)


def test_emergence_trajectory_with_obstacles(tmp_path, capsys):
    """``--obstacles`` propagates into the constraints dict."""
    obstacles = np.array([[0.5, 0.5, 0.5]])
    obs_path = _save_npz(tmp_path / "obstacles.npz", obstacles)

    code = main([
        "emergence", "trajectory",
        "[0.0, 0.0, 0.0]",
        "[1.0, 1.0, 1.0]",
        "--steps", "8",
        "--obstacles", str(obs_path),
    ])
    out = capsys.readouterr().out

    assert code == 0
    result = json.loads(out)
    # obstacle_violations is always present (defaults to 0 when no obstacles).
    assert "obstacle_violations" in result
    assert isinstance(result["obstacle_violations"], int)


def test_emergence_trajectory_invalid_json(capsys):
    """Malformed JSON start state exits 1 with a stderr message."""
    code = main([
        "emergence", "trajectory",
        "[not valid json",
        "[0.0, 0.0, 0.0]",
    ])
    err = capsys.readouterr().err

    assert code == 1
    assert err


def test_emergence_trajectory_start_end_shape_mismatch(capsys):
    """start (3D) and end (2D) of different lengths exits 1."""
    code = main([
        "emergence", "trajectory",
        "[0.0, 0.0, 0.0]",
        "[1.0, 2.0]",
    ])
    err = capsys.readouterr().err

    assert code == 1
    assert err


# ----------------------------------------------------------------------
# emergence cycle
# ----------------------------------------------------------------------

def test_emergence_cycle_success(tmp_path, capsys):
    """``cycle`` on a valid .npz emits the full emergence_cycle output."""
    obs = _make_observation(n_samples=40, n_features=4, seed=3)
    path = _save_npz(tmp_path / "obs.npz", obs)

    code = main(["emergence", "cycle", str(path)])
    out = capsys.readouterr().out

    assert code == 0
    result = json.loads(out)
    for key in (
        "perception",
        "causal_graph",
        "counterfactual",
        "posterior",
        "memory",
        "emergence_score",
        "warnings",
    ):
        assert key in result, f"missing key: {key}"
    assert isinstance(result["emergence_score"], float)
    assert 0.0 <= result["emergence_score"] <= 1.0
    assert isinstance(result["warnings"], list)


def test_emergence_cycle_missing_file(capsys):
    """``cycle`` on a nonexistent file exits 1 with a stderr message."""
    code = main(["emergence", "cycle", "/nonexistent/path/obs.npz"])
    err = capsys.readouterr().err

    assert code == 1
    assert err


def test_emergence_cycle_insufficient_data(tmp_path, capsys):
    """``cycle`` on 1D input returns score=0 with insufficient_data reason."""
    path = _save_npy(tmp_path / "obs.npy", np.array([1.0, 2.0, 3.0]))

    code = main(["emergence", "cycle", str(path)])
    out = capsys.readouterr().out

    assert code == 0
    result = json.loads(out)
    assert result["emergence_score"] == 0.0
    assert result["reason"] == "insufficient_data"


# ----------------------------------------------------------------------
# Subprocess smoke test (end-to-end CLI invocation)
# ----------------------------------------------------------------------

def test_subprocess_emergence_help_lists_subcommands():
    """``python -m zero_data_model emergence --help`` lists all 4 subcommands."""
    proc = subprocess.run(
        [sys.executable, "-m", "zero_data_model", "emergence", "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0
    for sub in ("perceive", "causal", "trajectory", "cycle"):
        assert sub in proc.stdout, f"subcommand {sub!r} not in help output"


def test_subprocess_perceive_executes_end_to_end(tmp_path):
    """``python -m zero_data_model emergence perceive <file>`` produces JSON."""
    obs = _make_observation(n_samples=20, n_features=3, seed=5)
    path = _save_npz(tmp_path / "obs.npz", obs)

    proc = subprocess.run(
        [sys.executable, "-m", "zero_data_model",
         "emergence", "perceive", str(path)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0
    result = json.loads(proc.stdout)
    assert "betti_numbers" in result
    assert result["n_points"] >= 1
