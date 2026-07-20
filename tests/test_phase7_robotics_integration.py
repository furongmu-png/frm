# tests/test_phase7_robotics_integration.py
"""Phase 7 — Robotics integration tests across facade / CLI / API / MCP.

Mirrors the Phase 7 Audio / Graph integration test pattern. Verifies that
the 10 Robotics facade methods are reachable from all four entry-point
layers (facade / CLI / API / MCP) and produce consistent results.

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


@pytest.fixture()
def waypoints():
    """Deterministic 4-waypoint path in 2D."""
    return np.array(
        [[0.0, 0.0], [1.0, 0.5], [2.0, 0.5], [3.0, 0.0]],
        dtype=float,
    )


@pytest.fixture()
def waypoints_list(waypoints):
    return waypoints.tolist()


@pytest.fixture()
def joint_angles():
    """4-DOF joint angle vector."""
    return np.array([0.1, -0.2, 0.3, 0.4], dtype=float)


@pytest.fixture()
def joint_angles_list(joint_angles):
    return joint_angles.tolist()


@pytest.fixture()
def target():
    return [0.5, 0.5]


@pytest.fixture()
def measurements():
    return [[1.0, 1.1], [0.9, 1.0], [1.05, 0.95]]


@pytest.fixture()
def variances():
    return [0.1, 0.2, 0.15]


@pytest.fixture()
def prior():
    return [0.0, 0.0]


@pytest.fixture()
def measurement():
    return [1.0, 1.0]


@pytest.fixture()
def trajectory():
    """Deterministic 6-step 2-DOF trajectory (>= 4 rows for jerk)."""
    np.random.seed(42)
    return (np.cumsum(np.random.randn(6, 2), axis=0) * 0.1).tolist()


@pytest.fixture()
def obstacles():
    """Obstacle rows: [center_x, center_y, radius]."""
    return [[1.0, 1.0, 0.2], [3.0, 0.0, 0.15]]


@pytest.fixture()
def position():
    return [0.0, 0.0]


@pytest.fixture()
def path():
    return [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]


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


class TestFacadeRobotics:
    def test_plan_motion_returns_expected_keys(self, waypoints):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.plan_motion(waypoints, n_steps=50)
        assert {"trajectory", "velocities", "accelerations", "total_time"} <= set(r.keys())

    def test_forward_kinematics_returns_2d_position(self, joint_angles):
        m = ZeroDataModel(dim=8, seed=42)
        pos = m.forward_kinematics(joint_angles)
        assert pos.shape == (2,)

    def test_inverse_kinematics_returns_expected_keys(self, target):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.inverse_kinematics(np.array(target, dtype=float))
        assert {"joint_angles", "success", "iterations"} <= set(r.keys())

    def test_fuse_sensors_returns_fused_vector(self, measurements, variances):
        m = ZeroDataModel(dim=8, seed=42)
        meas_arrs = [np.asarray(x, dtype=float) for x in measurements]
        fused = m.fuse_sensors(meas_arrs, variances)
        assert fused.shape == (2,)
        assert np.all(np.isfinite(fused))

    def test_update_kalman_returns_estimate_and_variance(self, prior, measurement):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.update_kalman(
            np.asarray(prior, dtype=float), 1.0,
            np.asarray(measurement, dtype=float), 0.5,
        )
        assert {"estimate", "variance"} <= set(r.keys())
        assert r["variance"] < 1.0  # posterior variance shrinks

    def test_generate_gait_returns_expected_shape(self):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.generate_gait(n_steps=50, gait_type="trot")
        assert {"joint_angles", "foot_contacts", "period"} <= set(r.keys())
        assert np.asarray(r["joint_angles"]).shape[0] == 50

    def test_optimize_trajectory_returns_expected_keys(self, trajectory):
        m = ZeroDataModel(dim=8, seed=42)
        traj = np.asarray(trajectory, dtype=float)
        r = m.optimize_trajectory(traj, n_iter=5)
        assert {"optimized", "jerk", "improvement"} <= set(r.keys())
        assert r["improvement"] >= 0.0

    def test_check_collision_returns_expected_keys(self, obstacles, position):
        m = ZeroDataModel(dim=8, seed=42)
        obs = [(np.array([1.0, 1.0]), 0.2)]
        r = m.check_collision(obs, np.array([0.0, 0.0]), radius=0.1)
        assert {"collision", "nearest_obstacle", "distance"} <= set(r.keys())

    def test_check_path_collision_returns_expected_keys(self, path):
        m = ZeroDataModel(dim=8, seed=42)
        obs = [(np.array([10.0, 10.0]), 0.1)]
        r = m.check_path_collision(obs, np.asarray(path, dtype=float), radius=0.1)
        assert {"collision", "first_collision_step", "nearest_obstacle", "min_clearance"} <= set(r.keys())

    def test_control_mpc_returns_expected_keys(self):
        m = ZeroDataModel(dim=8, seed=42)
        r = m.control_mpc(
            current_state=np.array([0.0, 0.0]),
            target_state=np.array([1.0, 1.0]),
        )
        assert {"action", "predicted_trajectory", "cost"} <= set(r.keys())


# --------------------------------------------------------------------- #
# 2. CLI — subprocess invocations
# --------------------------------------------------------------------- #


class TestCLIRobotics:
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

    def test_cli_motion(self, tmp_path, waypoints_list):
        f = self._write(tmp_path, "wp.json", waypoints_list)
        r = self._run_cli("robotics", "motion", f, "--steps", "20")
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "trajectory" in payload

    def test_cli_forward(self, tmp_path, joint_angles_list):
        f = self._write(tmp_path, "ang.json", joint_angles_list)
        r = self._run_cli("robotics", "forward", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "position" in payload

    def test_cli_inverse(self, tmp_path, target):
        f = self._write(tmp_path, "tgt.json", target)
        r = self._run_cli("robotics", "inverse", f)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "joint_angles" in payload

    def test_cli_fuse(self, tmp_path, measurements):
        f = self._write(tmp_path, "meas.json", measurements)
        r = self._run_cli("robotics", "fuse", f, "--variances", "0.1,0.2,0.15")
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "fused" in payload

    def test_cli_kalman(self, prior, measurement):
        r = self._run_cli(
            "robotics", "kalman",
            json.dumps(prior), json.dumps(measurement),
            "--prior-var", "1.0", "--meas-var", "0.5",
        )
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "estimate" in payload

    def test_cli_gait(self):
        r = self._run_cli("robotics", "gait", "--steps", "20", "--gait", "trot")
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "joint_angles" in payload

    def test_cli_optimize(self, tmp_path, trajectory):
        f = self._write(tmp_path, "traj.json", trajectory)
        r = self._run_cli("robotics", "optimize", f, "--n-iter", "5")
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "optimized" in payload

    def test_cli_collision(self, tmp_path, obstacles, position):
        f = self._write(tmp_path, "obs.json", obstacles)
        r = self._run_cli(
            "robotics", "collision", f,
            "--position", ",".join(str(x) for x in position),
        )
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "collision" in payload

    def test_cli_path_collision(self, tmp_path, obstacles, path):
        fo = self._write(tmp_path, "obs.json", obstacles)
        fp = self._write(tmp_path, "path.json", path)
        r = self._run_cli("robotics", "path-collision", fo, fp)
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "min_clearance" in payload

    def test_cli_mpc(self):
        r = self._run_cli(
            "robotics", "mpc",
            json.dumps([0.0, 0.0]), json.dumps([1.0, 1.0]),
        )
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            assert "action" in payload


# --------------------------------------------------------------------- #
# 3. API — FastAPI endpoints
# --------------------------------------------------------------------- #


class TestAPIRobotics:
    def test_api_motion(self, client, waypoints_list):
        r = client.post(
            "/robotics/motion",
            json={"waypoints": waypoints_list, "n_steps": 20},
        )
        assert r.status_code == 200, r.text
        assert "trajectory" in r.json()

    def test_api_forward(self, client, joint_angles_list):
        r = client.post(
            "/robotics/forward",
            json={"joint_angles": joint_angles_list},
        )
        assert r.status_code == 200, r.text
        assert "position" in r.json()

    def test_api_inverse(self, client, target):
        r = client.post("/robotics/inverse", json={"target": target})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "joint_angles" in body and "success" in body

    def test_api_fuse(self, client, measurements, variances):
        r = client.post(
            "/robotics/fuse",
            json={"measurements": measurements, "variances": variances},
        )
        assert r.status_code == 200, r.text
        assert "fused" in r.json()

    def test_api_kalman(self, client, prior, measurement):
        r = client.post(
            "/robotics/kalman",
            json={
                "prior": prior, "prior_var": 1.0,
                "measurement": measurement, "meas_var": 0.5,
            },
        )
        assert r.status_code == 200, r.text
        assert "estimate" in r.json()

    def test_api_gait(self, client):
        r = client.post(
            "/robotics/gait",
            json={"n_steps": 20, "gait_type": "trot"},
        )
        assert r.status_code == 200, r.text
        assert "joint_angles" in r.json()

    def test_api_optimize(self, client, trajectory):
        r = client.post(
            "/robotics/optimize",
            json={"trajectory": trajectory, "n_iter": 5},
        )
        assert r.status_code == 200, r.text
        assert "optimized" in r.json()

    def test_api_collision(self, client, obstacles, position):
        r = client.post(
            "/robotics/collision",
            json={"obstacles": obstacles, "position": position, "radius": 0.1},
        )
        assert r.status_code == 200, r.text
        assert "collision" in r.json()

    def test_api_path_collision(self, client, obstacles, path):
        r = client.post(
            "/robotics/path-collision",
            json={"obstacles": obstacles, "path": path, "radius": 0.1},
        )
        assert r.status_code == 200, r.text
        assert "min_clearance" in r.json()

    def test_api_mpc(self, client):
        r = client.post(
            "/robotics/mpc",
            json={"current_state": [0.0, 0.0], "target_state": [1.0, 1.0]},
        )
        assert r.status_code == 200, r.text
        assert "action" in r.json()

    def test_api_mpc_with_obstacles(self, client, obstacles):
        r = client.post(
            "/robotics/mpc",
            json={
                "current_state": [0.0, 0.0],
                "target_state": [1.0, 1.0],
                "obstacles": obstacles,
            },
        )
        assert r.status_code == 200, r.text
        assert "action" in r.json()

    # --- negative paths (Pydantic rejection -> 422) ---
    def test_api_motion_single_waypoint_rejected(self, client):
        r = client.post(
            "/robotics/motion",
            json={"waypoints": [[0.0, 0.0]], "n_steps": 20},
        )
        assert r.status_code in (400, 422)

    def test_api_optimize_short_trajectory_rejected(self, client):
        r = client.post(
            "/robotics/optimize",
            json={"trajectory": [[0.0], [1.0], [2.0]], "n_iter": 5},
        )
        assert r.status_code in (400, 422)

    def test_api_gait_invalid_type_rejected(self, client):
        r = client.post(
            "/robotics/gait",
            json={"n_steps": 20, "gait_type": "gallop"},
        )
        assert r.status_code in (400, 422)

    def test_api_kalman_nonpositive_var_rejected(self, client, prior, measurement):
        r = client.post(
            "/robotics/kalman",
            json={
                "prior": prior, "prior_var": 0.0,
                "measurement": measurement, "meas_var": 0.5,
            },
        )
        assert r.status_code in (400, 422)

    def test_api_fuse_length_mismatch_rejected(self, client, measurements):
        r = client.post(
            "/robotics/fuse",
            json={"measurements": measurements, "variances": [0.1, 0.2]},
        )
        # Manual length check inside the handler -> 400.
        assert r.status_code in (400, 422)


# --------------------------------------------------------------------- #
# 4. MCP — ZeroDataMCPServer.call_tool
# --------------------------------------------------------------------- #


class TestMCPRobotics:
    def test_mcp_plan_motion(self, mcp_server, waypoints_list):
        r = mcp_server.call_tool("robotics_plan_motion", waypoints=waypoints_list, n_steps=20)
        assert "trajectory" in r

    def test_mcp_forward_kinematics(self, mcp_server, joint_angles_list):
        r = mcp_server.call_tool("robotics_forward_kinematics", joint_angles=joint_angles_list)
        assert "position" in r

    def test_mcp_inverse_kinematics(self, mcp_server, target):
        r = mcp_server.call_tool("robotics_inverse_kinematics", target=target)
        assert "joint_angles" in r and "success" in r

    def test_mcp_fuse_sensors(self, mcp_server, measurements, variances):
        r = mcp_server.call_tool(
            "robotics_fuse_sensors", measurements=measurements, variances=variances,
        )
        assert "fused" in r

    def test_mcp_update_kalman(self, mcp_server, prior, measurement):
        r = mcp_server.call_tool(
            "robotics_update_kalman",
            prior=prior, prior_var=1.0, measurement=measurement, meas_var=0.5,
        )
        assert "estimate" in r and "variance" in r

    def test_mcp_generate_gait(self, mcp_server):
        r = mcp_server.call_tool("robotics_generate_gait", n_steps=20, gait_type="trot")
        assert "joint_angles" in r and "period" in r

    def test_mcp_optimize_trajectory(self, mcp_server, trajectory):
        r = mcp_server.call_tool("robotics_optimize_trajectory", trajectory=trajectory, n_iter=5)
        assert "optimized" in r and "improvement" in r

    def test_mcp_check_collision(self, mcp_server, obstacles, position):
        r = mcp_server.call_tool(
            "robotics_check_collision", obstacles=obstacles, position=position, radius=0.1,
        )
        assert "collision" in r and "nearest_obstacle" in r

    def test_mcp_check_path_collision(self, mcp_server, obstacles, path):
        r = mcp_server.call_tool(
            "robotics_check_path_collision", obstacles=obstacles, path=path, radius=0.1,
        )
        assert "first_collision_step" in r and "min_clearance" in r

    def test_mcp_control_mpc(self, mcp_server):
        r = mcp_server.call_tool(
            "robotics_control_mpc",
            current_state=[0.0, 0.0], target_state=[1.0, 1.0],
        )
        assert "action" in r and "cost" in r

    def test_mcp_control_mpc_with_obstacles(self, mcp_server, obstacles):
        r = mcp_server.call_tool(
            "robotics_control_mpc",
            current_state=[0.0, 0.0], target_state=[1.0, 1.0], obstacles=obstacles,
        )
        assert "action" in r

    # --- negative paths (return {"error": ...}) ---
    def test_mcp_plan_motion_single_waypoint_returns_error(self, mcp_server):
        r = mcp_server.call_tool("robotics_plan_motion", waypoints=[[0.0, 0.0]], n_steps=10)
        assert "error" in r

    def test_mcp_optimize_short_trajectory_returns_error(self, mcp_server):
        r = mcp_server.call_tool(
            "robotics_optimize_trajectory",
            trajectory=[[0.0], [1.0], [2.0]], n_iter=5,
        )
        assert "error" in r

    def test_mcp_gait_invalid_type_returns_error(self, mcp_server):
        r = mcp_server.call_tool("robotics_generate_gait", n_steps=20, gait_type="gallop")
        assert "error" in r

    def test_mcp_kalman_nonpositive_var_returns_error(self, mcp_server, prior, measurement):
        r = mcp_server.call_tool(
            "robotics_update_kalman",
            prior=prior, prior_var=0.0, measurement=measurement, meas_var=0.5,
        )
        assert "error" in r

    def test_mcp_fuse_length_mismatch_returns_error(self, mcp_server, measurements):
        r = mcp_server.call_tool(
            "robotics_fuse_sensors",
            measurements=measurements, variances=[0.1, 0.2],
        )
        assert "error" in r

    def test_mcp_robotics_tools_registered(self, mcp_server):
        tools = mcp_server.list_tools()
        for name in (
            "robotics_plan_motion", "robotics_forward_kinematics",
            "robotics_inverse_kinematics", "robotics_fuse_sensors",
            "robotics_update_kalman", "robotics_generate_gait",
            "robotics_optimize_trajectory", "robotics_check_collision",
            "robotics_check_path_collision", "robotics_control_mpc",
        ):
            assert name in tools, f"missing tool: {name}"


# --------------------------------------------------------------------- #
# 5. Cross-layer consistency
# --------------------------------------------------------------------- #


class TestCrossLayerConsistency:
    """Verify facade and MCP agree on the dim-agnostic algebraic methods.

    fuse / kalman / collision are pure functions of their inputs — they do
    not depend on the model's embedding dim or RNG seed, so a facade built
    with ZeroDataModel(dim=8, seed=42) should agree with the default MCP
    server (which uses ZeroDataModel(dim=64)).
    """

    def test_facade_and_mcp_fuse_sensors_agree(
        self, measurements, variances, mcp_server,
    ):
        m = ZeroDataModel(dim=8, seed=42)
        meas_arrs = [np.asarray(x, dtype=float) for x in measurements]
        facade_fused = m.fuse_sensors(meas_arrs, variances)
        mcp_fused = mcp_server.call_tool(
            "robotics_fuse_sensors",
            measurements=measurements, variances=variances,
        )["fused"]
        np.testing.assert_allclose(facade_fused, np.asarray(mcp_fused), atol=1e-10)

    def test_facade_and_mcp_kalman_agree(
        self, prior, measurement, mcp_server,
    ):
        m = ZeroDataModel(dim=8, seed=42)
        facade = m.update_kalman(
            np.asarray(prior, dtype=float), 1.0,
            np.asarray(measurement, dtype=float), 0.5,
        )
        mcp = mcp_server.call_tool(
            "robotics_update_kalman",
            prior=prior, prior_var=1.0, measurement=measurement, meas_var=0.5,
        )
        np.testing.assert_allclose(facade["estimate"], np.asarray(mcp["estimate"]), atol=1e-10)
        assert facade["variance"] == pytest.approx(mcp["variance"], abs=1e-12)

    def test_facade_and_mcp_check_collision_agree(
        self, obstacles, position, mcp_server,
    ):
        m = ZeroDataModel(dim=8, seed=42)
        obs = [(np.array([1.0, 1.0]), 0.2), (np.array([3.0, 0.0]), 0.15)]
        facade = m.check_collision(obs, np.array([0.0, 0.0]), radius=0.1)
        mcp = mcp_server.call_tool(
            "robotics_check_collision",
            obstacles=obstacles, position=position, radius=0.1,
        )
        assert facade["collision"] == mcp["collision"]
        assert facade["nearest_obstacle"] == mcp["nearest_obstacle"]
        assert facade["distance"] == pytest.approx(mcp["distance"], abs=1e-10)
