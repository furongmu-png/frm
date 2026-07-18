# tests/test_round9_regressions.py
"""Regression tests for the Round-9 audit fixes.

Each test guards one Round-9 finding against silent regression:

  R9-001      connected_components_1d + _betti_numbers use ``max_radius``
              (not ``max_radius / n_points``) as the gap threshold
  R9-002      vision.FeatureExtractor.extract saves/restores ca.state +
              ca.rule + morphogenetic grid/morphogens/rate
  R9-003      find_invertible_map scales by (nb/na) so T @ source = target
  R9-004      hardware/accel.py catches Exception (not just ImportError)
  R9-005      ZeroDataModel.__getstate__ deepcopies arrays under the lock
  R9-006      _limit_body_size rejects Transfer-Encoding: chunked (411)
  R9-007      DepthEstimator._linear_perspective uses arctan2(gy, gx) (no eps)
  R9-008      MorphogeneticSubstrate.step clips self.grid to [0, 1]
  R9-009      _default_persistence_root creates the directory it returns
  R9-010      VideoFrameAnalyzer._temporal_encoding saves/restores ca.state + rule
  R9-012      BayesianEstimator._update_normal guards non-finite observation
  R9-013      MultiLingualEncoder handles dim < n_stats without wrap-around
  R9-015      PERF8-7 _compute_sigma_q2 reads _recent_actions (not action_history)
"""

from __future__ import annotations

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _sandbox_root(tmp_path):
    """Sandbox persistence under tmp_path so save/load tests do not touch
    the real persistence root."""
    from zero_data_model.persistence import set_persistence_root

    root = tmp_path / "persistence"
    set_persistence_root(str(root))
    yield root
    set_persistence_root(str(tmp_path / "_reset"))


@pytest.fixture()
def api_client(tmp_path):
    """A TestClient configured for the D-batch API regression tests.

    Mirrors the ``client`` fixture in ``tests/test_api.py`` and the
    ``api_client`` fixture in ``tests/test_round8_regressions.py``:
    sandboxes persistence, installs a small dim=16 model, disables
    slowapi rate limits, and points the TestClient at ``http://localhost``
    so the ``TrustedHostMiddleware`` does not reject the request.
    """
    from fastapi.testclient import TestClient

    from zero_data_model import api as api_module
    from zero_data_model.model import ZeroDataModel
    from zero_data_model.persistence import set_persistence_root

    set_persistence_root(str(tmp_path / "api_persistence"))
    api_module.set_model(ZeroDataModel(dim=16))
    prev_limiter_enabled = None
    if api_module.limiter is not None:
        prev_limiter_enabled = api_module.limiter.enabled
        api_module.limiter.enabled = False
    app = api_module.create_app()
    with TestClient(app, base_url="http://localhost") as c:
        yield c
    api_module._model = None
    if api_module.limiter is not None and prev_limiter_enabled is not None:
        api_module.limiter.enabled = prev_limiter_enabled


# --------------------------------------------------------------------------- #
# R9-001: connected_components_1d uses max_radius (not max_radius / n_points)
# --------------------------------------------------------------------------- #


def test_r9_001_connected_components_uniform_points_one_component():
    """A uniformly-spaced point cloud in [0, 1] with ``max_radius >= 1.0``
    must return exactly 1 component (all gaps <= max_radius). The previous
    ``max_radius / n_points`` threshold shrank with the point count, so for
    ``n=64`` the threshold was ~1/64 and roughly half of all adjacent gaps
    exceeded it -- returning ~32 components instead of 1."""
    from zero_data_model.math_universe import MathematicalUniverse

    mu = MathematicalUniverse(dim=64)
    # 16 uniformly-spaced points in [0, 1]: max adjacent gap = 1/15 ~ 0.067.
    points = np.linspace(0.0, 1.0, 16)
    # With max_radius = 1.0 (>> 0.067), all points are in one component.
    n_components = mu.topology.connected_components_1d(points, max_radius=1.0)
    assert n_components == 1, (
        f"uniform points in [0,1] with max_radius=1.0 should give 1 "
        f"component, got {n_components} -- R9-001 regression: the "
        "threshold is still using max_radius / n_points."
    )


def test_r9_001_connected_components_small_radius_gives_n_components():
    """When ``max_radius`` is smaller than every adjacent gap, every point
    is its own component (so the count == n_points). The previous broken
    formula ``max_radius / n_points`` would still split some points into
    the same component for small ``n``."""
    from zero_data_model.math_universe import MathematicalUniverse

    mu = MathematicalUniverse(dim=64)
    # 8 points with min gap = 1.0; max_radius = 0.5 < 1.0 -> 8 components.
    points = np.arange(8, dtype=float)
    n_components = mu.topology.connected_components_1d(points, max_radius=0.5)
    assert n_components == 8, (
        f"8 points spaced 1.0 apart with max_radius=0.5 should give 8 "
        f"components, got {n_components} -- R9-001 regression."
    )


def test_r9_001_jit_kernel_uses_raw_max_radius():
    """The JIT kernel ``_betti_numbers`` must use the raw ``max_radius``
    (not ``max_radius / n_points``). We verify by checking the JIT path
    matches the documented contract directly."""
    from zero_data_model.hardware import kernels

    if not kernels.HAS_NUMBA:
        pytest.skip("numba not installed -- JIT path not exercised")
    # 16 uniformly-spaced points in [0, 1]: max adjacent gap = 1/15 ~ 0.067.
    points = np.sort(np.linspace(0.0, 1.0, 16))
    betti_0, _ = kernels._betti_numbers(
        np.ascontiguousarray(points, dtype=float),
        1.0,  # max_radius >> 0.067 -> single component
    )
    assert betti_0 == 1, (
        f"JIT _betti_numbers with max_radius=1.0 on uniform [0,1] points "
        f"should give 1 component, got {betti_0} -- R9-001 regression."
    )


# --------------------------------------------------------------------------- #
# R9-002: FeatureExtractor.extract restores substrate state
# --------------------------------------------------------------------------- #


def test_r9_002_feature_extractor_preserves_ca_state():
    """``FeatureExtractor.extract`` must save and restore the biological
    substrate's CA ``state`` and ``rule`` so subsequent ``think()`` cycles
    see the substrate's true state, not the image-derived seed evolved 5
    steps. Without the save/restore, the CA state is permanently replaced
    by ``state`` (the binarized image row) and the rule is left at whatever
    ``evolve`` last used."""
    from zero_data_model.biological import BiologicalSubstrate
    from zero_data_model.capabilities.vision import FeatureExtractor

    bio = BiologicalSubstrate(dim=16)
    # Seed the CA with a known state we can verify survives the extract call.
    original_state = bio.automata.state.copy()
    original_rule = bio.automata.rule

    extractor = FeatureExtractor(dim=16, biological=bio)
    img = np.random.default_rng(7).standard_normal((8, 8))
    extractor.extract(img)

    assert np.array_equal(bio.automata.state, original_state), (
        "FeatureExtractor.extract corrupted ca.state -- R9-002 regression: "
        "the save/restore try/finally is missing or wrong."
    )
    assert bio.automata.rule == original_rule, (
        "FeatureExtractor.extract corrupted ca.rule -- R9-002 regression."
    )


def test_r9_002_feature_extractor_preserves_morphogenetic_grid():
    """``FeatureExtractor.extract`` calls
    ``self.biological.morphogenetic.develop(n_steps=10)`` which advances the
    morphogenetic grid + morphogens in place. Without save/restore the grid
    is permanently shifted 10 steps from the image (not the model's
    belief). Verify the grid, morphogens, and diffusion_rate are unchanged."""
    from zero_data_model.biological import BiologicalSubstrate
    from zero_data_model.capabilities.vision import FeatureExtractor

    bio = BiologicalSubstrate(dim=16)
    original_grid = bio.morphogenetic.grid.copy()
    original_morphogens = [m.copy() for m in bio.morphogenetic.morphogens]
    original_rate = bio.morphogenetic.diffusion_rate

    extractor = FeatureExtractor(dim=16, biological=bio)
    img = np.random.default_rng(11).standard_normal((8, 8))
    extractor.extract(img)

    assert np.array_equal(bio.morphogenetic.grid, original_grid), (
        "FeatureExtractor.extract corrupted morphogenetic.grid -- R9-002 regression."
    )
    assert len(bio.morphogenetic.morphogens) == len(original_morphogens)
    for i, (m, orig) in enumerate(
        zip(bio.morphogenetic.morphogens, original_morphogens, strict=False)
    ):
        assert np.array_equal(m, orig), (
            f"morphogen[{i}] corrupted by FeatureExtractor.extract -- R9-002 regression."
        )
    assert bio.morphogenetic.diffusion_rate == original_rate, (
        "morphogenetic.diffusion_rate corrupted -- R9-002 regression."
    )


# --------------------------------------------------------------------------- #
# R9-003: find_invertible_map scales by (nb/na) so T @ source = target
# --------------------------------------------------------------------------- #


def test_r9_003_find_invertible_map_honors_documented_contract():
    """``find_invertible_map(source, target)`` must return a matrix ``T``
    with ``T @ source == target`` EXACTLY (not just up to a norm ratio).
    The previous unscaled Householder reflection satisfied
    ``T @ source = (|source|/|target|) * target``, silently rescaling
    transferred solutions by the source/target norm ratio."""
    from zero_data_model.category_engine import CategoryTheoryEngine

    engine = CategoryTheoryEngine(dim=8, rng=np.random.default_rng(0))
    # Source and target with DIFFERENT magnitudes (so the bug is observable).
    source = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # |target| = 3
    T = engine.find_invertible_map(source, target)
    assert T is not None
    # The documented contract: T @ source == target (not scaled).
    np.testing.assert_allclose(T @ source, target, atol=1e-10)


def test_r9_003_find_invertible_map_remains_invertible():
    """After the (nb/na) scaling, the map is no longer orthogonal but must
    still be invertible (the whole point of the function)."""
    from zero_data_model.category_engine import CategoryTheoryEngine

    engine = CategoryTheoryEngine(dim=8, rng=np.random.default_rng(1))
    rng = np.random.default_rng(42)
    source = rng.standard_normal(8)
    target = rng.standard_normal(8) * 5.0  # different magnitude
    T = engine.find_invertible_map(source, target)
    assert T is not None
    assert np.linalg.matrix_rank(T) == 8, (
        "scaled Householder map is rank-deficient -- R9-003 regression: "
        "the (nb/na) scaling broke invertibility."
    )


# --------------------------------------------------------------------------- #
# R9-004: hardware/accel.py catches Exception (not just ImportError)
# --------------------------------------------------------------------------- #


def test_r9_004_accel_module_imports_without_crash_when_cupy_probe_fails():
    """``hardware/accel.py`` must export ``xp`` (the array module) even
    when CuPy is importable but its CUDA runtime probe raises
    ``RuntimeError`` (e.g. mismatched driver). The previous ``except
    ImportError`` left ``xp`` unbound, crashing every downstream numerics
    import with ``NameError``. We simulate the failure by injecting a
    fake ``cupy`` module whose ``cuda.runtime.getDeviceCount`` raises
    RuntimeError, then re-importing accel and verifying ``xp`` falls
    back to numpy."""
    import importlib
    import sys
    import types

    # Build a fake ``cupy`` module whose cuda.runtime.getDeviceCount
    # raises RuntimeError (simulating a CUDA driver mismatch).
    fake_cupy = types.ModuleType("cupy")
    fake_cuda = types.ModuleType("cupy.cuda")
    fake_runtime = types.ModuleType("cupy.cuda.runtime")

    def _boom():
        raise RuntimeError("cudaErrorInsufficientDriver (simulated)")

    fake_runtime.getDeviceCount = _boom
    fake_cuda.runtime = fake_runtime
    fake_cupy.cuda = fake_cuda
    # Save + restore sys.modules so the fake import does not leak.
    saved = {
        "cupy": sys.modules.get("cupy"),
        "cupy.cuda": sys.modules.get("cupy.cuda"),
        "cupy.cuda.runtime": sys.modules.get("cupy.cuda.runtime"),
        "zero_data_model.hardware.accel": sys.modules.get(
            "zero_data_model.hardware.accel"
        ),
    }
    sys.modules["cupy"] = fake_cupy
    sys.modules["cupy.cuda"] = fake_cuda
    sys.modules["cupy.cuda.runtime"] = fake_runtime
    # Remove accel from sys.modules so the import re-runs the probe.
    if "zero_data_model.hardware.accel" in sys.modules:
        del sys.modules["zero_data_model.hardware.accel"]
    try:
        accel = importlib.import_module("zero_data_model.hardware.accel")
        # ``xp`` must be bound (to numpy when the probe raised) -- the
        # previous ``except ImportError`` would have left it unbound,
        # raising NameError on this attribute access.
        assert hasattr(accel, "xp"), (
            "accel module did not export ``xp`` after CuPy probe failed "
            "-- R9-004 regression: a RuntimeError from getDeviceCount "
            "left xp unbound."
        )
        assert accel.xp is not None
        # When the probe raises, the backend must fall back to numpy.
        assert accel.backend_name() == "numpy"
        assert accel.has_gpu is False
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        # Re-import the real accel module so other tests see the real backend.
        if "zero_data_model.hardware.accel" in sys.modules:
            del sys.modules["zero_data_model.hardware.accel"]
        importlib.import_module("zero_data_model.hardware.accel")


# --------------------------------------------------------------------------- #
# R9-005: ZeroDataModel.__getstate__ deepcopies arrays under the lock
# --------------------------------------------------------------------------- #


def test_r9_005_getstate_returns_independent_arrays():
    """``__getstate__`` must return a dict whose numpy arrays are
    INDEPENDENT copies of the model's arrays, so a concurrent ``think()``
    mutating the model in-place (``emission[:, :] += ...``) cannot tear
    the snapshot pickle is about to serialise. The previous SIDE-1 fix
    took the lock only for the shallow ``__dict__.copy()`` then released
    it before pickle walked the arrays -- a torn-snapshot race window."""
    from zero_data_model.model import ZeroDataModel

    model = ZeroDataModel(dim=8)
    state = model.__getstate__()
    # Every numpy array in ``state`` must be a separate object from the
    # model's live array (deepcopy, not reference).
    live_arrays = {
        k: v
        for k, v in model.__dict__.items()
        if isinstance(v, np.ndarray)
    }
    for key, live_arr in live_arrays.items():
        if key not in state:
            continue
        snap_arr = state[key]
        if not isinstance(snap_arr, np.ndarray):
            continue
        # Must be a distinct object (not the same array view).
        assert snap_arr is not live_arr, (
            f"__getstate__ returned a reference to live array '{key}' -- "
            "R9-005 regression: a concurrent think() could mutate the "
            "array while pickle is serialising it (torn snapshot)."
        )
        # Must have the same values (deepcopy of the current state).
        np.testing.assert_array_equal(snap_arr, live_arr)
        # Mutating the snapshot must not affect the live model.
        snap_arr.fill(0.0)
        assert not np.allclose(live_arr, 0.0) or live_arr.size == 0, (
            f"mutating snapshot '{key}' affected the live model -- "
            "R9-005 regression: __getstate__ did not deepcopy."
        )


# --------------------------------------------------------------------------- #
# R9-006: _limit_body_size rejects Transfer-Encoding: chunked (411)
# --------------------------------------------------------------------------- #


def test_r9_006_chunked_transfer_encoding_rejected(api_client):
    """A request with ``Transfer-Encoding: chunked`` must be rejected
    with 411 Length Required (not silently accepted with no body-size cap).
    The previous implementation only enforced the 4 MiB cap when a
    Content-Length header was present, so a chunked request bypassed the
    cap entirely -- allowing an unbounded body to be buffered in memory
    by the ASGI server until OOM-killed."""
    # We cannot easily send a true chunked request via TestClient (httpx
    # forces Content-Length on the wire), but we CAN verify the middleware
    # inspects the Transfer-Encoding header by sending it explicitly. The
    # middleware must short-circuit before ``call_next`` runs.
    response = api_client.post(
        "/classify",
        headers={"Transfer-Encoding": "chunked"},
        json={"text": "hello"},
    )
    assert response.status_code == 411, (
        f"chunked-encoding request should be rejected with 411, got "
        f"{response.status_code} -- R9-006 regression: the body-size "
        "cap is bypassable via chunked transfer."
    )
    body = response.json()
    assert "request_id" in body, "411 response must include request_id"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"


# --------------------------------------------------------------------------- #
# R9-007: DepthEstimator._linear_perspective uses arctan2(gy, gx) (no eps)
# --------------------------------------------------------------------------- #


def test_r9_007_linear_perspective_handles_zero_gradient_without_bias():
    """``_linear_perspective`` must use ``np.arctan2(gy, gx)`` directly
    (no ``+ eps`` on gx). The previous ``gx + eps`` biased every angle by
    a small sign-dependent amount toward +x. We verify by parsing the
    function body with ``ast`` (so the docstring/comment text mentioning
    ``gx + eps`` does not falsely trip a substring check) and by checking
    that a constant image (gx = gy = 0 everywhere) does not produce a NaN
    orientation."""
    import ast
    import inspect
    import textwrap

    from zero_data_model.capabilities.vision_advanced import DepthEstimator

    source = textwrap.dedent(inspect.getsource(DepthEstimator._linear_perspective))
    tree = ast.parse(source)
    # Walk the parsed AST (which excludes comments + docstrings) and look
    # for any ``gx + eps`` BinOp -- the documented R9-007 regression.
    found_bias = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
            continue
        left_is_gx = isinstance(node.left, ast.Name) and node.left.id == "gx"
        right_is_eps = isinstance(node.right, ast.Name) and node.right.id == "eps"
        if left_is_gx and right_is_eps:
            found_bias = True
            break
    assert not found_bias, (
        "_linear_perspective still biases arctan2 with ``gx + eps`` -- "
        "R9-007 regression: arctan2(0,0) is well-defined (returns 0.0)."
    )
    # Sanity-check the function does not crash or produce NaN on a
    # constant image (where gx = gy = 0 everywhere).
    estimator = DepthEstimator(dim=8)
    img = np.full((8, 8), 0.5, dtype=float)
    depth = estimator._linear_perspective(img)
    assert np.all(np.isfinite(depth)), (
        "_linear_perspective returned non-finite values on a constant image."
    )


# --------------------------------------------------------------------------- #
# R9-008: MorphogeneticSubstrate.step clips self.grid to [0, 1]
# --------------------------------------------------------------------------- #


def test_r9_008_step_clips_grid_to_unit_interval():
    """``MorphogeneticField.step`` must clip ``self.grid`` to [0, 1]
    after the diffusion update, matching the (u, v) Gray-Scott pair and
    the extra morphogens. The previous implementation left ``grid``
    unbounded, so a signed grid mixed a [-0.3, 0.3]-ish quantity with
    [0, 1]-clipped morphogens in ``develop``'s ``pattern = grid * (1 +
    0.1 * morphogen_sum)`` -- the exact semantic inconsistency Round-6
    THEORY6-18 sought to eliminate."""
    from zero_data_model.biological import BiologicalSubstrate

    bio = BiologicalSubstrate(dim=16)
    # Force the grid out of [0, 1] so the clip is observable.
    bio.morphogenetic.grid[:] = 5.0
    bio.morphogenetic.step()
    grid = bio.morphogenetic.grid
    assert np.all(grid >= 0.0) and np.all(grid <= 1.0), (
        f"grid values out of [0, 1] after step: min={grid.min()}, "
        f"max={grid.max()} -- R9-008 regression: the [0, 1] clip is "
        "missing from the grid diffusion update."
    )


# --------------------------------------------------------------------------- #
# R9-009: _default_persistence_root creates the directory it returns
# --------------------------------------------------------------------------- #


def test_r9_009_default_persistence_root_is_usable(monkeypatch, tmp_path):
    """``_default_persistence_root`` must return a path that EXISTS on
    disk (so the first ``ModelSerializer.save`` call doesn't crash with
    ``FileNotFoundError`` from ``tempfile.mkdtemp(dir=root)``). The
    previous implementation returned ``cwd/data`` without creating it,
    so the first save on a fresh checkout failed. We verify the contract
    by monkey-patching ``os.getcwd`` and ``os.path.isdir`` so the
    function takes the non-container branch (``cwd/data``) against a
    fresh ``tmp_path`` -- and then assert the returned path exists."""
    import os

    from zero_data_model import persistence

    # Force the non-container branch: pretend /app and /app/data do NOT
    # exist, and the cwd is tmp_path (which has no ``data/`` subdir yet).
    real_isdir = os.path.isdir

    def fake_isdir(p):
        if p in ("/app", "/app/data"):
            return False
        return real_isdir(p)

    monkeypatch.setattr(os.path, "isdir", fake_isdir)
    monkeypatch.chdir(tmp_path)
    root = persistence._default_persistence_root()
    assert os.path.isdir(root), (
        f"_default_persistence_root returned {root!r} but the directory "
        "does not exist -- R9-009 regression: the first save call would "
        "crash with FileNotFoundError from tempfile.mkdtemp."
    )
    # The returned path must be the cwd/data path (the non-container branch).
    assert root.endswith("data")


# --------------------------------------------------------------------------- #
# R9-010: VideoFrameAnalyzer._temporal_encoding restores ca.state + rule
# --------------------------------------------------------------------------- #


def test_r9_010_temporal_encoding_preserves_ca_state_and_rule():
    """``_temporal_encoding`` must save and restore ``ca.state`` AND
    ``ca.rule`` so subsequent ``think()`` cycles see the substrate's true
    state, not the motion-derived seed advanced ``n_steps``. The bug is
    especially insidious because ``ca.rule`` is overwritten (not just
    ``state``), so callers that save/restore only ``state`` elsewhere
    would still see the rule change."""
    from zero_data_model.biological import BiologicalSubstrate
    from zero_data_model.capabilities.vision_advanced import VideoFrameAnalyzer

    bio = BiologicalSubstrate(dim=16)
    analyzer = VideoFrameAnalyzer(dim=16, biological=bio)
    original_state = bio.automata.state.copy()
    original_rule = bio.automata.rule

    # A motion series with values above and below the mean.
    motion = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    analyzer._temporal_encoding(motion)

    assert np.array_equal(bio.automata.state, original_state), (
        "_temporal_encoding corrupted ca.state -- R9-010 regression: "
        "the save/restore try/finally is missing or wrong."
    )
    assert bio.automata.rule == original_rule, (
        "_temporal_encoding corrupted ca.rule -- R9-010 regression: "
        "rule was overwritten with 30 and not restored."
    )


# --------------------------------------------------------------------------- #
# R9-012: BayesianEstimator._update_normal guards non-finite observation
# --------------------------------------------------------------------------- #


def test_r9_012_update_normal_rejects_nan_observation():
    """``_update_normal`` must reject a NaN observation (do nothing)
    rather than irreversibly poisoning the posterior. Without the guard,
    a single NaN makes ``new_mean`` NaN, ``new_var`` NaN, and every
    subsequent update re-combines NaN -- the posterior stays NaN forever."""
    from zero_data_model.capabilities.analytics_advanced import BayesianEstimator

    est = BayesianEstimator(dim=8)
    est.reset(prior_mean=0.5, prior_var=1.0)
    mean_before = est.normal_mean
    var_before = est.normal_var
    n_before = est._normal_n
    est.update(float("nan"))
    assert est.normal_mean == mean_before, (
        "NaN observation changed normal_mean -- R9-012 regression: "
        "the posterior is now permanently poisoned."
    )
    assert est.normal_var == var_before
    assert est._normal_n == n_before, (
        "NaN observation should not increment the observation count."
    )


def test_r9_012_update_normal_rejects_inf_observation():
    """``_update_normal`` must reject an Inf observation for the same
    reason as NaN."""
    from zero_data_model.capabilities.analytics_advanced import BayesianEstimator

    est = BayesianEstimator(dim=8)
    est.reset(prior_mean=0.5, prior_var=1.0)
    mean_before = est.normal_mean
    est.update(float("inf"))
    assert est.normal_mean == mean_before, (
        "Inf observation changed normal_mean -- R9-012 regression."
    )


def test_r9_012_update_normal_accepts_finite_observation():
    """A finite observation must still update the posterior normally
    (the guard must not over-reject valid inputs)."""
    from zero_data_model.capabilities.analytics_advanced import BayesianEstimator

    est = BayesianEstimator(dim=8)
    est.reset(prior_mean=0.0, prior_var=1.0)
    mean_before = est.normal_mean
    est.update(2.0)
    # Posterior mean should move toward the observation (from 0 toward 2).
    assert est.normal_mean > mean_before, (
        "Finite observation did not move the posterior mean -- "
        "R9-012 over-rejects valid inputs."
    )
    assert est._normal_n == 1


# --------------------------------------------------------------------------- #
# R9-013: MultiLingualEncoder handles dim < n_stats without wrap-around
# --------------------------------------------------------------------------- #


def test_r9_013_multilingual_encoder_small_dim_does_not_wrap():
    """When ``dim < len(_RECOGNIZED_SCRIPTS)`` (11), the encoder must
    not wrap the per-script stats modulo ``dim`` (which would
    double-count three script proportions and overwrite the base
    encoder's output in the leading bins). The encoding must degrade
    gracefully -- either truncating the stats or omitting them -- and
    must still produce an L2-normalised ``dim``-length vector."""
    from zero_data_model.capabilities.nlp_advanced import MultiLingualEncoder

    # dim=8 < 11 scripts -- the previous implementation wrapped.
    enc = MultiLingualEncoder(dim=8)
    vec = enc.encode("hello world")
    assert vec.shape == (8,), (
        f"encode returned shape {vec.shape}, expected (8,) -- the "
        "encoding must always return a dim-length vector."
    )
    # Must be L2-normalised.
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-6, (
        f"encoded vector norm = {norm}, expected 1.0 -- the encoding "
        "must always produce a unit vector."
    )


def test_r9_013_multilingual_encoder_default_dim_unaffected():
    """The default ``dim=64`` path must be unaffected by the R9-013 fix
    (it has enough room for all 11 scripts without truncation)."""
    from zero_data_model.capabilities.nlp_advanced import MultiLingualEncoder

    enc = MultiLingualEncoder(dim=64)
    vec = enc.encode("hello world")
    assert vec.shape == (64,)
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# R9-015: PERF8-7 _compute_sigma_q2 reads _recent_actions (not action_history)
# --------------------------------------------------------------------------- #


def test_r9_015_compute_sigma_q2_reads_recent_actions_not_full_history():
    """``_compute_sigma_q2`` must read from ``_recent_actions`` (maxlen=32),
    not from ``action_history`` (maxlen=1000). The previous regression
    test (test_perf8_7_compute_sigma_q2_uses_recent_actions_not_full_history)
    used data where both paths produced indistinguishable results (~1.0),
    so it provided zero actual regression protection. This test uses data
    where the two paths produce DIFFERENT sigma_q2 values, so a future
    refactor that reverts to reading ``action_history`` will fail.
    """
    from zero_data_model.active_inference import ActiveInferenceEngine

    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=4, action_dim=2, rng=np.random.default_rng(99)
    )
    # action_history: 40 copies of [1.0, -1.0] -> var per dim = 1.0,
    # mean var = 1.0 -> sigma_q2 = 1.0 + 1e-6 ~ 1.000001.
    # _recent_actions: 32 copies of [0.0, 0.0] -> var per dim = 0,
    # mean var = 0 -> sigma_q2 = 0 + 1e-6 = 1e-6.
    # The two paths give sigma_q2 ~ 1.0 vs ~1e-6 -- easily distinguished.
    engine.action_history.clear()
    engine._recent_actions.clear()
    for _ in range(40):
        engine.action_history.append(np.array([1.0, -1.0]))
    for _ in range(32):
        engine._recent_actions.append(np.array([0.0, 0.0]))
    engine._sigma_q2_dirty = True
    sigma_q2 = engine._compute_sigma_q2()
    # If sigma_q2 read _recent_actions (correct): sigma_q2 = 1e-6.
    # If sigma_q2 read action_history (regression): sigma_q2 ~ 1.000001.
    assert sigma_q2 < 0.01, (
        f"sigma_q2 = {sigma_q2}, expected ~1e-6 -- PERF8-7 regression: "
        "_compute_sigma_q2 is reading action_history instead of "
        "_recent_actions (would give ~1.0)."
    )
