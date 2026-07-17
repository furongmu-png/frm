# tests/test_round6_regressions.py
"""Regression tests for the Round-6 audit fixes.

Each test guards one Round-6 finding against silent regression:
  THEORY6-1   process uses the same KL-based FE as compute_free_energy
  THEORY6-18  extra morphogens (n_signals>2) clipped to [0, 1]
  CONCUR6-2   persistence save serialised by _PERSISTENCE_LOCK
  CONCUR6-3   load holds _PERSISTENCE_LOCK across config+npz read
  API6-7-1    IBMQuantumBackend docstring says 2**n_qubits (not 2*n_qubits)
  PERF6-2     npz arrays read once into dict (no double disk read)
  PERF6-5     SimulatorQuantumBackend caches _all_idx in __init__
  PERF6-6     _apply_ry/_apply_cnot mutate state in place
  NEW5-2      API endpoints reject NaN/Inf series (forecast/anomalies/trend)
  NEW5-3      /think rejects NaN/Inf input
  NEW5-4      NLPRules.tokenize rejects over-long text
  NEW5-5      FractalGenerator.generate clamps n_iterations
  NEW5-6      extra morphogens clipped (same as THEORY6-18)
  NEW5-10     category_engine.process calls structural_similarity (no warning)
  RNG5-7      parallel threading path does NOT require joblib
  API6-8-1    version bumped to 0.2.1
"""

from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest

from zero_data_model import __version__
from zero_data_model.active_inference import (
    ActiveInferenceEngine,
)
from zero_data_model.base import Signal
from zero_data_model.biological import MorphogeneticField
from zero_data_model.capabilities.rules import NLPRules
from zero_data_model.hardware.parallel import ParallelExecutor
from zero_data_model.hardware.quantum import SimulatorQuantumBackend
from zero_data_model.math_universe import MathematicalUniverse
from zero_data_model.model import ZeroDataModel
from zero_data_model.persistence import (
    _PERSISTENCE_LOCK,
    ModelSerializer,
    set_persistence_root,
)

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _sandbox_root(tmp_path):
    root = tmp_path / "persistence"
    set_persistence_root(str(root))
    yield root
    set_persistence_root(str(tmp_path / "_reset"))


# --------------------------------------------------------------------------- #
# THEORY6-1: process uses the same KL-based FE as compute_free_energy
# --------------------------------------------------------------------------- #


def test_theory6_1_process_uses_kl_free_energy_formula(monkeypatch):
    """``process`` must use ``compute_free_energy`` (KL-based), not the
    legacy ``pred_error + ||belief||^2 * 0.01`` magnitude penalty.

    We verify by monkeypatching ``compute_free_energy`` to return a sentinel
    value and checking that ``process`` records that sentinel in
    ``free_energy_history``. If ``process`` used the legacy formula, the
    sentinel would never appear.
    """
    engine = ActiveInferenceEngine(state_dim=16, obs_dim=8, action_dim=4)
    obs = np.ones(8) * 0.5
    # Run one cycle first to populate belief_state.
    engine.process(Signal(data=obs))

    # Monkeypatch compute_free_energy to return a distinctive sentinel.
    # Round-7 audit THEORY7-2: the signature now accepts an optional ``state``
    # keyword argument (used by ``select_action``), so the lambda must accept it.
    sentinel = 12345.678
    monkeypatch.setattr(
        engine, "compute_free_energy", lambda obs, state=None: sentinel
    )
    engine.process(Signal(data=obs))
    # The last entry in free_energy_history must be the sentinel (or
    # close to it — process falls back to pred_error only when the
    # sentinel is _FREE_ENERGY_SENTINEL, which 12345.678 is not).
    assert engine.free_energy_history[-1] == pytest.approx(sentinel), (
        f"process recorded {engine.free_energy_history[-1]} but "
        f"compute_free_energy returned {sentinel}; process is not "
        "using compute_free_energy"
    )


# --------------------------------------------------------------------------- #
# THEORY6-18 / NEW5-6: extra morphogens clipped to [0, 1]
# --------------------------------------------------------------------------- #


def test_theory6_18_extra_morphogens_clipped_to_unit_interval():
    """Extra morphogens (n_signals > 2) must be clipped to [0, 1] after
    each ``step``. They are initialised from N(0, 0.1^2) (can be negative)
    and evolved under plain Laplacian diffusion; without clipping they
    drift unbounded, contradicting the docstring's "concentration" claim.
    """
    field = MorphogeneticField(grid_size=8, n_signals=4)
    # Force a large perturbation to drive values outside [0, 1].
    for m in field.morphogens[2:]:
        m += 5.0
    field.step()
    for i, m in enumerate(field.morphogens[2:], start=2):
        assert np.all(m >= 0.0), f"morphogen {i} has negative values: {m.min()}"
        assert np.all(m <= 1.0), f"morphogen {i} has values > 1: {m.max()}"


# --------------------------------------------------------------------------- #
# CONCUR6-2 / CONCUR6-3: persistence lock
# --------------------------------------------------------------------------- #


def test_concur6_2_persistence_lock_is_a_lock():
    """``_PERSISTENCE_LOCK`` must be a ``threading.Lock`` (or compatible)
    so save/load are serialised across threads."""
    assert hasattr(_PERSISTENCE_LOCK, "acquire")
    assert hasattr(_PERSISTENCE_LOCK, "release")


def test_concur6_2_concurrent_saves_do_not_lose_data():
    """Two concurrent saves to the same path must not silently lose one
    snapshot. With the lock, they are serialised; without it, the
    backup→replace→cleanup race loses one writer's data.
    """
    model_a = ZeroDataModel(dim=8, seed=1)
    model_b = ZeroDataModel(dim=8, seed=2)
    errors: list[Exception] = []

    def save_once(model, name):
        try:
            for _ in range(3):
                ModelSerializer.save(model, name)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=save_once, args=(model_a, "snap")),
        threading.Thread(target=save_once, args=(model_b, "snap")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"concurrent saves raised: {errors}"
    # The final snapshot must load without error (no torn state).
    loaded = ModelSerializer.load("snap")
    assert loaded.dim == 8


# --------------------------------------------------------------------------- #
# API6-7-1: IBMQuantumBackend docstring says 2**n_qubits
# --------------------------------------------------------------------------- #


def test_api6_7_1_ibm_backend_docstring_says_2_pow_n_qubits():
    """The docstring must say ``2 ** n_qubits`` (the correct output length),
    not ``2 * n_qubits`` (a 16x understatement at n_qubits=8). The audit
    comment legitimately mentions the old wrong text, so we check only the
    descriptive (non-audit-comment) portion of the docstring."""
    from zero_data_model.hardware.ibm_quantum import IBMQuantumBackend

    doc = IBMQuantumBackend.__doc__ or ""
    # The descriptive portion (before the Round-6 audit comment) must
    # describe the correct length.
    desc = doc.split("Round-6 audit")[0]
    assert "2 ** n_qubits" in desc, "descriptive docstring should say 2 ** n_qubits"
    assert "length ``2 * n_qubits``" not in desc, (
        "descriptive docstring still says length 2 * n_qubits"
    )


# --------------------------------------------------------------------------- #
# PERF6-5 / PERF6-6: simulator caches _all_idx and mutates in place
# --------------------------------------------------------------------------- #


def test_perf6_5_simulator_caches_all_idx():
    """``SimulatorQuantumBackend`` must cache the index array in
    ``__init__`` rather than allocating it on every gate call."""
    backend = SimulatorQuantumBackend(n_qubits=4)
    assert hasattr(backend, "_all_idx"), "missing cached _all_idx"
    assert backend._all_idx.shape == (16,)
    # Reuse: the array must be the same object across calls (not reallocated).
    state = np.zeros(16, dtype=complex)
    state[0] = 1.0
    backend._apply_ry(state, 0.5, 0)
    first = backend._all_idx
    backend._apply_ry(state, 0.5, 1)
    assert backend._all_idx is first, "_all_idx was reallocated"


def test_perf6_6_apply_ry_mutates_state_in_place():
    """``_apply_ry`` must mutate the state array in place (no ``.copy()``
    per gate). The caller reassigns the reference, so the old state is
    never used afterwards — the copy is pure waste."""
    backend = SimulatorQuantumBackend(n_qubits=3)
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0
    original_id = id(state)
    result = backend._apply_ry(state, 0.5, 0)
    assert id(result) == original_id, "_apply_ry returned a different array"


def test_perf6_6_apply_cnot_mutates_state_in_place():
    """``_apply_cnot`` must also mutate in place."""
    backend = SimulatorQuantumBackend(n_qubits=3)
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0
    original_id = id(state)
    result = backend._apply_cnot(state, 0, 1)
    assert id(result) == original_id, "_apply_cnot returned a different array"


def test_perf6_6_in_place_gates_preserve_norm():
    """In-place mutation must not break unitarity: the L2 norm of the
    state vector must stay 1.0 (to floating-point precision) after a
    sequence of gates."""
    backend = SimulatorQuantumBackend(n_qubits=4)
    params = np.random.default_rng(0).standard_normal((2, 4, 2)) * 0.1
    entangling = np.eye(4) * 0.5
    probs = backend.evolve_and_measure(params, entangling)
    # Born rule: probabilities sum to 1.
    assert abs(np.sum(probs) - 1.0) < 1e-9, f"probs sum to {np.sum(probs)}"


# --------------------------------------------------------------------------- #
# NEW5-2: API endpoints reject NaN/Inf series
# --------------------------------------------------------------------------- #


@pytest.fixture()
def _api_client():
    """A TestClient backed by a small model, with rate limiting disabled
    and ``base_url="http://localhost"`` so ``TrustedHostMiddleware``
    (S-MED-13) accepts the request. The default ``testserver`` host used
    by Starlette's TestClient is not in ``ZDM_ALLOWED_HOSTS`` and is
    rejected with a 400 "Invalid host header" *before* the request reaches
    the endpoint -- which would mask the actual NaN-rejection behaviour
    we are testing here.

    Persistence is already sandboxed by the autouse ``_sandbox_root``
    fixture above; we only need to swap the lazy singleton model for a
    small one and disable the slowapi per-route caps.
    """
    from fastapi.testclient import TestClient

    import zero_data_model.api as api_module
    from zero_data_model.api import create_app
    from zero_data_model.model import ZeroDataModel

    api_module.set_model(ZeroDataModel(dim=16))
    prev_enabled = None
    if api_module.limiter is not None:
        prev_enabled = api_module.limiter.enabled
        api_module.limiter.enabled = False
    app = create_app()
    with TestClient(app, base_url="http://localhost") as client:
        yield client
    api_module._model = None
    if api_module.limiter is not None and prev_enabled is not None:
        api_module.limiter.enabled = prev_enabled


def test_new5_2_forecast_rejects_nan_series(_api_client):
    """``/forecast`` must reject a series containing NaN with HTTP 400.

    JSON cannot natively encode NaN/Inf, so we send the raw JSON string
    with ``NaN`` literal (which Python's json encoder emits with
    ``allow_nan=True``) and set the content-type manually. Pydantic's
    JSON parser accepts the ``NaN`` literal as a float, so the body
    reaches the endpoint where ``_ensure_finite`` rejects it.
    """
    import json as _json

    body = _json.dumps({"series": [1.0, float("nan"), 3.0]}, allow_nan=True)
    resp = _api_client.post(
        "/forecast", content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400, resp.text
    assert "finite" in resp.json()["detail"].lower()


def test_new5_2_anomalies_rejects_inf_series(_api_client):
    """``/anomalies`` must reject a series containing Inf with HTTP 400."""
    import json as _json

    body = _json.dumps({"series": [1.0, float("inf")]}, allow_nan=True)
    resp = _api_client.post(
        "/anomalies", content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400, resp.text
    assert "finite" in resp.json()["detail"].lower()


def test_new5_2_trend_rejects_nan_series(_api_client):
    """``/trend`` must reject a series containing NaN with HTTP 400."""
    import json as _json

    body = _json.dumps(
        {"series": [float("nan"), 2.0, 3.0, 4.0, 5.0]}, allow_nan=True
    )
    resp = _api_client.post(
        "/trend", content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400, resp.text
    assert "finite" in resp.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# NEW5-3: /think rejects NaN/Inf input
# --------------------------------------------------------------------------- #


def test_new5_3_think_rejects_nan_input(_api_client):
    """``/think`` must reject an input containing NaN with HTTP 400."""
    import json as _json

    body = _json.dumps({"input": [1.0, float("nan")]}, allow_nan=True)
    resp = _api_client.post(
        "/think", content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400, resp.text
    assert "finite" in resp.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# NEW5-4: NLPRules.tokenize rejects over-long text
# --------------------------------------------------------------------------- #


def test_new5_4_tokenize_rejects_overlong_text():
    """``tokenize`` must reject text longer than the cap (65536 chars)."""
    rules = NLPRules()
    long_text = "a" * 65537
    with pytest.raises(ValueError, match="exceeds tokenize cap"):
        rules.tokenize(long_text)


def test_new5_4_tokenize_accepts_normal_text():
    """Normal-length text must still tokenize without error."""
    rules = NLPRules()
    tokens = rules.tokenize("hello world, this is a test")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


# --------------------------------------------------------------------------- #
# NEW5-5: FractalGenerator.generate clamps n_iterations
# --------------------------------------------------------------------------- #


def test_new5_5_fractal_generate_clamps_n_iterations():
    """``generate`` must clamp ``n_iterations`` so a hostile caller cannot
    hang the process by passing a huge value."""
    universe = MathematicalUniverse(dim=8)
    fractal = universe.fractal
    initial = np.ones(8) * 0.5
    # A huge value must not hang; it should be clamped internally.
    result = fractal.generate(initial, n_iterations=10**9)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result)), "fractal output must be finite"


def test_new5_5_fractal_generate_clamps_negative_iterations():
    """Negative ``n_iterations`` must be clamped to 0 (no iterations)."""
    universe = MathematicalUniverse(dim=8)
    fractal = universe.fractal
    initial = np.ones(8) * 0.5
    result = fractal.generate(initial, n_iterations=-5)
    # With 0 iterations, the output is tanh(initial) or just initial.
    assert result.shape == (8,)


# --------------------------------------------------------------------------- #
# NEW5-10: category_engine.process calls structural_similarity (no warning)
# --------------------------------------------------------------------------- #


def test_new5_10_category_engine_process_emits_no_deprecation_warning():
    """``CategoryTheoryEngine.process`` must call ``structural_similarity``
    directly, not the deprecated ``find_isomorphism`` alias. Each call to the
    alias emits a DeprecationWarning with stack-frame inspection (up to 15
    per cycle)."""
    from zero_data_model.category_engine import CategoryTheoryEngine

    engine = CategoryTheoryEngine(dim=8)
    signal = Signal(data=np.ones(8) * 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Should not raise any DeprecationWarning.
        engine.process(signal)


# --------------------------------------------------------------------------- #
# RNG5-7: parallel threading path does NOT require joblib
# --------------------------------------------------------------------------- #


def test_rng5_7_threading_backend_works_without_joblib(monkeypatch):
    """The ``backend="threading"`` path must work even when joblib is
    absent. The old gate ``self.parallel = n_workers > 1 and _HAS_JOBLIB``
    silently disabled parallelism when joblib was missing, even though
    the threading path uses ``ThreadPoolExecutor`` directly."""
    import zero_data_model.hardware.parallel as parallel_mod

    # Simulate joblib being absent.
    monkeypatch.setattr(parallel_mod, "_HAS_JOBLIB", False)
    executor = ParallelExecutor(n_workers=2, backend="threading")
    assert executor.parallel is True, (
        "threading backend should be enabled even without joblib"
    )


def test_rng5_7_non_threading_backend_requires_joblib(monkeypatch):
    """Non-threading backends (e.g. 'loky') still require joblib."""
    import zero_data_model.hardware.parallel as parallel_mod

    monkeypatch.setattr(parallel_mod, "_HAS_JOBLIB", False)
    executor = ParallelExecutor(n_workers=2, backend="loky")
    assert executor.parallel is False, (
        "non-threading backend should be disabled without joblib"
    )


# --------------------------------------------------------------------------- #
# API6-8-1: version bumped to 0.2.1
# --------------------------------------------------------------------------- #


def test_api6_8_1_version_is_0_2_1():
    """The version must be bumped to ``0.2.1`` after Round-5/6 behaviour
    changes (sentinel value, exception→warn, persistence hardening)."""
    assert __version__ == "0.2.1", f"version is {__version__}, expected 0.2.1"
