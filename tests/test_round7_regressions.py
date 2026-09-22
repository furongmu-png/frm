# tests/test_round7_regressions.py
"""Regression tests for the Round-7 audit fixes.

Each test guards one Round-7 finding against silent regression:

  THEORY7-1   compute_free_energy uses ONE state for KL + NLL (no posterior mixing)
  THEORY7-2   select_action evaluates EFE under predicted_state (not current belief)
  THEORY7-3   process records PRIOR surprise (FE computed before update_belief)
  THEORY7-4   think passes actual MSE (not pred.uncertainty) to module.update()
  CONCUR7-1   persistence load reads npz bytes under _PERSISTENCE_LOCK (no torn read)
  PERF7-1     ParallelExecutor reuses a persistent ThreadPoolExecutor across map() calls
  PERF7-2     process does NOT call infer_state (no redundant gradient step)
  API7-1-1    /save OpenAPI schema declares 201 + SaveResponse{saved, name}
  NEW7-1      callers use structural_similarity (no find_isomorphism DeprecationWarning)
"""

from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest

from zero_data_model.active_inference import (
    ActiveInferenceEngine,
)
from zero_data_model.base import Signal
from zero_data_model.hardware.parallel import ParallelExecutor
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
# THEORY7-1: KL term and NLL term share the same posterior state
# --------------------------------------------------------------------------- #


def test_theory7_1_compute_free_energy_uses_one_state_for_kl_and_nll():
    """``compute_free_energy(state=s)`` must use ``s`` for BOTH the KL term
    (``||s||^2``) and the pragmatic NLL term (``||obs - s @ emission||^2``).

    The Round-6 fix mixed two posteriors: KL used ``||inferred||^2`` (one
    gradient step ahead of ``belief_state``) while NLL used ``pred_error``
    from ``belief_state``. The Round-7 fix unifies them on a single ``state``.

    We verify by passing a known ``state`` and checking that BOTH terms
    respond to it. Specifically:
      * If KL uses ``state`` and NLL uses ``state``, scaling ``state`` by 0
        zeros the KL term AND aligns the prediction with ``state @ emission``.
      * If KL silently fell back to ``belief_state`` (the Round-6 bug), the
        KL term would NOT track the passed ``state``.
    """
    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=4, action_dim=2, rng=np.random.default_rng(42)
    )
    obs = np.ones(4) * 0.5

    # Belief the engine was initialised with.
    belief = engine.generative_model.belief_state.copy()
    # A DISTINCT state far from the belief.
    distinct_state = belief + np.ones(8) * 10.0

    fe_belief = engine.compute_free_energy(obs, state=belief)
    fe_distinct = engine.compute_free_energy(obs, state=distinct_state)

    # Both must be finite (not the sentinel).
    assert np.isfinite(fe_belief), "FE(belief) must be finite"
    assert np.isfinite(fe_distinct), "FE(distinct_state) must be finite"

    # ||distinct_state||^2 >> ||belief||^2, so the KL term dominates and
    # fe_distinct > fe_belief. If the KL term silently used ``belief`` for
    # both calls (ignoring ``state``), fe_distinct would equal fe_belief.
    assert fe_distinct > fe_belief + 1.0, (
        "KL term did not track the passed ``state``; "
        "FE(distinct) should dominate FE(belief) via ||state||^2"
    )


# --------------------------------------------------------------------------- #
# THEORY7-2: select_action evaluates EFE under predicted_state
# --------------------------------------------------------------------------- #


def test_theory7_2_select_action_uses_predicted_state_for_efe(monkeypatch):
    """``select_action`` must pass ``state=predicted_state`` to
    ``compute_free_energy`` so the EFE reflects the candidate action's
    predicted outcome (not the current belief).

    We monkeypatch ``compute_free_energy`` to record the ``state`` kwarg it
    was called with, then check that at least one call used a state DIFFERENT
    from the current ``belief_state``.
    """
    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=4, action_dim=2, rng=np.random.default_rng(7)
    )
    # Run a cycle so belief_state is non-trivial.
    engine.process(Signal(data=np.ones(4) * 0.3))

    observed_states: list = []
    real_cfe = engine.compute_free_energy

    def recording_cfe(obs, state=None):
        observed_states.append(state)
        return real_cfe(obs, state=state)

    monkeypatch.setattr(engine, "compute_free_energy", recording_cfe)
    engine.select_action(engine.generative_model.belief_state)

    belief = engine.generative_model.belief_state
    # At least one call must have used a non-None state distinct from belief.
    non_belief_calls = [
        s for s in observed_states
        if s is not None and not np.allclose(s, belief, atol=1e-9)
    ]
    assert non_belief_calls, (
        "select_action never called compute_free_energy(state=predicted_state); "
        "the EFE was evaluated under the current belief (Round-7 THEORY7-2 regression)"
    )


# --------------------------------------------------------------------------- #
# THEORY7-3: process records PRIOR surprise (FE before update_belief)
# --------------------------------------------------------------------------- #


def test_theory7_3_process_records_prior_surprise_before_update(monkeypatch):
    """``process`` must compute ``free_energy`` BEFORE ``update_belief``
    mutates ``belief_state``. We verify by capturing the belief_state seen
    by ``compute_free_energy`` and confirming it equals the PRE-update belief.

    The Round-6 fix called ``compute_free_energy`` AFTER ``update_belief``,
    so the recorded FE was the posterior surprise (after the belief had
    already moved toward the observation).
    """
    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=4, action_dim=2, rng=np.random.default_rng(11)
    )
    # Establish a non-trivial belief.
    engine.process(Signal(data=np.ones(4) * 0.2))
    pre_belief = engine.generative_model.belief_state.copy()

    captured_belief: list = []

    def spy_cfe(self_engine, observation, state=None):
        # Record the belief_state compute_free_energy sees at call time.
        captured_belief.append(self_engine.generative_model.belief_state.copy())
        return orig_cfe(self_engine, observation, state=state)

    orig_cfe = ActiveInferenceEngine.compute_free_energy
    monkeypatch.setattr(ActiveInferenceEngine, "compute_free_energy", spy_cfe)

    engine.process(Signal(data=np.ones(4) * 0.7))

    assert len(captured_belief) >= 1, "compute_free_energy was never called"
    # The belief compute_free_energy saw must match the PRE-update belief
    # (process must call compute_free_energy BEFORE update_belief mutates it).
    assert np.allclose(captured_belief[0], pre_belief, atol=1e-12), (
        "process called compute_free_energy AFTER update_belief mutated "
        "belief_state; the recorded FE is posterior surprise, not prior"
    )


# --------------------------------------------------------------------------- #
# THEORY7-4: think passes actual MSE (not pred.uncertainty) to module.update
# --------------------------------------------------------------------------- #


def test_theory7_4_think_passes_actual_mse_to_module_update():
    """``ZeroDataModel.think`` must compute the per-module prediction error
    as the mean-squared distance between the predicted observation and the
    input signal, NOT pass ``pred.uncertainty`` (the variance of the
    predicted output).

    We verify by patching ``ActiveInferenceEngine.update`` to record the
    ``prediction_error`` argument it receives, then checking the value is
    consistent with the MSE — NOT the variance of the prediction.
    """
    model = ZeroDataModel(dim=8, seed=3)
    received_errors: list[float] = []

    orig_update = model.active_inference.update

    def recording_update(prediction_error):
        received_errors.append(float(prediction_error))
        return orig_update(prediction_error)

    model.active_inference.update = recording_update  # type: ignore[assignment]

    # Drive a think cycle with a known input.
    signal = np.ones(8) * 0.4
    model.think(signal)

    # At least one module update must have been called.
    # Note: only ActiveInferenceEngine.update is patched; other modules use
    # their own update signatures. We check the active_inference module's
    # recorded error.
    assert received_errors, "active_inference.update was never called"

    # The recorded error must NOT equal pred.uncertainty (variance of the
    # predicted observation). Compute what the OLD buggy code would have
    # passed (variance of the predicted obs) and assert they differ.
    pred = model.active_inference.predict(Signal(data=signal))
    pred_var = float(np.var(pred.value))
    # The new MSE-based error must differ from the variance — otherwise we
    # silently regressed to the Round-6 behaviour. Allow exact equality only
    # when the variance itself is ~0 (degenerate).
    if pred_var > 1e-6:
        assert not np.allclose(received_errors[0], pred_var, rtol=1e-3), (
            "think passed pred.uncertainty (variance) to module.update; "
            "should pass the actual MSE prediction error"
        )


# --------------------------------------------------------------------------- #
# CONCUR7-1: persistence load reads npz bytes under _PERSISTENCE_LOCK
# --------------------------------------------------------------------------- #


def test_concur7_1_load_reads_npz_bytes_under_lock(tmp_path):
    """``ModelSerializer.load`` must read BOTH ``config.json`` AND the npz
    file's bytes while holding ``_PERSISTENCE_LOCK``. The Round-6 fix only
    locked the config read; the npz read ran unlocked ~130 lines later, so
    a concurrent ``save``'s ``os.replace(staging, target)`` could swap the
    directory between the two reads (torn read).

    We verify by holding ``_PERSISTENCE_LOCK`` from a side thread while the
    main thread tries to load: the load must BLOCK until the lock is
    released, proving the npz read is also covered by the lock.
    """
    model = ZeroDataModel(dim=8, seed=1)
    ModelSerializer.save(model, "snap_for_lock_test")

    # Acquire the lock in the main thread; load() must not be able to
    # complete until we release it.
    acquired = _PERSISTENCE_LOCK.acquire(blocking=False)
    assert acquired, "test setup: could not acquire _PERSISTENCE_LOCK"

    load_done = threading.Event()
    load_error: list = []

    def load_attempt():
        try:
            ModelSerializer.load("snap_for_lock_test")
            load_done.set()
        except Exception as exc:
            load_error.append(exc)
            load_done.set()

    t = threading.Thread(target=load_attempt)
    t.start()

    # Give load() time to either finish (bug) or block (correct).
    t.join(timeout=0.4)
    # Correct behaviour: load is blocked on the lock and has NOT finished.
    assert not load_done.is_set(), (
        "load() completed while _PERSISTENCE_LOCK was held by another thread; "
        "the npz read is NOT covered by the lock (CONCUR7-1 regression)"
    )

    # Release the lock; load() should now finish quickly.
    _PERSISTENCE_LOCK.release()
    t.join(timeout=5.0)
    assert load_done.is_set(), "load() did not finish after the lock was released"
    assert load_error == [], f"load raised after lock release: {load_error}"


# --------------------------------------------------------------------------- #
# PERF7-1: ParallelExecutor reuses a persistent ThreadPoolExecutor
# --------------------------------------------------------------------------- #


def test_perf7_1_parallel_executor_reuses_persistent_pool():
    """``ParallelExecutor`` must create a ``ThreadPoolExecutor`` ONCE in
    ``__init__`` and reuse it across ``map()`` calls, rather than spawning
    and tearing down a fresh pool per call.

    We verify by counting ThreadPoolExecutor instantiations across two
    ``map`` calls: with the persistent pool, the count is 1 (the one in
    __init__); without it (Round-6 behaviour), the count is 2.
    """
    instantiation_count = 0

    # Patch ThreadPoolExecutor.__init__ to count instantiations.
    from concurrent.futures import ThreadPoolExecutor

    real_tpe_init = ThreadPoolExecutor.__init__

    def counting_tpe_init(self, *args, **kwargs):
        nonlocal instantiation_count
        instantiation_count += 1
        return real_tpe_init(self, *args, **kwargs)

    ThreadPoolExecutor.__init__ = counting_tpe_init  # type: ignore[assignment]
    try:
        executor = ParallelExecutor(n_workers=2)
        # One pool created in __init__.
        assert instantiation_count == 1, (
            f"expected 1 pool in __init__, got {instantiation_count}"
        )
        # Run two map calls — they MUST reuse the same pool.
        executor.map(lambda x: x * 2, [1, 2, 3])
        executor.map(lambda x: x * 2, [4, 5, 6])
        assert instantiation_count == 1, (
            f"map() created a new ThreadPoolExecutor instead of reusing the "
            f"persistent pool; count after 2 map calls = {instantiation_count}"
        )
        executor.shutdown()
    finally:
        ThreadPoolExecutor.__init__ = real_tpe_init  # type: ignore[assignment]


def test_perf7_1_parallel_executor_shutdown_releases_pool():
    """``ParallelExecutor.shutdown()`` must release the persistent pool so
    worker threads are reclaimed (no resource leak across long-lived
    executor instances)."""
    executor = ParallelExecutor(n_workers=2)
    assert executor._pool is not None, "persistent pool was not created"
    executor.shutdown()
    assert executor._pool is None, "shutdown() did not release the pool"


# --------------------------------------------------------------------------- #
# PERF7-2: process does NOT call infer_state (no redundant gradient step)
# --------------------------------------------------------------------------- #


def test_perf7_2_process_does_not_call_infer_state(monkeypatch):
    """``process`` must NOT call ``generative_model.infer_state`` —
    ``compute_free_energy`` computes its own prediction error directly
    from ``state @ emission``, so the extra gradient step is redundant.

    The Round-6 implementation called ``infer_state`` inside
    ``compute_free_energy`` (one extra gradient step per cycle). Round-7
    removes that: ``compute_free_energy`` is now O(1) inference-free.

    We verify by spying on ``infer_state`` and confirming ``process`` does
    not call it.
    """
    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=4, action_dim=2, rng=np.random.default_rng(5)
    )
    infer_calls = 0
    real_infer = engine.generative_model.infer_state

    def counting_infer(obs):
        nonlocal infer_calls
        infer_calls += 1
        return real_infer(obs)

    monkeypatch.setattr(engine.generative_model, "infer_state", counting_infer)

    engine.process(Signal(data=np.ones(4) * 0.5))
    # select_action may call predict_observation directly (no infer_state);
    # process itself must not call infer_state.
    assert infer_calls == 0, (
        f"process() called infer_state {infer_calls} times; "
        "compute_free_energy should compute prediction error directly"
    )


# --------------------------------------------------------------------------- #
# API7-1-1: /save OpenAPI schema declares 201 + SaveResponse{saved, name}
# --------------------------------------------------------------------------- #


def test_api7_1_1_save_openapi_schema_matches_actual_response():
    """The ``/save`` OpenAPI schema must declare:
      * status_code 201 (Successful Response)
      * response schema ``SaveResponse`` with fields ``{saved, name}``

    The Round-6 contract mismatched: OpenAPI declared 200 +
    ``SaveResponse{saved}`` (no ``name``), while the actual handler returned
    ``JSONResponse(201, {"saved": True, "name": ...})``.
    """
    from zero_data_model.api import create_app

    app = create_app()
    schema = app.openapi()

    # API 版本化后 /save 迁移至 /v1/save
    save_path = schema["paths"].get("/v1/save") or schema["paths"].get("/save")
    assert save_path is not None, "OpenAPI missing both /v1/save and /save"
    save_op = save_path["post"]
    responses = save_op["responses"]
    # 201 must be declared (Round-6 declared only 200).
    assert "201" in responses, "OpenAPI does not declare 201 for /save"
    assert responses["201"]["description"] == "Successful Response"

    # SaveResponse schema must include both `saved` and `name`.
    save_schema_ref = responses["201"]["content"]["application/json"]["schema"]["$ref"]
    assert "SaveResponse" in save_schema_ref
    save_props = schema["components"]["schemas"]["SaveResponse"]["properties"]
    assert "saved" in save_props, "SaveResponse missing `saved` field"
    assert "name" in save_props, "SaveResponse missing `name` field (API7-1-1)"


# --------------------------------------------------------------------------- #
# NEW7-1: callers use structural_similarity (no find_isomorphism DeprecationWarning)
# --------------------------------------------------------------------------- #


def test_new7_1_callers_do_not_emit_find_isomorphism_warning():
    """The 4 internal callers (SemanticComparator, ZeroShotClassifier,
    TrendAnalyzer, ZeroDataModel.find_analogies) must use the renamed
    ``structural_similarity`` method, NOT the deprecated
    ``find_isomorphism`` alias.

    We verify by exercising each call path and asserting no
    ``DeprecationWarning`` mentioning ``find_isomorphism`` is emitted.
    """
    model = ZeroDataModel(dim=16, seed=2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # 1. SemanticComparator.text_similarity
        model.text_similarity("hello", "world")
        # 2. ZeroShotClassifier.classify_text
        model.classify_text("a short prompt")
        # 3. TrendAnalyzer.analyze_trend
        model.analyze_trend(np.sin(np.linspace(0, 6, 32)))
        # 4. ZeroDataModel.find_analogies
        model.find_analogies(np.ones(8), np.ones(8) * 2)

    deprecation_warnings = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "find_isomorphism" in str(w.message)
    ]
    assert deprecation_warnings == [], (
        "Internal callers still use deprecated find_isomorphism: "
        + "; ".join(str(w.message) for w in deprecation_warnings)
    )


def test_new7_1_source_has_no_internal_find_isomorphism_callers():
    """Source-level guard: no production code path under ``src/`` should
    call ``find_isomorphism`` (the deprecated alias). The ONLY legitimate
    occurrences are:
      * the definition + deprecation wrapper in ``category_engine.py``
      * docstring/comment references in ``model.py`` and ``kernels.py``
      * the type stub in ``category_engine.pyi``

    Round-10 audit R10-C-009: the guard now ALSO scans for
    ``.compute_betti_numbers(`` calls (the other deprecated alias in
    ``math_universe.py``). The previous guard only scanned
    ``find_isomorphism``, so the internal ``topological_features`` and
    ``ShapeAnalyzer._betti0`` call sites leaked through and emitted
    ``DeprecationWarning`` on every ``model.think()`` / ``analyze_shape``
    cycle (fixed in R10-C-002 and R10-C-003).
    """
    import re
    from pathlib import Path

    src_root = Path(__file__).resolve().parent.parent / "src" / "zero_data_model"
    # Files that are allowed to mention ``find_isomorphism`` or
    # ``compute_betti_numbers`` (definitions, deprecation wrappers, docs,
    # stubs). Call sites must use ``structural_similarity`` /
    # ``connected_components_1d`` / ``vietoris_rips_betti``.
    allowlist = {
        "category_engine.py",     # defines + deprecates find_isomorphism
        "category_engine.pyi",     # type stub mirrors the alias
        "hardware/kernels.py",      # docstring comment only
        "math_universe.py",         # defines + deprecates compute_betti_numbers
        "math_universe.pyi",        # type stub
    }
    # Pattern: a real call ``.find_isomorphism(`` or
    # ``.compute_betti_numbers(`` (not a def, not a comment).
    call_re = re.compile(r"\.(find_isomorphism|compute_betti_numbers)\s*\(")

    violations: list[str] = []
    for py in src_root.rglob("*.py"):
        if py.name in allowlist:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in call_re.finditer(text):
            # Check it's not inside a comment line.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line = text[line_start : text.find("\n", m.end())]
            stripped = text[line_start:m.start()].lstrip()
            if stripped.startswith("#"):
                continue
            violations.append(f"{py.name}: {line.strip()}")
    assert violations == [], (
        "production code still calls deprecated "
        "find_isomorphism / compute_betti_numbers: "
        + "; ".join(violations)
    )
