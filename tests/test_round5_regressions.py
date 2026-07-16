# tests/test_round5_regressions.py
"""Regression tests for the Round-5 audit fixes.

Each test guards one of the Round-5 audit findings against silent regression:

  TEST5-1/3    compute_free_energy returns the sentinel (not a sanitized
               small value) for NaN/Inf observation  [in test_round4_regressions]
  TEST5-2      update_belief rejects NaN/Inf observation without mutating
               belief_state
  PERSIST5-1   npz extra keys are rejected; dtype check runs AFTER key check
  PERSIST5-2   morph_counts upper bound + fresh-model comparison
  PERSIST5-3   shape checks for topos_classifier / classical_weights /
               morphogenetic_grid / automata_state / circuit_params
  PERSIST5-4   non-numeric dtype (string) is rejected
  PERSIST5-5   oversized config.json is rejected before parsing
  NEW5-1       Generator.spawn is available (guards numpy>=1.25 pin)
  RNG5-2       seeded long-run (200-cycle) reproducibility
  IBM5-1       JobError is a QiskitError subclass (comment factual accuracy)
  IBM5-3       TypeError dynamically propagates from the constructor's try block
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from zero_data_model.model import ZeroDataModel
from zero_data_model.persistence import ModelSerializer, set_persistence_root

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _sandbox_root(tmp_path):
    """Sandbox persistence under tmp_path for every test in this module."""
    root = tmp_path / "persistence"
    set_persistence_root(str(root))
    yield root
    set_persistence_root(str(tmp_path / "_reset"))


def _save_and_get_npz(model: ZeroDataModel, name: str) -> Path:
    """Save ``model`` and return the npz path for mutation."""
    ModelSerializer.save(model, name)
    from zero_data_model.persistence import _validate_path

    target = Path(_validate_path(name))
    return target / "arrays.npz"


def _save_and_get_config(model: ZeroDataModel, name: str) -> Path:
    """Save ``model`` and return the config.json path for mutation."""
    ModelSerializer.save(model, name)
    from zero_data_model.persistence import _validate_path

    target = Path(_validate_path(name))
    return target / "config.json"


def _rewrite_npz(npz_path: Path, mutator):
    """Read arrays.npz, apply ``mutator(dict) -> dict``, write it back."""
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays = mutator(arrays)
    np.savez(str(npz_path), **arrays)


def _rewrite_config(config_path: Path, mutator):
    """Read config.json, apply ``mutator(dict) -> dict``, write it back."""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = mutator(cfg)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)


# --------------------------------------------------------------------------- #
# TEST5-2: update_belief rejects NaN/Inf observation
# --------------------------------------------------------------------------- #


def test_test5_2_update_belief_rejects_nan_without_mutating_belief():
    """update_belief does NOT mutate belief_state for a NaN observation.

    Round-5 audit TEST5-2: ``update_belief`` is the learning path (called
    from ``process``). A NaN observation would permanently corrupt
    ``belief_state`` — every subsequent cycle inherits the NaN. The fix
    rejects the update entirely and returns the sentinel prediction_error.
    """
    from zero_data_model.active_inference import (
        _FREE_ENERGY_SENTINEL,
        ActiveInferenceEngine,
    )

    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=8, action_dim=4, rng=np.random.default_rng(0)
    )
    belief_before = engine.generative_model.belief_state.copy()
    nan_obs = np.full(8, np.nan)
    new_state, pred_error = engine.generative_model.update_belief(nan_obs)
    # belief_state must NOT have changed.
    assert np.array_equal(engine.generative_model.belief_state, belief_before), (
        "update_belief mutated belief_state despite a NaN observation"
    )
    # The returned prediction_error must be the sentinel.
    assert pred_error == _FREE_ENERGY_SENTINEL, (
        f"pred_error {pred_error} != sentinel; AnomalyDetector would not "
        "flag the anomaly"
    )
    # The returned state is the unchanged belief (copied, not mutated).
    assert np.array_equal(new_state, belief_before)


def test_test5_2_update_belief_rejects_inf_without_mutating_belief():
    """update_belief does NOT mutate belief_state for an Inf observation."""
    from zero_data_model.active_inference import (
        _FREE_ENERGY_SENTINEL,
        ActiveInferenceEngine,
    )

    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=8, action_dim=4, rng=np.random.default_rng(0)
    )
    belief_before = engine.generative_model.belief_state.copy()
    inf_obs = np.full(8, np.inf)
    _, pred_error = engine.generative_model.update_belief(inf_obs)
    assert np.array_equal(engine.generative_model.belief_state, belief_before)
    assert pred_error == _FREE_ENERGY_SENTINEL


def test_test5_2_update_belief_normal_observation_still_works():
    """A normal (finite) observation still updates belief_state.

    Ensures the NaN guard did not break the normal learning path.
    """
    from zero_data_model.active_inference import (
        _FREE_ENERGY_SENTINEL,
        ActiveInferenceEngine,
    )

    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=8, action_dim=4, rng=np.random.default_rng(0)
    )
    belief_before = engine.generative_model.belief_state.copy()
    obs = np.ones(8)
    _new_state, pred_error = engine.generative_model.update_belief(obs)
    # belief_state MUST have changed (normal path).
    assert not np.array_equal(engine.generative_model.belief_state, belief_before), (
        "update_belief did not mutate belief_state for a normal observation"
    )
    # pred_error should be finite and not the sentinel.
    assert np.isfinite(pred_error)
    assert pred_error != _FREE_ENERGY_SENTINEL


# --------------------------------------------------------------------------- #
# PERSIST5-1: npz extra keys are rejected
# --------------------------------------------------------------------------- #


def test_persist5_1_rejects_extra_npz_keys():
    """load() rejects an npz that contains unexpected keys.

    Round-5 audit PERSIST5-1: the old code only checked for MISSING keys.
    A hostile npz could stuff extra arrays; while they wouldn't be loaded,
    the dtype loop iterated ALL ``data.files`` (loading each array to
    inspect its dtype) before the key check — an OOM DoS vector.
    """
    model = ZeroDataModel(dim=8)
    npz = _save_and_get_npz(model, "snap")

    # Add a junk key to the npz.
    _rewrite_npz(npz, lambda a: {**a, "hostile_junk": np.zeros((999, 999))})
    with pytest.raises(ValueError, match="unexpected keys"):
        ModelSerializer.load("snap")


# --------------------------------------------------------------------------- #
# PERSIST5-2: morph_counts upper bound + fresh-model comparison
# --------------------------------------------------------------------------- #


def test_persist5_2_rejects_morph_count_above_cap():
    """load() rejects a functor_morphism_counts entry above the cap."""
    model = ZeroDataModel(dim=8)
    cfg = _save_and_get_config(model, "snap")
    with open(cfg, encoding="utf-8") as f:
        cfg_data = json.load(f)
    counts = list(cfg_data["functor_morphism_counts"])
    if not counts:
        pytest.skip("no functor morphism counts to mutate")
    counts[0] = 5000  # above _MAX_MORPH_COUNT (4096)
    _rewrite_config(cfg, lambda c: {**c, "functor_morphism_counts": counts})
    with pytest.raises(ValueError, match="exceeds cap"):
        ModelSerializer.load("snap")


def test_persist5_2_rejects_morph_count_mismatch_with_fresh_model():
    """load() rejects morph_counts that don't match the fresh model.

    Round-5 audit PERSIST5-2: the saved morph_counts must match the fresh
    model's functor morphism_map lengths, so a snapshot from a different
    model topology is rejected loudly (same pattern as n_consciousness_layers).
    """
    model = ZeroDataModel(dim=8)
    cfg = _save_and_get_config(model, "snap")
    with open(cfg, encoding="utf-8") as f:
        cfg_data = json.load(f)
    counts = list(cfg_data["functor_morphism_counts"])
    if not counts:
        pytest.skip("no functor morphism counts to mutate")
    # Set the first count to something != the fresh model's count. If the
    # fresh count is 0, use 1; otherwise use count+1.
    fresh_count = counts[0]
    counts[0] = fresh_count + 1 if fresh_count < 4096 else fresh_count - 1
    if counts[0] == fresh_count:
        pytest.skip("cannot construct a mismatch test (count at boundary)")
    _rewrite_config(cfg, lambda c: {**c, "functor_morphism_counts": counts})
    with pytest.raises(ValueError, match="fresh dim=.* model has"):
        ModelSerializer.load("snap")


# --------------------------------------------------------------------------- #
# PERSIST5-3: shape checks for arrays that previously lacked them
# --------------------------------------------------------------------------- #


def test_persist5_3_rejects_wrong_topos_classifier_shape():
    """load() rejects a topos_classifier with the wrong shape."""
    model = ZeroDataModel(dim=8)
    npz = _save_and_get_npz(model, "snap")
    _rewrite_npz(npz, lambda a: {**a, "category_engine_topos_classifier": np.zeros((1, 1))})
    with pytest.raises(ValueError, match="topos_classifier shape"):
        ModelSerializer.load("snap")


def test_persist5_3_rejects_wrong_classical_weights_shape():
    """load() rejects classical_weights with the wrong shape."""
    model = ZeroDataModel(dim=8)
    npz = _save_and_get_npz(model, "snap")
    _rewrite_npz(
        npz, lambda a: {**a, "quantum_hybrid_classical_weights": np.zeros((1, 1))}
    )
    with pytest.raises(ValueError, match="classical_weights shape"):
        ModelSerializer.load("snap")


def test_persist5_3_rejects_wrong_circuit_params_shape():
    """load() rejects quantum_circuit params with the wrong shape."""
    model = ZeroDataModel(dim=8)
    npz = _save_and_get_npz(model, "snap")
    _rewrite_npz(
        npz, lambda a: {**a, "quantum_hybrid_circuit_params": np.zeros((1, 1, 1))}
    )
    with pytest.raises(ValueError, match="circuit_params shape"):
        ModelSerializer.load("snap")


def test_persist5_3_rejects_wrong_morphogenetic_grid_shape():
    """load() rejects morphogenetic_grid with the wrong shape."""
    model = ZeroDataModel(dim=8)
    npz = _save_and_get_npz(model, "snap")
    _rewrite_npz(
        npz, lambda a: {**a, "biological_morphogenetic_grid": np.zeros((1, 1))}
    )
    with pytest.raises(ValueError, match="morphogenetic_grid shape"):
        ModelSerializer.load("snap")


def test_persist5_3_rejects_wrong_automata_state_shape():
    """load() rejects automata_state with the wrong shape."""
    model = ZeroDataModel(dim=8)
    npz = _save_and_get_npz(model, "snap")
    _rewrite_npz(
        npz, lambda a: {**a, "biological_automata_state": np.zeros((1,))}
    )
    with pytest.raises(ValueError, match="automata_state shape"):
        ModelSerializer.load("snap")


# --------------------------------------------------------------------------- #
# PERSIST5-4: non-numeric dtype is rejected
# --------------------------------------------------------------------------- #


def test_persist5_4_rejects_string_dtype_array():
    """load() rejects a string-dtype array (not just object-dtype).

    Round-5 audit PERSIST5-4: the old check only rejected dtype kind 'O'.
    String ('S'), unicode ('U') and void ('V') arrays could carry unexpected
    payloads. The fix broadens the rejection to all non-numeric kinds.
    """
    model = ZeroDataModel(dim=8)
    npz = _save_and_get_npz(model, "snap")
    # Replace a float array with a string array of the same key.
    _rewrite_npz(
        npz,
        lambda a: {
            **a,
            "quantum_hybrid_classical_weights": np.array(["junk", "data"]),
        },
    )
    with pytest.raises(ValueError, match="non-numeric dtype"):
        ModelSerializer.load("snap")


# --------------------------------------------------------------------------- #
# PERSIST5-5: oversized config.json is rejected
# --------------------------------------------------------------------------- #


def test_persist5_5_rejects_oversized_config():
    """load() rejects a config.json larger than _MAX_CONFIG_BYTES.

    Round-5 audit PERSIST5-5: a hostile multi-MB config could OOM the JSON
    parser before any validation runs.
    """
    model = ZeroDataModel(dim=8)
    cfg = _save_and_get_config(model, "snap")
    # Overwrite config.json with a file larger than 1 MiB.
    with open(cfg, "w", encoding="utf-8") as f:
        # Minimal valid JSON with a huge string value to exceed 1 MiB.
        f.write('{"dim": 8, "junk": "')
        f.write("x" * (1 << 20))  # 1 MiB of junk
        f.write('"}')
    with pytest.raises(ValueError, match="config.json too large"):
        ModelSerializer.load("snap")


# --------------------------------------------------------------------------- #
# NEW5-1: Generator.spawn is available (guards numpy>=1.25 pin)
# --------------------------------------------------------------------------- #


def test_new5_1_generator_spawn_is_available():
    """np.random.Generator.spawn exists (requires numpy>=1.25).

    Round-5 audit NEW5-1: ``ZeroDataModel.__init__`` calls
    ``self._rng.spawn(6)`` to give each cognitive module a bit-stream-
    independent child Generator. ``spawn`` was added in numpy 1.25; on
    1.24 the model raises ``AttributeError`` at construction. The
    pyproject.toml lower bound was bumped to ``>=1.25`` to match.
    """
    rng = np.random.default_rng(42)
    assert hasattr(rng, "spawn"), (
        "np.random.Generator.spawn is missing; numpy < 1.25 is installed "
        "despite the pyproject.toml >=1.25 pin"
    )
    children = rng.spawn(6)
    assert len(children) == 6
    # Children are independent bit-streams.
    for i in range(6):
        for j in range(i + 1, 6):
            assert children[i] is not children[j]


# --------------------------------------------------------------------------- #
# RNG5-2: seeded long-run (200-cycle) reproducibility
# --------------------------------------------------------------------------- #


def test_rng5_2_seeded_long_run_reproducibility():
    """Two seeded models produce identical think() output over 200 cycles.

    Round-5 audit RNG5-2: the existing test only checked 5 cycles, which
    is too short to catch BLAS thread drift or RNG state divergence over
    a realistic session. This test runs 200 cycles with a small dim (8)
    to keep it fast while exercising the spawn-children consumption order
    over a much longer horizon.
    """
    m1 = ZeroDataModel(dim=8, seed=999)
    m2 = ZeroDataModel(dim=8, seed=999)
    n_cycles = 200
    for i in range(n_cycles):
        a = m1.think().data
        b = m2.think().data
        assert np.allclose(a, b), (
            f"cycle {i} diverged in long-run seeded reproducibility test"
        )


# --------------------------------------------------------------------------- #
# IBM5-1: JobError is a QiskitError subclass (comment factual accuracy)
# --------------------------------------------------------------------------- #


def test_ibm5_1_joberror_is_qiskiterror_subclass():
    """JobError inherits from QiskitError (the comment was factually wrong).

    Round-5 audit IBM5-1: the Round-4 comment said JobError inherits "NOT
    from QiskitError", but ``JobError.__mro__`` shows it IS a QiskitError
    subclass. This test guards the factual accuracy of the exception
    hierarchy so a future qiskit refactoring that changes the hierarchy
    is caught.
    """
    from zero_data_model.hardware import ibm_quantum as ibm_mod

    # When qiskit-ibm-runtime IS installed, JobError is a real qiskit class.
    # When NOT installed, JobError is the sentinel (inherits Exception only).
    if ibm_mod._HAS_IBM_RUNTIME:
        from qiskit.exceptions import QiskitError

        assert issubclass(ibm_mod.JobError, QiskitError), (
            "JobError should be a QiskitError subclass (verified MRO: "
            f"{[c.__name__ for c in ibm_mod.JobError.__mro__]})"
        )
    else:
        pytest.skip("qiskit-ibm-runtime not installed; JobError is sentinel")


# --------------------------------------------------------------------------- #
# IBM5-3: TypeError dynamically propagates from the constructor's try block
# --------------------------------------------------------------------------- #


def test_ibm5_3_typeerror_dynamically_propagates(monkeypatch):
    """A TypeError raised inside the constructor's try block propagates.

    Round-5 audit IBM5-3: the existing Round-4 test only checked the
    exception hierarchy statically (``issubclass``). This test dynamically
    triggers a TypeError inside the try block and verifies it is NOT
    caught by the ``except (QiskitError, JobError, OSError, RuntimeError)``
    clause.
    """
    from zero_data_model.hardware import ibm_quantum as ibm_mod

    if not ibm_mod._HAS_IBM_RUNTIME:
        pytest.skip("qiskit-ibm-runtime not installed")

    # Remove any env token so only our explicit token is used.
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)

    def raise_typeerror(*args, **kwargs):
        raise TypeError("simulated programming bug")

    monkeypatch.setattr(ibm_mod, "QiskitRuntimeService", raise_typeerror)

    with pytest.raises(TypeError, match="simulated programming bug"):
        ibm_mod.IBMQuantumBackend(token="fake_token")


# --------------------------------------------------------------------------- #
# RNG5-1: unseeded model uses multiple workers when the environment supports it
# --------------------------------------------------------------------------- #


def test_rng5_1_unseeded_uses_multiple_workers_on_multicore():
    """In unseeded mode, the parallel executor uses > 1 worker on a multi-core
    machine, exercising the spawn-isolation fix under real concurrency.

    Round-5 audit RNG5-1: the existing ``test_rng1_unseeded_parallel_safe``
    only asserts ``n_workers >= 1``, which passes even on a single-core
    machine where no actual concurrency occurs. This test skips on
    single-core environments and verifies the spawn fix is exercised under
    real multi-worker parallelism.
    """
    m = ZeroDataModel(dim=8)  # unseeded -> n_workers=None -> cpu_count()
    if multiprocessing.cpu_count() <= 1:
        pytest.skip("single-core environment; cannot verify multi-worker concurrency")
    assert m.parallel_executor.n_workers > 1, (
        f"parallel_executor has {m.parallel_executor.n_workers} workers on a "
        f"{multiprocessing.cpu_count()}-core machine"
    )
    # Run several cycles to exercise the parallel path.
    for _ in range(5):
        out = m.think()
        assert np.all(np.isfinite(out.data)), "think() produced non-finite output"


# --------------------------------------------------------------------------- #
# IBM5-4: factory clamps n_qubits < 1 (does not propagate ValueError)
# --------------------------------------------------------------------------- #


def test_ibm5_4_factory_clamps_n_qubits_below_minimum():
    """get_quantum_backend(n_qubits=0) clamps to 1 instead of raising.

    Round-5 audit IBM5-4: the factory caught ``(ImportError, RuntimeError)``
    but not ``ValueError``. A call with ``n_qubits < 1`` would raise
    ``ValueError("n_qubits must be in [1, 20]")`` from the backend's
    ``__init__``, which propagated instead of degrading to the simulator.
    The fix clamps ``n_qubits < 1`` to 1 (consistent with the ``> 20``
    clamp).
    """
    import warnings

    from zero_data_model.hardware.quantum import get_quantum_backend

    # n_qubits=0 should clamp to 1 and return a working backend (not raise).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend = get_quantum_backend(n_qubits=0)
    assert backend.n_qubits == 1, f"expected n_qubits=1 after clamp, got {backend.n_qubits}"


# --------------------------------------------------------------------------- #
# TEST5-4: update_belief emits a warning when rejecting NaN
# --------------------------------------------------------------------------- #


def test_test5_4_update_belief_warns_on_nan_observation():
    """update_belief emits a UserWarning when it rejects a NaN observation.

    Round-5 audit TEST5-4: the NaN rejection was silent, making it hard
    for operators to see why the belief was not updating. The fix emits
    a warning so the rejection is observable.
    """
    from zero_data_model.active_inference import ActiveInferenceEngine

    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=8, action_dim=4, rng=np.random.default_rng(0)
    )
    nan_obs = np.full(8, np.nan)
    with pytest.warns(UserWarning, match="non-finite observation"):
        engine.generative_model.update_belief(nan_obs)

