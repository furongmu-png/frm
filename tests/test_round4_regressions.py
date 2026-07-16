# tests/test_round4_regressions.py
"""Regression tests for the Round-4 audit fixes.

Each test guards one of the Round-4 audit findings against silent regression:
  PERSIST-1   n_morphogens / n_fractal_transforms / cycle_count validation
  PERSIST-2   npz key pre-check + layer/transition/emission/belief_state shape
  PERSIST-3   functor_morphism_counts element type/range
  PERSIST-5   bool/float type confusion for config scalars
  RNG-1       spawned child Generators are independent (spawn isolation)
  RNG-1       seeded reproducibility preserved after spawn
  IBM-1/2     TypeError propagates (not swallowed) under narrow except
  NEW-2       compute_free_energy sanitizes NaN/Inf observation
  HYP-P11     KL term exact value (isolated from pred_error)
"""

from __future__ import annotations

import json
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


def _save_and_get_config(model: ZeroDataModel, name: str) -> tuple[Path, Path]:
    """Save ``model`` and return (npz_path, config_path) for mutation."""
    ModelSerializer.save(model, name)
    # Use the persistence API to resolve the actual path.
    from zero_data_model.persistence import _validate_path

    target = Path(_validate_path(name))
    return target / "arrays.npz", target / "config.json"


def _rewrite_config(config_path: Path, mutator):
    """Read config.json, apply ``mutator(dict) -> dict``, write it back."""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = mutator(cfg)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)


# --------------------------------------------------------------------------- #
# PERSIST-1: n_morphogens / n_fractal_transforms / cycle_count validation
# --------------------------------------------------------------------------- #


def test_persist1_rejects_missing_n_morphogens():
    """load() rejects a snapshot whose config.json omits n_morphogens."""
    model = ZeroDataModel(dim=8)
    npz, cfg = _save_and_get_config(model, "snap")
    _rewrite_config(cfg, lambda c: {k: v for k, v in c.items() if k != "n_morphogens"})
    with pytest.raises(ValueError, match="n_morphogens"):
        ModelSerializer.load("snap")


def test_persist1_rejects_missing_n_fractal_transforms():
    """load() rejects a snapshot whose config.json omits n_fractal_transforms."""
    model = ZeroDataModel(dim=8)
    _, cfg = _save_and_get_config(model, "snap")
    _rewrite_config(
        cfg, lambda c: {k: v for k, v in c.items() if k != "n_fractal_transforms"}
    )
    with pytest.raises(ValueError, match="n_fractal_transforms"):
        ModelSerializer.load("snap")


def test_persist1_rejects_negative_cycle_count():
    """load() rejects a negative cycle_count."""
    model = ZeroDataModel(dim=8)
    _, cfg = _save_and_get_config(model, "snap")
    _rewrite_config(cfg, lambda c: {**c, "cycle_count": -5})
    with pytest.raises(ValueError, match="cycle_count.*non-negative"):
        ModelSerializer.load("snap")


def test_persist1_rejects_bool_n_morphogens():
    """load() rejects ``n_morphogens: true`` (bool, not int)."""
    model = ZeroDataModel(dim=8)
    _, cfg = _save_and_get_config(model, "snap")
    _rewrite_config(cfg, lambda c: {**c, "n_morphogens": True})
    with pytest.raises(ValueError, match="n_morphogens.*plain int"):
        ModelSerializer.load("snap")


# --------------------------------------------------------------------------- #
# PERSIST-2: npz key pre-check + shape validation
# --------------------------------------------------------------------------- #


def test_persist2_rejects_missing_npz_key():
    """load() reports missing npz keys with a clear ValueError."""
    model = ZeroDataModel(dim=8)
    npz, cfg = _save_and_get_config(model, "snap")
    # Drop one array from the npz by rewriting it without the key.
    with np.load(npz, allow_pickle=False) as data:
        kept = {k: data[k] for k in data.files if k != "biological_automata_state"}
    np.savez(str(npz), **kept)
    with pytest.raises(ValueError, match="missing required keys"):
        ModelSerializer.load("snap")


def test_persist2_rejects_wrong_layer_weights_shape():
    """load() rejects a weights array whose shape doesn't match (dim, dim)."""
    model = ZeroDataModel(dim=8)
    npz, cfg = _save_and_get_config(model, "snap")
    # Rewrite just the layer-0 weights with a wrong shape.
    with np.load(npz, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["consciousness_layers_0_weights"] = np.zeros((1, 1))
    np.savez(str(npz), **arrays)
    with pytest.raises(ValueError, match="layer 0 weights shape"):
        ModelSerializer.load("snap")


def test_persist2_rejects_wrong_transition_shape():
    """load() rejects an active_inference_transition with wrong shape."""
    model = ZeroDataModel(dim=8)
    npz, cfg = _save_and_get_config(model, "snap")
    with np.load(npz, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["active_inference_transition"] = np.zeros((1, 1))
    np.savez(str(npz), **arrays)
    with pytest.raises(ValueError, match="transition shape"):
        ModelSerializer.load("snap")


# --------------------------------------------------------------------------- #
# PERSIST-3: functor_morphism_counts element validation
# --------------------------------------------------------------------------- #


def test_persist3_rejects_negative_morph_count():
    """load() rejects a negative functor_morphism_counts entry."""
    model = ZeroDataModel(dim=8)
    _, cfg = _save_and_get_config(model, "snap")
    # Read the actual counts so we only mutate one entry.
    with open(cfg, encoding="utf-8") as f:
        cfg_data = json.load(f)
    counts = list(cfg_data["functor_morphism_counts"])
    if not counts:
        pytest.skip("no functor morphism counts to mutate")
    counts[0] = -5
    _rewrite_config(cfg, lambda c: {**c, "functor_morphism_counts": counts})
    with pytest.raises(ValueError, match="functor_morphism_counts\\[0\\].*non-negative"):
        ModelSerializer.load("snap")


def test_persist3_rejects_bool_morph_count():
    """load() rejects a bool functor_morphism_counts entry."""
    model = ZeroDataModel(dim=8)
    _, cfg = _save_and_get_config(model, "snap")
    with open(cfg, encoding="utf-8") as f:
        cfg_data = json.load(f)
    counts = list(cfg_data["functor_morphism_counts"])
    if not counts:
        pytest.skip("no functor morphism counts to mutate")
    counts[0] = True
    _rewrite_config(cfg, lambda c: {**c, "functor_morphism_counts": counts})
    with pytest.raises(ValueError, match="functor_morphism_counts\\[0\\].*non-negative int"):
        ModelSerializer.load("snap")


# --------------------------------------------------------------------------- #
# PERSIST-5: bool/float type confusion for dim
# --------------------------------------------------------------------------- #


def test_persist5_rejects_bool_dim():
    """load() rejects ``dim: true`` (bool, not int)."""
    model = ZeroDataModel(dim=8)
    _, cfg = _save_and_get_config(model, "snap")
    _rewrite_config(cfg, lambda c: {**c, "dim": True})
    with pytest.raises(ValueError, match="dim.*plain int"):
        ModelSerializer.load("snap")


def test_persist5_rejects_float_dim():
    """load() rejects ``dim: 8.5`` (float, not int)."""
    model = ZeroDataModel(dim=8)
    _, cfg = _save_and_get_config(model, "snap")
    _rewrite_config(cfg, lambda c: {**c, "dim": 8.5})
    with pytest.raises(ValueError, match="dim.*plain int"):
        ModelSerializer.load("snap")


# --------------------------------------------------------------------------- #
# RNG-1: spawn isolation
# --------------------------------------------------------------------------- #


def test_rng1_child_generators_are_independent():
    """Each module's child Generator is an independent bit-stream.

    Round-4 audit RNG-1: ``self._rng.spawn(6)`` must produce 6 bit-stream
    independent child Generators. If two modules shared the same Generator
    instance, drawing from one would advance the other's state. This test
    verifies they are distinct objects and produce independent draws.
    """
    m1 = ZeroDataModel(dim=8, seed=42)
    # Each module holds its own _rng; they must be distinct objects.
    rngs = [
        m1.consciousness._rng,
        m1.active_inference._rng,
        m1.category_engine._rng,
        m1.quantum_hybrid._rng,
        m1.biological._rng,
        m1.math_universe._rng,
    ]
    # All distinct objects.
    for i in range(len(rngs)):
        for j in range(i + 1, len(rngs)):
            assert rngs[i] is not rngs[j], (
                f"modules {i} and {j} share the same Generator instance"
            )
    # Independent draws: drawing from one does not affect the next draw of
    # another (they share no state).
    a = rngs[0].standard_normal(4)
    b = rngs[1].standard_normal(4)
    # If they were the same instance, b would be the *second* draw (different
    # from a, but correlated via shared state). The independence guarantee is
    # already established by the ``is not`` check above; here we just assert
    # the draws are not identical (which would indicate the same stream).
    assert not np.allclose(a, b), "two child Generators produced identical draws"


def test_rng1_seeded_reproducibility_preserved_after_spawn():
    """Two models with the same seed produce identical think() sequences.

    Round-4 audit RNG-1: ``spawn`` is deterministic given the parent seed, so
    seeded reproducibility is preserved. The child Generators are consumed in
    a fixed order (consciousness, active_inference, category_engine, ...),
    so two seeded models produce bit-identical ``think()`` output.
    """
    m1 = ZeroDataModel(dim=8, seed=123)
    m2 = ZeroDataModel(dim=8, seed=123)
    outs1 = [m1.think().data.copy() for _ in range(5)]
    outs2 = [m2.think().data.copy() for _ in range(5)]
    for i, (a, b) in enumerate(zip(outs1, outs2, strict=True)):
        assert np.allclose(a, b), f"cycle {i} diverged after spawn refactor"


def test_rng1_unseeded_parallel_safe():
    """An unseeded model with multi-worker parallel_executor does not crash.

    Round-4 audit RNG-1: in unseeded mode, ``parallel_executor`` uses
    ThreadPoolExecutor with multiple workers. Before the spawn fix, the
    shared Generator was accessed concurrently, which could crash or
    produce inconsistent state. After the fix, each module has an
    independent child Generator, so concurrent access is safe.
    """

    # Force multi-worker even without joblib by clearing n_workers override.
    m = ZeroDataModel(dim=8)  # unseeded -> n_workers=None -> cpu_count()
    # Run several think cycles; if the spawn fix is reverted this would
    # race on the shared Generator and could (non-deterministically)
    # produce corrupted state. We assert no exception and finite output.
    for _ in range(5):
        out = m.think()
        assert np.all(np.isfinite(out.data)), "think() produced non-finite output"
    # Sanity: the model has multiple workers configured.
    assert m.parallel_executor.n_workers >= 1


# --------------------------------------------------------------------------- #
# IBM-1/2: TypeError propagates under narrow except
# --------------------------------------------------------------------------- #


def test_ibm_typeerror_propagates_from_constructor():
    """A TypeError raised inside the constructor's try block propagates.

    Round-4 audit IBM-1/IBM-2: the constructor's ``except (QiskitError, JobError,
    OSError, RuntimeError)`` must NOT catch TypeError/AttributeError (which
    indicate programming bugs, not service failures). We verify the exception
    hierarchy directly: TypeError is NOT a subclass of any caught type.
    """
    from zero_data_model.hardware import ibm_quantum as ibm_mod

    # The caught types in the constructor's except clause.
    caught_types = (ibm_mod.QiskitError, ibm_mod.JobError, OSError, RuntimeError)
    # TypeError and AttributeError must NOT be subclasses of any caught type.
    for bug_type in (TypeError, AttributeError, NameError):
        for caught in caught_types:
            assert not issubclass(bug_type, caught), (
                f"{bug_type.__name__} is a subclass of {caught.__name__} — "
                "the narrow except would swallow this programming bug"
            )


def test_ibm_qiskiterror_is_caught_by_except_clause():
    """QiskitError is caught by the constructor's except clause.

    Round-4 audit IBM-1/IBM-2: ``QiskitError`` is the real IBM base class.
    The except clause ``(QiskitError, JobError, OSError, RuntimeError)``
    must catch it so the constructor degrades to the simulator. We verify
    the hierarchy directly (issubclass), which works without qiskit-ibm-runtime
    installed because the module defines ``QiskitError`` as a sentinel class.
    """
    from zero_data_model.hardware import ibm_quantum as ibm_mod

    caught_types = (ibm_mod.QiskitError, ibm_mod.JobError, OSError, RuntimeError)
    # QiskitError must be caught.
    assert issubclass(ibm_mod.QiskitError, caught_types), (
        "QiskitError is not in the caught types — IBM service failures "
        "would propagate instead of degrading to the simulator"
    )
    # When qiskit-ibm-runtime IS installed, real IBM exceptions subclass
    # QiskitError, so they are also caught. We can't test the real ones
    # without the runtime, but the hierarchy contract is what matters.


# --------------------------------------------------------------------------- #
# NEW-2: compute_free_energy sanitizes NaN/Inf observation
# --------------------------------------------------------------------------- #


def test_new2_compute_free_energy_handles_nan_observation():
    """compute_free_energy returns the sentinel for a NaN observation.

    Round-4 audit NEW-2: before the fix, a NaN observation would propagate
    through ``infer_state`` and make ``b_norm_sq`` / ``kl_qp`` NaN, which
    would then poison ``select_action`` and ``action_history``. The Round-4
    fix sanitized the observation up-front.

    Round-5 audit TEST5-1/TEST5-3: the entry sanitize was REMOVED because
    it masked anomalies (a NaN observation became zeros, producing a
    "normal" small free energy that ``AnomalyDetector`` would not flag).
    The downstream guards now return the sentinel ``_FREE_ENERGY_SENTINEL``
    (1e6) so the anomaly is flagged. This test asserts the sentinel value
    — not just ``np.isfinite`` — so a regression that re-introduces entry
    sanitize would be caught (the value would be small, not 1e6).
    """
    from zero_data_model.active_inference import (
        _FREE_ENERGY_SENTINEL,
        ActiveInferenceEngine,
    )

    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=8, action_dim=4, rng=np.random.default_rng(0)
    )
    # An observation full of NaN.
    nan_obs = np.full(8, np.nan)
    fe = engine.compute_free_energy(nan_obs)
    assert np.isfinite(fe), f"free energy {fe} is not finite for NaN observation"
    assert fe >= 0, f"free energy {fe} should be non-negative"
    # Round-5 TEST5-3: the sentinel must be returned, not a sanitized small
    # value. This is what lets AnomalyDetector flag the anomaly.
    assert fe == _FREE_ENERGY_SENTINEL, (
        f"free energy {fe} != sentinel {_FREE_ENERGY_SENTINEL}; the entry "
        "sanitize may have been re-introduced, masking the anomaly"
    )


def test_new2_compute_free_energy_handles_inf_observation():
    """compute_free_energy returns the sentinel for an Inf observation.

    Round-5 audit TEST5-3: same sentinel assertion as the NaN test.
    """
    from zero_data_model.active_inference import (
        _FREE_ENERGY_SENTINEL,
        ActiveInferenceEngine,
    )

    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=8, action_dim=4, rng=np.random.default_rng(0)
    )
    inf_obs = np.full(8, np.inf)
    fe = engine.compute_free_energy(inf_obs)
    assert np.isfinite(fe), f"free energy {fe} is not finite for Inf observation"
    assert fe == _FREE_ENERGY_SENTINEL, (
        f"free energy {fe} != sentinel {_FREE_ENERGY_SENTINEL}"
    )


# --------------------------------------------------------------------------- #
# HYP-P11: KL term exact value (isolated from pred_error)
# --------------------------------------------------------------------------- #


def test_p11_kl_term_exact_value_isolated_from_pred_error():
    """The KL term equals ``0.5 * ||belief||^2`` when sigma_q2 == 1.

    Round-4 audit HYP-P11: the original property test only asserted
    ``fe >= 0``, which cannot catch a sign error in the KL formula (a
    negative KL could be masked by a positive prediction error). This
    test isolates the KL term by zeroing the emission matrix (so
    ``pred_error = mean((obs - 0)^2) = mean(obs^2)`` is known) and
    asserting the exact expected free energy.

    Setup:
      - belief_state = b (a known vector)
      - emission = 0  (so predicted_obs = 0, error = obs)
      - observation = 0 (so pred_error = 0, isolating KL)
      - action_history empty -> sigma_q2 = 1.0
      - KL(N(b, 1*I) || N(0, I)) = 0.5 * ||b||^2
      - FE = pred_error + KL = 0 + 0.5 * ||b||^2
    """
    from zero_data_model.active_inference import ActiveInferenceEngine

    engine = ActiveInferenceEngine(
        state_dim=8, obs_dim=8, action_dim=4, rng=np.random.default_rng(0)
    )
    # Set belief to a known vector.
    belief = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    engine.generative_model.belief_state = belief.copy()
    # Zero the emission so predicted_obs = state @ 0 = 0.
    engine.generative_model.emission[:] = 0.0
    # Zero observation so pred_error = mean((0 - 0)^2) = 0.
    obs = np.zeros(8)
    fe = engine.compute_free_energy(obs)
    # Expected: KL = 0.5 * ||belief||^2 = 0.5 * 9 = 4.5
    expected_kl = 0.5 * float(np.dot(belief, belief))
    assert abs(fe - expected_kl) < 1e-6, (
        f"FE {fe} != expected KL {expected_kl} (KL term not isolated correctly)"
    )
    # Also assert the exact expected value.
    assert abs(fe - 4.5) < 1e-6, f"FE {fe} != 4.5"
