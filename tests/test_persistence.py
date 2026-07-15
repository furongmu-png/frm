# tests/test_persistence.py
"""Tests for the numpy .npz + JSON model persistence layer."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from zero_data_model.model import ZeroDataModel
from zero_data_model.persistence import ModelSerializer, set_persistence_root


@pytest.fixture(autouse=True)
def _sandbox_root(tmp_path):
    """Sandbox persistence under tmp_path for every test in this module.

    The serializer now rejects absolute paths and ``..`` traversal, so each
    test must point the persistence root at a per-test temp dir and pass
    only a *relative* snapshot name to save()/load().
    """
    root = tmp_path / "persistence"
    set_persistence_root(str(root))
    yield root
    # Restore a clean default after the test so other modules aren't
    # affected by the redirect.
    set_persistence_root(str(tmp_path / "_reset"))


def _make_model(dim: int = 16, cycles: int = 3) -> ZeroDataModel:
    """Construct a small model and advance it a few cycles to bump state."""
    model = ZeroDataModel(dim=dim)
    for _ in range(cycles):
        model.think()
    return model


def test_save_load_roundtrip_preserves_dim_and_cycle_count(_sandbox_root):
    """save+load preserves dim and cycle_count."""
    model = _make_model(dim=16, cycles=3)
    name = "snap"

    ModelSerializer.save(model, name)
    loaded = ModelSerializer.load(name)

    assert loaded.dim == model.dim == 16
    assert loaded.cycle_count == model.cycle_count == 3


def test_save_load_preserves_think_result_shape(_sandbox_root):
    """After load, think() still returns a Signal of shape (dim,)."""
    model = _make_model(dim=16, cycles=2)
    name = "snap"

    ModelSerializer.save(model, name)
    loaded = ModelSerializer.load(name)

    signal = loaded.think()
    assert signal.data.shape == (16,)
    # The cycle counter advances past the pre-save value.
    assert signal.metadata["cycle"] == loaded.cycle_count == 3


def test_save_creates_arrays_npz_and_config_json(_sandbox_root):
    """save() writes both arrays.npz and config.json under the directory."""
    model = _make_model(dim=16, cycles=1)
    name = "snap"

    ModelSerializer.save(model, name)

    path = os.path.join(str(_sandbox_root), name)
    assert os.path.isdir(path)
    files = set(os.listdir(path))
    assert "arrays.npz" in files
    assert "config.json" in files

    # The npz must contain at least the listed numpy states.
    with np.load(os.path.join(path, "arrays.npz")) as data:
        keys = set(data.files)
    for required in (
        "consciousness_layers_0_weights",
        "active_inference_transition",
        "active_inference_emission",
        "active_inference_belief_state",
        "category_engine_topos_classifier",
        "quantum_hybrid_classical_weights",
        "quantum_hybrid_circuit_params",
        "quantum_hybrid_circuit_entangling",
        "quantum_hybrid_annealer_cost_matrix",
        "biological_morphogenetic_grid",
        "biological_automata_state",
        "math_universe_fractal_transforms_0_scale",
    ):
        assert required in keys, f"missing array key: {required}"

    # config.json must be valid JSON with the expected scalar fields.
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["dim"] == 16
    assert cfg["cycle_count"] == 1
    assert "quantum_backend_name" in cfg


def test_load_raises_clear_error_on_missing_name(_sandbox_root):
    """load() on a non-existent name raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ModelSerializer.load("does_not_exist")


def test_load_raises_when_npz_missing(_sandbox_root):
    """load() raises when the directory exists but arrays.npz is absent."""
    name = "partial"
    path = os.path.join(str(_sandbox_root), name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"dim": 16, "cycle_count": 0}, f)
    with pytest.raises(FileNotFoundError):
        ModelSerializer.load(name)


def test_save_load_preserves_numpy_state(_sandbox_root):
    """A representative numpy state is byte-identical after a round-trip."""
    model = _make_model(dim=16, cycles=2)
    name = "snap"

    ModelSerializer.save(model, name)
    loaded = ModelSerializer.load(name)

    np.testing.assert_array_equal(
        loaded.consciousness.layers[0].weights,
        model.consciousness.layers[0].weights,
    )
    np.testing.assert_array_equal(
        loaded.active_inference.generative_model.belief_state,
        model.active_inference.generative_model.belief_state,
    )
    np.testing.assert_array_equal(
        loaded.quantum_hybrid.annealer.cost_matrix,
        model.quantum_hybrid.annealer.cost_matrix,
    )
    np.testing.assert_array_equal(
        loaded.math_universe.fractal.transforms[0][0],
        model.math_universe.fractal.transforms[0][0],
    )


def test_save_load_preserves_full_state_across_subsystems(_sandbox_root):
    """Expanded round-trip: every documented subsystem array plus the scalar
    config fields (cycle_count, quantum_backend_name) survive save+load.

    Covers the gaps the original test left open:
      * consciousness biases for layers 1 and 2 (not just layer 0 weights)
      * active_inference transition AND emission matrices
      * quantum_hybrid classical_weights
      * biological morphogenetic_grid
      * math_universe fractal offsets (transforms[0][1] and transforms[1])
      * cycle_count and quantum_backend_name scalars
    """
    model = _make_model(dim=16, cycles=3)
    name = "full_snap"

    ModelSerializer.save(model, name)
    loaded = ModelSerializer.load(name)

    # --- Scalar config fields ---
    assert loaded.cycle_count == model.cycle_count == 3
    assert loaded.quantum_hybrid.quantum_backend_name == (
        model.quantum_hybrid.quantum_backend_name
    )

    # --- Consciousness biases for layers 1 and 2 (layer 0 weights already
    #     covered by test_save_load_preserves_numpy_state above). ---
    assert len(loaded.consciousness.layers) == len(model.consciousness.layers)
    for i in (1, 2):
        np.testing.assert_array_equal(
            loaded.consciousness.layers[i].bias,
            model.consciousness.layers[i].bias,
        )
        np.testing.assert_array_equal(
            loaded.consciousness.layers[i].weights,
            model.consciousness.layers[i].weights,
        )

    # --- Active inference transition and emission matrices. ---
    gm_orig = model.active_inference.generative_model
    gm_load = loaded.active_inference.generative_model
    np.testing.assert_array_equal(gm_load.transition, gm_orig.transition)
    np.testing.assert_array_equal(gm_load.emission, gm_orig.emission)
    np.testing.assert_array_equal(gm_load.belief_state, gm_orig.belief_state)

    # --- Quantum hybrid classical_weights (circuit params/entangling and
    #     annealer cost_matrix are covered by the npz-key test above). ---
    np.testing.assert_array_equal(
        loaded.quantum_hybrid.classical_weights,
        model.quantum_hybrid.classical_weights,
    )
    np.testing.assert_array_equal(
        loaded.quantum_hybrid.quantum_circuit.params,
        model.quantum_hybrid.quantum_circuit.params,
    )
    np.testing.assert_array_equal(
        loaded.quantum_hybrid.quantum_circuit.entangling,
        model.quantum_hybrid.quantum_circuit.entangling,
    )

    # --- Biological morphogenetic_grid (and automata state). ---
    np.testing.assert_array_equal(
        loaded.biological.morphogenetic.grid,
        model.biological.morphogenetic.grid,
    )
    np.testing.assert_array_equal(
        loaded.biological.automata.state,
        model.biological.automata.state,
    )

    # --- Math universe fractal offsets (transforms[0][1]) and a second
    #     transform pair (transforms[1]) to catch index-mixing bugs. ---
    fractal_orig = model.math_universe.fractal.transforms
    fractal_load = loaded.math_universe.fractal.transforms
    assert len(fractal_load) == len(fractal_orig)
    # transforms[0] offset (index 1 of the (scale, offset) tuple).
    np.testing.assert_array_equal(fractal_load[0][1], fractal_orig[0][1])
    # Full second transform pair (scale + offset).
    np.testing.assert_array_equal(fractal_load[1][0], fractal_orig[1][0])
    np.testing.assert_array_equal(fractal_load[1][1], fractal_orig[1][1])

    # --- Category engine topos classifier. ---
    np.testing.assert_array_equal(
        loaded.category_engine.topos.classifier,
        model.category_engine.topos.classifier,
    )


@pytest.mark.parametrize(
    "bad_name",
    [
        "/etc/passwd",                 # absolute path
        "../escape",                   # parent traversal
        "good/../../bad",              # nested parent traversal
        "",                            # empty
    ],
)
def test_save_rejects_paths_outside_root(_sandbox_root, bad_name):
    """save() rejects absolute / traversal paths with ValueError."""
    model = _make_model(dim=16, cycles=1)
    with pytest.raises(ValueError, match="path outside persistence root"):
        ModelSerializer.save(model, bad_name)


@pytest.mark.parametrize(
    "bad_name",
    [
        "/etc/passwd",
        "../escape",
        "good/../../bad",
    ],
)
def test_load_rejects_paths_outside_root(_sandbox_root, bad_name):
    """load() rejects absolute / traversal paths with ValueError."""
    with pytest.raises(ValueError, match="path outside persistence root"):
        ModelSerializer.load(bad_name)


def test_save_is_atomic_overwrites_existing(_sandbox_root):
    """Re-saving the same name replaces the snapshot atomically."""
    model = _make_model(dim=16, cycles=1)
    ModelSerializer.save(model, "snap")
    # Advance and re-save; the on-disk cycle_count must reflect the new state.
    model.think()
    model.think()
    ModelSerializer.save(model, "snap")
    loaded = ModelSerializer.load("snap")
    assert loaded.cycle_count == model.cycle_count == 3
