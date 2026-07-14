# tests/test_persistence.py
"""Tests for the numpy .npz + JSON model persistence layer."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from zero_data_model.model import ZeroDataModel
from zero_data_model.persistence import ModelSerializer


def _make_model(dim: int = 16, cycles: int = 3) -> ZeroDataModel:
    """Construct a small model and advance it a few cycles to bump state."""
    model = ZeroDataModel(dim=dim)
    for _ in range(cycles):
        model.think()
    return model


def test_save_load_roundtrip_preserves_dim_and_cycle_count(tmp_path):
    """save+load preserves dim and cycle_count."""
    model = _make_model(dim=16, cycles=3)
    path = str(tmp_path / "snap")

    ModelSerializer.save(model, path)
    loaded = ModelSerializer.load(path)

    assert loaded.dim == model.dim == 16
    assert loaded.cycle_count == model.cycle_count == 3


def test_save_load_preserves_think_result_shape(tmp_path):
    """After load, think() still returns a Signal of shape (dim,)."""
    model = _make_model(dim=16, cycles=2)
    path = str(tmp_path / "snap")

    ModelSerializer.save(model, path)
    loaded = ModelSerializer.load(path)

    signal = loaded.think()
    assert signal.data.shape == (16,)
    # The cycle counter advances past the pre-save value.
    assert signal.metadata["cycle"] == loaded.cycle_count == 3


def test_save_creates_arrays_npz_and_config_json(tmp_path):
    """save() writes both arrays.npz and config.json under the directory."""
    model = _make_model(dim=16, cycles=1)
    path = str(tmp_path / "snap")

    ModelSerializer.save(model, path)

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


def test_load_raises_clear_error_on_missing_path(tmp_path):
    """load() on a non-existent path raises FileNotFoundError."""
    missing = str(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        ModelSerializer.load(missing)


def test_load_raises_when_npz_missing(tmp_path):
    """load() raises when the directory exists but arrays.npz is absent."""
    path = str(tmp_path / "partial")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"dim": 16, "cycle_count": 0}, f)
    with pytest.raises(FileNotFoundError):
        ModelSerializer.load(path)


def test_save_load_preserves_numpy_state(tmp_path):
    """A representative numpy state is byte-identical after a round-trip."""
    model = _make_model(dim=16, cycles=2)
    path = str(tmp_path / "snap")

    ModelSerializer.save(model, path)
    loaded = ModelSerializer.load(path)

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
