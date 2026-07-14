# tests/test_biological.py
import numpy as np

from zero_data_model.base import Signal
from zero_data_model.biological import (
    BiologicalSubstrate,
    CellularAutomata,
    DNAStorage,
    MorphogeneticField,
)


def test_dna_store_retrieve():
    dna = DNAStorage()
    data = np.array([1.0, 2.0, 3.0, 4.0])
    dna.store("test", data)
    retrieved = dna.retrieve("test")
    assert retrieved is not None
    np.testing.assert_allclose(retrieved, data, atol=0.3)


def test_dna_generate():
    dna = DNAStorage()
    dna.store("a", np.random.randn(16))
    dna.store("b", np.random.randn(16))
    result = dna.generate(Signal(data=np.zeros(16)))
    assert len(result.data) > 0


def test_morphogenetic_develop():
    mf = MorphogeneticField(grid_size=8)
    pattern = mf.develop(n_steps=10)
    assert pattern.shape == (8, 8)


def test_cellular_automata():
    ca = CellularAutomata(size=32)
    history = ca.evolve(n_steps=10)
    assert history.shape == (11, 32)


def test_biological_substrate_process():
    bio = BiologicalSubstrate(dim=16)
    signal = Signal(data=np.random.randn(16))
    result = bio.process(signal)
    assert result.data.shape == (16,)
