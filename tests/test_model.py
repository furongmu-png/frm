# tests/test_model.py
import numpy as np
from zero_data_model.model import ZeroDataModel
from zero_data_model.base import Signal


def test_model_creation():
    model = ZeroDataModel(dim=16)
    assert len(model.modules) == 6


def test_model_think_no_input():
    model = ZeroDataModel(dim=16)
    result = model.think()
    assert result.data.shape == (16,)
    assert result.metadata["cycle"] == 1


def test_model_think_with_input():
    model = ZeroDataModel(dim=16)
    result = model.think(np.random.randn(16))
    assert result.data.shape == (16,)


def test_model_solve():
    model = ZeroDataModel(dim=16)
    result = model.solve(np.random.randn(16))
    assert "energy" in result.metadata


def test_model_find_analogies():
    model = ZeroDataModel(dim=16)
    a = np.random.randn(16)
    score = model.find_analogies(a, a)
    assert abs(score - 1.0) < 1e-6


def test_model_generate_knowledge():
    model = ZeroDataModel(dim=16)
    knowledge = model.generate_knowledge()
    assert knowledge.data.shape[0] > 0


def test_model_multiple_cycles():
    model = ZeroDataModel(dim=16)
    for i in range(5):
        result = model.think()
        assert result.metadata["cycle"] == i + 1
