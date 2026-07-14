# tests/test_category_engine.py
import numpy as np

from zero_data_model.base import Signal
from zero_data_model.category_engine import Category, CategoryTheoryEngine


def test_category_creation():
    cat = Category(name="Test")
    cat.add_object("a", np.array([1.0, 2.0]))
    assert "a" in cat.objects


def test_find_isomorphism():
    engine = CategoryTheoryEngine(dim=16)
    a = np.random.randn(16)
    similarity = engine.find_isomorphism(a, a)
    assert abs(similarity - 1.0) < 1e-6


def test_process():
    engine = CategoryTheoryEngine(dim=16)
    signal = Signal(data=np.random.randn(16))
    result = engine.process(signal)
    assert result.data.shape[0] > 0


def test_predict():
    engine = CategoryTheoryEngine(dim=16)
    signal = Signal(data=np.random.randn(16))
    pred = engine.predict(signal)
    assert pred.value.shape == (16,)
