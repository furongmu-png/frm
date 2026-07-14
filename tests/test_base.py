# tests/test_base.py
import numpy as np
import pytest
from zero_data_model.base import Signal, Prediction, CognitiveModule, KnowledgeStore


def test_signal_creation():
    s = Signal(data=np.array([1.0, 2.0]))
    assert s.confidence == 1.0
    assert s.metadata == {}


def test_prediction_creation():
    p = Prediction(value=np.array([1.0]), uncertainty=0.5)
    assert p.uncertainty == 0.5


def test_abstract_base_classes():
    with pytest.raises(TypeError):
        CognitiveModule()
    with pytest.raises(TypeError):
        KnowledgeStore()
