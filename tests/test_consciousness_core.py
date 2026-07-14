# tests/test_consciousness_core.py
import numpy as np

from zero_data_model.base import Signal
from zero_data_model.consciousness_core import ConsciousnessCore, GlobalWorkspace, SelfModel


def test_consciousness_core_process():
    core = ConsciousnessCore(dim=16, n_layers=2)
    signal = Signal(data=np.random.randn(16))
    result = core.process(signal)
    assert result.data.shape == (16,)
    assert "source" in result.metadata


def test_consciousness_core_predict():
    core = ConsciousnessCore(dim=16)
    signal = Signal(data=np.random.randn(16))
    pred = core.predict(signal)
    assert pred.value.shape == (16,)
    assert pred.uncertainty >= 0


def test_self_model_reflect():
    sm = SelfModel(dim=16)
    sm.update(np.random.randn(16))
    reflection = sm.reflect()
    assert reflection.data.shape == (16,)
    assert "self_confidence" in reflection.metadata


def test_global_workspace_broadcast():
    gw = GlobalWorkspace(dim=16)
    signal = Signal(data=np.random.randn(16))
    result = gw.broadcast(signal)
    assert result.data.shape == (16,)


def test_reflect():
    core = ConsciousnessCore(dim=16)
    reflection = core.reflect()
    assert reflection.data.shape == (16,)
