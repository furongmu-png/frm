# tests/test_active_inference.py
import numpy as np
from zero_data_model.active_inference import ActiveInferenceEngine, MarkovBlanket, GenerativeModel
from zero_data_model.base import Signal


def test_active_inference_process():
    engine = ActiveInferenceEngine(state_dim=16, obs_dim=8, action_dim=4)
    signal = Signal(data=np.random.randn(8))
    result = engine.process(signal)
    assert result.data.shape == (16,)
    assert "free_energy" in result.metadata


def test_free_energy_decreases():
    np.random.seed(42)
    engine = ActiveInferenceEngine(state_dim=16, obs_dim=8, action_dim=4)
    fixed_input = Signal(data=np.ones(8) * 0.5)
    energies = []
    for _ in range(20):
        engine.process(fixed_input)
        energies.append(engine.free_energy_history[-1])
    assert energies[-1] <= energies[0] + 1.0


def test_epistemic_foraging():
    engine = ActiveInferenceEngine(state_dim=16, obs_dim=8, action_dim=4)
    high_uncertainty = np.random.randn(16) * 10
    result = engine.epistemic_foraging(high_uncertainty)
    assert result is not None
    assert result.metadata["type"] == "epistemic"


def test_markov_blanket_creation():
    mb = MarkovBlanket.create(8, 4, 16)
    assert mb.sensory_weights.shape == (8, 16)
    assert mb.active_weights.shape == (16, 4)


def test_generative_model_infer():
    gm = GenerativeModel(state_dim=16, obs_dim=8)
    obs = np.random.randn(8)
    belief, error = gm.infer_state(obs)
    assert belief.shape == (16,)
    assert error >= 0
