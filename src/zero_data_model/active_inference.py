# src/zero_data_model/active_inference.py
"""Active Inference Engine based on Free Energy Principle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import CognitiveModule, Prediction, Signal


@dataclass
class MarkovBlanket:
    """Defines the boundary between internal and external states."""
    sensory_dim: int
    active_dim: int
    internal_dim: int
    sensory_weights: np.ndarray
    active_weights: np.ndarray

    @classmethod
    def create(cls, sensory_dim: int = 32, active_dim: int = 16, internal_dim: int = 64):
        return cls(
            sensory_dim=sensory_dim,
            active_dim=active_dim,
            internal_dim=internal_dim,
            sensory_weights=np.random.randn(sensory_dim, internal_dim) * 0.1,
            active_weights=np.random.randn(internal_dim, active_dim) * 0.1,
        )


class GenerativeModel:
    """Internal generative model — predicts sensory inputs from hidden states."""

    def __init__(self, state_dim: int = 64, obs_dim: int = 32):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.transition = np.random.randn(state_dim, state_dim) * 0.05
        self.emission = np.random.randn(state_dim, obs_dim) * 0.1
        self.belief_state = np.zeros(state_dim)

    def predict_observation(self, state: np.ndarray) -> np.ndarray:
        return state @ self.emission

    def predict_next_state(self, state: np.ndarray, action: np.ndarray | None = None) -> np.ndarray:
        next_state = state @ self.transition
        if action is not None:
            padded = np.zeros(self.state_dim)
            padded[: len(action)] = action
            next_state += padded
        return next_state

    def infer_state(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        predicted_obs = self.predict_observation(self.belief_state)
        error = observation[: self.obs_dim] - predicted_obs[: len(observation)]
        if len(error) < self.obs_dim:
            error = np.pad(error, (0, self.obs_dim - len(error)))
        prediction_error = float(np.mean(error ** 2))
        gradient = error @ self.emission.T
        self.belief_state += 0.1 * gradient
        return self.belief_state.copy(), prediction_error


class HomeostaticController:
    """Maintains internal equilibrium."""

    def __init__(self, dim: int = 64, target: np.ndarray | None = None):
        self.target = target if target is not None else np.zeros(dim)
        self.dim = dim
        self.tolerance = 0.5

    def deviation(self, state: np.ndarray) -> float:
        s = state[: self.dim]
        if len(s) < self.dim:
            s = np.pad(s, (0, self.dim - len(s)))
        return float(np.linalg.norm(s - self.target))

    def regulate(self, state: np.ndarray) -> np.ndarray:
        s = state[: self.dim]
        if len(s) < self.dim:
            s = np.pad(s, (0, self.dim - len(s)))
        correction = (self.target - s) * 0.1
        return correction


class ActiveInferenceEngine(CognitiveModule):
    """
    Active Inference Engine based on Free Energy Principle.
    - Minimizes variational free energy (prediction error + complexity)
    - Epistemic foraging: actively seeks information
    - Homeostatic regulation
    """

    def __init__(self, state_dim: int = 64, obs_dim: int = 32, action_dim: int = 16):
        self.blanket = MarkovBlanket.create(obs_dim, action_dim, state_dim)
        self.generative_model = GenerativeModel(state_dim, obs_dim)
        self.homeostasis = HomeostaticController(state_dim)
        self.action_history: list[np.ndarray] = []
        self.free_energy_history: list[float] = []

    def compute_free_energy(self, observation: np.ndarray) -> float:
        """F = complexity - accuracy (variational free energy)."""
        _, pred_error = self.generative_model.infer_state(observation)
        complexity = float(np.linalg.norm(self.generative_model.belief_state) ** 2) * 0.01
        return pred_error + complexity

    def select_action(self, belief: np.ndarray) -> np.ndarray:
        """Select action that minimizes expected free energy."""
        best_action = None
        best_efep = float("inf")
        for _ in range(8):
            candidate = np.random.randn(self.blanket.active_dim) * 0.5
            predicted_state = self.generative_model.predict_next_state(belief, candidate)
            predicted_obs = self.generative_model.predict_observation(predicted_state)
            efe = self.compute_free_energy(predicted_obs)
            homeostatic_dev = self.homeostasis.deviation(predicted_state)
            total = efe + 0.1 * homeostatic_dev
            if total < best_efep:
                best_efep = total
                best_action = candidate
        return best_action if best_action is not None else np.zeros(self.blanket.active_dim)

    def epistemic_foraging(self, belief: np.ndarray) -> Signal | None:
        """Actively seek information when uncertain."""
        uncertainty = float(np.var(belief))
        if uncertainty > 0.1:
            exploration = np.random.randn(self.blanket.active_dim) * uncertainty
            return Signal(
                data=exploration,
                metadata={"type": "epistemic", "uncertainty": uncertainty},
            )
        return None

    def process(self, signal: Signal) -> Signal:
        belief, pred_error = self.generative_model.infer_state(signal.data)
        free_energy = pred_error + float(np.linalg.norm(belief) ** 2) * 0.01
        self.free_energy_history.append(free_energy)
        action = self.select_action(belief)
        self.action_history.append(action)
        correction = self.homeostasis.regulate(belief)
        output = belief + correction
        return Signal(data=output, metadata={"free_energy": free_energy, "action": action.tolist()})

    def predict(self, signal: Signal) -> Prediction:
        gm = self.generative_model
        predicted_obs = gm.predict_observation(gm.belief_state)
        return Prediction(value=predicted_obs, uncertainty=float(np.var(predicted_obs)))

    def update(self, prediction_error: float) -> None:
        noise = np.random.randn(*self.generative_model.emission.shape) * prediction_error * 0.001
        self.generative_model.emission += noise
