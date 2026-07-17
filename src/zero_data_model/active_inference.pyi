from __future__ import annotations

from collections import deque
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
    def create(
        cls,
        sensory_dim: int = 32,
        active_dim: int = 16,
        internal_dim: int = 64,
        rng: np.random.Generator | None = None,
    ) -> MarkovBlanket: ...


class GenerativeModel:
    """Internal generative model — predicts sensory inputs from hidden states."""

    state_dim: int
    obs_dim: int
    transition: np.ndarray
    emission: np.ndarray
    belief_state: np.ndarray
    _last_state: np.ndarray | None
    _last_observation: np.ndarray | None
    _last_error: np.ndarray | None

    def __init__(
        self, state_dim: int = 64, obs_dim: int = 32, rng: np.random.Generator | None = None
    ) -> None: ...

    def predict_observation(self, state: np.ndarray) -> np.ndarray: ...

    def predict_next_state(
        self, state: np.ndarray, action: np.ndarray | None = None
    ) -> np.ndarray: ...

    def infer_state(self, observation: np.ndarray) -> tuple[np.ndarray, float]: ...

    def update_belief(self, observation: np.ndarray) -> tuple[np.ndarray, float]: ...

    def emission_gradient_step(self, lr: float) -> None: ...


class HomeostaticController:
    """Maintains internal equilibrium."""

    target: np.ndarray
    dim: int
    tolerance: float

    def __init__(self, dim: int = 64, target: np.ndarray | None = None) -> None: ...

    def deviation(self, state: np.ndarray) -> float: ...

    def regulate(self, state: np.ndarray) -> np.ndarray: ...


class ActiveInferenceEngine(CognitiveModule):
    """
    Active Inference Engine based on Free Energy Principle.
    - Minimizes variational free energy (prediction error + complexity)
    - Epistemic foraging: actively seeks information
    - Homeostatic regulation
    """

    blanket: MarkovBlanket
    generative_model: GenerativeModel
    homeostasis: HomeostaticController
    action_history: deque[np.ndarray]
    free_energy_history: deque[float]
    _cached_sigma_q2: float
    _sigma_q2_dirty: bool

    def __init__(
        self,
        state_dim: int = 64,
        obs_dim: int = 32,
        action_dim: int = 16,
        rng: np.random.Generator | None = None,
    ) -> None: ...

    def _compute_sigma_q2(self) -> float: ...

    def compute_free_energy(
        self, observation: np.ndarray, state: np.ndarray | None = None
    ) -> float:
        """F = KL(q || p) + prediction error (variational free energy).

        Round-7 audit THEORY7-2: ``state`` lets callers evaluate the free
        energy at an arbitrary state (e.g. a candidate next-state in
        ``select_action``) without mutating ``belief_state``. Defaults to
        ``belief_state`` when omitted so the read-only analytics path is
        unchanged.
        """
        ...

    def select_action(
        self,
        belief: np.ndarray,
        current_observation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select action that minimizes expected free energy.

        Round-8 audit THEORY8-1: ``current_observation`` lets the caller pass
        the CURRENT observation so the EFE's NLL term is non-degenerate
        (the previous loop passed the PREDICTED observation, which made
        ``error = 0`` inside ``compute_free_energy`` and silently disabled
        the active-inference driving signal).
        """
        ...

    def epistemic_foraging(self, belief: np.ndarray) -> Signal | None:
        """Actively seek information when uncertain."""
        ...

    def process(self, signal: Signal) -> Signal: ...

    def predict(self, signal: Signal) -> Prediction: ...

    def update(self, prediction_error: float) -> None: ...
