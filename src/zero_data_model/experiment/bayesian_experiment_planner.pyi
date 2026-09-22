from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

class SandboxInterface(Protocol):
    def apply_intervention(
        self, intervention_type: str, params: dict[str, Any]
    ) -> dict[str, Any]: ...

    def get_state_distribution(self) -> np.ndarray: ...


@dataclass
class CandidateIntervention:
    name: str
    intervention_type: str
    params: dict[str, Any] = field(default_factory=dict)
    predicted_info_gain: float = 0.0
    executed: bool = False
    actual_info_gain: float = 0.0


class BayesianExperimentPlannerV2:
    candidates: deque[CandidateIntervention]
    _eval_interval: int
    _n_samples: int
    _rng: np.random.Generator
    _lock: threading.RLock
    _history: deque[dict[str, Any]]
    _step_count: int
    _sandbox: SandboxInterface | None

    def __init__(
        self,
        eval_interval: int = 1000,
        n_samples: int = 10,
        seed: int = 42,
        max_candidates: int = 256,
        max_history: int = 1024,
    ) -> None: ...

    def attach_sandbox(self, sandbox: SandboxInterface) -> None: ...

    def detach_sandbox(self) -> None: ...

    def register_candidate(
        self,
        name: str,
        intervention_type: str,
        params: dict[str, Any] | None = None,
    ) -> None: ...

    def default_candidates(self, dim: int = 64) -> None: ...

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float: ...

    def estimate_info_gain(
        self,
        candidate: CandidateIntervention,
        current_state: np.ndarray,
        world_model_step: Any | None = None,
    ) -> float: ...

    def _simulate_intervention(
        self,
        current_state: np.ndarray,
        candidate: CandidateIntervention,
        natural_next: np.ndarray,
    ) -> np.ndarray: ...

    def select_best(
        self,
        current_state: np.ndarray,
        world_model_step: Any | None = None,
    ) -> CandidateIntervention | None: ...

    def execute(
        self,
        candidate: CandidateIntervention,
        current_state: np.ndarray,
    ) -> dict[str, Any]: ...

    def evaluate(
        self,
        current_state: np.ndarray,
        step: int,
        world_model_step: Any | None = None,
    ) -> dict[str, Any] | None: ...

    @property
    def stats(self) -> dict[str, Any]: ...


__all__: list[str] = [
    "BayesianExperimentPlannerV2",
    "CandidateIntervention",
    "SandboxInterface",
]
