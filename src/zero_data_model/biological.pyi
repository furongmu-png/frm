from __future__ import annotations

import numpy as np

from .base import CognitiveModule, KnowledgeStore, Prediction, Signal

class DNAStorage(KnowledgeStore):
    """DNA-inspired knowledge storage using quaternary encoding (A=0, T=1, C=2, G=3)."""

    capacity: int

    def __init__(self, capacity: int = 1024, rng: np.random.Generator | None = None) -> None: ...

    def _encode(self, data: np.ndarray) -> np.ndarray: ...

    def _decode(
        self, encoded: np.ndarray, nbytes: int, shape: tuple[int, ...]
    ) -> np.ndarray: ...

    def store(self, key: str, value: np.ndarray) -> None: ...

    def retrieve(self, key: str) -> np.ndarray | None: ...

    def generate(self, query: Signal) -> Signal:
        """Self-generate knowledge by recombining stored sequences."""
        ...


class MorphogeneticField:
    """Self-organizing structure development inspired by morphogenesis."""

    _DEFAULT_DU: float
    _DEFAULT_DV: float
    _DEFAULT_FEED: float
    _DEFAULT_KILL: float

    grid_size: int
    grid: np.ndarray
    morphogens: list[np.ndarray]
    diffusion_rate: float
    du_rate: float
    dv_rate: float
    feed_rate: float
    kill_rate: float

    def __init__(
        self, grid_size: int = 16, n_signals: int = 3, rng: np.random.Generator | None = None
    ) -> None: ...

    def step(self) -> None: ...

    def develop(self, n_steps: int = 50) -> np.ndarray: ...


class CellularAutomata:
    """Cellular automaton for distributed computation."""

    size: int
    rule: int
    state: np.ndarray

    def __init__(
        self, size: int = 64, rule: int = 30, rng: np.random.Generator | None = None
    ) -> None: ...

    def _apply_rule(self, left: int, center: int, right: int) -> int: ...

    def step(self) -> None: ...

    def evolve(self, n_steps: int = 50) -> np.ndarray: ...

    def step_n(self, n: int) -> np.ndarray: ...


class BiologicalSubstrate(CognitiveModule):
    """
    Biological computation substrate.
    - DNA storage with crossover-based self-generation
    - Morphogenetic field for self-organization
    - Cellular automata for distributed computation
    """

    dim: int
    dna_storage: DNAStorage
    morphogenetic: MorphogeneticField
    automata: CellularAutomata
    _cycle_count: int
    # Round-8 audit PERF8-3: cache for ``predict`` to reuse.
    _last_process_output: np.ndarray | None

    def __init__(self, dim: int = 64, rng: np.random.Generator | None = None) -> None: ...

    def process(self, signal: Signal) -> Signal: ...

    def predict(self, signal: Signal) -> Prediction: ...

    def update(self, prediction_error: float) -> None: ...
