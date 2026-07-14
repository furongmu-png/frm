# src/zero_data_model/biological.py
"""Biological Computation Substrate — DNA storage, morphogenetic fields, cellular automata."""

from __future__ import annotations
import numpy as np
from .base import Signal, Prediction, CognitiveModule, KnowledgeStore

# JIT kernels -- graceful fallback to pure numpy if numba missing.
try:
    from .hardware.kernels import _cellular_automata_step, _morphogenetic_laplacian
    _HAS_JIT = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_JIT = False


class DNAStorage(KnowledgeStore):
    """DNA-inspired knowledge storage using quaternary encoding (A=0, T=1, C=2, G=3)."""

    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self._store: dict[str, np.ndarray] = {}

    def _encode(self, data: np.ndarray) -> np.ndarray:
        quantized = np.clip(np.round((data - data.min()) / (data.max() - data.min() + 1e-8) * 3).astype(int), 0, 3)
        return quantized

    def _decode(self, encoded: np.ndarray, original_min: float, original_max: float) -> np.ndarray:
        return encoded / 3.0 * (original_max - original_min) + original_min

    def store(self, key: str, value: np.ndarray) -> None:
        self._store[key] = {
            "encoded": self._encode(value),
            "min": float(value.min()),
            "max": float(value.max()),
            "shape": value.shape,
        }

    def retrieve(self, key: str) -> np.ndarray | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        return self._decode(entry["encoded"], entry["min"], entry["max"]).reshape(entry["shape"])

    def generate(self, query: Signal) -> Signal:
        """Self-generate knowledge by recombining stored sequences."""
        if len(self._store) < 2:
            return Signal(data=np.random.randn(64), metadata={"source": "dna_random"})
        keys = list(self._store.keys())
        k1, k2 = np.random.choice(keys, 2, replace=False)
        v1 = self.retrieve(k1)
        v2 = self.retrieve(k2)
        if v1 is None or v2 is None:
            return Signal(data=np.random.randn(64), metadata={"source": "dna_random"})
        min_len = min(len(v1.flatten()), len(v2.flatten()))
        crossover = np.random.randint(1, min_len)
        flat1 = v1.flatten()[:min_len]
        flat2 = v2.flatten()[:min_len]
        child = np.concatenate([flat1[:crossover], flat2[crossover:]])
        return Signal(data=child, metadata={"source": "dna_crossover", "parents": [k1, k2]})


class MorphogeneticField:
    """Self-organizing structure development inspired by morphogenesis."""

    def __init__(self, grid_size: int = 16, n_signals: int = 3):
        self.grid_size = grid_size
        self.grid = np.random.randn(grid_size, grid_size) * 0.1
        self.morphogens = [np.random.randn(grid_size, grid_size) * 0.1 for _ in range(n_signals)]
        self.diffusion_rate = 0.05

    def step(self) -> None:
        if _HAS_JIT:
            laplacian = _morphogenetic_laplacian(
                np.ascontiguousarray(self.grid, dtype=float)
            )
        else:
            laplacian = (
                np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0)
                + np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1)
                - 4 * self.grid
            )
        self.grid += self.diffusion_rate * laplacian
        for m in self.morphogens:
            if _HAS_JIT:
                m_lap = _morphogenetic_laplacian(np.ascontiguousarray(m, dtype=float))
            else:
                m_lap = (
                    np.roll(m, 1, axis=0) + np.roll(m, -1, axis=0)
                    + np.roll(m, 1, axis=1) + np.roll(m, -1, axis=1)
                    - 4 * m
                )
            m += self.diffusion_rate * m_lap

    def develop(self, n_steps: int = 50) -> np.ndarray:
        for _ in range(n_steps):
            self.step()
        morphogen_sum = sum(self.morphogens)
        pattern = self.grid * (1.0 + 0.1 * morphogen_sum)
        return pattern


class CellularAutomata:
    """Cellular automaton for distributed computation."""

    def __init__(self, size: int = 64, rule: int = 30):
        self.size = size
        self.rule = rule
        self.state = np.random.randint(0, 2, size)

    def _apply_rule(self, left: int, center: int, right: int) -> int:
        index = (left << 2) | (center << 1) | right
        return (self.rule >> index) & 1

    def step(self) -> None:
        if _HAS_JIT:
            # JIT path: all cells updated in one pass -- the biggest win.
            self.state = _cellular_automata_step(
                np.ascontiguousarray(self.state),
                int(self.rule),
                int(self.size),
            )
            return
        new_state = np.zeros(self.size, dtype=int)
        for i in range(self.size):
            left = self.state[(i - 1) % self.size]
            center = self.state[i]
            right = self.state[(i + 1) % self.size]
            new_state[i] = self._apply_rule(left, center, right)
        self.state = new_state

    def evolve(self, n_steps: int = 50) -> np.ndarray:
        history = [self.state.copy()]
        for _ in range(n_steps):
            self.step()
            history.append(self.state.copy())
        return np.array(history)


class BiologicalSubstrate(CognitiveModule):
    """
    Biological computation substrate.
    - DNA storage with crossover-based self-generation
    - Morphogenetic field for self-organization
    - Cellular automata for distributed computation
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.dna_storage = DNAStorage(capacity=1024)
        self.morphogenetic = MorphogeneticField(grid_size=16)
        self.automata = CellularAutomata(size=dim)

    def process(self, signal: Signal) -> Signal:
        self.dna_storage.store("input", signal.data)
        generated = self.dna_storage.generate(signal)
        pattern = self.morphogenetic.develop(n_steps=10)
        pattern_flat = pattern.flatten()[: self.dim]
        if len(pattern_flat) < self.dim:
            pattern_flat = np.pad(pattern_flat, (0, self.dim - len(pattern_flat)))
        self.automata.evolve(n_steps=5)
        ca_signal = self.automata.state.astype(float)
        combined = 0.4 * generated.data[: self.dim] + 0.3 * pattern_flat + 0.3 * ca_signal
        if len(combined) < self.dim:
            combined = np.pad(combined, (0, self.dim - len(combined)))
        return Signal(data=combined[: self.dim], metadata={"source": "biological"})

    def predict(self, signal: Signal) -> Prediction:
        pattern = self.morphogenetic.develop(n_steps=5)
        predicted = pattern.flatten()[: self.dim]
        if len(predicted) < self.dim:
            predicted = np.pad(predicted, (0, self.dim - len(predicted)))
        return Prediction(value=predicted[: self.dim], uncertainty=float(np.var(predicted)))

    def update(self, prediction_error: float) -> None:
        self.morphogenetic.diffusion_rate = max(0.001, self.morphogenetic.diffusion_rate + prediction_error * 0.01)
