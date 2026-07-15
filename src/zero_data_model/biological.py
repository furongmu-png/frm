# src/zero_data_model/biological.py
"""Biological Computation Substrate — DNA storage, morphogenetic fields, cellular automata."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .base import CognitiveModule, KnowledgeStore, Prediction, Signal

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
        # ``OrderedDict`` so we can evict the oldest entry when ``capacity``
        # is exceeded (Fix 20) -- the original ``dict`` declared ``capacity``
        # but never enforced it.
        self._store: OrderedDict[str, dict] = OrderedDict()

    def _encode(self, data: np.ndarray) -> np.ndarray:
        normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
        quantized = np.clip(np.round(normalized * 3).astype(int), 0, 3)
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
        self._store.move_to_end(key)
        # Enforce capacity by evicting the oldest entries (Fix 20).
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

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
        # ``np.random.randint(1, min_len)`` crashes when ``min_len == 1``;
        # clamp the lower bound to 2 so crossover is always valid (Fix 19).
        crossover = int(np.random.randint(1, max(2, min_len)))
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

    def step_n(self, n: int) -> np.ndarray:
        """Advance ``n`` steps WITHOUT recording history (Fix 11).

        Returns the final state. Used by hot loops (e.g.
        ``PatternMiner._best_automaton_rule``) that only need the final state
        rather than the full evolution tape -- avoids building 256 histories.
        """
        for _ in range(n):
            self.step()
        return self.state


class BiologicalSubstrate(CognitiveModule):
    """
    Biological computation substrate.
    - DNA storage with crossover-based self-generation
    - Morphogenetic field for self-organization
    - Cellular automata for distributed computation
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.dna_storage = DNAStorage(capacity=16)
        self.morphogenetic = MorphogeneticField(grid_size=16)
        self.automata = CellularAutomata(size=dim)
        # Sequence counter so each ``process`` call stores under a unique key
        # (Fix 4): the original code always stored under "input", overwriting
        # the previous entry, so ``generate`` (which needs >=2 stored
        # sequences) never actually performed crossover.
        self._cycle_count = 0

    def process(self, signal: Signal) -> Signal:
        # Use a unique per-cycle key so the store accumulates a population of
        # signals and crossover/recombination actually runs (Fix 4). Capacity
        # is bounded by DNAStorage (oldest evicted automatically).
        self.dna_storage.store(f"signal_{self._cycle_count}", signal.data)
        self._cycle_count += 1
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
        # Seed the morphogenetic field development from the incoming signal
        # rather than ignoring it (Fix 17): project the signal onto the grid
        # via an outer product so the prediction actually reflects the input.
        seed = signal.data[: self.dim]
        if len(seed) < self.dim:
            seed = np.pad(seed, (0, self.dim - len(seed)))
        gs = self.morphogenetic.grid.shape[0]
        # Build a (gs, gs) perturbation from the first ``gs`` signal samples.
        seed_vec = seed[:gs]
        if seed_vec.shape[0] < gs:
            seed_vec = np.pad(seed_vec, (0, gs - seed_vec.shape[0]))
        perturb = 0.05 * np.outer(seed_vec, seed_vec)
        # Save/restore the morphogenetic grid so predict() stays read-only
        # w.r.t. substrate state across the parallel predict step.
        saved_grid = self.morphogenetic.grid.copy()
        try:
            self.morphogenetic.grid = self.morphogenetic.grid + perturb
            pattern = self.morphogenetic.develop(n_steps=5)
        finally:
            self.morphogenetic.grid = saved_grid
        predicted = pattern.flatten()[: self.dim]
        if len(predicted) < self.dim:
            predicted = np.pad(predicted, (0, self.dim - len(predicted)))
        return Prediction(value=predicted[: self.dim], uncertainty=float(np.var(predicted)))

    def update(self, prediction_error: float) -> None:
        new_rate = self.morphogenetic.diffusion_rate + prediction_error * 0.01
        self.morphogenetic.diffusion_rate = max(0.001, new_rate)
