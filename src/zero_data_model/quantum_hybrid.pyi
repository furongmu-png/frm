from __future__ import annotations

import numpy as np

from .base import CognitiveModule, Prediction, Signal
from .hardware.quantum import QuantumBackend

class VariationalQuantumCircuit:
    """Variational quantum circuit backed by Qiskit (or a simulator fallback)."""

    n_qubits: int
    n_layers: int
    params: np.ndarray
    entangling: np.ndarray
    backend: QuantumBackend

    def __init__(
        self, n_qubits: int = 8, n_layers: int = 3, backend: str | None = None
    ) -> None: ...

    @property
    def backend_name(self) -> str: ...

    def evolve(self, input_state: np.ndarray) -> np.ndarray:
        """Run the variational ansatz; returns a normalized state vector."""
        ...

    def measure(self, state: np.ndarray) -> np.ndarray: ...


# Backward-compatible alias for existing imports.
SimulatedQuantumCircuit = VariationalQuantumCircuit


class QuantumAnnealer:
    """Simulated quantum annealing with numba-JIT inner loop."""

    n_vars: int
    cost_matrix: np.ndarray
    jit: bool

    def __init__(self, n_vars: int = 16) -> None: ...

    def optimize(self, n_iterations: int = 100) -> tuple[np.ndarray, float]: ...

    def _energy(self, state: np.ndarray) -> float: ...


class QuantumClassicalHybrid(CognitiveModule):
    """
    Quantum-Classical Hybrid computation.
    - Real Qiskit variational quantum circuit (or simulator fallback)
    - Numba-JIT quantum annealing for optimization
    - Classical neural processing
    """

    dim: int
    n_qubits: int
    quantum_circuit: VariationalQuantumCircuit
    annealer: QuantumAnnealer
    classical_weights: np.ndarray

    def __init__(
        self,
        dim: int = 64,
        n_qubits: int = 8,
        quantum_backend: str | None = None,
    ) -> None: ...

    @property
    def quantum_backend_name(self) -> str: ...

    def process(self, signal: Signal) -> Signal: ...

    def predict(self, signal: Signal) -> Prediction: ...

    def update(self, prediction_error: float) -> None: ...

    def solve_optimization(self) -> tuple[np.ndarray, float]: ...
