# src/zero_data_model/quantum_hybrid.py
"""Quantum-Classical Hybrid Computation Layer.

Uses a real Qiskit variational quantum circuit when Qiskit is available
(``hardware.quantum.QiskitQuantumBackend``), falling back to a hand-rolled
state-vector simulator otherwise. The annealer's inner loop is JIT-compiled
with numba when installed for faster optimization.
"""

from __future__ import annotations

import numpy as np

from .base import CognitiveModule, Prediction, Signal
from .hardware.quantum import QuantumBackend, get_quantum_backend

# JIT kernels -- graceful fallback to pure numpy if numba missing.
try:
    from .hardware.kernels import _quantum_classical_forward
    _HAS_KERNELS_JIT = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_KERNELS_JIT = False


class VariationalQuantumCircuit:
    """Variational quantum circuit backed by Qiskit (or a simulator fallback).

    The ansatz parameters and entangling coupling matrix are owned here; the
    actual circuit construction and measurement are delegated to the
    ``QuantumBackend`` selected at construction time.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        backend: str | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        self.params = self._rng.standard_normal((n_layers, n_qubits, 2)) * 0.1
        self.entangling = self._rng.standard_normal((n_qubits, n_qubits)) * 0.05
        self.entangling = (self.entangling + self.entangling.T) / 2
        self.backend: QuantumBackend = get_quantum_backend(
            n_qubits=n_qubits, n_layers=n_layers, prefer=backend
        )

    @property
    def backend_name(self) -> str:
        return self.backend.name

    @property
    def is_real_quantum_hardware(self) -> bool:
        """True when the backend is real IBM Quantum hardware.

        Combines the backend's identity (``name == "ibm_quantum"``) with the
        backend's own availability flag so simulator fallbacks (no token,
        missing runtime, or a failed job) always report ``False``.
        """
        return self.backend.name == "ibm_quantum" and bool(
            getattr(self.backend, "is_real_hardware", False)
        )

    def evolve(self, input_state: np.ndarray) -> np.ndarray:
        """Run the variational ansatz; returns a normalized state vector.

        The input is folded into the first RY layer as initial rotation
        biases so external signals still steer the circuit.
        """
        # Bias the first-layer RY angles by the (scaled) input signal so the
        # circuit is responsive to incoming data without changing the backend API.
        params = self.params.copy()
        biased = input_state[: self.n_qubits]
        if biased.size:
            params[0, : biased.size, 0] += biased
        return self.backend.evolve_and_measure(
            params, self.entangling, n_shots=1024
        )

    def measure(self, state: np.ndarray) -> np.ndarray:
        # ``evolve`` already returns a probability vector when using the Qiskit
        # backend; normalize defensively for the simulator path.
        # Use np.iscomplexobj (Fix 22): np.any(np.iscomplex(state)) returns
        # False for a complex dtype with all-zero imaginary parts, which would
        # skip the |.|^2 squaring and feed complex values into the divider.
        probs = np.abs(state) ** 2 if np.iscomplexobj(state) else state
        return probs / (np.sum(probs) + 1e-8)


# Backward-compatible alias for existing imports.
SimulatedQuantumCircuit = VariationalQuantumCircuit


# Numba-accelerated annealing inner loop (JIT when available, pure-numpy
# fallback otherwise). Kept at module scope so the compiled cache persists.
#
# A-CRIT-02: ``np.random.*`` calls inside an ``@njit`` function use numba's
# *own* internal RNG, which is independent of numpy's global RNG and is NOT
# re-seeded by ``ZeroDataModel.think``/``solve``. To make seeded models
# reproducible, the random ``flips`` and ``accepts`` sequences are generated
# outside the JIT kernel (drawing from numpy's global RNG) and passed in as
# arrays.
try:  # pragma: no cover - optional dependency
    from numba import njit

    @njit(cache=True, nogil=True)
    def _anneal_inner(cost_matrix, best_state, best_energy, n_iterations, n_vars, flips, accepts):
        for t in range(1, n_iterations + 1):
            temperature = 1.0 / np.log(1.0 + t)
            candidate = best_state.copy()
            flip = flips[t - 1]
            candidate[flip] = -candidate[flip]
            candidate_energy = float(candidate @ cost_matrix @ candidate)
            delta = candidate_energy - best_energy
            if delta < 0.0 or accepts[t - 1] < np.exp(-delta / (temperature + 1e-8)):
                best_state = candidate
                best_energy = candidate_energy
        return best_state, best_energy

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def _anneal_inner(cost_matrix, best_state, best_energy, n_iterations, n_vars, flips, accepts):
        for t in range(1, n_iterations + 1):
            temperature = 1.0 / np.log(1 + t)
            candidate = best_state.copy()
            flip = flips[t - 1]
            candidate[flip] *= -1
            candidate_energy = float(candidate @ cost_matrix @ candidate)
            delta = candidate_energy - best_energy
            if delta < 0 or accepts[t - 1] < np.exp(-delta / (temperature + 1e-8)):
                best_state = candidate
                best_energy = candidate_energy
        return best_state, best_energy


class QuantumAnnealer:
    """Simulated quantum annealing with numba-JIT inner loop."""

    def __init__(self, n_vars: int = 16, rng: np.random.Generator | None = None):
        self.n_vars = n_vars
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        self.cost_matrix = self._rng.standard_normal((n_vars, n_vars)) * 0.1
        self.cost_matrix = (self.cost_matrix + self.cost_matrix.T) / 2
        self.jit = _HAS_NUMBA
        self._jit_warmed = False

    def optimize(self, n_iterations: int = 100) -> tuple[np.ndarray, float]:
        # Round-3 audit CRIT-1: per-module Generator
        best_state = self._rng.choice([-1, 1], size=self.n_vars).astype(float)
        best_energy = float(best_state @ self.cost_matrix @ best_state)
        # Generate the RNG sequence outside the JIT kernel so it draws from the
        # per-module Generator (CRIT-1) rather than numba's independent internal
        # RNG (A-CRIT-02). The Generator is seeded via ZeroDataModel when a seed
        # is set, so seeded models stay reproducible.
        flips = self._rng.integers(0, self.n_vars, size=n_iterations)
        accepts = self._rng.random(size=n_iterations)
        # Warm up the JIT cache on the first call with this array shape.
        if self.jit and not self._jit_warmed:
            _anneal_inner(
                self.cost_matrix, best_state.copy(), best_energy, 1, self.n_vars,
                flips[:1], accepts[:1],
            )
            self._jit_warmed = True
        best_state, best_energy = _anneal_inner(
            self.cost_matrix, best_state, best_energy, n_iterations, self.n_vars,
            flips, accepts,
        )
        return best_state, float(best_energy)

    def _energy(self, state: np.ndarray) -> float:
        return float(state @ self.cost_matrix @ state)


class QuantumClassicalHybrid(CognitiveModule):
    """
    Quantum-Classical Hybrid computation.
    - Real Qiskit variational quantum circuit (or simulator fallback)
    - Numba-JIT quantum annealing for optimization
    - Classical neural processing
    """

    def __init__(
        self,
        dim: int = 64,
        n_qubits: int = 8,
        quantum_backend: str | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.dim = dim
        self.n_qubits = n_qubits
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        self.quantum_circuit = VariationalQuantumCircuit(
            n_qubits=n_qubits, n_layers=3, backend=quantum_backend, rng=self._rng
        )
        self.annealer = QuantumAnnealer(dim, rng=self._rng)
        self.classical_weights = self._rng.standard_normal((dim, dim)) * 0.05

    @property
    def quantum_backend_name(self) -> str:
        return self.quantum_circuit.backend_name

    def process(self, signal: Signal) -> Signal:
        x = signal.data[: self.dim]
        if len(x) < self.dim:
            x = np.pad(x, (0, self.dim - len(x)))
        quantum_features = self.quantum_circuit.evolve(x[: 2 * self.n_qubits])
        qf = np.zeros(self.dim)
        qf[: len(quantum_features)] = quantum_features[: self.dim]
        if _HAS_KERNELS_JIT:
            classical = _quantum_classical_forward(
                np.ascontiguousarray(x, dtype=float),
                np.ascontiguousarray(self.classical_weights, dtype=float),
            )
        else:
            classical = np.tanh(x @ self.classical_weights)
        combined = 0.3 * qf + 0.7 * classical
        return Signal(
            data=combined,
            metadata={"quantum_features": True, "quantum_backend": self.quantum_backend_name},
        )

    def predict(self, signal: Signal) -> Prediction:
        x = signal.data[: self.dim]
        if len(x) < self.dim:
            x = np.pad(x, (0, self.dim - len(x)))
        if _HAS_KERNELS_JIT:
            predicted = _quantum_classical_forward(
                np.ascontiguousarray(x, dtype=float),
                np.ascontiguousarray(self.classical_weights, dtype=float),
            )
        else:
            predicted = np.tanh(x @ self.classical_weights)
        return Prediction(value=predicted, uncertainty=float(np.var(predicted)))

    def update(self, prediction_error: float) -> None:
        if not np.isfinite(prediction_error):
            return
        prediction_error = float(np.clip(prediction_error, -1e6, 1e6))
        # Round-3 audit CRIT-1: per-module Generator
        noise = self._rng.standard_normal(self.classical_weights.shape) * prediction_error * 0.001
        self.classical_weights += noise

    def solve_optimization(self) -> tuple[np.ndarray, float]:
        return self.annealer.optimize()
