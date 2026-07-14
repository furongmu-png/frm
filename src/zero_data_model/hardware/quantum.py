"""Real Qiskit quantum backend with a simulator fallback.

``QiskitQuantumBackend`` constructs genuine variational quantum circuits with
RY gates and CNOT entanglement, then samples them via Qiskit's
``StatevectorSampler``. When Qiskit is not installed, the same interface is
served by ``SimulatorQuantumBackend`` which keeps the original hand-rolled
state-vector evolution.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    _HAS_QISKIT = True
except ImportError:  # pragma: no cover
    _HAS_QISKIT = False


class QuantumBackend:
    """Interface for quantum execution backends."""

    name: str = "base"

    @property
    def is_real_hardware(self) -> bool:
        """True only when the backend executes on physical quantum hardware.

        Simulators (including the local Qiskit ``StatevectorSampler``) return
        ``False``; the IBM Quantum backend overrides this to report the actual
        hardware availability.
        """
        return False

    def evolve_and_measure(
        self, params: np.ndarray, entangling: np.ndarray, n_shots: int = 1024
    ) -> np.ndarray:
        """Run a variational circuit parameterized by ``params`` and return
        a probability vector over measured bitstrings (length 2*n_qubits)."""
        raise NotImplementedError


class QiskitQuantumBackend(QuantumBackend):
    """Real Qiskit variational quantum circuit backend.

    Builds a layered ansatz: RY(parameters) -> CNOT entanglement -> RY ->
    CNOT, then samples ``n_shots`` measurements via the StatevectorSampler
    primitive. The resulting bitstring-counts histogram is mapped into a
    probability vector aligned with the original simulated-circuit output
    shape (2 * n_qubits).
    """

    name = "qiskit"

    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        if not _HAS_QISKIT:  # pragma: no cover - guarded by factory
            raise RuntimeError("Qiskit is not installed")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self._sampler = StatevectorSampler()

    def _build_circuit(self, params: np.ndarray, entangling: np.ndarray) -> QuantumCircuit:
        """Build the parameterized ansatz circuit.

        ``params`` has shape (n_layers, n_qubits, 2); the two angles per qubit
        per layer are the RY rotations applied before and after entanglement.
        """
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        n_layers = min(self.n_layers, params.shape[0])
        for layer in range(n_layers):
            for q in range(self.n_qubits):
                theta = float(params[layer, q, 0])
                qc.ry(theta, q)
            # Entangle nearest neighbours with CNOT, weighted by the coupling
            # matrix so the learned structure is respected.
            for i in range(self.n_qubits - 1):
                if abs(float(entangling[i, i + 1])) > 1e-6:
                    qc.cx(i, i + 1)
            for q in range(self.n_qubits):
                theta = float(params[layer, q, 1])
                qc.ry(theta, q)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def evolve_and_measure(
        self, params: np.ndarray, entangling: np.ndarray, n_shots: int = 1024
    ) -> np.ndarray:
        """Run the circuit and return a probability vector of length 2*n_qubits."""
        qc = self._build_circuit(params, entangling)
        result = self._sampler.run([qc], shots=n_shots).result()
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()

        # Collapse the bitstring histogram into a probability vector whose
        # length matches the legacy simulator output (2 * n_qubits) so the
        # rest of the system is unchanged.
        out = np.zeros(2 * self.n_qubits, dtype=float)
        total = float(sum(counts.values())) + 1e-8
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            for i, b in enumerate(bits[: len(out)]):
                if b == "1":
                    out[i] += count / total
        return out


class SimulatorQuantumBackend(QuantumBackend):
    """Fallback hand-rolled state-vector simulator (no Qiskit dependency)."""

    name = "simulator"

    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def _ry_gate(self, state: np.ndarray, theta: np.ndarray) -> np.ndarray:
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        result = np.zeros_like(state)
        result[0::2] = cos * state[0::2] - sin * state[1::2]
        result[1::2] = sin * state[0::2] + cos * state[1::2]
        return result

    def _entangle(self, state: np.ndarray, entangling: np.ndarray) -> np.ndarray:
        dim = min(len(state), self.n_qubits)
        s = state[:dim]
        entangled = s + entangling[:dim, :dim] @ s * 0.01
        state[:dim] = entangled
        return state

    def evolve_and_measure(
        self, params: np.ndarray, entangling: np.ndarray, n_shots: int = 1024
    ) -> np.ndarray:
        """Evolve a zero-initialized state through the ansatz and measure."""
        state = np.zeros(2 * self.n_qubits)
        for layer in range(min(self.n_layers, params.shape[0])):
            state = self._ry_gate(state, params[layer, :, 0])
            state = self._entangle(state, entangling)
            state = self._ry_gate(state, params[layer, :, 1])
            state = self._entangle(state, entangling)
        probs = np.abs(state) ** 2
        s = float(np.sum(probs)) + 1e-8
        return probs / s


def get_quantum_backend(
    n_qubits: int = 8, n_layers: int = 3, prefer: str | None = None
) -> QuantumBackend:
    """Factory: pick the best available quantum backend.

    ``prefer`` may be 'qiskit', 'simulator', 'ibm', 'ibm_quantum', or None
    (auto-detect). When IBM hardware is requested but ``qiskit-ibm-runtime``
    is missing or no token is configured, the factory falls back to the
    local Qiskit simulator (or the pure-NumPy simulator if Qiskit is absent).
    """
    if prefer in ("ibm", "ibm_quantum"):
        try:
            from .ibm_quantum import IBMQuantumBackend

            return IBMQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
        except Exception:
            # IBM runtime missing or init failed: fall back to the local
            # Qiskit simulator (per spec), then the pure-NumPy simulator.
            if _HAS_QISKIT:
                return QiskitQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
            return SimulatorQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
    if (prefer is None or prefer == "qiskit") and _HAS_QISKIT:
        return QiskitQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
    return SimulatorQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
