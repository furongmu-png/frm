"""Real Qiskit quantum backend with a true state-vector simulator fallback.

``QiskitQuantumBackend`` constructs genuine variational quantum circuits with
RY gates and CNOT entanglement, then samples them via Qiskit's
``StatevectorSampler``. When Qiskit is not installed, the same interface is
served by ``SimulatorQuantumBackend`` which performs a *real* state-vector
evolution in the full ``2**n_qubits``-dimensional complex Hilbert space:
genuine RY rotations on each qubit, genuine CNOT entanglement between
nearest neighbours, and Born-rule measurement (``probs = |amplitude|^2``).

C-6 fix: the previous simulator returned a length-``2*n_qubits`` real vector
and "entangled" by adding ``entangling @ state * 0.01`` (a non-unitary
perturbation that broke normalisation). It also applied RY as ``n_qubits``
independent 2-level rotations on adjacent amplitude pairs, which is not the
action of RY on a multi-qubit state. The Qiskit backend likewise collapsed
its bitstring histogram down to ``2*n_qubits``. Both paths now return the
proper ``2**n_qubits``-dimensional probability vector over computational-basis
bitstrings.
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
        a probability vector over measured bitstrings (length ``2**n_qubits``)."""
        raise NotImplementedError


class QiskitQuantumBackend(QuantumBackend):
    """Real Qiskit variational quantum circuit backend.

    Builds a layered ansatz: RY(parameters) -> CNOT entanglement -> RY ->
    CNOT, then samples ``n_shots`` measurements via the StatevectorSampler
    primitive. The resulting bitstring-counts histogram is mapped into a
    probability vector indexed by the computational basis (length
    ``2**n_qubits``); index ``k`` holds the probability of measuring the
    bitstring for integer ``k``.
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
            # C-6 fix: gate the second CNOT on the coupling matrix to match
            # the first loop (was unconditional, asymmetric with the first).
            for i in range(self.n_qubits - 1):
                if abs(float(entangling[i, i + 1])) > 1e-6:
                    qc.cx(i, i + 1)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def evolve_and_measure(
        self, params: np.ndarray, entangling: np.ndarray, n_shots: int = 1024
    ) -> np.ndarray:
        """Run the circuit and return a probability vector of length 2**n_qubits.

        Index ``k`` of the output holds the probability of observing the
        computational-basis bitstring corresponding to integer ``k`` (Qiskit
        emits big-endian bitstrings, so ``int(bitstring, 2)`` is the index).
        """
        qc = self._build_circuit(params, entangling)
        result = self._sampler.run([qc], shots=n_shots).result()
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()

        dim = 1 << self.n_qubits  # 2**n_qubits
        out = np.zeros(dim, dtype=float)
        total = float(sum(counts.values())) + 1e-8
        for bitstring, count in counts.items():
            # Qiskit bitstrings are MSB-first and may contain spaces for
            # multi-register circuits; collapse to a single integer index.
            idx = int(bitstring.replace(" ", ""), 2)
            if 0 <= idx < dim:
                out[idx] += count / total
        # Defensive renormalisation: bitstrings outside the expected range
        # should not occur, but sum-of-probabilities must be exactly 1.
        return out / (float(np.sum(out)) + 1e-12)


class SimulatorQuantumBackend(QuantumBackend):
    """Hand-rolled state-vector simulator (no Qiskit dependency).

    Implements genuine quantum state evolution in the full
    ``2**n_qubits``-dimensional complex Hilbert space:

    - ``_apply_ry`` applies the 2x2 RY(theta) rotation to qubit ``q`` by
      pairing every basis index whose q-th bit is 0 with the index whose
      q-th bit is 1 (and all other bits equal), as required by the tensor
      product structure of multi-qubit Hilbert space.
    - ``_apply_cnot`` swaps amplitudes between basis states whose target bit
      differs, conditioned on the control bit being set — the textbook CNOT.
    - ``evolve_and_measure`` returns the Born-rule distribution
      ``probs = |state|^2`` (normalised).

    C-6 fix: the previous implementation treated the state as ``n_qubits``
    independent 2-level systems (length ``2*n_qubits`` real vector with
    ``result[0::2]`` / ``result[1::2]`` RY pairs) and "entangled" via a
    non-unitary ``state += 0.01 * entangling @ state`` perturbation that
    broke normalisation. Neither is a quantum operation; both are replaced
    by the correct tensor-product unitaries below.
    """

    name = "simulator"

    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        # 2**20 amplitudes * 16 bytes (complex128) = 16 MB — the upper bound
        # the factory enforces; reject explicitly if called directly with more.
        if n_qubits > 20:
            raise ValueError(
                f"n_qubits={n_qubits} would materialize a 2**n_qubits state "
                f"vector (>16 MB); the supported maximum is 20."
            )
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def _apply_ry(self, state: np.ndarray, theta: float, qubit: int) -> np.ndarray:
        """Apply the RY(theta) gate to ``qubit`` of the multi-qubit state.

        RY(theta) = [[cos(theta/2), -sin(theta/2)],
                     [sin(theta/2),  cos(theta/2)]]
        acts on the (|0>, |1>) subspace of the target qubit. In the full
        ``2**n`` Hilbert space, every basis index ``i`` whose q-th bit is 0
        is paired with ``j = i | (1 << q)`` (the same index with the q-th bit
        set); the (a_i, a_j) amplitude pair is rotated by the 2x2 matrix.

        Vectorised over all 2**n basis indices via NumPy bitwise ops so the
        gate applies in O(2**n) time with no Python-level loop (critical for
        n_qubits >= 15 where the state vector has >32K amplitudes).
        """
        cos_t = np.cos(theta / 2.0)
        sin_t = np.sin(theta / 2.0)
        mask = np.intp(1 << qubit)
        # Vectorised index selection: all indices whose q-th bit is unset.
        all_idx = np.arange(len(state), dtype=np.intp)
        bit_unset = (all_idx & mask) == 0
        idx0 = all_idx[bit_unset]
        idx1 = idx0 | mask  # corresponding indices with the q-th bit set
        a = state[idx0]
        b = state[idx1]
        new_state = state.copy()
        new_state[idx0] = cos_t * a - sin_t * b
        new_state[idx1] = sin_t * a + cos_t * b
        return new_state

    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT(control, target): flip the target bit conditioned on control.

        For every basis index ``i`` with the control bit set and the target
        bit unset, swap amplitude ``a_i`` with ``a_j`` where ``j`` differs
        from ``i`` only in the target bit (``j = i ^ (1 << target)``).

        Vectorised over all 2**n basis indices via NumPy bitwise ops.
        """
        c_mask = np.intp(1 << control)
        t_mask = np.intp(1 << target)
        all_idx = np.arange(len(state), dtype=np.intp)
        # Indices where control=1 and target=0 (the "low" partner of each pair).
        low_mask = ((all_idx & c_mask) != 0) & ((all_idx & t_mask) == 0)
        idx_low = all_idx[low_mask]
        idx_high = idx_low ^ t_mask
        new_state = state.copy()
        new_state[idx_low] = state[idx_high]
        new_state[idx_high] = state[idx_low]
        return new_state

    def evolve_and_measure(
        self, params: np.ndarray, entangling: np.ndarray, n_shots: int = 1024
    ) -> np.ndarray:
        """Evolve a |0...0>-initialised state through the ansatz and measure.

        Returns the Born-rule probability vector ``|amplitude|^2`` over the
        ``2**n_qubits`` computational basis states. ``n_shots`` is accepted
        for API compatibility with the Qiskit backend but ignored: the
        state-vector simulator returns exact probabilities, not samples.
        """
        del n_shots  # exact state-vector simulator; not used
        dim = 1 << self.n_qubits  # 2**n_qubits
        # Initialise to |0...0>: amplitude 1+0j on the first basis state.
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0 + 0.0j
        n_layers = min(self.n_layers, params.shape[0])
        for layer in range(n_layers):
            # First RY layer: rotate every qubit by params[layer, q, 0].
            for q in range(self.n_qubits):
                theta = float(params[layer, q, 0])
                state = self._apply_ry(state, theta, q)
            # CNOT entanglement: nearest-neighbour pairs gated by the coupling
            # matrix, so the learned structure selectively engages entanglement.
            for i in range(self.n_qubits - 1):
                if abs(float(entangling[i, i + 1])) > 1e-6:
                    state = self._apply_cnot(state, i, i + 1)
            # Second RY layer.
            for q in range(self.n_qubits):
                theta = float(params[layer, q, 1])
                state = self._apply_ry(state, theta, q)
            # Second CNOT entanglement (also gated by the coupling matrix —
            # C-6 fix: the previous code applied this unconditionally,
            # asymmetrically with the first loop and ignoring the learned
            # structure for the post-rotation entanglement).
            for i in range(self.n_qubits - 1):
                if abs(float(entangling[i, i + 1])) > 1e-6:
                    state = self._apply_cnot(state, i, i + 1)
        # Born rule: probabilities = |amplitude|^2 = Re^2 + Im^2.
        probs = (state.real * state.real + state.imag * state.imag).astype(float)
        total = float(np.sum(probs))
        # Numerical drift can move the total slightly off 1.0 (the operations
        # above are unitary, so theoretically total == 1.0 exactly, but
        # floating-point cos/sin can introduce ~1e-15 error).
        return probs / (total + 1e-12)


def get_quantum_backend(
    n_qubits: int = 8, n_layers: int = 3, prefer: str | None = None
) -> QuantumBackend:
    """Factory: pick the best available quantum backend.

    ``prefer`` may be 'qiskit', 'simulator', 'ibm', 'ibm_quantum', or None
    (auto-detect). When IBM hardware is requested but ``qiskit-ibm-runtime``
    is missing or no token is configured, the factory falls back to the
    local Qiskit simulator (or the pure-NumPy simulator if Qiskit is absent).

    Fix 24: ``n_qubits`` is clamped to <= 20 because the local state-vector
    simulator materializes a dense ``2**n_qubits`` complex amplitude vector
    (16 MB at n_qubits=20). A warning is emitted when clamping occurs.
    """
    import warnings

    if n_qubits > 20:
        warnings.warn(
            f"n_qubits={n_qubits} exceeds the supported maximum (20); "
            "clamping to 20 to avoid exponential state-vector blowup "
            "(2**n_qubits complex amplitudes).",
            stacklevel=2,
        )
        n_qubits = 20
    if prefer in ("ibm", "ibm_quantum"):
        try:
            from .ibm_quantum import IBMQuantumBackend

            return IBMQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
        except (ImportError, RuntimeError) as exc:
            # IBM runtime missing or init failed: fall back to the local
            # Qiskit simulator (per spec), then the pure-NumPy simulator.
            warnings.warn(
                f"IBM backend init failed ({type(exc).__name__}); falling back",
                stacklevel=2,
            )
            if _HAS_QISKIT:
                return QiskitQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
            return SimulatorQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
    if (prefer is None or prefer == "qiskit") and _HAS_QISKIT:
        return QiskitQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
    return SimulatorQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
