"""Tests for the IBM Quantum hardware backend.

All tests run WITHOUT a real IBM Quantum token and WITHOUT
``qiskit-ibm-runtime`` installed; they exercise the graceful fallback path
and the factory selection logic. When ``qiskit-ibm-runtime`` is installed
but no token is configured, the direct-instantiation tests verify that
``IBMQuantumBackend`` delegates to the local simulator.
"""

from __future__ import annotations

import numpy as np

from zero_data_model.hardware.quantum import QuantumBackend, get_quantum_backend

try:  # pragma: no cover - module always imports (its own deps are guarded)
    from zero_data_model.hardware.ibm_quantum import IBMQuantumBackend

    _HAS_IBM_MODULE = True
except Exception:  # pragma: no cover - defensive
    _HAS_IBM_MODULE = False
    IBMQuantumBackend = None  # type: ignore[assignment]


def _make_backend(n_qubits: int = 4, n_layers: int = 2):
    """Return an IBMQuantumBackend when possible, else the factory result.

    When ``qiskit-ibm-runtime`` is not installed, ``IBMQuantumBackend``
    raises ``RuntimeError`` from its constructor; in that case we fall back
    to the factory which itself selects the Qiskit simulator.
    """
    if _HAS_IBM_MODULE:
        try:
            return IBMQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
        except RuntimeError:
            pass  # qiskit-ibm-runtime not installed
    return get_quantum_backend(n_qubits=n_qubits, n_layers=n_layers, prefer="ibm")


def test_ibm_backend_factory_returns_simulator_without_token():
    """``get_quantum_backend(prefer="ibm")`` returns a working backend."""
    backend = get_quantum_backend(n_qubits=4, n_layers=2, prefer="ibm")
    assert isinstance(backend, QuantumBackend)
    params = np.random.randn(2, 4, 2) * 0.1
    entangling = np.random.randn(4, 4) * 0.05
    out = backend.evolve_and_measure(params, entangling, n_shots=64)
    # C-6 fix: outputs a full 2**n_qubits probability vector (was 2 * n_qubits).
    assert out.shape == (2 ** 4,)


def test_ibm_backend_name():
    """The returned backend's name is one of the known set."""
    backend = get_quantum_backend(n_qubits=4, n_layers=2, prefer="ibm")
    assert backend.name in {"ibm_quantum", "qiskit", "simulator"}


def test_ibm_backend_fallback_to_simulator():
    """Without a token, ``evolve_and_measure`` delegates to the local simulator."""
    backend = _make_backend()
    params = np.random.randn(2, 4, 2) * 0.1
    entangling = np.random.randn(4, 4) * 0.05
    out = backend.evolve_and_measure(params, entangling, n_shots=64)
    # C-6 fix: outputs a full 2**n_qubits probability vector (was 2 * n_qubits).
    assert out.shape == (2 ** 4,)


def test_ibm_backend_without_token_is_not_real_hardware():
    """``is_real_hardware`` is False when no token is configured."""
    backend = _make_backend()
    assert backend.is_real_hardware is False


def test_factory_prefer_ibm_works():
    """``get_quantum_backend(prefer="ibm")`` returns a runnable backend."""
    backend = get_quantum_backend(n_qubits=4, n_layers=2, prefer="ibm")
    params = np.random.randn(2, 4, 2) * 0.1
    entangling = np.random.randn(4, 4) * 0.05
    out = backend.evolve_and_measure(params, entangling, n_shots=64)
    # C-6 fix: outputs a full 2**n_qubits probability vector (was 2 * n_qubits).
    assert out.shape == (2 ** 4,)
    assert backend.name in {"ibm_quantum", "qiskit", "simulator"}
