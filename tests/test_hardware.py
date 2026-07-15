# tests/test_hardware.py
"""Tests for the hardware acceleration layer: Qiskit backend, parallel executor, accel."""

import numpy as np

from zero_data_model.hardware import (
    ParallelExecutor,
    QuantumBackend,
    SimulatorQuantumBackend,
    asnumpy,
    backend_name,
    to_gpu,
    xp,
)
from zero_data_model.hardware.quantum import get_quantum_backend
from zero_data_model.quantum_hybrid import (
    QuantumAnnealer,
    QuantumClassicalHybrid,
    VariationalQuantumCircuit,
)


def test_accel_backend_name():
    name = backend_name()
    assert name in {"numpy", "cupy"}


def test_accel_asnumpy_roundtrip():
    arr = np.arange(10, dtype=float)
    g = to_gpu(arr)
    back = asnumpy(g)
    assert np.allclose(back, arr)


def test_accel_xp_array():
    a = xp.arange(5)
    assert a.shape == (5,)
    assert asnumpy(a).shape == (5,)


def test_quantum_backend_factory_qiskit_or_simulator():
    backend = get_quantum_backend(n_qubits=4, n_layers=2)
    assert isinstance(backend, QuantumBackend)
    assert backend.name in {"qiskit", "simulator"}


def test_qiskit_backend_runs_circuit():
    backend = get_quantum_backend(n_qubits=4, n_layers=2, prefer="qiskit")
    params = np.random.randn(2, 4, 2) * 0.1
    entangling = np.random.randn(4, 4) * 0.05
    out = backend.evolve_and_measure(params, entangling, n_shots=256)
    # C-6 fix: outputs a full 2**n_qubits probability vector over the
    # computational basis (was incorrectly collapsed to 2 * n_qubits).
    assert out.shape == (2 ** 4,)
    assert np.all(out >= 0)
    assert np.all(out <= 1.0 + 1e-6)


def test_simulator_backend_fallback():
    backend = SimulatorQuantumBackend(n_qubits=4, n_layers=2)
    params = np.random.randn(2, 4, 2) * 0.1
    entangling = np.random.randn(4, 4) * 0.05
    out = backend.evolve_and_measure(params, entangling)
    # C-6 fix: outputs a full 2**n_qubits probability vector (was 2 * n_qubits).
    assert out.shape == (2 ** 4,)
    # The simulator normalizes the measured probabilities to sum to 1.0.
    # The previous ``or out.sum() >= 0`` clause was vacuous (always true for
    # a probability vector) and masked real normalization bugs.
    assert abs(out.sum() - 1.0) < 1e-3


def test_variational_circuit_uses_backend():
    circuit = VariationalQuantumCircuit(n_qubits=4, n_layers=2)
    state = circuit.evolve(np.random.randn(8))
    assert state.shape[0] > 0


def test_annealer_jit_runs():
    annealer = QuantumAnnealer(n_vars=8)
    state, energy = annealer.optimize(n_iterations=50)
    assert state.shape == (8,)
    assert np.isfinite(energy)


def test_parallel_executor_preserves_order():
    def f(x):
        return x * 2
    executor = ParallelExecutor()
    out = executor.map(f, list(range(10)))
    assert out == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


def test_parallel_executor_empty():
    executor = ParallelExecutor()
    assert executor.map(lambda x: x, []) == []


def test_parallel_executor_map_modules():
    from zero_data_model.base import Signal
    from zero_data_model.consciousness_core import ConsciousnessCore
    from zero_data_model.math_universe import MathematicalUniverse

    modules = [ConsciousnessCore(dim=8), MathematicalUniverse(dim=8)]
    executor = ParallelExecutor()
    signal = Signal(data=np.random.randn(8))
    results = executor.map_modules(modules, signal)
    assert len(results) == 2
    for r in results:
        assert r.data.shape[0] == 8


def test_hybrid_uses_real_qiskit_backend():
    hybrid = QuantumClassicalHybrid(dim=16, n_qubits=4)
    # Either qiskit is installed (real backend) or it gracefully falls back.
    assert hybrid.quantum_backend_name in {"qiskit", "simulator"}
    from zero_data_model.base import Signal
    out = hybrid.process(Signal(data=np.random.randn(16)))
    assert out.data.shape == (16,)
    assert out.metadata["quantum_backend"] == hybrid.quantum_backend_name


def test_annealer_jit_consistent_results_shape():
    """JIT-compiled annealer returns the same shape across calls."""
    annealer = QuantumAnnealer(n_vars=12)
    for _ in range(3):
        state, energy = annealer.optimize(n_iterations=20)
        assert state.shape == (12,)
        assert np.isfinite(energy)
