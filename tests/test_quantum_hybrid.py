# tests/test_quantum_hybrid.py
import numpy as np
from zero_data_model.quantum_hybrid import QuantumClassicalHybrid, SimulatedQuantumCircuit
from zero_data_model.base import Signal


def test_quantum_circuit_evolve():
    qc = SimulatedQuantumCircuit(n_qubits=4)
    state = np.random.randn(8)
    result = qc.evolve(state)
    assert len(result) == 8


def test_quantum_circuit_measure():
    qc = SimulatedQuantumCircuit(n_qubits=4)
    state = np.random.randn(8)
    probs = qc.measure(state)
    assert abs(np.sum(probs) - 1.0) < 1e-6


def test_hybrid_process():
    hybrid = QuantumClassicalHybrid(dim=16, n_qubits=4)
    signal = Signal(data=np.random.randn(16))
    result = hybrid.process(signal)
    assert result.data.shape == (16,)
    assert result.metadata["quantum_features"] is True


def test_optimization():
    hybrid = QuantumClassicalHybrid(dim=8)
    solution, energy = hybrid.solve_optimization()
    assert solution.shape == (8,)
    assert isinstance(energy, float)
