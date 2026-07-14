"""Hardware acceleration layer.

Auto-selects the best available backend:
- GPU via CuPy when CUDA is available, else NumPy on CPU.
- Real Qiskit quantum circuits when Qiskit is installed, else a simulator.
- Real IBM Quantum hardware via qiskit-ibm-runtime when configured.
- Parallel module execution via joblib when multiple cores are available.

All capabilities degrade gracefully to pure-NumPy on CPU when optional
dependencies are missing, so the core system stays zero-dependency.
"""

from __future__ import annotations

from .accel import asnumpy, backend_name, has_gpu, to_cpu, to_gpu, xp
from .parallel import ParallelExecutor
from .quantum import QiskitQuantumBackend, QuantumBackend, SimulatorQuantumBackend

try:  # pragma: no cover - optional dependency (qiskit-ibm-runtime)
    from .ibm_quantum import IBMQuantumBackend
except ImportError:  # pragma: no cover
    IBMQuantumBackend = None  # type: ignore[assignment]

__all__ = [
    "xp",
    "has_gpu",
    "backend_name",
    "to_gpu",
    "to_cpu",
    "asnumpy",
    "QuantumBackend",
    "QiskitQuantumBackend",
    "SimulatorQuantumBackend",
    "IBMQuantumBackend",
    "ParallelExecutor",
]
