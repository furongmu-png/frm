"""Hardware acceleration layer.

Auto-selects the best available backend:
- GPU via CuPy when CUDA is available, else NumPy on CPU.
- Real Qiskit quantum circuits when Qiskit is installed, else a simulator.
- Parallel module execution via joblib when multiple cores are available.

All capabilities degrade gracefully to pure-NumPy on CPU when optional
dependencies are missing, so the core system stays zero-dependency.
"""

from .accel import xp, has_gpu, backend_name, to_gpu, to_cpu, asnumpy
from .quantum import QuantumBackend, QiskitQuantumBackend, SimulatorQuantumBackend
from .parallel import ParallelExecutor

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
    "ParallelExecutor",
]
