from __future__ import annotations

from .accel import asnumpy, backend_name, has_gpu, to_cpu, to_gpu, xp
from .ibm_quantum import IBMQuantumBackend
from .parallel import ParallelExecutor
from .quantum import (
    QiskitQuantumBackend,
    QuantumBackend,
    SimulatorQuantumBackend,
)

# ``IBMQuantumBackend`` is conditionally available: it is set to ``None`` at
# runtime when ``qiskit-ibm-runtime`` is not installed (see ``__init__.py``).
IBMQuantumBackend: type[IBMQuantumBackend] | None

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
