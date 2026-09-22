from __future__ import annotations

from .accel import asnumpy, backend_name, has_gpu, to_cpu, to_gpu, xp
from .ibm_quantum import IBMQuantumBackend as _IBMQuantumBackend
from .parallel import ParallelExecutor
from .quantum import (
    QiskitQuantumBackend,
    QuantumBackend,
    SimulatorQuantumBackend,
)

# ``IBMQuantumBackend`` is conditionally available: it is set to ``None`` at
# runtime when ``qiskit-ibm-runtime`` is not installed (see ``__init__.py``).
# Aliasing the class import to ``_IBMQuantumBackend`` lets the type annotation
# below use the class as a TYPE while the runtime attribute
# ``IBMQuantumBackend`` is clearly a VALUE that may be the class or ``None`` --
# matching ``__init__.py``'s try/except exactly (same pattern as
# ``ZeroDataMCPServer`` in ``zero_data_model/__init__.pyi``).
IBMQuantumBackend: type[_IBMQuantumBackend] | None

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
