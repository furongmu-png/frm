"""IBM Quantum hardware backend.

Runs variational circuits on real IBM Quantum devices via qiskit-ibm-runtime.
Requires an IBM Quantum account and API token. When the token is not
configured, falls back to the local Qiskit StatevectorSampler.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from .quantum import QiskitQuantumBackend

try:  # pragma: no cover - optional dependency
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session
    from qiskit_ibm_runtime.exceptions import IBMApiError

    _HAS_IBM_RUNTIME = True
    _IMPORT_ERROR: str | None = None
except ImportError as _exc:  # pragma: no cover - exercised in CI without runtime
    _HAS_IBM_RUNTIME = False
    _IMPORT_ERROR = str(_exc)
    # Sentinel so that ``except IBMApiError`` stays syntactically valid even
    # when the runtime package is absent. The real-hardware code path is
    # unreachable in that case (guarded by ``self._available``), so this
    # alias is never actually used for matching.
    IBMApiError = Exception  # type: ignore[assignment, misc]


class IBMQuantumBackend(QiskitQuantumBackend):
    """Quantum backend that runs on real IBM Quantum hardware.

    Subclasses :class:`QiskitQuantumBackend` so it can reuse the variational
    ansatz (``_build_circuit``) and the local ``StatevectorSampler`` for the
    fallback path. When ``qiskit-ibm-runtime`` is not installed, or no IBM
    Quantum token is configured, the backend transparently degrades to the
    local simulator: ``evolve_and_measure`` still works and returns a
    probability vector of length ``2 * n_qubits``.
    """

    name = "ibm_quantum"

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        token: str | None = None,
        instance: str | None = None,
        backend_name: str | None = None,
    ):
        if not _HAS_IBM_RUNTIME:
            raise RuntimeError(
                "qiskit-ibm-runtime is not installed. Install it with "
                "`pip install 'qiskit-ibm-runtime>=0.20'` to use real IBM "
                "Quantum hardware; the factory will otherwise fall back to "
                f"the local Qiskit simulator. (import error: {_IMPORT_ERROR!r})"
            )
        # Initialize the Qiskit simulator base: provides ``_build_circuit``
        # and the local-sampler fallback used when no hardware is available.
        super().__init__(n_qubits=n_qubits, n_layers=n_layers)
        self._token_masked = token[:4] + "***" if token else None
        self.instance = instance
        self.backend_name = backend_name
        self._service = None
        self._backend = None
        self._available = False

        # Resolve the API token: explicit arg -> env var -> saved account.
        resolved_token = token
        if resolved_token is None:
            resolved_token = os.environ.get("IBM_QUANTUM_TOKEN")
        has_saved_account = False
        if resolved_token is None:
            try:
                saved = QiskitRuntimeService.saved_accounts()
                has_saved_account = bool(saved)
            except Exception:  # pragma: no cover - defensive
                has_saved_account = False

        if resolved_token is None and not has_saved_account:
            # No credentials anywhere: stay in simulator-fallback mode.
            self._available = False
            return

        # Credentials resolved: connect to the IBM Quantum service.
        # ``channel="ibm_quantum"`` is the classic channel for individual
        # IBM Quantum Experience accounts; ``"ibm_cloud"`` is an alternative
        # for IBM Cloud-provisioned resources.
        try:
            if resolved_token is not None:
                self._service = QiskitRuntimeService(
                    channel="ibm_quantum",
                    token=resolved_token,
                    instance=instance,
                )
            else:
                # Fall back to credentials saved via `saved_accounts()`.
                self._service = QiskitRuntimeService(
                    channel="ibm_quantum",
                    instance=instance,
                )
            # Select a backend: explicit name, else least-busy with enough qubits.
            if backend_name is not None:
                self._backend = self._service.backend(backend_name)
            else:
                self._backend = self._service.least_busy(min_num_qubits=n_qubits)
            self._available = True
        except (
            IBMApiError,
            OSError,
            RuntimeError,
        ) as exc:  # pragma: no cover - network/hardware path
            # Round-3 audit B-batch: narrow the catch so programming bugs
            # (TypeError/AttributeError/NameError) propagate instead of being
            # silently swallowed and degrading to the simulator. Only catch
            # errors that genuinely indicate the IBM service is unavailable:
            #   * IBMApiError  -- qiskit-ibm-runtime auth/API failures
            #   * OSError      -- network/transport/DNS errors
            #   * RuntimeError -- qiskit wraps some failures as RuntimeError
            warnings.warn(
                f"IBM Quantum service initialization failed ({type(exc).__name__}); "
                "falling back to the local simulator.",
                stacklevel=2,
            )
            self._available = False
            self._service = None
            self._backend = None

    @property
    def is_real_hardware(self) -> bool:
        """True only when running on actual IBM Quantum hardware."""
        return bool(self._available and self._backend is not None)

    def evolve_and_measure(
        self,
        params: np.ndarray,
        entangling: np.ndarray,
        n_shots: int = 1024,
    ) -> np.ndarray:
        """Run the variational circuit on real hardware (or fall back)."""
        if not self._available:
            # No credentials / service: delegate to the local simulator.
            return super().evolve_and_measure(params, entangling, n_shots)
        try:
            from qiskit import transpile

            # Reuse the shared ansatz builder from QiskitQuantumBackend.
            qc = self._build_circuit(params, entangling)
            tqc = transpile(qc, backend=self._backend)
            with Session(service=self._service, backend=self._backend) as session:
                sampler = SamplerV2(session=session)
                job = sampler.run([tqc], shots=n_shots)
                result = job.result()
            counts = self._extract_counts(result[0])
        except IBMApiError as exc:  # pragma: no cover - network/hardware path
            warnings.warn(
                f"IBM Quantum API error ({type(exc).__name__}); falling back to the "
                "local simulator.",
                stacklevel=2,
            )
            return super().evolve_and_measure(params, entangling, n_shots)
        except (
            OSError,
            RuntimeError,
        ) as exc:  # pragma: no cover - network/hardware path
            # Round-3 audit B-batch: same narrowing as the constructor. Lets
            # TypeError/AttributeError (programming bugs in the qiskit call
            # chain) propagate instead of silently degrading to the simulator.
            # IBMApiError is already caught by the except clause above.
            warnings.warn(
                f"IBM Quantum job failed ({type(exc).__name__}); "
                "falling back to the local simulator.",
                stacklevel=2,
            )
            return super().evolve_and_measure(params, entangling, n_shots)
        return self._counts_to_prob_vector(counts)

    @staticmethod
    def _extract_counts(pub_result) -> dict:
        """Extract the bitstring counts from a SamplerV2 pub result.

        SamplerV2 stores measurement data in ``pub_result.data``; the
        classical register name depends on the (transpiled) circuit. We
        prefer the default ``c`` register used by ``_build_circuit``, then
        ``meas``, then any register exposing ``get_counts``.
        """
        data = getattr(pub_result, "data", None)
        if data is None:
            return {}
        for attr in ("c", "meas"):
            obj = getattr(data, attr, None)
            if obj is not None and hasattr(obj, "get_counts"):
                return obj.get_counts()
        for name in dir(data):
            if name.startswith("_"):
                continue
            obj = getattr(data, name, None)
            if hasattr(obj, "get_counts"):
                return obj.get_counts()
        return {}

    def _counts_to_prob_vector(self, counts: dict) -> np.ndarray:
        """Collapse the bitstring histogram into a probability vector.

        Round-3 audit CRIT-2: the previous implementation allocated
        ``2 * n_qubits`` entries and indexed bit-by-bit, producing a per-qubit
        marginal that was semantically wrong AND dimensionally inconsistent
        with the simulator path (which returns ``2**n_qubits``). Now parses
        each bitstring as a binary basis-state index, matching the simulator's
        ``evolve_and_measure`` output exactly.
        """
        dim = 1 << self.n_qubits
        out = np.zeros(dim, dtype=float)
        total = float(sum(counts.values())) + 1e-8
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            # Parse the bitstring as a binary basis-state index (MSB-first,
            # matching Qiskit's convention). Guard against over-long strings.
            idx = int(bits, 2) if bits else 0
            if 0 <= idx < dim:
                out[idx] += count / total
        return out
