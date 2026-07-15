"""Parallel module execution engine.

Runs the cognitive modules' ``process`` step concurrently using joblib when
multiple cores are available. Falls back to sequential execution on a single
core or when joblib is missing, so behaviour is always correct.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

try:  # pragma: no cover - optional dependency
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except ImportError:  # pragma: no cover
    _HAS_JOBLIB = False

T = TypeVar("T")
R = TypeVar("R")


def _cpu_count() -> int:
    return max(1, (os.cpu_count() or 1))


class ParallelExecutor:
    """Execute callables concurrently, returning results in input order.

    Uses joblib's process pool when available and the core count is > 1;
    otherwise runs sequentially. Thread-pool fallback is used for light tasks
    to avoid IPC overhead.
    """

    def __init__(self, n_workers: int | None = None, backend: str = "threading"):
        self.n_workers = n_workers or _cpu_count()
        self.backend = backend
        self.parallel = self.n_workers > 1 and _HAS_JOBLIB

    def map(self, func: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """Apply ``func`` to each item, preserving input order."""
        items_list = list(items)
        if not items_list:
            return []
        if not self.parallel or len(items_list) == 1:
            return [func(x) for x in items_list]

        if self.backend == "threading":
            # Threads avoid pickling overhead for the GIL-bound numpy work
            # the cognitive modules perform, and are fastest for this load.
            with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                futures = {ex.submit(func, x): i for i, x in enumerate(items_list)}
                results: list[R | None] = [None] * len(items_list)
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
                # Preserve None results (Fix 21): cognitive modules may
                # legitimately return None, and dropping them desyncs the
                # output order from the input order.
                return results  # type: ignore[return-value]

        # Process-pool backend via joblib (heavier, true parallelism).
        return list(
            Parallel(n_jobs=self.n_workers, backend=self.backend)(
                delayed(func)(x) for x in items_list
            )
        )

    def map_modules(self, modules, signal):
        """Run ``module.process(signal)`` for every module concurrently."""
        return self.map(lambda m: m.process(signal), modules)

    @property
    def info(self) -> dict:
        return {
            "n_workers": self.n_workers,
            "backend": self.backend if self.parallel else "sequential",
            "joblib": _HAS_JOBLIB,
        }
