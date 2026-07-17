"""Parallel module execution engine.

Runs the cognitive modules' ``process`` step concurrently using joblib when
multiple cores are available. Falls back to sequential execution on a single
core or when joblib is missing, so behaviour is always correct.
"""

from __future__ import annotations

import contextlib
import os
import threading
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

    Round-7 audit PERF7-1: the threading backend now uses a PERSISTENT
    ``ThreadPoolExecutor`` created in ``__init__`` and reused across ``map``
    calls, instead of spawning and tearing down a fresh pool per call. At
    ``n_workers=8``, ``ZeroDataModel.think`` calls ``map`` twice per cycle,
    so the old code paid ~16 thread-spawn+join operations per cycle
    (~1-2 ms overhead) — eliminated.

    Round-8 audit CONCUR8-2: fork-safety. A persistent ``ThreadPoolExecutor``
    holds live worker threads, and ``os.fork`` (used by gunicorn ``--preload``
    and Python ``multiprocessing``) does NOT carry those threads into the
    child — the child inherits a ``ThreadPoolExecutor`` whose threads are
    dead but whose internal state looks alive, so the next ``map`` deadlocks
    on ``as_completed``. We record the PID at pool creation time and
    recreate the pool lazily inside ``map`` when the PID has changed.

    Round-8 audit CONCUR8-3/4: lazy recreation is now lock-protected, and
    ``shutdown`` flips ``self._pool = None`` BEFORE signaling the workers so
    a concurrent ``map`` does not observe a half-shutdown pool.
    """

    def __init__(self, n_workers: int | None = None, backend: str = "threading"):
        self.n_workers = n_workers or _cpu_count()
        self.backend = backend
        # Round-6 audit RNG5-7: decouple the threading path from joblib. The
        # ``backend="threading"`` path (the default) uses ``ThreadPoolExecutor``
        # directly, NOT joblib — yet the old gate ``self.parallel = n_workers>1
        # and _HAS_JOBLIB`` silently disabled all parallelism when joblib was
        # missing, even though threads would have worked fine. Only the
        # process-pool path (``backend != "threading"``) actually needs joblib.
        self.parallel = self.n_workers > 1 and (
            self.backend == "threading" or _HAS_JOBLIB
        )
        # Round-7 audit PERF7-1: persistent thread pool, reused across calls.
        # Round-8 audit CONCUR8-2: record the PID so a forked child can detect
        # that the inherited pool is dead and recreate it.
        self._pool: ThreadPoolExecutor | None = None
        self._pool_pid: int = os.getpid()
        # Round-8 audit CONCUR8-3: serialize lazy recreation so two concurrent
        # ``map`` calls in a forked child do not race to build two pools
        # (the loser's pool would leak threads).
        self._recreate_lock = threading.Lock()
        if self.parallel and self.backend == "threading":
            self._pool = ThreadPoolExecutor(max_workers=self.n_workers)

    def _ensure_pool(self) -> ThreadPoolExecutor | None:
        """Return a live thread pool for the CURRENT pid, or ``None`` if the
        threading backend is disabled.

        Lazily recreates the pool when:
        * it was shut down (``self._pool is None``), or
        * the process PID has changed since the pool was built (fork child).

        CONCUR8-3: the recreate path is serialised by ``_recreate_lock`` so
        two concurrent ``map`` calls cannot build two competing pools.
        """
        if not self.parallel or self.backend != "threading":
            return None
        current_pid = os.getpid()
        pool = self._pool
        if pool is not None and self._pool_pid == current_pid:
            return pool
        # Either the pool is None (shutdown) or the PID changed (fork).
        # CONCUR8-3: serialise the recreation.
        with self._recreate_lock:
            # Re-check under the lock: another thread may have recreated
            # the pool while we were waiting.
            pool = self._pool
            if pool is not None and self._pool_pid == current_pid:
                return pool
            pool = ThreadPoolExecutor(max_workers=self.n_workers)
            self._pool = pool
            self._pool_pid = current_pid
            return pool

    def _ensure_pool_locked(self) -> ThreadPoolExecutor | None:
        """Same as ``_ensure_pool`` but assumes ``_recreate_lock`` is already
        held by the caller.

        Round-8 audit CONCUR8-7: ``map`` needs to atomically "get pool +
        submit all futures" so a concurrent ``shutdown`` cannot land between
        ``_ensure_pool`` returning a (now-stale) pool handle and the first
        ``pool.submit`` -- which would raise
        ``RuntimeError: cannot schedule new futures after interpreter shutdown``.
        Holding ``_recreate_lock`` across submit makes the get+submit
        sequence atomic w.r.t. ``shutdown`` (which also takes the lock).

        ``_ensure_pool`` itself takes ``_recreate_lock``, so re-entry would
        deadlock on a non-reentrant ``Lock``. This locked variant does the
        same recreate work WITHOUT re-taking the lock -- the caller already
        holds it.
        """
        if not self.parallel or self.backend != "threading":
            return None
        current_pid = os.getpid()
        pool = self._pool
        if pool is not None and self._pool_pid == current_pid:
            return pool
        # Pool is None or PID changed; recreate (caller holds the lock).
        pool = ThreadPoolExecutor(max_workers=self.n_workers)
        self._pool = pool
        self._pool_pid = current_pid
        return pool

    def shutdown(self) -> None:
        """Shut down the persistent thread pool (if any).

        CONCUR8-4: clear ``self._pool`` BEFORE signalling the workers so a
        concurrent ``map`` that already grabbed the (now-stale) pool handle
        still completes its ``as_completed`` loop, while new ``map`` calls
        lazily build a fresh pool. ``wait=False`` keeps shutdown non-blocking
        — critical for ``atexit`` / ``__del__`` paths that must not block.
        """
        # Take the lock so we cannot race with ``_ensure_pool``'s recreate
        # path: if ``shutdown`` wins the lock, any concurrent recreator
        # will see ``self._pool is None`` after we release and build a
        # fresh pool (which is the desired outcome — shutdown is local).
        with self._recreate_lock:
            pool = self._pool
            if pool is None:
                return
            # Clear first so concurrent ``map`` calls bypass this pool.
            self._pool = None
        # Release the lock before ``shutdown(wait=False)`` — the join is
        # non-blocking and does not need to hold the recreate lock.
        # ``shutdown`` should not raise, but a partially-initialised pool
        # (e.g. constructor failure mid-init) might. Swallow so ``__del__``
        # / ``atexit`` paths never propagate.
        with contextlib.suppress(Exception):
            pool.shutdown(wait=False)

    def __del__(self) -> None:
        # __del__ must never raise — interpreter shutdown path.
        with contextlib.suppress(Exception):
            self.shutdown()

    def map(self, func: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """Apply ``func`` to each item, preserving input order.

        Round-8 audit CONCUR8-5: when one future raises, ``fut.result()``
        re-raises the exception and the loop bails out -- but the OTHER
        in-flight futures keep running on the pool, leaking resources and
        delaying the exception's propagation to the caller (``as_completed``
        would still drain them if we let it, but we never get there). Now we
        ``cancel`` every not-yet-started future on the way out so the pool
        is free for the next ``map`` call. (``cancel`` only kills pending
        futures -- already-running ones finish, but that is unavoidable.)

        Round-8 audit CONCUR8-7: the get-pool + submit-all step is wrapped
        in ``_recreate_lock`` so a concurrent ``shutdown`` cannot land
        between ``_ensure_pool_locked`` returning a (now-stale) pool handle
        and the first ``pool.submit`` -- the previous code raised
        ``RuntimeError: cannot schedule new futures after interpreter
        shutdown`` in that window.
        """
        items_list = list(items)
        if not items_list:
            return []
        if not self.parallel or len(items_list) == 1:
            return [func(x) for x in items_list]

        if self.backend == "threading":
            # Round-7 audit PERF7-1: reuse the persistent pool.
            # Round-8 audit CONCUR8-2/3: ``_ensure_pool`` recreates the pool
            # lazily when the PID has changed (fork) or it was shut down,
            # and the recreate path is lock-protected.
            # Round-8 audit CONCUR8-7: hold ``_recreate_lock`` across the
            # get-pool + submit-all step so ``shutdown`` cannot land
            # mid-submit and leave us calling ``pool.submit`` on a
            # half-shutdown pool (RuntimeError).
            with self._recreate_lock:
                pool = self._ensure_pool_locked()
                if pool is None:  # pragma: no cover - defensive
                    return [func(x) for x in items_list]
                futures = {
                    pool.submit(func, x): i for i, x in enumerate(items_list)
                }
            results: list[R | None] = [None] * len(items_list)
            try:
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
            except BaseException:
                # CONCUR8-5: a future raised (or the caller was cancelled
                # / interrupted). Cancel every not-yet-started future so the
                # pool is free for the next ``map`` call instead of letting
                # the remaining work run to completion in the background.
                # ``cancel`` returns False for already-running futures --
                # those will finish on their own and we cannot help that.
                for fut in futures:
                    fut.cancel()
                raise
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
