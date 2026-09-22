# src/zero_data_model/api.py
"""FastAPI REST service exposing the :class:`ZeroDataModel` as HTTP endpoints.

The app holds a single module-level :class:`ZeroDataModel` instance that is
lazily initialized on first use (``dim=32`` for fast startup). Numpy arrays
returned by the model are converted to plain Python lists so they serialize
to JSON cleanly (FastAPI / pydantic cannot serialize numpy directly).

Security & operational hardening:

* ``PathRequest.name`` is a *relative* name validated by a strict regex;
  the persistence layer sandboxes it under a configurable root, rejecting
  absolute paths, ``..`` traversal and symlinks.
* Optional API-key auth (``ZDM_API_KEY`` env var) gates every endpoint
  except ``/health`` and ``/ready``.
* Pydantic ``Field`` bounds cap every list/int input (CWE-400).
* Optional ``slowapi`` rate limits (CWE-770) -- skipped if not installed.
* Structured JSON logging + ``X-Request-ID`` tracing middleware.
* Optional ``prometheus-fastapi-instrumentator`` ``/metrics`` endpoint
  with custom ``zdm_*`` gauges/histograms.
* Centralized exception handler that logs details server-side and returns
  a generic ``{"detail": "internal error", "request_id": ...}`` body
  (no path/config leakage).
* Docs (``/docs``, ``/redoc``, ``/openapi.json``) are disabled when
  ``ZDM_ENV=production``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hmac
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from logging import Logger
from typing import Any

import numpy as np
from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .cache import timed_lru_cache
from .metrics import ZDM_METRICS
from .model import ZeroDataModel
from .persistence import ModelSerializer
from .security import (
    RateLimiter,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)

# --------------------------------------------------------------------------- #
# Optional dependencies -- imported defensively so the app keeps working
# when an observability/rate-limit library is not installed.
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - environment-dependent import
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    _HAS_SLOWAPI = True
except Exception:  # pragma: no cover - optional dep missing
    _HAS_SLOWAPI = False
    Limiter = None  # type: ignore[assignment, misc]
    RateLimitExceeded = None  # type: ignore[assignment, misc]
    _rate_limit_exceeded_handler = None  # type: ignore[assignment]
    get_remote_address = None  # type: ignore[assignment]

try:  # pragma: no cover - environment-dependent import
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    from prometheus_fastapi_instrumentator import Instrumentator

    _HAS_PROMETHEUS = True
except Exception:  # pragma: no cover - optional dep missing
    _HAS_PROMETHEUS = False
    Instrumentator = None  # type: ignore[assignment, misc]
    CONTENT_TYPE_LATEST = ""  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]

try:  # pragma: no cover - environment-dependent import
    import structlog

    _HAS_STRUCTLOG = True
except Exception:  # pragma: no cover - optional dep missing
    _HAS_STRUCTLOG = False
    structlog = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Logging setup -- JSON formatter via structlog if available, else stdlib
# JSON lines via a JsonFormatter. Level is read from ZDM_LOG_LEVEL.
# --------------------------------------------------------------------------- #


# LogRecord built-in attribute names that ``makeRecord`` refuses to
# overwrite via ``extra=`` (it raises ``KeyError``). Structured log fields
# sharing one of these names (e.g. ``module`` -- which the phase-7
# structured-logging decorator uses) must be stashed separately by
# ``_LoggerAdapter._emit`` so the stdlib fallback path does not crash.
_RESERVED_LOGRECORD_KEYS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
})


class _JsonFormatter(logging.Formatter):
    """Minimal stdlib JSON line formatter (used when structlog is absent)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Promote selected LogRecord extras to top-level keys so request
        # tracing fields (request_id, method, path, status, duration_ms)
        # land at the root of the JSON object, not nested under "extras".
        for key in (
            "request_id",
            "method",
            "path",
            "status",
            "duration_ms",
            "error",
        ):
            if key in record.__dict__:
                payload[key] = record.__dict__[key]
        # Merge structured fields whose names collide with reserved
        # LogRecord attributes (e.g. ``module``); ``_emit`` stashes them
        # here so they survive to the formatter without triggering
        # ``makeRecord``'s reserved-key overwrite guard.
        _overrides = record.__dict__.get("_structured_overrides")
        if isinstance(_overrides, dict):
            payload.update(_overrides)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _build_logger() -> Logger:
    """Build the module-level logger (JSON, level from ZDM_LOG_LEVEL)."""
    level_name = os.environ.get("ZDM_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    if _HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger("zero_data_model.api")  # type: ignore[return-value]

    logger = logging.getLogger("zero_data_model.api")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger


class _LoggerAdapter:
    """Thin adapter so call sites can use ``log.info(msg, k=v, ...)``.

    structlog accepts kwargs natively. stdlib logging does not -- it wants
    them under ``extra=``. This adapter normalizes the call site so the
    same ``log.info("request", request_id=..., method=...)`` works either
    way, and the JSON formatter (which reads those keys from
    ``LogRecord.__dict__``) picks them up.
    """

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def _emit(self, level: str, msg: str, **kwargs: Any) -> None:
        if _HAS_STRUCTLOG:
            getattr(self._logger, level)(msg, **kwargs)
        else:
            # stdlib: stash kwargs as LogRecord extras so _JsonFormatter
            # can promote them to top-level JSON keys. Keys that collide
            # with reserved LogRecord attributes (e.g. ``module``) would
            # raise ``KeyError`` in ``makeRecord``; collect those under a
            # single non-reserved attribute so the formatter can still
            # emit them without overwriting the real record field.
            extra: dict[str, Any] = {}
            reserved_overrides: dict[str, Any] = {}
            for _k, _v in kwargs.items():
                if _k == "exc_info":
                    continue
                if _k in _RESERVED_LOGRECORD_KEYS:
                    reserved_overrides[_k] = _v
                else:
                    extra[_k] = _v
            if reserved_overrides:
                extra["_structured_overrides"] = reserved_overrides
            exc_info = kwargs.get("exc_info")
            getattr(self._logger, level)(msg, extra=extra, exc_info=exc_info)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._emit("debug", msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._emit("info", msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._emit("warning", msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._emit("error", msg, **kwargs)


log = _LoggerAdapter(_build_logger())


# --------------------------------------------------------------------------- #
# Module-level singleton model
# --------------------------------------------------------------------------- #

# Module-level singleton. Lazily initialized so importing the module is cheap
# and the first request pays the construction cost.
_model: ZeroDataModel | None = None

# Round-8 audit CONCUR8-1: serialize get_model/set_model. FastAPI dispatches
# each request on a threadpool worker, so ``/load`` (which replaces the
# singleton) can race with a concurrent ``/think`` or ``/save`` — the latter
# could grab a half-constructed model (``__init__`` mid-flight) or hold a
# reference to the OLD model after ``set_model`` has swapped in the new one
# and the old model's ``ParallelExecutor`` has been shut down (R8-HIGH-2),
# causing a use-after-shutdown on ``map_modules``. The lock makes the swap
# atomic from the readers' perspective.
_model_lock = threading.Lock()


def get_model() -> ZeroDataModel:
    """Return the lazily-initialized module-level model instance.

    CONCUR8-1: takes ``_model_lock`` so a concurrent ``set_model`` cannot
    observe ``_model is None`` mid-construction (the previous ``if _model
    is None: _model = ZeroDataModel(...)`` would let two threads both build
    a model, then both assign — leaking the loser's ``ParallelExecutor``).
    """
    global _model
    with _model_lock:
        if _model is None:
            _model = ZeroDataModel(dim=32)
        return _model


def set_model(model: ZeroDataModel | None) -> None:
    """Replace the module-level model (used by /load and tests).

    R8-HIGH-2: shut down the OLD model's ``ParallelExecutor`` before the
    swap so its persistent ``ThreadPoolExecutor`` worker threads do not
    leak — the old code replaced ``_model`` and left the old model's
    thread pool running indefinitely (each ``/load`` leaked ``n_workers``
    threads + a ``ThreadPoolExecutor`` internal queue).

    CONCUR8-1: the swap + shutdown are atomic w.r.t. ``get_model``.
    """
    global _model
    with _model_lock:
        old = _model
        _model = model
    # Shutdown the old model OUTSIDE ``_model_lock`` so a slow
    # ``pool.shutdown(wait=False)`` does not block ``/think``. ``wait=False``
    # already makes this non-blocking, but the lock-drop is still cleaner.
    if old is not None:
        with contextlib.suppress(Exception):
            old.parallel_executor.shutdown()
    # P2.11 性能优化: 模型实例被替换后，所有读缓存的端点都必须失效，
    # 否则新模型的前若干次读会被旧模型的快照污染。统一清空缓存注册表。
    _clear_endpoint_caches()


# --------------------------------------------------------------------------- #
# P2.11 性能优化: 读端点缓存层
#
# 下列 ``_cached_*`` 辅助函数为"不频繁变化"的只读端点提供 TTL 缓存：
#   - ``/architect/stats`` 与 ``/architect/dormant``: 架构可塑性统计仅在
#     每 ``eval_interval`` 个 think() 周期更新一次，30s TTL 足够新鲜。
#   - ``/episodic_graph/node_count`` 与 ``/episodic_graph/recent``: 情节图
#     节点数随周期增长但变化缓慢，60s TTL 平衡新鲜度与开销。
#
# 设计要点：
#   * 缓存的是 ``_to_jsonable`` 后的纯 Python dict（与 numpy 解耦），
#     不会持有模型内部数组的引用。
#   * 用哨兵 ``_CACHE_DISABLED`` 表示"模块未启用"，避免缓存 JSONResponse
#     对象（FastAPI 不应复用响应实例）。端点根据哨兵重建 503 响应。
#   * ``set_model`` 替换模型实例时通过 ``_clear_endpoint_caches`` 清空所有
#     缓存，防止旧模型快照污染新模型的前几次读。
#   * 不缓存 ``/think``（每次结果不同）、``/health``、``/ready``（探针必须
#     反映实时状态）。
# --------------------------------------------------------------------------- #
_CACHE_DISABLED = object()
_ENDPOINT_CACHE_REGISTRY: list = []


def _register_cache(fn):
    """Register a timed_lru_cache-wrapped function for bulk invalidation."""
    _ENDPOINT_CACHE_REGISTRY.append(fn)
    return fn


@_register_cache
@timed_lru_cache(maxsize=1, ttl=30)
def _cached_architect_stats() -> Any:
    """Cached ``architect.stats`` (30s TTL). Returns ``_CACHE_DISABLED`` if
    the architect module is not enabled."""
    model = get_model()
    with model._lock:
        arch = model.architect
        if arch is None:
            return _CACHE_DISABLED
        return _to_jsonable(arch.stats)


@_register_cache
@timed_lru_cache(maxsize=1, ttl=30)
def _cached_architect_dormant() -> Any:
    """Cached dormant module names (30s TTL). Returns ``_CACHE_DISABLED`` if
    the architect module is not enabled."""
    model = get_model()
    with model._lock:
        arch = model.architect
        if arch is None:
            return _CACHE_DISABLED
        return {"dormant": list(arch.dormant_names())}


@_register_cache
@timed_lru_cache(maxsize=1, ttl=60)
def _cached_episodic_node_count() -> Any:
    """Cached episodic-graph node count (60s TTL). Returns
    ``_CACHE_DISABLED`` if the episodic_graph module is not enabled."""
    model = get_model()
    with model._lock:
        eg = model.episodic_graph
        if eg is None:
            return _CACHE_DISABLED
        return {"node_count": int(eg.node_count)}


@_register_cache
@timed_lru_cache(maxsize=32, ttl=60)
def _cached_episodic_recent(n: int) -> Any:
    """Cached recent episodes (60s TTL, keyed by ``n``). Returns
    ``_CACHE_DISABLED`` if the episodic_graph module is not enabled."""
    model = get_model()
    with model._lock:
        eg = model.episodic_graph
        if eg is None:
            return _CACHE_DISABLED
        episodes = eg.get_recent(n)
        return {"episodes": _to_jsonable([vars(e) for e in episodes])}


def _clear_endpoint_caches() -> None:
    """Clear all registered endpoint caches (called on ``set_model``)."""
    for fn in _ENDPOINT_CACHE_REGISTRY:
        with contextlib.suppress(Exception):
            fn.cache_clear()


# Round-6 audit NEW5-2 / NEW5-3: validate that an input array is finite
# (no NaN/Inf). NaN/Inf in a series poisons np.polyfit / np.mean / z-score;
# NaN/Inf in ``think.input`` or ``recognize.image`` propagates through the
# predictive hierarchy. Each array-accepting endpoint calls this helper
# right after ``np.asarray``.
def _ensure_finite(arr: np.ndarray, name: str) -> None:
    """Raise HTTPException(400) if ``arr`` contains NaN or Inf."""
    if not np.all(np.isfinite(arr)):
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be finite (no NaN or Inf)",
        )


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python natives.

    Used by the ``/emergence/cycle`` endpoint whose response shape is too
    dynamic for a strict pydantic schema. Non-finite floats (NaN, +/-Inf)
    become ``None`` because standard JSON has no representation for them
    (the topology module's persistence diagram uses +Inf for essential
    homology classes).
    """
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def _module_disabled_response(name: str) -> JSONResponse:
    """Return a 503 response body for an un-enabled cognitive-upgrade module.

    Phase 7 cognitive modules (architect / layered_predictor / temporal_memory
    / episodic_graph / semantic_index / logic_layer / causal_inference /
    meta_cognition / experiment_planner / hypothesis_tester) are gated behind
    ``enable_*`` feature flags on :class:`ZeroDataModel`, and the multiagent
    modules (world / communication / culture) are not yet integrated into the
    model at all. When the corresponding attribute is ``None`` the endpoint
    returns this 503 so callers can distinguish "not enabled" from a real
    error.
    """
    return JSONResponse(
        status_code=503,
        content={"detail": f"module {name} not enabled", "enabled": False},
    )


def _parse_obstacles(rows: list[list[float]]) -> list[tuple[np.ndarray, float]]:
    """Convert API ``obstacles`` rows to ``(center, radius)`` tuples.

    Each row is ``[center_x, center_y, ..., radius]`` — the LAST element
    is the obstacle radius, the leading elements form the center
    coordinates. Returns a list of ``(np.ndarray, float)`` matching the
    format expected by :meth:`ZeroDataModel.check_collision`.
    """
    parsed: list[tuple[np.ndarray, float]] = []
    for i, row in enumerate(rows):
        if len(row) < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"obstacles[{i}] must have >= 2 values "
                    f"(center + radius), got {len(row)}"
                ),
            )
        center = np.asarray(row[:-1], dtype=float)
        _ensure_finite(center, f"obstacles[{i}].center")
        radius = float(row[-1])
        if not np.isfinite(radius) or radius < 0.0:
            raise HTTPException(
                status_code=400,
                detail=f"obstacles[{i}].radius must be finite and >= 0, got {radius}",
            )
        parsed.append((center, radius))
    return parsed


# --------------------------------------------------------------------------- #
# API key auth dependency (CWE-306)
# --------------------------------------------------------------------------- #


def _configured_api_key() -> str | None:
    """Return the configured API key (env ZDM_API_KEY), or None if unset."""
    raw = os.environ.get("ZDM_API_KEY")
    if raw is not None and raw.strip() == "":
        # An explicitly empty value is a configuration error, not "unset".
        raise RuntimeError(
            "ZDM_API_KEY is set but empty; refusing to start with auth disabled"
        )
    return raw or None


def verify_api_key(api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """FastAPI dependency: reject requests when ZDM_API_KEY is set and the
    ``X-API-Key`` header does not match.

    When ``ZDM_API_KEY`` is unset (local dev), auth is disabled.
    """
    expected = _configured_api_key()
    if not expected:
        return
    ok = api_key is not None and hmac.compare_digest(
        api_key.encode("utf-8"), expected.encode("utf-8")
    )
    if not ok:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# --------------------------------------------------------------------------- #
# Rate limiting (CWE-770) -- slowapi if installed, else no-op decorator.
# --------------------------------------------------------------------------- #

if _HAS_SLOWAPI:
    limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
else:
    limiter = None


def _limit(rate: str):  # type: ignore[no-untyped-def]
    """Return a slowapi limit decorator, or a no-op if slowapi is missing."""
    if _HAS_SLOWAPI and limiter is not None:
        return limiter.limit(rate)

    def _wrap(func):  # type: ignore[no-untyped-def]
        return func

    return _wrap


# --------------------------------------------------------------------------- #
# Prometheus custom metrics.
#
# All ``zdm_*`` metric objects are centrally owned by
# :mod:`zero_data_model.metrics` (``ZDM_METRICS``) and re-exported here as
# backwards-compatible module-level aliases so existing call sites
# (``_think_duration`` / ``_cycle_count`` / ``_free_energy``) keep working
# unchanged. When ``prometheus_client`` is absent every object is a no-op
# (see ``metrics._NoopMetric``), so these aliases are always non-``None``.
# --------------------------------------------------------------------------- #

_think_duration = ZDM_METRICS.think_duration
_cycle_count = ZDM_METRICS.cycle_count
_free_energy = ZDM_METRICS.free_energy


# --------------------------------------------------------------------------- #
# Phase-7 endpoint structured-logging decorator.
#
# Wraps a phase-7 cognitive-upgrade endpoint so every call emits a
# structured ``phase7.endpoint.call`` log on entry, ``phase7.endpoint.ok``
# (with ``duration_ms``) on success, and ``phase7.endpoint.error`` on
# exception -- and increments the generic cognitive-module call / duration
# / error Prometheus counters (``zdm_cognitive_module_*``).
#
# ``functools.wraps`` preserves the wrapped function's signature so FastAPI
# dependency injection (``Request`` / ``Depends``) keeps working through the
# wrapper.
# --------------------------------------------------------------------------- #


def _log_phase7(module: str, method: str):  # type: ignore[no-untyped-def]
    """Decorate a phase-7 endpoint to log entry/exit/exception + latency."""

    def _decorator(func):  # type: ignore[no-untyped-def]
        @functools.wraps(func)
        async def _wrapper(*args: Any, **kwargs: Any) -> Any:
            log.info("phase7.endpoint.call", module=module, method=method)
            ZDM_METRICS.cognitive_module_calls_total.labels(
                module=module, method=method
            ).inc()
            _hist = ZDM_METRICS.cognitive_module_duration_seconds.labels(
                module=module, method=method
            )
            _start = time.perf_counter()
            try:
                _result = await func(*args, **kwargs)
            except Exception as exc:
                _hist.observe(time.perf_counter() - _start)
                ZDM_METRICS.cognitive_module_errors_total.labels(
                    module=module, method=method
                ).inc()
                log.warning(
                    "phase7.endpoint.error",
                    module=module,
                    method=method,
                    error=str(exc),
                )
                raise
            _elapsed = time.perf_counter() - _start
            _hist.observe(_elapsed)
            log.info(
                "phase7.endpoint.ok",
                module=module,
                method=method,
                duration_ms=round(_elapsed * 1000, 3),
            )
            return _result

        return _wrapper

    return _decorator


# --------------------------------------------------------------------------- #
# Request schemas (with input-validation bounds, CWE-400)
# --------------------------------------------------------------------------- #


class ThinkRequest(BaseModel):
    input: list[float] | None = Field(default=None, max_length=1024)


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8192)


class SimilarityRequest(BaseModel):
    a: str = Field(min_length=1, max_length=8192)
    b: str = Field(min_length=1, max_length=8192)


class GenerateRequest(BaseModel):
    seed: str = Field(min_length=1, max_length=4096)
    length: int = Field(default=32, ge=1, le=256)


class ForecastRequest(BaseModel):
    series: list[float] = Field(max_length=10000)
    horizon: int = Field(default=5, ge=1, le=1000)


class AnomaliesRequest(BaseModel):
    series: list[float] = Field(max_length=10000)


class TrendRequest(BaseModel):
    series: list[float] = Field(max_length=10000)


class RecognizeRequest(BaseModel):
    image: list[list[float]] = Field(max_length=512)

    @field_validator("image")
    @classmethod
    def _validate_rows(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("image must be non-empty")
        if len(v) > 512:
            raise ValueError("image must have at most 512 rows")
        # Round-8 audit API8-2: validate the OUTER list + per-row length cap
        # here (these are pure schema constraints and yield 422 on violation).
        # The STRUCTURAL checks (each row non-empty, rectangular shape) are
        # done in the ``/recognize`` endpoint so that ``[[]]`` -- the case
        # explicitly exercised by ``test_post_recognize_empty_image_returns_4xx``
        # -- returns 400 (endpoint business check) rather than 422 (schema).
        for row in v:
            if not isinstance(row, list):
                raise ValueError("image must be a list of rows")
            if len(row) > 512:
                raise ValueError("each image row must have at most 512 values")
        return v


class PathRequest(BaseModel):
    """Relative snapshot name (NOT a full path) for /save and /load.

    Only alphanumerics, underscores, hyphens and forward slashes are
    permitted; ``..`` is rejected by the regex AND by the persistence
    layer's sandbox. The persistence layer joins this name to its root
    directory.
    """

    name: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-/]+$",
    )


# --------------------------------------------------------------------------- #
# Response schemas
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    status: str


class RootResponse(BaseModel):
    status: str
    version: str
    hardware_info: dict


class ReadyResponse(BaseModel):
    status: str
    model_ready: bool


class ThinkResponse(BaseModel):
    cycle: int
    output: list[float]
    confidence: float


class ClassifyResponse(BaseModel):
    topic: str
    confidence: float


class SimilarityResponse(BaseModel):
    similarity: float


class GenerateResponse(BaseModel):
    text: str


class ForecastResponse(BaseModel):
    forecast: list[float]


class AnomaliesResponse(BaseModel):
    anomalies: list[bool]


class TrendResponse(BaseModel):
    trend_slope: float
    regime: str
    curvature: float
    geodesic_deviation: float
    isomorphism_score: float


class RecognizeResponse(BaseModel):
    shape: str
    confidence: float


class SaveResponse(BaseModel):
    saved: bool
    # Round-7 audit API7-1-1: the ``name`` field is part of the HTTP
    # response body (the ``Location`` header carries the same value as a
    # canonical URI). Declaring it here makes the OpenAPI schema match
    # the actual 201 response body ``{"saved": True, "name": ...}``.
    name: str


class LoadResponse(BaseModel):
    loaded: bool


# --------------------------------------------------------------------------- #
# Emergence schemas (spec §2.3) — /emergence/* endpoints
# --------------------------------------------------------------------------- #


class EmergencePerceiveRequest(BaseModel):
    data: list[list[float]] = Field(..., min_length=2, max_length=1000)
    max_dim: int | None = Field(None, ge=0, le=4)


class EmergencePerceiveResponse(BaseModel):
    betti_numbers: list[int]
    persistence_entropy: float
    euler_characteristic: int
    n_points: int
    max_eps: float
    # Omitted by default (too large); add ?full_diagram=true to include.
    persistence_diagram: list[list[float | None]] | None = None


class EmergenceCausalRequest(BaseModel):
    data: list[list[float]] = Field(..., min_length=2, max_length=1000)
    var_names: list[str] | None = Field(None, max_length=20)
    method: str | None = Field(None, pattern="^(pc|lingam|correlation)$")


class EmergenceCausalResponse(BaseModel):
    adjacency: list[list[int]]
    edges: list[list[int]]
    n_edges: int
    is_acyclic: bool
    method: str
    var_names: list[str]


class EmergenceTrajectoryRequest(BaseModel):
    start_state: list[float] = Field(..., min_length=1, max_length=100)
    end_state: list[float] = Field(..., min_length=1, max_length=100)
    n_steps: int = Field(32, ge=0, le=1000)
    obstacles: list[list[float]] | None = Field(None, max_length=50)
    margin: float | None = Field(None, ge=1e-9, le=10.0)


class EmergenceTrajectoryResponse(BaseModel):
    trajectory: list[list[float]]
    lagrangian: list[float]
    action: float
    converged: bool
    iterations: int
    obstacle_violations: int | None = None


class EmergenceSampleRequest(BaseModel):
    mean: list[float] = Field(..., min_length=1, max_length=100)
    std: float = Field(1.0, ge=1e-9, le=100.0)
    n_samples: int = Field(100, ge=1, le=1000)


class EmergenceSampleResponse(BaseModel):
    mean: list[float]
    std: list[float]
    accept_rate: float
    ess: float
    converged: bool
    # V4-NEW-L004 (v4.2): expose raw samples to match MCP output contract.
    # May be empty when n_samples=0 (engine returns shape (0, dim)).
    samples: list[list[float]] = Field(default_factory=list)


class EmergenceRecallRequest(BaseModel):
    query: list[float] = Field(..., min_length=1, max_length=100)
    n_steps: int = Field(100, ge=0, le=1000)


class EmergenceRecallResponse(BaseModel):
    label: str | int | None
    similarity: float
    emerged: bool
    trajectory: list[list[float]]
    converged: bool
    divergence: float
    # V4-NEW-L003 (v4.2): expose nearest_pattern to match MCP output contract.
    # None when memory is empty, query is non-finite, or engine-level
    # degradation kicks in (spec §7.3).
    nearest_pattern: list[float] | None = None


class EmergenceCycleRequest(BaseModel):
    observation: list[list[float]] = Field(..., min_length=2, max_length=1000)


# --------------------------------------------------------------------------- #
# Phase 6 — Memory / Planning / Multimodal / RL request schemas.
# --------------------------------------------------------------------------- #

class MemoryEncodeRequest(BaseModel):
    observation: list[float] = Field(..., min_length=1, max_length=4096)
    label: str | int | None = None


class MemoryRetrieveRequest(BaseModel):
    query: list[float] = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(5, ge=1, le=100)


class PlanningTrajectoryRequest(BaseModel):
    start_state: list[float] = Field(..., min_length=1, max_length=4096)
    goal_state: list[float] = Field(..., min_length=1, max_length=4096)
    n_steps: int = Field(32, ge=0, le=1000)
    obstacles: list[list[float]] | None = Field(None, max_length=200)
    margin: float | None = Field(None, ge=1e-9, le=10.0)


class PlanningDecomposeRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=256)
    max_depth: int | None = Field(None, ge=1, le=20)


class PlanningSequenceRequest(BaseModel):
    adjacency: list[list[int]] = Field(..., min_length=1, max_length=200)


class MultimodalAlignRequest(BaseModel):
    observations_a: list[list[float]] = Field(..., min_length=2, max_length=1000)
    observations_b: list[list[float]] = Field(..., min_length=2, max_length=1000)


class MultimodalFuseRequest(BaseModel):
    embeddings: list[list[float]] = Field(..., min_length=2, max_length=20)
    strategy: str | None = Field(None, pattern="^(mean|concat|weighted)$")


class MultimodalContrastiveRequest(BaseModel):
    batch_a: list[list[float]] = Field(..., min_length=2, max_length=256)
    batch_b: list[list[float]] = Field(..., min_length=2, max_length=256)


class RLStepRequest(BaseModel):
    state: int = Field(..., ge=0, le=4096)
    action: int = Field(..., ge=0, le=4096)


class RLTrainQRequest(BaseModel):
    n_episodes: int = Field(50, ge=1, le=2000)
    max_steps_per_episode: int = Field(50, ge=1, le=1000)


class RLSearchMCTSRequest(BaseModel):
    root_state: int = Field(..., ge=0, le=4096)
    n_simulations: int = Field(50, ge=1, le=2000)
    max_depth: int = Field(10, ge=1, le=200)


# ------------------------------------------------------------------
# Phase 7 — Audio request schemas.
# ------------------------------------------------------------------


class AudioSignalRequest(BaseModel):
    """Shared schema for all audio endpoints.

    Accepts a 1D or 2D (multi-channel) signal as a flat list (1D) or
    list-of-channels (2D). Sample rate must be > 0.
    """
    signal: list[float] = Field(..., min_length=1, max_length=200000)
    sample_rate: int = Field(16000, ge=1, le=192000)


# ------------------------------------------------------------------
# Phase 7 — Graph request schemas.
# ------------------------------------------------------------------


class GraphAdjacencyRequest(BaseModel):
    """Shared schema for graph endpoints taking a single adjacency matrix."""
    adjacency: list[list[float]] = Field(..., min_length=1, max_length=512)


class GraphEncodeRequest(BaseModel):
    adjacency: list[list[float]] = Field(..., min_length=1, max_length=512)
    node_features: list[list[float]] | None = Field(None)


class GraphPathRequest(BaseModel):
    adjacency: list[list[float]] = Field(..., min_length=1, max_length=512)
    source: int = Field(..., ge=0, le=4096)
    target: int = Field(..., ge=0, le=4096)


class GraphIsomorphismRequest(BaseModel):
    adjacency_a: list[list[float]] = Field(..., min_length=1, max_length=512)
    adjacency_b: list[list[float]] = Field(..., min_length=1, max_length=512)


class GraphTrackRequest(BaseModel):
    snapshots: list[list[list[float]]] = Field(..., min_length=2, max_length=64)


# ------------------------------------------------------------------
# Phase 7 — Robotics request schemas.
# ------------------------------------------------------------------


class RoboticsMotionRequest(BaseModel):
    """Plan a smooth trajectory through waypoints (cubic spline)."""
    waypoints: list[list[float]] = Field(..., min_length=2, max_length=512)
    n_steps: int = Field(100, ge=1, le=2000)


class RoboticsJointAnglesRequest(BaseModel):
    """Forward / inverse kinematics input."""
    joint_angles: list[float] = Field(..., min_length=1, max_length=64)


class RoboticsInverseKinematicsRequest(BaseModel):
    target: list[float] = Field(..., min_length=2, max_length=2)
    seed: list[float] | None = Field(None, min_length=1, max_length=64)


class RoboticsFuseRequest(BaseModel):
    """Inverse-variance weighted fusion of multiple sensor measurements."""
    measurements: list[list[float]] = Field(..., min_length=1, max_length=64)
    variances: list[float] = Field(..., min_length=1, max_length=64)


class RoboticsKalmanRequest(BaseModel):
    """Sequential Kalman-style Bayesian update."""
    prior: list[float] = Field(..., min_length=1, max_length=4096)
    prior_var: float = Field(..., gt=0.0, le=1e6)
    measurement: list[float] = Field(..., min_length=1, max_length=4096)
    meas_var: float = Field(..., gt=0.0, le=1e6)


class RoboticsGaitRequest(BaseModel):
    n_steps: int = Field(100, ge=1, le=2000)
    gait_type: str = Field("walk", pattern="^(walk|trot|bound)$")


class RoboticsOptimizeRequest(BaseModel):
    trajectory: list[list[float]] = Field(..., min_length=4, max_length=4096)
    n_iter: int = Field(10, ge=1, le=2000)


class RoboticsObstaclesRequest(BaseModel):
    """Shared schema for collision / MPC endpoints.

    Each obstacle row is ``[center_x, center_y, ..., radius]`` — the LAST
    element is the obstacle radius and the leading elements form the
    center coordinates.
    """
    obstacles: list[list[float]] = Field(default_factory=list, max_length=128)


class RoboticsCollisionRequest(RoboticsObstaclesRequest):
    position: list[float] = Field(..., min_length=1, max_length=64)
    radius: float = Field(0.1, ge=0.0, le=10.0)


class RoboticsPathCollisionRequest(RoboticsObstaclesRequest):
    path: list[list[float]] = Field(..., min_length=1, max_length=4096)
    radius: float = Field(0.1, ge=0.0, le=10.0)


class RoboticsMPCRequest(BaseModel):
    current_state: list[float] = Field(..., min_length=1, max_length=64)
    target_state: list[float] = Field(..., min_length=1, max_length=64)
    obstacles: list[list[float]] | None = Field(None, max_length=128)


# --------------------------------------------------------------------------- #
# Phase 7 — Time request schemas.
# --------------------------------------------------------------------------- #


class TimeSeriesRequest(BaseModel):
    """Shared schema for all Time endpoints that consume a single 1D series."""
    series: list[float] = Field(..., min_length=1, max_length=65536)


class TimeCyclePhaseRequest(BaseModel):
    """Cycle phase tracker accepts an optional explicit period."""
    series: list[float] = Field(..., min_length=1, max_length=65536)
    period: int | None = Field(None, ge=0, le=65536)


# --------------------------------------------------------------------------- #
# Phase 7 — Code request schemas.
# --------------------------------------------------------------------------- #


class CodeSourceRequest(BaseModel):
    """Shared schema for all Code endpoints that consume a single source."""
    source: str = Field(..., min_length=1, max_length=1048576)


class CodeCompareRequest(BaseModel):
    """Compare takes two source strings."""
    source_a: str = Field(..., min_length=1, max_length=1048576)
    source_b: str = Field(..., min_length=1, max_length=1048576)


# --------------------------------------------------------------------------- #
# Phase 7 — Reasoning request schemas.
# --------------------------------------------------------------------------- #


class ReasoningPropInferRequest(BaseModel):
    """Forward-chain propositional facts + rules to a fixpoint."""
    facts: dict[str, bool] = Field(default_factory=dict)
    rules: list[list[str]] = Field(default_factory=list)
    negation_rules: list[list[str]] = Field(default_factory=list)


class ReasoningSyllogismRequest(BaseModel):
    """Categorical syllogism (form, subject, predicate) for each premise."""
    major: list = Field(..., min_length=3, max_length=3)
    minor: list = Field(..., min_length=3, max_length=3)


class ReasoningInductRequest(BaseModel):
    """Induce a rule from labeled examples."""
    examples: list[dict] = Field(..., min_length=1, max_length=1024)
    labels: list[bool] = Field(..., min_length=1, max_length=1024)


class ReasoningAnalogizeRequest(BaseModel):
    """Find a structural analogy between source and target dicts."""
    source: dict = Field(..., min_length=1)
    target: dict = Field(..., min_length=1)


class ReasoningAbduceRequest(BaseModel):
    """Pick the best explanation for an observation."""
    observation: str = Field(..., min_length=1, max_length=65536)
    hypotheses: list[str] = Field(..., min_length=1, max_length=128)
    priors: list[float] | None = Field(None, max_length=128)


class ReasoningDefaultRule(BaseModel):
    """A single defeasible default rule."""
    rule: list = Field(..., min_length=2, max_length=3)
    exception: list | None = Field(None, max_length=8)


class ReasoningDefaultsRequest(BaseModel):
    """Apply defeasible default rules to a set of facts."""
    defaults: list[ReasoningDefaultRule] = Field(default_factory=list, max_length=128)
    facts: dict[str, bool] = Field(default_factory=dict)


class ReasoningCausalRequest(BaseModel):
    """Trace a causal chain from a starting node."""
    links: list[list[str]] = Field(default_factory=list, max_length=1024)
    start: str = Field(..., min_length=1, max_length=1024)
    max_depth: int = Field(5, ge=0, le=100)


# --------------------------------------------------------------------------- #
# Phase 7 — Causal request schemas.
# --------------------------------------------------------------------------- #


class CausalDecisionTreeRequest(BaseModel):
    """Fit a decision tree to (features, labels)."""
    features: list[list[float]] = Field(..., min_length=1, max_length=65536)
    labels: list[Any] = Field(..., min_length=1, max_length=65536)


class CausalGameRequest(BaseModel):
    """Analyze a 2-player normal-form game."""
    payoff_a: list[list[float]] = Field(..., min_length=1, max_length=1024)
    payoff_b: list[list[float]] | None = Field(None, max_length=1024)


class CausalCounterfactualRequest(BaseModel):
    """Estimate counterfactual outcome under do(X[index] = value)."""
    observed: list[float] = Field(..., min_length=1, max_length=65536)
    index: int = Field(0, ge=0, le=65535)
    value: float = 0.0


class CausalBanditRequest(BaseModel):
    """Select the next bandit arm given per-arm reward histories."""
    rewards_history: list[list[float]] = Field(
        default_factory=list, max_length=1024
    )


class CausalPOMDPRequest(BaseModel):
    """Solve a (PO)MDP via value iteration."""
    transitions: list[list[list[float]]] = Field(..., max_length=256)
    observations: list[list[float]] = Field(default_factory=list, max_length=256)
    rewards: list[Any] = Field(default_factory=list, max_length=256)


class CausalGraphRequest(BaseModel):
    """Discover a causal graph from observational data."""
    data: list[list[float]] = Field(..., min_length=2, max_length=65536)
    var_names: list[str] | None = Field(None, max_length=4096)


class CausalInterveneRequest(BaseModel):
    """Estimate the effect of do(X[intervention_var] = value)."""
    data: list[list[float]] = Field(..., min_length=2, max_length=65536)
    intervention_var: int = Field(..., ge=0, le=4095)
    intervention_value: float


# --------------------------------------------------------------------------- #
# Phase 7 cognitive-upgrade request schemas (plasticity / cogtime / cogmem /
# knowledge / metacog / experiment / multiagent).
# --------------------------------------------------------------------------- #


class ArchitectReactivateRequest(BaseModel):
    """Reactivate a dormant module by name."""
    name: str = Field(..., min_length=1, max_length=256)


class EpisodicPlanRequest(BaseModel):
    """Plan a minimum-free-energy path between two state vectors."""
    start_state: list[float] = Field(..., min_length=1, max_length=4096)
    goal_state: list[float] = Field(..., min_length=1, max_length=4096)
    horizon: int = Field(20, ge=1, le=4096)


class SemanticQueryRequest(BaseModel):
    """Query the semantic index for the k nearest entries."""
    vec: list[float] = Field(..., min_length=1, max_length=4096)
    k: int = Field(5, ge=1, le=1024)


class LogicRuleRequest(BaseModel):
    """Register a fuzzy logic rule."""
    name: str = Field(..., min_length=1, max_length=256)
    antecedents: list[str] = Field(default_factory=list, max_length=256)
    consequent: str = Field(..., min_length=1, max_length=256)
    weight: float = Field(1.0, ge=0.0, le=1000.0)
    description: str = Field("", max_length=4096)


class LogicPredicateRequest(BaseModel):
    """Set a predicate truth value (clamped to [0, 1])."""
    name: str = Field(..., min_length=1, max_length=256)
    value: float = Field(..., ge=0.0, le=1.0)


class CausalTransitionRequest(BaseModel):
    """Set the causal transition matrix (must be a square 2-D matrix)."""
    matrix: list[list[float]] = Field(..., min_length=1, max_length=4096)


class CausalDoRequest(BaseModel):
    """Apply a single do-intervention: do(X[index] = value)."""
    index: int = Field(..., ge=0, le=4095)
    value: float


class CausalInferenceCounterfactualRequest(BaseModel):
    """Estimate a counterfactual state under do(X[index] = value)."""
    observed: list[float] = Field(..., min_length=1, max_length=4096)
    index: int = Field(..., ge=0, le=4095)
    value: float


class CausalConfoundersRequest(BaseModel):
    """Identify potential confounders between two target variables."""
    var_a: int = Field(..., ge=0, le=4095)
    var_b: int = Field(..., ge=0, le=4095)


class ExperimentSelectBestRequest(BaseModel):
    """Select the best candidate experiment given current uncertainty."""
    current_uncertainty: float = Field(0.5, ge=0.0, le=1.0)


class ExperimentRecordRequest(BaseModel):
    """Record the outcome of an executed candidate experiment."""
    candidate_id: int = Field(..., ge=0, le=1000000)
    fe_before: float
    fe_after: float


# --------------------------------------------------------------------------- #
# Request tracing middleware (X-Request-ID, structured access log)
# --------------------------------------------------------------------------- #


def _request_id_from(request: Request) -> str:
    """Use the inbound X-Request-ID if present (and well-formed), else mint."""
    inbound = request.headers.get("x-request-id", "")
    if inbound and len(inbound) <= 64 and all(
        c.isalnum() or c in "-_" for c in inbound
    ):
        return inbound
    return uuid.uuid4().hex


async def request_tracing_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Attach an X-Request-ID to the response and log every request.

    Logs method, path, status, duration_ms and request_id for every
    request. Errors are re-raised so FastAPI's exception handlers still
    run; the access log line is emitted in a finally block.
    """
    request_id = _request_id_from(request)
    # Stash on state so endpoint code and the exception handler can read it.
    request.state.request_id = request_id

    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        log_fn = log.warning if duration_ms > 1000 else log.info
        log_fn(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=status_code,
            duration_ms=round(duration_ms, 3),
        )


# --------------------------------------------------------------------------- #
# Body size limit (S-HIGH-03 -- CWE-400 DoS via unbounded payload)
# --------------------------------------------------------------------------- #

MAX_BODY = 4 * 1024 * 1024  # 4 MiB


async def _limit_body_size(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Reject requests whose declared Content-Length exceeds MAX_BODY.

    Only the declared length is checked (the body is not buffered here) so
    huge uploads are rejected before they hit the application.

    Round-8 audit API8-3: ``int(cl)`` raises ``ValueError`` on a malformed
    Content-Length header (e.g. ``"12abc"``, ``"1, 2"``). Starlette does
    NOT validate this header before handing it to the app, so the previous
    code surfaced as a bare 500 with no diagnostic. Now we return 400 with
    an actionable message.

    Round-8 audit API8-4: the 413 / 400 response bodies now include
    ``request_id`` (read from ``request.state`` -- set by the OUTER
    ``request_tracing_middleware``) and the matching ``X-Request-ID``
    header, so clients can correlate the rejection with their access log
    line. Previously the body was ``{"detail": "payload too large"}`` with
    no request_id, so a client reporting a 413 could not be matched to a
    server-side log entry.

    Round-9 audit R9-006: the previous implementation only enforced the
    4 MiB cap when a ``Content-Length`` header was present. A client using
    ``Transfer-Encoding: chunked`` (no Content-Length) bypassed the cap
    entirely, allowing an unbounded body to be buffered in memory by the
    ASGI server until the process was OOM-killed. We reject chunked
    requests explicitly: the API contract requires ``Content-Length`` so
    the cap is enforceable pre-body, and no current client of the API
    uses streaming uploads.
    """
    te = request.headers.get("transfer-encoding", "").lower()
    if "chunked" in te:
        request_id = getattr(request.state, "request_id", "-")
        return JSONResponse(
            status_code=411,
            content={
                "detail": "chunked transfer-encoding not supported; use Content-Length",
                "request_id": request_id,
            },
            headers={
                "X-Request-ID": request_id,
                "X-Content-Type-Options": "nosniff",
            },
        )
    cl = request.headers.get("content-length")
    if cl:
        try:
            cl_int = int(cl)
        except (TypeError, ValueError):
            request_id = getattr(request.state, "request_id", "-")
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "invalid Content-Length header",
                    "request_id": request_id,
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Content-Type-Options": "nosniff",
                },
            )
        if cl_int > MAX_BODY:
            request_id = getattr(request.state, "request_id", "-")
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "payload too large",
                    "request_id": request_id,
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Content-Type-Options": "nosniff",
                },
            )
    return await call_next(request)


# --------------------------------------------------------------------------- #
# CORS / TrustedHost configuration (S-MED-13 -- CWE-942 overly-permissive
# cross-origin / host policy). Both are env-var configurable.
# --------------------------------------------------------------------------- #

_allowed_origins = [
    o.strip()
    for o in os.environ.get(
        "ZDM_CORS_ORIGINS", "http://localhost:8000,http://localhost:3000"
    ).split(",")
    if o.strip()
]
_allowed_hosts = [
    h.strip()
    for h in os.environ.get("ZDM_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if h.strip()
]


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #


def create_app() -> FastAPI:
    """Build and return a configured FastAPI application."""
    # Round-8 audit SIDE-2: fail CLOSED for docs. Previously the default was
    # the empty string, so ``is_production`` was ``False`` unless the operator
    # explicitly set ``ZDM_ENV=production`` -- leaking the full OpenAPI spec
    # (every endpoint, every Pydantic schema) to unauthenticated callers when
    # an operator forgot the env var. Defaulting to ``production`` keeps the
    # spec hidden unless ``ZDM_ENV=development`` (or ``=test``) is set.
    is_production = os.environ.get("ZDM_ENV", "production").lower() != "development"

    app = FastAPI(
        title="ZeroDataModel API",
        version=__version__,
        docs_url=None if is_production else "/docs",
        redoc_url=None if is_production else "/redoc",
        openapi_url=None if is_production else "/openapi.json",
        openapi_tags=[
            {"name": "core", "description": "Core cognitive operations"},
            {"name": "phase7", "description": "Phase 7 cognitive upgrade modules"},
            {"name": "versioning", "description": "API version information"},
        ],
    )

    # Register the rate-limit middleware + handler when slowapi is present.
    if _HAS_SLOWAPI and limiter is not None:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        # SlowAPIMiddleware is added lazily to avoid importing it when the
        # dep is missing.
        from slowapi.middleware import SlowAPIMiddleware  # noqa: PLC0415

        app.add_middleware(SlowAPIMiddleware)

    # Structured access log + X-Request-ID + body-size limit (S-HIGH-03).
    # Round-8 audit API8-4: middleware registration order was BACKWARDS.
    # Starlette's ``add_middleware`` / ``app.middleware("http")`` are LIFO:
    # the LAST registered middleware is the OUTERMOST (runs first on
    # inbound). The previous code registered ``request_tracing_middleware``
    # FIRST and ``_limit_body_size`` SECOND, so the actual execution order
    # was: ``_limit_body_size`` -> ``request_tracing`` -> endpoint. That
    # meant (a) 413 rejections from ``_limit_body_size`` were never captured
    # in the access log (they never reached ``request_tracing``), and (b)
    # ``request.state.request_id`` was unset when ``_limit_body_size`` ran,
    # so the 413 body had no request_id for client-side correlation.
    #
    # The fix: register ``_limit_body_size`` FIRST (so it is the INNER
    # middleware) and ``request_tracing_middleware`` SECOND (so it is the
    # OUTER one). Execution order is now: ``request_tracing`` ->
    # ``_limit_body_size`` -> endpoint. 413s are logged and carry the
    # request_id set by the tracing middleware.
    app.middleware("http")(_limit_body_size)
    app.middleware("http")(request_tracing_middleware)

    # CORS (S-MED-13): only the configured origins may issue credentialed
    # cross-origin requests. ``allow_credentials=False`` keeps the policy
    # strict; methods/headers are pinned to what the API actually uses.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )
    # TrustedHost (S-MED-13): reject requests whose Host header does not
    # match the configured allow-list (host-header injection / cache
    # poisoning).
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts)

    # ---------------------------------------------------------------- #
    # Security middleware layer (token-bucket rate limit, request-size
    # guard, browser-facing security response headers).
    #
    # Starlette ``add_middleware`` / ``app.middleware("http")`` are LIFO:
    # the LAST registered middleware is the OUTERMOST (runs first on
    # inbound, last on the outbound response). We register in this order
    # so that the OUTERMOST middleware is ``SecurityHeadersMiddleware`` --
    # that way *every* response (including 429 from the rate limiter and
    # 413 from the size guard, which short-circuit by returning a
    # ``JSONResponse`` directly) still carries the security headers on the
    # way out. Execution order (outermost first):
    #   SecurityHeaders -> RequestSizeLimit -> rate_limit
    #   -> TrustedHost -> CORS -> request_tracing -> _limit_body_size
    #   -> slowapi -> endpoint.
    # ---------------------------------------------------------------- #
    # Rate-limit middleware (token-bucket, per-client). Health/readiness
    # and metrics/docs paths are exempt so probes and Prometheus scrapes
    # (which do not carry credentials) are never throttled. Configurable
    # via ZDM_RATE_LIMIT_RPM (default 60) and ZDM_RATE_LIMIT_BURST
    # (default 10). WebSocket endpoints bypass the HTTP middleware chain
    # entirely; they must call ``RateLimiter.check`` in their own handler.
    _rate_limiter = RateLimiter(
        requests_per_minute=int(os.environ.get("ZDM_RATE_LIMIT_RPM", "60")),
        burst=int(os.environ.get("ZDM_RATE_LIMIT_BURST", "10")),
    )
    _rate_limit_exempt_paths = frozenset(
        {
            "/",
            "/health",
            "/ready",
            "/version",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            # v1 mirrors of health/readiness/version are also probe-class
            # paths and must never be throttled.
            "/v1/health",
            "/v1/ready",
            "/v1/version",
        }
    )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path in _rate_limit_exempt_paths:
            return await call_next(request)
        if not _rate_limiter.check(request):
            request_id = getattr(request.state, "request_id", "-")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Try again later.",
                    "request_id": request_id,
                },
                headers={"Retry-After": "60"},
            )
        return await call_next(request)

    # Coarse outer DoS guard on the declared Content-Length. The stricter
    # per-route cap (MAX_BODY = 4 MiB inside ``_limit_body_size``) remains
    # the authoritative application limit; this 10 MiB guard rejects
    # pathologically large uploads before they reach the inner stack.
    app.add_middleware(
        RequestSizeLimitMiddleware, max_size=10 * 1024 * 1024
    )
    # Security response headers on every response (outermost). Registered
    # last so it is the OUTERMOST middleware and wraps all short-circuit
    # responses (429 / 413 / 401 / 500) too.
    app.add_middleware(SecurityHeadersMiddleware)

    # Prometheus instrumentator (optional). The instrumentator still
    # collects the default HTTP metrics, but the /metrics endpoint is
    # registered separately below with API-key auth (S-HIGH-02).
    if _HAS_PROMETHEUS and Instrumentator is not None:
        Instrumentator(
            should_group_status_codes=False,
            should_ignore_untemplated=True,
            excluded_handlers=["/health", "/ready", "/metrics"],
        ).instrument(app)

    # Graceful-shutdown flag so /health can return 503 while draining
    # in-flight requests (O-HIGH-21).
    _shutting_down: bool = False

    # Startup banner: warn if auth is disabled (local dev mode).
    @app.on_event("startup")
    async def _on_startup() -> None:
        # Validate ZDM_ENV (O-MED-26): fail fast on a typo'd environment
        # name rather than silently running with docs/observability in an
        # unintended mode.
        env = os.environ.get("ZDM_ENV", "").lower()
        if env not in ("", "development", "dev", "production", "prod"):
            raise RuntimeError(
                f"Invalid ZDM_ENV={env!r}; expected development/production"
            )
        if not _configured_api_key():
            log.warning(
                "ZDM_API_KEY is not set -- authentication disabled (dev mode)"
            )
        else:
            log.info("API key authentication enabled")
        log.info(
            "startup_complete",
            env=env or "development",
            allowed_hosts=_allowed_hosts,
        )

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        nonlocal _shutting_down
        _shutting_down = True

    # ---------------------------------------------------------------- #
    # Centralized exception handlers (CWE-209 -- no internal leakage)
    # ---------------------------------------------------------------- #
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(  # type: ignore[no-untyped-def]
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        # Surface the request_id on every HTTPException response (Q-MED-12)
        # so error bodies share the same shape as the unhandled-exception
        # handler below. Headers (e.g. WWW-Authenticate) are preserved.
        request_id = getattr(request.state, "request_id", "-")
        # Round-8 audit API8-6c: merge ``X-Content-Type-Options: nosniff``
        # into whatever headers the HTTPException already carries (e.g.
        # WWW-Authenticate on 401). Without nosniff, a browser receiving
        # an error body that reflects user input could content-sniff the
        # body as HTML and execute it (XSS sink).
        exc_headers = dict(getattr(exc, "headers", None) or {})
        exc_headers.setdefault("X-Request-ID", request_id)
        exc_headers.setdefault("X-Content-Type-Options", "nosniff")
        return JSONResponse(
            status_code=exc.status_code,
            headers=exc_headers,
            content={"detail": exc.detail, "request_id": request_id},
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(  # type: ignore[no-untyped-def]
        request: Request, exc: Exception
    ) -> JSONResponse:
        # HTTPException is handled by FastAPI's default handler -- but if
        # one slips through here we still treat it generically. We log the
        # full detail server-side and return a generic 500 body.
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        # Round-8 audit SIDE-3: log only the exception TYPE, not its
        # message. ``str(exc)`` for many exception types carries file
        # paths, request payloads, or internal variable names that the
        # response body intentionally withheld. The full traceback is
        # retained at DEBUG level for ops triage.
        log.error(
            "unhandled_exception",
            request_id=request_id,
            error_type=type(exc).__name__,
        )
        log.debug("unhandled_exception_traceback", request_id=request_id, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal error", "request_id": request_id},
            headers={
                "X-Request-ID": request_id,
                "X-Content-Type-Options": "nosniff",
            },
        )

    # ---------------------------------------------------------------- #
    # API versioning (Phase: /v1 prefix + backward-compat root paths)
    # ---------------------------------------------------------------- #
    # All business endpoints are registered on ``v1_router`` and mounted
    # twice: once under ``/v1`` (canonical) and once at the root (legacy,
    # hidden from OpenAPI, marked deprecated by the middleware below).
    # ``/metrics``, ``/`` stay on ``app`` directly (root only, never
    # deprecated). ``/health`` and ``/ready`` live on ``v1_router`` so they
    # are reachable at both ``/health`` and ``/v1/health`` (health probes must
    # not be version-gated); they are exempt from the deprecation header.
    v1_router = APIRouter()

    # Root paths that must NOT carry the ``X-Deprecated`` header. Everything
    # else served at the root (i.e. legacy mirrors of /v1 endpoints) is
    # flagged deprecated with a ``Link`` pointing at the successor version.
    _non_deprecated_root_paths = frozenset(
        {
            "/",
            "/health",
            "/ready",
            "/version",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
    )

    # Version negotiation via Accept header. A caller requesting a future
    # version (e.g. ``application/vnd.zdm.v2+json``) gets a 501 pointing them
    # at v1. Registered outermost so it can short-circuit before auth/CORS.
    @app.middleware("http")
    async def version_negotiation(request: Request, call_next):  # type: ignore[no-untyped-def]
        accept = request.headers.get("accept", "")
        if "application/vnd.zdm.v2+json" in accept:
            return JSONResponse(
                status_code=501,
                content={"detail": "v2 not yet implemented. Use v1."},
                headers={"X-Content-Type-Options": "nosniff"},
            )
        return await call_next(request)

    # Backward-compat deprecation marker. Any non-/v1, non-exempt path served
    # at the root is a legacy mirror of the /v1 endpoint and is flagged with
    # ``X-Deprecated: true`` plus a ``Link`` to the successor version.
    @app.middleware("http")
    async def deprecate_root_paths(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        path = request.url.path
        if (
            not path.startswith("/v1")
            and path not in _non_deprecated_root_paths
        ):
            response.headers["X-Deprecated"] = "true"
            response.headers["Link"] = f'</v1{path}>; rel="successor-version"'
        return response

    # ---------------------------------------------------------------- #
    # Health / readiness (cheap liveness; readiness probes the model)
    # ---------------------------------------------------------------- #
    @app.get(
        "/metrics",
        include_in_schema=False,
        dependencies=[Depends(verify_api_key)],
    )
    def _metrics() -> Response:
        """Prometheus metrics endpoint (auth-gated, S-HIGH-02).

        Round-8 audit API8-1: when ``prometheus_client`` /
        ``prometheus_fastapi_instrumentator`` are not installed, the module
        sets ``generate_latest = None`` and ``CONTENT_TYPE_LATEST = ""``.
        Calling ``Response(None, media_type="")`` then raises inside
        Starlette (None body cannot be encoded) and surfaces as a bare 500
        with no diagnostic. Return an explicit 503 so the operator sees
        *why* metrics are unavailable.
        """
        if not _HAS_PROMETHEUS or generate_latest is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "prometheus metrics unavailable: install "
                    "'prometheus_client' and 'prometheus_fastapi_instrumentator'"
                ),
            )
        # Round-8 audit API8-6b: prometheus exposition format is text/plain.
        # ``Cache-Control: no-store`` prevents intermediate proxies (or the
        # scraper's own HTTP cache) from serving stale metric values during
        # an incident. ``X-Content-Type-Options: nosniff`` prevents browsers
        # that somehow hit /metrics (misconfigured ingress, dev port-forward)
        # from sniffing the body as HTML and executing any ``<script>`` that
        # might appear in a metric label value (stored-XSS via labels).
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @v1_router.get(
        "/health",
        response_model=HealthResponse,
        responses={
            503: {
                "description": "Service shutting down",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/HealthResponse"}
                    }
                },
            }
        },
        tags=["health"],
    )
    def health() -> Any:
        """Liveness probe. Cheap: no model construction."""
        if _shutting_down:
            return JSONResponse(
                status_code=503,
                content={"status": "shutting_down"},
                headers={"X-Content-Type-Options": "nosniff"},
            )
        return HealthResponse(status="ok")

    @v1_router.get(
        "/ready",
        response_model=ReadyResponse,
        responses={
            503: {
                "description": "Service degraded (model not ready)",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ReadyResponse"}
                    }
                },
            }
        },
        tags=["health"],
    )
    def ready() -> Any:
        """Readiness probe. Constructs the model if needed and reports
        whether it is ready to serve requests."""
        try:
            get_model()
            return ReadyResponse(status="ok", model_ready=True)
        except Exception as exc:  # pragma: no cover - defensive
            # SIDE-3: log only the type, not the message (which may carry
            # filesystem paths or internal variable names).
            log.warning("ready_probe_failed", error_type=type(exc).__name__)
            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "model_ready": False},
                headers={"X-Content-Type-Options": "nosniff"},
            )

    @app.get(
        "/",
        response_model=RootResponse,
        tags=["health"],
        dependencies=[Depends(verify_api_key)],
    )
    def root() -> RootResponse:
        """Root: status + active hardware backends (auth-gated)."""
        model = get_model()
        return RootResponse(
            status="ok",
            version=__version__,
            hardware_info=model.hardware_info,
        )

    # ---------------------------------------------------------------- #
    # Cognitive endpoints
    # ---------------------------------------------------------------- #
    @v1_router.post(
        "/think",
        response_model=ThinkResponse,
        tags=["cognitive"],
        dependencies=[Depends(verify_api_key)],
    )
    @_limit("10/minute")
    async def think(
        request: Request,
        req: ThinkRequest | None = Body(default=None),  # noqa: B008
    ) -> ThinkResponse:
        """Run one thought cycle. With no ``input`` the model self-generates.

        P2.16 异步改造: ``think()`` 是 CPU 密集的同步调用（毫秒级到
        秒级），原同步 handler 会阻塞 FastAPI 事件循环的工作线程，
        导致其他异步端点（``/perceive-topology``、``/discover-causal-dynamics``
        等）排队等待。改为 ``async def`` 并用 ``asyncio.to_thread``
        把 ``model.think`` 调度到默认线程池，释放事件循环处理其他
        请求。Prometheus 指标与 cycle_count 读取逻辑保持不变。
        """
        model = get_model()
        body = req or ThinkRequest()
        input_data = (
            np.asarray(body.input, dtype=float) if body.input else None
        )
        if input_data is not None:
            _ensure_finite(input_data, "input")
        start = time.perf_counter()
        # P2.16: 把同步 think() 卸载到线程池，避免阻塞事件循环。
        signal = await asyncio.to_thread(model.think, input_data)
        if _think_duration is not None:
            _think_duration.observe(time.perf_counter() - start)
        # Round-8 audit CONCUR8-8: read cycle_count and the last free energy
        # under ``model._lock`` so the two prometheus gauges are consistent
        # with each other AND with this think() call. Without the lock, a
        # concurrent think() could land BETWEEN our think() return and these
        # reads -- incrementing cycle_count and appending a new free energy
        # value -- so ``zdm_cycle_count`` would point at cycle N+1 while
        # ``zdm_free_energy_last`` would point at the next cycle's energy,
        # and neither would match the ``cycle`` field returned in the HTTP
        # response body. ``model._lock`` is an ``RLock`` so re-entering it
        # from the same thread (think() already released it) is safe.
        with model._lock:
            cycle_count_snapshot = model.cycle_count
            fe_history = model.active_inference.free_energy_history
            last_fe = fe_history[-1] if fe_history else None
        if _cycle_count is not None:
            _cycle_count.set(cycle_count_snapshot)
        if _free_energy is not None and last_fe is not None:
            _free_energy.set(float(last_fe))
        return ThinkResponse(
            cycle=int(signal.metadata.get("cycle", cycle_count_snapshot)),
            output=[float(x) for x in np.asarray(signal.data).flatten().tolist()],
            confidence=float(signal.confidence),
        )

    @v1_router.post(
        "/classify",
        response_model=ClassifyResponse,
        tags=["nlp"],
        dependencies=[Depends(verify_api_key)],
    )
    @_limit("30/minute")
    def classify(request: Request, req: ClassifyRequest) -> ClassifyResponse:
        """Zero-shot classify ``text`` into a topic."""
        model = get_model()
        # Round-8 audit CONCUR8-6: read endpoints must take ``model._lock``
        # so a concurrent ``/think`` cannot mutate the module arrays in
        # place (``+=``) while this handler reads them. NumPy ``+=`` is NOT
        # atomic, so an unsynchronised read can observe a half-mutated
        # ``topos.classifier`` / ``emission`` / ``classical_weights`` and
        # return garbled topic scores. ``_lock`` is an ``RLock`` so the
        # same thread can re-enter it (e.g. via ``model._lock`` inside
        # ``classify_text`` if it ever needs to).
        with model._lock:
            topic, conf = model.classify_text(req.text)
        return ClassifyResponse(topic=str(topic), confidence=float(conf))

    @v1_router.post(
        "/similarity",
        response_model=SimilarityResponse,
        tags=["nlp"],
        dependencies=[Depends(verify_api_key)],
    )
    @_limit("30/minute")
    def similarity(request: Request, req: SimilarityRequest) -> SimilarityResponse:
        """Semantic similarity in [0, 1] between two texts."""
        model = get_model()
        # CONCUR8-6: see /classify — read under model._lock.
        with model._lock:
            score = model.text_similarity(req.a, req.b)
        return SimilarityResponse(similarity=float(score))

    @v1_router.post(
        "/generate",
        response_model=GenerateResponse,
        tags=["nlp"],
        dependencies=[Depends(verify_api_key)],
    )
    @_limit("20/minute")
    def generate(request: Request, req: GenerateRequest) -> GenerateResponse:
        """Generate ``length`` printable characters from a seed."""
        model = get_model()
        # CONCUR8-6: see /classify — read under model._lock.
        with model._lock:
            text = model.generate_text(req.seed, length=req.length)
        return GenerateResponse(text=str(text))

    @v1_router.post(
        "/forecast",
        response_model=ForecastResponse,
        tags=["analytics"],
        dependencies=[Depends(verify_api_key)],
    )
    @_limit("30/minute")
    def forecast(request: Request, req: ForecastRequest) -> ForecastResponse:
        """Forecast ``horizon`` future values of the input series."""
        model = get_model()
        series = np.asarray(req.series, dtype=float).flatten()
        if series.size == 0:
            raise HTTPException(status_code=400, detail="series must be non-empty")
        _ensure_finite(series, "series")
        # CONCUR8-6: see /classify — read under model._lock.
        with model._lock:
            preds = model.forecast(series, horizon=req.horizon)
        return ForecastResponse(
            forecast=[float(x) for x in np.asarray(preds).flatten().tolist()]
        )

    @v1_router.post(
        "/anomalies",
        response_model=AnomaliesResponse,
        tags=["analytics"],
        dependencies=[Depends(verify_api_key)],
    )
    @_limit("30/minute")
    def anomalies(request: Request, req: AnomaliesRequest) -> AnomaliesResponse:
        """Flag anomalies in a 1D series (returns a boolean mask)."""
        model = get_model()
        series = np.asarray(req.series, dtype=float).flatten()
        if series.size == 0:
            raise HTTPException(status_code=400, detail="series must be non-empty")
        _ensure_finite(series, "series")
        # CONCUR8-6: see /classify — read under model._lock.
        with model._lock:
            mask = model.detect_anomalies(series)
        return AnomaliesResponse(anomalies=[bool(x) for x in np.asarray(mask).tolist()])

    @v1_router.post(
        "/trend",
        response_model=TrendResponse,
        tags=["analytics"],
        dependencies=[Depends(verify_api_key)],
    )
    @_limit("30/minute")
    def trend(request: Request, req: TrendRequest) -> TrendResponse:
        """Analyze trend / regime / curvature / geodesic / isomorphism."""
        model = get_model()
        series = np.asarray(req.series, dtype=float).flatten()
        if series.size == 0:
            raise HTTPException(status_code=400, detail="series must be non-empty")
        _ensure_finite(series, "series")
        # CONCUR8-6: see /classify — read under model._lock.
        with model._lock:
            out = model.analyze_trend(series)
        return TrendResponse(
            trend_slope=float(out["trend_slope"]),
            regime=str(out["regime"]),
            curvature=float(out["curvature"]),
            geodesic_deviation=float(out["geodesic_deviation"]),
            isomorphism_score=float(out["isomorphism_score"]),
        )

    @v1_router.post(
        "/recognize",
        response_model=RecognizeResponse,
        tags=["vision"],
        dependencies=[Depends(verify_api_key)],
    )
    @_limit("30/minute")
    def recognize(request: Request, req: RecognizeRequest) -> RecognizeResponse:
        """Recognize a shape/pattern in a 2D grayscale image (list of rows).

        Round-8 audit API8-2: the schema validator only checks the outer list
        length + per-row length cap. The STRUCTURAL checks (each row
        non-empty, rectangular shape) live here so that ``[[]]`` returns
        400 (endpoint business check) rather than 422 (schema), preserving
        the contract exercised by ``test_post_recognize_empty_image_returns_4xx``.
        A JAGGED image (``[[1,2],[3]]``) previously produced an ``object``
        dtype array that crashed inside ``model.recognize_pattern`` -- now
        it is rejected at the endpoint boundary with an actionable 400.
        """
        model = get_model()
        if not req.image or not req.image[0]:
            raise HTTPException(status_code=400, detail="image must be non-empty")
        # API8-2: every row must be non-empty (``[[]]`` is caught here).
        for i, row in enumerate(req.image):
            if not row:
                raise HTTPException(
                    status_code=400,
                    detail=f"image row {i} must be non-empty",
                )
        # API8-2: rectangular check (reject jagged arrays before np.asarray
        # silently produces an ``object`` dtype 1D array).
        expected_len = len(req.image[0])
        for i, row in enumerate(req.image):
            if len(row) != expected_len:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"image must be rectangular: row {i} has "
                        f"{len(row)} values, expected {expected_len}"
                    ),
                )
        image = np.asarray(req.image, dtype=float)
        if image.ndim != 2:
            raise HTTPException(status_code=400, detail="image must be 2D")
        _ensure_finite(image, "image")
        # CONCUR8-6: see /classify — read under model._lock.
        with model._lock:
            shape, conf = model.recognize_pattern(image)
        return RecognizeResponse(shape=str(shape), confidence=float(conf))

    @v1_router.post(
        "/save",
        response_model=SaveResponse,
        status_code=201,
        tags=["persistence"],
        dependencies=[Depends(verify_api_key)],
    )
    @_limit("5/minute")
    def save(request: Request, req: PathRequest) -> JSONResponse:
        """Persist the current model state to ``name`` (relative) on disk.

        Returns 201 Created with a ``Location`` header pointing at the
        canonical resource URI for the snapshot (Q-LOW-13).

        Round-7 audit API7-1-1: the OpenAPI schema now declares
        ``status_code=201`` and ``SaveResponse{saved, name}`` so the contract
        matches the actual ``JSONResponse(201, {"saved": True, "name": ...})``
        returned below. The ``JSONResponse`` is kept (rather than returning a
        ``SaveResponse`` instance) so the ``Location`` header survives —
        FastAPI only emits headers when the handler returns an explicit
        ``Response``.

        Round-3 audit: acquire ``model._lock`` while serialising so a
        concurrent ``think()`` cannot mutate arrays in-place (``+=``) while
        ``np.savez`` reads them — that race produced silently torn snapshots.
        """
        model = get_model()
        try:
            with model._lock:
                ModelSerializer.save(model, req.name)
        except ValueError as exc:
            # Sandbox violation -> 400, no path leakage in detail.
            raise HTTPException(status_code=400, detail="invalid snapshot name") from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail="save failed") from exc
        return JSONResponse(
            status_code=201,
            content={"saved": True, "name": req.name},
            headers={"Location": f"/v1/load/{req.name}"},
        )

    @v1_router.post(
        "/load",
        response_model=LoadResponse,
        tags=["persistence"],
        dependencies=[Depends(verify_api_key)],
    )
    @_limit("5/minute")
    def load(request: Request, req: PathRequest) -> LoadResponse:
        """Replace the live model with one loaded from ``name`` (relative).

        Round-8 audit API8-5: previously only ``ValueError`` (sandbox),
        ``FileNotFoundError`` and ``KeyError`` (pickle schema mismatch)
        were caught. Any OTHER ``OSError`` -- permission denied on the
        snapshot file, disk read error, ``EOFError`` during unpickling when
        the file is truncated, ``IsADirectoryError`` -- escaped as a bare
        500 with no ``request_id`` body and no server-side context (the
        centralized ``Exception`` handler would log it generically as
        ``unhandled_exception``). Catch ``OSError`` explicitly and return
        a clean 500 with the standard error body shape. ``KeyError`` is
        kept because it is not a subclass of ``OSError``.
        """
        try:
            new_model = ModelSerializer.load(req.name)
        except ValueError as exc:
            # Sandbox violation (ValueError) -> 400, no path leakage.
            raise HTTPException(status_code=400, detail="invalid snapshot name") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="snapshot not found") from exc
        except KeyError as exc:
            raise HTTPException(status_code=400, detail="load failed") from exc
        except OSError as exc:
            # API8-5: disk I/O errors (permission denied, disk full, IO
            # error during unpickling, ``IsADirectoryError``). Previously
            # escaped as a bare 500 via the unhandled-exception handler.
            # Round-8 audit SIDE-4: ``str(exc)`` for OSError subclasses
            # almost always includes the resolved filesystem path
            # (e.g. ``PermissionError: [Errno 13] Permission denied:
            # '/var/lib/zdm/snapshots/prod.pkl'``). The HTTP response body
            # returns a generic ``{"detail": "load failed"}`` with no path,
            # but the log line would leak the absolute path -- drop it and
            # log only the type + the user-supplied relative name (which
            # was already validated by the persistence sandbox).
            log.warning(
                "load_failed_oserror",
                snapshot_name=req.name,
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=500, detail="load failed") from exc
        set_model(new_model)
        return LoadResponse(loaded=True)

    # ---------------------------------------------------------------- #
    # Causal emergence endpoints (spec §2.3)
    # ---------------------------------------------------------------- #
    @v1_router.post(
        "/emergence/perceive",
        response_model=EmergencePerceiveResponse,
        tags=["emergence"],
    )
    @_limit("30/minute")
    async def perceive_topology(  # noqa: ANN202
        request: Request,
        req: EmergencePerceiveRequest,
        _api_key: str = Depends(verify_api_key),
        full_diagram: bool = False,
    ) -> EmergencePerceiveResponse:
        """Perceive topological invariants (Betti numbers, persistence entropy)."""
        arr = np.asarray(req.data, dtype=float)
        _ensure_finite(arr, "data")
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            raise HTTPException(
                status_code=400,
                detail="data must be 2D with shape (n>=2, d>=2)",
            )
        model = get_model()
        # CONCUR8-6: read under model._lock (same pattern as /classify etc.).
        with model._lock:
            result = model.perceive_topology(arr, max_dim=req.max_dim)
        diagram: list[list[float | None]] | None = None
        if full_diagram:
            diagram = _to_jsonable(result.get("persistence_diagram", []))
        return EmergencePerceiveResponse(
            betti_numbers=[int(x) for x in result["betti_numbers"]],
            persistence_entropy=float(result["persistence_entropy"]),
            euler_characteristic=int(result["euler_characteristic"]),
            n_points=int(result["n_points"]),
            max_eps=float(result["max_eps"]),
            persistence_diagram=diagram,
        )

    @v1_router.post(
        "/emergence/causal",
        response_model=EmergenceCausalResponse,
        tags=["emergence"],
    )
    @_limit("30/minute")
    async def discover_causal_dynamics(  # noqa: ANN202
        request: Request,
        req: EmergenceCausalRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> EmergenceCausalResponse:
        """Discover causal DAG from observational data (PC/LiNGAM/correlation)."""
        arr = np.asarray(req.data, dtype=float)
        _ensure_finite(arr, "data")
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            raise HTTPException(
                status_code=400,
                detail="data must be 2D with shape (n>=2, d>=2)",
            )
        model = get_model()
        with model._lock:
            result = model.discover_causal_dynamics(
                arr, var_names=req.var_names, method=req.method
            )
        adj = np.asarray(result["adjacency"])
        return EmergenceCausalResponse(
            adjacency=[[int(x) for x in row] for row in adj.tolist()],
            edges=[[int(x) for x in edge] for edge in result.get("edges", [])],
            n_edges=int(result["n_edges"]),
            is_acyclic=bool(result["is_acyclic"]),
            method=str(result["method"]),
            var_names=[str(x) for x in result["var_names"]],
        )

    @v1_router.post(
        "/emergence/trajectory",
        response_model=EmergenceTrajectoryResponse,
        tags=["emergence"],
    )
    @_limit("30/minute")
    async def generate_emergence_trajectory(  # noqa: ANN202
        request: Request,
        req: EmergenceTrajectoryRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> EmergenceTrajectoryResponse:
        """Generate a damped least-action trajectory between two states."""
        start = np.asarray(req.start_state, dtype=float)
        end = np.asarray(req.end_state, dtype=float)
        _ensure_finite(start, "start_state")
        _ensure_finite(end, "end_state")
        if start.shape != end.shape:
            raise HTTPException(
                status_code=400,
                detail="start_state and end_state must have the same shape",
            )
        constraints: dict | None = None
        if req.obstacles is not None or req.margin is not None:
            constraints = {}
            if req.obstacles is not None:
                obs_arr = np.asarray(req.obstacles, dtype=float)
                _ensure_finite(obs_arr, "obstacles")
                # V4-NEW-M001: validate obstacle dim matches start_state to
                # avoid engine ValueError surfacing as 500. Empty obstacles
                # (K=0) is allowed and skips projection (fix L2 back-compat).
                if obs_arr.ndim != 2 or obs_arr.shape[1] != start.shape[0]:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "obstacles must be 2D with shape (K, dim) where "
                            f"dim matches start_state; got shape "
                            f"{obs_arr.shape} for start_state dim {start.shape[0]}"
                        ),
                    )
                constraints["obstacles"] = obs_arr
            if req.margin is not None:
                constraints["margin"] = float(req.margin)
        model = get_model()
        with model._lock:
            result = model.generate_trajectory(
                start, end, n_steps=req.n_steps, constraints=constraints
            )
        traj = np.asarray(result["trajectory"])
        lagr = np.asarray(result["lagrangian"])
        return EmergenceTrajectoryResponse(
            trajectory=[[float(x) for x in row] for row in traj.tolist()],
            lagrangian=[float(x) for x in lagr.tolist()],
            action=float(result["action"]),
            converged=bool(result["converged"]),
            iterations=int(result["iterations"]),
            obstacle_violations=(
                int(result["obstacle_violations"])
                if "obstacle_violations" in result
                else None
            ),
        )

    @v1_router.post(
        "/emergence/sample",
        response_model=EmergenceSampleResponse,
        tags=["emergence"],
    )
    @_limit("30/minute")
    async def sample_posterior(  # noqa: ANN202
        request: Request,
        req: EmergenceSampleRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> EmergenceSampleResponse:
        """Sample a Gaussian posterior via HMC."""
        mean_arr = np.asarray(req.mean, dtype=float)
        _ensure_finite(mean_arr, "mean")
        std = float(req.std)

        # Gaussian log-prob: -0.5 * sum(((q - mean) / std) ** 2)
        def _gaussian_log_prob(q: np.ndarray) -> float:
            return -0.5 * float(np.sum(((q - mean_arr) / std) ** 2))

        model = get_model()
        with model._lock:
            result = model.sample_posterior(
                log_prob_fn=_gaussian_log_prob,
                initial_position=mean_arr,
                n_samples=req.n_samples,
            )
        return EmergenceSampleResponse(
            mean=[float(x) for x in np.asarray(result["mean"]).tolist()],
            std=[float(x) for x in np.asarray(result["std"]).tolist()],
            accept_rate=float(result["accept_rate"]),
            ess=float(result["ess"]),
            converged=bool(result["converged"]),
            samples=[
                [float(x) for x in row]
                for row in np.asarray(result["samples"]).tolist()
            ],  # V4-NEW-L004
        )

    @v1_router.post(
        "/emergence/recall",
        response_model=EmergenceRecallResponse,
        tags=["emergence"],
    )
    @_limit("30/minute")
    async def recall_memory(  # noqa: ANN202
        request: Request,
        req: EmergenceRecallRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> EmergenceRecallResponse:
        """Recall from chaotic associative memory."""
        query = np.asarray(req.query, dtype=float)
        _ensure_finite(query, "query")
        model = get_model()
        with model._lock:
            result = model.recall_memory(query, n_steps=req.n_steps)
        label: str | int | None = result.get("label")
        if isinstance(label, np.integer):
            label = int(label)
        traj = np.asarray(result["trajectory"])
        return EmergenceRecallResponse(
            label=label,
            similarity=float(result["similarity"]),
            emerged=bool(result["emerged"]),
            trajectory=[[float(x) for x in row] for row in traj.tolist()],
            converged=bool(result["converged"]),
            divergence=float(result["divergence"]),
            nearest_pattern=(
                [float(x) for x in np.asarray(result["nearest_pattern"]).tolist()]
                if result.get("nearest_pattern") is not None
                else None
            ),  # V4-NEW-L003
        )

    @v1_router.post(
        "/emergence/cycle",
        tags=["emergence"],
    )
    @_limit("10/minute")  # tighter limit — this is the expensive call
    async def emergence_cycle(  # noqa: ANN202
        request: Request,
        req: EmergenceCycleRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Run the full recursive emergence loop.

        Perception -> causal graph -> counterfactual trajectory ->
        posterior sampling -> memory recall -> emergence score.
        """
        arr = np.asarray(req.observation, dtype=float)
        _ensure_finite(arr, "observation")
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            raise HTTPException(
                status_code=400,
                detail="observation must be 2D with shape (n>=2, d>=2)",
            )
        model = get_model()
        with model._lock:
            result = model.emergence_cycle(arr)
        return _to_jsonable(result)

    # ------------------------------------------------------------------ #
    # Phase 6 — Memory / Planning / Multimodal / RL endpoints.
    # ------------------------------------------------------------------ #
    @v1_router.post("/memory/encode", tags=["memory"])
    @_limit("30/minute")
    async def memory_encode(  # noqa: ANN202
        request: Request,
        req: MemoryEncodeRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Encode an observation into the episodic memory store."""
        obs = np.asarray(req.observation, dtype=float)
        _ensure_finite(obs, "observation")
        model = get_model()
        with model._lock:
            result = model.encode_memory(obs, label=req.label)
        return _to_jsonable(result)

    @v1_router.post("/memory/retrieve", tags=["memory"])
    @_limit("30/minute")
    async def memory_retrieve(  # noqa: ANN202
        request: Request,
        req: MemoryRetrieveRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Retrieve top-k similar episodic memories."""
        query = np.asarray(req.query, dtype=float)
        _ensure_finite(query, "query")
        model = get_model()
        with model._lock:
            result = model.retrieve_memory(query, top_k=req.top_k)
        return _to_jsonable(result)

    @v1_router.post("/memory/consolidate", tags=["memory"])
    @_limit("10/minute")
    async def memory_consolidate(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Promote high-weight working-memory items into episodic memory."""
        model = get_model()
        with model._lock:
            result = model.consolidate_memory()
        return _to_jsonable(result)

    @v1_router.post("/planning/trajectory", tags=["planning"])
    @_limit("30/minute")
    async def planning_trajectory(  # noqa: ANN202
        request: Request,
        req: PlanningTrajectoryRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Plan a damped least-action trajectory with per-step actions."""
        start = np.asarray(req.start_state, dtype=float)
        goal = np.asarray(req.goal_state, dtype=float)
        _ensure_finite(start, "start_state")
        _ensure_finite(goal, "goal_state")
        if start.shape != goal.shape:
            raise HTTPException(
                status_code=400,
                detail="start_state and goal_state must have the same shape",
            )
        obstacles = None
        if req.obstacles is not None:
            obstacles = np.asarray(req.obstacles, dtype=float)
            _ensure_finite(obstacles, "obstacles")
            if obstacles.ndim != 2 or obstacles.shape[1] != start.shape[0]:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "obstacles must be 2D with shape (K, dim) where dim "
                        f"matches start_state; got {obstacles.shape}"
                    ),
                )
        model = get_model()
        with model._lock:
            result = model.plan_trajectory(
                start, goal, obstacles=obstacles, n_steps=req.n_steps,
                margin=req.margin,
            )
        return _to_jsonable(result)

    @v1_router.post("/planning/decompose", tags=["planning"])
    @_limit("30/minute")
    async def planning_decompose(  # noqa: ANN202
        request: Request,
        req: PlanningDecomposeRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Decompose a goal into an AND/OR tree of atomic actions."""
        model = get_model()
        with model._lock:
            result = model.decompose_goal(req.goal, max_depth=req.max_depth)
        return _to_jsonable(result)

    @v1_router.post("/planning/sequence", tags=["planning"])
    @_limit("30/minute")
    async def planning_sequence(  # noqa: ANN202
        request: Request,
        req: PlanningSequenceRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Topologically sort a DAG of actions."""
        adj = np.asarray(req.adjacency, dtype=int)
        _ensure_finite(adj.astype(float), "adjacency")
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise HTTPException(
                status_code=400,
                detail=f"adjacency must be square 2D, got {adj.shape}",
            )
        model = get_model()
        with model._lock:
            result = model.sequence_actions(adj)
        return _to_jsonable(result)

    @v1_router.post("/multimodal/align", tags=["multimodal"])
    @_limit("30/minute")
    async def multimodal_align(  # noqa: ANN202
        request: Request,
        req: MultimodalAlignRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Fit CCA + project the first sample of modality A into shared space."""
        a = np.asarray(req.observations_a, dtype=float)
        b = np.asarray(req.observations_b, dtype=float)
        _ensure_finite(a, "observations_a")
        _ensure_finite(b, "observations_b")
        if a.shape[0] != b.shape[0]:
            raise HTTPException(
                status_code=400,
                detail="observations_a and observations_b must have the same n_samples",
            )
        model = get_model()
        with model._lock:
            fit = model.fit_cross_modal(a, b)
            aligned = model.align_cross_modal(a[0], source="a")
        return _to_jsonable({"fit": fit, "aligned": aligned})

    @v1_router.post("/multimodal/fuse", tags=["multimodal"])
    @_limit("30/minute")
    async def multimodal_fuse(  # noqa: ANN202
        request: Request,
        req: MultimodalFuseRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Fuse multiple modality embeddings into a single vector."""
        embeddings = [np.asarray(e, dtype=float) for e in req.embeddings]
        for i, e in enumerate(embeddings):
            _ensure_finite(e, f"embeddings[{i}]")
        model = get_model()
        with model._lock:
            result = model.fuse_modalities(embeddings, strategy=req.strategy)
        return _to_jsonable(result)

    @v1_router.post("/multimodal/contrastive", tags=["multimodal"])
    @_limit("30/minute")
    async def multimodal_contrastive(  # noqa: ANN202
        request: Request,
        req: MultimodalContrastiveRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """InfoNCE contrastive alignment loss between paired batches."""
        a = np.asarray(req.batch_a, dtype=float)
        b = np.asarray(req.batch_b, dtype=float)
        _ensure_finite(a, "batch_a")
        _ensure_finite(b, "batch_b")
        if a.shape != b.shape:
            raise HTTPException(
                status_code=400,
                detail="batch_a and batch_b must have the same shape",
            )
        model = get_model()
        with model._lock:
            result = model.contrastive_loss(a, b)
        return _to_jsonable(result)

    @v1_router.post("/rl/step", tags=["rl"])
    @_limit("60/minute")
    async def rl_step(  # noqa: ANN202
        request: Request,
        req: RLStepRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Take one step in the synthetic MDP."""
        model = get_model()
        with model._lock:
            result = model.step_mdp(req.state, req.action)
        return _to_jsonable(result)

    @v1_router.post("/rl/train-q", tags=["rl"])
    @_limit("5/minute")  # expensive — training loop
    async def rl_train_q(  # noqa: ANN202
        request: Request,
        req: RLTrainQRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Full tabular Q-learning training loop."""
        model = get_model()
        with model._lock:
            result = model.train_q_learner(
                n_episodes=req.n_episodes,
                max_steps_per_episode=req.max_steps_per_episode,
            )
        return _to_jsonable(result)

    @v1_router.post("/rl/search-mcts", tags=["rl"])
    @_limit("10/minute")
    async def rl_search_mcts(  # noqa: ANN202
        request: Request,
        req: RLSearchMCTSRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """UCT MCTS over the synthetic MDP."""
        model = get_model()
        with model._lock:
            result = model.search_rl_mcts(
                req.root_state,
                n_simulations=req.n_simulations,
                max_depth=req.max_depth,
            )
        return _to_jsonable(result)

    # ------------------------------------------------------------------
    # Phase 7 — Audio endpoints.
    # ------------------------------------------------------------------
    @v1_router.post("/audio/encode", tags=["audio"])
    @_limit("30/minute")
    async def audio_encode(  # noqa: ANN202
        request: Request,
        req: AudioSignalRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Encode a 1D audio signal into a ``dim``-length L2-normalized vector."""
        signal = np.asarray(req.signal, dtype=float)
        _ensure_finite(signal, "signal")
        model = get_model()
        with model._lock:
            embedding = model.encode_audio(signal, sample_rate=req.sample_rate)
        return {"embedding": _to_jsonable(embedding)}

    @v1_router.post("/audio/onsets", tags=["audio"])
    @_limit("30/minute")
    async def audio_onsets(  # noqa: ANN202
        request: Request,
        req: AudioSignalRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Detect note/onset events via spectral flux."""
        signal = np.asarray(req.signal, dtype=float)
        _ensure_finite(signal, "signal")
        model = get_model()
        with model._lock:
            result = model.detect_onsets(signal, sample_rate=req.sample_rate)
        return _to_jsonable(result)

    @v1_router.post("/audio/pitch", tags=["audio"])
    @_limit("30/minute")
    async def audio_pitch(  # noqa: ANN202
        request: Request,
        req: AudioSignalRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Detect fundamental frequency via autocorrelation."""
        signal = np.asarray(req.signal, dtype=float)
        _ensure_finite(signal, "signal")
        model = get_model()
        with model._lock:
            result = model.detect_pitch(signal, sample_rate=req.sample_rate)
        return _to_jsonable(result)

    @v1_router.post("/audio/classify", tags=["audio"])
    @_limit("30/minute")
    async def audio_classify(  # noqa: ANN202
        request: Request,
        req: AudioSignalRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Classify audio texture (speech / music / noise / silence)."""
        signal = np.asarray(req.signal, dtype=float)
        _ensure_finite(signal, "signal")
        model = get_model()
        with model._lock:
            result = model.classify_audio(signal, sample_rate=req.sample_rate)
        return _to_jsonable(result)

    @v1_router.post("/audio/segment", tags=["audio"])
    @_limit("30/minute")
    async def audio_segment(  # noqa: ANN202
        request: Request,
        req: AudioSignalRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Segment audio into speech/silence regions via VAD."""
        signal = np.asarray(req.signal, dtype=float)
        _ensure_finite(signal, "signal")
        model = get_model()
        with model._lock:
            result = model.segment_speech(signal, sample_rate=req.sample_rate)
        return _to_jsonable(result)

    @v1_router.post("/audio/music", tags=["audio"])
    @_limit("30/minute")
    async def audio_music(  # noqa: ANN202
        request: Request,
        req: AudioSignalRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Analyze music for tempo and beats."""
        signal = np.asarray(req.signal, dtype=float)
        _ensure_finite(signal, "signal")
        model = get_model()
        with model._lock:
            result = model.analyze_music(signal, sample_rate=req.sample_rate)
        return _to_jsonable(result)

    # ------------------------------------------------------------------
    # Phase 7 — Graph endpoints.
    # ------------------------------------------------------------------
    @v1_router.post("/graph/encode", tags=["graph"])
    @_limit("30/minute")
    async def graph_encode(  # noqa: ANN202
        request: Request,
        req: GraphEncodeRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Encode a graph into a ``dim``-length L2-normalized vector."""
        adjacency = np.asarray(req.adjacency, dtype=float)
        _ensure_finite(adjacency, "adjacency")
        node_features = None
        if req.node_features is not None:
            node_features = np.asarray(req.node_features, dtype=float)
            _ensure_finite(node_features, "node_features")
        model = get_model()
        with model._lock:
            embedding = model.encode_graph(adjacency, node_features=node_features)
        return {"embedding": _to_jsonable(embedding)}

    @v1_router.post("/graph/communities", tags=["graph"])
    @_limit("30/minute")
    async def graph_communities(  # noqa: ANN202
        request: Request,
        req: GraphAdjacencyRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Detect communities via modularity optimization."""
        adjacency = np.asarray(req.adjacency, dtype=float)
        _ensure_finite(adjacency, "adjacency")
        model = get_model()
        with model._lock:
            result = model.detect_communities(adjacency)
        return _to_jsonable(result)

    @v1_router.post("/graph/path", tags=["graph"])
    @_limit("30/minute")
    async def graph_path(  # noqa: ANN202
        request: Request,
        req: GraphPathRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Find shortest path via Dijkstra."""
        adjacency = np.asarray(req.adjacency, dtype=float)
        _ensure_finite(adjacency, "adjacency")
        model = get_model()
        with model._lock:
            result = model.find_path(adjacency, req.source, req.target)
        return _to_jsonable(result)

    @v1_router.post("/graph/centrality", tags=["graph"])
    @_limit("30/minute")
    async def graph_centrality(  # noqa: ANN202
        request: Request,
        req: GraphAdjacencyRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Analyze degree / betweenness / closeness centrality."""
        adjacency = np.asarray(req.adjacency, dtype=float)
        _ensure_finite(adjacency, "adjacency")
        model = get_model()
        with model._lock:
            result = model.analyze_centrality(adjacency)
        return _to_jsonable(result)

    @v1_router.post("/graph/isomorphism", tags=["graph"])
    @_limit("30/minute")
    async def graph_isomorphism(  # noqa: ANN202
        request: Request,
        req: GraphIsomorphismRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Check if two graphs are likely isomorphic (Weisfeiler-Lehman)."""
        adj_a = np.asarray(req.adjacency_a, dtype=float)
        adj_b = np.asarray(req.adjacency_b, dtype=float)
        _ensure_finite(adj_a, "adjacency_a")
        _ensure_finite(adj_b, "adjacency_b")
        model = get_model()
        with model._lock:
            result = model.check_isomorphism(adj_a, adj_b)
        return _to_jsonable(result)

    @v1_router.post("/graph/track", tags=["graph"])
    @_limit("20/minute")
    async def graph_track(  # noqa: ANN202
        request: Request,
        req: GraphTrackRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Track community drift across graph snapshots."""
        snapshots = [np.asarray(s, dtype=float) for s in req.snapshots]
        for i, s in enumerate(snapshots):
            _ensure_finite(s, f"snapshots[{i}]")
        model = get_model()
        with model._lock:
            result = model.track_dynamic_graph(snapshots)
        return _to_jsonable(result)

    @v1_router.post("/graph/spanning", tags=["graph"])
    @_limit("30/minute")
    async def graph_spanning(  # noqa: ANN202
        request: Request,
        req: GraphAdjacencyRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Extract the minimum spanning tree via Kruskal."""
        adjacency = np.asarray(req.adjacency, dtype=float)
        _ensure_finite(adjacency, "adjacency")
        model = get_model()
        with model._lock:
            result = model.extract_spanning_tree(adjacency)
        return _to_jsonable(result)

    # ------------------------------------------------------------------
    # Phase 7 — Robotics endpoints.
    # ------------------------------------------------------------------
    @v1_router.post("/robotics/motion", tags=["robotics"])
    @_limit("30/minute")
    async def robotics_motion(  # noqa: ANN202
        request: Request,
        req: RoboticsMotionRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Plan a smooth trajectory through waypoints (cubic spline)."""
        waypoints = np.asarray(req.waypoints, dtype=float)
        _ensure_finite(waypoints, "waypoints")
        model = get_model()
        with model._lock:
            result = model.plan_motion(waypoints, n_steps=req.n_steps)
        return _to_jsonable(result)

    @v1_router.post("/robotics/forward", tags=["robotics"])
    @_limit("30/minute")
    async def robotics_forward(  # noqa: ANN202
        request: Request,
        req: RoboticsJointAnglesRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Forward kinematics: joint angles -> end-effector position."""
        angles = np.asarray(req.joint_angles, dtype=float)
        _ensure_finite(angles, "joint_angles")
        model = get_model()
        with model._lock:
            position = model.forward_kinematics(angles)
        return {"position": _to_jsonable(position)}

    @v1_router.post("/robotics/inverse", tags=["robotics"])
    @_limit("30/minute")
    async def robotics_inverse(  # noqa: ANN202
        request: Request,
        req: RoboticsInverseKinematicsRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Inverse kinematics via damped least squares."""
        target = np.asarray(req.target, dtype=float)
        _ensure_finite(target, "target")
        seed = None
        if req.seed is not None:
            seed = np.asarray(req.seed, dtype=float)
            _ensure_finite(seed, "seed")
        model = get_model()
        with model._lock:
            result = model.inverse_kinematics(target, seed=seed)
        return _to_jsonable(result)

    @v1_router.post("/robotics/fuse", tags=["robotics"])
    @_limit("30/minute")
    async def robotics_fuse(  # noqa: ANN202
        request: Request,
        req: RoboticsFuseRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Inverse-variance weighted sensor fusion."""
        if len(req.measurements) != len(req.variances):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"measurements ({len(req.measurements)}) and variances "
                    f"({len(req.variances)}) must have equal length"
                ),
            )
        measurements = [np.asarray(m, dtype=float) for m in req.measurements]
        for i, m in enumerate(measurements):
            _ensure_finite(m, f"measurements[{i}]")
        for i, v in enumerate(req.variances):
            if not np.isfinite(v) or v <= 0.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"variances[{i}] must be finite and > 0, got {v}",
                )
        model = get_model()
        with model._lock:
            fused = model.fuse_sensors(measurements, list(req.variances))
        return {"fused": _to_jsonable(fused)}

    @v1_router.post("/robotics/kalman", tags=["robotics"])
    @_limit("30/minute")
    async def robotics_kalman(  # noqa: ANN202
        request: Request,
        req: RoboticsKalmanRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Sequential Kalman-style Bayesian update."""
        prior = np.asarray(req.prior, dtype=float)
        meas = np.asarray(req.measurement, dtype=float)
        _ensure_finite(prior, "prior")
        _ensure_finite(meas, "measurement")
        model = get_model()
        with model._lock:
            result = model.update_kalman(
                prior, req.prior_var, meas, req.meas_var
            )
        return _to_jsonable(result)

    @v1_router.post("/robotics/gait", tags=["robotics"])
    @_limit("30/minute")
    async def robotics_gait(  # noqa: ANN202
        request: Request,
        req: RoboticsGaitRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Generate a periodic gait pattern (walk / trot / bound)."""
        model = get_model()
        with model._lock:
            result = model.generate_gait(n_steps=req.n_steps, gait_type=req.gait_type)
        return _to_jsonable(result)

    @v1_router.post("/robotics/optimize", tags=["robotics"])
    @_limit("30/minute")
    async def robotics_optimize(  # noqa: ANN202
        request: Request,
        req: RoboticsOptimizeRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Smooth a trajectory by minimizing jerk (gradient descent)."""
        trajectory = np.asarray(req.trajectory, dtype=float)
        _ensure_finite(trajectory, "trajectory")
        model = get_model()
        with model._lock:
            result = model.optimize_trajectory(trajectory, n_iter=req.n_iter)
        return _to_jsonable(result)

    @v1_router.post("/robotics/collision", tags=["robotics"])
    @_limit("30/minute")
    async def robotics_collision(  # noqa: ANN202
        request: Request,
        req: RoboticsCollisionRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Check collision at a single position against obstacles."""
        obstacles = _parse_obstacles(req.obstacles)
        position = np.asarray(req.position, dtype=float)
        _ensure_finite(position, "position")
        model = get_model()
        with model._lock:
            result = model.check_collision(obstacles, position, radius=req.radius)
        return _to_jsonable(result)

    @v1_router.post("/robotics/path-collision", tags=["robotics"])
    @_limit("30/minute")
    async def robotics_path_collision(  # noqa: ANN202
        request: Request,
        req: RoboticsPathCollisionRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Check collision along a path."""
        obstacles = _parse_obstacles(req.obstacles)
        path = np.asarray(req.path, dtype=float)
        _ensure_finite(path, "path")
        model = get_model()
        with model._lock:
            result = model.check_path_collision(obstacles, path, radius=req.radius)
        return _to_jsonable(result)

    @v1_router.post("/robotics/mpc", tags=["robotics"])
    @_limit("30/minute")
    async def robotics_mpc(  # noqa: ANN202
        request: Request,
        req: RoboticsMPCRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Pick the next control action via model predictive control."""
        current = np.asarray(req.current_state, dtype=float)
        target = np.asarray(req.target_state, dtype=float)
        _ensure_finite(current, "current_state")
        _ensure_finite(target, "target_state")
        obstacles = None
        if req.obstacles is not None:
            obstacles = _parse_obstacles(req.obstacles)
        model = get_model()
        with model._lock:
            result = model.control_mpc(current, target, obstacles=obstacles)
        return _to_jsonable(result)

    # ------------------------------------------------------------------
    # Phase 7 — Time endpoints.
    # ------------------------------------------------------------------
    @v1_router.post("/time/encode", tags=["time"])
    @_limit("60/minute")
    async def time_encode(  # noqa: ANN202
        request: Request,
        req: TimeSeriesRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Encode a 1D time series into a dim-length L2-normalized vector."""
        series = np.asarray(req.series, dtype=float)
        _ensure_finite(series, "series")
        model = get_model()
        with model._lock:
            emb = model.encode_time_series(series)
        return {"embedding": _to_jsonable(emb)}

    @v1_router.post("/time/seasonality", tags=["time"])
    @_limit("60/minute")
    async def time_seasonality(  # noqa: ANN202
        request: Request,
        req: TimeSeriesRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Detect seasonal periods via autocorrelation peaks."""
        series = np.asarray(req.series, dtype=float)
        _ensure_finite(series, "series")
        model = get_model()
        with model._lock:
            result = model.detect_seasonality(series)
        return _to_jsonable(result)

    @v1_router.post("/time/frequency", tags=["time"])
    @_limit("60/minute")
    async def time_frequency(  # noqa: ANN202
        request: Request,
        req: TimeSeriesRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Analyze FFT frequency content + spectral entropy."""
        series = np.asarray(req.series, dtype=float)
        _ensure_finite(series, "series")
        model = get_model()
        with model._lock:
            result = model.analyze_frequency(series)
        return _to_jsonable(result)

    @v1_router.post("/time/events", tags=["time"])
    @_limit("60/minute")
    async def time_events(  # noqa: ANN202
        request: Request,
        req: TimeSeriesRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Analyze event timestamps: inter-arrival, rate, burstiness."""
        ts = np.asarray(req.series, dtype=float)
        _ensure_finite(ts, "series")
        model = get_model()
        with model._lock:
            result = model.analyze_event_timestamps(ts)
        return _to_jsonable(result)

    @v1_router.post("/time/anomaly", tags=["time"])
    @_limit("60/minute")
    async def time_anomaly(  # noqa: ANN202
        request: Request,
        req: TimeSeriesRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Detect anomalous inter-arrival gaps via z-score."""
        ts = np.asarray(req.series, dtype=float)
        _ensure_finite(ts, "series")
        model = get_model()
        with model._lock:
            result = model.detect_anomalous_timing(ts)
        return _to_jsonable(result)

    @v1_router.post("/time/cycle", tags=["time"])
    @_limit("60/minute")
    async def time_cycle(  # noqa: ANN202
        request: Request,
        req: TimeCyclePhaseRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Track phase within a periodic cycle."""
        series = np.asarray(req.series, dtype=float)
        _ensure_finite(series, "series")
        model = get_model()
        with model._lock:
            result = model.track_cycle_phase(series, period=req.period)
        return _to_jsonable(result)

    @v1_router.post("/time/forecast", tags=["time"])
    @_limit("60/minute")
    async def time_forecast(  # noqa: ANN202
        request: Request,
        req: TimeSeriesRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Score forecastability: entropy + stationarity + autocorr."""
        series = np.asarray(req.series, dtype=float)
        _ensure_finite(series, "series")
        model = get_model()
        with model._lock:
            result = model.score_forecastability(series)
        return _to_jsonable(result)

    # ------------------------------------------------------------------
    # Phase 7 — Code endpoints.
    # ------------------------------------------------------------------
    @v1_router.post("/code/encode", tags=["code"])
    @_limit("60/minute")
    async def code_encode(  # noqa: ANN202
        request: Request,
        req: CodeSourceRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Encode Python source into a dim-length L2-normalized vector."""
        model = get_model()
        with model._lock:
            emb = model.encode_code(req.source)
        return {"embedding": _to_jsonable(emb)}

    @v1_router.post("/code/ast", tags=["code"])
    @_limit("60/minute")
    async def code_ast(  # noqa: ANN202
        request: Request,
        req: CodeSourceRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Analyze AST: functions, classes, imports, complexity, depth."""
        model = get_model()
        with model._lock:
            result = model.analyze_ast(req.source)
        return _to_jsonable(result)

    @v1_router.post("/code/compare", tags=["code"])
    @_limit("60/minute")
    async def code_compare(  # noqa: ANN202
        request: Request,
        req: CodeCompareRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Compare two code snippets via token + structure similarity."""
        model = get_model()
        with model._lock:
            result = model.compare_code(req.source_a, req.source_b)
        return _to_jsonable(result)

    @v1_router.post("/code/defects", tags=["code"])
    @_limit("60/minute")
    async def code_defects(  # noqa: ANN202
        request: Request,
        req: CodeSourceRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Detect defect patterns (mutable_default / bare_except / equals_none)."""
        model = get_model()
        with model._lock:
            result = model.detect_code_defects(req.source)
        return _to_jsonable(result)

    @v1_router.post("/code/control-flow", tags=["code"])
    @_limit("60/minute")
    async def code_control_flow(  # noqa: ANN202
        request: Request,
        req: CodeSourceRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Analyze per-function cyclomatic complexity and basic blocks."""
        model = get_model()
        with model._lock:
            result = model.analyze_control_flow(req.source)
        return _to_jsonable(result)

    @v1_router.post("/code/style", tags=["code"])
    @_limit("60/minute")
    async def code_style(  # noqa: ANN202
        request: Request,
        req: CodeSourceRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Run rule-based style checks (line length / naming / whitespace)."""
        model = get_model()
        with model._lock:
            result = model.analyze_code_style(req.source)
        return _to_jsonable(result)

    @v1_router.post("/code/dependencies", tags=["code"])
    @_limit("60/minute")
    async def code_dependencies(  # noqa: ANN202
        request: Request,
        req: CodeSourceRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Build a module-level import dependency graph."""
        model = get_model()
        with model._lock:
            result = model.build_dependency_graph(req.source)
        return _to_jsonable(result)

    # ------------------------------------------------------------------
    # Phase 7 — Reasoning endpoints.
    # ------------------------------------------------------------------
    @v1_router.post("/reasoning/prop-infer", tags=["reasoning"])
    @_limit("30/minute")
    async def reasoning_prop_infer(  # noqa: ANN202
        request: Request,
        req: ReasoningPropInferRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Forward-chain propositional facts + rules to a fixpoint."""
        model = get_model()
        with model._lock:
            for prop, val in req.facts.items():
                model.add_logical_fact(prop, val)
            for r in req.rules:
                if len(r) >= 2:
                    model.add_logical_rule(str(r[0]), str(r[1]))
            for r in req.negation_rules:
                if len(r) >= 2:
                    model.reasoning_propositional.add_negation_rule(
                        str(r[0]), str(r[1])
                    )
            result = model.infer_logical()
        return _to_jsonable(result)

    @v1_router.post("/reasoning/syllogism", tags=["reasoning"])
    @_limit("30/minute")
    async def reasoning_syllogism(  # noqa: ANN202
        request: Request,
        req: ReasoningSyllogismRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Run a categorical syllogism (Barbara / Celarent / Darii / Ferio)."""
        major = tuple(req.major)
        minor = tuple(req.minor)
        model = get_model()
        with model._lock:
            result = model.syllogism(major, minor)
        return _to_jsonable(result)

    @v1_router.post("/reasoning/induct", tags=["reasoning"])
    @_limit("30/minute")
    async def reasoning_induct(  # noqa: ANN202
        request: Request,
        req: ReasoningInductRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Induce a (key, value) rule from labeled examples."""
        if len(req.examples) != len(req.labels):
            raise HTTPException(
                status_code=400,
                detail="examples and labels must have the same length",
            )
        model = get_model()
        with model._lock:
            result = model.induct_rule(req.examples, req.labels)
        return _to_jsonable(result)

    @v1_router.post("/reasoning/analogize", tags=["reasoning"])
    @_limit("30/minute")
    async def reasoning_analogize(  # noqa: ANN202
        request: Request,
        req: ReasoningAnalogizeRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Find a structural analogy between a source and target dict."""
        model = get_model()
        with model._lock:
            result = model.analogize(req.source, req.target)
        return _to_jsonable(result)

    @v1_router.post("/reasoning/abduce", tags=["reasoning"])
    @_limit("30/minute")
    async def reasoning_abduce(  # noqa: ANN202
        request: Request,
        req: ReasoningAbduceRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Pick the best explanation for an observation."""
        if req.priors is not None and len(req.priors) != len(req.hypotheses):
            raise HTTPException(
                status_code=400,
                detail="priors length must match hypotheses length",
            )
        model = get_model()
        with model._lock:
            result = model.abduce(
                req.observation, req.hypotheses, priors=req.priors
            )
        return _to_jsonable(result)

    @v1_router.post("/reasoning/defaults", tags=["reasoning"])
    @_limit("30/minute")
    async def reasoning_defaults(  # noqa: ANN202
        request: Request,
        req: ReasoningDefaultsRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Apply defeasible default rules to a set of facts."""
        model = get_model()
        with model._lock:
            for d in req.defaults:
                rule_t = tuple(d.rule)
                exc_t = tuple(d.exception) if d.exception is not None else None
                model.add_default_rule(rule_t, exception=exc_t)
            result = model.conclude_defaults(req.facts)
        return _to_jsonable(result)

    @v1_router.post("/reasoning/causal", tags=["reasoning"])
    @_limit("30/minute")
    async def reasoning_causal(  # noqa: ANN202
        request: Request,
        req: ReasoningCausalRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Trace a causal chain from a starting node."""
        model = get_model()
        with model._lock:
            for link in req.links:
                if len(link) >= 2:
                    model.add_causal_link(str(link[0]), str(link[1]))
            result = model.trace_causal_chain(req.start, max_depth=req.max_depth)
        return _to_jsonable(result)

    # ------------------------------------------------------------------
    # Phase 7 — Causal endpoints.
    # ------------------------------------------------------------------
    @v1_router.post("/causal/decision-tree", tags=["causal"])
    @_limit("30/minute")
    async def causal_decision_tree(  # noqa: ANN202
        request: Request,
        req: CausalDecisionTreeRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Fit an ID3-style decision tree to (features, labels)."""
        if len(req.features) != len(req.labels):
            raise HTTPException(
                status_code=400,
                detail="features and labels must have the same length",
            )
        features = np.asarray(req.features, dtype=float)
        labels = np.asarray(req.labels)
        _ensure_finite(features, "features")
        model = get_model()
        with model._lock:
            result = model.fit_decision_tree(features, labels)
        return _to_jsonable(result)

    @v1_router.post("/causal/game", tags=["causal"])
    @_limit("30/minute")
    async def causal_game(  # noqa: ANN202
        request: Request,
        req: CausalGameRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Analyze a 2-player normal-form game; find pure Nash equilibria."""
        payoff_a = np.asarray(req.payoff_a, dtype=float)
        _ensure_finite(payoff_a, "payoff_a")
        payoff_b = None
        if req.payoff_b is not None:
            payoff_b = np.asarray(req.payoff_b, dtype=float)
            _ensure_finite(payoff_b, "payoff_b")
        model = get_model()
        with model._lock:
            result = model.analyze_game(payoff_a, payoff_b)
        return _to_jsonable(result)

    @v1_router.post("/causal/counterfactual", tags=["causal"])
    @_limit("30/minute")
    async def causal_counterfactual(  # noqa: ANN202
        request: Request,
        req: CausalCounterfactualRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Estimate counterfactual outcome under do(X[index] = value)."""
        observed = np.asarray(req.observed, dtype=float)
        _ensure_finite(observed, "observed")
        intervention = {"index": req.index, "value": req.value}
        model = get_model()
        with model._lock:
            result = model.counterfactual(observed, intervention)
        return _to_jsonable(result)

    @v1_router.post("/causal/bandit", tags=["causal"])
    @_limit("30/minute")
    async def causal_bandit(  # noqa: ANN202
        request: Request,
        req: CausalBanditRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Select the next bandit arm (epsilon-greedy + UCB1)."""
        model = get_model()
        with model._lock:
            result = model.select_bandit_arm(req.rewards_history)
        return _to_jsonable(result)

    @v1_router.post("/causal/pomdp", tags=["causal"])
    @_limit("30/minute")
    async def causal_pomdp(  # noqa: ANN202
        request: Request,
        req: CausalPOMDPRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Solve a (PO)MDP via value iteration."""
        transitions = np.asarray(req.transitions, dtype=float)
        observations = np.asarray(req.observations, dtype=float)
        rewards = np.asarray(req.rewards, dtype=float)
        if transitions.size > 0:
            _ensure_finite(transitions, "transitions")
        if observations.size > 0:
            _ensure_finite(observations, "observations")
        if rewards.size > 0:
            _ensure_finite(rewards, "rewards")
        model = get_model()
        with model._lock:
            result = model.solve_pomdp(transitions, observations, rewards)
        return _to_jsonable(result)

    @v1_router.post("/causal/discover-graph", tags=["causal"])
    @_limit("30/minute")
    async def causal_discover_graph(  # noqa: ANN202
        request: Request,
        req: CausalGraphRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Discover a causal graph from observational data (PC-style)."""
        data = np.asarray(req.data, dtype=float)
        _ensure_finite(data, "data")
        model = get_model()
        with model._lock:
            result = model.discover_causal_graph(data, var_names=req.var_names)
        return _to_jsonable(result)

    @v1_router.post("/causal/intervene", tags=["causal"])
    @_limit("30/minute")
    async def causal_intervene(  # noqa: ANN202
        request: Request,
        req: CausalInterveneRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Estimate the effect of do(X[intervention_var] = value)."""
        data = np.asarray(req.data, dtype=float)
        _ensure_finite(data, "data")
        model = get_model()
        with model._lock:
            result = model.intervene(
                data, req.intervention_var, req.intervention_value
            )
        return _to_jsonable(result)

    # ------------------------------------------------------------------ #
    # Phase 7 cognitive-upgrade endpoints (plasticity / cogtime / cogmem /
    # knowledge / metacog / experiment / multiagent). Each module is gated
    # behind an ``enable_*`` feature flag (or not yet integrated, for the
    # multiagent modules); when the attribute is ``None`` the endpoint
    # returns 503 ``{"detail": "module X not enabled", "enabled": false}``.
    # ------------------------------------------------------------------ #

    # --- plasticity / architect ---------------------------------------
    @v1_router.get("/architect/stats", tags=["architect"])
    @_limit("30/minute")
    @_log_phase7("architect", "stats")
    async def architect_stats(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the architecture optimizer's summary statistics."""
        # P2.11 性能优化: 架构统计仅在每 eval_interval 周期更新一次，
        # 用 30s TTL 缓存避免每次请求都获取模型锁并重算 stats。
        result = _cached_architect_stats()
        if result is _CACHE_DISABLED:
            return _module_disabled_response("architect")
        return result

    @v1_router.get("/architect/dormant", tags=["architect"])
    @_limit("30/minute")
    @_log_phase7("architect", "dormant")
    async def architect_dormant(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """List the names of currently dormant modules."""
        # P2.11 性能优化: 休眠模块列表变化缓慢，用 30s TTL 缓存。
        result = _cached_architect_dormant()
        if result is _CACHE_DISABLED:
            return _module_disabled_response("architect")
        return result

    @v1_router.post("/architect/reactivate", tags=["architect"])
    @_limit("30/minute")
    @_log_phase7("architect", "reactivate")
    async def architect_reactivate(  # noqa: ANN202
        request: Request,
        req: ArchitectReactivateRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Reactivate a dormant module by name."""
        model = get_model()
        with model._lock:
            arch = model.architect
            if arch is None:
                return _module_disabled_response("architect")
            ok = arch.reactivate(model.modules, req.name)
        return {"reactivated": bool(ok), "name": req.name}

    # --- cogtime / temporal_memory ------------------------------------
    @v1_router.get("/temporal_memory/context", tags=["temporal_memory"])
    @_limit("30/minute")
    @_log_phase7("temporal_memory", "context")
    async def temporal_memory_context(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the current temporal-memory hidden-state context vector."""
        model = get_model()
        with model._lock:
            tm = model.temporal_memory
            if tm is None:
                return _module_disabled_response("temporal_memory")
            return {"context": _to_jsonable(tm.get_context())}

    @v1_router.get("/temporal_memory/spectral_radius", tags=["temporal_memory"])
    @_limit("30/minute")
    @_log_phase7("temporal_memory", "spectral_radius")
    async def temporal_memory_spectral_radius(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the spectral radius of the temporal-memory recurrent weight."""
        model = get_model()
        with model._lock:
            tm = model.temporal_memory
            if tm is None:
                return _module_disabled_response("temporal_memory")
            return {"spectral_radius": float(tm.spectral_radius)}

    # --- cogtime / layered_predictor ----------------------------------
    @v1_router.get("/layered_predictor/context", tags=["layered_predictor"])
    @_limit("30/minute")
    @_log_phase7("layered_predictor", "context")
    async def layered_predictor_context(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the concatenated [L0, L1, L2] layered-predictor context."""
        model = get_model()
        with model._lock:
            lp = model.layered_predictor
            if lp is None:
                return _module_disabled_response("layered_predictor")
            return {"context": _to_jsonable(lp.get_context())}

    @v1_router.get("/layered_predictor/rhythm", tags=["layered_predictor"])
    @_limit("30/minute")
    @_log_phase7("layered_predictor", "rhythm")
    async def layered_predictor_rhythm(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the rhythmic-confidence heuristic of the layered predictor."""
        model = get_model()
        with model._lock:
            lp = model.layered_predictor
            if lp is None:
                return _module_disabled_response("layered_predictor")
            return {"rhythm": float(lp.predict_rhythm())}

    # --- cogmem / episodic_graph --------------------------------------
    @v1_router.get("/episodic_graph/recent", tags=["episodic_graph"])
    @_limit("30/minute")
    @_log_phase7("episodic_graph", "recent")
    async def episodic_graph_recent(  # noqa: ANN202
        request: Request,
        n: int = 10,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the ``n`` most recent episodes (by step)."""
        if n < 1 or n > 1000:
            raise HTTPException(
                status_code=400, detail="n must be in [1, 1000]"
            )
        # P2.11 性能优化: 情节图历史变化缓慢，用 60s TTL 缓存（按 n 分键）。
        result = _cached_episodic_recent(n)
        if result is _CACHE_DISABLED:
            return _module_disabled_response("episodic_graph")
        return result

    @v1_router.get("/episodic_graph/node_count", tags=["episodic_graph"])
    @_limit("30/minute")
    @_log_phase7("episodic_graph", "node_count")
    async def episodic_graph_node_count(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the current episodic-graph node count."""
        # P2.11 性能优化: 节点数随周期增长但变化缓慢，用 60s TTL 缓存。
        result = _cached_episodic_node_count()
        if result is _CACHE_DISABLED:
            return _module_disabled_response("episodic_graph")
        return result

    @v1_router.post("/episodic_graph/plan", tags=["episodic_graph"])
    @_limit("30/minute")
    @_log_phase7("episodic_graph", "plan")
    async def episodic_graph_plan(  # noqa: ANN202
        request: Request,
        req: EpisodicPlanRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Plan a minimum-free-energy path between two state vectors."""
        start_state = np.asarray(req.start_state, dtype=float)
        goal_state = np.asarray(req.goal_state, dtype=float)
        _ensure_finite(start_state, "start_state")
        _ensure_finite(goal_state, "goal_state")
        model = get_model()
        with model._lock:
            eg = model.episodic_graph
            if eg is None:
                return _module_disabled_response("episodic_graph")
            path = eg.plan(start_state, goal_state, horizon=req.horizon)
        return {"path": path}

    # --- cogmem / semantic_index --------------------------------------
    @v1_router.get("/semantic_index/size", tags=["semantic_index"])
    @_limit("30/minute")
    @_log_phase7("semantic_index", "size")
    async def semantic_index_size(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the number of entries in the semantic index."""
        model = get_model()
        with model._lock:
            si = model.semantic_index
            if si is None:
                return _module_disabled_response("semantic_index")
            return {"size": int(si.size)}

    @v1_router.post("/semantic_index/query", tags=["semantic_index"])
    @_limit("30/minute")
    @_log_phase7("semantic_index", "query")
    async def semantic_index_query(  # noqa: ANN202
        request: Request,
        req: SemanticQueryRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Query the semantic index for the k nearest entries."""
        vec = np.asarray(req.vec, dtype=float)
        _ensure_finite(vec, "vec")
        model = get_model()
        with model._lock:
            si = model.semantic_index
            if si is None:
                return _module_disabled_response("semantic_index")
            results = si.query(vec, k=req.k)
        return {
            "results": [
                {"node_id": nid, "similarity": float(sim), "metadata": meta}
                for nid, sim, meta in results
            ]
        }

    # --- knowledge / logic_layer --------------------------------------
    @v1_router.get("/logic_layer/rules", tags=["logic_layer"])
    @_limit("30/minute")
    @_log_phase7("logic_layer", "rules")
    async def logic_layer_rules(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """List all registered fuzzy logic rules."""
        model = get_model()
        with model._lock:
            ll = model.logic_layer
            if ll is None:
                return _module_disabled_response("logic_layer")
            return {"rules": _to_jsonable([vars(r) for r in ll.rules])}

    @v1_router.post("/logic_layer/rule", tags=["logic_layer"])
    @_limit("30/minute")
    @_log_phase7("logic_layer", "rule")
    async def logic_layer_add_rule(  # noqa: ANN202
        request: Request,
        req: LogicRuleRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Register a new fuzzy logic rule."""
        model = get_model()
        with model._lock:
            ll = model.logic_layer
            if ll is None:
                return _module_disabled_response("logic_layer")
            ll.add_rule(
                req.name,
                list(req.antecedents),
                req.consequent,
                weight=req.weight,
                description=req.description,
            )
        return {"registered": req.name}

    @v1_router.post("/logic_layer/predicate", tags=["logic_layer"])
    @_limit("30/minute")
    @_log_phase7("logic_layer", "predicate")
    async def logic_layer_set_predicate(  # noqa: ANN202
        request: Request,
        req: LogicPredicateRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Set a predicate truth value (clamped to [0, 1])."""
        model = get_model()
        with model._lock:
            ll = model.logic_layer
            if ll is None:
                return _module_disabled_response("logic_layer")
            ll.set_predicate(req.name, req.value)
        return {"name": req.name, "value": req.value}

    @v1_router.get("/logic_layer/check", tags=["logic_layer"])
    @_limit("30/minute")
    @_log_phase7("logic_layer", "check")
    async def logic_layer_check(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Evaluate all rules and summarize violations."""
        model = get_model()
        with model._lock:
            ll = model.logic_layer
            if ll is None:
                return _module_disabled_response("logic_layer")
            return _to_jsonable(ll.check_all())

    @v1_router.get("/logic_layer/penalty", tags=["logic_layer"])
    @_limit("30/minute")
    @_log_phase7("logic_layer", "penalty")
    async def logic_layer_penalty(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the per-rule penalty signal vector."""
        model = get_model()
        with model._lock:
            ll = model.logic_layer
            if ll is None:
                return _module_disabled_response("logic_layer")
            return {"penalty": _to_jsonable(ll.get_penalty_signal())}

    # --- knowledge / causal_inference ---------------------------------
    @v1_router.post("/causal_inference/transition", tags=["causal_inference"])
    @_limit("30/minute")
    @_log_phase7("causal_inference", "transition")
    async def causal_inference_set_transition(  # noqa: ANN202
        request: Request,
        req: CausalTransitionRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Set the causal transition matrix (must be a square 2-D matrix)."""
        matrix = np.asarray(req.matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise HTTPException(
                status_code=400,
                detail="matrix must be a square 2-D array",
            )
        _ensure_finite(matrix, "matrix")
        model = get_model()
        with model._lock:
            ci = model.causal_inference
            if ci is None:
                return _module_disabled_response("causal_inference")
            ci.set_transition(matrix)
        return {"shape": list(matrix.shape)}

    @v1_router.post("/causal_inference/do", tags=["causal_inference"])
    @_limit("30/minute")
    @_log_phase7("causal_inference", "do")
    async def causal_inference_do(  # noqa: ANN202
        request: Request,
        req: CausalDoRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Apply a single do-intervention do(X[index] = value)."""
        intervention = {req.index: req.value}
        model = get_model()
        with model._lock:
            ci = model.causal_inference
            if ci is None:
                return _module_disabled_response("causal_inference")
            result = ci.do_calculus(intervention)
        return _to_jsonable(result)

    @v1_router.post("/causal_inference/counterfactual", tags=["causal_inference"])
    @_limit("30/minute")
    @_log_phase7("causal_inference", "counterfactual")
    async def causal_inference_counterfactual(  # noqa: ANN202
        request: Request,
        req: CausalInferenceCounterfactualRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Estimate a counterfactual state under do(X[index] = value)."""
        observed = np.asarray(req.observed, dtype=float)
        _ensure_finite(observed, "observed")
        intervention = {req.index: req.value}
        model = get_model()
        with model._lock:
            ci = model.causal_inference
            if ci is None:
                return _module_disabled_response("causal_inference")
            result = ci.counterfactual(observed, intervention)
        return {"counterfactual": _to_jsonable(result)}

    @v1_router.post("/causal_inference/confounders", tags=["causal_inference"])
    @_limit("30/minute")
    @_log_phase7("causal_inference", "confounders")
    async def causal_inference_confounders(  # noqa: ANN202
        request: Request,
        req: CausalConfoundersRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Identify potential confounders between two target variables."""
        model = get_model()
        with model._lock:
            ci = model.causal_inference
            if ci is None:
                return _module_disabled_response("causal_inference")
            confs = ci.identify_confounders(req.var_a, req.var_b)
        return {"confounders": confs}

    # --- metacog / meta_cognition -------------------------------------
    @v1_router.get("/meta_cognition/confidence", tags=["meta_cognition"])
    @_limit("30/minute")
    @_log_phase7("meta_cognition", "confidence")
    async def meta_cognition_confidence(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the current meta-cognitive confidence in [0, 1]."""
        model = get_model()
        with model._lock:
            mc = model.meta_cognition
            if mc is None:
                return _module_disabled_response("meta_cognition")
            return {"confidence": float(mc.get_confidence())}

    @v1_router.get("/meta_cognition/uncertainty", tags=["meta_cognition"])
    @_limit("30/minute")
    @_log_phase7("meta_cognition", "uncertainty")
    async def meta_cognition_uncertainty(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the per-dimension uncertainty vector."""
        model = get_model()
        with model._lock:
            mc = model.meta_cognition
            if mc is None:
                return _module_disabled_response("meta_cognition")
            return {"uncertainty": _to_jsonable(mc.get_uncertainty_vector())}

    @v1_router.get("/meta_cognition/should_seek_info", tags=["meta_cognition"])
    @_limit("30/minute")
    @_log_phase7("meta_cognition", "should_seek_info")
    async def meta_cognition_should_seek_info(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return whether the model should actively seek information."""
        model = get_model()
        with model._lock:
            mc = model.meta_cognition
            if mc is None:
                return _module_disabled_response("meta_cognition")
            return {"should_seek_info": bool(mc.should_seek_info())}

    @v1_router.get("/meta_cognition/stats", tags=["meta_cognition"])
    @_limit("30/minute")
    @_log_phase7("meta_cognition", "stats")
    async def meta_cognition_stats(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the meta-cognition summary statistics."""
        model = get_model()
        with model._lock:
            mc = model.meta_cognition
            if mc is None:
                return _module_disabled_response("meta_cognition")
            return _to_jsonable(mc.stats)

    # --- experiment / experiment_planner ------------------------------
    @v1_router.get("/experiment_planner/candidates", tags=["experiment_planner"])
    @_limit("30/minute")
    @_log_phase7("experiment_planner", "candidates")
    async def experiment_planner_candidates(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """List all candidate experiments."""
        model = get_model()
        with model._lock:
            ep = model.experiment_planner
            if ep is None:
                return _module_disabled_response("experiment_planner")
            cands = list(ep.candidates)
        return {
            "candidates": _to_jsonable([vars(c) for c in cands]),
            "count": len(cands),
        }

    @v1_router.post("/experiment_planner/select_best", tags=["experiment_planner"])
    @_limit("30/minute")
    @_log_phase7("experiment_planner", "select_best")
    async def experiment_planner_select_best(  # noqa: ANN202
        request: Request,
        req: ExperimentSelectBestRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Select the best unexecuted candidate by predicted info gain."""
        model = get_model()
        with model._lock:
            ep = model.experiment_planner
            if ep is None:
                return _module_disabled_response("experiment_planner")
            best = ep.select_best(req.current_uncertainty)
        if best is None:
            return {"best": None}
        return {"best": _to_jsonable(vars(best))}

    @v1_router.post("/experiment_planner/record_result", tags=["experiment_planner"])
    @_limit("30/minute")
    @_log_phase7("experiment_planner", "record_result")
    async def experiment_planner_record_result(  # noqa: ANN202
        request: Request,
        req: ExperimentRecordRequest,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Record the outcome of an executed candidate experiment."""
        model = get_model()
        with model._lock:
            ep = model.experiment_planner
            if ep is None:
                return _module_disabled_response("experiment_planner")
            cands = list(ep.candidates)
            if req.candidate_id < 0 or req.candidate_id >= len(cands):
                raise HTTPException(
                    status_code=404,
                    detail=f"candidate_id {req.candidate_id} not found",
                )
            candidate = cands[req.candidate_id]
            ep.record_result(candidate, req.fe_before, req.fe_after)
        return {
            "recorded": True,
            "candidate_id": req.candidate_id,
            "actual_gain": float(candidate.actual_gain),
        }

    @v1_router.get("/experiment_planner/stats", tags=["experiment_planner"])
    @_limit("30/minute")
    @_log_phase7("experiment_planner", "stats")
    async def experiment_planner_stats(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the experiment-planner summary statistics."""
        model = get_model()
        with model._lock:
            ep = model.experiment_planner
            if ep is None:
                return _module_disabled_response("experiment_planner")
            return _to_jsonable(ep.stats)

    # --- experiment / hypothesis_tester -------------------------------
    @v1_router.get("/hypothesis_tester/hypotheses", tags=["hypothesis_tester"])
    @_limit("30/minute")
    @_log_phase7("hypothesis_tester", "hypotheses")
    async def hypothesis_tester_hypotheses(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """List all generated hypotheses."""
        model = get_model()
        with model._lock:
            ht = model.hypothesis_tester
            if ht is None:
                return _module_disabled_response("hypothesis_tester")
            hyps = list(ht.hypotheses)
        return {
            "hypotheses": _to_jsonable([vars(h) for h in hyps]),
            "count": len(hyps),
        }

    @v1_router.get("/hypothesis_tester/supported", tags=["hypothesis_tester"])
    @_limit("30/minute")
    @_log_phase7("hypothesis_tester", "supported")
    async def hypothesis_tester_supported(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """List the tested-and-supported hypotheses."""
        model = get_model()
        with model._lock:
            ht = model.hypothesis_tester
            if ht is None:
                return _module_disabled_response("hypothesis_tester")
            supported = ht.get_supported()
        return {
            "supported": _to_jsonable([vars(h) for h in supported]),
            "count": len(supported),
        }

    @v1_router.get("/hypothesis_tester/stats", tags=["hypothesis_tester"])
    @_limit("30/minute")
    @_log_phase7("hypothesis_tester", "stats")
    async def hypothesis_tester_stats(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the hypothesis-tester summary statistics."""
        model = get_model()
        with model._lock:
            ht = model.hypothesis_tester
            if ht is None:
                return _module_disabled_response("hypothesis_tester")
            return _to_jsonable(ht.stats)

    # --- multiagent / world (not integrated -> always 503) ------------
    @v1_router.get("/world/collaboration_stats", tags=["world"])
    @_limit("30/minute")
    @_log_phase7("world", "collaboration_stats")
    async def world_collaboration_stats(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return multiagent-world collaboration statistics."""
        return _module_disabled_response("world")

    @v1_router.get("/world/agent_count", tags=["world"])
    @_limit("30/minute")
    @_log_phase7("world", "agent_count")
    async def world_agent_count(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the number of agents in the multiagent world."""
        return _module_disabled_response("world")

    @v1_router.get("/world/step_count", tags=["world"])
    @_limit("30/minute")
    @_log_phase7("world", "step_count")
    async def world_step_count(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the multiagent-world step count."""
        return _module_disabled_response("world")

    # --- multiagent / communication (not integrated -> always 503) ----
    @v1_router.get("/communication/emergent_meanings", tags=["communication"])
    @_limit("30/minute")
    @_log_phase7("communication", "emergent_meanings")
    async def communication_emergent_meanings(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the emergent symbol->meaning mapping."""
        return _module_disabled_response("communication")

    @v1_router.get("/communication/stats", tags=["communication"])
    @_limit("30/minute")
    @_log_phase7("communication", "stats")
    async def communication_stats(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the communication-channel summary statistics."""
        return _module_disabled_response("communication")

    # --- multiagent / culture (not integrated -> always 503) ----------
    @v1_router.get("/culture/knowledge_curve", tags=["culture"])
    @_limit("30/minute")
    @_log_phase7("culture", "knowledge_curve")
    async def culture_knowledge_curve(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the [(gen_id, knowledge_graph_size), ...] knowledge curve."""
        return _module_disabled_response("culture")

    @v1_router.get("/culture/stats", tags=["culture"])
    @_limit("30/minute")
    @_log_phase7("culture", "stats")
    async def culture_stats(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """Return the culture-propagation summary statistics."""
        return _module_disabled_response("culture")

    @v1_router.get("/culture/generations", tags=["culture"])
    @_limit("30/minute")
    @_log_phase7("culture", "generations")
    async def culture_generations(  # noqa: ANN202
        request: Request,
        _api_key: str = Depends(verify_api_key),
    ) -> dict:
        """List all recorded generations."""
        return _module_disabled_response("culture")

    # ---------------------------------------------------------------- #
    # Versioning: /v1/version info endpoint (also mirrored at /version
    # via the legacy root mount, exempt from the deprecation header).
    # ---------------------------------------------------------------- #
    @v1_router.get("/version", tags=["versioning"])
    def version() -> dict:
        """Report the API version and the supported/deprecated version set."""
        return {
            "version": "1.0.0",
            "supported_versions": ["1"],
            "deprecated_versions": [],
            "latest": "1",
        }

    # ---------------------------------------------------------------- #
    # Router registration
    # ---------------------------------------------------------------- #
    # Canonical mount: every business endpoint under /v1 (appears in the
    # OpenAPI schema). Legacy mount: the SAME router is also served at the
    # root with ``include_in_schema=False`` so existing root-path callers
    # keep working, but the routes are hidden from the docs and flagged
    # deprecated by ``deprecate_root_paths`` above. /metrics and / stay on
    # ``app`` directly (root-only, never deprecated); /health, /ready and
    # /version live on v1_router and are therefore reachable at both /v1
    # and the root (exempt from deprecation).
    app.include_router(v1_router, prefix="/v1")
    app.include_router(v1_router, prefix="", include_in_schema=False)

    return app


# A module-level app instance so ``uvicorn zero_data_model.api:app`` works.
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
