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

import contextlib
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
from .model import ZeroDataModel
from .persistence import ModelSerializer

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
    from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
    from prometheus_fastapi_instrumentator import Instrumentator

    _HAS_PROMETHEUS = True
except Exception:  # pragma: no cover - optional dep missing
    _HAS_PROMETHEUS = False
    Instrumentator = None  # type: ignore[assignment, misc]
    Histogram = None  # type: ignore[assignment, misc]
    Gauge = None  # type: ignore[assignment, misc]
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
            # can promote them to top-level JSON keys.
            extra = {k: v for k, v in kwargs.items() if k != "exc_info"}
            exc_info = kwargs.get("exc_info")
            getattr(self._logger, level)(msg, extra=extra, exc_info=exc_info)

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
# Prometheus custom metrics (only constructed if prometheus is available)
# --------------------------------------------------------------------------- #

if _HAS_PROMETHEUS:
    _think_duration = Histogram(
        "zdm_think_duration_seconds",
        "Wall-clock duration of /think cycles in seconds.",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    )
    _cycle_count = Gauge(
        "zdm_cycle_count",
        "Current model cycle count.",
    )
    _free_energy = Gauge(
        "zdm_free_energy_last",
        "Last think() cycle free energy",
    )
else:
    _think_duration = None  # type: ignore[assignment]
    _cycle_count = None  # type: ignore[assignment]
    _free_energy = None  # type: ignore[assignment]


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

    @app.get(
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

    @app.get(
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
    @app.post(
        "/think",
        response_model=ThinkResponse,
        tags=["cognitive"],
        dependencies=[Depends(verify_api_key)],
    )
    @_limit("10/minute")
    def think(
        request: Request,
        req: ThinkRequest | None = Body(default=None),  # noqa: B008
    ) -> ThinkResponse:
        """Run one thought cycle. With no ``input`` the model self-generates."""
        model = get_model()
        body = req or ThinkRequest()
        input_data = (
            np.asarray(body.input, dtype=float) if body.input else None
        )
        if input_data is not None:
            _ensure_finite(input_data, "input")
        start = time.perf_counter()
        signal = model.think(input_data)
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

    @app.post(
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

    @app.post(
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

    @app.post(
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

    @app.post(
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

    @app.post(
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

    @app.post(
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

    @app.post(
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

    @app.post(
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
            headers={"Location": f"/load/{req.name}"},
        )

    @app.post(
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
                name=req.name,
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=500, detail="load failed") from exc
        set_model(new_model)
        return LoadResponse(loaded=True)

    # ---------------------------------------------------------------- #
    # Causal emergence endpoints (spec §2.3)
    # ---------------------------------------------------------------- #
    @app.post(
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

    @app.post(
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

    @app.post(
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

    @app.post(
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

    @app.post(
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

    @app.post(
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
    @app.post("/memory/encode", tags=["memory"])
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

    @app.post("/memory/retrieve", tags=["memory"])
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

    @app.post("/memory/consolidate", tags=["memory"])
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

    @app.post("/planning/trajectory", tags=["planning"])
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

    @app.post("/planning/decompose", tags=["planning"])
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

    @app.post("/planning/sequence", tags=["planning"])
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

    @app.post("/multimodal/align", tags=["multimodal"])
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

    @app.post("/multimodal/fuse", tags=["multimodal"])
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

    @app.post("/multimodal/contrastive", tags=["multimodal"])
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

    @app.post("/rl/step", tags=["rl"])
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

    @app.post("/rl/train-q", tags=["rl"])
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

    @app.post("/rl/search-mcts", tags=["rl"])
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
    @app.post("/audio/encode", tags=["audio"])
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

    @app.post("/audio/onsets", tags=["audio"])
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

    @app.post("/audio/pitch", tags=["audio"])
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

    @app.post("/audio/classify", tags=["audio"])
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

    @app.post("/audio/segment", tags=["audio"])
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

    @app.post("/audio/music", tags=["audio"])
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
    @app.post("/graph/encode", tags=["graph"])
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

    @app.post("/graph/communities", tags=["graph"])
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

    @app.post("/graph/path", tags=["graph"])
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

    @app.post("/graph/centrality", tags=["graph"])
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

    @app.post("/graph/isomorphism", tags=["graph"])
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

    @app.post("/graph/track", tags=["graph"])
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

    @app.post("/graph/spanning", tags=["graph"])
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
    @app.post("/robotics/motion", tags=["robotics"])
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

    @app.post("/robotics/forward", tags=["robotics"])
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

    @app.post("/robotics/inverse", tags=["robotics"])
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

    @app.post("/robotics/fuse", tags=["robotics"])
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

    @app.post("/robotics/kalman", tags=["robotics"])
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

    @app.post("/robotics/gait", tags=["robotics"])
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

    @app.post("/robotics/optimize", tags=["robotics"])
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

    @app.post("/robotics/collision", tags=["robotics"])
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

    @app.post("/robotics/path-collision", tags=["robotics"])
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

    @app.post("/robotics/mpc", tags=["robotics"])
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
    @app.post("/time/encode", tags=["time"])
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

    @app.post("/time/seasonality", tags=["time"])
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

    @app.post("/time/frequency", tags=["time"])
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

    @app.post("/time/events", tags=["time"])
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

    @app.post("/time/anomaly", tags=["time"])
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

    @app.post("/time/cycle", tags=["time"])
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

    @app.post("/time/forecast", tags=["time"])
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
    @app.post("/code/encode", tags=["code"])
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

    @app.post("/code/ast", tags=["code"])
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

    @app.post("/code/compare", tags=["code"])
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

    @app.post("/code/defects", tags=["code"])
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

    @app.post("/code/control-flow", tags=["code"])
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

    @app.post("/code/style", tags=["code"])
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

    @app.post("/code/dependencies", tags=["code"])
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

    return app


# A module-level app instance so ``uvicorn zero_data_model.api:app`` works.
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
