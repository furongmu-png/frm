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

import hmac
import json
import logging
import os
import time
import uuid
from datetime import datetime
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
            "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
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


def get_model() -> ZeroDataModel:
    """Return the lazily-initialized module-level model instance."""
    global _model
    if _model is None:
        _model = ZeroDataModel(dim=32)
    return _model


def set_model(model: ZeroDataModel | None) -> None:
    """Replace the module-level model (used by /load and tests)."""
    global _model
    _model = model


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


class LoadResponse(BaseModel):
    loaded: bool


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
    """
    cl = request.headers.get("content-length")
    if cl and int(cl) > MAX_BODY:
        return JSONResponse(status_code=413, content={"detail": "payload too large"})
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
    is_production = os.environ.get("ZDM_ENV", "").lower() == "production"

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

    # Structured access log + X-Request-ID.
    app.middleware("http")(request_tracing_middleware)

    # Body size limit (S-HIGH-03): reject payloads larger than MAX_BODY before
    # they reach the application. Registered after the tracing middleware so
    # 413 responses are still logged with a request_id.
    app.middleware("http")(_limit_body_size)

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
        return JSONResponse(
            status_code=exc.status_code,
            headers=exc.headers if getattr(exc, "headers", None) else None,
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
        log.error(
            "unhandled_exception",
            request_id=request_id,
            error=str(exc),
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
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
        """Prometheus metrics endpoint (auth-gated, S-HIGH-02)."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> Any:
        """Liveness probe. Cheap: no model construction."""
        if _shutting_down:
            return JSONResponse(
                status_code=503, content={"status": "shutting_down"}
            )
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse, tags=["health"])
    def ready() -> Any:
        """Readiness probe. Constructs the model if needed and reports
        whether it is ready to serve requests."""
        try:
            get_model()
            return ReadyResponse(status="ok", model_ready=True)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("ready_probe_failed", error=str(exc))
            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "model_ready": False},
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
        start = time.perf_counter()
        signal = model.think(input_data)
        if _think_duration is not None:
            _think_duration.observe(time.perf_counter() - start)
        if _cycle_count is not None:
            _cycle_count.set(model.cycle_count)
        if _free_energy is not None and model.active_inference.free_energy_history:
            _free_energy.set(model.active_inference.free_energy_history[-1])
        return ThinkResponse(
            cycle=int(signal.metadata.get("cycle", model.cycle_count)),
            output=[float(x) for x in np.asarray(signal.data).flatten().tolist()],
            confidence=float(signal.confidence),
        )

    @app.post(
        "/classify",
        response_model=ClassifyResponse,
        tags=["nlp"],
        dependencies=[Depends(verify_api_key)],
    )
    def classify(req: ClassifyRequest) -> ClassifyResponse:
        """Zero-shot classify ``text`` into a topic."""
        model = get_model()
        topic, conf = model.classify_text(req.text)
        return ClassifyResponse(topic=str(topic), confidence=float(conf))

    @app.post(
        "/similarity",
        response_model=SimilarityResponse,
        tags=["nlp"],
        dependencies=[Depends(verify_api_key)],
    )
    def similarity(req: SimilarityRequest) -> SimilarityResponse:
        """Semantic similarity in [0, 1] between two texts."""
        model = get_model()
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
        """Recognize a shape/pattern in a 2D grayscale image (list of rows)."""
        model = get_model()
        if not req.image or not req.image[0]:
            raise HTTPException(status_code=400, detail="image must be non-empty")
        image = np.asarray(req.image, dtype=float)
        if image.ndim != 2:
            raise HTTPException(status_code=400, detail="image must be 2D")
        shape, conf = model.recognize_pattern(image)
        return RecognizeResponse(shape=str(shape), confidence=float(conf))

    @app.post(
        "/save",
        response_model=SaveResponse,
        tags=["persistence"],
        dependencies=[Depends(verify_api_key)],
    )
    def save(req: PathRequest) -> JSONResponse:
        """Persist the current model state to ``name`` (relative) on disk.

        Returns 201 Created with a ``Location`` header pointing at the
        canonical resource URI for the snapshot (Q-LOW-13).

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
    def load(req: PathRequest) -> LoadResponse:
        """Replace the live model with one loaded from ``name`` (relative)."""
        try:
            new_model = ModelSerializer.load(req.name)
        except ValueError as exc:
            # Sandbox violation (ValueError) -> 400, no path leakage.
            raise HTTPException(status_code=400, detail="invalid snapshot name") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="snapshot not found") from exc
        except KeyError as exc:
            raise HTTPException(status_code=400, detail="load failed") from exc
        set_model(new_model)
        return LoadResponse(loaded=True)

    return app


# A module-level app instance so ``uvicorn zero_data_model.api:app`` works.
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
