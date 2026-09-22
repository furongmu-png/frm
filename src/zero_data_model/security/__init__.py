"""Security middleware layer for the ZeroDataModel FastAPI application.

This package bundles the composable security middlewares and helpers used to
harden the HTTP surface of ``zero_data_model.api``:

* :class:`RateLimiter` -- token-bucket per-client rate limiter.
* :class:`SecurityHeadersMiddleware` -- browser-facing security response headers.
* :class:`RequestSizeLimitMiddleware` -- 413 guard on oversized request bodies.
* :func:`verify_api_key` / :func:`get_configured_api_key` / :func:`is_public_path`
  -- opt-in API-key authentication dependency and helpers.
"""

from __future__ import annotations

from .auth import (
    API_KEY_HEADER,
    PUBLIC_PATHS,
    get_configured_api_key,
    is_public_path,
    verify_api_key,
)
from .headers import (
    DEFAULT_MAX_REQUEST_SIZE,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from .rate_limiter import RateLimiter, TokenBucket

__all__ = [
    "API_KEY_HEADER",
    "DEFAULT_MAX_REQUEST_SIZE",
    "PUBLIC_PATHS",
    "RateLimiter",
    "RequestSizeLimitMiddleware",
    "SecurityHeadersMiddleware",
    "TokenBucket",
    "get_configured_api_key",
    "is_public_path",
    "verify_api_key",
]
