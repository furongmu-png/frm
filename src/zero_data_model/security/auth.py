"""API Key and optional JWT authentication for FastAPI.

Provides a reusable, self-contained API-key authentication dependency for
FastAPI applications. Authentication is **opt-in**: when the ``ZDM_API_KEY``
environment variable is unset the dependency allows every request through
(local development mode). When set, every non-public endpoint requires an
``X-API-Key`` header that matches the configured value, compared in constant
time to mitigate timing attacks (CWE-208).

Public paths (health probes, metrics scrape, docs) are always accessible
without a key so that liveness/readiness checks and Prometheus scrapes work
without distributing a key to the prober.

This module is independent of ``zero_data_model.api``'s own ``verify_api_key``
dependency (which predates this package and is wired into every endpoint).
The two are functionally equivalent; this module exists as the canonical,
reusable implementation for new surfaces and for direct unit testing.
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Endpoints that don't require authentication. Health/readiness probes must
# stay open so orchestrators (k8s liveness/readiness) and Prometheus scrapers
# (which do not carry an API key) can reach them.
PUBLIC_PATHS = frozenset(
    {
        "/",
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


def is_public_path(path: str) -> bool:
    """Check if a path should be accessible without an API key."""
    if path in PUBLIC_PATHS:
        return True
    # WebSocket upgrade paths.
    if path.startswith("/ws"):
        return True
    # Swagger UI assets.
    return path.startswith("/docs") or path.startswith("/redoc")


def get_configured_api_key() -> str | None:
    """Get API key from environment. Returns ``None`` if not set (auth disabled)."""
    return os.environ.get("ZDM_API_KEY") or None


async def verify_api_key(
    request: Request, api_key: str | None = Security(API_KEY_HEADER)
) -> bool:
    """FastAPI dependency: verify the ``X-API-Key`` header.

    Behaviour:
    * If ``ZDM_API_KEY`` is unset, auth is disabled and every request is allowed
      (development mode).
    * If the request path is public (see :data:`PUBLIC_PATHS`), allow it.
    * Otherwise require a non-empty ``X-API-Key`` header that matches the
      configured key in constant time.
    """
    configured = get_configured_api_key()
    if configured is None:
        # Auth not configured -- allow all.
        return True
    if is_public_path(request.url.path):
        return True
    if not api_key:
        raise HTTPException(
            status_code=401, detail="API key required (X-API-Key header)"
        )
    # Constant-time comparison to prevent timing attacks.
    if not hmac.compare_digest(api_key, configured):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True
