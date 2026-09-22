"""Security headers and request-size-limit middlewares for FastAPI.

``SecurityHeadersMiddleware`` adds a baseline set of browser-facing security
response headers (``X-Content-Type-Options``, ``X-Frame-Options``,
``X-XSS-Protection``, ``Referrer-Policy``, ``Cache-Control``) to every
response, and adds ``Strict-Transport-Security`` (HSTS) only on HTTPS so a
plain-HTTP dev server never advertises an HSTS policy that would lock clients
out of the unencrypted port.

``RequestSizeLimitMiddleware`` rejects request bodies whose declared
``Content-Length`` exceeds a configurable cap with ``413 Payload Too Large``,
before the body is read. This complements (does not replace) the stricter
per-route body cap enforced inside ``api.py`` -- the outer cap is a coarse
DoS guard, the inner cap is the authoritative application limit.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

DEFAULT_MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MiB


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds security-related HTTP headers to all responses."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        # Only add HSTS in production (HTTPS). Adding it on plain HTTP would
        # pin clients to HTTPS for a host that doesn't serve HTTPS.
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects request bodies larger than ``max_size`` bytes.

    Only the declared ``Content-Length`` is inspected (the body is not
    buffered), so oversized uploads are rejected before they reach the
    application. A missing or malformed ``Content-Length`` is treated as
    "no declared size" and the request is allowed through; downstream
    middleware (the stricter per-route cap) handles chunked / streaming
    encodings explicitly.
    """

    def __init__(self, app, max_size: int = DEFAULT_MAX_REQUEST_SIZE):  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        cl = request.headers.get("content-length")
        if cl:
            try:
                cl_int = int(cl)
            except (TypeError, ValueError):
                return await call_next(request)
            if cl_int > self.max_size:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"Request body too large (max {self.max_size} bytes)"
                    },
                )
        return await call_next(request)


__all__ = [
    "DEFAULT_MAX_REQUEST_SIZE",
    "RequestSizeLimitMiddleware",
    "SecurityHeadersMiddleware",
    "Response",
]
