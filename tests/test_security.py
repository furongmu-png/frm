# tests/test_security.py
"""Tests for the ``zero_data_model.security`` middleware layer.

Covers:
* :class:`RateLimiter` (token-bucket) -- burst, refill, per-client isolation,
  stale-bucket cleanup.
* :func:`verify_api_key` (opt-in API-key auth) -- disabled-when-unset,
  correct/wrong/missing key, public-path bypass, constant-time comparison.
* :class:`SecurityHeadersMiddleware` -- header presence + HTTPS-only HSTS.
* :class:`RequestSizeLimitMiddleware` -- small body allowed / large body 413.
* Integration via ``create_app()`` + ``TestClient`` -- health bypasses auth,
  ``/think`` requires auth when ``ZDM_API_KEY`` is set, rate limit returns
  429, CORS headers are emitted.
"""

from __future__ import annotations

import asyncio
import hmac
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by fastapi.testclient.TestClient

from fastapi.testclient import TestClient  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import PlainTextResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from zero_data_model.security import (  # noqa: E402
    RateLimiter,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    TokenBucket,
    get_configured_api_key,
    is_public_path,
    verify_api_key,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_request(host: str = "1.2.3.4", forwarded: str | None = None):
    """Build a lightweight stand-in for ``fastapi.Request``.

    ``RateLimiter`` only inspects ``request.client.host`` and
    ``request.headers.get("x-forwarded-for")``; a ``SimpleNamespace`` with a
    dict for headers satisfies both without spinning up an ASGI scope.
    """
    headers: dict[str, str] = {}
    if forwarded:
        headers["x-forwarded-for"] = forwarded
    return SimpleNamespace(
        client=SimpleNamespace(host=host),
        headers=headers,
    )


def _make_auth_request(path: str = "/think"):
    """Build a stand-in for ``fastapi.Request`` for the auth dependency.

    ``verify_api_key`` only reads ``request.url.path``.
    """
    return SimpleNamespace(url=SimpleNamespace(path=path))


# --------------------------------------------------------------------------- #
# RateLimiter
# --------------------------------------------------------------------------- #

def test_rate_limiter_allows_within_limit():
    """A burst of 10 requests under a capacity-10 bucket all succeed."""
    limiter = RateLimiter(requests_per_minute=60, burst=10)
    req = _make_request(host="10.0.0.1")
    for _ in range(10):
        assert limiter.check(req) is True


def test_rate_limiter_blocks_exceeding_burst():
    """The 11th request in a capacity-10 bucket is rejected."""
    limiter = RateLimiter(requests_per_minute=60, burst=10)
    req = _make_request(host="10.0.0.2")
    for _ in range(10):
        assert limiter.check(req) is True
    # 11th request exceeds the burst capacity -> rejected.
    assert limiter.check(req) is False


def test_rate_limiter_refills_over_time():
    """After enough time elapses, the bucket refills and allows a request."""
    # 60 rpm => 1 token/second; burst=1 so we can exhaust it in one call.
    limiter = RateLimiter(requests_per_minute=60, burst=1)
    req = _make_request(host="10.0.0.3")
    assert limiter.check(req) is True  # consumes the only token
    assert limiter.check(req) is False  # empty
    # Sleep just over 1s so a full token refills.
    time.sleep(1.05)
    assert limiter.check(req) is True


def test_rate_limiter_per_client_isolation():
    """Distinct client IPs maintain independent buckets."""
    limiter = RateLimiter(requests_per_minute=60, burst=2)
    req_a = _make_request(host="10.0.0.10")
    req_b = _make_request(host="10.0.0.11")
    # Exhaust client A's bucket.
    assert limiter.check(req_a) is True
    assert limiter.check(req_a) is True
    assert limiter.check(req_a) is False  # A is throttled
    # Client B is unaffected.
    assert limiter.check(req_b) is True
    assert limiter.check(req_b) is True


def test_rate_limiter_x_forwarded_for_is_used():
    """``X-Forwarded-For`` (leftmost entry) identifies the client behind ingress."""
    limiter = RateLimiter(requests_per_minute=60, burst=1)
    # Same socket host but different X-Forwarded-For -> treated as 2 clients.
    req_a = _make_request(host="10.0.0.1", forwarded="203.0.113.1, 10.0.0.1")
    req_b = _make_request(host="10.0.0.1", forwarded="203.0.113.2, 10.0.0.1")
    assert limiter.check(req_a) is True
    assert limiter.check(req_b) is True  # different forwarded client
    # Same forwarded client as req_a -> throttled (bucket exhausted).
    req_a2 = _make_request(host="10.0.0.1", forwarded="203.0.113.1, 10.0.0.1")
    assert limiter.check(req_a2) is False


def test_rate_limiter_cleanup_stale():
    """``cleanup_stale`` removes buckets idle longer than the threshold."""
    limiter = RateLimiter(requests_per_minute=60, burst=5)
    req = _make_request(host="10.0.0.4")
    limiter.check(req)  # creates a bucket
    assert "10.0.0.4" in limiter.buckets
    # Forcibly age the bucket's last_refill into the past.
    limiter.buckets["10.0.0.4"].last_refill = time.monotonic() - 7200
    limiter.cleanup_stale(max_age_seconds=3600)
    assert "10.0.0.4" not in limiter.buckets


def test_token_bucket_consumes_in_units():
    """``TokenBucket.consume`` deducts the requested number of tokens."""
    bucket = TokenBucket(capacity=5, refill_rate=0.0)
    assert bucket.consume(3) is True
    assert bucket.tokens == pytest.approx(2.0)
    assert bucket.consume(3) is False  # only 2 left
    assert bucket.tokens == pytest.approx(2.0)  # unchanged on rejection


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #

def test_no_api_key_allows_all(monkeypatch):
    """When ``ZDM_API_KEY`` is unset, auth is disabled and everything is allowed."""
    monkeypatch.delenv("ZDM_API_KEY", raising=False)
    assert get_configured_api_key() is None
    req = _make_auth_request("/think")
    assert asyncio.run(verify_api_key(req, api_key=None)) is True


def test_api_key_correct_passes(monkeypatch):
    """A matching ``X-API-Key`` value is accepted."""
    monkeypatch.setenv("ZDM_API_KEY", "secret-key-123")
    req = _make_auth_request("/think")
    assert asyncio.run(verify_api_key(req, api_key="secret-key-123")) is True


def test_api_key_wrong_rejected(monkeypatch):
    """A mismatched key raises 403."""
    monkeypatch.setenv("ZDM_API_KEY", "secret-key-123")
    req = _make_auth_request("/think")
    with pytest.raises(Exception) as exc_info:
        asyncio.run(verify_api_key(req, api_key="wrong-key"))
    # The raised HTTPException must carry 403.
    assert exc_info.value.status_code == 403


def test_api_key_missing_rejected(monkeypatch):
    """A missing key (None) on a protected path raises 401."""
    monkeypatch.setenv("ZDM_API_KEY", "secret-key-123")
    req = _make_auth_request("/think")
    with pytest.raises(Exception) as exc_info:
        asyncio.run(verify_api_key(req, api_key=None))
    assert exc_info.value.status_code == 401


def test_public_paths_no_auth_needed(monkeypatch):
    """Public paths (health/ready/metrics/docs) bypass auth even when configured."""
    monkeypatch.setenv("ZDM_API_KEY", "secret-key-123")
    for path in ("/", "/health", "/ready", "/metrics", "/docs", "/redoc",
                 "/openapi.json"):
        assert is_public_path(path) is True
        req = _make_auth_request(path)
        # No key supplied, but path is public -> allowed.
        assert asyncio.run(verify_api_key(req, api_key=None)) is True


def test_is_public_path_ws_prefix():
    """WebSocket upgrade paths (``/ws*``) are public."""
    assert is_public_path("/ws") is True
    assert is_public_path("/ws/stream") is True
    assert is_public_path("/think") is False


def test_timing_safe_comparison(monkeypatch):
    """``verify_api_key`` compares keys via ``hmac.compare_digest`` (CWE-208)."""
    monkeypatch.setenv("ZDM_API_KEY", "secret-key-123")
    called: list[tuple[object, object]] = []
    real = hmac.compare_digest

    def spy(a, b):
        called.append((a, b))
        return real(a, b)

    monkeypatch.setattr(
        "zero_data_model.security.auth.hmac.compare_digest", spy
    )
    req = _make_auth_request("/think")
    asyncio.run(verify_api_key(req, api_key="secret-key-123"))
    assert called, "hmac.compare_digest was not invoked"
    # The configured key and the supplied key were the two arguments.
    assert called[0] == ("secret-key-123", "secret-key-123")


# --------------------------------------------------------------------------- #
# SecurityHeadersMiddleware
# --------------------------------------------------------------------------- #

async def _ok(request):  # type: ignore[no-untyped-def]
    return PlainTextResponse("ok")


def _headers_app() -> Starlette:
    app = Starlette(routes=[Route("/", _ok), Route("/health", _ok)])
    app.add_middleware(SecurityHeadersMiddleware)
    return app


def test_security_headers_present():
    """All baseline security headers are present on a normal response."""
    client = TestClient(_headers_app())
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-XSS-Protection"] == "1; mode=block"
    assert r.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert r.headers["Cache-Control"] == "no-store"


def test_hsts_only_https():
    """HSTS is added on HTTPS requests but NOT on plain HTTP."""
    # Plain HTTP: no HSTS.
    http_client = TestClient(_headers_app(), base_url="http://testserver")
    r = http_client.get("/")
    assert "Strict-Transport-Security" not in r.headers
    # HTTPS: HSTS present.
    https_client = TestClient(_headers_app(), base_url="https://testserver")
    r = https_client.get("/")
    assert r.status_code == 200
    hsts = r.headers.get("Strict-Transport-Security", "")
    assert "max-age=31536000" in hsts
    assert "includeSubDomains" in hsts


# --------------------------------------------------------------------------- #
# RequestSizeLimitMiddleware
# --------------------------------------------------------------------------- #

async def _echo(request):  # type: ignore[no-untyped-def]
    body = await request.body()
    return PlainTextResponse(f"got {len(body)} bytes")


def _size_app(max_size: int) -> Starlette:
    app = Starlette(routes=[Route("/", _echo, methods=["POST"])])
    app.add_middleware(RequestSizeLimitMiddleware, max_size=max_size)
    return app


def test_small_body_allowed():
    """A body under the cap is passed through."""
    client = TestClient(_size_app(max_size=100))
    r = client.post("/", content=b"x" * 10)
    assert r.status_code == 200
    assert "got 10 bytes" in r.text


def test_large_body_rejected_413():
    """A body over the cap is rejected with 413 before reaching the handler."""
    client = TestClient(_size_app(max_size=100))
    r = client.post("/", content=b"x" * 200)
    assert r.status_code == 413
    assert "too large" in r.json()["detail"].lower()


def test_request_size_missing_content_length_allowed():
    """A GET (no Content-Length) is allowed through the size guard."""
    client = TestClient(_size_app(max_size=100))
    r = client.get("/")
    # _echo is POST-only; GET falls through to 405, NOT 413 -- proving the
    # size guard did not short-circuit.
    assert r.status_code == 405


# --------------------------------------------------------------------------- #
# Integration via create_app() + TestClient
# --------------------------------------------------------------------------- #

@pytest.fixture()
def app_factory(tmp_path, monkeypatch):
    """Yield a factory that builds a fresh ``create_app()`` TestClient.

    Model + persistence are sandboxed per test; slowapi's per-route limiter is
    disabled so it doesn't compound with the new token-bucket limiter. The
    factory reads the *current* environment at call time, so tests can
    ``monkeypatch.setenv`` for ``ZDM_API_KEY`` / ``ZDM_RATE_LIMIT_*`` before
    invoking it.
    """
    from zero_data_model import api as api_module
    from zero_data_model.api import create_app
    from zero_data_model.model import ZeroDataModel
    from zero_data_model.persistence import set_persistence_root

    set_persistence_root(str(tmp_path / "sec_persistence"))
    api_module.set_model(ZeroDataModel(dim=16))
    prev_limiter_enabled = None
    if api_module.limiter is not None:
        prev_limiter_enabled = api_module.limiter.enabled
        api_module.limiter.enabled = False

    def make() -> TestClient:
        app = create_app()
        return TestClient(app, base_url="http://localhost")

    yield make

    api_module._model = None
    if api_module.limiter is not None and prev_limiter_enabled is not None:
        api_module.limiter.enabled = prev_limiter_enabled


def test_health_no_auth_needed(app_factory):
    """``/health`` and ``/ready`` must be reachable without an API key."""
    with app_factory() as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 200


def test_health_no_rate_limit(app_factory):
    """``/health`` is exempt from the rate limiter (many calls still 200)."""
    with app_factory() as c:
        for _ in range(15):
            assert c.get("/health").status_code == 200


def test_think_requires_auth_when_configured(app_factory, monkeypatch):
    """When ``ZDM_API_KEY`` is set, ``/v1/think`` rejects requests without a key."""
    monkeypatch.setenv("ZDM_API_KEY", "secret-key-123")
    with app_factory() as c:
        # No key -> 401.
        r = c.post("/v1/think")
        assert r.status_code == 401
        # Correct key -> 200.
        r = c.post("/v1/think", headers={"X-API-Key": "secret-key-123"})
        assert r.status_code == 200


def test_think_open_when_no_key(app_factory, monkeypatch):
    """Without ``ZDM_API_KEY``, ``/v1/think`` is open (development mode)."""
    monkeypatch.delenv("ZDM_API_KEY", raising=False)
    with app_factory() as c:
        assert c.post("/v1/think").status_code == 200


def test_rate_limit_returns_429(app_factory, monkeypatch):
    """Exceeding the burst returns 429 with a ``Retry-After`` header."""
    monkeypatch.setenv("ZDM_RATE_LIMIT_BURST", "2")
    monkeypatch.setenv("ZDM_RATE_LIMIT_RPM", "2")
    with app_factory() as c:
        # /v1/classify is a cheap non-exempt endpoint.
        for _ in range(2):
            r = c.post("/v1/classify", json={"text": "the algorithm computes"})
            assert r.status_code == 200
        # 3rd request exceeds the burst -> 429.
        r = c.post("/v1/classify", json={"text": "the algorithm computes"})
        assert r.status_code == 429
        assert r.headers.get("Retry-After") == "60"
        body = r.json()
        assert "rate limit" in body["detail"].lower()


def test_rate_limit_response_carries_security_headers(app_factory, monkeypatch):
    """A 429 response is still wrapped by SecurityHeadersMiddleware."""
    monkeypatch.setenv("ZDM_RATE_LIMIT_BURST", "1")
    monkeypatch.setenv("ZDM_RATE_LIMIT_RPM", "1")
    with app_factory() as c:
        c.post("/v1/classify", json={"text": "x"})  # consumes the only token
        r = c.post("/v1/classify", json={"text": "x"})
        assert r.status_code == 429
        # SecurityHeadersMiddleware is outermost, so 429 carries the headers.
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") == "DENY"


def test_cors_headers_present(app_factory):
    """CORS advertises the allowed origin for credentialed cross-origin requests."""
    with app_factory() as c:
        # The default allow-list includes http://localhost:8000.
        r = c.get("/", headers={"Origin": "http://localhost:8000"})
        assert r.status_code == 200
        assert r.headers.get("Access-Control-Allow-Origin") == "http://localhost:8000"


def test_cors_rejects_disallowed_origin(app_factory):
    """A non-allow-listed origin does not get an ``Access-Control-Allow-Origin``."""
    with app_factory() as c:
        r = c.get("/health", headers={"Origin": "http://evil.example"})
        # /health is reachable (public) but CORS does not echo the origin.
        assert r.status_code == 200
        assert "Access-Control-Allow-Origin" not in r.headers


def test_security_headers_on_normal_endpoint(app_factory):
    """Security headers are present on a normal cognitive-endpoint response."""
    with app_factory() as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") == "DENY"
        assert r.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
