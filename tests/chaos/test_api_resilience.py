# tests/chaos/test_api_resilience.py
"""API resilience tests — verify API handles adverse conditions gracefully.

These tests exercise the FastAPI app's resilience layer:

* Custom token-bucket rate limiter (burst=10, 60 rpm) returns 429 with a
  ``Retry-After`` header once the burst is exhausted.
* Malformed / oversized inputs are rejected with the correct status codes
  (422 / 413 / 404 / 405) instead of crashing the server.
* Concurrent requests are handled cleanly.

NOTE: the app uses ``TrustedHostMiddleware`` which rejects any Host header
not in ``ZDM_ALLOWED_HOSTS`` (default ``localhost,127.0.0.1``). The default
``TestClient`` base URL (``http://testserver``) would be rejected with 400,
so every client below is created with ``base_url="http://localhost"``.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from zero_data_model import api as api_module
from zero_data_model.api import create_app


def _make_client() -> TestClient:
    """Build a TestClient pointed at ``localhost`` (trusted host)."""
    return TestClient(create_app(), base_url="http://localhost")


class TestAPIRateLimitResilience:
    """API should handle rate limiting gracefully."""

    def test_rate_limit_returns_429_with_retry_after(self):
        """Bursting past the token-bucket capacity yields 429 + Retry-After.

        The custom ``RateLimiter`` middleware (configured with burst=10
        by default) runs BEFORE routing, so even though ``GET /think`` is a
        405 (the route is POST-only), each request still consumes a token.
        After 10 requests the bucket is empty and subsequent requests get a
        429 with ``Retry-After: 60``.
        """
        client = _make_client()
        # Burst 20 requests — exceeds the default burst capacity of 10.
        responses = [client.get("/think") for _ in range(20)]
        rate_limited = [r for r in responses if r.status_code == 429]
        assert len(rate_limited) > 0, "Expected at least one 429 after bursting"
        assert "Retry-After" in rate_limited[0].headers


class TestAPIMalformedInput:
    """API should handle malformed input gracefully."""

    @pytest.fixture()
    def client(self):
        """A fresh app + client per test.

        slowapi (if installed) is disabled so the module-level limiter's
        budget is not shared across tests — the custom per-app
        ``RateLimiter`` (fresh on every ``create_app()``) still applies,
        but each test only makes one request so it never trips.
        """
        prev_enabled = None
        if api_module.limiter is not None:
            prev_enabled = api_module.limiter.enabled
            api_module.limiter.enabled = False
        client = _make_client()
        yield client
        if api_module.limiter is not None and prev_enabled is not None:
            api_module.limiter.enabled = prev_enabled

    def test_invalid_json_body(self, client):
        """Non-JSON body with JSON content-type → 422 Unprocessable Entity."""
        r = client.post(
            "/think", data="not json", headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 422

    def test_oversized_body(self, client):
        """An 11 MiB body exceeds the 10 MiB RequestSizeLimitMiddleware → 413."""
        large_body = "x" * (11 * 1024 * 1024)
        r = client.post(
            "/think", data=large_body, headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 413

    def test_missing_content_type(self, client):
        """Missing Content-Type may be accepted (200) or rejected (422)."""
        r = client.post("/think", data='{"cycles": 1}')
        assert r.status_code in (200, 422)

    def test_unknown_endpoint(self, client):
        r = client.get("/nonexistent")
        assert r.status_code == 404

    def test_method_not_allowed(self, client):
        """DELETE on a GET-only endpoint → 405 Method Not Allowed."""
        r = client.delete("/health")
        assert r.status_code == 405


class TestAPIConcurrency:
    """API should handle concurrent requests."""

    def test_concurrent_health_checks(self):
        """10 concurrent GET /health requests should all return 200.

        Each thread uses its own ``TestClient`` (``httpx.Client`` is not
        guaranteed thread-safe), but they share the same FastAPI ``app``
        — exercising real server-side concurrency. ``/health`` is
        rate-limit-exempt, so no throttling interferes.
        """
        app = create_app()
        results: list[int] = []
        lock = threading.Lock()

        def check():
            # Per-thread client avoids httpx.Client shared-state races.
            c = TestClient(app, base_url="http://localhost")
            r = c.get("/health")
            with lock:
                results.append(r.status_code)

        threads = [threading.Thread(target=check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(s == 200 for s in results), f"Got statuses: {results}"
