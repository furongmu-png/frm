"""Token-bucket rate limiter for FastAPI.

A self-contained, dependency-light rate limiter that throttles per-client
request rates using the token-bucket algorithm. Each client (identified by
``X-Forwarded-For`` when behind an ingress, otherwise the socket peer) gets
its own bucket with a configurable capacity (burst) and refill rate.

This is intentionally framework-agnostic at the core (``RateLimiter.check``
just takes a ``Request``) so it can be wired into a FastAPI middleware or
called directly from a WebSocket handler -- WebSocket endpoints do NOT pass
through the HTTP middleware chain, so they must invoke ``check`` themselves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock

from fastapi import Request


@dataclass
class TokenBucket:
    """A single token bucket.

    ``capacity`` tokens can accumulate (the burst size); ``refill_rate`` tokens
    are added per second up to ``capacity``. ``consume(n)`` removes ``n`` tokens
    if available and returns ``True``, otherwise returns ``False`` without
    mutating the bucket.
    """

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def consume(self, n: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class RateLimiter:
    """Per-client rate limiter using the token-bucket algorithm.

    Parameters
    ----------
    requests_per_minute:
        Sustained request rate ceiling. Converted to a per-second refill rate.
    burst:
        Maximum number of requests allowed in a short window (bucket capacity).
    """

    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        self.capacity = burst
        self.buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()

    def _get_client_id(self, request: Request) -> str:
        # Use X-Forwarded-For (behind ingress) or client host.
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, request: Request) -> bool:
        """Return ``True`` if the request is allowed, ``False`` if rate-limited."""
        client_id = self._get_client_id(request)
        with self._lock:
            if client_id not in self.buckets:
                self.buckets[client_id] = TokenBucket(
                    capacity=self.capacity,
                    refill_rate=self.refill_rate,
                )
            return self.buckets[client_id].consume(1)

    def cleanup_stale(self, max_age_seconds: int = 3600) -> None:
        """Remove buckets that haven't been used recently."""
        now = time.monotonic()
        with self._lock:
            stale = [
                cid
                for cid, bucket in self.buckets.items()
                if now - bucket.last_refill > max_age_seconds
            ]
            for cid in stale:
                del self.buckets[cid]
