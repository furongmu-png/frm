"""Caching utilities for ZeroDataModel API.

Provides an LRU cache with TTL (time-to-live) so that infrequently-changing
endpoints (e.g. ``/architect/stats``, knowledge-graph shape) can be served
from cache instead of recomputing on every request.

The standard ``functools.lru_cache`` has no expiry — once an entry is cached
it stays until evicted by LRU replacement. For endpoints whose underlying
data changes slowly but does change (e.g. the architect evaluates every
``eval_interval`` cycles), a TTL-bounded cache keeps the data fresh without
the cost of recomputing on every call.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def timed_lru_cache(maxsize: int = 128, ttl: int = 300):
    """
    LRU cache with TTL (time-to-live) in seconds.
    Items expire after ttl seconds, even if not evicted by LRU.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cached = functools.lru_cache(maxsize=maxsize)(func)
        expiry: dict[Any, float] = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            now = time.monotonic()
            if key in expiry and now > expiry[key]:
                cached.cache_clear()
                # Re-cache only this key
                expiry.clear()
            expiry[key] = now + ttl
            return cached(*args, **kwargs)

        wrapper.cache_clear = cached.cache_clear  # type: ignore
        wrapper.cache_info = cached.cache_info  # type: ignore
        return wrapper

    return decorator


__all__: list[str] = ["timed_lru_cache"]
