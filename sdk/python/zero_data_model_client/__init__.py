"""ZeroDataModel Python SDK.

A thin synchronous client for the ZeroDataModel REST API covering the core
cognitive endpoints and the phase-7 cognitive-upgrade modules.
"""
from __future__ import annotations

from .client import ZeroDataModelClient
from .exceptions import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    ZeroDataModelError,
)

__version__ = "1.0.0"

__all__ = [
    "ZeroDataModelClient",
    "ZeroDataModelError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "ValidationError",
    "ServerError",
]
