"""Exception hierarchy for the ZeroDataModel SDK.

All SDK errors derive from :class:`ZeroDataModelError`, so callers can catch
every SDK failure with a single ``except ZeroDataModelError`` block while still
being able to branch on the specific HTTP failure class.
"""
from __future__ import annotations


class ZeroDataModelError(Exception):
    """Base exception for all SDK errors."""


class AuthenticationError(ZeroDataModelError):
    """API key missing or invalid (HTTP 401/403)."""


class RateLimitError(ZeroDataModelError):
    """Rate limit exceeded (HTTP 429).

    Attributes:
        retry_after: Suggested wait time in seconds before retrying.
    """

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class NotFoundError(ZeroDataModelError):
    """Resource not found (HTTP 404)."""


class ValidationError(ZeroDataModelError):
    """Request validation failed (HTTP 422)."""


class ServerError(ZeroDataModelError):
    """Server error (HTTP 5xx)."""
