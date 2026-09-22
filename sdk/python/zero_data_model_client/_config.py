"""Configuration for the ZeroDataModel SDK client."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClientConfig:
    """Client configuration.

    Attributes:
        base_url: Base URL of the ZeroDataModel REST API (no trailing slash).
            Defaults to the local ``/v1`` versioned endpoint; callers talking
            to a remote host should pass e.g. ``https://api.example.com/v1``.
        api_key: Optional API key sent as the ``X-API-Key`` header.
        timeout: Per-request timeout in seconds.
        max_retries: Maximum number of retries for retriable failures.
        retry_backoff: Base backoff (seconds) between retries.
    """

    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 0.5
