"""HTTP transport layer for the ZeroDataModel SDK.

Wraps :mod:`httpx` so the :class:`~zero_data_model_client.client.ZeroDataModelClient`
stays focused on endpoint methods. A custom ``transport`` may be injected
(used by the test-suite via :class:`httpx.MockTransport`) so the client can be
exercised without a live server.
"""
from __future__ import annotations

from typing import Any

import httpx

from ._config import ClientConfig


def create_client(
    config: ClientConfig,
    headers: dict[str, str],
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Build an :class:`httpx.Client` for the given configuration.

    Args:
        config: Resolved client configuration.
        headers: Default headers (including ``X-API-Key`` when set).
        transport: Optional custom transport (e.g. ``httpx.MockTransport``).
    """
    kwargs: dict[str, Any] = {
        "base_url": config.base_url,
        "timeout": config.timeout,
        "headers": headers,
    }
    if transport is not None:
        kwargs["transport"] = transport
    return httpx.Client(**kwargs)
