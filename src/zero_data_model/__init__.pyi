from __future__ import annotations

from . import capabilities, hardware
from .base import CognitiveModule, Prediction, Signal
from .mcp_server import ZeroDataMCPServer
from .model import ZeroDataModel

__version__: str

# ``ZeroDataMCPServer`` is conditionally available: it is set to ``None`` at
# runtime when the optional ``mcp`` package (or any of its dependencies) is
# missing (see ``__init__.py``).
ZeroDataMCPServer: type[ZeroDataMCPServer] | None

__all__ = [
    "ZeroDataModel",
    "ZeroDataMCPServer",
    "Signal",
    "Prediction",
    "CognitiveModule",
    "capabilities",
    "hardware",
    "__version__",
]
