# src/zero_data_model/__init__.py
"""Zero-Data Model: A self-sufficient cognitive system."""
__version__ = "0.2.0"

# Lazy imports of the public surface. Importing the full stack at package
# import time would pull in optional accelerators; we defer to the actual
# submodules so `import zero_data_model` stays cheap and side-effect free.
from . import capabilities, hardware
from .base import CognitiveModule, Prediction, Signal
from .model import ZeroDataModel

# MCP server wrapper. Imported gracefully so the package keeps working even if
# the (optional) mcp package is absent -- the module itself degrades cleanly.
try:
    from .mcp_server import ZeroDataMCPServer
except Exception:  # pragma: no cover - defensive: mcp_server only needs numpy.
    ZeroDataMCPServer = None  # type: ignore[assignment, misc]

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
