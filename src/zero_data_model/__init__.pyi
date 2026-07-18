from __future__ import annotations

from . import capabilities, hardware
from .base import CognitiveModule, Prediction, Signal
from .mcp_server import ZeroDataMCPServer as _ZeroDataMCPServer
from .model import ZeroDataModel

__version__: str

# Round-10 audit R10-C-005: the previous stub imported ``ZeroDataMCPServer``
# as a class (line 5) and then redeclared the same name as a runtime
# attribute of type ``type[ZeroDataMCPServer] | None`` (line 13). Type
# checkers see the same name bound twice with different meanings: as a
# TYPE (referring to the class) and as a VALUE (the runtime attribute
# that may be ``None`` when the optional ``mcp`` package is missing).
# The runtime resolves this at import time via a try/except that
# conditionally rebinds the name to ``None``, but the stub cannot express
# that binding under one name. Aliasing the class import to
# ``_ZeroDataMCPServer`` lets the type annotation below use the class as
# a TYPE while the runtime attribute ``ZeroDataMCPServer`` is clearly a
# VALUE that may be the class or ``None`` — matching ``__init__.py``'s
# try/except exactly.
ZeroDataMCPServer: type[_ZeroDataMCPServer] | None

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
