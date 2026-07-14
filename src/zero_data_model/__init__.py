# src/zero_data_model/__init__.py
"""Zero-Data Model: A self-sufficient cognitive system."""
__version__ = "0.2.0"

# Lazy imports of the public surface. Importing the full stack at package
# import time would pull in optional accelerators; we defer to the actual
# submodules so `import zero_data_model` stays cheap and side-effect free.
from .base import Signal, Prediction, CognitiveModule
from .model import ZeroDataModel
from . import capabilities
from . import hardware

__all__ = [
    "ZeroDataModel",
    "Signal",
    "Prediction",
    "CognitiveModule",
    "capabilities",
    "hardware",
    "__version__",
]
