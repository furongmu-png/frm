from __future__ import annotations

from .model import ZeroDataModel

class ModelSerializer:
    """Serialize a :class:`ZeroDataModel` to a directory of npz + json."""

    @staticmethod
    def save(model: ZeroDataModel, path: str) -> None:
        """Save model state to a directory: arrays.npz + config.json."""
        ...

    @staticmethod
    def load(path: str) -> ZeroDataModel:
        """Reconstruct a :class:`ZeroDataModel` from a saved directory."""
        ...
