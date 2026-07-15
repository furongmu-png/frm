from __future__ import annotations

from .model import ZeroDataModel

def set_persistence_root(path: str) -> None:
    """Override the persistence root at runtime (used by tests)."""
    ...


def get_persistence_root() -> str:
    """Return the current persistence root (resolved absolute path)."""
    ...


class ModelSerializer:
    """Serialize a :class:`ZeroDataModel` to a directory of npz + json."""

    @staticmethod
    def save(model: ZeroDataModel, path: str) -> None:
        """Save model state to a directory: arrays.npz + config.json.

        ``path`` must be a *relative* name under the persistence root.
        """
        ...

    @staticmethod
    def load(path: str) -> ZeroDataModel:
        """Reconstruct a :class:`ZeroDataModel` from a saved directory.

        ``path`` must be a *relative* name under the persistence root.
        """
        ...
