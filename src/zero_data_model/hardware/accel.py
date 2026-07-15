"""GPU/CPU array abstraction.

Exports ``xp`` -- the array module to use for numerics. It is CuPy when a CUDA
GPU is available, otherwise NumPy. Code written against ``xp`` runs unchanged
on both backends. Use ``to_gpu`` / ``to_cpu`` / ``asnumpy`` to move arrays
between devices when interoperating with libraries that require a specific
device.
"""

from __future__ import annotations

from typing import Any

import numpy as np

has_gpu = False
_backend_name = "numpy"

try:  # pragma: no cover - environment dependent
    import cupy as _cupy  # type: ignore

    if _cupy.cuda.runtime.getDeviceCount() > 0:
        xp = _cupy
        has_gpu = True
        _backend_name = "cupy"
    else:
        xp = np
except ImportError:
    xp = np


def backend_name() -> str:
    """Return the name of the active array backend ('cupy' or 'numpy')."""
    return _backend_name


def to_gpu(array: np.ndarray) -> Any:
    """Move a NumPy array to the GPU device (no-op when no GPU)."""
    if has_gpu:
        return _cupy.asarray(array)
    return array


def to_cpu(array: np.ndarray) -> Any:
    """Move a GPU array back to a NumPy array on the host (no-op on CPU)."""
    if has_gpu and hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def asnumpy(array) -> np.ndarray:
    """Coerce any backend array into a host NumPy array."""
    if has_gpu and hasattr(array, "get"):
        return np.asarray(array.get())
    return np.asarray(array)
