"""Repo-root pytest configuration.

Ensures the ``src`` layout is importable during test collection. This is
normally handled by ``[tool.pytest.ini_options] pythonpath = ["src"]`` in
pyproject.toml (pytest >= 7) and by the editable install
(``pip install -e .``). This conftest is a defensive fallback for older pytest
versions or environments that run tests without an install.

An autouse fixture re-seeds the global numpy RNG before every test so the
30+ tests across 9 files that call ``np.random.randn(...)`` without an
explicit seed become deterministic and stop flaking under load.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(autouse=True)
def _seed_numpy() -> None:
    """Pin the global ``np.random`` RNG before every test.

    Re-seeding before each test eliminates flakiness in tests that rely on
    the legacy global RNG (``np.random.randn``, ``np.random.choice``, ...) by
    giving them a fresh, deterministic stream every time. Tests that prefer
    their own ``np.random.default_rng(seed)`` are unaffected because the
    ``default_rng`` generator is independent of the global state.
    """
    np.random.seed(42)
    yield
