"""Repo-root pytest configuration.

Ensures the ``src`` layout is importable during test collection. This is
normally handled by ``[tool.pytest.ini_options] pythonpath = ["src"]`` in
pyproject.toml (pytest >= 7) and by the editable install
(``pip install -e .``). This conftest is a defensive fallback for older pytest
versions or environments that run tests without an install.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
