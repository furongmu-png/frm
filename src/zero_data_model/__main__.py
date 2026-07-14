"""Runnable entry point: ``python -m zero_data_model``.

Prints the installed version and runs a quick zero-data demo: one
``think()`` cycle, a zero-shot text classification, and a hardware
backend report. Use ``--version`` to print only the version string.
Pass ``--mcp`` to start the MCP server instead of the mini-demo.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _print_version() -> int:
    from . import __version__

    print(__version__)
    return 0


def _run_mcp() -> int:
    # Imported lazily so ``--version`` does not pay the import cost.
    from .mcp_server import ZeroDataMCPServer

    ZeroDataMCPServer().run()
    return 0


def _run_demo() -> int:
    # Imported lazily so ``--version`` does not pay the import cost.
    import numpy as np

    from . import __version__
    from .model import ZeroDataModel

    print("=" * 60)
    print(f"  Zero-Data Model v{__version__}")
    print("  A self-sufficient cognitive system — no training data")
    print("=" * 60)

    model = ZeroDataModel(dim=32)

    print("\n[1] Hardware acceleration report")
    hw = model.hardware_info
    print(f"    array backend:   {hw.get('array_backend')}")
    print(f"    gpu available:   {hw.get('gpu')}")
    print(f"    quantum backend: {hw.get('quantum_backend')}")
    print(f"    annealer jit:    {hw.get('annealer_jit')}")
    print(f"    parallel:        {hw.get('backend')} ({hw.get('n_workers')} workers)")

    print("\n[2] One self-generated thought cycle (no input)")
    signal = model.think()
    cycle = signal.metadata.get("cycle")
    reflection = signal.metadata.get("self_reflection", {})
    print(f"    cycle:          {cycle}")
    print(f"    output norm:    {float(np.linalg.norm(signal.data)):.4f}")
    print(
        f"    self-confidence: {float(reflection.get('self_confidence', 0.0)):.4f}"
    )

    print("\n[3] Zero-shot text classification")
    sample = "the algorithm computes the network"
    topic, confidence = model.classify_text(sample)
    print(f"    input:      {sample!r}")
    print(f"    topic:      {topic!r}")
    print(f"    confidence: {confidence:.4f}")

    print("\n" + "=" * 60)
    print("  Mini-demo complete — no external data was used.")
    print("=" * 60)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zero_data_model",
        description="Zero-Data Model: a self-sufficient cognitive system.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed zero-data-model version and exit.",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Start the MCP server exposing the model's tools to AI agents "
        "(requires the optional `mcp` package for live serving).",
    )
    args = parser.parse_args(argv)

    if args.version:
        return _print_version()
    if args.mcp:
        return _run_mcp()
    return _run_demo()


if __name__ == "__main__":
    sys.exit(main())
