"""Runnable entry point: ``python -m zero_data_model``.

Prints the installed version and runs a quick zero-data demo: one
``think()`` cycle, a zero-shot text classification, and a hardware
backend report. Use ``--version`` to print only the version string.
Pass ``--mcp`` to start the MCP server instead of the mini-demo.

The ``emergence`` subcommand group exposes the causal-emergence engine
(spec §2.3) on the command line: ``perceive``, ``causal``, ``trajectory``,
``cycle``. All emergence calls go through ``ZeroDataModel``'s facade
methods so the model-level ``self._lock`` serializes them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from typing import Any


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


# ------------------------------------------------------------------ #
# Helpers for the ``emergence`` subcommand group
# ------------------------------------------------------------------ #

# Deterministic model config for CLI reproducibility (spec constraint 4).
_CLI_MODEL_DIM: int = 16
_CLI_MODEL_SEED: int = 42


def _build_model():
    """Construct a deterministic ``ZeroDataModel`` for CLI use."""
    from .model import ZeroDataModel

    return ZeroDataModel(dim=_CLI_MODEL_DIM, seed=_CLI_MODEL_SEED)


def _load_observation(path: str):
    """Load a 2D ndarray observation from .npz / .npy / .csv.

    For ``.npz`` archives with multiple arrays, the alphabetically-first
    key is loaded. 1D inputs (e.g. a single-row CSV) are returned as-is;
    downstream engine methods decide whether to accept them.
    """
    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(f"observation file not found: {path}")

    if path.endswith(".npz"):
        with np.load(path) as data:
            keys = sorted(data.files)
            if not keys:
                raise ValueError(f"{path}: empty .npz archive")
            arr = data[keys[0]]
    elif path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".csv"):
        arr = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(
            f"unsupported observation format: {path} "
            "(expected .npz / .npy / .csv)"
        )
    return np.ascontiguousarray(arr, dtype=float)


def _load_vector(arg: str):
    """Load a 1D vector from an inline JSON string or a file path.

    Inline JSON: ``"[0.0, 1.0, 2.0]"``. Files: ``.json`` / ``.npy`` /
    ``.npz`` / ``.csv`` (single-row CSVs collapse to 1D automatically).
    Raises ``ValueError`` for non-1D inputs.
    """
    import numpy as np

    if os.path.exists(arg):
        if arg.endswith(".json"):
            with open(arg) as f:
                data = json.load(f)
        elif arg.endswith(".npy"):
            data = np.load(arg)
        elif arg.endswith(".npz"):
            with np.load(arg) as f:
                keys = sorted(f.files)
                if not keys:
                    raise ValueError(f"{arg}: empty .npz archive")
                data = f[keys[0]]
        elif arg.endswith(".csv"):
            data = np.loadtxt(arg, delimiter=",")
        else:
            # Fall back to JSON parse for unknown extensions.
            with open(arg) as f:
                data = json.load(f)
    else:
        data = json.loads(arg)

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"expected 1D vector, got {arr.ndim}D shape {tuple(arr.shape)}"
        )
    return arr


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types.

    Non-finite floats (NaN / +Inf / -Inf) are mapped to ``None`` so the
    emitted JSON stays valid (``json.dumps`` would otherwise emit bare
    ``NaN`` / ``Infinity`` tokens that break strict parsers).
    """
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _emit_json(payload: dict) -> int:
    """Print ``payload`` as a single-line JSON document to stdout."""
    print(json.dumps(_to_jsonable(payload)))
    return 0


# ------------------------------------------------------------------ #
# emergence subcommand handlers
# ------------------------------------------------------------------ #

def _run_emergence_perceive(args: argparse.Namespace) -> int:
    """``emergence perceive <file>``: topological invariants of a point cloud."""
    data = _load_observation(args.file)
    model = _build_model()
    result = model.perceive_topology(data)
    payload = {
        "betti_numbers": result.get("betti_numbers", []),
        "persistence_entropy": result.get("persistence_entropy", 0.0),
        "euler_characteristic": result.get("euler_characteristic", 0),
        "n_points": result.get("n_points", 0),
        "max_eps": result.get("max_eps", 0.0),
    }
    if args.full_diagram:
        # persistence_diagram can be long; only emit on explicit request.
        payload["persistence_diagram"] = result.get("persistence_diagram", [])
    return _emit_json(payload)


def _run_emergence_causal(args: argparse.Namespace) -> int:
    """``emergence causal <file>``: discover causal DAG from observations."""
    data = _load_observation(args.file)
    model = _build_model()
    result = model.discover_causal_dynamics(data)
    payload = {
        "adjacency": result.get("adjacency", []),
        "edges": result.get("edges", []),
        "n_edges": result.get("n_edges", 0),
        "is_acyclic": result.get("is_acyclic", True),
        "method": result.get("method", "unknown"),
    }
    return _emit_json(payload)


def _run_emergence_trajectory(args: argparse.Namespace) -> int:
    """``emergence trajectory <start> <end>``: damped least-action path."""
    start = _load_vector(args.start)
    end = _load_vector(args.end)
    constraints: dict | None = None
    if args.obstacles:
        obstacles = _load_observation(args.obstacles)
        constraints = {"obstacles": obstacles, "margin": 0.1}
    model = _build_model()
    result = model.generate_trajectory(
        start, end, n_steps=args.steps, constraints=constraints
    )
    payload = {
        "trajectory": result.get("trajectory", []),
        "action": result.get("action", 0.0),
        "converged": result.get("converged", False),
        "iterations": result.get("iterations", 0),
        "obstacle_violations": result.get("obstacle_violations", 0),
    }
    return _emit_json(payload)


def _run_emergence_cycle(args: argparse.Namespace) -> int:
    """``emergence cycle <file>``: full recursive emergence loop."""
    data = _load_observation(args.file)
    model = _build_model()
    result = model.emergence_cycle(data)
    return _emit_json(result)


# ------------------------------------------------------------------ #
# argparse wiring
# ------------------------------------------------------------------ #

def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser, including ``emergence`` subcommands."""
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

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        help="Available subcommands (run with `<command> --help` for details).",
    )

    emergence_parser = subparsers.add_parser(
        "emergence",
        help="Causal emergence engine (perceive / causal / trajectory / cycle).",
        description="Causal emergence engine — topological perception, causal "
        "discovery, counterfactual trajectories, and the full recursive "
        "emergence loop. All commands route through ZeroDataModel's facade "
        "methods so they serialize under the model-level lock.",
    )
    emergence_sub = emergence_parser.add_subparsers(
        dest="subcommand",
        required=True,
        metavar="<subcommand>",
        help="perceive | causal | trajectory | cycle",
    )

    # emergence perceive <file>
    perceive_parser = emergence_sub.add_parser(
        "perceive",
        help="Perceive topological invariants of a point cloud.",
        description="Compute persistent homology (betti numbers, persistence "
        "entropy, Euler characteristic) of the input point cloud via the "
        "Vietoris-Rips filtration.",
    )
    perceive_parser.add_argument(
        "file", help="Observation file (.npz / .npy / .csv)."
    )
    perceive_parser.add_argument(
        "--full-diagram",
        action="store_true",
        help="Include the (potentially long) persistence_diagram in the output.",
    )
    perceive_parser.set_defaults(func=_run_emergence_perceive)

    # emergence causal <file>
    causal_parser = emergence_sub.add_parser(
        "causal",
        help="Discover causal DAG from observational data.",
        description="Discover a directed acyclic graph (PC / LiNGAM / "
        "correlation) from multivariate observational data.",
    )
    causal_parser.add_argument(
        "file", help="Observation file (.npz / .npy / .csv), shape (n_samples, n_features)."
    )
    causal_parser.set_defaults(func=_run_emergence_causal)

    # emergence trajectory <start> <end>
    trajectory_parser = emergence_sub.add_parser(
        "trajectory",
        help="Generate a damped least-action trajectory between two states.",
        description="Solve a discretized Euler-Lagrange boundary value problem "
        "with damping to connect start_state to end_state.",
    )
    trajectory_parser.add_argument(
        "start",
        help="Start state (inline JSON like '[0.0, 1.0, 2.0]' or a .json / "
        ".npy / .npz / .csv file path).",
    )
    trajectory_parser.add_argument(
        "end",
        help="End state (same format as <start>).",
    )
    trajectory_parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Number of interior steps (trajectory has steps+1 rows). Default: 32.",
    )
    trajectory_parser.add_argument(
        "--obstacles",
        default=None,
        help="Optional obstacles file (.npz / .npy / .csv), shape (K, dim). "
        "Sets constraints={'obstacles': <array>, 'margin': 0.1}.",
    )
    trajectory_parser.set_defaults(func=_run_emergence_trajectory)

    # emergence cycle <file>
    cycle_parser = emergence_sub.add_parser(
        "cycle",
        help="Run the full recursive emergence loop on one observation.",
        description="Run perception -> causal discovery -> counterfactual -> "
        "posterior -> memory -> emergence_score. Input must be 2D with "
        "n_samples >= 2 and n_features >= 2.",
    )
    cycle_parser.add_argument(
        "file", help="Observation file (.npz / .npy / .csv), shape (n_samples, n_features)."
    )
    cycle_parser.set_defaults(func=_run_emergence_cycle)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        return _print_version()
    if args.mcp:
        return _run_mcp()

    if getattr(args, "command", None) == "emergence":
        # ``func`` is set via ``set_defaults`` on each emergence subparser.
        try:
            return args.func(args)
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    return _run_demo()


if __name__ == "__main__":
    sys.exit(main())
