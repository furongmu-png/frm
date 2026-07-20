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
    """Load a 2D ndarray observation from .json / .npz / .npy / .csv.

    For ``.npz`` archives with multiple arrays, the alphabetically-first
    key is loaded. 1D inputs (e.g. a single-row CSV) are returned as-is;
    downstream engine methods decide whether to accept them.

    V4-NEW-L007 (v4.2): added ``.json`` support to match ``_load_vector``.
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
    elif path.endswith(".json"):
        with open(path) as f:
            arr = np.asarray(json.load(f), dtype=float)
    else:
        raise ValueError(
            f"unsupported observation format: {path} "
            "(expected .json / .npz / .npy / .csv)"
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
        # V4-NEW-L001 (v4.2): respect --margin instead of hardcoding 0.1.
        constraints = {"obstacles": obstacles}
        if args.margin is not None:
            constraints["margin"] = args.margin
    model = _build_model()
    result = model.generate_trajectory(
        start, end, n_steps=args.steps, constraints=constraints
    )
    payload = {
        "trajectory": result.get("trajectory", []),
        "action": result.get("action", 0.0),
        "converged": result.get("converged", False),
        "iterations": result.get("iterations", 0),
    }
    # V4-NEW-L002 (v4.2): only include obstacle_violations when obstacles
    # were active, matching the engine's backward-compat contract (the key
    # is absent from the result dict when constraints is None / {}).
    if "obstacle_violations" in result:
        payload["obstacle_violations"] = result["obstacle_violations"]
    return _emit_json(payload)


def _run_emergence_cycle(args: argparse.Namespace) -> int:
    """``emergence cycle <file>``: full recursive emergence loop."""
    data = _load_observation(args.file)
    model = _build_model()
    result = model.emergence_cycle(data)
    return _emit_json(result)


# ------------------------------------------------------------------ #
# Phase 6 — Memory / Planning / Multimodal / RL subcommand handlers
# ------------------------------------------------------------------ #

def _run_memory_encode(args: argparse.Namespace) -> int:
    obs = _load_observation(args.file)
    model = _build_model()
    result = model.encode_memory(obs, label=args.label)
    return _emit_json(result)


def _run_memory_retrieve(args: argparse.Namespace) -> int:
    query = _load_observation(args.file)
    model = _build_model()
    result = model.retrieve_memory(query, top_k=args.top_k)
    return _emit_json(result)


def _run_memory_consolidate(args: argparse.Namespace) -> int:
    model = _build_model()
    result = model.consolidate_memory()
    return _emit_json(result)


def _run_planning_trajectory(args: argparse.Namespace) -> int:
    start = _load_vector(args.start)
    end = _load_vector(args.end)
    obstacles = _load_observation(args.obstacles) if args.obstacles else None
    model = _build_model()
    result = model.plan_trajectory(
        start, end, obstacles=obstacles, n_steps=args.steps, margin=args.margin
    )
    return _emit_json(result)


def _run_planning_decompose(args: argparse.Namespace) -> int:
    model = _build_model()
    result = model.decompose_goal(args.goal, max_depth=args.max_depth)
    return _emit_json(result)


def _run_planning_sequence(args: argparse.Namespace) -> int:
    adjacency = _load_observation(args.file)
    model = _build_model()
    result = model.sequence_actions(adjacency)
    return _emit_json(result)


def _run_multimodal_align(args: argparse.Namespace) -> int:
    a = _load_observation(args.file_a)
    b = _load_observation(args.file_b)
    model = _build_model()
    fit = model.fit_cross_modal(a, b)
    # Align a single sample (the first row) — align() expects a 1D vector.
    sample = a[0] if a.ndim == 2 else a
    aligned = model.align_cross_modal(sample, source="a")
    return _emit_json({"fit": fit, "aligned": aligned})


def _run_multimodal_fuse(args: argparse.Namespace) -> int:
    a = _load_observation(args.file_a)
    b = _load_observation(args.file_b)
    model = _build_model()
    result = model.fuse_modalities([a, b], strategy=args.strategy)
    return _emit_json(result)


def _run_multimodal_contrastive(args: argparse.Namespace) -> int:
    a = _load_observation(args.file_a)
    b = _load_observation(args.file_b)
    model = _build_model()
    result = model.contrastive_loss(a, b)
    return _emit_json(result)


# ------------------------------------------------------------------ #
# Phase 7 — Audio CLI handlers.
# ------------------------------------------------------------------ #


def _run_audio_encode(args: argparse.Namespace) -> int:
    signal = _load_observation(args.file)
    model = _build_model()
    result = model.encode_audio(signal, sample_rate=int(args.sample_rate))
    return _emit_json({"embedding": result})


def _run_audio_detect_onsets(args: argparse.Namespace) -> int:
    signal = _load_observation(args.file)
    model = _build_model()
    result = model.detect_onsets(signal, sample_rate=int(args.sample_rate))
    return _emit_json(result)


def _run_audio_detect_pitch(args: argparse.Namespace) -> int:
    signal = _load_observation(args.file)
    model = _build_model()
    result = model.detect_pitch(signal, sample_rate=int(args.sample_rate))
    return _emit_json(result)


def _run_audio_classify(args: argparse.Namespace) -> int:
    signal = _load_observation(args.file)
    model = _build_model()
    result = model.classify_audio(signal, sample_rate=int(args.sample_rate))
    return _emit_json(result)


def _run_audio_segment_speech(args: argparse.Namespace) -> int:
    signal = _load_observation(args.file)
    model = _build_model()
    result = model.segment_speech(signal, sample_rate=int(args.sample_rate))
    return _emit_json(result)


def _run_audio_analyze_music(args: argparse.Namespace) -> int:
    signal = _load_observation(args.file)
    model = _build_model()
    result = model.analyze_music(signal, sample_rate=int(args.sample_rate))
    return _emit_json(result)


# ------------------------------------------------------------------ #
# Phase 7 — Graph CLI handlers.
# ------------------------------------------------------------------ #


def _run_graph_encode(args: argparse.Namespace) -> int:
    adjacency = _load_observation(args.file)
    model = _build_model()
    node_features = _load_observation(args.features) if args.features else None
    result = model.encode_graph(adjacency, node_features=node_features)
    return _emit_json({"embedding": result})


def _run_graph_communities(args: argparse.Namespace) -> int:
    adjacency = _load_observation(args.file)
    model = _build_model()
    result = model.detect_communities(adjacency)
    return _emit_json(result)


def _run_graph_path(args: argparse.Namespace) -> int:
    adjacency = _load_observation(args.file)
    model = _build_model()
    result = model.find_path(adjacency, int(args.source), int(args.target))
    return _emit_json(result)


def _run_graph_centrality(args: argparse.Namespace) -> int:
    adjacency = _load_observation(args.file)
    model = _build_model()
    result = model.analyze_centrality(adjacency)
    return _emit_json(result)


def _run_graph_isomorphism(args: argparse.Namespace) -> int:
    adj_a = _load_observation(args.file_a)
    adj_b = _load_observation(args.file_b)
    model = _build_model()
    result = model.check_isomorphism(adj_a, adj_b)
    return _emit_json(result)


def _run_graph_track_dynamic(args: argparse.Namespace) -> int:
    """Track community drift across graph snapshots.

    Each snapshot is loaded from a separate file listed via positional args
    or via --files (comma-separated). For simplicity, we accept multiple
    positional paths after the command.
    """
    # Reuse argparse REMAINDER for snapshot files.
    files = args.files
    if not files:
        print("error: at least two snapshot files are required", file=sys.stderr)
        return 1
    snapshots = [_load_observation(f) for f in files]
    model = _build_model()
    result = model.track_dynamic_graph(snapshots)
    return _emit_json(result)


def _run_graph_spanning(args: argparse.Namespace) -> int:
    adjacency = _load_observation(args.file)
    model = _build_model()
    result = model.extract_spanning_tree(adjacency)
    return _emit_json(result)


# ------------------------------------------------------------------ #
# Phase 7 — Robotics CLI handlers.
# ------------------------------------------------------------------ #


def _parse_obstacles_file(path: str) -> list:
    """Load obstacles from a JSON/CSV file.

    Each obstacle is a row ``[x, y, ..., radius]`` where the LAST column
    is the obstacle radius and the leading columns are its center
    coordinates. Returns a list of ``(np.ndarray center, float radius)``
    tuples matching the format expected by the robotics facade.
    """
    import numpy as np

    arr = _load_observation(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    obstacles: list[tuple[np.ndarray, float]] = []
    for row in arr:
        if row.size < 2:
            continue
        center = np.asarray(row[:-1], dtype=float)
        radius = float(row[-1])
        obstacles.append((center, radius))
    return obstacles


def _run_robotics_motion(args: argparse.Namespace) -> int:
    waypoints = _load_observation(args.file)
    model = _build_model()
    result = model.plan_motion(waypoints, n_steps=int(args.steps))
    return _emit_json(result)


def _run_robotics_forward(args: argparse.Namespace) -> int:
    joint_angles = _load_observation(args.file)
    model = _build_model()
    result = model.forward_kinematics(joint_angles)
    return _emit_json({"position": result})


def _run_robotics_inverse(args: argparse.Namespace) -> int:
    target = _load_observation(args.file)
    seed = _load_vector(args.seed) if args.seed else None
    model = _build_model()
    result = model.inverse_kinematics(target, seed=seed)
    return _emit_json(result)


def _run_robotics_fuse(args: argparse.Namespace) -> int:
    """Fuse multiple sensor measurements loaded from a 2D file.

    ``--variances`` accepts comma-separated floats (one per row of the
    input file). When omitted, equal variances (1.0) are used.
    """
    import numpy as np

    measurements_2d = _load_observation(args.file)
    if measurements_2d.ndim == 1:
        measurements_2d = measurements_2d.reshape(1, -1)
    measurements = [measurements_2d[i] for i in range(measurements_2d.shape[0])]
    if args.variances:
        variances = [float(v) for v in args.variances.split(",")]
    else:
        variances = [1.0] * len(measurements)
    model = _build_model()
    result = model.fuse_sensors(measurements, variances)
    return _emit_json({"fused": result})


def _run_robotics_kalman(args: argparse.Namespace) -> int:
    prior = _load_vector(args.prior)
    measurement = _load_vector(args.measurement)
    model = _build_model()
    result = model.update_kalman(
        prior, float(args.prior_var), measurement, float(args.meas_var)
    )
    return _emit_json(result)


def _run_robotics_gait(args: argparse.Namespace) -> int:
    model = _build_model()
    result = model.generate_gait(n_steps=int(args.steps), gait_type=args.gait)
    return _emit_json(result)


def _run_robotics_optimize(args: argparse.Namespace) -> int:
    trajectory = _load_observation(args.file)
    model = _build_model()
    result = model.optimize_trajectory(trajectory, n_iter=int(args.n_iter))
    return _emit_json(result)


def _run_robotics_collision(args: argparse.Namespace) -> int:
    """Check collision at a single position against a list of obstacles.

    The obstacles file is a 2D array ``[[x, y, ..., radius], ...]`` where
    the LAST column is the obstacle radius and the rest is its center.
    Position is parsed from the ``--position`` flag (comma-separated).
    """
    import numpy as np

    obstacles = _parse_obstacles_file(args.file)
    position = np.asarray(
        [float(v) for v in args.position.split(",")], dtype=float
    )
    model = _build_model()
    result = model.check_collision(obstacles, position, radius=float(args.radius))
    return _emit_json(result)


def _run_robotics_path_collision(args: argparse.Namespace) -> int:
    """Check collision along a path. Obstacles and path are each loaded
    from a separate JSON/CSV file."""
    obstacles = _parse_obstacles_file(args.obstacles)
    path = _load_observation(args.path)
    model = _build_model()
    result = model.check_path_collision(obstacles, path, radius=float(args.radius))
    return _emit_json(result)


def _run_robotics_mpc(args: argparse.Namespace) -> int:
    current_state = _load_vector(args.current)
    target_state = _load_vector(args.target)
    obstacles = None
    if args.obstacles:
        obstacles = _parse_obstacles_file(args.obstacles)
    model = _build_model()
    result = model.control_mpc(current_state, target_state, obstacles=obstacles)
    return _emit_json(result)


# ------------------------------------------------------------------ #
# Phase 7 — Time CLI handlers.
# ------------------------------------------------------------------ #


def _run_time_encode(args: argparse.Namespace) -> int:
    series = _load_observation(args.file)
    model = _build_model()
    result = model.encode_time_series(series)
    return _emit_json({"embedding": result})


def _run_time_seasonality(args: argparse.Namespace) -> int:
    series = _load_observation(args.file)
    model = _build_model()
    result = model.detect_seasonality(series)
    return _emit_json(result)


def _run_time_frequency(args: argparse.Namespace) -> int:
    series = _load_observation(args.file)
    model = _build_model()
    result = model.analyze_frequency(series)
    return _emit_json(result)


def _run_time_events(args: argparse.Namespace) -> int:
    timestamps = _load_observation(args.file)
    model = _build_model()
    result = model.analyze_event_timestamps(timestamps)
    return _emit_json(result)


def _run_time_anomaly(args: argparse.Namespace) -> int:
    timestamps = _load_observation(args.file)
    model = _build_model()
    result = model.detect_anomalous_timing(timestamps)
    return _emit_json(result)


def _run_time_cycle(args: argparse.Namespace) -> int:
    series = _load_observation(args.file)
    period = int(args.period) if args.period is not None else None
    model = _build_model()
    result = model.track_cycle_phase(series, period=period)
    return _emit_json(result)


def _run_time_forecast(args: argparse.Namespace) -> int:
    series = _load_observation(args.file)
    model = _build_model()
    result = model.score_forecastability(series)
    return _emit_json(result)


# ------------------------------------------------------------------ #
# Phase 7 — Code CLI handlers.
# ------------------------------------------------------------------ #


def _load_source(path: str) -> str:
    """Read a Python source file as text.

    Accepts any text file (typically ``.py``); raises ``FileNotFoundError``
    if the path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"source file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _run_code_encode(args: argparse.Namespace) -> int:
    source = _load_source(args.file)
    model = _build_model()
    emb = model.encode_code(source)
    return _emit_json({"embedding": emb})


def _run_code_ast(args: argparse.Namespace) -> int:
    source = _load_source(args.file)
    model = _build_model()
    result = model.analyze_ast(source)
    return _emit_json(result)


def _run_code_compare(args: argparse.Namespace) -> int:
    source_a = _load_source(args.file_a)
    source_b = _load_source(args.file_b)
    model = _build_model()
    result = model.compare_code(source_a, source_b)
    return _emit_json(result)


def _run_code_defects(args: argparse.Namespace) -> int:
    source = _load_source(args.file)
    model = _build_model()
    result = model.detect_code_defects(source)
    return _emit_json(result)


def _run_code_control_flow(args: argparse.Namespace) -> int:
    source = _load_source(args.file)
    model = _build_model()
    result = model.analyze_control_flow(source)
    return _emit_json(result)


def _run_code_style(args: argparse.Namespace) -> int:
    source = _load_source(args.file)
    model = _build_model()
    result = model.analyze_code_style(source)
    return _emit_json(result)


def _run_code_dependencies(args: argparse.Namespace) -> int:
    source = _load_source(args.file)
    model = _build_model()
    result = model.build_dependency_graph(source)
    return _emit_json(result)


# ------------------------------------------------------------------ #
# Phase 7 — Reasoning CLI handlers.
#
# Unlike other Phase 7 domains which are stateless queries, the
# Reasoning facade has 4 stateful `add_*` methods that mutate engine
# state (propositional facts/rules, defeasible defaults, causal links).
# CLI subprocesses don't persist that state between invocations, so the
# CLI design is "compositional": each stateful query subcommand accepts
# a JSON config file that supplies all the facts / rules / defaults /
# causal links upfront, applies them to a fresh model, then runs the
# query and emits the result.
# ------------------------------------------------------------------ #


def _load_json_file(path: str) -> dict | list:
    """Read a JSON file as a Python object (dict or list)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_reasoning_prop_infer(args: argparse.Namespace) -> int:
    """Run propositional inference from a config file.

    Config schema (JSON):
        {
            "facts": {"proposition": true/false, ...},
            "rules": [["antecedent", "consequent"], ...],
            "negation_rules": [["antecedent", "consequent"], ...]
        }
    All three keys are optional; missing keys default to empty.
    """
    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    facts = config.get("facts", {}) or {}
    rules = config.get("rules", []) or []
    neg_rules = config.get("negation_rules", []) or []
    model = _build_model()
    for prop, val in facts.items():
        model.add_logical_fact(str(prop), bool(val))
    for r in rules:
        if len(r) >= 2:
            model.add_logical_rule(str(r[0]), str(r[1]))
    for r in neg_rules:
        if len(r) >= 2:
            model.reasoning_propositional.add_negation_rule(str(r[0]), str(r[1]))
    result = model.infer_logical()
    return _emit_json(result)


def _run_reasoning_syllogism(args: argparse.Namespace) -> int:
    """Run a categorical syllogism (Barbara/Celarent/Darii/Ferio).

    Args are JSON files containing tuples of (form, subject, predicate):
        major: ["A", "human", "mortal"]
        minor: ["A", "socrates", "human"]
    """
    major = tuple(_load_json_file(args.major_file))
    minor = tuple(_load_json_file(args.minor_file))
    model = _build_model()
    result = model.syllogism(major, minor)
    return _emit_json(result)


def _run_reasoning_induct(args: argparse.Namespace) -> int:
    """Induce a rule from labeled examples.

    Config schema (JSON):
        {
            "examples": [{"color": "red"}, ...],
            "labels": [true, false, ...]
        }
    """
    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    examples = config.get("examples", [])
    labels = config.get("labels", [])
    model = _build_model()
    result = model.induct_rule(examples, [bool(l) for l in labels])
    return _emit_json(result)


def _run_reasoning_analogize(args: argparse.Namespace) -> int:
    """Find a structural analogy between a source and target dict.

    Two JSON files: `source_file` and `target_file`, each containing an
    object mapping attribute names to values (numeric, string, etc.).
    """
    source = _load_json_file(args.source_file)
    target = _load_json_file(args.target_file)
    model = _build_model()
    result = model.analogize(source, target)
    return _emit_json(result)


def _run_reasoning_abduce(args: argparse.Namespace) -> int:
    """Pick the best explanation for an observation.

    Config schema (JSON):
        {
            "observation": "patient has fever and cough",
            "hypotheses": ["flu", "cold", "allergy"],
            "priors": [0.3, 0.4, 0.3]  // optional
        }
    """
    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    observation = str(config.get("observation", ""))
    hypotheses = config.get("hypotheses", []) or []
    priors = config.get("priors")
    model = _build_model()
    result = model.abduce(observation, hypotheses, priors=priors)
    return _emit_json(result)


def _run_reasoning_defaults(args: argparse.Namespace) -> int:
    """Apply defeasible default rules to a set of facts.

    Config schema (JSON):
        {
            "defaults": [
                {"rule": ["bird", "flies", true],
                 "exception": ["penguin"]},
                ...
            ],
            "facts": {"bird": true, "penguin": false}
        }
    Both keys are optional.
    """
    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    defaults = config.get("defaults", []) or []
    facts = config.get("facts", {}) or {}
    model = _build_model()
    for d in defaults:
        if not isinstance(d, dict):
            continue
        rule = d.get("rule")
        exception = d.get("exception")
        if rule is None:
            continue
        rule_t = tuple(rule)
        exc_t = tuple(exception) if exception is not None else None
        model.add_default_rule(rule_t, exception=exc_t)
    result = model.conclude_defaults(facts)
    return _emit_json(result)


def _run_reasoning_causal(args: argparse.Namespace) -> int:
    """Trace a causal chain from a starting node.

    Config schema (JSON):
        {
            "links": [["a", "b"], ["b", "c"], ...],
            "start": "a",
            "max_depth": 5
        }
    `max_depth` is optional (default 5).
    """
    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    links = config.get("links", []) or []
    start = str(config.get("start", ""))
    max_depth = int(config.get("max_depth", 5))
    model = _build_model()
    for link in links:
        if len(link) >= 2:
            model.add_causal_link(str(link[0]), str(link[1]))
    result = model.trace_causal_chain(start, max_depth=max_depth)
    return _emit_json(result)


# ------------------------------------------------------------------ #
# Phase 7 — Causal CLI handlers.
# ------------------------------------------------------------------ #


def _run_causal_decision_tree(args: argparse.Namespace) -> int:
    """Fit an ID3-style decision tree to ``(features, labels)``.

    Config schema (JSON):
        {
            "features": [[1.0, 2.0], [3.0, 4.0], ...],  // 2D (n_samples, n_features)
            "labels":   ["a", "b", "a", ...]              // 1D (n_samples,)
        }
    """
    import numpy as np

    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    features = np.asarray(config.get("features", []), dtype=float)
    labels = np.asarray(config.get("labels", []))
    model = _build_model()
    result = model.fit_decision_tree(features, labels)
    return _emit_json(result)


def _run_causal_game(args: argparse.Namespace) -> int:
    """Analyze a 2-player normal-form game; find pure Nash equilibria.

    Config schema (JSON):
        {
            "payoff_a": [[3, 0], [5, 1]],   // 2D (n_rows, n_cols)
            "payoff_b": [[3, 5], [0, 1]]     // optional, same shape
        }
    ``payoff_b`` omitted => zero-sum game (``payoff_b = -payoff_a``).
    """
    import numpy as np

    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    payoff_a = np.asarray(config.get("payoff_a", []), dtype=float)
    payoff_b_raw = config.get("payoff_b")
    payoff_b = np.asarray(payoff_b_raw, dtype=float) if payoff_b_raw is not None else None
    model = _build_model()
    result = model.analyze_game(payoff_a, payoff_b)
    return _emit_json(result)


def _run_causal_counterfactual(args: argparse.Namespace) -> int:
    """Estimate the counterfactual outcome under ``do(X[index] = value)``.

    Config schema (JSON):
        {
            "observed": [1.0, 2.0, 3.0, 4.0],   // 1D
            "index": 1,                          // int
            "value": 9.0                         // float
        }
    """
    import numpy as np

    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    observed = np.asarray(config.get("observed", []), dtype=float)
    intervention = {
        "index": int(config.get("index", 0)),
        "value": float(config.get("value", 0.0)),
    }
    model = _build_model()
    result = model.counterfactual(observed, intervention)
    return _emit_json(result)


def _run_causal_bandit(args: argparse.Namespace) -> int:
    """Select the next bandit arm (epsilon-greedy + UCB1).

    Config schema (JSON):
        {
            "rewards_history": [
                [1.0, 0.5, 0.8],   // rewards observed for arm 0
                [0.2],              // rewards observed for arm 1
                []                  // arm 2 never pulled
            ]
        }
    """
    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    rewards_history = config.get("rewards_history", []) or []
    if not isinstance(rewards_history, list):
        raise ValueError("rewards_history must be a list of lists")
    rewards_history = [list(h) for h in rewards_history]
    model = _build_model()
    result = model.select_bandit_arm(rewards_history)
    return _emit_json(result)


def _run_causal_pomdp(args: argparse.Namespace) -> int:
    """Solve a (PO)MDP via value iteration.

    Config schema (JSON):
        {
            "transitions":   [[[...], ...], ...],  // 3D (S, A, S)
            "observations":  [[...], ...],         // 2D (S, O)
            "rewards":        [...]                 // 1D (S,) or 2D (S, A)
        }
    """
    import numpy as np

    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    transitions = np.asarray(config.get("transitions", []), dtype=float)
    observations = np.asarray(config.get("observations", []), dtype=float)
    rewards = np.asarray(config.get("rewards", []), dtype=float)
    model = _build_model()
    result = model.solve_pomdp(transitions, observations, rewards)
    return _emit_json(result)


def _run_causal_discover_graph(args: argparse.Namespace) -> int:
    """Discover a causal graph from observational data (PC-style).

    Config schema (JSON):
        {
            "data": [[...], ...],                 // 2D (n_samples, n_vars)
            "var_names": ["x", "y", "z"]           // optional, len = n_vars
        }
    """
    import numpy as np

    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    data = np.asarray(config.get("data", []), dtype=float)
    var_names = config.get("var_names")
    if var_names is not None:
        var_names = [str(n) for n in var_names]
    model = _build_model()
    result = model.discover_causal_graph(data, var_names=var_names)
    return _emit_json(result)


def _run_causal_intervene(args: argparse.Namespace) -> int:
    """Estimate the effect of ``do(X[intervention_var] = value)``.

    Config schema (JSON):
        {
            "data": [[...], ...],                 // 2D (n_samples, n_vars)
            "intervention_var": 1,                // int (column index)
            "intervention_value": 5.0             // float
        }
    """
    import numpy as np

    config = _load_json_file(args.file)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    data = np.asarray(config.get("data", []), dtype=float)
    intervention_var = int(config.get("intervention_var", 0))
    intervention_value = float(config.get("intervention_value", 0.0))
    model = _build_model()
    result = model.intervene(data, intervention_var, intervention_value)
    return _emit_json(result)


def _run_rl_step(args: argparse.Namespace) -> int:
    model = _build_model()
    result = model.step_mdp(int(args.state), int(args.action))
    return _emit_json(result)


def _run_rl_train_q(args: argparse.Namespace) -> int:
    model = _build_model()
    result = model.train_q_learner(
        n_episodes=args.episodes, max_steps_per_episode=args.max_steps
    )
    return _emit_json(result)


def _run_rl_search_mcts(args: argparse.Namespace) -> int:
    model = _build_model()
    result = model.search_rl_mcts(
        int(args.root_state),
        n_simulations=args.simulations,
        max_depth=args.depth,
    )
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
        help="Optional obstacles file (.json / .npz / .npy / .csv), shape "
        "(K, dim). Activates constraints avoidance.",
    )
    trajectory_parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Safety margin around obstacles (V4-NEW-L001). If omitted, the "
        "engine uses rules.differential_obstacle_margin (default 1e-3).",
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

    # ------------------------------------------------------------------ #
    # Phase 6 — Memory / Planning / Multimodal / RL subparsers.
    # ------------------------------------------------------------------ #
    memory_parser = subparsers.add_parser(
        "memory",
        help="Memory capabilities (encode / retrieve / consolidate).",
        description="Episodic + working memory facade.",
    )
    memory_sub = memory_parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>",
        help="encode | retrieve | consolidate",
    )
    mem_enc = memory_sub.add_parser("encode", help="Encode an observation.")
    mem_enc.add_argument("file", help="Observation file (.npz / .npy / .csv / .json).")
    mem_enc.add_argument("--label", default=None, help="Optional label for the memory.")
    mem_enc.set_defaults(func=_run_memory_encode)
    mem_ret = memory_sub.add_parser("retrieve", help="Retrieve top-k similar memories.")
    mem_ret.add_argument("file", help="Query file (.npz / .npy / .csv / .json).")
    mem_ret.add_argument("--top-k", type=int, default=5, help="Top-k (default 5).")
    mem_ret.set_defaults(func=_run_memory_retrieve)
    mem_con = memory_sub.add_parser("consolidate", help="Promote working → episodic.")
    mem_con.set_defaults(func=_run_memory_consolidate)

    planning_parser = subparsers.add_parser(
        "planning",
        help="Planning capabilities (trajectory / decompose / sequence).",
        description="Hierarchical / trajectory / action sequencing facade.",
    )
    planning_sub = planning_parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>",
        help="trajectory | decompose | sequence",
    )
    plan_traj = planning_sub.add_parser("trajectory", help="Plan a trajectory.")
    plan_traj.add_argument("start", help="Start state (inline JSON or file).")
    plan_traj.add_argument("end", help="End state (inline JSON or file).")
    plan_traj.add_argument("--steps", type=int, default=32)
    plan_traj.add_argument("--obstacles", default=None)
    plan_traj.add_argument("--margin", type=float, default=None)
    plan_traj.set_defaults(func=_run_planning_trajectory)
    plan_dec = planning_sub.add_parser("decompose", help="Decompose a goal.")
    plan_dec.add_argument("goal", help="Goal name.")
    plan_dec.add_argument("--max-depth", type=int, default=None)
    plan_dec.set_defaults(func=_run_planning_decompose)
    plan_seq = planning_sub.add_parser("sequence", help="Topologically sort actions.")
    plan_seq.add_argument("file", help="Adjacency matrix file (.npz / .npy / .csv).")
    plan_seq.set_defaults(func=_run_planning_sequence)

    multimodal_parser = subparsers.add_parser(
        "multimodal",
        help="Multimodal capabilities (align / fuse / contrastive).",
        description="Cross-modal alignment / fusion / contrastive learning facade.",
    )
    multimodal_sub = multimodal_parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>",
        help="align | fuse | contrastive",
    )
    mm_align = multimodal_sub.add_parser("align", help="CCA alignment of two modalities.")
    mm_align.add_argument("file_a", help="Observation file for modality A.")
    mm_align.add_argument("file_b", help="Observation file for modality B.")
    mm_align.set_defaults(func=_run_multimodal_align)
    mm_fuse = multimodal_sub.add_parser("fuse", help="Fuse two modality embeddings.")
    mm_fuse.add_argument("file_a")
    mm_fuse.add_argument("file_b")
    mm_fuse.add_argument("--strategy", default=None, help="mean | concat | weighted.")
    mm_fuse.set_defaults(func=_run_multimodal_fuse)
    mm_con = multimodal_sub.add_parser("contrastive", help="InfoNCE contrastive loss.")
    mm_con.add_argument("file_a")
    mm_con.add_argument("file_b")
    mm_con.set_defaults(func=_run_multimodal_contrastive)

    # ------------------------------------------------------------------
    # Phase 7 — Audio subparsers.
    # ------------------------------------------------------------------
    audio_parser = subparsers.add_parser(
        "audio",
        help="Audio capabilities (encode / onsets / pitch / classify / segment / music).",
        description="Audio encoder + onset/pitch detection + classifier + VAD + music analyzer.",
    )
    audio_sub = audio_parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>",
        help="encode | onsets | pitch | classify | segment | music",
    )
    au_enc = audio_sub.add_parser("encode", help="Encode a 1D audio signal into a dim-length vector.")
    au_enc.add_argument("file", help="Audio signal file (.npz / .npy / .csv / .json).")
    au_enc.add_argument("--sample-rate", type=int, default=16000)
    au_enc.set_defaults(func=_run_audio_encode)
    au_onsets = audio_sub.add_parser("onsets", help="Detect note/onset events via spectral flux.")
    au_onsets.add_argument("file", help="Audio signal file.")
    au_onsets.add_argument("--sample-rate", type=int, default=16000)
    au_onsets.set_defaults(func=_run_audio_detect_onsets)
    au_pitch = audio_sub.add_parser("pitch", help="Detect fundamental frequency via autocorrelation.")
    au_pitch.add_argument("file", help="Audio signal file.")
    au_pitch.add_argument("--sample-rate", type=int, default=16000)
    au_pitch.set_defaults(func=_run_audio_detect_pitch)
    au_class = audio_sub.add_parser("classify", help="Classify audio texture (speech/music/noise/silence).")
    au_class.add_argument("file", help="Audio signal file.")
    au_class.add_argument("--sample-rate", type=int, default=16000)
    au_class.set_defaults(func=_run_audio_classify)
    au_seg = audio_sub.add_parser("segment", help="Segment audio into speech/silence regions via VAD.")
    au_seg.add_argument("file", help="Audio signal file.")
    au_seg.add_argument("--sample-rate", type=int, default=16000)
    au_seg.set_defaults(func=_run_audio_segment_speech)
    au_music = audio_sub.add_parser("music", help="Analyze music for tempo and beats.")
    au_music.add_argument("file", help="Audio signal file.")
    au_music.add_argument("--sample-rate", type=int, default=16000)
    au_music.set_defaults(func=_run_audio_analyze_music)

    # ------------------------------------------------------------------
    # Phase 7 — Graph subparsers.
    # ------------------------------------------------------------------
    graph_parser = subparsers.add_parser(
        "graph",
        help="Graph capabilities (encode / communities / path / centrality / isomorphism / track / spanning).",
        description="Graph encoder + community detection + path + centrality + isomorphism + dynamic tracking + MST.",
    )
    graph_sub = graph_parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>",
        help="encode | communities | path | centrality | isomorphism | track | spanning",
    )
    gr_enc = graph_sub.add_parser("encode", help="Encode a graph into a dim-length vector.")
    gr_enc.add_argument("file", help="Adjacency matrix file (.npz / .npy / .csv / .json).")
    gr_enc.add_argument("--features", default=None, help="Optional node features file.")
    gr_enc.set_defaults(func=_run_graph_encode)
    gr_com = graph_sub.add_parser("communities", help="Detect communities via modularity.")
    gr_com.add_argument("file", help="Adjacency matrix file.")
    gr_com.set_defaults(func=_run_graph_communities)
    gr_path = graph_sub.add_parser("path", help="Find shortest path via Dijkstra.")
    gr_path.add_argument("file", help="Adjacency matrix file.")
    gr_path.add_argument("source", type=int, help="Source node index.")
    gr_path.add_argument("target", type=int, help="Target node index.")
    gr_path.set_defaults(func=_run_graph_path)
    gr_cent = graph_sub.add_parser("centrality", help="Analyze degree / betweenness / closeness centrality.")
    gr_cent.add_argument("file", help="Adjacency matrix file.")
    gr_cent.set_defaults(func=_run_graph_centrality)
    gr_iso = graph_sub.add_parser("isomorphism", help="Check if two graphs are isomorphic (WL hash).")
    gr_iso.add_argument("file_a", help="First adjacency matrix file.")
    gr_iso.add_argument("file_b", help="Second adjacency matrix file.")
    gr_iso.set_defaults(func=_run_graph_isomorphism)
    gr_track = graph_sub.add_parser("track", help="Track community drift across graph snapshots.")
    gr_track.add_argument("files", nargs="+", help="Two or more snapshot adjacency files.")
    gr_track.set_defaults(func=_run_graph_track_dynamic)
    gr_span = graph_sub.add_parser("spanning", help="Extract minimum spanning tree via Kruskal.")
    gr_span.add_argument("file", help="Adjacency matrix file.")
    gr_span.set_defaults(func=_run_graph_spanning)

    # ------------------------------------------------------------------
    # Phase 7 — Robotics subparsers.
    # ------------------------------------------------------------------
    robotics_parser = subparsers.add_parser(
        "robotics",
        help="Robotics capabilities (motion / forward / inverse / fuse / kalman / gait / optimize / collision / path-collision / mpc).",
        description="Motion planning + IK + sensor fusion + Kalman + gait + trajectory optimization + collision + MPC facade.",
    )
    robotics_sub = robotics_parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>",
        help="motion | forward | inverse | fuse | kalman | gait | optimize | collision | path-collision | mpc",
    )
    rb_motion = robotics_sub.add_parser(
        "motion", help="Plan a smooth trajectory through waypoints (cubic spline)."
    )
    rb_motion.add_argument("file", help="Waypoints file (2D: n_wp x n_dof).")
    rb_motion.add_argument("--steps", type=int, default=100)
    rb_motion.set_defaults(func=_run_robotics_motion)
    rb_fwd = robotics_sub.add_parser(
        "forward", help="Forward kinematics: joint angles -> end-effector position."
    )
    rb_fwd.add_argument("file", help="Joint angles file (1D).")
    rb_fwd.set_defaults(func=_run_robotics_forward)
    rb_inv = robotics_sub.add_parser(
        "inverse", help="Inverse kinematics via damped least squares."
    )
    rb_inv.add_argument("file", help="Target position file (1D, length 2).")
    rb_inv.add_argument(
        "--seed", default=None,
        help="Optional seed joint-angles file or inline JSON vector.",
    )
    rb_inv.set_defaults(func=_run_robotics_inverse)
    rb_fuse = robotics_sub.add_parser(
        "fuse", help="Fuse multiple sensor measurements by inverse-variance weighting."
    )
    rb_fuse.add_argument("file", help="2D measurements file (n_sensors x n_dims).")
    rb_fuse.add_argument(
        "--variances", default=None,
        help="Comma-separated variances (one per row). Defaults to 1.0 each.",
    )
    rb_fuse.set_defaults(func=_run_robotics_fuse)
    rb_kal = robotics_sub.add_parser(
        "kalman", help="Sequential Kalman-style Bayesian update."
    )
    rb_kal.add_argument(
        "prior", help="Prior estimate (inline JSON or file)."
    )
    rb_kal.add_argument(
        "measurement", help="Measurement (inline JSON or file)."
    )
    rb_kal.add_argument("--prior-var", type=float, required=True)
    rb_kal.add_argument("--meas-var", type=float, required=True)
    rb_kal.set_defaults(func=_run_robotics_kalman)
    rb_gait = robotics_sub.add_parser(
        "gait", help="Generate a periodic gait pattern (walk / trot / bound)."
    )
    rb_gait.add_argument("--steps", type=int, default=100)
    rb_gait.add_argument(
        "--gait", default="walk",
        choices=["walk", "trot", "bound"],
    )
    rb_gait.set_defaults(func=_run_robotics_gait)
    rb_opt = robotics_sub.add_parser(
        "optimize", help="Smooth a trajectory by minimizing jerk (gradient descent)."
    )
    rb_opt.add_argument("file", help="Trajectory file (2D: n_steps x n_dof).")
    rb_opt.add_argument("--n-iter", type=int, default=10)
    rb_opt.set_defaults(func=_run_robotics_optimize)
    rb_col = robotics_sub.add_parser(
        "collision", help="Check collision at a single position against obstacles."
    )
    rb_col.add_argument(
        "file",
        help="Obstacles file (2D: each row is [x, y, ..., radius]).",
    )
    rb_col.add_argument(
        "--position", required=True,
        help="Body position as comma-separated values (e.g. 0.0,0.0).",
    )
    rb_col.add_argument("--radius", type=float, default=0.1, help="Body radius.")
    rb_col.set_defaults(func=_run_robotics_collision)
    rb_pcol = robotics_sub.add_parser(
        "path-collision", help="Check collision along a path."
    )
    rb_pcol.add_argument(
        "obstacles",
        help="Obstacles file (2D: each row is [x, y, ..., radius]).",
    )
    rb_pcol.add_argument("path", help="Path file (2D: n_steps x n_dof).")
    rb_pcol.add_argument("--radius", type=float, default=0.1, help="Body radius.")
    rb_pcol.set_defaults(func=_run_robotics_path_collision)
    rb_mpc = robotics_sub.add_parser(
        "mpc", help="Pick the next control action via model predictive control."
    )
    rb_mpc.add_argument(
        "current", help="Current state (inline JSON or file).",
    )
    rb_mpc.add_argument(
        "target", help="Target state (inline JSON or file).",
    )
    rb_mpc.add_argument(
        "--obstacles", default=None,
        help="Optional obstacles file (2D: each row is [x, y, ..., radius]).",
    )
    rb_mpc.set_defaults(func=_run_robotics_mpc)

    rl_parser = subparsers.add_parser(
        "rl",
        help="Reinforcement learning capabilities (step / train-q / search-mcts).",
        description="Synthetic MDP + Q-learning + MCTS facade.",
    )
    rl_sub = rl_parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>",
        help="step | train-q | search-mcts",
    )
    rl_step = rl_sub.add_parser("step", help="Take one MDP step.")
    rl_step.add_argument("state", type=int, help="Current state index.")
    rl_step.add_argument("action", type=int, help="Action index.")
    rl_step.set_defaults(func=_run_rl_step)
    rl_train = rl_sub.add_parser("train-q", help="Train the Q-learner.")
    rl_train.add_argument("--episodes", type=int, default=100)
    rl_train.add_argument("--max-steps", type=int, default=100)
    rl_train.set_defaults(func=_run_rl_train_q)
    rl_mcts = rl_sub.add_parser("search-mcts", help="MCTS over the synthetic MDP.")
    rl_mcts.add_argument("root_state", type=int)
    rl_mcts.add_argument("--simulations", type=int, default=100)
    rl_mcts.add_argument("--depth", type=int, default=10)
    rl_mcts.set_defaults(func=_run_rl_search_mcts)

    # ------------------------------------------------------------------ #
    # Phase 7 — Time subparser.
    # ------------------------------------------------------------------ #
    time_parser = subparsers.add_parser(
        "time",
        help="Time-series capabilities (encode / seasonality / frequency / events / anomaly / cycle / forecast).",
        description="Time-series encoder + seasonality + FFT + event timing + anomaly + cycle phase + forecastability facade.",
    )
    time_sub = time_parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>",
        help="encode | seasonality | frequency | events | anomaly | cycle | forecast",
    )
    t_enc = time_sub.add_parser(
        "encode", help="Encode a 1D time series into a dim-length L2-normalized vector."
    )
    t_enc.add_argument("file", help="Time-series file (1D).")
    t_enc.set_defaults(func=_run_time_encode)

    t_seas = time_sub.add_parser(
        "seasonality", help="Detect seasonal periods via autocorrelation peaks."
    )
    t_seas.add_argument("file", help="Time-series file (1D).")
    t_seas.set_defaults(func=_run_time_seasonality)

    t_freq = time_sub.add_parser(
        "frequency", help="FFT frequency analysis + spectral entropy."
    )
    t_freq.add_argument("file", help="Time-series file (1D).")
    t_freq.set_defaults(func=_run_time_frequency)

    t_evt = time_sub.add_parser(
        "events", help="Analyze event timestamps (inter-arrival, rate, burstiness)."
    )
    t_evt.add_argument("file", help="Event timestamps file (1D).")
    t_evt.set_defaults(func=_run_time_events)

    t_anom = time_sub.add_parser(
        "anomaly", help="Detect anomalous inter-arrival gaps via z-score."
    )
    t_anom.add_argument("file", help="Event timestamps file (1D).")
    t_anom.set_defaults(func=_run_time_anomaly)

    t_cyc = time_sub.add_parser(
        "cycle", help="Track phase within a periodic cycle."
    )
    t_cyc.add_argument("file", help="Time-series file (1D).")
    t_cyc.add_argument(
        "--period", type=int, default=None,
        help="Explicit cycle period; auto-detected when omitted.",
    )
    t_cyc.set_defaults(func=_run_time_cycle)

    t_fc = time_sub.add_parser(
        "forecast", help="Score forecastability (entropy + stationarity + autocorr)."
    )
    t_fc.add_argument("file", help="Time-series file (1D).")
    t_fc.set_defaults(func=_run_time_forecast)

    # ------------------------------------------------------------------ #
    # Phase 7 — Code subparser.
    # ------------------------------------------------------------------ #
    code_parser = subparsers.add_parser(
        "code",
        help="Code-analysis capabilities (encode / ast / compare / defects / control-flow / style / dependencies).",
        description="Python source encoder + AST analyzer + similarity + defects + control flow + style + dependency graph facade.",
    )
    code_sub = code_parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>",
        help="encode | ast | compare | defects | control-flow | style | dependencies",
    )
    c_enc = code_sub.add_parser(
        "encode", help="Encode Python source into a dim-length L2-normalized vector."
    )
    c_enc.add_argument("file", help="Python source file (.py).")
    c_enc.set_defaults(func=_run_code_encode)

    c_ast = code_sub.add_parser(
        "ast", help="Analyze AST (functions / classes / imports / complexity / depth)."
    )
    c_ast.add_argument("file", help="Python source file (.py).")
    c_ast.set_defaults(func=_run_code_ast)

    c_cmp = code_sub.add_parser(
        "compare", help="Compare two code snippets via token + structure similarity."
    )
    c_cmp.add_argument("file_a", help="First Python source file (.py).")
    c_cmp.add_argument("file_b", help="Second Python source file (.py).")
    c_cmp.set_defaults(func=_run_code_compare)

    c_def = code_sub.add_parser(
        "defects", help="Detect defect patterns (mutable_default / bare_except / equals_none)."
    )
    c_def.add_argument("file", help="Python source file (.py).")
    c_def.set_defaults(func=_run_code_defects)

    c_cf = code_sub.add_parser(
        "control-flow", help="Analyze per-function cyclomatic complexity and basic blocks."
    )
    c_cf.add_argument("file", help="Python source file (.py).")
    c_cf.set_defaults(func=_run_code_control_flow)

    c_st = code_sub.add_parser(
        "style", help="Run rule-based style checks (line length / naming / whitespace)."
    )
    c_st.add_argument("file", help="Python source file (.py).")
    c_st.set_defaults(func=_run_code_style)

    c_dep = code_sub.add_parser(
        "dependencies", help="Build a module-level import dependency graph."
    )
    c_dep.add_argument("file", help="Python source file (.py).")
    c_dep.set_defaults(func=_run_code_dependencies)

    # ------------------------------------------------------------------ #
    # Phase 7 — Reasoning subparser.
    # ------------------------------------------------------------------ #
    reasoning_parser = subparsers.add_parser(
        "reasoning",
        help="Reasoning capabilities (prop-infer / syllogism / induct / analogize / abduce / defaults / causal).",
        description="Propositional + deductive + inductive + analogical + abductive + defeasible + causal-chain reasoning facade.",
    )
    reasoning_sub = reasoning_parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>",
        help="prop-infer | syllogism | induct | analogize | abduce | defaults | causal",
    )

    r_pi = reasoning_sub.add_parser(
        "prop-infer",
        help="Forward-chain propositional facts and rules (modus ponens).",
    )
    r_pi.add_argument(
        "file",
        help="JSON config with `facts`, `rules`, `negation_rules` keys.",
    )
    r_pi.set_defaults(func=_run_reasoning_prop_infer)

    r_syl = reasoning_sub.add_parser(
        "syllogism",
        help="Run a categorical syllogism (Barbara / Celarent / Darii / Ferio).",
    )
    r_syl.add_argument(
        "major_file",
        help="JSON array [form, subject, predicate] for the major premise.",
    )
    r_syl.add_argument(
        "minor_file",
        help="JSON array [form, subject, predicate] for the minor premise.",
    )
    r_syl.set_defaults(func=_run_reasoning_syllogism)

    r_ind = reasoning_sub.add_parser(
        "induct", help="Induce a (key, value) rule from labeled examples."
    )
    r_ind.add_argument(
        "file",
        help="JSON config with `examples` (list of dicts) and `labels` (list of bools).",
    )
    r_ind.set_defaults(func=_run_reasoning_induct)

    r_ana = reasoning_sub.add_parser(
        "analogize", help="Find a structural analogy between a source and target dict."
    )
    r_ana.add_argument("source_file", help="JSON object (source attributes).")
    r_ana.add_argument("target_file", help="JSON object (target attributes).")
    r_ana.set_defaults(func=_run_reasoning_analogize)

    r_abd = reasoning_sub.add_parser(
        "abduce", help="Pick the best explanation for an observation."
    )
    r_abd.add_argument(
        "file",
        help="JSON config with `observation`, `hypotheses`, and optional `priors`.",
    )
    r_abd.set_defaults(func=_run_reasoning_abduce)

    r_def = reasoning_sub.add_parser(
        "defaults", help="Apply defeasible default rules to a set of facts."
    )
    r_def.add_argument(
        "file",
        help="JSON config with `defaults` (list of {rule, exception}) and `facts`.",
    )
    r_def.set_defaults(func=_run_reasoning_defaults)

    r_cau = reasoning_sub.add_parser(
        "causal", help="Trace a causal chain from a starting node."
    )
    r_cau.add_argument(
        "file",
        help="JSON config with `links`, `start`, and optional `max_depth`.",
    )
    r_cau.set_defaults(func=_run_reasoning_causal)

    # ------------------------------------------------------------------ #
    # Phase 7 — Causal subparser.
    # ------------------------------------------------------------------ #
    causal_parser = subparsers.add_parser(
        "causal",
        help="Causal/decision capabilities (decision-tree / game / counterfactual / bandit / pomdp / discover-graph / intervene).",
        description="Decision trees, game theory, counterfactual reasoning, multi-armed bandits, POMDP solving, causal-graph discovery, and intervention analysis.",
    )
    causal_sub = causal_parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>",
        help="decision-tree | game | counterfactual | bandit | pomdp | discover-graph | intervene",
    )

    c_dt = causal_sub.add_parser(
        "decision-tree",
        help="Fit an ID3-style decision tree to (features, labels).",
    )
    c_dt.add_argument(
        "file",
        help="JSON config with `features` (2D) and `labels` (1D).",
    )
    c_dt.set_defaults(func=_run_causal_decision_tree)

    c_game = causal_sub.add_parser(
        "game",
        help="Find pure-strategy Nash equilibria of a 2-player normal-form game.",
    )
    c_game.add_argument(
        "file",
        help="JSON config with `payoff_a` (2D) and optional `payoff_b` (2D).",
    )
    c_game.set_defaults(func=_run_causal_game)

    c_cf = causal_sub.add_parser(
        "counterfactual",
        help="Estimate counterfactual outcome under do(X[index] = value).",
    )
    c_cf.add_argument(
        "file",
        help="JSON config with `observed`, `index`, and `value`.",
    )
    c_cf.set_defaults(func=_run_causal_counterfactual)

    c_band = causal_sub.add_parser(
        "bandit", help="Select the next bandit arm (epsilon-greedy + UCB1)."
    )
    c_band.add_argument(
        "file",
        help="JSON config with `rewards_history` (list of lists).",
    )
    c_band.set_defaults(func=_run_causal_bandit)

    c_pomdp = causal_sub.add_parser(
        "pomdp", help="Solve a (PO)MDP via value iteration."
    )
    c_pomdp.add_argument(
        "file",
        help="JSON config with `transitions` (3D), `observations` (2D), `rewards`.",
    )
    c_pomdp.set_defaults(func=_run_causal_pomdp)

    c_disc = causal_sub.add_parser(
        "discover-graph",
        help="Discover a causal graph from observational data (PC-style).",
    )
    c_disc.add_argument(
        "file",
        help="JSON config with `data` (2D) and optional `var_names`.",
    )
    c_disc.set_defaults(func=_run_causal_discover_graph)

    c_int = causal_sub.add_parser(
        "intervene",
        help="Estimate the effect of do(X[intervention_var] = value).",
    )
    c_int.add_argument(
        "file",
        help="JSON config with `data` (2D), `intervention_var`, `intervention_value`.",
    )
    c_int.set_defaults(func=_run_causal_intervene)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        return _print_version()
    if args.mcp:
        return _run_mcp()

    if getattr(args, "command", None) in (
        "emergence", "memory", "planning", "multimodal", "rl",
        "audio", "graph", "robotics", "time", "code", "reasoning", "causal",
    ):
        # ``func`` is set via ``set_defaults`` on each subparser.
        try:
            return args.func(args)
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    return _run_demo()


if __name__ == "__main__":
    sys.exit(main())
