"""MCP Server exposing zero-data model capabilities as AI agent tools.

Uses the FastMCP SDK (pip install mcp) if available. When mcp is not installed,
the tool functions are still defined as plain callables for direct use.
"""
from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .model import ZeroDataModel

# Optional FastMCP SDK. The module must import and work even when ``mcp`` is
# not installed; in that case tools are exposed as plain callables.
try:  # pragma: no cover - exercised only when mcp is installed.
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - graceful degradation path.
    FastMCP = None  # type: ignore[assignment, misc]


def _to_py(obj: Any) -> Any:
    """Recursively convert numpy values to JSON-serializable Python types.

    Non-finite floats (NaN, +Inf, -Inf) are mapped to ``None`` to keep
    output strict-JSON-safe — MCP clients using strict parsers (e.g.
    standard ``json.loads``) reject non-finite floats, and persistence
    diagrams routinely contain ``+Inf`` for essential homology classes
    (V4-NEW-M002).
    """
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_py(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def _ensure_finite(arr: np.ndarray, name: str) -> None:
    """Raise ValueError if ``arr`` contains NaN or Inf.

    Mirrors ``api._ensure_finite`` but raises plain ValueError (the
    ``@_error_to_dict`` wrapper converts it to ``{"error": ...}`` for
    MCP clients). V4-NEW-L006 (v4.2): MCP tool inputs previously did
    only ``np.asarray(dtype=float)`` without finiteness checking, so
    NaN/Inf silently entered the engine and surfaced as opaque
    RuntimeWarnings. Now they return a clean error dict instead.
    """
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN or Inf)")


def _parse_obstacles(rows: list[list[float]]) -> list[tuple[np.ndarray, float]]:
    """Convert MCP ``obstacles`` rows to ``(center, radius)`` tuples.

    Each row is ``[center_x, center_y, ..., radius]`` — the LAST element
    is the obstacle radius, the leading elements form the center
    coordinates. Returns a list of ``(np.ndarray, float)`` matching the
    format expected by :meth:`ZeroDataModel.check_collision`.
    """
    parsed: list[tuple[np.ndarray, float]] = []
    for i, row in enumerate(rows):
        if len(row) < 2:
            raise ValueError(
                f"obstacles[{i}] must have >= 2 values (center + radius), got {len(row)}"
            )
        center = np.asarray(row[:-1], dtype=float)
        _ensure_finite(center, f"obstacles[{i}].center")
        radius = float(row[-1])
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError(f"obstacles[{i}].radius must be finite and >= 0, got {radius}")
        parsed.append((center, radius))
    return parsed


def _error_to_dict(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Wrap a tool so any exception is surfaced as ``{"error": str}``.

    ``functools.wraps`` preserves the original signature (via ``__wrapped__``)
    so FastMCP can introspect the real parameter schema, and the docstring is
    retained for use as the MCP tool description.

    Round-3 audit: log the full exception with traceback server-side before
    returning the error dict, so operators have visibility into MCP tool
    failures (previously the traceback was permanently lost).
    """
    import logging

    _logger = logging.getLogger(__name__)

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface as a dict.
            _logger.exception("MCP tool %s failed", fn.__name__)
            return {"error": f"{fn.__name__}: {exc}"}

    return _wrapper


class ZeroDataMCPServer:
    """Wrap a :class:`ZeroDataModel` as a set of MCP tools for AI agents.

    When the ``mcp`` package is installed, tools are registered with a
    :class:`FastMCP` server (call :meth:`run` to serve them over MCP). When it
    is not installed, ``self.mcp`` is ``None`` and the tools remain available as
    plain callables via :meth:`call_tool` (useful for testing and scripting).
    """

    def __init__(self, dim: int = 64) -> None:
        self.model = ZeroDataModel(dim=dim)
        self.dim = dim
        # Map of tool name -> bound callable. Always populated so the server is
        # usable without the MCP SDK (graceful degradation).
        self.tools: dict[str, Callable[..., dict]] = self._build_tools()

        if FastMCP is not None:  # pragma: no cover - requires mcp installed.
            self.mcp = FastMCP("zero-data-model")
            for name, func in self.tools.items():
                # Register each callable as an MCP tool; the docstring is used
                # as the tool description and the explicit name overrides the
                # private method name.
                self.mcp.tool(name=name)(func)
        else:
            self.mcp = None

    # ------------------------------------------------------------------
    # Tool implementations. Each method accepts JSON-friendly inputs (lists,
    # strings, ints), converts to numpy as needed, calls the underlying model,
    # normalizes outputs to JSON-serializable Python types, and returns a dict.
    # A docstring on each tool doubles as its MCP description. The
    # ``_error_to_dict`` decorator converts any raised exception into an
    # ``{"error": str}`` response.
    # ------------------------------------------------------------------

    @_error_to_dict
    def think(self, input_data: list[float] | None = None) -> dict:
        """Run one cognitive cycle of the zero-data model.

        If ``input_data`` is omitted the model self-generates a thought from
        its internal state. Returns the integrated signal, a confidence score
        and per-cycle metadata.
        """
        arr = np.asarray(input_data, dtype=float) if input_data is not None else None
        signal = self.model.think(arr)
        return {
            "signal_data": _to_py(signal.data),
            "confidence": float(signal.confidence),
            "metadata": _to_py(signal.metadata),
        }

    @_error_to_dict
    def classify_text(self, text: str) -> dict:
        """Zero-shot classify ``text`` into a topic (tech/nature/emotion/science).

        Returns the predicted topic and a confidence score in [0, 1].
        """
        topic, confidence = self.model.classify_text(text)
        return {"topic": str(topic), "confidence": float(confidence)}

    @_error_to_dict
    def text_similarity(self, text_a: str, text_b: str) -> dict:
        """Compute the semantic similarity between two texts in [0, 1]."""
        similarity = self.model.text_similarity(text_a, text_b)
        return {"similarity": float(similarity)}

    @_error_to_dict
    def generate_text(self, seed: str, length: int = 32) -> dict:
        """Generate text from a ``seed`` with no external training data."""
        text = self.model.generate_text(seed, length=length)
        return {"text": str(text)}

    @_error_to_dict
    def encode_text(self, text: str) -> dict:
        """Encode ``text`` into a fixed-length L2-normalized embedding vector."""
        vec = self.model.encode_text(text)
        return {"embedding": _to_py(vec), "dim": int(vec.shape[0])}

    @_error_to_dict
    def recognize_pattern(self, image: list[list[float]]) -> dict:
        """Recognize a shape/pattern in a 2D image via self-synthesized prototypes.

        Returns the best-matching pattern name and a confidence score.
        """
        arr = np.asarray(image, dtype=float)
        pattern, confidence = self.model.recognize_pattern(arr)
        return {"pattern": str(pattern), "confidence": float(confidence)}

    @_error_to_dict
    def analyze_shape(self, image: list[list[float]]) -> dict:
        """Analyze geometric/topological shape properties of a 2D image.

        Returns aspect ratio, symmetry, complexity and betti-0.
        """
        arr = np.asarray(image, dtype=float)
        return _to_py(self.model.analyze_shape(arr))

    @_error_to_dict
    def forecast(self, series: list[float], horizon: int = 5) -> dict:
        """Forecast ``horizon`` future values of a 1D time series (zero-data)."""
        arr = np.asarray(series, dtype=float)
        forecast = self.model.forecast(arr, horizon=horizon)
        return {"forecast": _to_py(forecast), "horizon": int(horizon)}

    @_error_to_dict
    def detect_anomalies(self, series: list[float]) -> dict:
        """Detect anomalies in a 1D series; returns a boolean mask and count."""
        arr = np.asarray(series, dtype=float)
        mask = self.model.detect_anomalies(arr)
        anomalies = [bool(x) for x in np.asarray(mask, dtype=bool).tolist()]
        return {"anomalies": anomalies, "count": int(sum(anomalies))}

    @_error_to_dict
    def analyze_trend(self, series: list[float]) -> dict:
        """Analyze trend, regime, curvature, geodesic deviation and isomorphism."""
        arr = np.asarray(series, dtype=float)
        return _to_py(self.model.analyze_trend(arr))

    @_error_to_dict
    def find_analogies(self, series_a: list[float], series_b: list[float]) -> dict:
        """Compute structural analogy strength between two series in [0, 1]."""
        a = np.asarray(series_a, dtype=float)
        b = np.asarray(series_b, dtype=float)
        strength = self.model.find_analogies(a, b)
        return {"analogy_strength": float(strength)}

    @_error_to_dict
    def detect_script(self, text: str) -> dict:
        """Detect the dominant script of ``text``.

        Returns one of 11 recognized script names (latin, cyrillic, greek,
        hebrew, arabic, devanagari, thai, hiragana, katakana, cjk, hangul),
        ``'mixed'`` (when no single script dominates the recognized chars),
        or ``'unknown'`` (when no recognized-script character is present).

        Round-10 audit R10-C-006: the previous docstring listed only 5
        scripts (latin/cyrillic/cjk/arabic/mixed); the implementation has
        supported 11 scripts since Fix 18, but the docstring was never
        updated.
        """
        script = self.model.detect_script(text)
        return {"script": str(script)}

    @_error_to_dict
    def analyze_syntax(self, text: str) -> dict:
        """Rule-based syntactic analysis (sentences, POS guesses, SVO hint)."""
        return _to_py(self.model.analyze_syntax(text))

    @_error_to_dict
    def infer_cause(self, cause: list[float], effect: list[float]) -> dict:
        """Granger-style causal inference between a cause and effect series.

        Returns the best lag, a signed causal strength in [-1, 1] and a
        rule-based p-value approximation.
        """
        c = np.asarray(cause, dtype=float)
        e = np.asarray(effect, dtype=float)
        return _to_py(self.model.infer_cause(c, e))

    @_error_to_dict
    def detect_change_points(self, series: list[float]) -> dict:
        """Detect distributional change points in a 1D series.

        Returns the list of change-point indices and the total count.
        """
        arr = np.asarray(series, dtype=float)
        cps = self.model.detect_change_points(arr)
        change_points = [int(x) for x in np.asarray(cps, dtype=int).tolist()]
        return {"change_points": change_points, "count": int(len(change_points))}

    @_error_to_dict
    def hardware_info(self) -> dict:
        """Report the active hardware backends (array, gpu, quantum, parallel)."""
        return _to_py(self.model.hardware_info)

    # ------------------------------------------------------------------
    # Causal emergence tools (spec §2.3). These delegate to the engine via
    # the model facade so callers never construct CausalEmergenceEngine
    # directly. Inputs are JSON-friendly lists; outputs go through _to_py.
    # ------------------------------------------------------------------

    @_error_to_dict
    def perceive_topology(
        self, data: list[list[float]], max_dim: int | None = None
    ) -> dict:
        """Compute topological invariants of a point cloud.

        Calculates Betti numbers (β0=components, β1=loops, β2=voids),
        persistence entropy (topological complexity in [0, 1]), Euler
        characteristic (alternating sum β0-β1+β2), point count and the
        largest persistence epsilon.

        Args:
            data: 2D array of shape ``(n_points, n_features)``. Must be
                at least 2x2 — smaller or non-2D input raises ValueError.
            max_dim: Optional maximum homology dimension (default 2).

        Returns:
            Dict with keys ``betti_numbers`` (list[int], len 3),
            ``persistence_entropy`` (float), ``euler_characteristic``
            (int), ``n_points`` (int), ``max_eps`` (float).

        Failure mode: returns ``{"error": "perceive_topology: ..."}`` on
        invalid input or internal failure.
        """
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            raise ValueError(
                f"data must be 2D with shape (n>=2, d>=2), got {arr.shape}"
            )
        _ensure_finite(arr, "data")  # V4-NEW-L006
        result = self.model.perceive_topology(arr, max_dim=max_dim)
        return _to_py({
            "betti_numbers": result["betti_numbers"],
            "persistence_entropy": result["persistence_entropy"],
            "euler_characteristic": result["euler_characteristic"],
            "n_points": result["n_points"],
            "max_eps": result["max_eps"],
        })

    @_error_to_dict
    def discover_causal_dynamics(
        self,
        data: list[list[float]],
        var_names: list[str] | None = None,
        method: str | None = None,
    ) -> dict:
        """Discover a causal DAG from observational data.

        Builds a directed acyclic graph over the columns of ``data``
        using PC (default, conditional independence), LiNGAM
        (``method='lingam'``) or Pearson correlation
        (``method='correlation'``).

        Args:
            data: 2D array of shape ``(n_samples, n_vars)``. Must be at
                least 2x2 — smaller or non-2D input raises ValueError.
            var_names: Optional names for each column (defaults to
                ``x0, x1, ...``).
            method: ``'pc'`` | ``'lingam'`` | ``'correlation'`` (default
                ``'pc'``).

        Returns:
            Dict with keys ``adjacency`` (list[list[int]] shape
            ``(n_vars, n_vars)``; ``adjacency[i][j]==1`` means ``i -> j``),
            ``edges`` (list of ``[src, dst]`` pairs), ``n_edges`` (int),
            ``is_acyclic`` (bool), ``method`` (str), ``var_names``
            (list[str]).

        Failure mode: returns ``{"error": "discover_causal_dynamics: ..."}``
        on invalid input or internal failure.
        """
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            raise ValueError(
                f"data must be 2D with shape (n>=2, d>=2), got {arr.shape}"
            )
        _ensure_finite(arr, "data")  # V4-NEW-L006
        result = self.model.discover_causal_dynamics(
            arr, var_names=var_names, method=method
        )
        adj = np.asarray(result.get("adjacency", np.zeros((0, 0))))
        return _to_py({
            "adjacency": adj,
            "edges": result.get("edges", []),
            "n_edges": int(result.get("n_edges", 0)),
            "is_acyclic": bool(result.get("is_acyclic", True)),
            "method": str(result.get("method", "none")),
            "var_names": list(result.get("var_names", [])),
        })

    @_error_to_dict
    def generate_trajectory(
        self,
        start_state: list[float],
        end_state: list[float],
        n_steps: int = 32,
        obstacles: list[list[float]] | None = None,
        margin: float | None = None,
    ) -> dict:
        """Generate a damped least-action trajectory between two states.

        Solves a discretized Euler-Lagrange boundary value problem with
        damping (spec §5). Optional obstacle avoidance projects interior
        points outside the safety margin after each relaxation sweep.

        Args:
            start_state, end_state: 1D boundary vectors of the same
                dimension. Mismatched dimensions raise ValueError.
            n_steps: Number of interior steps; ``trajectory`` will have
                ``n_steps + 1`` rows.
            obstacles: Optional list of obstacle centers, each of length
                ``dim``. When provided, ``obstacle_violations`` is added
                to the output.
            margin: Optional safety distance around each obstacle
                (defaults to ``rules.differential_obstacle_margin``).

        Returns:
            Dict with keys ``trajectory`` (list[list[float]] shape
            ``(n_steps + 1, dim)``), ``lagrangian`` (list[float] length
            ``n_steps``), ``action`` (float), ``converged`` (bool),
            ``iterations`` (int). ``obstacle_violations`` (int) is added
            only when ``obstacles`` is provided.

        Failure mode: returns ``{"error": "generate_trajectory: ..."}`` on
        invalid input or internal failure.
        """
        start = np.asarray(start_state, dtype=float)
        end = np.asarray(end_state, dtype=float)
        if start.shape != end.shape:
            raise ValueError(
                f"start_state shape {start.shape} != end_state shape {end.shape}"
            )
        _ensure_finite(start, "start_state")  # V4-NEW-L006
        _ensure_finite(end, "end_state")  # V4-NEW-L006
        constraints: dict | None = None
        if obstacles is not None or margin is not None:
            constraints = {}
            if obstacles is not None:
                obs_arr = np.asarray(obstacles, dtype=float)
                _ensure_finite(obs_arr, "obstacles")  # V4-NEW-L006
                constraints["obstacles"] = obs_arr
            if margin is not None:
                constraints["margin"] = float(margin)
        result = self.model.generate_trajectory(
            start, end, n_steps=int(n_steps), constraints=constraints
        )
        return _to_py(result)

    @_error_to_dict
    def sample_posterior(
        self, mean: list[float], std: float = 1.0, n_samples: int = 100
    ) -> dict:
        """Sample from a Gaussian posterior via Hamiltonian Monte Carlo.

        MCP does NOT expose arbitrary ``log_prob_fn`` callables (those
        cannot be JSON-serialized). Only a Gaussian posterior centered at
        ``mean`` with standard deviation ``std`` is supported.

        Args:
            mean: 1D target mean vector. Non-1D input raises ValueError.
            std: Gaussian standard deviation (must be > 0).
            n_samples: Number of HMC samples (must be >= 1).

        Returns:
            Dict with keys ``samples`` (list[list[float]] shape
            ``(n_samples, len(mean))``), ``mean`` (list[float]), ``std``
            (list[float], per-dimension sample std), ``accept_rate``
            (float in [0, 1]), ``ess`` (float, effective sample size),
            ``converged`` (bool).

        Failure mode: returns ``{"error": "sample_posterior: ..."}`` on
        invalid input or internal failure.
        """
        mean_arr = np.asarray(mean, dtype=float)
        if mean_arr.ndim != 1:
            raise ValueError(f"mean must be 1D, got shape {mean_arr.shape}")
        _ensure_finite(mean_arr, "mean")  # V4-NEW-L006
        std_f = float(std)
        if std_f <= 0.0:
            raise ValueError(f"std must be > 0, got {std_f}")
        if not math.isfinite(std_f):
            raise ValueError("std must be finite (no NaN or Inf)")  # V4-NEW-L006
        n_s = int(n_samples)
        if n_s < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_s}")

        def _gaussian_log_prob(q: np.ndarray) -> float:
            return -0.5 * float(np.sum(((q - mean_arr) / std_f) ** 2))

        result = self.model.sample_posterior(
            log_prob_fn=_gaussian_log_prob,
            initial_position=mean_arr,
            n_samples=n_s,
        )
        return _to_py({
            "samples": result["samples"],
            "mean": result["mean"],
            "std": result["std"],
            "accept_rate": result["accept_rate"],
            "ess": result["ess"],
            "converged": result["converged"],
        })

    @_error_to_dict
    def recall_memory(self, query: list[float], n_steps: int = 100) -> dict:
        """Recall from chaotic associative memory via a Lorenz attractor.

        The query is encoded as the initial state of a Lorenz system;
        integrating forward yields a chaotic trajectory whose settled
        state identifies the nearest stored pattern. Empty memory
        returns ``label=None, emerged=True`` (spec §7.3).

        Args:
            query: 1D query vector. Non-1D input raises ValueError.
            n_steps: Number of RK4 integration steps (must be >= 0).

        Returns:
            Dict with keys ``label`` (str | int | None), ``similarity``
            (float in [0, 1]), ``emerged`` (bool — True means chaotic
            divergence was detected), ``trajectory`` (list[list[float]]
            shape ``(n_steps + 1, 3)``), ``converged`` (bool),
            ``divergence`` (float), ``nearest_pattern`` (list[float] |
            None).

        Failure mode: returns ``{"error": "recall_memory: ..."}`` on
        invalid input or internal failure.
        """
        query_arr = np.asarray(query, dtype=float)
        if query_arr.ndim != 1:
            raise ValueError(f"query must be 1D, got shape {query_arr.shape}")
        _ensure_finite(query_arr, "query")  # V4-NEW-L006
        result = self.model.recall_memory(query_arr, n_steps=int(n_steps))
        return _to_py(result)

    @_error_to_dict
    def emergence_cycle(self, observation: list[list[float]]) -> dict:
        """Run the full recursive emergence loop on one observation.

        Pipeline: perceive topology → discover causal DAG → generate
        counterfactual trajectory → sample posterior → recall memory →
        compute emergence score. Module failures are logged to
        ``warnings`` and degraded to zero-value placeholders (spec §8.2).

        Args:
            observation: 2D array of shape ``(n_samples, n_features)``.
                Must be at least 2x2 — smaller or non-2D input raises
                ValueError.

        Returns:
            Dict with keys ``perception``, ``causal_graph``,
            ``counterfactual``, ``posterior``, ``memory`` (each a dict
            mirroring the corresponding tool's output),
            ``emergence_score`` (float in [0, 1]) and ``warnings``
            (list[str]).

        Failure mode: returns ``{"error": "emergence_cycle: ..."}`` on
        invalid input or internal failure.
        """
        arr = np.asarray(observation, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            raise ValueError(
                f"observation must be 2D with shape (n>=2, d>=2), got {arr.shape}"
            )
        _ensure_finite(arr, "observation")  # V4-NEW-L006
        result = self.model.emergence_cycle(arr)
        return _to_py(result)

    # ------------------------------------------------------------------
    # Phase 6 — Memory / Planning / Multimodal / RL tools (spec §12).
    # Inputs are JSON-friendly lists/ints/strings; outputs go through
    # _to_py. Each tool mirrors a Phase 6 API endpoint and CLI handler.
    # ------------------------------------------------------------------

    @_error_to_dict
    def memory_encode(
        self, observation: list[float], label: str | int | None = None
    ) -> dict:
        """Encode an observation into the episodic memory store.

        Args:
            observation: 1D feature vector (must be non-empty). Non-1D
                input raises ValueError.
            label: Optional label (string or int) tagging the memory.

        Returns:
            Dict with keys ``id`` (int), ``label``, ``strength`` (float),
            ``timestamp`` (int), ``size`` (int = ``len(observation)``).

        Failure mode: returns ``{"error": "memory_encode: ..."}`` on
        invalid input or internal failure.
        """
        arr = np.asarray(observation, dtype=float)
        if arr.ndim != 1 or arr.shape[0] < 1:
            raise ValueError(
                f"observation must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "observation")
        result = self.model.encode_memory(arr, label=label)
        return _to_py(result)

    @_error_to_dict
    def memory_retrieve(
        self, query: list[float], top_k: int = 5
    ) -> dict:
        """Retrieve the top-k most similar episodic memories.

        Args:
            query: 1D query vector. Non-1D input raises ValueError.
            top_k: Number of nearest neighbors to return (must be >= 1).

        Returns:
            Dict with keys ``matches`` (list of dicts with ``id``,
            ``label``, ``similarity``, ``observation``), ``count`` (int).

        Failure mode: returns ``{"error": "memory_retrieve: ..."}`` on
        invalid input or internal failure.
        """
        arr = np.asarray(query, dtype=float)
        if arr.ndim != 1 or arr.shape[0] < 1:
            raise ValueError(
                f"query must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "query")
        result = self.model.retrieve_memory(arr, top_k=int(top_k))
        return _to_py(result)

    @_error_to_dict
    def memory_consolidate(self) -> dict:
        """Promote high-weight working-memory items into episodic memory.

        Takes no arguments. Returns the consolidation summary dict
        produced by :class:`MemoryConsolidator`.

        Failure mode: returns ``{"error": "memory_consolidate: ..."}``
        on internal failure.
        """
        result = self.model.consolidate_memory()
        return _to_py(result)

    @_error_to_dict
    def planning_trajectory(
        self,
        start_state: list[float],
        goal_state: list[float],
        n_steps: int = 32,
        obstacles: list[list[float]] | None = None,
        margin: float | None = None,
    ) -> dict:
        """Plan a damped least-action trajectory with per-step actions.

        Mirrors the emergence ``generate_trajectory`` tool but routes
        through the Phase 6 ``TrajectoryPlanner`` (which emits per-step
        actions suitable for downstream RL/control).

        Args:
            start_state, goal_state: 1D boundary vectors of equal length.
                Mismatched dimensions raise ValueError.
            n_steps: Number of interior steps (must be >= 0).
            obstacles: Optional list of obstacle centers, each of the
                same dimensionality as ``start_state``.
            margin: Optional safety distance around obstacles.

        Returns:
            Dict with keys ``trajectory``, ``actions``, ``converged``,
            ``iterations``, plus ``obstacle_violations`` when obstacles
            are provided.

        Failure mode: returns ``{"error": "planning_trajectory: ..."}``
        on invalid input or internal failure.
        """
        start = np.asarray(start_state, dtype=float)
        end = np.asarray(goal_state, dtype=float)
        if start.shape != end.shape:
            raise ValueError(
                f"start_state shape {start.shape} != goal_state shape {end.shape}"
            )
        _ensure_finite(start, "start_state")
        _ensure_finite(end, "goal_state")
        obs_arr: np.ndarray | None = None
        if obstacles is not None:
            obs_arr = np.asarray(obstacles, dtype=float)
            _ensure_finite(obs_arr, "obstacles")
        result = self.model.plan_trajectory(
            start,
            end,
            obstacles=obs_arr,
            n_steps=int(n_steps),
            margin=margin,
        )
        return _to_py(result)

    @_error_to_dict
    def planning_decompose(
        self, goal: str, max_depth: int | None = None
    ) -> dict:
        """Decompose an abstract goal into an AND/OR tree of sub-goals.

        Args:
            goal: A short goal description string (must be non-empty).
            max_depth: Optional recursion depth cap (must be >= 1 when set).

        Returns:
            Dict with keys ``tree`` (nested dict), ``n_nodes`` (int),
            ``depth`` (int).

        Failure mode: returns ``{"error": "planning_decompose: ..."}``
        on invalid input or internal failure.
        """
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        result = self.model.decompose_goal(goal, max_depth=max_depth)
        return _to_py(result)

    @_error_to_dict
    def planning_sequence(self, adjacency: list[list[int]]) -> dict:
        """Topologically sort a DAG of actions with greedy cycle breaking.

        Args:
            adjacency: Square 2D adjacency matrix (list of lists of
                ``0``/``1``). Must be at least 1x1.

        Returns:
            Dict with keys ``order`` (list[int]), ``cycles_broken``
            (int), ``valid`` (bool).

        Failure mode: returns ``{"error": "planning_sequence: ..."}``
        on invalid input or internal failure.
        """
        arr = np.asarray(adjacency, dtype=int)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
            raise ValueError(
                f"adjacency must be square 2D, got shape {arr.shape}"
            )
        result = self.model.sequence_actions(arr)
        return _to_py(result)

    @_error_to_dict
    def multimodal_align(
        self,
        observations_a: list[list[float]],
        observations_b: list[list[float]],
    ) -> dict:
        """Fit CCA on paired observations across two modalities.

        Args:
            observations_a, observations_b: 2D arrays of shape
                ``(n_samples, dim_a)`` and ``(n_samples, dim_b)``. Both
                must have at least 2 rows and 2 columns. The row counts
                must match (CCA is supervised on the sample pairing).

        Returns:
            Dict with keys ``correlations`` (list[float]), ``n_components``
            (int), ``mean_correlation`` (float in [0, 1]).

        Failure mode: returns ``{"error": "multimodal_align: ..."}`` on
        invalid input or internal failure.
        """
        a = np.asarray(observations_a, dtype=float)
        b = np.asarray(observations_b, dtype=float)
        if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
            raise ValueError(
                f"observations_a must be 2D with shape (n>=2, d>=2), got {a.shape}"
            )
        if b.ndim != 2 or b.shape[0] < 2 or b.shape[1] < 2:
            raise ValueError(
                f"observations_b must be 2D with shape (n>=2, d>=2), got {b.shape}"
            )
        if a.shape[0] != b.shape[0]:
            raise ValueError(
                f"row count mismatch: a={a.shape[0]} vs b={b.shape[0]}"
            )
        _ensure_finite(a, "observations_a")
        _ensure_finite(b, "observations_b")
        return _to_py(self.model.fit_cross_modal(a, b))

    @_error_to_dict
    def multimodal_fuse(
        self,
        embeddings: list[list[float]],
        strategy: str | None = None,
    ) -> dict:
        """Fuse multiple modality embeddings via mean / concat / weighted.

        Args:
            embeddings: List of 1D embedding vectors (at least 2). All
                vectors must have the same length (mean/weighted modes)
                — concat mode accepts variable lengths.
            strategy: ``'mean'`` | ``'concat'`` | ``'weighted'``
                (default: rules-based).

        Returns:
            Dict with keys ``fused`` (list[float]), ``strategy`` (str),
            ``n_inputs`` (int).

        Failure mode: returns ``{"error": "multimodal_fuse: ..."}`` on
        invalid input or internal failure.
        """
        if not isinstance(embeddings, list) or len(embeddings) < 2:
            raise ValueError("embeddings must be a list of >= 2 vectors")
        arrs = [np.asarray(e, dtype=float) for e in embeddings]
        for i, e in enumerate(arrs):
            if e.ndim != 1:
                raise ValueError(
                    f"embeddings[{i}] must be 1D, got shape {e.shape}"
                )
            _ensure_finite(e, f"embeddings[{i}]")
        return _to_py(self.model.fuse_modalities(arrs, strategy=strategy))

    @_error_to_dict
    def multimodal_contrastive(
        self,
        batch_a: list[list[float]],
        batch_b: list[list[float]],
    ) -> dict:
        """InfoNCE contrastive alignment loss between paired batches.

        Args:
            batch_a, batch_b: 2D arrays of shape ``(n_samples, dim)``.
                Must have at least 2 rows and matching shapes.

        Returns:
            Dict with keys ``loss`` (float), ``accuracy`` (float in
            [0, 1]), ``similarity_matrix`` (list[list[float]]).

        Failure mode: returns ``{"error": "multimodal_contrastive: ..."}`
        on invalid input or internal failure.
        """
        a = np.asarray(batch_a, dtype=float)
        b = np.asarray(batch_b, dtype=float)
        if a.ndim != 2 or a.shape[0] < 2:
            raise ValueError(
                f"batch_a must be 2D with n>=2 rows, got {a.shape}"
            )
        if b.shape != a.shape:
            raise ValueError(
                f"batch shape mismatch: a={a.shape} vs b={b.shape}"
            )
        _ensure_finite(a, "batch_a")
        _ensure_finite(b, "batch_b")
        return _to_py(self.model.contrastive_loss(a, b))

    @_error_to_dict
    def rl_step(self, state: int, action: int) -> dict:
        """Take one step in the synthetic MDP.

        Args:
            state: Source state index (must be >= 0).
            action: Action index (must be >= 0).

        Returns:
            Dict with keys ``next_state`` (int), ``reward`` (float),
            ``done`` (bool).

        Failure mode: returns ``{"error": "rl_step: ..."}`` on invalid
        input or internal failure.
        """
        s = int(state)
        a = int(action)
        if s < 0:
            raise ValueError(f"state must be >= 0, got {s}")
        if a < 0:
            raise ValueError(f"action must be >= 0, got {a}")
        return _to_py(self.model.step_mdp(s, a))

    @_error_to_dict
    def rl_train_q(
        self,
        n_episodes: int = 100,
        max_steps_per_episode: int = 100,
    ) -> dict:
        """Full tabular Q-learning training loop on the synthetic MDP.

        Args:
            n_episodes: Number of training episodes (must be >= 1).
            max_steps_per_episode: Per-episode step cap (must be >= 1).

        Returns:
            Dict with keys ``episode_rewards`` (list[float]),
            ``final_policy`` (list[int]), ``final_value`` (list[float]),
            ``mean_reward`` (float).

        Failure mode: returns ``{"error": "rl_train_q: ..."}`` on
        invalid input or internal failure.
        """
        if n_episodes < 1:
            raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
        if max_steps_per_episode < 1:
            raise ValueError(
                f"max_steps_per_episode must be >= 1, got {max_steps_per_episode}"
            )
        return _to_py(
            self.model.train_q_learner(
                n_episodes=int(n_episodes),
                max_steps_per_episode=int(max_steps_per_episode),
            )
        )

    @_error_to_dict
    def rl_search_mcts(
        self,
        root_state: int,
        n_simulations: int = 50,
        max_depth: int = 10,
    ) -> dict:
        """UCT Monte-Carlo Tree Search over the known synthetic MDP.

        Args:
            root_state: MDP state index to search from (must be >= 0).
            n_simulations: Number of MCTS rollouts (must be >= 1).
            max_depth: Maximum tree depth (must be >= 1).

        Returns:
            Dict with keys ``best_action`` (int), ``visits`` (int),
            ``value`` (float), ``tree_size`` (int).

        Failure mode: returns ``{"error": "rl_search_mcts: ..."}`` on
        invalid input or internal failure.
        """
        s = int(root_state)
        if s < 0:
            raise ValueError(f"root_state must be >= 0, got {s}")
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        return _to_py(
            self.model.search_rl_mcts(
                s,
                n_simulations=int(n_simulations),
                max_depth=int(max_depth),
            )
        )

    # ------------------------------------------------------------------
    # Phase 7 — Audio tools.
    # ------------------------------------------------------------------

    @_error_to_dict
    def audio_encode(self, signal: list[float], sample_rate: int = 16000) -> dict:
        """Encode a 1D audio signal into a ``dim``-length L2-normalized vector.

        Args:
            signal: 1D audio samples (must be non-empty, finite).
            sample_rate: Samples per second (must be >= 1).

        Returns:
            Dict with key ``embedding`` (list[float] of length ``dim``).

        Failure mode: returns ``{"error": "audio_encode: ..."}`` on
        invalid input or internal failure.
        """
        arr = np.asarray(signal, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"signal must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "signal")
        if sample_rate < 1:
            raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")
        result = self.model.encode_audio(arr, sample_rate=int(sample_rate))
        return {"embedding": _to_py(result)}

    @_error_to_dict
    def audio_detect_onsets(
        self, signal: list[float], sample_rate: int = 16000
    ) -> dict:
        """Detect note/onset events via spectral flux.

        Args:
            signal: 1D audio samples (must be non-empty, finite).
            sample_rate: Samples per second (must be >= 1).

        Returns:
            Dict with keys ``onset_frames``, ``onset_times``,
            ``spectral_flux``, ``mean_flux``.

        Failure mode: returns ``{"error": "audio_detect_onsets: ..."}``.
        """
        arr = np.asarray(signal, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"signal must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "signal")
        if sample_rate < 1:
            raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")
        return _to_py(
            self.model.detect_onsets(arr, sample_rate=int(sample_rate))
        )

    @_error_to_dict
    def audio_detect_pitch(
        self, signal: list[float], sample_rate: int = 16000
    ) -> dict:
        """Detect fundamental frequency via autocorrelation.

        Args:
            signal: 1D audio samples (must be non-empty, finite).
            sample_rate: Samples per second (must be >= 1).

        Returns:
            Dict with keys ``pitch_hz``, ``confidence``, ``f0_candidates``.

        Failure mode: returns ``{"error": "audio_detect_pitch: ..."}``.
        """
        arr = np.asarray(signal, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"signal must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "signal")
        if sample_rate < 1:
            raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")
        return _to_py(
            self.model.detect_pitch(arr, sample_rate=int(sample_rate))
        )

    @_error_to_dict
    def audio_classify(
        self, signal: list[float], sample_rate: int = 16000
    ) -> dict:
        """Classify audio texture (speech / music / noise / silence).

        Args:
            signal: 1D audio samples (must be non-empty, finite).
            sample_rate: Samples per second (must be >= 1).

        Returns:
            Dict with keys ``label`` (str), ``confidence`` (float),
            ``features`` (dict of spectral centroid / ZCR / RMS / flatness).

        Failure mode: returns ``{"error": "audio_classify: ..."}``.
        """
        arr = np.asarray(signal, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"signal must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "signal")
        if sample_rate < 1:
            raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")
        return _to_py(
            self.model.classify_audio(arr, sample_rate=int(sample_rate))
        )

    @_error_to_dict
    def audio_segment_speech(
        self, signal: list[float], sample_rate: int = 16000
    ) -> dict:
        """Segment audio into speech / silence regions via VAD.

        Args:
            signal: 1D audio samples (must be non-empty, finite).
            sample_rate: Samples per second (must be >= 1).

        Returns:
            Dict with keys ``segments`` (list of {start, end, label}),
            ``total_duration`` (float), ``speech_ratio`` (float).

        Failure mode: returns ``{"error": "audio_segment_speech: ..."}``.
        """
        arr = np.asarray(signal, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"signal must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "signal")
        if sample_rate < 1:
            raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")
        return _to_py(
            self.model.segment_speech(arr, sample_rate=int(sample_rate))
        )

    @_error_to_dict
    def audio_analyze_music(
        self, signal: list[float], sample_rate: int = 16000
    ) -> dict:
        """Analyze music for tempo and beats.

        Args:
            signal: 1D audio samples (must be non-empty, finite).
            sample_rate: Samples per second (must be >= 1).

        Returns:
            Dict with keys ``tempo_bpm``, ``beat_frames``,
            ``beat_times``, ``onset_envelope``.

        Failure mode: returns ``{"error": "audio_analyze_music: ..."}``.
        """
        arr = np.asarray(signal, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"signal must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "signal")
        if sample_rate < 1:
            raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")
        return _to_py(
            self.model.analyze_music(arr, sample_rate=int(sample_rate))
        )

    # ------------------------------------------------------------------
    # Phase 7 — Graph tools.
    # ------------------------------------------------------------------

    @_error_to_dict
    def graph_encode(
        self,
        adjacency: list[list[float]],
        node_features: list[list[float]] | None = None,
    ) -> dict:
        """Encode a graph into a ``dim``-length L2-normalized vector.

        Args:
            adjacency: 2D adjacency matrix (square, non-empty, finite).
            node_features: Optional 2D node feature matrix aligned with
                ``adjacency`` rows (n_nodes x n_features).

        Returns:
            Dict with key ``embedding`` (list[float] of length ``dim``).

        Failure mode: returns ``{"error": "graph_encode: ..."}``.
        """
        adj = np.asarray(adjacency, dtype=float)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.shape[0] < 1:
            raise ValueError(
                f"adjacency must be a non-empty 2D square matrix, got {adj.shape}"
            )
        _ensure_finite(adj, "adjacency")
        feats = None
        if node_features is not None:
            feats = np.asarray(node_features, dtype=float)
            if feats.ndim != 2 or feats.shape[0] != adj.shape[0]:
                raise ValueError(
                    f"node_features must be 2D with {adj.shape[0]} rows, got {feats.shape}"
                )
            _ensure_finite(feats, "node_features")
        result = self.model.encode_graph(adj, node_features=feats)
        return {"embedding": _to_py(result)}

    @_error_to_dict
    def graph_detect_communities(self, adjacency: list[list[float]]) -> dict:
        """Detect communities via modularity optimization.

        Args:
            adjacency: 2D adjacency matrix (square, non-empty, finite).

        Returns:
            Dict with keys ``communities`` (list of node lists),
            ``modularity`` (float), ``n_communities`` (int).

        Failure mode: returns ``{"error": "graph_detect_communities: ..."}``.
        """
        adj = np.asarray(adjacency, dtype=float)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.shape[0] < 1:
            raise ValueError(
                f"adjacency must be a non-empty 2D square matrix, got {adj.shape}"
            )
        _ensure_finite(adj, "adjacency")
        return _to_py(self.model.detect_communities(adj))

    @_error_to_dict
    def graph_find_path(
        self,
        adjacency: list[list[float]],
        source: int,
        target: int,
    ) -> dict:
        """Find the shortest path via Dijkstra.

        Args:
            adjacency: 2D adjacency matrix (square, non-empty, finite).
            source: Source node index (must be in [0, n_nodes)).
            target: Target node index (must be in [0, n_nodes)).

        Returns:
            Dict with keys ``path`` (list of node indices),
            ``distance`` (float), ``visited`` (int).

        Failure mode: returns ``{"error": "graph_find_path: ..."}``.
        """
        adj = np.asarray(adjacency, dtype=float)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.shape[0] < 1:
            raise ValueError(
                f"adjacency must be a non-empty 2D square matrix, got {adj.shape}"
            )
        _ensure_finite(adj, "adjacency")
        n = adj.shape[0]
        if not (0 <= int(source) < n):
            raise ValueError(f"source {source} out of range [0, {n})")
        if not (0 <= int(target) < n):
            raise ValueError(f"target {target} out of range [0, {n})")
        return _to_py(self.model.find_path(adj, int(source), int(target)))

    @_error_to_dict
    def graph_analyze_centrality(self, adjacency: list[list[float]]) -> dict:
        """Analyze degree / betweenness / closeness centrality.

        Args:
            adjacency: 2D adjacency matrix (square, non-empty, finite).

        Returns:
            Dict with keys ``degree``, ``betweenness``, ``closeness``,
            ``most_central`` (int node index).

        Failure mode: returns ``{"error": "graph_analyze_centrality: ..."}``.
        """
        adj = np.asarray(adjacency, dtype=float)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.shape[0] < 1:
            raise ValueError(
                f"adjacency must be a non-empty 2D square matrix, got {adj.shape}"
            )
        _ensure_finite(adj, "adjacency")
        return _to_py(self.model.analyze_centrality(adj))

    @_error_to_dict
    def graph_check_isomorphism(
        self,
        adjacency_a: list[list[float]],
        adjacency_b: list[list[float]],
    ) -> dict:
        """Check if two graphs are likely isomorphic (Weisfeiler-Lehman).

        Args:
            adjacency_a: First 2D adjacency matrix.
            adjacency_b: Second 2D adjacency matrix.

        Returns:
            Dict with keys ``isomorphic`` (bool), ``confidence`` (float),
            ``wl_hash_a`` (str), ``wl_hash_b`` (str).

        Failure mode: returns ``{"error": "graph_check_isomorphism: ..."}``.
        """
        adj_a = np.asarray(adjacency_a, dtype=float)
        adj_b = np.asarray(adjacency_b, dtype=float)
        for name, arr in (("adjacency_a", adj_a), ("adjacency_b", adj_b)):
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
                raise ValueError(
                    f"{name} must be a non-empty 2D square matrix, got {arr.shape}"
                )
            _ensure_finite(arr, name)
        return _to_py(self.model.check_isomorphism(adj_a, adj_b))

    @_error_to_dict
    def graph_track_dynamic(self, snapshots: list[list[list[float]]]) -> dict:
        """Track community drift across graph snapshots.

        Args:
            snapshots: List of 2D adjacency matrices (>= 2 snapshots).

        Returns:
            Dict with keys ``community_drift``, ``node_migrations``,
            ``stability`` (float).

        Failure mode: returns ``{"error": "graph_track_dynamic: ..."}``.
        """
        if not isinstance(snapshots, list) or len(snapshots) < 2:
            raise ValueError(
                f"snapshots must be a list of >= 2 adjacency matrices, got len={len(snapshots) if hasattr(snapshots, '__len__') else 'N/A'}"
            )
        snap_arrs = []
        for i, s in enumerate(snapshots):
            arr = np.asarray(s, dtype=float)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
                raise ValueError(
                    f"snapshots[{i}] must be a non-empty 2D square matrix, got {arr.shape}"
                )
            _ensure_finite(arr, f"snapshots[{i}]")
            snap_arrs.append(arr)
        return _to_py(self.model.track_dynamic_graph(snap_arrs))

    @_error_to_dict
    def graph_extract_spanning_tree(self, adjacency: list[list[float]]) -> dict:
        """Extract the minimum spanning tree via Kruskal.

        Args:
            adjacency: 2D adjacency matrix (square, non-empty, finite).

        Returns:
            Dict with keys ``mst_edges`` (list of [u, v, w]),
            ``total_weight`` (float), ``mst_adjacency`` (2D list).

        Failure mode: returns ``{"error": "graph_extract_spanning_tree: ..."}``.
        """
        adj = np.asarray(adjacency, dtype=float)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.shape[0] < 1:
            raise ValueError(
                f"adjacency must be a non-empty 2D square matrix, got {adj.shape}"
            )
        _ensure_finite(adj, "adjacency")
        return _to_py(self.model.extract_spanning_tree(adj))

    # ------------------------------------------------------------------
    # Phase 7 — Robotics tools.
    # ------------------------------------------------------------------

    @_error_to_dict
    def robotics_plan_motion(
        self, waypoints: list[list[float]], n_steps: int = 100
    ) -> dict:
        """Plan a smooth trajectory through waypoints (cubic spline).

        Args:
            waypoints: 2D array of shape ``(n_waypoints, n_dof)`` with at
                least 2 rows. 1D input is reshaped to ``(n, 1)``.
            n_steps: Number of trajectory samples (must be >= 1).

        Returns:
            Dict with keys ``trajectory``, ``velocities``,
            ``accelerations`` (each a 2D list of shape
            ``(n_steps, n_dof)``), ``total_time`` (float).

        Failure mode: returns ``{"error": "robotics_plan_motion: ..."}``.
        """
        wp = np.asarray(waypoints, dtype=float)
        if wp.ndim != 2 or wp.shape[0] < 2:
            raise ValueError(
                f"waypoints must be 2D with >= 2 rows, got shape {wp.shape}"
            )
        _ensure_finite(wp, "waypoints")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        return _to_py(self.model.plan_motion(wp, n_steps=int(n_steps)))

    @_error_to_dict
    def robotics_forward_kinematics(
        self, joint_angles: list[float]
    ) -> dict:
        """Forward kinematics: joint angles -> end-effector position.

        Computes the planar N-DOF arm end-effector position from
        ``joint_angles`` (length ``arm_segments``, default 4).

        Args:
            joint_angles: 1D joint angle vector (must be non-empty).

        Returns:
            Dict with key ``position`` (list[float] of length 2:
            ``[x, y]``).

        Failure mode: returns ``{"error": "robotics_forward_kinematics: ..."}``.
        """
        arr = np.asarray(joint_angles, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"joint_angles must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "joint_angles")
        pos = self.model.forward_kinematics(arr)
        return {"position": _to_py(pos)}

    @_error_to_dict
    def robotics_inverse_kinematics(
        self,
        target: list[float],
        seed: list[float] | None = None,
    ) -> dict:
        """Inverse kinematics via damped least squares.

        Args:
            target: 1D target position (must be length 2; extras are
                truncated).
            seed: Optional 1D seed joint angle vector.

        Returns:
            Dict with keys ``joint_angles`` (list[float]),
            ``success`` (bool), ``iterations`` (int).

        Failure mode: returns ``{"error": "robotics_inverse_kinematics: ..."}``.
        """
        tgt = np.asarray(target, dtype=float)
        if tgt.ndim != 1 or tgt.size < 2:
            raise ValueError(
                f"target must be 1D with len>=2, got shape {tgt.shape}"
            )
        _ensure_finite(tgt, "target")
        seed_arr = None
        if seed is not None:
            seed_arr = np.asarray(seed, dtype=float)
            _ensure_finite(seed_arr, "seed")
        return _to_py(self.model.inverse_kinematics(tgt, seed=seed_arr))

    @_error_to_dict
    def robotics_fuse_sensors(
        self,
        measurements: list[list[float]],
        variances: list[float],
    ) -> dict:
        """Inverse-variance weighted sensor fusion.

        Args:
            measurements: List of 1D sensor readings (>= 1).
            variances: Per-sensor variance (>= 1). Must be same length
                as ``measurements``, each > 0.

        Returns:
            Dict with key ``fused`` (list[float] of fused measurements).

        Failure mode: returns ``{"error": "robotics_fuse_sensors: ..."}``.
        """
        if not isinstance(measurements, list) or len(measurements) < 1:
            raise ValueError("measurements must be a list of >= 1 vectors")
        if not isinstance(variances, list) or len(variances) != len(measurements):
            raise ValueError(
                f"variances (len={len(variances) if hasattr(variances, '__len__') else 'N/A'}) "
                f"must match measurements (len={len(measurements)})"
            )
        meas_arrs = []
        for i, m in enumerate(measurements):
            arr = np.asarray(m, dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"measurements[{i}] must be 1D, got shape {arr.shape}")
            _ensure_finite(arr, f"measurements[{i}]")
            meas_arrs.append(arr)
        for i, v in enumerate(variances):
            v_f = float(v)
            if not math.isfinite(v_f) or v_f <= 0.0:
                raise ValueError(f"variances[{i}] must be finite and > 0, got {v}")
        fused = self.model.fuse_sensors(meas_arrs, [float(v) for v in variances])
        return {"fused": _to_py(fused)}

    @_error_to_dict
    def robotics_update_kalman(
        self,
        prior: list[float],
        prior_var: float,
        measurement: list[float],
        meas_var: float,
    ) -> dict:
        """Sequential Kalman-style Bayesian update.

        Args:
            prior: 1D prior estimate (must be non-empty).
            prior_var: Prior variance (must be > 0).
            measurement: 1D measurement (must be non-empty).
            meas_var: Measurement variance (must be > 0).

        Returns:
            Dict with keys ``estimate`` (list[float]), ``variance`` (float).

        Failure mode: returns ``{"error": "robotics_update_kalman: ..."}``.
        """
        prior_arr = np.asarray(prior, dtype=float)
        meas_arr = np.asarray(measurement, dtype=float)
        if prior_arr.ndim != 1 or prior_arr.size < 1:
            raise ValueError(
                f"prior must be 1D with len>=1, got shape {prior_arr.shape}"
            )
        if meas_arr.ndim != 1 or meas_arr.size < 1:
            raise ValueError(
                f"measurement must be 1D with len>=1, got shape {meas_arr.shape}"
            )
        _ensure_finite(prior_arr, "prior")
        _ensure_finite(meas_arr, "measurement")
        pv = float(prior_var)
        mv = float(meas_var)
        if not math.isfinite(pv) or pv <= 0.0:
            raise ValueError(f"prior_var must be finite and > 0, got {pv}")
        if not math.isfinite(mv) or mv <= 0.0:
            raise ValueError(f"meas_var must be finite and > 0, got {mv}")
        return _to_py(self.model.update_kalman(prior_arr, pv, meas_arr, mv))

    @_error_to_dict
    def robotics_generate_gait(
        self, n_steps: int = 100, gait_type: str = "walk"
    ) -> dict:
        """Generate a periodic gait pattern (walk / trot / bound).

        Args:
            n_steps: Number of gait samples (must be >= 1).
            gait_type: One of ``"walk"``, ``"trot"``, ``"bound"``.
                Unknown values default to ``"walk"``.

        Returns:
            Dict with keys ``joint_angles`` (2D list shape
            ``(n_steps, 8)``), ``foot_contacts`` (2D list shape
            ``(n_steps, 4)``), ``period`` (float).

        Failure mode: returns ``{"error": "robotics_generate_gait: ..."}``.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if gait_type not in ("walk", "trot", "bound"):
            raise ValueError(
                f"gait_type must be 'walk' | 'trot' | 'bound', got {gait_type!r}"
            )
        return _to_py(
            self.model.generate_gait(n_steps=int(n_steps), gait_type=gait_type)
        )

    @_error_to_dict
    def robotics_optimize_trajectory(
        self, trajectory: list[list[float]], n_iter: int = 10
    ) -> dict:
        """Smooth a trajectory by minimizing jerk (gradient descent).

        Args:
            trajectory: 2D trajectory array of shape ``(n_steps, n_dof)``
                with at least 4 rows (must be >= 4 for 3rd differences).
            n_iter: Number of optimization iterations (must be >= 1).

        Returns:
            Dict with keys ``optimized`` (2D list, same shape as input),
            ``jerk`` (float), ``improvement`` (float in [0, 1]).

        Failure mode: returns ``{"error": "robotics_optimize_trajectory: ..."}``.
        """
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim != 2 or traj.shape[0] < 4:
            raise ValueError(
                f"trajectory must be 2D with >= 4 rows, got shape {traj.shape}"
            )
        _ensure_finite(traj, "trajectory")
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        return _to_py(self.model.optimize_trajectory(traj, n_iter=int(n_iter)))

    @_error_to_dict
    def robotics_check_collision(
        self,
        obstacles: list[list[float]],
        position: list[float],
        radius: float = 0.1,
    ) -> dict:
        """Check collision at a single position against circular obstacles.

        Args:
            obstacles: List of obstacle rows, each of the form
                ``[center_x, center_y, ..., radius]`` — the LAST element
                is the obstacle radius, the leading elements are the
                center coordinates. Empty list is allowed (no obstacles).
            position: 1D body position (must be non-empty).
            radius: Body radius (must be >= 0).

        Returns:
            Dict with keys ``collision`` (bool),
            ``nearest_obstacle`` (int, -1 if no obstacles),
            ``distance`` (float, signed clearance; negative = penetration).

        Failure mode: returns ``{"error": "robotics_check_collision: ..."}``.
        """
        obs = _parse_obstacles(obstacles) if obstacles else []
        pos = np.asarray(position, dtype=float)
        if pos.ndim != 1 or pos.size < 1:
            raise ValueError(
                f"position must be 1D with len>=1, got shape {pos.shape}"
            )
        _ensure_finite(pos, "position")
        r = float(radius)
        if not math.isfinite(r) or r < 0.0:
            raise ValueError(f"radius must be finite and >= 0, got {r}")
        return _to_py(self.model.check_collision(obs, pos, radius=r))

    @_error_to_dict
    def robotics_check_path_collision(
        self,
        obstacles: list[list[float]],
        path: list[list[float]],
        radius: float = 0.1,
    ) -> dict:
        """Check collision along a path of body positions.

        Args:
            obstacles: Same format as ``robotics_check_collision``.
            path: 2D path array of shape ``(n_steps, n_dof)``.
            radius: Body radius (must be >= 0).

        Returns:
            Dict with keys ``collision`` (bool),
            ``first_collision_step`` (int, -1 if no collision),
            ``nearest_obstacle`` (int), ``min_clearance`` (float).

        Failure mode: returns ``{"error": "robotics_check_path_collision: ..."}``.
        """
        obs = _parse_obstacles(obstacles) if obstacles else []
        p = np.asarray(path, dtype=float)
        if p.ndim != 2 or p.shape[0] < 1:
            raise ValueError(
                f"path must be 2D with >= 1 row, got shape {p.shape}"
            )
        _ensure_finite(p, "path")
        r = float(radius)
        if not math.isfinite(r) or r < 0.0:
            raise ValueError(f"radius must be finite and >= 0, got {r}")
        return _to_py(self.model.check_path_collision(obs, p, radius=r))

    @_error_to_dict
    def robotics_control_mpc(
        self,
        current_state: list[float],
        target_state: list[float],
        obstacles: list[list[float]] | None = None,
    ) -> dict:
        """Pick the next control action via model predictive control.

        Evaluates 9 candidate actions (8 directions + zero) over a
        prediction horizon and picks the action with the lowest predicted
        cost (tracking error + collision penalty + free-energy surprisal).

        Args:
            current_state: 1D current state (must be non-empty).
            target_state: 1D target state (must be non-empty).
            obstacles: Optional obstacles in the same row format as
                ``robotics_check_collision``.

        Returns:
            Dict with keys ``action`` (list[float] of length 2),
            ``predicted_trajectory`` (2D list), ``cost`` (float).

        Failure mode: returns ``{"error": "robotics_control_mpc: ..."}``.
        """
        current = np.asarray(current_state, dtype=float)
        target = np.asarray(target_state, dtype=float)
        if current.ndim != 1 or current.size < 1:
            raise ValueError(
                f"current_state must be 1D with len>=1, got shape {current.shape}"
            )
        if target.ndim != 1 or target.size < 1:
            raise ValueError(
                f"target_state must be 1D with len>=1, got shape {target.shape}"
            )
        _ensure_finite(current, "current_state")
        _ensure_finite(target, "target_state")
        obs = None
        if obstacles is not None:
            obs = _parse_obstacles(obstacles)
        return _to_py(self.model.control_mpc(current, target, obstacles=obs))

    # ------------------------------------------------------------------
    # Phase 7 — Time tools.
    # ------------------------------------------------------------------

    @_error_to_dict
    def time_encode(self, series: list[float]) -> dict:
        """Encode a 1D time series into a dim-length L2-normalized vector.

        Args:
            series: 1D time series (must be non-empty). 2D input is
                flattened internally.

        Returns:
            Dict with key ``embedding`` (list[float] of length ``dim``).
            Empty input returns a zero vector of length ``dim``.

        Failure mode: returns ``{"error": "time_encode: ..."}``.
        """
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"series must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "series")
        emb = self.model.encode_time_series(arr)
        return {"embedding": _to_py(emb)}

    @_error_to_dict
    def time_detect_seasonality(self, series: list[float]) -> dict:
        """Detect seasonal periods via autocorrelation peaks.

        Args:
            series: 1D time series (must be non-empty).

        Returns:
            Dict with keys ``periods`` (list[int]), ``strengths``
            (list[float]), ``dominant_period`` (int, 0 if none found).
            ``n < 4`` returns empty lists and ``dominant_period=0``.

        Failure mode: returns ``{"error": "time_detect_seasonality: ..."}``.
        """
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"series must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "series")
        return _to_py(self.model.detect_seasonality(arr))

    @_error_to_dict
    def time_analyze_frequency(self, series: list[float]) -> dict:
        """Analyze FFT frequency content + spectral entropy.

        Args:
            series: 1D time series (must be non-empty).

        Returns:
            Dict with keys ``frequencies`` (list[float]),
            ``power`` (list[float]), ``dominant_freq`` (float),
            ``spectral_entropy`` (float in [0, 1]). ``n < 2`` returns
            empty arrays and zeroed scalars.

        Failure mode: returns ``{"error": "time_analyze_frequency: ..."}``.
        """
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"series must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "series")
        return _to_py(self.model.analyze_frequency(arr))

    @_error_to_dict
    def time_analyze_event_timestamps(self, series: list[float]) -> dict:
        """Analyze event timing: inter-arrival, rate, burstiness.

        Args:
            series: 1D event timestamps (must be non-empty). Timestamps
            are sorted internally, so unsorted input is OK.

        Returns:
            Dict with keys ``inter_arrival`` (list[float] of length
            ``n-1``), ``rate`` (float, events per time unit),
            ``burstiness`` (float in [-1, 1]), ``total_events`` (int).
            ``n < 2`` returns empty inter_arrival and zeroed scalars.

        Failure mode: returns ``{"error": "time_analyze_event_timestamps: ..."}``.
        """
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"series must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "series")
        return _to_py(self.model.analyze_event_timestamps(arr))

    @_error_to_dict
    def time_detect_anomalous_timing(self, series: list[float]) -> dict:
        """Detect anomalous inter-arrival gaps via z-score.

        Args:
            series: 1D event timestamps (must be non-empty).

        Returns:
            Dict with keys ``anomalies`` (list[bool] of length
            ``max(0, n-1)``), ``scores`` (list[float] = |z-score|),
            ``threshold`` (float, ``event_threshold_std`` default 2.0).
            ``ts.size < 3`` returns zeroed arrays of shape
            ``(max(0, n-1),)``.

        Failure mode: returns ``{"error": "time_detect_anomalous_timing: ..."}``.
        """
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"series must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "series")
        return _to_py(self.model.detect_anomalous_timing(arr))

    @_error_to_dict
    def time_track_cycle_phase(
        self, series: list[float], period: int | None = None
    ) -> dict:
        """Track phase within a periodic cycle.

        Args:
            series: 1D time series (must be non-empty).
            period: Optional explicit cycle period. When omitted or
                ``<= 0``, the period is auto-detected via the
                ``SeasonalityDetector``; if detection fails, falls back
                to ``max(2, n // 2)``.

        Returns:
            Dict with keys ``phases`` (list[float] of length ``n``, in
            ``[0, 1)``), ``period`` (int, the period used),
            ``phase_coherence`` (float in [0, 1]). ``n < 4`` returns
            zero phases, ``period=0``, ``coherence=0.0``.

        Failure mode: returns ``{"error": "time_track_cycle_phase: ..."}``.
        """
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"series must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "series")
        p: int | None = None
        if period is not None:
            p_int = int(period)
            if p_int < 0:
                raise ValueError(f"period must be >= 0, got {p_int}")
            p = p_int
        return _to_py(self.model.track_cycle_phase(arr, period=p))

    @_error_to_dict
    def time_score_forecastability(self, series: list[float]) -> dict:
        """Score how forecastable a series is.

        Blends three signals into a [0, 1] score: normalized Shannon
        entropy (sample), stationarity (mean-stability across halves),
        and autocorrelation strength (max autocorr at non-zero lag).

        Args:
            series: 1D time series (must be non-empty).

        Returns:
            Dict with keys ``forecastability`` (float in [0, 1]),
            ``entropy`` (float in [0, 1]), ``stationarity`` (float in
            [0, 1]), ``autocorr_strength`` (float in [0, 1]).
            ``n < forecast_min_samples`` (default 8) returns
            ``{0.0, 1.0, 0.0, 0.0}``.

        Failure mode: returns ``{"error": "time_score_forecastability: ..."}``.
        """
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(
                f"series must be 1D with len>=1, got shape {arr.shape}"
            )
        _ensure_finite(arr, "series")
        return _to_py(self.model.score_forecastability(arr))

    # ------------------------------------------------------------------
    # Phase 7 — Code tools.
    # ------------------------------------------------------------------

    @_error_to_dict
    def code_encode(self, source: str) -> dict:
        """Encode Python source code into a dim-length L2-normalized vector.

        Args:
            source: Python source string (must be non-empty). Syntax
                errors are swallowed — AST-based features produce zero
                contribution, but regex-based features (lines, tokens,
                keyword density) still produce a non-trivial vector.

        Returns:
            Dict with key ``embedding`` (list[float] of length ``dim``).
            Empty source returns a zero vector of length ``dim``.

        Failure mode: returns ``{"error": "code_encode: ..."}``.
        """
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        if not source:
            raise ValueError("source must be non-empty")
        emb = self.model.encode_code(source)
        return {"embedding": _to_py(emb)}

    @_error_to_dict
    def code_analyze_ast(self, source: str) -> dict:
        """Analyze AST structure: functions, classes, imports, complexity.

        Args:
            source: Python source string (must be non-empty).

        Returns:
            Dict with keys ``functions`` (list[str]), ``classes``
            (list[str]), ``imports`` (list[str]), ``complexity`` (int,
            starting at 1), ``max_depth`` (int), ``node_count`` (int).
            Empty source or SyntaxError returns all-zero values.

        Failure mode: returns ``{"error": "code_analyze_ast: ..."}``.
        """
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        if not source:
            raise ValueError("source must be non-empty")
        return _to_py(self.model.analyze_ast(source))

    @_error_to_dict
    def code_compare(self, source_a: str, source_b: str) -> dict:
        """Compare two code snippets via token + structure similarity.

        Overall similarity is ``0.5 * token_overlap + 0.5 *
        structure_similarity``.

        Args:
            source_a: First Python source string (must be non-empty).
            source_b: Second Python source string (must be non-empty).

        Returns:
            Dict with keys ``similarity`` (float in [0, 1]),
            ``token_overlap`` (float in [0, 1], Jaccard index of token
            sets), ``structure_similarity`` (float in [0, 1], cosine
            similarity of AST node-type Counter vectors).

        Failure mode: returns ``{"error": "code_compare: ..."}``.
        """
        if not isinstance(source_a, str) or not isinstance(source_b, str):
            raise ValueError("source_a and source_b must be strings")
        if not source_a or not source_b:
            raise ValueError("source_a and source_b must be non-empty")
        return _to_py(self.model.compare_code(source_a, source_b))

    @_error_to_dict
    def code_detect_defects(self, source: str) -> dict:
        """Detect rule-based defect patterns.

        Scans the AST for three patterns:
        - ``mutable_default``: function default arg is a mutable literal
          (list/dict/set/listcomp).
        - ``bare_except``: ``except:`` clause with no exception type.
        - ``equals_none``: ``x == None`` (should be ``x is None``).

        Args:
            source: Python source string (must be non-empty).

        Returns:
            Dict with keys ``defects`` (list of dicts each with
            ``pattern`` (str), ``line`` (int), ``col`` (int),
            ``snippet`` (str)), and ``total`` (int == len(defects)).
            Empty source or SyntaxError returns ``{"defects": [], "total": 0}``.

        Failure mode: returns ``{"error": "code_detect_defects: ..."}``.
        """
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        if not source:
            raise ValueError("source must be non-empty")
        return _to_py(self.model.detect_code_defects(source))

    @_error_to_dict
    def code_analyze_control_flow(self, source: str) -> dict:
        """Analyze per-function cyclomatic complexity and basic blocks.

        Args:
            source: Python source string (must be non-empty).

        Returns:
            Dict with keys ``functions`` (list of dicts each with
            ``name`` (str), ``complexity`` (int), ``lines`` (int),
            ``basic_blocks`` (int = complexity + 1)),
            ``avg_complexity`` (float), ``max_complexity`` (int),
            ``total_functions`` (int). Empty / SyntaxError / no-functions
            returns zero-state.

        Failure mode: returns ``{"error": "code_analyze_control_flow: ..."}``.
        """
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        if not source:
            raise ValueError("source must be non-empty")
        return _to_py(self.model.analyze_control_flow(source))

    @_error_to_dict
    def code_analyze_style(self, source: str) -> dict:
        """Run rule-based style checks.

        Checks line length (default 100 chars), trailing whitespace,
        mixed indentation (tabs + spaces), and function naming
        (snake_case required).

        Args:
            source: Python source string (must be non-empty).

        Returns:
            Dict with keys ``violations`` (list of dicts each with
            ``rule`` (str), ``line`` (int, 1-indexed), ``message``
            (str)), ``total`` (int == len(violations)), ``score``
            (float in [0, 1], 1.0 = no violations).

        Failure mode: returns ``{"error": "code_analyze_style: ..."}``.
        """
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        if not source:
            raise ValueError("source must be non-empty")
        return _to_py(self.model.analyze_code_style(source))

    @_error_to_dict
    def code_build_dependency_graph(self, source: str) -> dict:
        """Build a module-level import dependency graph.

        The source file is treated as a central ``__main__`` node that
        imports each imported module.

        Args:
            source: Python source string (must be non-empty).

        Returns:
            Dict with keys ``nodes`` (list[str], includes ``__main__``
            at index 0 plus sorted unique imports), ``edges`` (list of
            ``[source, target]`` pairs from ``__main__`` to each import;
            duplicates possible if an import is repeated), ``adjacency``
            (2D list of floats, shape ``n_nodes x n_nodes``), and
            ``n_modules`` (int == len(nodes), including ``__main__``).
            Empty / SyntaxError / no-imports returns empty-state with
            ``adjacency`` shape ``(0, 0)``.

        Failure mode: returns ``{"error": "code_build_dependency_graph: ..."}``.
        """
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        if not source:
            raise ValueError("source must be non-empty")
        return _to_py(self.model.build_dependency_graph(source))

    # ------------------------------------------------------------------
    # Phase 7 — Reasoning tools.
    # ------------------------------------------------------------------

    @_error_to_dict
    def reasoning_infer_logical(
        self,
        facts: dict[str, bool] | None = None,
        rules: list[list[str]] | None = None,
        negation_rules: list[list[str]] | None = None,
    ) -> dict:
        """Forward-chain propositional facts + rules to a fixpoint.

        Applies the supplied ``facts`` and ``rules`` to a fresh model
        instance (state does not persist across calls) and runs
        modus-ponens inference up to ``max_inference_depth`` iterations.

        Args:
            facts: Dict mapping proposition strings to their truth
                values (``True`` / ``False``). Defaults to ``{}``.
            rules: List of ``[antecedent, consequent]`` pairs. When
                ``antecedent`` resolves True, ``consequent`` is set True.
                Defaults to ``[]``.
            negation_rules: List of ``[antecedent, consequent]`` pairs.
                When ``antecedent`` resolves True, ``consequent`` is set
                False. Defaults to ``[]``.

        Returns:
            Dict with keys ``facts`` (dict[str, bool], the resulting
            knowledge base after inference), ``inferences`` (list of
            ``[antecedent, consequent, value]`` triples actually derived),
            ``contradictions`` (list[str], sorted deduplicated
            propositions inferred as both True and False).

        Failure mode: returns ``{"error": "reasoning_infer_logical: ..."}``.
        """
        from zero_data_model.model import ZeroDataModel

        facts = facts or {}
        rules = rules or []
        negation_rules = negation_rules or []
        if not isinstance(facts, dict):
            raise ValueError("facts must be a dict")
        if not isinstance(rules, list):
            raise ValueError("rules must be a list")
        if not isinstance(negation_rules, list):
            raise ValueError("negation_rules must be a list")
        # Fresh model to avoid leaking state across MCP calls.
        m = ZeroDataModel(dim=self.model.dim, seed=self.model._seed)
        for prop, val in facts.items():
            m.add_logical_fact(str(prop), bool(val))
        for r in rules:
            if not isinstance(r, (list, tuple)) or len(r) < 2:
                raise ValueError(f"each rule must be [antecedent, consequent], got {r}")
            m.add_logical_rule(str(r[0]), str(r[1]))
        for r in negation_rules:
            if not isinstance(r, (list, tuple)) or len(r) < 2:
                raise ValueError(
                    f"each negation_rule must be [antecedent, consequent], got {r}"
                )
            m.reasoning_propositional.add_negation_rule(str(r[0]), str(r[1]))
        return _to_py(m.infer_logical())

    @_error_to_dict
    def reasoning_syllogism(self, major: list, minor: list) -> dict:
        """Run a categorical syllogism (Barbara / Celarent / Darii / Ferio).

        Args:
            major: ``[form, subject, predicate]`` for the major premise.
                ``form`` must be one of ``"A"`` (All S are P), ``"E"``
                (No S are P), ``"I"`` (Some S are P), ``"O"`` (Some S
                are not P).
            minor: ``[form, subject, predicate]`` for the minor premise.
                The minor's predicate must equal the major's subject
                (the middle term M).

        Returns:
            Dict with keys ``conclusion`` (str, the rendered conclusion
            like ``"All socrates are mortal"``), ``valid`` (bool), and
            ``form`` (str like ``"Barbara (AAA-1)"`` or ``"invalid"``).
            Invalid forms / mismatched middle terms return
            ``{"conclusion": None, "valid": False, "form": "invalid"}``.

        Failure mode: returns ``{"error": "reasoning_syllogism: ..."}``.
        """
        if not isinstance(major, (list, tuple)) or len(major) != 3:
            raise ValueError("major must be a 3-element [form, subject, predicate]")
        if not isinstance(minor, (list, tuple)) or len(minor) != 3:
            raise ValueError("minor must be a 3-element [form, subject, predicate]")
        return _to_py(self.model.syllogism(tuple(major), tuple(minor)))

    @_error_to_dict
    def reasoning_induct_rule(
        self, examples: list[dict], labels: list[bool]
    ) -> dict:
        """Induce a (key, value) rule from labeled examples.

        Args:
            examples: List of dicts, each mapping feature names to
                values. Must be non-empty and same length as ``labels``.
            labels: List of booleans, ``True`` for positive examples
                (rule should cover them) and ``False`` for negatives.

        Returns:
            Dict with keys ``rule`` (str like ``"color == 'red'"``),
            ``confidence`` (float in [0, 1]), ``support`` (int, number
            of positives matching the rule), ``coverage`` (float in
            [0, 1], fraction of positives covered).
            Empty / mismatched / no-positives returns
            ``{"rule": "", "confidence": 0.0, "support": 0, "coverage": 0.0}``.

        Failure mode: returns ``{"error": "reasoning_induct_rule: ..."}``.
        """
        if not isinstance(examples, list):
            raise ValueError("examples must be a list")
        if not isinstance(labels, list):
            raise ValueError("labels must be a list")
        if len(examples) != len(labels):
            raise ValueError(
                f"examples ({len(examples)}) and labels ({len(labels)}) "
                "must have the same length"
            )
        return _to_py(self.model.induct_rule(examples, [bool(l) for l in labels]))

    @_error_to_dict
    def reasoning_analogize(self, source: dict, target: dict) -> dict:
        """Find a structural analogy between a source and target dict.

        Args:
            source: Dict mapping attribute names to values (numeric,
                string, list, dict, or other hashable). Must be non-empty.
            target: Same shape as ``source``. Must be non-empty.

        Returns:
            Dict with keys ``mapping`` (dict[str, str], source key to
            matched target key), ``similarity`` (float in [0, 1]),
            ``transfer`` (dict[str, value], target key to source value
            for matched numeric pairs only).

        Failure mode: returns ``{"error": "reasoning_analogize: ..."}``.
        """
        if not isinstance(source, dict):
            raise ValueError("source must be a dict")
        if not isinstance(target, dict):
            raise ValueError("target must be a dict")
        if not source:
            raise ValueError("source must be non-empty")
        if not target:
            raise ValueError("target must be non-empty")
        return _to_py(self.model.analogize(source, target))

    @_error_to_dict
    def reasoning_abduce(
        self,
        observation: str,
        hypotheses: list[str],
        priors: list[float] | None = None,
    ) -> dict:
        """Pick the best explanation for an observation.

        Args:
            observation: A free-text observation (tokenized by
                alphanumeric word boundaries, case-insensitive).
            hypotheses: List of candidate hypothesis strings. Must be
                non-empty.
            priors: Optional priors aligned with ``hypotheses``. When
                omitted, uniform priors are used. Must be same length
                as ``hypotheses``.

        Returns:
            Dict with keys ``best`` (str, the winning hypothesis; or
            ``None`` on empty/mismatched priors), ``scores`` (list[float]
            aligned with ``hypotheses``), ``confidence`` (float in
            [0, 1], the normalized score of the winner).

        Failure mode: returns ``{"error": "reasoning_abduce: ..."}``.
        """
        if not isinstance(observation, str):
            raise ValueError("observation must be a string")
        if not isinstance(hypotheses, list):
            raise ValueError("hypotheses must be a list")
        if not hypotheses:
            raise ValueError("hypotheses must be non-empty")
        if priors is not None:
            if not isinstance(priors, list):
                raise ValueError("priors must be a list")
            if len(priors) != len(hypotheses):
                raise ValueError(
                    f"priors ({len(priors)}) must match hypotheses "
                    f"({len(hypotheses)})"
                )
        return _to_py(
            self.model.abduce(observation, hypotheses, priors=priors)
        )

    @_error_to_dict
    def reasoning_conclude_defaults(
        self,
        defaults: list[dict] | None = None,
        facts: dict[str, bool] | None = None,
    ) -> dict:
        """Apply defeasible default rules to a set of facts.

        Args:
            defaults: List of dicts each with ``rule`` (a list
                ``[antecedent, consequent]`` or
                ``[antecedent, consequent, value]`` — value defaults to
                True when omitted) and optional ``exception`` (a list
                whose first element is the exception proposition; when
                True in ``facts``, the rule is defeated). Defaults to
                ``[]``.
            facts: Dict of proposition -> truth value. Missing facts
                are treated as False. Defaults to ``{}``.

        Returns:
            Dict with keys ``conclusions`` (dict[str, bool], the
            non-ambiguous inferred propositions), ``defeated`` (list of
            ``[antecedent, consequent]`` pairs blocked by exceptions),
            ``ambiguous`` (list[str], sorted deduplicated propositions
            inferred as both True and False by different rules).

        Failure mode: returns ``{"error": "reasoning_conclude_defaults: ..."}``.
        """
        from zero_data_model.model import ZeroDataModel

        defaults = defaults or []
        facts = facts or {}
        if not isinstance(defaults, list):
            raise ValueError("defaults must be a list")
        if not isinstance(facts, dict):
            raise ValueError("facts must be a dict")
        # Fresh model to avoid leaking state across MCP calls.
        m = ZeroDataModel(dim=self.model.dim, seed=self.model._seed)
        for d in defaults:
            if not isinstance(d, dict):
                raise ValueError(f"each default must be a dict, got {type(d).__name__}")
            rule = d.get("rule")
            exception = d.get("exception")
            if rule is None:
                raise ValueError("each default must have a 'rule' key")
            if not isinstance(rule, (list, tuple)):
                raise ValueError("rule must be a list")
            rule_t = tuple(rule)
            exc_t = tuple(exception) if exception is not None else None
            m.add_default_rule(rule_t, exception=exc_t)
        return _to_py(m.conclude_defaults(facts))

    @_error_to_dict
    def reasoning_trace_causal(
        self,
        links: list[list[str]] | None = None,
        start: str = "root",
        max_depth: int = 5,
    ) -> dict:
        """Trace a causal chain from a starting node.

        Args:
            links: List of ``[cause, effect]`` pairs defining the causal
                graph. Defaults to ``[]`` (empty graph — ``start`` will
                return a single-node chain).
            start: The starting node for DFS. Coerced to ``str``.
            max_depth: Maximum DFS depth. Default 5. Must be ``>= 0``.

        Returns:
            Dict with keys ``chain`` (list[str], DFS visit order
            starting with ``start``), ``effects`` (list[str], sorted
            distinct downstream effects), ``depth`` (int, max depth
            reached), ``cycles`` (bool, True if a back-edge was
            detected during traversal).

        Failure mode: returns ``{"error": "reasoning_trace_causal: ..."}``.
        """
        from zero_data_model.model import ZeroDataModel

        links = links or []
        if not isinstance(links, list):
            raise ValueError("links must be a list")
        if not isinstance(start, str):
            raise ValueError("start must be a string")
        if not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError(f"max_depth must be a non-negative int, got {max_depth}")
        # Fresh model to avoid leaking state across MCP calls.
        m = ZeroDataModel(dim=self.model.dim, seed=self.model._seed)
        for link in links:
            if not isinstance(link, (list, tuple)) or len(link) < 2:
                raise ValueError(f"each link must be [cause, effect], got {link}")
            m.add_causal_link(str(link[0]), str(link[1]))
        return _to_py(m.trace_causal_chain(start, max_depth=max_depth))

    # ------------------------------------------------------------------
    # Registration / public API.
    # ------------------------------------------------------------------

    def _build_tools(self) -> dict[str, Callable[..., dict]]:
        """Build the name -> bound-callable map for all registered tools."""
        return {
            "think": self.think,
            "classify_text": self.classify_text,
            "text_similarity": self.text_similarity,
            "generate_text": self.generate_text,
            "encode_text": self.encode_text,
            "recognize_pattern": self.recognize_pattern,
            "analyze_shape": self.analyze_shape,
            "forecast": self.forecast,
            "detect_anomalies": self.detect_anomalies,
            "analyze_trend": self.analyze_trend,
            "find_analogies": self.find_analogies,
            "detect_script": self.detect_script,
            "analyze_syntax": self.analyze_syntax,
            "infer_cause": self.infer_cause,
            "detect_change_points": self.detect_change_points,
            "hardware_info": self.hardware_info,
            # Causal emergence tools (spec §2.3).
            "perceive_topology": self.perceive_topology,
            "discover_causal_dynamics": self.discover_causal_dynamics,
            "generate_trajectory": self.generate_trajectory,
            "sample_posterior": self.sample_posterior,
            "recall_memory": self.recall_memory,
            "emergence_cycle": self.emergence_cycle,
            # Phase 6 — Memory / Planning / Multimodal / RL (spec §12).
            "memory_encode": self.memory_encode,
            "memory_retrieve": self.memory_retrieve,
            "memory_consolidate": self.memory_consolidate,
            "planning_trajectory": self.planning_trajectory,
            "planning_decompose": self.planning_decompose,
            "planning_sequence": self.planning_sequence,
            "multimodal_align": self.multimodal_align,
            "multimodal_fuse": self.multimodal_fuse,
            "multimodal_contrastive": self.multimodal_contrastive,
            "rl_step": self.rl_step,
            "rl_train_q": self.rl_train_q,
            "rl_search_mcts": self.rl_search_mcts,
            # Phase 7 — Audio (encode / onsets / pitch / classify / segment / music).
            "audio_encode": self.audio_encode,
            "audio_detect_onsets": self.audio_detect_onsets,
            "audio_detect_pitch": self.audio_detect_pitch,
            "audio_classify": self.audio_classify,
            "audio_segment_speech": self.audio_segment_speech,
            "audio_analyze_music": self.audio_analyze_music,
            # Phase 7 — Graph (encode / communities / path / centrality /
            # isomorphism / track / spanning).
            "graph_encode": self.graph_encode,
            "graph_detect_communities": self.graph_detect_communities,
            "graph_find_path": self.graph_find_path,
            "graph_analyze_centrality": self.graph_analyze_centrality,
            "graph_check_isomorphism": self.graph_check_isomorphism,
            "graph_track_dynamic": self.graph_track_dynamic,
            "graph_extract_spanning_tree": self.graph_extract_spanning_tree,
            # Phase 7 — Robotics (plan_motion / forward / inverse / fuse /
            # kalman / gait / optimize / collision / path-collision / mpc).
            "robotics_plan_motion": self.robotics_plan_motion,
            "robotics_forward_kinematics": self.robotics_forward_kinematics,
            "robotics_inverse_kinematics": self.robotics_inverse_kinematics,
            "robotics_fuse_sensors": self.robotics_fuse_sensors,
            "robotics_update_kalman": self.robotics_update_kalman,
            "robotics_generate_gait": self.robotics_generate_gait,
            "robotics_optimize_trajectory": self.robotics_optimize_trajectory,
            "robotics_check_collision": self.robotics_check_collision,
            "robotics_check_path_collision": self.robotics_check_path_collision,
            "robotics_control_mpc": self.robotics_control_mpc,
            # Phase 7 — Time (encode / seasonality / frequency /
            # events / anomaly / cycle / forecast).
            "time_encode": self.time_encode,
            "time_detect_seasonality": self.time_detect_seasonality,
            "time_analyze_frequency": self.time_analyze_frequency,
            "time_analyze_event_timestamps": self.time_analyze_event_timestamps,
            "time_detect_anomalous_timing": self.time_detect_anomalous_timing,
            "time_track_cycle_phase": self.time_track_cycle_phase,
            "time_score_forecastability": self.time_score_forecastability,
            # Phase 7 — Code (encode / ast / compare / defects /
            # control-flow / style / dependencies).
            "code_encode": self.code_encode,
            "code_analyze_ast": self.code_analyze_ast,
            "code_compare": self.code_compare,
            "code_detect_defects": self.code_detect_defects,
            "code_analyze_control_flow": self.code_analyze_control_flow,
            "code_analyze_style": self.code_analyze_style,
            "code_build_dependency_graph": self.code_build_dependency_graph,
            # Phase 7 — Reasoning (prop-infer / syllogism / induct /
            # analogize / abduce / defaults / causal).
            "reasoning_infer_logical": self.reasoning_infer_logical,
            "reasoning_syllogism": self.reasoning_syllogism,
            "reasoning_induct_rule": self.reasoning_induct_rule,
            "reasoning_analogize": self.reasoning_analogize,
            "reasoning_abduce": self.reasoning_abduce,
            "reasoning_conclude_defaults": self.reasoning_conclude_defaults,
            "reasoning_trace_causal": self.reasoning_trace_causal,
        }

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self.tools.keys())

    def call_tool(self, name: str, **kwargs: Any) -> dict:
        """Call a registered tool by name with keyword arguments.

        Returns ``{"error": str}`` when the tool is unknown or raises.
        """
        if name not in self.tools:
            return {"error": f"Unknown tool: {name}"}
        # The decorated callable already converts exceptions to {"error": ...}.
        return self.tools[name](**kwargs)

    def run(self) -> None:
        """Start the MCP server, or print guidance when the SDK is missing."""
        if self.mcp is not None:  # pragma: no cover - requires mcp installed.
            self.mcp.run()
            return
        print("MCP SDK not installed. Install with: pip install mcp")
        print("Available tools:")
        for name in self.list_tools():
            print(f"  - {name}")


def main() -> int:
    """Console entry point: build a server and run it."""
    ZeroDataMCPServer().run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
