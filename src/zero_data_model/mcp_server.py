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
