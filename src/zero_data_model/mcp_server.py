"""MCP Server exposing zero-data model capabilities as AI agent tools.

Uses the FastMCP SDK (pip install mcp) if available. When mcp is not installed,
the tool functions are still defined as plain callables for direct use.
"""
from __future__ import annotations

import functools
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
    """Recursively convert numpy values to JSON-serializable Python types."""
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
        return float(obj)
    return obj


def _error_to_dict(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Wrap a tool so any exception is surfaced as ``{"error": str}``.

    ``functools.wraps`` preserves the original signature (via ``__wrapped__``)
    so FastMCP can introspect the real parameter schema, and the docstring is
    retained for use as the MCP tool description.
    """

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface as a dict.
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
        """Detect the dominant script of ``text`` (latin/cyrillic/cjk/arabic/mixed)."""
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
