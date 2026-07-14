# tests/test_mcp_server.py
"""Tests for the MCP server wrapper around the zero-data model.

These tests exercise the server through its plain-callable API
(``call_tool`` / ``list_tools``), which works whether or not the optional
``mcp`` package is installed (graceful degradation).
"""
import json

from zero_data_model.mcp_server import ZeroDataMCPServer


def test_server_initializes():
    server = ZeroDataMCPServer(dim=16)
    assert server.model.dim == 16
    # mcp attribute always exists; None when the SDK is absent.
    assert hasattr(server, "mcp")
    assert isinstance(server.tools, dict)


def test_list_tools():
    server = ZeroDataMCPServer(dim=16)
    tools = server.list_tools()
    assert len(tools) >= 16
    # A few representative tool names must be present.
    for expected in (
        "think",
        "classify_text",
        "text_similarity",
        "forecast",
        "detect_anomalies",
        "detect_script",
        "hardware_info",
    ):
        assert expected in tools


def test_call_tool_classify_text():
    server = ZeroDataMCPServer(dim=16)
    result = server.call_tool("classify_text", text="the algorithm computes")
    assert isinstance(result, dict)
    assert "topic" in result
    assert isinstance(result["topic"], str)
    assert "confidence" in result
    assert isinstance(result["confidence"], float)


def test_call_tool_text_similarity():
    server = ZeroDataMCPServer(dim=16)
    result = server.call_tool("text_similarity", text_a="hello", text_b="hi")
    assert isinstance(result, dict)
    assert "similarity" in result
    assert isinstance(result["similarity"], float)
    assert 0.0 <= result["similarity"] <= 1.0


def test_call_tool_forecast():
    server = ZeroDataMCPServer(dim=16)
    result = server.call_tool("forecast", series=[1, 2, 3, 4, 5], horizon=3)
    assert isinstance(result, dict)
    assert "forecast" in result
    assert isinstance(result["forecast"], list)
    assert len(result["forecast"]) == 3
    assert all(isinstance(v, float) for v in result["forecast"])
    assert result["horizon"] == 3


def test_call_tool_detect_anomalies():
    server = ZeroDataMCPServer(dim=16)
    result = server.call_tool("detect_anomalies", series=[1, 1, 1, 10, 1, 1])
    assert isinstance(result, dict)
    assert "anomalies" in result
    assert isinstance(result["anomalies"], list)
    assert all(isinstance(v, bool) for v in result["anomalies"])
    assert len(result["anomalies"]) == 6
    assert "count" in result
    assert isinstance(result["count"], int)


def test_call_tool_think():
    server = ZeroDataMCPServer(dim=16)
    result = server.call_tool("think")
    assert isinstance(result, dict)
    assert "signal_data" in result
    assert isinstance(result["signal_data"], list)
    assert len(result["signal_data"]) == 16
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert "metadata" in result


def test_call_tool_hardware_info():
    server = ZeroDataMCPServer(dim=16)
    result = server.call_tool("hardware_info")
    assert isinstance(result, dict)
    # The hardware report must include the array backend key.
    assert "array_backend" in result
    assert isinstance(result["array_backend"], str)


def test_call_tool_detect_script():
    server = ZeroDataMCPServer(dim=16)
    result = server.call_tool("detect_script", text="hello мир")
    assert isinstance(result, dict)
    assert "script" in result
    assert isinstance(result["script"], str)
    # Mixed latin + cyrillic input resolves to one of the known labels.
    assert result["script"] in {"latin", "cyrillic", "cjk", "arabic", "mixed"}


def test_call_nonexistent_tool_returns_error():
    server = ZeroDataMCPServer(dim=16)
    result = server.call_tool("nonexistent")
    assert isinstance(result, dict)
    assert "error" in result
    assert isinstance(result["error"], str)


def test_tool_outputs_are_json_serializable():
    """Every tool result must round-trip through json.dumps (no numpy types)."""
    server = ZeroDataMCPServer(dim=16)
    cases = [
        ("think", {}),
        ("classify_text", {"text": "the network learns"}),
        ("text_similarity", {"text_a": "abc", "text_b": "abd"}),
        ("generate_text", {"seed": "seed", "length": 8}),
        ("encode_text", {"text": "data"}),
        ("recognize_pattern", {"image": [[0.0, 1.0], [1.0, 0.0]]}),
        ("analyze_shape", {"image": [[0.0, 1.0], [1.0, 0.0]]}),
        ("forecast", {"series": [1.0, 2.0, 3.0], "horizon": 2}),
        ("detect_anomalies", {"series": [1.0, 1.0, 9.0]}),
        ("analyze_trend", {"series": [1.0, 2.0, 3.0, 4.0]}),
        ("find_analogies", {"series_a": [1.0, 2.0], "series_b": [1.0, 2.0]}),
        ("detect_script", {"text": "hello"}),
        ("analyze_syntax", {"text": "the cat sat."}),
        ("infer_cause", {"cause": [1.0, 2.0, 3.0, 4.0], "effect": [2.0, 3.0, 4.0, 5.0]}),
        ("detect_change_points", {"series": [1.0, 1.0, 5.0, 5.0, 1.0]}),
        ("hardware_info", {}),
    ]
    for name, kwargs in cases:
        result = server.call_tool(name, **kwargs)
        assert isinstance(result, dict)
        # Must not raise: confirms no numpy scalars/arrays leaked through.
        json.dumps(result)


def test_call_tool_think_with_input():
    server = ZeroDataMCPServer(dim=16)
    result = server.call_tool("think", input_data=[float(i) for i in range(16)])
    assert isinstance(result, dict)
    assert "signal_data" in result
    assert len(result["signal_data"]) == 16


def test_call_tool_failing_returns_error_dict():
    """A tool that raises internally must surface an error dict, not raise."""
    server = ZeroDataMCPServer(dim=16)
    # Force a genuine failure by replacing the underlying model method.
    original = server.model.classify_text

    def _boom(_text):
        raise RuntimeError("forced failure")

    server.model.classify_text = _boom  # type: ignore[method-assign]
    try:
        result = server.call_tool("classify_text", text="anything")
    finally:
        server.model.classify_text = original  # type: ignore[method-assign]
    assert isinstance(result, dict)
    assert "error" in result
    assert "forced failure" in result["error"]


def test_tools_dict_matches_list_tools():
    server = ZeroDataMCPServer(dim=16)
    assert set(server.tools.keys()) == set(server.list_tools())
    # Each registered value is callable.
    for name, func in server.tools.items():
        assert callable(func), f"{name} is not callable"
        # Docstring is preserved (used as the MCP tool description).
        assert func.__doc__, f"{name} is missing a docstring"
