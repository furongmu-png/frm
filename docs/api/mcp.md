# MCP Tools

ZeroDataModel ships an MCP (Model Context Protocol) server that exposes the
model's capabilities as tools for AI agents. It uses the FastMCP SDK
(`pip install mcp`) when available, and degrades gracefully — when `mcp` is not
installed the tools remain available as plain callables for direct scripting
and testing.

## Running the MCP server

```bash
pip install -e ".[mcp]"
PYTHONPATH=src python -m zero_data_model.mcp_server
# or:
PYTHONPATH=src python -m zero_data_model.mcp_server  # prints available tools if SDK missing
```

## Auto-generated reference

::: zero_data_model.mcp_server.ZeroDataMCPServer

## Tool catalogue

The server registers **40+ tools** mirroring the REST surface. Each tool
accepts JSON-friendly inputs (lists, strings, ints), converts them to numpy as
needed, calls the underlying model, normalises outputs to JSON-serialisable
Python types, and returns a dict. Non-finite floats (NaN, ±Inf) are mapped to
`null` to keep output strict-JSON-safe — MCP clients using strict parsers reject
non-finite floats, and persistence diagrams routinely contain `+Inf`.

Any raised exception is wrapped by the `@_error_to_dict` decorator and surfaced
as `{"error": "<tool_name>: <message>"}` — the full traceback is logged
server-side.

### Core cognitive tools

| Tool | Input | Returns |
| --- | --- | --- |
| `think` | `input_data: list[float] \| None` | `{signal_data, confidence, metadata}` |
| `classify_text` | `text: str` | `{topic, confidence}` |
| `text_similarity` | `text_a, text_b: str` | `{similarity}` |
| `generate_text` | `seed: str, length: int = 32` | `{text}` |
| `encode_text` | `text: str` | `{embedding, dim}` |
| `recognize_pattern` | `image: list[list[float]]` | `{pattern, confidence}` |

### Analytics tools

| Tool | Input | Returns |
| --- | --- | --- |
| `forecast` | `series: list[float], horizon: int = 5` | `{forecast}` |
| `detect_anomalies` | `series: list[float]` | `{anomalies}` |
| `analyze_trend` | `series: list[float]` | `{slope, r2, ...}` |
| `mine_patterns` | `series: list[float]` | `{patterns}` |

### Vision tools

| Tool | Input | Returns |
| --- | --- | --- |
| `encode_image` | `image: list[list[float]]` | `{embedding, dim}` |
| `extract_image_features` | `image: list[list[float]]` | `{features}` |
| `analyze_shape` | `image: list[list[float]]` | `{edges, corners, ...}` |

### Phase 7 upgrade readouts

The Phase 7 modules expose per-module readout tools prefixed `phase7_`:

| Prefix | Tools |
| --- | --- |
| `phase7_architect_*` | `stats`, `dormant`, `reactivate` |
| `phase7_layered_predictor_*` | `predict`, `stats` |
| `phase7_temporal_memory_*` | `recall`, `stats` |
| `phase7_episodic_graph_*` | `insert`, `recent`, `count`, `plan` |
| `phase7_semantic_index_*` | `add`, `search`, `stats` |
| `phase7_logic_layer_*` | `add_rule`, `check_all`, `stats` |
| `phase7_causal_inference_*` | `transition`, `do`, `counterfactual`, `confounders` |
| `phase7_meta_cognition_*` | `confidence`, `uncertainty`, `should_seek_info`, `stats` |
| `phase7_experiment_planner_*` | `candidates`, `select_best`, `record_result`, `stats` |
| `phase7_hypothesis_tester_*` | `hypotheses`, `supported`, `stats` |
| `phase7_world_*` | `collaboration_stats`, `agent_count`, `step_count` |
| `phase7_communication_*` | `emergent_meanings`, `stats` |
| `phase7_culture_*` | `knowledge_curve`, `stats`, `generations` |

## Programmatic usage (no SDK required)

```python
from zero_data_model.mcp_server import ZeroDataMCPServer

server = ZeroDataMCPServer(dim=64)
print(server.list_tools())                       # ['think', 'classify_text', ...]

result = server.call_tool("classify_text", text="the algorithm computes the network")
print(result)                                    # {'topic': 'tech', 'confidence': 0.83}

err = server.call_tool("unknown_tool")
print(err)                                       # {'error': 'Unknown tool: unknown_tool'}
```

## Registering with a real MCP client

When the `mcp` package is installed, each callable is registered with a
`FastMCP("zero-data-model")` server via `self.mcp.tool(name=name)(func)`. The
docstring doubles as the tool description; the explicit `name=` overrides the
private method name. Call `server.run()` to serve the tools over MCP.

```python
server = ZeroDataMCPServer(dim=64)
server.run()        # blocks; serves over MCP
```
