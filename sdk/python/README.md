# ZeroDataModel Python SDK

Official synchronous Python client for the ZeroDataModel REST API. Covers the
core cognitive endpoints (`/think`, `/health`, `/ready`, `/knowledge-graph`,
`/story-milestones`) and the phase-7 cognitive-upgrade modules (`/architect`,
`/temporal_memory`, `/layered_predictor`, `/episodic_graph`, `/semantic_index`,
`/logic_layer`, `/causal_inference`, `/meta_cognition`, `/experiment_planner`,
`/hypothesis_tester`, `/world`, `/communication`, `/culture`).

## Install

```bash
pip install zero-data-model-client
```

## Quick start

```python
from zero_data_model_client import ZeroDataModelClient

with ZeroDataModelClient(base_url="http://localhost:8000", api_key="...") as client:
    health = client.health()
    result = client.think(cycles=1)
    print(result["free_energy"])

    # Phase-7 example
    beliefs = client.layered_predictor_beliefs()
    client.architect_action("activate", "memory")
```

## Authentication

If `api_key` is provided it is sent as the `X-API-Key` header on every request.
Health and readiness endpoints are typically unauthenticated server-side.

## Error handling

All failures derive from `ZeroDataModelError`:

| HTTP status | Exception |
| ----------- | --------- |
| 401 / 403   | `AuthenticationError` |
| 404         | `NotFoundError` |
| 422         | `ValidationError` |
| 429         | `RateLimitError` (exposes `.retry_after`) |
| 5xx         | `ServerError` |
| transport   | `ZeroDataModelError` (wrapped) |

```python
from zero_data_model_client import ZeroDataModelClient, RateLimitError, ZeroDataModelError

try:
    client.think(cycles=1)
except RateLimitError as e:
    wait = e.retry_after
except ZeroDataModelError as e:
    ...
```

## Configuration

| Option | Default | Description |
| ------ | ------- | ----------- |
| `base_url` | `http://localhost:8000` | API base URL |
| `api_key` | `None` | Optional API key |
| `timeout` | `30.0` | Per-request timeout (seconds) |

## Development

```bash
cd sdk/python
pip install -e ".[test]"
pytest tests/ -xvs
```
