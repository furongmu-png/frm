# REST API Reference

ZeroDataModel ships a FastAPI service that exposes the model as HTTP endpoints.
The app holds a single module-level `ZeroDataModel` instance that is lazily
initialized on first use (`dim=32` for fast startup). Numpy arrays returned by
the model are converted to plain Python lists so they serialize to JSON cleanly.

## Running the service

```bash
pip install -e ".[web]"
PYTHONPATH=src uvicorn zero_data_model.api:app --port 8000
# Swagger UI: http://localhost:8000/docs   (only when ZDM_ENV=development)
# ReDoc:      http://localhost:8000/redoc
# OpenAPI:    http://localhost:8000/openapi.json
```

Set `ZDM_ENV=production` (the default) to hide the docs routes. Set `ZDM_API_KEY`
to enable `X-API-Key` authentication on every cognitive endpoint. See
[Configuration](../getting-started/configuration.md) for the full env-var table.

## Versioning

The REST API is versioned under the `/v1` path prefix. This isolates breaking
changes — a future `/v2` can ship without disturbing existing callers — and
gives clients a stable, machine-readable signal for deprecation.

### Canonical paths

Every business endpoint (`/think`, `/classify`, `/architect/stats`,
`/emergence/cycle`, …) is served under `/v1`. The canonical, documented form
is:

```
POST /v1/think
POST /v1/classify
GET  /v1/architect/stats
POST /v1/emergence/cycle
…
```

A small set of operational endpoints intentionally stay at the **root** and
are not versioned (so k8s liveness probes, Prometheus scrapers, and the
Swagger UI don't have to track versions):

| Method | Root path | Also under `/v1`? | Notes |
| --- | --- | --- | --- |
| `GET` | `/health` | yes | Liveness probe. |
| `GET` | `/ready` | yes | Readiness probe. |
| `GET` | `/` | no | Root index (status + hardware info). |
| `GET` | `/metrics` | no | Prometheus scrape (auth-gated). |
| `GET` | `/docs`, `/redoc`, `/openapi.json` | no | Swagger UI / ReDoc / spec (dev mode only). |

Health and readiness are the only endpoints reachable at **both** root and
`/v1` — probes must never be version-gated.

### Version info endpoint

```
GET /v1/version
```

```json
{
  "version": "1.0.0",
  "supported_versions": ["1"],
  "deprecated_versions": [],
  "latest": "1"
}
```

### Version negotiation (Accept header)

A caller may request a specific API version by sending a vendor Accept header.
Today only `v1` is implemented; asking for a future version short-circuits
before auth/CORS with a `501 Not Implemented`:

```bash
curl -s -X POST http://localhost:8000/v1/think \
  -H "Accept: application/vnd.zdm.v2+json"
# -> 501 {"detail": "v2 not yet implemented. Use v1."}
```

Supported Accept values:

| Accept | Behaviour |
| --- | --- |
| (default) `application/json` | Served by `v1`. |
| `application/vnd.zdm.v1+json` | Served by `v1` (explicit). |
| `application/vnd.zdm.v2+json` | `501 Not Implemented` (future). |

### Backward compatibility & deprecation

Every business endpoint registered on `/v1` is **also** mirrored at the root
path (`POST /think` works alongside `POST /v1/think`). The legacy root mirror
is hidden from the OpenAPI schema (`include_in_schema=False`) and is flagged
deprecated so callers can migrate:

```
$ curl -i -X POST http://localhost:8000/think
HTTP/1.1 200 OK
X-Deprecated: true
Link: </v1/think>; rel="successor-version"
…
```

The deprecation marker is added by the `deprecate_root_paths` middleware for any
path that is **not** under `/v1` and **not** in the exempt set
(`/`, `/health`, `/ready`, `/version`, `/metrics`, `/docs`, `/redoc`,
`/openapi.json`). Clients should treat the presence of `X-Deprecated: true`
as a signal to switch to the `/v1` successor advertised in the `Link` header.

The legacy root mount will be removed in a future major release; new code
should target `/v1` directly.

### SDK & frontend defaults

- **Python SDK**: `ZeroDataModelClient()` defaults its `base_url` to
  `http://localhost:8000/v1`. Passing `base_url="http://host:port"` (no `/v1`)
  will hit the deprecated root mount and receive `X-Deprecated` headers.
- **Frontend (vite + nginx)**: `apiUrl("/think")` returns `/api/v1/think` in
  proxied mode. The nginx ingress rewrites `/api/* → /v1/*` (stripping `/api`),
  so the backend receives `/v1/think`. In direct mode, `VITE_API_BASE` must
  include the version prefix, e.g. `http://localhost:8000/v1`.

## Auto-generated reference

The block below is rendered by mkdocstrings directly from the source docstrings
in `src/zero_data_model/api.py` and `src/zero_data_model/model.py`.

### `create_app`

::: zero_data_model.api.create_app

### `ZeroDataModel`

The integration entry point imported into the API module. See
[Python SDK](python-sdk.md) for the full method catalogue.

::: zero_data_model.api.ZeroDataModel

## Endpoint catalogue

The full service registers 80+ endpoints across the cognitive core, Phase 6
extended capabilities (memory, planning, multimodal, rl, audio, graph, robotics,
time, code, reasoning, causal), the causal-emergence engine, and the Phase 7
upgrade readouts. The tables below summarise the most-used ones; the live
OpenAPI spec at `/openapi.json` is authoritative.

### Health & operations

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | public | Liveness probe. Cheap: no model construction. Returns 503 while shutting down. |
| `GET` | `/ready` | public | Readiness probe. Constructs the model if needed. |
| `GET` | `/` | API key | Root: status + active hardware backends. |
| `GET` | `/metrics` | API key | Prometheus metrics (24 `zdm_*` gauges/histograms). 503 if prometheus_client not installed. |
| `POST` | `/save` | API key | Persist the model to disk (`name` is a relative path, sandboxed). |
| `POST` | `/load` | API key | Load a previously saved model. |

### Cognitive core

| Method | Path | Tag | Rate limit | Description |
| --- | --- | --- | --- | --- |
| `POST` | `/think` | cognitive | 10/min | Run one thought cycle. With no `input` the model self-generates. Async: offloaded to a thread pool. |
| `POST` | `/classify` | nlp | 30/min | Zero-shot classify `text` into a topic. |
| `POST` | `/similarity` | nlp | 30/min | Semantic similarity between two texts in [0, 1]. |
| `POST` | `/generate` | nlp | 30/min | Generate text from a seed. |
| `POST` | `/forecast` | analytics | 30/min | Forecast a time series `horizon` steps ahead. |
| `POST` | `/anomalies` | analytics | 30/min | Detect anomalies in a series. |
| `POST` | `/trend` | analytics | 30/min | Analyse the trend of a series. |
| `POST` | `/recognize` | vision | 30/min | Recognize a shape/pattern in a 2D image. |

### Causal-emergence engine

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/emergence/perceive` | Topological feature extraction (Betti numbers, persistence). |
| `POST` | `/emergence/causal` | Causal structure discovery. |
| `POST` | `/emergence/trajectory` | Generate a trajectory through belief space. |
| `POST` | `/emergence/sample` | Sample from the posterior. |
| `POST` | `/emergence/recall` | Recall from chaotic-memory attractor store. |
| `POST` | `/emergence/cycle` | Run one full emergence cycle. |

### Phase 7 upgrade readouts

| Method | Path | Tag | Description |
| --- | --- | --- | --- |
| `GET` | `/architect/stats` | architect | ArchitectureOptimizer error history + pending plan. |
| `GET` | `/architect/dormant` | architect | List dormant (pruned) modules. |
| `POST` | `/architect/reactivate` | architect | Reactivate a dormant module by name. |
| `POST` | `/episodic/plan` | episodic | Plan a trajectory through the episodic graph. |
| `POST` | `/semantic/query` | semantic | Semantic similarity retrieval over belief vectors. |
| `POST` | `/logic/rule` | logic | Add a symbolic rule to the `LogicLayer`. |
| `POST` | `/logic/predicate` | logic | Add a logical predicate. |
| `POST` | `/causal/transition` | causal | Update the causal transition matrix. |
| `POST` | `/causal/do` | causal | Apply a do-intervention. |
| `POST` | `/causal/counterfactual` | causal | Counterfactual query. |
| `POST` | `/causal/confounders` | causal | List confounders. |
| `POST` | `/experiment/select-best` | experiment | Pick the next best experiment (Bayesian). |
| `POST` | `/experiment/record` | experiment | Record an experiment outcome. |

### Phase 6 extended capabilities

Each tag below has 3–11 endpoints. See `/openapi.json` for the full request
schemas.

| Tag | Endpoints | Sample paths |
| --- | --- | --- |
| `memory` | 3 | `/memory/encode`, `/memory/retrieve`, `/memory/consolidate` |
| `planning` | 3 | `/planning/trajectory`, `/planning/decompose`, `/planning/sequence` |
| `multimodal` | 3 | `/multimodal/align`, `/multimodal/fuse`, `/multimodal/contrastive` |
| `rl` | 3 | `/rl/step`, `/rl/train-q`, `/rl/search-mcts` |
| `audio` | 6 | `/audio/encode`, `/audio/onsets`, `/audio/pitch`, `/audio/classify`, `/audio/segment`, `/audio/music` |
| `graph` | 7 | `/graph/encode`, `/graph/communities`, `/graph/path`, `/graph/centrality`, `/graph/isomorphism`, `/graph/track`, `/graph/spanning` |
| `robotics` | 11 | `/robotics/{motion,forward,inverse,fuse,kalman,gait,optimize,collision,path-collision,mpc}` |
| `time` | 8 | `/time/{encode,seasonality,frequency,events,anomaly,cycle,forecast}` |
| `code` | 7 | `/code/{encode,ast,compare,defects,control-flow,style,dependencies}` |
| `reasoning` | 7 | `/reasoning/{prop-infer,syllogism,induct,analogize,abduce,defaults,causal}` |
| `causal` | 7 | `/causal/{decision-tree,game,counterfactual,bandit,pomdp,discover-graph,intervene}` |

## Examples

### Run a self-generated thought

```bash
curl -s -X POST http://localhost:8000/v1/think \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ZDM_API_KEY" \
  -d '{}'
```

```json
{
  "cycle": 1,
  "output": [0.0123, -0.0456, ...],
  "confidence": 0.78
}
```

### Classify text

```bash
curl -s -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ZDM_API_KEY" \
  -d '{"text": "the algorithm computes the network"}'
```

```json
{ "topic": "tech", "confidence": 0.83 }
```

### Python (httpx)

```python
import httpx

API = "http://localhost:8000/v1"
HEADERS = {"X-API-Key": "your-key"}

r = httpx.post(f"{API}/think", headers=HEADERS, json={}, timeout=30)
print(r.json())
```

> The legacy root paths (`POST /think`, `POST /classify`, …) still work but
> return `X-Deprecated: true` + a `Link` to the `/v1` successor. New code should
> target `/v1` directly.

## Error responses

All errors share a uniform body shape:

```json
{ "detail": "invalid or missing API key", "request_id": "abc123" }
```

| Status | When |
| --- | --- |
| `401` | Missing or wrong `X-API-Key` (when `ZDM_API_KEY` is set). |
| `413` | Request `Content-Length` exceeds 10 MiB (or 4 MiB per-route cap). |
| `422` | Pydantic validation error on the request body. |
| `429` | Rate limit exceeded (`Retry-After: 60` header set). |
| `500` | Unhandled exception. Only the exception **type** is logged server-side; the body returns a generic `internal error` (CWE-209: no internal leakage). |
| `501` | `Accept: application/vnd.zdm.v2+json` (or any not-yet-implemented version) — version negotiation short-circuit. |
| `503` | `/health` while shutting down; `/ready` while model not ready; `/metrics` when prometheus_client not installed. |
