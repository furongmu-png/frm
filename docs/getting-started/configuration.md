# Configuration

ZeroDataModel is configured at three layers:

1. **Constructor flags** on `ZeroDataModel.__init__` — turn core modules and
   cognitive-upgrade phases on/off.
2. **Environment variables** — runtime behaviour of the FastAPI service
   (auth, CORS, rate limits, observability).
3. **Optional extras** — installed dependencies that unlock acceleration paths
   (see [Installation](installation.md)).

## Constructor feature flags

All flags default to `False`; the baseline `think()` cycle is unchanged when
none are set.

| Flag | Phase | Module | Effect |
| --- | --- | --- | --- |
| `enable_architect` | 1 | `ArchitectureOptimizer` | Module split / prune driven by accumulated prediction error. |
| `enable_layered_predictor` | 2 | `LayeredPredictor`, `TemporalMemory` | Multi-timescale predictive coding + Echo-State-style temporal memory. |
| `enable_episodic_memory` | 3 | `EpisodicGraph`, `SemanticIndex` | Episode graph + retrieval index over belief vectors. |
| `enable_logic_layer` | 4 | `LogicLayer`, `CausalInference` | Rule constraints (Gödel t-norm) + causal transition inference. |
| `enable_meta_cognition` | 5 | `MetaCognition` | Second-order beliefs over uncertainty and parameter drift. |
| `enable_experiment_planner` | 6 | `BayesianExperimentPlanner`, `HypothesisTester` | Bayesian-optimal experiment selection + Bayes-factor testing. |
| `enable_multiagent` | 7 | `MultiAgentWorld`, `CommunicationChannel`, `CulturePropagation` | Multi-agent world (library modules; `think()` hook staged). |

```python
from zero_data_model.model import ZeroDataModel

m = ZeroDataModel(
    dim=64,
    seed=42,                       # reproducible cycles
    enable_architect=True,
    enable_meta_cognition=True,
)
```

When a phase is enabled, the corresponding module is **lazily imported** and
constructed; when disabled it is left as `None` so the import cost is paid only
when used.

## Environment variables

These govern the FastAPI service (`zero_data_model.api`). The Docker / K8s
deployments set them via ConfigMap / Secret; for local development export them
in your shell.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ZDM_API_KEY` | *(unset — auth disabled)* | Set to a non-empty value to enable `X-API-Key` authentication. Use `openssl rand -hex 32`. |
| `ZDM_CORS_ORIGINS` | `http://localhost:8000,http://localhost:3000` | Comma-separated list of allowed CORS origins. |
| `ZDM_RATE_LIMIT_RPM` | `60` | Sustained per-client request ceiling (requests per minute). |
| `ZDM_RATE_LIMIT_BURST` | `10` | Token-bucket burst capacity (max requests in a short window). |
| `ZDM_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated TrustedHost allow-list. |
| `ZDM_ENV` | `production` | `development` enables `/docs`, `/redoc`, `/openapi.json`; `production` hides them. Must be one of `development`/`dev`/`production`/`prod` or startup raises. |
| `ZDM_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`/`INFO`/`WARNING`/`ERROR`). |
| `ZDM_DIM` | `64` | (K8s ConfigMap) model vector dimension passed to the backend. |
| `ZDM_ENABLE_ARCHITECT` | `false` | (K8s ConfigMap) enable the plasticity module in the backend. |
| `ZDM_ENABLE_LAYERED_PREDICTOR` | `false` | (K8s ConfigMap) enable the cogtime module. |
| `ZDM_ENABLE_CAUSAL_EMERGENCE` | `false` | (K8s ConfigMap) enable the causal-emergence engine. |
| `ZDM_ENABLE_METACOG` | `false` | (K8s ConfigMap) enable the meta-cognition module. |
| `ZDM_ENABLE_MULTIAGENT` | `false` | (K8s ConfigMap) enable the multi-agent module. |
| `IBM_QUANTUM_TOKEN` | *(unset)* | IBM Quantum token for the real-hardware quantum backend (optional). |

### Production example

```bash
export ZDM_API_KEY=$(openssl rand -hex 32)
export ZDM_CORS_ORIGINS="https://app.example.com"
export ZDM_RATE_LIMIT_RPM=120
export ZDM_RATE_LIMIT_BURST=20
export ZDM_ENV=production
PYTHONPATH=src uvicorn zero_data_model.api:app --port 8000
```

### Development example

```bash
export ZDM_ENV=development        # exposes /docs, /redoc, /openapi.json
# ZDM_API_KEY left unset — auth disabled, every endpoint open
PYTHONPATH=src uvicorn zero_data_model.api:app --port 8000 --reload
```

## Security middleware behaviour

`create_app()` assembles a layered middleware stack (outermost first):

```
SecurityHeaders -> RequestSizeLimit -> rate_limit
  -> TrustedHost -> CORS -> request_tracing -> _limit_body_size
  -> slowapi -> endpoint
```

- **API-key auth** — every cognitive endpoint is gated by an `X-API-Key` header
  checked in constant time (`hmac.compare_digest`). `/health`, `/ready`,
  `/metrics`, and the docs routes stay public so probes and Prometheus scrapes
  work without a key.
- **Rate limiting** — token-bucket per-client (identified by `X-Forwarded-For`
  behind an ingress, else the socket peer). Health/metrics/docs paths are
  exempt. WebSocket endpoints bypass the HTTP middleware chain and must call
  `RateLimiter.check` in their own handler.
- **Request size limit** — declared `Content-Length` over 10 MiB is rejected
  with `413` before the body is read; a stricter 4 MiB per-route cap applies
  inside `_limit_body_size`.
- **Security response headers** — `X-Content-Type-Options`, `X-Frame-Options`,
  `X-XSS-Protection`, `Referrer-Policy`, `Cache-Control: no-store` on every
  response, plus `Strict-Transport-Security` (HSTS) on HTTPS only.
- **CORS / TrustedHost** — origins and hosts are env-configurable; the default
  CORS policy is `allow_credentials=False`.

Authentication is **opt-in**: when `ZDM_API_KEY` is unset the API runs in
development mode with no auth (every endpoint is open). Set it to enable
authentication.
