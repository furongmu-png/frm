# Zero-Data Model

> A self-sufficient cognitive system that requires **no external training data**.

The Zero-Data Model composes six theory-driven core modules (consciousness,
active inference, category theory, quantum-classical hybrid, biological
substrate, mathematical universe) into twelve domain capabilities (NLP, vision,
analytics). It self-generates its inputs via DNA crossover, fractal iteration
and morphogenesis, and augments them with small hand-written rule priors
instead of learned weights.

![Python 3.14](https://img.shields.io/badge/python-3.14-blue)
![numpy](https://img.shields.io/badge/numpy-required-green)
![qiskit](https://img.shields.io/badge/qiskit-optional-orange)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

## Highlights

- **Zero-data by construction** — `think()` with no argument self-generates an
  input from `DNAStorage.generate` + `FractalGenerator.generate`; no datasets
  and no pre-trained weights are ever loaded.
- **6 core modules + 12 domain capabilities** — ConsciousnessCore,
  ActiveInferenceEngine, CategoryTheoryEngine, QuantumClassicalHybrid,
  BiologicalSubstrate, MathematicalUniverse, composed into NLP / Vision /
  Analytics capabilities backed by `NLPRules` / `VisionRules` / `AnalyticsRules`.
- **Hardware acceleration with graceful degradation** — real Qiskit
  variational circuits (`StatevectorSampler`), numba-JIT annealing, GPU arrays
  via CuPy, and joblib parallelism; every path falls back to pure NumPy.
- **7-phase cognitive-upgrade layer** — an optional, feature-flagged stack
  layered on top of the six core modules. All upgrades default to **off** so
  the baseline is zero-regression; enable them individually to grow the system
  from a fixed architecture into a fully plastic, time-layered, structured,
  neuro-symbolic, meta-cognitive, actively experimenting, multi-agent one:
  - **Architecture plasticity** (`ArchitectureOptimizer`) — module split / prune
    driven by accumulated prediction error.
  - **Layered time** (`LayeredPredictor`, `TemporalMemory`) — multi-timescale
    predictive coding plus an Echo-State-style temporal memory.
  - **Structured memory** (`EpisodicGraph`, `SemanticIndex`) — episode graph +
    retrieval index over belief vectors.
  - **Neuro-symbolic fusion** (`LogicLayer`, `CausalInference`) — rule
    constraints + causal transition inference.
  - **Meta-cognition** (`MetaCognition`) — second-order beliefs over
    uncertainty and parameter drift.
  - **Active experiment** (`BayesianExperimentPlanner`, `HypothesisTester`) —
    Bayesian-optimal experiment selection + Bayes-factor hypothesis testing.
  - **Multi-agent** (`MultiAgentWorld`, `CommunicationChannel`,
    `CulturePropagation`) — multi-agent world, message channel, cultural
    transmission.

## Quick install & run

```bash
pip install numpy scipy                      # required (only these two)
pip install qiskit numba joblib cupy         # optional accelerators (pick any)
PYTHONPATH=src python demo.py                # end-to-end zero-data demo
PYTHONPATH=src python benchmark.py           # hardware-acceleration benchmarks
```

Optional REST API (FastAPI):

```bash
pip install fastapi uvicorn
PYTHONPATH=src uvicorn zero_data_model.api:app --port 8000   # Swagger UI at /docs
```

A 30-second taste:

```python
from zero_data_model.model import ZeroDataModel
m = ZeroDataModel(dim=64)
print(m.think().metadata)                    # self-generated thought, no input
print(m.classify_text("the algorithm computes the network"))   # ('tech', conf)
```

Enable cognitive-upgrade phases individually with feature flags (all default
to `False`; the baseline `think()` cycle is unchanged when none are set):

```python
m = ZeroDataModel(dim=64, enable_architect=True, enable_meta_cognition=True)
```

Each enabled phase appends keys to `Signal.metadata["cognitive_upgrades"]`
without altering the six core modules' computation.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — four-layer diagrams
  (hardware / core / capabilities / cognitive-upgrade), module dependency
  graph, the `think()` data flow, hardware stack, file tree, design principles.
- [`docs/superpowers/specs/2026-07-22-phase7-cognitive-upgrade-design.md`](docs/superpowers/specs/2026-07-22-phase7-cognitive-upgrade-design.md)
  — Phase 7 design spec: 7 phases × 13 modules, theory basis, feature flags,
  architecture decisions, military-grade fix summary, test-coverage matrix.
- [`docs/superpowers/reviews/2026-07-22-phase7-cognitive-upgrade-review.md`](docs/superpowers/reviews/2026-07-22-phase7-cognitive-upgrade-review.md)
  — Phase 7 implementation review: 13 modules, ~80 defect fixes, verification
  results, residual bugs, follow-up suggestions.
- [`docs/tutorial.md`](docs/tutorial.md) — installation, quick start, NLP /
  vision / analytics examples, cross-domain transfer, hardware acceleration,
  benchmarks and tests.
- [`docs/theory.md`](docs/theory.md) — the science behind each module
  (GWT, predictive processing, free energy, category/topos theory, VQCs,
  biological computation, information geometry / TDA / fractals) with the
  mapping from theory to class/method.

## Project structure

```
src/zero_data_model/
  base.py                # Signal, Prediction, CognitiveModule, KnowledgeStore
  model.py               # ZeroDataModel — integration entry point
  consciousness_core.py  # ConsciousnessCore, GlobalWorkspace, SelfModel, PredictiveLayer
  active_inference.py    # ActiveInferenceEngine, GenerativeModel, MarkovBlanket, HomeostaticController
  category_engine.py     # CategoryTheoryEngine, Category, Functor, ToposEngine
  quantum_hybrid.py      # QuantumClassicalHybrid, VariationalQuantumCircuit, QuantumAnnealer
  biological.py          # BiologicalSubstrate, DNAStorage, MorphogeneticField, CellularAutomata
  math_universe.py       # MathematicalUniverse, InformationGeometry, TopologicalAnalyzer, FractalGenerator
  persistence.py         # ModelSerializer — npz + json save/load (no pickle)
  api.py                 # FastAPI app exposing ZeroDataModel as HTTP endpoints
  capabilities/          # rules.py, nlp.py, vision.py, analytics.py + advanced extensions
  hardware/              # accel.py, quantum.py, parallel.py
  plasticity/            # architect.py — ArchitectureOptimizer (module split / prune)
  cogtime/               # layered_predictor.py, temporal_memory.py (multi-timescale)
  cogmem/                # episodic_graph.py, semantic_index.py (structured memory)
  knowledge/             # logic_layer.py, causal_inference.py (neuro-symbolic)
  metacog/               # meta_cognition.py (second-order beliefs)
  experiment/            # experiment_planner.py, hypothesis_tester.py (Bayesian)
  multiagent/            # world.py, communication.py, culture.py (multi-agent)
  causal_emergence/      # engine.py + 5 submodules (topology/causal/differential/hmc/chaotic)
  security/              # rate_limiter.py, auth.py, headers.py (middleware layer)
tests/                   # one test file per module + integration
demo.py  benchmark.py    # runnable demo + benchmarks
docs/                    # architecture, tutorial, theory, superpowers/
```

## Security

The FastAPI app (`zero_data_model.api`) ships a layered security middleware
stack assembled in `create_app()`:

- **API-key auth** — every cognitive endpoint is gated by an `X-API-Key`
  header checked in constant time (`hmac.compare_digest`). `/health`,
  `/ready`, `/metrics`, and the docs routes stay public so liveness/readiness
  probes and Prometheus scrapes work without a key.
- **Rate limiting** — a token-bucket limiter (`security/rate_limiter.py`)
  throttles per-client request rates (identified by `X-Forwarded-For` behind
  an ingress, else the socket peer). Health/metrics/docs paths are exempt.
  WebSocket endpoints bypass the HTTP middleware chain and must call
  `RateLimiter.check` in their own handler.
- **Request size limit** — requests whose declared `Content-Length` exceeds
  the cap are rejected with `413` before the body is read.
- **Security response headers** — `X-Content-Type-Options`, `X-Frame-Options`,
  `X-XSS-Protection`, `Referrer-Policy`, `Cache-Control: no-store` on every
  response, plus `Strict-Transport-Security` (HSTS) on HTTPS only.
- **CORS / TrustedHost** — origins and hosts are env-configurable; the
  default CORS policy is `allow_credentials=False`.

Authentication is **opt-in**: when `ZDM_API_KEY` is unset the API runs in
development mode with no auth (every endpoint is open). Set it to enable
authentication.

| Environment variable        | Default                              | Meaning                                                          |
| --------------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| `ZDM_API_KEY`               | *(unset — auth disabled)*            | Set to a non-empty value to enable `X-API-Key` authentication.   |
| `ZDM_CORS_ORIGINS`          | `http://localhost:8000,http://localhost:3000` | Comma-separated list of allowed CORS origins.            |
| `ZDM_RATE_LIMIT_RPM`        | `60`                                 | Sustained per-client request ceiling (requests per minute).      |
| `ZDM_RATE_LIMIT_BURST`      | `10`                                 | Token-bucket burst capacity (max requests in a short window).    |
| `ZDM_ALLOWED_HOSTS`         | `localhost,127.0.0.1`                | Comma-separated TrustedHost allow-list.                          |
| `ZDM_ENV`                   | `production`                         | `development` enables `/docs`, `/redoc`, `/openapi.json`.        |

```bash
# Production example: auth on, tight CORS, generous rate limit.
export ZDM_API_KEY=$(openssl rand -hex 32)
export ZDM_CORS_ORIGINS="https://app.example.com"
export ZDM_RATE_LIMIT_RPM=120
export ZDM_RATE_LIMIT_BURST=20
PYTHONPATH=src uvicorn zero_data_model.api:app --port 8000
```

The security middlewares and auth helpers are also available as a reusable
package (`zero_data_model.security`) for new surfaces.

## Testing

```bash
pip install pytest
PYTHONPATH=src pytest -q                   # full suite (78 test files)
PYTHONPATH=src pytest tests/test_model.py -v
```

## License

MIT (placeholder) — see `LICENSE` when added.
