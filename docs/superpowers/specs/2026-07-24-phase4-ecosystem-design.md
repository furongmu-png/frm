# Phase 4 — Open Ecosystem Platform Design Spec

- **Date:** 2026-07-24
- **Status:** Draft (awaiting user review)
- **Scope:** Transform the lab-grade ZeroDataModel (Phases 1–7 core + multi-agent + tools + observability) into a scalable, multi-domain, distributed open ecosystem with ethics & safety and sustainable-evolution infrastructure.
- **Related:**
  - [Phase 7 cognitive-upgrade spec](2026-07-22-phase7-cognitive-upgrade-design.md)
  - [REST API reference](../../api/rest.md)
  - [Python SDK reference](../../api/python-sdk.md)
  - Existing v1 API: [`src/zero_data_model/api.py`](../../../src/zero_data_model/api.py)
  - Existing auth: [`src/zero_data_model/security/auth.py`](../../../src/zero_data_model/security/auth.py)
  - Existing audit: [`src/zero_data_model/observability/audit.py`](../../../src/zero_data_model/observability/audit.py)

---

## 1. Overview

### 1.1 Motivation

After three core upgrade stages (S4 + predictive coding + Hopfield;
neuro-symbolic + meta-cognition + experiment; multi-agent + tools +
observability) ZeroDataModel is a complete *cognitive architecture*, but it is
still a single-process library:

- The only HTTP surface is the **v1** FastAPI app in `api.py` — a 4100-line
  monolith that mixes 122 endpoints, auth, persistence, observability, and
  metrics. There is no batch inference, no remote intervention, no audit-log
  query, no knowledge-graph export.
- The v2 negotiation hook returns `501 Not Implemented`.
- There is no SDK; the `sdk/python/` package ships a README but no client.
- There is no federated learning story; every model instance is isolated.
- There are no domain templates; every consumer re-derives the wiring.
- Ethics & safety are scattered across `observability/audit.py`
  (`ValueVector`, `AuditEventType`), `tools/safety.py`, and ad-hoc checks;
  there is no emergency-stop, no bias detection, no alignment audit API.
- There is no model registry or regression gate; a `git push` is the only
  "release" and there is no automated check that a change regresses cognitive
  scores.

Phase 4 turns the artifact into a **platform** along five axes, each behind a
feature flag so the v1 baseline keeps its zero-regression contract.

### 1.2 Design contract (extends Phase 7 invariants)

Every Phase 4 subsystem obeys:

- **Feature-flagged.** Constructed only when the flag is set; otherwise the
  import is lazy and the runtime cost is zero. The v1 app and the v1 API
  behaviour are unchanged when all Phase 4 flags are off.
- **Read-only on the v1 core.** v2 endpoints observe already-computed model
  state; they never call back into a core module's `process`/`predict`/
  `update` synchronously from a request handler. Long-running work (batch
  inference, federated rounds, bias scans) runs on a background task pool
  and exposes status via polling/SSE.
- **Try/except isolated.** Every external surface wraps third-party calls
  (HTTP server, JWT lib, opentelemetry, flower, networkx) individually; a
  failure warns and degrades, never crashes `think()` or the v1 app.
- **Metadata-only side-effects on `Signal`.** New cognitive contributions
  land under `Signal.metadata["phase4"]`; that key is only attached when at
  least one Phase 4 cognitive surface is active, so the no-upgrade metadata
  key-set is unchanged.
- **No backwards-compat shims for unused code.** Where Phase 4 replaces a v1
  behaviour, the v1 path is *removed* under the v2 prefix, not parallel-
  maintained. v1 callers keep using `/v1/...`.

### 1.3 Non-goals

- No re-implementation of the cognitive core. Phase 4 wraps; it does not
  rewrite `ZeroDataModel`, `HierarchicalZeroDataModel`, or any Phase 1–7
  module.
- No new cognitive algorithm. New "intelligence" in Phase 4 is *organizational*
  (registry, federation, templates), not *algorithmic*.
- No multi-tenant SaaS control plane. The platform is single-tenant per
  process; multi-tenancy is a deployment concern handled by the operator
  (one pod per tenant) and is explicitly out of scope.

---

## 2. Open Platform (`src/zero_data_model/platform/`)

A new top-level package that owns the v2 HTTP surface, the Python SDK, and the
shared client/server schema. It is imported lazily by `api.py` so the v1 app
picks v2 up only when `ZDM_ENABLE_PLATFORM=1`.

### 2.1 Package layout

```
src/zero_data_model/platform/
  __init__.py              # public re-exports: create_app, ZeroDataClient
  app.py                   # v2 FastAPI factory: mounts v2_router under /v2
  v2_router.py             # all /v2/* endpoints
  auth.py                  # API-Key + JWT, read/write scope separation
  schemas.py               # pydantic v2 request/response models (shared w/ SDK)
  serial.py                # HierarchicalSerializer: full model state ⇆ JSON+weights
  knowledge_export.py      # ReasoningGraph → GraphML / JSON-LD
  audit_query.py           # AuditLogger → time-range / event-type query
  openapi.py               # OpenAPI 3.0 doc generator + Swagger UI mount
  sdk.py                   # ZeroDataClient (sync + async)
  exceptions.py            # ZeroDataError hierarchy
```

The v1 file `api.py` keeps its existing `version_negotiation` middleware but
the 501 branch is replaced with a route to `platform.app.v2_router` when the
platform flag is on, and keeps the 501 when the flag is off (so an un-flagged
build is byte-for-byte identical to today).

### 2.2 REST API v2

All v2 endpoints live under the `/v2` prefix. v1 endpoints are untouched.
Version selection is *path-based* (`/v2/...`), **not** Accept-header based —
the Accept-header hook in v1 was a placeholder and is removed in favour of
explicit paths, which are simpler to route, log, and rate-limit.

#### 2.2.1 Endpoint catalogue

| Method | Path | Auth scope | Purpose |
| --- | --- | --- | --- |
| `POST` | `/v2/think` | `cog:read` | Batch inference: one request, N observations → N actions. |
| `POST` | `/v2/think/stream` | `cog:read` | SSE streaming variant for long rolls. |
| `GET`  | `/v2/model/state` | `model:read` | Export full model state: JSON manifest + binary weights blob. |
| `PUT`  | `/v2/model/state` | `model:write` | Load a previously exported state (atomic, version-checked). |
| `POST` | `/v2/intervention` | `model:write` | Remote intervention into a named internal module (belief, memory, value vector). |
| `GET`  | `/v2/metrics` | `metrics:read` | Prometheus exposition + cognitive gauges (free energy, prediction error, layer errors). |
| `GET`  | `/v2/knowledge-graph` | `kg:read` | Export the reasoning graph as GraphML or JSON-LD (`?format=graphml\|jsonld`). |
| `GET`  | `/v2/audit-log` | `audit:read` | Query audit log by time range and event type. |
| `GET`  | `/v2/health` | public | Liveness + readiness combined (k8s probe target). |
| `GET`  | `/v2/version` | public | Build, model-version, schema-version, capability flags. |
| `GET`  | `/v2/openapi.json` | public | OpenAPI 3.0 document (dev only). |
| `GET`  | `/v2/docs` | public | Swagger UI (dev only). |

The `/metrics`, `/health`, `/ready` *root* paths from v1 stay unchanged so
existing probes keep working; v2 only adds the `/v2/`-prefixed variants.

#### 2.2.2 Batch inference — `POST /v2/think`

Request:

```json
{
  "observations": [[0.1, -0.2, ...], [0.3, 0.0, ...]],
  "return_metadata": true,
  "modules": ["active_inference", "category_engine"]
}
```

- `observations`: 1–64 arrays per request (hard cap, CWE-400). Each array
  length must equal the running model's `dim`.
- `return_metadata`: when `true`, the response carries `metadata` per item
  (free energy, prediction error, PCN layer errors, phase-3 audit ref).
- `modules`: optional allow-list restricting which cognitive modules run
  for this batch (perf opt; default = all enabled).

Response (200):

```json
{
  "results": [
    {"action": [0.12, ...], "metadata": {"free_energy": 0.43, ...}},
    ...
  ],
  "batch_id": "01HXY...",
  "server_processing_ms": 12.4
}
```

The handler dispatches the batch onto a `ThreadPoolExecutor` (size =
`ZDM_PLATFORM_WORKERS`, default 4) so a 64-item batch does not block the
event loop. Each item's `model.think()` call is wrapped in try/except; a
single failure yields a per-item `{"error": "..."}` entry, not a 500 for
the whole batch.

#### 2.2.3 Model state — `GET /v2/model/state`

Returns a multipart response:

- Part 1 (`application/json`): a manifest
  ```json
  {
    "schema_version": "zdm.state/v1",
    "model_version": "0.14.2",
    "created_at": "2026-07-24T08:00:00Z",
    "dim": 32,
    "modules": ["consciousness", "active_inference", ...],
    "phase_flags": {"use_pcn": true, "use_phase2": true, ...},
    "weights_blob": "weights.npz",
    "weights_sha256": "ab12...",
    "weights_size": 245678
  }
  ```
- Part 2 (`application/octet-stream`): the binary `.npz` weights blob.

This is produced by a new `HierarchicalSerializer`
(`platform/serial.py`) that extends the existing `ModelSerializer` to also
serialize Phase 2/3 state (logic rules, second-order belief, experiment
planner state, tool registry, audit ring buffer, thought chain). The v1
`ModelSerializer` is kept as-is for v1 callers; `HierarchicalSerializer`
delegates to it for the core and adds the layered state on top.

#### 2.2.4 Model state — `PUT /v2/model/state`

Accepts the same multipart body. Validation:

1. `schema_version` must equal `"zdm.state/v1"`.
2. `model_version` must be compatible (same major; minor drift allowed).
3. Recompute `weights_sha256` and reject on mismatch.
4. Load into a *staging* model instance, run a sanity `think()` on a fixed
   probe input, compare the action against a stored reference; only on
   match does the staging instance atomically swap in as the live model
   (under the existing module-level lock pattern from `persistence.py`).

This is the same atomic-swap discipline already used by `ModelSerializer`
(Stage→backup→replace→cleanup under a module-level lock); Phase 4 reuses
that lock rather than introducing a new one.

#### 2.2.5 Remote intervention — `POST /v2/intervention`

Request:

```json
{
  "module": "belief",
  "operation": "set",
  "path": ["active_inference", "generative_model", "belief_state"],
  "value": [0.1, -0.2, ...],
  "reason": "human override during debugging session #42"
}
```

- `module`: one of `belief`, `memory`, `value`, `category`.
- `operation`: `set` | `add` | `clamp` | `release`.
- `path`: dotted access path into the named module (validated against an
  allow-list of writable paths; everything else is 403).
- `value`: JSON-serializable; cast to `np.ndarray` server-side when the
  target is numeric.
- `reason`: **mandatory**, free-text, ≥ 8 chars. Recorded in the audit log
  alongside the caller identity (from JWT `sub` claim) so every
  intervention is attributable.

Every intervention writes an `AuditEntry` of type `HUMAN_INTERVENTION`
**before** the mutation is applied (intent log) and a second one **after**
(outcome log), so a crash mid-intervention leaves a recoverable trail.

#### 2.2.6 Metrics — `GET /v2/metrics`

Prometheus exposition text, identical format to v1 `/metrics` *plus* the
cognitive gauges that today live only inside `signal.metadata`:

```
# HELP zdm_free_energy Last free energy reported by the model
# TYPE zdm_free_energy gauge
zdm_free_energy{layer="L0"} 0.12
zdm_free_energy{layer="L1"} 0.08
zdm_free_energy{layer="L2"} 0.04
# HELP zdm_prediction_error Last prediction error norm
# TYPE zdm_prediction_error gauge
zdm_prediction_error 0.21
# HELP zdm_audit_events_total Total audit events by type
# TYPE zdm_audit_events_total counter
zdm_audit_events_total{type="ACTION_SELECTION"} 1283
zdm_audit_events_total{type="HUMAN_INTERVENTION"} 4
```

The endpoint reuses the existing `prometheus_client` integration; new
gauges are registered once at app startup in `platform/app.py`.

#### 2.2.7 Knowledge graph — `GET /v2/knowledge-graph`

Query params: `format=graphml|jsonld` (default `graphml`), `depth=N`
(max hops from root, default 3), `node_filter=...` (optional JSON
predicate).

Backed by `platform/knowledge_export.py`, which wraps
`knowledge.reasoning_graph.ReasoningGraph` and walks it via `networkx`.
GraphML output is a string; JSON-LD output uses the
`https://schema.zerodata.ai/v1/` context with `Node`/`Edge`/`Inference`
types. The endpoint streams the response (chunked) so a 1M-edge graph does
not buffer in memory.

#### 2.2.8 Audit log — `GET /v2/audit-log`

Query params: `from=<ISO8601>`, `to=<ISO8601>`, `type=<EventType>`
(repeatable), `module=<name>` (repeatable), `limit=N` (default 100, max
1000), `cursor=<token>` (pagination).

Backed by `platform/audit_query.py`, which reads from the in-memory
`AuditLogger` ring buffer (existing) when `ZDM_AUDIT_BACKEND=memory` (the
default), or from an append-only JSONL file when
`ZDM_AUDIT_BACKEND=file`. The file backend is a thin reader; no new
storage engine is introduced in Phase 4.

Response:

```json
{
  "events": [
    {"entry_id": 42, "timestamp": 1784835390.1, "event_type": "HUMAN_INTERVENTION", "module": "belief", "context": {...}, "result": {...}},
    ...
  ],
  "next_cursor": "eyJ0cyI6MTc4..."
}
```

### 2.3 Authentication & authorization

`platform/auth.py` defines a unified dependency that supports two schemes,
both resolved from the `Authorization` header:

1. **API Key** — `Authorization: Bearer zdm_<key>`. Lookup in a static
   keyfile (`ZDM_API_KEYS_FILE`, JSON: `{"keys": [{"id": "k1", "secret":
   "...", "scopes": ["cog:read", "model:read"]}]}`). Constant-time compare
   (reuses `security/auth.py`'s existing helper).
2. **JWT** — `Authorization: Bearer <jwt>`. HS256 signed with
   `ZDM_JWT_SECRET`. Claims: `sub` (caller id), `scopes` (list), `exp`
   (15-min default, configurable). Verified with `pyjwt` (already an
   optional dep via `requirements.lock`).

Scope model — every endpoint declares required scopes (see §2.2.1 table).
The dependency returns `403` when the resolved identity lacks a required
scope, `401` when no credential is presented and `ZDM_REQUIRE_AUTH=1`
(the default for v2; v1 keeps its opt-in behaviour).

Read/write separation:

- Read scopes: `cog:read`, `model:read`, `metrics:read`, `kg:read`,
  `audit:read`.
- Write scopes: `model:write`. (Intervention is the only write surface in
  v2.1; later versions may add `cog:write` for online learning.)

A single key may carry both read and write scopes (operator pattern) or
only read (consumer pattern). The default dev key (`ZDM_API_KEY`) is
auto-issued all scopes so the local `quickstart` flow is unchanged.

### 2.4 OpenAPI documentation

`platform/openapi.py` produces an OpenAPI 3.0 document from the FastAPI
auto-generated spec, then post-processes it to:

- Add the `Bearer` security scheme (API Key + JWT).
- Tag every endpoint with its scope, so the rendered Swagger UI groups
  endpoints by access level.
- Attach examples for every request body (drawn from `schemas.py`).

The Swagger UI is mounted at `/v2/docs` and the spec at
`/v2/openapi.json`. Both are hidden when `ZDM_ENV=production`, matching
the v1 convention.

### 2.5 Python SDK — `ZeroDataClient`

`platform/sdk.py` ships a single class with sync and async surfaces. The
SDK is re-exported from the top-level `sdk/python/` package so existing
users of that package pick it up without a new install.

```python
from zero_data_model.platform import ZeroDataClient

client = ZeroDataClient(
    api_key="zdm_...",
    endpoint="https://api.zerodata.ai/v2",
    timeout=30.0,
    retry={"total": 3, "backoff_factor": 0.5},
)

# Sync surface
action, info = client.think(observation)            # single
results = client.think_batch([obs_a, obs_b, obs_c]) # batch
client.intervene(module="belief", operation="set",
                 path=["active_inference", "belief_state"],
                 value=[0.1, -0.2], reason="debug #42")
kg = client.get_knowledge_graph(format="jsonld")
state = client.export_state()                       # -> StateBundle
client.import_state(state)
audit = client.query_audit_log(
    from_="2026-07-24T00:00:00Z", type="HUMAN_INTERVENTION")

# Async surface (same methods, `_async` suffix)
import asyncio
async def main():
    async with ZeroDataClient(api_key="...", endpoint="...") as c:
        action, info = await c.think_async(observation)
asyncio.run(main())
```

Design notes:

- **Transport:** `httpx` (sync + async from one library; already a
  transitive dep of `fastapi`). No `requests`.
- **Auth:** the client accepts `api_key=` or `jwt=` (a callable returning a
  fresh JWT when the previous one expires). It injects `Authorization`
  and transparently retries once on `401` with a refreshed token.
- **Retries:** exponential backoff on 5xx and 429, idempotency-key header
  on `think`/`intervene` (UUID5 of the request body) so a retry is
  safe even if the server applied the first attempt.
- **Streaming:** `think_stream` returns an iterator; the SDK yields
  per-item results as they arrive.
- **Errors:** `ZeroDataError` hierarchy
  (`AuthenticationError`, `AuthorizationError`, `RateLimitError`,
  `ServerError`, `ClientError`, `ValidationError`). Each carries the
  request id, status, and the server's `detail` field. No raw `httpx`
  exceptions leak.
- **State bundle:** `export_state()` returns a `StateBundle` dataclass
  holding `manifest: dict`, `weights: bytes`, `weights_sha256: str`.
  `import_state(state)` round-trips it through `PUT /v2/model/state`.
  The bytes are kept in memory; the SDK does not write temp files (the
  server's multipart parser handles streaming).

The TypeScript SDK (`sdk/typescript/`) gets the same surface in a
follow-up task; it is out of scope for this spec because the user named
only the Python SDK.

---

## 3. Domain Application Templates (`src/zero_data_model/templates/`)

Pre-configured, runnable entry points for the three named scenarios. Each
template is a *thin* wrapper: it constructs a `HierarchicalZeroDataModel`
with the right Phase 1–7 flags, wires an environment adapter, and
registers a benchmark. It does **not** add new cognitive logic.

### 3.1 Package layout

```
src/zero_data_model/templates/
  __init__.py
  base.py                   # TemplateBase: shared run/eval harness
  scientific_discovery/
    __init__.py
    config.yaml             # model flags, dim, scenario params
    environment.py          # HypothesisTestEnv adapter
    benchmark.py            # eval: novelty, falsification rate, coverage
    README.md
  education_simulation/
    __init__.py
    config.yaml
    environment.py          # StudentModelEnv adapter
    benchmark.py            # eval: knowledge gain, misconception count
    README.md
  game_npc/
    __init__.py
    config.yaml
    environment.py          # GameWorldEnv adapter (OpenAI Gym-like)
    benchmark.py            # eval: task completion, dialogue coherence, latency
    README.md
```

### 3.2 `TemplateBase` contract

Every template exposes:

```python
class TemplateBase:
    name: str                          # "scientific_discovery"
    def build_model(self) -> HierarchicalZeroDataModel: ...
    def build_env(self, seed: int): ...
    def run_episode(self, model, env, max_steps: int = 1000) -> EpisodeLog: ...
    def evaluate(self, log: EpisodeLog) -> BenchmarkReport: ...
    def cli(self) -> None: ...          # `python -m zero_data_model.templates.<name>`
```

`EpisodeLog` and `BenchmarkReport` are dataclasses in `base.py`; both
serialize to JSON. The CLI runs `build_model → build_env → run_episode →
evaluate` and prints a Markdown report to stdout, writing artifacts to
`experiments/output/<name>/`.

### 3.3 Scientific discovery template

- **Scenario.** The agent is given a phenomenon (a black-box function
  `f: R^n → R` synthesised from a small library of physics-flavoured
  primitives: harmonic, decay, saturation, noise) and must propose
  experiments, observe outcomes, and converge on the governing law.
- **Model flags.** `use_pcn=True, use_phase2=True, enable_tools=True,
  enable_observability=True`. Phase 2's `BayesianExperimentPlannerV2`
  is the workhorse; `LogicLayer` validates hypotheses for internal
  consistency before they are tested.
- **Env.** `HypothesisTestEnv`: exposes `propose(experiment) → outcome`,
  tracks a budget (default 50 experiments), scores by the held-out law's
  log-likelihood under the agent's final hypothesis.
- **Benchmark.** Three metrics:
  1. **Novelty** — fraction of proposed experiments that explore regions
     not covered by random sampling (measured by KDE overlap).
  2. **Falsification rate** — fraction of hypotheses the agent
     *self-rejects* via the logic layer or Bayes factor before spending
     budget on them (a higher rate is better — it means the agent is
     reasoning, not just probing).
  3. **Coverage** — KL divergence between the agent's final posterior over
     laws and the true law's distribution (lower is better).

### 3.4 Educational simulation template

- **Scenario.** The agent plays a tutor: it observes a simulated
  student's response to a problem, infers the student's belief state
  (using Phase 7's `SecondOrderBelief` over the *student*, not over the
  agent itself), and picks the next problem to maximise expected
  knowledge gain.
- **Model flags.** `use_pcn=True, use_phase2=True, enable_teaching=True,
  enable_observability=True`. `HumanFeedback` is the channel through
  which the *simulated* student gives correctness feedback.
- **Env.** `StudentModelEnv`: a hidden Markov student with K skills, each
  with a latent mastery in [0, 1]; the agent observes correctness of
  responses and must decide which skill to probe next.
- **Benchmark.**
  1. **Knowledge gain** — sum over skills of the agent's posterior mean
     mastery movement toward the true mastery, per session.
  2. **Misconception count** — number of skills the agent ends the
     session still wrong about (absolute error > 0.2).
  3. **Question efficiency** — average information gain per question
     (bits), compared to a random-question baseline.

### 3.5 Game NPC template

- **Scenario.** The agent is an NPC in a small grid world with a player
  avatar. The NPC must (a) complete its own task (collect N items) and
  (b) converse with the player via a fixed-vocabulary dialogue to either
  help or misdirect, depending on a configured "alignment" value.
- **Model flags.** `use_pcn=True, use_phase2=False, enable_tools=False,
  enable_observability=True`. This is the lightest template — NPCs must
  be cheap. Phase 2 modules are off by default but can be enabled via
  `config.yaml` for "boss" NPCs.
- **Env.** `GameWorldEnv`: a 10×10 grid, a player, N items, an
  `act(action) → observation, reward, done` interface (Gym-like; we
  intentionally do not pull in `gymnasium` to keep the dep graph flat —
  the env is a 60-line class).
- **Benchmark.**
  1. **Task completion** — fraction of items collected within the step
     budget.
  2. **Dialogue coherence** — fraction of NPC utterances that are
     topically consistent with the player's last utterance (evaluated by
     a fixed keyword-overlap scorer; we deliberately avoid an LLM judge
     to keep the benchmark deterministic and offline).
  3. **Latency** — p50/p95 `think()` wall time per step. Hard target:
     p95 ≤ 5 ms on a laptop CPU with `dim=16`. A regression beyond this
     budget fails CI (see §6.3).

### 3.6 Template evaluation harness

`base.py` ships a tiny harness that runs every template's benchmark 5
times with seeds `[0, 1, 2, 3, 4]` and reports mean ± std. The harness is
exposed as a pytest fixture (`tests/conftest.py::template_runner`) so a
template's benchmark can be invoked from CI as
`pytest tests/test_templates.py -k scientific_discovery`.

---

## 4. Federated Learning (`src/zero_data_model/federation/`)

Decentralised knowledge sharing across model instances, with privacy
preservation via differential privacy on the shared artefact.

### 4.1 What is shared — and what is not

Phase 4 shares **knowledge-graph deltas**, not raw weights. Rationale:

- The model's neuro-symbolic substrate makes its knowledge graph the
  natural unit of meaning; raw weight tensors are opaque, large, and
  encode training-data-derivable information (privacy risk).
- The existing `CulturePropagation` mechanism already shares
  "abstractions" across agents in a single world; federation extends
  that pattern across *processes*.
- A graph delta is small (typically < 10 KB per round) and is naturally
  amenable to differential privacy: each edge's weight is clamped and
  noised rather than the full tensor.

**Not shared:** model weights, episodic memory contents (which may carry
PII from the local environment), audit logs, intervention history.

### 4.2 Topology — hybrid trusted-core + untrusted-edge

```
            ┌──────────────────────────────────────────────┐
            │  Trusted core (operator-run, mTLS, audited)   │
            │  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
            │  │ Aggregator │  │ Registry  │  │ Audit   │   │
            │  └────┬────┘  └─────┬───┘  └────┬────┘       │
            └───────┼─────────────┼────────────┼───────────┘
                    │             │            │
        ┌───────────┴───┐    ┌────┴────┐   ┌───┴────┐
        │  Edge A       │    │ Edge B   │   │ Edge C │   (untrusted,
        │  (tenant 1)   │    │ (t2)     │   │ (t3)   │    no mTLS to
        └───────────────┘    └─────────┘   └────────┘    each other)
```

- **Trusted core** runs the `Aggregator`, `Registry`, and `Audit`
  services. It is operator-owned, mTLS-secured, and logs every round.
- **Edges** are tenant model instances. They talk only to the core
  (never peer-to-peer), so a compromised edge cannot poison another
  edge directly — only via the aggregator, which validates deltas.
- Edges authenticate to the core with the same API-Key + JWT scheme as
  §2.3, carrying a `fed:contribute` scope.

### 4.3 Package layout

```
src/zero_data_model/federation/
  __init__.py
  delta.py            # KnowledgeDelta: diff + DP noise
  aggregator.py       # Aggregator: receive, validate, merge, publish
  contributor.py      # Contributor: local KG → delta, push to core
  consumer.py         # Consumer: pull merged delta, apply to local KG
  registry.py         # Registry: round metadata, participant roster
  privacy.py          # DPBudget, clip, noise, accountant
  transport.py        # HTTP transport over the v2 API (reuses ZeroDataClient)
  config.py           # FedConfig dataclass
```

### 4.4 Round protocol

A federation round is a single request/response between an edge and the
core. The core drives rounds on a fixed cadence
(`ZDM_FED_ROUND_INTERVAL_SECONDS`, default 300 s); edges may contribute
opportunistically.

1. **Local delta extraction** (edge). `contributor.extract_delta(model)`
   walks the local `ReasoningGraph` and emits a `KnowledgeDelta`:
   - `added`: edges added since the last acknowledged round.
   - `updated`: edges whose weight moved by more than `ε` (default 0.01).
   - `removed`: edges deleted since the last round.
   Each edge carries `(subject, predicate, object, weight, evidence_count)`.

2. **Differential privacy** (edge). `privacy.apply_dp(delta, budget)`:
   - Clips every edge weight to `[-C, +C]` (default `C = 1.0`).
   - Adds Gaussian noise `N(0, σ²)` to each weight, with `σ = C · √(2 ln(1.25/δ)) / ε`.
   - Drops edges whose noised weight falls below a threshold `τ`
     (default 0.05) — these are too noisy to be useful and would
     inflate the published graph.
   - The privacy budget `(ε, δ)` is tracked by a `PrivacyAccountant`
     across rounds; an edge refuses to contribute when the cumulative
     `ε` exceeds its configured cap (default `ε_total = 8.0`, the
     standard zCDP-composed cap for a research setting).

3. **Submission** (edge → core). `transport.submit_delta(delta)` POSTs
   to `/v2/federation/deltas` on the core. The core authenticates the
   edge, validates the delta's signature (HMAC over the canonical JSON),
   and enqueues it for the next round.

4. **Aggregation** (core). `aggregator.merge(deltas)`:
   - Groups edges by `(subject, predicate, object)` across all
     contributing edges.
   - Computes the merged weight as the **median** of contributing
     weights (robust to a single poisoned edge; mean would be cheaper
     but median is the standard robust aggregator for federated KG
     settings).
   - Computes `evidence_count` as the sum of contributing counts.
   - Drops edges supported by fewer than `k_min` edges (default 2) —
     this is the core's defence against a lone poisoned contributor.

5. **Publication** (core → edges). `aggregator.publish()` writes the
   merged delta to `/v2/federation/rounds/latest`. Edges poll this on
   their own cadence.

6. **Application** (edge). `consumer.apply_delta(model, delta)`:
   - For `added`/`updated` edges, upsert into the local `ReasoningGraph`
     with a `provenance = "fed:round-N"` tag.
   - For `removed` edges, mark as `superseded` rather than hard-delete
     (so a local episode that referenced the edge still resolves).
   - Does **not** trigger a `think()` cycle; the new knowledge is
     available on the next local `think()`.

### 4.5 Threat model & defences

| Threat | Defence |
| --- | --- |
| Poisoned edge injects false edges. | Median aggregation + `k_min` support threshold (§4.4 step 4). A lone poisoned edge cannot move a merged weight by more than `C`. |
| Poisoned edge floods the core. | Per-edge rate limit (1 delta / round) + delta size cap (256 KB) at the transport layer. |
| Edges re-identify training data from a delta. | Differential privacy on weights (§4.4 step 2). The published weight distribution is provably indistinguishable from one trained without any single edge's contribution, up to `(ε, δ)`. |
| Core is compromised. | Out of scope for this spec — the core is trusted by definition. A future spec may add a multi-core threshold scheme. |
| Stale delta replayed. | Each delta carries a `round_id` monotonic with the registry; the core rejects any delta whose `round_id` is older than the current round. |
| Edge contributes more than its privacy budget allows. | `PrivacyAccountant` raises `BudgetExhausted` and the edge logs + skips; it does not crash. |

### 4.6 Configuration

`FedConfig` (in `federation/config.py`) is loaded from YAML:

```yaml
federation:
  enabled: true
  role: edge                      # edge | core
  core_endpoint: https://core.zerodata.ai/v2
  round_interval_seconds: 300
  privacy:
    epsilon_per_round: 0.5
    delta: 1.0e-5
    epsilon_total_cap: 8.0
    clip_bound: 1.0
    drop_threshold: 0.05
  aggregation:
    k_min: 2
  transport:
    api_key: zdm_...
    timeout_seconds: 30
    retry: {total: 3, backoff_factor: 0.5}
```

A core node sets `role: core` and additionally configures the
aggregator's storage (`backend: memory | file`, default `memory`).

---

## 5. Ethics & Safety (`src/zero_data_model/ethics/`)

A unified ethics layer that consolidates the scattered safety logic and
adds the missing pieces: value alignment as a first-class constraint,
emergency stop, and bias detection.

### 5.1 Package layout

```
src/zero_data_model/ethics/
  __init__.py
  alignment.py        # ValueAlignment: constrains action selection
  emergency_stop.py   # EmergencyStop: kill-switch + safe-state
  bias_detector.py    # BiasDetector: distributional fairness scan
  policy.py           # EthicsPolicy: configurable rule set
  audit_bridge.py     # bridges ethics events → AuditLogger
```

### 5.2 Value alignment (`alignment.py`)

`ValueAlignment` extends the existing `ValueVector` from
`observability/audit.py` (which today only *records* the preferred state)
into an **active constraint** on action selection:

```python
class ValueAlignment:
    def __init__(self, dimensions: list[ValueDimension]): ...
    def score_action(self, action: np.ndarray, context: dict) -> float:
        """Return a real-valued alignment score in [-1, +1]."""
    def constrain(self, action: np.ndarray, context: dict) -> np.ndarray:
        """Project the proposed action onto the value-aligned subspace."""
```

- `score_action` computes a weighted sum over the configured dimensions
  (e.g. `avoid_harm`, `pursue_truth`, `respect_autonomy`), each a
  function of `(action, context)` returning `[-1, +1]`.
- `constrain` is the operative hook: it nudges the action toward the
  alignment-maximising direction by gradient ascent on `score_action`,
  step size bounded by a configurable `max_nudge` (default `0.1`).
  This is a soft constraint — it does not *forbid* misaligned actions,
  it makes them less likely.
- Wired into `think()` after `active_inference` produces its proposed
  action and before the action is returned, gated by the
  `enable_value_alignment` feature flag.

The default dimensions and their scoring functions live in `policy.py`
and are overridable via the `ZDM_ETHICS_POLICY` YAML file. This keeps the
value system auditable and version-controlled rather than baked into
code.

### 5.3 Emergency stop (`emergency_stop.py`)

A two-level kill-switch:

1. **Soft stop** (`POST /v2/emergency/stop?level=soft`). The model enters
   a *read-only* mode: `think()` still runs but interventions are
   refused, tool calls are refused, federation contributions are
   paused. The model continues to serve observations so operators can
   inspect its state. Reversible via `POST /v2/emergency/resume`.
2. **Hard stop** (`POST /v2/emergency/stop?level=hard`). The model
   enters a *frozen* state: `think()` returns the last action without
   any cognitive work; all writes (state, audit, KG) are flushed to
   disk and the process exits with code `0` after a configurable grace
   period (`ZDM_EMERGENCY_GRACE_SECONDS`, default 10). Requires a
   separate `ethics:emergency` scope so a normal operator key cannot
   trigger it.

Both endpoints write an `AuditEntry` of type `EMERGENCY_STOP` *before*
acting and again after the stop state is reached. The soft-stop state
is sticky across restarts: a `state.json` marker is written to the
persistence root and consulted on startup, so a crashed-and-restarted
soft-stopped model comes back soft-stopped.

The hard-stop endpoint also fires a `SIGTERM` to itself after the grace
period. Phase 4's implementation task must verify that `__main__.py`
handles `SIGTERM` gracefully (drain in-flight requests, flush audit,
exit 0); if it does not today, adding a signal handler is in scope for
the §5 implementation chunk.

### 5.4 Bias detection (`bias_detector.py`)

`BiasDetector` runs as a background task on a configurable cadence
(`ZDM_BIAS_SCAN_INTERVAL_SECONDS`, default 600 s). For each protected
attribute configured in the ethics policy (e.g. `gender`, `age_band` —
the attribute names are abstract; the operator is responsible for
populating them in `context`), it computes:

- **Demographic parity** — `P(action | attr=a) − P(action | attr=b)` for
  each pair, on the last N actions (default 1000) drawn from the audit
  log.
- **Action distribution divergence** — KL divergence between the action
  distributions conditioned on each attribute value.

When any metric exceeds the configured threshold
(`ZDM_BIAS_PARITY_THRESHOLD`, default 0.1; `ZDM_BIAS_KL_THRESHOLD`,
default 0.05), the detector:

1. Writes an `AuditEntry` of type `VALUE_VIOLATION` with the offending
   metric and the action window.
2. Emits a `zdm_bias_violation` Prometheus gauge so operators get an
   alert.
3. *Does not* auto-correct. Auto-correction is out of scope — the
   detector surfaces, the operator decides.

### 5.5 Ethics policy (`policy.py`)

A single YAML-deserialisable dataclass that configures all of §5.2–5.4:

```yaml
ethics:
  enable_value_alignment: true
  value_dimensions:
    - {name: avoid_harm, weight: 1.0, scorer: harm_projection}
    - {name: pursue_truth, weight: 0.5, scorer: information_gain}
  max_nudge: 0.1
  emergency:
    grace_seconds: 10
  bias:
    scan_interval_seconds: 600
    parity_threshold: 0.1
    kl_threshold: 0.05
    protected_attributes: [gender, age_band]
```

The policy is loaded once at startup and is **immutable** for the
process lifetime — to change it, restart. This prevents an in-flight
intervention from silently disabling safety.

### 5.6 Audit bridge (`audit_bridge.py`)

A thin adapter so every ethics event flows through the existing
`AuditLogger`. This means `/v2/audit-log` (§2.2.8) already surfaces
ethics events; no separate ethics-event endpoint is needed.

---

## 6. Sustainable Evolution (`src/zero_data_model/registry/`)

Model registry, version management, and automated regression detection.

### 6.1 Package layout

```
src/zero_data_model/registry/
  __init__.py
  model_card.py       # ModelCard: metadata about a model version
  store.py            # RegistryStore: on-disk registry (file backend)
  version.py          # semantic versioning + capability negotiation
  regression.py      # RegressionDetector: runs benchmarks, compares
  ci_hook.py          # pre-commit / CI entry point
```

### 6.2 Model registry (`store.py`, `model_card.py`)

A `ModelCard` captures everything needed to reproduce and reason about
a model version:

```python
@dataclass
class ModelCard:
    version: str                       # semver, e.g. "0.14.2"
    created_at: datetime
    git_sha: str
    git_dirty: bool                    # uncommitted changes at build time
    code_version: str                  # ZeroDataModel.__version__
    phase_flags: dict[str, bool]       # which Phase 1–7 flags were on
    dim: int
    modules: list[str]
    state_bundle_hash: str             # sha256 of export_state() output
    benchmarks: dict[str, BenchmarkReport]   # the §3 benchmarks at this version
    capabilities: list[str]            # declared capability strings
    deprecated: bool = False
    successor: str | None = None       # version that replaces this one
```

`RegistryStore` writes cards to `experiments/registry/cards/<version>.json`
(file backend, default) or to a configured SQLite db
(`ZDM_REGISTRY_BACKEND=sqlite`). The file backend is append-only; a card
is never overwritten — supersession is recorded via `successor` on a
*new* card.

Cards are produced:

- **On every `git tag`.** The release CI workflow already exists
  (`.github/workflows/release.yml`); Phase 4 adds a step that builds a
  card, runs the §3 benchmarks, and commits the card to the registry.
- **On every main-branch build.** A "dev" card (version
  `0.14.2-dev.<short_sha>`) is produced for regression comparison.
- **On demand.** `python -m zero_data_model.registry.ci_hook build-card`
  produces a card for the current working tree.

### 6.3 Regression detection (`regression.py`)

`RegressionDetector.compare(candidate: ModelCard, baseline: ModelCard)`
runs the candidate's benchmarks against the baseline's and produces a
`RegressionReport`:

```python
@dataclass
class RegressionReport:
    baseline_version: str
    candidate_version: str
    metrics: dict[str, MetricDelta]    # name → {baseline, candidate, delta, pct}
    verdict: Literal["pass", "warn", "fail"]
    failures: list[str]               # human-readable failure reasons
```

**Pass/fail rules** (each is a hard gate; any failure → verdict `fail`):

- **Latency.** Game-NPC template's p95 `think()` must not increase by
  more than 10% vs baseline. (The 5 ms target in §3.5 is the absolute
  cap; the regression gate is the *delta*.)
- **Cognitive fidelity.** For a fixed probe input (the same one used by
  the state-load sanity check in §2.2.4), the candidate's action vector
  must have cosine similarity ≥ 0.95 to the baseline's. This catches
  silent refactors that change behaviour without changing tests.
- **Benchmark scores.** No benchmark metric may regress by more than 5%.
  A regression between 5% and 10% yields `warn`; above 10% is `fail`.
- **Audit invariants.** Every `think()` cycle must still produce an
  `ACTION_SELECTION` audit entry when observability is on. Missing
  audit entries → `fail` (this catches silent observability regressions).

The detector is wired into CI as a job that:

1. Loads the latest non-dev card as baseline.
2. Builds a candidate card for the PR.
3. Runs `compare` and posts the `RegressionReport` as a PR comment.
4. Fails the PR check if `verdict == "fail"`.

### 6.4 Version negotiation (`version.py`)

`ZeroDataModel.__version__` (already exists) is the *code* version. Phase
4 adds a *schema* version for the on-wire formats:

- `zdm.state/v1` — the model state bundle schema (§2.2.3).
- `zdm.federation/v1` — the federation delta schema (§4.4).
- `zdm.audit/v1` — the audit log entry schema (already stable from
  Phase 3; Phase 4 just names it).

The `GET /v2/version` endpoint reports all three. A v2 server refuses a
state bundle or delta whose schema version it does not recognise,
returning `409 Conflict` with the supported versions in the body.

---

## 7. File-by-file change map

### 7.1 New files (Phase 4)

```
src/zero_data_model/platform/         # §2 — 11 files (see §2.1)
src/zero_data_model/templates/        # §3 — 4 sub-packages (see §3.1)
src/zero_data_model/federation/       # §4 — 9 files (see §4.3)
src/zero_data_model/ethics/           # §5 — 6 files (see §5.1)
src/zero_data_model/registry/         # §6 — 6 files (see §6.1)
docs/superpowers/specs/2026-07-24-phase4-ecosystem-design.md  # this doc
tests/test_platform_api.py            # §2 endpoints
tests/test_platform_sdk.py            # §2.5 client
tests/test_templates.py               # §3 benchmark harness
tests/test_federation.py              # §4 round protocol + DP
tests/test_ethics.py                  # §5 alignment + e-stop + bias
tests/test_registry.py                # §6 cards + regression
```

### 7.2 Modified files (Phase 4)

- `src/zero_data_model/api.py` — replace the 501 branch in
  `version_negotiation` with a conditional mount of
  `platform.app.v2_router` when `ZDM_ENABLE_PLATFORM=1`. ~10 lines.
- `src/zero_data_model/pcn/hierarchical_model.py` — add optional
  `value_alignment`, `emergency_stop`, `bias_detector` hooks in
  `think()`, gated by flags. Each hook is a single try/except block
  mirroring the Phase 3 pattern. ~60 lines.
- `src/zero_data_model/__init__.py` — re-export `ZeroDataClient` and
  `create_app` from `platform` when the flag is on.
- `sdk/python/` — re-export `ZeroDataClient` from
  `zero_data_model.platform.sdk`.
- `pyproject.toml` — add optional extras group `[platform]` (httpx,
  pyjwt, multipart), `[federation]` (networkx), `[templates]` (pyyaml).
- `.github/workflows/ci.yml` — add a `phase4` matrix leg that sets
  `ZDM_ENABLE_PLATFORM=1` and runs the new test files.
- `.github/workflows/release.yml` — add the card-build step from §6.2.
- `docs/api/rest.md` — add a v2 section.
- `docs/api/python-sdk.md` — document `ZeroDataClient`.

### 7.3 Unchanged files

Every file not listed above is untouched. In particular:

- `src/zero_data_model/model.py` — no change. Phase 4 wraps; it does not
  touch the cognitive core.
- All Phase 1–7 module files — no change.
- `src/zero_data_model/security/` — reused as-is; `platform/auth.py`
  builds on `security/auth.py`'s helpers.
- `src/zero_data_model/observability/audit.py` — reused as-is;
  `ethics/audit_bridge.py` writes to the existing `AuditLogger`.
- `src/zero_data_model/persistence.py` — reused as-is by v1;
  `platform/serial.py` extends (does not replace) `ModelSerializer`.

---

## 8. Migration & backward compatibility

### 8.1 v1 callers

- **No breaking change.** Every v1 path (`/v1/...`, root `/health`,
  `/ready`, `/metrics`) behaves identically when Phase 4 flags are off.
- The `version_negotiation` middleware's 501 branch is the *only*
  observable change: a v1 caller sending
  `Accept: application/vnd.zdm.v2+json` to a Phase 4-enabled server now
  gets routed to v2 instead of a 501. This is the intended behaviour
  and is documented as the deprecation path for the Accept-header
  scheme in favour of explicit `/v2/...` paths.

### 8.2 v1 SDK consumers

- The existing `sdk/python/` README documents a planned client; Phase 4
  ships the actual `ZeroDataClient` and re-exports it from the same
  package. Anyone who built a client against the README's planned
  signature should re-check; the shipped signature is in §2.5.

### 8.3 Federation early adopters

- None exist today (federation is new in Phase 4), so there is no
  migration burden. The schema version `zdm.federation/v1` is the
  initial contract.

### 8.4 Audit log consumers

- The `AuditEntry` shape (§observability/audit.py) is unchanged. Phase 4
  *adds* event types only via `ethics/audit_bridge.py`
  (`VALUE_VIOLATION`, `EMERGENCY_STOP` already exist in the enum), so no
  existing consumer breaks.

---

## 9. Testing strategy

### 9.1 Unit tests (per package)

- **`test_platform_api.py`** — every v2 endpoint hit with a TestClient;
  auth scope matrix (read key → 200 on read, 403 on write; write key →
  200 on both; no key → 401); batch inference with mixed success/failure
  (one bad observation → per-item error, batch still 200).
- **`test_platform_sdk.py`** — `ZeroDataClient` against a mock
  `httpx.MockTransport`; retry on 503; idempotency-key on retry; JWT
  refresh on 401; `StateBundle` round-trip through `export_state` →
  `import_state`.
- **`test_templates.py`** — each template's `run_episode` produces a
  non-empty `EpisodeLog`; each benchmark returns finite floats; the
  Game-NPC latency assertion (p95 ≤ 5 ms) is enforced as a hard
  assertion in this file.
- **`test_federation.py`** — round protocol with a 3-edge in-memory
  cluster; DP noise distribution sanity (mean ≈ 0, variance ≈ σ²);
  `k_min` rejection of a lone poisoned edge; privacy accountant refuses
  to exceed budget.
- **`test_ethics.py`** — `ValueAlignment.constrain` moves an action
  toward the alignment gradient; soft-stop blocks interventions but not
  `think()`; hard-stop flushes audit and exits; bias detector flags a
  synthetic parity violation.
- **`test_registry.py`** — card build/compare; regression verdict on a
  latency regression; cognitive-fidelity failure on a behavioural
  change.

### 9.2 Integration test

A single end-to-end test (`tests/test_phase4_e2e.py`) wires:

1. A core node with federation + ethics on.
2. Two edge nodes contributing deltas.
3. One round completes; edges consume the merged delta.
4. An intervention is issued, audited, and visible via `/v2/audit-log`.
5. A bias scan runs and (with a crafted context) flags a violation.
6. A regression report is built comparing the live model to a baseline
   card.

This test runs only in the `phase4` CI matrix leg (it needs the
optional deps).

### 9.3 Performance regression

`benchmark.py` (already at the repo root) is extended with a
`--phase4` flag that runs the Game-NPC latency benchmark and asserts
the p95 cap. This is the same gate the CI regression detector uses
(§6.3), run from the same harness so local and CI numbers are
directly comparable.

---

## 10. Rollout plan

Phase 4 ships behind feature flags, in four independently-reviewable
chunks:

1. **Platform v2 + SDK** (§2). Unblocks external consumers. No cognitive
   change. Flag: `ZDM_ENABLE_PLATFORM`.
2. **Ethics & safety** (§5). Unblocks safety-gated deployments. Adds
   `think()` hooks. Flags: `enable_value_alignment`,
   `enable_emergency_stop`, `enable_bias_detector`.
3. **Templates** (§3). Unblocks the three named scenarios. No new
   cognitive code; pure configuration + adapters. Flag:
   `ZDM_ENABLE_TEMPLATES`.
4. **Federation** (§4). Unblocks multi-instance deployments. Flag:
   `ZDM_ENABLE_FEDERATION`.

Each chunk is a separate PR with its own spec section, tests, and CI
matrix leg. The registry (§6) ships with chunk 1 because the regression
detector is needed to gate the other chunks.

---

## 11. Open questions

These are resolved by the design above but are called out so the
reviewer can challenge them:

1. **Median vs. mean for federation aggregation.** Median is robust to a
   single poisoned edge but costs O(N log N) per edge group. With
   default `k_min=2` and expected < 50 edges per round, the cost is
   negligible. If round sizes grow, switch to a trimmed mean.
2. **Soft constraint vs. hard veto for value alignment.** This spec
   chooses soft (nudge). A hard veto is more safety-preserving but can
   deadlock an agent whose every action violates a dimension. The
   `max_nudge` parameter is the escape valve.
3. **File vs. SQLite for the registry.** File is simpler and append-only;
   SQLite scales better. The spec defaults to file and makes SQLite an
   env-var switch. If we ever need range queries over cards, switch the
   default.
4. **Accept-header version negotiation removal.** §2 says v2 uses path-
   based versioning. The v1 Accept-header hook is removed. No caller is
   known to use it (the 501 has been the response since the hook was
   added), so this is safe; flagged here for visibility.

---

## 12. Glossary

- **Aggregator** — the core-side service that merges federation deltas.
- **Card** — a `ModelCard`; the registry's unit of a model version.
- **Delta** — a `KnowledgeDelta`; the federated-learning unit of
  transfer.
- **Edge** — a tenant model instance that contributes to and consumes
  from federation.
- **Hard stop** — the irreversible emergency-stop level that exits the
  process.
- **Intervention** — a remote write into a named internal module via
  `/v2/intervention`.
- **Round** — one federation cycle: extract → DP → submit → aggregate →
  publish → apply.
- **Soft stop** — the reversible emergency-stop level that pauses writes
  but keeps serving reads.
- **State bundle** — the multipart payload of `GET /v2/model/state`.
