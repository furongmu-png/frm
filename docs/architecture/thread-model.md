# Thread Model

The model supports concurrent callers (a long-running `think()` plus concurrent
read APIs like `classify_text`, `detect_anomalies`) via a **Phase D lockless**
design with narrow per-module locks.

## Phase D lockless core

`think()` does **not** hold a global lock for the duration of the cycle. Instead:

- Each core `CognitiveModule` owns a private `threading.RLock` that serializes
  only its own `process` / `predict` / `update`. Two different modules can run
  concurrently (via `ParallelExecutor.map_modules`), and a read API that touches
  only module A does not block a `think()` writing to module B.
- `ZeroDataModel._lock` (a single `threading.RLock`) is retained but acquired
  only around the few operations that are genuinely model-wide: the
  `cycle_count` counter increment, `pickle`/`__getstate__` (so pickle cannot
  serialize a half-mutated array), and the `architect` hook (the one
  upgrade hook that is write-adjacent to `self.modules`). `RLock` is reentrant,
  so a caller that already holds `_lock` (e.g. a future `think` that calls
  pickle internally) does not self-deadlock.
- The read-only analytics methods (`compute_free_energy`, `classify_text`, …)
  are pure functions of their arguments and never touch the shared RNG, so they
  remain safe to call concurrently without any lock.

## Cognitive-upgrade module locking

Each Phase 7 module that holds mutable state carries its own
`threading.RLock` and takes it around every state-mutating method, matching the
core-module pattern:

| Module | Lock scope |
| --- | --- |
| `ArchitectureOptimizer` | `record_errors` / `evaluate` (error history + pending split/prune plan read by `think()`'s architect hook, must not be torn) |
| `LayeredPredictor` / `TemporalMemory` | `update` (in-place weight / reservoir updates) |
| `EpisodicGraph` | `insert` and `_next_id` (the monotonically increasing episode id is the CRITICAL race fixed in this phase; the id counter is read-then-incremented, so without the lock two concurrent `insert` calls could hand out the same id) |
| `SemanticIndex` | `add` / `search` |
| `LogicLayer` / `CausalInference` | `add_rule` / `check_all` and the transition-matrix update |
| `MetaCognition` | `update` |
| `BayesianExperimentPlanner` / `HypothesisTester` | `evaluate` / `record_result` / `get_supported` |
| `MultiAgentWorld` / `CommunicationChannel` / `CulturePropagation` | `step` / `send` / `propagate` |

Because each lock is **per-module** (not a single global lock), enabling several
phases does not serialize the hooks against each other any more than necessary;
the only model-wide serialization is the architect hook's brief `self._lock`
acquisition. NaN guards (`_last_valid_context`) and length-capped histories are
applied under the same lock so a concurrent read never observes a NaN that a
writer is in the middle of purging.

## Concurrency in the FastAPI service

The REST API builds on this thread model:

- `POST /think` is an `async def` handler that offloads the CPU-bound
  `model.think(...)` call to a thread pool via `asyncio.to_thread`, so the
  event loop is not blocked while other async endpoints (e.g.
  `/emergence/perceive-topology`, `/discover-causal-dynamics`) are served.
- Read endpoints (`/classify`, `/similarity`, `/forecast`, …) take
  `model._lock` so a concurrent `/think` cannot mutate module arrays in place
  (`+=`) while a handler reads them. NumPy `+=` is **not** atomic, so an
  unsynchronised read can observe a half-mutated `topos.classifier` /
  `emission` / `classical_weights` and return garbled topic scores. `_lock` is
  an `RLock` so the same thread can re-enter it.
- After `think()` returns, the handler reads `cycle_count` and the last free
  energy under `model._lock` so the two Prometheus gauges are consistent with
  each other AND with this `think()` call. Without the lock, a concurrent
  `think()` could land between the call return and these reads — incrementing
  `cycle_count` and appending a new free energy value — so `zdm_cycle_count`
  would point at cycle N+1 while `zdm_free_energy_last` would point at the next
  cycle's energy, and neither would match the `cycle` field returned in the HTTP
  response body.

## Per-module RNG

Round-4 audit RNG-1: `ZeroDataModel` spawns an **independent child Generator**
for each cognitive module via `np.random.default_rng(seed).spawn(n)`. NumPy's
`Generator` is **not** thread-safe, so the earlier approach of sharing one
`self._rng` across all modules was still a race when `parallel_executor`
dispatched `module.process()` to `ThreadPoolExecutor` workers in unseeded mode.

`spawn(n)` produces `n` bit-stream-independent child Generators that can be used
concurrently from different threads. Seeded mode keeps reproducibility (spawn is
deterministic given the parent seed); the children are consumed in a fixed order
so a seeded model's `think()` sequence remains identical across runs.
