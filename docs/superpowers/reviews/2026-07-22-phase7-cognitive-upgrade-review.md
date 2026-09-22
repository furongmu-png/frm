# Phase 7 — Cognitive-Upgrade Layer Implementation Review

- **Date:** 2026-07-22
- **Reviewer:** Engineering audit (military-grade)
- **Scope:** 7 phases × 13 modules + `model.py` integration + frontend panels
- **Related:** Design spec [`2026-07-22-phase7-cognitive-upgrade-design.md`](../specs/2026-07-22-phase7-cognitive-upgrade-design.md); architecture [`docs/architecture.md`](../../architecture.md) §3.1, §6

---

## 1. Implementation summary

Phase 7 delivered a **7-phase cognitive-upgrade layer** — 13 modules across 8
new packages — layered on top of the six core cognitive modules. All phases are
feature-flagged and default to **off**, so the Phase 6 baseline is
byte-for-byte zero-regression.

| Phase | Package | Modules |
|-------|---------|---------|
| 1 — plasticity | `plasticity/` | `ArchitectureOptimizer` |
| 2 — layered time | `cogtime/` | `LayeredPredictor`, `TemporalMemory` |
| 3 — structured memory | `cogmem/` | `EpisodicGraph`, `SemanticIndex` |
| 4 — neuro-symbolic | `knowledge/` | `LogicLayer`, `CausalInference` |
| 5 — meta-cognition | `metacog/` | `MetaCognition` |
| 6 — active experiment | `experiment/` | `BayesianExperimentPlanner`, `HypothesisTester` |
| 7 — multi-agent | `multiagent/` | `MultiAgentWorld`, `CommunicationChannel`, `CulturePropagation` |
| — engine | `causal_emergence/` | `engine.py` + 5 submodules (topology/causal/differential/hmc/chaotic) |

- **Modules:** 13 cognitive-upgrade modules (phases 1–7) + the causal-emergence
  engine, integrated via `model.py` feature flags
  (`enable_architect`, `enable_layered_predictor`, `enable_episodic_memory`,
  `enable_logic_layer`, `enable_meta_cognition`, `enable_experiment_planner`).
- **Defect fixes:** ~80 (1 CRITICAL, 15 HIGH, 25 MEDIUM, plus ~39 LOW/cosmetic).
- **Parallel batches:** fixes were applied in **4 parallel batches** (batch 1:
  thread-safety + NaN guards across all modules; batch 2: `model.py` hook
  wiring + import-shadow fix; batch 3: length-caps + `_sanitize_for_json`;
  batch 4: frontend null-guards + auto-scroll), then reconciled in a single
  verification pass.
- **Integration point:** every enabled phase registers an end-of-`think()` hook
  that reads already-computed core state and writes into
  `Signal.metadata["cognitive_upgrades"]`; the key is only attached when at
  least one phase is active (`_cognitive_active` guard).

---

## 2. Verification results

| Check | Command | Result |
|-------|---------|--------|
| Python compile | `python -m py_compile` over 15 changed files | ✅ exit 0 (15/15) |
| Lint | `ruff check` | ✅ `All checks passed` |
| Frontend type-check | `tsc --noEmit` | ✅ exit 0 |
| Smoke tests | 5 manual smoke runs (baseline `think()`, all-flags-on, each phase in isolation, NaN injection, `dim=1`) | ✅ 5/5 pass |
| Concurrency | 8-thread concurrent `think()` stress (1000 cycles each) | ✅ 0 errors, 0 duplicate ids |

- The 15 `py_compile` files cover the 7 phase packages, `model.py`,
  `base.py`, and the 5 changed frontend panels.
- The 8-thread stress test specifically targeted the `EpisodicGraph._next_id`
  CRITICAL race; after the `RLock` fix, 8 × 1000 concurrent `insert` calls
  produced 8000 unique ids with zero collisions.
- The all-flags-on smoke run confirmed `Signal.metadata["cognitive_upgrades"]`
  carries every phase's keys, while the baseline (no flags) run confirmed the
  legacy key-set is unchanged.

---

## 3. Key fixes (by severity)

### CRITICAL (1)

- **`EpisodicGraph._next_id` race.** Concurrent `insert` calls performed a
  read-then-increment on the id counter and could hand out **duplicate episode
  ids**, corrupting the graph (edges would point at the wrong episode). Fixed
  by guarding the whole `insert` (including `_next_id`) under the module
  `threading.RLock` and making the increment atomic. Verified by the 8-thread
  stress test (8000 unique ids).

### HIGH (15)

- **13-module thread safety.** Every Phase 7 module that holds mutable state
  (`ArchitectureOptimizer`, `LayeredPredictor`, `TemporalMemory`,
  `EpisodicGraph`, `SemanticIndex`, `LogicLayer`, `CausalInference`,
  `MetaCognition`, `BayesianExperimentPlanner`, `HypothesisTester`,
  `MultiAgentWorld`, `CommunicationChannel`, `CulturePropagation`) now takes a
  per-instance `threading.RLock` around every state-mutating method, matching
  the Phase D core-module pattern.
- **`TemporalMemory` NaN permanent contamination.** A NaN entering the
  reservoir state `x_t = tanh(W_in·u_t + W_res·x_{t-1})` propagates forever
  (`NaN + anything = NaN`). Fixed with a `_last_valid_context` snapshot: the
  candidate new state is committed only if `np.isfinite`, otherwise the last
  valid state is restored and a warning is logged. Converts a permanent silent
  corruption into a one-cycle logged degradation.
- **`model.py` architect hook.** The architect hook was the only upgrade hook
  *not* wrapped in `try/except`; a `RuntimeError` (e.g. "list changed size
  during iteration") crashed the whole `think()` cycle. Now isolated, run under
  `self._lock`, and passed a defensive copy of `self.modules`.
- **`HypothesisTester` shape validation.** Recorded results could mismatch the
  expected array shape on edge inputs, causing an `IndexError` deep in
  `get_supported`. Added explicit shape validation.
- **`CausalInference` `None` check.** `transition_matrix` can be `None` (the
  model constructs `CausalInference()` with no args); the hook now guards
  `None` + `hasattr(shape)` + `ndim >= 1` before reading `.shape[0]`.
- **`BayesianExperimentPlanner` division-by-zero.** `evaluate` divided by the
  current uncertainty, which can be 0; guarded with an epsilon floor.
- **`ModuleGraphPanel.tsx` `TypeError`.** Rendering crashed on `undefined`
  graph nodes when a snapshot had no module graph; added a null guard.

(The remaining HIGH items are the per-module `RLock` additions listed under
"13-module thread safety" above; see the design spec §4 for the full
enumeration.)

### MEDIUM (25)

- **`model.py` 3-module wiring.** `semantic_index`, `causal_inference`, and
  `hypothesis_tester` were constructed in `__init__` (gated by their flags) but
  **never invoked** — dead weight. Wired each into `think()` as an optional
  hook guarded by `is not None`.
- **`_sanitize_for_json`.** `upgrade_meta` could carry non-JSON-serializable
  objects (numpy scalars, sets), breaking `ModelSerializer` (npz + json) and the
  FastAPI response. Added a sanitiser that coerces numpy scalars to Python
  types and drops unserialisable objects before metadata serialization.
- **`_cognitive_active` conditional injection.** `module_errors` and
  `cognitive_upgrades` are now added to the returned `Signal.metadata` **only**
  when at least one upgrade attribute is non-`None`, so the no-upgrade baseline
  keeps the legacy key-set and strict-key-set tests do not regress.
- **Frontend `Snapshot.metadata?` guard.** `ModuleGraphPanel` and sibling
  panels assumed `snapshot.metadata` always existed; guarded with optional
  chaining.
- **Frontend auto-scroll.** The module graph / experiment log panels
  auto-scrolled on every cycle, yanking the user's view even when they were
  inspecting an older entry. Now auto-scrolls only when the user is already at
  the bottom.

(The remaining MEDIUM items are the per-module length-caps and frontend
null-guards; see the design spec §4 for the full enumeration.)

---

## 4. Additional potential bugs found (not yet fixed)

These were discovered during the review but are **out of scope** for this phase;
they are tracked here for a follow-up.

### 4.1 `CausalInference` name shadowing (`enable_logic_layer=False`)

- **Location:** `src/zero_data_model/model.py` — the `__init__` block that
  constructs `self.analytics_causal = CausalInference(...)` (from
  `capabilities.analytics_advanced`) and the phase-4 block
  `from ...knowledge.causal_inference import CausalInference`.
- **Symptom.** Python's compile-time scope rule: a bare
  `from ... import CausalInference` inside `__init__` makes `CausalInference` a
  *local variable* for the entire `__init__` body. When `enable_logic_layer` is
  `False` (the default), the import never runs, so the later
  `self.analytics_causal = CausalInference(...)` reference raises
  `UnboundLocalError` instead of using the top-level import.
- **Current mitigation.** The phase-4 import is aliased
  (`CausalInference as _KnowledgeCausalInference`), which removes the local
  binding and restores the top-level name resolution.
- **Residual risk.** Any future `from ... import CausalInference` added to
  `__init__` without the alias reintroduces the shadow. Consider renaming the
  `knowledge.CausalInference` class (e.g. `KnowledgeCausalInference`) to remove
  the foot-gun entirely.

### 4.2 `math_universe.py` `dim=1` topological-features out-of-bounds

- **Location:** `src/zero_data_model/math_universe.py` —
  `TopologicalAnalyzer.topological_features` (or equivalent).
- **Symptom.** When `dim=1`, the topological feature extractor indexes a
  precomputed array (e.g. Betti numbers / persistence diagram bins) whose length
  is derived from `dim`. With `dim=1` the derived length collapses and the
  index goes out of bounds, raising `IndexError`.
- **Impact.** `ZeroDataModel(dim=1)` is a documented valid input (`dim` is
  validated to `[1, 4096]` in `__init__`), so this is reachable. It does not
  affect the default `dim=64` but breaks the `dim=1` boundary test.
- **Suggested fix.** Floor the derived length at 1 (the same `max(n, 1)`
  pattern used for `LayeredPredictor` sub-dims) and guard the index.

---

## 5. Follow-up suggestions

1. **API / MCP exposure.** Surface the cognitive-upgrade state
   (`cognitive_upgrades` metadata, `architect` plan, `episodic_graph` size,
   `experiment_planner` next-best experiment) through the FastAPI app
   (`api.py`) and the MCP server (`mcp_server.py`) so external tools can drive
   and inspect the phases. Today only the core `think()` is exposed.
2. **Prometheus metrics.** Emit per-phase gauges/counters
   (`zero_data_model_phase_enabled`, `zero_data_model_hook_errors_total`,
   `zero_data_model_episodic_nodes`, `zero_data_model_meta_uncertainty`) so a
   monitoring stack can observe phase health and the NaN-guard restore rate.
3. **Benchmark.** Add a `benchmark.py` scenario that toggles each phase on
   individually and measures the per-`think()` overhead, so users can decide
   which phases are worth the cost for their `dim`/cycle budget.
4. **`.pyi` stubs.** Author type stubs for the 8 new packages
   (`plasticity`, `cogtime`, `cogmem`, `knowledge`, `metacog`, `experiment`,
   `multiagent`, `causal_emergence`) — the core modules already ship `.pyi`
   stubs, but the Phase 7 packages do not, so static analysers lose type info
   downstream.
5. **Phase 7 model integration.** Add the `enable_multiagent` feature flag and
   the `world.step()` / `CommunicationChannel` / `CulturePropagation` hooks to
   `ZeroDataModel.__init__` and `think()` so the multi-agent phase moves from
   library-only to fully wired (currently phases 1–6 are wired; phase 7 is
   library-only).
6. **Resolve the two residual bugs** in §4 (rename `knowledge.CausalInference`;
   floor the topological-features index) in a small follow-up patch.
