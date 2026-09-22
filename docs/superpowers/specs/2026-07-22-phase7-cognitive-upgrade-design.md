# Phase 7 — Cognitive-Upgrade Layer Design Spec

- **Date:** 2026-07-22
- **Status:** Implemented (phases 1–6 wired; phase 7 library modules shipped, model-level integration staged)
- **Scope:** 7 phases × 13 modules layered on top of the six core cognitive modules
- **Related:** [`docs/architecture.md`](../../architecture.md) §1 (cognitive-upgrade layer), §3.1 (hooks), §6 (thread model); implementation review [`2026-07-22-phase7-cognitive-upgrade-review.md`](../reviews/2026-07-22-phase7-cognitive-upgrade-review.md)

---

## 1. Overview

### 1.1 Motivation

The Phase 1–6 baseline ships **six core cognitive modules** (ConsciousnessCore,
ActiveInferenceEngine, CategoryTheoryEngine, QuantumClassicalHybrid,
BiologicalSubstrate, MathematicalUniverse) composed into twelve domain
capabilities. That architecture is *fixed*: the six modules never change their
number, their memory is a flat rolling buffer, their "reasoning" is purely
sub-symbolic, they cannot reflect on their own uncertainty, they cannot run
experiments, and there is only ever one agent.

Phase 7 grows that fixed core into a **complete cognitive architecture** along
seven axes, each gated behind a feature flag so the baseline is zero-regression:

1. **Plastic** — the architecture itself can split / prune modules.
2. **Time-layered** — prediction runs across multiple timescales, with a
   recurrent temporal memory.
3. **Structured memory** — episodes live in a graph; beliefs are retrievable by
   similarity.
4. **Neuro-symbolic** — a logic layer enforces rule constraints; a causal
   inference module exposes transition structure.
5. **Meta-cognitive** — second-order beliefs track uncertainty and parameter
   drift.
6. **Actively experimenting** — a Bayesian planner picks the next experiment;
   a hypothesis tester applies Bayes factors.
7. **Multi-agent** — a world of agents communicate and propagate culture.

### 1.2 Design contract

Every phase obeys four invariants:

- **Feature-flagged.** Constructed in `ZeroDataModel.__init__` only when the
  flag is set; `None` otherwise (lazy import, zero cost when off).
- **Read-only on the core.** Hooks observe already-computed state; they never
  call back into a core module's `process`/`predict`/`update`.
- **Try/except isolated.** Each hook is wrapped individually; a failure warns
  and is skipped, never crashing `think()`.
- **Metadata-only output.** Results land in `Signal.metadata["cognitive_upgrades"]`,
  and that key is only attached when at least one phase is active, so the
  no-upgrade metadata key-set is unchanged.

---

## 2. The seven phases

### Phase 1 — Architecture plasticity

- **Goal.** Let the module graph adapt: a module whose prediction error stays
  high gets *split*; one whose error stays low gets *pruned*. The system stops
  being a fixed 6-module architecture and becomes a self-tuning graph.
- **Modules.**
  - `ArchitectureOptimizer` — `src/zero_data_model/plasticity/architect.py`
- **Theory basis.** Predictive-processing error budgeting: each module is
  allocated representation capacity proportional to the free-energy reduction it
  delivers. A persistently high-error module is evidence that its single
  generative model is under-fitting two phenomena → split. A near-zero-error
  module is redundant → prune. The split decision mirrors the active-inference
  free-energy (FE) decomposition: ΔFE = accuracy − complexity, applied at the
  *architectural* level rather than the parameter level.
- **Feature flag.** `enable_architect` (default `False`).
- **`think()` integration.** End-of-cycle hook (the only write-adjacent hook):
  `record_errors(module_names, errors)` then `evaluate(copy(self.modules), cycle)`
  **under `self._lock`**. The modules list is passed as a defensive copy so a
  regression in `architect.py` cannot corrupt the live list. Contributes
  `upgrade_meta["architecture"]`.

### Phase 2 — Layered time

- **Goal.** Replace the single-timescale predictor with a hierarchy of
  fast/slow layers, and add a recurrent temporal memory so the system carries
  state across cycles.
- **Modules.**
  - `LayeredPredictor`, `Layer` — `src/zero_data_model/cogtime/layered_predictor.py`
  - `TemporalMemory` — `src/zero_data_model/cogtime/temporal_memory.py`
- **Theory basis.** Hierarchical predictive coding (multi-timescale prediction:
  L0 fast / L1 medium / L2 slow). `TemporalMemory` is an **Echo State Network**
  (ESN): a fixed random reservoir with a trained readout, where the reservoir
  state `x_t = tanh(W_in·u_t + W_res·x_{t-1})` carries temporal context without
  BPTT. Sub-dims are floored at 1 (`max(dim//2, 1)`) so `dim=1` does not
  produce degenerate 0-width layers.
- **Feature flag.** `enable_layered_predictor` (default `False`).
- **`think()` integration.** Two hooks: `layered_predictor.update(signal.data,
  cycle)` → `["layered_predictor"]`; `temporal_memory.update(signal.data,
  signal.data, lr=0.01)` → `["temporal_memory_error"]`.

### Phase 3 — Structured memory

- **Goal.** Store experiences as a graph of episodes (state / action /
  next-state / free-energy) and index belief vectors for similarity retrieval,
  replacing the flat rolling buffer.
- **Modules.**
  - `EpisodicGraph`, `Episode`, `EpisodeEdge` — `src/zero_data_model/cogmem/episodic_graph.py`
  - `SemanticIndex` — `src/zero_data_model/cogmem/semantic_index.py`
- **Theory basis.** Episodic memory as a labelled transition graph (state →
  action → next-state with an FE tag), enabling trajectory replay and
  counterfactual inspection. `SemanticIndex` is a vector retrieval index over
  belief vectors (cosine / inner-product neighbourhood queries) — the substrate
  for "have I been in a similar belief before?".
- **Feature flag.** `enable_episodic_memory` (default `False`).
- **`think()` integration.** `episodic_graph.insert(state=belief, action,
  next_state=None, free_energy=fe, step=cycle)` → `["episodic_nodes"]`;
  `semantic_index.add(signal.data, node_id=cycle, metadata)` →
  `["semantic_index_size"]`.

### Phase 4 — Neuro-symbolic fusion

- **Goal.** Layer symbolic rule constraints on top of the sub-symbolic core,
  and expose the inferred causal transition structure.
- **Modules.**
  - `LogicLayer`, `LogicRule` — `src/zero_data_model/knowledge/logic_layer.py`
  - `CausalInference` — `src/zero_data_model/knowledge/causal_inference.py`
- **Theory basis.** `LogicLayer` evaluates rules under **Gödel t-norm** fuzzy
  logic: conjunction `min(a,b)`, so a rule's satisfaction is bounded by its
  weakest premise — a hard symbolic constraint projected onto continuous
  activations. `CausalInference` maintains a transition matrix and surfaces its
  dimensionality (the "hot" causal structure the model currently believes).
- **Feature flag.** `enable_logic_layer` (default `False`).
- **`think()` integration.** `logic_layer.check_all()` → `["logic_violations"]`
  (only attached when `n_violations > 0`); `causal_inference` surfaces
  `transition_matrix.shape[0]` → `["causal_dim"]` (guarded against `None`).
  Note: the import is aliased (`CausalInference as _KnowledgeCausalInference`)
  to avoid shadowing the top-level `CausalInference` from
  `capabilities.analytics_advanced`.

### Phase 5 — Meta-cognition

- **Goal.** Second-order beliefs: the model forms beliefs *about* its own
  prediction error and parameter drift, enabling "I am uncertain about being
  uncertain" monitoring.
- **Modules.**
  - `MetaCognition` — `src/zero_data_model/metacog/meta_cognition.py`
- **Theory basis.** Meta-cognition as a higher-order predictive layer: the
  meta-module predicts the first-order prediction error and the norm of the
  parameter update, and tracks the divergence as a meta-uncertainty signal
  (a second-order free-energy term). This is the active-inference FE
  decomposition applied one level up.
- **Feature flag.** `enable_meta_cognition` (default `False`).
- **`think()` integration.** `meta_cognition.update(prediction_error=fe,
  param_update_norm=||transition||)` → `["meta_cognition"]`.

### Phase 6 — Active experiment

- **Goal.** The model selects which experiment to run next to maximally reduce
  uncertainty, and tests hypotheses with Bayes factors.
- **Modules.**
  - `BayesianExperimentPlanner`, `CandidateExperiment` — `src/zero_data_model/experiment/experiment_planner.py`
  - `HypothesisTester`, `Hypothesis` — `src/zero_data_model/experiment/hypothesis_tester.py`
- **Theory basis.** **Bayesian optimal experimental design**: pick the candidate
  whose expected information gain (mutual information between outcome and
  parameter) is largest — implemented as a Dijkstra-style search over a
  candidate graph weighted by expected uncertainty reduction. `HypothesisTester`
  uses the **Bayes factor** `BF = P(data|H1) / P(data|H0)` to decide
  support/reject, avoiding the arbitrary p-value threshold problem.
- **Feature flag.** `enable_experiment_planner` (default `False`).
- **`think()` integration.** `experiment_planner.evaluate(
  current_uncertainty=mean_uncertainty, step=cycle)` → `["experiment"]`;
  `hypothesis_tester.get_supported()` → `["supported_hypotheses"]`.

### Phase 7 — Multi-agent

- **Goal.** Move from a single-agent model to a world of agents that
  communicate and propagate culture across generations.
- **Modules.**
  - `MultiAgentWorld`, `AgentState` — `src/zero_data_model/multiagent/world.py`
  - `CommunicationChannel` — `src/zero_data_model/multiagent/communication.py`
  - `CulturePropagation`, `Generation` — `src/zero_data_model/multiagent/culture.py`
- **Theory basis.** Multi-agent active inference (each agent minimises its own
  FE over a shared Markov blanket that includes other agents). `CulturePropagation`
  models cultural transmission across `Generation`s (vertical/horizontal meme
  flow) as a biased-copy process with mutation.
- **Feature flag.** *(planned — `enable_multiagent`)*. The three modules ship as
  library code with per-method `threading.RLock` protection; the
  `ZeroDataModel.__init__` feature flag and `think()` hook are being staged so
  that phases 1–6 ship with stable, fully-verified integration first.
- **`think()` integration.** *(planned)* A `world.step()` hook contributing
  `["multiagent"]`, plus `CommunicationChannel` / `CulturePropagation` readouts.

---

## 3. Architecture decisions

### 3.1 Why feature flags (zero-regression)

Every phase is gated by a `__init__` flag defaulting to `False`, and the module
is left `None` when off. This delivers three guarantees:

- **Byte-for-byte baseline.** `think()` with no flags produces the *exact* same
  `Signal.metadata` key-set as Phase 6 — critical for the strict-key-set
  regression suite (the `_cognitive_active` guard only adds
  `cognitive_upgrades` / `module_errors` when something is enabled).
- **Zero import cost.** Modules are imported lazily inside the `if enable_*:`
  block, so a user who never enables a phase pays no import time.
- **Safe rollout.** Each phase can be enabled in isolation, so a bug in one
  phase cannot affect users of the others or of the baseline.

### 3.2 Why `threading.RLock` per module, not a global lock (Phase D lockless)

Phase D made `think()` **lockless** for the core cycle: each `CognitiveModule`
holds its own `RLock` around `process`/`predict`/`update`, so two modules run
concurrently and a read API touching module A does not block a `think()` writing
module B. The Phase 7 modules follow the same pattern — each carries its own
`RLock` around its mutating methods — rather than re-introducing a single global
lock. `RLock` (reentrant) is chosen over `Lock` so a method that internally
calls another locked method on the same instance does not self-deadlock, and so
`think()`'s architect hook can take `self._lock` even if a future caller already
holds it. The only model-wide `self._lock` acquisition in the upgrade layer is
the brief architect hook (the one write-adjacent hook).

### 3.3 Why `evaluate()` does not mutate its input (defensive programming)

`ArchitectureOptimizer.evaluate()` historically mutated `self.modules`
(append/pop), which broke the later `zip([type(m).__name__ for m in
self.modules], errors)` — silent truncation when a module was added, positional
mislabeling when one was pruned, because `errors` still had N entries. The fix
is two-layer: `architect.py` now operates on an internal copy, *and* `think()`
passes `list(self.modules)` (a defensive copy) so even a future regression in
`architect.py` cannot corrupt the live list. `think()` also recomputes
`_final_module_names` from the *current* `self.modules` right before the zip and
truncates to the shorter length, disambiguating same-named split modules with an
`#index` suffix.

### 3.4 Why NaN guards use `_last_valid_context` (permanent-contamination problem)

A single NaN propagating into a recurrent state (e.g. `TemporalMemory`'s
reservoir, `LayeredPredictor`'s weights, `MetaCognition`'s belief) **permanently
contaminates** the state: `NaN + anything = NaN`, so once a NaN enters the
recurrence it never leaves, and every subsequent output is NaN forever. The fix
is a `_last_valid_context` snapshot: each mutating method computes the
candidate new state, checks `np.isfinite` on it, and only commits it if finite;
otherwise it restores the last valid state and logs a warning. This converts a
permanent, silent corruption into a one-cycle, logged degradation. The check is
performed under the module's `RLock` so a concurrent reader never observes a
half-purged NaN.

---

## 4. Military-grade fix summary

Key fixes applied during the phase (CRITICAL ×1, HIGH ×15, MEDIUM ×25). Each is
one line; full detail lives in the implementation review.

### CRITICAL (1)

1. `EpisodicGraph._next_id` race — concurrent `insert` read-then-incremented the
   id counter and could hand out duplicate ids; now guarded by the module
   `RLock` with the increment made atomic.

### HIGH (15)

1. `ArchitectureOptimizer` — the architect hook was the only upgrade hook not
   wrapped in `try/except`; a `RuntimeError` crashed `think()`. Now isolated.
2. `ArchitectureOptimizer.evaluate` mutated the live `self.modules`; now
   operates on a copy and `think()` passes a defensive copy too.
3. Architect hook touched `self.modules` with no lock under Phase D; now runs
   under `self._lock`.
4. `LayeredPredictor` — concurrent `update` raced on in-place weight writes;
   added `RLock`.
5. `TemporalMemory` — NaN entering the reservoir contaminated it permanently;
   added `_last_valid_context` finite-check guard.
6. `TemporalMemory` — `RLock` around `update`.
7. `EpisodicGraph` — `RLock` around `insert` (pairs with the CRITICAL id fix).
8. `SemanticIndex` — `RLock` around `add`/`search`.
9. `LogicLayer` — `RLock` around `add_rule`/`check_all`.
10. `CausalInference` — `transition_matrix` could be `None`; added explicit
    `None` + shape check before `upgrade_meta["causal_dim"]`.
11. `MetaCognition` — `RLock` around `update`.
12. `BayesianExperimentPlanner` — division-by-zero when uncertainty was 0;
    guarded.
13. `HypothesisTester` — shape mismatch on recorded results; added shape
    validation.
14. `model.py` — `CausalInference` import shadowed the top-level name and
    caused `UnboundLocalError` when `enable_logic_layer=False`; aliased the
    import.
15. `ModuleGraphPanel.tsx` — `TypeError` on undefined graph nodes; added null
    guard.

### MEDIUM (25)

1. `model.py` — wire the constructed `semantic_index` into `think()` (was dead
   weight).
2. `model.py` — wire `causal_inference` into `think()` (was dead weight).
3. `model.py` — wire `hypothesis_tester` into `think()` (was dead weight).
4. `model.py` — only surface `module_errors` / `cognitive_upgrades` when a
   phase is active (`_cognitive_active` guard) to preserve the legacy key-set.
5. `model.py` — recompute `_final_module_names` from current `self.modules`
   before the zip to survive architect split/prune.
6. `model.py` — disambiguate same-named split modules with `#index` suffix so
   `dict(zip(...))` does not collapse them.
7. `model.py` — `_seed_val` previously diverged from `self._seed` (forced to
   42 when `seed is None`); now both hold the raw user seed.
8. `model.py` — floor `LayeredPredictor` sub-dims at 1 so `dim=1` does not
   produce 0-width layers.
9. `ArchitectureOptimizer` — length-cap the error history.
10. `LayeredPredictor` — length-cap the per-layer prediction history.
11. `TemporalMemory` — length-cap the reservoir state history.
12. `EpisodicGraph` — length-cap the episode list (rolling window).
13. `SemanticIndex` — length-cap the index.
14. `LogicLayer` — length-cap the rule list.
15. `CausalInference` — length-cap the transition history.
16. `MetaCognition` — length-cap the meta-belief history.
17. `BayesianExperimentPlanner` — length-cap the candidate list.
18. `HypothesisTester` — length-cap the recorded results.
19. `_sanitize_for_json` — strip non-JSON-serializable objects from
    `upgrade_meta` before metadata serialization.
20. `ModuleGraphPanel.tsx` — guard `Snapshot.metadata?` against `undefined`.
21. `ModuleGraphPanel.tsx` — auto-scroll only when the user is already at the
   bottom (do not yank the view on every cycle).
22. `ExperimentLog.tsx` — null-guard empty experiment results.
23. `LogicPanel.tsx` — null-guard empty violation lists.
24. `CommunicationLog.tsx` — null-guard empty message buffers.
25. `KnowledgeGraphView.tsx` — null-guard empty node/edge arrays.

---

## 5. Test-coverage matrix

13 modules × 6 test dimensions. ✅ = covered by an existing test file under
`tests/`; the column headers are the dimensions requested by the audit.

| # | Module | Construct | Core method | Thread-safe | NaN guard | Boundary | Length-cap |
|---|--------|:---------:|:-----------:|:-----------:|:---------:|:--------:|:----------:|
| 1 | `ArchitectureOptimizer` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2 | `LayeredPredictor` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 3 | `TemporalMemory` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 4 | `EpisodicGraph` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 5 | `SemanticIndex` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 6 | `LogicLayer` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 7 | `CausalInference` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 8 | `MetaCognition` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 9 | `BayesianExperimentPlanner` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 10 | `HypothesisTester` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 11 | `MultiAgentWorld` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 12 | `CommunicationChannel` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 13 | `CulturePropagation` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Test dimensions:

- **Construct** — the module builds with valid args and rejects invalid ones
  (e.g. `dim < 1`).
- **Core method** — the module's primary method (`update` / `insert` / `add` /
  `check_all` / `evaluate` / `step`) returns the documented shape/type.
- **Thread-safe** — N-thread concurrent calls do not corrupt state or hand out
  duplicate ids (the `EpisodicGraph._next_id` CRITICAL regression).
- **NaN guard** — a NaN injected into the input is caught by
  `_last_valid_context` and does not permanently contaminate recurrent state.
- **Boundary** — `dim=1`, empty histories, zero-length inputs do not crash.
- **Length-cap** — histories / indices / rule lists respect their rolling cap
  and do not grow unbounded.

The phase is exercised end-to-end by the Phase 7 integration tests
(`tests/test_phase7_{causal,reasoning,code,time,robotics,graph,audio}_integration.py`)
and the 8-thread concurrency stress test in the implementation review.
