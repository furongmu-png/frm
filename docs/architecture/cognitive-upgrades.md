# Cognitive Upgrades

Phase 7 grows the fixed six-module core into a complete cognitive architecture
along seven axes. Each phase is gated behind a `ZeroDataModel.__init__` feature
flag (all default `False`), and every phase obeys four invariants:

- **Feature-flagged.** Constructed only when the flag is set; `None` otherwise
  (lazy import, zero cost when off).
- **Read-only on the core.** Hooks observe already-computed state; they never
  call back into a core module's `process`/`predict`/`update`.
- **Try/except isolated.** Each hook is wrapped individually; a failure warns
  and is skipped, never crashing `think()`.
- **Metadata-only output.** Results land in `Signal.metadata["cognitive_upgrades"]`,
  and that key is only attached when at least one phase is active, so the
  no-upgrade metadata key-set is unchanged (zero-regression for strict-key-set
  tests).

## The seven phases

| # | Phase | Flag | Module(s) | Theory basis |
| --- | --- | --- | --- | --- |
| 1 | Architecture plasticity | `enable_architect` | `ArchitectureOptimizer` | Predictive-processing error budgeting (FE decomposition at the architectural level) |
| 2 | Layered time | `enable_layered_predictor` | `LayeredPredictor`, `TemporalMemory` | Hierarchical predictive coding + Echo State Network |
| 3 | Structured memory | `enable_episodic_memory` | `EpisodicGraph`, `SemanticIndex` | Episodic memory as labelled transition graph + vector retrieval |
| 4 | Neuro-symbolic fusion | `enable_logic_layer` | `LogicLayer`, `CausalInference` | Gödel t-norm fuzzy logic + causal transition inference |
| 5 | Meta-cognition | `enable_meta_cognition` | `MetaCognition` | Higher-order predictive layer (second-order free energy) |
| 6 | Active experiment | `enable_experiment_planner` | `BayesianExperimentPlanner`, `HypothesisTester` | Bayesian optimal experimental design + Bayes factors |
| 7 | Multi-agent | `enable_multiagent` *(planned)* | `MultiAgentWorld`, `CommunicationChannel`, `CulturePropagation` | Multi-agent active inference + cultural transmission |

### Phase 1 — Architecture plasticity

Let the module graph adapt: a module whose prediction error stays high gets
*split*; one whose error stays low gets *pruned*.

- **`think()` integration.** End-of-cycle hook (the only write-adjacent hook):
  `record_errors(module_names, errors)` then `evaluate(copy(self.modules), cycle)`
  **under `self._lock`**. The modules list is passed as a defensive copy so a
  regression in `architect.py` cannot corrupt the live list.
- **Output.** `upgrade_meta["architecture"]`.

See [Architecture Plasticity](../modules/plasticity.md).

### Phase 2 — Layered time

Replace the single-timescale predictor with a hierarchy of fast/slow layers, and
add a recurrent temporal memory so the system carries state across cycles.

- **Theory.** Hierarchical predictive coding (L0 fast / L1 medium / L2 slow).
  `TemporalMemory` is an **Echo State Network**: a fixed random reservoir with a
  trained readout, where the reservoir state `x_t = tanh(W_in·u_t + W_res·x_{t-1})`
  carries temporal context without BPTT. Sub-dims are floored at 1 (`max(dim//2, 1)`)
  so `dim=1` does not produce degenerate 0-width layers.
- **`think()` integration.** Two hooks: `layered_predictor.update(signal.data,
  cycle)` → `["layered_predictor"]`; `temporal_memory.update(signal.data,
  signal.data, lr=0.01)` → `["temporal_memory_error"]`.

See [Layered Time](../modules/layered-time.md).

### Phase 3 — Structured memory

Store experiences as a graph of episodes (state / action / next-state /
free-energy) and index belief vectors for similarity retrieval, replacing the
flat rolling buffer.

- **Theory.** Episodic memory as a labelled transition graph (state → action →
  next-state with an FE tag), enabling trajectory replay and counterfactual
  inspection. `SemanticIndex` is a vector retrieval index over belief vectors
  (cosine / inner-product neighbourhood queries) — the substrate for "have I
  been in a similar belief before?".
- **`think()` integration.** `episodic_graph.insert(state=belief, action,
  next_state=None, free_energy=fe, step=cycle)` → `["episodic_nodes"]`;
  `semantic_index.add(signal.data, node_id=cycle, metadata)` →
  `["semantic_index_size"]`.

See [Structured Memory](../modules/memory.md).

### Phase 4 — Neuro-symbolic fusion

Layer symbolic rule constraints on top of the sub-symbolic core, and expose the
inferred causal transition structure.

- **Theory.** `LogicLayer` evaluates rules under **Gödel t-norm** fuzzy logic:
  conjunction `min(a,b)`, so a rule's satisfaction is bounded by its weakest
  premise — a hard symbolic constraint projected onto continuous activations.
  `CausalInference` maintains a transition matrix and surfaces its dimensionality
  (the "hot" causal structure the model currently believes).
- **`think()` integration.** `logic_layer.check_all()` → `["logic_violations"]`
  (only attached when `n_violations > 0`); `causal_inference` surfaces
  `transition_matrix.shape[0]` → `["causal_dim"]` (guarded against `None`).

See [Neuro-Symbolic](../modules/neuro-symbolic.md).

### Phase 5 — Meta-cognition

Second-order beliefs: the model forms beliefs *about* its own prediction error
and parameter drift, enabling "I am uncertain about being uncertain" monitoring.

- **Theory.** Meta-cognition as a higher-order predictive layer: the meta-module
  predicts the first-order prediction error and the norm of the parameter
  update, and tracks the divergence as a meta-uncertainty signal (a
  second-order free-energy term).
- **`think()` integration.** `meta_cognition.update(prediction_error=fe,
  param_update_norm=||transition||)` → `["meta_cognition"]`.

See [Meta-Cognition](../modules/meta-cognition.md).

### Phase 6 — Active experiment

The model selects which experiment to run next to maximally reduce uncertainty,
and tests hypotheses with Bayes factors.

- **Theory.** **Bayesian optimal experimental design**: pick the candidate whose
  expected information gain (mutual information between outcome and parameter) is
  largest — implemented as a Dijkstra-style search over a candidate graph
  weighted by expected uncertainty reduction. `HypothesisTester` uses the **Bayes
  factor** `BF = P(data|H1) / P(data|H0)` to decide support/reject, avoiding the
  arbitrary p-value threshold problem.
- **`think()` integration.** `experiment_planner.evaluate(
  current_uncertainty=mean_uncertainty, step=cycle)` → `["experiment"]`;
  `hypothesis_tester.get_supported()` → `["supported_hypotheses"]`.

See [Active Experiment](../modules/experiment.md).

### Phase 7 — Multi-agent

Move from a single-agent model to a world of agents that communicate and
propagate culture across generations.

- **Theory.** Multi-agent active inference (each agent minimises its own FE over
  a shared Markov blanket that includes other agents). `CulturePropagation`
  models cultural transmission across `Generation`s (vertical/horizontal meme
  flow) as a biased-copy process with mutation.
- **Status.** The three modules ship as library code with per-method
  `threading.RLock` protection; the `ZeroDataModel.__init__` feature flag and
  `think()` hook are being staged so that phases 1–6 ship with stable,
  fully-verified integration first.
- **`think()` integration.** *(planned)* A `world.step()` hook contributing
  `["multiagent"]`, plus `CommunicationChannel` / `CulturePropagation` readouts.

See [Multi-Agent](../modules/multi-agent.md).

## `think()` hook call order

After the core `process → integrate → reflect → predict → update` cycle
completes, `think()` runs the cognitive-upgrade hooks — but only for the phases
whose feature flag was set. Each hook reads already-computed state and writes
into a local `upgrade_meta` dict; that dict is only attached to the returned
`Signal.metadata["cognitive_upgrades"]` when at least one phase is active.

The hook call order (each guarded by `is not None`, each wrapped in its own
`try/except`; failures only `warning`-log and `pass`, never crash `think()`):

```
core cycle done
      │
      ▼
1. architect            record_errors(names, errors) → evaluate(copy(modules), cycle)
                        [under self._lock; result → upgrade_meta["architecture"]]
2. layered_predictor    update(signal.data, cycle)    → upgrade_meta["layered_predictor"]
   temporal_memory      update(signal.data, signal.data, lr=0.01)
                                                    → upgrade_meta["temporal_memory_error"]
3. episodic_graph       insert(state=belief, action, free_energy, step=cycle)
                                                    → upgrade_meta["episodic_nodes"]
   semantic_index       add(signal.data, node_id=cycle, metadata) → ["semantic_index_size"]
4. logic_layer          check_all()  → upgrade_meta["logic_violations"] (only if n>0)
   causal_inference     surface transition_matrix shape → upgrade_meta["causal_dim"]
5. meta_cognition       update(prediction_error=fe, param_update_norm) → ["meta_cognition"]
6. experiment_planner   evaluate(current_uncertainty=mean_uncertainty, step=cycle) → ["experiment"]
   hypothesis_tester    get_supported() → upgrade_meta["supported_hypotheses"]
      │
      ▼
upgrade_meta  ──►  Signal.metadata["cognitive_upgrades"]   (only if any phase active)
```

## Key invariants

- **Read-only on the core.** Hooks observe `errors`, `signal.data`,
  `active_inference.free_energy_history`, etc., but never call back into a core
  module's `process`/`predict`/`update`. The architect is the one
  write-adjacent hook (it can split/prune `self.modules`); it is therefore run
  under `self._lock` and is passed a **defensive copy** of the modules list so a
  regression in `architect.py` cannot corrupt the live list.
- **Try/except per hook.** Every hook is wrapped individually. A failure in
  `meta_cognition.update` cannot prevent `hypothesis_tester.get_supported` from
  running, and none of them can abort `think()`.
- **Conditional metadata.** `_cognitive_active` checks whether any upgrade
  attribute is non-`None`; only then are `module_errors` and
  `cognitive_upgrades` added to the returned metadata, preserving the legacy
  key-set otherwise.
