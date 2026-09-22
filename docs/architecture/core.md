# Core Modules

The six core modules are `CognitiveModule` subclasses that each implement the
`process(Signal) -> Signal`, `predict(Signal) -> Prediction`,
`update(prediction_error) -> None` trio defined in `src/zero_data_model/base.py`.
`ZeroDataModel` owns one instance of each and fans every `think()` cycle out to
all six in parallel via `ParallelExecutor.map_modules()`.

| Module | Source file | Theory basis |
| --- | --- | --- |
| `ConsciousnessCore` | `consciousness_core.py` | Global Workspace Theory (GWT), self-model, predictive layer |
| `ActiveInferenceEngine` | `active_inference.py` | Free Energy Principle, generative model, Markov blanket, homeostatic controller |
| `CategoryTheoryEngine` | `category_engine.py` | Category / topos theory, functors, isomorphism detection |
| `QuantumClassicalHybrid` | `quantum_hybrid.py` | Variational quantum circuits (VQC), quantum annealing |
| `BiologicalSubstrate` | `biological.py` | DNA storage crossover, morphogenetic field, cellular automata |
| `MathematicalUniverse` | `math_universe.py` | Information geometry, topological data analysis (TDA), fractal generation |

## 1. ConsciousnessCore

`consciousness_core.py` implements a Global-Workspace-style integration layer
with a `SelfModel` and a `PredictiveLayer`.

- `GlobalWorkspace` integrates per-module outputs into a broadcast signal.
- `SelfModel` carries `self_confidence` and history length — the metacognitive
  readout surfaced in `Signal.metadata["self_reflection"]`.
- `PredictiveLayer` predicts the next signal; its prediction error feeds the
  model-wide `mean_err` that nudges every module's parameters.

`think()` calls `consciousness.reflect()` after integration, which returns a
`Signal` whose `metadata` carries `self_confidence` and `history_len`.

## 2. ActiveInferenceEngine

`active_inference.py` implements the Free Energy Principle. It owns a
`GenerativeModel`, a `MarkovBlanket` separating internal state from observations,
and a `HomeostaticController` that drives actions to minimise free energy.

- `free_energy_history` — appended every cycle; the last value is exposed via
  the `zdm_free_energy_last` Prometheus gauge and the `last_fe` field returned
  by `POST /think`.
- The mean prediction error across all modules (`mean_err`) is fed back into
  every module's `update()`, closing the predictive-processing loop.

## 3. CategoryTheoryEngine

`category_engine.py` implements category / topos theory primitives — `Category`,
`Functor`, and a `ToposEngine` that classifies signals by isomorphism. This is
what powers cross-domain transfer: the same functor can map an NLP embedding
into a vision embedding space without any learned alignment.

`ZeroShotClassifier` composes `TextEncoder` + `ActiveInferenceEngine` +
`CategoryTheoryEngine` to classify text into a topic with no training data.

## 4. QuantumClassicalHybrid

`quantum_hybrid.py` pairs a `VariationalQuantumCircuit` (RY + CNOT ansatz run
through `StatevectorSampler` when Qiskit is available) with a
`QuantumAnnealer` whose inner loop is JIT-compiled with `numba.njit(cache=True)`
when numba is available, with a pure-NumPy fallback (`_anneal_inner`).

The annealer exposes `.jit` so callers can tell which path is active — see
`ZeroDataModel.hardware_info["annealer_jit"]`.

When a `seed` is passed to `ZeroDataModel`, the deterministic
`SimulatorQuantumBackend` is forced (the Qiskit sampler performs stochastic
shot-based measurement that does not draw from numpy's global RNG, so
re-seeding cannot make it reproducible).

## 5. BiologicalSubstrate

`biological.py` is the **zero-data input source**. It owns:

- `DNAStorage` — `generate()` produces a DNA-crossover signal. Combined with
  the fractal signal (50/50 blend) in `ZeroDataModel._self_generate()`, this is
  how `think(None)` produces a coherent output with no input.
- `MorphogeneticField` — `develop()` applies morphogenetic diffusion.
- `CellularAutomata` — discrete evolution step used by `PatternMiner` and the
  causal-emergence engine.

## 6. MathematicalUniverse

`math_universe.py` is the second half of the zero-data input source:

- `InformationGeometry` — Fisher-metric-style geometry over belief vectors.
- `TopologicalAnalyzer` — topological data analysis (Betti numbers, persistence
  diagrams) used by the causal-emergence engine.
- `FractalGenerator` — `generate()` produces a fractal-iterated signal that
  blends 50/50 with the DNA-crossover signal in `_self_generate()`.

## `think()` data flow

```
            input_data (np.ndarray | None)
                      │
           ┌──────────┴──────────┐
           │  None?              │ yes -> _self_generate()
           │  else: pad to dim   │       DNAStorage.generate() + FractalGenerator.generate()
           └──────────┬──────────┘
                      ▼
                 Signal(data)
                      │
    ┌─────────────────────┼─────────────────────┐   ParallelExecutor.map_modules()
    │                     │                     │   (concurrent when n_workers > 1)
    ▼                     ▼                     ▼
ConsciousnessCore   ActiveInferenceEngine   ...4 more modules
.process(signal)    .process(signal)        .process(signal)
    │                     │                     │
    └──────────┬──────────┴─────────────────────┘
               ▼
     _integrate(results)        mean of padded module outputs -> integrated Signal
               │
      ┌────────┴────────┐
      ▼                 ▼
consciousness.reflect()  ParallelExecutor.map(m.predict, modules)
(SelfModel metacognition)  -> list[Prediction]
      │                 │
      │                 ▼
      │        mean(pred.uncertainty) -> mean_err
      │                 │
      │                 ▼
      │        for m in modules: m.update(mean_err)
      │
      ▼
return Signal(data=integrated.data,
             metadata={cycle, self_reflection, module_count})
```

Key points (all in `src/zero_data_model/model.py`):

- `_self_generate()` blends 50/50 a DNA-crossover signal
  (`biological.dna_storage.generate`) with a fractal-iterated signal
  (`math_universe.fractal.generate`). This is the **zero-data** input source.
- `_integrate()` pads each module output to a common length and takes the
  per-coordinate mean — a simple Global-Workspace-style integration.
- `consciousness.reflect()` returns a `Signal` whose `metadata` carries
  `self_confidence` and `history_len` from `SelfModel` — the system's
  metacognitive readout.
- Every module's `update(mean_err)` nudges its internal parameters by a
  prediction-error-scaled noise term, closing the predictive-processing loop.

## Module dependency graph (capabilities → core modules)

| Capability | Core modules used |
| --- | --- |
| `TextEncoder` | `MathematicalUniverse` |
| `SemanticComparator` | `TextEncoder` + `CategoryTheoryEngine` + `MathematicalUniverse` |
| `ZeroShotClassifier` | `TextEncoder` + `ActiveInferenceEngine` + `CategoryTheoryEngine` |
| `TextGenerator` | `TextEncoder` + `BiologicalSubstrate` + `MathematicalUniverse` |
| `ImageEncoder` | `MathematicalUniverse` |
| `FeatureExtractor` | `BiologicalSubstrate` |
| `PatternRecognizer` | `ImageEncoder` + `ActiveInferenceEngine` |
| `ShapeAnalyzer` | `CategoryTheoryEngine` |
| `TimeSeriesForecaster` | `ActiveInferenceEngine` + `QuantumClassicalHybrid` |
| `AnomalyDetector` | `ActiveInferenceEngine` |
| `PatternMiner` | `BiologicalSubstrate` + `MathematicalUniverse` |
| `TrendAnalyzer` | `MathematicalUniverse` + `CategoryTheoryEngine` |

See the [per-module docs](../modules/consciousness.md) for the cognitive-upgrade
phases layered on top of these six cores.
