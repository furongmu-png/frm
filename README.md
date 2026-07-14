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

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — three-layer diagrams,
  module dependency graph, the `think()` data flow, hardware stack, file tree,
  design principles.
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
  capabilities/          # rules.py, nlp.py, vision.py, analytics.py
  hardware/              # accel.py, quantum.py, parallel.py
tests/                   # one test file per module + integration
demo.py  benchmark.py    # runnable demo + benchmarks
docs/                    # architecture, tutorial, theory
```

## Testing

```bash
pip install pytest
PYTHONPATH=src pytest -q                   # full suite (12 test files)
PYTHONPATH=src pytest tests/test_model.py -v
```

## License

MIT (placeholder) — see `LICENSE` when added.
