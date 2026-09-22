# ZeroDataModel

> A self-sufficient cognitive system that requires **no external training data**.

ZeroDataModel composes six theory-driven core modules (consciousness, active
inference, category theory, quantum-classical hybrid, biological substrate,
mathematical universe) into twelve domain capabilities (NLP, vision, analytics).
It self-generates its inputs via DNA crossover, fractal iteration and
morphogenesis, and augments them with small hand-written rule priors instead of
learned weights.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
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
  the baseline is zero-regression.
- **Production-ready API** — FastAPI service with API-key auth, rate limiting,
  security headers, Prometheus metrics, and 40+ MCP tools for AI agents.

## A 30-second taste

```python
from zero_data_model.model import ZeroDataModel

m = ZeroDataModel(dim=64)
print(m.think().metadata)                                # self-generated thought, no input
print(m.classify_text("the algorithm computes the network"))   # ('tech', conf)
```

Enable cognitive-upgrade phases individually with feature flags (all default
to `False`; the baseline `think()` cycle is unchanged when none are set):

```python
m = ZeroDataModel(dim=64, enable_architect=True, enable_meta_cognition=True)
```

## Where to go next

- **New to ZeroDataModel?** Start with [Installation](getting-started/installation.md)
  and the [Quick Start](getting-started/quickstart.md).
- **Want the big picture?** Read the [Architecture Overview](architecture/overview.md).
- **Building an integration?** See the [REST API](api/rest.md) and
  [Python SDK](api/python-sdk.md) references.
- **Deploying to production?** Jump to [Docker](deployment/docker.md) or
  [Kubernetes](deployment/kubernetes.md).

## License

MIT — see `LICENSE` in the repository root.
