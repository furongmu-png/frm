# Zero-Data Model — Practical Tutorial

A hands-on guide to the Zero-Data Model: install it, run a thought cycle, and
exercise every domain capability (NLP, Vision, Analytics) plus the
hardware-acceleration stack. Every code snippet below mirrors the patterns used
in `demo.py` and `benchmark.py` and references the real source files in
`src/zero_data_model/`.

> The system needs **no datasets and no pre-trained weights**. The only inputs
> to the examples below are arrays and strings you type yourself.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [NLP capabilities](#3-nlp-capabilities)
4. [Vision capabilities](#4-vision-capabilities)
5. [Analytics capabilities](#5-analytics-capabilities)
6. [Cross-domain transfer](#6-cross-domain-transfer)
7. [Hardware acceleration](#7-hardware-acceleration)
8. [Persistence](#8-persistence)
9. [REST API and Web UI](#9-rest-api-and-web-ui)
10. [Running benchmarks and demos](#10-running-benchmarks-and-demos)
11. [Running the test suite](#11-running-the-test-suite)

---

## 1. Installation

### Required dependencies

Only two packages are strictly required:

```bash
pip install numpy scipy
```

### Optional dependencies (each one transparently enables an acceleration path)

| Package | What it unlocks | Source file that uses it |
|---|---|---|
| `qiskit` | Real variational quantum circuits via `StatevectorSampler` | `src/zero_data_model/hardware/quantum.py` |
| `numba` | JIT-compiled inner loop for the quantum annealer | `src/zero_data_model/quantum_hybrid.py` |
| `joblib` | Process/thread-pool parallel module execution | `src/zero_data_model/hardware/parallel.py` |
| `cupy` | GPU array backend (`xp = cupy`) when CUDA is available | `src/zero_data_model/hardware/accel.py` |
| `fastapi`, `uvicorn` | REST API service (`api.py`) — see §9 | `src/zero_data_model/api.py` |

```bash
pip install qiskit numba joblib cupy        # pick whichever you want
```

If any of these is missing the system simply falls back to a pure-NumPy path,
so you can always start with just `numpy scipy` and add accelerators later.

### Making the package importable

There is no `pyproject.toml`/`setup.py` yet, so put `src` on your path. From the
project root:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

or, equivalently, prefix every command:

```bash
PYTHONPATH=src python demo.py
```

Both `demo.py` and `benchmark.py` do this automatically with
`sys.path.insert(0, "src")`, so you can also just run them directly.

---

## 2. Quick start

Create a model and run one self-generated thought cycle. With `think()` called
**without arguments** the model generates its own input from internal structure
(DNA crossover + fractal iteration) — this is the zero-data contract.

```python
import numpy as np
from zero_data_model.model import ZeroDataModel

model = ZeroDataModel(dim=64)        # dim is the internal signal width

# A self-generated thought (no input data at all):
result = model.think()
print(result.data.shape)                              # (64,)
print(result.metadata["cycle"])                       # 1
print(result.metadata["self_reflection"]["self_confidence"])

# Feed an external signal in:
signal = np.sin(np.linspace(0, 2 * np.pi, 64))
result = model.think(signal)
print(result.metadata["cycle"])                       # 2
```

You can also classify text in one line:

```python
topic, confidence = model.classify_text("the algorithm computes the network")
print(topic, confidence)                              # e.g. 'tech' 0.83
```

### What just happened?

- `ZeroDataModel.__init__` (in `src/zero_data_model/model.py`) instantiated the
  six core modules and the twelve domain capabilities, wiring them together.
- `think()` fanned the `Signal` out to all six core modules in parallel
  (`ParallelExecutor.map_modules`), integrated their outputs, ran
  `consciousness.reflect()` for a metacognitive readout, and updated each
  module's parameters from the mean prediction error.

For the full walkthrough, run the bundled demo:

```bash
PYTHONPATH=src python demo.py
```

---

## 3. NLP capabilities

All NLP capabilities live in `src/zero_data_model/capabilities/nlp.py` and
compose `TextEncoder` with `NLPRules` (stop-words, sentiment lexicon, topic
keywords from `capabilities/rules.py`) plus core modules. No external NLP
library is used.

### 3.1 Encode text

`TextEncoder.encode` produces an L2-normalized `dim`-length vector from three
feature groups: char n-gram hashing (n=2,3), Unicode codepoint statistics, and
rule-based token/topic features.

```python
from zero_data_model.capabilities.nlp import TextEncoder

enc = TextEncoder(dim=64)
vec = enc.encode("code data model algorithm")
print(vec.shape, round(float(np.linalg.norm(vec)), 4))   # (64,) 1.0
```

### 3.2 Semantic similarity

`SemanticComparator.similarity` blends cosine similarity (routed through
`CategoryTheoryEngine.find_isomorphism`) with a KL-divergence-based similarity
(`exp(-KL)` from `InformationGeometry.kl_divergence`). Returns a float in
`[0, 1]`; identical texts score ~1.0.

```python
from zero_data_model.capabilities.nlp import SemanticComparator
from zero_data_model.math_universe import MathematicalUniverse
from zero_data_model.category_engine import CategoryTheoryEngine

comparator = SemanticComparator(enc, MathematicalUniverse(dim=64), CategoryTheoryEngine(dim=64))
print(round(comparator.similarity("code data model", "code data model"), 4))   # ~1.0
print(round(comparator.similarity("code data model", "tree river mountain"), 4))  # lower
```

Or, via the top-level model:

```python
print(model.text_similarity("code data model", "code data model"))
```

### 3.3 Zero-shot classification

`ZeroShotClassifier` builds one prototype vector per topic
(`tech`, `nature`, `emotion`, `science`) by averaging the encoded keywords from
`NLPRules.topic_keywords` — these prototypes are **self-generated**, not
learned. `classify` scores the input encoding against each prototype (cosine +
free-energy penalty from `ActiveInferenceEngine.compute_free_energy`) and
returns `(best_topic, confidence)`.

```python
samples = [
    "the algorithm computes the network",   # -> tech
    "trees rivers mountains forests",       # -> nature
    "love joy happiness hope",              # -> emotion
    "energy force mass quantum field",      # -> science
]
for s in samples:
    topic, conf = model.classify_text(s)
    print(f"{s!r:45s} -> {topic:8s} {conf:.3f}")
```

### 3.4 Text generation (no training data)

`TextGenerator.generate` stores the seed encoding in `DNAStorage`, recombines
stored sequences via crossover (`DNAStorage.generate`), refines the result with
`FractalGenerator.generate`, then maps each scalar to a printable ASCII char
(`_to_printable`). Output is always exactly `length` printable chars.

```python
print(repr(model.generate_text("seed", length=48)))
```

---

## 4. Vision capabilities

All vision capabilities live in `src/zero_data_model/capabilities/vision.py`
and compose `MathematicalUniverse` / `BiologicalSubstrate` /
`ActiveInferenceEngine` / `CategoryTheoryEngine` with `VisionRules` (Sobel
kernels, Gaussian kernel, shape names from `capabilities/rules.py`). Inputs are
2D grayscale arrays; 1D arrays are reshaped to `(1, -1)`.

### 4.1 Encode an image

`ImageEncoder.encode` downsamples the image, then concatenates fractal
compression stats (`FractalGenerator.compress`) with topological features
(`TopologicalAnalyzer.topological_features`). Output is L2-normalized.

```python
import numpy as np
yy, xx = np.indices((16, 16))
circle = np.zeros((16, 16))
circle[(yy - 8) ** 2 + (xx - 8) ** 2 <= 25] = 1.0

enc = model.encode_image(circle)
print(enc.shape, round(float(np.linalg.norm(enc)), 4))   # (64,) 1.0
```

### 4.2 Extract rule-based features

`FeatureExtractor.extract` returns a dict with four arrays:

- `edges` — Sobel magnitude (`VisionRules.sobel_x`, `sobel_y` convolved via `VisionRules.convolve`).
- `texture` — cellular-automaton evolution seeded by a binarized image row
  (`BiologicalSubstrate.automata.evolve`).
- `morphology` — self-organized morphogenetic pattern
  (`BiologicalSubstrate.morphogenetic.develop`).
- `stats` — `[mean, std, min, max]` of pixel intensities.

```python
feats = model.extract_image_features(circle)
for k, v in feats.items():
    print(f"{k:10s} shape={v.shape}")
```

### 4.3 Pattern recognition (zero-shot)

`PatternRecognizer.recognize` synthesizes a deterministic prototype image for
each shape in `VisionRules.shapes` (`circle`, `square`, `triangle`, `line`,
`blob`) via `_synthesize_prototype`, encodes both input and prototype, and
picks the best cosine match (adjusted by a free-energy penalty). Returns
`(shape_name, confidence)`.

```python
shape, conf = model.recognize_pattern(circle)
print(shape, round(conf, 4))                              # e.g. 'circle' 0.7

square = np.zeros((16, 16)); square[4:12, 4:12] = 1.0
print(model.recognize_pattern(square))                    # e.g. ('square', 0.6)
```

### 4.4 Shape analysis

`ShapeAnalyzer.analyze` returns a dict of geometric / topological properties:

- `aspect_ratio` — bounding-box height / width of non-zero pixels.
- `symmetry` — correlation between the image and its left-right mirror.
- `complexity` — std of the Sobel gradient magnitude.
- `beti0` — connected-components count (rule-based) or Betti-0 from
  `CategoryTheoryEngine.topology` when available.

```python
analysis = model.analyze_shape(circle)
print({k: round(float(v), 4) for k, v in analysis.items()})
```

---

## 5. Analytics capabilities

All analytics capabilities live in `src/zero_data_model/capabilities/analytics.py`
and compose `ActiveInferenceEngine` / `QuantumClassicalHybrid` /
`BiologicalSubstrate` / `MathematicalUniverse` / `CategoryTheoryEngine` with
`AnalyticsRules` (moving-average window, z-score threshold, seasonal lags).

### 5.1 Time-series forecasting

`TimeSeriesForecaster.forecast` embeds the series into the active-inference
belief state, rolls `GenerativeModel.predict_next_state` forward `horizon`
steps, decodes each to a scalar, and blends 50/50 with a linearly extrapolated
moving-average trend.

```python
rng = np.random.default_rng(42)
series = np.arange(20, dtype=float) + rng.standard_normal(20) * 0.5
forecast = model.forecast(series, horizon=5)
print(forecast)                                  # shape (5,)
```

### 5.2 Anomaly detection

`AnomalyDetector.detect` computes a per-point free energy (resetting the belief
state before each point so each surprisal is independent), z-scores the
free-energy series, and flags points whose `|z|` exceeds
`AnalyticsRules.anomaly_z_threshold` (default 2.0). Returns a boolean mask.

```python
anomaly_series = np.array([1, 1, 1, 1, 1, 100, 1, 1, 1, 1], dtype=float)
mask = model.detect_anomalies(anomaly_series)
print(mask)                                      # False except at index 5
print("anomaly at index", np.where(mask)[0].tolist())
```

### 5.3 Pattern mining

`PatternMiner.mine` returns a dict with four structural descriptors:

- `self_similarity` — first-half / second-half correlation from `FractalGenerator.compress`.
- `topology` — `TopologicalAnalyzer.topological_features` (Betti numbers + moments).
- `automaton_rule` — the Wolfram rule (0–255) whose evolution best reproduces
  the binarized series (`_best_automaton_rule` sweeps all 256 rules).
- `periodicity` — autocorrelation peak lag in `1..len//2`.

```python
mined = model.mine_patterns(series)
print({k: (round(float(v), 4) if np.isscalar(v) else v) for k, v in mined.items() if k != 'topology'})
```

### 5.4 Trend analysis

`TrendAnalyzer.analyze` returns:

- `trend_slope` — degree-1 `np.polyfit` slope.
- `regime` — `"up"` / `"down"` / `"flat"` from `AnalyticsRules.trend_threshold`.
- `curvature` — mean of the second difference.
- `geodesic_deviation` — `InformationGeometry.kl_divergence` between the two
  abs-valued halves of the series.
- `isomorphism_score` — cosine similarity between the two halves
  (`CategoryTheoryEngine.find_isomorphism`).

```python
up_series = np.arange(50, dtype=float)
print(model.analyze_trend(up_series)["regime"])  # 'up'
```

---

## 6. Cross-domain transfer

The category-theory engine (`src/zero_data_model/category_engine.py`) lets the
model detect structural similarity between two problems and transfer a solution
vector across domains via a `Functor`.

### 6.1 Analogy detection

`ZeroDataModel.find_analogies(a, b)` delegates to
`CategoryTheoryEngine.find_isomorphism`, which pads both arrays to `dim` and
returns their cosine similarity in `[-1, 1]`.

```python
a = np.random.randn(64)
b = a + np.random.randn(64) * 0.1     # near-duplicate
c = np.random.randn(64)               # unrelated
print(model.find_analogies(a, b))     # ~1.0
print(model.find_analogies(a, c))     # lower
```

### 6.2 Functorial transfer

`CategoryTheoryEngine.transfer_solution(source_cat, target_cat, solution)`
applies the functor mapping `source_cat -> target_cat` (a default
`NLP -> CV` functor is created in `_init_default_categories`). When no matching
functor exists the solution is returned unchanged.

```python
ce = model.category_engine
sol = np.random.randn(64)
transferred = ce.transfer_solution("NLP", "CV", sol)
print(transferred.shape)              # (64,)
```

### 6.3 Optimization via quantum annealing

`ZeroDataModel.solve(problem)` calls `QuantumClassicalHybrid.solve_optimization`,
which runs `QuantumAnnealer.optimize` (numba-JIT simulated annealing over a
symmetric cost matrix). Returns a `Signal` whose `metadata["energy"]` is the
final Ising-style energy.

```python
sol = model.solve(np.random.randn(64))
print("energy:", round(sol.metadata["energy"], 4))
```

---

## 7. Hardware acceleration

`ZeroDataModel.hardware_info` returns a single diagnostic dict combining all
three acceleration backends. Check it before assuming any acceleration is
active:

```python
hw = model.hardware_info
print(hw)
# {'array_backend': 'numpy', 'gpu': False, 'quantum_backend': 'qiskit',
#  'annealer_jit': True, 'n_workers': 8, 'backend': 'threading', 'joblib': True}
```

### 7.1 GPU arrays (`hardware/accel.py`)

The array module is exposed as `xp` (CuPy on CUDA, else NumPy). Use
`to_gpu` / `to_cpu` / `asnumpy` to move arrays between devices. Code written
against `xp` runs unchanged on both backends.

```python
from zero_data_model.hardware import xp, has_gpu, backend_name, to_gpu, asnumpy

print(backend_name(), has_gpu)        # 'cupy' True  -- or 'numpy' False
a = xp.arange(10, dtype=float)
host = asnumpy(a)                     # always a host np.ndarray
```

### 7.2 Quantum backend (`hardware/quantum.py`)

`get_quantum_backend(prefer=None)` auto-selects `QiskitQuantumBackend` (real
`StatevectorSampler`, RY+CNOT ansatz) when Qiskit is installed, otherwise
`SimulatorQuantumBackend`. Force one with `prefer="qiskit"` or
`prefer="simulator"`.

```python
from zero_data_model.hardware.quantum import get_quantum_backend
import numpy as np

backend = get_quantum_backend(n_qubits=6, n_layers=3)
print(backend.name)                   # 'qiskit' or 'simulator'
params = np.random.randn(3, 6, 2) * 0.1
entangling = np.random.randn(6, 6) * 0.05
probs = backend.evolve_and_measure(params, entangling, n_shots=1024)
print(probs.shape)                    # (12,)  == 2 * n_qubits
```

You can also pin the backend for the whole model. Note that
`ZeroDataModel.__init__` only accepts `dim` — the quantum backend is selected
inside `QuantumClassicalHybrid`, so to force a backend you rebuild that module
after construction:

```python
from zero_data_model.quantum_hybrid import QuantumClassicalHybrid
# Force the pure-NumPy simulator even if Qiskit is installed:
model.quantum_hybrid = QuantumClassicalHybrid(dim=64, quantum_backend="simulator")
print(model.quantum_hybrid.quantum_backend_name)   # 'simulator'
```

### 7.3 Parallel execution (`hardware/parallel.py`)

`ParallelExecutor` runs the six module `process()`/`predict()` calls
concurrently via `ThreadPoolExecutor` (joblib threading) when `n_workers > 1`.
Force sequential execution to compare:

```python
from zero_data_model.hardware.parallel import ParallelExecutor

seq_exec = ParallelExecutor(n_workers=1)        # always sequential
print(seq_exec.info)                            # {'n_workers': 1, 'backend': 'sequential', 'joblib': ...}
```

The benchmark script (see §10) measures the parallel-vs-sequential speedup
directly.

---

## 8. Persistence

`ModelSerializer` (`src/zero_data_model/persistence.py`) saves and loads the
full `ZeroDataModel` state to a directory on disk using numpy `.npz` for arrays
and JSON for configuration metadata. **Pickle is intentionally not used**, so
the on-disk format is portable, auditable, and safe to load from untrusted
sources.

### 8.1 Save / load from Python

`ModelSerializer.save(model, path)` writes `arrays.npz` + `config.json` under
`path`. It serializes every module's numeric state: consciousness-core layer
weights/biases, active-inference transition/emission/belief_state, category
topos classifier + functor transforms, quantum-hybrid classical weights +
circuit params/entangling + annealer cost matrix, biological morphogenetic
grid/morphogens + automata state, and math-universe fractal transforms.
`ModelSerializer.load(path)` reconstructs a fresh `ZeroDataModel(dim=...)` and
overwrites every saved array in place.

```python
from zero_data_model.persistence import ModelSerializer
from zero_data_model.model import ZeroDataModel

model = ZeroDataModel(dim=32)
model.think(); model.think()           # evolve state a little
print("cycle before save:", model.cycle_count)

ModelSerializer.save(model, "saved_model")          # writes saved_model/{arrays.npz,config.json}

restored = ModelSerializer.load("saved_model")
print("cycle after load: ", restored.cycle_count)   # matches
print(restored.hardware_info["quantum_backend"])    # re-detected on this machine
```

Notes:

- The `config.json` records `dim`, `cycle_count`, `quantum_backend_name`, and
  the per-module counts (`n_consciousness_layers`, `n_functors`,
  `functor_morphism_counts`, `n_morphogens`, `n_fractal_transforms`) needed to
  walk the npz on load.
- The quantum backend is **re-detected on the loading machine** (the saved
  `quantum_backend_name` is informational only), so a model saved where Qiskit
  is installed will load and run on a machine with only the simulator.

### 8.2 Save / load via the REST API

The same serializer is wired into the `/save` and `/load` HTTP endpoints (see
§9), so you can persist and restore the live server-side model without writing
Python.

---

## 9. REST API and Web UI

The FastAPI service in `src/zero_data_model/api.py` exposes the `ZeroDataModel`
as HTTP endpoints. A single module-level `ZeroDataModel(dim=32)` is lazily
initialized on first use; numpy arrays returned by the model are converted to
plain Python lists so they serialize to JSON cleanly.

### 9.1 Install the extra deps

```bash
pip install fastapi uvicorn
```

### 9.2 Start the server

The module exposes a ready-made `app`, so you can run it directly:

```bash
PYTHONPATH=src uvicorn zero_data_model.api:app --port 8000
# or, equivalently:
PYTHONPATH=src python -m zero_data_model.api        # runs uvicorn on 0.0.0.0:8000
```

Interactive docs are auto-generated at `http://localhost:8000/docs` (Swagger UI)
and `http://localhost:8000/redoc`.

### 9.3 Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | — | `{status, hardware_info}` |
| POST | `/think` | `{input: [floats]}` (optional) | `{cycle, output, confidence}` |
| POST | `/classify` | `{text: str}` | `{topic, confidence}` |
| POST | `/similarity` | `{a: str, b: str}` | `{similarity}` |
| POST | `/generate` | `{seed: str, length: int=32}` | `{text}` |
| POST | `/forecast` | `{series: [floats], horizon: int=5}` | `{forecast: [floats]}` |
| POST | `/anomalies` | `{series: [floats]}` | `{anomalies: [bool]}` |
| POST | `/trend` | `{series: [floats]}` | `{trend_slope, regime, curvature, geodesic_deviation, isomorphism_score}` |
| POST | `/recognize` | `{image: [[floats]]}` | `{shape, confidence}` |
| POST | `/save` | `{path: str}` | `{saved: true}` |
| POST | `/load` | `{path: str}` | `{loaded: true}` |

### 9.4 curl examples

```bash
# Health + active hardware backends.
curl http://localhost:8000/

# Self-generated thought (no input -> model self-generates).
curl -X POST http://localhost:8000/think

# Feed an external signal into think().
curl -X POST http://localhost:8000/think \
  -H 'Content-Type: application/json' \
  -d '{"input":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}'

# Zero-shot text classification.
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"the algorithm computes the network"}'

# Semantic similarity between two texts.
curl -X POST http://localhost:8000/similarity \
  -H 'Content-Type: application/json' \
  -d '{"a":"code data model","b":"code data model"}'

# Self-generated text.
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"seed":"seed","length":48}'

# Time-series forecast.
curl -X POST http://localhost:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{"series":[1,2,3,4,5,6,7,8,9,10],"horizon":3}'

# Anomaly detection (index 5 is the spike).
curl -X POST http://localhost:8000/anomalies \
  -H 'Content-Type: application/json' \
  -d '{"series":[1,1,1,1,1,100,1,1,1,1]}'

# Trend analysis.
curl -X POST http://localhost:8000/trend \
  -H 'Content-Type: application/json' \
  -d '{"series":[1,2,3,4,5,6,7,8,9,10]}'

# Pattern recognition on a 2D grayscale image (list of rows).
curl -X POST http://localhost:8000/recognize \
  -H 'Content-Type: application/json' \
  -d '{"image":[[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]]}'

# Persist and restore the live model.
curl -X POST http://localhost:8000/save -H 'Content-Type: application/json' -d '{"path":"saved_model"}'
curl -X POST http://localhost:8000/load -H 'Content-Type: application/json' -d '{"path":"saved_model"}'
```

### 9.5 Web UI

> **Status: not yet implemented.** The design spec lists a `web/index.html`
> browser UI as planned engineering-layer work, but no such file exists in the
> repository today. Until it ships, drive the model from the browser via the
> Swagger UI at `http://localhost:8000/docs`, or call the endpoints above from
> any HTTP client.

---

## 10. Running benchmarks and demos

Two runnable scripts ship with the project.

### 10.1 `demo.py` — end-to-end zero-data walkthrough

Runs the full capability tour: hardware info, self-generated thoughts, external
signal, cross-domain analogies, quantum-annealing optimization, NLP
(similarity / classification / generation), CV (encoding / features /
recognition / shape analysis) and Analytics (forecast / anomalies / patterns /
trend). No arguments needed.

```bash
PYTHONPATH=src python demo.py
# or simply:
python demo.py            # demo.py adds 'src' to sys.path itself
```

### 10.2 `benchmark.py` — hardware-acceleration benchmarks

Measures four things:

1. Quantum backend: Qiskit vs `SimulatorQuantumBackend` (`bench_quantum_backend`).
2. Quantum annealer: numba-JIT vs pure-NumPy inner loop (`bench_annealer_jit`).
3. Parallel vs sequential `think()` cycles (`bench_parallel_vs_sequential`).
4. Full-model NLP / CV / Analytics throughput (`bench_full_model`).

```bash
PYTHONPATH=src python benchmark.py
```

Typical output (numbers depend on your machine and which optional deps are
installed):

```
[1] Quantum backend: Qiskit vs Simulator
  Active backend: qiskit
  Evolve+measure (n_qubits=6, shots=1024): 12.34 ms
  Simulator fallback:                       0.45 ms
[2] Quantum annealer: numba-JIT vs pure numpy
  n_vars=64, iterations=500, JIT=True
  Best energy: -12.3456
  Time: 8.90 ms
[3] Parallel vs sequential module execution
  Parallel think() (cycle):   3.21 ms
  Sequential think() (cycle): 7.84 ms
  Speedup: 2.44x
[4] Full model: NLP/CV/Analytics throughput
  NLP:      20 classifications in ... ms
  CV:       10 pattern recognitions in ... ms
  Analytics: 10 forecasts in ... ms
```

---

## 11. Running the test suite

The tests in `tests/` import the package directly
(`from zero_data_model.model import ZeroDataModel`), so `src` must be on the
path. There is one test file per module plus integration, persistence, API and
end-to-end scenario tests (15 files in total):

```bash
pip install pytest
PYTHONPATH=src pytest -q
```

Or run a single module's tests:

```bash
PYTHONPATH=src pytest tests/test_quantum_hybrid.py -v
PYTHONPATH=src pytest tests/test_capabilities_analytics.py -v
```

The test files cover:

| Test file | Covers |
|---|---|
| `tests/test_base.py` | `Signal`, `Prediction`, `CognitiveModule`, `KnowledgeStore` |
| `tests/test_consciousness_core.py` | `ConsciousnessCore`, `GlobalWorkspace`, `SelfModel`, `PredictiveLayer` |
| `tests/test_active_inference.py` | `ActiveInferenceEngine`, `GenerativeModel`, `MarkovBlanket`, `HomeostaticController` |
| `tests/test_category_engine.py` | `CategoryTheoryEngine`, `Category`, `Functor`, `ToposEngine` |
| `tests/test_quantum_hybrid.py` | `QuantumClassicalHybrid`, `VariationalQuantumCircuit`, `QuantumAnnealer` |
| `tests/test_biological.py` | `BiologicalSubstrate`, `DNAStorage`, `MorphogeneticField`, `CellularAutomata` |
| `tests/test_math_universe.py` | `MathematicalUniverse`, `InformationGeometry`, `TopologicalAnalyzer`, `FractalGenerator` |
| `tests/test_hardware.py` | `accel`, `quantum` backends, `ParallelExecutor` |
| `tests/test_model.py` | `ZeroDataModel` integration + every domain capability |
| `tests/test_capabilities_nlp.py` | `TextEncoder`, `SemanticComparator`, `ZeroShotClassifier`, `TextGenerator` |
| `tests/test_capabilities_vision.py` | `ImageEncoder`, `FeatureExtractor`, `PatternRecognizer`, `ShapeAnalyzer` |
| `tests/test_capabilities_analytics.py` | `TimeSeriesForecaster`, `AnomalyDetector`, `PatternMiner`, `TrendAnalyzer` |
| `tests/test_persistence.py` | `ModelSerializer.save/load` roundtrip, file layout, error handling |
| `tests/test_api.py` | FastAPI endpoints (health, classify, similarity, forecast, think, generate, anomalies, trend, recognize, save+load) |
| `tests/test_scenarios.py` | End-to-end scenarios: zero-shot classification, cross-domain transfer, anomaly/trend/forecast, self-generation |

---

## Where to go next

- **`docs/architecture.md`** — three-layer diagrams, the `think()` data flow,
  the hardware stack, and design principles.
- **`docs/theory.md`** — the scientific theory behind each core module, with
  the mapping from theory to class/method.
