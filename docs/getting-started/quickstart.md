# Quick Start

This guide walks through constructing a `ZeroDataModel`, running a self-generated
thought cycle, exercising the NLP / Vision / Analytics capabilities, and turning
on cognitive-upgrade phases. Every snippet needs **no datasets and no pre-trained
weights** — the only inputs are arrays and strings you type yourself.

## 1. Construct the model

```python
import numpy as np
from zero_data_model.model import ZeroDataModel

model = ZeroDataModel(dim=64)        # dim is the internal signal width
```

`dim` drives every cognitive module to allocate `dim × dim` arrays. Valid range
is `[1, 4096]`; values outside that range raise `ValueError`.

Pass a `seed` for reproducible cycles (this also pins BLAS to 1 thread and forces
the deterministic pure-NumPy quantum simulator):

```python
model = ZeroDataModel(dim=64, seed=42)
```

## 2. Run a thought cycle

`think()` is the cognitive heartbeat. With **no argument** the model
self-generates an input from `BiologicalSubstrate.dna_storage` and
`MathematicalUniverse.fractal`:

```python
result = model.think()
print(result.data.shape)                                  # (64,)
print(result.metadata["cycle"])                           # 1
print(result.metadata["self_reflection"]["self_confidence"])
```

Feed an external signal in:

```python
signal = np.sin(np.linspace(0, 2 * np.pi, 64))
result = model.think(signal)
print(result.metadata["cycle"])                           # 2
```

Each cycle:

1. Fans the signal out to the six core modules in parallel
   (`ParallelExecutor.map_modules`).
2. Integrates the per-module outputs via a per-coordinate mean
   (Global-Workspace-style integration).
3. Runs `consciousness.reflect()` for a metacognitive readout.
4. Predicts the next state, computes the mean prediction error, and nudges
   every module's parameters (`update(mean_err)`).

## 3. Use the domain capabilities

### NLP

```python
print(model.classify_text("the algorithm computes the network"))   # ('tech', 0.xx)
print(model.text_similarity("a neuron fires", "a cell spikes"))    # 0.xx
print(model.generate_text("the", length=32))                       # generated string
print(model.encode_text("hello world").shape)                      # (64,)
```

### Vision

```python
image = np.random.rand(16, 16)
print(model.recognize_pattern(image))                              # ('shape', conf)
print(model.analyze_shape(image))                                  # {edges, corners, ...}
print(model.extract_image_features(image).keys())
```

### Analytics

```python
series = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
print(model.forecast(series, horizon=5))                           # next 5 values
print(model.detect_anomalies(series))                              # anomaly mask
print(model.analyze_trend(series))                                 # {slope, r2, ...}
```

## 4. Enable cognitive-upgrade phases

All seven phases default to **off** so the baseline `think()` cycle is
byte-for-byte zero-regression. Enable them individually:

```python
m = ZeroDataModel(
    dim=64,
    enable_architect=True,            # Phase 1 — architecture plasticity
    enable_layered_predictor=True,    # Phase 2 — layered time + temporal memory
    enable_episodic_memory=True,      # Phase 3 — structured memory
    enable_logic_layer=True,          # Phase 4 — neuro-symbolic fusion
    enable_meta_cognition=True,       # Phase 5 — meta-cognition
    enable_experiment_planner=True,   # Phase 6 — active experiment
    # enable_multiagent=True,         # Phase 7 — multi-agent (library modules)
)
```

Each enabled phase appends keys to `Signal.metadata["cognitive_upgrades"]`
without altering the six core modules' computation:

```python
sig = m.think()
print(sig.metadata.get("cognitive_upgrades", {}).keys())
# dict_keys(['architecture', 'layered_predictor', 'temporal_memory_error',
#            'episodic_nodes', 'semantic_index_size', 'logic_violations',
#            'meta_cognition', 'experiment', 'supported_hypotheses'])
```

See the [Cognitive Upgrades](../architecture/cognitive-upgrades.md) page for the
theory basis and `think()` integration of each phase.

## 5. Inspect the hardware stack

```python
print(model.hardware_info)
# {
#   'array_backend': 'numpy' | 'cupy',
#   'gpu': bool,
#   'quantum_backend': 'qiskit' | 'simulator',
#   'annealer_jit': bool,
#   'n_workers': int,
#   'backend': 'threading' | 'sequential',
#   'joblib': bool,
# }
```

## 6. Persist and restore state

`ModelSerializer` saves/loads model state as `npz + json` (no pickle):

```python
from zero_data_model.persistence import ModelSerializer

ModelSerializer.save(model, "checkpoint")        # writes checkpoint.npz + checkpoint.json
restored = ModelSerializer.load("checkpoint")
```

## Next steps

- [Configuration](configuration.md) — environment variables and feature flags.
- [Architecture Overview](../architecture/overview.md) — four-layer stack and
  module dependency graph.
- [Python SDK](../api/python-sdk.md) — full Python API reference.
- [REST API](../api/rest.md) — HTTP endpoints with mkdocstrings auto-generated
  reference.
