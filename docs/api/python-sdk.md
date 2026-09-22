# Python SDK

The Python SDK is the `zero_data_model` package itself. Install it with
`pip install -e .` (see [Installation](../getting-started/installation.md)) and
import the integration entry point:

```python
from zero_data_model.model import ZeroDataModel
```

## Auto-generated reference

The block below is rendered by mkdocstrings directly from the source docstring
of `ZeroDataModel` in `src/zero_data_model/model.py`. It covers the constructor
and every public method.

::: zero_data_model.model.ZeroDataModel

## Common patterns

### Construct with cognitive upgrades

```python
from zero_data_model.model import ZeroDataModel

m = ZeroDataModel(
    dim=64,
    seed=42,
    enable_architect=True,
    enable_layered_predictor=True,
    enable_episodic_memory=True,
    enable_logic_layer=True,
    enable_meta_cognition=True,
    enable_experiment_planner=True,
)
```

### Run a self-generated thought cycle

```python
sig = m.think()                       # no argument -> self-generated input
print(sig.data.shape)                 # (64,)
print(sig.metadata["cycle"])
print(sig.metadata.get("cognitive_upgrades", {}))
```

### Cross-domain transfer

The same set of core modules backs every capability, so a signal encoded in one
domain can be inspected through another:

```python
import numpy as np

text_vec = m.encode_text("a neuron fires")         # NLP embedding
img = np.random.rand(16, 16)
img_vec = m.encode_image(img)                       # Vision embedding
print(m.text_similarity("a neuron fires", "a cell spikes"))
print(m.recognize_pattern(img))
```

### Inspect hardware acceleration

```python
print(m.hardware_info)
# {'array_backend': 'numpy', 'gpu': False, 'quantum_backend': 'simulator',
#  'annealer_jit': False, 'n_workers': 1, 'backend': 'sequential', 'joblib': False}
```

### Persist and restore

```python
from zero_data_model.persistence import ModelSerializer

ModelSerializer.save(m, "checkpoint")          # checkpoint.npz + checkpoint.json
restored = ModelSerializer.load("checkpoint")  # no pickle involved
```

## Auto-generated persistence reference

::: zero_data_model.persistence.ModelSerializer

## Base types

The `Signal`, `Prediction`, `CognitiveModule` and `KnowledgeStore` types live in
`zero_data_model.base`. They are the uniform contract every core module
implements.

::: zero_data_model.base.Signal

::: zero_data_model.base.CognitiveModule
