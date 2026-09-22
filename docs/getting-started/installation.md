# Installation

ZeroDataModel runs anywhere NumPy runs. Only `numpy` and `scipy` are strictly
required; every other dependency is optional and transparently enables an
acceleration path.

## Requirements

- Python **3.10+** (tested on 3.11, 3.12, 3.14)
- `numpy>=1.25` (uses `Generator.spawn` for per-module independent RNGs)
- `scipy>=1.10`

## 1. Install the Python package

The package lives under `src/` and ships a standard `pyproject.toml`. From the
repository root:

```bash
pip install -e .                 # core only (numpy + scipy pulled in automatically)
```

### Optional extras

Each extra unlocks an acceleration or integration path. Install whichever you
need — missing ones silently fall back to pure NumPy.

| Extra | What it unlocks | Install |
| --- | --- | --- |
| `quantum` | Real Qiskit variational circuits (`StatevectorSampler`) | `pip install -e ".[quantum]"` |
| `jit` | numba-JIT inner loop for the quantum annealer | `pip install -e ".[jit]"` |
| `parallel` | joblib thread/process-pool parallel module execution | `pip install -e ".[parallel]"` |
| `gpu` | CuPy GPU array backend when CUDA is available | `pip install -e ".[gpu]"` |
| `web` | FastAPI REST service + structured logging + Prometheus metrics | `pip install -e ".[web]"` |
| `mcp` | MCP server exposing capabilities as AI agent tools | `pip install -e ".[mcp]"` |
| `all` | All of the above | `pip install -e ".[all]"` |

```bash
# Example: API service + parallel execution
pip install -e ".[web,parallel]"
```

## 2. (Optional) Build the web frontend

The browser UI is a React + Vite + TypeScript app under `frontend/`. It is not
required to use the Python package or the REST API.

```bash
cd frontend
npm install
npm run build        # outputs to frontend/dist/
npm run dev          # or run the dev server with HMR
```

Node.js **18+** is required.

## 3. Verify the install

```bash
PYTHONPATH=src python demo.py            # end-to-end zero-data demo
PYTHONPATH=src python benchmark.py       # hardware-acceleration benchmarks
```

Or, in Python:

```python
from zero_data_model.model import ZeroDataModel
m = ZeroDataModel(dim=64)
print(m.think().metadata["cycle"])       # 1
print(m.hardware_info)                   # shows active acceleration backends
```

## 4. Run the REST API (optional)

```bash
pip install -e ".[web]"
PYTHONPATH=src uvicorn zero_data_model.api:app --port 8000
# Swagger UI at http://localhost:8000/docs (when ZDM_ENV=development)
```

See [Configuration](configuration.md) for environment variables, and
[REST API](../api/rest.md) for the endpoint reference.

## Troubleshooting

- **`ModuleNotFoundError: numpy`** — the core install requires numpy>=1.25; run
  `pip install -e .` again from the repo root.
- **GPU not detected** — `cupy-cuda12x` only loads when CUDA 12.x runtime is
  present and `cupy.cuda.runtime.getDeviceCount() > 0`. The system silently
  uses NumPy otherwise.
- **Quantum backend silently falls back to simulator** — Qiskit must be
  importable AND, when a `seed` is passed to `ZeroDataModel`, the deterministic
  `SimulatorQuantumBackend` is forced for reproducibility (the Qiskit sampler
  draws from its own RNG, not numpy's).
