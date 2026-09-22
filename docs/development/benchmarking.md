# Benchmarking

Two complementary benchmark surfaces: `benchmark.py` (Python, cognitive modules
+ hardware) and `tests/load/k6-*.js` (HTTP, REST API under load). Both feed
into CI.

## `benchmark.py`

Runs six benchmark groups and prints a per-dimension table. Lives at the repo
root.

| Group | What it measures |
| --- | --- |
| `[1] Quantum backend` | Qiskit vs `SimulatorQuantumBackend` evolve+measure |
| `[2] Annealer JIT` | `QuantumAnnealer.optimize` (numba-JIT vs numpy) |
| `[3] Parallel vs sequential` | `think()` cycle with `ParallelExecutor` vs 1 worker |
| `[4] Full model` | NLP classify / CV pattern / Analytics forecast throughput |
| `[JIT] numba kernels` | CellularAutomata, MorphogeneticField, FractalGenerator JIT vs reference |
| `[Cognitive Upgrade]` | 7-dimension on/off overhead (construct, think median, memory, metadata) |

### Running

```bash
# Full benchmark (all groups)
python benchmark.py

# Cognitive-upgrade dimensions only
python benchmark.py --cognitive-only

# Custom dim / cycle count
python benchmark.py --dim 128 --cycles 20

# Save current run as the regression baseline
python benchmark.py --save-baseline

# Compare current run against stored baseline (.benchmarks/baseline.json)
python benchmark.py --compare
```

### Cognitive-upgrade output

Each of the 7 dimensions (`architect`, `layered_predictor`, `episodic_memory`,
`logic_layer`, `meta_cognition`, `experiment_planner`, `all_upgrades`) reports:

| Metric | Meaning |
| --- | --- |
| `construct_ms` | Model `__init__` time (on = flag enabled, off = baseline) |
| `think_median_ms` | Median of 10 single-cycle `think()` runs |
| `think_total_ms` | `cycles` consecutive `think()` total |
| `mem_delta_kb` | `tracemalloc` peak during construction |
| `metadata_bytes` | `json.dumps(signal.metadata)` size |
| `overhead_ms` / `overhead_pct` | `on - off` delta vs baseline |

A regression is flagged when `think_median_ms`, `think_total_ms`, or
`construct_ms` worsens by more than **10%** vs the baseline.

### Baseline storage

`python benchmark.py --save-baseline` writes `.benchmarks/baseline.json`:

```json
{
  "timestamp": "2025-...",
  "python_version": "3.13.x",
  "numpy_version": "2.x.x",
  "platform": "...",
  "cognitive_upgrade": { "dim": 64, "cycles": 10, "dimensions": [...] }
}
```

On subsequent `--compare` runs, `benchmark.py` loads the baseline, recomputes
the same metrics, and prints a regression report:

```
REGRESSIONS (> 10% slower):
  dimension            metric                     old         new    %slower
  architect            think_median_ms          1.234       1.567        27.0
```

## CI integration

The `Benchmark` workflow (`.github/workflows/benchmark.yml`) runs:

- **Weekly** (Sunday 03:00 UTC) and on `workflow_dispatch`
- On PRs that touch `benchmark.py`, `model.py`, or any of the 7 cognitive
  module packages

On `main`, it commits an updated `.benchmarks/baseline.json` back to the repo.
On PRs, it fetches the `main` baseline, runs `--compare`, and posts a PR
comment with the regression report (updating the same comment if it already
exists).

## k6 load tests

Four k6 scenarios in `tests/load/`:

| Script | VUs × duration | Thresholds |
| --- | --- | --- |
| `k6-smoke.js` | 1 VU × 5 iterations | none (shape checks only) |
| `k6-load.js` | 10 VU × 3 min (ramp up/hold/down) | `p(95)<500ms`, `error rate<1%` |
| `k6-stress.js` | 50→100 VU × 6.5 min | `p(99)<2000ms`, `error rate<5%` |
| `k6-phase7.js` | 10 VU × 2 min | per-endpoint `p(95)<300ms`, aggregate `p(95)<500ms`, `error rate<1%` |

### Running k6 locally

```bash
# Install k6 (macOS)
brew install k6

# Or via Docker (no install)
docker run --rm --network host -v $(pwd)/tests/load:/scripts \
  grafana/k6 run /scripts/k6-smoke.js -e BASE_URL=http://localhost:8000

# Start the API server
ZDM_ENV=development uvicorn zero_data_model.api:app --host 0.0.0.0 --port 8000

# Run a scenario
k6 run tests/load/k6-load.js
k6 run tests/load/k6-phase7.js -e BASE_URL=http://staging:8000
```

### Expected status codes

k6 scripts declare `expectedStatuses` explicitly so expected responses are not
counted as failures:

| Endpoint | Expected | Reason |
| --- | --- | --- |
| `/health`, `/ready` | 200 | Always available |
| `/think` | 200, 429 | 429 = rate limiter (`10/minute`) working as designed |
| Phase 7 GET endpoints | 200, 503 | 503 = module disabled (feature flag off), not a server error |

### Adding a new endpoint to `k6-phase7.js`

Append to the `ENDPOINTS` array:

```js
const ENDPOINTS = [
  // ...
  { path: '/your/endpoint', tag: 'your_endpoint' },
];
```

`buildThresholds()` auto-generates a `p(95)<300` per-endpoint threshold from
the tag.

## Profiling

For deeper investigation, `profile_hotspots.py` uses `line_profiler` to find
hot lines in the cognitive modules:

```bash
pip install -e ".[dev]"   # line_profiler is in dev extras
python profile_hotspots.py
```

## Interpreting results

| Symptom | Likely cause |
| --- | --- |
| `think_median_ms` regresses >10% | A hot-path change in a cognitive module; check the diff against the baseline |
| k6 `p(95)` exceeds threshold under load | Rate limiter, GIL contention, or unbounded growth in a deque |
| `mem_delta_kb` grows over time | Unbounded cache / deque; check `maxlen` on `deque(...)` constructors |
| Phase 7 endpoints all return 503 | Feature flags disabled — start the model with `enable_*=True` |
