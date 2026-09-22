# Testing

ZeroDataModel uses four complementary test layers: pytest (Python unit +
integration), vitest + Playwright (frontend), k6 (load/stress), and a chaos
suite (fault injection). All gates run in CI on every push and pull request.

## Test layout

| Path | Layer | Runner | What it covers |
| --- | --- | --- | --- |
| `tests/test_*.py` | Unit + integration | pytest | Core modules, REST API, MCP server, capabilities |
| `tests/chaos/` | Resilience | pytest | Fault injection, API resilience under errors |
| `tests/load/` | Load / stress | k6 | REST API throughput, latency, error-rate thresholds |
| `frontend/e2e/*.spec.ts` | E2E | Playwright | Frontend launch, tab navigation, no-crash, new panels |
| `frontend/src/**` | Unit | vitest | React component unit tests |
| `sdk/python/tests/` | SDK | pytest | Generated Python client |
| `sdk/typescript/tests/` | SDK | vitest | Generated TypeScript client |

## pytest (Python)

### Setup

```bash
# Install with dev + web extras (httpx is required by TestClient)
pip install -e ".[dev,web,mcp]"
```

`pyproject.toml` configures pytest:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

`conftest.py` adds an **autouse fixture** that re-seeds the global numpy RNG
(`np.random.seed(42)`) before every test. This keeps the 30+ legacy tests that
call `np.random.randn(...)` without an explicit seed deterministic. Tests that
use their own `np.random.default_rng(seed)` are unaffected.

### Running tests

```bash
# Full suite, quiet
python -m pytest tests/ -q

# A single module
python -m pytest tests/test_multiagent_world.py -v

# With coverage (CI gate is 70%)
python -m pytest tests/ \
  --cov=src/zero_data_model \
  --cov-report=term-missing \
  --cov-fail-under=70
```

### Test categories

The 90+ test files break down into:

| Prefix / group | Files | Covers |
| --- | --- | --- |
| `test_capabilities_*` | 18 | NLP, Vision, Audio, Code, Robotics, Analytics, Causal, Graph, Reasoning, Time, Rules |
| `test_phase6_*` | 5 | Phase 6 multimodal / memory / planning / RL / integration |
| `test_phase7_*` | 9 | Phase 7 module integrations (audio, causal, code, graph, multiagent, reasoning, robotics, time) |
| `test_causal_emergence_*` | 8 | Causal emergence engine, topology, HMC, chaotic memory, differential |
| `test_cogmem_*` / `test_cogtime_*` | 4 | Episodic graph, semantic index, layered predictor, temporal memory |
| `test_multiagent_*` | 3 | World, communication, culture |
| `test_round{4..10}_regressions.py` | 7 | Round-by-round regression suites (do not delete) |
| `test_mcp_*` | 3 | MCP server, integration, emergence |
| `test_*_integration.py` | several | Cross-module end-to-end |
| `test_property_invariants.py` | 1 | Hypothesis-based property tests |

### Property-based testing

`test_property_invariants.py` uses [Hypothesis](https://hypothesis.readthedocs.io).
The `.hypothesis/` directory stores the failing-example cache; commit it so
previously discovered counter-examples are replayed on every run.

## Frontend (vitest + Playwright)

### vitest unit tests

```bash
cd frontend
npm install
npm test                # vitest run
npm run test:watch      # vitest (watch mode)
```

Configuration: `vite.config.ts` (jsdom environment), setup file
`frontend/src/test-setup.ts`.

### Playwright E2E

```bash
cd frontend
npx playwright install --with-deps chromium
npx playwright test          # headless
npx playwright test --ui     # interactive
```

E2E specs in `frontend/e2e/`:

| Spec | Verifies |
| --- | --- |
| `app-launch.spec.ts` | App boots, renders without crash |
| `tab-navigation.spec.ts` | Tab switching works |
| `new-panels.spec.ts` | New panels render correctly |
| `no-crash.spec.ts` | No uncaught exceptions during navigation |

The E2E suite serves the production bundle via a static server (configured in
`playwright.config.ts`) — no backend required, the app must degrade gracefully.

### Frontend lint

```bash
cd frontend
npm run lint        # oxlint
```

## Load testing (k6)

Four k6 scenarios live in `tests/load/`. See
[Benchmarking](benchmarking.md#k6-load-tests) for the full table; the CI
`chaos` job runs `k6-smoke.js` against a live uvicorn server on every push.

```bash
# Local run (start the API first, see below)
ZDM_ENV=development uvicorn zero_data_model.api:app --host 0.0.0.0 --port 8000

# In another shell
k6 run tests/load/k6-smoke.js
k6 run tests/load/k6-load.js
k6 run tests/load/k6-stress.js
k6 run tests/load/k6-phase7.js
```

> `python -m zero_data_model` runs the mini-demo, **not** the FastAPI server.
> Always start the API with `uvicorn zero_data_model.api:app`.

## Chaos suite

```bash
python -m pytest tests/chaos/ -v
```

| File | What it does |
| --- | --- |
| `tests/chaos/test_fault_injection.py` | Injects faults into cognitive modules and asserts graceful degradation |
| `tests/chaos/test_api_resilience.py` | API resilience under error conditions |

## CI gates

The `CI` workflow (`.github/workflows/ci.yml`) runs four parallel jobs on every
push and pull request:

| Job | Matrix | Gate |
| --- | --- | --- |
| `test` | Python 3.11, 3.12, 3.13, 3.14 | `pytest` + `--cov-fail-under=70` + `ruff check` + `mypy` (7 strict modules) |
| `e2e` | Node 20 | `npx playwright test` |
| `security` | Python 3.13 | `pip-audit` + `bandit -r src/` (SARIF uploaded, non-blocking) |
| `chaos` | Python 3.13 | `pytest tests/chaos/` + `k6-smoke.js` |

`fail-fast: false` on the test matrix so a 3.13 failure does not hide a 3.12
regression.

## Writing a new test

1. **Pick the right folder.** Unit tests for a module go in `tests/test_<module>.py`.
   Integration tests across modules use the `_integration.py` suffix.
2. **Use explicit seeds.** Prefer `np.random.default_rng(seed)` over the global
   RNG. The autouse fixture covers legacy code, but new code should not rely on
   it.
3. **Mark slow tests** with `@pytest.mark.slow` if they exceed ~1 s so they can
   be filtered in local runs.
4. **Run the linters** before pushing:

   ```bash
   ruff check src/ tests/
   ruff format src/ tests/
   ```
