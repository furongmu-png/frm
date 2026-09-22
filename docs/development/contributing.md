# Contributing

Thanks for contributing to ZeroDataModel. This guide covers setup, the local
dev loop, branching, version bumps, and releases.

## Prerequisites

- Python **3.10+** (CI tests 3.11, 3.12, 3.13, 3.14)
- Node **20+** (for the frontend)
- Docker + Docker Compose (for deployment / monitoring stacks)

## Local setup

```bash
# Clone
git clone <repo-url> zero-data-model
cd zero-data-model

# Python env (use whatever env manager you prefer)
python -m venv .venv
source .venv/bin/activate

# Install with all extras (fastest path to a working dev env)
pip install -e ".[dev,web,mcp,quantum,jit,parallel]"

# Frontend
cd frontend && npm install && cd ..

# Pre-commit hooks (ruff + bandit)
pip install pre-commit
pre-commit install
```

Verify the install:

```bash
python -c "from zero_data_model.api import create_app; print(create_app())"
pytest tests/ -q
```

## Pre-commit hooks

`.pre-commit-config.yaml` mirrors the CI lint and security gates so they run
locally before every commit:

| Hook | What it does |
| --- | --- |
| `ruff` (`--fix`) | Auto-fix lint issues in `src/` and `tests/` |
| `ruff-format` | Enforce formatting |
| `bandit` (`-r src/ -ll -ii`) | Static security analysis (low-severity, recursive) |

To run all hooks manually:

```bash
pre-commit run --all-files
```

## Linting and type checking

```bash
# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Type check (7 strict modules)
mypy src/zero_data_model/cogmem/ \
     src/zero_data_model/cogtime/ \
     src/zero_data_model/plasticity/ \
     src/zero_data_model/knowledge/ \
     src/zero_data_model/metacog/ \
     src/zero_data_model/experiment/ \
     src/zero_data_model/multiagent/

# Frontend lint
cd frontend && npm run lint
```

The 7 cognitive-upgrade module packages have `strict = true` in
`pyproject.toml` `[tool.mypy.overrides]`. The legacy modules
(`model.py`, `hardware/`, `capabilities/`) use `follow_imports = "silent"` to
avoid blocking on pre-existing typing debt.

## Branching and PRs

1. **Branch from `main`.** Use a descriptive name:
   `feat/<short-desc>`, `fix/<short-desc>`, `docs/<short-desc>`,
   `chore/<short-desc>`.
2. **Keep PRs small.** One logical change per PR makes review faster and
   rollbacks safer.
3. **Run all gates locally** before pushing:

   ```bash
   pre-commit run --all-files
   pytest tests/ -q --cov-fail-under=70
   cd frontend && npm test && npx playwright test && cd ..
   ```
4. **Write tests for new behavior.** Unit tests in `tests/test_<module>.py`,
   integration tests with the `_integration.py` suffix.
5. **Update `CHANGELOG.md`** under the `[Unreleased]` section using
   [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) categories
   (`Added`, `Changed`, `Fixed`, `Removed`).
6. **Open the PR.** CI runs the `test`, `e2e`, `security`, and `chaos` jobs.
   All must pass (security SARIF is uploaded but non-blocking).

### Commit message style

Conventional Commits are preferred but not enforced:

```
feat(multiagent): add culture propagation acceleration metric
fix(api): correct rate-limit exemption for /health
docs(api): expand REST endpoint table
chore: bump numpy upper bound to <3
```

## Versioning

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The version lives in three places that must stay in sync:

| File | Field |
| --- | --- |
| `pyproject.toml` | `[project] version` |
| `deploy/k8s/helm/Chart.yaml` | `version` and `appVersion` |

Use the bump script to update all of them at once:

```bash
./scripts/bump-version.sh 0.3.0
```

The script validates the version matches `MAJOR.MINOR.PATCH` and prints the
next steps (commit, tag, push).

## Release process

Releases are triggered by pushing a `v*.*.*` tag. The `Release` workflow
(`.github/workflows/release.yml`) runs five jobs:

1. **`validate-version`** — extracts the version from the tag and asserts
   `pyproject.toml` and `Chart.yaml` match.
2. **`test`** — runs the full pytest suite on Python 3.11 and 3.12 as a gate.
3. **`build-images`** — builds `Dockerfile.backend` and `Dockerfile.frontend`,
   pushes to `ghcr.io/<owner>/<repo>-{backend,frontend}:{version,latest,sha}`.
4. **`build-helm`** — packages the Helm chart and pushes it to GHCR as an OCI
   artifact.
5. **`github-release`** — creates a GitHub Release with auto-generated notes
   and the Helm chart `.tgz` attached.

### Cutting a release

```bash
# 1. Bump version (updates pyproject.toml + Chart.yaml)
./scripts/bump-version.sh 0.3.0

# 2. Commit and tag
git add pyproject.toml deploy/k8s/helm/Chart.yaml
git commit -m "chore: bump version to 0.3.0"
git tag v0.3.0

# 3. Push (this triggers the Release workflow)
git push origin v0.3.0
```

## Updating the changelog

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The `[Unreleased]` section accumulates entries during development; on release,
rename it to `[0.3.0] - 2025-XX-XX` and add a fresh `[Unreleased]` block above
it.

```markdown
## [Unreleased]

### Added
- New feature X

### Fixed
- Bug Y in module Z

## [0.2.1] - 2025-01-15

### Added
- 7-stage cognitive upgrade modules
```

## Project layout

```
src/zero_data_model/        # Python package (src layout)
├── model.py                # ZeroDataModel orchestrator
├── api.py                  # FastAPI REST API
├── mcp_server.py           # FastMCP server
├── base.py                 # CognitiveModule ABC, Signal
├── persistence.py          # ModelSerializer
├── metrics.py              # 24 Prometheus metrics
├── cogmem/                 # Phase 3: episodic graph, semantic index
├── cogtime/                # Phase 2: layered predictor, temporal memory
├── plasticity/             # Phase 1: architecture optimizer
├── knowledge/              # Phase 4: logic layer, causal inference
├── metacog/                # Phase 5: meta-cognition
├── experiment/             # Phase 6: experiment planner, hypothesis tester
└── multiagent/             # Phase 7: world, communication, culture
tests/                      # pytest + chaos + k6 load
frontend/                   # React + Vite + Playwright
sdk/{python,typescript}/    # Generated client SDKs
deploy/                     # k8s, monitoring, docker-compose
docs/                       # This documentation site
```

## Getting help

- Browse the [Architecture](../architecture/overview.md) docs to understand
  the four-layer stack.
- Check existing tests in `tests/` for usage patterns.
- For module-specific behavior, see the [Modules](../modules/consciousness.md)
  section.
