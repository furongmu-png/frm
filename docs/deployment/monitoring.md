# Monitoring

ZeroDataModel ships a complete Prometheus + Grafana + AlertManager monitoring
stack under `deploy/monitoring/`. It covers all 24 custom `zdm_*` metrics
exposed by the backend's `/metrics` endpoint.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ZDM Backend (:8000)                     │
│              /metrics endpoint (prometheus_client)           │
│         exposes 24 zdm_* metrics (metrics.py)                │
└──────────────────────────┬──────────────────────────────────┘
                           │ scrape (15s)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Prometheus (:9090)                         │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │  scrape configs │  │  alerting rules  │                  │
│  │ prometheus.yml  │  │   alerts.yml     │                  │
│  └─────────────────┘  └────────┬─────────┘                  │
└────────────────────────┼────────────────────────────────────┘
                         │ fire alerts
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                AlertManager (:9093)                          │
│          routing / grouping / inhibition / dedup            │
│      receiver: default (configure webhook/email/slack)      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Grafana (:3000)                           │
│  ┌────────────────┐  ┌──────────────────────────────────┐   │
│  │  provisioning  │  │          dashboards              │   │
│  │  datasources   │  │  zdm-overview.json (21 panels)   │   │
│  │  dashboards    │  │  zdm-errors.json    (7 panels)   │   │
│  └────────────────┘  └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Data flow:** ZDM Backend → Prometheus (scrape) → Grafana (visualise) /
AlertManager (alert).

## Quick start

### Prerequisites

- Docker and Docker Compose installed
- ZDM Backend running and exposing `/metrics` on `backend:8000`

### Start the stack

```bash
# From the repository root
docker-compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
```

### Verify

```bash
curl http://localhost:9090/-/healthy        # Prometheus
curl http://localhost:9093/-/healthy        # AlertManager
curl http://localhost:3000/api/health       # Grafana
```

### Stop

```bash
docker-compose -f deploy/monitoring/docker-compose.monitoring.yml down
```

## Access

| Service | URL | Credentials |
| --- | --- | --- |
| Grafana | http://localhost:3000 | `admin` / `admin` |
| Prometheus | http://localhost:9090 | none |
| AlertManager | http://localhost:9093 | none |

## Dashboards

### 1. Cognitive System Overview (`zdm-overview`)

5 rows, 21 panels, covering every ZDM subsystem:

| Row | Panels | What it shows |
| --- | --- | --- |
| Core system status | Free Energy trend / Think cycle count / Think P95 latency / latency heatmap | `think()` loop health |
| Cognitive module calls | call rate / error rate / P95 latency / top-5 error rate | 13 cognitive-upgrade modules |
| Architecture & memory | Architect dormant+ops / Temporal spectral radius+MSE / Layered belief norm / Episodic nodes+edges / Semantic index | plasticity + time/memory subsystems |
| Cognition & reasoning | Meta-cognition confidence vs uncertainty / Logic violations+penalty / Experiment candidates+info gain / Hypothesis support count | meta-cog, logic, experiment, hypothesis |
| Multi-agent | Collaboration events / Communication usage / Culture generations | multi-agent collaboration |

### 2. Errors & Performance (`zdm-errors`)

7 panels for deep error/performance analysis:

1. Think latency P50 / P95 / P99 comparison
2. Per-module error-rate heatmap over time
3. Error-rate alert threshold line (5% reference)
4. Per-module P95 latency comparison
5. Top-10 cumulative module errors (bar gauge)
6. Total errors (1h) — stat
7. Total calls (1h) — stat

## Alert rules

Defined in `prometheus/alerts.yml` — 8 rules, classified by severity:

| Alert | Severity | Trigger | For |
| --- | --- | --- | --- |
| `ZDMBackendDown` | critical | backend scrape failed | 1m |
| `ZDMModuleErrorRateHigh` | critical | module error rate > 5% | 2m |
| `ZDMThinkLatencyHigh` | warning | Think P95 > 500ms | 5m |
| `ZDMFreeEnergyNotConverging` | warning | Free energy 10min stddev > 1.0 | 10m |
| `ZDMLogicViolationsIncreasing` | warning | > 10 logic violations in 5min | 5m |
| `ZDMMetaCognitionLowConfidence` | warning | meta-cognition confidence < 0.2 | 5m |
| `ZDMArchitectDormantSpike` | info | dormant modules increase > 5 in 10min | 5m |
| `ZDMExperimentNoProgress` | info | no new experiment candidate in 30min | 30m |

### Alert routing (AlertManager)

Configured in `alertmanager/alertmanager.yml`:

- Group by `alertname` and `service`
- `group_wait: 30s` (first alert wait)
- `group_interval: 5m`
- `repeat_interval: 4h`
- Default receiver `default` — you must configure webhook / email / slack.

## Directory layout

```
deploy/monitoring/
├── README.md                          # this doc's source
├── docker-compose.monitoring.yml      # stack orchestration
├── prometheus/
│   ├── prometheus.yml                 # scrape config
│   └── alerts.yml                     # 8 alert rules
├── alertmanager/
│   └── alertmanager.yml               # alert routing
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml # auto datasource
    │   └── dashboards/zdm.yml         # auto dashboard loader
    └── dashboards/
        ├── zdm-overview.json          # 21-panel overview
        └── zdm-errors.json            # 7-panel error/perf
```

## Securing `/metrics`

The `/metrics` endpoint is auth-gated (`S-HIGH-02`): it requires an `X-API-Key`
header when `ZDM_API_KEY` is set, so Prometheus scrapes must include the key.
Add the key to the Prometheus scrape job's `bearer_token` or a custom
`Authorization` header in `prometheus/prometheus.yml`, or leave `ZDM_API_KEY`
unset on a trusted internal network.
