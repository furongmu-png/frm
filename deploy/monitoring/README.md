# ZeroDataModel 监控栈

基于 Prometheus + Grafana + AlertManager 的完整监控可视化与告警方案，覆盖 ZDM 全部 24 个自定义指标。

## 监控架构

```
┌─────────────────────────────────────────────────────────────┐
│                      ZDM Backend (:8000)                     │
│              /metrics 端点 (prometheus_client)               │
│         暴露 24 个 zdm_* 指标 (metrics.py)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ scrape (15s)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Prometheus (:9090)                         │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │  scrape configs │  │  alerting rules  │                  │
│  │ prometheus.yml  │  │   alerts.yml     │                  │
│  └─────────────────┘  └────────┬─────────┘                  │
└────────────────────────────────┼────────────────────────────┘
                                 │ fire alerts
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                AlertManager (:9093)                          │
│          路由 / 分组 / 抑制 / 去重                            │
│      receiver: default (webhook/email/slack 自行配置)        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Grafana (:3000)                           │
│  ┌────────────────┐  ┌──────────────────────────────────┐   │
│  │  provisioning  │  │          dashboards              │   │
│  │  datasources   │  │  zdm-overview.json (21 panels)   │   │
│  │  dashboards    │  │  zdm-errors.json    (7 panels)   │   │
│  └────────────────┘  └──────────────────────────────────┘   │
│         查询 Prometheus → 可视化所有 zdm_* 指标              │
└─────────────────────────────────────────────────────────────┘
```

**数据流**: ZDM Backend → Prometheus (采集) → Grafana (可视化) / AlertManager (告警)

## 目录结构

```
deploy/monitoring/
├── README.md                          # 本文档
├── docker-compose.monitoring.yml      # 监控栈编排
├── prometheus/
│   ├── prometheus.yml                 # 采集配置
│   └── alerts.yml                     # 告警规则 (8 条)
├── alertmanager/
│   └── alertmanager.yml               # 告警路由配置
└── grafana/
    ├── provisioning/
    │   ├── datasources/
    │   │   └── prometheus.yml         # 数据源自动配置
    │   └── dashboards/
    │       └── zdm.yml                # 仪表盘自动加载
    └── dashboards/
        ├── zdm-overview.json          # 总览仪表盘 (21 面板)
        └── zdm-errors.json            # 错误与性能仪表盘 (7 面板)
```

## 快速启动

### 前置条件

- Docker 与 Docker Compose 已安装
- ZDM Backend 已运行并暴露 `/metrics` 端点（在 `backend:8000` 上）

### 启动监控栈

```bash
# 从项目根目录执行
docker-compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
```

### 验证服务状态

```bash
# Prometheus
curl http://localhost:9090/-/healthy

# AlertManager
curl http://localhost:9093/-/healthy

# Grafana
curl http://localhost:3000/api/health
```

### 停止监控栈

```bash
docker-compose -f deploy/monitoring/docker-compose.monitoring.yml down
```

## 访问入口

| 服务           | 地址                   | 凭据          |
|----------------|------------------------|---------------|
| Grafana        | http://localhost:3000  | admin / admin |
| Prometheus     | http://localhost:9090  | 无需认证      |
| AlertManager   | http://localhost:9093  | 无需认证      |

## 仪表盘说明

### 1. ZeroDataModel — Cognitive System Overview (zdm-overview)

总览仪表盘，5 行 21 个面板，覆盖 ZDM 全部子系统：

| 行 | 面板 | 说明 |
|----|------|------|
| **Row 1: 核心系统状态** | Free Energy 趋势 / Think 周期数 / Think P95 延迟 / Think 延迟分布热力图 | `think()` 循环核心健康度 |
| **Row 2: 认知模块调用** | 各模块调用速率 / 错误率 / P95 延迟 / 错误率排行 (Top 5) | 13 个认知升级模块的通用调用监控 |
| **Row 3: 架构与记忆** | Architect dormant / Architect 操作 / Temporal 谱半径+MSE / Layered belief norm / Episodic 节点边数 / Semantic Index | 可塑性与时序/记忆子系统 |
| **Row 4: 认知与推理** | Meta-cognition 置信度 vs 不确定性 / Logic 违规+Penalty / Experiment 候选+Info Gain / Hypothesis 支持数 | 元认知、逻辑、实验与假设 |
| **Row 5: Multi-agent** | Collaboration events / Communication usage / Culture generations | 多智能体协作 |

### 2. ZeroDataModel — Errors & Performance (zdm-errors)

错误与性能深度分析仪表盘，7 个面板：

1. **Think 延迟 P50/P95/P99 对比** — 完整延迟分位数曲线
2. **各模块错误率热力图** — 按模块的错误率时间分布
3. **错误率告警阈值线** — 含 5% 告警阈值参考线
4. **模块调用延迟分布对比** — 各模块 P95 延迟横向对比
5. **模块累计错误数 (Top 10)** — Bar gauge 排行
6. **总错误数 (1h)** — Stat 汇总
7. **总调用数 (1h)** — Stat 汇总

## 告警规则说明

告警规则定义在 `prometheus/alerts.yml`，共 8 条，按严重等级分类：

| 告警名称 | 严重级别 | 触发条件 | 持续时间 |
|----------|----------|----------|----------|
| `ZDMBackendDown` | critical | backend scrape 失败 | 1m |
| `ZDMModuleErrorRateHigh` | critical | 模块错误率 > 5% | 2m |
| `ZDMThinkLatencyHigh` | warning | Think P95 > 500ms | 5m |
| `ZDMFreeEnergyNotConverging` | warning | Free energy 10min 标准差 > 1.0 | 10m |
| `ZDMLogicViolationsIncreasing` | warning | 5min 内逻辑违规 > 10 | 5m |
| `ZDMMetaCognitionLowConfidence` | warning | 元认知置信度 < 0.2 | 5m |
| `ZDMArchitectDormantSpike` | info | 10min 内 dormant 模块增加 > 5 | 5m |
| `ZDMExperimentNoProgress` | info | 30min 无新实验候选 | 30m |

### 告警路由

AlertManager 配置 (`alertmanager/alertmanager.yml`)：
- 按 `alertname` 和 `service` 分组
- 首次告警等待 30s (`group_wait`)
- 分组间隔 5min (`group_interval`)
- 重复通知间隔 4h (`repeat_interval`)
- 默认接收器 `default`：需用户自行配置 webhook/email/slack

## 自定义告警添加方法

### 1. 编辑告警规则

在 `prometheus/alerts.yml` 的 `zdm-critical` 组下添加新规则：

```yaml
groups:
  - name: zdm-critical
    rules:
      # ... 现有规则 ...

      # 新增告警示例：Episodic graph 过大
      - alert: ZDMEpisodicGraphTooLarge
        expr: zdm_episodic_node_count > 10000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Episodic graph node count exceeds 10000"
          description: "Current node count: {{ $value }}, consider consolidation"
```

也可创建新的告警组：

```yaml
groups:
  - name: zdm-critical
    rules: [...]
  - name: zdm-custom
    rules:
      - alert: MyCustomAlert
        expr: ...
```

### 2. 重载 Prometheus 配置

```bash
# 方式一：发送 reload 信号
docker-compose -f deploy/monitoring/docker-compose.monitoring.yml exec prometheus \
  kill -HUP 1

# 方式二：使用 Prometheus reload API (需启用 --web.enable-lifecycle)
curl -X POST http://localhost:9090/-/reload
```

### 3. 配置告警接收器

编辑 `alertmanager/alertmanager.yml`，在 `receivers` 中添加通知渠道：

```yaml
receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://your-webhook:5000/alert'
        send_resolved: true
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'
        channel: '#zdm-alerts'
```

更新路由规则将不同严重级别路由到不同接收器：

```yaml
route:
  group_by: ['alertname', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'
  routes:
    - matchers: ['severity="critical"']
      receiver: 'slack'
      continue: true
```

重载 AlertManager 配置：

```bash
docker-compose -f deploy/monitoring/docker-compose.monitoring.yml restart alertmanager
```

## 指标参考

所有 24 个自定义指标定义在 `src/zero_data_model/metrics.py`：

| 指标 | 类型 | 标签 |
|------|------|------|
| `zdm_think_duration_seconds` | Histogram | — |
| `zdm_cycle_count` | Gauge | — |
| `zdm_free_energy_last` | Gauge | — |
| `zdm_cognitive_module_calls_total` | Counter | module, method |
| `zdm_cognitive_module_duration_seconds` | Histogram | module, method |
| `zdm_cognitive_module_errors_total` | Counter | module, method |
| `zdm_architect_dormant_count` | Gauge | — |
| `zdm_architect_actions_total` | Counter | action_type |
| `zdm_temporal_spectral_radius` | Gauge | — |
| `zdm_temporal_mse` | Gauge | — |
| `zdm_layered_belief_norm` | Gauge | — |
| `zdm_episodic_node_count` | Gauge | — |
| `zdm_episodic_edge_count` | Gauge | — |
| `zdm_semantic_index_size` | Gauge | — |
| `zdm_logic_penalty` | Gauge | — |
| `zdm_logic_violation_count` | Gauge | — |
| `zdm_metacog_confidence` | Gauge | — |
| `zdm_metacog_uncertainty` | Gauge | — |
| `zdm_experiment_candidates` | Gauge | — |
| `zdm_experiment_info_gain` | Gauge | — |
| `zdm_hypothesis_supported` | Gauge | — |
| `zdm_multiagent_collaboration_events` | Gauge | — |
| `zdm_communication_usage` | Gauge | — |
| `zdm_culture_generations` | Gauge | — |
