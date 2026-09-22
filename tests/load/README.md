# 负载测试 (k6)

本目录包含 ZeroDataModel REST API 的 [k6](https://k6.io) 负载 / 压力测试脚本。
k6 是 Grafana 出品的开源负载测试工具，使用 ES 模块语法的 JavaScript 编写。

## 目录

| 脚本 | 用途 | VU / 时长 | 阈值 |
|------|------|-----------|------|
| `k6-smoke.js` | 冒烟测试：验证服务已启动且 `/health`、`/ready`、`/think` 返回预期形状 | 1 VU × 5 次迭代 | 无（仅 check 断言） |
| `k6-load.js` | 持续负载：模拟正常流量，验证 p(95) 延迟与错误率 | 10 VU × 3 分钟（含爬升/保持/下降） | `p(95)<500ms`、`error rate<1%` |
| `k6-stress.js` | 压力测试：突破正常容量，寻找断裂点 | 50→100 VU × 6.5 分钟 | `p(99)<2000ms`、`error rate<5%` |
| `k6-phase7.js` | Phase 7 认知升级模块端点逐端点延迟基准 | 10 VU × 2 分钟 | 每 endpoint `p(95)<300ms`，聚合 `p(95)<500ms`、`error rate<1%` |

---

## 1. 安装 k6

### macOS (Homebrew)

```bash
brew install k6
```

### Linux (apt — Debian/Ubuntu)

```bash
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E65071
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt update
sudo apt install k6
```

### Linux (二进制下载)

```bash
# 从 https://github.com/grafana/k6/releases 下载最新 release
curl -sSL https://github.com/grafana/k6/releases/download/v0.51.0/k6-v0.51.0-linux-amd64.tar.gz | tar xz
sudo mv k6-v0.51.0-linux-amd64/k6 /usr/local/bin/
```

### Docker（无需安装）

```bash
docker run --rm --network host -v $(pwd)/tests/load:/scripts \
  grafana/k6 run /scripts/k6-smoke.js -e BASE_URL=http://localhost:8000
```

验证安装：

```bash
k6 version
```

---

## 2. 启动 API 服务器

所有脚本默认指向 `http://localhost:8000`，可通过 `BASE_URL` 环境变量覆盖。

```bash
# 安装依赖（含 web extra：fastapi/uvicorn/slowapi）
pip install -e ".[dev,web]"

# 开发模式启动（暴露 /docs，关闭 API key 认证）
ZDM_ENV=development uvicorn zero_data_model.api:app --host 0.0.0.0 --port 8000
```

> **注意**：`python -m zero_data_model` 运行的是 mini-demo，**不会**启动 FastAPI 服务器。
> 必须使用 uvicorn 启动 `zero_data_model.api:app`。

验证服务已就绪：

```bash
curl -fsS http://localhost:8000/health
# {"status":"ok",...}
```

---

## 3. 运行脚本

### 冒烟测试

```bash
k6 run tests/load/k6-smoke.js
```

覆盖默认地址：

```bash
k6 run tests/load/k6-smoke.js -e BASE_URL=http://staging:8000
```

### 负载测试

```bash
k6 run tests/load/k6-load.js
```

### 压力测试

```bash
k6 run tests/load/k6-stress.js
```

### Phase 7 端点基准

```bash
k6 run tests/load/k6-phase7.js
```

> 该脚本会输出每个 endpoint 的 p(95) 延迟与 PASS/FAIL 表格。

---

## 4. 脚本说明

### k6-smoke.js

冒烟测试。1 个虚拟用户 (VU) 执行 5 次迭代，每次：
1. `GET /health` — 断言 200
2. `GET /ready` — 断言 200
3. `POST /think` (body `{"cycles":1}`) — 断言 200 且响应含 `free_energy` 字段

使用固定迭代数（而非 `duration`）是因为 `/think` 端点有 `10/minute` 速率限制（见
`api.py` 的 `@_limit("10/minute")`）。5 次请求远低于此上限，避免触发 429。

### k6-load.js

持续负载测试。10 VU、3 分钟（30s 爬升 → 2m 保持 → 30s 下降）。每组迭代命中 4 个分组：

| 分组 | 端点 | 说明 |
|------|------|------|
| `health` | `GET /health` | 限流豁免路径 |
| `ready` | `GET /ready` | 就绪探针 |
| `think` | `POST /think` | 接受 200 或 429（限流正常工作） |
| `phase7 architect` | `GET /architect/stats` | 接受 200 或 503（模块未启用） |

阈值：`http_req_duration p(95)<500ms`、`http_req_failed rate<0.01`。

### k6-stress.js

压力测试。50→100 VU、6.5 分钟。与负载测试相同的 4 个分组，但 pacing 更紧
（`sleep(0.1)`）和更宽松的阈值（`p(99)<2000ms`、`error rate<5%`），
反映高压下更高的错误容忍度。

### k6-phase7.js

针对 40 个 phase7 端点中的 29 个只读 GET 端点做逐端点延迟基准。
覆盖模块：architect、temporal_memory、layered_predictor、episodic_graph、
semantic_index、logic_layer、meta_cognition、experiment_planner、
hypothesis_tester、world、communication、culture。

每个端点使用 `endpoint:<tag>` 标签，k6 的 per-tag 阈值
（`http_req_duration{endpoint:<tag>}: p(95)<300`）能精确锁定哪个端点慢，
而不是被聚合 p(95) 掩盖。

`handleSummary()` 在运行结束后输出表格：

```
=== Per-endpoint latency (p95) ===
  architect_stats                          p95=12.34ms  [PASS]
  temporal_memory_context                  p95=8.91ms   [PASS]
  ...
```

---

## 5. 阈值说明

k6 阈值 (thresholds) 是硬性通过/失败门禁。运行结束时若任一阈值未满足，
k6 退出码非零（CI 中会标红）。

| 脚本 | 阈值 | 含义 |
|------|------|------|
| smoke | 无 | 仅用 `check()` 断言响应形状 |
| load | `http_req_duration: p(95)<500` | 95% 请求在 500ms 内完成 |
| load | `http_req_failed: rate<0.01` | 错误率低于 1% |
| stress | `http_req_duration: p(99)<2000` | 99% 请求在 2s 内完成 |
| stress | `http_req_failed: rate<0.05` | 错误率低于 5% |
| phase7 | `http_req_duration: p(95)<500` | 聚合 p(95) < 500ms |
| phase7 | `http_req_failed: rate<0.01` | 聚合错误率 < 1% |
| phase7 | `http_req_duration{endpoint:<tag>}: p(95)<300` | 每 endpoint p(95) < 300ms |

### 预期状态码

脚本使用 `expectedStatuses` 显式声明可接受的状态码，避免预期响应被误计为失败：

| 端点 | 预期状态码 | 原因 |
|------|-----------|------|
| `/health`, `/ready` | 200 | 始终可用 |
| `/think` | 200, 429 | 429 = 速率限制器正常工作（`10/minute`） |
| phase7 GET 端点 | 200, 503 | 503 = 模块未启用（feature flag 关闭），不是服务器错误 |

---

## 6. 解读结果

### 关键指标

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `http_req_duration` | 请求总耗时（含 TLS、发送、等待、接收） | p(95)/p(99) 百分位 |
| `http_req_failed` | 失败请求比例 | `expectedStatuses` 之外的状态码 |
| `http_reqs` | 总请求数 | 吞吐量参考 |
| `iterations` | 完整迭代次数 | VU × 时长 / 单次迭代耗时 |
| `vus` | 当前活跃虚拟用户数 | stages 爬升轨迹 |

### 退出码

| 退出码 | 含义 |
|--------|------|
| 0 | 全部 check 通过、全部阈值满足 |
| 非 0 | 有 check 失败或阈值未满足 |

CI 中 `grafana/k6-action@v0.3.1` 会根据退出码决定 job 成功/失败。

### 常见问题

**Q: smoke 测试的 `/think` 返回 429？**
A: 冒烟脚本已限制为 5 次迭代，不应触发 `10/minute` 限制。若仍出现 429，
   检查是否有其他进程在并发请求 `/think`。

**Q: load/stress 测试 `http_req_failed` 超标？**
A: 检查是否所有 `expectedStatuses` 都已正确声明。若 phase7 端点返回 504
   或 500（而非 503），说明模块配置错误，需检查 `enable_*` feature flag。

**Q: phase7 脚本全部返回 503？**
A: 这是正常的——phase7 认知模块默认未启用。如需测试 200 响应，
   在启动模型时设置 feature flag（如 `enable_architect=True`），
   或使用 `ZeroDataModel(dim=32, enable_architect=True, ...)` 构造模型。

**Q: 如何在 CI 中运行？**
A: `.github/workflows/ci.yml` 的 `chaos` job 会自动运行 `k6-smoke.js`。
   详见 CI 配置中的 `Start FastAPI server` 与 `Run k6 smoke load test` 步骤。

---

## 7. CI 集成

`.github/workflows/ci.yml` 中的 `chaos` job 会在每次 push/PR 时：

1. 安装依赖（`[dev,web]` extras）
2. `ruff check tests/chaos/` 检查混沌测试代码风格
3. `pytest tests/chaos/ -v` 运行混沌工程测试
4. 用 uvicorn 后台启动 API 服务器（`ZDM_ENV=development`）
5. 等待 `/health` 就绪（最多 30s）
6. 使用 `grafana/k6-action@v0.3.1` 运行 `k6-smoke.js`
7. 无论成功失败，停止 API 服务器

---

## 8. 自定义阈值与扩展

添加新端点到 `k6-phase7.js`：在 `ENDPOINTS` 数组中追加
`{ path: '/your/endpoint', tag: 'your_endpoint' }`，
`buildThresholds()` 会自动为其生成 `p(95)<300` 的 per-endpoint 阈值。

调整负载强度：修改 `k6-load.js` / `k6-stress.js` 的 `stages` 数组中的
`target`（VU 数）和 `duration`（持续时长）。
