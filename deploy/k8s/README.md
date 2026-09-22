# ZeroDataModel — Kubernetes 部署指南

本目录包含 ZeroDataModel 的 Kubernetes 编排配置，提供两种部署方式：

| 方式 | 路径 | 适用场景 |
| --- | --- | --- |
| **Helm Chart** | `helm/` | 生产环境，支持参数化、版本管理、滚动升级 |
| **原生 Manifest** | `manifests/` | 快速试用，无需安装 Helm，直接 `kubectl apply` |

## 架构概览

```
                    ┌──────────────┐
   浏览器 ─── 80/443 │   Ingress    │  (nginx ingress controller)
                    └──────┬───────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         /  (SPA)     /api/*        /ws
              │            │            │
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────────────────┐
        │frontend │  │      backend        │
        │ (nginx) │  │ REST :8000  WS :8765│
        │  :80    │  │  (FastAPI/uvicorn)  │
        └─────────┘  └─────────────────────┘
```

- **frontend**：nginx 服务静态文件（SPA），由 `frontend-nginx` ConfigMap 提供仅静态文件配置。
- **backend**：运行 `run_visualization_demo.py`，暴露 REST API（`:8000`）与 WebSocket 流（`:8765`）。
- **Ingress**：`/` → frontend；`/api/*` → backend REST（剥离 `/api` 前缀，匹配后端根路由如 `/health`）；`/ws` → backend WebSocket。

> **注意**：镜像内嵌的 `nginx.conf` 将 `/api`、`/ws` 代理到 compose 服务名 `backend`，该名称在 K8s 中无法解析（会导致 nginx 启动失败）。因此在 K8s 中通过 ConfigMap 挂载一份仅静态文件的 nginx 配置覆盖它，路由统一由 Ingress 完成。

---

## 前置条件

- Kubernetes 集群（≥ 1.23）
- `kubectl` 已配置并连接到目标集群
- **Helm 3.x**（仅 Helm 部署方式需要）
- **nginx Ingress Controller** 已安装在集群中（Ingress 路由依赖它）
  ```bash
  # 检查是否已安装
  kubectl get pods -n ingress-nginx
  ```
- 已构建并推送到集群可访问的镜像仓库的 backend / frontend 镜像
  ```bash
  # 在项目根目录构建
  docker build -f Dockerfile.backend  -t <registry>/zero-data-model-backend:1.0.0  .
  docker build -f Dockerfile.frontend -t <registry>/zero-data-model-frontend:1.0.0 .
  docker push <registry>/zero-data-model-backend:1.0.0
  docker push <registry>/zero-data-model-frontend:1.0.0
  ```

---

## 方式一：Helm 部署

### 1. 创建命名空间

```bash
kubectl create namespace zdm
```

### 2. 安装 Chart

使用默认配置安装：

```bash
helm install zdm ./deploy/k8s/helm -n zdm
```

使用自定义镜像仓库和域名安装：

```bash
helm install zdm ./deploy/k8s/helm -n zdm \
  --set backend.image.repository=<registry>/zero-data-model-backend \
  --set backend.image.tag=1.0.0 \
  --set frontend.image.repository=<registry>/zero-data-model-frontend \
  --set frontend.image.tag=1.0.0 \
  --set ingress.host=zdm.example.com \
  --set backend.secret.ZDM_API_KEY="your-secret-key"
```

或通过 values 文件覆盖（推荐用于生产）：

```bash
# 创建 values-prod.yaml（示例见下方"配置说明"）
helm install zdm ./deploy/k8s/helm -n zdm -f values-prod.yaml
```

### 3. 验证部署

```bash
kubectl -n zdm get pods
kubectl -n zdm get svc
kubectl -n zdm get ingress
kubectl -n zdm get hpa
```

### 4. 访问服务

将 Ingress host 解析到集群入口后访问 `http://<ingress-host>/`。

本地测试可使用 port-forward：

```bash
kubectl -n zdm port-forward svc/zdm-frontend 8080:80
# 访问 http://localhost:8080
```

---

## 方式二：原生 Manifest 部署

无需 Helm，直接 `kubectl apply`。所有资源创建在 `zdm` namespace。

```bash
# 1. 创建 namespace
kubectl apply -f deploy/k8s/manifests/namespace.yaml

# 2. 应用 ConfigMap / Secret（先编辑 configmap.yaml 填入真实 ZDM_API_KEY）
kubectl apply -f deploy/k8s/manifests/configmap.yaml

# 3. 部署 backend
kubectl apply -f deploy/k8s/manifests/backend.yaml

# 4. 部署 frontend
kubectl apply -f deploy/k8s/manifests/frontend.yaml

# 5. 部署 Ingress（先编辑 ingress.yaml 修改 host）
kubectl apply -f deploy/k8s/manifests/ingress.yaml
```

> 原生 manifest 使用默认镜像名 `zero-data-model-backend:latest` / `zero-data-model-frontend:latest`。如需使用私有仓库，请编辑对应文件中的 `image:` 字段。

一键部署全部（按依赖顺序）：

```bash
kubectl apply -f deploy/k8s/manifests/
```

---

## 配置说明（values.yaml）

### backend

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `backend.image.repository` | `zero-data-model-backend` | 后端镜像仓库 |
| `backend.image.tag` | `latest` | 镜像标签 |
| `backend.image.pullPolicy` | `IfNotPresent` | 镜像拉取策略 |
| `backend.replicas` | `1` | 副本数（HPA 启用时被覆盖） |
| `backend.restPort` | `8000` | REST API 端口 |
| `backend.wsPort` | `8765` | WebSocket 端口 |
| `backend.resources.requests` | `cpu: 500m, memory: 512Mi` | 资源请求 |
| `backend.resources.limits` | `cpu: 2000m, memory: 2Gi` | 资源上限 |
| `backend.env.*` | 见下 | 非敏感环境变量（ConfigMap） |
| `backend.secret.*` | 见下 | 敏感环境变量（Secret） |
| `backend.livenessProbe` | `GET /health :8000` | 存活探针 |
| `backend.readinessProbe` | `GET /ready :8000` | 就绪探针 |

**环境变量（backend.env）：**

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ZDM_DIM` | `64` | 模型向量维度 |
| `ZDM_ENV` | `production` | 运行环境（`production` 隐藏 /docs；`development` 暴露） |
| `ZDM_LOG_LEVEL` | `INFO` | 日志级别 |
| `ZDM_CORS_ORIGINS` | `http://localhost:8000,http://localhost:3000` | CORS 允许来源（逗号分隔） |
| `ZDM_ALLOWED_HOSTS` | `localhost,127.0.0.1` | TrustedHost 允许主机（逗号分隔） |
| `ZDM_ENABLE_ARCHITECT` | `false` | 启用架构师（plasticity）模块 |
| `ZDM_ENABLE_LAYERED_PREDICTOR` | `false` | 启用分层预测器（cogtime） |
| `ZDM_ENABLE_CAUSAL_EMERGENCE` | `false` | 启用因果涌现引擎 |
| `ZDM_ENABLE_METACOG` | `false` | 启用元认知模块 |
| `ZDM_ENABLE_MULTIAGENT` | `false` | 启用多智能体模块 |
| `PYTHONUNBUFFERED` | `1` | 禁用 stdout 缓冲 |

**敏感变量（backend.secret）：**

| 变量 | 说明 |
| --- | --- |
| `ZDM_API_KEY` | API 访问密钥（留空则不启用密钥校验） |
| `IBM_QUANTUM_TOKEN` | IBM Quantum 令牌（可选，用于量子硬件后端） |

### frontend

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `frontend.image.repository` | `zero-data-model-frontend` | 前端镜像仓库 |
| `frontend.image.tag` | `latest` | 镜像标签 |
| `frontend.replicas` | `2` | 副本数 |
| `frontend.port` | `80` | nginx 监听端口 |
| `frontend.resources.requests` | `cpu: 100m, memory: 64Mi` | 资源请求 |
| `frontend.resources.limits` | `cpu: 500m, memory: 256Mi` | 资源上限 |

### ingress

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `ingress.enabled` | `true` | 是否创建 Ingress |
| `ingress.className` | `nginx` | IngressClass 名称 |
| `ingress.host` | `zdm.local` | 主机名 |
| `ingress.tls.enabled` | `false` | 是否启用 TLS |
| `ingress.tls.secretName` | `""` | TLS Secret 名称（需预先创建） |
| `ingress.annotations` | 见 values | nginx ingress 注解（rewrite-target 等） |

### autoscaling

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `autoscaling.enabled` | `true` | 是否为 backend 创建 HPA |
| `autoscaling.minReplicas` | `1` | 最小副本数 |
| `autoscaling.maxReplicas` | `4` | 最大副本数 |
| `autoscaling.targetCPUUtilizationPercentage` | `70` | CPU 触发阈值 |

### podDisruptionBudget

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `podDisruptionBudget.enabled` | `true` | 是否创建 PDB |
| `podDisruptionBudget.minAvailable` | `1` | 最少可用 Pod 数 |

---

## values-prod.yaml 示例

```yaml
backend:
  image:
    repository: registry.example.com/zero-data-model-backend
    tag: "1.0.0"
    pullPolicy: IfNotPresent
  secret:
    ZDM_API_KEY: "prod-secret-key"
  env:
    ZDM_ENV: "production"
    ZDM_LOG_LEVEL: "WARNING"

frontend:
  image:
    repository: registry.example.com/zero-data-model-frontend
    tag: "1.0.0"
  replicas: 3

ingress:
  host: zdm.example.com
  tls:
    enabled: true
    secretName: zdm-tls

autoscaling:
  maxReplicas: 8
  targetCPUUtilizationPercentage: 60
```

---

## 升级 / 回滚

### 升级

```bash
# 修改 values 后滚动升级
helm upgrade zdm ./deploy/k8s/helm -n zdm -f values-prod.yaml

# 仅更新镜像标签
helm upgrade zdm ./deploy/k8s/helm -n zdm \
  --set backend.image.tag=1.1.0 \
  --reuse-values
```

### 查看历史版本

```bash
helm history zdm -n zdm
```

### 回滚

```bash
# 回滚到上一个版本
helm rollback zdm -n zdm

# 回滚到指定版本
helm rollback zdm <REVISION> -n zdm
```

### 原生 manifest 升级

```bash
# 编辑 manifest 后重新 apply（Deployment 支持 RollingUpdate）
kubectl apply -f deploy/k8s/manifests/backend.yaml
kubectl rollout status deployment/zdm-backend -n zdm

# 回滚
kubectl rollout undo deployment/zdm-backend -n zdm
kubectl rollout undo deployment/zdm-backend -n zdm --to-revision=2
```

---

## 故障排查

### Pod 处于 Pending 状态

```bash
kubectl -n zdm describe pod <pod-name>
# 常见原因：资源不足、镜像拉取失败
```

检查事件：

```bash
kubectl -n zdm get events --sort-by='.lastTimestamp'
```

### 镜像拉取失败（ImagePullBackOff）

确认镜像仓库地址正确且集群有权限拉取。私有仓库需配置 imagePullSecrets：

```bash
kubectl -n zdm create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<user> \
  --docker-password=<pass>

# values.yaml
helm upgrade zdm ./deploy/k8s/helm -n zdm \
  --set backend.imagePullSecrets[0].name=regcred \
  --set frontend.imagePullSecrets[0].name=regcred
```

### 后端健康检查失败

```bash
# 查看 backend 日志
kubectl -n zdm logs -l app.kubernetes.io/component=backend --tail=50

# 进入 Pod 检查
kubectl -n zdm exec -it <backend-pod> -- python -c \
  "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').status)"
```

探针路径：liveness `GET /health`，readiness `GET /ready`（均走 REST 端口 8000）。

### Ingress 无法访问

1. 确认 nginx ingress controller 正在运行：
   ```bash
   kubectl get pods -n ingress-nginx
   ```
2. 确认 Ingress 资源已创建且 backend 就绪：
   ```bash
   kubectl -n zdm get ingress
   kubectl -n zdm describe ingress zdm
   ```
3. 确认 host 解析到集群入口（LoadBalancer IP / NodePort）：
   ```bash
   kubectl get svc -n ingress-nginx
   ```
4. 本地测试可在 `/etc/hosts` 添加：
   ```
   <ingress-controller-ip>  zdm.local
   ```

### WebSocket 连接失败

- 确认 Ingress 注解包含长超时（`proxy-read-timeout: 86400`，已在默认 values 中配置）。
- nginx ingress controller 默认支持 WebSocket 升级，无需额外配置。
- 后端 WS streamer 监听 8765 端口，接受任意路径。

### 前端连接后端失败（生产环境注意事项）

前端代码默认直连 `ws://<host>:8765` 和 `http://<host>:8000`。在 K8s + Ingress 环境下这些端口未对外暴露，需要通过 Ingress 路由访问：

- REST 请求应走 `http://<ingress-host>/api/*`（Ingress 剥离 `/api` 后转发到后端根路由）
- WebSocket 应走 `ws://<ingress-host>/ws`

如前端构建时未配置这些路径，可能需要调整前端 `useModelStore.ts` 中的 `WS_URL` / `REST_URL`，或通过构建期环境变量注入。

### 卸载

```bash
# Helm
helm uninstall zdm -n zdm
kubectl delete namespace zdm

# 原生 manifest
kubectl delete -f deploy/k8s/manifests/
kubectl delete namespace zdm
```

---

## 目录结构

```
deploy/k8s/
├── README.md                     # 本文档
├── helm/                         # Helm Chart
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── _helpers.tpl
│       ├── backend-deployment.yaml
│       ├── backend-service.yaml
│       ├── frontend-deployment.yaml
│       ├── frontend-service.yaml
│       ├── ingress.yaml
│       ├── configmap.yaml
│       ├── hpa.yaml
│       ├── pdb.yaml
│       ├── secret.yaml
│       └── serviceaccount.yaml
└── manifests/                    # 原生 Manifest（kubectl apply）
    ├── namespace.yaml
    ├── configmap.yaml
    ├── backend.yaml
    ├── frontend.yaml
    └── ingress.yaml
```
