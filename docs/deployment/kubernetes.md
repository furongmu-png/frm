# Kubernetes

ZeroDataModel ships Kubernetes configuration under `deploy/k8s/` with two
deployment paths:

| Path | Location | Use when |
| --- | --- | --- |
| **Helm Chart** | `deploy/k8s/helm/` | Production — parameterised, version-managed, rolling upgrades. |
| **Raw Manifests** | `deploy/k8s/manifests/` | Quick trial — no Helm required, just `kubectl apply`. |

## Architecture

```
                ┌──────────────┐
   Browser ─── 80/443 │   Ingress    │  (nginx ingress controller)
                └──────┬───────┘
              ┌────────┼────────────┐
              ▼        ▼            ▼
         /  (SPA)   /api/*        /ws
              │        │            │
              ▼        ▼            ▼
        ┌─────────┐  ┌─────────────────────┐
        │frontend │  │      backend        │
        │ (nginx) │  │ REST :8000  WS :8765│
        │  :80    │  │  (FastAPI/uvicorn)  │
        └─────────┘  └─────────────────────┘
```

- **frontend** — nginx serves the static SPA; a ConfigMap overrides the
  image-embedded `nginx.conf` (which proxies to the compose service name
  `backend`, unresolvable in K8s) with a static-files-only config. Routing is
  done by the Ingress.
- **backend** — runs the visualization demo, exposing REST (`:8000`) and
  WebSocket (`:8765`).
- **Ingress** — `/` → frontend; `/api/*` → backend REST (strips `/api` prefix
  so the backend root routes like `/health` work); `/ws` → backend WebSocket.

## Prerequisites

- Kubernetes **≥ 1.23**
- `kubectl` configured and connected to the target cluster
- **Helm 3.x** (Helm path only)
- **nginx Ingress Controller** installed in the cluster
- Backend / frontend images built and pushed to a registry the cluster can
  reach:

  ```bash
  docker build -f Dockerfile.backend  -t <registry>/zero-data-model-backend:1.0.0  .
  docker build -f Dockerfile.frontend -t <registry>/zero-data-model-frontend:1.0.0 .
  docker push <registry>/zero-data-model-backend:1.0.0
  docker push <registry>/zero-data-model-frontend:1.0.0
  ```

## Path A: Helm

```bash
kubectl create namespace zdm

# Default install
helm install zdm ./deploy/k8s/helm -n zdm

# Customised install
helm install zdm ./deploy/k8s/helm -n zdm \
  --set backend.image.repository=<registry>/zero-data-model-backend \
  --set backend.image.tag=1.0.0 \
  --set frontend.image.repository=<registry>/zero-data-model-frontend \
  --set frontend.image.tag=1.0.0 \
  --set ingress.host=zdm.example.com \
  --set backend.secret.ZDM_API_KEY="your-secret-key"
```

Or via a values file (recommended for production):

```bash
helm install zdm ./deploy/k8s/helm -n zdm -f values-prod.yaml
```

### Verify

```bash
kubectl -n zdm get pods
kubectl -n zdm get svc
kubectl -n zdm get ingress
kubectl -n zdm get hpa
```

Local port-forward for testing:

```bash
kubectl -n zdm port-forward svc/zdm-frontend 8080:80
# then open http://localhost:8080
```

## Path B: Raw manifests

```bash
kubectl apply -f deploy/k8s/manifests/namespace.yaml
# Edit configmap.yaml to set ZDM_API_KEY first:
kubectl apply -f deploy/k8s/manifests/configmap.yaml
kubectl apply -f deploy/k8s/manifests/backend.yaml
kubectl apply -f deploy/k8s/manifests/frontend.yaml
# Edit ingress.yaml to set the host first:
kubectl apply -f deploy/k8s/manifests/ingress.yaml
```

Or all at once (defaults: image names `zero-data-model-backend:latest` /
`zero-data-model-frontend:latest`):

```bash
kubectl apply -f deploy/k8s/manifests/
```

## Key `values.yaml` parameters

### backend

| Parameter | Default | Description |
| --- | --- | --- |
| `backend.image.repository` | `zero-data-model-backend` | Backend image repository |
| `backend.image.tag` | `latest` | Image tag |
| `backend.replicas` | `1` | Replica count (overridden by HPA when enabled) |
| `backend.restPort` | `8000` | REST API port |
| `backend.wsPort` | `8765` | WebSocket port |
| `backend.resources.requests` | `cpu: 500m, memory: 512Mi` | Resource requests |
| `backend.resources.limits` | `cpu: 2000m, memory: 2Gi` | Resource limits |
| `backend.livenessProbe` | `GET /health :8000` | Liveness probe |
| `backend.readinessProbe` | `GET /ready :8000` | Readiness probe |

Non-sensitive backend env vars land in a ConfigMap; sensitive ones in a Secret:

| Variable | Default | Description |
| --- | --- | --- |
| `ZDM_DIM` | `64` | Model vector dimension |
| `ZDM_ENV` | `production` | `production` hides `/docs`; `development` exposes it |
| `ZDM_LOG_LEVEL` | `INFO` | Log level |
| `ZDM_CORS_ORIGINS` | `http://localhost:8000,http://localhost:3000` | CORS allow-list |
| `ZDM_ALLOWED_HOSTS` | `localhost,127.0.0.1` | TrustedHost allow-list |
| `ZDM_ENABLE_ARCHITECT` | `false` | Enable plasticity module |
| `ZDM_ENABLE_LAYERED_PREDICTOR` | `false` | Enable cogtime module |
| `ZDM_ENABLE_CAUSAL_EMERGENCE` | `false` | Enable causal-emergence engine |
| `ZDM_ENABLE_METACOG` | `false` | Enable meta-cognition module |
| `ZDM_ENABLE_MULTIAGENT` | `false` | Enable multi-agent module |

Secret values:

| Variable | Description |
| --- | --- |
| `ZDM_API_KEY` | API access key (empty = auth disabled) |
| `IBM_QUANTUM_TOKEN` | IBM Quantum token (optional) |

### ingress

| Parameter | Default | Description |
| --- | --- | --- |
| `ingress.enabled` | `true` | Create an Ingress |
| `ingress.className` | `nginx` | IngressClass name |
| `ingress.host` | `zdm.local` | Hostname |
| `ingress.tls.enabled` | `false` | Enable TLS |
| `ingress.tls.secretName` | `""` | TLS Secret name (must pre-exist) |

### autoscaling (HPA)

| Parameter | Default | Description |
| --- | --- | --- |
| `autoscaling.enabled` | `true` | Create a backend HPA |
| `autoscaling.minReplicas` | `1` | Minimum replicas |
| `autoscaling.maxReplicas` | `4` | Maximum replicas |
| `autoscaling.targetCPUUtilizationPercentage` | `70` | CPU trigger threshold |

### podDisruptionBudget

| Parameter | Default | Description |
| --- | --- | --- |
| `podDisruptionBudget.enabled` | `true` | Create a PDB |
| `podDisruptionBudget.minAvailable` | `1` | Minimum available pods |

## `values-prod.yaml` example

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

## Upgrade / rollback

```bash
# Rolling upgrade after a values change
helm upgrade zdm ./deploy/k8s/helm -n zdm -f values-prod.yaml

# Bump just the image tag
helm upgrade zdm ./deploy/k8s/helm -n zdm \
  --set backend.image.tag=1.1.0 --reuse-values

# History & rollback
helm history zdm -n zdm
helm rollback zdm -n zdm                 # previous revision
helm rollback zdm <REVISION> -n zdm      # specific revision
```

Raw manifest rollback:

```bash
kubectl apply -f deploy/k8s/manifests/backend.yaml
kubectl rollout status deployment/zdm-backend -n zdm
kubectl rollout undo deployment/zdm-backend -n zdm
kubectl rollout undo deployment/zdm-backend -n zdm --to-revision=2
```

## Troubleshooting

- **Pod stuck `Pending`** — `kubectl -n zdm describe pod <pod>`; usually
  insufficient resources or image pull failure. Check events with
  `kubectl -n zdm get events --sort-by='.lastTimestamp'`.
- **`ImagePullBackOff`** — confirm the registry address and that the cluster
  has pull credentials. For private registries:
  ```bash
  kubectl -n zdm create secret docker-registry regcred \
    --docker-server=<registry> \
    --docker-username=<user> --docker-password=<pass>
  helm upgrade zdm ./deploy/k8s/helm -n zdm \
    --set backend.imagePullSecrets[0].name=regcred \
    --set frontend.imagePullSecrets[0].name=regcred
  ```
- **Backend healthcheck failing** —
  `kubectl -n zdm logs -l app.kubernetes.io/component=backend --tail=50`.
  Probes: liveness `GET /health`, readiness `GET /ready` (both on REST port
  8000).
- **Ingress unreachable** — confirm the nginx ingress controller is running
  (`kubectl get pods -n ingress-nginx`), the Ingress resource exists
  (`kubectl -n zdm describe ingress zdm`), and the host resolves to the
  cluster entry (LoadBalancer IP / NodePort). For local testing add
  `<ingress-controller-ip>  zdm.local` to `/etc/hosts`.
- **WebSocket connection failed** — confirm the Ingress annotations include a
  long timeout (`proxy-read-timeout: 86400`, set in the default values). The
  nginx ingress controller supports WS upgrade natively. The backend WS
  streamer listens on 8765 and accepts any path.
- **Frontend cannot reach backend** — in K8s + Ingress the 8000/8765 ports are
  not exposed externally. REST must go through `http://<ingress-host>/api/*`
  (Ingress strips `/api`); WebSocket through `ws://<ingress-host>/ws`. Adjust
  `frontend/src/store/useModelStore.ts` `WS_URL` / `REST_URL` or inject via
  build-time env vars if the frontend was built with different defaults.

## Uninstall

```bash
# Helm
helm uninstall zdm -n zdm
kubectl delete namespace zdm

# Raw manifests
kubectl delete -f deploy/k8s/manifests/
kubectl delete namespace zdm
```
