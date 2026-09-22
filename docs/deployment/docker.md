# Docker

ZeroDataModel ships a `docker-compose.yml` that brings up the backend (FastAPI
REST + WebSocket), the frontend (nginx-served SPA), and a Caddy reverse proxy
that terminates TLS on ports 80/443.

## Architecture

```
浏览器 ─── 80/443 ── Caddy (TLS, reverse proxy)
                     ├── /        → frontend (nginx static SPA)
                     ├── /api/*   → backend :8000 (REST)
                     └── /ws      → backend :8765 (WebSocket)
```

- **backend** — built from `Dockerfile.backend`, runs the visualization demo
  entry point exposing REST on `:8000` and WebSocket on `:8765`. Healthcheck
  hits `GET /health`.
- **frontend** — built from `Dockerfile.frontend`, serves the static SPA from
  `frontend/dist` via nginx.
- **caddy** — official `caddy:2-alpine` image, configured via `Caddyfile`.
  Domain defaults to `localhost` (override with the `DOMAIN` env var).

## Quick start

From the repository root:

```bash
DOMAIN=localhost docker compose up --build -d
```

Verify:

```bash
curl -sf http://localhost:8000/health        # {"status":"ok"}
docker compose ps
docker compose logs -f backend
```

Endpoints after startup:

| Service | URL |
| --- | --- |
| Frontend | http://localhost/ |
| REST API | http://localhost:8000 |
| WebSocket | ws://localhost:8765 |
| Swagger UI (dev) | http://localhost:8000/docs (only when `ZDM_ENV=development`) |

## Configuration via environment variables

Edit `docker-compose.yml` `backend.environment:` or pass env vars on the
command line. The most common ones (see [Configuration](../getting-started/configuration.md)
for the full table):

```bash
ZDM_API_KEY=$(openssl rand -hex 32) \
ZDM_ENV=production \
ZDM_RATE_LIMIT_RPM=120 \
DOMAIN=zdm.example.com \
docker compose up --build -d
```

The `model-data` named volume persists backend experiment output at
`/app/experiments/output` across container restarts.

## Dev compose

A separate `docker-compose.dev.yml` exists for hot-reload development
(backend with `--reload`, frontend with Vite HMR). See `Dockerfile.dev` and
`docker-compose.dev.yml` for details.

## Other Dockerfiles

| File | Purpose |
| --- | --- |
| `Dockerfile.backend` | Backend image (Python + uvicorn entry point) |
| `Dockerfile.frontend` | Frontend image (Node build + nginx static serve) |
| `Dockerfile` | Combined image (backend + frontend in one container) |
| `Dockerfile.dev` | Development image with hot reload |
| `Caddyfile` | Caddy reverse-proxy config (TLS, `/api` and `/ws` routing) |
| `nginx.conf` | nginx config embedded in the frontend image |
| `.dockerignore` | Build-context exclusions |

## Stop / clean up

```bash
docker compose down                  # stop containers, keep volumes
docker compose down -v               # also remove the model-data volume
```
