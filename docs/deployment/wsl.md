# WSL (Windows Subsystem for Linux)

The `deploy-wsl.sh` script is a one-shot deployer for WSL2. It auto-detects
whether Docker is available and falls back to a native Python + Node.js deploy
when it is not. The frontend uses `window.location.hostname` to derive the
backend address, so no reverse proxy is needed for LAN access.

## Usage

From the repository root:

```bash
./deploy-wsl.sh             # auto-detect environment and deploy
./deploy-wsl.sh stop        # stop all services
./deploy-wsl.sh status      # show service status
./deploy-wsl.sh logs        # tail logs (backend by default; or: logs frontend)
```

## Architecture

```
backend  experiments/run_visualization_demo.py  →  :8000 REST + :8765 WebSocket
frontend  frontend/dist (static)                →  :8080 HTTP
```

The frontend derives the backend URL from `window.location.hostname`, so a
browser hitting `http://<wsl-ip>:8080` automatically talks to
`http://<wsl-ip>:8000` and `ws://<wsl-ip>:8765`.

## Detection logic

1. **WSL2 detection** — `grep -qiE 'microsoft|wsl' /proc/version`.
2. **Docker detection** — `docker` and `docker compose version` both succeed.
3. **Native fallback** — Python ≥ 3.10 AND Node.js ≥ 18 must be present.

### Docker path

Runs `DOMAIN=localhost docker compose up --build -d` and waits for the backend
healthcheck (`GET /health` on :8000) up to 30 × 2s.

### Native path

When Docker is absent the script:

1. Installs backend deps: `pip install -e ".[web,parallel,jit]"` (falls back to
   `.[web]` if the full extras fail).
2. Builds the frontend: `cd frontend && npm install && npm run build`.
3. Starts the backend with `nohup python experiments/run_visualization_demo.py
   --steps 100000 --ws_host 0.0.0.0 --rest_host 0.0.0.0 --ws_port 8765
   --rest_port 8000` and writes the PID to `.wsl-pids/backend.pid`.
4. Starts a static HTTP server for the frontend on `:8080` from
   `frontend/dist`.
5. Waits for backend readiness (20 × 2s).

PID files and logs land in `.wsl-pids/` (`backend.pid`, `backend.log`,
`frontend.pid`, `frontend.log`).

## LAN access from Windows / other devices

WSL2's default NAT mode hides the WSL IP behind a virtual switch. Two options:

### Option A: Mirrored networking (Win11 22H2+, recommended)

Create or edit `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

Then `wsl --shutdown` and restart. LAN devices can hit the Windows host's IP
on the exposed ports directly — no port forwarding needed.

### Option B: Port forwarding (all WSL2 versions)

In an elevated Windows PowerShell, forward the Windows host ports to the WSL IP
(the script prints the exact commands after deploy):

```powershell
$wslIp = "<printed-by-deploy-wsl.sh>"
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=8765 listenaddress=0.0.0.0 connectport=8765 connectaddress=$wslIp

New-NetFirewallRule -DisplayName 'ZeroDataModel' -Direction Inbound -LocalPort 8080,8000,8765 -Protocol TCP -Action Allow
```

Inspect / clear:

```powershell
netsh interface portproxy show all
netsh interface portproxy reset
```

## Endpoints after deploy

| Service | Native path | Docker path |
| --- | --- | --- |
| Frontend | http://localhost:8080 | http://localhost/ |
| REST API | http://localhost:8000 | http://localhost:8000 |
| WebSocket | ws://localhost:8765 | ws://localhost:8765 |

## Stop / status / logs

```bash
./deploy-wsl.sh stop        # stops Docker compose AND native processes
./deploy-wsl.sh status      # shows Docker compose ps + native PID status
./deploy-wsl.sh logs        # tails .wsl-pids/backend.log (or: logs frontend)
```
