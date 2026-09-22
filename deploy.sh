#!/usr/bin/env bash
set -euo pipefail

# ZeroDataModel — one-click deploy script
# Usage: DOMAIN=your.domain.com ./deploy.sh
#        ./deploy.sh  (local, no HTTPS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  ZeroDataModel — Deployment"
echo "=========================================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: docker is not installed"
    exit 1
fi
if ! docker compose version &> /dev/null; then
    echo "ERROR: docker compose is not available"
    exit 1
fi

# Build and start
echo "[1/3] Building images..."
docker compose build

echo "[2/3] Starting services..."
docker compose up -d

echo "[3/3] Waiting for backend health..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health &> /dev/null; then
        echo "  Backend is healthy!"
        break
    fi
    sleep 2
    if [ $i -eq 30 ]; then
        echo "  WARNING: Backend health check timed out"
    fi
done

echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
if [ -n "${DOMAIN:-}" ]; then
    echo "  HTTPS:  https://$DOMAIN"
    echo "  HTTP:   http://$DOMAIN"
else
    echo "  Frontend: http://localhost"
    echo "  API:      http://localhost:8000"
    echo "  WebSocket: ws://localhost:8765"
fi
echo ""
echo "  Logs:     docker compose logs -f"
echo "  Stop:     docker compose down"
echo "=========================================="
