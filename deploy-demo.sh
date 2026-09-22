#!/usr/bin/env bash
#
# deploy-demo.sh — 认知起源演示一键部署脚本
#
# 功能：
#   1. 构建 Docker 镜像（含 ffmpeg + matplotlib + 演示脚本）
#   2. 运行认知起源 demo（4 阶段，可指定步数）
#   3. 自动生成素材索引 + 演示视频
#   4. （可选）启动 HTTPS 反代对外发布
#
# 用法：
#   ./deploy-demo.sh                       # 默认：1000 步 + 视频
#   ./deploy-demo.sh --steps 5000          # 全量 5000 步
#   ./deploy-demo.sh --serve               # 同时启动 HTTPS 服务
#   ./deploy-demo.sh --domain demo.example.com
#   ./deploy-demo.sh --cleanup             # 清理所有产物
#
# 依赖：Docker 20.10+ 与 docker compose v2

set -euo pipefail

# ---------------------------------------------------------------- #
# 参数解析
# ---------------------------------------------------------------- #
STEPS="${STEPS:-1000}"
SERVE=false
DOMAIN="${DOMAIN:-localhost}"
CLEANUP=false
BUILD_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)      STEPS="$2"; shift 2 ;;
        --serve)      SERVE=true; shift ;;
        --domain)     DOMAIN="$2"; shift 2 ;;
        --cleanup)    CLEANUP=true; shift ;;
        --build-only) BUILD_ONLY=true; shift ;;
        --help|-h)
            cat <<EOF
认知起源演示部署脚本

用法: $0 [选项]

选项:
  --steps N        demo 每阶段步数（默认 1000；全量 5000）
  --serve          同时启动 HTTPS 反代（Caddy，自动 TLS）
  --domain DOMAIN  HTTPS 域名（配合 --serve，默认 localhost）
  --build-only     仅构建镜像不运行
  --cleanup        清理所有容器与卷
  --help           显示此帮助

示例:
  $0                                  # 默认部署
  $0 --steps 5000 --serve --domain demo.example.com
  $0 --cleanup
EOF
            exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------- #
# 颜色输出
# ---------------------------------------------------------------- #
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
info() { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $*" >&2; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)]${NC} $*" >&2; }

# ---------------------------------------------------------------- #
# 前置检查
# ---------------------------------------------------------------- #
if ! command -v docker >/dev/null 2>&1; then
    err "未找到 docker，请先安装：https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    err "未找到 docker compose v2，请升级 Docker 或安装 docker-compose-plugin"
    exit 1
fi

# ---------------------------------------------------------------- #
# 清理模式
# ---------------------------------------------------------------- #
if $CLEANUP; then
    log "清理演示产物..."
    docker compose --profile demo down -v --remove-orphans 2>/dev/null || true
    docker compose down -v --remove-orphans 2>/dev/null || true
    rm -rf demo_output video_assets
    log "清理完成"
    exit 0
fi

# ---------------------------------------------------------------- #
# 主流程
# ---------------------------------------------------------------- #
cd "$(dirname "$0")"

log "=== 认知起源演示部署 ==="
info "步数: $STEPS"
info "域名: $DOMAIN"
info "启动 HTTPS 反代: $SERVE"
echo

# 1. 构建镜像
log "[1/4] 构建 Docker 镜像（zdm-demo:latest）..."
docker build -f Dockerfile.demo -t zdm-demo:latest .

if $BUILD_ONLY; then
    log "镜像构建完成（--build-only）"
    exit 0
fi

# 2. 运行 demo + 生成视频
log "[2/4] 运行认知起源 demo（4 阶段，每阶段 $STEPS 步）..."
docker run --rm \
    -v "$PWD/demo_output:/app/demo_output" \
    -v "$PWD/video_assets:/app/video_assets" \
    -e PYTHONUNBUFFERED=1 \
    -e MPLBACKEND=Agg \
    zdm-demo:latest \
    sh -c "
        python demo_cognitive_origin.py --steps $STEPS --no-streamer --output demo_output &&
        echo '--- 生成素材索引 ---' &&
        python build_material_index.py --input demo_output --out video_assets &&
        echo '--- 生成演示视频 ---' &&
        python render_demo_video.py --input demo_output --out video_assets
    "

# 3. 验证产物
log "[3/4] 验证产物..."
if [[ -f video_assets/cognitive_origin_demo.mp4 ]]; then
    SIZE=$(du -h video_assets/cognitive_origin_demo.mp4 | cut -f1)
    DUR=$(ffprobe -v error -show_entries format=duration \
          -of csv=p=0 video_assets/cognitive_origin_demo.mp4 2>/dev/null || echo "?")
    info "演示视频: video_assets/cognitive_origin_demo.mp4"
    info "  大小: $SIZE"
    info "  时长: ${DUR}s"
else
    warn "未生成演示视频"
fi
if [[ -f demo_output/report.json ]]; then
    STEPS_TOTAL=$(python3 -c "import json; print(json.load(open('demo_output/report.json'))['total_steps'])" 2>/dev/null || echo "?")
    MILESTONES=$(python3 -c "import json; print(json.load(open('demo_output/report.json'))['milestone_count'])" 2>/dev/null || echo "?")
    info "演示报告: demo_output/report.json"
    info "  总步数: $STEPS_TOTAL"
    info "  里程碑: $MILESTONES"
fi

# 4. 可选：启动 HTTPS 反代
if $SERVE; then
    log "[4/4] 启动 HTTPS 反代（Caddy）..."
    export DOMAIN
    docker compose --profile demo up -d backend frontend caddy
    info "服务已启动："
    info "  前端 + HTTPS:  https://$DOMAIN"
    info "  REST API:      https://$DOMAIN/api/"
    info "  WebSocket:     wss://$DOMAIN/ws"
    info "查看日志: docker compose logs -f"
else
    log "[4/4] 跳过 HTTPS 服务（如需启动加 --serve）"
fi

echo
log "=== 部署完成 ==="
info "素材目录: ./demo_output/"
info "视频目录: ./video_assets/"
info "重新运行: $0 --steps $STEPS"
info "查看报告: cat demo_output/report.json | python3 -m json.tool"
