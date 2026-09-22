#!/usr/bin/env bash
# deploy-wsl.sh — WSL2 一键部署 ZeroDataModel（自动检测 Docker / 原生降级）
#
# 用法:
#   ./deploy-wsl.sh            # 自动检测环境并部署
#   ./deploy-wsl.sh stop       # 停止所有服务
#   ./deploy-wsl.sh status     # 查看服务状态
#   ./deploy-wsl.sh logs       # 查看日志
#
# 架构:
#   后端  experiments/run_visualization_demo.py  →  :8000 REST + :8765 WebSocket
#   前端  frontend/dist (静态)                    →  :8080 HTTP
#   前端用 window.location.hostname 动态拼接后端地址，无需反向代理。
#
# 局域网访问:
#   WSL2 NAT 模式下需在 Windows PowerShell (管理员) 执行 portproxy 转发
#   3 个端口到 WSL IP。脚本部署后自动打印命令。
#   Win11 22H2+ 可在 ~/.wslconfig 设 networkingMode=mirrored 免转发。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
readonly _G=$'\033[32m'  # 绿
readonly _Y=$'\033[33m'  # 黄
readonly _R=$'\033[31m'  # 红
readonly _B=$'\033[36m'  # 青
readonly _N=$'\033[0m'   # 复位

info()  { printf "${_B}[INFO]${_N}  %s\n"  "$*"; }
ok()    { printf "${_G}[OK]${_N}    %s\n"  "$*"; }
warn()  { printf "${_Y}[WARN]${_N}  %s\n"  "$*"; }
err()   { printf "${_R}[ERROR]${_N} %s\n" "$*" >&2; }

# 端口常量
REST_PORT=8000
WS_PORT=8765
WEB_PORT=8080

# PID 文件目录（记录原生路径的进程）
PID_DIR="$SCRIPT_DIR/.wsl-pids"

# ------------------------------------------------------------------ #
# 环境检测
# ------------------------------------------------------------------ #

detect_wsl() {
    if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
        return 0
    fi
    return 1
}

get_wsl_ip() {
    # hostname -I 返回空格分隔的 IP 列表，取第一个
    local ip
    ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [[ -z "$ip" ]]; then
        ip="127.0.0.1"
    fi
    echo "$ip"
}

detect_docker() {
    if command -v docker &>/dev/null && docker compose version &>/dev/null; then
        return 0
    fi
    return 1
}

detect_python() {
    local py
    for py in python3 python; do
        if command -v "$py" &>/dev/null; then
            local ver
            ver=$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
            local major minor
            major=${ver%%.*}
            minor=${ver#*.}
            minor=${minor%%.*}
            if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
                echo "$py"
                return 0
            fi
        fi
    done
    return 1
}

detect_node() {
    if command -v node &>/dev/null; then
        local ver
        ver=$(node -v 2>/dev/null | sed 's/v//' | cut -d. -f1)
        if [[ "$ver" -ge 18 ]]; then
            return 0
        fi
    fi
    return 1
}

check_port_free() {
    local port=$1
    if command -v ss &>/dev/null; then
        if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
            return 1
        fi
    elif command -v netstat &>/dev/null; then
        if netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
            return 1
        fi
    fi
    return 0
}

# ------------------------------------------------------------------ #
# Docker 路径
# ------------------------------------------------------------------ #

deploy_docker() {
    info "检测到 Docker，使用 docker-compose 部署..."

    # 检查端口占用（Caddy 用 80/443，backend 用 8000/8765）
    local p
    for p in 80 443 ${REST_PORT} ${WS_PORT}; do
        if ! check_port_free "$p"; then
            warn "端口 ${p} 已被占用，docker compose 启动可能失败"
        fi
    done

    info "构建并启动容器..."
    DOMAIN=localhost docker compose up --build -d

    info "等待后端健康检查..."
    local i
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${REST_PORT}/health" &>/dev/null; then
            ok "后端健康检查通过"
            break
        fi
        sleep 2
        if [[ $i -eq 30 ]]; then
            warn "后端健康检查超时（可能仍在初始化）"
        fi
    done

    local wsl_ip
    wsl_ip=$(get_wsl_ip)

    echo ""
    echo "${_G}========================================${_N}"
    echo "${_G}  Docker 部署完成！${_N}"
    echo "${_G}========================================${_N}"
    echo "  前端 (本机):   http://localhost"
    echo "  REST API:      http://localhost:${REST_PORT}"
    echo "  WebSocket:     ws://localhost:${WS_PORT}"
    echo ""
    print_lan_guide "$wsl_ip" "80 ${REST_PORT} ${WS_PORT}"
    echo ""
    echo "  日志:  docker compose logs -f"
    echo "  停止:  docker compose down"
    echo "${_G}========================================${_N}"
}

stop_docker() {
    info "停止 Docker 服务..."
    docker compose down
    ok "Docker 服务已停止"
}

status_docker() {
    docker compose ps
}

# ------------------------------------------------------------------ #
# 原生路径（无 Docker）
# ------------------------------------------------------------------ #

deploy_native() {
    info "未检测到 Docker，使用原生部署..."

    local py
    if ! py=$(detect_python); then
        err "需要 Python >= 3.10。请安装: sudo apt update && sudo apt install -y python3 python3-pip python3-venv"
        return 1
    fi
    ok "Python: $($py --version)"

    if ! detect_node; then
        err "需要 Node.js >= 18。请安装: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install -y nodejs"
        return 1
    fi
    ok "Node.js: $(node -v)"

    # 端口检查
    local p
    for p in ${REST_PORT} ${WS_PORT} ${WEB_PORT}; do
        if ! check_port_free "$p"; then
            err "端口 ${p} 已被占用，请先释放或修改 deploy-wsl.sh 中的端口常量"
            return 1
        fi
    done

    # 安装后端依赖
    info "安装后端 Python 依赖..."
    $py -m pip install --quiet -e ".[web,parallel,jit]" 2>&1 | tail -3 || {
        warn "pip install 部分失败，尝试仅安装核心 web 依赖..."
        $py -m pip install --quiet -e ".[web]" || {
            err "后端依赖安装失败"
            return 1
        }
    }
    ok "后端依赖安装完成"

    # 构建前端
    info "构建前端..."
    (
        cd "$SCRIPT_DIR/frontend"
        if [[ ! -d node_modules ]]; then
            info "安装 npm 依赖（首次较慢）..."
            npm install --silent 2>&1 | tail -3
        fi
        npm run build 2>&1 | tail -5
    )
    if [[ ! -f "$SCRIPT_DIR/frontend/dist/index.html" ]]; then
        err "前端构建失败，dist/index.html 不存在"
        return 1
    fi
    ok "前端构建完成"

    # 创建 PID 目录
    mkdir -p "$PID_DIR"

    # 启动后端（REST + WebSocket + model loop）
    info "启动后端服务 (REST :${REST_PORT} + WebSocket :${WS_PORT})..."
    PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR" \
        nohup "$py" "$SCRIPT_DIR/experiments/run_visualization_demo.py" \
        --steps 100000 \
        --ws_host 0.0.0.0 \
        --rest_host 0.0.0.0 \
        --ws_port "${WS_PORT}" \
        --rest_port "${REST_PORT}" \
        > "$PID_DIR/backend.log" 2>&1 &
    local backend_pid=$!
    echo "$backend_pid" > "$PID_DIR/backend.pid"
    info "后端 PID: $backend_pid (日志: .wsl-pids/backend.log)"

    # 等待后端就绪
    info "等待后端就绪..."
    local i
    for i in $(seq 1 20); do
        if curl -sf "http://localhost:${REST_PORT}/health" &>/dev/null; then
            ok "后端健康检查通过"
            break
        fi
        # 检查进程是否还活着
        if ! kill -0 "$backend_pid" 2>/dev/null; then
            err "后端进程已退出，查看日志: cat .wsl-pids/backend.log"
            tail -20 "$PID_DIR/backend.log" >&2 || true
            return 1
        fi
        sleep 2
        if [[ $i -eq 20 ]]; then
            warn "后端健康检查超时（可能仍在初始化量子/JIT 模块）"
            info "查看日志: tail -f .wsl-pids/backend.log"
        fi
    done

    # 启动前端静态服务
    info "启动前端静态服务 (HTTP :${WEB_PORT})..."
    (
        cd "$SCRIPT_DIR/frontend/dist"
        nohup "$py" -m http.server "${WEB_PORT}" --bind 0.0.0.0 \
            > "$PID_DIR/frontend.log" 2>&1 &
        echo $! > "$PID_DIR/frontend.pid"
    )
    ok "前端 PID: $(cat "$PID_DIR/frontend.pid") (日志: .wsl-pids/frontend.log)"

    local wsl_ip
    wsl_ip=$(get_wsl_ip)

    echo ""
    echo "${_G}========================================${_N}"
    echo "${_G}  原生部署完成！${_N}"
    echo "${_G}========================================${_N}"
    echo "  前端 (本机):   http://localhost:${WEB_PORT}"
    echo "  REST API:      http://localhost:${REST_PORT}"
    echo "  WebSocket:     ws://localhost:${WS_PORT}"
    echo ""
    print_lan_guide "$wsl_ip" "${WEB_PORT} ${REST_PORT} ${WS_PORT}"
    echo ""
    echo "  日志:  tail -f .wsl-pids/backend.log  或  .wsl-pids/frontend.log"
    echo "  停止:  ./deploy-wsl.sh stop"
    echo "${_G}========================================${_N}"
}

stop_native() {
    info "停止原生服务..."
    local stopped=0
    for name in backend frontend; do
        local pidfile="$PID_DIR/${name}.pid"
        if [[ -f "$pidfile" ]]; then
            local pid
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                sleep 1
                kill -9 "$pid" 2>/dev/null || true
                ok "已停止 ${name} (PID: $pid)"
                stopped=1
            fi
            rm -f "$pidfile"
        fi
    done
    if [[ $stopped -eq 0 ]]; then
        warn "没有找到运行中的原生服务"
    fi
}

status_native() {
    local any_running=0
    for name in backend frontend; do
        local pidfile="$PID_DIR/${name}.pid"
        if [[ -f "$pidfile" ]]; then
            local pid
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                printf "  ${_G}●${_N} %-10s PID:%-8s 端口:" "$name" "$pid"
                case $name in
                    backend)  echo " ${REST_PORT}(REST) ${WS_PORT}(WS)";;
                    frontend) echo " ${WEB_PORT}(HTTP)";;
                esac
                any_running=1
            else
                printf "  ${_R}●${_N} %-10s PID:%-8s (已退出)\n" "$name" "$pid"
            fi
        fi
    done
    if [[ $any_running -eq 0 ]]; then
        echo "  没有运行中的原生服务"
    fi
}

show_logs_native() {
    local name=${1:-backend}
    local logfile="$PID_DIR/${name}.log"
    if [[ -f "$logfile" ]]; then
        info "实时日志: $logfile (Ctrl+C 退出)"
        tail -f "$logfile"
    else
        err "日志文件不存在: $logfile"
        echo "可用日志:"
        ls -1 "$PID_DIR"/*.log 2>/dev/null || echo "  (无)"
    fi
}

# ------------------------------------------------------------------ #
# 局域网访问指引
# ------------------------------------------------------------------ #

print_lan_guide() {
    local wsl_ip=$1
    shift
    local ports=("$@")

    echo "${_B}--- 局域网访问指引 ---${_N}"
    echo ""
    echo "  WSL IP: ${_Y}${wsl_ip}${_N}"
    echo ""
    echo "  ${_Y}方案 A: Win11 22H2+ 镜像网络（推荐，免转发）${_N}"
    echo "  在 Windows 用户目录创建/编辑 %USERPROFILE%\.wslconfig，加入:"
    echo "    [wsl2]"
    echo "    networkingMode=mirrored"
    echo "  然后 WSL 重启 (wsl --shutdown)，局域网设备直接访问 Windows IP 的端口即可。"
    echo ""
    echo "  ${_Y}方案 B: 端口转发（兼容所有 WSL2 版本）${_N}"
    echo "  在 Windows PowerShell (管理员) 执行以下命令，把 Windows 主机端口"
    echo "  转发到 WSL IP ${wsl_ip}:"
    echo ""
    local p
    for p in "${ports[@]}"; do
        echo "    netsh interface portproxy add v4tov4 listenport=${p} listenaddress=0.0.0.0 connectport=${p} connectaddress=${wsl_ip}"
    done
    echo ""
    echo "  然后开放 Windows 防火墙:"
    echo "    New-NetFirewallRule -DisplayName 'ZeroDataModel' -Direction Inbound -LocalPort $(
        IFS=,; echo "${ports[*]}"
    ) -Protocol TCP -Action Allow"
    echo ""
    echo "  局域网设备访问: http://<Windows主机IP>:$( [[ " ${ports[*]} " =~ 80 ]] && echo 80 || echo "${ports[0]}" )"
    echo "  查看已设置的转发: netsh interface portproxy show all"
    echo "  清除转发: netsh interface portproxy reset"
}

# ------------------------------------------------------------------ #
# 主逻辑
# ------------------------------------------------------------------ #

main() {
    echo "${_B}========================================${_N}"
    echo "${_B}  ZeroDataModel — WSL 部署${_N}"
    echo "${_B}========================================${_N}"
    echo ""

    # 环境检测
    if detect_wsl; then
        ok "运行环境: WSL2"
    else
        warn "未检测到 WSL 环境（/proc/version 无 microsoft 标记）"
        warn "本脚本为 WSL 设计，继续执行可能部分功能不可用..."
    fi

    local wsl_ip
    wsl_ip=$(get_wsl_ip)
    info "WSL IP: ${wsl_ip}"
    echo ""

    # 子命令处理
    local cmd=${1:-deploy}
    case "$cmd" in
        stop)
            # 优先停止 Docker，再停止原生
            if detect_docker && docker compose ls 2>/dev/null | grep -q "$(basename "$SCRIPT_DIR")"; then
                stop_docker
            fi
            stop_native
            exit 0
            ;;
        status)
            if detect_docker; then
                info "Docker 服务:"
                status_docker 2>/dev/null || true
                echo ""
            fi
            info "原生服务:"
            status_native
            exit 0
            ;;
        logs)
            shift
            if detect_docker && docker compose ls 2>/dev/null | grep -q "$(basename "$SCRIPT_DIR")"; then
                docker compose logs -f "$@"
            else
                show_logs_native "${1:-backend}"
            fi
            exit 0
            ;;
        deploy|"")
            ;;
        *)
            err "未知命令: $cmd"
            echo "用法: ./deploy-wsl.sh [deploy|stop|status|logs]"
            exit 1
            ;;
    esac

    # 选路径: Docker 优先
    if detect_docker; then
        deploy_docker
    else
        warn "未检测到 Docker"
        info "降级到原生部署 (Python + Node.js)..."
        echo ""
        deploy_native
    fi
}

main "$@"
