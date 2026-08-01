#!/usr/bin/env bash
#
# RunPod 上启停服务。放在仓库根目录，在 Pod 的终端里跑：
#
#   bash run_pod.sh              # 重启服务（先杀旧进程再起，最常用）
#   bash run_pod.sh stop         # 停掉
#   bash run_pod.sh status       # 看进程 / 端口 / GPU 是否生效
#   bash run_pod.sh logs         # 跟踪日志
#   PORT=8080 bash run_pod.sh    # 换端口
#
# 用 nohup 起，SSH 断开或 Web Terminal 关掉都不会带走进程。
# -u 是关键：不加的话 Python 输出会被缓冲，日志里长时间看不到东西，
# 容易误以为服务卡住了。
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${PORT:-8000}"
LOG="${LOG:-/workspace/cardface.log}"
PATTERN="server/app.py"          # pkill 用的匹配串，别改动得和启动命令不一致

# 顺带把可调参数集中在这里，改完重跑本脚本即可生效（默认值见各模块注释）
export MAX_USERS="${MAX_USERS:-20}"
export FACE_MIN_SHARP="${FACE_MIN_SHARP:-20}"   # 设 0 可关掉人脸清晰度闸门

log()  { printf '\033[36m%s\033[0m\n' "$*"; }
ok()   { printf '\033[32m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
err()  { printf '\033[31m%s\033[0m\n' "$*"; }

pod_url() {
  # RunPod 会注入 RUNPOD_POD_ID；拿不到就留空，让调用方自己看 Connect 页
  if [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net"
  fi
}

do_stop() {
  if pkill -f "$PATTERN" 2>/dev/null; then
    log "已发送停止信号，等待退出…"
    for _ in $(seq 1 20); do
      pgrep -f "$PATTERN" >/dev/null 2>&1 || break
      sleep 0.5
    done
    if pgrep -f "$PATTERN" >/dev/null 2>&1; then
      warn "还没退出，强制 kill -9"
      pkill -9 -f "$PATTERN" 2>/dev/null || true
      sleep 1
    fi
    ok "已停止"
  else
    log "没有在跑的进程"
  fi
}

do_status() {
  echo
  if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    ok "进程在跑: PID $(pgrep -f "$PATTERN" | tr '\n' ' ')"
  else
    err "进程没在跑"
  fi

  local h
  h=$(curl -s -m 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)
  if [ -n "$h" ]; then
    ok "本地 /health 正常"
    echo "  $h"
    # 这一行才是 GPU 是否真的生效的判据 —— 不要用 get_available_providers()，
    # 它列的是编译时支持，即使运行时加载失败也照样打印 CUDA。
    if echo "$h" | grep -q '"gpu": *true\|"gpu":true'; then
      ok "GPU 生效 (CUDAExecutionProvider)"
    else
      err "跑在 CPU 上！单帧会慢约 3 倍"
      echo "  常见原因：装了 CUDA 13 版的 onnxruntime-gpu 但镜像是 CUDA 12"
      echo "  修复： pip uninstall -y onnxruntime-gpu && pip install \"onnxruntime-gpu<1.27\""
      echo "  （反之 nvidia-smi 显示 CUDA 13.x 就装 >=1.27）"
    fi
  else
    err "本地 /health 无响应"
    echo "  首次启动要等 CUDA 初始化 + 模型预热，实测约 100 秒，属正常"
    echo "  看日志： bash run_pod.sh logs"
  fi

  local u
  u=$(pod_url)
  [ -n "$u" ] && { echo; log "手机访问: ${u}/"; log "WS 地址:  ${u/https/wss}/ws"; }
  echo
}

do_start() {
  [ -f cardpose.onnx ] || { err "找不到 cardpose.onnx（39MB，走 Git LFS 或单独传）"; exit 1; }

  if ! python -c "import fastapi, uvicorn, cv2, onnxruntime" 2>/dev/null; then
    warn "依赖不全，安装 GPU 版…"
    pip install -q -r server/requirements-gpu.txt || {
      err "依赖安装失败"; exit 1; }
  fi

  do_stop

  log "启动 (port=${PORT}, MAX_USERS=${MAX_USERS}, FACE_MIN_SHARP=${FACE_MIN_SHARP})"
  mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
  nohup python -u server/app.py --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
  log "PID $! ，日志 $LOG"

  # 等就绪。uvicorn 是 lifespan 跑完才开始监听，所以这段时间端口是不通的
  log "等待就绪（冷启动约 100 秒）…"
  for i in $(seq 1 90); do
    if curl -s -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      ok "就绪，耗时约 $((i*2)) 秒"
      do_status
      return 0
    fi
    if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
      err "进程已退出，日志末尾："
      tail -n 30 "$LOG" 2>/dev/null
      exit 1
    fi
    sleep 2
  done
  err "180 秒还没就绪，日志末尾："
  tail -n 30 "$LOG" 2>/dev/null
  exit 1
}

case "${1:-start}" in
  start|restart|"") do_start ;;
  stop)            do_stop ;;
  status)          do_status ;;
  logs)            tail -f "$LOG" ;;
  *)               echo "用法: bash run_pod.sh [start|stop|status|logs]"; exit 1 ;;
esac
