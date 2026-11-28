#!/usr/bin/env bash
# ExhibitFlow GPU Worker — One-Command Vast.ai Launcher
#
# Usage:
#   HF_TOKEN=hf_xxx bash start_vast.sh
#
# Optional env vars:
#   PORT=8001  MODE=full|tracking_seg|tracking  LOCAL=1
#   TUNNEL=quick|named|none  CLOUDFLARE_TUNNEL_TOKEN=...  PUBLIC_GPU_URL=...
#   DATA_DIR=/workspace/gpu_data  HF_HOME=/workspace/.hf_home  GPU_TOKEN=...
set -Eeuo pipefail
trap 'echo "[vast] ERROR at line $LINENO (cmd: $BASH_COMMAND)" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f /venv/main/bin/activate ]; then
    # shellcheck disable=SC1091
    . /venv/main/bin/activate
fi
PY="$(command -v python3)"

PORT="${PORT:-8001}"
MODE="${MODE:-full}"
TUNNEL="${TUNNEL:-quick}"
LOCAL="${LOCAL:-}"
DATA_DIR="${DATA_DIR:-/workspace/gpu_data}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/.make/worker.log}"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export EF_WEIGHTS_DIR="${EF_WEIGHTS_DIR:-$SCRIPT_DIR/weights}"
export EXHIBITFLOW_WORK="${EXHIBITFLOW_WORK:-$SCRIPT_DIR/.make/work}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[vast] Checking GPU & idempotent setup (deps, externals, weights, llama, mapmodels)..."
make setup PY="$PY"

mkdir -p "$DATA_DIR" "$(dirname "$LOG_FILE")"
if [ -f "$SCRIPT_DIR/.make/worker.pid" ] && kill -0 "$(cat "$SCRIPT_DIR/.make/worker.pid")" 2>/dev/null; then
    echo "[vast] Stopping previous background worker..."
    make stop PY="$PY"
fi

ARGS=(
    launch.py
    --host 0.0.0.0
    --port "$PORT"
    --mode "$MODE"
    --tunnel "$TUNNEL"
    --data-dir "$DATA_DIR"
    --log-file "$LOG_FILE"
)
if [ -n "$LOCAL" ]; then
    ARGS+=(--local)
fi

echo "[vast] Launching GPU worker in foreground (Ctrl+C to stop)..."
exec "$PY" "${ARGS[@]}" "$@"
