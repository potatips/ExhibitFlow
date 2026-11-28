#!/usr/bin/env bash
# Standalone ExhibitFlow GPU Worker setup script for Vast.ai / Ubuntu CUDA machines.
#
# Can be used as Vast.ai PROVISIONING_SCRIPT or run manually inside GPU_server/:
#   HF_TOKEN=hf_xxx START=1 bash setup.sh
set -Eeuo pipefail
trap 'echo "[setup] ERROR at line $LINENO (cmd: $BASH_COMMAND)" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-https://github.com/khangkaka066/ExhibitFlow.git}"
BRANCH="${BRANCH:-main}"
REPO_DIR="${REPO_DIR:-/workspace/ExhibitFlow}"
START="${START:-0}"
PORT="${PORT:-8001}"
MODE="${MODE:-full}"

if [ -f /venv/main/bin/activate ]; then
    # shellcheck disable=SC1091
    . /venv/main/bin/activate
fi
PY="$(command -v python3)"
echo "[setup] python=$PY port=$PORT mode=$MODE"

command -v git >/dev/null  || { apt-get update -qq && apt-get install -y -qq git; }
command -v make >/dev/null || { apt-get update -qq && apt-get install -y -qq make; }

if [ -f "$SCRIPT_DIR/Makefile" ] && [ -d "$SCRIPT_DIR/worker" ]; then
    GPU_DIR="$SCRIPT_DIR"
else
    if [ -d "$REPO_DIR/.git" ]; then
        git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
        git -C "$REPO_DIR" checkout -q -f -B "$BRANCH" FETCH_HEAD
    else
        mkdir -p "$(dirname "$REPO_DIR")"
        git clone --depth 1 -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
    fi
    GPU_DIR="$REPO_DIR/GPU_server"
fi

cd "$GPU_DIR"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[setup] warning: HF_TOKEN is unset; private HuggingFace weights will fail to download"
fi
export HF_TOKEN="${HF_TOKEN:-}"

make setup PY="$PY"
echo "[setup] Done. Start worker with: cd $GPU_DIR && bash start_vast.sh"

if [ "$START" = "1" ]; then
    PORT="$PORT" MODE="$MODE" PY="$PY" bash "$GPU_DIR/start_vast.sh"
fi
