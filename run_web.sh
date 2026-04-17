#!/usr/bin/env bash

set -euo pipefail

DEFAULT_LAYOUT_MODEL="models/doclayout_yolo.pt"
DEFAULT_CONDA_ENV="CV"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
PADDLEX_HOME="${PADDLEX_HOME:-/tmp/paddlex}"
PADDLE_HOME="${PADDLE_HOME:-/tmp/paddle}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics}"
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"

if [[ -z "${DOCLAYOUT_YOLO_MODEL:-}" && -f "$DEFAULT_LAYOUT_MODEL" ]]; then
  export DOCLAYOUT_YOLO_MODEL="$DEFAULT_LAYOUT_MODEL"
fi

mkdir -p "$PADDLEX_HOME" "$PADDLE_HOME" "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$DEFAULT_CONDA_ENV"; then
  exec conda run --no-capture-output -n "$DEFAULT_CONDA_ENV" env \
    DOCLAYOUT_YOLO_MODEL="${DOCLAYOUT_YOLO_MODEL:-}" \
    PADDLEX_HOME="$PADDLEX_HOME" \
    PADDLE_HOME="$PADDLE_HOME" \
    MPLCONFIGDIR="$MPLCONFIGDIR" \
    YOLO_CONFIG_DIR="$YOLO_CONFIG_DIR" \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="$PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK" \
    python -m uvicorn web.app:app --host "$HOST" --port "$PORT" --timeout-graceful-shutdown 3
else
  exec env \
    DOCLAYOUT_YOLO_MODEL="${DOCLAYOUT_YOLO_MODEL:-}" \
    PADDLEX_HOME="$PADDLEX_HOME" \
    PADDLE_HOME="$PADDLE_HOME" \
    MPLCONFIGDIR="$MPLCONFIGDIR" \
    YOLO_CONFIG_DIR="$YOLO_CONFIG_DIR" \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="$PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK" \
    python -m uvicorn web.app:app --host "$HOST" --port "$PORT" --timeout-graceful-shutdown 3
fi
