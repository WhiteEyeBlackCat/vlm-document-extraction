#!/usr/bin/env bash

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"

python -m uvicorn web.app:app --host "$HOST" --port "$PORT"
