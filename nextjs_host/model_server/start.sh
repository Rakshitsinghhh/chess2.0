#!/usr/bin/env bash
# ─── Chess2.0 FastAPI server ──────────────────────────────────────────────────
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
WORKERS="${WORKERS:-1}"

echo "Starting Chess2.0 model server on ${HOST}:${PORT} …"
exec uvicorn server:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --log-level info
