#!/usr/bin/env bash
#
# dev.sh -- run the Ragsody backend (API + MCP) and the Vite frontend
# together from a single terminal. Ctrl+C stops both.
#
# Usage:
#   ./dev.sh
#   CLIENT_DIR=/path/to/ragsody_client ./dev.sh
#   SERVER_PYTHON=/path/to/venv/bin/python ./dev.sh
#
# The script locates the backend virtualenv in its own directory and
# auto-detects the frontend as a sibling folder named `client` or
# `ragsody_client`. Override with CLIENT_DIR if your layout differs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- locate frontend -----------------------------------------------------
CLIENT_DIR="${CLIENT_DIR:-}"
if [[ -z "$CLIENT_DIR" ]]; then
  for candidate in "$ROOT_DIR/../client" "$ROOT_DIR/../ragsody_client"; do
    if [[ -d "$candidate" ]]; then
      CLIENT_DIR="$(cd "$candidate" && pwd)"
      break
    fi
  done
fi
if [[ -z "$CLIENT_DIR" ]]; then
  echo "error: frontend directory not found next to '$ROOT_DIR'." >&2
  echo "       pass it explicitly, e.g. CLIENT_DIR=/path/to/ragsody_client ./dev.sh" >&2
  exit 1
fi

# ---- locate backend python ------------------------------------------------
SERVER_PYTHON="${SERVER_PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$SERVER_PYTHON" ]]; then
  echo "error: backend python not found at '$SERVER_PYTHON'." >&2
  echo "       create it first: cd '$ROOT_DIR' && python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  echo "       or point SERVER_PYTHON at an existing interpreter." >&2
  exit 1
fi

# ---- locate package manager -----------------------------------------------
if command -v pnpm >/dev/null 2>&1; then
  FRONTEND_CMD=(pnpm dev --host)
elif command -v npm >/dev/null 2>&1; then
  FRONTEND_CMD=(npm run dev -- --host)
else
  echo "error: neither pnpm nor npm found in PATH." >&2
  exit 1
fi

BACKEND_PID=""
FRONTEND_PID=""

shutdown() {
  echo ""
  echo "Stopping backend and frontend..."
  trap - INT TERM
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
  kill 0 2>/dev/null || true
}
trap shutdown INT TERM

echo "Ragsody dev"
echo "  backend  : $ROOT_DIR   ($SERVER_PYTHON -m src.main server)"
echo "  frontend : $CLIENT_DIR (${FRONTEND_CMD[*]})"
echo ""
echo "Press Ctrl+C to stop both."
echo ""

(
  cd "$ROOT_DIR"
  exec "$SERVER_PYTHON" -m src.main server
) &
BACKEND_PID=$!

(
  cd "$CLIENT_DIR"
  exec "${FRONTEND_CMD[@]}"
) &
FRONTEND_PID=$!

# Poll instead of blocking on `wait -n`: bash defers traps while a `wait`
# is running, so a signal sent only to this script (e.g. SIGTERM) would
# never reach the trap until a child exits. The loop lets the INT/TERM
# trap fire between `sleep`s and kill the children itself.
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do
  sleep 0.5
done

# At least one process exited on its own -- stop the other if it survived.
kill 0 2>/dev/null || true
wait 2>/dev/null || true
