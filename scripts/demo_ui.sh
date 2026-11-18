#!/usr/bin/env bash
# Serve the PathSnap UI (Laptop 2) and open it in the browser.
# Usage: scripts/demo_ui.sh [API_URL]      e.g. scripts/demo_ui.sh http://10.20.201.72:8000
set -e
cd "$(dirname "$0")/../ui"
PORT="${PORT:-5500}"
API="${1:-http://localhost:8000}"
URL="http://localhost:$PORT/?api=$API"
echo "PathSnap UI | $URL"
( sleep 1; (open "$URL" || xdg-open "$URL") >/dev/null 2>&1 || true ) &
exec python3 -m http.server "$PORT" --bind 127.0.0.1
