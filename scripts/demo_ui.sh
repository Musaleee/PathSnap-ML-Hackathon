#!/usr/bin/env bash
# Serve the PathSnap UI (Laptop 2) at http://localhost:5500 and open it in the browser.
set -e
cd "$(dirname "$0")/../ui"
PORT="${PORT:-5500}"
echo "PathSnap UI | http://localhost:$PORT  (API box: http://localhost:8000 or http://<laptop-1-ip>:8000)"
( sleep 1; open "http://localhost:$PORT" 2>/dev/null || true ) &
exec python3 -m http.server "$PORT" --bind 127.0.0.1
