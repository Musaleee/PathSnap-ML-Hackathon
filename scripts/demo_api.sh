#!/usr/bin/env bash
# Start the PathSnap API (Laptop 1). Keep this terminal visible: it prints one log line per request.
# Usage: scripts/demo_api.sh [trips_folder]      (default: ../evaluator_data/evaluator_test)
# HOST=0.0.0.0 scripts/demo_api.sh ...           to accept requests from another laptop
set -e
cd "$(dirname "$0")/.."
DATA="${1:-../evaluator_data/evaluator_test}"
if [ ! -d "$DATA" ]; then echo "Trips folder not found: $DATA"; exit 1; fi
export PATHSNAP_DATA_DIR="$(cd "$DATA" && pwd)"
export PYTHONWARNINGS=ignore
echo "PathSnap API | data: $PATHSNAP_DATA_DIR | http://${HOST:-127.0.0.1}:${PORT:-8000}"
exec .venv/bin/uvicorn api.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
