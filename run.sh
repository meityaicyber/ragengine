#!/usr/bin/env sh
# Thin wrapper: ./run.sh <command> [args]   (same commands as python run.py)
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
exec "$PY" "$DIR/run.py" "$@"
