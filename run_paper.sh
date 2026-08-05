#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
"$PY" scripts/verify_data.py --strict-lock
"$PY" scripts/run_v51.py --config configs/paper_main.yaml --skip-download
