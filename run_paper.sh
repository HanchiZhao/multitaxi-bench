#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Virtual environment not found. Run ./setup.sh first." >&2
  exit 1
fi
"$PY" scripts/verify_data.py --strict-lock
"$PY" scripts/run_v51.py --config configs/paper_main.yaml --skip-download
"$PY" scripts/final_experiment_acceptance.py --config processed_data/resolved_v51_config.yaml
echo "PAPER PIPELINE AND ACCEPTANCE PASSED"
