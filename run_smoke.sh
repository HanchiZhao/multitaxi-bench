#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${MULTITAXI_PYTHON:-$ROOT/.venv/bin/python}"
SMOKE="$ROOT/_smoke_workspace"
rm -rf "$SMOKE"
mkdir -p "$SMOKE"
cp -R "$ROOT/scripts" "$SMOKE/scripts"
cp -R "$ROOT/configs" "$SMOKE/configs"
cp -R "$ROOT/provenance" "$SMOKE/provenance"
cd "$SMOKE"
"$PY" scripts/generate_mock_data.py --trips-per-month 1200 --force
"$PY" scripts/run_v52.py --config configs/smoke.yaml --skip-download
echo "SAFE SMOKE TEST PASSED; real data were never touched."
