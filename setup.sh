#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-dqn.txt
.venv/bin/python scripts/verify_v4_preservation.py
.venv/bin/python scripts/preflight.py --config configs/smoke.yaml
