# Reproducibility

## Environment

- Python 3.11 x64;
- pinned core dependencies in `requirements.txt`;
- pinned CPU PyTorch in `requirements-dqn.txt`;
- deterministic CPU mode in the formal paper configuration.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Setup creates `.venv`, installs
the pinned dependencies, verifies the audited V4 core manifest, and runs preflight checks.

## Data

The experiment uses NYC TLC Yellow Taxi records from January through June 2025. Raw
Parquet files are not committed. `data_manifest.json` records official download URLs and
`data_lock.json` records accepted SHA256 values and sizes for every raw month and Taxi Zone
asset.

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\verify_data.py --strict-lock
```

Strict verification does not rewrite data. Version 5.2 additionally checks the declared
pickup-month boundary: 181 service days, 134 clean out-of-month pickups excluded, and zero
retained. Dropoff timestamps are not month-restricted.

## Smoke run

`run_smoke.ps1` or `run_smoke.sh` creates an isolated `_smoke_workspace/`, generates mock
data there, and runs the v5.2 orchestration with reduced settings. It never writes mock
files into the official `data/` directory.

## Formal paper run

`run_paper.ps1` and `run_paper.sh` execute:

1. strict frozen-data verification;
2. audited V4 core and static research-contract checks;
3. v5.2 environment construction and pickup-month boundary validation;
4. Q-learning and Double DQN training;
5. 300 common-scenario evaluation for all seven policies;
6. Global Dynamic Zone Shapley and validation;
7. cost re-accounting and accounting-identity audit;
8. algorithm-conditioned Path-occurrence Shapley and path validation;
9. sensitivity analyses and canonical figures;
10. reproduction verification and final hard acceptance.

The formal configuration is fixed at Taxi Zone 132, 08:00, 120 minutes, 300 scenarios,
medium competition, 15,000 Q-learning episodes, 12,000 DQN episodes, ten Global Shapley
players, and 512 configured permutations for each Shapley layer. Exact games are enumerated
when their player count permits it.

`processed_data/resolved_v52_config.yaml` records the active objective and effective
download behavior. The accepted copy is also stored under
`paper_artifacts/final_20260811/reports/`.

## Expected terminal states

```text
DATA VERIFICATION PASSED
V4 CORE WITH AUTHORIZED V5.2 DATE FIX PASSED
RESEARCH CONTRACT PASSED
V5.2 DATE BOUNDARY VALIDATION PASSED
REPRODUCTION PASSED
MULTITAXI-BENCH V5.2 PIPELINE COMPLETED SUCCESSFULLY
FINAL EXPERIMENT ACCEPTANCE PASSED
V5.2 PAPER PIPELINE AND ACCEPTANCE PASSED
```

## Verify the committed evidence

This command hashes every curated artifact and independently checks the accepted status,
date boundary, seven-policy set, 2,100 evaluation rows, and common scenario bank:

```powershell
.\.venv\Scripts\python.exe scripts\verify_release_artifacts.py
```

Runtime directories (`processed_data/`, `results/`, and `models/`) are intentionally
ignored. The committed evidence is review material; the canonical reproduction path is
still strict data verification followed by `run_paper`.
