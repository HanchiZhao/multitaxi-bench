# Reproducibility

## Environment

- Python 3.11 x64
- pinned core dependencies in `requirements.txt`
- pinned CPU PyTorch in `requirements-dqn.txt`
- deterministic CPU mode for the paper configuration

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Setup installs dependencies,
checks the audited V4 core manifest, and runs preflight validation.

## Data

The experiment uses NYC TLC Yellow Taxi records from January through June 2025. Raw
Parquet files are not committed. `data_manifest.json` records official download URLs and
`data_lock.json` records the accepted SHA256 and size for every raw month and Taxi Zone
asset.

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\verify_data.py --strict-lock
```

Never overwrite a previously verified six-month dataset merely to repair an unrelated
code or visualization issue.

## Smoke run

`run_smoke.ps1` or `run_smoke.sh` creates `_smoke_workspace/`, generates mock data there,
and runs the same V5.1 orchestration with reduced settings. It never writes mock files into
the real `data/` directory.

## Paper run

`run_paper.ps1` and `run_paper.sh` execute:

1. strict frozen-data verification;
2. audited V4 core verification and static research-contract checks;
3. environment build and runtime contract checks;
4. Q-learning and Double DQN training;
5. 300 common-scenario policy evaluation;
6. Global Dynamic Zone Shapley and validation;
7. cost re-accounting and accounting identity audit;
8. Path-occurrence Shapley and path validation;
9. all canonical figures;
10. reproduction verification and final acceptance.

The main configuration is fixed at Taxi Zone 132, 08:00, 120 minutes, 300 scenarios,
medium competition, 15,000 Q-learning episodes, 12,000 DQN episodes, 10 Global Shapley
players, and 512 permutations for both Shapley layers.

`processed_data/resolved_v51_config.yaml` records the active objective plus effective CLI
download behavior (`skip_download_cli` and `effective_download_performed`).

## Expected terminal states

```text
DATA VERIFICATION PASSED
V4 PRESERVATION PASSED
RESEARCH CONTRACT PASSED
REPRODUCTION PASSED
MULTITAXI-BENCH V5.1 PIPELINE COMPLETED SUCCESSFULLY
FINAL EXPERIMENT ACCEPTANCE PASSED
PAPER PIPELINE AND ACCEPTANCE PASSED
```

