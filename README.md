# MultiTaxi-Bench 5.1

MultiTaxi-Bench evaluates sequential taxi repositioning policies and two complementary
Shapley explanations on six months of NYC TLC Yellow Taxi data.

## Research contract

A single empty driver starts at any NYC Taxi Zone and clock time. During a fixed
120-minute horizon, the driver repeatedly chooses **WAIT** or empty **REPOSITION**.
Passenger destinations are sampled exogenously from the learned OD distribution. Only
completed trips earn revenue, and all waiting, empty, occupied, unfinished-trip, and
terminal time must account for exactly 120 minutes.

The primary objective is **two-hour operating net earnings**:

```text
0.6979 × modeled fare-and-tip receipts
− 0.25 USD × occupied miles
− 0.25 USD × empty miles
```

The fare-share deduction is embodied in the 0.6979 retained-receipts coefficient and is
never subtracted a second time.

## Methods retained in the final pipeline

- Hierarchical empirical-Bayes OD estimation and duration calibration;
- latent vacant-taxi competition and a common scenario bank;
- Wait Only, Highest Demand, Highest Income, Greedy Net Earnings,
  finite-horizon value iteration, Q-learning, and Double DQN;
- paired policy evaluation over common scenarios;
- Global Dynamic Zone Shapley for reposition-option value;
- algorithm-conditioned Path-occurrence Shapley for one representative trajectory per
  policy, with signed contributions and occurrence identity preserved;
- imputation, ablation, baseline, competition, axiom, accounting, and reproduction checks.

## Quick start

Python 3.11 is required.

### Windows PowerShell

```powershell
git clone https://github.com/HanchiZhao/multitaxi-bench.git
cd multitaxi-bench
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1
```

Download the official January-June 2025 TLC trip files and verify the frozen lock:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\verify_data.py --strict-lock
```

Run the full 300-scenario paper experiment:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_paper.ps1
```

The paper run never downloads or overwrites raw data implicitly. It regenerates runtime
outputs under `processed_data/`, `results/`, and `models/`, then executes reproduction and
final acceptance checks.

## Final verified result

The 2026-08-07 formal run passed all research-contract, accounting, reproduction, Global
Shapley, and Path Shapley hard checks. DQN had the highest mean operating net earnings at
72.5035 USD/2h; finite-horizon value iteration had 72.3656 USD/2h; Wait Only had
71.5437 USD/2h. The DQN and value-iteration paired 95% confidence intervals versus Wait
Only both crossed zero, so the result supports slightly higher means, not demonstrated
statistical superiority.

Curated final reports, tables, and figures are under `paper_artifacts/`. Full generated
outputs remain reproducible but are not committed.

## Documentation

- [`docs/RESEARCH_CONTRACT.md`](docs/RESEARCH_CONTRACT.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md)
- [`docs/DEPENDENCY_AUDIT.md`](docs/DEPENDENCY_AUDIT.md)

The audited V4 core is protected by `provenance/v4_core_manifest.json`; the final
repository does not keep a duplicate source snapshot.
