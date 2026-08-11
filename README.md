# MultiTaxi-Bench 5.2

MultiTaxi-Bench evaluates sequential taxi repositioning policies and two complementary
Shapley explanations on NYC TLC Yellow Taxi data. Version 5.2 is the final
date-boundary-corrected paper pipeline and supersedes the v5.1 formal results.

## Final paper task

One empty driver starts at JFK Airport (Taxi Zone 132) at 08:00 and operates for a fixed
120-minute horizon. At each decision point, the policy chooses **WAIT** or empty
**REPOSITION**. Passenger destinations are exogenous outcomes sampled from the learned
origin-destination distribution; only completed passenger trips earn revenue. All
waiting, empty, occupied, unfinished-trip, and terminal time must sum to exactly 120
minutes.

The primary objective is two-hour **operating net earnings** under the fare-share model:

```text
0.6979 × modeled fare-and-tip receipts
− 0.25 USD × occupied miles
− 0.25 USD × empty miles
```

The 0.3021 fare-share rate is the complement of the retained-receipts coefficient and is
not deducted a second time. Owner-operator, fixed-lease, and V4-legacy definitions are
reported as post-hoc sensitivity models.

## Data and methods

- NYC TLC Yellow Taxi trips from January through June 2025;
- pickup timestamps restricted to each file's declared half-open calendar month;
- 21,753,862 retained trips across 181 service days;
- 134 quality-clean out-of-month pickups excluded and zero retained;
- seven policies evaluated on the same 300 scenarios (2,100 episodes total);
- Q-learning trained for 15,000 episodes and Double DQN for 12,000 episodes;
- Global Dynamic Zone Shapley for reposition-option value;
- algorithm-conditioned Path-occurrence Shapley with signed, occurrence-specific values;
- strict research-contract, accounting, reproduction, data-boundary, and Shapley checks.

## Final verified results

The accepted 2026-08-11 run produced the following primary results:

| Rank | Policy | Mean earnings (USD/2h) | Paired gain vs WAIT-only (95% CI) |
|---:|---|---:|---:|
| 1 | Finite-horizon value iteration | 72.3245 | +0.9803 [-0.3935, 2.3541] |
| 2 | WAIT-only | 71.3443 | reference |
| 3 | DQN | 71.0296 | -0.3147 [-1.8500, 1.2206] |
| 4 | Q-learning | 64.9298 | -6.4144 [-8.2801, -4.5488] |
| 5 | Greedy net earnings | 64.3991 | -6.9452 [-8.8134, -5.0770] |
| 6 | Highest demand | 58.7053 | -12.6389 [-14.7057, -10.5722] |
| 7 | Highest income | 55.3020 | -16.0423 [-18.2375, -13.8470] |

Finite-horizon value iteration has the highest sample mean, but its paired confidence
interval versus WAIT-only crosses zero. DQN is also not distinguishable from WAIT-only at
the 95% level. The result therefore does not establish statistical superiority for either
method. The four remaining repositioning policies are significantly below WAIT-only in
this fixed JFK 08:00 task.

## Quick start

Python 3.11 x64 is required. Clone the final branch explicitly:

```powershell
git clone --branch v5.2-final --single-branch https://github.com/HanchiZhao/multitaxi-bench.git
cd multitaxi-bench
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1
```

Download and verify the official six monthly TLC files:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\verify_data.py --strict-lock
```

Run the complete paper experiment:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_paper.ps1
```

The paper runner never downloads or overwrites raw data implicitly. It regenerates
`processed_data/`, `results/`, and `models/`, then executes the final hard acceptance
checks. Full CPU training can take several hours.

## Repository evidence

`paper_artifacts/final_20260811/` contains the accepted reports, key tables, raw 2,100-row
evaluation table, and publication figures. Its SHA256 manifest can be checked with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_release_artifacts.py
```

Further documentation:

- [`docs/RESEARCH_CONTRACT.md`](docs/RESEARCH_CONTRACT.md)
- [`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/V5.2_RELEASE_AUDIT.md`](docs/V5.2_RELEASE_AUDIT.md)
- [`docs/DEPENDENCY_AUDIT.md`](docs/DEPENDENCY_AUDIT.md)

The audited V4 core remains protected by `provenance/v4_core_manifest.json`. The GitHub
`main` branch remains the V4 final baseline; v5.1 history is preserved in its existing
branches, and v5.2 is published independently as `v5.2-final`.
