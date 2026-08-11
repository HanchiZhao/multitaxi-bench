# Final verified experiment — 2026-08-11

MultiTaxi-Bench v5.2 is the accepted, date-boundary-corrected paper run. The machine
acceptance status is `FINAL EXPERIMENT ACCEPTANCE PASSED` with no failed check.

## Frozen evaluation contract

- start state: one empty driver at JFK Airport, Taxi Zone 132;
- start time: 08:00;
- horizon: 120 minutes with 5-minute WAIT re-evaluation;
- primary metric: completed-trip operating net earnings under the fare-share model;
- evaluation: seven policies on an identical bank of 300 scenarios each;
- training: Q-learning 15,000 episodes; Double DQN 12,000 episodes;
- Global Shapley: ten reposition candidates, exact 1,024-coalition evaluation;
- Path Shapley: one representative path per policy with unique occurrence identities.

## Primary policy results

| Rank | Policy | Mean (USD/2h) | Mean paired gain vs WAIT-only | Paired 95% CI |
|---:|---|---:|---:|---:|
| 1 | Finite-horizon value iteration | 72.3245 | +0.9803 | [-0.3935, 2.3541] |
| 2 | WAIT-only | 71.3443 | reference | reference |
| 3 | DQN | 71.0296 | -0.3147 | [-1.8500, 1.2206] |
| 4 | Q-learning | 64.9298 | -6.4144 | [-8.2801, -4.5488] |
| 5 | Greedy net earnings | 64.3991 | -6.9452 | [-8.8134, -5.0770] |
| 6 | Highest demand | 58.7053 | -12.6389 | [-14.7057, -10.5722] |
| 7 | Highest income | 55.3020 | -16.0423 | [-18.2375, -13.8470] |

The value-iteration and DQN intervals cross zero. Their observed means therefore do not
demonstrate superiority over WAIT-only at the 95% level. Q-learning, Greedy net earnings,
Highest demand, and Highest income have intervals entirely below zero.

The operational mechanism is consistent with the fixed airport task: mean empty time is
11.06 minutes for value iteration, compared with 25.14 for Q-learning, 25.91 for Greedy,
38.77 for Highest demand, and 41.54 for Highest income. WAIT-only has no empty reposition
time and completes the most passenger trips on average (3.07).

## Data-boundary repair

The v5.1 run used 194 rather than 181 service days because 134 quality-clean pickups fell
outside the month declared by their source files. Version 5.2 applies
`month_start <= pickup < next_month_start`, does not constrain dropoff timestamps, and
uses the same boundary for environment construction and fare-share calibration.

| Boundary check | Final value |
|---|---:|
| Retained cleaned trips | 21,753,862 |
| Service days | 181 |
| Clean out-of-month pickups excluded | 134 |
| Out-of-month pickups retained | 0 |
| Dynamic environment schema | `4.1-date-boundary-fix` |

## Shapley results

The Global Dynamic Zone game assigns 0.71130 USD/2h of total reposition-option value.
Penn Station/Madison Sq West (Zone 186) is first with 0.41484 USD/2h. Removing the five
highest-Shapley zones reduces the full reposition-option value by 86.2%.

The algorithm-conditioned Path-occurrence analysis contains 64 unique occurrences across
seven representative trajectories. Forty occurrences have negative signed values; these
are retained rather than clipped. Global Efficiency error is 3.33e-16, all seven Path
Efficiency checks pass, and no hard Global or Path axiom fails.

## Acceptance evidence

- 2,100 evaluation rows and exactly 300 scenarios for every policy;
- identical scenario identifiers across all seven policies;
- maximum time-accounting error: 4.26e-14 minutes;
- Q-learning table-action rate: 99.8865%;
- DQN trained-model action rate: 100%;
- configured retained-receipts share: 0.6979;
- independently derived retained-receipts share: 0.6979058177;
- final failed-check list: empty.

The canonical evidence is stored in `paper_artifacts/final_20260811/` and protected by
`SHA256SUMS.csv`. The v5.1 evidence remains available through Git history and the preserved
v5.1 branch, but it is not the final paper result.
