# Final verified experiment — 2026-08-07

The formal fare-share-primary run used 300 common scenarios per algorithm and passed all
machine acceptance checks.

| Rank | Algorithm | Mean operating net earnings (USD/2h) | 95% CI |
|---:|---|---:|---:|
| 1 | DQN | 72.5035 | [70.4970, 74.5099] |
| 2 | Finite-Horizon Value Iteration | 72.3656 | [70.3355, 74.3958] |
| 3 | Wait Only | 71.5437 | [69.4525, 73.6349] |
| 4 | Greedy Net Earnings | 64.8055 | [62.7607, 66.8503] |
| 5 | Q-Learning | 64.5307 | [62.5267, 66.5348] |
| 6 | Highest Demand | 58.6555 | [56.9249, 60.3860] |
| 7 | Highest Income | 55.4260 | [53.3225, 57.5294] |

Paired against Wait Only:

- DQN: +0.9598 USD/2h, 95% CI [-0.4329, 2.3525].
- Finite-Horizon Value Iteration: +0.8219 USD/2h,
  95% CI [-0.5418, 2.1857].

Both intervals cross zero. The evidence supports slightly higher sample means, not a claim
that either method is statistically superior to Wait Only.

## Validation evidence

- 7 algorithms × 300 scenarios = 2,100 episodes;
- maximum time-accounting error: 2.84e-14;
- maximum primary re-accounting gap: 4.26e-14;
- fare-share deduction count: exactly 1;
- maximum Path Efficiency gap: 4.26e-13;
- 41 negative Path Shapley occurrences retained;
- Q-learning fallback/unseen rate: 0.2661%;
- DQN trained-model action rate: 100%;
- no Global or Path hard-axiom failure.

See `paper_artifacts/` for the immutable reports, selected tables, high-resolution bars,
and local-detail Taxi Zone maps from the accepted run.

