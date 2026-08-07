# Research contract

## Decision problem

Given any NYC Taxi Zone and any start time from 00:00 through 24:00, one empty driver
operates for a fixed 120-minute horizon. At every decision point, the policy may:

1. `WAIT`, after which the environment is re-evaluated; or
2. empty `REPOSITION` to an available Taxi Zone.

Passenger destinations are exogenous outcomes sampled from the learned historical OD
distribution. After a completed occupied trip, the driver continues from its destination.
Revenue is credited only to completed trips.

The accounting identity is:

```text
waiting
+ empty reposition
+ completed occupied
+ unfinished occupied within horizon
+ terminal unused
= 120 minutes
```

JFK Taxi Zone 132 at 08:00 is the main paper scenario, not a hard-coded objective.

## Objective and cost models

The primary metric is **two-hour operating net earnings**. The main fare-share model uses
a 0.6979 retained share of modeled fare-and-tip receipts and 0.25 USD/mile for occupied
and empty movement. `fare_share_rate=0.3021` is a disclosure identity, not a second cost.

Owner-operator, fixed-lease, and V4-legacy definitions are post-hoc sensitivity models.
A fixed hourly lease cannot be injected as the active policy objective because the audited
V4 core does not propagate that term through every decision value.

## Two Shapley games

### Global Dynamic Zone Shapley

- Player: a Taxi Zone available as an active empty-reposition option.
- Empty coalition: WAIT-only, with no active reposition option.
- Passenger destinations remain unrestricted and exogenous.
- Value: optimal expected operating net earnings under the restricted action set.

### Algorithm-conditioned Path-occurrence Shapley

- Player: one node/event occurrence in one representative algorithm trajectory.
- Identity: algorithm + scenario + path position; repeated visits to one zone are distinct.
- Value: replayed path earnings after retaining a coalition of occurrences and repairing
  gaps with the audited route model.
- Contributions are signed. Negative values are valid and are never clipped to zero.

The two games answer different questions and must not be merged.

