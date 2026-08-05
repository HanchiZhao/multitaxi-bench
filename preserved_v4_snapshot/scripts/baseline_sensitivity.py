"""Baseline-action sensitivity analysis for Dynamic Zone Shapley.

Canonical Shapley uses WAIT as the only baseline action. Sensitivity variants add one
always-available reposition target selected by demand, income, or greedy net earning.
The Shapley player set itself is kept fixed for comparable rankings.
"""
from __future__ import annotations

import argparse
import math
from itertools import combinations

import numpy as np
import pandas as pd

from config import HORIZON_MINUTES, PROCESSED_DIR, RANDOM_SEED, ensure_directories
from shapley_explainer import CoalitionValueCache, exact_shapley, monte_carlo_shapley
from two_hour_environment import DynamicTaxiEnvironment, FiniteHorizonModel, parse_clock_time


def choose_external_baselines(env, start_zone, start_time, players):
    pool = env.candidate_zones(start_zone, start_time, n=len(players) + 12)
    external = [z for z in pool if z not in set(players)]
    if not external:
        return {"wait_only": []}
    absolute = parse_clock_time(start_time)
    demand_zone = max(external, key=lambda z: env.node_metric(z, absolute)["pickup_rate_index"])
    income_zone = max(
        external,
        key=lambda z: env.node_metric(z, absolute)["expected_trip_revenue"]
        - 0.35 * env.node_metric(z, absolute)["expected_trip_distance_miles"],
    )
    greedy_zone = max(
        external,
        key=lambda z: (
            env.node_metric(z, absolute)["expected_trip_revenue"]
            - 0.35 * env.node_metric(z, absolute)["expected_trip_distance_miles"]
            - 0.35 * env.route(start_zone, z, absolute_minutes=absolute)["distance_miles"]
        )
        / max(
            1.0,
            env.route(start_zone, z, absolute_minutes=absolute)["duration_min"]
            + env.node_metric(z, absolute)["expected_wait_min"]
            + env.node_metric(z, absolute)["expected_trip_duration_min"],
        ),
    )
    return {
        "wait_only": [],
        "highest_demand_external": [int(demand_zone)],
        "highest_income_external": [int(income_zone)],
        "greedy_external": [int(greedy_zone)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic Shapley baseline sensitivity")
    parser.add_argument("--permutations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    ensure_directories()
    shapley = pd.read_csv(PROCESSED_DIR / "node_shapley_values.csv").sort_values("shapley_rank")
    start_zone = int(shapley["start_zone"].iloc[0])
    start_time = str(shapley["start_time"].iloc[0])
    players = shapley["zone_id"].astype(int).tolist()
    env = DynamicTaxiEnvironment.load()
    if "competition_scenario" in shapley.columns:
        env = env.with_competition_scenario(str(shapley["competition_scenario"].iloc[0]))
    model = FiniteHorizonModel(env, start_time, HORIZON_MINUTES, players)
    baselines = choose_external_baselines(env, start_zone, start_time, players)

    long_rows = []
    for name, base_zones in baselines.items():
        print(f"Computing baseline: {name} {base_zones}")
        cache = CoalitionValueCache(model, start_zone, players, base_allowed_zones=base_zones)
        if len(players) <= 10:
            phi, se = exact_shapley(cache)
            method = "exact"
        else:
            phi, se, _ = monte_carlo_shapley(cache, args.permutations, args.seed)
            method = "mc"
        for zone, value, standard_error in zip(players, phi, se):
            long_rows.append(
                {
                    "baseline": name,
                    "base_allowed_zone_ids": "|".join(str(z) for z in base_zones),
                    "zone_id": zone,
                    "zone_name": env.zone_names.get(zone, str(zone)),
                    "shapley_value": value,
                    "standard_error": standard_error,
                    "method": method,
                    "baseline_coalition_value": cache.value(0),
                    "full_coalition_value": cache.value((1 << len(players)) - 1),
                }
            )
    long_df = pd.DataFrame(long_rows)
    long_df["rank"] = long_df.groupby("baseline")["shapley_value"].rank(method="min", ascending=False)
    long_df.to_csv(PROCESSED_DIR / "baseline_sensitivity_results.csv", index=False)

    summary_rows = []
    pivot_value = long_df.pivot(index="zone_id", columns="baseline", values="shapley_value")
    pivot_rank = long_df.pivot(index="zone_id", columns="baseline", values="rank")
    for a, b in combinations(baselines.keys(), 2):
        rank_corr = float(pivot_rank[a].corr(pivot_rank[b], method="spearman"))
        value_corr = float(pivot_value[a].corr(pivot_value[b], method="spearman"))
        top_a = set(pivot_value[a].nlargest(min(5, len(players))).index)
        top_b = set(pivot_value[b].nlargest(min(5, len(players))).index)
        overlap = len(top_a & top_b) / max(1, len(top_a | top_b))
        summary_rows.append(
            {
                "baseline_a": a,
                "baseline_b": b,
                "spearman_rank_correlation": rank_corr,
                "spearman_value_correlation": value_corr,
                "top5_jaccard_overlap": overlap,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PROCESSED_DIR / "baseline_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {PROCESSED_DIR / 'baseline_sensitivity_results.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'baseline_sensitivity_summary.csv'}")


if __name__ == "__main__":
    main()
