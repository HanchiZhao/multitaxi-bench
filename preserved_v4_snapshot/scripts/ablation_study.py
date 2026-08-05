"""Ablation study for Dynamic Zone Shapley.

High-Shapley zones are removed from the active reposition action set and the resulting
loss in optimal expected two-hour net earnings is measured.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import HORIZON_MINUTES, PROCESSED_DIR, RANDOM_SEED, TOP_K_FOR_REPORTING, ensure_directories
from two_hour_environment import DynamicTaxiEnvironment, FiniteHorizonModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dynamic Zone Shapley ablation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    ensure_directories()

    shapley_path = PROCESSED_DIR / "node_shapley_values.csv"
    if not shapley_path.exists():
        raise FileNotFoundError(f"Missing {shapley_path}; run shapley_explainer.py first")
    shapley = pd.read_csv(shapley_path).sort_values("shapley_rank")
    start_zone = int(shapley["start_zone"].iloc[0])
    start_time = str(shapley["start_time"].iloc[0])
    players = shapley["zone_id"].astype(int).tolist()
    top_k = min(int(args.top_k), len(players))

    env = DynamicTaxiEnvironment.load()
    if "competition_scenario" in shapley.columns:
        env = env.with_competition_scenario(str(shapley["competition_scenario"].iloc[0]))
    model = FiniteHorizonModel(env, start_time, HORIZON_MINUTES, players)
    full_value = model.start_value(start_zone, players)
    rows = []

    for rank, zone in enumerate(players[:top_k], start=1):
        allowed = [z for z in players if z != zone]
        value = model.start_value(start_zone, allowed)
        rows.append(
            {
                "ablation_type": "single_top_shapley",
                "ablation_rank": rank,
                "removed_zone_ids": str(zone),
                "removed_zone_names": env.zone_names.get(zone, str(zone)),
                "baseline_full_value": full_value,
                "ablated_value": value,
                "absolute_drop": full_value - value,
                "percentage_drop": 100.0 * (full_value - value) / max(abs(full_value), 1e-9),
            }
        )

    for k in range(1, top_k + 1):
        removed = players[:k]
        allowed = [z for z in players if z not in set(removed)]
        value = model.start_value(start_zone, allowed)
        rows.append(
            {
                "ablation_type": "cumulative_top_shapley",
                "ablation_rank": k,
                "removed_zone_ids": "|".join(str(z) for z in removed),
                "removed_zone_names": "|".join(env.zone_names.get(z, str(z)) for z in removed),
                "baseline_full_value": full_value,
                "ablated_value": value,
                "absolute_drop": full_value - value,
                "percentage_drop": 100.0 * (full_value - value) / max(abs(full_value), 1e-9),
            }
        )

    rng = np.random.default_rng(args.seed)
    for k in range(1, top_k + 1):
        for rep in range(int(args.random_controls)):
            removed = sorted(int(x) for x in rng.choice(players, size=k, replace=False))
            allowed = [z for z in players if z not in set(removed)]
            value = model.start_value(start_zone, allowed)
            rows.append(
                {
                    "ablation_type": "random_control",
                    "ablation_rank": k,
                    "random_rep": rep,
                    "removed_zone_ids": "|".join(str(z) for z in removed),
                    "removed_zone_names": "|".join(env.zone_names.get(z, str(z)) for z in removed),
                    "baseline_full_value": full_value,
                    "ablated_value": value,
                    "absolute_drop": full_value - value,
                    "percentage_drop": 100.0 * (full_value - value) / max(abs(full_value), 1e-9),
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(PROCESSED_DIR / "ablation_study_results.csv", index=False)
    summary = (
        results.groupby(["ablation_type", "ablation_rank"], as_index=False)
        .agg(
            mean_absolute_drop=("absolute_drop", "mean"),
            std_absolute_drop=("absolute_drop", "std"),
            mean_percentage_drop=("percentage_drop", "mean"),
            n=("absolute_drop", "size"),
        )
    )
    summary.to_csv(PROCESSED_DIR / "ablation_study_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {PROCESSED_DIR / 'ablation_study_results.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'ablation_study_summary.csv'}")


if __name__ == "__main__":
    main()
