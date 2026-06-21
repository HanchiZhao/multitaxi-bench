r"""
Evaluate recommendation algorithms against node-level Shapley explanations.

This script is the comparison layer of the revised single-driver taxi-zone project.
It assumes that scripts/route_environment.py has created the processed environment
and that scripts/shapley_explainer.py has created node_shapley_values.csv.

Recommended usage from project root:
    python scripts/evaluate_recommendations.py

Optional:
    python scripts/evaluate_recommendations.py --include-q-learning
    python scripts/evaluate_recommendations.py --origin 132 --time-budget-min 120

Output:
    processed_data/algorithm_recommendation_comparison.csv
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

DEFAULT_ORIGIN = 132
DEFAULT_TIME_BUDGET_MIN = 120.0
DEFAULT_COMPARISON_OUTPUT = os.path.join(PROCESSED_DIR, "algorithm_recommendation_comparison.csv")
DEFAULT_SHAPLEY_OUTPUT = os.path.join(PROCESSED_DIR, "node_shapley_values.csv")
DEFAULT_PATH_TABLE = os.path.join(PROCESSED_DIR, "shapley_ready_candidate_paths.csv")


def ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def parse_path(path_value) -> List[int]:
    if isinstance(path_value, list):
        return [int(x) for x in path_value]
    s = str(path_value).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
        return [int(float(p.strip())) for p in s.split(",") if p.strip()]
    if "-" in s:
        return [int(float(p.strip())) for p in s.split("-") if p.strip()]
    if "," in s:
        return [int(float(p.strip())) for p in s.split(",") if p.strip()]
    return [int(float(s))]


def load_shapley_table(path: str = DEFAULT_SHAPLEY_OUTPUT) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Warning: Shapley table not found: {path}")
        print("Run: python scripts/shapley_explainer.py")
        return pd.DataFrame(columns=["zone_id", "shapley_value", "rank"])
    df = pd.read_csv(path)
    if "zone_id" not in df.columns:
        return pd.DataFrame(columns=["zone_id", "shapley_value", "rank"])
    if "shapley_value" not in df.columns:
        df["shapley_value"] = 0.0
    if "rank" not in df.columns:
        df = df.sort_values("shapley_value", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    return df


def shapley_metrics_for_path(path: Sequence[int], shapley_df: pd.DataFrame) -> Dict[str, float]:
    if shapley_df.empty or not path:
        return {
            "destination_shapley_value": np.nan,
            "destination_shapley_rank": np.nan,
            "path_total_shapley": np.nan,
            "path_mean_shapley": np.nan,
            "path_positive_shapley_nodes": np.nan,
        }

    sh = shapley_df.set_index("zone_id")
    unique_nodes = list(dict.fromkeys(int(z) for z in path))
    dest = int(path[-1])
    values = []
    for z in unique_nodes:
        if z in sh.index:
            values.append(float(sh.loc[z, "shapley_value"]))
    dest_value = float(sh.loc[dest, "shapley_value"]) if dest in sh.index else np.nan
    dest_rank = float(sh.loc[dest, "rank"]) if dest in sh.index else np.nan
    return {
        "destination_shapley_value": dest_value,
        "destination_shapley_rank": dest_rank,
        "path_total_shapley": float(np.sum(values)) if values else np.nan,
        "path_mean_shapley": float(np.mean(values)) if values else np.nan,
        "path_positive_shapley_nodes": int(np.sum(np.array(values) > 0)) if values else 0,
    }


def path_record_from_result(env, label: str, path_result, shapley_df: pd.DataFrame, extra: Optional[Dict] = None) -> Dict:
    path = [int(x) for x in path_result.path]
    dest = int(path[-1]) if path else -1
    info = env.node_info.get(dest, {})
    rec = {
        "algorithm": label,
        "origin": int(path[0]) if path else None,
        "recommended_destination": dest,
        "destination_name": info.get("zone_name", ""),
        "borough": info.get("borough", ""),
        "path": "-".join(str(x) for x in path),
        "num_edges": int(getattr(path_result, "num_edges", max(0, len(path) - 1))),
        "travel_time_min": float(getattr(path_result, "total_travel_time", np.nan)),
        "distance_miles": float(getattr(path_result, "total_distance", np.nan)),
        "dest_expected_income": float(getattr(path_result, "total_expected_income", np.nan)),
        "route_utility": float(getattr(path_result, "route_utility", np.nan)),
        "pickup_rate_per_hour": float(info.get("pickup_rate_per_hour", np.nan)),
        "income_score": float(info.get("income_score", np.nan)),
        "demand_score": float(info.get("demand_score", np.nan)),
        "opportunity_score": float(info.get("opportunity_score", np.nan)),
    }
    rec.update(shapley_metrics_for_path(path, shapley_df))
    if extra:
        rec.update(extra)
    return rec


def load_env(rebuild: bool = False):
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from route_environment import build_environment, load_environment_from_processed  # type: ignore

    if rebuild:
        return build_environment()
    try:
        return load_environment_from_processed()
    except Exception as exc:
        print("Processed environment not found or not loadable. Rebuilding from raw data...")
        print(f"Reason: {exc}")
        return build_environment()


def add_static_algorithm_rows(env, origin: int, time_budget_min: float, shapley_df: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []
    ranking_algs = ["utility", "dijkstra_time", "dijkstra_distance", "highest_income", "highest_demand"]

    for alg in ranking_algs:
        ranked = env.rank_opportunity_nodes(
            origin=origin,
            time_budget_min=time_budget_min,
            top_k=1,
            algorithm=alg,
        )
        if ranked.empty:
            continue
        dest = int(ranked.iloc[0]["destination"])
        try:
            if alg == "dijkstra_distance":
                pr = env.shortest_distance_path(origin, dest, algorithm="dijkstra", time_budget_min=time_budget_min)
            else:
                pr = env.shortest_time_path(origin, dest, algorithm="dijkstra", time_budget_min=time_budget_min)
            rows.append(path_record_from_result(env, alg, pr, shapley_df, extra={"source": "rank_opportunity_nodes"}))
        except Exception as exc:
            rows.append(
                {
                    "algorithm": alg,
                    "origin": origin,
                    "recommended_destination": dest,
                    "path": "",
                    "error": str(exc),
                    "source": "rank_opportunity_nodes",
                }
            )
    return rows


def add_candidate_path_row(env, shapley_df: pd.DataFrame, path_table_path: str = DEFAULT_PATH_TABLE) -> List[Dict]:
    if not os.path.exists(path_table_path):
        return []
    df = pd.read_csv(path_table_path)
    if df.empty or "route_utility" not in df.columns or "path" not in df.columns:
        return []
    row = df.sort_values(["route_utility", "travel_time_min"], ascending=[False, True]).iloc[0]
    path = parse_path(row["path"])
    if len(path) < 2:
        return []
    try:
        pr = env.compute_path_result(path, algorithm="candidate_path_highest_utility")
        return [
            path_record_from_result(
                env,
                "candidate_path_highest_utility",
                pr,
                shapley_df,
                extra={"source": "shapley_ready_candidate_paths"},
            )
        ]
    except Exception as exc:
        return [
            {
                "algorithm": "candidate_path_highest_utility",
                "origin": path[0],
                "recommended_destination": path[-1],
                "path": "-".join(str(x) for x in path),
                "error": str(exc),
                "source": "shapley_ready_candidate_paths",
            }
        ]


def add_value_iteration_row(
    env,
    origin: int,
    time_budget_min: float,
    shapley_df: pd.DataFrame,
    candidate_limit: int = 40,
) -> List[Dict]:
    try:
        candidates = env.candidate_opportunity_nodes(origin=origin, time_budget_min=time_budget_min, max_candidates=candidate_limit)
        vi_df, _V, _policy = env.finite_horizon_value_iteration(
            origin=origin,
            time_budget_min=time_budget_min,
            candidate_nodes=candidates,
            max_actions_per_state=8,
        )
        current = vi_df[(vi_df["zone"] == origin) & (vi_df["time_bin"] == 0)]
        if current.empty:
            return []
        action = int(current.iloc[0]["best_action_zone"])
        if action == origin:
            # Stay is a valid action. Represent it as a zero-edge path.
            class SimplePath:
                path = [origin, origin]
                total_travel_time = 0.0
                total_distance = 0.0
                total_expected_income = float(env.node_info[origin].get("avg_driver_income_per_trip", 0.0))
                route_utility = float(current.iloc[0].get("value", 0.0))
                num_edges = 0

            pr = SimplePath()
        else:
            pr = env.shortest_time_path(origin, action, algorithm="dijkstra", time_budget_min=time_budget_min)
        return [
            path_record_from_result(
                env,
                "value_iteration_first_action",
                pr,
                shapley_df,
                extra={
                    "source": "finite_horizon_value_iteration",
                    "policy_value": float(current.iloc[0].get("value", np.nan)),
                    "immediate_reward": float(current.iloc[0].get("immediate_reward", np.nan)),
                },
            )
        ]
    except Exception as exc:
        return [{"algorithm": "value_iteration_first_action", "origin": origin, "error": str(exc)}]


def add_q_learning_row(
    env,
    origin: int,
    time_budget_min: float,
    shapley_df: pd.DataFrame,
    episodes: int,
    candidate_limit: int = 40,
) -> List[Dict]:
    try:
        candidates = env.candidate_opportunity_nodes(origin=origin, time_budget_min=time_budget_min, max_candidates=candidate_limit)
        q_df, _Q = env.tabular_q_learning(
            origin=origin,
            time_budget_min=time_budget_min,
            candidate_nodes=candidates,
            episodes=episodes,
            max_actions_per_state=8,
        )
        current = q_df[(q_df["zone"] == origin) & (q_df["time_bin"] == 0)]
        if current.empty:
            return []
        action = int(current.iloc[0]["best_action_zone"])
        if action == origin:
            class SimplePath:
                path = [origin, origin]
                total_travel_time = 0.0
                total_distance = 0.0
                total_expected_income = float(env.node_info[origin].get("avg_driver_income_per_trip", 0.0))
                route_utility = float(current.iloc[0].get("q_value", 0.0))
                num_edges = 0

            pr = SimplePath()
        else:
            pr = env.shortest_time_path(origin, action, algorithm="dijkstra", time_budget_min=time_budget_min)
        return [
            path_record_from_result(
                env,
                "q_learning_first_action",
                pr,
                shapley_df,
                extra={"source": "tabular_q_learning", "q_value": float(current.iloc[0].get("q_value", np.nan))},
            )
        ]
    except Exception as exc:
        return [{"algorithm": "q_learning_first_action", "origin": origin, "error": str(exc)}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare recommendation algorithms with Shapley explanation metrics.")
    parser.add_argument("--origin", type=int, default=DEFAULT_ORIGIN)
    parser.add_argument("--time-budget-min", type=float, default=DEFAULT_TIME_BUDGET_MIN)
    parser.add_argument("--output", type=str, default=DEFAULT_COMPARISON_OUTPUT)
    parser.add_argument("--shapley-table", type=str, default=DEFAULT_SHAPLEY_OUTPUT)
    parser.add_argument("--path-table", type=str, default=DEFAULT_PATH_TABLE)
    parser.add_argument("--rebuild-env", action="store_true", help="Rebuild processed environment from raw data before evaluation.")
    parser.add_argument("--include-q-learning", action="store_true", help="Also run a small Q-learning comparison.")
    parser.add_argument("--q-episodes", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    env = load_env(rebuild=args.rebuild_env)
    shapley_df = load_shapley_table(args.shapley_table)

    rows: List[Dict] = []
    rows.extend(add_static_algorithm_rows(env, args.origin, args.time_budget_min, shapley_df))
    rows.extend(add_candidate_path_row(env, shapley_df, args.path_table))
    rows.extend(add_value_iteration_row(env, args.origin, args.time_budget_min, shapley_df))
    if args.include_q_learning:
        rows.extend(add_q_learning_row(env, args.origin, args.time_budget_min, shapley_df, episodes=args.q_episodes))

    out = pd.DataFrame(rows)
    if not out.empty and "route_utility" in out.columns:
        out = out.sort_values(["route_utility", "destination_shapley_value"], ascending=[False, False]).reset_index(drop=True)
        out.insert(0, "comparison_rank", np.arange(1, len(out) + 1))
    out.to_csv(args.output, index=False)

    print("\n=== Recommendation comparison complete ===")
    print(f"Saved comparison table: {args.output}")
    if not out.empty:
        cols = [
            c
            for c in [
                "comparison_rank",
                "algorithm",
                "recommended_destination",
                "destination_name",
                "travel_time_min",
                "distance_miles",
                "route_utility",
                "destination_shapley_value",
                "destination_shapley_rank",
                "path_total_shapley",
            ]
            if c in out.columns
        ]
        print(out[cols].to_string(index=False))
    else:
        print("No recommendation rows were produced.")


if __name__ == "__main__":
    main()
