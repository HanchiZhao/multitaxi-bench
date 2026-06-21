r"""
Node-level Shapley explainer for the single-driver opportunity-routing project.

Goal
----
Given a Shapley-ready candidate path table, explain which taxi-zone nodes
increase the best available route utility.

This module matches the revised project design:
    origin + time budget -> candidate opportunity paths -> Shapley explanation

Coalition value
---------------
For a coalition S of optional nodes:
    v(S) = max route_utility among candidate paths activated by S

A path is activated when all of its required optional nodes are included in S.
The baseline route is always available, so every coalition has a valid value.

Recommended usage from project root:
    python scripts/shapley_explainer.py

Optional examples:
    python scripts/shapley_explainer.py --origin 132 --baseline-mode shortest_time
    python scripts/shapley_explainer.py --origin 132 --baseline-mode zero --permutations 2000

Outputs
-------
    processed_data/node_shapley_values.csv
    processed_data/shapley_run_summary.csv
    processed_data/shapley_rebuilt_path_table.csv
    results/figures/top_shapley_nodes.png
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Paths and defaults
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

DEFAULT_PATH_TABLE = os.path.join(PROCESSED_DIR, "shapley_ready_candidate_paths.csv")
DEFAULT_NODE_TABLE = os.path.join(PROCESSED_DIR, "opportunity_nodes_2025-01_2025-06.csv")
DEFAULT_OUTPUT = os.path.join(PROCESSED_DIR, "node_shapley_values.csv")
DEFAULT_SUMMARY_OUTPUT = os.path.join(PROCESSED_DIR, "shapley_run_summary.csv")
DEFAULT_REBUILT_PATH_TABLE = os.path.join(PROCESSED_DIR, "shapley_rebuilt_path_table.csv")
DEFAULT_FIGURE = os.path.join(FIGURES_DIR, "top_shapley_nodes.png")

DEFAULT_ORIGIN = 132
DEFAULT_TIME_BUDGET_MIN = 120.0
DEFAULT_EXACT_THRESHOLD = 18
DEFAULT_PERMUTATIONS = 1000
DEFAULT_SEED = 42


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class PathRecord:
    path_id: int
    algorithm: str
    destination: int
    path: List[int]
    travel_time_min: float
    distance_miles: float
    dest_expected_income: float
    route_utility: float
    required_nodes: List[int]
    required_mask: int


# =============================================================================
# General helpers
# =============================================================================

def ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def parse_path(path_value) -> List[int]:
    """Parse a path stored as '132-93-161' or '[132, 93, 161]'."""
    if isinstance(path_value, list):
        return [int(x) for x in path_value]
    s = str(path_value).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [int(float(p)) for p in parts]
    if "-" in s:
        return [int(float(p.strip())) for p in s.split("-") if p.strip()]
    if "," in s:
        return [int(float(p.strip())) for p in s.split(",") if p.strip()]
    return [int(float(s))]


def safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def bit_count(mask: int) -> int:
    return int(mask.bit_count())


def path_contains_node(path: Sequence[int], node: int) -> bool:
    return int(node) in {int(x) for x in path}


# =============================================================================
# Input preparation
# =============================================================================

def try_auto_generate_path_table(
    path_table_path: str,
    origin: int,
    time_budget_min: float,
    m_per_family: int,
    final_k: int,
    uniform_income: bool,
) -> None:
    """Generate shapley_ready_candidate_paths.csv if it is missing.

    This uses route_environment.py if the processed opportunity environment already
    exists. It intentionally avoids rebuilding raw parquet data unless the user's
    environment file chooses to do so.
    """
    if os.path.exists(path_table_path):
        return

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        from route_environment import load_environment_from_processed  # type: ignore
    except Exception as exc:
        raise FileNotFoundError(
            f"Missing path table: {path_table_path}\n"
            "I also could not import load_environment_from_processed from scripts/route_environment.py.\n"
            "Please run: python scripts\\route_environment.py"
        ) from exc

    try:
        env = load_environment_from_processed()
    except Exception as exc:
        raise FileNotFoundError(
            f"Missing path table: {path_table_path}\n"
            "Processed environment files were not found. Please run:\n"
            "    python scripts\\route_environment.py"
        ) from exc

    print("Path table not found. Auto-generating candidate paths from processed environment...")
    candidates = env.candidate_opportunity_nodes(
        origin=origin,
        time_budget_min=time_budget_min,
        max_candidates=80,
        uniform_income=uniform_income,
    )
    paths = env.build_candidate_path_pool(
        origin=origin,
        time_budget_min=time_budget_min,
        candidate_nodes=candidates,
        m_per_family=m_per_family,
        final_k=final_k,
        uniform_income=uniform_income,
    )
    path_table, _node_to_bit = env.shapley_ready_path_table(origin, paths)
    path_table.to_csv(path_table_path, index=False)
    print(f"Saved auto-generated path table: {path_table_path}")


def load_path_table(path_table_path: str) -> pd.DataFrame:
    if not os.path.exists(path_table_path):
        raise FileNotFoundError(f"Missing path table: {path_table_path}")
    df = pd.read_csv(path_table_path)
    required_cols = ["path", "route_utility"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Path table is missing required columns: {missing}")

    df = df.copy()
    if "path_id" not in df.columns:
        df["path_id"] = range(len(df))
    if "algorithm" not in df.columns:
        df["algorithm"] = "unknown"
    if "destination" not in df.columns:
        df["destination"] = df["path"].apply(lambda x: parse_path(x)[-1] if parse_path(x) else -1)
    for c in ["travel_time_min", "distance_miles", "dest_expected_income", "route_utility"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["parsed_path"] = df["path"].apply(parse_path)
    df = df[df["parsed_path"].apply(lambda p: len(p) >= 1)].reset_index(drop=True)
    if df.empty:
        raise ValueError("Path table contains no valid paths.")
    return df


def choose_baseline_row(df: pd.DataFrame, baseline_mode: str) -> Optional[pd.Series]:
    """Choose baseline path row.

    baseline_mode:
    - shortest_time: shortest travel time among candidate paths
    - shortest_distance: shortest distance among candidate paths
    - first: first row in the path table
    - highest_utility: highest utility path; useful as a sanity check but often gives zero Shapley
    - zero: no path baseline, v(empty)=0 and only the origin is always available
    """
    mode = baseline_mode.lower()
    if mode == "zero":
        return None
    if mode == "shortest_time":
        idx = df.sort_values(["travel_time_min", "route_utility"], ascending=[True, False]).index[0]
    elif mode == "shortest_distance":
        idx = df.sort_values(["distance_miles", "route_utility"], ascending=[True, False]).index[0]
    elif mode == "highest_utility":
        idx = df.sort_values(["route_utility", "travel_time_min"], ascending=[False, True]).index[0]
    elif mode == "first":
        idx = df.index[0]
    else:
        raise ValueError(
            "baseline-mode must be one of: shortest_time, shortest_distance, first, highest_utility, zero"
        )
    return df.loc[idx]


def rebuild_path_records(
    df: pd.DataFrame,
    origin: int,
    baseline_mode: str = "shortest_time",
) -> Tuple[List[PathRecord], Dict[int, int], Dict[int, object]]:
    """Rebuild optional-node masks from raw path strings.

    Important: this ignores the required_mask already in the path table because
    older environment versions may have chosen the highest-utility path as the
    baseline, which would make all Shapley values collapse to zero.
    """
    baseline_row = choose_baseline_row(df, baseline_mode)
    if baseline_row is None:
        baseline_path = [int(origin)]
        baseline_value = 0.0
        baseline_path_id = None
        baseline_algorithm = "zero"
    else:
        baseline_path = [int(x) for x in baseline_row["parsed_path"]]
        baseline_value = safe_float(baseline_row["route_utility"])
        baseline_path_id = safe_int(baseline_row["path_id"])
        baseline_algorithm = str(baseline_row.get("algorithm", "unknown"))

    baseline_nodes = set(int(x) for x in baseline_path)
    all_path_nodes = sorted({int(z) for path in df["parsed_path"] for z in path})

    # Origin and baseline path nodes are always available, so they are not players.
    optional_nodes = [z for z in all_path_nodes if z not in baseline_nodes]
    node_to_bit = {z: i for i, z in enumerate(optional_nodes)}

    records: List[PathRecord] = []
    for _, row in df.iterrows():
        path = [int(x) for x in row["parsed_path"]]
        required_nodes = [int(z) for z in path if int(z) in node_to_bit]
        mask = 0
        for z in required_nodes:
            mask |= 1 << node_to_bit[z]
        records.append(
            PathRecord(
                path_id=safe_int(row.get("path_id", len(records))),
                algorithm=str(row.get("algorithm", "unknown")),
                destination=safe_int(row.get("destination", path[-1] if path else -1)),
                path=path,
                travel_time_min=safe_float(row.get("travel_time_min", 0.0)),
                distance_miles=safe_float(row.get("distance_miles", 0.0)),
                dest_expected_income=safe_float(row.get("dest_expected_income", 0.0)),
                route_utility=safe_float(row.get("route_utility", 0.0)),
                required_nodes=required_nodes,
                required_mask=mask,
            )
        )

    meta: Dict[int, object] = {}
    summary = {
        "origin": int(origin),
        "baseline_mode": baseline_mode,
        "baseline_path_id": baseline_path_id,
        "baseline_algorithm": baseline_algorithm,
        "baseline_path": "-".join(str(x) for x in baseline_path),
        "baseline_value": baseline_value,
        "baseline_nodes": sorted(baseline_nodes),
        "num_paths": len(records),
        "num_optional_players": len(optional_nodes),
    }
    meta.update(summary)
    return records, node_to_bit, meta


def save_rebuilt_path_table(records: Sequence[PathRecord], output_path: str) -> None:
    rows = []
    for r in records:
        rows.append(
            {
                "path_id": r.path_id,
                "algorithm": r.algorithm,
                "destination": r.destination,
                "path": "-".join(str(x) for x in r.path),
                "required_optional_nodes": ",".join(str(x) for x in r.required_nodes),
                "required_mask": r.required_mask,
                "travel_time_min": r.travel_time_min,
                "distance_miles": r.distance_miles,
                "dest_expected_income": r.dest_expected_income,
                "route_utility": r.route_utility,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


# =============================================================================
# Coalition value and Shapley calculation
# =============================================================================

class CoalitionValueFunction:
    def __init__(self, path_records: Sequence[PathRecord], baseline_value: float):
        self.path_records = list(path_records)
        self.baseline_value = float(baseline_value)
        self.cache: Dict[int, float] = {}
        self.best_path_cache: Dict[int, Optional[int]] = {}

    def value(self, coalition_mask: int) -> float:
        coalition_mask = int(coalition_mask)
        if coalition_mask in self.cache:
            return self.cache[coalition_mask]
        best_value = self.baseline_value
        best_path_id: Optional[int] = None
        for pr in self.path_records:
            if (pr.required_mask & coalition_mask) == pr.required_mask:
                if pr.route_utility > best_value:
                    best_value = pr.route_utility
                    best_path_id = pr.path_id
        self.cache[coalition_mask] = best_value
        self.best_path_cache[coalition_mask] = best_path_id
        return best_value

    def best_path(self, coalition_mask: int) -> Optional[int]:
        _ = self.value(coalition_mask)
        return self.best_path_cache.get(int(coalition_mask))


def exact_shapley(
    n_players: int,
    value_fn: CoalitionValueFunction,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact Shapley by summing all coalitions.

    Returns:
        shapley_values, standard_errors, positive_counts, marginal_counts
    """
    shap = np.zeros(n_players, dtype=float)
    positive_counts = np.zeros(n_players, dtype=float)
    marginal_counts = np.zeros(n_players, dtype=float)

    if n_players == 0:
        return shap, np.zeros(0), positive_counts, marginal_counts

    # Precompute all coalition values once.
    all_masks = range(1 << n_players)
    for mask in all_masks:
        value_fn.value(mask)

    for i in range(n_players):
        bit = 1 << i
        total = 0.0
        for mask in range(1 << n_players):
            if mask & bit:
                continue
            k = bit_count(mask)
            # Shapley weight for all coalitions of size k not containing i.
            weight = 1.0 / (n_players * math.comb(n_players - 1, k))
            marginal = value_fn.value(mask | bit) - value_fn.value(mask)
            total += weight * marginal
            if marginal > 1e-12:
                positive_counts[i] += 1.0
            marginal_counts[i] += 1.0
        shap[i] = total

    # Exact enumeration has no Monte Carlo standard error.
    stderr = np.zeros(n_players, dtype=float)
    return shap, stderr, positive_counts, marginal_counts


def monte_carlo_shapley(
    n_players: int,
    value_fn: CoalitionValueFunction,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    antithetic: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo Shapley over random permutations.

    If antithetic=True, every sampled permutation is paired with its reverse.
    This typically lowers variance by balancing early and late insertion effects.
    """
    rng = random.Random(seed)
    sums = np.zeros(n_players, dtype=float)
    sums_sq = np.zeros(n_players, dtype=float)
    counts = np.zeros(n_players, dtype=float)
    positive_counts = np.zeros(n_players, dtype=float)

    if n_players == 0:
        return sums, np.zeros(0), positive_counts, counts

    def process_order(order: Sequence[int]) -> None:
        mask = 0
        prev_value = value_fn.value(mask)
        for i in order:
            new_mask = mask | (1 << i)
            new_value = value_fn.value(new_mask)
            marginal = new_value - prev_value
            sums[i] += marginal
            sums_sq[i] += marginal * marginal
            counts[i] += 1.0
            if marginal > 1e-12:
                positive_counts[i] += 1.0
            mask = new_mask
            prev_value = new_value

    players = list(range(n_players))
    for _ in range(int(permutations)):
        order = players[:]
        rng.shuffle(order)
        process_order(order)
        if antithetic:
            process_order(list(reversed(order)))

    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.zeros(n_players, dtype=float)
    valid = counts > 1
    variances[valid] = (sums_sq[valid] - counts[valid] * means[valid] ** 2) / (counts[valid] - 1)
    variances = np.maximum(variances, 0.0)
    stderr = np.divide(np.sqrt(variances), np.sqrt(counts), out=np.zeros_like(variances), where=counts > 0)
    return means, stderr, positive_counts, counts


# =============================================================================
# Output construction
# =============================================================================

def load_node_metadata(node_table_path: str) -> pd.DataFrame:
    if not os.path.exists(node_table_path):
        return pd.DataFrame(columns=["zone_id", "zone_name", "borough"])
    nodes = pd.read_csv(node_table_path)
    if "zone_id" not in nodes.columns:
        return pd.DataFrame(columns=["zone_id", "zone_name", "borough"])
    return nodes


def build_result_table(
    shap_values: np.ndarray,
    stderr: np.ndarray,
    positive_counts: np.ndarray,
    marginal_counts: np.ndarray,
    node_to_bit: Dict[int, int],
    path_records: Sequence[PathRecord],
    nodes_meta: pd.DataFrame,
) -> pd.DataFrame:
    bit_to_node = {bit: node for node, bit in node_to_bit.items()}
    rows = []
    for bit in range(len(bit_to_node)):
        node = int(bit_to_node[bit])
        containing = [pr for pr in path_records if path_contains_node(pr.path, node)]
        best_pr = max(containing, key=lambda p: p.route_utility) if containing else None
        rows.append(
            {
                "zone_id": node,
                "shapley_value": float(shap_values[bit]),
                "standard_error": float(stderr[bit]),
                "positive_marginal_count": int(positive_counts[bit]),
                "marginal_count": int(marginal_counts[bit]),
                "positive_marginal_fraction": float(positive_counts[bit] / marginal_counts[bit])
                if marginal_counts[bit] > 0
                else 0.0,
                "appeared_in_paths_count": len(containing),
                "best_path_including_node_id": best_pr.path_id if best_pr is not None else None,
                "best_path_including_node_utility": best_pr.route_utility if best_pr is not None else None,
                "best_path_including_node": "-".join(str(x) for x in best_pr.path) if best_pr is not None else "",
            }
        )
    out = pd.DataFrame(rows)
    if not nodes_meta.empty:
        keep_cols = [
            c
            for c in [
                "zone_id",
                "zone_name",
                "borough",
                "service_zone",
                "avg_driver_income_per_trip",
                "pickup_rate_per_hour",
                "expected_income_2h_proxy",
                "income_score",
                "demand_score",
                "opportunity_score",
            ]
            if c in nodes_meta.columns
        ]
        out = out.merge(nodes_meta[keep_cols], on="zone_id", how="left")
    out = out.sort_values(["shapley_value", "appeared_in_paths_count"], ascending=[False, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    # Put rank and metadata near the front.
    front = [
        c
        for c in [
            "rank",
            "zone_id",
            "zone_name",
            "borough",
            "shapley_value",
            "standard_error",
            "positive_marginal_fraction",
            "appeared_in_paths_count",
        ]
        if c in out.columns
    ]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def save_summary(summary: Dict[str, object], output_path: str) -> None:
    pd.DataFrame([summary]).to_csv(output_path, index=False)


def save_top_plot(result_df: pd.DataFrame, figure_path: str, top_n: int = 15) -> None:
    if result_df.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping figure.")
        return

    top = result_df.head(top_n).copy()
    label_col = "zone_name" if "zone_name" in top.columns else "zone_id"
    labels = []
    for _, row in top.iterrows():
        z = int(row["zone_id"])
        name = str(row.get(label_col, ""))
        if not name or name.lower() == "nan":
            labels.append(str(z))
        else:
            labels.append(f"{z} {name}")

    plt.figure(figsize=(10, max(5, 0.35 * len(top))))
    plt.barh(labels[::-1], top["shapley_value"].values[::-1])
    plt.xlabel("Node Shapley value")
    plt.ylabel("Taxi zone")
    plt.title(f"Top {len(top)} Shapley Nodes")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute node-level Shapley values for candidate taxi-zone paths.")
    parser.add_argument("--origin", type=int, default=DEFAULT_ORIGIN, help="Origin zone ID, default 132 JFK Airport.")
    parser.add_argument("--time-budget-min", type=float, default=DEFAULT_TIME_BUDGET_MIN)
    parser.add_argument("--path-table", type=str, default=DEFAULT_PATH_TABLE)
    parser.add_argument("--node-table", type=str, default=DEFAULT_NODE_TABLE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=str, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--rebuilt-path-output", type=str, default=DEFAULT_REBUILT_PATH_TABLE)
    parser.add_argument("--figure-output", type=str, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--baseline-mode",
        type=str,
        default="shortest_time",
        choices=["shortest_time", "shortest_distance", "first", "highest_utility", "zero"],
        help=(
            "Baseline route mode. shortest_time is recommended. "
            "highest_utility usually makes Shapley values collapse toward zero."
        ),
    )
    parser.add_argument("--exact-threshold", type=int, default=DEFAULT_EXACT_THRESHOLD)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-antithetic", action="store_true", help="Disable reverse-permutation pairing.")
    parser.add_argument("--auto-generate", action="store_true", help="Auto-generate path table if missing.")
    parser.add_argument("--m-per-family", type=int, default=5, help="Used only when auto-generating path table.")
    parser.add_argument("--final-k", type=int, default=30, help="Used only when auto-generating path table.")
    parser.add_argument("--uniform-income", action="store_true", help="Used only when auto-generating path table.")
    parser.add_argument("--top-plot-n", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if args.auto_generate or not os.path.exists(args.path_table):
        if not os.path.exists(args.path_table):
            print(f"Path table not found: {args.path_table}")
            print("Attempting to auto-generate it from the processed environment...")
        try_auto_generate_path_table(
            path_table_path=args.path_table,
            origin=args.origin,
            time_budget_min=args.time_budget_min,
            m_per_family=args.m_per_family,
            final_k=args.final_k,
            uniform_income=args.uniform_income,
        )

    path_df = load_path_table(args.path_table)
    path_records, node_to_bit, meta = rebuild_path_records(
        path_df,
        origin=args.origin,
        baseline_mode=args.baseline_mode,
    )
    save_rebuilt_path_table(path_records, args.rebuilt_path_output)

    n_players = len(node_to_bit)
    baseline_value = float(meta.get("baseline_value", 0.0))
    value_fn = CoalitionValueFunction(path_records, baseline_value=baseline_value)

    if n_players <= args.exact_threshold:
        method = "exact"
        shap_values, stderr, positive_counts, marginal_counts = exact_shapley(n_players, value_fn)
    else:
        method = "monte_carlo"
        shap_values, stderr, positive_counts, marginal_counts = monte_carlo_shapley(
            n_players,
            value_fn,
            permutations=args.permutations,
            seed=args.seed,
            antithetic=not args.no_antithetic,
        )

    nodes_meta = load_node_metadata(args.node_table)
    result = build_result_table(
        shap_values=shap_values,
        stderr=stderr,
        positive_counts=positive_counts,
        marginal_counts=marginal_counts,
        node_to_bit=node_to_bit,
        path_records=path_records,
        nodes_meta=nodes_meta,
    )
    result.to_csv(args.output, index=False)
    save_top_plot(result, args.figure_output, top_n=args.top_plot_n)

    grand_coalition = (1 << n_players) - 1 if n_players > 0 else 0
    v_empty = value_fn.value(0)
    v_full = value_fn.value(grand_coalition)
    shap_sum = float(np.sum(shap_values))
    explanation_gap = float((v_full - v_empty) - shap_sum)

    summary = dict(meta)
    summary.update(
        {
            "method": method,
            "exact_threshold": args.exact_threshold,
            "permutations_requested": args.permutations if method == "monte_carlo" else 0,
            "antithetic": (not args.no_antithetic) if method == "monte_carlo" else False,
            "seed": args.seed,
            "v_empty": v_empty,
            "v_full": v_full,
            "v_full_minus_v_empty": v_full - v_empty,
            "sum_shapley_values": shap_sum,
            "efficiency_gap": explanation_gap,
            "path_table": args.path_table,
            "rebuilt_path_table": args.rebuilt_path_output,
            "node_shapley_output": args.output,
            "figure_output": args.figure_output,
        }
    )
    # Lists are not CSV-friendly.
    if isinstance(summary.get("baseline_nodes"), list):
        summary["baseline_nodes"] = ",".join(str(x) for x in summary["baseline_nodes"])
    save_summary(summary, args.summary_output)

    print("\n=== Shapley explainer complete ===")
    print(f"Method: {method}")
    print(f"Baseline mode: {args.baseline_mode}")
    print(f"Baseline path: {summary.get('baseline_path')}")
    print(f"Baseline value v(empty): {v_empty:.6f}")
    print(f"Full coalition value v(N): {v_full:.6f}")
    print(f"Sum Shapley values: {shap_sum:.6f}")
    print(f"Efficiency gap: {explanation_gap:.8f}")
    print(f"Optional player nodes: {n_players}")
    print(f"Saved node Shapley values: {args.output}")
    print(f"Saved run summary: {args.summary_output}")
    print(f"Saved rebuilt path table: {args.rebuilt_path_output}")
    print(f"Saved figure: {args.figure_output}")

    if not result.empty:
        print("\nTop Shapley nodes:")
        cols = [c for c in ["rank", "zone_id", "zone_name", "borough", "shapley_value", "standard_error", "appeared_in_paths_count"] if c in result.columns]
        print(result[cols].head(10).to_string(index=False))

    if args.baseline_mode == "highest_utility":
        print(
            "\nNote: baseline-mode=highest_utility often makes v(empty) already equal to the best path, "
            "so most Shapley values may be near zero. For explanation, shortest_time is usually better."
        )


if __name__ == "__main__":
    main()
