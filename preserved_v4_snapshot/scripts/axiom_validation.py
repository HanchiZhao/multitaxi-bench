"""Axiom and diagnostic validation for Dynamic Zone Shapley.

The script preserves the historical AIM/FMD/CG/DIM/DME labels while adapting their
meaning to the dynamic two-hour net-earnings game. It also adds standard Shapley tests:
Efficiency, Dummy, and Symmetry.
"""
from __future__ import annotations

import argparse
import math
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    AXIOM_TOLERANCE,
    DIM_REVENUE_INCREASE,
    DME_REVENUE_STEPS,
    HORIZON_MINUTES,
    LOW_CONFIDENCE_THRESHOLD,
    PROCESSED_DIR,
    RANDOM_SEED,
    ensure_directories,
)
from shapley_explainer import CoalitionValueCache
from two_hour_environment import DynamicTaxiEnvironment, FiniteHorizonModel


def status_row(name, category, status, statistic, threshold, details, hard_axiom=True):
    return {
        "test": name,
        "category": category,
        "status": status,
        "statistic": statistic,
        "threshold": threshold,
        "hard_axiom": bool(hard_axiom),
        "details": details,
    }


def all_or_sampled_masks(n: int, max_masks: int, seed: int) -> List[int]:
    total = 1 << n
    if total <= max_masks:
        return list(range(total))
    rng = np.random.default_rng(seed)
    selected = {0, total - 1}
    while len(selected) < max_masks:
        selected.add(int(rng.integers(0, total)))
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dynamic Shapley axioms")
    parser.add_argument("--skip-dim", action="store_true")
    parser.add_argument("--skip-dme", action="store_true")
    parser.add_argument("--max-coalitions", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    ensure_directories()

    shapley_path = PROCESSED_DIR / "node_shapley_values.csv"
    if not shapley_path.exists():
        raise FileNotFoundError(f"Missing {shapley_path}; run shapley_explainer.py first")
    shapley = pd.read_csv(shapley_path).sort_values("shapley_rank")
    players = shapley["zone_id"].astype(int).tolist()
    phi = dict(zip(shapley["zone_id"].astype(int), shapley["shapley_value"].astype(float)))
    start_zone = int(shapley["start_zone"].iloc[0])
    start_time = str(shapley["start_time"].iloc[0])
    env = DynamicTaxiEnvironment.load()
    if "competition_scenario" in shapley.columns:
        env = env.with_competition_scenario(str(shapley["competition_scenario"].iloc[0]))
    model = FiniteHorizonModel(env, start_time, HORIZON_MINUTES, players)
    cache = CoalitionValueCache(model, start_zone, players)
    n = len(players)
    full_mask = (1 << n) - 1
    masks = all_or_sampled_masks(n, args.max_coalitions, args.seed)
    results = []

    # Populate values used by multiple tests.
    for mask in masks:
        cache.value(mask)
    cache.value(full_mask)

    # ------------------------------------------------------------------
    # Efficiency (hard Shapley axiom)
    # ------------------------------------------------------------------
    gain = cache.value(full_mask) - cache.value(0)
    phi_sum = sum(phi.values())
    efficiency_gap = phi_sum - gain
    efficiency_pass = abs(efficiency_gap) <= max(AXIOM_TOLERANCE, 1e-6 * max(1.0, abs(gain)))
    results.append(
        status_row(
            "Efficiency",
            "standard_shapley",
            "PASS" if efficiency_pass else "FAIL",
            efficiency_gap,
            "abs(gap) <= numerical tolerance",
            f"sum(phi)={phi_sum:.8f}; v(N)-v(empty)={gain:.8f}",
            True,
        )
    )

    # ------------------------------------------------------------------
    # AIM: action-set monotonicity
    # ------------------------------------------------------------------
    min_marginal = math.inf
    violation_count = 0
    comparison_count = 0
    for mask in masks:
        for i in range(n):
            bit = 1 << i
            if mask & bit:
                continue
            marginal = cache.value(mask | bit) - cache.value(mask)
            min_marginal = min(min_marginal, marginal)
            comparison_count += 1
            if marginal < -AXIOM_TOLERANCE:
                violation_count += 1
    aim_pass = violation_count == 0
    results.append(
        status_row(
            "AIM",
            "dynamic_monotonicity",
            "PASS" if aim_pass else "FAIL",
            min_marginal if min_marginal < math.inf else np.nan,
            f">= {-AXIOM_TOLERANCE}",
            f"violations={violation_count}/{comparison_count}; adding a zone only enlarges the action set",
            True,
        )
    )

    # ------------------------------------------------------------------
    # Dummy
    # ------------------------------------------------------------------
    dummy_zones = []
    dummy_failures = []
    for i, zone in enumerate(players):
        bit = 1 << i
        marginals = []
        for mask in masks:
            if mask & bit:
                continue
            marginals.append(cache.value(mask | bit) - cache.value(mask))
        if marginals and max(abs(x) for x in marginals) <= 1e-8:
            dummy_zones.append(zone)
            if abs(phi.get(zone, 0.0)) > 1e-6:
                dummy_failures.append(zone)
    if dummy_zones:
        dummy_status = "PASS" if not dummy_failures else "FAIL"
        details = f"dummy_zones={dummy_zones}; failures={dummy_failures}"
    else:
        dummy_status = "N/A"
        details = "No exact/near-dummy zones were present in the selected player set"
    results.append(status_row("Dummy", "standard_shapley", dummy_status, len(dummy_failures), "0 failures", details, True))

    # ------------------------------------------------------------------
    # Symmetry
    # ------------------------------------------------------------------
    symmetric_pairs = []
    symmetry_failures = []
    # Exact profile comparison is affordable for the default 10 players.
    if n <= 12:
        for i, j in combinations(range(n), 2):
            differences = []
            for mask in range(1 << n):
                if mask & (1 << i) or mask & (1 << j):
                    continue
                mi = cache.value(mask | (1 << i)) - cache.value(mask)
                mj = cache.value(mask | (1 << j)) - cache.value(mask)
                differences.append(abs(mi - mj))
            if differences and max(differences) <= 1e-8:
                pair = (players[i], players[j])
                symmetric_pairs.append(pair)
                if abs(phi[players[i]] - phi[players[j]]) > 1e-6:
                    symmetry_failures.append(pair)
    if symmetric_pairs:
        symmetry_status = "PASS" if not symmetry_failures else "FAIL"
        symmetry_details = f"symmetric_pairs={symmetric_pairs}; failures={symmetry_failures}"
    else:
        symmetry_status = "N/A"
        symmetry_details = "No empirically symmetric pair exists in this scenario; synthetic unit tests remain in test_dynamic_pipeline.py"
    results.append(status_row("Symmetry", "standard_shapley", symmetry_status, len(symmetry_failures), "0 failures", symmetry_details, True))

    # ------------------------------------------------------------------
    # FMD: dynamic singleton dominance (diagnostic)
    # ------------------------------------------------------------------
    singleton_gains = []
    shapley_values = []
    for i, zone in enumerate(players):
        singleton_gains.append(cache.value(1 << i) - cache.value(0))
        shapley_values.append(phi[zone])
    fmd_corr = float(pd.Series(singleton_gains).rank().corr(pd.Series(shapley_values).rank(), method="pearson"))
    if np.isnan(fmd_corr):
        fmd_status = "N/A"
        fmd_details = "Singleton gains or Shapley values are constant, so rank correlation is undefined"
    else:
        fmd_status = "PASS" if fmd_corr >= 0.5 else "WARN"
        fmd_details = "Spearman correlation between singleton strategic gain and Shapley value; complementarity can reduce correlation"
    results.append(
        status_row(
            "FMD",
            "dynamic_dominance_diagnostic",
            fmd_status,
            fmd_corr,
            ">= 0.50 recommended",
            fmd_details,
            False,
        )
    )

    # ------------------------------------------------------------------
    # CG: dependence on low-confidence imputed transitions
    # ------------------------------------------------------------------
    filtered_env = env.with_min_od_confidence(LOW_CONFIDENCE_THRESHOLD)
    filtered_model = FiniteHorizonModel(filtered_env, start_time, HORIZON_MINUTES, players)
    filtered_full = filtered_model.start_value(start_zone, players)
    original_full = cache.value(full_mask)
    value_drop_pct = 100.0 * (original_full - filtered_full) / max(abs(original_full), 1e-9)
    original_singletons = {z: cache.value(1 << i) - cache.value(0) for i, z in enumerate(players)}
    filtered_baseline = filtered_model.start_value(start_zone, [])
    filtered_singletons = {z: filtered_model.start_value(start_zone, [z]) - filtered_baseline for z in players}
    top_k = min(5, n)
    top_original = set(sorted(players, key=lambda z: original_singletons[z], reverse=True)[:top_k])
    top_filtered = set(sorted(players, key=lambda z: filtered_singletons[z], reverse=True)[:top_k])
    overlap = len(top_original & top_filtered) / max(1, len(top_original | top_filtered))
    cg_pass = value_drop_pct <= 20.0 and overlap >= 0.5
    results.append(
        status_row(
            "CG",
            "imputation_robustness",
            "PASS" if cg_pass else "WARN",
            value_drop_pct,
            "value drop <=20% and top-5 Jaccard >=0.50",
            f"filtered_full={filtered_full:.4f}; original_full={original_full:.4f}; top5_jaccard={overlap:.3f}",
            False,
        )
    )

    # ------------------------------------------------------------------
    # DIM: realistic revenue perturbation
    # ------------------------------------------------------------------
    if args.skip_dim:
        results.append(status_row("DIM", "income_monotonicity", "SKIPPED", np.nan, "not run", "Requested by --skip-dim", False))
    else:
        target_zone = int(shapley.iloc[0]["zone_id"])
        perturbed_env = env.with_origin_revenue_multiplier(target_zone, 1.0 + DIM_REVENUE_INCREASE)
        perturbed_model = FiniteHorizonModel(perturbed_env, start_time, HORIZON_MINUTES, players)
        perturbed_full = perturbed_model.start_value(start_zone, players)
        original_singleton = cache.value(1 << players.index(target_zone)) - cache.value(0)
        perturbed_base = perturbed_model.start_value(start_zone, [])
        perturbed_singleton = perturbed_model.start_value(start_zone, [target_zone]) - perturbed_base
        dim_pass = perturbed_full >= original_full - AXIOM_TOLERANCE and perturbed_singleton >= original_singleton - AXIOM_TOLERANCE
        results.append(
            status_row(
                "DIM",
                "income_monotonicity",
                "PASS" if dim_pass else "FAIL",
                perturbed_full - original_full,
                ">= 0",
                f"zone={target_zone}; singleton change={perturbed_singleton-original_singleton:.6f}; revenue +{DIM_REVENUE_INCREASE*100:.1f}%",
                True,
            )
        )

    # ------------------------------------------------------------------
    # DME: diagnostic saturation under repeated revenue increases
    # ------------------------------------------------------------------
    if args.skip_dme:
        results.append(status_row("DME", "diminishing_returns_diagnostic", "SKIPPED", np.nan, "not run", "Requested by --skip-dme", False))
    else:
        target_zone = int(shapley.iloc[0]["zone_id"])
        values = [original_full]
        for increase in DME_REVENUE_STEPS:
            e = env.with_origin_revenue_multiplier(target_zone, 1.0 + increase)
            m = FiniteHorizonModel(e, start_time, HORIZON_MINUTES, players)
            values.append(m.start_value(start_zone, players))
        increments = np.diff(values)
        diminishing = all(increments[i + 1] <= increments[i] + 1e-6 for i in range(len(increments) - 1))
        results.append(
            status_row(
                "DME",
                "diminishing_returns_diagnostic",
                "PASS" if diminishing else "INFO",
                float(increments[-1]) if len(increments) else np.nan,
                "non-increasing increments (diagnostic only)",
                f"zone={target_zone}; values={values}; increments={increments.tolist()}; network complementarity may legitimately violate DME",
                False,
            )
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(PROCESSED_DIR / "axiom_validation_results.csv", index=False)
    summary = (
        results_df.groupby(["status", "hard_axiom"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    summary.to_csv(PROCESSED_DIR / "axiom_validation_summary.csv", index=False)
    print(results_df[["test", "status", "statistic", "details"]].to_string(index=False))
    print(f"Saved: {PROCESSED_DIR / 'axiom_validation_results.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'axiom_validation_summary.csv'}")


if __name__ == "__main__":
    main()
