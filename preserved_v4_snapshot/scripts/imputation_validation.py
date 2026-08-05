"""Holdout validation for hierarchical EB OD estimates.

Dense observed cells are reconstructed without their same-bin observation. Duration is
recombined on a log scale and includes the calibrated structural duration prior, reducing
the previous over-shrinkage of long trips.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RANDOM_SEED, ensure_directories


def safe_weighted(values, weights, default):
    valid = [(float(v), float(w)) for v, w in zip(values, weights) if pd.notna(v) and np.isfinite(v) and float(w) > 0]
    if not valid:
        return float(default)
    return float(np.average([v for v, _ in valid], weights=[w for _, w in valid]))


def safe_geometric_weighted(values, weights, default):
    valid = [(float(v), float(w)) for v, w in zip(values, weights)
             if pd.notna(v) and np.isfinite(v) and float(v) > 0 and float(w) > 0]
    if not valid:
        return float(default)
    return float(math.exp(np.average([math.log(v) for v, _ in valid], weights=[w for _, w in valid])))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    nonzero = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100.0) if nonzero.any() else np.nan
    spearman = float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank(), method="pearson")) if len(y_true) > 1 else np.nan
    return {"mae": mae, "rmse": rmse, "bias": bias, "mape_percent": mape, "spearman_rank_correlation": spearman}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate hierarchical EB imputation")
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    ensure_directories()
    path = PROCESSED_DIR / "dynamic_od_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run route_environment.py first")
    df = pd.read_csv(path)
    dense = df[df["observed_trip_count"] >= args.min_count].copy()
    effective_min_count = int(args.min_count)
    if dense.empty:
        observed_positive = df[df["observed_trip_count"] > 0].copy()
        if observed_positive.empty:
            raise RuntimeError("No observed OD cells available for validation")
        effective_min_count = max(1, int(observed_positive["observed_trip_count"].quantile(0.75)))
        dense = observed_positive[observed_positive["observed_trip_count"] >= effective_min_count].copy()
        print(f"Warning: using adaptive observed-count threshold {effective_min_count}.")
    if len(dense) > args.sample:
        dense = dense.sample(args.sample, random_state=args.seed)

    reconstruct_rows = []
    for row in dense.itertuples(index=False):
        prior_weights = [
            float(row.temporal_weight), float(row.spatial_weight),
            float(row.global_weight), float(row.gravity_weight),
        ]
        pred_revenue = safe_weighted(
            [row.temporal_revenue_prior, row.spatial_revenue_prior, row.global_revenue_prior, row.gravity_revenue_prior],
            prior_weights, row.global_revenue_prior,
        )
        calibrated_duration = float(getattr(row, "calibrated_duration_prior", row.gravity_duration_prior))
        duration_values = [
            row.temporal_duration_prior, row.spatial_duration_prior,
            row.global_duration_prior, calibrated_duration,
        ]
        pred_duration = safe_geometric_weighted(duration_values, prior_weights, calibrated_duration)
        pred_duration = max(pred_duration, 0.85 * calibrated_duration)
        pred_distance = safe_weighted(
            [row.temporal_distance_prior, row.spatial_distance_prior, row.global_distance_prior, row.gravity_distance_prior],
            prior_weights, row.global_distance_prior,
        )
        reconstruct_rows.append({
            "time_bin": int(row.time_bin), "origin": int(row.origin), "destination": int(row.destination),
            "observed_trip_count": float(row.observed_trip_count),
            "true_revenue": float(row.observed_avg_driver_revenue),
            "predicted_revenue_without_same_bin_observation": pred_revenue,
            "true_duration_min": float(row.observed_avg_duration_min),
            "predicted_duration_without_same_bin_observation": pred_duration,
            "calibrated_duration_prior": calibrated_duration,
            "duration_calibration_multiplier": float(getattr(row, "duration_calibration_multiplier", 1.0)),
            "duration_distance_band": str(getattr(row, "duration_distance_band", "unknown")),
            "duration_borough_pair": str(getattr(row, "duration_borough_pair", "unknown")),
            "true_distance_miles": float(row.observed_avg_distance_miles),
            "predicted_distance_without_same_bin_observation": pred_distance,
            "temporal_weight": float(row.temporal_weight), "spatial_weight": float(row.spatial_weight),
            "global_weight": float(row.global_weight), "gravity_weight": float(row.gravity_weight),
        })
    validation = pd.DataFrame(reconstruct_rows)
    validation.to_csv(PROCESSED_DIR / "imputation_holdout_validation.csv", index=False)

    summary_rows = []
    for metric_name, true_col, pred_col in [
        ("driver_revenue", "true_revenue", "predicted_revenue_without_same_bin_observation"),
        ("duration_min", "true_duration_min", "predicted_duration_without_same_bin_observation"),
        ("distance_miles", "true_distance_miles", "predicted_distance_without_same_bin_observation"),
    ]:
        result = metrics(validation[true_col].to_numpy(float), validation[pred_col].to_numpy(float))
        result.update({"metric": metric_name, "n_holdout_cells": len(validation), "min_observed_trip_count": effective_min_count})
        summary_rows.append(result)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PROCESSED_DIR / "imputation_holdout_summary.csv", index=False)

    # Explicit tail diagnostics show whether long-duration cells remain systematically low.
    if not validation.empty:
        validation["duration_truth_quartile"] = pd.qcut(
            validation["true_duration_min"], q=4, labels=["Q1_short", "Q2", "Q3", "Q4_long"], duplicates="drop"
        )
        tail_rows = []
        for label, group in validation.groupby("duration_truth_quartile", observed=True):
            result = metrics(
                group["true_duration_min"].to_numpy(float),
                group["predicted_duration_without_same_bin_observation"].to_numpy(float),
            )
            result.update({"duration_group": str(label), "n": len(group)})
            tail_rows.append(result)
        pd.DataFrame(tail_rows).to_csv(PROCESSED_DIR / "duration_tail_validation_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Saved: {PROCESSED_DIR / 'imputation_holdout_validation.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'imputation_holdout_summary.csv'}")


if __name__ == "__main__":
    main()
