"""Verify the canonical v5.2 output set and key numerical identities."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from config import FIGURES_DIR, PROCESSED_DIR
from cost_models import load_yaml, primary_cost_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    model = primary_cost_model(config)

    required = [
        PROCESSED_DIR / "algorithm_recommendation_comparison.csv",
        PROCESSED_DIR / "paired_policy_comparisons.csv",
        PROCESSED_DIR / "node_shapley_values.csv",
        PROCESSED_DIR / "algorithm_cost_model_comparison.csv",
        PROCESSED_DIR / "path_shapley_values.csv",
        PROCESSED_DIR / "path_value_calibration.csv",
        PROCESSED_DIR / "path_axiom_validation_results.csv",
        PROCESSED_DIR / "data_date_boundary_gate.json",
        FIGURES_DIR / "algorithm_net_earnings_comparison.png",
        FIGURES_DIR / "policy_operating_net_earnings_comparison.png",
        FIGURES_DIR / "algorithm_path_shapley_bars.png",
        FIGURES_DIR / "algorithm_conditioned_path_shapley_maps.png",
    ]
    if config.get("run", {}).get("strict_data_lock", False):
        required.extend(
            [
                PROCESSED_DIR / "date_boundary_fix_validation.json",
                PROCESSED_DIR / "data_lock_verification.json",
            ]
        )
    if not config.get("run", {}).get("skip_validation", False):
        required.extend(
            [
                PROCESSED_DIR / "axiom_validation_results.csv",
                PROCESSED_DIR / "imputation_holdout_summary.csv",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("REPRODUCTION FAILED: missing outputs\n" + "\n".join(missing))

    episodes = pd.read_csv(PROCESSED_DIR / "policy_episode_results.csv")
    max_time_error = float(episodes["time_accounting_error"].abs().max())
    if max_time_error > 1e-6:
        raise SystemExit("REPRODUCTION FAILED: time accounting")

    efficiency = pd.read_csv(PROCESSED_DIR / "path_shapley_efficiency_checks.csv")
    max_efficiency_gap = float(efficiency["efficiency_gap"].abs().max())
    if max_efficiency_gap > 1e-6:
        raise SystemExit("REPRODUCTION FAILED: path Shapley efficiency")

    calibration = pd.read_csv(PROCESSED_DIR / "path_value_calibration.csv")
    if len(calibration) and float(calibration["calibration_gap"].abs().max()) > 1e-6:
        raise SystemExit("REPRODUCTION FAILED: path full-value calibration")

    path_values = pd.read_csv(PROCESSED_DIR / "path_shapley_values.csv")
    if path_values["node_occurrence_id"].duplicated().any():
        raise SystemExit("REPRODUCTION FAILED: duplicate path occurrence IDs")
    if not np.isfinite(path_values["shapley_value"].astype(float)).all():
        raise SystemExit("REPRODUCTION FAILED: non-finite path Shapley value")

    report = {
        "status": "REPRODUCTION PASSED",
        "project_version": "5.2.0",
        "primary_cost_model": model.name,
        "algorithms": sorted(episodes["algorithm"].unique().tolist()),
        "max_time_error": max_time_error,
        "max_path_efficiency_gap": max_efficiency_gap,
    }
    (PROCESSED_DIR / "reproduction_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
