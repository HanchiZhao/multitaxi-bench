"""Machine-check the formal 300-scenario V5.1 experiment deliverables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import pandas as pd

from config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR
from cost_models import load_yaml, primary_cost_model


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
        handle.seek(-12, 2)
        ending = handle.read(12)
    if len(signature) < 24 or signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG: {path}")
    if len(ending) != 12 or ending[4:8] != b"IEND":
        raise ValueError(f"Truncated PNG (missing IEND): {path}")
    return struct.unpack(">II", signature[16:24])


def hard_axiom_failures(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "hard_axiom" not in frame or "status" not in frame:
        raise ValueError(f"Axiom table has unexpected schema: {path}")
    hard_mask = frame["hard_axiom"].astype(str).str.lower().eq("true")
    hard = frame[hard_mask]
    return hard[hard["status"].astype(str).eq("FAIL")]["test"].astype(str).tolist()


def dataframe_records(frame: pd.DataFrame) -> list[dict]:
    """Return JSON-safe records with missing values represented as null."""
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lower", type=float, default=60.0)
    parser.add_argument("--upper", type=float, default=80.0)
    args = parser.parse_args()

    config = load_yaml(args.config)
    primary = primary_cost_model(config)
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    checks["primary_is_fare_share"] = primary.name == "fare_share"
    reproduction = read_json(PROCESSED_DIR / "reproduction_report.json")
    checks["reproduction_passed"] = reproduction.get("status") == "REPRODUCTION PASSED"
    research = read_json(PROCESSED_DIR / "research_contract_report.json")
    checks["research_contract_passed"] = research.get("status") == "RESEARCH CONTRACT PASSED"
    v4 = read_json(PROCESSED_DIR / "v4_preservation_report.json")
    checks["v4_preservation_passed"] = v4.get("status") == "V4 PRESERVATION PASSED"
    cost_audit = read_json(PROCESSED_DIR / "cost_accounting_audit.json")
    checks["cost_accounting_passed"] = cost_audit.get("status") == "COST ACCOUNTING PASSED"
    calibration = read_json(RESULTS_DIR / "fare_share_calibration.json")
    checks["fare_share_calibration_passed"] = calibration.get("status") == "FARE SHARE CALIBRATION PASSED"

    episodes = pd.read_csv(PROCESSED_DIR / "policy_episode_results.csv")
    counts = episodes.groupby("algorithm")["scenario_id"].size()
    checks["all_algorithms_have_300_scenarios"] = bool((counts == 300).all())
    checks["strict_120_minute_accounting"] = bool(
        episodes["time_accounting_error"].abs().max() <= 1e-6
        and (episodes["horizon_minutes"] == 120).all()
    )
    details["scenario_counts"] = {str(key): int(value) for key, value in counts.items()}
    details["max_abs_time_accounting_error"] = float(
        episodes["time_accounting_error"].abs().max()
    )

    summary = pd.read_csv(PROCESSED_DIR / "algorithm_cost_model_comparison.csv")
    value_column = (
        "mean_primary_operating_net_earnings_2h"
        if "mean_primary_operating_net_earnings_2h" in summary
        else "mean_primary_take_home_2h"
    )
    summary = summary.sort_values(value_column, ascending=False)
    top_mean = float(summary.iloc[0][value_column])
    checks["primary_earnings_scale_reasonable"] = args.lower <= top_mean <= args.upper
    details["primary_cost_model"] = primary.name
    details["top_algorithm"] = str(summary.iloc[0]["algorithm"])
    details["top_mean_operating_net_earnings_2h"] = top_mean
    details["earnings_acceptance_window"] = [args.lower, args.upper]
    details["algorithm_means"] = {
        str(row.algorithm): float(getattr(row, value_column))
        for row in summary.itertuples(index=False)
    }

    global_failures = hard_axiom_failures(
        PROCESSED_DIR / "axiom_validation_results.csv"
    )
    path_failures = hard_axiom_failures(
        PROCESSED_DIR / "path_axiom_validation_results.csv"
    )
    checks["global_hard_axioms_passed"] = not global_failures
    checks["path_hard_axioms_passed"] = not path_failures
    details["global_hard_axiom_failures"] = global_failures
    details["path_hard_axiom_failures"] = path_failures

    path_efficiency = pd.read_csv(
        PROCESSED_DIR / "path_shapley_efficiency_checks.csv"
    )
    checks["path_efficiency"] = bool(
        path_efficiency["efficiency_gap"].abs().max() <= 1e-6
    )
    path_values = pd.read_csv(PROCESSED_DIR / "path_shapley_values.csv")
    checks["path_occurrence_ids_unique"] = not bool(
        path_values["node_occurrence_id"].duplicated().any()
    )
    checks["signed_negative_values_preserved"] = bool(
        (path_values["shapley_value"] < -1e-9).any()
    )
    details["negative_path_occurrence_count"] = int(
        (path_values["shapley_value"] < -1e-9).sum()
    )

    diagnostics = pd.read_csv(PROCESSED_DIR / "learning_policy_diagnostics.csv")
    q_rows = diagnostics[diagnostics["algorithm"].astype(str).eq("q_learning")]
    dqn_rows = diagnostics[diagnostics["algorithm"].astype(str).eq("dqn")]
    checks["q_learning_policy_coverage_reasonable"] = bool(
        len(q_rows)
        and float(q_rows.iloc[0]["fallback_action_rate"]) <= 0.05
        and float(q_rows.iloc[0]["unseen_state_rate"]) <= 0.05
    )
    checks["dqn_uses_trained_model"] = bool(
        len(dqn_rows) and float(dqn_rows.iloc[0]["model_action_rate"]) >= 0.99
    )
    details["learning_policy_diagnostics"] = dataframe_records(diagnostics)

    imputation = pd.read_csv(PROCESSED_DIR / "imputation_holdout_summary.csv")
    imputation_numeric = imputation.select_dtypes(include="number")
    checks["imputation_validation_finite"] = bool(
        len(imputation) >= 3
        and not imputation_numeric.isna().any().any()
        and (imputation["spearman_rank_correlation"] > 0.50).all()
    )
    details["imputation_holdout_summary"] = dataframe_records(imputation)

    competition = pd.read_csv(
        PROCESSED_DIR / "competition_sensitivity_results.csv"
    )
    competition_scenarios = set(
        competition["competition_scenario"].astype(str).str.lower()
    )
    checks["competition_sensitivity_complete"] = bool(
        competition_scenarios == {"low", "medium", "high"}
        and competition["mean_net_earnings_2h"].notna().all()
        and competition["max_abs_time_accounting_error"].max() <= 1e-6
    )
    details["competition_scenarios"] = sorted(competition_scenarios)
    details["competition_rows"] = int(len(competition))

    ablation = pd.read_csv(PROCESSED_DIR / "ablation_study_summary.csv")
    baseline_sensitivity = pd.read_csv(
        PROCESSED_DIR / "baseline_sensitivity_summary.csv"
    )
    checks["ablation_and_baseline_sensitivity_present"] = bool(
        len(ablation) > 0 and len(baseline_sensitivity) > 0
    )
    details["ablation_summary"] = dataframe_records(ablation)
    details["baseline_sensitivity_summary"] = dataframe_records(
        baseline_sensitivity
    )

    figure_paths = {
        "bars": FIGURES_DIR / "algorithm_path_shapley_bars.png",
        "map": FIGURES_DIR / "algorithm_conditioned_path_shapley_maps.png",
    }
    dimensions = {}
    figure_ok = True
    for name, path in figure_paths.items():
        try:
            width, height = png_dimensions(path)
            dimensions[name] = [width, height]
            figure_ok = figure_ok and width >= 1800 and height >= 1200
        except Exception as exc:
            dimensions[name] = str(exc)
            figure_ok = False
    checks["bars_and_map_png_generated"] = figure_ok
    details["figure_dimensions"] = dimensions

    paired = pd.read_csv(PROCESSED_DIR / "paired_cost_model_comparisons.csv")
    details["paired_vs_wait_only"] = dataframe_records(paired)
    details["global_axioms"] = dataframe_records(
        pd.read_csv(PROCESSED_DIR / "axiom_validation_results.csv")
    )
    details["path_axioms"] = dataframe_records(
        pd.read_csv(PROCESSED_DIR / "path_axiom_validation_results.csv")
    )

    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "FINAL EXPERIMENT ACCEPTANCE PASSED"
        if not failed
        else "FINAL EXPERIMENT ACCEPTANCE FAILED",
        "checks": checks,
        "failed": failed,
        "details": details,
    }
    output = RESULTS_DIR / "final_experiment_acceptance.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
