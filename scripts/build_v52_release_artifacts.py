"""Build a compact v5.2 release summary and a v5.1-to-v5.2 change table."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MEAN_COLUMN = "mean_primary_operating_net_earnings_2h"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def hard_axiom_failures(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    hard = frame[frame["hard_axiom"].astype(str).str.lower().eq("true")]
    return hard[hard["status"].astype(str).eq("FAIL")]["test"].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--old-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    new_root = Path(args.new_root).resolve()
    old_root = Path(args.old_root).resolve() if args.old_root else None
    output = Path(args.output_dir).resolve() if args.output_dir else new_root / "results"
    output.mkdir(parents=True, exist_ok=True)

    acceptance = read_json(new_root / "results" / "final_experiment_acceptance.json")
    date_fix = read_json(new_root / "processed_data" / "date_boundary_fix_validation.json")
    calibration = read_json(new_root / "results" / "fare_share_calibration.json")
    algorithms = pd.read_csv(new_root / "processed_data" / "algorithm_cost_model_comparison.csv")
    algorithms = algorithms.sort_values(MEAN_COLUMN, ascending=False).reset_index(drop=True)
    paired = pd.read_csv(new_root / "processed_data" / "paired_cost_model_comparisons.csv")
    path_values = pd.read_csv(new_root / "processed_data" / "path_shapley_values.csv")
    global_failures = hard_axiom_failures(new_root / "processed_data" / "axiom_validation_results.csv")
    path_failures = hard_axiom_failures(new_root / "processed_data" / "path_axiom_validation_results.csv")

    algorithm_change = None
    paired_change = None
    old_available = False
    if old_root is not None:
        old_alg_path = old_root / "processed_data" / "algorithm_cost_model_comparison.csv"
        old_pair_path = old_root / "processed_data" / "paired_cost_model_comparisons.csv"
        if old_alg_path.is_file() and old_pair_path.is_file():
            old_available = True
            old_algorithms = pd.read_csv(old_alg_path)[["algorithm", MEAN_COLUMN]].rename(
                columns={MEAN_COLUMN: "v5_1_mean_operating_net_earnings_2h"}
            )
            new_algorithms = algorithms[["algorithm", MEAN_COLUMN]].rename(
                columns={MEAN_COLUMN: "v5_2_mean_operating_net_earnings_2h"}
            )
            algorithm_change = old_algorithms.merge(new_algorithms, on="algorithm", how="outer")
            algorithm_change["absolute_change_usd_2h"] = (
                algorithm_change["v5_2_mean_operating_net_earnings_2h"]
                - algorithm_change["v5_1_mean_operating_net_earnings_2h"]
            )
            algorithm_change["percent_change"] = 100.0 * algorithm_change[
                "absolute_change_usd_2h"
            ] / algorithm_change["v5_1_mean_operating_net_earnings_2h"].abs().clip(lower=1e-12)
            algorithm_change = algorithm_change.sort_values(
                "v5_2_mean_operating_net_earnings_2h", ascending=False
            )
            algorithm_change.to_csv(output / "v51_v52_algorithm_comparison.csv", index=False)

            old_paired = pd.read_csv(old_pair_path)[
                ["algorithm", "baseline", "mean_paired_gain", "paired_ci_lower", "paired_ci_upper"]
            ].rename(
                columns={
                    "mean_paired_gain": "v5_1_mean_paired_gain",
                    "paired_ci_lower": "v5_1_ci_lower",
                    "paired_ci_upper": "v5_1_ci_upper",
                }
            )
            new_paired = paired[
                ["algorithm", "baseline", "mean_paired_gain", "paired_ci_lower", "paired_ci_upper"]
            ].rename(
                columns={
                    "mean_paired_gain": "v5_2_mean_paired_gain",
                    "paired_ci_lower": "v5_2_ci_lower",
                    "paired_ci_upper": "v5_2_ci_upper",
                }
            )
            paired_change = old_paired.merge(new_paired, on=["algorithm", "baseline"], how="outer")
            paired_change["paired_gain_change"] = (
                paired_change["v5_2_mean_paired_gain"] - paired_change["v5_1_mean_paired_gain"]
            )
            paired_change.to_csv(output / "v51_v52_paired_comparison.csv", index=False)

    top = algorithms.iloc[0]
    comparison = {
        "old_results_available": old_available,
        "algorithm_changes": json_records(algorithm_change) if algorithm_change is not None else [],
        "paired_changes": json_records(paired_change) if paired_change is not None else [],
    }
    (output / "v51_v52_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    summary = {
        "status": "V5.2 RELEASE ARTIFACTS PASSED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_version": "5.2.0",
        "formal_acceptance": acceptance.get("status"),
        "date_boundary_validation": date_fix.get("status"),
        "service_days": date_fix.get("details", {}).get("environment_service_days"),
        "excluded_out_of_month_clean_rows": date_fix.get("details", {}).get(
            "excluded_out_of_month_clean_rows"
        ),
        "top_algorithm": str(top["algorithm"]),
        "top_mean_operating_net_earnings_2h": float(top[MEAN_COLUMN]),
        "algorithm_results": json_records(
            algorithms[["algorithm", MEAN_COLUMN, "ci_lower", "ci_upper"]]
        ),
        "paired_vs_wait_only": json_records(paired),
        "global_hard_axiom_failures": global_failures,
        "path_hard_axiom_failures": path_failures,
        "negative_path_occurrence_count": int((path_values["shapley_value"] < -1e-9).sum()),
        "configured_driver_revenue_share": calibration.get("configured_driver_revenue_share"),
        "derived_driver_revenue_share": calibration.get("derived_effective_driver_revenue_share"),
        "v51_comparison_available": old_available,
    }
    if acceptance.get("status") != "FINAL EXPERIMENT ACCEPTANCE PASSED":
        raise SystemExit("Cannot publish release summary: final acceptance did not pass")
    (output / "v52_release_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    rows = [
        "# MultiTaxi-Bench v5.2 final results",
        "",
        "v5.2 supersedes the v5.1 formal results after correcting the monthly pickup-date "
        "boundary and the 194-versus-181 service-day exposure defect.",
        "",
        f"- Formal acceptance: `{summary['formal_acceptance']}`",
        f"- Date-boundary validation: `{summary['date_boundary_validation']}`",
        f"- Service days: **{summary['service_days']}**",
        f"- Excluded out-of-month quality-clean trips: **{summary['excluded_out_of_month_clean_rows']}**",
        f"- Highest sample mean: **{summary['top_algorithm']}**, "
        f"{summary['top_mean_operating_net_earnings_2h']:.4f} USD/2h",
        "",
        "## Seven-policy results",
        "",
        "| Rank | Algorithm | Mean operating net earnings (USD/2h) | 95% CI |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(algorithms.itertuples(index=False), start=1):
        rows.append(
            f"| {rank} | {row.algorithm} | {getattr(row, MEAN_COLUMN):.4f} | "
            f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}] |"
        )
    rows.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "Paired differences versus Wait Only must be interpreted from their paired 95% "
            "confidence intervals; a higher sample mean alone is not evidence of statistical superiority.",
            "",
            "## Validation",
            "",
            f"- Global hard-axiom failures: {len(global_failures)}",
            f"- Path hard-axiom failures: {len(path_failures)}",
            f"- Signed negative path occurrences retained: {summary['negative_path_occurrence_count']}",
            "- The 120-minute task, seven policies, cost model, scenario bank design, and both "
            "Shapley game definitions are unchanged from the approved research contract.",
        ]
    )
    (output / "FINAL_RESULTS_V52.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
