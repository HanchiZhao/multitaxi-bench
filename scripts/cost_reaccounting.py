"""Re-account V4 policy episodes under all explicit V5.1 cost models.

The script first validates the active V4 accounting identity. It then recovers raw
modeled receipts by reversing the active revenue share and applies each alternative
cost definition once. Fare-share disclosure is never subtracted a second time.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, ensure_directories
from cost_models import (
    all_cost_models,
    load_yaml,
    operating_net_earnings,
    primary_cost_model,
    raw_receipts_from_effective,
)


AUDIT_TOLERANCE = 1e-7


def summarize(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for algorithm, group in df.groupby("algorithm", sort=False):
        values = group[value_col].astype(float)
        n = len(values)
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        se = std / math.sqrt(max(1, n))
        mean = float(values.mean())
        rows.append(
            {
                "algorithm": algorithm,
                "episodes": n,
                "mean_primary_operating_net_earnings_2h": mean,
                "std_primary_operating_net_earnings_2h": std,
                "ci_lower": mean - 1.96 * se,
                "ci_upper": mean + 1.96 * se,
                "median_primary_operating_net_earnings_2h": float(
                    values.median()
                ),
                "worst_10_percent": float(values.quantile(0.10)),
                "mean_raw_driver_receipts_2h": float(
                    group["raw_driver_receipts_2h"].mean()
                ),
                "mean_total_miles": float(
                    (group["occupied_miles"] + group["empty_miles"]).mean()
                ),
                "mean_completed_trips": float(group["completed_trips"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        "mean_primary_operating_net_earnings_2h", ascending=False
    )
    out = out.reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def _runtime_objective(cfg: dict, models: dict):
    runtime = cfg.get("runtime", {})
    required = {
        "active_cost_model",
        "active_revenue_share",
        "active_occupied_cost_per_mile",
        "active_empty_cost_per_mile",
        "active_fixed_lease_per_hour",
    }
    missing = sorted(required - set(runtime))
    if missing:
        raise SystemExit(
            "Resolved v5.2 config is missing runtime accounting fields: "
            f"{missing}. Run scripts/run_v52.py; do not re-account an ambiguous "
            "policy_episode_results.csv directly from paper_main.yaml."
        )
    active_name = str(runtime["active_cost_model"])
    if active_name not in models:
        raise SystemExit(f"Unknown runtime active_cost_model={active_name!r}")
    active = models[active_name]
    recorded = {
        "revenue_share": float(runtime["active_revenue_share"]),
        "occupied_cost_per_mile": float(
            runtime["active_occupied_cost_per_mile"]
        ),
        "empty_cost_per_mile": float(runtime["active_empty_cost_per_mile"]),
        "fixed_lease_per_hour": float(runtime["active_fixed_lease_per_hour"]),
    }
    for field, value in recorded.items():
        if abs(float(getattr(active, field)) - value) > AUDIT_TOLERANCE:
            raise SystemExit(
                f"Runtime {field}={value} disagrees with active model "
                f"{active_name}.{field}={getattr(active, field)}"
            )
    if active.fixed_lease_per_hour > AUDIT_TOLERANCE:
        raise SystemExit(
            "The preserved V4 objective does not propagate fixed hourly lease cost; "
            "it cannot be the active primary model."
        )
    return active


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    ensure_directories()
    cfg = load_yaml(args.config)
    models = all_cost_models(cfg)
    primary = primary_cost_model(cfg)
    active = _runtime_objective(cfg, models)

    episode_path = PROCESSED_DIR / "policy_episode_results.csv"
    if not episode_path.exists():
        raise FileNotFoundError(f"Missing {episode_path}; run V4 evaluation first")
    df = pd.read_csv(episode_path)
    required_columns = {
        "algorithm",
        "scenario_id",
        "gross_revenue_2h",
        "operating_cost_2h",
        "net_earnings_2h",
        "occupied_miles",
        "empty_miles",
        "horizon_minutes",
        "completed_trips",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise SystemExit(f"Episode table is missing columns: {missing_columns}")

    # Audit the objective actually used by V4 before reversing revenue share.
    expected_active_cost = (
        df["occupied_miles"].astype(float) * active.occupied_cost_per_mile
        + df["empty_miles"].astype(float) * active.empty_cost_per_mile
    )
    active_cost_gap = (
        expected_active_cost - df["operating_cost_2h"].astype(float)
    ).abs()
    active_net_gap = (
        df["gross_revenue_2h"].astype(float)
        - df["operating_cost_2h"].astype(float)
        - df["net_earnings_2h"].astype(float)
    ).abs()

    df["raw_driver_receipts_2h"] = [
        raw_receipts_from_effective(value, active.revenue_share)
        for value in df["gross_revenue_2h"].astype(float)
    ]
    raw_roundtrip_gap = (
        df["raw_driver_receipts_2h"] * active.revenue_share
        - df["gross_revenue_2h"].astype(float)
    ).abs()

    identity_gaps = []
    for name, model in models.items():
        values = [
            operating_net_earnings(
                row.raw_driver_receipts_2h,
                row.occupied_miles,
                row.empty_miles,
                row.horizon_minutes,
                model,
            )
            for row in df.itertuples(index=False)
        ]
        df[f"{name}_effective_receipts_2h"] = [
            value["effective_driver_receipts"] for value in values
        ]
        df[f"{name}_mileage_cost_2h"] = [
            value["mileage_cost"] for value in values
        ]
        df[f"{name}_fixed_lease_cost_2h"] = [
            value["fixed_lease_cost"] for value in values
        ]
        df[f"{name}_disclosed_fare_share_cost_2h"] = [
            value["disclosed_fare_share_cost"] for value in values
        ]
        canonical = f"{name}_operating_net_earnings_2h"
        df[canonical] = [value["operating_net_earnings"] for value in values]
        identity_gaps.extend(
            abs(value["fare_share_identity_gap"]) for value in values
        )

    primary_col = f"{primary.name}_operating_net_earnings_2h"
    df["primary_cost_model"] = primary.name
    df["primary_operating_net_earnings_2h"] = df[primary_col]

    # Because primary is the active propagated objective, re-accounting it must
    # reproduce V4 episode net earnings exactly. This catches both missed and double
    # fare-share deductions.
    primary_reaccount_gap = (
        df["primary_operating_net_earnings_2h"]
        - df["net_earnings_2h"].astype(float)
    ).abs()

    audit = {
        "status": "COST ACCOUNTING PASSED",
        "active_cost_model": active.name,
        "primary_cost_model": primary.name,
        "rows": int(len(df)),
        "max_active_mileage_cost_gap": float(active_cost_gap.max()),
        "max_active_net_identity_gap": float(active_net_gap.max()),
        "max_raw_receipt_roundtrip_gap": float(raw_roundtrip_gap.max()),
        "max_fare_share_identity_gap": float(max(identity_gaps, default=0.0)),
        "max_primary_reaccounting_gap": float(primary_reaccount_gap.max()),
        "fare_share_deduction_count": 1,
        "fare_share_disclosure_subtracted_again": False,
        "tolerance": AUDIT_TOLERANCE,
    }
    failed = {
        key: value
        for key, value in audit.items()
        if key.startswith("max_") and float(value) > AUDIT_TOLERANCE
    }
    if failed:
        audit["status"] = "COST ACCOUNTING FAILED"
        audit["failed"] = failed
    (PROCESSED_DIR / "cost_accounting_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if failed:
        raise SystemExit(json.dumps(audit, indent=2))

    df.to_csv(PROCESSED_DIR / "policy_episode_cost_models.csv", index=False)
    summary = summarize(df, "primary_operating_net_earnings_2h")
    summary.to_csv(
        PROCESSED_DIR / "algorithm_cost_model_comparison.csv", index=False
    )

    # Paired common-scenario comparison under the configured primary model.
    pivot = df.pivot_table(
        index="scenario_id",
        columns="algorithm",
        values="primary_operating_net_earnings_2h",
        aggfunc="first",
    )
    baseline = "wait_only"
    rows = []
    if baseline in pivot.columns:
        for algorithm in pivot.columns:
            if algorithm == baseline:
                continue
            values = (pivot[algorithm] - pivot[baseline]).dropna()
            n = len(values)
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            se = std / math.sqrt(max(1, n))
            rows.append(
                {
                    "algorithm": algorithm,
                    "baseline": baseline,
                    "paired_scenarios": n,
                    "mean_paired_gain": float(values.mean()),
                    "paired_ci_lower": float(values.mean() - 1.96 * se),
                    "paired_ci_upper": float(values.mean() + 1.96 * se),
                    "win_rate": float((values > 1e-9).mean()),
                    "tie_rate": float((values.abs() <= 1e-9).mean()),
                    "loss_rate": float((values < -1e-9).mean()),
                }
            )
    pd.DataFrame(rows).to_csv(
        PROCESSED_DIR / "paired_cost_model_comparisons.csv", index=False
    )

    # Event-level fields for representative trajectories. Under the byte-preserved V4
    # contract, episode occupied_miles and occupied operating cost cover completed trips;
    # UNFINISHED_TRIP contributes in-horizon time but no completed-trip revenue or episode
    # mileage cost. Keep V5.1 event accounting identical to that preserved definition.
    trajectory_path = PROCESSED_DIR / "policy_trajectories.csv"
    if trajectory_path.exists():
        trajectories = pd.read_csv(trajectory_path)
        trajectories["raw_revenue"] = (
            trajectories["revenue"].astype(float) / active.revenue_share
        )
        is_occupied = trajectories["event_type"].astype(str).eq(
            "OCCUPIED_TRIP"
        )
        is_empty = trajectories["event_type"].astype(str).eq(
            "EMPTY_REPOSITION"
        )
        for name, model in models.items():
            trajectories[f"{name}_effective_revenue"] = (
                trajectories["raw_revenue"] * model.revenue_share
            )
            trajectories[f"{name}_movement_cost"] = np.where(
                is_occupied,
                trajectories["distance_miles"] * model.occupied_cost_per_mile,
                np.where(
                    is_empty,
                    trajectories["distance_miles"] * model.empty_cost_per_mile,
                    0.0,
                ),
            )
            canonical = f"{name}_event_operating_net_earnings"
            trajectories[canonical] = (
                trajectories[f"{name}_effective_revenue"]
                - trajectories[f"{name}_movement_cost"]
            )
        trajectories.to_csv(
            PROCESSED_DIR / "policy_trajectories_cost_models.csv", index=False
        )

    print(json.dumps(audit, indent=2))
    print(summary.to_string(index=False))
    print("Cost re-accounting complete.")


if __name__ == "__main__":
    main()
