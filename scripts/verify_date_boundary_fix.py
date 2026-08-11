"""Validate the approved v5.2 pickup-month boundary repair end to end."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import MONTHS, PROCESSED_DIR, RESULTS_DIR
from month_boundaries import expected_service_days
from two_hour_environment import DynamicTaxiEnvironment


EXPECTED_LOCKED_EXCLUDED_ROWS = 134
EXPECTED_SCHEMA = "4.1-date-boundary-fix"


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(PROCESSED_DIR / "date_boundary_fix_validation.json"),
    )
    args = parser.parse_args()

    gate = read_json(PROCESSED_DIR / "data_date_boundary_gate.json")
    lock = read_json(PROCESSED_DIR / "data_lock_verification.json")
    calibration = read_json(RESULTS_DIR / "fare_share_calibration.json")
    summary = pd.read_csv(PROCESSED_DIR / "environment_build_summary.csv").iloc[0]
    environment = DynamicTaxiEnvironment.load(
        PROCESSED_DIR / "dynamic_environment.pkl"
    )

    expected_days = expected_service_days(MONTHS)
    checks = {
        "strict_data_lock_passed": (
            lock.get("status") == "DATA LOCK VERIFICATION PASSED"
            and lock.get("strict_lock") is True
            and len(lock.get("verified_files", {})) == 12
        ),
        "date_gate_passed": gate.get("status") == "DATE BOUNDARY FIX PASSED",
        "expected_service_days_is_181": expected_days == 181,
        "gate_service_days_is_181": (
            int(gate.get("retained_unique_service_days", -1)) == expected_days
        ),
        "environment_service_days_is_181": (
            int(summary.get("service_days", -1)) == expected_days
            and int(environment.metadata.get("service_days", -1)) == expected_days
        ),
        "all_134_locked_outliers_excluded": (
            int(gate.get("clean_rows_out_of_month_excluded", -1))
            == EXPECTED_LOCKED_EXCLUDED_ROWS
        ),
        "no_out_of_month_rows_retained": (
            int(gate.get("retained_out_of_month_rows", -1)) == 0
            and int(summary.get("retained_out_of_month_rows", -1)) == 0
            and int(calibration.get("retained_out_of_month_rows", -1)) == 0
        ),
        "fare_share_uses_same_boundary": (
            calibration.get("status") == "FARE SHARE CALIBRATION PASSED"
            and int(calibration.get("out_of_month_clean_rows_excluded", -1))
            == EXPECTED_LOCKED_EXCLUDED_ROWS
        ),
        "environment_schema_is_date_fixed": (
            str(summary.get("environment_schema_version")) == EXPECTED_SCHEMA
            and str(environment.metadata.get("environment_schema_version"))
            == EXPECTED_SCHEMA
        ),
        "pickup_only_boundary_documented": (
            "dropoff datetime is not month-restricted"
            in str(gate.get("filter_definition", ""))
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": (
            "V5.2 DATE BOUNDARY VALIDATION PASSED"
            if not failed
            else "V5.2 DATE BOUNDARY VALIDATION FAILED"
        ),
        "project_version": "5.2.0",
        "checks": checks,
        "failed": failed,
        "details": {
            "months": list(MONTHS),
            "expected_service_days": expected_days,
            "environment_service_days": int(summary.get("service_days", -1)),
            "excluded_out_of_month_clean_rows": int(
                gate.get("clean_rows_out_of_month_excluded", -1)
            ),
            "configured_driver_revenue_share": calibration.get(
                "configured_driver_revenue_share"
            ),
            "derived_driver_revenue_share": calibration.get(
                "derived_effective_driver_revenue_share"
            ),
            "environment_schema_version": environment.metadata.get(
                "environment_schema_version"
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
