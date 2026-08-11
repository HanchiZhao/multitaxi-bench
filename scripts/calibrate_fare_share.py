"""Verify the effective fare-share rate against the locked 2025H1 TLC data.

TLC's fare-share lease cap is defined on farebox revenue and excludes tips. V4
intentionally stores fare plus tip in one modeled-receipt field. This read-only check
applies the preserved V4 cleaning rules to the six locked monthly Parquet files and
converts a farebox-only withholding into its effective share of fare+tip receipts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from config import (
    DATA_DIR,
    MAX_DISTANCE_MILES,
    MAX_DURATION_MIN,
    MAX_FARE,
    MIN_DISTANCE_MILES,
    MIN_DURATION_MIN,
    MIN_FARE,
    MONTHS,
    RESULTS_DIR,
    ZONE_SHP,
)
from cost_models import load_yaml, primary_cost_model
from month_boundaries import month_bounds, pickup_in_declared_month


REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
    "tip_amount",
]


def month_totals(
    path: Path, valid_zones: set[int], month: str
) -> dict[str, float]:
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    missing = sorted(set(REQUIRED_COLUMNS[:-1]) - available)
    if missing:
        raise SystemExit(f"{path.name} is missing columns: {missing}")
    columns = [column for column in REQUIRED_COLUMNS if column in available]
    clean_rows = 0
    clean_rows_before_month_filter = 0
    out_of_month_clean_rows_excluded = 0
    retained_out_of_month_rows = 0
    fare_sum = 0.0
    tip_sum = 0.0
    for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
        frame = batch.to_pandas()
        if "tip_amount" not in frame:
            frame["tip_amount"] = 0.0
        frame["tpep_pickup_datetime"] = pd.to_datetime(
            frame["tpep_pickup_datetime"], errors="coerce"
        )
        frame["tpep_dropoff_datetime"] = pd.to_datetime(
            frame["tpep_dropoff_datetime"], errors="coerce"
        )
        for column in (
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "fare_amount",
            "tip_amount",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(
            subset=[
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "trip_distance",
                "fare_amount",
            ]
        )
        frame["PULocationID"] = frame["PULocationID"].astype(int)
        frame["DOLocationID"] = frame["DOLocationID"].astype(int)
        duration = (
            frame["tpep_dropoff_datetime"] - frame["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60.0
        keep = (
            frame["PULocationID"].isin(valid_zones)
            & frame["DOLocationID"].isin(valid_zones)
            & frame["fare_amount"].between(MIN_FARE, MAX_FARE)
            & frame["trip_distance"].between(
                MIN_DISTANCE_MILES, MAX_DISTANCE_MILES
            )
            & duration.between(MIN_DURATION_MIN, MAX_DURATION_MIN)
        )
        quality_clean = frame.loc[keep].copy()
        clean_rows_before_month_filter += int(len(quality_clean))
        in_month = pickup_in_declared_month(
            quality_clean["tpep_pickup_datetime"], month
        )
        out_of_month_clean_rows_excluded += int((~in_month).sum())
        clean = quality_clean.loc[in_month]
        retained_out_of_month_rows += int(
            (~pickup_in_declared_month(clean["tpep_pickup_datetime"], month)).sum()
        )
        clean_rows += int(len(clean))
        fare_sum += float(clean["fare_amount"].clip(lower=0.0).sum())
        tip_sum += float(
            clean["tip_amount"].fillna(0.0).clip(lower=0.0).sum()
        )
    return {
        "clean_rows": clean_rows,
        "clean_rows_before_month_filter": clean_rows_before_month_filter,
        "out_of_month_clean_rows_excluded": out_of_month_clean_rows_excluded,
        "retained_out_of_month_rows": retained_out_of_month_rows,
        "fare_sum": fare_sum,
        "tip_sum": tip_sum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tolerance", type=float, default=0.002)
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "fare_share_calibration.json"),
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    model = primary_cost_model(config)
    if model.source_farebox_share_rate <= 0.0:
        raise SystemExit(
            "Primary model has no source_farebox_share_rate to calibrate"
        )
    if not model.tips_retained_fully:
        raise SystemExit("This calibration requires tips_retained_fully=true")

    zones = gpd.read_file(ZONE_SHP)
    if "LocationID" not in zones.columns:
        raise SystemExit(f"{ZONE_SHP} has no LocationID field")
    valid_zones = set(zones["LocationID"].astype(int))

    monthly = []
    for month in MONTHS:
        path = DATA_DIR / f"yellow_tripdata_{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        totals = month_totals(path, valid_zones, month)
        start, end = month_bounds(month)
        monthly.append(
            {
                "month": month,
                "expected_start_inclusive": start.isoformat(),
                "expected_end_exclusive": end.isoformat(),
                **totals,
            }
        )

    fare = sum(row["fare_sum"] for row in monthly)
    tips = sum(row["tip_sum"] for row in monthly)
    modeled_receipts = fare + tips
    if modeled_receipts <= 0.0:
        raise SystemExit("No cleaned modeled receipts were found")

    effective_withheld = (
        model.source_farebox_share_rate * fare / modeled_receipts
    )
    effective_retained = 1.0 - effective_withheld
    share_gap = abs(model.revenue_share - effective_retained)
    disclosed_gap = abs(model.fare_share_rate - effective_withheld)
    retained_out_of_month = int(
        sum(row["retained_out_of_month_rows"] for row in monthly)
    )
    passed = (
        share_gap <= args.tolerance
        and disclosed_gap <= args.tolerance
        and retained_out_of_month == 0
    )
    report = {
        "status": "FARE SHARE CALIBRATION PASSED"
        if passed
        else "FARE SHARE CALIBRATION FAILED",
        "months": MONTHS,
        "monthly": monthly,
        "clean_rows": int(sum(row["clean_rows"] for row in monthly)),
        "clean_rows_before_month_filter": int(
            sum(row["clean_rows_before_month_filter"] for row in monthly)
        ),
        "out_of_month_clean_rows_excluded": int(
            sum(row["out_of_month_clean_rows_excluded"] for row in monthly)
        ),
        "retained_out_of_month_rows": retained_out_of_month,
        "clean_fare_sum": fare,
        "clean_tip_sum": tips,
        "tip_share_of_modeled_receipts": tips / modeled_receipts,
        "source_farebox_share_rate": model.source_farebox_share_rate,
        "derived_effective_withheld_share_of_fare_plus_tip": effective_withheld,
        "derived_effective_driver_revenue_share": effective_retained,
        "configured_fare_share_rate": model.fare_share_rate,
        "configured_driver_revenue_share": model.revenue_share,
        "configured_revenue_share_gap": share_gap,
        "configured_fare_share_gap": disclosed_gap,
        "tolerance": args.tolerance,
        "definition": (
            "35% of cleaned farebox revenue, excluding tips; all cleaned observed "
            "tips retained by driver; converted to a single V4-compatible share of "
            "fare+tip modeled receipts"
        ),
        "date_boundary_definition": (
            "declared_month_start <= pickup datetime < next_month_start; "
            "dropoff datetime is not month-restricted"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
