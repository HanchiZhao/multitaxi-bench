"""Validate arbitrary origin-destination-time coverage.

The compact passenger-transition table does not enumerate all 6.6M possible
96 × origin × destination cells. DynamicTaxiEnvironment.estimate_od() must nevertheless
return a finite estimate, confidence and provenance for every valid query.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from config import OD_COVERAGE_TEST_PAIRS, PROCESSED_DIR, RANDOM_SEED, ensure_directories
from two_hour_environment import DynamicTaxiEnvironment, format_clock_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Check arbitrary OD-time estimate coverage")
    parser.add_argument("--sample", type=int, default=OD_COVERAGE_TEST_PAIRS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    ensure_directories()
    env = DynamicTaxiEnvironment.load()
    rng = np.random.default_rng(args.seed)
    rows = []
    failures = []
    for i in range(max(1, int(args.sample))):
        origin = int(rng.choice(env.zone_ids))
        destination = int(rng.choice(env.zone_ids))
        minute = float(rng.uniform(0.0, 1440.0))
        try:
            estimate = env.estimate_od(origin, destination, minute)
            numeric = [
                "expected_driver_revenue", "expected_duration_minutes",
                "expected_distance_miles", "revenue_std", "duration_std",
                "distance_std", "confidence", "observed_trip_count",
            ]
            finite = all(np.isfinite(float(estimate[k])) for k in numeric)
            valid = (
                finite
                and float(estimate["expected_driver_revenue"]) >= 0
                and float(estimate["expected_duration_minutes"]) > 0
                and float(estimate["expected_distance_miles"]) > 0
                and 0 < float(estimate["confidence"]) <= 1
            )
            row = dict(estimate)
            row["query_index"] = i
            row["valid"] = bool(valid)
            rows.append(row)
            if not valid:
                failures.append(f"invalid estimate {origin}->{destination} at {format_clock_time(minute)}")
        except Exception as exc:  # validation must report every failure instead of stopping early
            failures.append(f"{origin}->{destination} at {format_clock_time(minute)}: {exc}")

    detail = pd.DataFrame(rows)
    detail.to_csv(PROCESSED_DIR / "full_od_coverage_samples.csv", index=False)
    source_counts = detail["estimate_source"].value_counts(dropna=False) if not detail.empty else pd.Series(dtype=int)
    summary = pd.DataFrame([
        {
            "queries_requested": int(args.sample),
            "queries_returned": int(len(detail)),
            "valid_queries": int(detail["valid"].sum()) if not detail.empty else 0,
            "failed_queries": int(len(failures) + ((~detail["valid"]).sum() if not detail.empty else 0)),
            "coverage_rate": float(detail["valid"].mean()) if not detail.empty else 0.0,
            "mean_confidence": float(detail["confidence"].mean()) if not detail.empty else np.nan,
            "directly_observed_share": float(detail["is_directly_observed"].mean()) if not detail.empty else np.nan,
            "estimate_source_counts": "|".join(f"{k}:{int(v)}" for k, v in source_counts.items()),
            "failure_examples": " || ".join(failures[:10]),
        }
    ])
    summary.to_csv(PROCESSED_DIR / "full_od_coverage_summary.csv", index=False)
    print(summary.to_string(index=False))
    if failures or (not detail.empty and not detail["valid"].all()):
        raise RuntimeError("Arbitrary OD coverage validation found invalid queries; inspect full_od_coverage_samples.csv")
    print(f"All {len(detail):,} sampled arbitrary OD-time queries returned valid estimates.")


if __name__ == "__main__":
    main()
