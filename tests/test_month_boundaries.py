from datetime import datetime
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from month_boundaries import expected_service_days, month_bounds, pickup_in_declared_month


def test_half_open_pickup_month_boundary() -> None:
    values = pd.Series(
        pd.to_datetime(
            [
                "2024-12-31 23:59:59",
                "2025-01-01 00:00:00",
                "2025-01-31 23:59:59",
                "2025-02-01 00:00:00",
            ]
        )
    )
    assert pickup_in_declared_month(values, "2025-01").tolist() == [
        False,
        True,
        True,
        False,
    ]


def test_january_to_june_2025_has_181_days() -> None:
    months = [f"2025-{month:02d}" for month in range(1, 7)]
    assert expected_service_days(months) == 181


def test_dropoff_is_not_part_of_pickup_boundary() -> None:
    pickup = pd.Series(pd.to_datetime(["2025-01-31 23:59:50"]))
    dropoff = datetime.fromisoformat("2025-02-01 00:10:00")
    assert bool(pickup_in_declared_month(pickup, "2025-01").iloc[0])
    assert dropoff.month == 2


def test_month_bounds_are_half_open() -> None:
    start, end = month_bounds("2025-02")
    assert start.isoformat() == "2025-02-01T00:00:00"
    assert end.isoformat() == "2025-03-01T00:00:00"
