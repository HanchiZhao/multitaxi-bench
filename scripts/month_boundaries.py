"""Shared pickup-month boundary helpers for the audited 2025H1 dataset.

Every monthly TLC file is treated as a half-open pickup-time interval:
``month_start <= pickup_time < next_month_start``. Drop-off timestamps are not
restricted because a trip picked up near midnight may legitimately end in the next
month.
"""
from __future__ import annotations

import calendar
from datetime import date
from typing import Iterable

import pandas as pd


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the inclusive start and exclusive end for a strict ``YYYY-MM`` label."""
    try:
        start = pd.Timestamp(f"{month}-01")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid month label {month!r}; expected YYYY-MM") from exc
    if start.strftime("%Y-%m") != str(month):
        raise ValueError(f"Invalid month label {month!r}; expected YYYY-MM")
    return start, start + pd.offsets.MonthBegin(1)


def pickup_in_declared_month(
    pickup_times: pd.Series, month: str
) -> pd.Series:
    """Return a boolean mask for pickups inside the file's declared month."""
    start, end = month_bounds(month)
    return pickup_times.ge(start) & pickup_times.lt(end)


def expected_calendar_days(month: str) -> int:
    start, _ = month_bounds(month)
    return int(calendar.monthrange(start.year, start.month)[1])


def expected_service_days(months: Iterable[str]) -> int:
    """Return the non-overlapping calendar exposure for the declared months."""
    return int(sum(expected_calendar_days(month) for month in months))


def declared_dates(month: str) -> set[date]:
    start, end = month_bounds(month)
    return set(pd.date_range(start, end - pd.Timedelta(days=1), freq="D").date)
