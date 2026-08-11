"""Build the 15-minute dynamic taxi environment for two-hour operations.

Required data files:
- data/taxi_zones.shp (+ .shx/.dbf/.prj/.cpg)
- data/taxi_zone_lookup.csv
- data/yellow_tripdata_2025-01.parquet ... 2025-06.parquet

Run from project root:
    python scripts/route_environment.py
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from config import (
    ADJACENCY_BUFFER_FEET,
    DATA_DIR,
    AIRPORT_QUEUE_MULTIPLIER,
    COMPETITION_SCENARIO_MULTIPLIERS,
    DEFAULT_COMPETITION_SCENARIO,
    DRIVER_REVENUE_SHARE,
    DURATION_CALIBRATION_MIN_COUNT,
    DURATION_CALIBRATION_SHRINKAGE,
    DURATION_DISTANCE_BINS_MILES,
    DURATION_MAX_TAIL_MULTIPLIER,
    DURATION_MIN_TAIL_MULTIPLIER,
    FALLBACK_SPEED_MPH,
    GLOBAL_OD_DISCOUNT,
    GRAVITY_PRIOR_STRENGTH,
    MAX_DESTINATIONS_PER_ORIGIN,
    MAX_SPEED_MPH,
    MIN_SPEED_MPH,
    MAX_DISTANCE_MILES,
    MAX_DURATION_MIN,
    MAX_EXPECTED_WAIT_MINUTES,
    MAX_FARE,
    MIN_CONFIDENCE,
    MIN_DISTANCE_MILES,
    MIN_DURATION_MIN,
    MIN_EXPECTED_WAIT_MINUTES,
    MIN_FARE,
    MONTHS,
    NODE_PRIOR_STRENGTH,
    N_TIME_BINS,
    OD_DIRICHLET_STRENGTH,
    OD_OBS_PRIOR_STRENGTH,
    PROCESSED_DIR,
    REFERENCE_WAIT_MINUTES,
    NEIGHBOR_SUPPLY_DIFFUSION,
    SUPPLY_BASE_FLOOR,
    SPATIAL_DECAY_MILES,
    SPATIAL_NEIGHBOR_COUNT,
    TEMPORAL_DECAY_BINS,
    TEMPORAL_WINDOW_BINS,
    TIME_BIN_MINUTES,
    VACANT_DEMAND_ATTRACTION,
    VACANT_HISTORY_BINS,
    VACANT_PICKUP_DEPLETION,
    VACANT_STOCK_PERSISTENCE,
    ZONE_LOOKUP,
    ZONE_SHP,
    ensure_directories,
)
from two_hour_environment import DynamicTaxiEnvironment
from month_boundaries import (
    declared_dates,
    expected_calendar_days,
    expected_service_days,
    month_bounds,
    pickup_in_declared_month,
)

TRIP_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
    "tip_amount",
]


def circular_bin_distance(a: int, b: int) -> int:
    d = abs(int(a) - int(b))
    return min(d, N_TIME_BINS - d)


def weighted_mean(values: Sequence[float], weights: Sequence[float], default: float) -> float:
    vals = []
    wts = []
    for v, w in zip(values, weights):
        if pd.notna(v) and np.isfinite(v) and float(w) > 0:
            vals.append(float(v))
            wts.append(float(w))
    if not wts or sum(wts) <= 0:
        return float(default)
    return float(np.average(vals, weights=wts))


def geometric_weighted_mean(values: Sequence[float], weights: Sequence[float], default: float) -> float:
    """Weighted mean on a log scale for positive, right-skewed durations."""
    vals: List[float] = []
    wts: List[float] = []
    for value, weight in zip(values, weights):
        if pd.notna(value) and np.isfinite(value) and float(value) > 0 and float(weight) > 0:
            vals.append(math.log(float(value)))
            wts.append(float(weight))
    if not wts or sum(wts) <= 0:
        return float(default)
    return float(math.exp(np.average(vals, weights=wts)))


def duration_distance_band(distance_miles: float) -> str:
    edges = list(DURATION_DISTANCE_BINS_MILES)
    d = float(max(0.0, distance_miles))
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= d < high:
            return f"{low:g}-{high:g}"
    return f"{edges[-2]:g}-{edges[-1]:g}"


def _shrunk_ratio(group: pd.DataFrame, global_ratio: float) -> float:
    if group.empty:
        return 1.0
    count = float(group["trip_count"].sum())
    raw = weighted_mean(group["duration_ratio"], group["trip_count"], global_ratio)
    reliability = count / (count + DURATION_CALIBRATION_SHRINKAGE)
    target = reliability * raw + (1.0 - reliability) * global_ratio
    return float(np.clip(target / max(global_ratio, 1e-9), 0.75, 1.35))


def build_duration_calibration(
    observed: pd.DataFrame, centroids_df: pd.DataFrame, time_profiles: pd.DataFrame
) -> Dict[str, object]:
    """Learn residual duration multipliers from observed OD-time cells.

    The base duration is distance / time-bin speed. Multipliers describe persistent residual
    congestion by distance band, borough pair, airport involvement and time bin. Every factor
    is empirically shrunk toward one to avoid unstable sparse-cell corrections.
    """
    zone_meta = centroids_df.set_index("zone_id")[["borough", "zone_name"]].to_dict("index")
    speed_by_bin = dict(zip(time_profiles["time_bin"].astype(int), time_profiles["speed_mph"].astype(float)))
    work = observed[(observed["observed_avg_distance_miles"] > 0) & (observed["observed_avg_duration_min"] > 0)].copy()
    if work.empty:
        return {"global_ratio": 1.0, "distance": {}, "borough_pair": {}, "airport": {}, "time_bin": {}}
    work["structural_duration"] = [
        max(1.0, float(d) / max(MIN_SPEED_MPH, speed_by_bin.get(int(tb), FALLBACK_SPEED_MPH)) * 60.0)
        for d, tb in zip(work["observed_avg_distance_miles"], work["time_bin"])
    ]
    work["duration_ratio"] = (work["observed_avg_duration_min"] / work["structural_duration"]).clip(0.35, 3.0)
    work["distance_band"] = work["observed_avg_distance_miles"].map(duration_distance_band)
    work["origin_borough"] = work["origin"].map(lambda z: str(zone_meta.get(int(z), {}).get("borough", "Unknown")))
    work["destination_borough"] = work["destination"].map(lambda z: str(zone_meta.get(int(z), {}).get("borough", "Unknown")))
    work["borough_pair"] = work["origin_borough"] + "->" + work["destination_borough"]
    work["airport_pair"] = [
        "airport" if ("airport" in str(zone_meta.get(int(o), {}).get("zone_name", "")).lower() or
                      "airport" in str(zone_meta.get(int(d), {}).get("zone_name", "")).lower()) else "non_airport"
        for o, d in zip(work["origin"], work["destination"])
    ]
    global_ratio = weighted_mean(work["duration_ratio"], work["trip_count"], 1.0)
    def factors(column: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, group in work.groupby(column, sort=False):
            if float(group["trip_count"].sum()) >= DURATION_CALIBRATION_MIN_COUNT:
                out[str(key)] = _shrunk_ratio(group, global_ratio)
        return out
    return {
        "global_ratio": float(np.clip(global_ratio, DURATION_MIN_TAIL_MULTIPLIER, DURATION_MAX_TAIL_MULTIPLIER)),
        "distance": factors("distance_band"),
        "borough_pair": factors("borough_pair"),
        "airport": factors("airport_pair"),
        "time_bin": factors("time_bin"),
    }


def calibrated_duration_prior(
    origin: int, destination: int, time_bin: int, distance_miles: float,
    speed_mph: float, zone_meta: Mapping[int, Mapping[str, object]], calibration: Mapping[str, object]
) -> Tuple[float, float, str, str]:
    origin_meta, destination_meta = zone_meta.get(int(origin), {}), zone_meta.get(int(destination), {})
    pair = f"{origin_meta.get('borough', 'Unknown')}->{destination_meta.get('borough', 'Unknown')}"
    airport = "airport" if ("airport" in str(origin_meta.get("zone_name", "")).lower() or
                            "airport" in str(destination_meta.get("zone_name", "")).lower()) else "non_airport"
    band = duration_distance_band(distance_miles)
    multiplier = float(calibration.get("global_ratio", 1.0))
    multiplier *= float(dict(calibration.get("distance", {})).get(band, 1.0))
    multiplier *= float(dict(calibration.get("borough_pair", {})).get(pair, 1.0))
    multiplier *= float(dict(calibration.get("airport", {})).get(airport, 1.0))
    multiplier *= float(dict(calibration.get("time_bin", {})).get(str(int(time_bin)), 1.0))
    multiplier = float(np.clip(multiplier, DURATION_MIN_TAIL_MULTIPLIER, DURATION_MAX_TAIL_MULTIPLIER))
    base = max(1.0, float(distance_miles) / max(MIN_SPEED_MPH, float(speed_mph)) * 60.0)
    return max(1.0, base * multiplier), multiplier, band, pair


def pooled_std(sum_x: float, sum_x2: float, n: float) -> float:
    if n <= 1:
        return 0.0
    mean = sum_x / n
    variance = max(0.0, sum_x2 / n - mean * mean)
    return float(math.sqrt(variance))


def read_zone_data() -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    if not ZONE_SHP.exists():
        raise FileNotFoundError(f"Missing shapefile: {ZONE_SHP}")
    if not ZONE_LOOKUP.exists():
        raise FileNotFoundError(f"Missing lookup: {ZONE_LOOKUP}")
    zones = gpd.read_file(ZONE_SHP)
    lookup = pd.read_csv(ZONE_LOOKUP)
    loc_col = next((c for c in zones.columns if c.lower() == "locationid"), None)
    if loc_col is None:
        raise ValueError("taxi_zones.shp does not contain LocationID")
    zones = zones.rename(columns={loc_col: "LocationID"})
    zones["LocationID"] = pd.to_numeric(zones["LocationID"], errors="coerce").astype("Int64")
    lookup["LocationID"] = pd.to_numeric(lookup["LocationID"], errors="coerce").astype("Int64")
    zones = zones.dropna(subset=["LocationID", "geometry"]).copy()
    zones["LocationID"] = zones["LocationID"].astype(int)
    zones = zones[zones.geometry.is_valid].copy()
    keep = [c for c in ["LocationID", "Borough", "Zone", "service_zone"] if c in lookup.columns]
    zones = zones.merge(lookup[keep], on="LocationID", how="left", suffixes=("", "_lookup"))
    if zones.crs is None:
        raise ValueError("Taxi-zone shapefile has no CRS")
    # NYC TLC taxi zones are usually projected in feet; use EPSG:2263 if the source is geographic.
    projected = zones.to_crs(2263) if zones.crs.is_geographic else zones.copy()
    centroid_proj = projected.geometry.centroid
    centroid_ll = gpd.GeoSeries(centroid_proj, crs=projected.crs).to_crs(4326)
    centroids = pd.DataFrame(
        {
            "zone_id": projected["LocationID"].astype(int),
            "zone_name": projected.get("Zone", pd.Series("Unknown", index=projected.index)).fillna("Unknown").astype(str),
            "borough": projected.get("Borough", pd.Series("Unknown", index=projected.index)).fillna("Unknown").astype(str),
            "service_zone": projected.get("service_zone", pd.Series("Unknown", index=projected.index)).fillna("Unknown").astype(str),
            "centroid_x": centroid_proj.x.astype(float),
            "centroid_y": centroid_proj.y.astype(float),
            "centroid_lon": centroid_ll.x.astype(float),
            "centroid_lat": centroid_ll.y.astype(float),
            "area_sqft": projected.geometry.area.astype(float),
        }
    ).sort_values("zone_id").reset_index(drop=True)
    return projected.sort_values("LocationID").reset_index(drop=True), centroids


def available_parquet_columns(path: Path) -> List[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        return list(pd.read_parquet(path, engine="auto").columns)


def clean_month(path: Path, valid_zones: set[int], month: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing trip file: {path}")
    if path.suffix.lower() == ".csv":
        header = pd.read_csv(path, nrows=0)
        available = set(header.columns)
    else:
        available = set(available_parquet_columns(path))
    required = {"tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "trip_distance", "fare_amount"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    use = [c for c in TRIP_COLUMNS if c in available]
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=use)
    else:
        df = pd.read_parquet(path, columns=use)
    if "tip_amount" not in df.columns:
        df["tip_amount"] = 0.0
    raw_count = len(df)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    valid_pickup_timestamp_rows = int(df["tpep_pickup_datetime"].notna().sum())
    for c in ["PULocationID", "DOLocationID", "trip_distance", "fare_amount", "tip_amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "trip_distance", "fare_amount"])
    df["PULocationID"] = df["PULocationID"].astype(int)
    df["DOLocationID"] = df["DOLocationID"].astype(int)
    df = df[df["PULocationID"].isin(valid_zones) & df["DOLocationID"].isin(valid_zones)].copy()
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    df = df[
        df["fare_amount"].between(MIN_FARE, MAX_FARE)
        & df["trip_distance"].between(MIN_DISTANCE_MILES, MAX_DISTANCE_MILES)
        & df["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN)
    ].copy()
    df["tip_amount"] = df["tip_amount"].fillna(0.0).clip(lower=0.0)
    # Revenue model intentionally excludes tolls/taxes/surcharges and remains configurable.
    df["driver_revenue"] = DRIVER_REVENUE_SHARE * (df["fare_amount"].clip(lower=0.0) + df["tip_amount"])
    df = df[df["driver_revenue"] > 0].copy()
    clean_rows_before_month_filter = int(len(df))
    month_start, next_month_start = month_bounds(month)
    in_month = pickup_in_declared_month(df["tpep_pickup_datetime"], month)
    clean_rows_out_of_month = int((~in_month).sum())
    df = df.loc[in_month].copy()
    df["time_bin"] = (
        df["tpep_pickup_datetime"].dt.hour * (60 // TIME_BIN_MINUTES)
        + df["tpep_pickup_datetime"].dt.minute // TIME_BIN_MINUTES
    ).astype(int)
    df["dropoff_time_bin"] = (
        df["tpep_dropoff_datetime"].dt.hour * (60 // TIME_BIN_MINUTES)
        + df["tpep_dropoff_datetime"].dt.minute // TIME_BIN_MINUTES
    ).astype(int)
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
    retained_out_of_month = int(
        (~pickup_in_declared_month(df["tpep_pickup_datetime"], month)).sum()
    )
    audit = {
        "month": month,
        "expected_start_inclusive": month_start.isoformat(),
        "expected_end_exclusive": next_month_start.isoformat(),
        "expected_calendar_days": expected_calendar_days(month),
        "raw_rows": int(raw_count),
        "raw_valid_pickup_timestamp_rows": valid_pickup_timestamp_rows,
        "clean_rows_before_month_filter": clean_rows_before_month_filter,
        "clean_rows_out_of_month_excluded": clean_rows_out_of_month,
        "clean_rows_retained": int(len(df)),
        "retained_out_of_month_rows": retained_out_of_month,
        "retained_service_days": int(df["pickup_date"].nunique()),
    }
    df.attrs["date_boundary_audit"] = audit
    print(
        f"{month}: raw {raw_count:,} -> quality-clean "
        f"{clean_rows_before_month_filter:,} -> in-month {len(df):,} "
        f"(excluded {clean_rows_out_of_month:,})"
    )
    return df


def monthly_aggregates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Return OD, pickup-node and dropoff-node aggregates for one month."""
    work = df.copy()
    for col in ["driver_revenue", "duration_min", "trip_distance"]:
        work[f"{col}_sq"] = work[col] ** 2
    od = (
        work.groupby(["time_bin", "PULocationID", "DOLocationID"], sort=False)
        .agg(
            trip_count=("PULocationID", "size"),
            revenue_sum=("driver_revenue", "sum"),
            revenue_sq_sum=("driver_revenue_sq", "sum"),
            duration_sum=("duration_min", "sum"),
            duration_sq_sum=("duration_min_sq", "sum"),
            distance_sum=("trip_distance", "sum"),
            distance_sq_sum=("trip_distance_sq", "sum"),
        )
        .reset_index()
        .rename(columns={"PULocationID": "origin", "DOLocationID": "destination"})
    )
    pickups = (
        work.groupby(["time_bin", "PULocationID"], sort=False)
        .size().reset_index(name="pickup_count")
        .rename(columns={"PULocationID": "zone_id"})
    )
    dropoffs = (
        work.groupby(["dropoff_time_bin", "DOLocationID"], sort=False)
        .size().reset_index(name="dropoff_count")
        .rename(columns={"dropoff_time_bin": "time_bin", "DOLocationID": "zone_id"})
    )
    service_days = int(work["pickup_date"].nunique())
    return od, pickups, dropoffs, service_days

def combine_aggregates(frames: Sequence[pd.DataFrame], keys: Sequence[str]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    numeric = [c for c in combined.columns if c not in keys]
    return combined.groupby(list(keys), as_index=False, sort=False)[numeric].sum()


def make_global_od(od_agg: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in od_agg.columns if c not in ["time_bin", "origin", "destination"]]
    global_od = od_agg.groupby(["origin", "destination"], as_index=False)[numeric].sum()
    n = global_od["trip_count"].clip(lower=1)
    global_od["avg_driver_revenue"] = global_od["revenue_sum"] / n
    global_od["avg_duration_min"] = global_od["duration_sum"] / n
    global_od["avg_distance_miles"] = global_od["distance_sum"] / n
    global_od["revenue_std"] = [pooled_std(a, b, c) for a, b, c in zip(global_od["revenue_sum"], global_od["revenue_sq_sum"], global_od["trip_count"])]
    global_od["duration_std"] = [pooled_std(a, b, c) for a, b, c in zip(global_od["duration_sum"], global_od["duration_sq_sum"], global_od["trip_count"])]
    global_od["distance_std"] = [pooled_std(a, b, c) for a, b, c in zip(global_od["distance_sum"], global_od["distance_sq_sum"], global_od["trip_count"])]
    return global_od


def fit_gravity_prior(global_od: pd.DataFrame) -> Dict[str, float]:
    if global_od.empty:
        return {"intercept": 3.0, "distance": 2.5, "duration": 0.25, "mean_revenue": 15.0, "mean_duration": 20.0, "mean_distance": 4.0}
    x = np.column_stack(
        [
            np.ones(len(global_od)),
            global_od["avg_distance_miles"].to_numpy(float),
            global_od["avg_duration_min"].to_numpy(float),
        ]
    )
    y = global_od["avg_driver_revenue"].to_numpy(float)
    w = np.sqrt(global_od["trip_count"].clip(lower=1).to_numpy(float))
    try:
        beta, *_ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
    except np.linalg.LinAlgError:
        beta = np.array([np.average(y, weights=w), 0.0, 0.0])
    return {
        "intercept": float(beta[0]),
        "distance": float(beta[1]),
        "duration": float(beta[2]),
        "mean_revenue": float(np.average(y, weights=w)),
        "mean_duration": float(np.average(global_od["avg_duration_min"], weights=w)),
        "mean_distance": float(np.average(global_od["avg_distance_miles"], weights=w)),
    }


def centroid_distance_miles(a: int, b: int, centroids: Mapping[int, Tuple[float, float]]) -> float:
    x1, y1 = centroids[int(a)]
    x2, y2 = centroids[int(b)]
    return float(math.hypot(x1 - x2, y1 - y2) / 5280.0)


def nearest_neighbors(centroids_df: pd.DataFrame, k: int = SPATIAL_NEIGHBOR_COUNT) -> Dict[int, List[Tuple[int, float]]]:
    rows = centroids_df[["zone_id", "centroid_x", "centroid_y"]].to_numpy(float)
    out: Dict[int, List[Tuple[int, float]]] = {}
    for zone, x, y in rows:
        distances = []
        for other, ox, oy in rows:
            if int(zone) == int(other):
                continue
            miles = math.hypot(x - ox, y - oy) / 5280.0
            distances.append((int(other), float(miles)))
        distances.sort(key=lambda p: p[1])
        out[int(zone)] = distances[: int(k)]
    return out


def build_node_metrics(
    pickup_agg: pd.DataFrame,
    dropoff_agg: pd.DataFrame,
    od_agg: pd.DataFrame,
    centroids_df: pd.DataFrame,
    service_days: int,
    competition_scenario: str = DEFAULT_COMPETITION_SCENARIO,
) -> pd.DataFrame:
    """Build demand and latent vacant-supply metrics for every 15-minute zone cell.

    TLC trip data observe successful pickups and dropoffs, not idle taxis. The supply model
    therefore produces a *relative latent competition index*, not a claimed exact fleet
    count. Recent dropoffs add potential vacant taxis, pickups consume them, and positive
    stocks diffuse partially to neighbouring zones. The absolute scale is calibrated so the
    median medium-scenario wait equals REFERENCE_WAIT_MINUTES.
    """
    if competition_scenario not in COMPETITION_SCENARIO_MULTIPLIERS:
        raise ValueError(f"Unknown competition scenario: {competition_scenario}")
    zones = centroids_df["zone_id"].astype(int).tolist()
    full = pd.MultiIndex.from_product([range(N_TIME_BINS), zones], names=["time_bin", "zone_id"]).to_frame(index=False)
    full = full.merge(pickup_agg, on=["time_bin", "zone_id"], how="left")
    full = full.merge(dropoff_agg, on=["time_bin", "zone_id"], how="left")
    full[["pickup_count", "dropoff_count"]] = full[["pickup_count", "dropoff_count"]].fillna(0.0)
    exposure = max(1.0, float(service_days))
    full["raw_pickups_per_bin_day"] = full["pickup_count"] / exposure
    full["raw_dropoffs_per_bin_day"] = full["dropoff_count"] / exposure

    zone_global = full.groupby("zone_id")["pickup_count"].sum() / (exposure * N_TIME_BINS)
    time_global = full.groupby("time_bin")["pickup_count"].sum() / (exposure * max(1, len(zones)))
    grand = float(full["pickup_count"].sum() / (exposure * N_TIME_BINS * max(1, len(zones))))
    nearest = nearest_neighbors(centroids_df)
    raw_map = {(int(r.time_bin), int(r.zone_id)): float(r.raw_pickups_per_bin_day) for r in full.itertuples(index=False)}

    posterior, temporal_priors, spatial_priors, node_conf, layers = [], [], [], [], []
    for r in full.itertuples(index=False):
        tb, zone = int(r.time_bin), int(r.zone_id)
        temporal_vals, temporal_w = [], []
        for delta in range(1, TEMPORAL_WINDOW_BINS + 1):
            weight = math.exp(-delta / TEMPORAL_DECAY_BINS)
            for other_tb in ((tb - delta) % N_TIME_BINS, (tb + delta) % N_TIME_BINS):
                temporal_vals.append(raw_map[(other_tb, zone)])
                temporal_w.append(weight)
        temporal = weighted_mean(temporal_vals, temporal_w, float(zone_global.get(zone, grand)))
        spatial_vals = [raw_map[(tb, n)] for n, _ in nearest.get(zone, [])]
        spatial_w = [math.exp(-dist / SPATIAL_DECAY_MILES) for _, dist in nearest.get(zone, [])]
        spatial = weighted_mean(spatial_vals, spatial_w, grand)
        prior = weighted_mean(
            [temporal, spatial, float(zone_global.get(zone, grand)), float(time_global.get(tb, grand)), grand],
            [4.0, 2.0, 3.0, 2.0, 1.0], grand,
        )
        count = float(r.pickup_count)
        reliability = count / (count + NODE_PRIOR_STRENGTH)
        post = reliability * float(r.raw_pickups_per_bin_day) + (1.0 - reliability) * prior
        conf = float(np.clip(count / (count + NODE_PRIOR_STRENGTH), MIN_CONFIDENCE, 0.995))
        posterior.append(max(post, 1e-8)); temporal_priors.append(temporal); spatial_priors.append(spatial); node_conf.append(conf)
        layers.append("observed_dense_eb" if count >= 20 else "observed_sparse_eb" if count > 0 else "temporal_spatial_eb" if temporal > 0 else "global_prior")

    full["posterior_pickups_per_bin_day"] = posterior
    full["temporal_pickup_prior"] = temporal_priors
    full["spatial_pickup_prior"] = spatial_priors
    full["node_confidence"] = node_conf
    full["node_imputation_layer"] = layers
    full["service_days"] = int(service_days)

    # Latent vacant-taxi stock. Successful dropoffs create potential vacant taxis, recent
    # demand attracts additional taxis, and successful pickups deplete the local stock. The
    # quantity is a calibrated relative supply proxy, not an observed fleet count.
    pickup_matrix = full.pivot(index="time_bin", columns="zone_id", values="posterior_pickups_per_bin_day").reindex(index=range(N_TIME_BINS), columns=zones).fillna(0.0).to_numpy(float)
    drop_matrix = full.pivot(index="time_bin", columns="zone_id", values="raw_dropoffs_per_bin_day").reindex(index=range(N_TIME_BINS), columns=zones).fillna(0.0).to_numpy(float)

    # Initialise from a decayed history so the first time bin is not treated as an empty city.
    stock = np.full(len(zones), SUPPLY_BASE_FLOOR, dtype=float)
    for lag in range(1, VACANT_HISTORY_BINS + 1):
        tb = (-lag) % N_TIME_BINS
        weight = VACANT_STOCK_PERSISTENCE ** (lag - 1)
        stock += weight * (drop_matrix[tb] + VACANT_DEMAND_ATTRACTION * pickup_matrix[tb])

    stock_history = np.zeros_like(pickup_matrix)
    # Iterate synthetic days until the daily cycle is effectively stable.
    for _ in range(30):
        for tb in range(N_TIME_BINS):
            prev_tb = (tb - 1) % N_TIME_BINS
            incoming = drop_matrix[prev_tb] + VACANT_DEMAND_ATTRACTION * pickup_matrix[prev_tb]
            stock = np.maximum(
                SUPPLY_BASE_FLOOR,
                VACANT_STOCK_PERSISTENCE * stock + incoming - VACANT_PICKUP_DEPLETION * pickup_matrix[tb],
            )
            stock_history[tb] = stock

    zone_index = {z: i for i, z in enumerate(zones)}
    diffused = stock_history.copy()
    for tb in range(N_TIME_BINS):
        for z in zones:
            zi = zone_index[z]
            vals, wts = [], []
            for neigh, dist in nearest.get(z, []):
                vals.append(stock_history[tb, zone_index[neigh]])
                wts.append(math.exp(-dist / SPATIAL_DECAY_MILES))
            neighbour_supply = weighted_mean(vals, wts, stock_history[tb, zi])
            diffused[tb, zi] = (1.0 - NEIGHBOR_SUPPLY_DIFFUSION) * stock_history[tb, zi] + NEIGHBOR_SUPPLY_DIFFUSION * neighbour_supply

    # Queue-pressure proxy: more latent taxis per passenger arrival means longer wait.
    demand_floor = max(0.05, float(np.nanmedian(pickup_matrix[pickup_matrix > 0])) * 0.05 if np.any(pickup_matrix > 0) else 0.05)
    queue_pressure = (diffused + SUPPLY_BASE_FLOOR) / (pickup_matrix + demand_floor)
    positive_pressure = queue_pressure[np.isfinite(queue_pressure) & (queue_pressure > 0)]
    calibration = REFERENCE_WAIT_MINUTES / float(np.median(positive_pressure) if len(positive_pressure) else 1.0)

    # Airport zones have institutional queues not fully recoverable from trips alone.
    airport_mask = centroids_df.set_index("zone_id").reindex(zones)["zone_name"].fillna("").str.contains("airport", case=False).to_numpy(bool)
    airport_adjustment = np.where(airport_mask[None, :], AIRPORT_QUEUE_MULTIPLIER, 1.0)
    waits = {}
    for scenario, multiplier in COMPETITION_SCENARIO_MULTIPLIERS.items():
        waits[scenario] = np.clip(calibration * queue_pressure * multiplier * airport_adjustment, MIN_EXPECTED_WAIT_MINUTES, MAX_EXPECTED_WAIT_MINUTES)

    supply_rows, pressure_rows = [], []
    wait_low, wait_med, wait_high = [], [], []
    for r in full.itertuples(index=False):
        ti, zi = int(r.time_bin), zone_index[int(r.zone_id)]
        supply_rows.append(float(diffused[ti, zi]))
        pressure_rows.append(float(queue_pressure[ti, zi]))
        wait_low.append(float(waits["low"][ti, zi])); wait_med.append(float(waits["medium"][ti, zi])); wait_high.append(float(waits["high"][ti, zi]))
    full["latent_vacant_supply_index"] = supply_rows
    full["competition_ratio"] = pressure_rows
    full["expected_wait_min_low"] = wait_low
    full["expected_wait_min_medium"] = wait_med
    full["expected_wait_min_high"] = wait_high
    full["expected_wait_min"] = full[f"expected_wait_min_{competition_scenario}"]
    full["competition_scenario"] = competition_scenario
    full["pickup_probability_15m"] = 1.0 - np.exp(-TIME_BIN_MINUTES / full["expected_wait_min"].clip(lower=1e-6))
    # Queue-equivalent competitors follow E[W] ≈ (V+1)/lambda. They are model-implied
    # equivalents used for interpretation and sensitivity analysis, not directly observed taxis.
    arrival_rate_per_minute = full["posterior_pickups_per_bin_day"].clip(lower=1e-8) / TIME_BIN_MINUTES
    for scenario in COMPETITION_SCENARIO_MULTIPLIERS:
        full[f"estimated_competitors_equivalent_{scenario}"] = (
            full[f"expected_wait_min_{scenario}"] * arrival_rate_per_minute - 1.0
        ).clip(lower=0.0)
    full["estimated_competitors_equivalent"] = full[f"estimated_competitors_equivalent_{competition_scenario}"]

    full["expected_trip_revenue"] = 0.0
    full["expected_trip_duration_min"] = 15.0
    full["expected_trip_distance_miles"] = 2.0
    return full

def prepare_observed_tables(od_agg: pd.DataFrame) -> pd.DataFrame:
    out = od_agg.copy()
    n = out["trip_count"].clip(lower=1)
    out["observed_avg_driver_revenue"] = out["revenue_sum"] / n
    out["observed_avg_duration_min"] = out["duration_sum"] / n
    out["observed_avg_distance_miles"] = out["distance_sum"] / n
    out["observed_revenue_std"] = [pooled_std(a, b, c) for a, b, c in zip(out["revenue_sum"], out["revenue_sq_sum"], out["trip_count"])]
    out["observed_duration_std"] = [pooled_std(a, b, c) for a, b, c in zip(out["duration_sum"], out["duration_sq_sum"], out["trip_count"])]
    out["observed_distance_std"] = [pooled_std(a, b, c) for a, b, c in zip(out["distance_sum"], out["distance_sq_sum"], out["trip_count"])]
    return out


def build_time_profiles(observed: pd.DataFrame) -> pd.DataFrame:
    """Estimate time-of-day speed and revenue-per-mile profiles from observed trips."""
    work = observed.copy()
    work = work[(work["observed_avg_distance_miles"] > 0) & (work["observed_avg_duration_min"] > 0)].copy()
    work["speed_mph"] = 60.0 * work["observed_avg_distance_miles"] / work["observed_avg_duration_min"]
    work["revenue_per_mile"] = work["observed_avg_driver_revenue"] / work["observed_avg_distance_miles"].clip(lower=0.1)
    rows = []
    global_speed = weighted_mean(work["speed_mph"], work["trip_count"], FALLBACK_SPEED_MPH)
    global_rpm = weighted_mean(work["revenue_per_mile"], work["trip_count"], 4.0)
    for tb in range(N_TIME_BINS):
        g = work[work["time_bin"] == tb]
        speed = weighted_mean(g["speed_mph"], g["trip_count"], global_speed)
        rpm = weighted_mean(g["revenue_per_mile"], g["trip_count"], global_rpm)
        rows.append({"time_bin": tb, "speed_mph": float(np.clip(speed, MIN_SPEED_MPH, MAX_SPEED_MPH)), "revenue_per_mile": max(0.1, rpm)})
    return pd.DataFrame(rows)


def build_dynamic_od_metrics(
    observed: pd.DataFrame,
    global_od: pd.DataFrame,
    node_metrics: pd.DataFrame,
    centroids_df: pd.DataFrame,
    time_profiles: pd.DataFrame,
) -> pd.DataFrame:
    zones = centroids_df["zone_id"].astype(int).tolist()
    centroids = {int(r.zone_id): (float(r.centroid_x), float(r.centroid_y)) for r in centroids_df.itertuples(index=False)}
    neighbors = nearest_neighbors(centroids_df)
    gravity = fit_gravity_prior(global_od)
    speed_by_bin = dict(zip(time_profiles["time_bin"].astype(int), time_profiles["speed_mph"].astype(float)))
    global_speed = float(np.average(time_profiles["speed_mph"]))
    duration_calibration = build_duration_calibration(observed, centroids_df, time_profiles)
    zone_meta = centroids_df.set_index("zone_id")[["borough", "zone_name"]].to_dict("index")

    obs_map = {(int(r.time_bin), int(r.origin), int(r.destination)): r for r in observed.itertuples(index=False)}
    by_pair: Dict[Tuple[int, int], List[object]] = defaultdict(list)
    for r in observed.itertuples(index=False):
        by_pair[(int(r.origin), int(r.destination))].append(r)
    global_map = {(int(r.origin), int(r.destination)): r for r in global_od.itertuples(index=False)}
    global_origin_lists: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    dest_popularity = global_od.groupby("destination")["trip_count"].sum().sort_values(ascending=False)
    popular_dests = [int(x) for x in dest_popularity.head(MAX_DESTINATIONS_PER_ORIGIN).index]
    for r in global_od.itertuples(index=False):
        global_origin_lists[int(r.origin)].append((int(r.destination), float(r.trip_count)))
    for origin in global_origin_lists:
        global_origin_lists[origin].sort(key=lambda x: x[1], reverse=True)

    current_groups = {(int(tb), int(origin)): g for (tb, origin), g in observed.groupby(["time_bin", "origin"], sort=False)}
    node_rate = {(int(r.time_bin), int(r.zone_id)): float(r.posterior_pickups_per_bin_day) for r in node_metrics.itertuples(index=False)}
    records: List[Dict[str, object]] = []

    for tb in range(N_TIME_BINS):
        for origin in zones:
            score: Dict[int, float] = defaultdict(float)
            group = current_groups.get((tb, origin))
            if group is not None:
                for r in group.itertuples(index=False):
                    score[int(r.destination)] += float(r.trip_count)
            for dest, cnt in global_origin_lists.get(origin, [])[: MAX_DESTINATIONS_PER_ORIGIN]:
                score[dest] += GLOBAL_OD_DISCOUNT * cnt
            for rank, dest in enumerate(popular_dests):
                score[dest] += max(0.1, 1.0 - rank / max(1, len(popular_dests)))
            if not score:
                for dest in popular_dests:
                    score[dest] = 1.0
            candidates = [d for d, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)[:MAX_DESTINATIONS_PER_ORIGIN]]
            if origin not in candidates and len(candidates) < MAX_DESTINATIONS_PER_ORIGIN:
                candidates.append(origin)

            # Prior destination probabilities from full-history origin flows plus gravity distance decay.
            global_total = sum(cnt for _, cnt in global_origin_lists.get(origin, []))
            prior_probs = []
            obs_counts = []
            row_estimates = []
            for destination in candidates:
                obs = obs_map.get((tb, origin, destination))
                obs_count = float(obs.trip_count) if obs is not None else 0.0
                obs_counts.append(obs_count)
                global_row = global_map.get((origin, destination))
                global_count = float(global_row.trip_count) if global_row is not None else 0.0
                distance_struct = centroid_distance_miles(origin, destination, centroids)
                gravity_prob = math.exp(-distance_struct / 8.0)
                global_prob = global_count / global_total if global_total > 0 else 0.0
                prior_probs.append(0.8 * global_prob + 0.2 * gravity_prob)

                temporal_rows = []
                temporal_weights = []
                for r in by_pair.get((origin, destination), []):
                    dist = circular_bin_distance(tb, int(r.time_bin))
                    if 0 < dist <= TEMPORAL_WINDOW_BINS:
                        w = float(r.trip_count) * math.exp(-dist / TEMPORAL_DECAY_BINS)
                        temporal_rows.append(r)
                        temporal_weights.append(w)
                temporal_count = float(sum(temporal_weights))

                spatial_values_rev = []
                spatial_values_dur = []
                spatial_values_dist = []
                spatial_weights = []
                for neigh, dist in neighbors.get(origin, []):
                    r = global_map.get((neigh, destination))
                    if r is not None:
                        w = float(r.trip_count) * math.exp(-dist / SPATIAL_DECAY_MILES)
                        spatial_values_rev.append(float(r.avg_driver_revenue))
                        spatial_values_dur.append(float(r.avg_duration_min))
                        spatial_values_dist.append(float(r.avg_distance_miles))
                        spatial_weights.append(w)
                for neigh, dist in neighbors.get(destination, []):
                    r = global_map.get((origin, neigh))
                    if r is not None:
                        w = float(r.trip_count) * math.exp(-dist / SPATIAL_DECAY_MILES)
                        spatial_values_rev.append(float(r.avg_driver_revenue))
                        spatial_values_dur.append(float(r.avg_duration_min))
                        spatial_values_dist.append(float(r.avg_distance_miles))
                        spatial_weights.append(w)

                time_speed = float(np.clip(speed_by_bin.get(tb, FALLBACK_SPEED_MPH), MIN_SPEED_MPH, MAX_SPEED_MPH))
                fallback_duration, duration_calibration_multiplier, duration_band, duration_borough_pair = calibrated_duration_prior(
                    origin, int(destination), tb, distance_struct, time_speed, zone_meta, duration_calibration
                )
                gravity_revenue = max(
                    0.0,
                    gravity["intercept"] + gravity["distance"] * distance_struct + gravity["duration"] * fallback_duration,
                )
                global_revenue = float(global_row.avg_driver_revenue) if global_row is not None else gravity["mean_revenue"]
                global_duration = (
                    float(global_row.avg_duration_min) * global_speed / time_speed
                    if global_row is not None else fallback_duration
                )
                global_distance = float(global_row.avg_distance_miles) if global_row is not None else distance_struct
                global_std_rev = float(global_row.revenue_std) if global_row is not None else max(1.0, 0.35 * gravity_revenue)
                global_std_dur = float(global_row.duration_std) if global_row is not None else max(2.0, 0.30 * fallback_duration)
                global_std_dist = float(global_row.distance_std) if global_row is not None else max(0.25, 0.25 * distance_struct)

                temporal_rev = weighted_mean(
                    [float(r.observed_avg_driver_revenue) for r in temporal_rows],
                    temporal_weights,
                    global_revenue,
                )
                temporal_dur = weighted_mean(
                    [float(r.observed_avg_duration_min) for r in temporal_rows],
                    temporal_weights,
                    global_duration,
                )
                temporal_dist = weighted_mean(
                    [float(r.observed_avg_distance_miles) for r in temporal_rows],
                    temporal_weights,
                    global_distance,
                )
                spatial_rev = weighted_mean(spatial_values_rev, spatial_weights, global_revenue)
                spatial_dur = weighted_mean(spatial_values_dur, spatial_weights, global_duration)
                spatial_dist = weighted_mean(spatial_values_dist, spatial_weights, global_distance)
                spatial_weight = min(sum(spatial_weights), 20.0)
                global_weight = min(global_count * GLOBAL_OD_DISCOUNT, 25.0)

                obs_rev = float(obs.observed_avg_driver_revenue) if obs is not None else np.nan
                obs_dur = float(obs.observed_avg_duration_min) if obs is not None else np.nan
                obs_dist = float(obs.observed_avg_distance_miles) if obs is not None else np.nan
                weights = [obs_count, min(temporal_count, 20.0), spatial_weight, global_weight, GRAVITY_PRIOR_STRENGTH]
                rev = weighted_mean([obs_rev, temporal_rev, spatial_rev, global_revenue, gravity_revenue], weights, gravity_revenue)
                dur = geometric_weighted_mean(
                    [obs_dur, temporal_dur, spatial_dur, global_duration, fallback_duration],
                    weights, fallback_duration,
                )
                # Keep a calibrated structural floor so sparse long trips are not collapsed
                # toward the city-wide mean by empirical-Bayes shrinkage.
                dur = max(dur, 0.85 * fallback_duration)
                dist_value = weighted_mean([obs_dist, temporal_dist, spatial_dist, global_distance, distance_struct], weights, distance_struct)
                effective_n = obs_count + 0.5 * temporal_count + 0.25 * spatial_weight + 0.2 * global_count
                confidence = float(np.clip(effective_n / (effective_n + OD_OBS_PRIOR_STRENGTH), MIN_CONFIDENCE, 0.995))
                if obs_count >= 20:
                    layer = "observed_dense_eb"
                elif obs_count > 0:
                    layer = "observed_sparse_eb"
                elif temporal_count > 0:
                    layer = "temporal_eb"
                elif spatial_weight > 0:
                    layer = "spatial_eb"
                elif global_count > 0:
                    layer = "global_od_eb"
                else:
                    layer = "gravity_prior"

                row_estimates.append(
                    {
                        "time_bin": tb,
                        "origin": origin,
                        "destination": int(destination),
                        "observed_trip_count": obs_count,
                        "avg_driver_revenue": rev,
                        "avg_duration_min": max(1.0, dur),
                        "avg_distance_miles": max(0.01, dist_value),
                        "revenue_std": float(obs.observed_revenue_std) if obs is not None and obs_count >= 2 else global_std_rev,
                        "duration_std": float(obs.observed_duration_std) if obs is not None and obs_count >= 2 else global_std_dur,
                        "distance_std": float(obs.observed_distance_std) if obs is not None and obs_count >= 2 else global_std_dist,
                        "confidence": confidence,
                        "imputation_layer": layer,
                        "effective_sample_size": effective_n,
                        "observed_weight": obs_count,
                        "temporal_weight": min(temporal_count, 20.0),
                        "spatial_weight": spatial_weight,
                        "global_weight": global_weight,
                        "gravity_weight": GRAVITY_PRIOR_STRENGTH,
                        "observed_avg_driver_revenue": obs_rev,
                        "observed_avg_duration_min": obs_dur,
                        "observed_avg_distance_miles": obs_dist,
                        "temporal_revenue_prior": temporal_rev,
                        "temporal_duration_prior": temporal_dur,
                        "temporal_distance_prior": temporal_dist,
                        "spatial_revenue_prior": spatial_rev,
                        "spatial_duration_prior": spatial_dur,
                        "spatial_distance_prior": spatial_dist,
                        "global_revenue_prior": global_revenue,
                        "global_duration_prior": global_duration,
                        "global_distance_prior": global_distance,
                        "gravity_revenue_prior": gravity_revenue,
                        "gravity_duration_prior": fallback_duration,
                        "gravity_distance_prior": distance_struct,
                        "calibrated_duration_prior": fallback_duration,
                        "duration_calibration_multiplier": duration_calibration_multiplier,
                        "duration_distance_band": duration_band,
                        "duration_borough_pair": duration_borough_pair,
                    }
                )

            prior_sum = sum(max(0.0, p) for p in prior_probs)
            if prior_sum <= 0:
                prior_probs = [1.0 / len(candidates)] * len(candidates)
            else:
                prior_probs = [max(0.0, p) / prior_sum for p in prior_probs]
            total_obs = sum(obs_counts)
            numerators = [c + OD_DIRICHLET_STRENGTH * p for c, p in zip(obs_counts, prior_probs)]
            denom = sum(numerators)
            probs = [x / denom if denom > 0 else 1.0 / len(numerators) for x in numerators]
            origin_expected_count = max(0.0, node_rate.get((tb, origin), 0.0))
            for rec, prob in zip(row_estimates, probs):
                rec["destination_probability"] = float(prob)
                rec["estimated_trip_count_per_bin_day"] = float(prob * origin_expected_count)
                records.append(rec)

    return pd.DataFrame(records)


def attach_node_trip_expectations(node_metrics: pd.DataFrame, od_metrics: pd.DataFrame) -> pd.DataFrame:
    weighted = od_metrics.copy()
    for metric in ["avg_driver_revenue", "avg_duration_min", "avg_distance_miles"]:
        weighted[f"weighted_{metric}"] = weighted["destination_probability"] * weighted[metric]
    agg = (
        weighted.groupby(["time_bin", "origin"], as_index=False)[
            ["weighted_avg_driver_revenue", "weighted_avg_duration_min", "weighted_avg_distance_miles"]
        ]
        .sum()
        .rename(
            columns={
                "origin": "zone_id",
                "weighted_avg_driver_revenue": "expected_trip_revenue",
                "weighted_avg_duration_min": "expected_trip_duration_min",
                "weighted_avg_distance_miles": "expected_trip_distance_miles",
            }
        )
    )
    out = node_metrics.drop(columns=["expected_trip_revenue", "expected_trip_duration_min", "expected_trip_distance_miles"]).merge(
        agg, on=["time_bin", "zone_id"], how="left"
    )
    out["expected_trip_revenue"] = out["expected_trip_revenue"].fillna(0.0)
    out["expected_trip_duration_min"] = out["expected_trip_duration_min"].fillna(15.0)
    out["expected_trip_distance_miles"] = out["expected_trip_distance_miles"].fillna(2.0)
    return out


def build_reposition_edges(zones: gpd.GeoDataFrame, centroids_df: pd.DataFrame, global_od: pd.DataFrame) -> pd.DataFrame:
    centroids = {int(r.zone_id): (float(r.centroid_x), float(r.centroid_y)) for r in centroids_df.itertuples(index=False)}
    global_map = {(int(r.origin), int(r.destination)): r for r in global_od.itertuples(index=False)}
    records = []
    n = len(zones)
    for i in range(n):
        a = int(zones.iloc[i]["LocationID"])
        ga = zones.iloc[i].geometry
        for j in range(i + 1, n):
            b = int(zones.iloc[j]["LocationID"])
            gb = zones.iloc[j].geometry
            boundary_distance = float(ga.distance(gb))
            if not (ga.touches(gb) or ga.intersects(gb) or boundary_distance <= ADJACENCY_BUFFER_FEET):
                continue
            for origin, destination in ((a, b), (b, a)):
                global_row = global_map.get((origin, destination))
                centroid_miles = centroid_distance_miles(origin, destination, centroids)
                if global_row is not None:
                    duration = float(global_row.avg_duration_min)
                    distance = float(global_row.avg_distance_miles)
                    confidence = float(np.clip(global_row.trip_count / (global_row.trip_count + 10.0), 0.2, 0.99))
                    source = "historical_adjacency"
                else:
                    distance = centroid_miles
                    duration = max(1.0, distance / FALLBACK_SPEED_MPH * 60.0)
                    confidence = 0.20
                    source = "spatial_fallback"
                records.append(
                    {
                        "origin": origin,
                        "destination": destination,
                        "duration_min": duration,
                        "distance_miles": distance,
                        "confidence": confidence,
                        "edge_source": source,
                    }
                )
    edges = pd.DataFrame(records)
    if edges.empty:
        raise RuntimeError("No reposition adjacency edges were constructed")
    return edges


def build_environment(clean_first: bool = True, competition_scenario: str = DEFAULT_COMPETITION_SCENARIO) -> DynamicTaxiEnvironment:
    ensure_directories()
    if competition_scenario not in COMPETITION_SCENARIO_MULTIPLIERS:
        raise ValueError(f"Unknown competition scenario: {competition_scenario}")
    if clean_first:
        for p in PROCESSED_DIR.glob("dynamic_*.csv"):
            p.unlink(missing_ok=True)
        for name in ["dynamic_environment.pkl", "environment_build_summary.csv", "zone_centroids.csv", "reposition_edges.csv", "global_od_metrics.csv", "observed_od_metrics.pkl", "time_bin_profiles.csv"]:
            (PROCESSED_DIR / name).unlink(missing_ok=True)

    zones, centroids = read_zone_data()
    valid_zones = set(centroids["zone_id"].astype(int))
    od_frames, pickup_frames, dropoff_frames = [], [], []
    all_service_dates: set[date] = set()
    monthly_date_audits: List[Dict[str, object]] = []
    for month in MONTHS:
        parquet_path = DATA_DIR / f"yellow_tripdata_{month}.parquet"
        csv_path = DATA_DIR / f"yellow_tripdata_{month}.csv"
        trip_path = parquet_path if parquet_path.exists() else csv_path
        df = clean_month(trip_path, valid_zones, month)
        monthly_date_audits.append(dict(df.attrs["date_boundary_audit"]))
        all_service_dates.update(df["pickup_date"].unique().tolist())
        od, pickups, dropoffs, days = monthly_aggregates(df)
        od_frames.append(od); pickup_frames.append(pickups); dropoff_frames.append(dropoffs)
        if days != expected_calendar_days(month):
            raise RuntimeError(
                f"Date-boundary gate failed for {month}: retained {days} service days, "
                f"expected {expected_calendar_days(month)}"
            )
        del df

    total_service_days = int(len(all_service_dates))
    expected_dates: set[date] = set()
    for month in MONTHS:
        expected_dates.update(declared_dates(month))
    expected_days = expected_service_days(MONTHS)
    excluded_out_of_month = int(
        sum(int(row["clean_rows_out_of_month_excluded"]) for row in monthly_date_audits)
    )
    retained_out_of_month = int(
        sum(int(row["retained_out_of_month_rows"]) for row in monthly_date_audits)
    )
    date_checks = {
        "all_months_present": len(monthly_date_audits) == len(MONTHS),
        "retained_out_of_month_rows_is_zero": retained_out_of_month == 0,
        "service_days_matches_calendar": total_service_days == expected_days,
        "retained_dates_match_declared_window": all_service_dates == expected_dates,
    }
    failed_date_checks = [name for name, passed in date_checks.items() if not passed]
    date_boundary_report = {
        "status": (
            "DATE BOUNDARY FIX PASSED"
            if not failed_date_checks
            else "DATE BOUNDARY FIX FAILED"
        ),
        "project_version": "5.2.0",
        "filter_definition": (
            "declared_month_start <= tpep_pickup_datetime < next_month_start; "
            "dropoff datetime is not month-restricted"
        ),
        "months": list(MONTHS),
        "expected_service_days": expected_days,
        "retained_unique_service_days": total_service_days,
        "clean_rows_out_of_month_excluded": excluded_out_of_month,
        "retained_out_of_month_rows": retained_out_of_month,
        "checks": date_checks,
        "failed": failed_date_checks,
        "monthly": monthly_date_audits,
    }
    date_gate_path = PROCESSED_DIR / "data_date_boundary_gate.json"
    date_gate_path.write_text(
        json.dumps(date_boundary_report, indent=2), encoding="utf-8"
    )
    if failed_date_checks:
        raise RuntimeError(
            "Date-boundary gate failed: " + ", ".join(failed_date_checks)
        )

    od_agg = combine_aggregates(od_frames, ["time_bin", "origin", "destination"])
    pickup_agg = combine_aggregates(pickup_frames, ["time_bin", "zone_id"])
    dropoff_agg = combine_aggregates(dropoff_frames, ["time_bin", "zone_id"])
    observed = prepare_observed_tables(od_agg)
    global_od = make_global_od(od_agg)
    time_profiles = build_time_profiles(observed)
    print("Building passenger demand + latent vacant-taxi competition metrics...")
    node_metrics = build_node_metrics(pickup_agg, dropoff_agg, od_agg, centroids, total_service_days, competition_scenario)
    print("Building hierarchical empirical-Bayes OD metrics...")
    od_metrics = build_dynamic_od_metrics(observed, global_od, node_metrics, centroids, time_profiles)
    node_metrics = attach_node_trip_expectations(node_metrics, od_metrics)
    print("Building zone-adjacency reposition graph...")
    reposition_edges = build_reposition_edges(zones, centroids, global_od)

    gravity = fit_gravity_prior(global_od)
    duration_calibration = build_duration_calibration(observed, centroids, time_profiles)
    metadata = {
        "months": MONTHS,
        "service_days": int(total_service_days),
        "time_bin_minutes": TIME_BIN_MINUTES,
        "n_time_bins": N_TIME_BINS,
        "driver_revenue_share": DRIVER_REVENUE_SHARE,
        "environment_schema_version": "4.1-date-boundary-fix",
        "project_version": "5.2.0",
        "date_boundary_filter": date_boundary_report,
        "competition_scenario": competition_scenario,
        "competition_multipliers": COMPETITION_SCENARIO_MULTIPLIERS,
        "wait_model": "latent supply from decayed dropoffs-pickups plus neighbour diffusion; median-calibrated queue pressure",
        "wait_model_limitation": "latent relative competition, not directly observed absolute vacant-taxi count",
        "gravity_model": gravity,
        "duration_calibration": duration_calibration,
        "duration_model": "log-scale EB with calibrated structural floor by time, distance, borough pair and airport involvement",
        "time_speed_mph": dict(zip(time_profiles["time_bin"].astype(int), time_profiles["speed_mph"].astype(float))),
        "time_revenue_per_mile": dict(zip(time_profiles["time_bin"].astype(int), time_profiles["revenue_per_mile"].astype(float))),
        "global_speed_mph": float(np.average(time_profiles["speed_mph"])),
        "pickup_scale": float(max(1.0, node_metrics["posterior_pickups_per_bin_day"].quantile(0.95))),
        "revenue_note": "Driver revenue uses configurable share × (fare_amount + tip_amount); tolls/taxes/surcharges excluded.",
    }
    observed_export = observed[[
        "time_bin", "origin", "destination", "trip_count",
        "observed_avg_driver_revenue", "observed_avg_duration_min",
        "observed_avg_distance_miles", "observed_revenue_std",
        "observed_duration_std", "observed_distance_std",
    ]].copy()
    env = DynamicTaxiEnvironment(
        node_metrics=node_metrics,
        od_metrics=od_metrics,
        reposition_edges=reposition_edges,
        zone_centroids=centroids,
        metadata=metadata,
        global_od_metrics=global_od,
        observed_od_metrics=observed_export,
    )

    node_metrics.to_csv(PROCESSED_DIR / "dynamic_node_metrics.csv", index=False)
    od_metrics.to_csv(PROCESSED_DIR / "dynamic_od_metrics.csv", index=False)
    global_od.to_csv(PROCESSED_DIR / "global_od_metrics.csv", index=False)
    observed_export.to_pickle(PROCESSED_DIR / "observed_od_metrics.pkl")
    time_profiles.to_csv(PROCESSED_DIR / "time_bin_profiles.csv", index=False)
    reposition_edges.to_csv(PROCESSED_DIR / "reposition_edges.csv", index=False)
    centroids.to_csv(PROCESSED_DIR / "zone_centroids.csv", index=False)
    env.save(PROCESSED_DIR / "dynamic_environment.pkl")

    summary = pd.DataFrame([{
        "zones": len(centroids), "time_bins": N_TIME_BINS,
        "dynamic_node_rows": len(node_metrics), "dynamic_od_rows": len(od_metrics),
        "global_observed_od_pairs": len(global_od), "observed_od_time_cells": len(observed_export), "reposition_edges": len(reposition_edges),
        "service_days": total_service_days,
        "environment_schema_version": "4.1-date-boundary-fix",
        "date_boundary_gate_status": date_boundary_report["status"],
        "out_of_month_clean_rows_excluded": excluded_out_of_month,
        "retained_out_of_month_rows": retained_out_of_month,
        "observed_od_cells": int((od_metrics["observed_trip_count"] > 0).sum()),
        "imputed_od_cells": int((od_metrics["observed_trip_count"] <= 0).sum()),
        "mean_od_confidence": float(od_metrics["confidence"].mean()),
        "mean_node_confidence": float(node_metrics["node_confidence"].mean()),
        "competition_scenario": competition_scenario,
        "mean_wait_minutes": float(node_metrics["expected_wait_min"].mean()),
        "median_wait_minutes": float(node_metrics["expected_wait_min"].median()),
        "arbitrary_od_query_supported": True,
    }])
    summary.to_csv(PROCESSED_DIR / "environment_build_summary.csv", index=False)
    print("\nEnvironment built successfully.")
    for name in ["dynamic_node_metrics.csv", "dynamic_od_metrics.csv", "global_od_metrics.csv", "observed_od_metrics.pkl", "time_bin_profiles.csv", "reposition_edges.csv", "zone_centroids.csv", "dynamic_environment.pkl", "environment_build_summary.csv"]:
        print(f"  {PROCESSED_DIR / name}")
    return env

def main() -> None:
    parser = argparse.ArgumentParser(description="Build dynamic 15-minute taxi environment")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove previous dynamic outputs")
    parser.add_argument("--competition-scenario", choices=sorted(COMPETITION_SCENARIO_MULTIPLIERS), default=DEFAULT_COMPETITION_SCENARIO)
    args = parser.parse_args()
    build_environment(clean_first=not args.no_clean, competition_scenario=args.competition_scenario)



if __name__ == "__main__":
    main()
