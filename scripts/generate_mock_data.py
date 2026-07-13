"""Generate a small reproducible mock dataset for smoke-testing the full pipeline.

Never run this against a directory containing real TLC files unless --force is intended.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from config import DATA_DIR, MONTHS, RANDOM_SEED


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock MultiTaxi data")
    parser.add_argument("--trips-per-month", type=int, default=4000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    protected = [DATA_DIR / f"yellow_tripdata_{m}.parquet" for m in MONTHS]
    if any(p.exists() for p in protected) and not args.force:
        raise SystemExit("Raw parquet files already exist. Refusing to overwrite without --force.")

    rng = np.random.default_rng(args.seed)
    zone_ids = np.arange(1, 13)
    geometries = []
    rows = []
    size = 9000.0
    origin_x, origin_y = 970000.0, 190000.0
    for idx, zone_id in enumerate(zone_ids):
        col = idx % 4
        row = idx // 4
        x0 = origin_x + col * size
        y0 = origin_y + row * size
        geometries.append(box(x0, y0, x0 + size, y0 + size))
        rows.append(
            {
                "LocationID": int(zone_id),
                "Borough": ["Queens", "Manhattan", "Brooklyn"][row],
                "Zone": f"Mock Zone {zone_id}",
                "service_zone": "Yellow Zone",
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:2263")
    gdf[["LocationID", "geometry"]].to_file(DATA_DIR / "taxi_zones.shp")
    pd.DataFrame(rows).to_csv(DATA_DIR / "taxi_zone_lookup.csv", index=False)

    attractiveness = np.linspace(0.7, 1.4, len(zone_ids))
    for month_index, month in enumerate(MONTHS):
        year, mon = map(int, month.split("-"))
        start = datetime(year, mon, 1)
        next_month = datetime(year + (mon == 12), (mon % 12) + 1, 1)
        span_minutes = int((next_month - start).total_seconds() // 60)
        pickup_minutes = rng.integers(0, span_minutes, size=args.trips_per_month)
        pickup_times = [start + timedelta(minutes=int(x)) for x in pickup_minutes]
        hours = np.array([t.hour for t in pickup_times])
        demand_boost = 1.0 + 0.9 * np.exp(-((hours - 8) / 2.0) ** 2) + 0.7 * np.exp(-((hours - 18) / 2.5) ** 2)
        origins = []
        destinations = []
        distances = []
        durations = []
        fares = []
        tips = []
        for boost in demand_boost:
            origin_prob = attractiveness * boost
            origin_prob = origin_prob / origin_prob.sum()
            o = int(rng.choice(zone_ids, p=origin_prob))
            # Nearby zones are more likely, but occasional long trips remain.
            dest_weight = np.array([np.exp(-abs(int(d) - o) / 3.0) * attractiveness[int(d) - 1] for d in zone_ids])
            dest_weight = dest_weight / dest_weight.sum()
            d = int(rng.choice(zone_ids, p=dest_weight))
            ox, oy = (o - 1) % 4, (o - 1) // 4
            dx, dy = (d - 1) % 4, (d - 1) // 4
            dist = max(0.3, float(np.hypot(dx - ox, dy - oy) * 2.1 + rng.gamma(2.0, 0.7)))
            duration = max(3.0, dist / 14.0 * 60.0 + rng.normal(0.0, 3.0))
            fare = max(3.0, 3.0 + 2.6 * dist + 0.22 * duration + rng.normal(0.0, 2.0))
            tip = max(0.0, fare * rng.normal(0.16, 0.07))
            origins.append(o)
            destinations.append(d)
            distances.append(dist)
            durations.append(duration)
            fares.append(fare)
            tips.append(tip)
        pickup_dt = pd.to_datetime(pickup_times)
        dropoff_dt = pickup_dt + pd.to_timedelta(durations, unit="m")
        df = pd.DataFrame(
            {
                "tpep_pickup_datetime": pickup_dt,
                "tpep_dropoff_datetime": dropoff_dt,
                "PULocationID": origins,
                "DOLocationID": destinations,
                "trip_distance": distances,
                "fare_amount": fares,
                "tip_amount": tips,
                "total_amount": np.array(fares) + np.array(tips),
            }
        )
        parquet_path = DATA_DIR / f"yellow_tripdata_{month}.parquet"
        try:
            df.to_parquet(parquet_path, index=False)
            saved_path = parquet_path
        except ImportError:
            saved_path = DATA_DIR / f"yellow_tripdata_{month}.csv"
            df.to_csv(saved_path, index=False)
        print(f"Saved {len(df)} mock trips for {month}: {saved_path.name}")
    print(f"Mock data written to {DATA_DIR}")


if __name__ == "__main__":
    main()
