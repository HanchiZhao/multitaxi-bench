from pathlib import Path
import hashlib, json
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MONTHS = [f"2025-{m:02d}" for m in range(1, 7)]
EXPECTED_COLUMNS = {
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID",
    "trip_distance", "fare_amount", "tip_amount"
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    required_shape = ["taxi_zones.shp", "taxi_zones.shx", "taxi_zones.dbf", "taxi_zones.prj", "taxi_zones.cpg"]
    missing_shape = [x for x in required_shape if not (DATA / x).exists()]
    if missing_shape:
        raise SystemExit(f"Incomplete shapefile set: {missing_shape}")

    lookup = DATA / "taxi_zone_lookup.csv"
    if not lookup.exists():
        raise SystemExit("taxi_zone_lookup.csv missing")
    lookup_df = pd.read_csv(lookup)
    if "LocationID" not in lookup_df.columns or lookup_df["LocationID"].nunique() < 263:
        raise SystemExit("Taxi-zone lookup is invalid")

    present, missing = [], []
    schemas = {}
    for month in MONTHS:
        p = DATA / f"yellow_tripdata_{month}.parquet"
        if not p.exists():
            missing.append(p.name)
            continue
        try:
            pf = pq.ParquetFile(p)
            cols = set(pf.schema_arrow.names)
        except Exception as e:
            raise SystemExit(f"Cannot read {p.name}: {e}")
        miss = sorted(EXPECTED_COLUMNS - cols)
        if miss:
            raise SystemExit(f"{p.name} schema missing: {miss}")
        present.append(p.name)
        schemas[p.name] = sorted(cols)

    report = {
        "status": "PARTIAL_DATA_OK" if missing else "COMPLETE_DATA_OK",
        "present_trip_files": present,
        "missing_trip_files": missing,
        "lookup_rows": int(len(lookup_df)),
        "lookup_unique_location_ids": int(lookup_df["LocationID"].nunique()),
        "files": {},
    }
    for p in sorted(DATA.iterdir()):
        if p.is_file() and p.name != "checksums.lock.json":
            report["files"][p.name] = {"size_bytes": p.stat().st_size, "sha256": sha256(p)}
    (ROOT / "MIGRATION_DATA_AUDIT_RUNTIME.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "present_trip_files", "missing_trip_files", "lookup_rows"]}, indent=2))
    if missing:
        print("Partial migration is valid. download_data.py will fetch only missing official files.")
    else:
        print("All January-June trip files are present.")

if __name__ == "__main__":
    main()
