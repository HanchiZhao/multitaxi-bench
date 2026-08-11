"""Verify the fixed January-June 2025 TLC input bundle without loading whole parquet files."""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd
import pyarrow.parquet as pq
from config import DATA_DIR, MONTHS, PROCESSED_DIR

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PARQUET_COLUMNS = {
    'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID',
    'trip_distance', 'fare_amount', 'tip_amount'
}
REQUIRED_SHAPE_FILES = ['taxi_zones.shp','taxi_zones.shx','taxi_zones.dbf','taxi_zones.prj','taxi_zones.cpg']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--strict-lock', action='store_true')
    args = ap.parse_args()

    required = [DATA_DIR / f'yellow_tripdata_{m}.parquet' for m in MONTHS]
    required += [DATA_DIR / 'taxi_zone_lookup.csv']
    required += [DATA_DIR / name for name in REQUIRED_SHAPE_FILES]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit('Missing data files:\n' + '\n'.join(missing))

    # Validate every monthly parquet from metadata only; never load a full month into RAM here.
    row_counts = {}
    schemas = {}
    for p in [DATA_DIR / f'yellow_tripdata_{m}.parquet' for m in MONTHS]:
        try:
            pf = pq.ParquetFile(p)
            cols = set(pf.schema_arrow.names)
        except Exception as exc:
            raise SystemExit(f'Cannot open parquet {p.name}: {exc}') from exc
        miss = sorted(REQUIRED_PARQUET_COLUMNS - cols)
        if miss:
            raise SystemExit(f'{p.name} schema missing required columns: {miss}')
        if pf.metadata.num_rows <= 0:
            raise SystemExit(f'{p.name} has no rows')
        row_counts[p.name] = int(pf.metadata.num_rows)
        schemas[p.name] = sorted(cols)

    lookup = pd.read_csv(DATA_DIR / 'taxi_zone_lookup.csv')
    if 'LocationID' not in lookup.columns or lookup['LocationID'].nunique() < 263:
        raise SystemExit('taxi_zone_lookup.csv is invalid or incomplete')

    verified_hashes = {}
    lock_path = None
    if args.strict_lock:
        root_lock = ROOT / 'data_lock.json'
        local_lock = DATA_DIR / 'checksums.lock.json'
        lockp = root_lock if root_lock.exists() else local_lock
        lock_path = str(lockp)
        if not lockp.exists():
            raise SystemExit('No data checksum lock found. Expected data_lock.json or data/checksums.lock.json')
        lock = json.loads(lockp.read_text(encoding='utf-8'))
        bad = []
        for name, meta in lock.items():
            p = DATA_DIR / name
            if not p.exists():
                bad.append(f'{name}: missing')
                continue
            actual_size = p.stat().st_size
            actual_sha = sha256(p)
            verified_hashes[name] = {
                'sha256': actual_sha,
                'size_bytes': int(actual_size),
            }
            if actual_sha != meta.get('sha256') or actual_size != meta.get('size_bytes'):
                bad.append(f'{name}: checksum/size mismatch')
        if bad:
            raise SystemExit('Data lock mismatch:\n' + '\n'.join(bad))

    report = {
        'status': 'DATA LOCK VERIFICATION PASSED' if args.strict_lock else 'DATA VERIFICATION PASSED',
        'strict_lock': bool(args.strict_lock),
        'lock_path': lock_path,
        'months': list(MONTHS),
        'row_counts': row_counts,
        'verified_files': verified_hashes,
        'lookup_location_ids': int(lookup['LocationID'].nunique()),
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / 'data_lock_verification.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8'
    )

    print('DATA VERIFICATION PASSED')
    print(f'Months verified: {len(MONTHS)} | lookup LocationIDs: {lookup["LocationID"].nunique()}')
    for name in sorted(row_counts):
        print(f'  {name}: {row_counts[name]:,} rows')


if __name__ == '__main__':
    main()
