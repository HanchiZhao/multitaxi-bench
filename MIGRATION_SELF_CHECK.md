# Migration Self-Check

This package was rebuilt after the June 2025 parquet was supplied.

Checks completed before packaging:

- Six monthly files (2025-01 ... 2025-06) are present.
- All six files have valid Apache Parquet `PAR1` head and tail magic.
- Expected schema-name tokens are present in all six parquet metadata regions.
- `taxi_zone_lookup.csv`: 265 unique LocationIDs.
- Taxi-zone shapefile: 263 features, 263 valid geometries, CRS EPSG:2263.
- Exact SHA256 and byte sizes were frozen in `data_lock.json`.
- V4 preservation audit: PASSED (23 required V4 scripts; 22 byte-identical; only `config.py` contains the previously allowed environment-variable extension).
- Research contract static check: PASSED.
- Python source compilation check: PASSED.
- `run_smoke.ps1` was changed so mock data are generated only in `_smoke_workspace`, never in the real `data` directory.
- `run_paper.ps1` was changed to verify the locked local data and run with `--skip-download`, so supplied raw data are not overwritten.

The first local run of `START_MIGRATION.ps1` will perform the stronger runtime checks after the pinned Python environment is installed, including PyArrow schema validation for every month and exact SHA256 verification.
