# Curated paper artifacts

`final_20260811/` is the accepted MultiTaxi-Bench v5.2 evidence set. It contains selected
machine reports, compact and episode-level tables, and final figures copied or redrawn
from the accepted 2026-08-11 run. Redrawn figures use the same accepted CSV values; only
label placement and layout changed.

`SHA256SUMS.csv` records the byte identity of every artifact in the folder. Verify it with
`python scripts/verify_release_artifacts.py`.

These files support review but do not replace the canonical reproduction path: strict
data verification followed by `run_paper.ps1` or `run_paper.sh`. Raw TLC Parquet, full
environment/OD caches, trained weights, and logs are intentionally excluded.
