# Accepted v5.2 evidence — 2026-08-11

This folder is the curated evidence set for the accepted MultiTaxi-Bench v5.2 paper run.

- `reports/`: final acceptance, date/data checks, research contract, accounting,
  calibration, reproduction, resolved configuration, and v5.1-to-v5.2 comparison;
- `tables/`: final summaries, 2,100 episode-level policy rows, paired differences,
  learning diagnostics, Shapley values and checks, sensitivity summaries, and training
  histories;
- `figures/`: the complete final figure set.

The formal numerical outputs are unchanged. Four figures were redrawn from the accepted
CSV tables to prevent label/error-bar overlap, move the competition legend outside the
data area, and use a compact Path Shapley panel layout.

Run `python scripts/verify_release_artifacts.py` from the repository root to verify the
SHA256 manifest and the core scientific contract.
