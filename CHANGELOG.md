# Changelog

## 5.2.0 — 2026-08-11

- Corrected monthly pickup-date boundaries to use each declared half-open month while
  leaving dropoff timestamps unrestricted.
- Rebuilt the six-month environment with 181 service days, excluded all 134 clean
  out-of-month pickup rows, and retained zero residual out-of-month pickups.
- Added shared month-boundary utilities, validation tests, environment schema
  `4.1-date-boundary-fix`, and final hard acceptance gates for the repair.
- Added the mandatory `run_v52.py` pipeline; the legacy `run_v51.py` entry point now
  redirects to v5.2 rather than allowing the superseded exposure definition.
- Recalibrated fare share on the same corrected trip population without changing the
  primary cost definition or deducting fare share twice.
- Re-ran Q-learning (15,000 episodes), Double DQN (12,000 episodes), seven-policy paired
  evaluation (300 common scenarios per policy), both Shapley games, all sensitivities,
  figures, reproduction checks, and final acceptance.
- Replaced the 2026-08-07 v5.1 evidence set with the accepted 2026-08-11 v5.2 reports,
  tables, raw evaluation episodes, and figures.
- Updated all public documentation to the v5.2 results and added an independently
  verifiable release-artifact manifest.
- Improved figure label placement, competition-plot legend placement, and the Path
  Shapley panel layout without changing any computed value.

## 5.1.0 — 2026-08-07

- Frozen the 120-minute sequential WAIT/REPOSITION research contract.
- Retained the complete audited V4 operational core and replaced the duplicate source
  snapshot with a SHA256 core manifest.
- Added the data-derived fare-share primary objective and accounting identities that
  prevent double deduction.
- Added algorithm-conditioned Path-occurrence Shapley and path-specific validation.
- Reworked signed Shapley bars and NYC Taxi Zone path maps, including local-detail maps
  with locator insets.
- Standardized user-facing terminology on `operating net earnings`.
- Made `run_paper` perform strict data verification, full reproduction, and final
  experiment acceptance.
- Isolated smoke data from the official data directory and aligned smoke/fast configs
  with the final fare-share model.
- Removed migration-only documents, duplicate requirements, stale generated outputs,
  and the invalid fixed-lease-primary sensitivity config.
- Added explicit runtime metadata for download-related CLI overrides.
