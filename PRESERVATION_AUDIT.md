# V4 Preservation Audit

The `preserved_v4_snapshot/scripts` directory is the immutable V4 Final source.

- 22 live V4 Python files are byte-identical to that snapshot.
- `scripts/config.py` is the only modified V4 file.
- Its only behavioral extension is reading revenue-share and mileage-cost coefficients from environment variables.
- With no environment variables, the defaults remain exactly V4: share `1.00`, occupied cost `$0.35/mile`, empty cost `$0.35/mile`.
- All new functionality is implemented in additional scripts and does not replace V4 global Dynamic Zone Shapley or V4 validations.

`python scripts/verify_v4_preservation.py` fails if any other V4 file changes.
`python scripts/verify_research_contract.py` fails if the 120-minute empty-driver WAIT/REPOSITION objective is altered.
