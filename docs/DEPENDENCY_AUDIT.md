# Repository dependency and cleanup audit

## Operational call chain

```text
run_paper.ps1 / run_paper.sh
  -> verify_data.py --strict-lock
  -> run_v52.py
       -> verify_v4_preservation.py
       -> verify_research_contract.py
       -> preflight.py
       -> calibrate_fare_share.py
       -> run_all.py
            -> clean_outputs.py
            -> route_environment.py -> test_dynamic_pipeline.py
            -> validate_full_od_coverage.py -> scenario_bank.py
            -> train_policies.py / train_dqn.py
            -> evaluate_recommendations.py
            -> shapley_explainer.py
            -> imputation / ablation / baseline / axiom / competition checks
            -> visualize_results.py
       -> verify_data.py -> verify_date_boundary_fix.py
       -> cost_reaccounting.py
       -> path_shapley_explainer.py
       -> path_axiom_validation.py
       -> visualize_path_shapley.py
       -> verify_reproduction.py
  -> final_experiment_acceptance.py
```

The compatibility entry point `run_v51.py` redirects to `run_v52.py`; it cannot execute
the superseded 194-service-day exposure. Smoke, setup, tests, CI, data verification, and
all scripts reachable from this chain are retained.

## Removed after dependency tracing

- duplicate `preserved_v4_snapshot/`, replaced by the reviewed SHA256 core manifest;
- migration-only PowerShell scripts, status files, audit bundles, and transfer guides;
- duplicate unpinned requirement variants;
- the invalid fixed-lease-primary configuration (cost sensitivities remain post hoc);
- stale generated `processed_data/`, `results/`, trained models, caches, and logs;
- retired CleanRoom R2-R4 orchestration and failure evidence;
- the superseded `paper_artifacts/final_20260807` evidence folder in the v5.2 branch.

The mock generator remains because isolated smoke and CI depend on it. `query_od.py`
remains as a supported inspection utility. No protected V4 environment, policy, learning,
Global Shapley, or validation implementation was removed.

## Generated-output policy

Runtime outputs are ignored. A curated, hash-manifested copy of the accepted 2026-08-11
reports, tables, evaluation rows, and figures is stored in
`paper_artifacts/final_20260811/`. Raw TLC Parquet, full OD/environment caches, trained
weights, logs, virtual environments, archives, and backups are never committed.
