# Repository dependency and cleanup audit

## Operational call chain

```text
run_paper.ps1 / run_paper.sh
  -> verify_data.py --strict-lock
  -> run_v51.py
       -> verify_v4_preservation.py
       -> verify_research_contract.py
       -> preflight.py
       -> run_all.py
            -> clean_outputs.py
            -> route_environment.py -> test_dynamic_pipeline.py
            -> validate_full_od_coverage.py -> scenario_bank.py
            -> train_policies.py / train_dqn.py
            -> evaluate_recommendations.py
            -> shapley_explainer.py
            -> imputation / ablation / baseline / axiom / competition checks
            -> visualize_results.py
       -> cost_reaccounting.py
       -> path_shapley_explainer.py
       -> path_axiom_validation.py
       -> visualize_path_shapley.py
       -> verify_reproduction.py
  -> final_experiment_acceptance.py
```

Smoke, setup, tests, CI, data verification, and all scripts reachable from this chain are
retained.

## Removed after dependency tracing

- `preserved_v4_snapshot/`: duplicate source replaced by the reviewed SHA256 manifest;
- migration-only PowerShell scripts, status files, audit bundles, and transfer guides;
- duplicate unpinned requirement variants;
- `paper_sensitivity.yaml`, whose fixed-lease primary objective is rejected by the current
  orchestrator and is unnecessary because the paper run already re-accounts every cost
  sensitivity post hoc;
- stale committed `processed_data/` and `results/` generated under an older cost model;
- migration-only `migration_data_audit.py` and lock-authoring helper `freeze_data_lock.py`.

The mock generator remains because isolated smoke and CI depend on it. `query_od.py` remains
as a supported inspection utility. No protected V4 environment, policy, learning, Global
Shapley, or validation implementation was removed.

## Generated-output policy

Runtime outputs are ignored. A curated, hash-manifested copy of the accepted 2026-08-07
reports, tables, and figures is stored in `paper_artifacts/`. Raw TLC Parquet, trained model
weights, caches, logs, virtual environments, and backups are never committed.

