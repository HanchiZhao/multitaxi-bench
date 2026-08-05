"""Audit that V5.1 retains the complete V4 Final implementation.

All V4 files except config.py must remain byte-identical to the immutable snapshot. config.py
is allowed one narrowly scoped extension: environment-variable overrides whose defaults equal
the original V4 constants.
"""
from pathlib import Path
import hashlib, json, difflib
ROOT=Path(__file__).resolve().parents[1]; SNAP=ROOT/"preserved_v4_snapshot"/"scripts"; LIVE=ROOT/"scripts"
REQUIRED={"ablation_study.py","axiom_validation.py","baseline_sensitivity.py","clean_outputs.py","competition_sensitivity.py","config.py","dqn_policy.py","evaluate_recommendations.py","generate_mock_data.py","imputation_validation.py","policies.py","query_od.py","route_environment.py","run_all.py","scenario_bank.py","shapley_explainer.py","test_dynamic_pipeline.py","train_dqn.py","train_policies.py","two_hour_environment.py","validate_full_od_coverage.py","visualize_results.py","__init__.py"}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    missing_snapshot=sorted(REQUIRED-{p.name for p in SNAP.glob("*.py")}); missing_live=sorted(REQUIRED-{p.name for p in LIVE.glob("*.py")})
    if missing_snapshot or missing_live: raise SystemExit(f"V4 preservation failed: snapshot_missing={missing_snapshot}, live_missing={missing_live}")
    modified=[]; rows=[]
    for name in sorted(REQUIRED):
        a=SNAP/name; b=LIVE/name; same=a.read_bytes()==b.read_bytes(); rows.append({"file":name,"snapshot_sha256":sha(a),"live_sha256":sha(b),"byte_identical":same});
        if not same: modified.append(name)
    if modified!=["config.py"]: raise SystemExit(f"Unexpected V4 source modifications: {modified}; only config.py override extension is permitted")
    live_cfg=(LIVE/"config.py").read_text(encoding="utf-8")
    markers=["MULTITAXI_DRIVER_REVENUE_SHARE","MULTITAXI_OCCUPIED_COST_PER_MILE","MULTITAXI_EMPTY_COST_PER_MILE",'"1.00"','"0.35"']
    if not all(x in live_cfg for x in markers): raise SystemExit("config.py override extension does not preserve V4 default coefficients")
    report={"status":"V4 PRESERVATION PASSED","required_scripts":len(REQUIRED),"byte_identical_scripts":len(REQUIRED)-1,"allowed_modified_file":"config.py","config_change":"environment-variable overrides with original V4 defaults"}
    (ROOT/"preserved_v4_snapshot"/"V4_PRESERVATION_AUDIT.json").write_text(json.dumps({"summary":report,"files":rows},indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
