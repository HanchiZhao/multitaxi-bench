from pathlib import Path
import argparse, json, numpy as np, pandas as pd
from config import PROCESSED_DIR, FIGURES_DIR
from cost_models import load_yaml, primary_cost_model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args(); cfg=load_yaml(args.config); m=primary_cost_model(cfg)
    required=[
        PROCESSED_DIR/'algorithm_recommendation_comparison.csv',PROCESSED_DIR/'paired_policy_comparisons.csv',
        PROCESSED_DIR/'node_shapley_values.csv',PROCESSED_DIR/'algorithm_cost_model_comparison.csv',
        PROCESSED_DIR/'path_shapley_values.csv',PROCESSED_DIR/'path_value_calibration.csv',PROCESSED_DIR/'path_axiom_validation_results.csv',
        FIGURES_DIR/'algorithm_net_earnings_comparison.png',FIGURES_DIR/'policy_take_home_comparison.png',FIGURES_DIR/'algorithm_path_shapley_bars.png',FIGURES_DIR/'algorithm_conditioned_path_shapley_maps.png']
    if not cfg.get('run',{}).get('skip_validation',False):
        required += [PROCESSED_DIR/'axiom_validation_results.csv', PROCESSED_DIR/'imputation_holdout_summary.csv']
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise SystemExit('REPRODUCTION FAILED: missing outputs\n'+'\n'.join(missing))
    ep=pd.read_csv(PROCESSED_DIR/'policy_episode_results.csv')
    if ep.time_accounting_error.abs().max()>1e-6: raise SystemExit('REPRODUCTION FAILED: time accounting')
    eff=pd.read_csv(PROCESSED_DIR/'path_shapley_efficiency_checks.csv')
    if eff.efficiency_gap.abs().max()>1e-6: raise SystemExit('REPRODUCTION FAILED: path Shapley efficiency')
    cal=pd.read_csv(PROCESSED_DIR/'path_value_calibration.csv')
    if len(cal) and cal.calibration_gap.abs().max()>1e-6: raise SystemExit('REPRODUCTION FAILED: path full-value calibration')
    if pd.read_csv(PROCESSED_DIR/'path_shapley_values.csv').node_occurrence_id.duplicated().any(): raise SystemExit('REPRODUCTION FAILED: duplicate path occurrence IDs')
    report={'status':'REPRODUCTION PASSED','primary_cost_model':m.name,'algorithms':sorted(ep.algorithm.unique().tolist()),'max_time_error':float(ep.time_accounting_error.abs().max()),'max_path_efficiency_gap':float(eff.efficiency_gap.abs().max())}
    (PROCESSED_DIR/'reproduction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
