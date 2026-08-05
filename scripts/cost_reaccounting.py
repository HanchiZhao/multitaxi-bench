"""Re-account every V4 policy episode under all explicit V5.1 cost models."""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from cost_models import load_yaml, all_cost_models, primary_cost_model, take_home
from config import PROCESSED_DIR, ensure_directories


def summarize(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows=[]
    for algorithm,g in df.groupby('algorithm',sort=False):
        x=g[value_col].astype(float); n=len(x); std=float(x.std(ddof=1)) if n>1 else 0.0; se=std/math.sqrt(max(1,n))
        rows.append({'algorithm':algorithm,'episodes':n,'mean_primary_take_home_2h':float(x.mean()),
                     'std_primary_take_home_2h':std,'ci_lower':float(x.mean()-1.96*se),'ci_upper':float(x.mean()+1.96*se),
                     'median_primary_take_home_2h':float(x.median()),'worst_10_percent':float(x.quantile(.10)),
                     'mean_raw_driver_receipts_2h':float(g.raw_driver_receipts_2h.mean()),
                     'mean_total_miles':float((g.occupied_miles+g.empty_miles).mean()),
                     'mean_completed_trips':float(g.completed_trips.mean())})
    out=pd.DataFrame(rows).sort_values('mean_primary_take_home_2h',ascending=False).reset_index(drop=True)
    out.insert(0,'rank',np.arange(1,len(out)+1)); return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args()
    ensure_directories(); cfg=load_yaml(args.config); models=all_cost_models(cfg); primary=primary_cost_model(cfg)
    p=PROCESSED_DIR/'policy_episode_results.csv'
    if not p.exists(): raise FileNotFoundError(f'Missing {p}; run V4 evaluation first')
    df=pd.read_csv(p)
    active_share=float(cfg.get('runtime',{}).get('active_revenue_share',primary.revenue_share))
    # V4 gross revenue was built after DRIVER_REVENUE_SHARE. Recover raw fare+tip receipts.
    df['raw_driver_receipts_2h']=df['gross_revenue_2h'].astype(float)/max(active_share,1e-12)
    for name,m in models.items():
        vals=[take_home(r.raw_driver_receipts_2h,r.occupied_miles,r.empty_miles,r.horizon_minutes,m) for r in df.itertuples(index=False)]
        df[f'{name}_effective_receipts_2h']=[v['effective_driver_receipts'] for v in vals]
        df[f'{name}_mileage_cost_2h']=[v['mileage_cost'] for v in vals]
        df[f'{name}_fixed_lease_cost_2h']=[v['fixed_lease_cost'] for v in vals]
        df[f'{name}_disclosed_fare_share_cost_2h']=[v['disclosed_fare_share_cost'] for v in vals]
        df[f'{name}_take_home_2h']=[v['take_home'] for v in vals]
    primary_col=f'{primary.name}_take_home_2h'
    df['primary_cost_model']=primary.name; df['primary_take_home_2h']=df[primary_col]
    df.to_csv(PROCESSED_DIR/'policy_episode_cost_models.csv',index=False)
    summary=summarize(df,'primary_take_home_2h'); summary.to_csv(PROCESSED_DIR/'algorithm_cost_model_comparison.csv',index=False)
    # paired common-scenario comparison under the configured primary model
    pivot=df.pivot_table(index='scenario_id',columns='algorithm',values='primary_take_home_2h',aggfunc='first')
    baseline='wait_only'; rows=[]
    if baseline in pivot.columns:
        for alg in pivot.columns:
            if alg==baseline: continue
            x=(pivot[alg]-pivot[baseline]).dropna(); n=len(x); std=float(x.std(ddof=1)) if n>1 else 0.0; se=std/math.sqrt(max(1,n))
            rows.append({'algorithm':alg,'baseline':baseline,'paired_scenarios':n,'mean_paired_gain':float(x.mean()),
                         'paired_ci_lower':float(x.mean()-1.96*se),'paired_ci_upper':float(x.mean()+1.96*se),
                         'win_rate':float((x>1e-9).mean()),'tie_rate':float((x.abs()<=1e-9).mean()),'loss_rate':float((x<-1e-9).mean())})
    pd.DataFrame(rows).to_csv(PROCESSED_DIR/'paired_cost_model_comparisons.csv',index=False)
    # add event-level cost fields for representative trajectories
    tp=PROCESSED_DIR/'policy_trajectories.csv'
    if tp.exists():
        t=pd.read_csv(tp); t['raw_revenue']=t['revenue'].astype(float)/max(active_share,1e-12)
        for name,m in models.items():
            is_occ=t.event_type.astype(str).isin(['OCCUPIED_TRIP','UNFINISHED_TRIP'])
            is_empty=t.event_type.astype(str).eq('EMPTY_REPOSITION')
            t[f'{name}_effective_revenue']=t.raw_revenue*m.revenue_share
            t[f'{name}_movement_cost']=np.where(is_occ,t.distance_miles*m.occupied_cost_per_mile,np.where(is_empty,t.distance_miles*m.empty_cost_per_mile,0.0))
            t[f'{name}_event_take_home']=t[f'{name}_effective_revenue']-t[f'{name}_movement_cost']
        t.to_csv(PROCESSED_DIR/'policy_trajectories_cost_models.csv',index=False)
    print(summary.to_string(index=False)); print('Cost re-accounting complete.')
if __name__=='__main__': main()
