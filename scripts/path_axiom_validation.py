"""Validation checks for the path-instance attribution extension.

The original V4 axiom suite remains unchanged in scripts/axiom_validation.py. This file adds
path-specific checks and performs an actual +5% event-revenue perturbation with Shapley
recomputation for one representative path.
"""
from __future__ import annotations
import argparse, numpy as np, pandas as pd
from config import PROCESSED_DIR, RANDOM_SEED, HORIZON_MINUTES
from cost_models import load_yaml, primary_cost_model
from two_hour_environment import DynamicTaxiEnvironment
from path_shapley_explainer import Occurrence, PathGame, exact, mc

def row(test,status,stat,thr,details,hard): return {"test":test,"status":status,"statistic":stat,"threshold":thr,"details":details,"hard_axiom":hard}
def build_occ(g):
    return [Occurrence(i,int(r.event_index),str(r.event_type),int(r.from_zone),int(r.to_zone),float(r.event_duration_minutes),float(r.raw_revenue),float(r.distance_miles),float(r.confidence)) for i,r in enumerate(g.itertuples(index=False))]
def compute(game,n,seed):
    if n<=10: return exact(game,n)[0]
    return mc(game,n,128,seed)[0]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args(); cfg=load_yaml(args.config); model=primary_cost_model(cfg)
    sv=pd.read_csv(PROCESSED_DIR/"path_shapley_values.csv"); ef=pd.read_csv(PROCESSED_DIR/"path_shapley_efficiency_checks.csv"); cv=pd.read_csv(PROCESSED_DIR/"path_coalition_values.csv")
    rows=[]
    maxgap=float(ef.efficiency_gap.abs().max()) if len(ef) else np.nan
    rows.append(row("Path Efficiency","PASS" if maxgap<=1e-6 else "FAIL",maxgap,"<=1e-6","sum(phi)=v(full)-v(empty) for each algorithm-conditioned path",True))
    duplicate=int(sv.node_occurrence_id.duplicated().sum())
    rows.append(row("Occurrence Identity","PASS" if duplicate==0 else "FAIL",duplicate,"0 duplicate IDs","same zone may recur, but algorithm/scenario/path-position IDs remain unique",True))
    finite=bool(np.isfinite(cv.value).all()); horizon_ok=bool((cv.elapsed_minutes<=HORIZON_MINUTES+1e-6).all())
    rows.append(row("Repair Feasibility","PASS" if finite and horizon_ok else "FAIL",int(not(finite and horizon_ok)),"finite and elapsed<=120","baseline bridges use the preserved V4 route model; horizon truncation is feasible, not a failure",True))
    calp=PROCESSED_DIR/"path_value_calibration.csv"
    if calp.exists() and len(pd.read_csv(calp)):
        cal=pd.read_csv(calp); gap=float(cal.calibration_gap.abs().max()); rows.append(row("Full-Path Replay Calibration","PASS" if gap<=1e-6 else "FAIL",gap,"<=1e-6","replaying the full occurrence set must reproduce the representative episode under the same cost model",True))
    neg=float((sv.shapley_value<-1e-9).mean()) if len(sv) else np.nan
    rows.append(row("Signed Contribution","INFO",neg,"diagnostic","negative values are legitimate for costly or redundant path occurrences",False))
    context=sv.groupby("zone_id").filter(lambda x:x[["algorithm","path_id","path_position"]].drop_duplicates().shape[0]>1)
    if context.empty: rows.append(row("Same-Zone Context Sensitivity","N/A",np.nan,"not available","no zone appeared in multiple path contexts",False))
    else:
        spans=context.groupby("zone_id").shapley_value.agg(lambda x:float(x.max()-x.min())); nonzero=float((spans>1e-8).mean())
        rows.append(row("Same-Zone Context Sensitivity","PASS" if nonzero>0 else "WARN",nonzero,">0 expected","same zone is not collapsed across algorithms, scenarios, or positions",False))
    # Actual path DIM with recomputation. Choose a path with a completed passenger event and fewest players.
    traj=pd.read_csv(PROCESSED_DIR/"policy_trajectories_cost_models.csv"); reps=pd.read_csv(PROCESSED_DIR/"representative_trajectories.csv"); env=DynamicTaxiEnvironment.load(); candidates=[]
    for rep in reps.itertuples(index=False):
        g=traj[(traj.algorithm==rep.algorithm)&(traj.scenario_id==rep.scenario_id)].sort_values("event_index")
        passenger=np.flatnonzero(g.event_type.astype(str).eq("OCCUPIED_TRIP").to_numpy())
        if len(passenger): candidates.append((len(g),rep,g,int(passenger[np.argmax(g.iloc[passenger].raw_revenue.to_numpy())])))
    if not candidates: rows.append(row("Path DIM","N/A",np.nan,"not available","no completed passenger event",True))
    else:
        _,rep,g,target=min(candidates,key=lambda x:x[0]); occ=build_occ(g); start_zone=int(g.iloc[0].from_zone); start_time=str(cfg.get("experiment",{}).get("start_time","08:00")); seed=RANDOM_SEED+int(rep.scenario_id)+sum(ord(c) for c in str(rep.algorithm))
        game0=PathGame(env,start_zone,start_time,occ,model); phi0=compute(game0,len(occ),seed)
        pert=[Occurrence(**o.__dict__) for o in occ]; pert[target].raw_revenue*=1.05; game1=PathGame(env,start_zone,start_time,pert,model); phi1=compute(game1,len(pert),seed)
        full0=game0.value((1<<len(occ))-1)[0]; full1=game1.value((1<<len(occ))-1)[0]; dphi=float(phi1[target]-phi0[target]); dfull=float(full1-full0); passed=dfull>=-1e-8 and dphi>=-1e-6
        rows.append(row("Path DIM","PASS" if passed else "FAIL",dphi,">=-1e-6",f"actual +5% revenue perturbation and Shapley recomputation for {rep.algorithm} occurrence {target}; full_value_change={dfull:.6f}",True))
    out=pd.DataFrame(rows); out.to_csv(PROCESSED_DIR/"path_axiom_validation_results.csv",index=False); print(out.to_string(index=False))
if __name__=="__main__": main()
