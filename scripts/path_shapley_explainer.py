"""Algorithm-conditioned, path-instance Shapley extension for the preserved V4 pipeline.

Each player is an event/node occurrence, not a global taxi-zone ID. The same zone in another
algorithm, scenario, or path position is a distinct player. A removed occurrence is skipped;
when the next retained event begins in a different zone, the value function inserts the V4
shortest-time route as an empty baseline bridge. Passenger destinations remain marked as
exogenous OCCUPIED_TRIP outcomes, while EMPTY_REPOSITION nodes are algorithmic choices.
"""
from __future__ import annotations
import argparse, math, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np, pandas as pd
from cost_models import load_yaml, primary_cost_model
from config import PROCESSED_DIR, HORIZON_MINUTES, RANDOM_SEED, ensure_directories
from two_hour_environment import DynamicTaxiEnvironment, parse_clock_time

@dataclass
class Occurrence:
    index:int; event_index:int; event_type:str; from_zone:int; to_zone:int
    duration:float; raw_revenue:float; original_distance:float; confidence:float


def role(event_type:str)->str:
    if event_type=='EMPTY_REPOSITION': return 'algorithm_reposition_target'
    if event_type in {'OCCUPIED_TRIP','UNFINISHED_TRIP'}: return 'passenger_dropoff'
    if event_type.startswith('WAIT') or event_type in {'TERMINAL_UNUSED','TERMINAL_OR_INVALID_ROUTE_WAIT'}: return 'waiting_location'
    return 'other'

class PathGame:
    def __init__(self,env,start_zone,start_time,occurrences,model,horizon=HORIZON_MINUTES):
        self.env=env; self.start_zone=int(start_zone); self.start_abs=parse_clock_time(start_time)
        self.occ=list(occurrences); self.model=model; self.horizon=float(horizon); self.cache={}
    def value(self,mask:int)->Tuple[float,Dict[str,float]]:
        mask=int(mask)
        if mask in self.cache: return self.cache[mask]
        zone=self.start_zone; elapsed=0.0; effective=0.0; mileage=0.0; bridge_minutes=0.0; valid=True
        for i,o in enumerate(self.occ):
            if not (mask&(1<<i)): continue
            # baseline bridge from the previous retained endpoint to this event's original origin
            if zone!=o.from_zone:
                r=self.env.route(zone,o.from_zone,absolute_minutes=self.start_abs+elapsed)
                dur=float(r['duration_min']); dist=float(r['distance_miles'])
                if elapsed+dur>self.horizon+1e-9: valid=False; elapsed=self.horizon; break
                elapsed+=dur; bridge_minutes+=dur; mileage+=dist*self.model.empty_cost_per_mile; zone=o.from_zone
            dur=max(0.0,float(o.duration))
            if o.event_type=='OCCUPIED_TRIP':
                if elapsed+dur>self.horizon+1e-9: elapsed=self.horizon; break
                effective+=o.raw_revenue*self.model.revenue_share
                mileage+=o.original_distance*self.model.occupied_cost_per_mile
                elapsed+=dur; zone=o.to_zone
            elif o.event_type=='UNFINISHED_TRIP':
                elapsed=min(self.horizon,elapsed+dur); zone=o.to_zone
            elif o.event_type=='EMPTY_REPOSITION':
                if elapsed+dur>self.horizon+1e-9: elapsed=self.horizon; break
                mileage+=o.original_distance*self.model.empty_cost_per_mile
                elapsed+=dur; zone=o.to_zone
            else:
                elapsed=min(self.horizon,elapsed+dur); zone=o.to_zone
            if elapsed>=self.horizon-1e-9: break
        fixed=self.horizon/60.0*self.model.fixed_lease_per_hour
        val=effective-mileage-fixed
        info={'elapsed_minutes':elapsed,'bridge_minutes':bridge_minutes,'effective_receipts':effective,
              'mileage_cost':mileage,'fixed_cost':fixed,'valid':float(valid),'end_zone':zone}
        self.cache[mask]=(float(val),info); return self.cache[mask]


def exact(game:PathGame,n:int):
    vals={m:game.value(m)[0] for m in range(1<<n)}; phi=np.zeros(n)
    den=math.factorial(n)
    for i in range(n):
        bit=1<<i
        for m in range(1<<n):
            if m&bit: continue
            k=m.bit_count(); w=math.factorial(k)*math.factorial(n-k-1)/den
            phi[i]+=w*(vals[m|bit]-vals[m])
    return phi,np.zeros(n),vals

def mc(game:PathGame,n:int,permutations:int,seed:int):
    rng=np.random.default_rng(seed); samples=[]
    for _ in range((permutations+1)//2):
        p=rng.permutation(n)
        for order in (p,p[::-1]):
            c=np.zeros(n); m=0; before=game.value(0)[0]
            for idx in order:
                nm=m|(1<<int(idx)); after=game.value(nm)[0]; c[int(idx)]=after-before; m=nm; before=after
            samples.append(c)
            if len(samples)>=permutations: break
        if len(samples)>=permutations: break
    a=np.vstack(samples); return a.mean(0),a.std(0,ddof=1)/math.sqrt(len(a)) if len(a)>1 else np.zeros(n),dict((k,v[0]) for k,v in game.cache.items())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--permutations',type=int,default=None); ap.add_argument('--exact-max',type=int,default=10); args=ap.parse_args()
    ensure_directories(); cfg=load_yaml(args.config); model=primary_cost_model(cfg); perms=int(args.permutations or cfg.get('path_shapley',{}).get('permutations',512))
    traj_path=PROCESSED_DIR/'policy_trajectories_cost_models.csv'
    reps_path=PROCESSED_DIR/'representative_trajectories.csv'
    if not traj_path.exists() or not reps_path.exists(): raise FileNotFoundError('Run evaluate_recommendations.py and cost_reaccounting.py first')
    traj=pd.read_csv(traj_path); reps=pd.read_csv(reps_path); env=DynamicTaxiEnvironment.load()
    out_rows=[]; coalition_rows=[]; eff_rows=[]; path_rows=[]; calibration_rows=[]
    episode_cost_path=PROCESSED_DIR/'policy_episode_cost_models.csv'
    episode_cost=pd.read_csv(episode_cost_path) if episode_cost_path.exists() else pd.DataFrame()
    for rep in reps.itertuples(index=False):
        g=traj[(traj.algorithm==rep.algorithm)&(traj.scenario_id==rep.scenario_id)].sort_values('event_index').copy()
        if g.empty: continue
        occ=[]
        for pos,r in enumerate(g.itertuples(index=False)):
            occ.append(Occurrence(pos,int(r.event_index),str(r.event_type),int(r.from_zone),int(r.to_zone),
                                  float(r.event_duration_minutes),float(r.raw_revenue),float(r.distance_miles),float(r.confidence)))
        n=len(occ); game=PathGame(env,int(rep.start_zone) if hasattr(rep,'start_zone') else int(g.iloc[0].from_zone),str(rep.start_time) if hasattr(rep,'start_time') else cfg.get('experiment',{}).get('start_time','08:00'),occ,model)
        seed=RANDOM_SEED+int(rep.scenario_id)+sum(ord(c) for c in str(rep.algorithm))
        if n<=args.exact_max: phi,se,vals=exact(game,n); method='exact'
        else: phi,se,vals=mc(game,n,perms,seed); method='antithetic_mc'
        empty=game.value(0)[0]; full=game.value((1<<n)-1)[0]; gap=float(phi.sum()-(full-empty))
        path_id=f"{rep.algorithm}__scenario_{int(rep.scenario_id)}"
        eff_rows.append({'algorithm':rep.algorithm,'scenario_id':int(rep.scenario_id),'path_id':path_id,'players':n,'method':method,'v_empty':empty,'v_full':full,'sum_shapley':float(phi.sum()),'efficiency_gap':gap,'passed':abs(gap)<=1e-6})
        for i,o in enumerate(occ):
            zone_name=env.zone_names.get(o.to_zone,str(o.to_zone))
            out_rows.append({'algorithm':rep.algorithm,'scenario_id':int(rep.scenario_id),'path_id':path_id,
                'node_occurrence_id':f'{path_id}@{i:03d}','path_position':i,'event_index':o.event_index,
                'zone_id':o.to_zone,'zone_name':zone_name,'node_role':role(o.event_type),'event_type':o.event_type,
                'from_zone':o.from_zone,'to_zone':o.to_zone,'event_duration_minutes':o.duration,
                'raw_event_revenue':o.raw_revenue,'event_distance_miles':o.original_distance,'confidence':o.confidence,
                'shapley_value':float(phi[i]),'shapley_se':float(se[i]),'ci_lower':float(phi[i]-1.96*se[i]),'ci_upper':float(phi[i]+1.96*se[i]),
                'cost_model':model.name,'method':method})
        # only save cached coalition values; exact saves all, MC saves visited values
        for mask,val in sorted(vals.items()):
            info=game.value(mask)[1]
            coalition_rows.append({'algorithm':rep.algorithm,'scenario_id':int(rep.scenario_id),'path_id':path_id,'mask':int(mask),'coalition_size':int(mask).bit_count(),'value':float(val),**info})
        path_rows.append({'algorithm':rep.algorithm,'scenario_id':int(rep.scenario_id),'path_id':path_id,'node_occurrences':n,'v_empty':empty,'v_full':full,'path_gain':full-empty,'cost_model':model.name})
        if not episode_cost.empty:
            match=episode_cost[(episode_cost.algorithm==rep.algorithm)&(episode_cost.scenario_id==int(rep.scenario_id))]
            if not match.empty:
                expected=float(match.iloc[0].primary_take_home_2h); calibration_rows.append({'algorithm':rep.algorithm,'scenario_id':int(rep.scenario_id),'path_id':path_id,'replayed_full_value':full,'episode_primary_take_home':expected,'calibration_gap':full-expected,'passed':abs(full-expected)<=1e-6})
    sv=pd.DataFrame(out_rows); cv=pd.DataFrame(coalition_rows); ef=pd.DataFrame(eff_rows); ps=pd.DataFrame(path_rows)
    sv.to_csv(PROCESSED_DIR/'path_shapley_values.csv',index=False); cv.to_csv(PROCESSED_DIR/'path_coalition_values.csv',index=False); ef.to_csv(PROCESSED_DIR/'path_shapley_efficiency_checks.csv',index=False); ps.to_csv(PROCESSED_DIR/'path_shapley_summary.csv',index=False); pd.DataFrame(calibration_rows).to_csv(PROCESSED_DIR/'path_value_calibration.csv',index=False)
    # same-zone, cross-context table; no aggregation that overwrites occurrence identity
    if not sv.empty:
        context=sv.groupby('zone_id').filter(lambda x: x[['algorithm','path_id','path_position']].drop_duplicates().shape[0]>1).copy()
        context.to_csv(PROCESSED_DIR/'same_zone_cross_path_context.csv',index=False)
    print(ef.to_string(index=False)); print('Algorithm-conditioned path Shapley complete.')
if __name__=='__main__': main()
