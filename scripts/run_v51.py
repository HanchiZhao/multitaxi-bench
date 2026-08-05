"""Preservation-first orchestrator: run the complete V4 pipeline, then V5.1 extensions."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
from cost_models import load_yaml, primary_cost_model
ROOT=Path(__file__).resolve().parents[1]
def call(script,*args,env=None):
    cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)]; print('\n'+'='*96+'\nRUN: '+' '.join(cmd)+'\n'+'='*96); subprocess.run(cmd,cwd=ROOT,check=True,env=env)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--skip-download',action='store_true'); ap.add_argument('--skip-v4',action='store_true'); args=ap.parse_args()
    cfg=load_yaml(args.config); exp=cfg.get('experiment',{}); run=cfg.get('run',{}); model=primary_cost_model(cfg); env=os.environ.copy(); env.update(model.env()); env['PYTHONHASHSEED']=str(exp.get('random_seed',20260712))
    if run.get('deterministic_cpu',True): env['CUDA_VISIBLE_DEVICES']=''
    # communicate active revenue share to post-hoc accounting without mutating source config
    cfg.setdefault('runtime',{})['active_revenue_share']=model.revenue_share
    # save resolved config used by this run
    import yaml
    resolved=ROOT/'processed_data'/'resolved_v51_config.yaml'; resolved.parent.mkdir(exist_ok=True); resolved.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    call('verify_v4_preservation.py',env=env); call('verify_research_contract.py','--static-only',env=env); call('preflight.py','--config',str(resolved),env=env)
    if run.get('download_data',False) and not args.skip_download: call('download_data.py',env=env); call('verify_data.py','--strict-lock',env=env)
    if not args.skip_v4:
        v4=['--start-zone',str(exp.get('start_zone',132)),'--start-time',str(exp.get('start_time','08:00')),'--episodes',str(exp.get('episodes',300)),'--players',str(exp.get('global_shapley_players',10)),'--permutations',str(exp.get('global_shapley_permutations',512)),'--competition-scenario',str(exp.get('competition_scenario','medium'))]
        if run.get('include_q_learning',True): v4.append('--include-q-learning'); v4 += ['--q-episodes',str(run.get('q_episodes',10000))]
        if run.get('include_dqn',True): v4.append('--include-dqn'); v4 += ['--dqn-episodes',str(run.get('dqn_episodes',8000))]
        if run.get('include_legacy',False): v4.append('--include-legacy')
        if run.get('fast',False): v4.append('--fast')
        if run.get('skip_environment',False): v4.append('--skip-environment')
        if run.get('skip_validation',False): v4.append('--skip-validation')
        if run.get('skip_visualization',False): v4.append('--skip-visualization')
        if run.get('skip_competition_sensitivity',False): v4.append('--skip-competition-sensitivity')
        call('run_all.py',*v4,env=env)
    call('verify_research_contract.py',env=env)
    call('cost_reaccounting.py','--config',str(resolved),env=env)
    call('path_shapley_explainer.py','--config',str(resolved),'--permutations',str(cfg.get('path_shapley',{}).get('permutations',512)),env=env)
    call('path_axiom_validation.py','--config',str(resolved),env=env)
    call('visualize_path_shapley.py','--config',str(resolved),env=env)
    call('verify_reproduction.py','--config',str(resolved),env=env)
    print('\nV5.1 PRESERVATION-FIRST PIPELINE COMPLETED SUCCESSFULLY')
if __name__=='__main__': main()
