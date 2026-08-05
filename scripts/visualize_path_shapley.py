"""Visualizations for realistic cost accounting and algorithm-conditioned path Shapley."""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from config import PROCESSED_DIR, FIGURES_DIR, ensure_directories
from cost_models import load_yaml, primary_cost_model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args(); ensure_directories(); cfg=load_yaml(args.config); model=primary_cost_model(cfg)
    summary=pd.read_csv(PROCESSED_DIR/'algorithm_cost_model_comparison.csv')
    fig,ax=plt.subplots(figsize=(10,5)); ax.bar(summary.algorithm,summary.mean_primary_take_home_2h,yerr=[summary.mean_primary_take_home_2h-summary.ci_lower,summary.ci_upper-summary.mean_primary_take_home_2h],capsize=4); ax.axhline(0,linewidth=.8); ax.set_ylabel(f'Mean two-hour take-home (USD)\nPrimary cost model: {model.name}'); ax.set_title('Policy Performance After Explicit Cost Accounting'); ax.tick_params(axis='x',rotation=35); fig.tight_layout(); fig.savefig(FIGURES_DIR/'policy_take_home_comparison.png',dpi=220); plt.close(fig)
    episodes=pd.read_csv(PROCESSED_DIR/'policy_episode_cost_models.csv'); model_cols=[c for c in episodes if c.endswith('_take_home_2h') and c!='primary_take_home_2h']
    rows=[]
    for c in model_cols:
        name=c[:-len('_take_home_2h')]
        for alg,g in episodes.groupby('algorithm'): rows.append({'cost_model':name,'algorithm':alg,'mean':g[c].mean()})
    cm=pd.DataFrame(rows); pivot=cm.pivot(index='algorithm',columns='cost_model',values='mean')
    fig,ax=plt.subplots(figsize=(10,5)); pivot.plot(kind='bar',ax=ax); ax.set_ylabel('Mean two-hour take-home (USD)'); ax.set_title('Cost-Model Sensitivity'); ax.tick_params(axis='x',rotation=35); fig.tight_layout(); fig.savefig(FIGURES_DIR/'cost_model_comparison.png',dpi=220); plt.close(fig)
    sv=pd.read_csv(PROCESSED_DIR/'path_shapley_values.csv')
    algs=list(sv.algorithm.drop_duplicates()); n=len(algs); cols=2; rows_n=max(1,math.ceil(n/cols)); fig,axes=plt.subplots(rows_n,cols,figsize=(14,4*rows_n),squeeze=False)
    for ax,alg in zip(axes.flat,algs):
        g=sv[sv.algorithm==alg].sort_values('path_position'); labels=[f"{int(r.path_position)}:{r.zone_name}\n{r.node_role}" for r in g.itertuples(index=False)]; ax.barh(labels,g.shapley_value); ax.axvline(0,linewidth=.8); ax.set_title(f'{alg}: path-occurrence Shapley'); ax.set_xlabel('Contribution to path take-home (USD)')
    for ax in axes.flat[len(algs):]: ax.axis('off')
    fig.tight_layout(); fig.savefig(FIGURES_DIR/'algorithm_path_shapley_bars.png',dpi=220); plt.close(fig)
    # Cross-algorithm heatmap. Cells are mean only within an algorithm; missing remains NaN, not zero.
    heat=sv.pivot_table(index=['zone_id','zone_name'],columns='algorithm',values='shapley_value',aggfunc='mean')
    if not heat.empty:
        fig,ax=plt.subplots(figsize=(max(8,len(heat.columns)*1.2),max(5,len(heat)*.35))); im=ax.imshow(np.ma.masked_invalid(heat.to_numpy()),aspect='auto'); ax.set_xticks(range(len(heat.columns)),heat.columns,rotation=35,ha='right'); ax.set_yticks(range(len(heat.index)),[f'{z} {n}' for z,n in heat.index]); ax.set_title('Same Taxi Zone Can Have Different Path-Conditioned Shapley Values'); fig.colorbar(im,ax=ax,label='Mean occurrence Shapley (USD)'); fig.tight_layout(); fig.savefig(FIGURES_DIR/'cross_algorithm_zone_shapley_heatmap.png',dpi=220); plt.close(fig)

    # Algorithm-conditioned path maps with a common Shapley scale.
    cent_path=PROCESSED_DIR/'zone_centroids.csv'
    if cent_path.exists() and not sv.empty:
        cent=pd.read_csv(cent_path).set_index('zone_id'); vmax=max(1e-9,float(sv.shapley_value.abs().max())); cols_n=2; rows_map=max(1,math.ceil(len(algs)/cols_n)); fig,axes=plt.subplots(rows_map,cols_n,figsize=(14,5*rows_map),squeeze=False)
        for ax,alg in zip(axes.flat,algs):
            g=sv[sv.algorithm==alg].sort_values('path_position').copy(); xs=[]; ys=[]
            start_zone=int(g.iloc[0].from_zone) if len(g) else None
            if start_zone in cent.index: xs.append(float(cent.loc[start_zone].centroid_lon)); ys.append(float(cent.loc[start_zone].centroid_lat))
            for r in g.itertuples(index=False):
                if int(r.zone_id) in cent.index: xs.append(float(cent.loc[int(r.zone_id)].centroid_lon)); ys.append(float(cent.loc[int(r.zone_id)].centroid_lat))
            if len(xs)>1: ax.plot(xs,ys,linewidth=1,alpha=.55)
            for r in g.itertuples(index=False):
                if int(r.zone_id) not in cent.index: continue
                x=float(cent.loc[int(r.zone_id)].centroid_lon); y=float(cent.loc[int(r.zone_id)].centroid_lat); marker='D' if r.node_role=='algorithm_reposition_target' else ('o' if r.node_role=='passenger_dropoff' else 's'); size=35+160*abs(float(r.shapley_value))/vmax
                sc=ax.scatter([x],[y],c=[float(r.shapley_value)],vmin=-vmax,vmax=vmax,cmap='coolwarm',s=size,marker=marker,edgecolors='black',linewidths=.4); ax.annotate(str(int(r.path_position)),(x,y),fontsize=7)
            if start_zone in cent.index: ax.scatter([float(cent.loc[start_zone].centroid_lon)],[float(cent.loc[start_zone].centroid_lat)],marker='*',s=180,edgecolors='black',linewidths=.6)
            ax.set_title(f'{alg} | path-specific node occurrences'); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        for ax in axes.flat[len(algs):]: ax.axis('off')
        fig.colorbar(sc,ax=axes.ravel().tolist(),label='Path-conditioned Shapley (USD)',shrink=.75); fig.suptitle('Algorithm-Conditioned Complete-Trajectory Shapley Maps',y=.995); fig.savefig(FIGURES_DIR/'algorithm_conditioned_path_shapley_maps.png',dpi=220,bbox_inches='tight'); plt.close(fig)

    print('V5.1 extension figures saved to',FIGURES_DIR)
if __name__=='__main__': main()
