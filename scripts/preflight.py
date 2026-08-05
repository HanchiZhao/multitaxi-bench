from pathlib import Path
import argparse, importlib, platform, shutil, sys
from cost_models import load_yaml, primary_cost_model
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--require-data',action='store_true'); args=ap.parse_args()
    cfg=load_yaml(args.config); m=primary_cost_model(cfg)
    mods=['numpy','pandas','networkx','matplotlib','geopandas','shapely','yaml']
    if args.require_data: mods.append('pyarrow')
    missing=[]
    for x in mods:
        try: importlib.import_module(x)
        except Exception: missing.append(x)
    if missing: raise SystemExit(f'Missing Python packages: {missing}')
    if sys.version_info<(3,10): raise SystemExit('Python 3.10+ required; 3.11 recommended')
    if args.require_data and not (ROOT/'data'/'yellow_tripdata_2025-01.parquet').exists(): raise SystemExit('Official data missing; run scripts/download_data.py')
    print(f'PREFLIGHT PASSED | Python={platform.python_version()} | cost_model={m.name} | free_GB={shutil.disk_usage(ROOT).free/1e9:.1f}')
if __name__=='__main__': main()
