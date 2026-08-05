from pathlib import Path
import json, shutil
ROOT=Path(__file__).resolve().parents[1]; src=ROOT/"data"/"checksums.lock.json"; dst=ROOT/"data_lock.json"
if not src.exists(): raise SystemExit("Run download_data.py first")
shutil.copy2(src,dst); print(f"Frozen reproducibility lock: {dst}")
