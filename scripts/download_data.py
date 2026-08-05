"""Download the fixed January-June 2025 TLC dataset directly into the V4-compatible data directory."""
from pathlib import Path
import argparse, hashlib, json, time, zipfile
import requests
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'
def sha256(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default=str(ROOT/'data_manifest.json')); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8')); expected_lock_path=ROOT/'data_lock.json'; expected=json.loads(expected_lock_path.read_text(encoding='utf-8')) if expected_lock_path.exists() else {}; DATA.mkdir(parents=True,exist_ok=True); lock={}
    for item in manifest['files']:
        p=DATA/item['name'];
        if args.force and p.exists(): p.unlink()
        if not p.exists():
            part=p.with_suffix(p.suffix+'.part')
            for attempt in range(4):
                try:
                    with requests.get(item['url'],stream=True,timeout=(20,180)) as r:
                        r.raise_for_status(); total=int(r.headers.get('content-length',0));
                        with part.open('wb') as f:
                            for b in tqdm(r.iter_content(1024*1024),total=max(1,math.ceil(total/(1024*1024))) if total else None,desc=p.name):
                                if b: f.write(b)
                    part.replace(p); break
                except Exception:
                    if attempt==3: raise
                    time.sleep(2**attempt)
        digest=sha256(p); lock[p.name]={'sha256':digest,'size_bytes':p.stat().st_size,'url':item['url']}
        if p.name in expected and (digest!=expected[p.name].get('sha256') or p.stat().st_size!=expected[p.name].get('size_bytes')): raise RuntimeError(f'Data lock mismatch for {p.name}')
    z=DATA/'taxi_zones.zip'
    if z.exists():
        with zipfile.ZipFile(z) as f: f.extractall(DATA)
    (DATA/'checksums.lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    print('DATA DOWNLOAD PASSED' + (' WITH FROZEN LOCK' if expected else ' (no root data_lock.json yet; run freeze_data_lock.py and commit it)'))
if __name__=='__main__':
    import math
    main()
