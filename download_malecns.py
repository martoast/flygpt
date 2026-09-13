"""Download the official Janelia MaleCNS v1.0 full connection graph.
Public CC-BY data. Requires: pip install gcsfs tqdm
"""
from pathlib import Path
import shutil
import gcsfs

BUCKET='flyem-male-cns'
REMOTE='v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather'
OUT=Path('data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather')
OUT.parent.mkdir(parents=True, exist_ok=True)
fs=gcsfs.GCSFileSystem(token='anon')
print(f'Downloading gs://{BUCKET}/{REMOTE} -> {OUT}')
with fs.open(f'{BUCKET}/{REMOTE}','rb') as src, OUT.open('wb') as dst:
    shutil.copyfileobj(src,dst,length=16*1024*1024)
print('done', OUT, OUT.stat().st_size)
