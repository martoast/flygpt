"""Storage-only adapter: preserve all checkpoints while a USB archive is offline."""
import argparse
import importlib
import os
from pathlib import Path
import shutil
import time
from src.provenance import save_json,sha256

DISK=Path('/Volumes/Seagate');BACKUP=DISK/'FlyGPT Backups/G2c-overnight'
PENDING=Path('artifacts/compiler_pending').absolute()
RESERVE=3*1024**3

def copy_verified(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True);digest=sha256(source)
    if destination.exists():
        assert sha256(destination)==digest;return digest
    temp=destination.with_suffix('.pt.partial')
    # A failed prior copy is never accepted as a completed checkpoint.
    with source.open('rb') as src,temp.open('wb') as dst:
        shutil.copyfileobj(src,dst,8*1024*1024);dst.flush();os.fsync(dst.fileno())
    assert sha256(temp)==digest;temp.replace(destination);return digest

def flush_pending():
    if not DISK.is_mount():return
    migrated=[]
    for p in PENDING.rglob('step_*.pt'):
        if p.is_symlink():continue
        dest=BACKUP/p.relative_to(PENDING);digest=copy_verified(p,dest)
        link=p.with_suffix('.pt.link');link.symlink_to(dest);link.replace(p)
        migrated.append({'path':str(p),'external':str(dest),'sha256':digest})
    if migrated:
        stamp=time.time_ns();save_json(f'results/compiler_v1/storage_migrations/{stamp}.json',{'verified_external':True,'checkpoints':migrated})

def archive(checkpoint,spec,step):
    relative=Path(spec['job_dir'])/f'step_{step:05d}.pt'
    if relative.is_absolute():raise ValueError('Archive job paths must be relative to the repository')
    flush_pending()
    notice=False
    while not DISK.is_mount() and shutil.disk_usage(PENDING.parent if PENDING.parent.exists() else Path('.')).free<RESERVE+checkpoint.stat().st_size:
        if not notice:print('WAITING: local archive reserve reached; reconnect Seagate to preserve checkpoint and continue.',flush=True);notice=True
        time.sleep(20)
    external=DISK.is_mount();destination=(BACKUP if external else PENDING)/relative
    digest=copy_verified(checkpoint,destination)
    return {'step':step,'path':str(destination),'sha256':digest,'storage':'verified_external' if external else 'local_pending_external_backup'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',required=True);p.add_argument('mode',choices=['train']);p.add_argument('--spec',required=True);p.add_argument('--until',type=int,required=True);a=p.parse_args()
    engine=importlib.import_module(a.engine);engine.archive=archive
    engine.train(a.spec,a.until)

if __name__=='__main__':main()
