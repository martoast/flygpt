"""Hash-verified off-machine backup of M1 checkpoints while USB is on the mini."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from src.provenance import save_json,sha256
from scripts.distributed_confirmation import SSH,ROOT

PENDING=Path('artifacts/compiler_pending')
BACKUP='/Volumes/Seagate/FlyGPT Backups/G2c-overnight/'

def command(code,*args):
    return '/usr/bin/python3 -c '+shlex.quote(code)+' '+ ' '.join(shlex.quote(str(a)) for a in args)

def copy(p,digest):
    target=BACKUP+str(p.relative_to(PENDING));temp=target+'.m1-incoming'
    # System Python on the mini is 3.9, so use a streaming implementation.
    probe="import pathlib,sys,hashlib; p=pathlib.Path(sys.argv[1]); h=hashlib.sha256(); f=p.open('rb') if p.exists() else None\nif f:\n for b in iter(lambda:f.read(8388608),b''): h.update(b)\nprint(h.hexdigest() if f else 'missing')"
    old=subprocess.check_output([*SSH,command(probe,target)],text=True).strip()
    if old==digest:return target
    if old!='missing':raise RuntimeError('Existing external checkpoint differs: '+target)
    mkdir="from pathlib import Path; import sys; Path(sys.argv[1]).parent.mkdir(parents=True,exist_ok=True)"
    subprocess.run([*SSH,command(mkdir,target)],check=True)
    with p.open('rb') as f:subprocess.run([*SSH,'cat > '+shlex.quote(temp)],stdin=f,check=True)
    got=subprocess.check_output([*SSH,command(probe,temp)],text=True).strip()
    if got!=digest:raise RuntimeError('Transferred checkpoint hash mismatch')
    finalize="import os,sys; f=open(sys.argv[1],'rb'); os.fsync(f.fileno()); f.close(); os.replace(sys.argv[1],sys.argv[2])"
    subprocess.run([*SSH,command(finalize,temp,target)],check=True)
    return target

def run():
    import fcntl
    lock=(ROOT/'remote_backup.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipts=ROOT/'remote_backup.json';done=json.loads(receipts.read_text()) if receipts.exists() else {}
    while True:
        try:
            for p in sorted((PENDING/ROOT).rglob('step_*.pt')):
                if p.is_symlink() or str(p) in done:continue
                digest=sha256(p);target=copy(p,digest)
                done[str(p)]={'sha256':digest,'external_path':target,'verified_utc_epoch':time.time()}
                save_json(receipts,done)
            if (ROOT/'completion.json').exists():break
        except Exception as exc:
            # Keep local originals intact and retry transport failures.
            save_json(ROOT/'remote_backup_last_error.json',{'error':repr(exc),'time':time.time()})
        time.sleep(60)

if __name__=='__main__':run()
