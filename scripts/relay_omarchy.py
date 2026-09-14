"""Publish Omarchy commits and stream hash-verified backups to mini-attached Seagate."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time
from src.provenance import save_json

LOCAL = Path(__file__).resolve().parents[1]
ROOT = '/home/alex/flygpt_efficiency'
BRANCH = 'experiment/compiler-efficiency'
REF = 'refs/remotes/omarchy/' + BRANCH
KNOWN = str(Path.home()/'.ssh/flygpt_omarchy_known_hosts')
LINUX = ['ssh','-o','UserKnownHostsFile='+KNOWN,'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','alex@omarchy.tail61505e.ts.net']
MINI = ['ssh','-i',str(Path.home()/'.ssh/flygpt_mac_mini_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','alex@alexs-mac-mini']
BACKUP = '/Volumes/Seagate/FlyGPT Backups/G2c-overnight/'


def command(*args):
    return subprocess.check_output(args,cwd=LOCAL,text=True,stderr=subprocess.STDOUT,timeout=120).strip()


def python_command(code,linux=False):
    return (ROOT+'/.venv/bin/python' if linux else '/Users/alex/flygpt/.venv/bin/python')+' -c '+shlex.quote(code)


def inventory():
    code = f'''from pathlib import Path
import json,hashlib
root=Path({ROOT!r}); study=root/'results/compiler_efficiency'; files={{}}
def add(p,relative,digest=None):
 if p.is_file():
  if digest is None:
   with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  files[str(p)]={{'source':str(p),'relative':str(relative),'sha256':digest}}
for p in study.rglob('progress.json'):
 d=json.loads(p.read_text())
 for a in d.get('archives',[]):
  source=Path(a['path']); add(source,source.relative_to(root/'artifacts/compiler_pending'),a['sha256'])
for p in study.rglob('*'):
 if p.suffix in ('.json','.jsonl','.md','.py','.log') or p.name=='teacher_cache.pt':add(p,p.relative_to(root))
plan=study/'frozen_plan.json'
if plan.exists():
 for name in json.loads(plan.read_text())['inputs']:
  p=root/name
  if p.suffix in ('.json','.py','.md'):add(p,name)
for name in ['COMPILER_EFFICIENCY_REPORT.md','COMPILER_EFFICIENCY_PREREGISTRATION.md']:
 add(root/name,name)
print(json.dumps(list(files.values())))
'''
    return json.loads(command(*LINUX,python_command(code,True)))


def transfer(item):
    relative=Path(item['relative'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid backup path')
    target=BACKUP+str(relative); temporary=target+'.omarchy-incoming'; digest=item['sha256']
    probe=f'''from pathlib import Path
import hashlib
assert Path('/Volumes/Seagate').is_mount()
p=Path({target!r}); p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():
 with p.open('rb') as f:print(hashlib.file_digest(f,'sha256').hexdigest())
else:print('missing')
'''
    old=command(*MINI,python_command(probe))
    if old==digest:return
    if old!='missing' and target.endswith('.pt'):raise RuntimeError('Immutable checkpoint conflict: '+target)
    producer=subprocess.Popen([*LINUX,'cat -- '+shlex.quote(item['source'])],stdout=subprocess.PIPE)
    consumer=subprocess.Popen([*MINI,'cat > '+shlex.quote(temporary)],stdin=producer.stdout)
    producer.stdout.close()
    try:
        c=consumer.wait(timeout=600); p=producer.wait(timeout=30)
        if c or p:raise RuntimeError(f'Transfer failed: {p}, {c}')
    finally:
        for process in [producer,consumer]:
            if process.poll() is None:process.kill();process.wait()
    check=f'''from pathlib import Path
import hashlib,os
p=Path({temporary!r})
with p.open('rb') as f:
 assert hashlib.file_digest(f,'sha256').hexdigest()=={digest!r}
 os.fsync(f.fileno())
os.replace(p,{target!r})
'''
    command(*MINI,python_command(check))


def main():
    state=LOCAL/'results/compiler_efficiency_relay';state.mkdir(parents=True,exist_ok=True)
    lock=(state/'relay.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipts=state/'backups.json';done=json.loads(receipts.read_text()) if receipts.exists() else {}
    while True:
        try:
            ssh=' '.join(shlex.quote(x) for x in LINUX[:-1])
            command('git','-c','core.sshCommand='+ssh,'fetch','ssh://alex@omarchy.tail61505e.ts.net'+ROOT,BRANCH+':'+REF)
            command('git','push','origin',REF+':refs/heads/'+BRANCH)
            for item in inventory():
                if done.get(item['source'],{}).get('sha256')==item['sha256']:continue
                transfer(item)
                done[item['source']]=dict(sha256=item['sha256'],external=BACKUP+item['relative'],verified_epoch=time.time())
                save_json(receipts,done)
            print(time.time(),'published and backed up',len(done),'artifacts',flush=True)
            # Inventory and successful backup happen after completion publication.
            if 'results/compiler_efficiency/completion.json' in command('git','ls-tree','-r','--name-only',REF).splitlines():return
        except Exception as exc:
            save_json(state/'last_error.json',dict(error=repr(exc),epoch=time.time()))
            print(time.time(),repr(exc),flush=True)
        time.sleep(300)


if __name__=='__main__':main()
