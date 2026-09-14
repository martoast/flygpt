"""Back up M1 checkpoints, reclaiming only verified older local versions."""
import fcntl
import json
from pathlib import Path
import time
from scripts import mirror_m1_checkpoints as mirror
from src.provenance import save_json,sha256

ROOT=Path('results/topology_replication')
STATE=Path('results/topology_replication_backup')
mirror.SSH[-1]='alex@alexs-mac-mini'


def sweep(done):
    for p in sorted(ROOT.glob('pairs/seed_*/*/progress.json')):
        progress=json.loads(p.read_text())
        for archive in progress['archives']:
            source=Path(archive['path']); key=str(source)
            if key not in done:
                assert sha256(source)==archive['sha256']
                relative=source.relative_to(Path.cwd()) if source.is_absolute() else source
                target=mirror.copy(relative,archive['sha256'])
                done[key]=dict(sha256=archive['sha256'],external=target,verified_epoch=time.time())
                save_json(STATE/'verified.json',done)
            assert done[key]['sha256']==archive['sha256']
            # The newer progress/checkpoint was already atomically published.
            # Never remove the latest version or the final model's target.
            if archive['step']<progress['step'] and source.exists():source.unlink()


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    lock=(STATE/'backup.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=STATE/'verified.json';done=json.loads(p.read_text()) if p.exists() else {}
    while True:
        try:
            sweep(done)
            if (ROOT/'completion.json').exists():break
        except Exception as exc:
            save_json(STATE/'last_error.json',dict(error=repr(exc),epoch=time.time()))
        time.sleep(30)


if __name__=='__main__':main()
