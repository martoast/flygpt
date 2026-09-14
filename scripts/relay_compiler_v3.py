"""Relay M4 compiler artifacts to GitHub without merging or touching training."""
import datetime
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSH = ('ssh -i /Users/alex/.ssh/flygpt_mac_mini_ed25519 '
       '-o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10')
REF = 'refs/remotes/macmini/experiment/compiler-v3'


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=120).strip()


def main():
    import fcntl
    directory = ROOT / 'results/compiler_v3_relay'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'relay.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Independent of the frozen experiment; safe to restart after laptop sleep.
    while True:
        try:
            command('git', '-c', 'core.sshCommand=' + SSH, 'fetch',
                    'ssh://alex@100.98.214.41/Users/alex/flygpt_compiler_v3',
                    'experiment/compiler-v3:' + REF)
            revision = command('git', 'rev-parse', REF)
            command('git', 'push', 'origin', REF + ':refs/heads/experiment/compiler-v3')
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(), revision, 'published', flush=True)
            files = command('git', 'ls-tree', '-r', '--name-only', REF).splitlines()
            if 'results/compiler_v3/completion.json' in files:
                return
        except (subprocess.SubprocessError, OSError) as exc:
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(), repr(exc), flush=True)
        time.sleep(300)


if __name__ == '__main__':
    main()
