"""Relay M4 compiler artifacts to GitHub without merging or touching training."""
import datetime
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSH = ('ssh -i /Users/alex/.ssh/flygpt_mac_mini_ed25519 '
       '-o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10')


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=120).strip()


def main():
    import argparse
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    study = 'compiler_confirmation' if args.confirmation else 'compiler_v3'
    branch = 'experiment/compiler-confirmation' if args.confirmation else 'experiment/compiler-v3'
    ref = 'refs/remotes/macmini/' + branch
    directory = ROOT / ('results/' + study + '_relay')
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'relay.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Independent of the frozen experiment; safe to restart after laptop sleep.
    while True:
        try:
            command('git', '-c', 'core.sshCommand=' + SSH, 'fetch',
                    'ssh://alex@alexs-mac-mini/Users/alex/flygpt_' + study,
                    branch + ':' + ref)
            revision = command('git', 'rev-parse', ref)
            command('git', 'push', 'origin', ref + ':refs/heads/' + branch)
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(), revision, 'published', flush=True)
            files = command('git', 'ls-tree', '-r', '--name-only', ref).splitlines()
            if 'results/' + study + '/completion.json' in files:
                return
        except (subprocess.SubprocessError, OSError) as exc:
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(), repr(exc), flush=True)
        time.sleep(300)


if __name__ == '__main__':
    main()
