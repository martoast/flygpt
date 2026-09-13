"""Small, explicit manifests for computational (never wetware) experiments."""
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def manifest(config, inputs=()):
    import torch
    return {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'code_commit': git_output('rev-parse', 'HEAD'),
        'worktree_dirty': bool(git_output('status', '--porcelain', '--untracked-files=no')),
        'config': config,
        'inputs': {str(p): sha256(p) for p in inputs},
        'hardware': {'platform': platform.platform(), 'machine': platform.machine(),
                     'torch': torch.__version__, 'mps': torch.backends.mps.is_available(),
                     'cuda': torch.cuda.is_available(), 'torch_threads': torch.get_num_threads()},
        'evidence_domain': 'computational; see graph provenance for synthetic versus MaleCNS',
    }


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
