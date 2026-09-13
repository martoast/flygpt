"""Copy the frozen G1 bundle to the user's mounted Seagate and verify it."""
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from src.provenance import save_json, sha256


def main():
    volume = Path('/Volumes/Seagate')
    source = Path('artifacts/FlyGPT-G1-seed0-step0512.tar')
    if not volume.is_mount() or volume.stat().st_dev == source.stat().st_dev:
        raise RuntimeError('Seagate must be mounted on a separate filesystem')
    expected = 'e680821b3cc82f2798f84320e253470113251b667e948b36fae83df446b06735'
    assert sha256(source) == expected, 'Local bundle changed'
    directory = volume/'FlyGPT Backups/G1-Finite-Grammar-Functional-Encoding/seed0-step0512'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/source.name
    if not target.exists():
        temporary = target.with_suffix('.tar.partial')
        with source.open('rb') as src, temporary.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=8*1024*1024)
            dst.flush()
            os.fsync(dst.fileno())
        if sha256(temporary) != expected:
            raise RuntimeError('USB copy checksum mismatch; partial retained for inspection')
        temporary.rename(target)
    assert sha256(target) == expected, 'Existing backup differs; refusing to overwrite'
    # Read each scientific artifact from the external tar, not the source disk.
    prefix = 'malecns_seed0_step0512/'
    with tarfile.open(target) as bundle:
        manifest = json.load(bundle.extractfile(prefix+'manifest.json'))
        members = {prefix+entry['source']: entry['sha256'] for entry in manifest['files']}
        members[prefix+'training_commit.tar'] = manifest['code_archive_sha256']
        members[prefix+'sampling.json'] = manifest['sampling_sha256']
        for name, digest in members.items():
            with bundle.extractfile(name) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
    receipt = {'experiment': 'G1 — Finite-Grammar Functional Encoding',
               'checkpoint_step': 512, 'destination': str(target),
               'archive_sha256': expected, 'bytes': target.stat().st_size,
               'verified_embedded_artifacts': len(members),
               'checkpoint_sha256': manifest['checkpoint_sha256'],
               'storage': 'User-selected external Seagate USB disk; separate physical device, not cloud/off-site storage',
               'verified_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    save_json(directory/'verification.json', receipt)
    save_json('results/malecns_v1/target/external_backup_0512.json', receipt)
    (directory/'SHA256SUMS').write_text(expected+'  '+target.name+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
