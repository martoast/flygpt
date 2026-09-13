"""Preserve G1's fixed 1024-update target and final test on the Seagate disk."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import torch
from src.provenance import save_json, sha256


def main():
    root = Path('results/malecns_v1/target')
    ckpath = root/'snapshots/real_0_step_1024.pt'
    result = json.loads((root/'real_0.json').read_text())
    assert json.loads((root/'conclusion.json').read_text())['final_step']==1024
    assert sha256(ckpath)==result['checkpoint_sha256']
    ck = torch.load(ckpath,weights_only=True,map_location='cpu')
    assert ck['completed_steps']==1024 and ck['seed']==0
    assert {int(s['step']) for s in ck['optimizer']['state'].values()}=={1024}
    generator = torch.Generator().manual_seed(20000)
    length = len(Path('data/raw/grammar_v1/train.txt').read_bytes())
    for _ in range(1024): torch.randint(0,length-32,(1,),generator=generator)
    assert torch.equal(generator.get_state(),ck['data_rng'])
    commit = ck['code_commit']; del ck
    files = [ckpath,root/'real_0.json',root/'trajectory/step_1024.json',root/'final_test.json',root/'conclusion.json',root/'extension_decisions.json',
             Path('data/processed/malecns.npz'),Path('data/processed/malecns_body_ids.npy'),Path('results/malecns_v1/teacher.pt'),
             Path('results/fly_real.pt'),Path('results/malecns_v1/graph_provenance.json'),Path('results/malecns_v1/download.json'),
             Path('data/raw/grammar_v1/train.txt'),Path('data/raw/grammar_v1/validation.txt'),Path('data/raw/grammar_v1/test.txt'),
             Path('requirements-lock.txt'),Path('scripts/train_target.py')]
    directory = Path('artifacts/g1-target-1024'); directory.mkdir(parents=True,exist_ok=True)
    code = directory/'training_commit.tar'
    subprocess.run(['git','archive','--format=tar','--output',str(code),commit],check=True)
    metadata = {'experiment':'G1 — Finite-Grammar Functional Encoding','target_step':1024,
                'training_commit':commit,'checkpoint_sha256':sha256(ckpath),
                'optimizer_steps_verified':1024,'sampling_state_reproduced':True,
                'stopping':'Pre-existing near-teacher threshold crossed; no 2048 continuation',
                'files':{str(p):sha256(p) for p in files},'code_archive_sha256':sha256(code),
                'source_caveat':'Historical dirty-worktree flag retained; recorded per-file hashes supplement training commit.'}
    save_json(directory/'manifest.json',metadata)
    archive = directory/'FlyGPT-G1-target-seed0-step1024.tar'
    if not archive.exists():
        with tarfile.open(archive,'w') as bundle:
            for p in files: bundle.add(p,arcname=str(p))
            bundle.add(code,arcname='training_commit.tar'); bundle.add(directory/'manifest.json',arcname='manifest.json')
    disk = Path('/Volumes/Seagate')
    assert disk.is_mount() and disk.stat().st_dev!=archive.stat().st_dev
    destination = disk/'FlyGPT Backups/G1-Finite-Grammar-Functional-Encoding/seed0-step1024'
    destination.mkdir(parents=True,exist_ok=True); target = destination/archive.name
    digest = sha256(archive)
    if not target.exists():
        partial = target.with_suffix('.tar.partial')
        with archive.open('rb') as src, partial.open('xb') as dst:
            shutil.copyfileobj(src,dst,8*1024*1024); dst.flush(); os.fsync(dst.fileno())
        assert sha256(partial)==digest; partial.rename(target)
    assert sha256(target)==digest
    with tarfile.open(target) as bundle:
        stored = json.load(bundle.extractfile('manifest.json'))
        assert stored==metadata
        for name, expected in {**metadata['files'],'training_commit.tar':metadata['code_archive_sha256']}.items():
            with bundle.extractfile(name) as stream:
                assert __import__('hashlib').file_digest(stream,'sha256').hexdigest()==expected
    receipt = {**metadata,'destination':str(target),'archive_sha256':digest,'bytes':target.stat().st_size,
               'verified_external_members':len(files)+1,'storage':'External Seagate USB; separate physical disk'}
    save_json(destination/'verification.json',receipt)
    save_json(root/'external_backup_1024.json',receipt)
    (destination/'SHA256SUMS').write_text(digest+'  '+target.name+'\n')
    print(json.dumps({k:receipt[k] for k in ('destination','archive_sha256','bytes','verified_external_members')},indent=2))


if __name__=='__main__':main()
