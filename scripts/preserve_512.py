"""Create a verified independent local copy of the 512-update milestone."""
import json
import shutil
import subprocess
from pathlib import Path
import torch
from src.provenance import sha256, save_json


def main():
    root = Path('results/malecns_v1')
    archive = Path('artifacts/malecns_seed0_step0512')
    archive.mkdir(parents=True, exist_ok=True)
    snapshot = root/'target/snapshots/real_0_step_0512.pt'
    metrics = json.loads((root/'target/trajectory/step_0512.json').read_text())
    assert sha256(snapshot) == metrics['inputs'][str(snapshot)]
    ck = torch.load(snapshot, map_location='cpu', weights_only=True)
    assert ck['completed_steps'] == 512 and ck['seed'] == 0
    assert len(ck['trace']) == 512
    optimizer_steps = sorted({int(s['step']) for s in ck['optimizer']['state'].values()})
    assert optimizer_steps == [512]
    train = Path('data/raw/grammar_v1/train.txt').read_bytes()
    validation = Path('data/raw/grammar_v1/validation.txt').read_bytes()
    generator = torch.Generator().manual_seed(20000)
    sampled_starts = [int(torch.randint(0, len(train)-32, (1,), generator=generator)) for _ in range(512)]
    assert torch.equal(generator.get_state(), ck['data_rng'])
    starts = metrics['config']['evaluation_starts']
    save_json(archive/'sampling.json', {'seed': 20000, 'training_window_starts': sampled_starts,
              'validation_windows': [{'start': s, 'input_bytes': list(validation[s:s+32]),
                                      'target_bytes': list(validation[s+1:s+33])} for s in starts]})
    files = [snapshot, root/'teacher.pt', Path('data/processed/malecns.npz'),
             Path('data/processed/malecns_body_ids.npy'), Path('results/fly_real.pt'),
             root/'target/trajectory/step_0512.json', root/'target/real_0.json',
             root/'graph_provenance.json', root/'download.json',
             Path('data/raw/grammar_v1/train.txt'), Path('data/raw/grammar_v1/validation.txt'),
             Path('requirements-lock.txt'), Path('scripts/train_target.py')]
    entries = []
    for source in files:
        expected = metrics['inputs'].get(str(source))
        digest = sha256(source)
        if expected is not None:
            assert digest == expected, source
        target = archive/source
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        assert sha256(target) == digest, target
        assert source.stat().st_ino != target.stat().st_ino
        entries.append({'source': str(source), 'copy': str(target), 'sha256': digest, 'bytes': target.stat().st_size})
    code = archive/'training_commit.tar'
    if not code.exists():
        subprocess.run(['git', 'archive', '--format=tar', '--output', str(code), ck['code_commit']], check=True)
    record = {'milestone': 'MaleCNS seed 0, 512 additional updates',
              'archive': str(archive.resolve()), 'checkpoint_sha256': sha256(snapshot),
              'training_code_commit': ck['code_commit'], 'optimizer_steps': optimizer_steps,
              'sampling_state_reproduced': True, 'files': entries,
              'code_archive_sha256': sha256(code), 'sampling_sha256': sha256(archive/'sampling.json'),
              'limitations': 'Independent copies on the same disk, not an off-machine backup. Original run reported a dirty worktree; commit archive alone is not proof of the entire runtime source state. Recorded source hashes remain authoritative.'}
    save_json(archive/'manifest.json', record)
    save_json(root/'target/milestone_0512_archive.json', record)
    print(json.dumps({'archive': str(archive), 'files_verified': len(entries),
                      'optimizer_steps': optimizer_steps, 'sampling_state_reproduced': True}))


if __name__ == '__main__':
    main()
