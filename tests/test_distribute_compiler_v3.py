import pytest

from scripts import distribute_compiler_v3 as d
from src.provenance import save_json, sha256


def imported_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    method = 'C_hidden'
    job = d.ROOT / 'seed_300' / method
    job.mkdir(parents=True)
    checkpoint = tmp_path / 'external.pt'
    checkpoint.write_bytes(b'checkpoint')
    archive = dict(path=str(checkpoint), sha256=sha256(checkpoint))
    save_json(job / 'spec.json', {'method': method})
    save_json(job / 'progress.json', dict(step=512, checkpoint_sha256=archive['sha256']))
    save_json(d.DIST / (method + '_imports.json'), {'512': archive})
    receipt = dict(method=method, step=512, checkpoint=archive,
                   spec_sha256=sha256(job / 'spec.json'),
                   inputs=d.runner.paths_hashes([job / 'progress.json', d.DIST / (method + '_imports.json')]))
    save_json(d.DIST / (method + '_complete.json'), receipt)
    return method, checkpoint, job


def test_import_accepts_verified_checkpoint(tmp_path, monkeypatch):
    method, checkpoint, _ = imported_job(tmp_path, monkeypatch)
    assert d.imported_checkpoint(method) == str(checkpoint)


def test_import_rejects_corrupted_checkpoint(tmp_path, monkeypatch):
    method, checkpoint, _ = imported_job(tmp_path, monkeypatch)
    checkpoint.write_bytes(b'corruption')
    with pytest.raises(AssertionError):
        d.imported_checkpoint(method)


def test_import_rejects_changed_progress(tmp_path, monkeypatch):
    method, _, job = imported_job(tmp_path, monkeypatch)
    save_json(job / 'progress.json', {'step': 256})
    with pytest.raises(RuntimeError, match='Frozen input changed'):
        d.imported_checkpoint(method)
