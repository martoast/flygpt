from pathlib import Path
import subprocess
import pytest
from scripts import backup_topology_replication as backup
from scripts.run_topology_replication import summarize
from scripts.run_topology_confirmation_v2 import assign_test
from src.provenance import save_json,sha256


def test_last_unused_test_allocation():
    ids,available=assign_test({'all_previous':[{'id':i} for i in range(3968)]},128,'locked')
    assert available==128 and set(ids)==set(range(3968,4096))


def test_requires_ten_new_pairs():
    with pytest.raises(ValueError):summarize({})


def test_analysis_pairs_by_seed():
    rows={str(s):{'real_ce':{'accuracy':.8},'rewired_ce':{'accuracy':.5+(s-600)*.01}} for s in range(600,610)}
    result=summarize(rows)
    assert result['n']==10 and result['positive_pairs']==10
    assert result['mean_benefit']==pytest.approx(.255)


def checkpoint_fixture(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    job=backup.ROOT/'pairs/seed_600/real_ce';job.mkdir(parents=True)
    older=tmp_path/'old.pt';latest=tmp_path/'latest.pt'
    older.write_bytes(b'old');latest.write_bytes(b'latest')
    save_json(job/'progress.json',dict(step=128,archives=[dict(path=str(p),step=s,sha256=sha256(p)) for p,s in [(older,64),(latest,128)]]))
    return older,latest


def test_delete_only_verified_older_checkpoint(tmp_path,monkeypatch):
    older,latest=checkpoint_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(backup.mirror,'copy',lambda p,h:'/verified/'+p.name)
    done={};backup.sweep(done)
    assert not older.exists() and latest.exists() and len(done)==2
    backup.sweep(done)
    assert latest.exists()


def test_failed_copy_keeps_all_local_checkpoints(tmp_path,monkeypatch):
    older,latest=checkpoint_fixture(tmp_path,monkeypatch)
    def fail(p,h):raise subprocess.CalledProcessError(1,'ssh')
    monkeypatch.setattr(backup.mirror,'copy',fail)
    with pytest.raises(subprocess.CalledProcessError):backup.sweep({})
    assert older.exists() and latest.exists()
