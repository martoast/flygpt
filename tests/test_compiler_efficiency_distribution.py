import pytest
import json
from pathlib import Path
from scripts import compiler_efficiency_distribution as d
from scripts.compiler_efficiency_distribution import owner
from src.provenance import sha256


def test_complete_seed_blocks_partition_without_overlap():
    linux={s for s in range(500,510) if owner(s)=='omarchy'}
    mac={s for s in range(500,510) if owner(s)=='macbook'}
    assert linux==set(range(500,505))
    assert mac==set(range(505,510))
    assert not linux & mac and len(linux | mac)==10
    for bad in [499,510]:
        with pytest.raises(ValueError):owner(bad)


@pytest.mark.parametrize('corrupt',[False,True])
def test_coordinator_import_gate_before_training(tmp_path,monkeypatch,corrupt):
    """Remote assignments must verify receipt and archives, never train afresh."""
    from scripts import run_compiler_efficiency as r
    monkeypatch.chdir(tmp_path)
    root=Path('results/compiler_efficiency');job=root/'seed_505/A_hard';job.mkdir(parents=True)
    Path(d.DOC).write_text('frozen amendment')
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'optimizer and RNG fixture')
    digest=sha256(checkpoint)
    spec=job/'spec.json';spec.write_text(json.dumps(dict(seed=505,job_dir=str(job))))
    progress=job/'progress.json';progress.write_text(json.dumps(dict(step=256,checkpoint_sha256=digest,
        archives=[dict(step=256,path=str(checkpoint),sha256=digest)])))
    (job/'macbook_complete.json').write_text(json.dumps(dict(amendment_sha256=sha256(d.DOC),
        progress_sha256='bad' if corrupt else sha256(progress),checkpoint_sha256=digest)))
    calls=[]
    def original_train(path,engine):
        assert (job/'model.pt').is_symlink()
        calls.append(path)
        return str(checkpoint)
    monkeypatch.setattr(r.v,'train',original_train)
    # Register originals with monkeypatch so controller's wrappers are restored.
    for obj,name in [(r,'publish'),(r,'report'),(r.c,'freeze')]:monkeypatch.setattr(obj,name,getattr(obj,name))
    monkeypatch.setattr(r,'run',lambda:r.v.train(str(spec),'unused'))
    if corrupt:
        with pytest.raises(AssertionError):d.controller(0)
        assert calls==[] and not (job/'model.pt').exists()
    else:
        d.controller(0)
        assert calls==[str(spec)]
