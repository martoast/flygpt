import pytest
from scripts.compiler_efficiency_distribution import owner


def test_complete_seed_blocks_partition_without_overlap():
    linux={s for s in range(500,510) if owner(s)=='omarchy'}
    mac={s for s in range(500,510) if owner(s)=='macbook'}
    assert linux==set(range(500,505))
    assert mac==set(range(505,510))
    assert not linux & mac and len(linux | mac)==10
    for bad in [499,510]:
        with pytest.raises(ValueError):owner(bad)
