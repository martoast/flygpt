"""Linux storage/measurement adapter; the frozen v3 learning implementation is reused."""
import resource
import sys
from types import SimpleNamespace
from scripts import compiler_v3_engine as original
from scripts.compiler_storage import archive


def train(spec_path, until):
    original.archive = archive
    if sys.platform.startswith('linux'):
        # Linux reports ru_maxrss in KiB; macOS reports bytes.
        original.resource = SimpleNamespace(RUSAGE_SELF=resource.RUSAGE_SELF,
            getrusage=lambda who: SimpleNamespace(ru_maxrss=resource.getrusage(who).ru_maxrss*1024))
    return original.train(spec_path, until)
