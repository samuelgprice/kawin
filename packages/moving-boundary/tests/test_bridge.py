"""Exact facade parity for the analytic example and minimal-import contracts."""

from pathlib import Path
import subprocess
import sys

import numpy as np

EXAMPLE = Path(__file__).parents[1] / "examples" / "binary_constant.py"


def test_binary_example():
    """The shipped example runs using local mesh additions and stock host services.

    Full legacy fixture and bulk comparisons belong to acceptance.py LEGACY,
    where the oracle is detached and each package worker is separately installed.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location('binary_example', EXAMPLE)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    model = example.build_model()
    model.solve(.5, minDtFrac=1e-14)
    assert model.currentTime == .5
    assert model.__class__.__module__ == 'moving_boundary.solvers.binary'


def test_namespace_import_is_lazy():
    subprocess.run([sys.executable, "-c",
                    "import sys, moving_boundary, moving_boundary.solvers; "
                    "assert 'kawin' not in sys.modules"], check=True, timeout=30)


def test_missing_fork_error():
    code = """
import sys
class BlockKawin:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'kawin':
            raise ModuleNotFoundError('blocked for contract test', name='kawin')
sys.meta_path.insert(0, BlockKawin())
try:
    from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel
except ImportError as exc:
    assert 'public kawin 0.5.0' in str(exc)
else:
    raise AssertionError('missing fork accepted')
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
