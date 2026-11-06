"""Exact facade parity for the analytic example and minimal-import contracts."""

from pathlib import Path
import subprocess
import sys

import numpy as np

EXAMPLE = Path(__file__).parents[1] / "examples" / "binary_constant.py"


def test_binary_parity(tmp_path):
    """Compare separate legacy/bridge processes and the frozen binary fixture.

    Separate processes cover all three binary modes and both bulk upwind
    branches without sharing solver globals or changing the frozen fixtures.
    """
    worker = EXAMPLE.parents[1] / "tools/binary_cases.py"
    from kawin.tests.illingworth_baseline_cases import BASELINE_PATH
    assert BASELINE_PATH.is_file(), "Required binary baseline fixture is missing"
    for entry in ("legacy", "bridge"):
        subprocess.run([sys.executable, str(worker), entry,
                        str(tmp_path / f"{entry}.npz")],
                       check=True, timeout=60)
    with np.load(tmp_path / "legacy.npz") as legacy, \
            np.load(tmp_path / "bridge.npz") as bridge, \
            np.load(BASELINE_PATH) as baseline:
        keys = {key for key in baseline.files if key.startswith("binary_")}
        assert keys == {key for key in legacy.files if key.startswith("binary_")}
        assert set(legacy.files) == set(bridge.files)
        for key in legacy.files:
            np.testing.assert_array_equal(bridge[key], legacy[key], err_msg=key)
            if key in keys:
                np.testing.assert_array_equal(legacy[key], baseline[key], err_msg=key)


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
    assert '5f4ad574' in str(exc)
else:
    raise AssertionError('missing fork accepted')
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
