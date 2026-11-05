"""Exact facade parity for the analytic example and minimal-import contracts."""

from pathlib import Path
import subprocess
import sys

import numpy as np

EXAMPLE = Path(__file__).parents[1] / "examples" / "binary_constant.py"


def test_binary_parity(tmp_path):
    """Compare separate legacy/bridge processes and the frozen binary fixture.

    This remains an alias checkpoint. Separate processes prevent shared module
    state from masking differences when the bridge becomes an extracted class.
    """
    worker = """
import importlib.util, sys
import numpy as np
from kawin.tests.illingworth_baseline_cases import _collect
spec = importlib.util.spec_from_file_location("binary_constant", sys.argv[1])
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
if sys.argv[2] == "legacy":
    from kawin.diffusion.MovingBoundaryIllingworthFDM import MovingBoundaryIllingworthFD1DModel as Model
else:
    from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel as Model
model = example.build_model(Model)
model.solve(0.5, minDtFrac=1e-14)
assert model.currentTime == 0.5
arrays = _collect("binary_phase_uniform", model,
                  ("interfaceData", "concData", "pData", "qData"))
np.savez(sys.argv[3], **arrays)
"""
    from kawin.tests.illingworth_baseline_cases import BASELINE_PATH
    assert BASELINE_PATH.is_file(), "Required binary baseline fixture is missing"
    for entry in ("legacy", "bridge"):
        subprocess.run([sys.executable, "-c", worker, str(EXAMPLE), entry,
                        str(tmp_path / f"{entry}.npz")],
                       check=True, timeout=60)
    with np.load(tmp_path / "legacy.npz") as legacy, \
            np.load(tmp_path / "bridge.npz") as bridge, \
            np.load(BASELINE_PATH) as baseline:
        keys = {key for key in baseline.files if key.startswith("binary_phase_uniform__")}
        assert keys == set(legacy.files) == set(bridge.files)
        for key in keys:
            np.testing.assert_array_equal(legacy[key], baseline[key])
            np.testing.assert_array_equal(bridge[key], baseline[key])


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
    assert '06dc7b4a' in str(exc)
else:
    raise AssertionError('missing fork accepted')
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
