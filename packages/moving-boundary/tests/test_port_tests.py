"""M8: the research test port (tools/port_tests.py) on synthetic sources, and the ported files' record.

The port itself is historical (it reads the frozen research checkout by path); these tests need no
research checkout. ``test_ported_files_match_their_record`` keeps hand edits out of ported files.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
HEADER = "Ported for a test."


def _tool():
    spec = importlib.util.spec_from_file_location("port_tests_tool", PROJECT / "tools/port_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = '''"""Research docstring."""
import os
import numpy as np
import pytest

import kawin.diffusion.SurrogateArtifacts as artifact_module
from kawin.diffusion import MovingBoundaryIllingworthFD1DModel, prepare_surrogate_artifact
from kawin.diffusion.mesh import CartesianFD1D
from kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D import integrate_planar_transformed_molar_inventories as integrate
from kawin.diffusion.DiffusionParameters import TemperatureParameters
from kawin.tests.illingworth_baseline_cases import _binary_model


@pytest.mark.skipif(__import__("kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D", fromlist=["x"]) is None, reason="x")
def test_kept():
    from kawin.tests.illingworth_baseline_cases import _three_phase_model
    assert np and artifact_module and MovingBoundaryIllingworthFD1DModel and CartesianFD1D and integrate
    assert TemperatureParameters and _binary_model and _three_phase_model and prepare_surrogate_artifact


@pytest.mark.skipif(os.environ.get("X") != "1", reason="licence")
def test_deferred():
    from examples.ThermoCalc.tc_python_adapter import TCPythonThermodynamics
    assert TCPythonThermodynamics
'''


def test_port_rewrites_imports_strings_and_removes_deferred_tests():
    tool = _tool()
    text, transforms, helpers, counts = tool.port_text(SOURCE, "kawin/tests/test_x.py",
                                                       deferred={"test_deferred": "needs TC"}, header=HEADER)
    assert text.startswith('"""' + HEADER + "\n\nResearch docstring.")
    assert "import moving_boundary.surrogate_artifacts as artifact_module" in text
    assert "from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel" in text
    assert "from moving_boundary import prepare_surrogate_artifact" in text
    assert "from moving_boundary.mesh import CartesianFD1D" in text
    assert ("from moving_boundary.solvers.illingworth.numerics.ternary import "
            "integrate_planar_transformed_molar_inventories as integrate") in text
    assert "from kawin.diffusion.DiffusionParameters import TemperatureParameters" in text  # stock kawin, kept
    assert "from ported_helpers.illingworth_baseline_cases import _binary_model" in text
    assert "    from ported_helpers.illingworth_baseline_cases import _three_phase_model" in text
    assert '__import__("moving_boundary.solvers.illingworth.numerics.ternary", fromlist=["x"])' in text
    assert "test_deferred" not in text and "import os" not in text and "examples" not in text
    assert text.endswith("prepare_surrogate_artifact\n")
    assert helpers == ["illingworth_baseline_cases"]
    assert counts == dict(total=2, deferred=1, ported=1)
    kinds = [item["kind"] for item in transforms]
    assert kinds.count("deferred test") == 1 and kinds.count("string") == 1
    assert {"kind": "dropped import (only used by deferred tests)", "name": "os"} in transforms
    compile(text, "ported", "exec")


@pytest.mark.parametrize("source, deferred, message", [
    ("from examples.ThermoCalc import adapter\n", {}, "research example import"),
    ("import kawin.diffusion.SurrogateArtifacts\n", {}, "without an alias cannot be renamed"),
    ("from kawin.diffusion.MovingBoundaryIllingworthFDM import no_such_name\n", {}, "has no package counterpart"),
    ("def test_a():\n    pass\n", {"test_b": "x"}, "deferred tests not found"),
    ('X = "see kawin.diffusion.MovingBoundaryIllingworthFDM for details"\n', {}, "research module paths left"),
])
def test_unportable_sources_raise(source, deferred, message):
    tool = _tool()
    with pytest.raises(tool.PortError, match=message):
        tool.port_text(source, "kawin/tests/test_x.py", deferred=deferred, header=HEADER)


def test_ported_files_match_their_record():
    """Every ported file is byte-for-byte (newlines normalized) what the recorded port produced."""
    record = json.loads((PROJECT / "docs/test-port-provenance.json").read_text(encoding="utf-8"))
    assert record["files"]
    for target, entry in record["files"].items():
        text = (PROJECT / target).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == entry["output_sha256"], target
        if entry["counts"] is not None:
            counts = entry["counts"]
            deferred = [item for item in entry["transforms"] if item["kind"] == "deferred test"]
            assert counts["ported"] + counts["deferred"] == counts["total"] and len(deferred) == counts["deferred"], target
            assert not any(f"def {item['name']}(" in text for item in deferred), target
