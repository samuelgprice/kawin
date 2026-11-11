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


HELPER_SOURCE = '''import os
import numpy as np
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder

SCALE = 2.0


class Base:
    def value(self):
        return np.float64(SCALE)


class Wanted(Base):
    pass


def test_not_copied():
    assert os


def unrelated():
    return ProfileBuilder
'''


def test_subset_helpers_copy_only_what_the_selected_names_need():
    tool = _tool()
    text = tool.extract_subset(HELPER_SOURCE, ["Wanted"], "kawin/tests/test_helper.py")
    assert "class Wanted(Base)" in text and "class Base" in text and "SCALE = 2.0" in text
    assert "import numpy as np" in text and "test_not_copied" not in text and "unrelated" not in text
    assert "import os" not in text and "CartesianFD1D" not in text
    assert "# --- kawin/tests/test_helper.py:13-14" in text and "# --- kawin/tests/test_helper.py:8-10" in text
    compile(text, "subset", "exec")
    with pytest.raises(tool.PortError, match="not defined at top level"):
        tool.extract_subset(HELPER_SOURCE, ["Missing"], "kawin/tests/test_helper.py")
    with pytest.raises(tool.PortError, match="star import"):
        tool.extract_subset("from data import *\n\ndef f():\n    return TDB\n", ["f"], "kawin/tests/test_helper.py")


def test_patches_follow_the_package_reimport_chain(monkeypatch):
    """A research module global now re-imported from another package module is patched on both."""
    import moving_boundary.solvers.illingworth.shared as shared
    import moving_boundary.solvers.illingworth.ternary as ternary
    from ported_helpers.relocation import owner_chain, setattr_owners
    assert owner_chain(ternary, "_COMPONENTWISE_ROUNDOFF_FACTOR") == [
        (ternary, "_COMPONENTWISE_ROUNDOFF_FACTOR"), (shared, "_COMPONENTWISE_ROUNDOFF_FACTOR")]
    assert owner_chain(ternary, "MovingBoundaryIllingworthTernaryFD1DModel") == [(ternary, "MovingBoundaryIllingworthTernaryFD1DModel")]
    marker = object()
    with monkeypatch.context() as patch:
        setattr_owners(patch, ternary, "_COMPONENTWISE_ROUNDOFF_FACTOR", marker)
        assert ternary._COMPONENTWISE_ROUNDOFF_FACTOR is marker and shared._COMPONENTWISE_ROUNDOFF_FACTOR is marker
    assert shared._COMPONENTWISE_ROUNDOFF_FACTOR == 64.0 and ternary._COMPONENTWISE_ROUNDOFF_FACTOR == 64.0
    tool = _tool()
    source = ("import kawin.diffusion.MovingBoundaryIllingworthTernaryFDM as fdm\n\n\ndef test_a(monkeypatch, name):\n"
              "    monkeypatch.setattr(fdm, \"_COMPONENTWISE_ROUNDOFF_FACTOR\", 1e-300)\n"
              "    monkeypatch.setattr(fdm, \"MovingBoundaryIllingworthTernaryFD1DModel\", None)\n"
              "    monkeypatch.setattr(fdm, \"_validate_ternary_diffusivity_matrix\", None)\n"
              "    monkeypatch.setattr(fdm, name, None, raising=False)\n")
    # Research defines the first two names itself; it only imported _validate_ternary_diffusivity_matrix here.
    defines = {"_COMPONENTWISE_ROUNDOFF_FACTOR", "MovingBoundaryIllingworthTernaryFD1DModel"}
    text, transforms, _, _ = tool.port_text(source, "kawin/tests/test_x.py", header=HEADER,
                                            research_defines=lambda module: defines)
    assert "from ported_helpers.relocation import setattr_owners" in text
    assert 'setattr_owners(monkeypatch, fdm, "_COMPONENTWISE_ROUNDOFF_FACTOR", 1e-300)' in text
    assert 'monkeypatch.setattr(fdm, "MovingBoundaryIllingworthTernaryFD1DModel", None)' in text  # nothing moved
    assert 'monkeypatch.setattr(fdm, "_validate_ternary_diffusivity_matrix", None)' in text  # research only imported it
    assert ("setattr_owners(monkeypatch, fdm, name, None, raising=False, "
            "research_defines=_RESEARCH_DEFINES_MovingBoundaryIllingworthTernaryFDM)") in text
    assert ("_RESEARCH_DEFINES_MovingBoundaryIllingworthTernaryFDM = frozenset({'MovingBoundaryIllingworthTernaryFD1DModel', "
            "'_COMPONENTWISE_ROUNDOFF_FACTOR'})") in text
    assert sum(item["kind"] == "monkeypatch through the package re-import chain" for item in transforms) == 2
    with pytest.raises(tool.PortError, match="needs research_defines"):
        tool.port_text(source, "kawin/tests/test_x.py", header=HEADER)
    with monkeypatch.context() as patch:  # a name research did not define is patched on the module only
        setattr_owners(patch, ternary, "_COMPONENTWISE_ROUNDOFF_FACTOR", marker, research_defines=frozenset())
        assert ternary._COMPONENTWISE_ROUNDOFF_FACTOR is marker and shared._COMPONENTWISE_ROUNDOFF_FACTOR == 64.0


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


MIGRATION_SOURCE = '''def build(R, options):
    params = {"tolerance": 1.0e-8}
    return dict(tolerance=1e-10, record=True, **options)
'''
MIGRATION_EDITS = [
    dict(function="build", op="dict_value", key="tolerance", old="1.0e-8", new="2.0e-9"),
    dict(function="build", op="keyword", call="dict", arg="tolerance", old="1e-10", new="1e-10 / R"),
    dict(function="build", op="insert_keyword", call="dict", after="tolerance", arg="bulk_picard_rtol", new="1e-10"),
    dict(function="build", op="kwargs_default", call="dict", arg="extra", new="1"),
]


def test_api_migration_edits_are_applied_and_recorded(monkeypatch):
    """Recorded ``api_migration`` edits rewrite the listed call sites only, and a stale ``old`` value is an error."""
    tool = _tool()
    monkeypatch.setattr(tool.API, "EDITS", {"kawin/tests/test_x.py": MIGRATION_EDITS})
    text, transforms, _, _ = tool.port_text(MIGRATION_SOURCE, "kawin/tests/test_x.py", header=HEADER)
    assert 'params = {"tolerance": 2.0e-9}' in text
    assert 'dict(tolerance=1e-10 / R, bulk_picard_rtol=1e-10, record=True, **{"extra": 1, **options})' in text
    migrations = [item for item in transforms if item["kind"] == "api_migration"]
    assert [item["op"] for item in migrations] == [edit["op"] for edit in MIGRATION_EDITS]
    assert all(item["reason"] == tool.API.REASON for item in migrations)
    namespace = {}
    exec(compile(text, "test_x", "exec"), namespace)
    assert namespace["build"](5.0, {"extra": 2}) == dict(tolerance=1e-10 / 5.0, bulk_picard_rtol=1e-10, record=True, extra=2)

    stale = [dict(MIGRATION_EDITS[1], old="1e-9")]
    monkeypatch.setattr(tool.API, "EDITS", {"kawin/tests/test_x.py": stale})
    with pytest.raises(tool.API.MigrationError, match="matched 0 places"):
        tool.port_text(MIGRATION_SOURCE, "kawin/tests/test_x.py", header=HEADER)
