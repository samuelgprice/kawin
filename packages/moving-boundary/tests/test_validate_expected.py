"""M6: normal runs validate expected artifacts and adopted inputs without any research checkout."""
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _tool(tmp_path, monkeypatch):
    """acceptance.py on a synthetic project: one case, one adopted input, one definitions file."""
    tool = _load("validate_expected_tool", PROJECT / "tools/acceptance.py")
    output = tmp_path / ".verification/acceptance"
    (output / "reference").mkdir(parents=True)
    (output / "inputs").mkdir()
    (output / "reference/C24.json").write_bytes(b"{}")
    (output / "reference/C24.npz").write_bytes(b"npz")
    (output / "inputs/open-surrogate.npz").write_bytes(b"surrogate")
    (tmp_path / "tools/definitions").mkdir(parents=True)
    (tmp_path / "tools/definitions/d.py").write_bytes(b"x = 1\n")
    (tmp_path / "docs").mkdir()
    provenance = dict(definitions={"d": dict(file="tools/definitions/d.py", sha256=_sha(b"x = 1\n"))})
    (tmp_path / "docs/definitions-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    manifest = dict(expected_artifacts=dict(cases={"C24": dict(json=_sha(b"{}"), npz=_sha(b"npz"))}),
                    adopted_inputs=dict(files={"open-surrogate.npz": dict(sha256=_sha(b"surrogate"), required_by=["C24"])}))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(tool, "PROJECT", tmp_path)
    monkeypatch.setattr(tool, "OUTPUT", output)
    monkeypatch.setattr(tool, "MANIFEST", tmp_path / "manifest.json")
    return tool, output


def test_valid_project_passes_and_unknown_case_fails(tmp_path, monkeypatch):
    tool, _ = _tool(tmp_path, monkeypatch)
    assert "expected_artifacts" in tool.validate_expected(["C24"])
    with pytest.raises(AssertionError, match="C01: no expected_artifacts entry"):
        tool.validate_expected(["C01"])


@pytest.mark.parametrize("relative, message", [
    ("reference/C24.npz", "C24: expected artifact changed: reference/C24.npz"),
    ("inputs/open-surrogate.npz", "adopted input changed: inputs/open-surrogate.npz")])
def test_changed_artifacts_fail(tmp_path, monkeypatch, relative, message):
    tool, output = _tool(tmp_path, monkeypatch)
    (output / relative).write_bytes(b"changed")
    with pytest.raises(AssertionError, match=message):
        tool.validate_expected(["C24"])


def test_missing_required_input_fails_only_its_cases(tmp_path, monkeypatch):
    tool, output = _tool(tmp_path, monkeypatch)
    (output / "inputs/open-surrogate.npz").unlink()
    tool.validate_expected([])  # no case needs it
    with pytest.raises(AssertionError, match=r"required input missing: inputs/open-surrogate.npz \(needed by C24"):
        tool.validate_expected(["C24"])


def test_changed_definitions_file_fails(tmp_path, monkeypatch):
    tool, _ = _tool(tmp_path, monkeypatch)
    (tmp_path / "tools/definitions/d.py").write_bytes(b"x = 2\n")
    with pytest.raises(AssertionError, match="definitions file changed: tools/definitions/d.py"):
        tool.validate_expected([])


def test_package_workers_get_no_research_root_and_references_are_never_overwritten(tmp_path, monkeypatch):
    tool, output = _tool(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(tool, "run", lambda args, **kwargs: calls.append(args) or "")
    def report(case):
        (output / "package").mkdir(exist_ok=True)
        (output / "package" / (case + ".json")).write_text(json.dumps(dict(status="PASS")))
    report("C01")
    tool.worker("C01", "package", state=dict(target="t", host="h"))
    assert "--root" not in calls[-1]
    with pytest.raises(AssertionError, match="C24 fixture is frozen; refusing overwrite"):
        tool.worker("C24", "reference", research=tmp_path)
    with pytest.raises(AssertionError, match="need --research"):
        tool.worker("C99", "reference")


def test_bindings_have_no_research_solver_fallback(monkeypatch):
    bindings = _load("bindings_no_fallback", PROJECT / "tools/definitions/_bindings.py")
    context = bindings.Context()
    monkeypatch.setitem(sys.modules, "moving_boundary.solvers", types.ModuleType("moving_boundary.solvers"))
    with pytest.raises(LookupError, match="package solver moving_boundary.solvers.MovingBoundaryIllingworthFD1DModel"):
        context.model("MovingBoundaryIllingworthFD1DModel")
    context.family = "binary"
    with pytest.raises(LookupError, match="no research fallback"):
        context.symbol("kawin.diffusion.mesh", "CartesianFD1D")([], [0.0, 1.0], 3)
    monkeypatch.setitem(bindings.MODULES, "kawin.diffusion.Missing", "moving_boundary.no_such_module_for_test")
    with pytest.raises(LookupError, match="package module moving_boundary.no_such_module_for_test"):
        context.imported_module("kawin.diffusion.Missing")
    with pytest.raises(LookupError, match="has no package counterpart"):
        context.symbol("kawin.diffusion.Missing", "Anything")
    assert context.imported_module("json") is json  # ordinary imports are unchanged
