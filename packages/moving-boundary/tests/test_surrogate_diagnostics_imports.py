"""Import, optional-dependency and debugger contracts for ``moving_boundary.surrogate_diagnostics``.

Hand port (FUTURE_PORTS entry 1) of the diagnostics check in research
``kawin/tests/test_no_debugger_dependencies.py`` at ``7af24f4``:
``test_diffusion_import_without_research_or_debugger_modules`` and
``test_production_has_no_debugger_references``. The two solver tests in that file are not part of
this port (see FUTURE_PORTS.md). Deviations, each because the package differs from research:

- The guarded subprocess imports ``moving_boundary.surrogate_diagnostics`` instead of
  ``kawin.diffusion``. It receives the parent's resolved ``sys.path`` (the package and its host may
  come from ``PYTHONPATH``, which ``-I`` ignores) and asserts that it loads the same package.
- It also rejects ``pdb`` and ``plotly``, runs the saved-data evaluators on a synthetic surrogate,
  and requires every plotting entry point to fail with the actionable missing-extra error.
- The source scan covers ``src/moving_boundary`` and also rejects ``pdb``/``ipdb``/``debugpy``
  imports and ``breakpoint()`` calls.

Also holds the package replacements for two research tests of ``test_surrogate_diagnostics.py``
that ``tools/port_tests.py`` records as superseded.
"""
import ast
import builtins
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "src" / "moving_boundary"

SYNTHETIC = r'''
import numpy as np
from moving_boundary.surrogates import TernaryMovingBoundaryThermodynamicsSurrogate

PHASES = ("ALPHA", "BETA")
ELEMENTS = ("Z", "X", "Y")
ETA = np.asarray([0.0, 0.5, 1.0])
LEFT = np.asarray([[0.20, 0.10], [0.25, 0.10], [0.30, 0.10]])
RIGHT = np.asarray([[0.40, 0.20], [0.45, 0.20], [0.50, 0.20]])
GENERAL = np.asarray([[0.20, 0.10], [0.30, 0.10], [0.20, 0.20], [0.30, 0.20]])
MATRIX = np.asarray([[2.0, 0.2], [0.1, 1.0]])


def matrices(count, scale=1.0):
    return np.broadcast_to(scale * MATRIX, (count, 2, 2)).copy()


def surrogate():
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=ELEMENTS, phases=PHASES, tieline_phases=PHASES, temperature=1000.0,
        eta_samples=ETA, tieline_compositions={PHASES[0]: LEFT, PHASES[1]: RIGHT},
        diffusivity_compositions={"interface": {PHASES[0]: LEFT, PHASES[1]: RIGHT},
                                  "general": {PHASES[0]: GENERAL, PHASES[1]: GENERAL}},
        diffusivities={"interface": {PHASES[0]: matrices(3), PHASES[1]: matrices(3, 0.5)},
                       "general": {PHASES[0]: matrices(4), PHASES[1]: matrices(4, 0.5)}},
        diffusivity_interpolation="nearest",
        metadata={"source": "from_database", "precipitate_phase": PHASES[1],
                  "probe_start": [0.20, 0.15], "probe_end": [0.30, 0.15]},
        validity_policy="raise")


def kinetics_records(model):
    records = []
    for phase in PHASES:
        for context in ("interface", "general"):
            for point, matrix in zip(model.diffusivity_compositions[context][phase],
                                     model.diffusivities[context][phase]):
                records.append({
                    "record_type": "kinetics", "query_index": len(records), "requested_phase": phase,
                    "input_composition": point.tolist(),
                    "phase_composition": [1.0 - float(np.sum(point)), *point.tolist()],
                    "interdiffusivity": matrix.tolist(),
                    "site_fractions": [{"sublattice": 1,
                                        "constituents": {"X": float(point[0]), "Y": float(point[1])}}],
                })
    return records


SITE_FRACTIONS = {
    "metadata": {"element_order": list(ELEMENTS)},
    "records": [{
        "status": "ok", "kind": "equilibrium", "calculation_index": 0, "requested_phase": "ALPHA",
        "input_full_composition": [0.7, 0.2, 0.1], "stable_composition_sets": ["ALPHA"],
        "phases": [{"phase": "ALPHA", "stable": True, "phase_amount": 1.0,
                    "phase_composition": [0.7, 0.2, 0.1], "diagnostic_errors": {},
                    "site_fractions": [{"sublattice": 1, "site_ratio": 1.0,
                                        "constituents": {"Z": 0.7, "X": 0.2, "Y": 0.1}}]}],
    }],
}
'''

GUARDED = r'''
import json, sys
sys.path[:] = json.loads(sys.argv[1])
expected_package = sys.argv[2]
BLOCKED = {"examples", "debugpy", "pdb", "plotly"}


class NoResearchDebuggerOrPlotly:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("blocked for contract test: " + fullname, name=fullname)


sys.meta_path.insert(0, NoResearchDebuggerOrPlotly())
from pathlib import Path
import moving_boundary
import moving_boundary.surrogate_diagnostics as diagnostics
assert str(Path(moving_boundary.__file__).resolve().parent) == expected_package, moving_boundary.__file__
''' + SYNTHETIC + r'''
model = surrogate()
tieline = diagnostics.evaluate_tieline_diagnostics(model, eta_count=3)
diffusivity = diagnostics.evaluate_diffusivity_diagnostics(model, interface_eta_count=3, bulk_grid_counts=(3, 3))
loo = diagnostics.evaluate_diffusivity_leave_one_out(model, phases="ALPHA")
consistency = diagnostics.evaluate_interface_bulk_diffusivity_consistency(model, neighbor_count=2)
construction = diagnostics.evaluate_surrogate_construction_diagnostics(model)
assert tieline and diffusivity and loo and consistency and construction is not None

calls = {
    "plot_tieline_diagnostics": lambda: diagnostics.plot_tieline_diagnostics(tieline),
    "plot_interface_diffusivity_diagnostics": lambda: diagnostics.plot_interface_diffusivity_diagnostics(diffusivity),
    "plot_bulk_diffusivity_diagnostics": lambda: diagnostics.plot_bulk_diffusivity_diagnostics(diffusivity, "ALPHA"),
    "plot_diffusivity_leave_one_out": lambda: diagnostics.plot_diffusivity_leave_one_out(loo, "ALPHA"),
    "plot_surrogate_construction_diagnostics": lambda: diagnostics.plot_surrogate_construction_diagnostics(construction),
    "plot_selected_diffusivity_calculations":
        lambda: diagnostics.plot_selected_diffusivity_calculations(model, kinetics_records(model)),
    "plot_calculation_site_fractions": lambda: diagnostics.plot_calculation_site_fractions(SITE_FRACTIONS, "ALPHA"),
    "plot_surrogate_diagnostics": lambda: diagnostics.plot_surrogate_diagnostics(
        model, eta_count=3, interface_eta_count=3, bulk_grid_counts=(3, 3)),
}
assert sorted(calls) == sorted(name for name in diagnostics.__all__ if name.startswith("plot_")), sorted(calls)
for name, call in calls.items():
    try:
        call()
    except ImportError as exc:
        assert "moving-boundary[diagnostics]" in str(exc), (name, str(exc))
    else:
        raise AssertionError(name + " did not report the missing Plotly extra")
leaked = sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED)
assert not leaked, leaked
print("guarded diagnostics ok")
'''


def test_diagnostics_import_without_research_debugger_or_plotly(tmp_path):
    """Saved-data diagnostics run with research, debugger and Plotly imports all rejected."""
    import moving_boundary
    package = str(Path(moving_boundary.__file__).resolve().parent)
    path = [entry for entry in sys.path if entry and Path(entry).resolve() != Path.cwd().resolve()]
    result = subprocess.run([sys.executable, "-I", "-S", "-c", GUARDED, json.dumps(path), package],
                            cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "guarded diagnostics ok" in result.stdout


def test_production_has_no_debugger_references():
    """No package source module refers to debugInPlace, imports a debugger or calls breakpoint()."""
    debuggers = {"pdb", "ipdb", "debugpy"}
    sources = sorted(SOURCE.rglob("*.py"))
    assert SOURCE / "surrogate_diagnostics.py" in sources
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id != "debugInPlace", path
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "debugInPlace" not in module and module.split(".")[0] not in debuggers, path
                assert all(alias.name != "debugInPlace" for alias in node.names), path
            if isinstance(node, ast.Import):
                assert all("debugInPlace" not in alias.name and alias.name.split(".")[0] not in debuggers
                           for alias in node.names), path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "breakpoint", path


def test_plotly_optional_dependency_is_declared_by_the_package():
    """Replaces research ``test_plotly_optional_dependency_is_declared`` (superseded)."""
    with (PROJECT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    assert project["optional-dependencies"]["diagnostics"] == ["plotly>=6.0"]
    assert "plotly" not in " ".join(project["dependencies"])
    pins = (PROJECT / "examples" / "requirements-tested.txt").read_text(encoding="utf-8").splitlines()
    assert "plotly==6.8.0" in pins


def test_plotly_import_error_keeps_evaluation_available_and_names_the_package_extra(monkeypatch):
    """Replaces research ``test_plotly_import_error_keeps_evaluation_available`` (superseded).

    Same body; only the expected message names the package extra instead of ``kawin[diagnostics]``.
    """
    namespace = {}
    exec(compile(SYNTHETIC, "synthetic_surrogate", "exec"), namespace)
    from moving_boundary.surrogate_diagnostics import evaluate_tieline_diagnostics, plot_tieline_diagnostics

    report = evaluate_tieline_diagnostics(namespace["surrogate"](), eta_count=3)
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"moving-boundary\[diagnostics\]"):
        plot_tieline_diagnostics(report)
