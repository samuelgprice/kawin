"""Import, optional-dependency and debugger contracts for surrogate diagnostics and comparison.

Covers ``moving_boundary.surrogate_diagnostics`` (FUTURE_PORTS entry 1) and
``moving_boundary.surrogate_comparison`` (entry 2).

Hand port (FUTURE_PORTS entry 1) of the diagnostics check in research
``kawin/tests/test_no_debugger_dependencies.py`` at ``7af24f4``:
``test_diffusion_import_without_research_or_debugger_modules`` and
``test_production_has_no_debugger_references``. The two solver tests in that file are not part of
this port (see FUTURE_PORTS.md). Deviations, each because the package differs from research:

- The guarded subprocess imports ``moving_boundary.surrogate_diagnostics`` and
  ``moving_boundary.surrogate_comparison`` instead of ``kawin.diffusion``. It receives the parent's
  resolved ``sys.path`` (the package and its host may come from ``PYTHONPATH``, which ``-I``
  ignores) and asserts that it loads the same ``moving_boundary`` and the same public kawin.
- Parent audit hooks do not reach a new interpreter, so when the environment requests the
  research-independence guard (``MB_FORBID_RESEARCH=1``), the child installs the existing
  ``tools/batch_safety.guard_independence`` itself before importing the package or kawin.
- It also rejects ``pdb`` and ``plotly``, runs the saved-data evaluators on a synthetic surrogate
  and a comparison of two synthetic saved bundles, and requires every plotting entry point to
  fail with the actionable missing-extra error.
- Further tests check that the child fails for the intended reason when the kawin origin is
  wrong or a forbidden research root is on its path under the guard.
- The source scan covers ``src/moving_boundary`` and also rejects ``pdb``/``ipdb``/``debugpy``
  imports and ``breakpoint()`` calls.

Also holds the package replacements for two research tests of ``test_surrogate_diagnostics.py``
that ``tools/port_tests.py`` records as superseded.
"""
import ast
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "src" / "moving_boundary"
SAFETY = PROJECT / "tools" / "batch_safety.py"

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
import json, os, runpy, sys
path, expected_package, expected_kawin, safety = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
# Parent audit hooks do not carry into this interpreter: install the requested guard here, first.
guards = runpy.run_path(safety)["guard_independence"]()
if os.environ.get("MB_FORBID_RESEARCH") == "1":
    assert "research" in guards, guards
print("independence guards:", sorted(guards))
sys.path[:] = path
BLOCKED = {"examples", "debugpy", "pdb", "plotly"}


class NoResearchDebuggerOrPlotly:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("blocked for contract test: " + fullname, name=fullname)


sys.meta_path.insert(0, NoResearchDebuggerOrPlotly())
from pathlib import Path
import moving_boundary
assert str(Path(moving_boundary.__file__).resolve().parent) == expected_package, \
    "moving_boundary origin: " + moving_boundary.__file__
import kawin
assert str(Path(kawin.__file__).resolve().parent) == expected_kawin, "kawin origin: " + kawin.__file__
import moving_boundary.surrogate_diagnostics as diagnostics
import moving_boundary.surrogate_comparison as comparison
from moving_boundary.surrogate_artifacts import SurrogateArtifactBundle
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

# Comparison of two saved bundles (identical model, scaled model) from the saved files alone.
first = SurrogateArtifactBundle.publish(Path("first.surrogate"), surrogate(), {"system": "contract"})
scaled = surrogate()
for context in scaled.diffusivities:
    for phase in scaled.diffusivities[context]:
        scaled.diffusivities[context][phase] = 2.0 * scaled.diffusivities[context][phase]
second = SurrogateArtifactBundle.publish(Path("second.surrogate"), scaled, {"system": "contract", "scaled": 2.0})
same = comparison.evaluate_saved_surrogate_comparison(Path("first.surrogate"), Path("first.surrogate"),
                                                      interface_eta_count=3, bulk_grid_counts=(3, 3))
assert same["tieline"]["summary"]["maximum_endpoint_displacement"] == 0.0
report = comparison.evaluate_saved_surrogate_comparison(Path("first.surrogate"), Path("second.surrogate"),
                                                        interface_eta_count=3, bulk_grid_counts=(3, 3))
assert [item["path"] for item in report["build_spec_differences"]] == ["build_spec.scaled"]
assert report["tieline"]["summary"]["maximum_endpoint_displacement"] == 0.0
for name, call in {
    "plot_saved_surrogate_comparison": lambda: comparison.plot_saved_surrogate_comparison(report),
    "compare_saved_surrogates": lambda: comparison.compare_saved_surrogates(
        Path("first.surrogate"), Path("second.surrogate"), interface_eta_count=3, bulk_grid_counts=(3, 3)),
}.items():
    try:
        call()
    except ImportError as exc:
        assert "moving-boundary[diagnostics]" in str(exc), (name, str(exc))
    else:
        raise AssertionError(name + " did not report the missing Plotly extra")
leaked = sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED)
assert not leaked, leaked
print("guarded diagnostics and comparison ok")
'''


def _origins():
    """The parent's ``moving_boundary`` and kawin package directories (the intended targets)."""
    import kawin
    import moving_boundary
    return (str(Path(moving_boundary.__file__).resolve().parent), str(Path(kawin.__file__).resolve().parent))


def _run_guarded(tmp_path, *, path=None, package=None, kawin=None, environ=None):
    """Run ``GUARDED`` in a fresh ``-I -S`` interpreter with explicit path and expected origins."""
    expected_package, expected_kawin = _origins()
    if path is None:
        path = [entry for entry in sys.path if entry and Path(entry).resolve() != Path.cwd().resolve()]
    env = dict(os.environ if environ is None else environ)
    return subprocess.run([sys.executable, "-I", "-S", "-c", GUARDED, json.dumps(path),
                           package or expected_package, kawin or expected_kawin, str(SAFETY)],
                          cwd=tmp_path, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=180)


def test_diagnostics_import_without_research_debugger_or_plotly(tmp_path):
    """Saved-data diagnostics and comparison run with research, debugger and Plotly imports rejected.

    Uses the parent's environment, so the child installs the independence guard whenever the run
    requests it (``MB_FORBID_RESEARCH=1``, as in the fresh-target and gate runs).
    """
    result = _run_guarded(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "guarded diagnostics and comparison ok" in result.stdout


def test_guarded_child_passes_with_the_research_guard_forced_on(tmp_path):
    """With the guard explicitly requested, the intended installed targets still pass."""
    forbidden = tmp_path / "forbidden-research-root"
    forbidden.mkdir()
    environ = dict(os.environ, MB_FORBID_RESEARCH="1", MB_FORBID_ROOTS=str(forbidden))
    work = tmp_path / "work"
    work.mkdir()
    result = _run_guarded(work, environ=environ)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "independence guards: ['research'" in result.stdout
    assert "guarded diagnostics and comparison ok" in result.stdout


def test_guarded_child_rejects_an_unexpected_kawin_origin(tmp_path):
    """A kawin other than the intended public host fails the origin assertion."""
    result = _run_guarded(tmp_path, kawin=str(tmp_path / "not-the-public-kawin"))
    assert result.returncode != 0
    assert "AssertionError: kawin origin:" in result.stderr, result.stderr


def test_guarded_child_rejects_forbidden_research_access_under_the_guard(tmp_path):
    """Under ``MB_FORBID_RESEARCH=1`` a forbidden root on the child's path is refused at import."""
    forbidden = tmp_path / "forbidden-research-root"
    forbidden.mkdir()
    environ = dict(os.environ, MB_FORBID_RESEARCH="1", MB_FORBID_ROOTS=str(forbidden))
    work = tmp_path / "work"
    work.mkdir()
    path = [str(forbidden)] + [entry for entry in sys.path
                               if entry and Path(entry).resolve() != Path.cwd().resolve()]
    result = _run_guarded(work, path=path, environ=environ)
    assert result.returncode != 0
    assert "IndependenceViolation: research path on sys.path" in result.stderr, result.stderr
    assert "guarded diagnostics and comparison ok" not in result.stdout


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
