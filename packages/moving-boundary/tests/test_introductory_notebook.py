"""Execute the introductory notebook as ordered cells against the public package."""

import json
from pathlib import Path

import numpy as np
import pytest

from moving_boundary.analytic import recorded_history
from moving_boundary.run_diagnostics import evaluate_bulk_coverage


@pytest.mark.parametrize("start_directory", ["repository", "examples"])
def test_fecrni_tutorial_solves_and_plots(start_directory, monkeypatch):
    """Exercise fresh sampling, the full solve and both historical/current plots.

    Execute without file metadata or an examples import path, matching the
    two supported working directories. Conservation and final sample coverage are workflow
    checks for this case, not evidence of physical accuracy.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    package = Path(__file__).resolve().parents[1]
    repository = package.parents[1]
    notebook = json.loads((package / "examples/fecrni_first_simulation.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    cwd = repository if start_directory == "repository" else package / "examples"
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(plt, "show", lambda: plt.gcf().canvas.draw())
    previous_figures = set(plt.get_fignums())
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                exec(compile(source, f"fecrni_first_simulation.ipynb:cell-{index}", "exec"), namespace)

        model = namespace["model"]
        times, positions = recorded_history(model.interfaceData)
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(3600.0)
        assert model.currentTime == pytest.approx(3600.0)
        assert np.all(np.isfinite(positions))
        assert 0.0 < float(model.getInterfacePosition()) < namespace["domain_length"]
        assert abs(float(model.getInterfacePosition()) - namespace["interface_position"]) > 1e-7
        assert namespace["relative_drift"] < 1e-8
        for _, fractions in model.getPhysicalPhaseProfiles():
            assert np.all(np.isfinite(fractions))
            assert np.all((fractions >= 0.0) & (fractions <= 1.0))
            np.testing.assert_allclose(fractions.sum(axis=1), 1.0, atol=1e-14)
        coverage = evaluate_bulk_coverage(
            namespace["surrogate"], dict(zip(namespace["phases"], model.getTransformedState())))
        assert all(report["outside"] == 0 for report in coverage.values())
        assert len(set(plt.get_fignums()) - previous_figures) == 2
        assert namespace["database"].resolve() == package / "examples/FeCrNi_MatCalc_open.tdb"
        assert Path.cwd() == cwd
        assert "__file__" not in namespace
    finally:
        for number in set(plt.get_fignums()) - previous_figures:
            plt.close(number)


PACKAGE = Path(__file__).resolve().parents[1]
EXAMPLES = PACKAGE / "examples"
TUTORIALS = ["binary_tlp_bonding", "fecrni_surrogate_reuse", "fecrni_surrogate_assessment",
             "fecrni_resolution_study", "ternary_interstitial_diffusion", "three_phase_growth"]


def _cells(name):
    """Read executable cells without requiring Jupyter in the normal test environment."""
    notebook = json.loads((EXAMPLES / f"{name}.ipynb").read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _execute(cell, namespace):
    exec(compile("".join(cell["source"]), "tutorial-cell", "exec"), namespace)


@pytest.fixture
def notebook_environment(monkeypatch, tmp_path):
    """Isolate notebook outputs and draw each figure without opening GUI windows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    previous = set(plt.get_fignums())
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(plt, "show", lambda: plt.gcf().canvas.draw())
    yield
    for number in set(plt.get_fignums()) - previous:
        plt.close(number)


@pytest.mark.parametrize("start_directory", ["repository", "examples"])
@pytest.mark.parametrize("name", TUTORIALS)
def test_standalone_tutorial(name, start_directory, monkeypatch, notebook_environment):
    """Execute the unchanged full tutorials and check their scientific workflow contracts."""
    cwd = PACKAGE.parents[1] if start_directory == "repository" else EXAMPLES
    monkeypatch.chdir(cwd)
    namespace = {"__name__": "__main__"}
    for cell in _cells(name):
        _execute(cell, namespace)
    assert Path.cwd() == cwd
    assert "__file__" not in namespace

    if "model" in namespace:
        model = namespace["model"]
        assert model.currentTime == pytest.approx(namespace["duration"])
        times, positions = recorded_history(model.interfaceData)
        assert times[0] == 0
        assert times[-1] == pytest.approx(namespace["duration"])
        assert np.all(np.isfinite(positions))
        assert np.all((positions > 0) & (positions < namespace["domain_length"]))
        assert namespace["relative_drift"] < 1e-7
        if hasattr(model, "getPhysicalPhaseProfiles"):
            for _, fractions in model.getPhysicalPhaseProfiles():
                assert np.all(np.isfinite(fractions))
                assert np.all((fractions >= 0) & (fractions <= 1))
                np.testing.assert_allclose(fractions.sum(axis=1), 1, atol=1e-14)
        else:
            fractions = np.asarray(model.data.currentY)
            assert np.all(np.isfinite(fractions))
            assert np.all((fractions >= 0) & (fractions <= 1))

    if name == "binary_tlp_bonding":
        positions = np.ravel(positions)
        assert namespace["interface_position"] < positions.max() < namespace["width_bound"]
        assert 0 < positions[-1] < namespace["interface_position"]
    elif name == "fecrni_surrogate_reuse":
        from copy import deepcopy
        import shutil
        from moving_boundary.surrogate_artifacts import (
            SurrogateArtifactCompatibilityError, SurrogateArtifactIntegrityError,
        )

        def forbidden(*args, **kwargs):
            pytest.fail("Reload must not sample or construct a backend")

        namespace["sample_surrogate"] = forbidden
        build = next(cell for cell in _cells(name) if "build-bundle" in cell.get("metadata", {}).get("tags", []))
        # Run just the actual load-or-build call, keeping the builder replaced.
        source = "".join(build["source"])
        exec(source[source.index("bundle, action ="):], namespace)
        assert namespace["action"] == "reloaded"
        before = {path.name: path.read_bytes() for path in namespace["output_dir"].iterdir()}
        altered = deepcopy(namespace["build_spec"])
        altered["sampling"]["temperature"] += 1
        with pytest.raises(SurrogateArtifactCompatibilityError):
            namespace["load_or_build_diagnostic_surrogate"](
                namespace["output_dir"], altered, forbidden, report_roles=namespace["report_roles"])
        assert before == {path.name: path.read_bytes() for path in namespace["output_dir"].iterdir()}

        damaged = namespace["output_dir"].parent / "damaged-copy"
        shutil.copytree(namespace["output_dir"], damaged)
        model_member = namespace["bundle"].manifest["members"]["model"]["path"]
        (damaged / model_member).write_bytes(b"damaged model")
        with pytest.raises(SurrogateArtifactIntegrityError):
            namespace["load_or_build_diagnostic_surrogate"](
                damaged, namespace["build_spec"], forbidden, report_roles=namespace["report_roles"])

        # Restart semantics: settings/import cells plus reload, without executing the builder cell.
        import moving_boundary.providers.pycalphad as provider
        from moving_boundary.surrogates import TernaryMovingBoundaryThermodynamicsSurrogate
        monkeypatch.setattr(provider, "create_pycalphad_thermodynamics_source", forbidden)
        monkeypatch.setattr(TernaryMovingBoundaryThermodynamicsSurrogate, "from_database", forbidden)
        cold = {"__name__": "__main__"}
        for cell in _cells(name):
            tags = cell.get("metadata", {}).get("tags", [])
            if "build-bundle" in tags:
                continue
            _execute(cell, cold)
            if "reload-bundle" in tags:
                break
        assert "source" not in cold
        for phase in namespace["phases"]:
            for before_array, after_array in zip(
                namespace["surrogate"].get_diffusivity_samples(phase),
                cold["surrogate"].get_diffusivity_samples(phase),
            ):
                np.testing.assert_array_equal(before_array, after_array)
        for phase, point in zip(namespace["phases"], (namespace["left_composition"], namespace["right_composition"])):
            np.testing.assert_array_equal(
                namespace["surrogate"].getInterdiffusivity(point, namespace["temperature"], phase=phase),
                cold["surrogate"].getInterdiffusivity(point, namespace["temperature"], phase=phase))
    elif name == "fecrni_surrogate_assessment":
        assert len(namespace["construction"]["records"]) > 0
        assert all(report["outside"] == 0 for report in namespace["coverage"].values())
        assert all(len(indices) <= 20 for indices in namespace["sample_indices"].values())
        for phase, report in namespace["loo"]["phase_reports"].items():
            assert len(report["prediction_status"]) == len(namespace["sample_indices"][phase])
        assert {row[2] for row in namespace["fresh_summary"]} == {"training", "held-out"}
        assert any(row[2] == "held-out" and row[4] > 0 for row in namespace["fresh_summary"])
        assert any(row[2] == "training" and row[4] > 0 for row in namespace["fresh_summary"])
        for _, _, _, requested, compared, excluded, mean in namespace["fresh_summary"]:
            assert requested == compared + excluded
            assert (mean is None) == (compared == 0)
    elif name == "fecrni_resolution_study":
        spatial = [model for _, model in namespace["spatial_models"]]
        temporal = [model for _, model in namespace["temporal_models"]]
        sampling = [model for _, model in namespace["sampling_models"]]
        assert [(m.phaseANodes, m.phaseBNodes) for m in spatial] == [(21, 41), (41, 81), (81, 161)]
        assert [m.semiLog_dt for m in temporal] == [0.05, 0.025, 0.0125]
        assert all(m.therm is namespace["surrogate"] for m in spatial + temporal)
        assert all((m.phaseANodes, m.phaseBNodes, m.semiLog_dt) == (21, 41, 0.05) for m in sampling)
        np.testing.assert_array_equal(sampling[1].therm.eta_samples, namespace["surrogate"].eta_samples)
        assert len({len(m.therm.get_diffusivity_samples("BCC_A2")[0]) for m in sampling}) == 3
        for candidate in spatial + temporal + sampling:
            assert candidate.currentTime == pytest.approx(namespace["duration"])
            assert np.isfinite(candidate.getInterfacePosition())
            assert len(candidate.mesh.z) == 61
    elif name == "ternary_interstitial_diffusion":
        for state, (_, fractions) in zip(namespace["states"], namespace["physical_profiles"]):
            np.testing.assert_allclose(namespace["x_from_u"](state), fractions[:, 1:], atol=1e-14)
        assert namespace["comparison"]["interface_rel_error"] < 1e-2
    elif name == "three_phase_growth":
        assert namespace["exact"].k_ab == max(root.k_ab for root in namespace["roots"])
        assert np.all(positions[:, 0] < positions[:, 1])
        assert positions[-1, 0] < positions[0, 0] and positions[-1, 1] > positions[0, 1]
        assert model.currentTime + namespace["start_time"] == pytest.approx(namespace["final_time"])
        np.testing.assert_allclose(positions[-1], namespace["reference_positions"][-1], atol=1e-3)


@pytest.mark.parametrize("failure", [None, "preflight", "sampling"])
def test_thermocalc_tutorial_closes_session(failure, monkeypatch, notebook_environment):
    """Check the notebook's real configuration and context-manager flow without a licence."""
    import tempfile
    from moving_boundary.surrogates import TernaryMovingBoundaryThermodynamicsSurrogate

    monkeypatch.chdir(EXAMPLES)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(Path(Path.cwd().anchor) / "tc-notebook-test"))
    namespace = {"__name__": "__main__"}
    cells = _cells("fecrni_thermocalc")
    sampling_cell = next(cell for cell in cells if "tc-sample" in cell.get("metadata", {}).get("tags", []))
    for cell in cells:
        if cell is sampling_cell:
            break
        _execute(cell, namespace)
    assert namespace["tc_config"].phases == ("BCC_A2#1", "FCC_A1#1")
    assert namespace["tc_config"].reference_element == "FE"
    assert not namespace["tc_config"].use_default_phases
    sessions = []

    class Session:
        def __init__(self, config):
            self.config, self.closed = config, False
            sessions.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def preflight(self, **kwargs):
            if failure == "preflight":
                raise RuntimeError("preflight failed")
            return {"ok": True}

    def sample(**kwargs):
        assert kwargs["thermodynamics"] is sessions[0]
        if failure == "sampling":
            raise RuntimeError("sampling failed")
        return object()

    namespace["TCPythonThermodynamics"] = Session
    monkeypatch.setattr(TernaryMovingBoundaryThermodynamicsSurrogate, "from_database", sample)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            _execute(sampling_cell, namespace)
    else:
        _execute(sampling_cell, namespace)
    assert sessions[0].closed


def test_collection_uses_public_package_apis_and_keeps_licensed_outputs_local():
    """Reject sibling-example dependencies and accidental saved licensed outputs."""
    import ast

    sibling_names = {path.stem for path in EXAMPLES.glob("*.py")} | {"advanced"}
    for name in [*TUTORIALS, "fecrni_thermocalc"]:
        for cell in _cells(name):
            source = "".join(cell["source"])
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] not in sibling_names
                elif isinstance(node, ast.Import):
                    assert not sibling_names & {alias.name.split(".")[0] for alias in node.names}
            if name == "fecrni_thermocalc":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None
