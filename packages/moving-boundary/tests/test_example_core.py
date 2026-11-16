"""Short teaching paths and public plotting across the shipped solver families."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from moving_boundary.analytic import recorded_history
from moving_boundary.plotting import plot_composition_profiles, plot_interface_history
from test_examples import load


@pytest.mark.parametrize("name", ["binary_planar_growth", "binary_tlp_bonding",
                                  "ternary_interstitial_similarity", "three_phase_similarity"])
def test_core_solves_once_without_detailed_validation_and_plots_physical_fractions(name, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = load(name)
    monkeypatch.setattr(example, "evaluate", lambda *args: pytest.fail("core must not run detailed validation"))
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("plot helpers must not display implicitly"))
    result = example.run_core(example.FAST)
    assert result.model.currentTime == pytest.approx(example.FAST.duration)
    assert set(result.metrics) == {"final_time", "inventory_abs", "inventory_rel"}
    assert result.metrics["inventory_rel"] < 1e-6
    figure, (left, right) = plt.subplots(1, 2)
    try:
        assert plot_interface_history(result.model, ax=left, time_offset=2, time_scale=3,
                                      position_offset=0.1, position_scale=4) is left
        times, positions = recorded_history(result.model.interfaceData)
        np.testing.assert_array_equal(left.lines[0].get_xdata(), (times + 2) * 3)
        expected_positions = np.asarray(positions).reshape(len(times), -1)
        np.testing.assert_array_equal(left.lines[0].get_ydata(), (expected_positions[:, 0] - 0.1) * 4)
        assert plot_composition_profiles(result.model, ax=right) is right
        if hasattr(result.model, "getPhysicalPhaseProfiles"):
            profiles = result.model.getPhysicalPhaseProfiles()
            independent_count = len(result.model.allElements) - 1
            assert len(right.lines) == len(profiles) * independent_count
            for index, (_, fractions) in enumerate(profiles):
                np.testing.assert_array_equal(right.lines[index * independent_count].get_ydata(), fractions[:, 1])
        else:
            np.testing.assert_array_equal(right.lines[0].get_ydata(), np.ravel(result.model.data.currentY))
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_minimal_binary_uses_shared_provider_and_default_axes():
    import matplotlib.pyplot as plt
    example = load("binary_constant")
    model = example.build_model()
    model.solve(0.1, minDtFrac=1e-14)
    ax = plot_composition_profiles(model, elements=["NI", "CR"])
    try:
        np.testing.assert_allclose(ax.lines[0].get_ydata() + ax.lines[1].get_ydata(), 1)
        with pytest.raises(ValueError, match="elements"):
            plot_composition_profiles(model, ax=ax, elements=["missing"])
    finally:
        plt.close(ax.figure)


def test_lifecycle_core_builds_and_reloads_without_twin_or_corruption_checks(tmp_path, monkeypatch):
    example = load("surrogate_lifecycle")
    monkeypatch.setattr(example, "integrity_demo", lambda *args: pytest.fail("no corruption demo in core"))
    monkeypatch.setattr(example, "run_differences", lambda *args: pytest.fail("no twin solve in core"))
    first = example.run_core(example.FAST, output_dir=tmp_path)
    second = example.run_core(example.FAST, output_dir=tmp_path)
    assert first.metrics["action"] == "built" and second.metrics["action"] == "reloaded"
    np.testing.assert_array_equal(first.model.getInterfacePosition(), second.model.getInterfacePosition())


@pytest.fixture(scope="module")
def open_bundle(tmp_path_factory):
    example = load("surrogate_workflow_fecrni_open")
    output = tmp_path_factory.mktemp("core_open")
    bundle, _ = example.prepare_pycalphad_bundle(example.WORKFLOW, None, output_dir=output)
    return example, output, bundle


def test_open_core_reloads_without_backend_queries_or_advanced_diagnostics(open_bundle, monkeypatch):
    example, output, bundle = open_bundle
    def forbidden(*args, **kwargs):
        pytest.fail("core reload must not query a backend or run advanced diagnostics")
    monkeypatch.setattr(example, "build_source", forbidden)
    monkeypatch.setattr(example, "saved_bundle_diagnostics", forbidden)
    monkeypatch.setattr(example, "fresh_query_diagnostics", forbidden)
    result = example.run_core(example.WORKFLOW, output_dir=output)
    assert result.metrics["action"] == "reloaded"
    assert result.metrics["final_time"] == pytest.approx(example.WORKFLOW.duration)
    assert result.metrics["inventory_rel_max"] < 1e-8
    assert all(value == 0 for value in result.metrics["final_state_outside_hull"].values())


@pytest.mark.parametrize("frozen", [False, True])
def test_lee_core_accepts_supplied_surrogate_and_skips_validation_twins(open_bundle, monkeypatch, frozen):
    _, _, bundle = open_bundle
    example = load("ternary_fecrni_lee1996")
    def forbidden(*args, **kwargs):
        pytest.fail("core must not prepare another surrogate or run validation twins")
    monkeypatch.setattr(example, "prepare_surrogate", forbidden)
    monkeypatch.setattr(example, "lagged_vs_frozen", forbidden)
    monkeypatch.setattr(example, "workflow_report", forbidden)
    config = dataclasses.replace(example.FROZEN_FAST if frozen else example.LAGGED_FAST, duration=60.0)
    result = example.run_core(config, surrogate=bundle.surrogate)
    assert result.model.currentTime == pytest.approx(config.duration)
    assert result.metrics["inventory_rel"] < 1e-8


def test_advanced_companions_use_public_surrogate_access_and_do_not_import_main_scripts():
    import ast
    examples = Path(__file__).parents[1] / "examples"
    names = {path.stem for path in examples.glob("*.py")}
    for path in [*examples.glob("*.py"), *examples.glob("advanced/*.py")]:
        text = path.read_text(encoding="utf-8")
        assert "._fitSupport" not in text and "._fit_diffusivity_samples" not in text
        if path.parent.name == "advanced":
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] not in names
                elif isinstance(node, ast.Import):
                    assert not names & {alias.name.split(".")[0] for alias in node.names}
