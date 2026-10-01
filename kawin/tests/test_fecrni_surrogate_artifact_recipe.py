import ast
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from kawin.diffusion import (
    TernaryMovingBoundaryThermodynamicsSurrogate,
    surrogate_build_fingerprint,
)


RECIPE_PATH = (
    Path("examples")
    / "ternaryExamples"
    / "FeCrNi"
    / "IllingworthTernaryTwoPhase_FeCrNi_Lee1996Cases.py"
)


def _load_recipe_definitions():
    """Load the notebook-style recipe without executing its simulation cells."""
    source = RECIPE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RECIPE_PATH))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "validated_phase_molar_volumes"
            for target in node.targets
        ):
            break
        definitions.append(node)
    module_name = "_fecrni_surrogate_recipe_test_module"
    module = types.ModuleType(module_name)
    module.__file__ = str(RECIPE_PATH.resolve())
    sys.modules[module_name] = module
    try:
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(RECIPE_PATH), "exec"),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def test_pycalphad_build_spec_is_deterministic_and_backend_specific(tmp_path):
    recipe = _load_recipe_definitions()
    source = recipe.build_source_thermodynamics()
    sampling = {
        "diffusivity_interpolation": "simplex_positive_2x2",
        "diffusivity_bulk_points": np.asarray(
            [[0.1, 0.1], [0.2, 0.1], [0.1, 0.2]], dtype=np.float64
        ),
    }

    first = recipe._surrogate_build_spec(source, sampling)
    second = recipe._surrogate_build_spec(source, sampling)
    canonical_first, fingerprint_first = surrogate_build_fingerprint(first)
    canonical_second, fingerprint_second = surrogate_build_fingerprint(second)

    assert fingerprint_first == fingerprint_second
    assert canonical_first == canonical_second
    assert first["runtime"]["backend"] == "pycalphad"
    assert first["runtime"]["database"]["sha256"]
    assert first["runtime"]["equilibrium_phases"] == ["BCC_A2", "FCC_A1"]
    assert {
        "pycalphad_adapter",
        "pycalphad_thermodynamics",
        "mobility",
        "free_energy_hessian",
    } <= set(first["implementation_sources"])

    changed_sampling = dict(sampling)
    changed_sampling["diffusivity_bulk_points"] = sampling["diffusivity_bulk_points"][::-1]
    assert surrogate_build_fingerprint(
        recipe._surrogate_build_spec(source, changed_sampling)
    )[1] != fingerprint_first

    changed_database = tmp_path / recipe.TDB_PATH.name
    changed_database.write_bytes(recipe.TDB_PATH.read_bytes() + b"\n$ fingerprint test\n")
    recipe.TDB_PATH = changed_database
    assert surrogate_build_fingerprint(
        recipe._surrogate_build_spec(source, sampling)
    )[1] != fingerprint_first


def test_tc_build_spec_retains_tc_runtime_identity_without_starting_backend():
    recipe = _load_recipe_definitions()
    recipe.THERM_ENGINE = "TC"

    class Backend:
        @staticmethod
        def get_runtime_version():
            return "fake-2026a"

    source = recipe.TCPythonThermodynamics(
        recipe._make_tc_config(recipe.TIELINE_PHASES), backend=Backend()
    )
    spec = recipe._surrogate_build_spec(source, {})

    assert spec["runtime"]["backend"] == "tc_python"
    assert spec["runtime"]["tc_python_version"] == "fake-2026a"
    assert "thermocalc_config" in spec["runtime"]
    assert "tc_python_adapter" in spec["implementation_sources"]
    assert "pycalphad_adapter" not in spec["implementation_sources"]


def test_backend_qualified_artifact_stems_do_not_collide():
    recipe = _load_recipe_definitions()

    recipe.THERM_ENGINE = "PYCALPHAD"
    pycalphad_stem = recipe._diffusivity_artifact_stem()
    recipe.THERM_ENGINE = "TC"
    tc_stem = recipe._diffusivity_artifact_stem()

    assert "pycalphad" in pycalphad_stem
    assert "tc" in tc_stem
    assert pycalphad_stem != tc_stem


def test_raw_pycalphad_source_uses_noop_context_manager():
    recipe = _load_recipe_definitions()
    source = recipe.build_source_thermodynamics()

    with recipe._thermodynamics_context(source) as active:
        assert active is source


def test_pycalphad_recipe_writes_compatible_kinetics_jsonl(tmp_path):
    recipe = _load_recipe_definitions()
    source = recipe.build_source_thermodynamics()
    path = tmp_path / "kinetics.jsonl"

    with recipe._capture_surrogate_kinetics(source, path):
        matrix = source.getInterdiffusivity(
            [0.38, 0.001], recipe.TEMPERATURE, phase="FCC_A1"
        )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["record_type"] == "metadata"
    assert records[0]["schema_version"] == 2
    assert records[0]["backend"] == "pycalphad"
    assert records[0]["matrix_elements"] == ["CR", "NI"]
    assert records[1]["record_type"] == "kinetics"
    np.testing.assert_allclose(records[1]["interdiffusivity"], matrix)
    assert records[1]["mobilities"]
    assert records[1]["thermodynamic_factors"]
    assert records[1]["site_fractions"]


def test_pycalphad_recipe_builds_then_strictly_reloads_without_rebuilding(tmp_path):
    recipe = _load_recipe_definitions()
    source = recipe.build_source_thermodynamics()
    recipe.OUTPUTS = tmp_path
    recipe.BULK_DIFFUSIVITY_MODE = "phase_uniform"
    recipe.TC_CAPTURE_DIFFUSIVITY_DIAGNOSTICS = False
    recipe.TC_CHECK_INTERFACE_BULK_DIFFUSIVITY_CONSISTENCY = False
    recipe.TC_DROP_FAILED_BULK_CALCULATIONS = False
    recipe.TC_DROP_INVALID_BULK_MATRICES = False
    calls = []

    def fake_builder(source_thermodynamics, *, diffusivity_sampling=None, artifact_directory=None):
        calls.append(source_thermodynamics)
        eta = np.asarray([0.0, 1.0])
        tielines = {
            "BCC_A2": np.asarray([[0.10, 0.20], [0.15, 0.25]]),
            "FCC_A1": np.asarray([[0.20, 0.10], [0.30, 0.10]]),
        }
        compositions = {
            context: {phase: values.copy() for phase, values in tielines.items()}
            for context in ("interface", "general")
        }
        matrices = {
            context: {
                phase: np.repeat(np.eye(2)[None, :, :], 2, axis=0)
                for phase in tielines
            }
            for context in ("interface", "general")
        }
        surrogate = TernaryMovingBoundaryThermodynamicsSurrogate(
            elements=("FE", "CR", "NI"),
            phases=("BCC_A2", "FCC_A1"),
            tieline_phases=("BCC_A2", "FCC_A1"),
            temperature=recipe.TEMPERATURE,
            eta_samples=eta,
            tieline_compositions=tielines,
            diffusivity_compositions=compositions,
            diffusivities=matrices,
            validity_policy="legacy",
        )
        return surrogate, {}

    recipe.build_tieline_surrogate = fake_builder
    recipe.RELOAD_TIELINE_SURROGATE = False
    built = recipe.prepare_tieline_surrogate(source)
    recipe.RELOAD_TIELINE_SURROGATE = True
    loaded = recipe.prepare_tieline_surrogate(source)

    assert len(calls) == 1
    assert built.path == loaded.path
    np.testing.assert_allclose(
        loaded.surrogate.interface_compositions(0.5)[0], [0.125, 0.225]
    )


def test_fecrni_figure9_configuration_matches_alpha_left_geometry():
    recipe = _load_recipe_definitions()

    assert recipe.TIELINE_PHASES == ("BCC_A2", "FCC_A1")
    assert recipe.PLOT_LEE_OH_FIG9_COMPARISON is True
    assert recipe.INTERFACE_POSITION == pytest.approx(12.0e-6, abs=2.0e-12)
    alpha_fraction = recipe.INTERFACE_POSITION / recipe.HALF_LENGTH
    overall = alpha_fraction * recipe.A_BULK + (1.0 - alpha_fraction) * recipe.B_BULK
    np.testing.assert_allclose(overall, [0.23, 0.0904], atol=1.0e-7, rtol=0.0)


@pytest.mark.parametrize(
    "contents, message",
    [
        ("1,2,3\n4,5,6\n", "two-column"),
        ("1,nan\n2,1\n", "finite"),
    ],
)
def test_lee_oh_figure9_loader_rejects_invalid_data(tmp_path, contents, message):
    recipe = _load_recipe_definitions()
    path = tmp_path / "invalid.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        recipe._load_lee_oh_fig9_curve(path)


def test_lee_oh_figure9_loader_rejects_missing_file(tmp_path):
    recipe = _load_recipe_definitions()

    with pytest.raises(FileNotFoundError, match="Could not find"):
        recipe._load_lee_oh_fig9_curve(tmp_path / "missing.csv")


def test_lee_oh_figure9_comparison_normalizes_and_reports_exact_metrics(tmp_path):
    recipe = _load_recipe_definitions()
    curve_path = tmp_path / "curve.csv"
    np.savetxt(
        curve_path,
        np.asarray([[100.0, 0.5], [1.0, 1.25], [10.0, 1.0]]),
        delimiter=",",
    )

    interface_data = types.SimpleNamespace(
        N=3,
        _time=np.asarray([0.0, 3600.0, 36000.0, 360000.0, 0.0]),
        _y=np.asarray([12.0e-6, 15.0e-6, 12.0e-6, 6.0e-6, 0.0]),
    )
    model = types.SimpleNamespace(phases=["BCC_A2", "FCC_A1"], interfaceData=interface_data)

    figure, axes, metrics = recipe.plot_lee_oh_fig9_comparison(
        model,
        cr_path=curve_path,
        ni_path=curve_path,
        show=False,
    )
    try:
        np.testing.assert_allclose(axes.lines[0].get_xdata(), [1.0, 10.0, 100.0])
        np.testing.assert_allclose(axes.lines[0].get_ydata(), [1.25, 1.0, 0.5])
        assert axes.get_xscale() == "log"
        assert "alpha/BCC" in axes.get_ylabel()
        assert "Cr correction" in axes.lines[1].get_label()
        assert "Ni correction" in axes.lines[2].get_label()
        assert metrics["model"]["peak_normalized_alpha_thickness"] == pytest.approx(1.25)
        assert metrics["model"]["peak_time_hours"] == pytest.approx(1.0)
        assert metrics["model"]["terminal_equilibrium_deviation"] == pytest.approx(0.0)
        for reference in metrics["references"].values():
            assert reference["comparison"]["overlap_count"] == 3
            assert reference["comparison"]["mean_absolute_error"] == pytest.approx(0.0)
            assert reference["comparison"]["root_mean_square_error"] == pytest.approx(0.0)
            assert reference["comparison"]["maximum_absolute_error"] == pytest.approx(0.0)
    finally:
        recipe.plt.close(figure)


def test_lee_oh_figure9_comparison_rejects_non_bcc_left_phase(tmp_path):
    recipe = _load_recipe_definitions()
    model = types.SimpleNamespace(phases=["FCC_A1", "BCC_A2"], interfaceData=None)

    with pytest.raises(ValueError, match="first phase must be BCC_A2"):
        recipe.plot_lee_oh_fig9_comparison(model, show=False)
