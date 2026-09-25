import builtins
import io
from pathlib import Path
import tomllib

import numpy as np
import pytest

from kawin.diffusion import (
    MergedPhaseDiffusivitySurrogate,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    evaluate_diffusivity_diagnostics,
    evaluate_diffusivity_leave_one_out,
    evaluate_tieline_diagnostics,
    load_calculation_site_fractions,
    plot_bulk_diffusivity_diagnostics,
    plot_calculation_site_fractions,
    plot_diffusivity_leave_one_out,
    plot_interface_diffusivity_diagnostics,
    plot_surrogate_diagnostics,
    plot_tieline_diagnostics,
)


PHASES = ("ALPHA", "BETA")
ELEMENTS = ("Z", "X", "Y")
ETA = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
LEFT = np.asarray([[0.20, 0.10], [0.25, 0.10], [0.30, 0.10]], dtype=np.float64)
RIGHT = np.asarray([[0.40, 0.20], [0.45, 0.20], [0.50, 0.20]], dtype=np.float64)
GENERAL = np.asarray([[0.20, 0.10], [0.30, 0.10], [0.20, 0.20], [0.30, 0.20]], dtype=np.float64)
MATRIX = np.asarray([[2.0, 0.2], [0.1, 1.0]], dtype=np.float64)


def test_plotly_optional_dependency_is_declared():
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    assert project["optional-dependencies"]["diagnostics"] == ["plotly>=6.0"]
    assert "plotly=6.8.0" in (root / "environment.yml").read_text(encoding="utf-8")


def _matrices(count, scale=1.0):
    return np.broadcast_to(scale * MATRIX, (count, 2, 2)).copy()


def _surrogate(*, source="from_database", interpolation="nearest", validity_policy="raise"):
    metadata = {
        "source": source,
        "precipitate_phase": PHASES[1],
    }
    if source == "from_database":
        metadata.update({"probe_start": [0.20, 0.15], "probe_end": [0.30, 0.15]})
    else:
        metadata.update({
            "probe_point": [0.25, 0.15],
            "generated_probe_points": [[0.20, 0.15], [0.25, 0.15], [0.30, 0.15]],
        })
    general = GENERAL if interpolation == "nearest" else np.asarray(
        [[0.20, 0.10], [0.40, 0.10], [0.20, 0.30]], dtype=np.float64
    )
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=ELEMENTS,
        phases=PHASES,
        tieline_phases=PHASES,
        temperature=1000.0,
        eta_samples=ETA,
        tieline_compositions={PHASES[0]: LEFT, PHASES[1]: RIGHT},
        diffusivity_compositions={
            "interface": {PHASES[0]: LEFT, PHASES[1]: RIGHT},
            "general": {PHASES[0]: general, PHASES[1]: general},
        },
        diffusivities={
            "interface": {PHASES[0]: _matrices(3), PHASES[1]: _matrices(3, 0.5)},
            "general": {PHASES[0]: _matrices(len(general)), PHASES[1]: _matrices(len(general), 0.5)},
        },
        diffusivity_interpolation=interpolation,
        metadata=metadata,
        validity_policy=validity_policy,
    )


class _Truth:
    def __init__(self, *, fail_tieline=False, fail_diffusivity=False, full_tieline_compositions=False, scale=1.0):
        self.fail_tieline = fail_tieline
        self.fail_diffusivity = fail_diffusivity
        self.full_tieline_compositions = full_tieline_compositions
        self.scale = float(scale)
        self.tieline_calls = 0
        self.diffusivity_calls = 0

    def getInterfacialComposition(self, x, T, precPhase=None, returnMeta=False):
        self.tieline_calls += 1
        point = np.asarray(x, dtype=np.float64)
        if self.fail_tieline and point[0] > 0.24:
            raise RuntimeError("synthetic tie-line failure")
        fraction = (point[0] - 0.20) / 0.10
        left = LEFT[0] + fraction * (LEFT[-1] - LEFT[0])
        right = RIGHT[0] + fraction * (RIGHT[-1] - RIGHT[0])
        if self.full_tieline_compositions:
            left = np.concatenate(([1.0 - np.sum(left)], left))
            right = np.concatenate(([1.0 - np.sum(right)], right))
        metadata = {
            "endpoint_phases": PHASES,
            "endpoints": (
                {"phase": PHASES[0], "composition": left},
                {"phase": PHASES[1], "composition": right},
            ),
        }
        return (left, right, metadata) if returnMeta else (left, right)

    def getInterdiffusivity(self, x, T, phase=None):
        self.diffusivity_calls += 1
        point = np.asarray(x, dtype=np.float64)
        if self.fail_diffusivity and point[0] > 0.25:
            raise RuntimeError("synthetic diffusivity failure")
        phase_scale = 1.0 if phase == PHASES[0] else 0.5
        return self.scale * phase_scale * MATRIX


def _merged(interpolation="nearest"):
    points = GENERAL if interpolation == "nearest" else np.asarray(
        [[0.20, 0.10], [0.40, 0.10], [0.20, 0.30]], dtype=np.float64
    )
    return MergedPhaseDiffusivitySurrogate(
        elements=ELEMENTS,
        phase="BETA",
        temperature=1000.0,
        diffusivity_compositions={"interface": points, "general": points},
        diffusivities={"interface": _matrices(len(points), 0.5), "general": _matrices(len(points), 0.5)},
        diffusivity_interpolation=interpolation,
    )


def _loo_surrogate(interpolation="simplex_linear", validity_policy="raise"):
    """Build a nonconstant, positive-matrix bulk sample set for holdout tests."""
    if interpolation == "continuous_grid":
        axes = (np.asarray([0.20, 0.30, 0.40]), np.asarray([0.10, 0.20, 0.30]))
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    else:
        axes = None
        points = np.asarray([
            [0.20, 0.10], [0.40, 0.10], [0.20, 0.30], [0.40, 0.30], [0.30, 0.20],
        ])
    matrices = np.asarray([
        [[2.0 + x, 0.1], [0.05, 1.0 + y]] for x, y in points
    ])
    matrices[len(points) // 2] *= 1.5
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=ELEMENTS,
        phases=PHASES,
        tieline_phases=PHASES,
        temperature=1000.0,
        eta_samples=ETA,
        tieline_compositions={PHASES[0]: LEFT, PHASES[1]: RIGHT},
        diffusivity_compositions={
            "interface": {PHASES[0]: LEFT, PHASES[1]: RIGHT},
            "general": {PHASES[0]: points, PHASES[1]: points},
        },
        diffusivities={
            "interface": {PHASES[0]: _matrices(3), PHASES[1]: _matrices(3, 0.5)},
            "general": {PHASES[0]: matrices, PHASES[1]: matrices * 0.5},
        },
        diffusivity_interpolation=interpolation,
        diffusivity_bulk_grids=axes,
        validity_policy=validity_policy,
    )


def test_diffusivity_leave_one_out_strictly_reports_missing_retained_hull_support():
    surrogate = _loo_surrogate()
    original = surrogate.diffusivities["general"]["ALPHA"].copy()
    report = evaluate_diffusivity_leave_one_out(surrogate, phases="ALPHA")
    phase = report["phase_reports"]["ALPHA"]

    assert report["phases"] == ("ALPHA",)
    assert np.array_equal(phase["sample_indices"], np.arange(5))
    np.testing.assert_array_equal(phase["actual_matrices"], original)
    np.testing.assert_array_equal(surrogate.diffusivities["general"]["ALPHA"], original)
    assert phase["predicted_matrices"].shape == (5, 2, 2)
    assert phase["prediction_status"][0] == "insufficient_fit_support"
    assert not phase["fallback"][0]
    assert np.isnan(phase["predicted_matrices"][0]).all()
    assert not phase["fallback"][4]
    assert phase["refit_interpolation"][4] == "simplex_linear"
    assert phase["prediction_status"][4] == "ok"
    assert phase["matrix_relative_error"][4] > 0.0
    assert phase["summary"]["nearest_fallback_count"] == 0
    assert phase["summary"]["insufficient_fit_support_count"] == 4
    assert phase["summary"]["failure_count"] == 0
    assert np.all(phase["actual_valid"])
    np.testing.assert_allclose(
        np.sort(phase["actual_eigenvalues"][4].real),
        np.sort(np.linalg.eigvals(original[4]).real),
    )


def test_diffusivity_leave_one_out_nearest_subset_and_merged_surrogate():
    surrogate = _loo_surrogate(interpolation="nearest")
    report = evaluate_diffusivity_leave_one_out(
        surrogate, phases="ALPHA", sample_indices={"ALPHA": [4, 0]}
    )["phase_reports"]["ALPHA"]
    assert report["sample_indices"].tolist() == [4, 0]
    assert np.all(report["refit_interpolation"] == "nearest")
    assert report["prediction_status"].tolist() == ["ok", "insufficient_fit_support"]
    for row, index in enumerate(report["sample_indices"]):
        points = np.delete(surrogate._fitSupport["ALPHA"]["points"], index, axis=0)
        matrices = np.delete(surrogate._fitSupport["ALPHA"]["matrices"], index, axis=0)
        nearest = np.argmin(np.sum((points - report["compositions"][row]) ** 2, axis=1))
        if report["prediction_status"][row] == "ok":
            np.testing.assert_array_equal(report["predicted_matrices"][row], matrices[nearest])

    merged = evaluate_diffusivity_leave_one_out(_merged())
    assert merged["phases"] == ("BETA",)
    assert merged["phase_reports"]["BETA"]["summary"]["sample_count"] == len(GENERAL)


def test_diffusivity_leave_one_out_regular_grid_uses_scattered_refit():
    surrogate = _loo_surrogate(interpolation="continuous_grid")
    report = evaluate_diffusivity_leave_one_out(
        surrogate, phases="ALPHA", sample_indices=[4]
    )
    phase = report["phase_reports"]["ALPHA"]
    assert report["source_interpolation"] == "continuous_grid"
    assert phase["refit_interpolation"].tolist() == ["simplex_linear"]
    assert phase["refit_note"].tolist() == ["continuous_grid_missing_node_downgrade"]
    assert not phase["fallback"][0]
    assert phase["matrix_relative_error"][0] > 0.0
    assert phase["summary"]["failure_count"] == 0


def test_diffusivity_leave_one_out_reports_unavailable_single_sample():
    point = GENERAL[:1]
    surrogate = MergedPhaseDiffusivitySurrogate(
        elements=ELEMENTS,
        phase="BETA",
        temperature=1000.0,
        diffusivity_compositions={"interface": point, "general": point},
        diffusivities={"interface": _matrices(1), "general": _matrices(1)},
    )
    phase = evaluate_diffusivity_leave_one_out(surrogate)["phase_reports"]["BETA"]
    assert phase["summary"]["failure_count"] == 0
    assert phase["summary"]["insufficient_fit_support_count"] == 1
    assert phase["summary"]["invalid_prediction_count"] == 0
    assert np.isnan(phase["predicted_matrices"]).all()
    assert phase["prediction_status"].tolist() == ["insufficient_fit_support"]

    with pytest.raises(ValueError, match="missing phase"):
        evaluate_diffusivity_leave_one_out(surrogate, sample_indices={"ALPHA": [0]})


def test_diffusivity_leave_one_out_exposes_invalid_interpolated_matrix():
    points = np.asarray([
        [0.20, 0.10], [0.40, 0.10], [0.20, 0.30], [0.2666666667, 0.1666666667],
    ])
    matrices = np.asarray([
        [[2.0, 100.0], [0.01, 2.0]],
        [[2.0, 0.01], [100.0, 2.0]],
        [[2.0, 0.0], [0.0, 2.0]],
        [[2.0, 0.0], [0.0, 2.0]],
    ])
    surrogate = MergedPhaseDiffusivitySurrogate(
        elements=ELEMENTS,
        phase="BETA",
        temperature=1000.0,
        diffusivity_compositions={"interface": points, "general": points},
        diffusivities={"interface": matrices, "general": matrices},
        diffusivity_interpolation="simplex_linear",
    )
    phase = evaluate_diffusivity_leave_one_out(
        surrogate, sample_indices=[3]
    )["phase_reports"]["BETA"]
    assert phase["actual_valid"].tolist() == [True]
    assert phase["predicted_valid"].tolist() == [False]
    assert np.min(phase["predicted_eigenvalues"].real) < 0.0
    assert phase["summary"]["invalid_prediction_count"] == 1


def test_diffusivity_leave_one_out_refits_simplex_positive_2x2_with_production_evaluator(monkeypatch):
    surrogate = _loo_surrogate(interpolation="simplex_positive_2x2")
    import kawin.diffusion.SurrogateDiagnostics as diagnostics_module

    monkeypatch.setattr(
        diagnostics_module, "_BulkDiffusivitySimplexLinear2D",
        lambda *args: (_ for _ in ()).throw(AssertionError("simplex-linear refit was used")),
    )
    phase = evaluate_diffusivity_leave_one_out(
        surrogate, phases="ALPHA", sample_indices=[4]
    )["phase_reports"]["ALPHA"]

    assert phase["refit_interpolation"].tolist() == ["simplex_positive_2x2"]
    assert phase["prediction_status"].tolist() == ["ok"]
    assert phase["predicted_valid"].tolist() == [True]
    assert not phase["fallback"][0]


def test_diffusivity_leave_one_out_positive_refit_avoids_simplex_linear_crossing():
    points = np.asarray([
        [0.20, 0.10], [0.40, 0.10], [0.20, 0.30], [0.2666666667, 0.1666666667],
    ])
    matrices = np.asarray([
        [[2.0, 100.0], [0.01, 2.0]],
        [[2.0, 0.01], [100.0, 2.0]],
        [[2.0, 0.0], [0.0, 2.0]],
        [[2.0, 0.0], [0.0, 2.0]],
    ])

    def make(mode):
        return MergedPhaseDiffusivitySurrogate(
            elements=ELEMENTS, phase="BETA", temperature=1000.0,
            diffusivity_compositions={"interface": points, "general": points},
            diffusivities={"interface": matrices, "general": matrices},
            diffusivity_interpolation=mode,
        )

    linear = evaluate_diffusivity_leave_one_out(make("simplex_linear"), sample_indices=[3])["phase_reports"]["BETA"]
    positive = evaluate_diffusivity_leave_one_out(make("simplex_positive_2x2"), sample_indices=[3])["phase_reports"]["BETA"]
    assert linear["refit_interpolation"].tolist() == ["simplex_linear"]
    assert linear["prediction_status"].tolist() == ["invalid_prediction"]
    assert positive["refit_interpolation"].tolist() == ["simplex_positive_2x2"]
    assert positive["prediction_status"].tolist() == ["ok"]
    assert positive["predicted_valid"].tolist() == [True]


def test_diffusivity_leave_one_out_legacy_marks_vertex_nearest_fallback():
    strict = _loo_surrogate(validity_policy="raise")
    legacy = _loo_surrogate(validity_policy="legacy")
    strict_phase = evaluate_diffusivity_leave_one_out(strict, phases="ALPHA", sample_indices=[0])["phase_reports"]["ALPHA"]
    legacy_phase = evaluate_diffusivity_leave_one_out(legacy, phases="ALPHA", sample_indices=[0])["phase_reports"]["ALPHA"]
    assert strict_phase["prediction_status"].tolist() == ["insufficient_fit_support"]
    assert not strict_phase["fallback"][0]
    assert np.isnan(strict_phase["predicted_matrices"][0]).all()
    assert legacy_phase["prediction_status"].tolist() == ["nearest_fallback"]
    assert legacy_phase["fallback"].tolist() == [True]
    assert np.all(np.isfinite(legacy_phase["predicted_matrices"][0]))


def test_diffusivity_leave_one_out_uses_canonical_solver_usable_fit_population():
    corners = np.asarray([[0.10, 0.10], [0.35, 0.10], [0.10, 0.35], [0.35, 0.35]])
    points = np.vstack((corners, [0.22, 0.12], [0.12, 0.22], [0.24, 0.24], corners[0]))
    matrices = np.asarray([
        np.eye(2), np.eye(2), np.eye(2), np.eye(2),
        [[1.0, 0.0], [0.0, -1.0]], np.eye(2) * 3.0, np.eye(2) * 2.0, np.eye(2),
    ])
    validity = {
        "coordinates": points,
        "statuses": ("VALID",) * 4 + ("KNOWN_INVALID", "UNKNOWN", "VALID", "VALID"),
        "fit_usable": (True,) * 6 + (False, True),
    }
    surrogate = MergedPhaseDiffusivitySurrogate(
        elements=ELEMENTS, phase="BETA", temperature=1000.0,
        diffusivity_compositions={"interface": corners, "general": points},
        diffusivities={"interface": _matrices(4), "general": matrices},
        diffusivity_validity={"interface": None, "general": validity},
    )
    phase = evaluate_diffusivity_leave_one_out(surrogate)["phase_reports"]["BETA"]
    assert phase["source_sample_count"] == 4
    assert phase["loo_fit_sample_count"] == 3
    np.testing.assert_allclose(phase["compositions"], corners)


def test_diffusivity_leave_one_out_plot_shows_matrices_and_point_status():
    pytest.importorskip("plotly")
    report = evaluate_diffusivity_leave_one_out(_loo_surrogate(), phases="ALPHA")
    figure = plot_diffusivity_leave_one_out(report, "ALPHA", renderer=None)
    trace = figure.data[0]

    assert trace.type == "scatterternary"
    assert len(trace.a) == 1
    assert np.allclose(np.asarray(trace.a) + np.asarray(trace.b) + np.asarray(trace.c), 1.0)
    assert figure.layout.ternary.aaxis.title.text == ELEMENTS[2]
    assert figure.layout.ternary.baxis.title.text == ELEMENTS[0]
    assert figure.layout.ternary.caxis.title.text == ELEMENTS[1]
    assert trace.marker.symbol[0] == "circle"
    assert "Actual D" in trace.text[0]
    assert "Predicted D" in trace.text[0]
    assert "Absolute error" in trace.text[0]
    assert "Predicted eigenvalues" in trace.text[0]
    assert "prediction status: ok" in trace.text[0]
    assert trace.marker.showscale
    assert figure.data[1].name == "Insufficient fit support"
    assert figure.data[1].marker.symbol == "square-open"

    with pytest.raises(ValueError, match="does not contain phase"):
        plot_diffusivity_leave_one_out(report, "MISSING", renderer=None)


def test_diffusivity_leave_one_out_plot_marks_failed_refit():
    pytest.importorskip("plotly")
    point = GENERAL[:1]
    surrogate = MergedPhaseDiffusivitySurrogate(
        elements=ELEMENTS,
        phase="BETA",
        temperature=1000.0,
        diffusivity_compositions={"interface": point, "general": point},
        diffusivities={"interface": _matrices(1), "general": _matrices(1)},
    )
    report = evaluate_diffusivity_leave_one_out(surrogate)
    figure = plot_diffusivity_leave_one_out(report, "BETA", renderer=None)
    assert len(figure.data) == 1
    assert figure.data[0].name == "Insufficient fit support"
    assert figure.data[0].marker.symbol == "square-open"
    assert "insufficient_fit_support" in figure.data[0].text[0]


def test_site_fraction_plot_compares_equivalent_b2_sublattices_with_disordered_state():
    pytest.importorskip("plotly")
    phase_state = {
        "phase": "BCC_B2#3", "stable": True, "phase_amount": 1.0,
        "phase_composition": [0.5, 0.3, 0.2], "diagnostic_errors": {},
        "site_fractions": [
            {"sublattice": 1, "site_ratio": 1.0, "constituents": {"W": 0.8, "TI": 0.15, "FE": 0.05}},
            {"sublattice": 2, "site_ratio": 1.0, "constituents": {"W": 0.2, "TI": 0.45, "FE": 0.35}},
            {"sublattice": 3, "site_ratio": 3.0, "constituents": {"VA": 1.0}},
        ],
    }
    report = {
        "metadata": {"record_type": "metadata", "element_order": ["W", "TI", "FE"]},
        "records": [{
            "record_type": "calculation_site_fractions", "calculation_index": 0,
            "kind": "equilibrium", "requested_phase": None, "temperature": 1973.0,
            "input_full_composition": [0.5, 0.3, 0.2], "status": "ok",
            "stable_composition_sets": ["BCC_B2#3"], "phases": [phase_state],
        }],
    }
    fig = plot_calculation_site_fractions(report, "BCC_B2", renderer=None)

    assert len(fig.data) == 8
    assert fig.data[0].name == "Maximum ordering deviation"
    assert fig.data[0].marker.color[0] == pytest.approx(0.3)
    assert fig.data[0].marker.symbol[0] == "circle"
    assert fig.data[1].name == "s1 W"
    assert fig.data[1].marker.color[0] == pytest.approx(0.3)
    assert fig.data[1].marker.colorbar.title.text == "actual − disordered"
    assert fig.data[0].b[0] == pytest.approx(0.5)
    assert "BCC_B2#3" in fig.data[0].text[0]
    assert "disordered: 0.5" in fig.data[0].text[0]
    assert len(fig.layout.updatemenus[0].buttons) == 8

    phase_state["site_fractions"][1]["constituents"] = {"VA": 1.0}
    fig_without_reference = plot_calculation_site_fractions(report, "BCC_B2", renderer=None)
    assert fig_without_reference.data[0].marker.colorbar.title.text == "site fraction"


def test_site_fraction_sidecar_loader_checks_ordered_records(monkeypatch):
    lines = (
        '{"record_type":"metadata","element_order":["W","TI","FE"]}\n'
        '{"record_type":"calculation_site_fractions","calculation_index":0}\n'
    )
    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: io.StringIO(lines))
    loaded = load_calculation_site_fractions("unused.jsonl")
    assert loaded["records"][0]["calculation_index"] == 0


@pytest.mark.parametrize("source", ["from_database", "from_database_seed_point"])
def test_tieline_report_reconstructs_probe_path_and_exact_truth(source):
    truth = _Truth()
    report = evaluate_tieline_diagnostics(_surrogate(source=source), thermodynamics=truth, eta_count=5)

    assert np.allclose(report["probe_compositions"][:, 0], np.linspace(0.20, 0.30, 5))
    assert report["summary"]["max_endpoint_error"] == pytest.approx(0.0)
    assert report["summary"]["failure_count"] == 0
    assert truth.tieline_calls == 5


def test_tieline_report_accepts_reference_first_full_truth_compositions():
    report = evaluate_tieline_diagnostics(
        _surrogate(), thermodynamics=_Truth(full_tieline_compositions=True), eta_count=5
    )

    assert report["summary"]["max_endpoint_error"] == pytest.approx(0.0)
    assert np.allclose(report["truth"]["endpoint_compositions"][PHASES[0]], np.linspace(LEFT[0], LEFT[-1], 5))


def test_tieline_report_requires_probe_metadata_only_when_truth_requested():
    surrogate = _surrogate()
    surrogate.metadata = {}
    report = evaluate_tieline_diagnostics(surrogate, eta_count=3)
    assert report["probe_compositions"] is None

    with pytest.raises(ValueError, match="requires probe_start"):
        evaluate_tieline_diagnostics(surrogate, thermodynamics=_Truth(), eta_count=3)


def test_tieline_truth_failures_can_record_or_raise():
    surrogate = _surrogate()
    with pytest.raises(ValueError, match="eta="):
        evaluate_tieline_diagnostics(surrogate, thermodynamics=_Truth(fail_tieline=True), eta_count=3)

    report = evaluate_tieline_diagnostics(
        surrogate,
        thermodynamics=_Truth(fail_tieline=True),
        eta_count=3,
        on_truth_error="record",
    )
    assert report["summary"]["failure_count"] == 2
    assert np.isnan(report["truth"]["endpoint_compositions"][PHASES[0]][-1]).all()


def test_evaluation_without_truth_makes_no_backend_calls():
    truth = _Truth()
    surrogate = _surrogate()
    evaluate_tieline_diagnostics(surrogate, eta_count=3)
    evaluate_diffusivity_diagnostics(
        surrogate,
        interface_eta_count=3,
        bulk_axes=(np.asarray([0.20, 0.30]), np.asarray([0.10, 0.20])),
    )
    assert truth.tieline_calls == 0
    assert truth.diffusivity_calls == 0


def test_diffusivity_report_contains_truth_validity_and_errors():
    report = evaluate_diffusivity_diagnostics(
        _surrogate(),
        thermodynamics=_Truth(),
        interface_eta_count=5,
        bulk_axes=(np.asarray([0.20, 0.30]), np.asarray([0.10, 0.20])),
    )

    alpha_interface = report["interface"]["phases"][PHASES[0]]
    assert alpha_interface["matrices"].shape == (5, 2, 2)
    assert np.all(alpha_interface["valid"])
    assert np.allclose(alpha_interface["truth"]["matrix_relative_error"], 0.0)
    assert report["bulk"]["grid_shape"] == (2, 2)
    assert report["summary"]["failure_count"] == 0


def test_diffusivity_truth_failures_can_record():
    report = evaluate_diffusivity_diagnostics(
        _surrogate(),
        thermodynamics=_Truth(fail_diffusivity=True),
        interface_eta_count=3,
        bulk_axes=(np.asarray([0.20, 0.30]), np.asarray([0.10])),
        on_truth_error="record",
    )
    assert report["summary"]["failure_count"] > 0
    assert np.isnan(report["bulk"]["phases"][PHASES[0]]["truth"]["matrices"][-1]).all()


def test_simplex_linear_and_merged_reports_include_fallback_and_bulk_only():
    simplex = evaluate_diffusivity_diagnostics(
        _surrogate(interpolation="simplex_linear", validity_policy="legacy"),
        interface_eta_count=3,
        bulk_axes=(np.asarray([0.20, 0.45]), np.asarray([0.10, 0.30])),
    )
    assert np.any(simplex["bulk"]["phases"][PHASES[0]]["fallback"])

    merged = evaluate_diffusivity_diagnostics(
        _merged(),
        bulk_axes=(np.asarray([0.20, 0.30]), np.asarray([0.10, 0.20])),
    )
    assert merged["interface"] is None
    assert merged["phases"] == ("BETA",)


def test_plotly_figures_have_expected_subplots_hover_and_layer_buttons():
    pytest.importorskip("plotly")
    surrogate = _surrogate()
    truth = _Truth()
    tie_report = evaluate_tieline_diagnostics(surrogate, thermodynamics=truth, eta_count=5)
    diff_report = evaluate_diffusivity_diagnostics(
        surrogate,
        thermodynamics=truth,
        interface_eta_count=5,
        bulk_axes=(np.asarray([0.20, 0.30]), np.asarray([0.10, 0.20])),
    )

    tie = plot_tieline_diagnostics(tie_report, hover_fields=("phase", "eta", "x1"), display_tieline_count=3)
    interface = plot_interface_diffusivity_diagnostics(diff_report, hover_fields="minimal")
    bulk = plot_bulk_diffusivity_diagnostics(diff_report, PHASES[0], hover_fields="diagnostic")

    assert "ternary" in tie.layout
    alpha_ternary = next(trace for trace in tie.data if trace.name == "ALPHA surrogate endpoints" and trace.type == "scatterternary")
    assert alpha_ternary.a[0] == pytest.approx(LEFT[0, 1])
    assert alpha_ternary.b[0] == pytest.approx(1.0 - np.sum(LEFT[0]))
    assert alpha_ternary.c[0] == pytest.approx(LEFT[0, 0])
    assert tie.layout.ternary.aaxis.title.text == ELEMENTS[2]
    assert tie.layout.ternary.baxis.title.text == ELEMENTS[0]
    assert tie.layout.ternary.caxis.title.text == ELEMENTS[1]
    assert tie.layout.ternary.domain.y[1] < tie.layout.annotations[0].y
    assert tie.layout.width == 1250
    assert tie.layout.height == 950
    tie_truth = [trace for trace in tie.data if " truth " in f" {trace.name} " and "error" not in trace.name]
    assert tie_truth
    assert all(trace.mode == "markers" and trace.visible is None for trace in tie_truth)
    alpha_truth_index, alpha_truth = next(
        (index, trace)
        for index, trace in enumerate(tie.data)
        if trace.name == "ALPHA truth endpoints" and trace.type == "scatterternary"
    )
    alpha_surrogate_index = next(
        index
        for index, trace in enumerate(tie.data)
        if trace.name == "ALPHA surrogate endpoints" and trace.type == "scatterternary"
    )
    assert alpha_truth_index < alpha_surrogate_index
    assert alpha_truth.marker.color != alpha_ternary.line.color
    assert alpha_truth.marker.size < next(trace.marker.size for trace in tie.data if trace.name == "ALPHA training endpoints")
    assert any(trace.customdata is not None and trace.customdata.shape[1] == 3 for trace in tie.data)
    alpha_interface = next(trace for trace in interface.data if trace.name == "ALPHA surrogate")
    beta_interface = next(trace for trace in interface.data if trace.name == "BETA surrogate")
    assert alpha_interface.xaxis == beta_interface.xaxis
    assert alpha_interface.yaxis != beta_interface.yaxis
    interface_truth = [trace for trace in interface.data if trace.name in {"ALPHA truth", "BETA truth"}]
    assert interface_truth
    assert all(trace.mode == "markers" and trace.visible is None for trace in interface_truth)
    assert interface.data.index(interface_truth[0]) < interface.data.index(alpha_interface)
    assert interface_truth[0].marker.color != alpha_interface.line.color
    assert interface_truth[0].marker.size < next(trace.marker.size for trace in interface.data if trace.name == "ALPHA training")
    assert "training_distance" not in "".join(str(trace.hovertemplate) for trace in interface.data)
    assert len(bulk.layout.updatemenus[0].buttons) == 4
    assert bulk.layout.width == 1500
    prediction_maps = [trace for trace in bulk.data if trace.name == "Prediction"]
    assert len(prediction_maps) == 4
    for trace, (row, col) in zip(prediction_maps, ((1, 2), (1, 3), (1, 4), (2, 1))):
        subplot = bulk.get_subplot(row, col)
        assert trace.type == "scatterternary"
        assert np.allclose(np.asarray(trace.a) + np.asarray(trace.b) + np.asarray(trace.c), 1.0)
        assert trace.marker.showscale
        assert trace.marker.colorbar.x == pytest.approx(subplot.domain.x[1] + 0.006)
        assert trace.marker.colorbar.y == pytest.approx(0.5 * sum(subplot.domain.y))
        assert trace.marker.colorbar.len == pytest.approx(subplot.domain.y[1] - subplot.domain.y[0])
    distance_map = next(trace for trace in bulk.data if trace.name == "Training distance")
    distance_subplot = bulk.get_subplot(2, 2)
    assert distance_map.type == "scatterternary"
    assert distance_map.marker.colorbar.x == pytest.approx(distance_subplot.domain.x[1] + 0.006)
    relative_error_maps = [trace for trace in bulk.data if trace.name == "Relative Error"]
    assert len(relative_error_maps) == 4
    assert all(trace.marker.colorbar.title.text == "relative (log)" for trace in relative_error_maps)
    assert all(np.allclose(np.asarray(trace.marker.color, dtype=np.float64), -300.0) for trace in relative_error_maps)
    truth_error_map = next(trace for trace in bulk.data if trace.name == "Matrix relative error")
    assert truth_error_map.marker.colorbar.title.text == "relative (log)"
    assert np.allclose(np.asarray(truth_error_map.marker.color, dtype=np.float64), -300.0)
    assert all(trace.type == "scatterternary" for trace in bulk.data)
    assert all(
        (f"ternary{index}" if index > 1 else "ternary") in bulk.layout
        for index in range(1, 9)
    )
    assert "data" in bulk.to_plotly_json()
    assert isinstance(bulk.to_json(), str)


def test_hover_fields_reject_unknown_name():
    pytest.importorskip("plotly")
    report = evaluate_tieline_diagnostics(_surrogate(), eta_count=3)
    with pytest.raises(ValueError, match="Unsupported hover fields"):
        plot_tieline_diagnostics(report, hover_fields=("not_a_field",))


def test_bulk_diffusivity_color_scale_is_selected_per_matrix_entry():
    pytest.importorskip("plotly")
    report = evaluate_diffusivity_diagnostics(
        _surrogate(),
        bulk_axes=(np.asarray([0.20, 0.30]), np.asarray([0.10, 0.20])),
    )
    phase_report = report["bulk"]["phases"][PHASES[0]]
    phase_report["matrices"][:, 0, 1] *= -1.0

    figure = plot_bulk_diffusivity_diagnostics(report, PHASES[0], renderer=None)
    prediction_maps = [trace for trace in figure.data if trace.name == "Prediction"]
    assert len(prediction_maps) == 4
    d00, d01, d10, d11 = prediction_maps

    assert d00.marker.colorbar.title.text == "m^2/s (log)"
    assert d01.marker.colorbar.title.text == "m^2/s (symlog)"
    assert d10.marker.colorbar.title.text == "m^2/s (log)"
    assert d11.marker.colorbar.title.text == "m^2/s (log)"
    assert np.allclose(np.asarray(d00.marker.color, dtype=np.float64), np.log10(2.0))
    assert np.any(np.asarray(d01.marker.color, dtype=np.float64) < 0.0)
    assert 0.0 in np.asarray(d01.marker.colorbar.tickvals, dtype=np.float64)
    assert all(value is not None for value in d01.marker.colorbar.ticktext)


def test_plotly_helpers_default_to_browser_renderer_and_allow_preserving_existing():
    pio = pytest.importorskip("plotly.io")
    report = evaluate_tieline_diagnostics(_surrogate(), eta_count=3)
    previous = pio.renderers.default
    try:
        pio.renderers.default = "json"
        plot_tieline_diagnostics(report)
        assert pio.renderers.default == "browser"

        pio.renderers.default = "json"
        plot_tieline_diagnostics(report, renderer=None)
        assert pio.renderers.default == "json"
    finally:
        pio.renderers.default = previous


def test_plotly_import_error_keeps_evaluation_available(monkeypatch):
    report = evaluate_tieline_diagnostics(_surrogate(), eta_count=3)
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"kawin\[diagnostics\]"):
        plot_tieline_diagnostics(report)


def test_high_level_plot_returns_reports_and_merged_bulk_only():
    pytest.importorskip("plotly")
    result = plot_surrogate_diagnostics(
        _surrogate(),
        eta_count=3,
        interface_eta_count=3,
        bulk_axes=(np.asarray([0.20, 0.30]), np.asarray([0.10, 0.20])),
        display_tieline_count=2,
    )
    assert set(result["figures"]) == {"construction", "thermodynamics", "interface_diffusivity", "bulk_diffusivity"}

    merged = plot_surrogate_diagnostics(
        _merged(),
        bulk_axes=(np.asarray([0.20, 0.30]), np.asarray([0.10, 0.20])),
    )
    assert set(merged["figures"]) == {"construction", "bulk_diffusivity"}
    assert set(merged["reports"]) == {"construction", "diffusivity"}


def test_three_phase_example_helper_routes_interfaces_truth_and_merged(monkeypatch):
    from examples.ternaryExamples import IllingworthTernaryThreePhaseStable as example

    surrogate_ab = _surrogate()
    surrogate_bc = _surrogate()
    merged = _merged()
    therm_ab = object()
    therm_bc = object()
    bulk_provider = type("Provider", (), {"phase_sources": {"ALPHA": surrogate_ab, "BETA": merged}})()
    calls = []

    def diagnostic_spy(surrogate, **kwargs):
        calls.append((surrogate, kwargs))
        return {"surrogate": surrogate, "kwargs": kwargs}

    monkeypatch.setattr(example, "plot_surrogate_diagnostics", diagnostic_spy)
    result = example.plot_surrogate_diagnostics_for_run(
        {
            "surrogate_ab": surrogate_ab,
            "surrogate_bc": surrogate_bc,
            "therm_ab": therm_ab,
            "therm_bc": therm_bc,
            "bulk_thermodynamics": bulk_provider,
        },
        compare_ground_truth=True,
        hover_fields="minimal",
        eta_count=7,
    )

    assert set(result) == {"ab", "bc", "merged_bulk"}
    assert result["merged_bulk"].keys() == {"BETA"}
    assert calls[0][1]["thermodynamics"] is therm_ab
    assert calls[1][1]["thermodynamics"] is therm_bc
    assert calls[2][1]["thermodynamics"] is therm_ab
    assert all(call[1]["hover_fields"] == "minimal" for call in calls)
