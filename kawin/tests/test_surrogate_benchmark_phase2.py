import numpy as np
import pytest

from kawin.diffusion import (
    DiffusivityDataset,
    IllingworthRun,
    SurrogateScheme,
    attach_flux_metrics,
    capture_illingworth_run,
    compare_illingworth_runs,
    flux_error,
    make_loo_plan,
    run_illingworth_benchmark,
    run_paired_benchmark,
    run_refinement_study,
)


POINTS = np.asarray([[0.1, 0.1], [0.3, 0.1], [0.1, 0.3], [0.3, 0.3], [0.2, 0.2]])
MATRIX = np.asarray([[2.0, 0.2], [0.1, 1.0]])


class _ConstantPredictor:
    def predict(self, compositions, *, phase, context, temperature):
        return np.broadcast_to(MATRIX, (len(compositions), 2, 2)).copy()


def _dataset():
    return DiffusivityDataset(POINTS, np.broadcast_to(MATRIX, (len(POINTS), 2, 2)).copy(),
                              phases="ALPHA", temperatures=1000.0)


def test_flux_error_and_optional_attachment_use_the_diffusivity_action():
    result = flux_error(2.0 * np.eye(2), np.eye(2), [[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(result["gt_flux"], [[-1.0, 0.0], [0.0, -2.0]])
    np.testing.assert_allclose(result["relative_error"], [1.0, 1.0])
    benchmark = run_paired_benchmark(make_loo_plan(_dataset()), [SurrogateScheme("constant", lambda training: _ConstantPredictor())])
    attached = attach_flux_metrics(benchmark, np.asarray([1.0, 0.0]))
    assert all(np.allclose(row["flux_relative_error"], [0.0]) for row in attached["constant"]["rows"])
    summary = attached["constant"]["flux_summary"]["overall"]
    assert summary["absolute_flux_error"]["maximum"] == pytest.approx(0.0)
    assert summary["evaluated_gradient_count"] == len(POINTS)
    assert summary["low_signal_count"] == 0


def test_flux_error_keeps_gt_low_signal_status_when_prediction_is_nonfinite():
    result = flux_error(np.asarray([[np.nan, 0.0], [0.0, 1.0]]), np.eye(2), [1.0, 0.0])
    np.testing.assert_allclose(result["gt_flux"], [[-1.0, 0.0]])
    assert not result["low_signal"][0]
    assert np.isnan(result["predicted_flux"]).all()
    assert np.isnan(result["absolute_error"][0])
    assert np.isnan(result["relative_error"][0])
    unavailable = flux_error(np.eye(2), np.asarray([[np.nan, 0.0], [0.0, 1.0]]), [1.0, 0.0])
    assert unavailable["low_signal"][0] is None


def test_refinement_study_labels_an_independent_reference_without_promoting_fine_data_to_truth():
    source = _dataset()
    validation = DiffusivityDataset(np.asarray([[0.15, 0.15]]), np.asarray([MATRIX]), phases="ALPHA", temperatures=1000.0)
    report = run_refinement_study(source, validation, [SurrogateScheme("constant", lambda training: _ConstantPredictor())],
                                  {"coarse": [0, 1, 2], "fine": [0, 1, 2, 3, 4]})
    assert report["kind"] == "refinement_comparison"
    assert report["reference"] == "independent_ground_truth"
    assert [level["training_sample_count"] for level in report["levels"]] == [3, 5]


def test_illingworth_run_comparison_reports_interface_and_profile_errors():
    x = np.asarray([0.0, 1.0])
    reference = IllingworthRun([0.0, 1.0, 2.0], [0.5, 0.6, 0.7], np.zeros((3, 2, 2)), x, "reference")
    candidate = IllingworthRun([0.0, 1.0, 2.0], [0.5, 0.65, 0.8], np.ones((3, 2, 2)), x, "candidate")
    report = compare_illingworth_runs(reference, candidate)
    assert report["completed"]
    assert report["reference_reached_target_time"]
    assert report["candidate_reached_target_time"]
    assert report["maximum_interface_position_error"] == pytest.approx(0.1)
    assert report["final_interface_position_error"] == pytest.approx(0.1)
    assert report["composition_error"]["final_absolute_l2_profile_error"] == pytest.approx(np.sqrt(2.0))


def test_illingworth_capture_and_runner_preserve_candidate_failures():
    class History:
        N = 1
        _time = np.asarray([0.0, 1.0])
        _y = np.asarray([0.5, 0.6])

    class Data:
        _y = np.zeros((2, 3, 2))

    class Model:
        interfaceData = History()
        data = Data()

    Model._z = np.asarray([0.0, 0.5, 1.0])
    captured = capture_illingworth_run(Model(), label="fake")
    assert captured.compositions.shape == (2, 3, 2)
    result = run_illingworth_benchmark(lambda: captured, {"ok": lambda: captured, "bad": lambda: (_ for _ in ()).throw(RuntimeError("failed"))})
    assert result["results"]["ok"]["completed"]
    assert not result["results"]["bad"]["completed"]


def test_different_time_grids_and_low_signal_flux_are_handled_without_extrapolation():
    x = np.asarray([0.0, 1.0])
    reference = IllingworthRun([0.0, 1.0, 2.0], [0.5, 0.6, 0.7], np.zeros((3, 2, 2)), x)
    candidate = IllingworthRun([0.0, 0.5, 1.5, 2.0], [0.5, 0.55, 0.65, 0.7], np.zeros((4, 2, 2)), x)
    comparison = compare_illingworth_runs(reference, candidate)
    assert comparison["completed"] and comparison["maximum_absolute_interface_position_error"] == pytest.approx(0.0)
    assert comparison["composition_error"]["final_normalized_l2_profile_error"] is None
    assert comparison["composition_error"]["final_normalized_l2_profile_error_reason"] == "reference composition norm is effectively zero"
    short = IllingworthRun([0.0, 0.5], [0.5, 0.55], np.zeros((2, 2, 2)), x)
    short_report = compare_illingworth_runs(reference, short)
    assert not short_report["reached_target_time"]
    assert not short_report["candidate_reached_target_time"]
    flux = flux_error(np.eye(2), np.eye(2), [0.0, 0.0], gt_flux_floor=1e-12)
    assert flux["low_signal"][0] and np.isnan(flux["relative_error"][0])


def test_illingworth_comparison_rejects_spatial_extrapolation_and_reports_target_coverage():
    reference = IllingworthRun(
        [0.0, 1.0, 2.0], [0.5, 0.6, 0.7], np.zeros((3, 2, 2)), [0.0, 1.0]
    )
    incompatible = IllingworthRun(
        [0.0, 1.0], [0.5, 0.6], np.zeros((2, 2, 2)), [0.1, 1.1]
    )
    report = compare_illingworth_runs(reference, incompatible, target_time=3.0)
    assert not report["completed"]
    assert not report["reference_reached_target_time"]
    assert not report["candidate_reached_target_time"]
    assert report["composition_error"] is None
    assert "spatial extrapolation" in report["metric_reason"]


def test_illingworth_profile_comparison_interpolates_matching_physical_domains():
    reference_x = np.asarray([0.0, 0.5, 1.0])
    candidate_x = np.asarray([0.0, 0.2, 0.8, 1.0])

    def linear_profile(x):
        return np.column_stack([0.25 + 2.0 * x, 1.0 - 0.5 * x])

    reference = IllingworthRun(
        [0.0, 1.0], [0.4, 0.5], np.asarray([linear_profile(reference_x)] * 2), reference_x
    )
    candidate = IllingworthRun(
        [0.0, 1.0], [0.4, 0.5], np.asarray([linear_profile(candidate_x)] * 2), candidate_x
    )
    report = compare_illingworth_runs(reference, candidate)
    assert report["composition_error"]["final_absolute_l2_profile_error"] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"compositions": np.asarray([[[np.nan, 0.0]], [[0.0, 0.0]]]), "spatial_coordinates": [0.0]},
        {"compositions": np.zeros((2, 1, 2)), "spatial_coordinates": [np.nan]},
    ],
)
def test_illingworth_run_rejects_nonfinite_profile_fields(kwargs):
    with pytest.raises(ValueError):
        IllingworthRun([0.0, 1.0], [0.5, 0.6], **kwargs)


def test_capture_and_compare_real_two_phase_ternary_illingworth_history():
    from kawin.tests.test_illingworth_ternary import _make_residual_identity_state

    model, _, _ = _make_residual_identity_state()
    model.solve(2.0e-4, minDtFrac=1.0e-10)
    run = capture_illingworth_run(model, label="real")
    assert run.compositions.shape[0] == len(run.times)
    assert run.spatial_coordinates.shape[1] == run.compositions.shape[1]
    comparison = compare_illingworth_runs(run, run)
    assert comparison["completed"]
    assert comparison["maximum_absolute_interface_position_error"] == pytest.approx(0.0)
