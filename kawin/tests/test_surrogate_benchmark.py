import numpy as np
import pytest
from pathlib import Path
from uuid import uuid4

from kawin.diffusion import (
    DiffusivityDataset,
    SpectralCriteria,
    SurrogateScheme,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    ThermodynamicsPredictor,
    build_neighborhoods,
    classify_matrix,
    eligible_training_mask,
    make_blocked_plan,
    make_independent_holdout_plan,
    make_loo_plan,
    pointwise_metrics,
    robust_positive_mask,
    run_paired_benchmark,
)


POINTS = np.asarray([
    [0.10, 0.10], [0.30, 0.10], [0.10, 0.30], [0.30, 0.30], [0.20, 0.20],
])
POSITIVE = np.asarray([[2.0, 0.2], [0.1, 1.0]])


def _dataset(matrices=None):
    if matrices is None:
        matrices = np.broadcast_to(POSITIVE, (len(POINTS), 2, 2)).copy()
    return DiffusivityDataset(POINTS, matrices, phases="ALPHA", temperatures=1000.0,
                              metadata={"source": "synthetic"})


def test_spectral_classification_detects_positive_negative_near_zero_repeated_and_complex():
    criteria = SpectralCriteria(positive_margin=1e-8, near_zero_margin=1e-6)
    assert classify_matrix(np.diag([2.0, 1.0]), criteria).classification == "positive"
    assert classify_matrix(np.diag([2.0, -1.0]), criteria).classification == "invalid"
    assert classify_matrix(np.diag([2.0, 1e-12]), criteria).classification == "near_boundary"
    assert classify_matrix(np.eye(2), criteria).classification == "positive"
    assert classify_matrix(np.asarray([[0.0, -1.0], [1.0, 0.0]]), criteria).classification == "invalid"


def test_pointwise_metrics_match_simple_diagonal_case_and_skip_ill_conditioned_operator():
    truth = np.diag([2.0, 1.0])
    predicted = np.diag([4.0, 1.0])
    metrics = pointwise_metrics(predicted, truth)
    assert metrics["frobenius_relative_error"] == pytest.approx(2.0 / np.sqrt(5.0))
    assert metrics["spectral_relative_error"] == pytest.approx(1.0)
    assert metrics["operator_relative_error"] == pytest.approx(1.0)
    np.testing.assert_allclose(metrics["log_eigenvalue_error"], [0.0, np.log(2.0)])
    singularish = pointwise_metrics(predicted, np.diag([1.0, 1e-15]))
    assert not singularish["operator_error_reliable"]
    assert singularish["operator_relative_error"] is None


def test_eigenvalue_errors_preserve_signed_bias_and_absolute_accuracy_for_real_negative_predictions():
    metrics = pointwise_metrics(np.diag([-0.1, 1.5]), np.diag([1.0, 2.0]))
    np.testing.assert_allclose(metrics["eigenvalue_signed_error"], [-1.1, -0.5])
    np.testing.assert_allclose(metrics["eigenvalue_absolute_error"], [1.1, 0.5])
    np.testing.assert_allclose(metrics["eigenvalue_relative_error"], [1.1, 0.25])
    assert metrics["minimum_eigenvalue_signed_error"] == pytest.approx(-1.1)
    assert metrics["minimum_eigenvalue_absolute_error"] == pytest.approx(1.1)
    assert metrics["log_eigenvalue_error"] is None

    complex_metrics = pointwise_metrics(np.asarray([[0.0, -1.0], [1.0, 0.0]]), np.diag([1.0, 2.0]))
    assert complex_metrics["eigenvalue_signed_error"] is None
    assert complex_metrics["eigenvalue_absolute_error"] is None


def test_dataset_round_trip_is_immutable_and_retains_invalid_gt():
    matrices = np.broadcast_to(POSITIVE, (len(POINTS), 2, 2)).copy()
    matrices[2] = np.diag([1.0, -1.0])
    dataset = _dataset(matrices)
    path = Path.cwd() / f"surrogate_benchmark_{uuid4().hex}.npz"
    try:
        path = dataset.save(path)
        loaded = DiffusivityDataset.load(path)
        np.testing.assert_array_equal(loaded.matrices, matrices)
        assert not loaded.matrices.flags.writeable
        assert loaded.metadata == {"source": "synthetic"}
        assert eligible_training_mask(loaded).tolist() == [True, True, False, True, True]
    finally:
        path.unlink(missing_ok=True)


def test_delaunay_robust_positive_and_blocked_splits_are_deterministic():
    matrices = np.broadcast_to(POSITIVE, (len(POINTS), 2, 2)).copy()
    matrices[0] = np.diag([1.0, -1.0])
    dataset = _dataset(matrices)
    adjacency = build_neighborhoods(dataset)
    assert len(adjacency) == len(dataset)
    robust = robust_positive_mask(dataset, neighborhoods=adjacency)
    assert not robust[0]
    assert not robust[4]  # Center point shares a Delaunay triangle with index 0.
    first = make_blocked_plan(dataset, neighborhoods=adjacency)
    second = make_blocked_plan(dataset, neighborhoods=adjacency)
    assert [split.name for split in first.splits] == [split.name for split in second.splits]
    for left, right in zip(first.splits, second.splits):
        np.testing.assert_array_equal(left.training_indices, right.training_indices)
        np.testing.assert_array_equal(left.validation_indices, right.validation_indices)
        assert not np.intersect1d(left.training_indices, left.validation_indices).size


class _ConstantPredictor:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float64)

    def predict(self, compositions, *, phase, context, temperature):
        return np.broadcast_to(self.matrix, (len(compositions), 2, 2)).copy()


def _constant_scheme(name, matrix):
    return SurrogateScheme(name, lambda training: _ConstantPredictor(matrix))


def test_loo_excludes_invalid_gt_from_fit_but_records_invalid_prediction_on_robust_point():
    matrices = np.broadcast_to(POSITIVE, (len(POINTS), 2, 2)).copy()
    matrices[0] = np.diag([1.0, -1.0])
    dataset = _dataset(matrices)
    plan = make_loo_plan(dataset)
    assert all(0 not in split.training_indices for split in plan.splits)
    report = run_paired_benchmark(plan, [_constant_scheme("bad", np.diag([1.0, -1.0]))])["bad"]
    assert report["summary"]["negative_prediction_count"] == len(plan.splits)
    assert report["summary"]["robust_positive_invalid_prediction_rate"] == pytest.approx(1.0)


def test_paired_schemes_share_exact_validation_rows_and_independent_holdout():
    training = _dataset()
    validation = DiffusivityDataset(np.asarray([[0.15, 0.15], [0.25, 0.15]]),
                                    np.broadcast_to(POSITIVE, (2, 2, 2)).copy(),
                                    phases="ALPHA", temperatures=1000.0)
    plan = make_independent_holdout_plan(training, validation)
    result = run_paired_benchmark(plan, [_constant_scheme("one", POSITIVE), _constant_scheme("two", POSITIVE)])
    assert [row["dataset_index"] for row in result["one"]["rows"]] == [0, 1]
    assert [row["dataset_index"] for row in result["one"]["rows"]] == [row["dataset_index"] for row in result["two"]["rows"]]
    assert result["one"]["summary"]["frobenius_relative_error"]["maximum"] == pytest.approx(0.0)


def test_thermodynamics_predictor_uses_existing_get_interdiffusivity_api():
    class Source:
        def getInterdiffusivity(self, composition, temperature, phase=None, query_context=None):
            assert phase == "ALPHA"
            assert temperature == 1000.0
            assert query_context == "general"
            return POSITIVE

    values = ThermodynamicsPredictor(Source()).predict(POINTS[:2], phase="ALPHA", context="general", temperature=1000.0)
    np.testing.assert_array_equal(values, np.broadcast_to(POSITIVE, (2, 2, 2)))


def test_low_margin_warning_is_not_spectral_invalid_for_an_exact_positive_prediction():
    criteria = SpectralCriteria(positive_margin=1e-4, near_zero_margin=1e-2)
    matrix = np.diag([1.0, 5e-4])
    dataset = DiffusivityDataset(POINTS, np.broadcast_to(matrix, (len(POINTS), 2, 2)).copy(),
                                 phases="ALPHA", temperatures=1000.0)
    plan = make_loo_plan(dataset, criteria)
    report = run_paired_benchmark(plan, [_constant_scheme("exact", matrix)], criteria)["exact"]
    assert np.isnan(report["summary"]["robust_positive_spectral_invalid_rate"])
    assert report["summary"]["negative_prediction_count"] == 0
    assert report["summary"]["low_margin_prediction_count"] == len(plan.splits)
    assert all(row["prediction_failure"] is None and row["prediction_low_margin_warning"] for row in report["rows"])


def test_robust_positive_margin_is_stricter_than_fitting_eligibility_for_points_and_neighbors():
    criteria = SpectralCriteria(positive_margin=1e-4, near_zero_margin=1e-2, robust_positive_margin=5e-3)
    low = np.diag([1.0, 1e-3])
    high = np.diag([1.0, 1e-2])
    matrices = np.asarray([low, high, high, high, high])
    adjacency = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2], []]
    mask = robust_positive_mask(_dataset(matrices), criteria, neighborhoods=adjacency)
    assert eligible_training_mask(_dataset(matrices), criteria)[0]
    assert not mask[0]
    matrices[1] = low
    assert not robust_positive_mask(_dataset(matrices), criteria, neighborhoods=adjacency)[0]


def test_robust_positive_requires_enough_same_state_neighbors_and_rejects_degenerate_geometry():
    isolated = DiffusivityDataset(POINTS[:2], np.broadcast_to(POSITIVE, (2, 2, 2)).copy(), phases="ALPHA")
    assert not np.any(robust_positive_mask(isolated))
    collinear = DiffusivityDataset(np.asarray([[0.1, 0.1], [0.2, 0.1], [0.3, 0.1]]),
                                  np.broadcast_to(POSITIVE, (3, 2, 2)).copy(), phases="ALPHA")
    assert not np.any(robust_positive_mask(collinear))
    adjacency = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2], []]
    assert robust_positive_mask(_dataset(), neighborhoods=adjacency).tolist() == [True, True, True, True, False]
    negative = np.broadcast_to(POSITIVE, (len(POINTS), 2, 2)).copy()
    negative[1] = np.diag([1.0, -1.0])
    assert not robust_positive_mask(_dataset(negative), neighborhoods=adjacency)[0]


def test_independent_holdout_robustness_comes_from_enclosing_training_simplex_not_validation_adjacency():
    training = DiffusivityDataset(np.asarray([[0.1, 0.1], [0.5, 0.1], [0.1, 0.5]]),
                                  np.broadcast_to(POSITIVE, (3, 2, 2)).copy(), phases="ALPHA", temperatures=1000.0)
    validation = DiffusivityDataset(np.asarray([[0.2, 0.2]]), np.asarray([POSITIVE]), phases="ALPHA", temperatures=1000.0)
    plan = make_independent_holdout_plan(training, validation)
    assert plan.validation_robust_positive_mask.tolist() == [True]
    outside = DiffusivityDataset(np.asarray([[0.7, 0.2]]), np.asarray([POSITIVE]), phases="ALPHA", temperatures=1000.0)
    assert not make_independent_holdout_plan(training, outside).validation_robust_positive_mask[0]


def test_independent_holdout_rejects_unmatched_physical_states():
    training = _dataset()
    validation = DiffusivityDataset(np.asarray([[0.2, 0.2]]), np.asarray([POSITIVE]), phases="BETA", temperatures=1000.0)
    with pytest.raises(ValueError, match="no eligible training samples"):
        make_independent_holdout_plan(training, validation)


def test_eigenvalue_summary_uses_absolute_minimum_error_magnitudes():
    dataset = _dataset()
    report = run_paired_benchmark(make_loo_plan(dataset), [_constant_scheme("under", np.diag([0.5, 1.0]))])["under"]
    summary = report["summary"]["minimum_eigenvalue_absolute_error"]
    assert summary["median"] > 0.0
    assert summary["maximum"] > 0.0


def test_robust_positive_unusable_rate_includes_evaluator_failures_but_not_spectral_invalidity():
    def failing_factory(training):
        raise RuntimeError("synthetic predictor failure")

    summary = run_paired_benchmark(make_loo_plan(_dataset()), [SurrogateScheme("fail", failing_factory)])["fail"]["summary"]
    assert summary["robust_positive_unusable_rate"] == pytest.approx(1.0)
    assert summary["robust_positive_spectral_invalid_rate"] == pytest.approx(0.0)


def test_state_groups_prevent_cross_context_and_temperature_training_or_neighbors():
    points = np.asarray([[0.1, 0.1], [0.3, 0.1], [0.1, 0.3], [0.1, 0.1], [0.3, 0.1], [0.1, 0.3]])
    dataset = DiffusivityDataset(points, np.broadcast_to(POSITIVE, (6, 2, 2)).copy(), phases="ALPHA",
                                 contexts=["general", "general", "general", "interface", "interface", "interface"],
                                 temperatures=[1000.0, 1000.0, 1000.0, 1100.0, 1100.0, 1100.0])
    neighborhoods = build_neighborhoods(dataset)
    assert all(neighbor < 3 for neighbor in neighborhoods[0])
    plan = make_loo_plan(dataset)
    first = next(split for split in plan.splits if split.validation_indices[0] == 0)
    assert first.context == "general" and first.temperature == 1000.0
    assert first.training_indices.tolist() == [1, 2]

    temperature_split = DiffusivityDataset(points[:4], np.broadcast_to(POSITIVE, (4, 2, 2)).copy(), phases="ALPHA",
                                           contexts="general", temperatures=[1000.0, 1000.0, 1100.0, 1100.0])
    first_temperature_split = next(split for split in make_loo_plan(temperature_split).splits if split.validation_indices[0] == 0)
    assert first_temperature_split.training_indices.tolist() == [1]


def test_sampling_counts_and_strata_are_not_summed_across_fits():
    matrices = np.broadcast_to(POSITIVE, (len(POINTS), 2, 2)).copy()
    matrices[0] = np.diag([1.0, 1e-16])
    matrices[1] = np.diag([1.0, -1.0])
    dataset = _dataset(matrices)
    plan = make_loo_plan(dataset)
    summary = run_paired_benchmark(plan, [_constant_scheme("constant", POSITIVE)])["constant"]["summary"]
    assert summary["gt_sample_count"] == len(dataset)
    assert summary["eligible_training_pool_count"] == 3
    assert summary["per_fit_training_size"]["minimum"] == 2
    assert summary["per_fit_training_size"]["maximum"] == 2
    assert summary["strata"]["gt_positive"]["frobenius_relative_error"]["count"] == 3
    assert summary["strata"]["near_boundary"]["frobenius_relative_error"]["count"] == 0
    assert summary["strata"]["gt_invalid"]["frobenius_relative_error"]["count"] == 0


def test_cached_dataset_runs_through_actual_simplex_linear_surrogate():
    matrices = np.asarray([POSITIVE * (1.0 + 0.05 * x + 0.03 * y) for x, y in POINTS])
    dataset = _dataset(matrices)

    def factory(training):
        interface = np.asarray([[0.12, 0.12], [0.16, 0.12]])
        surrogate = TernaryMovingBoundaryThermodynamicsSurrogate(
            elements=("Z", "X", "Y"), phases=("ALPHA", "BETA"), tieline_phases=("ALPHA", "BETA"),
            temperature=1000.0, eta_samples=np.asarray([0.0, 1.0]),
            tieline_compositions={"ALPHA": interface, "BETA": interface + np.asarray([0.1, 0.0])},
            diffusivity_compositions={"interface": {"ALPHA": interface, "BETA": interface},
                                      "general": {"ALPHA": training.compositions, "BETA": training.compositions}},
            diffusivities={"interface": {"ALPHA": np.asarray([POSITIVE, POSITIVE]), "BETA": np.asarray([POSITIVE, POSITIVE])},
                           "general": {"ALPHA": training.matrices, "BETA": training.matrices}},
            diffusivity_interpolation="simplex_linear",
        )
        return ThermodynamicsPredictor(surrogate)

    report = run_paired_benchmark(make_loo_plan(dataset), [SurrogateScheme("simplex", factory)])["simplex"]
    assert not report["failures"]
    assert len(report["rows"]) == len(dataset)
