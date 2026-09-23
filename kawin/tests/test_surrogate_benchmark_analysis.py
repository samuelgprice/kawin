import numpy as np
import pytest

from kawin.diffusion.SurrogateBenchmarkAnalysis import (
    _record_matrix, active_vertex_diagnostics, barycentric_coordinates,
    negative_prediction_breakdown, pointwise_level_rows, simplex_prediction,
)
from kawin.diffusion import SpectralCriteria


def test_barycentric_signed_cuberoot_reconstruction_matches_known_prediction():
    vertices = np.asarray([[0., 0.], [1., 0.], [0., 1.]])
    matrices = np.asarray([[[1., -8.], [27., 64.]], [[8., -1.], [1., 8.]], [[27., -27.], [8., 1.]]])
    point = np.asarray([.25, .5])
    weights, transformed_vertices, transformed, prediction = simplex_prediction(vertices, matrices, point)
    np.testing.assert_allclose(weights, [.25, .25, .5])
    np.testing.assert_allclose(transformed, np.tensordot(weights, transformed_vertices, axes=(0, 0)))
    np.testing.assert_allclose(prediction, transformed ** 3)


def test_barycentric_rejects_outside_simplex():
    with pytest.raises(ValueError, match="outside"):
        barycentric_coordinates([[0., 0.], [1., 0.], [0., 1.]], [1., 1.])


def test_negative_breakdown_preserves_gt_strata_and_robust_subset():
    level = np.asarray(["fine"] * 4)
    scheme = np.asarray(["kawin_simplex_linear"] * 4)
    data = {"level": level, "scheme": scheme, "prediction_failure": np.asarray(["negative", "negative", "negative", "none"]),
            "truth_classification": np.asarray(["positive", "positive", "invalid", "positive"]),
            "robust_positive": np.asarray([True, False, False, True])}
    rows = [row for row in negative_prediction_breakdown(data) if row["level"] == "fine"]
    values = {row["stratum"]: row for row in rows}
    assert values["robust_positive"]["negative_prediction_count"] == 1
    assert values["other_gt_positive"]["negative_prediction_count"] == 1
    assert values["gt_invalid"]["negative_prediction_count"] == 1


def test_active_edge_diagnostics_exclude_zero_weight_delaunay_vertex():
    vertices = np.asarray([[0., 0.], [1., 0.], [0., 1.]])
    matrices = np.asarray([np.eye(2), np.eye(2) * 2., np.eye(2) * 100.])
    diagnostics = active_vertex_diagnostics(vertices, matrices, [.5, .5, 0.], SpectralCriteria())
    assert diagnostics["active_vertex_indices"] == [0, 1]
    assert diagnostics["active_vertex_count"] == 2
    assert diagnostics["active_distance_min"] == diagnostics["active_distance_max"] == 1.
    assert diagnostics["active_matrix_entry_range"] == 1.


def test_trace_determinant_and_discriminant_diagnostics_are_exact():
    record = _record_matrix([[3., 1.], [1., 2.]], SpectralCriteria())
    assert record["trace"] == 5.
    assert record["determinant"] == pytest.approx(5.)
    assert record["discriminant"] == pytest.approx(5.)


def test_medium_independent_selection_never_uses_fine_rows():
    class Pointwise(dict):
        files = ("level", "scheme", "value")

    pointwise = Pointwise(level=np.asarray(["medium", "fine", "medium"]), scheme=np.asarray(["kawin_nearest"] * 3), value=np.asarray([1., 2., 3.]))
    selected = pointwise_level_rows(pointwise, "medium", "kawin_nearest")
    np.testing.assert_allclose(selected["value"], [1., 3.])
