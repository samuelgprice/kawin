"""Cache-only post-processing helpers for ternary diffusivity benchmark artifacts.

These helpers read completed benchmark outputs.  They never create
Thermo-Calc calculations, choose new splits, or rerun surrogate predictions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .MovingBoundarySurrogates import _signed_cuberoot
from .SurrogateBenchmark import SpectralCriteria
from .SurrogateBenchmarkExperiment import load_thermocalc_surrogate_dataset


SCHEMES = ("kawin_nearest", "kawin_simplex_linear", "idw_signed_cuberoot_p2", "kawin_simplex_positive_2x2")
LEVELS = ("coarse", "medium", "fine")
STRATA = ("robust_positive", "gt_positive", "near_boundary", "gt_invalid")


def load_benchmark_analysis(benchmark_directory, dataset_path=None):
    """Load completed benchmark artifacts and their recorded cached dataset.

    The dataset path defaults to ``manifest.json``'s source archive.  This
    validates paired pointwise alignment but does not refit any predictor.
    """
    directory = Path(benchmark_directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    source = Path(dataset_path or manifest["dataset_metadata"]["source_archive"])
    criteria = SpectralCriteria(**manifest["spectral_criteria"])
    dataset = load_thermocalc_surrogate_dataset(
        source, phase=manifest["state"]["phase"], context=manifest["state"]["context"], criteria=criteria,
        **manifest.get("duplicate_tolerances", {}),
    )
    pointwise = np.load(directory / "pointwise_results.npz")
    loo = np.load(directory / "loo_pointwise_results.npz")
    summary = list(csv.DictReader((directory / "summary.csv").open(encoding="utf-8")))
    validation_indices = np.asarray(manifest["validation_master_row_indices"], dtype=int)
    validation = dataset.subset(validation_indices)
    _validate_paired_alignment(pointwise, validation.compositions)
    return {"directory": directory, "manifest": manifest, "dataset": dataset, "criteria": criteria,
            "pointwise": pointwise, "loo": loo, "summary": summary, "validation": validation,
            "validation_indices": validation_indices}


def _validate_paired_alignment(pointwise, validation_compositions):
    """Reject artifact rows that do not preserve paired validation identities."""
    for level in LEVELS:
        for scheme in np.unique(pointwise["scheme"]):
            mask = (pointwise["level"] == level) & (pointwise["scheme"] == scheme)
            points = pointwise["composition"][mask]
            if len(points) != len(validation_compositions) or not np.allclose(points, validation_compositions, rtol=0., atol=1e-14):
                raise ValueError(f"pointwise rows for {level}/{scheme} are not aligned with manifest validation identities.")


def pointwise_level_rows(pointwise, level, scheme):
    """Return a field-preserving view of one recorded independent-holdout level."""
    if level not in LEVELS or scheme not in set(pointwise["scheme"]):
        raise ValueError("requested level or scheme is not present in the benchmark analysis contract.")
    mask = (pointwise["level"] == level) & (pointwise["scheme"] == scheme)
    return {name: pointwise[name][mask] for name in pointwise.files}


def _eigenvalues(matrices):
    """Return consistently sorted real eigenvalues for a batch of 2x2 matrices."""
    return np.sort(np.real_if_close(np.linalg.eigvals(np.asarray(matrices)), tol=1000).real, axis=-1)


def _record_matrix(matrix, criteria):
    """Return trace/determinant/discriminant and normalized spectral diagnostics."""
    matrix = np.asarray(matrix, dtype=float)
    trace = float(np.trace(matrix))
    determinant = float(np.linalg.det(matrix))
    discriminant = trace ** 2 - 4. * determinant
    scale = float(np.linalg.norm(matrix, ord=np.inf))
    eigenvalues = _eigenvalues(matrix[None])[0]
    condition = float(np.linalg.cond(matrix)) if np.all(np.isfinite(matrix)) else np.nan
    normalized_minimum = float(eigenvalues[0] / scale) if scale > 0 else np.nan
    return {"eigenvalues": eigenvalues, "trace": trace, "determinant": determinant,
            "discriminant": discriminant, "condition_number": condition,
            "normalized_minimum_eigenvalue": normalized_minimum}


def negative_prediction_breakdown(pointwise):
    """Break simplex negative predictions down by recorded GT stratum and level."""
    records = []
    for level in LEVELS:
        mask = (pointwise["level"] == level) & (pointwise["scheme"] == "kawin_simplex_linear")
        failure = pointwise["prediction_failure"][mask] == "negative"
        truth = pointwise["truth_classification"][mask]
        robust = pointwise["robust_positive"][mask]
        groups = {
            "robust_positive": robust,
            "other_gt_positive": (truth == "positive") & ~robust,
            "near_boundary": truth == "near_boundary",
            "gt_invalid": truth == "invalid",
        }
        for stratum, membership in groups.items():
            denominator = int(np.count_nonzero(membership))
            count = int(np.count_nonzero(failure & membership))
            records.append({"level": level, "stratum": stratum, "negative_prediction_count": count,
                            "stratum_count": denominator, "negative_fraction": np.nan if denominator == 0 else count / denominator})
    return records


def barycentric_coordinates(vertices, point, *, tolerance=1e-10):
    """Return triangle barycentric weights, rejecting extrapolation explicitly."""
    vertices = np.asarray(vertices, dtype=float)
    point = np.asarray(point, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("barycentric reconstruction requires a nondegenerate 3x2 simplex.")
    affine = np.vstack((vertices.T, np.ones(3)))
    weights = np.linalg.solve(affine, np.r_[point, 1.])
    if np.any(weights < -tolerance) or np.any(weights > 1. + tolerance):
        raise ValueError("point lies outside the reconstructed simplex.")
    return weights


def simplex_prediction(vertices, matrices, point):
    """Reconstruct production simplex-linear's signed-cube-root entry interpolation."""
    weights = barycentric_coordinates(vertices, point)
    transformed_vertices = _signed_cuberoot(np.asarray(matrices).reshape(3, 4)).reshape(3, 2, 2)
    transformed = np.tensordot(weights, transformed_vertices, axes=(0, 0))
    return weights, transformed_vertices, transformed, transformed ** 3


def active_vertex_diagnostics(vertices, matrices, weights, criteria, *, active_vertex_weight_tol=1e-12):
    """Summarize only vertices with material barycentric contribution.

    A Delaunay triangle can represent an edge interpolation with one zero
    weight.  Keeping all vertices remains useful for provenance, whereas these
    diagnostics avoid attributing the edge prediction to that inactive vertex.
    """
    vertices, matrices, weights = np.asarray(vertices), np.asarray(matrices), np.asarray(weights)
    active = np.flatnonzero(np.abs(weights) > active_vertex_weight_tol)
    if not len(active):
        raise ValueError("simplex reconstruction has no active barycentric vertices.")
    spectra = [_record_matrix(matrix, criteria) for matrix in matrices[active]]
    distances = np.linalg.norm(vertices[active, None] - vertices[None, active], axis=2)
    pairwise = distances[np.triu_indices(len(active), 1)]
    eigen_minima = np.asarray([item["eigenvalues"][0] for item in spectra])
    conditions = np.asarray([item["condition_number"] for item in spectra])
    return {"active_vertex_indices": active.tolist(), "active_vertex_count": int(len(active)),
            "active_barycentric_weights": weights[active].tolist(), "active_vertex_compositions": vertices[active].tolist(),
            "active_distance_min": float(np.min(pairwise)) if len(pairwise) else 0., "active_distance_max": float(np.max(pairwise)) if len(pairwise) else 0.,
            "active_lambda_min_range": float(np.ptp(eigen_minima)), "active_condition_number_range": float(np.ptp(conditions)),
            "active_matrix_entry_range": float(np.ptp(matrices[active], axis=0).max()),
            "active_vertices_positive_determinant": bool(all(item["determinant"] > 0 for item in spectra)),
            "active_vertex_diagnostics": spectra}


def fine_simplex_failures(analysis, *, active_vertex_weight_tol=1e-12):
    """Reconstruct fine robust-positive simplex failures from recorded effective support."""
    from scipy.spatial import Delaunay

    data, manifest, dataset, criteria = analysis["pointwise"], analysis["manifest"], analysis["dataset"], analysis["criteria"]
    support_indices = np.asarray(manifest["refinement_levels"]["fine"]["fit_eligible_master_row_indices"], dtype=int)
    support_points, support_matrices = dataset.compositions[support_indices], dataset.matrices[support_indices]
    triangulation = Delaunay(support_points)
    records = dataset.spectral_records(criteria)
    invalid_points = dataset.compositions[np.asarray([record.classification == "invalid" for record in records])]
    mask = ((data["level"] == "fine") & (data["scheme"] == "kawin_simplex_linear") &
            data["robust_positive"] & (data["prediction_failure"] == "negative"))
    fine_simplex_rows = np.flatnonzero((data["level"] == "fine") & (data["scheme"] == "kawin_simplex_linear"))
    output, vertices = [], []
    for row_index in np.flatnonzero(mask):
        point = data["composition"][row_index]
        simplex_index = int(triangulation.find_simplex(point))
        if simplex_index < 0:
            raise ValueError("saved fine validation failure lies outside recorded effective support.")
        local = triangulation.simplices[simplex_index]
        master = support_indices[local]
        vertex_points, vertex_matrices = support_points[local], support_matrices[local]
        weights, transformed_vertices, transformed, reconstructed = simplex_prediction(vertex_points, vertex_matrices, point)
        saved = data["predicted_matrix"][row_index]
        if not np.allclose(reconstructed, saved, rtol=1e-10, atol=1e-24):
            raise ValueError("signed-cube-root simplex reconstruction does not reproduce saved prediction.")
        truth = data["truth_matrix"][row_index]
        gt = _record_matrix(truth, criteria)
        predicted = _record_matrix(saved, criteria)
        distances = np.linalg.norm(point - invalid_points, axis=1) if len(invalid_points) else np.empty(0)
        cache_distances = np.linalg.norm(point - dataset.compositions, axis=1)
        nearest = np.argsort(cache_distances)[:6]
        nearest_spectra = [_record_matrix(dataset.matrices[index], criteria) for index in nearest]
        vertex_spectra = [_record_matrix(matrix, criteria) for matrix in vertex_matrices]
        active_diagnostic = active_vertex_diagnostics(vertex_points, vertex_matrices, weights, criteria,
                                                       active_vertex_weight_tol=active_vertex_weight_tol)
        failure_id = len(output)
        validation_local = int(np.flatnonzero(fine_simplex_rows == row_index)[0])
        output.append({"failure_id": failure_id, "master_validation_index": int(analysis["validation_indices"][validation_local]),
                       "composition_x": point[0], "composition_y": point[1], "truth_matrix": truth.tolist(), "predicted_matrix": saved.tolist(),
                       "truth_eigenvalues": gt["eigenvalues"].tolist(), "predicted_eigenvalues": predicted["eigenvalues"].tolist(),
                       "truth_trace": gt["trace"], "truth_determinant": gt["determinant"], "truth_discriminant": gt["discriminant"],
                       "predicted_trace": predicted["trace"], "predicted_determinant": predicted["determinant"], "predicted_discriminant": predicted["discriminant"],
                       "truth_condition_number": gt["condition_number"], "truth_normalized_minimum_eigenvalue": gt["normalized_minimum_eigenvalue"],
                       "frobenius_relative_error": float(data["frobenius_relative_error"][row_index]), "spectral_relative_error": float(data["spectral_relative_error"][row_index]),
                       "operator_relative_error": float(data["operator_relative_error"][row_index]),
                       "minimum_eigenvalue_absolute_error": float(data["minimum_eigenvalue_absolute_error"][row_index]),
                       "flux_relative_error": data["flux_relative_error"][row_index].tolist(), "flux_absolute_error": data["flux_absolute_error"][row_index].tolist(),
                       "nearest_invalid_distance": float(np.min(distances)) if len(distances) else np.nan,
                       "nearest_cached_master_indices": nearest.tolist(), "nearest_cached_distances": cache_distances[nearest].tolist(),
                       "nearest_cached_classifications": [records[index].classification for index in nearest],
                       "local_lambda_min_range": float(np.ptp([item["eigenvalues"][0] for item in nearest_spectra])),
                       "local_lambda_max_range": float(np.ptp([item["eigenvalues"][1] for item in nearest_spectra])),
                       "local_matrix_entry_range": float(np.ptp(dataset.matrices[nearest], axis=0).max()),
                       "simplex_area": float(abs(np.cross(vertex_points[1] - vertex_points[0], vertex_points[2] - vertex_points[0])) / 2.),
                       "barycentric_minimum": float(np.min(weights)), "barycentric_maximum": float(np.max(weights)),
                       "vertex_lambda_min_range": float(np.ptp([item["eigenvalues"][0] for item in vertex_spectra])),
                       "vertex_lambda_max_range": float(np.ptp([item["eigenvalues"][1] for item in vertex_spectra])),
                       "active_vertex_weight_tol": active_vertex_weight_tol, "predicted_negative_determinant": bool(predicted["determinant"] < 0),
                       "trace_remains_positive": bool(predicted["trace"] > 0), **active_diagnostic,
                       "reconstruction_max_abs_error": float(np.max(np.abs(reconstructed - saved)))})
        for vertex_number, (source, vertex, matrix, transformed_vertex, weight) in enumerate(zip(master, vertex_points, vertex_matrices, transformed_vertices, weights)):
            diagnostic = _record_matrix(matrix, criteria)
            vertices.append({"failure_id": failure_id, "vertex_number": vertex_number, "master_index": int(source),
                             "source_identity": dataset.metadata.get("source_rows", [])[int(source)], "composition_x": vertex[0], "composition_y": vertex[1],
                             "barycentric_weight": float(weight), "distance_to_validation": float(np.linalg.norm(point - vertex)),
                             "matrix": matrix.tolist(), "transformed_matrix": transformed_vertex.tolist(), "eigenvalues": diagnostic["eigenvalues"].tolist(),
                             "normalized_minimum_eigenvalue": diagnostic["normalized_minimum_eigenvalue"], "condition_number": diagnostic["condition_number"],
                             "trace": diagnostic["trace"], "determinant": diagnostic["determinant"], "discriminant": diagnostic["discriminant"],
                             "active": bool(vertex_number in active_diagnostic["active_vertex_indices"]),
                             "weighted_transformed_matrix": (weight * transformed_vertex).tolist()})
    return output, vertices


def matched_failure_success_summary(analysis, failures):
    """Compare failures with unique deterministic matched fine simplex successes.

    Candidates are ordered by absolute normalized-margin difference, then
    composition distance, then pointwise row identity. A success is consumed
    after matching, so the table makes any reuse impossible and auditable.
    """
    pointwise, criteria = analysis["pointwise"], analysis["criteria"]
    mask = (pointwise["level"] == "fine") & (pointwise["scheme"] == "kawin_simplex_linear") & pointwise["robust_positive"]
    candidates = np.flatnonzero(mask & (pointwise["prediction_failure"] != "negative"))
    rows = np.flatnonzero(mask)
    fine_simplex_rows = np.flatnonzero((pointwise["level"] == "fine") & (pointwise["scheme"] == "kawin_simplex_linear"))
    truth = pointwise["truth_matrix"]
    margins = np.asarray([_record_matrix(matrix, criteria)["normalized_minimum_eigenvalue"] for matrix in truth])
    support = np.asarray(analysis["manifest"]["refinement_levels"]["fine"]["fit_eligible_master_row_indices"], dtype=int)
    support_points = analysis["dataset"].compositions[support]
    invalid_points = analysis["dataset"].compositions[np.asarray([record.classification == "invalid" for record in analysis["dataset"].spectral_records(criteria)])]
    successes, matches, used = [], [], set()
    for failure in failures:
        point = np.asarray([failure["composition_x"], failure["composition_y"]])
        margin = failure["truth_normalized_minimum_eigenvalue"]
        ranking = sorted((index for index in candidates if index not in used), key=lambda index: (abs(margins[index] - margin), np.linalg.norm(pointwise["composition"][index] - point), int(index)))
        selected = int(ranking[0]); used.add(selected); successes.append(selected)
        success_point = pointwise["composition"][selected]
        matches.append({"failure_id": failure["failure_id"], "failure_master_validation_index": failure["master_validation_index"],
                        "matched_success_pointwise_index": selected,
                        "matched_success_master_validation_index": int(analysis["validation_indices"][np.flatnonzero(fine_simplex_rows == selected)[0]]),
                        "matching_margin_difference": float(abs(margins[selected] - margin)),
                        "matching_composition_distance": float(np.linalg.norm(success_point - point)),
                        "matched_success_composition": success_point.tolist(),
                        "matched_success_nearest_support_distance": float(np.min(np.linalg.norm(success_point - support_points, axis=1))),
                        "matched_success_nearest_invalid_distance": float(np.min(np.linalg.norm(success_point - invalid_points, axis=1)))})
    def summarize(indices):
        indices = np.asarray(indices, dtype=int)
        values = {"lambda_min_gt": _eigenvalues(truth[indices])[:, 0], "normalized_spectral_margin": margins[indices], "condition_number": np.asarray([_record_matrix(truth[index], criteria)["condition_number"] for index in indices]),
                  "frobenius_error": pointwise["frobenius_relative_error"][indices], "minimum_eigenvalue_error": pointwise["minimum_eigenvalue_absolute_error"][indices]}
        return {name: {"median": float(np.nanmedian(value)), "p95": float(np.nanpercentile(value, 95))} for name, value in values.items()}
    failure_indices = [index for index in rows if pointwise["prediction_failure"][index] == "negative"]
    return {"matching_rule": "unique nearest normalized spectral margin, then composition distance, then pointwise identity",
            "failure_count": len(failure_indices), "matched_success_count": len(successes), "success_reuse_count": len(successes) - len(set(successes)),
            "failure": summarize(failure_indices), "matched_success": summarize(successes), "matches": matches}
