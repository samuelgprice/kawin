"""Numerically compare two verified ternary surrogate artifact bundles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .SurrogateArtifacts import SurrogateArtifactBundle


_TOLERANCE_KEYS = {
    "endpoint_composition_max",
    "interface_matrix_symmetric_relative_max",
    "bulk_matrix_symmetric_relative_max",
    "minimum_bulk_shared_fraction",
    "minimum_interface_cross_fraction",
}


def _positive_count(value, name):
    value = int(value)
    if value < 2:
        raise ValueError(f"{name} must be at least 2.")
    return value


def _build_spec_differences(reference, candidate, path="build_spec"):
    differences = []
    if isinstance(reference, dict) and isinstance(candidate, dict):
        for key in sorted(set(reference) | set(candidate)):
            child = f"{path}.{key}"
            if key not in reference:
                differences.append({"path": child, "reference": None, "candidate": candidate[key]})
            elif key not in candidate:
                differences.append({"path": child, "reference": reference[key], "candidate": None})
            else:
                differences.extend(_build_spec_differences(reference[key], candidate[key], child))
        return differences
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            differences.append({"path": path, "reference": reference, "candidate": candidate})
        else:
            for index, (left, right) in enumerate(zip(reference, candidate)):
                differences.extend(_build_spec_differences(left, right, f"{path}[{index}]"))
        return differences
    if reference != candidate:
        differences.append({"path": path, "reference": reference, "candidate": candidate})
    return differences


def _normalized_eta(surrogate, normalized):
    lower, upper = surrogate.eta_bounds
    return lower + normalized * (upper - lower)


def _interface_compositions(surrogate, eta):
    values = {phase: [] for phase in surrogate.tieline_phases}
    for eta_value in eta:
        for phase, composition in zip(surrogate.tieline_phases, surrogate.interface_compositions(eta_value)):
            values[phase].append(composition)
    return {phase: np.asarray(points, dtype=np.float64) for phase, points in values.items()}


def _tie_geometry(first, second):
    vector = np.asarray(second) - np.asarray(first)
    return np.linalg.norm(vector, axis=1), np.degrees(np.arctan2(vector[:, 1], vector[:, 0]))


def _angle_difference(first, second):
    return np.abs((np.asarray(second) - np.asarray(first) + 180.0) % 360.0 - 180.0)


def _evaluate_matrices(surrogate, points, phase, context):
    matrices = np.full((len(points), 2, 2), np.nan, dtype=np.float64)
    success = np.zeros(len(points), dtype=bool)
    errors = np.full(len(points), "", dtype=object)
    for index, point in enumerate(np.asarray(points, dtype=np.float64)):
        try:
            matrix = np.asarray(
                surrogate.getInterdiffusivity(
                    point, surrogate.temperature, phase=phase, query_context=context
                ),
                dtype=np.float64,
            )
            if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"expected a finite 2x2 matrix, received shape {matrix.shape}")
            matrices[index] = matrix
            success[index] = True
        except Exception as exc:
            errors[index] = f"{type(exc).__name__}: {exc}"
    return {"matrices": matrices, "success": success, "errors": errors}


def _eigenvalues(matrices):
    output = np.full((len(matrices), 2), np.nan + 0j, dtype=np.complex128)
    for index, matrix in enumerate(matrices):
        if np.all(np.isfinite(matrix)):
            output[index] = np.linalg.eigvals(matrix)
    return output


def _distribution(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else np.nan,
        "median": float(np.median(finite)) if finite.size else np.nan,
        "p95": float(np.percentile(finite, 95.0)) if finite.size else np.nan,
        "maximum": float(np.max(finite)) if finite.size else np.nan,
    }


def _matrix_comparison(reference, candidate, reference_success, candidate_success, floor):
    shared = np.asarray(reference_success, dtype=bool) & np.asarray(candidate_success, dtype=bool)
    difference = np.full_like(reference, np.nan)
    absolute = np.full_like(reference, np.nan)
    reference_relative = np.full(len(reference), np.nan)
    symmetric_relative = np.full(len(reference), np.nan)
    if np.any(shared):
        difference[shared] = candidate[shared] - reference[shared]
        absolute[shared] = np.abs(difference[shared])
        delta_norm = np.linalg.norm(difference[shared], axis=(1, 2))
        reference_norm = np.linalg.norm(reference[shared], axis=(1, 2))
        candidate_norm = np.linalg.norm(candidate[shared], axis=(1, 2))
        reference_relative[shared] = delta_norm / np.maximum(reference_norm, floor)
        symmetric_relative[shared] = delta_norm / np.maximum(
            np.maximum(reference_norm, candidate_norm), floor
        )
    return {
        "reference_matrices": np.asarray(reference, dtype=np.float64),
        "candidate_matrices": np.asarray(candidate, dtype=np.float64),
        "difference_matrices": difference,
        "absolute_component_difference": absolute,
        "reference_eigenvalues": _eigenvalues(reference),
        "candidate_eigenvalues": _eigenvalues(candidate),
        "reference_success": np.asarray(reference_success, dtype=bool),
        "candidate_success": np.asarray(candidate_success, dtype=bool),
        "shared": shared,
        "reference_relative_frobenius_error": reference_relative,
        "symmetric_relative_frobenius_error": symmetric_relative,
        "summary": {
            "sample_count": int(len(reference)),
            "shared_count": int(np.count_nonzero(shared)),
            "shared_fraction": float(np.mean(shared)) if len(shared) else 0.0,
            "reference_relative_frobenius_error": _distribution(reference_relative),
            "symmetric_relative_frobenius_error": _distribution(symmetric_relative),
        },
    }


def _support_points(surrogate, phase):
    support = surrogate._fitSupport[phase]
    return np.asarray(support["points"], dtype=np.float64)


def _point_matches(points, targets, atol=1e-12):
    if not len(points) or not len(targets):
        return np.zeros(len(points), dtype=bool)
    return np.any(
        np.all(np.abs(points[:, None, :] - targets[None, :, :]) <= atol, axis=2), axis=1
    )


def _bulk_points(reference, candidate, phase, grid_counts, bulk_axes):
    reference_training = _support_points(reference, phase)
    candidate_training = _support_points(candidate, phase)
    if not len(reference_training) or not len(candidate_training):
        grid = np.empty((0, 2), dtype=np.float64)
    else:
        lower = np.maximum(np.min(reference_training, axis=0), np.min(candidate_training, axis=0))
        upper = np.minimum(np.max(reference_training, axis=0), np.max(candidate_training, axis=0))
        if bulk_axes is None:
            if np.any(upper < lower):
                grid = np.empty((0, 2), dtype=np.float64)
            else:
                axes = tuple(np.linspace(lower[i], upper[i], grid_counts[i]) for i in range(2))
                x0, x1 = np.meshgrid(*axes, indexing="ij")
                grid = np.column_stack((x0.ravel(), x1.ravel()))
        else:
            axes = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in bulk_axes)
            if len(axes) != 2 or any(axis.size == 0 for axis in axes):
                raise ValueError("bulk_axes must contain two nonempty one-dimensional arrays.")
            if any(not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0) for axis in axes):
                raise ValueError("bulk_axes must be finite and strictly increasing.")
            x0, x1 = np.meshgrid(*axes, indexing="ij")
            grid = np.column_stack((x0.ravel(), x1.ravel()))
            grid = grid[np.all((grid >= lower) & (grid <= upper), axis=1)]
    minimum = max(float(reference.min_composition), float(candidate.min_composition))
    if len(grid):
        grid = grid[
            (grid[:, 0] >= minimum)
            & (grid[:, 1] >= minimum)
            & (np.sum(grid, axis=1) <= 1.0 - minimum)
        ]
    combined = np.vstack((grid, reference_training, candidate_training))
    if len(combined):
        combined = combined[
            (combined[:, 0] >= minimum)
            & (combined[:, 1] >= minimum)
            & (np.sum(combined, axis=1) <= 1.0 - minimum)
        ]
    points = np.unique(combined, axis=0) if len(combined) else np.empty((0, 2), dtype=np.float64)
    source = np.full(len(points), "grid", dtype=object)
    in_reference = _point_matches(points, reference_training)
    in_candidate = _point_matches(points, candidate_training)
    source[in_reference] = "reference training"
    source[in_candidate] = "candidate training"
    source[in_reference & in_candidate] = "both training"
    training_on_grid = _point_matches(points, grid) & (in_reference | in_candidate)
    source[training_on_grid] = np.asarray(
        [f"{value} + grid" for value in source[training_on_grid]], dtype=object
    )
    return points, source, reference_training, candidate_training


def _validate_identity(reference_bundle, candidate_bundle):
    reference = reference_bundle.surrogate
    candidate = candidate_bundle.surrogate
    if tuple(reference.elements) != tuple(candidate.elements):
        raise ValueError("Cannot compare surrogates with different element order.")
    if tuple(reference.tieline_phases) != tuple(candidate.tieline_phases):
        raise ValueError("Cannot compare surrogates with different tie-line phases or phase order.")
    if tuple(reference.phases) != tuple(candidate.phases):
        raise ValueError("Cannot compare surrogates with different configured phases or phase order.")
    if not np.isclose(reference.temperature, candidate.temperature, rtol=0.0, atol=1e-8):
        raise ValueError("Cannot compare surrogates at different temperatures.")


def _validate_tolerances(tolerances):
    if tolerances is None:
        return None
    if not isinstance(tolerances, Mapping):
        raise TypeError("tolerances must be a mapping or None.")
    unknown = sorted(set(tolerances) - _TOLERANCE_KEYS)
    if unknown:
        raise ValueError(f"Unknown surrogate comparison tolerances: {unknown}.")
    output = {}
    for key, value in tolerances.items():
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Tolerance {key!r} must be finite and nonnegative.")
        if key.startswith("minimum_") and value > 1.0:
            raise ValueError(f"Coverage tolerance {key!r} must not exceed 1.")
        output[key] = value
    return output


def _maximum_metric(phase_reports, section, metric):
    values = [report[section]["summary"][metric]["maximum"] for report in phase_reports.values()]
    finite = [value for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else np.nan


def _verdict(report, tolerances):
    if not tolerances:
        return {"equivalent": None, "criteria": {}}
    observations = {
        "endpoint_composition_max": report["tieline"]["summary"]["maximum_endpoint_displacement"],
        "interface_matrix_symmetric_relative_max": _maximum_metric(
            report["interface"]["phases"], "own_paths", "symmetric_relative_frobenius_error"
        ),
        "bulk_matrix_symmetric_relative_max": _maximum_metric(
            report["bulk"]["phases"], "comparison", "symmetric_relative_frobenius_error"
        ),
        "minimum_bulk_shared_fraction": report["bulk"]["summary"]["shared_fraction"],
        "minimum_interface_cross_fraction": report["interface"]["summary"]["cross_shared_fraction"],
    }
    criteria = {}
    for key, threshold in tolerances.items():
        observed = observations[key]
        passed = bool(np.isfinite(observed) and (observed >= threshold if key.startswith("minimum_") else observed <= threshold))
        criteria[key] = {"threshold": threshold, "observed": observed, "passed": passed}
    return {"equivalent": bool(criteria) and all(item["passed"] for item in criteria.values()), "criteria": criteria}


def evaluate_saved_surrogate_comparison(
    reference_bundle_path,
    candidate_bundle_path,
    *,
    interface_eta_count=201,
    bulk_grid_counts=(51, 51),
    bulk_axes=None,
    relative_error_floor=1e-300,
    tolerances=None,
):
    """Compare predictions from two verified saved surrogate bundles.

    The artifacts are validated against their own manifests, so differing
    build fingerprints are permitted. Interface calculations retain each
    surrogate's validity policy and bulk comparisons use only compositions
    evaluated successfully by both models. No thermodynamic backend is used.
    """
    reference_bundle = (
        reference_bundle_path if isinstance(reference_bundle_path, SurrogateArtifactBundle)
        else SurrogateArtifactBundle.load_published(reference_bundle_path)
    )
    candidate_bundle = (
        candidate_bundle_path if isinstance(candidate_bundle_path, SurrogateArtifactBundle)
        else SurrogateArtifactBundle.load_published(candidate_bundle_path)
    )
    _validate_identity(reference_bundle, candidate_bundle)
    reference = reference_bundle.surrogate
    candidate = candidate_bundle.surrogate
    count = _positive_count(interface_eta_count, "interface_eta_count")
    if np.isscalar(bulk_grid_counts):
        bulk_grid_counts = (bulk_grid_counts, bulk_grid_counts)
    bulk_grid_counts = tuple(_positive_count(value, "bulk_grid_counts") for value in bulk_grid_counts)
    if len(bulk_grid_counts) != 2:
        raise ValueError("bulk_grid_counts must contain exactly two values.")
    floor = float(relative_error_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("relative_error_floor must be positive and finite.")
    tolerances = _validate_tolerances(tolerances)

    normalized = np.linspace(0.0, 1.0, count)
    reference_eta = _normalized_eta(reference, normalized)
    candidate_eta = _normalized_eta(candidate, normalized)
    reference_endpoints = _interface_compositions(reference, reference_eta)
    candidate_endpoints = _interface_compositions(candidate, candidate_eta)
    displacements = {
        phase: np.linalg.norm(candidate_endpoints[phase] - reference_endpoints[phase], axis=1)
        for phase in reference.tieline_phases
    }
    reference_length, reference_orientation = _tie_geometry(
        reference_endpoints[reference.tieline_phases[0]], reference_endpoints[reference.tieline_phases[1]]
    )
    candidate_length, candidate_orientation = _tie_geometry(
        candidate_endpoints[candidate.tieline_phases[0]], candidate_endpoints[candidate.tieline_phases[1]]
    )
    all_displacements = np.concatenate(tuple(displacements.values()))
    tieline = {
        "normalized_eta": normalized,
        "reference_eta": reference_eta,
        "candidate_eta": candidate_eta,
        "reference_endpoints": reference_endpoints,
        "candidate_endpoints": candidate_endpoints,
        "endpoint_displacement": displacements,
        "reference_length": reference_length,
        "candidate_length": candidate_length,
        "length_absolute_difference": np.abs(candidate_length - reference_length),
        "reference_orientation_degrees": reference_orientation,
        "candidate_orientation_degrees": candidate_orientation,
        "orientation_absolute_difference": _angle_difference(reference_orientation, candidate_orientation),
        "summary": {
            "endpoint_displacement": _distribution(all_displacements),
            "maximum_endpoint_displacement": float(np.max(all_displacements)),
            "length_absolute_difference": _distribution(np.abs(candidate_length - reference_length)),
            "orientation_absolute_difference": _distribution(_angle_difference(reference_orientation, candidate_orientation)),
        },
    }

    interface_phases = {}
    cross_total = 0
    cross_shared = 0
    for phase in reference.tieline_phases:
        reference_own = _evaluate_matrices(reference, reference_endpoints[phase], phase, "interface")
        candidate_own = _evaluate_matrices(candidate, candidate_endpoints[phase], phase, "interface")
        own = _matrix_comparison(
            reference_own["matrices"], candidate_own["matrices"],
            reference_own["success"], candidate_own["success"], floor,
        )
        candidate_on_reference = _evaluate_matrices(candidate, reference_endpoints[phase], phase, "interface")
        at_reference = _matrix_comparison(
            reference_own["matrices"], candidate_on_reference["matrices"],
            reference_own["success"], candidate_on_reference["success"], floor,
        )
        reference_on_candidate = _evaluate_matrices(reference, candidate_endpoints[phase], phase, "interface")
        at_candidate = _matrix_comparison(
            reference_on_candidate["matrices"], candidate_own["matrices"],
            reference_on_candidate["success"], candidate_own["success"], floor,
        )
        cross_total += 2 * count
        cross_shared += at_reference["summary"]["shared_count"] + at_candidate["summary"]["shared_count"]
        interface_phases[phase] = {
            "reference_compositions": reference_endpoints[phase],
            "candidate_compositions": candidate_endpoints[phase],
            "composition_displacement": displacements[phase],
            "own_paths": own,
            "at_reference_compositions": at_reference,
            "at_candidate_compositions": at_candidate,
            "errors": {
                "reference_own": reference_own["errors"],
                "candidate_own": candidate_own["errors"],
                "candidate_on_reference": candidate_on_reference["errors"],
                "reference_on_candidate": reference_on_candidate["errors"],
            },
        }

    bulk_phases = {}
    bulk_total = 0
    bulk_shared = 0
    bulk_coverage_counts = {
        "shared": 0,
        "reference only": 0,
        "candidate only": 0,
        "neither": 0,
    }
    for phase in reference.tieline_phases:
        points, source, reference_training, candidate_training = _bulk_points(
            reference, candidate, phase, bulk_grid_counts, bulk_axes
        )
        reference_values = _evaluate_matrices(reference, points, phase, "general")
        candidate_values = _evaluate_matrices(candidate, points, phase, "general")
        comparison = _matrix_comparison(
            reference_values["matrices"], candidate_values["matrices"],
            reference_values["success"], candidate_values["success"], floor,
        )
        coverage = np.full(len(points), "neither", dtype=object)
        coverage[reference_values["success"]] = "reference only"
        coverage[candidate_values["success"]] = "candidate only"
        coverage[comparison["shared"]] = "shared"
        coverage_counts = {
            label: int(np.count_nonzero(coverage == label))
            for label in ("shared", "reference only", "candidate only", "neither")
        }
        bulk_total += len(points)
        bulk_shared += comparison["summary"]["shared_count"]
        for label, value in coverage_counts.items():
            bulk_coverage_counts[label] += value
        bulk_phases[phase] = {
            "compositions": points,
            "sample_source": source,
            "coverage": coverage,
            "coverage_counts": coverage_counts,
            "reference_training_compositions": reference_training,
            "candidate_training_compositions": candidate_training,
            "comparison": comparison,
            "reference_errors": reference_values["errors"],
            "candidate_errors": candidate_values["errors"],
        }

    differences = _build_spec_differences(
        reference_bundle.manifest["build_spec"], candidate_bundle.manifest["build_spec"]
    )
    report = {
        "kind": "saved_surrogate_comparison",
        "elements": tuple(reference.elements),
        "phases": tuple(reference.tieline_phases),
        "temperature": float(reference.temperature),
        "reference": {
            "path": str(reference_bundle.path),
            "fingerprint": reference_bundle.manifest["build_fingerprint"],
            "created_utc": reference_bundle.manifest.get("created_utc"),
            "interpolation": reference.diffusivityInterpolation,
            "validity_policy": reference.validity_policy,
        },
        "candidate": {
            "path": str(candidate_bundle.path),
            "fingerprint": candidate_bundle.manifest["build_fingerprint"],
            "created_utc": candidate_bundle.manifest.get("created_utc"),
            "interpolation": candidate.diffusivityInterpolation,
            "validity_policy": candidate.validity_policy,
        },
        "build_spec_differences": differences,
        "only_implementation_sources_differ": bool(differences) and all(
            item["path"].startswith("build_spec.implementation_sources") for item in differences
        ),
        "tieline": tieline,
        "interface": {
            "phases": interface_phases,
            "summary": {
                "cross_sample_count": int(cross_total),
                "cross_shared_count": int(cross_shared),
                "cross_shared_fraction": float(cross_shared / cross_total) if cross_total else 0.0,
            },
        },
        "bulk": {
            "phases": bulk_phases,
            "summary": {
                "sample_count": int(bulk_total),
                "shared_count": int(bulk_shared),
                "shared_fraction": float(bulk_shared / bulk_total) if bulk_total else 0.0,
                "coverage_counts": bulk_coverage_counts,
            },
        },
        "settings": {
            "interface_eta_count": count,
            "bulk_grid_counts": bulk_grid_counts,
            "relative_error_floor": floor,
        },
    }
    report["verdict"] = _verdict(report, tolerances)
    return report


def _matrix_text(matrix):
    if not np.all(np.isfinite(matrix)):
        return "unavailable"
    return "[" + "; ".join(", ".join(f"{value:.6g}" for value in row) for row in matrix) + "]"


def _comparison_hover(comparison, index, prefix, reference_error="", candidate_error=""):
    reference = comparison["reference_matrices"][index]
    candidate = comparison["candidate_matrices"][index]
    difference = comparison["difference_matrices"][index]
    symmetric = comparison["symmetric_relative_frobenius_error"][index]
    reference_relative = comparison["reference_relative_frobenius_error"][index]
    return (
        f"{prefix}<br>Reference: {_matrix_text(reference)}<br>Candidate: {_matrix_text(candidate)}"
        f"<br>Candidate - reference: {_matrix_text(difference)}"
        f"<br>Symmetric relative: {symmetric:.6g}<br>Reference-relative: {reference_relative:.6g}"
        f"<br>Reference eigenvalues: {comparison['reference_eigenvalues'][index]}"
        f"<br>Candidate eigenvalues: {comparison['candidate_eigenvalues'][index]}"
        f"<br>Reference failure: {reference_error or 'none'}"
        f"<br>Candidate failure: {candidate_error or 'none'}"
    )


def plot_saved_surrogate_comparison(report, *, renderer="browser"):
    """Create one interactive Plotly figure for a saved-surrogate report.

    ``renderer`` selects Plotly's default renderer but does not display the
    figure; callers retain control over when to call ``figure.show()``.
    """
    if report.get("kind") != "saved_surrogate_comparison":
        raise ValueError("plot_saved_surrogate_comparison requires a saved-surrogate comparison report.")
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:
        raise ImportError("Plotly is required for saved surrogate comparison plots.") from exc
    if renderer is not None:
        pio.renderers.default = renderer
    figure = go.Figure()
    groups = []
    titles = []

    def add_group(label, traces):
        start = len(figure.data)
        for trace in traces:
            trace.visible = len(groups) == 0
            figure.add_trace(trace)
        groups.append(list(range(start, len(figure.data))))
        titles.append(label)

    differences = report["build_spec_differences"]
    verdict = report["verdict"]["equivalent"]
    difference_paths = ", ".join(item["path"] for item in differences) or "none"
    summary_rows = [
        ("Reference fingerprint", report["reference"]["fingerprint"]),
        ("Candidate fingerprint", report["candidate"]["fingerprint"]),
        ("Build-spec differences", len(differences)),
        ("Build-spec difference paths", difference_paths),
        ("Only implementation sources differ", report["only_implementation_sources_differ"]),
        ("Maximum endpoint displacement", report["tieline"]["summary"]["maximum_endpoint_displacement"]),
        ("Maximum interface matrix error", _maximum_metric(
            report["interface"]["phases"], "own_paths", "symmetric_relative_frobenius_error"
        )),
        ("Maximum bulk matrix error", _maximum_metric(
            report["bulk"]["phases"], "comparison", "symmetric_relative_frobenius_error"
        )),
        ("Shared bulk coverage", report["bulk"]["summary"]["shared_fraction"]),
        ("Interface cross coverage", report["interface"]["summary"]["cross_shared_fraction"]),
        ("Equivalent", "not evaluated" if verdict is None else verdict),
    ]
    add_group("Overview", [go.Table(
        header={"values": ["Quantity", "Value"]},
        cells={"values": [[row[0] for row in summary_rows], [str(row[1]) for row in summary_rows]]},
        name="Overview",
    )])

    normalized = report["tieline"]["normalized_eta"]
    tie_traces = []
    for phase in report["phases"]:
        reference_points = report["tieline"]["reference_endpoints"][phase]
        candidate_points = report["tieline"]["candidate_endpoints"][phase]
        for label, points, dash in (("Reference", reference_points, "solid"), ("Candidate", candidate_points, "dash")):
            full = np.column_stack((1.0 - np.sum(points, axis=1), points))
            hover = [
                f"{label} {phase}<br>u={u:.6g}<br>Composition={point.tolist()}"
                f"<br>Reference composition={reference_points[index].tolist()}"
                f"<br>Candidate composition={candidate_points[index].tolist()}"
                f"<br>Displacement={report['tieline']['endpoint_displacement'][phase][index]:.6g}"
                for index, (u, point) in enumerate(zip(normalized, points))
            ]
            tie_traces.append(go.Scatterternary(
                a=full[:, 2], b=full[:, 0], c=full[:, 1], mode="lines+markers",
                line={"dash": dash}, name=f"{label} {phase}", text=hover,
                hovertemplate="%{text}<extra></extra>",
            ))
        tie_traces.append(go.Scatter(
            x=normalized, y=report["tieline"]["endpoint_displacement"][phase],
            mode="lines+markers", name=f"{phase} endpoint displacement",
            xaxis="x3", yaxis="y4",
            text=[
                f"{phase}<br>u={u:.6g}<br>Endpoint displacement={value:.6g}"
                for u, value in zip(normalized, report["tieline"]["endpoint_displacement"][phase])
            ],
            hovertemplate="%{text}<extra></extra>",
        ))
    add_group("Tie lines", tie_traces)

    for phase in report["phases"]:
        phase_report = report["interface"]["phases"][phase]
        traces = []
        for label, comparison, dash, reference_errors, candidate_errors in (
            ("Own paths", phase_report["own_paths"], "solid",
             phase_report["errors"]["reference_own"], phase_report["errors"]["candidate_own"]),
            ("At reference compositions", phase_report["at_reference_compositions"], "dot",
             phase_report["errors"]["reference_own"], phase_report["errors"]["candidate_on_reference"]),
            ("At candidate compositions", phase_report["at_candidate_compositions"], "dash",
             phase_report["errors"]["reference_on_candidate"], phase_report["errors"]["candidate_own"]),
        ):
            hover = [
                _comparison_hover(
                    comparison, i,
                    f"{phase} - {label}<br>u={normalized[i]:.6g}"
                    f"<br>Reference interpolation={report['reference']['interpolation']}"
                    f"<br>Candidate interpolation={report['candidate']['interpolation']}",
                    reference_errors[i], candidate_errors[i],
                )
                for i in range(len(normalized))
            ]
            traces.append(go.Scatter(
                x=normalized, y=comparison["symmetric_relative_frobenius_error"],
                mode="lines+markers", line={"dash": dash}, name=label,
                text=hover, hovertemplate="%{text}<extra></extra>", xaxis="x", yaxis="y",
            ))
        own = phase_report["own_paths"]
        traces.extend([
            go.Scatter(
                x=normalized, y=np.linalg.norm(own["reference_matrices"], axis=(1, 2)),
                mode="lines", name="Reference matrix norm", xaxis="x2", yaxis="y2",
                text=[_comparison_hover(own, i, f"{phase} - own paths<br>u={normalized[i]:.6g}",
                                        phase_report["errors"]["reference_own"][i],
                                        phase_report["errors"]["candidate_own"][i]) for i in range(len(normalized))],
                hovertemplate="%{text}<extra></extra>",
            ),
            go.Scatter(
                x=normalized, y=np.linalg.norm(own["candidate_matrices"], axis=(1, 2)),
                mode="lines", name="Candidate matrix norm", xaxis="x2", yaxis="y2",
                text=[_comparison_hover(own, i, f"{phase} - own paths<br>u={normalized[i]:.6g}",
                                        phase_report["errors"]["reference_own"][i],
                                        phase_report["errors"]["candidate_own"][i]) for i in range(len(normalized))],
                hovertemplate="%{text}<extra></extra>",
            ),
            go.Scatter(
                x=normalized, y=phase_report["composition_displacement"],
                mode="lines+markers", line={"dash": "dot"}, name="Composition displacement",
                xaxis="x", yaxis="y3",
                text=[
                    f"{phase} - own paths<br>u={normalized[i]:.6g}"
                    f"<br>Composition displacement={phase_report['composition_displacement'][i]:.6g}"
                    for i in range(len(normalized))
                ],
                hovertemplate="%{text}<extra></extra>",
            ),
        ])
        add_group(f"Interface: {phase}", traces)

    for phase in report["phases"]:
        phase_report = report["bulk"]["phases"][phase]
        points = phase_report["compositions"]
        comparison = phase_report["comparison"]
        full = np.column_stack((1.0 - np.sum(points, axis=1), points))
        shared = comparison["shared"]
        hover = [
            _comparison_hover(
                comparison, i,
                f"{phase} - general<br>Composition={points[i].tolist()}"
                f"<br>Source={phase_report['sample_source'][i]}<br>Coverage={phase_report['coverage'][i]}"
                f"<br>Reference interpolation={report['reference']['interpolation']}"
                f"<br>Candidate interpolation={report['candidate']['interpolation']}",
                phase_report["reference_errors"][i], phase_report["candidate_errors"][i],
            )
            for i in range(len(points))
        ]
        values = comparison["symmetric_relative_frobenius_error"]
        colors = np.log10(np.maximum(values[shared], np.finfo(float).tiny))
        traces = [go.Scatterternary(
            a=full[shared, 2], b=full[shared, 0], c=full[shared, 1], mode="markers",
            name="Shared predictions", text=np.asarray(hover, dtype=object)[shared],
            hovertemplate="%{text}<extra></extra>",
            marker={"color": colors, "colorscale": "Viridis", "showscale": True,
                    "colorbar": {"title": "log10 symmetric error"}},
        )]
        missing = ~shared
        if np.any(missing):
            traces.append(go.Scatterternary(
                a=full[missing, 2], b=full[missing, 0], c=full[missing, 1], mode="markers",
                name="Coverage gap", text=np.asarray(hover, dtype=object)[missing],
                hovertemplate="%{text}<extra></extra>", marker={"color": "lightgray", "symbol": "x"},
            ))
        add_group(f"Bulk: {phase}", traces)

    buttons = []
    for index, indices in enumerate(groups):
        visible = [False] * len(figure.data)
        for trace_index in indices:
            visible[trace_index] = True
        title = titles[index]
        is_tieline = title == "Tie lines"
        is_interface = title.startswith("Interface:")
        buttons.append({
            "label": title,
            "method": "update",
            "args": [
                {"visible": visible},
                {
                    "title": title,
                    "ternary.domain.x": [0.0, 0.58] if is_tieline else [0.0, 1.0],
                    "xaxis.visible": is_interface,
                    "yaxis.visible": is_interface,
                    "xaxis2.visible": is_interface,
                    "yaxis2.visible": is_interface,
                    "yaxis3.visible": is_interface,
                    "xaxis3.visible": is_tieline,
                    "yaxis4.visible": is_tieline,
                },
            ],
        })
    figure.update_layout(
        title=titles[0], template="plotly_white", width=1150, height=820,
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.01, "y": 1.13}],
        ternary={"sum": 1, "domain": {"x": [0.0, 1.0]},
                 "aaxis": {"title": report["elements"][2]},
                 "baxis": {"title": report["elements"][0]}, "caxis": {"title": report["elements"][1]}},
        xaxis={"domain": [0.0, 0.46], "title": "normalized eta", "visible": False},
        yaxis={"domain": [0.0, 1.0], "title": "symmetric relative error", "visible": False},
        yaxis3={"overlaying": "y", "side": "right", "title": "composition displacement", "visible": False},
        xaxis2={"domain": [0.55, 1.0], "title": "normalized eta", "anchor": "y2", "visible": False},
        yaxis2={"domain": [0.0, 1.0], "title": "matrix Frobenius norm", "anchor": "x2", "type": "log", "visible": False},
        xaxis3={"domain": [0.64, 1.0], "title": "normalized eta", "anchor": "y4", "visible": False},
        yaxis4={"domain": [0.0, 1.0], "title": "endpoint composition displacement", "anchor": "x3", "visible": False},
    )
    return figure


def compare_saved_surrogates(reference_bundle_path, candidate_bundle_path, *, renderer="browser", **kwargs):
    """Return a numerical report and Plotly figure for two verified bundles.

    Additional keyword arguments are forwarded to
    :func:`evaluate_saved_surrogate_comparison`. No Thermo-Calc session is
    started and neither loaded surrogate is modified.
    """
    report = evaluate_saved_surrogate_comparison(reference_bundle_path, candidate_bundle_path, **kwargs)
    return {"report": report, "figure": plot_saved_surrogate_comparison(report, renderer=renderer)}
