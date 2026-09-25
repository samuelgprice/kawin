"""Interactive diagnostics for ternary moving-boundary surrogate models.

The evaluation helpers in this module depend only on NumPy and SciPy so they
remain available when Plotly is not installed.  Plotting helpers import Plotly
lazily and return figures without displaying or writing them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from html import escape
import json
from pathlib import Path

import numpy as np
import tqdm
from scipy.spatial import Delaunay

from .MovingBoundarySurrogates import (
    MergedPhaseDiffusivitySurrogate,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    _BulkDiffusivitySimplexLinear2D,
    _BulkDiffusivitySimplexPositive2x2,
    _matrix_validity_diagnostics,
)

import importlib
import examples.debugInPlace as debug_module

importlib.invalidate_caches()
debug_module = importlib.reload(debug_module)

# Required if you previously used:
# from examples.debugInPlace import debugInPlace
debugInPlace = debug_module.debugInPlace

_HOVER_PRESETS = {
    "tieline": {
        "minimal": ("source", "phase", "eta", "x1", "x2"),
        "diagnostic": (
            "source", "phase", "eta", "x_ref", "x1", "x2", "probe_x1", "probe_x2",
            "partner_x1", "partner_x2", "training", "endpoint_error", "tieline_length",
            "orientation", "length_error", "angle_error",
        ),
        "all": (
            "source", "phase", "eta", "x_ref", "x1", "x2", "probe_ref", "probe_x1",
            "probe_x2", "partner_x1", "partner_x2", "training", "endpoint_error",
            "tieline_length", "orientation", "length_error", "angle_error", "failure",
        ),
    },
    "diffusivity": {
        "minimal": ("source", "phase", "eta", "x1", "x2", "value"),
        "diagnostic": (
            "source", "phase", "eta", "x_ref", "x1", "x2", "value", "d00", "d01",
            "d10", "d11", "valid", "eigenvalue_0", "eigenvalue_1", "training_distance",
            "training", "fallback", "truth_value", "absolute_error", "relative_error", "matrix_relative_error",
        ),
        "all": (
            "source", "phase", "context", "eta", "x_ref", "x1", "x2", "value", "d00",
            "d01", "d10", "d11", "valid", "eigenvalue_0", "eigenvalue_1",
            "training_distance", "fallback", "truth_value", "absolute_error", "relative_error",
            "training", "matrix_relative_error", "truth_valid", "failure",
        ),
    },
}

_FIELD_LABELS = {
    "source": "Source",
    "phase": "Phase",
    "context": "Context",
    "eta": "eta",
    "x_ref": "X(reference)",
    "x1": "X(component 1)",
    "x2": "X(component 2)",
    "probe_ref": "Probe X(reference)",
    "probe_x1": "Probe X(component 1)",
    "probe_x2": "Probe X(component 2)",
    "partner_x1": "Partner X(component 1)",
    "partner_x2": "Partner X(component 2)",
    "training": "Training sample",
    "endpoint_error": "Endpoint error norm",
    "tieline_length": "Tie-line length",
    "orientation": "Tie-line angle (deg)",
    "length_error": "Tie-line length error",
    "angle_error": "Tie-line angle error (deg)",
    "value": "Displayed value",
    "d00": "D[0,0] (m^2/s)",
    "d01": "D[0,1] (m^2/s)",
    "d10": "D[1,0] (m^2/s)",
    "d11": "D[1,1] (m^2/s)",
    "valid": "Valid matrix",
    "truth_valid": "Valid truth matrix",
    "eigenvalue_0": "Eigenvalue 0",
    "eigenvalue_1": "Eigenvalue 1",
    "training_distance": "Nearest training distance",
    "fallback": "Nearest fallback",
    "truth_value": "Ground truth",
    "absolute_error": "Absolute error",
    "relative_error": "Relative error",
    "matrix_relative_error": "Matrix relative error",
    "failure": "Truth failure",
}

_TEXT_FIELDS = {"source", "phase", "context", "failure"}
_BOOL_FIELDS = {"training", "valid", "truth_valid", "fallback"}


def _positive_int(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _truth_error_policy(value):
    value = str(value).lower()
    if value not in {"raise", "record"}:
        raise ValueError("on_truth_error must be either 'raise' or 'record'.")
    return value


def _independent_points(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    return values


def _training_mask(values, training, atol=1e-12):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    training = np.asarray(training, dtype=np.float64).reshape(-1)
    return np.any(np.isclose(values[:, None], training[None, :], rtol=0.0, atol=atol), axis=1)


def _probe_path(surrogate, eta, probe_compositions=None, *, required=False):
    """Reconstruct the physical bulk-composition query associated with each eta."""
    eta = np.asarray(eta, dtype=np.float64)
    if callable(probe_compositions):
        points = np.asarray([probe_compositions(float(value)) for value in eta], dtype=np.float64)
        return _independent_points(points, "probe_compositions callable output")
    if probe_compositions is not None:
        points = _independent_points(probe_compositions, "probe_compositions")
        if points.shape[0] == eta.size:
            return points.copy()
        if points.shape[0] == surrogate.eta_samples.size:
            return np.column_stack(
                [np.interp(eta, surrogate.eta_samples, points[:, component]) for component in range(2)]
            )
        raise ValueError("probe_compositions must match eta_count or the surrogate training eta count.")

    metadata = getattr(surrogate, "metadata", {})
    if "generated_probe_points" in metadata:
        points = _independent_points(metadata["generated_probe_points"], "generated_probe_points metadata")
        if points.shape[0] != surrogate.eta_samples.size:
            raise ValueError("generated_probe_points metadata must match eta_samples.")
        return np.column_stack(
            [np.interp(eta, surrogate.eta_samples, points[:, component]) for component in range(2)]
        )
    if "probe_start" in metadata and "probe_end" in metadata:
        start = np.asarray(metadata["probe_start"], dtype=np.float64).reshape(2)
        end = np.asarray(metadata["probe_end"], dtype=np.float64).reshape(2)
        lower, upper = surrogate.eta_bounds
        fraction = (eta - lower) / (upper - lower)
        return start[None, :] + fraction[:, None] * (end - start)[None, :]
    if required:
        raise ValueError(
            "Ground-truth tie-line comparison requires probe_start/probe_end or generated_probe_points "
            "metadata, or an explicit probe_compositions array/callable."
        )
    return None


def _tie_geometry(left, right):
    vectors = np.asarray(right) - np.asarray(left)
    length = np.linalg.norm(vectors, axis=1)
    orientation = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return length, orientation


def _angle_difference(first, second):
    """Return the wrapped absolute angular difference in degrees."""
    return np.abs((np.asarray(first) - np.asarray(second) + 180.0) % 360.0 - 180.0)


def _independent_truth_endpoint(value, elements):
    """Normalize an independent or reference-first full ternary endpoint."""
    composition = np.asarray(value, dtype=np.float64).reshape(-1)
    if composition.size == len(elements) == 3:
        composition = composition[1:]
    if composition.size != 2 or not np.all(np.isfinite(composition)):
        raise ValueError(
            "Ground-truth tie-line endpoints must contain either two independent "
            "components or three finite full components in surrogate element order."
        )
    return composition


def _truth_endpoints(result, phases, elements):
    """Validate and phase-order a thermodynamics interface-composition result."""
    if not isinstance(result, tuple) or len(result) not in {2, 3}:
        raise ValueError("getInterfacialComposition must return two endpoints and optional metadata.")
    raw = tuple(_independent_truth_endpoint(value, elements) for value in result[:2])
    if len(result) == 2:
        return raw
    metadata = result[2]
    endpoints = metadata.get("endpoints", ()) if isinstance(metadata, Mapping) else ()
    by_phase = {
        str(item.get("phase")): _independent_truth_endpoint(item.get("composition"), elements)
        for item in endpoints
        if isinstance(item, Mapping) and item.get("phase") is not None
    }
    if by_phase:
        missing = [phase for phase in phases if phase not in by_phase]
        extra = [phase for phase in by_phase if phase not in phases]
        if missing or extra:
            raise ValueError(f"Ground-truth endpoint phases do not match {phases}: missing={missing}, extra={extra}.")
        ordered = tuple(by_phase[phase] for phase in phases)
        return ordered
    endpoint_phases = tuple(str(value) for value in metadata.get("endpoint_phases", ())) if isinstance(metadata, Mapping) else ()
    if endpoint_phases and endpoint_phases != tuple(phases): 
        debugInPlace()
        raise ValueError(f"Ground-truth endpoint phases {endpoint_phases} do not match {tuple(phases)}.")
    return raw


def evaluate_tieline_diagnostics(
    surrogate,
    *,
    thermodynamics=None,
    probe_compositions=None,
    eta_count=101,
    on_truth_error="raise",
):
    """Evaluate a ternary surrogate tie-line family and optional ground truth.

    Ground-truth points are evaluated at the original bulk probe path recorded
    by line or seed-point surrogate construction. Ground-truth endpoints may
    contain either the two independent components or all three components in
    ``surrogate.elements`` order. With ``on_truth_error`` set to ``"record"``,
    failed samples are retained as NaNs and described in ``failures``.
    """
    if not isinstance(surrogate, TernaryMovingBoundaryThermodynamicsSurrogate):
        raise TypeError("Tie-line diagnostics require TernaryMovingBoundaryThermodynamicsSurrogate.")
    policy = _truth_error_policy(on_truth_error)
    eta_count = _positive_int(eta_count, "eta_count")
    if eta_count < 2:
        raise ValueError("eta_count must be at least 2.")
    eta = np.linspace(surrogate.eta_bounds[0], surrogate.eta_bounds[1], eta_count)
    phases = tuple(surrogate.tieline_phases)
    endpoints = {
        phase: np.column_stack(
            [np.interp(eta, surrogate.eta_samples, surrogate.tieline_compositions[phase][:, i]) for i in range(2)]
        )
        for phase in phases
    }
    probe = _probe_path(surrogate, eta, probe_compositions, required=thermodynamics is not None)
    length, orientation = _tie_geometry(endpoints[phases[0]], endpoints[phases[1]])
    report = {
        "kind": "tieline",
        "elements": tuple(surrogate.elements),
        "phases": phases,
        "temperature": float(surrogate.temperature),
        "eta": eta,
        "probe_compositions": probe,
        "training_mask": _training_mask(eta, surrogate.eta_samples),
        "endpoint_compositions": endpoints,
        "tieline_length": length,
        "orientation_degrees": orientation,
        "truth": None,
        "failures": [],
    }
    if thermodynamics is None:
        report["summary"] = {"truth_evaluated": False, "sample_count": eta_count, "failure_count": 0}
        return report

    truth_endpoints = {phase: np.full((eta_count, 2), np.nan) for phase in phases}
    failures = []
    precipitate_phase = surrogate.metadata.get("precipitate_phase", phases[1])
    for index, (eta_value, point) in enumerate(zip(eta, probe)):
        try:
            result = thermodynamics.getInterfacialComposition(
                point,
                surrogate.temperature,
                precPhase=precipitate_phase,
                returnMeta=True,
            )
            if result[2]['endpoint_phases'][0] is None:
                debugInPlace()

                result = thermodynamics.getInterfacialComposition(
                                point,
                                surrogate.temperature,
                                precPhase=precipitate_phase,
                                returnMeta=True,
                            )
            ordered = _truth_endpoints(result, phases, surrogate.elements)
            for phase, composition in zip(phases, ordered):
                truth_endpoints[phase][index] = composition
        except Exception as exc:
            message = (
                f"Ground-truth tie-line query failed at sample {index}, eta={eta_value:.8g}, "
                f"composition={point.tolist()}: {exc}"
            )
            if policy == "raise":
                raise ValueError(message) from exc
            failures.append({"index": index, "eta": float(eta_value), "composition": point.copy(), "error": str(exc)})

    endpoint_abs_error = {
        phase: np.abs(endpoints[phase] - truth_endpoints[phase]) for phase in phases
    }
    endpoint_error_norm = {
        phase: np.linalg.norm(endpoints[phase] - truth_endpoints[phase], axis=1) for phase in phases
    }
    truth_length, truth_orientation = _tie_geometry(truth_endpoints[phases[0]], truth_endpoints[phases[1]])
    valid = np.all(np.isfinite(np.column_stack(tuple(truth_endpoints.values()))), axis=1)
    report["truth"] = {
        "endpoint_compositions": truth_endpoints,
        "endpoint_absolute_error": endpoint_abs_error,
        "endpoint_error_norm": endpoint_error_norm,
        "tieline_length": truth_length,
        "orientation_degrees": truth_orientation,
        "length_absolute_error": np.abs(length - truth_length),
        "orientation_absolute_error": _angle_difference(orientation, truth_orientation),
        "valid": valid,
    }
    report["failures"] = failures
    finite_errors = np.concatenate([values[np.isfinite(values)] for values in endpoint_error_norm.values()])
    report["summary"] = {
        "truth_evaluated": True,
        "sample_count": eta_count,
        "valid_count": int(np.count_nonzero(valid)),
        "failure_count": len(failures),
        "max_endpoint_error": float(np.max(finite_errors)) if finite_errors.size else np.nan,
        "max_orientation_error_degrees": float(np.nanmax(report["truth"]["orientation_absolute_error"])) if np.any(valid) else np.nan,
    }
    return report


def _surrogate_phases(surrogate, phases):
    available = (
        tuple(surrogate.tieline_phases)
        if isinstance(surrogate, TernaryMovingBoundaryThermodynamicsSurrogate)
        else (surrogate.phase,)
    )
    if phases is None:
        return available
    if isinstance(phases, str):
        phases = (phases,)
    phases = tuple(str(phase) for phase in phases)
    unknown = [phase for phase in phases if phase not in available]
    if unknown:
        raise ValueError(f"Unknown diagnostic phases {unknown}; available phases are {available}.")
    return phases


def _nearest_distances(points, training, chunk_size=10000):
    """Compute composition-space distance to the nearest training point in chunks."""
    points = np.asarray(points, dtype=np.float64)
    training = np.asarray(training, dtype=np.float64)
    out = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        delta = points[start:stop, None, :] - training[None, :, :]
        out[start:stop] = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    return out


def _diffusivity_training_compositions(surrogate, context, phase):
    values = surrogate.diffusivity_compositions[context]
    if isinstance(values, Mapping):
        values = values[phase]
    return np.asarray(values, dtype=np.float64)


def _diffusivity_training_matrices(surrogate, context, phase):
    """Return stored matrices in the same row order as training compositions."""
    values = surrogate.diffusivities[context]
    if isinstance(values, Mapping):
        values = values[phase]
    return np.asarray(values, dtype=np.float64)


def _loo_sample_indices(sample_indices, phase, count):
    """Select unique held-out row indices, retaining the caller's order."""
    if sample_indices is None:
        return np.arange(count, dtype=np.int64)
    if isinstance(sample_indices, Mapping) and phase not in sample_indices:
        raise ValueError(f"sample_indices is missing phase {phase}.")
    values = sample_indices[phase] if isinstance(sample_indices, Mapping) else sample_indices
    indices = np.asarray(values)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("sample_indices must be a one-dimensional sequence of integers.")
    if np.any(indices < 0) or np.any(indices >= count) or np.unique(indices).size != indices.size:
        raise ValueError(f"sample_indices for phase {phase} must be unique and between 0 and {count - 1}.")
    return indices.astype(np.int64, copy=True)


def _loo_refit_supports_point(points, point):
    """Return whether retained samples provide nondegenerate 2-D support.

    This intentionally rebuilds the geometry after each holdout.  The source
    surrogate's cached hull contains the held-out row and therefore cannot
    establish support for a leave-one-out prediction.
    """
    if len(points) < 3 or np.linalg.matrix_rank(points - points[0]) < 2:
        return False
    try:
        return int(Delaunay(points).find_simplex(point)) >= 0
    except Exception:
        return False


def _loo_bulk_matrix(points, matrices, index, interpolation, validity_policy):
    """Refit the configured bulk evaluator after deleting one fit sample.

    ``simplex_positive_2x2`` uses the production spectral evaluator directly.
    ``continuous_grid`` cannot be reconstructed after deleting one node and is
    explicitly downgraded to scattered ``simplex_linear``.  Strict sources do
    not extrapolate from an unsupported retained cloud; legacy sources retain
    the historical nearest-neighbor diagnostic fallback.
    """
    retained = np.arange(len(points)) != index
    remaining_points = points[retained]
    remaining_matrices = matrices[retained]
    if not len(remaining_points):
        return None, "unavailable", False, np.nan, "insufficient_fit_support", None
    point = points[index]
    distance_squared = np.sum((remaining_points - point) ** 2, axis=1)
    nearest = int(np.argmin(distance_squared))
    distance = float(np.sqrt(distance_squared[nearest]))
    refit_mode = "simplex_linear" if interpolation == "continuous_grid" else interpolation
    refit_note = "continuous_grid_missing_node_downgrade" if interpolation == "continuous_grid" else None
    supported = _loo_refit_supports_point(remaining_points, point)
    if not supported:
        if validity_policy == "raise":
            return None, refit_mode, False, distance, "insufficient_fit_support", refit_note
        return remaining_matrices[nearest].copy(), "nearest", True, distance, "nearest_fallback", refit_note

    if interpolation == "nearest":
        return remaining_matrices[nearest].copy(), "nearest", False, distance, "ok", None
    evaluator = (
        _BulkDiffusivitySimplexPositive2x2
        if refit_mode == "simplex_positive_2x2"
        else _BulkDiffusivitySimplexLinear2D
    )(remaining_points, remaining_matrices)
    prediction = np.asarray(evaluator.evaluate(point[None, :]), dtype=np.float64)[0]
    return prediction, refit_mode, False, distance, "ok", refit_note


def evaluate_diffusivity_leave_one_out(
    surrogate,
    *,
    phases=None,
    sample_indices=None,
    relative_error_floor=1e-300,
    eigen_imag_tol=1e-12,
    eigen_real_min=1e-14,
):
    """Compare stored bulk matrices with predictions from leave-one-out refits.

    Uses the source surrogate's cached canonical solver-usable general fit
    population, not raw provenance rows. It makes no thermodynamics queries or
    changes to the original model. ``sample_indices`` indexes that fit
    population. ``continuous_grid`` is explicitly downgraded to scattered
    simplex-linear after a node is removed. With ``validity_policy='raise'``,
    a point outside the rebuilt retained 2-D hull is reported as
    ``insufficient_fit_support``. ``legacy`` retains an explicitly labeled
    nearest-neighbor fallback in that case.

    Returns one report per phase with held-out compositions, actual and
    predicted 2x2 matrices, component and Frobenius relative errors, physical
    eigenvalues, and fallback flags. Failed refits have NaN predictions and
    entries in ``failures``. Rebuilding a triangulation for each held-out row
    can be slow for large grids; use ``sample_indices`` for a subset if needed.
    """
    if not isinstance(surrogate, (TernaryMovingBoundaryThermodynamicsSurrogate, MergedPhaseDiffusivitySurrogate)):
        raise TypeError("Leave-one-out diagnostics require a supported ternary diffusivity surrogate.")
    floor = float(relative_error_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("relative_error_floor must be positive and finite.")
    selected_phases = _surrogate_phases(surrogate, phases)
    interpolation = str(surrogate.diffusivityInterpolation)
    phase_reports = {}
    for phase in selected_phases:
        support = surrogate._fitSupport[phase if isinstance(surrogate, TernaryMovingBoundaryThermodynamicsSurrogate) else "general"]
        points = np.asarray(support["points"], dtype=np.float64)
        matrices = np.asarray(support["matrices"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or matrices.shape != (len(points), 2, 2):
            raise ValueError(f"Stored general diffusivity samples for phase {phase} have incompatible shapes.")
        indices = _loo_sample_indices(sample_indices, phase, len(points))
        held_out = matrices[indices].copy()
        predicted = np.full((len(indices), 2, 2), np.nan, dtype=np.float64)
        nearest_distance = np.full(len(indices), np.nan, dtype=np.float64)
        fallback = np.zeros(len(indices), dtype=bool)
        refit_interpolation = np.full(len(indices), "failed", dtype=object)
        refit_note = np.full(len(indices), None, dtype=object)
        prediction_status = np.full(len(indices), "refit_failure", dtype=object)
        failures = []
        for row, index in tqdm.tqdm(enumerate(indices), total=len(indices)):
            try:
                prediction, mode, used_fallback, distance, status, note = _loo_bulk_matrix(
                    points, matrices, int(index), interpolation, surrogate.validity_policy
                )
                if prediction is not None:
                    predicted[row] = prediction
                refit_interpolation[row] = mode
                fallback[row] = used_fallback
                nearest_distance[row] = distance
                refit_note[row] = note
                prediction_status[row] = status
            except Exception as exc:
                failures.append({
                    "index": int(index), "phase": phase,
                    "composition": points[index].copy(), "error": str(exc),
                })
        predicted_validity = _matrix_validity_diagnostics(
            predicted, eigen_imag_tol=eigen_imag_tol, eigen_real_min=eigen_real_min
        )
        actual_validity = _matrix_validity_diagnostics(
            held_out, eigen_imag_tol=eigen_imag_tol, eigen_real_min=eigen_real_min
        )
        absolute = np.abs(predicted - held_out)
        relative = absolute / np.maximum(np.abs(held_out), floor)
        matrix_relative = np.linalg.norm(predicted - held_out, axis=(1, 2)) / np.maximum(
            np.linalg.norm(held_out, axis=(1, 2)), floor
        )
        finite_error = matrix_relative[np.isfinite(matrix_relative)]
        attempted_prediction = np.isin(prediction_status, ("ok", "nearest_fallback"))
        invalid_prediction = attempted_prediction & ~predicted_validity["valid"]
        prediction_status[invalid_prediction] = "invalid_prediction"
        phase_reports[phase] = {
            "sample_indices": indices,
            "compositions": points[indices].copy(),
            "actual_matrices": held_out,
            "predicted_matrices": predicted,
            "actual_valid": actual_validity["valid"],
            "predicted_valid": predicted_validity["valid"],
            "actual_eigenvalues": actual_validity["eigenvalues"] * actual_validity["scales"][:, None],
            "predicted_eigenvalues": predicted_validity["eigenvalues"] * predicted_validity["scales"][:, None],
            "absolute_error": absolute,
            "relative_error": relative,
            "matrix_relative_error": matrix_relative,
            "nearest_training_distance": nearest_distance,
            "fallback": fallback,
            "refit_interpolation": refit_interpolation,
            "refit_note": refit_note,
            "prediction_status": prediction_status,
            "source_sample_count": len(points),
            "loo_fit_sample_count": max(len(points) - 1, 0),
            "failures": failures,
            "summary": {
                "sample_count": len(indices),
                "failure_count": len(failures),
                "nearest_fallback_count": int(np.count_nonzero(fallback)),
                "insufficient_fit_support_count": int(np.count_nonzero(prediction_status == "insufficient_fit_support")),
                "invalid_prediction_count": int(np.count_nonzero(invalid_prediction)),
                "mean_matrix_relative_error": float(np.mean(finite_error)) if finite_error.size else np.nan,
                "max_matrix_relative_error": float(np.max(finite_error)) if finite_error.size else np.nan,
            },
        }
    return {
        "kind": "diffusivity_leave_one_out",
        "elements": tuple(surrogate.elements),
        "phases": selected_phases,
        "temperature": float(surrogate.temperature),
        "source_interpolation": interpolation,
        "source_validity_policy": str(surrogate.validity_policy),
        "context": "general",
        "phase_reports": phase_reports,
    }


def _fallback_mask(surrogate, points, phase, context):
    """Identify simplex-linear queries that use nearest-neighbor hull fallback."""
    if getattr(surrogate, "diffusivityInterpolation", None) != "simplex_linear":
        return np.zeros(len(points), dtype=bool)
    training = _diffusivity_training_compositions(surrogate, context, phase)
    if training.shape[0] < 3:
        return np.ones(len(points), dtype=bool)
    try:
        from scipy.spatial import Delaunay

        return Delaunay(training).find_simplex(points) < 0
    except Exception:
        return np.ones(len(points), dtype=bool)


def _bulk_grid(surrogate, phases, counts, axes):
    """Build a rectangular plotting grid while retaining only ternary-simplex points."""
    if np.isscalar(counts):
        counts = (counts, counts)
    counts = tuple(_positive_int(value, "bulk grid count") for value in counts)
    if len(counts) != 2:
        raise ValueError("bulk_grid_counts must contain exactly two values.")
    if axes is None:
        training = np.concatenate(
            [_diffusivity_training_compositions(surrogate, "general", phase) for phase in phases], axis=0
        )
        lower = np.min(training, axis=0)
        upper = np.max(training, axis=0)
        span = upper - lower
        for component in range(2):
            if span[component] <= 0.0:
                delta = max(1e-6, abs(lower[component]) * 1e-3)
                lower[component] = max(float(getattr(surrogate, "min_composition", 0.0)), lower[component] - delta)
                upper[component] += delta
        axes = tuple(np.linspace(lower[i], upper[i], counts[i]) for i in range(2))
    else:
        axes = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in axes)
        if len(axes) != 2 or any(axis.size == 0 for axis in axes):
            raise ValueError("bulk_axes must contain two nonempty arrays.")
        if any(not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0) for axis in axes):
            raise ValueError("bulk_axes must be finite and strictly increasing.")
    grid_x, grid_y = np.meshgrid(axes[0], axes[1], indexing="ij")
    all_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    minimum = float(getattr(surrogate, "min_composition", 0.0))
    mask = (
        (all_points[:, 0] >= minimum)
        & (all_points[:, 1] >= minimum)
        & (np.sum(all_points, axis=1) <= 1.0 - minimum)
    )
    indices = np.column_stack(np.unravel_index(np.flatnonzero(mask), grid_x.shape))
    return tuple(axis.copy() for axis in axes), all_points[mask], indices, grid_x.shape


def _matrix_predictions(surrogate, compositions, phase, context):
    values = surrogate.getInterdiffusivity(
        compositions,
        getattr(surrogate, "temperature", None),
        phase=phase,
        query_context=context,
    )
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (2, 2) and len(compositions) == 1:
        values = values[None, :, :]
    if values.shape != (len(compositions), 2, 2):
        raise ValueError(f"Surrogate diffusivity for phase {phase} returned shape {values.shape}, expected {(len(compositions), 2, 2)}.")
    return values


def _truth_matrices(thermodynamics, compositions, temperature, phase, policy, context):
    """Evaluate truth matrices pointwise so individual backend failures can be recorded."""
    matrices = np.full((len(compositions), 2, 2), np.nan)
    failures = []
    for index, point in enumerate(compositions):
        try:
            matrix = np.asarray(thermodynamics.getInterdiffusivity(point, temperature, phase=phase), dtype=np.float64)
            if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"expected a finite 2x2 matrix, received shape {matrix.shape}")
            matrices[index] = matrix
        except Exception as exc:
            message = (
                f"Ground-truth diffusivity query failed for phase {phase}, context={context}, "
                f"sample {index}, composition={point.tolist()}: {exc}"
            )
            if policy == "raise":
                raise ValueError(message) from exc
            failures.append({"index": index, "phase": phase, "context": context, "composition": point.copy(), "error": str(exc)})
    return matrices, failures


def _diffusivity_phase_report(
    surrogate,
    phase,
    context,
    compositions,
    thermodynamics,
    policy,
    relative_error_floor,
    eigen_imag_tol,
    eigen_real_min,
    eta=None,
):
    """Evaluate one phase/context and attach coverage, validity, and truth diagnostics."""
    matrices = _matrix_predictions(surrogate, compositions, phase, context)
    diagnostics = _matrix_validity_diagnostics(
        matrices, eigen_imag_tol=eigen_imag_tol, eigen_real_min=eigen_real_min
    )
    training_context = "interface" if context == "interface" else "general"
    training = _diffusivity_training_compositions(surrogate, training_context, phase)
    report = {
        "phase": phase,
        "context": context,
        "eta": None if eta is None else np.asarray(eta, dtype=np.float64),
        "training_mask": None if eta is None else _training_mask(eta, surrogate.eta_samples),
        "compositions": np.asarray(compositions, dtype=np.float64),
        "matrices": matrices,
        "valid": diagnostics["valid"],
        "finite": diagnostics["finite"],
        "eigenvalues": diagnostics["eigenvalues"],
        "nearest_training_distance": _nearest_distances(compositions, training),
        "training_compositions": training.copy(),
        "fallback": _fallback_mask(surrogate, compositions, phase, training_context),
        "truth": None,
        "failures": [],
    }
    if thermodynamics is None:
        return report
    truth, failures = _truth_matrices(
        thermodynamics, compositions, surrogate.temperature, phase, policy, context
    )
    truth_diagnostics = _matrix_validity_diagnostics(
        truth, eigen_imag_tol=eigen_imag_tol, eigen_real_min=eigen_real_min
    )
    absolute = np.abs(matrices - truth)
    relative = absolute / np.maximum(np.abs(truth), relative_error_floor)
    numerator = np.linalg.norm(matrices - truth, axis=(1, 2))
    denominator = np.maximum(np.linalg.norm(truth, axis=(1, 2)), relative_error_floor)
    report["truth"] = {
        "matrices": truth,
        "valid": truth_diagnostics["valid"],
        "eigenvalues": truth_diagnostics["eigenvalues"],
        "absolute_error": absolute,
        "relative_error": relative,
        "matrix_relative_error": numerator / denominator,
    }
    report["failures"] = failures
    return report


def evaluate_diffusivity_diagnostics(
    surrogate,
    *,
    thermodynamics=None,
    phases=None,
    interface_eta_count=101,
    bulk_grid_counts=(51, 51),
    bulk_axes=None,
    relative_error_floor=1e-300,
    eigen_imag_tol=1e-12,
    eigen_real_min=1e-14,
    on_truth_error="raise",
):
    """Evaluate interface and bulk diffusivity diagnostics for a ternary surrogate.

    ``MergedPhaseDiffusivitySurrogate`` objects produce a bulk-only report.
    Matrix validity follows the positive-real-eigenvalue assumptions used by
    the ternary Illingworth solver.
    """
    if not isinstance(surrogate, (TernaryMovingBoundaryThermodynamicsSurrogate, MergedPhaseDiffusivitySurrogate)):
        raise TypeError("Diffusivity diagnostics require a supported ternary moving-boundary surrogate.")
    policy = _truth_error_policy(on_truth_error)
    floor = float(relative_error_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("relative_error_floor must be positive and finite.")
    selected_phases = _surrogate_phases(surrogate, phases)
    failures = []
    interface = None
    if isinstance(surrogate, TernaryMovingBoundaryThermodynamicsSurrogate):
        count = _positive_int(interface_eta_count, "interface_eta_count")
        if count < 2:
            raise ValueError("interface_eta_count must be at least 2.")
        eta = np.linspace(surrogate.eta_bounds[0], surrogate.eta_bounds[1], count)
        phase_reports = {}
        for phase in selected_phases:
            compositions = np.column_stack(
                [np.interp(eta, surrogate.eta_samples, surrogate.tieline_compositions[phase][:, i]) for i in range(2)]
            )
            phase_report = _diffusivity_phase_report(
                surrogate, phase, "interface", compositions, thermodynamics, policy, floor,
                eigen_imag_tol, eigen_real_min, eta=eta,
            )
            phase_reports[phase] = phase_report
            failures.extend(phase_report["failures"])
        interface = {"eta": eta, "phases": phase_reports}

    axes, points, grid_indices, grid_shape = _bulk_grid(
        surrogate, selected_phases, bulk_grid_counts, bulk_axes
    )
    bulk_phases = {}
    for phase in selected_phases:
        phase_report = _diffusivity_phase_report(
            surrogate, phase, "general", points, thermodynamics, policy, floor,
            eigen_imag_tol, eigen_real_min,
        )
        bulk_phases[phase] = phase_report
        failures.extend(phase_report["failures"])
    bulk = {
        "axes": axes,
        "grid_indices": grid_indices,
        "grid_shape": grid_shape,
        "points": points,
        "phases": bulk_phases,
    }
    phase_reports = list(bulk_phases.values())
    if interface is not None:
        phase_reports += list(interface["phases"].values())
    valid = np.concatenate([item["valid"] for item in phase_reports])
    relative = []
    for item in phase_reports:
        if item["truth"] is not None:
            relative.extend(item["truth"]["matrix_relative_error"][np.isfinite(item["truth"]["matrix_relative_error"])] )
    return {
        "kind": "diffusivity",
        "elements": tuple(surrogate.elements),
        "phases": selected_phases,
        "temperature": float(surrogate.temperature),
        "interpolation": str(surrogate.diffusivityInterpolation),
        "interface": interface,
        "bulk": bulk,
        "failures": failures,
        "summary": {
            "truth_evaluated": thermodynamics is not None,
            "sample_count": int(valid.size),
            "invalid_count": int(np.count_nonzero(~valid)),
            "failure_count": len(failures),
            "max_matrix_relative_error": float(np.max(relative)) if relative else np.nan,
        },
    }


def _require_plotly(renderer="browser"):
    """Import Plotly lazily and configure the requested default renderer."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Plotly surrogate diagnostics require the optional dependency; install 'kawin[diagnostics]'."
        ) from exc
    if renderer is not None:
        pio.renderers.default = str(renderer)
    return go, make_subplots


def _selected_hover_fields(kind, hover_fields):
    presets = _HOVER_PRESETS[kind]
    if isinstance(hover_fields, str):
        if hover_fields not in presets:
            raise ValueError(f"Unknown hover preset '{hover_fields}'; expected one of {tuple(presets)}.")
        return presets[hover_fields]
    if not isinstance(hover_fields, Sequence):
        raise TypeError("hover_fields must be a preset name or a sequence of field names.")
    fields = tuple(str(field) for field in hover_fields)
    allowed = set(presets["all"])
    unknown = [field for field in fields if field not in allowed]
    if unknown:
        raise ValueError(f"Unsupported hover fields {unknown}; valid fields are {tuple(presets['all'])}.")
    return fields


def _broadcast_field(value, count):
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(count, array.item(), dtype=object if array.dtype.kind in "OUS" else None)
    if array.shape[0] != count:
        raise ValueError(f"Hover field has length {array.shape[0]}, expected {count}.")
    return array


def _hover_payload(available, selected, count, hover_format):
    """Pack only selected point fields into Plotly customdata and its template."""
    names = [name for name in selected if name in available]
    arrays = [_broadcast_field(available[name], count) for name in names]
    customdata = np.column_stack(arrays) if arrays else None
    lines = []
    for index, name in enumerate(names):
        label = _FIELD_LABELS[name]
        if name in _TEXT_FIELDS or name in _BOOL_FIELDS:
            lines.append(f"{label}: %{{customdata[{index}]}}")
        else:
            lines.append(f"{label}: %{{customdata[{index}]:{hover_format}}}")
    return customdata, "<br>".join(lines) + "<extra></extra>"


def _composition_fields(compositions):
    compositions = np.asarray(compositions, dtype=np.float64)
    return {
        "x_ref": 1.0 - np.sum(compositions, axis=1),
        "x1": compositions[:, 0],
        "x2": compositions[:, 1],
    }


def _ternary_coordinates(compositions):
    """Map reference-first ternary compositions to the diagnostic corner order."""
    fields = _composition_fields(compositions)
    return {"a": fields["x2"], "b": fields["x_ref"], "c": fields["x1"]}


def plot_diffusivity_leave_one_out(report, phase, *, hover_format=".6g", renderer="browser"):
    """Plot held-out bulk samples on a ternary with full per-point diagnostics.

    Marker color is the base-10 log of Frobenius relative matrix error, with
    zero displayed at a 1e-16 floor. Circles mark valid predictions, diamonds
    explicit legacy nearest fallbacks, crosses invalid predicted matrices, open
    squares insufficient retained fit support, and black X markers unexpected
    refit failures. The returned Plotly figure is not displayed automatically.
    """
    if report.get("kind") != "diffusivity_leave_one_out" or phase not in report.get("phase_reports", {}):
        raise ValueError(f"Leave-one-out diffusivity report does not contain phase '{phase}'.")
    go, _ = _require_plotly(renderer)
    values = report["phase_reports"][phase]
    points = np.asarray(values["compositions"], dtype=np.float64)
    errors = np.asarray(values["matrix_relative_error"], dtype=np.float64)
    failures = {int(item["index"]): str(item["error"]) for item in values["failures"]}
    statuses = np.asarray(values["prediction_status"], dtype=object)
    failed_mask = statuses == "refit_failure"
    support_mask = statuses == "insufficient_fit_support"
    finite = np.isfinite(errors) & ~failed_mask & ~support_mask
    log_errors = np.full(len(points), np.nan, dtype=np.float64)
    log_errors[finite] = np.log10(np.maximum(errors[finite], 1e-16))
    log_errors[~finite & ~failed_mask & ~support_mask] = float(np.max(log_errors[finite]) + 1.0) if np.any(finite) else 0.0
    displayed_colors = log_errors[~failed_mask & ~support_mask]
    color_min = float(np.min(displayed_colors)) if displayed_colors.size else -16.0
    color_max = float(np.max(displayed_colors)) if displayed_colors.size else -15.0
    if color_min == color_max:
        color_min -= 0.5
        color_max += 0.5

    symbols = []
    hover = []

    def matrix_text(matrix):
        return "[" + ", ".join(format(float(value), hover_format) for value in matrix[0]) + "]<br>[" + ", ".join(
            format(float(value), hover_format) for value in matrix[1]
        ) + "]"

    def eigen_text(eigenvalues):
        return ", ".join(
            format(float(value.real), hover_format)
            + (f" {format(float(value.imag), '+' + hover_format)}i" if abs(value.imag) > 0 else "")
            for value in eigenvalues
        )

    for row, index in enumerate(values["sample_indices"]):
        failed = statuses[row] == "refit_failure"
        invalid = statuses[row] == "invalid_prediction"
        symbols.append("x" if failed else "square-open" if support_mask[row] else "cross" if invalid else "diamond" if values["fallback"][row] else "circle")
        composition = points[row]
        fields = _composition_fields(composition[None, :])
        lines = [
            f"<b>{escape(str(phase))} · sample {int(index)}</b>",
            f"{escape(str(report['elements'][0]))}: {format(fields['x_ref'][0], hover_format)}; "
            f"{escape(str(report['elements'][1]))}: {format(composition[0], hover_format)}; "
            f"{escape(str(report['elements'][2]))}: {format(composition[1], hover_format)}",
            f"Actual D (m²/s):<br>{matrix_text(values['actual_matrices'][row])}",
            f"Predicted D (m²/s):<br>{matrix_text(values['predicted_matrices'][row])}",
            f"Absolute error (m²/s):<br>{matrix_text(values['absolute_error'][row])}",
            f"Relative component error:<br>{matrix_text(values['relative_error'][row])}",
            f"Frobenius relative error: {format(errors[row], hover_format)}",
            f"Actual valid: {bool(values['actual_valid'][row])}; predicted valid: {bool(values['predicted_valid'][row])}",
            f"Actual eigenvalues (m²/s): {eigen_text(values['actual_eigenvalues'][row])}",
            f"Predicted eigenvalues (m²/s): {eigen_text(values['predicted_eigenvalues'][row])}",
            f"Source interpolation: {escape(str(report['source_interpolation']))}; source validity policy: {escape(str(report.get('source_validity_policy', 'unknown')))}; context: {escape(str(report['context']))}",
            f"Nearest retained sample distance: {format(values['nearest_training_distance'][row], hover_format)}",
            f"Refit: {escape(str(values['refit_interpolation'][row]))}; prediction status: {escape(str(statuses[row]))}; nearest fallback: {bool(values['fallback'][row])}",
            f"LOO fit samples: {int(values['loo_fit_sample_count'])} / source fit samples: {int(values['source_sample_count'])}",
        ]
        if values.get("refit_note") is not None and values["refit_note"][row] is not None:
            lines.append(f"Refit note: {escape(str(values['refit_note'][row]))}")
        if support_mask[row]:
            lines.append("Support reason: held-out composition is outside the retained 2-D fit hull.")
        if failed:
            lines.append(f"Failure: {escape(failures[int(index)])}")
        hover.append("<br>".join(lines))

    fig = go.Figure()
    normal_mask = ~failed_mask & ~support_mask
    if np.any(normal_mask):
        good = normal_mask
        fig.add_trace(go.Scatterternary(
            **_ternary_coordinates(points[good]),
            mode="markers",
            name=f"{phase} held-out samples",
            marker={
                "size": 8,
                "symbol": np.asarray(symbols, dtype=object)[good],
                "color": log_errors[good],
                "colorscale": "Viridis",
                "cmin": color_min,
                "cmax": color_max,
                "showscale": True,
                "colorbar": {"title": "log10 relative error"},
            },
            text=np.asarray(hover, dtype=object)[good],
            hovertemplate="%{text}<extra></extra>",
        ))
    if np.any(failed_mask):
        fig.add_trace(go.Scatterternary(
            **_ternary_coordinates(points[failed_mask]),
            mode="markers",
            name="Failed refit",
            marker={"size": 10, "symbol": "x", "color": "black"},
            text=np.asarray(hover, dtype=object)[failed_mask],
            hovertemplate="%{text}<extra></extra>",
        ))
    if np.any(support_mask):
        fig.add_trace(go.Scatterternary(
            **_ternary_coordinates(points[support_mask]),
            mode="markers", name="Insufficient fit support",
            marker={"size": 10, "symbol": "square-open", "color": "#9467bd"},
            text=np.asarray(hover, dtype=object)[support_mask],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig.update_layout(
        title=f"Diffusivity leave-one-out: {phase} at {report['temperature']:g} K",
        template="plotly_white",
        width=1050,
        height=850,
        margin={"t": 100, "b": 50, "l": 50, "r": 140},
        uirevision=f"diffusivity-leave-one-out-{phase}",
        ternary={
            "sum": 1,
            "aaxis": {"title": report["elements"][2]},
            "baxis": {"title": report["elements"][0]},
            "caxis": {"title": report["elements"][1]},
        },
        annotations=[{
            "text": "Circle: valid prediction; Diamond: nearest fallback; Cross: invalid predicted matrix; Open square: insufficient fit support; X: failed refit",
            "x": 0.5, "y": 1.03, "xref": "paper", "yref": "paper", "showarrow": False,
        }],
    )
    return fig


def load_calculation_site_fractions(path):
    """Return metadata and ordered records from a TC-Python site-fraction sidecar."""
    with Path(path).open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or rows[0].get("record_type") != "metadata":
        raise ValueError("Site-fraction sidecar must begin with a metadata record.")
    if any(row.get("record_type") != "calculation_site_fractions" for row in rows[1:]):
        raise ValueError("Site-fraction sidecar contains an unexpected record type.")
    return {"metadata": rows[0], "records": rows[1:]}


def _disordered_site_reference(phase_state, elements):
    """Return random occupancies only for equivalent elemental sublattices.

    Variable substitutional sublattices must contain all selected elements;
    any extra constituents there must have zero occupancy. Other sublattices
    may contain only vacant sites. Weighted
    observed occupancies must reproduce the reported phase composition. This
    avoids inventing a random reference for more complex sublattice models.
    """
    sublattices = phase_state.get("site_fractions")
    composition = phase_state.get("phase_composition")
    if not sublattices or composition is None or len(composition) != len(elements):
        return None
    if any(value is None or not np.isfinite(value) for value in composition):
        return None
    element_names = tuple(str(element).upper() for element in elements)
    x = dict(zip(element_names, (float(value) for value in composition)))
    if not np.isclose(sum(x.values()), 1.0, atol=1e-6):
        return None
    variable = []
    for sublattice in sublattices:
        constituents = sublattice.get("constituents")
        if not constituents:
            return None
        values = {str(name).upper(): value for name, value in constituents.items()}
        selected = set(values) & set(element_names)
        if selected:
            if selected != set(element_names):
                return None
            if any(value is None or abs(value) > 1e-8 for name, value in values.items()
                   if name not in element_names):
                return None
            variable.append(sublattice)
        else:
            if ("VA" not in values or values["VA"] is None
                    or not np.isclose(values["VA"], 1.0, atol=1e-8)
                    or any(value is None or abs(value) > 1e-8
                           for name, value in values.items() if name != "VA")):
                return None
    if not variable:
        return None
    ratios = [sublattice.get("site_ratio") for sublattice in variable]
    if any(value is None or not np.isfinite(value) or value <= 0 for value in ratios):
        return None
    for element in element_names:
        if any(next((value for name, value in sublattice["constituents"].items()
                     if str(name).upper() == element), None) is None for sublattice in variable):
            return None
        weighted = sum(
            float(ratio) * float(next(value for name, value in sublattice["constituents"].items()
                                      if str(name).upper() == element))
            for sublattice, ratio in zip(variable, ratios)
        ) / sum(ratios)
        if not np.isclose(weighted, x[element], rtol=1e-4, atol=1e-4):
            return None
    return {
        (int(sublattice["sublattice"]), str(name).upper()): x[str(name).upper()]
        for sublattice in variable
        for name in sublattice["constituents"]
        if str(name).upper() in element_names
    }


def plot_calculation_site_fractions(
    source, phase, *, kinds=("equilibrium", "kinetics"), hover_format=".6g", renderer="browser",
):
    """Plot construction-time site occupancies and disordered deviations.

    ``source`` is a path to the JSON Lines sidecar or the result of
    :func:`load_calculation_site_fractions`. ``phase`` matches the base phase
    name, so BCC_B2 includes BCC_B2#1, #2, etc. A dropdown selects a
    summary of maximum ordering deviation or a sublattice/constituent. Color
    shows actual minus random occupancy for a selected constituent when a
    composition-matched disordered reference is justified by equivalent
    elemental sublattices; otherwise it shows actual occupancy. Hover gives
    the full recorded phase constitution, result and input compositions, and
    composition-set identity. Circles denote equilibrium calculations and
    diamonds denote kinetics calculations. Failed calculations have no site
    fractions and are excluded from the markers.
    """
    data = load_calculation_site_fractions(source) if isinstance(source, (str, Path)) else source
    metadata = data["metadata"]
    elements = tuple(metadata["element_order"])
    selected_kinds = {kinds} if isinstance(kinds, str) else {str(kind) for kind in kinds}
    base = str(phase).split("#", 1)[0].upper()
    entries = []
    for record in data["records"]:
        if record.get("status") != "ok" or record.get("kind") not in selected_kinds:
            continue
        for state in record.get("phases", []):
            if str(state["phase"]).split("#", 1)[0].upper() != base or not state.get("site_fractions"):
                continue
            entries.append((record, state, _disordered_site_reference(state, elements)))
    if not entries:
        raise ValueError(f"No recorded site fractions for phase '{phase}' and kinds {sorted(selected_kinds)}.")

    def format_value(value):
        return "unavailable" if value is None else format(float(value), hover_format)

    points = []
    hover = []
    site_keys = set()
    marker_symbols = []
    for record, state, reference in entries:
        point = np.asarray(record["input_full_composition"], dtype=np.float64)
        points.append(point)
        marker_symbols.append({"equilibrium": "circle", "kinetics": "diamond", "driving_force": "square"}.get(record["kind"], "cross"))
        lines = [
            f"<b>{escape(str(state['phase']))} · calculation {record['calculation_index']}</b>",
            f"Kind: {escape(str(record['kind']))}; requested: {escape(str(record['requested_phase']))}",
            "Input: " + ", ".join(f"{escape(str(e))}={format_value(v)}" for e, v in zip(elements, point)),
            "Phase: " + ", ".join(f"{escape(str(e))}={format_value(v)}" for e, v in zip(elements, state.get("phase_composition") or [None] * len(elements))),
            f"Phase amount: {format_value(state.get('phase_amount'))}; stable: {bool(state['stable'])}",
            "Stable sets: " + escape(", ".join(record.get("stable_composition_sets") or [])),
        ]
        for sublattice in state["site_fractions"]:
            index = int(sublattice["sublattice"])
            lines.append(f"Sublattice {index} (sites={format_value(sublattice.get('site_ratio'))}):")
            for constituent, actual in (sublattice.get("constituents") or {}).items():
                key = (index, str(constituent).upper())
                site_keys.add(key)
                random = None if reference is None else reference.get(key)
                lines.append(
                    f"&nbsp;&nbsp;{escape(str(constituent))}: {format_value(actual)}; "
                    f"disordered: {format_value(random)}; Δ: "
                    f"{format_value(None if actual is None or random is None else actual - random)}"
                )
        if state.get("diagnostic_errors"):
            lines.append("Diagnostic errors: " + escape(str(state["diagnostic_errors"])))
        hover.append("<br>".join(lines))

    go, _ = _require_plotly(renderer)
    fig = go.Figure()
    ordered_rows = []
    order_magnitude = []
    for row, (_, state, reference) in enumerate(entries):
        if reference is None:
            continue
        observed = {
            (int(site["sublattice"]), str(name).upper()): value
            for site in state["site_fractions"]
            for name, value in (site.get("constituents") or {}).items()
        }
        if any(observed.get(key) is None for key in reference):
            continue
        ordered_rows.append(row)
        order_magnitude.append(max(abs(float(observed[key]) - target) for key, target in reference.items()))
    if ordered_rows:
        ordered_points = np.asarray(points)[ordered_rows]
        fig.add_trace(go.Scatterternary(
            a=ordered_points[:, 2], b=ordered_points[:, 0], c=ordered_points[:, 1],
            mode="markers", name="Maximum ordering deviation", visible=True,
            marker={
                "size": 7, "symbol": np.asarray(marker_symbols, dtype=object)[ordered_rows],
                "color": order_magnitude, "colorscale": "Viridis",
                "cmin": 0, "cmax": 1, "showscale": True,
                "colorbar": {"title": "max |actual − disordered|"},
            },
            text=np.asarray(hover, dtype=object)[ordered_rows],
            hovertemplate="%{text}<extra></extra>",
        ))
    ordered_elements = tuple(str(element).upper() for element in elements)
    ordered_keys = sorted(site_keys, key=lambda key: (key[0], ordered_elements.index(key[1]) if key[1] in ordered_elements else len(elements), key[1]))
    for key in ordered_keys:
        rows, actual_values, random_values = [], [], []
        for row, (_, state, reference) in enumerate(entries):
            sublattice = next((site for site in state["site_fractions"] if int(site["sublattice"]) == key[0]), None)
            constituents = {} if sublattice is None else (sublattice.get("constituents") or {})
            actual = next((value for name, value in constituents.items() if str(name).upper() == key[1]), None)
            if actual is None:
                continue
            rows.append(row)
            actual_values.append(float(actual))
            random_values.append(None if reference is None else reference.get(key))
        if not rows:
            continue
        comparable = all(value is not None for value in random_values)
        color = np.asarray(actual_values) - np.asarray(random_values, dtype=np.float64) if comparable else actual_values
        selected_points = np.asarray(points)[rows]
        fig.add_trace(go.Scatterternary(
            a=selected_points[:, 2], b=selected_points[:, 0], c=selected_points[:, 1],
            mode="markers", name=f"s{key[0]} {key[1]}", visible=len(fig.data) == 0,
            marker={
                "size": 7, "symbol": np.asarray(marker_symbols, dtype=object)[rows],
                "color": color,
                "colorscale": "RdBu_r" if comparable else "Viridis",
                "cmin": -1 if comparable else 0, "cmax": 1,
                "showscale": True,
                "colorbar": {"title": "actual − disordered" if comparable else "site fraction"},
            },
            text=np.asarray(hover, dtype=object)[rows],
            hovertemplate="%{text}<extra></extra>",
        ))
    buttons = [
        {"label": trace.name, "method": "update", "args": [{"visible": [i == index for i in range(len(fig.data))]}]}
        for index, trace in enumerate(fig.data)
    ]
    fig.update_layout(
        title=f"Site fractions: {base} during surrogate construction",
        template="plotly_white", width=1050, height=850,
        margin={"t": 125, "r": 150},
        ternary={
            "sum": 1,
            "aaxis": {"title": elements[2]},
            "baxis": {"title": elements[0]},
            "caxis": {"title": elements[1]},
        },
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.02, "y": 1.10}],
        annotations=[
            {"text": "Sublattice / constituent", "x": 0.02, "y": 1.16, "xref": "paper", "yref": "paper", "showarrow": False},
            {"text": "Circle: equilibrium · Diamond: kinetics · Square: driving force", "x": 0.5, "y": 1.02,
             "xref": "paper", "yref": "paper", "showarrow": False},
        ],
    )
    return fig


def _leave_room_above_ternary(fig, layout_name="ternary", gap=0.07):
    """Lower a ternary subplot so its top-axis label clears its subplot title."""
    ternary = getattr(fig.layout, layout_name)
    lower, upper = ternary.domain.y
    ternary.domain.y = (lower, upper - gap)


def _draw_truth_below_predictions(fig):
    """Place ground-truth traces below surrogate and training traces."""
    def layer(trace):
        legendgroup = str(trace.legendgroup or "")
        if legendgroup == "phase-regions":
            return 0
        if legendgroup == "truth" or legendgroup.startswith("truth-"):
            return 1
        return 2

    fig.data = tuple(sorted(fig.data, key=layer))


def _matrix_hover_fields(phase_report, component=None, source="surrogate"):
    """Flatten matrix diagnostics into point-aligned fields for Plotly hover data."""
    matrices = phase_report["matrices"]
    fields = {
        "source": source,
        "phase": phase_report["phase"],
        "context": phase_report["context"],
        "eta": np.full(len(matrices), np.nan) if phase_report["eta"] is None else phase_report["eta"],
        **_composition_fields(phase_report["compositions"]),
        "d00": matrices[:, 0, 0],
        "d01": matrices[:, 0, 1],
        "d10": matrices[:, 1, 0],
        "d11": matrices[:, 1, 1],
        "valid": phase_report["valid"],
        "eigenvalue_0": np.real(phase_report["eigenvalues"][:, 0]),
        "eigenvalue_1": np.real(phase_report["eigenvalues"][:, 1]),
        "training_distance": phase_report["nearest_training_distance"],
        "training": np.isclose(phase_report["nearest_training_distance"], 0.0, rtol=0.0, atol=1e-12),
        "fallback": phase_report["fallback"],
        "failure": _failure_strings(phase_report["failures"], len(matrices)),
    }
    if component is not None:
        fields["value"] = matrices[:, component[0], component[1]]
    truth = phase_report["truth"]
    if truth is not None:
        fields["truth_valid"] = truth["valid"]
        fields["matrix_relative_error"] = truth["matrix_relative_error"]
        if component is not None:
            i, j = component
            fields["truth_value"] = truth["matrices"][:, i, j]
            fields["absolute_error"] = truth["absolute_error"][:, i, j]
            fields["relative_error"] = truth["relative_error"][:, i, j]
    return fields


def _failure_strings(failures, count):
    values = np.full(count, "", dtype=object)
    for failure in failures:
        index = int(failure["index"])
        if 0 <= index < count:
            values[index] = str(failure["error"])
    return values


def _zoom_ranges(point_sets):
    points = np.concatenate([np.asarray(points, dtype=np.float64) for points in point_sets if points is not None], axis=0)
    lower = np.nanmin(points, axis=0)
    upper = np.nanmax(points, axis=0)
    span = np.maximum(upper - lower, 1e-4)
    padding = 0.08 * span
    return [lower[0] - padding[0], upper[0] + padding[0]], [lower[1] - padding[1], upper[1] + padding[1]]


def plot_tieline_diagnostics(
    report,
    *,
    hover_fields="diagnostic",
    hover_format=".6g",
    display_tieline_count=21,
    phase_region_report=None,
    renderer="browser",
):
    """Create an interactive Plotly tie-line figure.

    The ternary overview places the first independent component at bottom
    right, the second independent component at top, and the reference
    component at bottom left. Ground-truth endpoint samples are shown as
    unconnected markers above the surrogate curves. Plotly's browser renderer
    is selected by default.
    """
    go, make_subplots = _require_plotly(renderer)
    if report.get("kind") != "tieline":
        raise ValueError("plot_tieline_diagnostics requires a tie-line diagnostic report.")
    selected = _selected_hover_fields("tieline", hover_fields)
    phases = report["phases"]
    eta = report["eta"]
    endpoints = report["endpoint_compositions"]
    truth = report["truth"]
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "ternary"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy", "secondary_y": True}]],
        column_widths=(0.56, 0.44),
        row_heights=(0.62, 0.38),
        horizontal_spacing=0.10,
        subplot_titles=("Full ternary overview", "Zoomed composition view", "Endpoint components", "Tie-line geometry / error"),
    )
    colors = {phase: f"#{color}" for phase, color in zip(phases, ("1f77b4", "d62728"))}
    truth_colors = {phase: f"#{color}" for phase, color in zip(phases, ("5b9bd5", "e56b6f"))}

    def endpoint_fields(phase, source, values, partner, phase_index):
        fields = {"source": source, "phase": phase, "eta": eta, **_composition_fields(values)}
        fields["partner_x1"] = partner[:, 0]
        fields["partner_x2"] = partner[:, 1]
        fields["training"] = report["training_mask"]
        fields["tieline_length"] = report["tieline_length"]
        fields["orientation"] = report["orientation_degrees"]
        if report["probe_compositions"] is not None:
            fields["probe_ref"] = 1.0 - np.sum(report["probe_compositions"], axis=1)
            fields["probe_x1"] = report["probe_compositions"][:, 0]
            fields["probe_x2"] = report["probe_compositions"][:, 1]
        if truth is not None:
            fields["endpoint_error"] = truth["endpoint_error_norm"][phase]
            fields["length_error"] = truth["length_absolute_error"]
            fields["angle_error"] = truth["orientation_absolute_error"]
            fields["failure"] = _failure_strings(report["failures"], len(eta))
        return fields

    for index, phase in enumerate(phases):
        values = endpoints[phase]
        partner = endpoints[phases[1 - index]]
        fields = endpoint_fields(phase, "surrogate", values, partner, index)
        custom, template = _hover_payload(fields, selected, len(eta), hover_format)
        ternary_values = _ternary_coordinates(values)
        fig.add_trace(
            go.Scatterternary(
                **ternary_values, mode="lines",
                name=f"{phase} surrogate endpoints", legendgroup="surrogate",
                line={"color": colors[phase]}, customdata=custom, hovertemplate=template,
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=values[:, 0], y=values[:, 1], mode="lines",
                name=f"{phase} surrogate endpoints", legendgroup="surrogate", showlegend=False,
                line={"color": colors[phase]}, customdata=custom, hovertemplate=template,
            ), row=1, col=2,
        )
        train = report["training_mask"]
        train_fields = {key: _broadcast_field(value, len(eta))[train] for key, value in fields.items()}
        train_custom, train_template = _hover_payload(train_fields, selected, int(np.count_nonzero(train)), hover_format)
        fig.add_trace(
            go.Scatterternary(
                **_ternary_coordinates(values[train]), mode="markers",
                name=f"{phase} training endpoints", legendgroup="training",
                marker={"color": colors[phase], "symbol": "diamond", "size": 8},
                customdata=train_custom, hovertemplate=train_template,
            ), row=1, col=1,
        )
        for component in range(2):
            component_fields = dict(fields)
            custom_component, template_component = _hover_payload(component_fields, selected, len(eta), hover_format)
            fig.add_trace(
                go.Scatter(
                    x=eta, y=values[:, component], mode="lines",
                    name=f"{phase} X({report['elements'][component + 1]})",
                    legendgroup=f"components-{phase}", line={"color": colors[phase], "dash": "solid" if component == 0 else "dot"},
                    customdata=custom_component, hovertemplate=template_component,
                ), row=2, col=1,
            )

    display_count = min(_positive_int(display_tieline_count, "display_tieline_count"), len(eta))
    display_indices = np.unique(np.linspace(0, len(eta) - 1, display_count).round().astype(int))
    for target_col, trace_type in ((1, "ternary"), (2, "xy")):
        for position, index in enumerate(display_indices):
            left, right = endpoints[phases[0]][index], endpoints[phases[1]][index]
            segment_compositions = np.vstack((left, right))
            segment_fields = {
                "source": "surrogate tie line",
                "phase": np.asarray(phases, dtype=object),
                "eta": float(eta[index]),
                **_composition_fields(segment_compositions),
                "partner_x1": segment_compositions[::-1, 0],
                "partner_x2": segment_compositions[::-1, 1],
                "training": bool(report["training_mask"][index]),
                "tieline_length": float(report["tieline_length"][index]),
                "orientation": float(report["orientation_degrees"][index]),
            }
            segment_custom, segment_template = _hover_payload(segment_fields, selected, 2, hover_format)
            if trace_type == "ternary":
                trace = go.Scatterternary(
                    **_ternary_coordinates(segment_compositions),
                    mode="lines", line={"color": "rgba(80,80,80,0.35)", "width": 1},
                    name="Surrogate tie lines", legendgroup="tie-lines", showlegend=position == 0,
                    customdata=segment_custom, hovertemplate=segment_template,
                )
            else:
                trace = go.Scatter(
                    x=[left[0], right[0]], y=[left[1], right[1]], mode="lines",
                    line={"color": "rgba(80,80,80,0.35)", "width": 1},
                    name="Surrogate tie lines", legendgroup="tie-lines", showlegend=False,
                    customdata=segment_custom, hovertemplate=segment_template,
                )
            fig.add_trace(trace, row=1, col=target_col)

    if report["probe_compositions"] is not None:
        probe = report["probe_compositions"]
        probe_fields = {
            "source": "probe path",
            "phase": "bulk",
            "eta": eta,
            **_composition_fields(probe),
            "probe_ref": 1.0 - np.sum(probe, axis=1),
            "probe_x1": probe[:, 0],
            "probe_x2": probe[:, 1],
            "training": report["training_mask"],
            "failure": _failure_strings(report["failures"], len(eta)),
        }
        probe_custom, probe_template = _hover_payload(
            probe_fields, selected, len(eta), hover_format
        )
        fig.add_trace(
            go.Scatterternary(
                **_ternary_coordinates(probe), mode="lines+markers",
                name="Probe path", legendgroup="probe", marker={"size": 4}, line={"dash": "dash"},
                customdata=probe_custom, hovertemplate=probe_template,
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=probe[:, 0], y=probe[:, 1], mode="lines+markers", name="Probe path",
                       legendgroup="probe", showlegend=False, marker={"size": 4}, line={"dash": "dash"},
                       customdata=probe_custom, hovertemplate=probe_template),
            row=1, col=2,
        )

    if truth is not None:
        for phase_index, phase in enumerate(phases):
            values = truth["endpoint_compositions"][phase]
            partner = truth["endpoint_compositions"][phases[1 - phase_index]]
            truth_fields = endpoint_fields(phase, "ground truth", values, partner, phase_index)
            truth_fields.update(_composition_fields(values))
            truth_custom, truth_template = _hover_payload(truth_fields, selected, len(eta), hover_format)
            fig.add_trace(
                go.Scatterternary(
                    **_ternary_coordinates(values), mode="markers",
                    name=f"{phase} truth endpoints", legendgroup="truth",
                    marker={"color": truth_colors[phase], "symbol": "x", "size": 7, "line": {"width": 1}},
                    customdata=truth_custom, hovertemplate=truth_template,
                ), row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=values[:, 0], y=values[:, 1], mode="markers", name=f"{phase} truth endpoints",
                           legendgroup="truth", showlegend=False,
                           marker={"color": truth_colors[phase], "symbol": "x", "size": 7, "line": {"width": 1}},
                           customdata=truth_custom, hovertemplate=truth_template),
                row=1, col=2,
            )
            for component in range(2):
                fig.add_trace(
                    go.Scatter(x=eta, y=values[:, component], mode="markers", name=f"{phase} truth X({report['elements'][component + 1]})",
                               legendgroup="truth",
                               marker={"color": truth_colors[phase], "symbol": "x" if component == 0 else "cross", "size": 6, "line": {"width": 1}},
                               customdata=truth_custom, hovertemplate=truth_template),
                    row=2, col=1,
                )
        fig.add_trace(go.Scatter(x=eta, y=truth["length_absolute_error"], name="Length error", legendgroup="truth-error"), row=2, col=2)
        fig.add_trace(go.Scatter(x=eta, y=truth["orientation_absolute_error"], name="Angle error", legendgroup="truth-error"), row=2, col=2, secondary_y=True)
        fig.update_yaxes(title_text="Length error", row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Angle error (deg)", row=2, col=2, secondary_y=True)
    else:
        fig.add_trace(go.Scatter(x=eta, y=report["tieline_length"], name="Tie-line length", legendgroup="geometry"), row=2, col=2)
        fig.add_trace(go.Scatter(x=eta, y=report["orientation_degrees"], name="Orientation", legendgroup="geometry"), row=2, col=2, secondary_y=True)
        fig.update_yaxes(title_text="Tie-line length", row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Orientation (deg)", row=2, col=2, secondary_y=True)

    if phase_region_report is not None:
        points = _independent_points(phase_region_report["points"], "phase_region_report points")
        labels = np.asarray(phase_region_report["labels"], dtype=object)
        fig.add_trace(
            go.Scatterternary(
                **_ternary_coordinates(points), mode="markers",
                marker={"size": 4, "opacity": 0.25}, text=labels,
                hovertemplate="Stable region: %{text}<extra></extra>", name="Phase regions", legendgroup="phase-regions",
            ), row=1, col=1,
        )

    x_range, y_range = _zoom_ranges([*endpoints.values(), report["probe_compositions"]])
    fig.update_xaxes(title_text=f"X({report['elements'][1]})", range=x_range, row=1, col=2)
    fig.update_yaxes(title_text=f"X({report['elements'][2]})", range=y_range, scaleanchor="x", scaleratio=1, row=1, col=2)
    fig.update_xaxes(title_text="eta", row=2, col=1)
    fig.update_yaxes(title_text="Mole fraction", row=2, col=1)
    fig.update_xaxes(title_text="eta", row=2, col=2)
    fig.update_layout(
        title=f"Tie-line surrogate diagnostics: {phases[0]} | {phases[1]} at {report['temperature']:g} K",
        template="plotly_white", width=1250, height=950, margin={"t": 130}, uirevision="tieline-diagnostics",
        ternary={"sum": 1, "aaxis": {"title": report["elements"][2]}, "baxis": {"title": report["elements"][0]}, "caxis": {"title": report["elements"][1]}},
    )
    _leave_room_above_ternary(fig)
    _draw_truth_below_predictions(fig)
    return fig


def plot_interface_diffusivity_diagnostics(
    report,
    *,
    hover_fields="diagnostic",
    hover_format=".6g",
    renderer="browser",
):
    """Create an interface diffusivity figure with one y-axis per phase.

    The two phase axes are independently scaled in every matrix-entry subplot
    while sharing that subplot's eta axis. Ground-truth samples are shown as
    unconnected markers above the surrogate curves. Plotly's browser renderer
    is selected by default.
    """
    go, make_subplots = _require_plotly(renderer)
    interface = report.get("interface")
    if report.get("kind") != "diffusivity" or interface is None:
        raise ValueError("Interface diffusivity plotting requires a full diffusivity report with interface data.")
    selected = _selected_hover_fields("diffusivity", hover_fields)
    phase_items = tuple(interface["phases"].items())
    if len(phase_items) > 2:
        raise ValueError("Interface diffusivity plots support at most two phase-specific y-axes.")
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=("D[0,0]", "D[0,1]", "D[1,0]", "D[1,1]"),
    )
    colors = ("#1f77b4", "#d62728", "#2ca02c")
    truth_colors = ("#5b9bd5", "#e56b6f", "#70ad75")
    for phase_index, (phase, phase_report) in enumerate(phase_items):
        secondary_y = phase_index == 1
        for flat_index, component in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            row, col = divmod(flat_index, 2)
            fields = _matrix_hover_fields(phase_report, component)
            custom, template = _hover_payload(fields, selected, len(interface["eta"]), hover_format)
            fig.add_trace(
                go.Scatter(
                    x=interface["eta"], y=phase_report["matrices"][:, component[0], component[1]], mode="lines",
                    name=f"{phase} surrogate", legendgroup=f"surrogate-{phase}", showlegend=flat_index == 0,
                    line={"color": colors[phase_index % len(colors)]}, customdata=custom, hovertemplate=template,
                ), row=row + 1, col=col + 1, secondary_y=secondary_y,
            )
            training = phase_report["training_mask"]
            if np.any(training):
                training_fields = {
                    key: _broadcast_field(value, len(interface["eta"]))[training]
                    for key, value in fields.items()
                }
                training_custom, training_template = _hover_payload(
                    training_fields, selected, int(np.count_nonzero(training)), hover_format
                )
                fig.add_trace(
                    go.Scatter(
                        x=interface["eta"][training],
                        y=phase_report["matrices"][training, component[0], component[1]],
                        mode="markers", marker={"symbol": "diamond", "size": 7, "color": colors[phase_index % len(colors)]},
                        name=f"{phase} training", legendgroup=f"training-{phase}", showlegend=flat_index == 0,
                        customdata=training_custom, hovertemplate=training_template,
                    ), row=row + 1, col=col + 1, secondary_y=secondary_y,
                )
            invalid = ~phase_report["valid"]
            if np.any(invalid):
                invalid_fields = {
                    key: _broadcast_field(value, len(interface["eta"]))[invalid]
                    for key, value in fields.items()
                }
                invalid_custom, invalid_template = _hover_payload(
                    invalid_fields, selected, int(np.count_nonzero(invalid)), hover_format
                )
                fig.add_trace(
                    go.Scatter(
                        x=interface["eta"][invalid], y=phase_report["matrices"][invalid, component[0], component[1]],
                        mode="markers", marker={"symbol": "x", "size": 10, "color": "black"},
                        name=f"{phase} invalid", legendgroup=f"invalid-{phase}", showlegend=flat_index == 0,
                        customdata=invalid_custom, hovertemplate=invalid_template,
                    ), row=row + 1, col=col + 1, secondary_y=secondary_y,
                )
            truth = phase_report["truth"]
            if truth is not None:
                truth_fields = _matrix_hover_fields(phase_report, component, source="ground truth")
                truth_fields["value"] = truth["matrices"][:, component[0], component[1]]
                truth_fields["d00"] = truth["matrices"][:, 0, 0]
                truth_fields["d01"] = truth["matrices"][:, 0, 1]
                truth_fields["d10"] = truth["matrices"][:, 1, 0]
                truth_fields["d11"] = truth["matrices"][:, 1, 1]
                truth_custom, truth_template = _hover_payload(
                    truth_fields, selected, len(interface["eta"]), hover_format
                )
                error_fields = _matrix_hover_fields(phase_report, component, source="absolute error")
                error_fields["value"] = truth["absolute_error"][:, component[0], component[1]]
                error_custom, error_template = _hover_payload(
                    error_fields, selected, len(interface["eta"]), hover_format
                )
                fig.add_trace(
                    go.Scatter(
                        x=interface["eta"], y=truth["matrices"][:, component[0], component[1]], mode="markers",
                        name=f"{phase} truth", legendgroup=f"truth-{phase}", showlegend=flat_index == 0,
                        marker={"color": truth_colors[phase_index % len(truth_colors)], "symbol": "x", "size": 6, "line": {"width": 1}},
                        customdata=truth_custom, hovertemplate=truth_template,
                    ), row=row + 1, col=col + 1, secondary_y=secondary_y,
                )
                fig.add_trace(
                    go.Scatter(
                        x=interface["eta"], y=truth["absolute_error"][:, component[0], component[1]], mode="lines",
                        name=f"{phase} absolute error", legendgroup=f"error-{phase}", showlegend=flat_index == 0,
                        line={"color": colors[phase_index % len(colors)], "dash": "dot"}, visible="legendonly",
                        customdata=error_custom, hovertemplate=error_template,
                    ), row=row + 1, col=col + 1, secondary_y=secondary_y,
                )
    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(title_text="eta", row=row, col=col)
            for phase_index, (phase, _) in enumerate(phase_items):
                color = colors[phase_index % len(colors)]
                fig.update_yaxes(
                    title_text=f"{phase} (m^2/s)",
                    exponentformat="e",
                    title_font_color=color,
                    tickfont_color=color,
                    row=row,
                    col=col,
                    secondary_y=phase_index == 1,
                )
    fig.update_layout(
        title=f"Interface diffusivity diagnostics at {report['temperature']:g} K",
        template="plotly_white", height=750, uirevision="interface-diffusivity",
    )
    _draw_truth_below_predictions(fig)
    return fig


def _grid_values(bulk, values):
    """Restore flat simplex-valid values to a NaN-masked rectangular grid."""
    grid = np.full(bulk["grid_shape"], np.nan, dtype=np.float64)
    indices = bulk["grid_indices"]
    grid[indices[:, 0], indices[:, 1]] = np.asarray(values, dtype=np.float64)
    return grid.T


def _grid_customdata(bulk, fields, selected, hover_format):
    """Map flat simplex-valid fields back onto a masked rectangular Plotly grid."""
    count = len(bulk["points"])
    chosen = [name for name in selected if name in fields]
    arrays = [_broadcast_field(fields[name], count) for name in chosen]
    custom = np.full((bulk["grid_shape"][1], bulk["grid_shape"][0], len(arrays)), None, dtype=object)
    for field_index, values in enumerate(arrays):
        for point_index, (i, j) in enumerate(bulk["grid_indices"]):
            custom[j, i, field_index] = values[point_index]
    template_lines = []
    for index, name in enumerate(chosen):
        if name in _TEXT_FIELDS or name in _BOOL_FIELDS:
            template_lines.append(f"{_FIELD_LABELS[name]}: %{{customdata[{index}]}}")
        else:
            template_lines.append(f"{_FIELD_LABELS[name]}: %{{customdata[{index}]:{hover_format}}}")
    return custom, "<br>".join(template_lines) + "<extra></extra>"


def _subplot_colorbar(fig, row, col, title):
    """Position a compact colorbar beside its owning Cartesian or ternary subplot."""
    subplot = fig.get_subplot(row, col)
    if hasattr(subplot, "xaxis"):
        x_domain = subplot.xaxis.domain
        y_domain = subplot.yaxis.domain
    else:
        x_domain = subplot.domain.x
        y_domain = subplot.domain.y
    return {
        "title": {"text": title},
        "x": x_domain[1] + 0.006,
        "xanchor": "left",
        "y": 0.5 * (y_domain[0] + y_domain[1]),
        "yanchor": "middle",
        "len": y_domain[1] - y_domain[0],
        "lenmode": "fraction",
        "thickness": 10,
        "thicknessmode": "pixels",
        "outlinewidth": 0.5,
        "xpad": 0,
    }


def _validate_diffusivity_color_scale(value):
    """Validate the color transform used for diffusivity fields."""
    mode = str(value).lower()
    allowed = ("auto", "linear", "log", "symlog")
    if mode not in allowed:
        raise ValueError(
            "diffusivity_color_scale must be one of 'auto', 'linear', 'log', or 'symlog'."
        )
    return mode


def _validate_symlog_linthresh(value):
    """Validate a positive finite signed-log linear-core threshold."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("symlog_linthresh must be a positive finite value.")
    return value


def _signed_log10_color_values(values, linthresh):
    """Map finite signed values to a symmetric-log color coordinate."""
    values = np.asarray(values, dtype=np.float64)
    transformed = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    magnitude = np.abs(values[finite])
    linear = magnitude <= linthresh
    finite_values = np.empty(magnitude.shape, dtype=np.float64)
    finite_values[linear] = magnitude[linear] / linthresh
    finite_values[~linear] = 1.0 + np.log10(magnitude[~linear] / linthresh)
    transformed[finite] = np.sign(values[finite]) * finite_values
    return transformed


def _select_symlog_linthresh(values, explicit):
    """Select a stable signed-log threshold from the plotted finite values."""
    explicit = _validate_symlog_linthresh(explicit)
    if explicit is not None:
        return explicit
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    maximum = float(np.max(np.abs(finite))) if finite.size else 0.0
    if maximum == 0.0:
        return 1.0
    exponent = np.floor(np.log10(maximum)) - 3.0
    return max(float(10.0**exponent), float(np.finfo(np.float64).tiny))


def _symlog_color_ticks(values, linthresh):
    """Return symmetric original-value ticks for a signed-log colorbar."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    maximum = float(np.max(np.abs(finite))) if finite.size else 0.0
    if maximum == 0.0:
        return np.asarray([0.0], dtype=np.float64)
    ratio = maximum / linthresh
    if ratio <= 1.0:
        magnitudes = np.asarray([maximum], dtype=np.float64)
    else:
        power_count = int(np.floor(np.log10(ratio))) + 1
        powers = np.arange(power_count, dtype=np.int64)
        if len(powers) > 5:
            powers = powers[np.unique(np.linspace(0, len(powers) - 1, 5).round().astype(int))]
        magnitudes = linthresh * np.power(10.0, powers)
        if magnitudes[-1] < maximum:
            magnitudes = np.append(magnitudes, maximum)
    return np.concatenate((-magnitudes[::-1], np.asarray([0.0]), magnitudes))


def _log_color_ticks(values):
    """Return readable positive original-value ticks for a log colorbar."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return np.asarray([], dtype=np.float64)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower == upper:
        return np.asarray([lower], dtype=np.float64)
    exponents = np.arange(
        int(np.floor(np.log10(lower))),
        int(np.ceil(np.log10(upper))) + 1,
        dtype=np.int64,
    )
    if len(exponents) > 7:
        exponents = exponents[np.unique(np.linspace(0, len(exponents) - 1, 7).round().astype(int))]
    ticks = np.power(10.0, exponents)
    ticks = ticks[(ticks >= lower) & (ticks <= upper)]
    return np.unique(np.concatenate((ticks, np.asarray([lower, upper]))))


def _format_colorbar_tick(value):
    """Format an original diffusivity value for a transformed colorbar."""
    return "0" if value == 0.0 else f"{value:.3g}"


def _diffusivity_color_mapping(values, color_scale="auto", symlog_linthresh=None):
    """Build a per-entry color mapping while retaining original-value labels."""
    requested = _validate_diffusivity_color_scale(color_scale)
    symlog_linthresh = _validate_symlog_linthresh(symlog_linthresh)
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if requested == "auto":
        mode = "symlog" if np.any(finite < 0.0) else "log"
    else:
        mode = requested

    if mode == "log":
        if np.any(finite < 0.0):
            raise ValueError(
                "Logarithmic diffusivity coloring cannot represent negative values; "
                "use diffusivity_color_scale='symlog' or 'auto'."
            )
        positive = finite[finite > 0.0]
        if positive.size == 0:
            if requested == "log" and finite.size:
                floor = 1.0e-300
                log_floor = float(np.log10(floor))
                return {
                    "mode": "log",
                    "colorscale": "Viridis",
                    "zmin": log_floor - 0.5,
                    "zmax": log_floor + 0.5,
                    "zmid": None,
                    "floor": floor,
                    "linthresh": None,
                    "tickvals": np.asarray([log_floor], dtype=np.float64),
                    "ticktext": ["0"],
                }
            return {
                "mode": "linear",
                "colorscale": "Viridis",
                "zmin": None,
                "zmax": None,
                "zmid": None,
                "floor": None,
                "linthresh": None,
                "tickvals": None,
                "ticktext": None,
            }
        floor = max(float(np.min(positive)) * 1.0e-3, float(np.finfo(np.float64).tiny))
        if np.any(finite == 0.0):
            original_ticks = np.concatenate((np.asarray([0.0]), _log_color_ticks(positive)))
        else:
            original_ticks = _log_color_ticks(positive)
        tickvals = np.log10(np.maximum(original_ticks, floor))
        zmin = float(np.log10(floor if np.any(finite == 0.0) else np.min(positive)))
        zmax = float(np.log10(np.max(positive)))
        if zmin == zmax:
            zmin -= 0.5
            zmax += 0.5
        return {
            "mode": "log",
            "colorscale": "Viridis",
            "zmin": zmin,
            "zmax": zmax,
            "zmid": None,
            "floor": floor,
            "linthresh": None,
            "tickvals": tickvals,
            "ticktext": [_format_colorbar_tick(value) for value in original_ticks],
        }

    if mode == "symlog":
        linthresh = _select_symlog_linthresh(values, symlog_linthresh)
        transformed = _signed_log10_color_values(values, linthresh)
        finite_transformed = transformed[np.isfinite(transformed)]
        limit = float(np.max(np.abs(finite_transformed))) if finite_transformed.size else 1.0
        limit = max(limit, 1.0)
        original_ticks = _symlog_color_ticks(values, linthresh)
        return {
            "mode": "symlog",
            "colorscale": "RdBu",
            "zmin": -limit,
            "zmax": limit,
            "zmid": 0.0,
            "floor": None,
            "linthresh": linthresh,
            "tickvals": _signed_log10_color_values(original_ticks, linthresh),
            "ticktext": [_format_colorbar_tick(value) for value in original_ticks],
        }

    return {
        "mode": "linear",
        "colorscale": "RdBu",
        "zmin": None,
        "zmax": None,
        "zmid": 0.0 if np.any(finite < 0.0) else None,
        "floor": None,
        "linthresh": None,
        "tickvals": None,
        "ticktext": None,
    }


def _transform_diffusivity_color_values(values, mapping):
    """Apply a color transform without changing the original hover-data values."""
    values = np.asarray(values, dtype=np.float64)
    if mapping["mode"] == "log":
        transformed = values.copy()
        finite = np.isfinite(values)
        transformed[finite] = np.log10(np.maximum(values[finite], mapping["floor"]))
        return transformed
    if mapping["mode"] == "symlog":
        return _signed_log10_color_values(values, mapping["linthresh"])
    return values


def _diffusivity_heatmap_options(fig, row, col, mapping, colorbar_title, colorscale=None):
    """Return Plotly heatmap options for a transformed diffusivity component."""
    colorbar = _subplot_colorbar(fig, row, col, colorbar_title)
    if mapping["mode"] != "linear":
        colorbar["title"] = {"text": f"{colorbar_title} ({mapping['mode']})"}
        colorbar["tickmode"] = "array"
        colorbar["tickvals"] = mapping["tickvals"].tolist()
        colorbar["ticktext"] = mapping["ticktext"]
    options = {
        "colorscale": mapping["colorscale"] if colorscale is None else colorscale,
        "colorbar": colorbar,
    }
    if mapping["zmin"] is not None:
        options["zmin"] = mapping["zmin"]
    if mapping["zmax"] is not None:
        options["zmax"] = mapping["zmax"]
    if mapping["zmid"] is not None:
        options["zmid"] = mapping["zmid"]
    return options


def _diffusivity_marker_options(fig, row, col, mapping, colorbar_title, colorscale=None, size=9):
    """Return ternary-marker options for a transformed diffusivity component.

    Plotly does not provide a native ternary heatmap trace. The bulk diagnostic
    maps therefore use dense ``Scatterternary`` markers; the color transform
    uses marker ``cmin``/``cmax``/``cmid`` equivalents of the heatmap limits.
    """
    colorbar = _subplot_colorbar(fig, row, col, colorbar_title)
    if mapping["mode"] != "linear":
        colorbar["title"] = {"text": f"{colorbar_title} ({mapping['mode']})"}
        colorbar["tickmode"] = "array"
        colorbar["tickvals"] = mapping["tickvals"].tolist()
        colorbar["ticktext"] = mapping["ticktext"]
    options = {
        "size": size,
        "colorscale": mapping["colorscale"] if colorscale is None else colorscale,
        "colorbar": colorbar,
        "showscale": True,
    }
    if mapping["zmin"] is not None:
        options["cmin"] = mapping["zmin"]
    if mapping["zmax"] is not None:
        options["cmax"] = mapping["zmax"]
    if mapping["zmid"] is not None:
        options["cmid"] = mapping["zmid"]
    return options


def plot_bulk_diffusivity_diagnostics(
    report,
    phase,
    *,
    hover_fields="diagnostic",
    hover_format=".6g",
    diffusivity_color_scale="auto",
    symlog_linthresh=None,
    renderer="browser",
):
    """Create a bulk diffusivity dashboard using the browser renderer by default.

    All composition fields are actual barycentric ternary subplots and use the
    same corner ordering as :func:`plot_tieline_diagnostics`. Plotly has no
    native ternary heatmap trace, so each colored field is rendered as a dense
    ``Scatterternary`` marker field at the valid simplex sample points. This
    preserves the triangular geometry and masked simplex boundary, but does
    not interpolate between samples. Colorbars are sized and positioned
    relative to their owning subplot, including switchable truth and error
    layers. Diffusivity component fields use per-component base-10 logarithmic
    coloring by default; components with negative finite values automatically
    use a signed-log transform so both signs remain visible. Relative-error
    layers use logarithmic coloring under ``"auto"``; an all-zero layer is
    placed at a small positive transform floor and retains a ``0`` colorbar
    label because zero itself has no logarithm. The ``Validity / truth error``
    panel applies the same rule to its matrix-relative-error values when truth
    data are available; without truth data it remains an invalid-matrix mask.
    Set ``diffusivity_color_scale`` to ``"linear"``, ``"log"``, or
    ``"symlog"`` to override the automatic choice. ``symlog_linthresh``
    controls the linear core of the signed-log transform.
    """
    go, make_subplots = _require_plotly(renderer)
    if report.get("kind") != "diffusivity" or phase not in report["bulk"]["phases"]:
        raise ValueError(f"Bulk diffusivity report does not contain phase '{phase}'.")
    selected = _selected_hover_fields("diffusivity", hover_fields)
    requested_color_scale = _validate_diffusivity_color_scale(diffusivity_color_scale)
    bulk = report["bulk"]
    phase_report = bulk["phases"][phase]
    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"type": "ternary"}, {"type": "ternary"}, {"type": "ternary"}, {"type": "ternary"}],
               [{"type": "ternary"}, {"type": "ternary"}, {"type": "ternary"}, {"type": "ternary"}]],
        subplot_titles=("Training coverage", "D[0,0]", "D[0,1]", "D[1,0]", "D[1,1]", "Training distance", "Validity / truth error", "Fallback mask"),
    )
    for row in (1, 2):
        for col in (1, 2, 3, 4):
            fig.update_ternaries(
                sum=1,
                aaxis={"title": report["elements"][2]},
                baxis={"title": report["elements"][0]},
                caxis={"title": report["elements"][1]},
                row=row,
                col=col,
            )
    for index in range(1, 9):
        layout_name = "ternary" if index == 1 else f"ternary{index}"
        _leave_room_above_ternary(fig, layout_name, gap=0.04)
    points = phase_report["compositions"]
    coverage_fields = _matrix_hover_fields(phase_report)
    coverage_custom, coverage_template = _hover_payload(coverage_fields, selected, len(points), hover_format)
    fig.add_trace(
        go.Scatterternary(
            **_ternary_coordinates(points), mode="markers",
            marker={"size": 5, "color": phase_report["nearest_training_distance"], "colorscale": "Viridis", "showscale": False},
            name="Evaluation grid", legendgroup="coverage", customdata=coverage_custom, hovertemplate=coverage_template,
        ), row=1, col=1,
    )
    training = phase_report["training_compositions"]
    training_fields = {
        "source": "training",
        "phase": phase,
        "context": "general",
        "eta": np.nan,
        **_composition_fields(training),
        "training": True,
    }
    training_custom, training_template = _hover_payload(
        training_fields, selected, len(training), hover_format
    )
    fig.add_trace(
        go.Scatterternary(
            **_ternary_coordinates(training),
            mode="markers", marker={"symbol": "diamond", "size": 9, "color": "black"},
            name="Training samples", legendgroup="training",
            customdata=training_custom, hovertemplate=training_template,
        ), row=1, col=1,
    )
    if training.shape[0] >= 3:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(training)
            boundary = training[np.append(hull.vertices, hull.vertices[0])]
            fig.add_trace(
                go.Scatterternary(
                    **_ternary_coordinates(boundary),
                    mode="lines", line={"color": "black", "dash": "dash"},
                    name="Training convex hull", legendgroup="training-hull", hoverinfo="skip",
                ), row=1, col=1,
            )
        except Exception:
            pass

    layer_indices = {"prediction": [], "truth": [], "absolute_error": [], "relative_error": []}
    components = ((0, 0), (0, 1), (1, 0), (1, 1))
    positions = ((1, 2), (1, 3), (1, 4), (2, 1))
    truth = phase_report["truth"]
    for component, (row, col) in zip(components, positions):
        layers = {
            "prediction": phase_report["matrices"][:, component[0], component[1]],
        }
        if truth is not None:
            layers.update({
                "truth": truth["matrices"][:, component[0], component[1]],
                "absolute_error": truth["absolute_error"][:, component[0], component[1]],
                "relative_error": truth["relative_error"][:, component[0], component[1]],
            })
        component_values = [layers["prediction"]]
        if truth is not None:
            component_values.append(layers["truth"])
        diffusivity_mapping = _diffusivity_color_mapping(
            np.concatenate(component_values),
            color_scale=diffusivity_color_scale,
            symlog_linthresh=symlog_linthresh,
        )
        for layer, values in layers.items():
            layer_fields = _matrix_hover_fields(
                phase_report, component, source=layer.replace("_", " ")
            )
            layer_fields["value"] = values
            if layer == "truth":
                layer_fields["d00"] = truth["matrices"][:, 0, 0]
                layer_fields["d01"] = truth["matrices"][:, 0, 1]
                layer_fields["d10"] = truth["matrices"][:, 1, 0]
                layer_fields["d11"] = truth["matrices"][:, 1, 1]
            layer_custom, layer_template = _hover_payload(
                layer_fields, selected, len(points), hover_format
            )
            trace_index = len(fig.data)
            layer_indices[layer].append(trace_index)
            mapping = diffusivity_mapping if layer in {"prediction", "truth"} else _diffusivity_color_mapping(
                values,
                color_scale=(
                    "log"
                    if layer == "relative_error" and requested_color_scale == "auto"
                    else requested_color_scale
                ),
                symlog_linthresh=symlog_linthresh,
            )
            layer_colorscale = mapping["colorscale"]
            if layer in {"absolute_error", "relative_error"} and mapping["mode"] == "linear":
                layer_colorscale = "Viridis"
            fig.add_trace(
                go.Scatterternary(
                    **_ternary_coordinates(points),
                    mode="markers",
                    marker={
                        "color": _transform_diffusivity_color_values(values, mapping),
                        **_diffusivity_marker_options(
                            fig,
                            row,
                            col,
                            mapping,
                            "m^2/s" if layer != "relative_error" else "relative",
                            colorscale=layer_colorscale,
                        ),
                    },
                    name=layer.replace("_", " ").title(),
                    visible=layer == "prediction",
                    customdata=layer_custom,
                    hovertemplate=layer_template,
                ), row=row, col=col,
            )

    distance_fields = _matrix_hover_fields(phase_report)
    distance_fields["value"] = phase_report["nearest_training_distance"]
    distance_custom, distance_template = _hover_payload(
        distance_fields, selected, len(points), hover_format
    )
    fig.add_trace(
        go.Scatterternary(
            **_ternary_coordinates(points),
            mode="markers",
            marker={
                "color": phase_report["nearest_training_distance"],
                "size": 9,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": _subplot_colorbar(fig, 2, 2, "distance"),
            },
            name="Training distance",
            customdata=distance_custom,
            hovertemplate=distance_template,
        ), row=2, col=2,
    )
    diagnostic = truth["matrix_relative_error"] if truth is not None else (~phase_report["valid"]).astype(float)
    diagnostic_fields = _matrix_hover_fields(phase_report)
    diagnostic_fields["value"] = diagnostic
    diagnostic_custom, diagnostic_template = _hover_payload(
        diagnostic_fields, selected, len(points), hover_format
    )
    if truth is not None:
        diagnostic_mapping = _diffusivity_color_mapping(
            diagnostic,
            color_scale="log" if requested_color_scale == "auto" else requested_color_scale,
            symlog_linthresh=symlog_linthresh,
        )
        diagnostic_marker = {
            "color": _transform_diffusivity_color_values(diagnostic, diagnostic_mapping),
            **_diffusivity_marker_options(
                fig, 2, 3, diagnostic_mapping, "relative", colorscale="Magma"
            ),
        }
    else:
        diagnostic_marker = {
            "color": diagnostic,
            "size": 9,
            "colorscale": "Magma",
            "showscale": True,
            "colorbar": _subplot_colorbar(fig, 2, 3, "invalid"),
        }
    fig.add_trace(
        go.Scatterternary(
            **_ternary_coordinates(points),
            mode="markers",
            marker=diagnostic_marker,
            name="Matrix relative error" if truth is not None else "Invalid matrix",
            customdata=diagnostic_custom,
            hovertemplate=diagnostic_template,
        ), row=2, col=3,
    )
    fallback_fields = _matrix_hover_fields(phase_report)
    fallback_fields["value"] = phase_report["fallback"].astype(float)
    fallback_custom, fallback_template = _hover_payload(
        fallback_fields, selected, len(points), hover_format
    )
    fig.add_trace(
        go.Scatterternary(
            **_ternary_coordinates(points),
            mode="markers",
            marker={
                "color": phase_report["fallback"].astype(float),
                "size": 9,
                "colorscale": [[0.0, "white"], [1.0, "#ff7f0e"]],
                "cmin": 0,
                "cmax": 1,
                "showscale": False,
            },
            name="Nearest fallback",
            customdata=fallback_custom,
            hovertemplate=fallback_template,
        ), row=2, col=4,
    )

    if truth is not None:
        fixed_indices = set(range(len(fig.data))) - set(sum(layer_indices.values(), []))
        buttons = []
        for layer in ("prediction", "truth", "absolute_error", "relative_error"):
            visible = [index in fixed_indices or index in layer_indices[layer] for index in range(len(fig.data))]
            buttons.append({"label": layer.replace("_", " ").title(), "method": "update", "args": [{"visible": visible}]})
        fig.update_layout(updatemenus=[{"type": "buttons", "direction": "right", "buttons": buttons, "x": 0.42, "y": 1.10}])

    fig.update_layout(
        title=f"Bulk diffusivity diagnostics: {phase} at {report['temperature']:g} K",
        template="plotly_white", width=1500, height=850, margin={"t": 130}, uirevision=f"bulk-diffusivity-{phase}",
    )
    return fig


def _hover_for_section(hover_fields, section):
    if isinstance(hover_fields, Mapping):
        return hover_fields.get(section, "diagnostic")
    return hover_fields


def evaluate_surrogate_construction_diagnostics(surrogate):
    """Return persisted construction attempts and outcome counts for a surrogate.

    This inspection is entirely metadata-based and never calls the original
    thermodynamics provider.  Surrogates made before construction provenance
    was introduced return an empty, schema-versioned report.
    """
    payload = getattr(surrogate, "metadata", {}).get("construction_diagnostics", {})
    records = [dict(record) for record in payload.get("records", [])]
    counts = {}
    for record in records:
        key = (record.get("kind", "unknown"), record.get("phase", ""), record.get("outcome", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    settings = dict(payload.get("settings", {}))
    metadata = getattr(surrogate, "metadata", {})
    thermocalc_config = metadata.get("thermocalc_config")
    if isinstance(thermocalc_config, Mapping):
        settings.update({
            f"thermocalc.{key}": value
            for key, value in thermocalc_config.items()
        })
        settings["thermocalc.capture_diffusivity_diagnostics"] = (
            "kinetics_diagnostics_sidecar" in metadata
        )
        settings["thermocalc.capture_site_fractions"] = "site_fractions_sidecar" in metadata
    return {
        "kind": "surrogate_construction", "schema_version": int(payload.get("schema_version", 1)),
        "elements": tuple(getattr(surrogate, "elements", ())),
        "settings": settings, "records": records, "counts": counts,
    }


def plot_surrogate_construction_diagnostics(report, *, hover_format=".6g", renderer="browser"):
    """Create a tabbed Plotly map of recorded surrogate-construction attempts."""
    if report.get("kind") != "surrogate_construction":
        raise ValueError("Construction report has an unexpected kind.")
    go, _ = _require_plotly(renderer)
    elements = tuple(report.get("elements", ("reference", "x1", "x2")))
    if len(elements) != 3:
        elements = ("reference", "x1", "x2")
    fig = go.Figure()
    records = report["records"]
    outcomes = {"success": "#2ca02c", "failed": "#d62728", "invalid": "#ff7f0e"}
    groups = [("Equilibrium", [r for r in records if r.get("kind") == "equilibrium"])]
    phases = list(report["settings"].get("tieline_phases", []))
    phases.extend(sorted({r.get("phase") for r in records if r.get("kind") == "kinetics" and r.get("phase") not in phases}))
    groups.extend([(f"Kinetics: {phase}", [r for r in records if r.get("kind") == "kinetics" and r.get("phase") == phase]) for phase in phases])
    count_lines = [
        f"{kind} / {phase or 'all phases'} / {outcome}: {value}"
        for (kind, phase, outcome), value in sorted(report["counts"].items())
    ]
    settings = report["settings"]
    setting_lines = [
        f"{escape(str(key))}: {escape(str(value))}"
        for key, value in settings.items()
        if key != "probe_parameters"
    ]
    overview_annotations = [
        {
            "text": "<b>Outcome counts</b><br>" + "<br>".join(count_lines or ["No persisted construction records."]),
            "showarrow": False, "x": 0.03, "y": 0.94, "xref": "paper", "yref": "paper",
            "xanchor": "left", "yanchor": "top", "align": "left",
        },
        {
            "text": "<b>Build settings</b><br>" + "<br>".join(setting_lines or ["No persisted settings."]),
            "showarrow": False, "x": 0.53, "y": 0.94, "xref": "paper", "yref": "paper",
            "xanchor": "left", "yanchor": "top", "align": "left",
        },
    ]
    overview_shapes = [
        {"type": "rect", "x0": 0.01, "x1": 0.49, "y0": 0.08, "y1": 0.98, "xref": "paper", "yref": "paper", "fillcolor": "#eef3fa", "line": {"width": 0}, "layer": "below"},
        {"type": "rect", "x0": 0.51, "x1": 0.99, "y0": 0.08, "y1": 0.98, "xref": "paper", "yref": "paper", "fillcolor": "#eef3fa", "line": {"width": 0}, "layer": "below"},
    ]
    trace_tabs = {"Overview": []}
    for tab, rows in groups:
        trace_tabs[tab] = []
        for outcome in outcomes:
            subset = [r for r in rows if r.get("outcome") == outcome and len(r.get("composition", ())) == 2]
            if not subset:
                continue
            points = np.asarray([r["composition"] for r in subset], dtype=float)
            hover = ["<br>".join(f"{escape(str(k))}: {escape(str(v))}" for k, v in row.items() if v is not None) for row in subset]
            symbols = ["diamond" if r.get("context") == "interface" else "circle" for r in subset]
            index = len(fig.data)
            fig.add_trace(go.Scatterternary(
                **_ternary_coordinates(points), mode="markers", name=f"{tab} {outcome}",
                marker={"size": 9, "color": outcomes[outcome], "symbol": symbols},
                text=hover, hovertemplate="%{text}<extra></extra>", visible=False,
            ))
            trace_tabs[tab].append(index)
    buttons = []
    for tab, indices in trace_tabs.items():
        visible = [i in indices for i in range(len(fig.data))]
        annotation = overview_annotations if tab == "Overview" else [{
            "text": "Diamond: interface endpoint; circle: bulk query.",
            "showarrow": False, "x": 0.5, "y": 0.88,
            "xref": "paper", "yref": "paper", "align": "center",
        }]
        layout_update = {
            "annotations": annotation,
            "shapes": overview_shapes if tab == "Overview" else [],
            "ternary.domain.y": [0.0, 0.01] if tab == "Overview" else [0.0, 0.80],
        }
        buttons.append({
            "label": tab,
            "method": "update",
            "args": [
                {"visible": visible},
                layout_update,
            ],
        })
    fig.update_layout(
        title="Surrogate construction diagnostics", template="plotly_white", height=760,
        ternary={"sum": 1, "domain": {"y": [0.0, 0.01]}, "aaxis": {"title": elements[2]}, "baxis": {"title": elements[0]}, "caxis": {"title": elements[1]}},
        updatemenus=[{"type": "buttons", "direction": "right", "buttons": buttons, "x": 0.5, "xanchor": "center", "y": 1.12}],
        annotations=overview_annotations, shapes=overview_shapes,
        margin={"t": 105}, uirevision="surrogate-construction",
    )
    return fig


def plot_surrogate_diagnostics(
    surrogate,
    *,
    thermodynamics=None,
    probe_compositions=None,
    eta_count=101,
    interface_eta_count=101,
    bulk_grid_counts=(51, 51),
    bulk_axes=None,
    on_truth_error="raise",
    hover_fields="diagnostic",
    hover_format=".6g",
    display_tieline_count=21,
    phase_region_report=None,
    diffusivity_color_scale="auto",
    symlog_linthresh=None,
    renderer="browser",
):
    """Build all reports and figures, defaulting Plotly display to the browser.

    ``diffusivity_color_scale`` is passed to the bulk diffusivity fields.
    With the default ``"auto"`` setting, each matrix-entry panel uses base-10
    logarithmic coloring when its finite diffusivity values are nonnegative,
    and signed-log (symlog) coloring when any finite value is negative.
    Relative-error layers are logarithmic even when every error is zero; zero
    is represented at a small positive transform floor.
    ``symlog_linthresh`` sets the positive linear-core threshold when symlog
    coloring is used.
    """
    figures = {}
    reports = {}
    construction = evaluate_surrogate_construction_diagnostics(surrogate)
    reports["construction"] = construction
    figures["construction"] = plot_surrogate_construction_diagnostics(construction, renderer=renderer)
    if isinstance(surrogate, TernaryMovingBoundaryThermodynamicsSurrogate):
        tieline = evaluate_tieline_diagnostics(
            surrogate,
            thermodynamics=thermodynamics,
            probe_compositions=probe_compositions,
            eta_count=eta_count,
            on_truth_error=on_truth_error,
        )
        reports["thermodynamics"] = tieline
        figures["thermodynamics"] = plot_tieline_diagnostics(
            tieline,
            hover_fields=_hover_for_section(hover_fields, "thermodynamics"),
            hover_format=hover_format,
            display_tieline_count=display_tieline_count,
            phase_region_report=phase_region_report,
            renderer=renderer,
        )
    diffusivity = evaluate_diffusivity_diagnostics(
        surrogate,
        thermodynamics=thermodynamics,
        interface_eta_count=interface_eta_count,
        bulk_grid_counts=bulk_grid_counts,
        bulk_axes=bulk_axes,
        on_truth_error=on_truth_error,
    )
    reports["diffusivity"] = diffusivity
    if diffusivity["interface"] is not None:
        figures["interface_diffusivity"] = plot_interface_diffusivity_diagnostics(
            diffusivity,
            hover_fields=_hover_for_section(hover_fields, "interface_diffusivity"),
            hover_format=hover_format,
            renderer=renderer,
        )
    figures["bulk_diffusivity"] = {
        phase: plot_bulk_diffusivity_diagnostics(
            diffusivity,
            phase,
            hover_fields=_hover_for_section(hover_fields, "bulk_diffusivity"),
            hover_format=hover_format,
            diffusivity_color_scale=diffusivity_color_scale,
            symlog_linthresh=symlog_linthresh,
            renderer=renderer,
        )
        for phase in diffusivity["phases"]
    }
    return {"figures": figures, "reports": reports}


__all__ = [
    "evaluate_tieline_diagnostics",
    "evaluate_diffusivity_diagnostics",
    "evaluate_diffusivity_leave_one_out",
    "plot_diffusivity_leave_one_out",
    "load_calculation_site_fractions",
    "plot_calculation_site_fractions",
    "plot_tieline_diagnostics",
    "plot_interface_diffusivity_diagnostics",
    "plot_bulk_diffusivity_diagnostics",
    "evaluate_surrogate_construction_diagnostics",
    "plot_surrogate_construction_diagnostics",
    "plot_surrogate_diagnostics",
]
