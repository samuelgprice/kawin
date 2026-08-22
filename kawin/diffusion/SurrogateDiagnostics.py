"""Interactive diagnostics for ternary moving-boundary surrogate models.

The evaluation helpers in this module depend only on NumPy and SciPy so they
remain available when Plotly is not installed.  Plotting helpers import Plotly
lazily and return figures without displaying or writing them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .MovingBoundarySurrogates import (
    MergedPhaseDiffusivitySurrogate,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    _matrix_validity_diagnostics,
)


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
    """Position a compact heatmap colorbar beside its owning subplot."""
    subplot = fig.get_subplot(row, col)
    x_domain = subplot.xaxis.domain
    y_domain = subplot.yaxis.domain
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


def plot_bulk_diffusivity_diagnostics(
    report,
    phase,
    *,
    hover_fields="diagnostic",
    hover_format=".6g",
    renderer="browser",
):
    """Create a bulk diffusivity dashboard using the browser renderer by default.

    Its ternary coverage view uses the same corner ordering as
    :func:`plot_tieline_diagnostics`. Each heatmap colorbar is sized and
    positioned relative to its own subplot, including switchable truth and
    error layers.
    """
    go, make_subplots = _require_plotly(renderer)
    if report.get("kind") != "diffusivity" or phase not in report["bulk"]["phases"]:
        raise ValueError(f"Bulk diffusivity report does not contain phase '{phase}'.")
    selected = _selected_hover_fields("diffusivity", hover_fields)
    bulk = report["bulk"]
    phase_report = bulk["phases"][phase]
    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"type": "ternary"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Training coverage", "D[0,0]", "D[0,1]", "D[1,0]", "D[1,1]", "Training distance", "Validity / truth error", "Fallback mask"),
    )
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
            layer_custom, layer_template = _grid_customdata(
                bulk, layer_fields, selected, hover_format
            )
            trace_index = len(fig.data)
            layer_indices[layer].append(trace_index)
            fig.add_trace(
                go.Heatmap(
                    x=bulk["axes"][0], y=bulk["axes"][1], z=_grid_values(bulk, values),
                    colorscale="RdBu" if layer in {"prediction", "truth"} else "Viridis",
                    zmid=0.0 if layer in {"prediction", "truth"} else None,
                    colorbar=_subplot_colorbar(
                        fig, row, col, "m^2/s" if layer != "relative_error" else "relative"
                    ),
                    name=layer.replace("_", " ").title(), showscale=True,
                    visible=layer == "prediction", customdata=layer_custom, hovertemplate=layer_template,
                ), row=row, col=col,
            )

    distance_fields = _matrix_hover_fields(phase_report)
    distance_fields["value"] = phase_report["nearest_training_distance"]
    distance_custom, distance_template = _grid_customdata(bulk, distance_fields, selected, hover_format)
    fig.add_trace(
        go.Heatmap(
            x=bulk["axes"][0], y=bulk["axes"][1], z=_grid_values(bulk, phase_report["nearest_training_distance"]),
            colorscale="Viridis", name="Training distance", showscale=True,
            colorbar=_subplot_colorbar(fig, 2, 2, "distance"),
            customdata=distance_custom, hovertemplate=distance_template,
        ), row=2, col=2,
    )
    diagnostic = truth["matrix_relative_error"] if truth is not None else (~phase_report["valid"]).astype(float)
    diagnostic_fields = _matrix_hover_fields(phase_report)
    diagnostic_fields["value"] = diagnostic
    diagnostic_custom, diagnostic_template = _grid_customdata(bulk, diagnostic_fields, selected, hover_format)
    fig.add_trace(
        go.Heatmap(
            x=bulk["axes"][0], y=bulk["axes"][1], z=_grid_values(bulk, diagnostic),
            colorscale="Magma", name="Matrix relative error" if truth is not None else "Invalid matrix", showscale=True,
            colorbar=_subplot_colorbar(fig, 2, 3, "relative" if truth is not None else "invalid"),
            customdata=diagnostic_custom, hovertemplate=diagnostic_template,
        ), row=2, col=3,
    )
    fallback_fields = _matrix_hover_fields(phase_report)
    fallback_fields["value"] = phase_report["fallback"].astype(float)
    fallback_custom, fallback_template = _grid_customdata(bulk, fallback_fields, selected, hover_format)
    fig.add_trace(
        go.Heatmap(
            x=bulk["axes"][0], y=bulk["axes"][1], z=_grid_values(bulk, phase_report["fallback"].astype(float)),
            colorscale=[[0.0, "white"], [1.0, "#ff7f0e"]], zmin=0, zmax=1,
            name="Nearest fallback", showscale=False, customdata=fallback_custom, hovertemplate=fallback_template,
        ), row=2, col=4,
    )

    if truth is not None:
        fixed_indices = set(range(len(fig.data))) - set(sum(layer_indices.values(), []))
        buttons = []
        for layer in ("prediction", "truth", "absolute_error", "relative_error"):
            visible = [index in fixed_indices or index in layer_indices[layer] for index in range(len(fig.data))]
            buttons.append({"label": layer.replace("_", " ").title(), "method": "update", "args": [{"visible": visible}]})
        fig.update_layout(updatemenus=[{"type": "buttons", "direction": "right", "buttons": buttons, "x": 0.42, "y": 1.10}])

    x_range, y_range = _zoom_ranges([points])
    for row, col in (*positions, (2, 2), (2, 3), (2, 4)):
        fig.update_xaxes(title_text=f"X({report['elements'][1]})", range=x_range, row=row, col=col)
        fig.update_yaxes(title_text=f"X({report['elements'][2]})", range=y_range, row=row, col=col)
    fig.update_layout(
        title=f"Bulk diffusivity diagnostics: {phase} at {report['temperature']:g} K",
        template="plotly_white", width=1500, height=850, margin={"t": 130}, uirevision=f"bulk-diffusivity-{phase}",
        ternary={"sum": 1, "aaxis": {"title": report["elements"][2]}, "baxis": {"title": report["elements"][0]}, "caxis": {"title": report["elements"][1]}},
    )
    _leave_room_above_ternary(fig)
    return fig


def _hover_for_section(hover_fields, section):
    if isinstance(hover_fields, Mapping):
        return hover_fields.get(section, "diagnostic")
    return hover_fields


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
    renderer="browser",
):
    """Build all reports and figures, defaulting Plotly display to the browser."""
    figures = {}
    reports = {}
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
            renderer=renderer,
        )
        for phase in diffusivity["phases"]
    }
    return {"figures": figures, "reports": reports}


__all__ = [
    "evaluate_tieline_diagnostics",
    "evaluate_diffusivity_diagnostics",
    "plot_tieline_diagnostics",
    "plot_interface_diffusivity_diagnostics",
    "plot_bulk_diffusivity_diagnostics",
    "plot_surrogate_diagnostics",
]
