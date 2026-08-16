import json
from pathlib import Path

import numpy as np
from scipy import optimize
try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, PchipInterpolator, RectBivariateSpline
except ImportError:  # pragma: no cover - SciPy is a package dependency.
    LinearNDInterpolator = None
    NearestNDInterpolator = None
    PchipInterpolator = None
    RectBivariateSpline = None

from kawin.thermo import MulticomponentThermodynamics


_DIFFUSIVITY_INTERPOLATION_NEAREST = "nearest"
_DIFFUSIVITY_INTERPOLATION_CONTINUOUS_GRID = "continuous_grid"
_DIFFUSIVITY_INTERPOLATION_SIMPLEX_LINEAR = "simplex_linear"


def _as_path_with_npz_suffix(path):
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    return path


def _as_independent_ternary_components(composition, elements, label):
    """
    Returns the two independent components from a ternary composition vector.

    Full ternary vectors are interpreted using the diffusion convention where
    ``elements[0]`` is dependent and the remaining entries are independent.
    """
    values = np.asarray(composition, dtype=np.float64).reshape(-1)
    if values.size == 2:
        out = values
    elif values.size == len(elements) == 3:
        out = values[1:]
    else:
        raise ValueError(f"{label} must contain either 2 independent or 3 full ternary components.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{label} contains non-finite composition values.")
    return out.astype(np.float64)


def _validate_independent_composition(composition, min_composition, label):
    """
    Validates a ternary independent-component vector.

    The dependent component is reconstructed as ``1 - x0 - x1``.
    """
    values = np.asarray(composition, dtype=np.float64).reshape(-1)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be a finite two-component vector.")
    dependent = 1.0 - float(np.sum(values))
    if np.any(values < min_composition) or dependent < min_composition:
        raise ValueError(f"{label} violates ternary composition bounds.")
    return values


def _validate_2x2_matrix(matrix, label):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 2x2 matrix.")
    return matrix


def _validate_positive_2x2_matrix(matrix, label):
    """
    Validates a finite 2x2 matrix with positive real eigenvalues.

    The ternary Illingworth bulk solver requires interdiffusivity matrices whose
    normalized eigenvalues stay positive. Continuous surrogate splines are
    checked at build time so runtime evaluation can remain a cheap array call.
    """
    matrix = _validate_2x2_matrix(matrix, label)
    scale = float(np.linalg.norm(matrix, ord=np.inf))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{label} must have nonzero norm.")
    eigenvalues = np.linalg.eigvals(matrix / scale)
    if np.any(np.abs(np.imag(eigenvalues)) > 1e-12) or np.any(np.real(eigenvalues) <= 1e-14):
        raise ValueError(f"{label} must have positive real eigenvalues.")
    return matrix


def _coerce_diffusivity_interpolation(mode):
    mode = _DIFFUSIVITY_INTERPOLATION_NEAREST if mode is None else str(mode)
    if mode not in {
        _DIFFUSIVITY_INTERPOLATION_NEAREST,
        _DIFFUSIVITY_INTERPOLATION_CONTINUOUS_GRID,
        _DIFFUSIVITY_INTERPOLATION_SIMPLEX_LINEAR,
    }:
        raise ValueError(
            "diffusivity_interpolation must be 'nearest', 'continuous_grid', or 'simplex_linear'."
        )
    return mode


def _signed_cuberoot(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.cbrt(np.abs(values))


def _coerce_positive_int(value, name):
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def _coerce_grid_counts(value, name):
    values = np.asarray(value, dtype=np.int64).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, 2)
    if values.size != 2 or np.any(values < 1):
        raise ValueError(f"{name} must be a positive integer or two positive integers.")
    return int(values[0]), int(values[1])


def _bulk_grid_axes(diffusivity_bulk_grids, min_composition):
    if diffusivity_bulk_grids is None:
        return None
    axes = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in diffusivity_bulk_grids)
    if len(axes) != 2:
        raise ValueError("diffusivity_bulk_grids must contain exactly two component axes.")
    for i, axis in enumerate(axes):
        if axis.size < 2:
            raise ValueError(f"diffusivity_bulk_grids axis {i} must contain at least two samples.")
        if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
            raise ValueError(f"diffusivity_bulk_grids axis {i} must be finite and strictly increasing.")
        if axis[0] < min_composition:
            raise ValueError("diffusivity_bulk_grids violates ternary composition bounds.")
    if axes[0][-1] + axes[1][-1] > 1.0 - min_composition:
        raise ValueError("diffusivity_bulk_grids rectangular domain violates ternary composition bounds.")
    return tuple(axis.copy() for axis in axes)


def _bulk_points_from_axes(axes):
    return np.asarray(np.meshgrid(*axes, indexing="ij"), dtype=np.float64).reshape(2, -1).T


def _bulk_points_from_simplex_axes(axes, min_composition):
    if axes is None:
        return None
    points = _bulk_points_from_axes(axes)
    points = points[_valid_simplex_mask(points, min_composition)]
    if points.shape[0] < 3:
        raise ValueError("diffusivity_bulk_grids must contain at least three simplex-valid points.")
    return points


def _densified_axes_from_bounds(bounds, counts):
    return tuple(
        np.linspace(float(bounds[i, 0]), float(bounds[i, 1]), int(counts[i]), dtype=np.float64)
        for i in range(2)
    )


def _valid_simplex_mask(points, min_composition):
    points = np.asarray(points, dtype=np.float64)
    return (
        np.all(np.isfinite(points), axis=1)
        & np.all(points >= min_composition, axis=1)
        & (np.sum(points, axis=1) <= 1.0 - min_composition)
    )


def _matrix_validity_diagnostics(matrices, *, eigen_imag_tol=1e-12, eigen_real_min=1e-14):
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (2, 2):
        raise ValueError("matrix validity diagnostics require shape (n_samples, 2, 2).")
    finite = np.all(np.isfinite(matrices), axis=(1, 2))
    scales = np.linalg.norm(np.where(np.isfinite(matrices), matrices, 0.0), ord=np.inf, axis=(1, 2))
    nonzero = np.isfinite(scales) & (scales > 0.0)
    normalized = np.zeros_like(matrices, dtype=np.float64)
    valid_scale = finite & nonzero
    normalized[valid_scale] = matrices[valid_scale] / scales[valid_scale, np.newaxis, np.newaxis]
    eigenvalues = np.full((matrices.shape[0], 2), np.nan + 0j, dtype=np.complex128)
    if np.any(valid_scale):
        eigenvalues[valid_scale] = np.linalg.eigvals(normalized[valid_scale])
    real_positive = np.all(np.real(eigenvalues) > float(eigen_real_min), axis=1)
    imaginary_small = np.all(np.abs(np.imag(eigenvalues)) <= float(eigen_imag_tol), axis=1)
    valid = finite & nonzero & real_positive & imaginary_small
    return {
        "valid": valid,
        "finite": finite,
        "nonzero": nonzero,
        "eigenvalues": eigenvalues,
        "scales": scales,
        "eigen_real_min": float(eigen_real_min),
        "eigen_imag_tol": float(eigen_imag_tol),
    }


class _InterfaceDiffusivitySpline1D:
    """
    Fast tie-line endpoint diffusivity interpolation over eta.

    Endpoint composition queries are mapped back to eta using a monotone
    endpoint component when available; otherwise the query is projected onto the
    nearest sampled endpoint-curve segment.
    """

    def __init__(self, eta_samples, endpoint_compositions, matrices):
        self.eta_samples = np.asarray(eta_samples, dtype=np.float64).reshape(-1)
        self.endpoint_compositions = np.asarray(endpoint_compositions, dtype=np.float64)
        values = np.asarray(matrices, dtype=np.float64)
        if self.endpoint_compositions.shape != (self.eta_samples.size, 2):
            raise ValueError("interface endpoint compositions must have shape (n_eta, 2).")
        if values.shape != (self.eta_samples.size, 2, 2):
            raise ValueError("interface diffusivity matrices must have shape (n_eta, 2, 2).")

        transformed = _signed_cuberoot(values.reshape(self.eta_samples.size, 4))
        if PchipInterpolator is not None:
            self._spline = PchipInterpolator(self.eta_samples, transformed, axis=0, extrapolate=True)
            self._transformed = None
        else:
            self._spline = None
            self._transformed = transformed

        self._inverse_component = None
        self._inverse_x = None
        self._inverse_eta = None
        for component in range(2):
            diffs = np.diff(self.endpoint_compositions[:, component])
            if np.all(diffs >= 0.0) and np.any(diffs > 0.0):
                self._inverse_component = component
                self._inverse_x = self.endpoint_compositions[:, component]
                self._inverse_eta = self.eta_samples
                break
            if np.all(diffs <= 0.0) and np.any(diffs < 0.0):
                self._inverse_component = component
                self._inverse_x = self.endpoint_compositions[::-1, component]
                self._inverse_eta = self.eta_samples[::-1]
                break

    def _eta_from_composition(self, values):
        if self._inverse_component is not None:
            component_values = values[:, self._inverse_component]
            return np.interp(
                component_values,
                self._inverse_x,
                self._inverse_eta,
                left=self.eta_samples[0],
                right=self.eta_samples[-1],
            )

        segments = self.endpoint_compositions[1:] - self.endpoint_compositions[:-1]
        segment_norms = np.sum(segments * segments, axis=1)
        out = np.empty(values.shape[0], dtype=np.float64)
        for i, value in enumerate(values):
            best_eta = self.eta_samples[0]
            best_distance = np.inf
            for j, segment in enumerate(segments):
                if segment_norms[j] <= 0.0:
                    t = 0.0
                else:
                    t = float(np.clip(np.dot(value - self.endpoint_compositions[j], segment) / segment_norms[j], 0.0, 1.0))
                projected = self.endpoint_compositions[j] + t * segment
                distance = float(np.sum((value - projected) ** 2))
                if distance < best_distance:
                    best_distance = distance
                    best_eta = self.eta_samples[j] + t * (self.eta_samples[j + 1] - self.eta_samples[j])
            out[i] = best_eta
        return out

    def evaluate(self, values):
        values = np.asarray(values, dtype=np.float64)
        eta = np.clip(self._eta_from_composition(values), self.eta_samples[0], self.eta_samples[-1])
        return self.evaluate_eta(eta)

    def evaluate_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64).reshape(-1)
        if self._spline is not None:
            transformed = self._spline(eta)
        else:
            transformed = np.vstack(
                [np.interp(eta, self.eta_samples, self._transformed[:, component]) for component in range(4)]
            ).T
        return (transformed ** 3).reshape(eta.shape[0], 2, 2)

    def validate_dense(self, label):
        eta = np.linspace(self.eta_samples[0], self.eta_samples[-1], max(25, 4 * self.eta_samples.size))
        matrices = self.evaluate_eta(eta)
        for i, matrix in enumerate(matrices):
            _validate_positive_2x2_matrix(matrix, f"{label} dense interface sample {i}")


class _BulkDiffusivityGridSpline2D:
    """
    Tensor-product spline for regular-grid ternary bulk diffusivity samples.

    Matrix components are interpolated after a signed cube-root transform and
    cubed on output. Query points must remain inside the stored rectangular
    composition window, which itself is required to lie inside the ternary
    simplex.
    """

    def __init__(self, x_axis, y_axis, matrices):
        if RectBivariateSpline is None:  # pragma: no cover - SciPy is a package dependency.
            raise ImportError("RectBivariateSpline is required for continuous_grid diffusivity interpolation.")
        self.x_axis = np.asarray(x_axis, dtype=np.float64).reshape(-1)
        self.y_axis = np.asarray(y_axis, dtype=np.float64).reshape(-1)
        values = np.asarray(matrices, dtype=np.float64)
        if values.shape != (self.x_axis.size, self.y_axis.size, 2, 2):
            raise ValueError("bulk diffusivity matrices must have shape (n_x, n_y, 2, 2).")
        transformed = _signed_cuberoot(values)
        kx = min(3, self.x_axis.size - 1)
        ky = min(3, self.y_axis.size - 1)
        self._splines = [
            RectBivariateSpline(self.x_axis, self.y_axis, transformed[:, :, i, j], kx=kx, ky=ky, s=0.0)
            for i in range(2)
            for j in range(2)
        ]

    def evaluate(self, values):
        values = np.asarray(values, dtype=np.float64)
        if np.any(values[:, 0] < self.x_axis[0]) or np.any(values[:, 0] > self.x_axis[-1]):
            raise ValueError("bulk diffusivity query lies outside continuous diffusivity grid.")
        if np.any(values[:, 1] < self.y_axis[0]) or np.any(values[:, 1] > self.y_axis[-1]):
            raise ValueError("bulk diffusivity query lies outside continuous diffusivity grid.")
        transformed = np.empty((values.shape[0], 4), dtype=np.float64)
        for component, spline in enumerate(self._splines):
            transformed[:, component] = spline.ev(values[:, 0], values[:, 1])
        return (transformed ** 3).reshape(values.shape[0], 2, 2)

    def validate_dense(self, label):
        dense_x = np.unique(np.concatenate((self.x_axis, 0.5 * (self.x_axis[:-1] + self.x_axis[1:]))))
        dense_y = np.unique(np.concatenate((self.y_axis, 0.5 * (self.y_axis[:-1] + self.y_axis[1:]))))
        points = _bulk_points_from_axes((dense_x, dense_y))
        matrices = self.evaluate(points)
        for i, matrix in enumerate(matrices):
            _validate_positive_2x2_matrix(matrix, f"{label} dense bulk sample {i}")


class _BulkDiffusivitySimplexLinear2D:
    """
    Linear scattered interpolator over simplex-valid ternary bulk samples.

    This mode is intended for composition regions near the ternary simplex
    boundary where a rectangular ``continuous_grid`` would require invalid
    corner compositions. Matrix components are interpolated after the same
    signed cube-root transform used by the regular-grid spline. Queries outside
    the sampled convex hull fall back to the nearest sampled point.
    """

    def __init__(self, points, matrices):
        if LinearNDInterpolator is None or NearestNDInterpolator is None:  # pragma: no cover - SciPy is a package dependency.
            raise ImportError("LinearNDInterpolator and NearestNDInterpolator are required for simplex_linear diffusivity interpolation.")
        self.points = np.asarray(points, dtype=np.float64)
        values = np.asarray(matrices, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("simplex_linear bulk diffusivity points must have shape (n_points, 2).")
        if self.points.shape[0] < 3:
            raise ValueError("simplex_linear bulk diffusivity requires at least three sample points.")
        if values.shape != (self.points.shape[0], 2, 2):
            raise ValueError("simplex_linear bulk diffusivity matrices must have shape (n_points, 2, 2).")
        transformed = _signed_cuberoot(values.reshape(self.points.shape[0], 4))
        self._linear = LinearNDInterpolator(self.points, transformed, fill_value=np.nan)
        self._nearest = NearestNDInterpolator(self.points, transformed)

    def evaluate(self, values):
        values = np.asarray(values, dtype=np.float64)
        transformed = np.asarray(self._linear(values), dtype=np.float64)
        missing = ~np.all(np.isfinite(transformed), axis=1)
        if np.any(missing):
            transformed[missing] = np.asarray(self._nearest(values[missing]), dtype=np.float64)
        return (transformed ** 3).reshape(values.shape[0], 2, 2)

    def validate_dense(self, label):
        matrices = self.evaluate(self.points)
        for i, matrix in enumerate(matrices):
            _validate_positive_2x2_matrix(matrix, f"{label} training sample {i}")


def _validate_tieline_phases(tieline_phases):
    phases = tuple(str(p) for p in tieline_phases)
    if len(phases) != 2:
        raise ValueError("tieline_phases must contain exactly two phases.")
    if phases[0] == phases[1]:
        raise ValueError("tieline_phases must contain two distinct phases.")
    return phases


def _sample_label(eta=None, composition=None):
    pieces = []
    if eta is not None:
        pieces.append(f"eta={float(eta):.8g}")
    if composition is not None:
        pieces.append(f"composition={np.asarray(composition, dtype=np.float64).reshape(-1).tolist()}")
    return ", ".join(pieces) if pieces else "sample"


def _extract_expected_tieline(meta, tieline_phases, elements, min_composition, eta=None, composition=None):
    """
    Extracts a phase-ordered tie-line from equilibrium metadata.

    Sampling a tie-line is only valid when the equilibrium result contains
    exactly the two expected phases. This helper rejects missing, extra,
    duplicate, or unlabeled endpoint phases instead of trying to infer intent.
    """
    label = _sample_label(eta=eta, composition=composition)
    expected = tuple(tieline_phases)
    expected_set = set(expected)
    if not isinstance(meta, dict) or "endpoints" not in meta:
        raise ValueError(f"Tie-line metadata is missing endpoints for {label}; expected phases {expected}.")
    endpoints = meta["endpoints"]
    if len(endpoints) != 2:
        observed = [endpoint.get("phase", None) if isinstance(endpoint, dict) else None for endpoint in endpoints]
        raise ValueError(f"Expected exactly two tie-line endpoints for {label}; expected phases {expected}, observed {observed}.")

    by_phase = {}
    observed = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict) or "phase" not in endpoint or "composition" not in endpoint:
            raise ValueError(f"Unlabeled tie-line endpoint for {label}; expected phases {expected}.")
        phase = endpoint["phase"]
        observed.append(phase)
        if phase in by_phase:
            raise ValueError(f"Duplicate tie-line phase for {label}; expected phases {expected}, observed {observed}.")
        by_phase[phase] = endpoint["composition"]

    observed_set = set(observed)
    if observed_set != expected_set:
        raise ValueError(f"Unexpected tie-line phases for {label}; expected {expected}, observed {observed}.")

    out = []
    for phase in expected:
        comp = _as_independent_ternary_components(by_phase[phase], elements, f"tie-line endpoint for phase {phase}")
        out.append(_validate_independent_composition(comp, min_composition, f"tie-line endpoint for phase {phase}"))
    return tuple(out)


def _call_interfacial_composition(thermodynamics, composition, temperature, precipitate_phase, eta):
    """
    Calls a thermodynamics object's interfacial-composition API with metadata.
    """
    try:
        result = thermodynamics.getInterfacialComposition(
            composition,
            temperature,
            0,
            precPhase=precipitate_phase,
            returnMeta=True,
        )
    except TypeError:
        result = thermodynamics.getInterfacialComposition(
            composition,
            temperature,
            precPhase=precipitate_phase,
            returnMeta=True,
        )
    if len(result) != 3:
        raise ValueError(f"getInterfacialComposition did not return metadata at eta={float(eta):.8g}.")
    return result


def _bulk_points_from_grids(diffusivity_bulk_grids):
    if diffusivity_bulk_grids is None:
        return None
    axes = [np.asarray(axis, dtype=np.float64).reshape(-1) for axis in diffusivity_bulk_grids]
    if len(axes) != 2:
        raise ValueError("diffusivity_bulk_grids must contain exactly two component axes.")
    return np.asarray(np.meshgrid(*axes), dtype=np.float64).T.reshape(-1, 2)


def _bulk_points_from_bbox(diffusivity_bulk_bbox, diffusivity_bulk_spacing):
    if diffusivity_bulk_bbox is None:
        return None
    if diffusivity_bulk_spacing is None:
        raise ValueError("diffusivity_bulk_spacing is required when diffusivity_bulk_bbox is provided.")
    bbox = np.asarray(diffusivity_bulk_bbox, dtype=np.float64)
    if bbox.shape != (2, 2):
        raise ValueError("diffusivity_bulk_bbox must have shape (2, 2).")
    spacing = float(diffusivity_bulk_spacing)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("diffusivity_bulk_spacing must be a positive finite value.")
    axes = [np.arange(bbox[i, 0], bbox[i, 1] + 0.5 * spacing, spacing, dtype=np.float64) for i in range(2)]
    return np.asarray(np.meshgrid(*axes), dtype=np.float64).T.reshape(-1, 2)


class TernaryMovingBoundaryThermodynamicsSurrogate:
    """
    Isothermal ternary surrogate for moving-boundary tie-lines and diffusivity.

    The surrogate stores a phase-ordered family of interface tie-lines indexed
    by scalar ``eta`` and phase-labeled interdiffusivity matrices indexed by
    independent composition. Tie-lines are linearly interpolated in ``eta``.
    By default diffusivities are selected by nearest sampled composition for
    archive compatibility. With ``diffusivity_interpolation='continuous_grid'``,
    interface diffusivities are interpolated only from tie-line endpoint samples
    and general/bulk diffusivities are evaluated from a regular rectangular
    composition grid that lies inside the ternary simplex. Continuous mode runs
    dense build-time matrix validation so implicit composition-dependent bulk
    solves can use cheap runtime evaluations without matrix repair. With
    ``diffusivity_interpolation='simplex_linear'``, general/bulk diffusivities
    are linearly interpolated over simplex-valid scattered samples with nearest
    fallback outside the sampled convex hull.
    """

    def __init__(
        self,
        elements,
        phases,
        tieline_phases,
        temperature,
        eta_samples,
        tieline_compositions,
        diffusivity_compositions,
        diffusivities,
        min_composition=1e-10,
        metadata=None,
        diffusivity_interpolation=_DIFFUSIVITY_INTERPOLATION_NEAREST,
        diffusivity_bulk_grids=None,
    ):
        self.elements = tuple(str(e) for e in elements)
        self.phases = tuple(str(p) for p in phases)
        self.tieline_phases = _validate_tieline_phases(tieline_phases)
        if len(self.elements) != 3:
            raise ValueError("TernaryMovingBoundaryThermodynamicsSurrogate requires exactly three elements.")
        for phase in self.tieline_phases:
            if phase not in self.phases:
                raise ValueError(f"tieline phase '{phase}' must be included in phases.")
        self.temperature = float(temperature)
        if not np.isfinite(self.temperature):
            raise ValueError("temperature must be finite.")
        self.min_composition = float(min_composition)
        if not np.isfinite(self.min_composition) or self.min_composition < 0.0:
            raise ValueError("min_composition must be nonnegative and finite.")

        self.eta_samples = np.asarray(eta_samples, dtype=np.float64).reshape(-1)
        if self.eta_samples.size < 2:
            raise ValueError("At least two eta_samples are required.")
        if not np.all(np.isfinite(self.eta_samples)) or not np.all(np.diff(self.eta_samples) > 0.0):
            raise ValueError("eta_samples must be finite and strictly increasing.")
        self.eta_bounds = (float(self.eta_samples[0]), float(self.eta_samples[-1]))

        self.tieline_compositions = {}
        for phase in self.tieline_phases:
            values = np.asarray(tieline_compositions[phase], dtype=np.float64)
            if values.shape != (self.eta_samples.size, 2):
                raise ValueError(f"tieline_compositions[{phase!r}] must have shape (n_eta, 2).")
            for i, comp in enumerate(values):
                _validate_independent_composition(comp, self.min_composition, f"tie-line sample {i} for phase {phase}")
            self.tieline_compositions[phase] = values.copy()

        self.diffusivity_compositions = self._coerce_diffusivity_samples(diffusivity_compositions, "diffusivity_compositions")
        self.diffusivities = self._coerce_diffusivity_samples(diffusivities, "diffusivities", matrices=True)
        self.metadata = {} if metadata is None else dict(metadata)
        self.diffusivityInterpolation = _coerce_diffusivity_interpolation(diffusivity_interpolation)
        self.diffusivityBulkGridAxes = None
        self._interfaceDiffusivityInterpolators = {}
        self._bulkDiffusivityInterpolators = {}
        if self.diffusivityInterpolation == _DIFFUSIVITY_INTERPOLATION_CONTINUOUS_GRID:
            self._build_continuous_diffusivity_interpolators(diffusivity_bulk_grids)
        elif self.diffusivityInterpolation == _DIFFUSIVITY_INTERPOLATION_SIMPLEX_LINEAR:
            self._build_simplex_linear_diffusivity_interpolators()

    @classmethod
    def from_database(
        cls,
        thermodynamics=None,
        *,
        database=None,
        elements=None,
        phases=None,
        tieline_phases=None,
        temperature=None,
        probe_start=None,
        probe_end=None,
        eta_samples=None,
        precipitate_phase=None,
        diffusivity_bulk_points=None,
        diffusivity_bulk_grids=None,
        diffusivity_bulk_bbox=None,
        diffusivity_bulk_spacing=None,
        diffusivity_interpolation=_DIFFUSIVITY_INTERPOLATION_NEAREST,
        min_composition=1e-10,
        thermodynamics_kwargs=None,
        validation_database=None,
        validation_thermodynamics_kwargs=None,
    ):
        """
        Samples tie-lines and diffusivities from a thermodynamics source.

        Every tie-line sample is accepted only if equilibrium metadata contains
        exactly the two explicitly requested ``tieline_phases``. Continuous
        bulk diffusivity interpolation requires explicit ``diffusivity_bulk_grids``
        so the rectangular sampling domain and array ordering are reproducible.
        Simplex-linear interpolation accepts scattered ``diffusivity_bulk_points``
        and can also sample the simplex-valid subset of ``diffusivity_bulk_grids``.
        """
        if tieline_phases is None:
            raise ValueError("tieline_phases must be provided explicitly.")
        tieline_phases = _validate_tieline_phases(tieline_phases)
        diffusivity_interpolation = _coerce_diffusivity_interpolation(diffusivity_interpolation)
        bulk_grid_axes = None
        if diffusivity_interpolation == _DIFFUSIVITY_INTERPOLATION_CONTINUOUS_GRID:
            if diffusivity_bulk_grids is None:
                raise ValueError("diffusivity_bulk_grids is required when diffusivity_interpolation is 'continuous_grid'.")
            if diffusivity_bulk_points is not None or diffusivity_bulk_bbox is not None or diffusivity_bulk_spacing is not None:
                raise ValueError("continuous_grid diffusivity interpolation uses diffusivity_bulk_grids only.")
            bulk_grid_axes = _bulk_grid_axes(diffusivity_bulk_grids, float(min_composition))
        elif diffusivity_interpolation == _DIFFUSIVITY_INTERPOLATION_SIMPLEX_LINEAR and diffusivity_bulk_grids is not None:
            bulk_grid_axes = tuple(np.asarray(axis, dtype=np.float64).reshape(-1).copy() for axis in diffusivity_bulk_grids)
            for i, axis in enumerate(bulk_grid_axes):
                if axis.size < 2:
                    raise ValueError(f"diffusivity_bulk_grids axis {i} must contain at least two samples.")
                if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
                    raise ValueError(f"diffusivity_bulk_grids axis {i} must be finite and strictly increasing.")
        validation_database_source = validation_database
        if validation_database_source is None and thermodynamics is None and isinstance(database, (str, Path)):
            validation_database_source = database
        validation_metadata = {}
        if validation_database_source is not None:
            validation_metadata["validation_database"] = str(validation_database_source)
            validation_metadata["validation_thermodynamics_kwargs"] = (
                dict(thermodynamics_kwargs or {}) if validation_thermodynamics_kwargs is None else dict(validation_thermodynamics_kwargs)
            )

        if temperature is None:
            raise ValueError("temperature must be provided.")
        temperature = float(temperature)
        if thermodynamics is None:
            if database is None or elements is None or phases is None:
                raise ValueError("database, elements, and phases are required when thermodynamics is not provided.")
            kwargs = {} if thermodynamics_kwargs is None else dict(thermodynamics_kwargs)
            thermodynamics = MulticomponentThermodynamics(database, list(elements), list(phases), **kwargs)
        if elements is None:
            elements = getattr(thermodynamics, "elements", None)
        if phases is None:
            phases = getattr(thermodynamics, "phases", None)
        if elements is None or phases is None:
            raise ValueError("elements and phases must be provided or available on thermodynamics.")
        elements = tuple(str(e) for e in elements)
        phases = tuple(str(p) for p in phases)
        if len(elements) != 3:
            raise ValueError("from_database currently supports ternary systems only.")
        for phase in tieline_phases:
            if phase not in phases:
                raise ValueError(f"tieline phase '{phase}' must be included in phases.")

        if eta_samples is None: 
            raise ValueError("eta_samples must be provided.")
        eta_samples = np.asarray(eta_samples, dtype=np.float64).reshape(-1)
        if eta_samples.size < 2:
            raise ValueError("At least two eta_samples are required.")
        if not np.all(np.isfinite(eta_samples)) or not np.all(np.diff(eta_samples) > 0.0):
            raise ValueError("eta_samples must be finite and strictly increasing.")
        probe_start = _as_independent_ternary_components(probe_start, elements, "probe_start")
        probe_end = _as_independent_ternary_components(probe_end, elements, "probe_end")
        precipitate_phase = tieline_phases[1] if precipitate_phase is None else str(precipitate_phase)

        tieline_values = {phase: [] for phase in tieline_phases}
        interface_diff_x = {phase: [] for phase in tieline_phases}
        interface_diff_d = {phase: [] for phase in tieline_phases}
        for eta in eta_samples:
            fraction = (float(eta) - float(eta_samples[0])) / (float(eta_samples[-1]) - float(eta_samples[0]))
            probe = probe_start + fraction * (probe_end - probe_start)
            _, _, meta = _call_interfacial_composition(thermodynamics, probe, temperature, precipitate_phase, eta)
            endpoints = _extract_expected_tieline(
                meta,
                tieline_phases,
                elements,
                float(min_composition),
                eta=eta,
                composition=probe,
            )
            for phase, comp in zip(tieline_phases, endpoints):
                tieline_values[phase].append(comp)
                interface_diff_x[phase].append(comp)
                interface_diff_d[phase].append(
                    _validate_2x2_matrix(
                        thermodynamics.getInterdiffusivity(comp, temperature, phase=phase),
                        f"interface diffusivity for phase {phase} at eta={float(eta):.8g}",
                    )
                )

        tieline_values = {phase: np.asarray(values, dtype=np.float64) for phase, values in tieline_values.items()}
        interface_diff_x = {phase: np.asarray(values, dtype=np.float64) for phase, values in interface_diff_x.items()}
        interface_diff_d = {phase: np.asarray(values, dtype=np.float64) for phase, values in interface_diff_d.items()}

        if diffusivity_interpolation == _DIFFUSIVITY_INTERPOLATION_CONTINUOUS_GRID:
            bulk_points = _bulk_points_from_axes(bulk_grid_axes)
            general_diff_x = {phase: [] for phase in tieline_phases}
            general_diff_d = {phase: [] for phase in tieline_phases}
            for phase in tieline_phases:
                for point in bulk_points:
                    general_diff_x[phase].append(point)
                    general_diff_d[phase].append(
                        _validate_2x2_matrix(
                            thermodynamics.getInterdiffusivity(point, temperature, phase=phase),
                            f"bulk diffusivity for phase {phase} at composition {point.tolist()}",
                        )
                    )
        else:
            grid_points = (
                _bulk_points_from_simplex_axes(bulk_grid_axes, float(min_composition))
                if diffusivity_interpolation == _DIFFUSIVITY_INTERPOLATION_SIMPLEX_LINEAR
                else _bulk_points_from_grids(diffusivity_bulk_grids)
            )
            bulk_points = cls._merge_bulk_points(
                elements,
                min_composition,
                diffusivity_bulk_points,
                grid_points,
                _bulk_points_from_bbox(diffusivity_bulk_bbox, diffusivity_bulk_spacing),
            )
            general_diff_x = {phase: [*interface_diff_x[phase]] for phase in tieline_phases}
            general_diff_d = {phase: [*interface_diff_d[phase]] for phase in tieline_phases}
            if bulk_points is not None:
                for phase in tieline_phases:
                    for point in bulk_points:
                        general_diff_x[phase].append(point)
                        general_diff_d[phase].append(
                            _validate_2x2_matrix(
                                thermodynamics.getInterdiffusivity(point, temperature, phase=phase),
                                f"bulk diffusivity for phase {phase} at composition {point.tolist()}",
                            )
                        )
        general_diff_x = {phase: np.asarray(values, dtype=np.float64) for phase, values in general_diff_x.items()}
        general_diff_d = {phase: np.asarray(values, dtype=np.float64) for phase, values in general_diff_d.items()}

        return cls(
            elements=elements,
            phases=phases,
            tieline_phases=tieline_phases,
            temperature=temperature,
            eta_samples=eta_samples,
            tieline_compositions=tieline_values,
            diffusivity_compositions={"interface": interface_diff_x, "general": general_diff_x},
            diffusivities={"interface": interface_diff_d, "general": general_diff_d},
            min_composition=min_composition,
            diffusivity_interpolation=diffusivity_interpolation,
            diffusivity_bulk_grids=bulk_grid_axes,
            metadata={
                "source": "from_database",
                "probe_start": probe_start.tolist(),
                "probe_end": probe_end.tolist(),
                "precipitate_phase": precipitate_phase,
                **validation_metadata,
            },
        )

    @staticmethod
    def _merge_bulk_points(elements, min_composition, *point_sets):
        merged = []
        for points in point_sets:
            if points is None:
                continue
            arr = np.asarray(points, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("bulk diffusivity sample points must have shape (n_points, 2).")
            for i, point in enumerate(arr):
                comp = _as_independent_ternary_components(point, elements, f"bulk diffusivity point {i}")
                _validate_independent_composition(comp, float(min_composition), f"bulk diffusivity point {i}")
                merged.append(comp)
        if not merged:
            return None
        return np.unique(np.asarray(merged, dtype=np.float64), axis=0)

    def _coerce_diffusivity_samples(self, samples, name, matrices=False):
        out = {}
        for context in ("interface", "general"):
            if context not in samples:
                raise ValueError(f"{name} must include context '{context}'.")
            out[context] = {}
            for phase in self.tieline_phases:
                if phase not in samples[context]:
                    raise ValueError(f"{name}[{context!r}] must include phase '{phase}'.")
                values = np.asarray(samples[context][phase], dtype=np.float64)
                if matrices:
                    if values.ndim != 3 or values.shape[1:] != (2, 2):
                        raise ValueError(f"{name}[{context!r}][{phase!r}] must have shape (n_samples, 2, 2).")
                    for i, matrix in enumerate(values):
                        _validate_2x2_matrix(matrix, f"{name}[{context!r}][{phase!r}][{i}]")
                else:
                    if values.ndim != 2 or values.shape[1] != 2:
                        raise ValueError(f"{name}[{context!r}][{phase!r}] must have shape (n_samples, 2).")
                    for i, comp in enumerate(values):
                        _validate_independent_composition(comp, self.min_composition, f"{name}[{context!r}][{phase!r}][{i}]")
                if values.shape[0] < 1:
                    raise ValueError(f"{name}[{context!r}][{phase!r}] must contain at least one sample.")
                out[context][phase] = values.copy()
        return out

    def _build_continuous_diffusivity_interpolators(self, diffusivity_bulk_grids):
        """
        Builds interface and bulk spline evaluators for continuous diffusivity.

        The bulk mode assumes general diffusivity samples are ordered as an
        ``indexing='ij'`` mesh over ``diffusivity_bulk_grids``. Interface mode
        remains tied only to the phase endpoint samples along ``eta``.
        """
        axes = _bulk_grid_axes(diffusivity_bulk_grids, self.min_composition)
        if axes is None:
            raise ValueError("diffusivity_bulk_grids is required for continuous_grid diffusivity interpolation.")
        self.diffusivityBulkGridAxes = axes
        expected_points = axes[0].size * axes[1].size
        for phase in self.tieline_phases:
            if self.diffusivity_compositions["interface"][phase].shape[0] != self.eta_samples.size:
                raise ValueError(f"continuous interface diffusivity samples for phase {phase} must match eta_samples.")
            if self.diffusivity_compositions["general"][phase].shape[0] != expected_points:
                raise ValueError(
                    f"continuous bulk diffusivity samples for phase {phase} must match the regular grid size."
                )
            expected_grid = _bulk_points_from_axes(axes)
            if not np.allclose(self.diffusivity_compositions["general"][phase], expected_grid, rtol=0.0, atol=1e-14):
                raise ValueError(
                    f"continuous bulk diffusivity compositions for phase {phase} must be ordered on diffusivity_bulk_grids."
                )

            interface = _InterfaceDiffusivitySpline1D(
                self.eta_samples,
                self.tieline_compositions[phase],
                self.diffusivities["interface"][phase],
            )
            bulk = _BulkDiffusivityGridSpline2D(
                axes[0],
                axes[1],
                self.diffusivities["general"][phase].reshape(axes[0].size, axes[1].size, 2, 2),
            )
            interface.validate_dense(f"continuous interface diffusivity for phase {phase}")
            bulk.validate_dense(f"continuous bulk diffusivity for phase {phase}")
            self._interfaceDiffusivityInterpolators[phase] = interface
            self._bulkDiffusivityInterpolators[phase] = bulk

    def _build_simplex_linear_diffusivity_interpolators(self):
        """
        Builds interface splines and simplex-linear bulk diffusivity evaluators.

        The bulk samples may be scattered over the valid ternary simplex. This
        avoids the rectangular-grid restriction of ``continuous_grid`` for
        systems whose useful composition path sits near ``x0 + x1 = 1``.
        """
        for phase in self.tieline_phases:
            if self.diffusivity_compositions["interface"][phase].shape[0] != self.eta_samples.size:
                raise ValueError(f"simplex_linear interface diffusivity samples for phase {phase} must match eta_samples.")
            interface = _InterfaceDiffusivitySpline1D(
                self.eta_samples,
                self.tieline_compositions[phase],
                self.diffusivities["interface"][phase],
            )
            bulk = _BulkDiffusivitySimplexLinear2D(
                self.diffusivity_compositions["general"][phase],
                self.diffusivities["general"][phase],
            )
            interface.validate_dense(f"simplex-linear interface diffusivity for phase {phase}")
            bulk.validate_dense(f"simplex-linear bulk diffusivity for phase {phase}")
            self._interfaceDiffusivityInterpolators[phase] = interface
            self._bulkDiffusivityInterpolators[phase] = bulk

    def _validate_temperature(self, T):
        if T is None:
            return
        values = np.asarray(T, dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("T must be finite for the isothermal surrogate.")
        if not np.allclose(values, self.temperature, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"TernaryMovingBoundaryThermodynamicsSurrogate is isothermal at {self.temperature}; "
                f"received T={values.reshape(-1).tolist()}."
            )

    def _phase_index(self, phase):
        phase = self.tieline_phases[0] if phase is None else str(phase)
        if phase not in self.tieline_phases:
            raise ValueError(f"Unknown phase '{phase}'. Expected one of {self.tieline_phases}.")
        return phase

    def interface_compositions(self, eta):
        """Returns the phase-ordered independent interface compositions."""
        eta = float(np.clip(float(eta), self.eta_bounds[0], self.eta_bounds[1]))
        return tuple(
            np.asarray(
                [
                    np.interp(eta, self.eta_samples, self.tieline_compositions[phase][:, component])
                    for component in range(2)
                ],
                dtype=np.float64,
            )
            for phase in self.tieline_phases
        )

    def _signed_tieline_distance(self, composition, eta):
        """
        Returns the signed normal distance from a composition to a tie-line.

        A zero means the composition is collinear with the phase-endpoint
        segment at ``eta``; normalization keeps the value in composition units.
        """
        left, right = self.interface_compositions(eta)
        direction = right - left
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length <= 0.0:
            return np.nan
        delta = composition - left
        return float(direction[0] * delta[1] - direction[1] * delta[0]) / length

    def getTielineOfGlobalComposition(
        self,
        composition,
        T=None,
        returnMeta=False,
        *,
        tolerance=1e-8,
        xtol=1e-10,
        maxiter=100,
        eta_bracket=None,
        **kwargs,
    ):
        """
        Finds the surrogate tie-line containing a global composition.

        The method solves a 1D collinearity condition over ``eta`` and then
        checks the lever-rule fraction, rejecting compositions that sit on a
        tie-line extension rather than inside the two-phase segment.
        """
        self._validate_temperature(T)
        target = _as_independent_ternary_components(composition, self.elements, "global composition")
        target = _validate_independent_composition(target, self.min_composition, "global composition")
        tolerance, xtol, maxiter = float(tolerance), float(xtol), int(maxiter)
        for name, value in (("tolerance", tolerance), ("xtol", xtol)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if maxiter <= 0:
            raise ValueError("maxiter must be positive.")

        if eta_bracket is None:
            eta_lower, eta_upper = self.eta_bounds
        else:
            bracket = np.asarray(eta_bracket, dtype=np.float64).reshape(-1)
            if bracket.size != 2:
                raise ValueError("eta_bracket must contain exactly two values.")
            eta_lower, eta_upper = float(bracket[0]), float(bracket[1])
            if (
                not np.isfinite(eta_lower)
                or not np.isfinite(eta_upper)
                or eta_upper <= eta_lower
                or eta_lower < self.eta_bounds[0]
                or eta_upper > self.eta_bounds[1]
            ):
                raise ValueError(f"eta_bracket must lie within eta_bounds={self.eta_bounds} and be increasing.")

        interior = self.eta_samples[(self.eta_samples > eta_lower) & (self.eta_samples < eta_upper)]
        eta_grid = np.asarray([eta_lower, *interior.tolist(), eta_upper], dtype=np.float64)
        signed_values = np.asarray([self._signed_tieline_distance(target, eta) for eta in eta_grid], dtype=np.float64)

        candidates = []
        eta_mid = 0.5 * (eta_lower + eta_upper)

        def add_candidate(eta):
            left, right = self.interface_compositions(float(eta))
            direction = right - left
            denom = float(np.dot(direction, direction))
            if not np.isfinite(denom) or denom <= 0.0:
                return
            phase_fraction = float(np.dot(target - left, direction) / denom)
            residual = left + phase_fraction * direction - target
            residual_norm = float(np.max(np.abs(residual)))
            if not np.isfinite(phase_fraction) or not np.isfinite(residual_norm):
                return
            candidates.append((residual_norm, abs(float(eta) - eta_mid), float(eta), phase_fraction, left, right, residual))

        for eta, value in zip(eta_grid, signed_values):
            if np.isfinite(value) and abs(value) <= tolerance:
                add_candidate(eta)

        for i in range(eta_grid.size - 1):
            f0 = signed_values[i]
            f1 = signed_values[i + 1]
            if not np.isfinite(f0) or not np.isfinite(f1) or f0 == 0.0 or f1 == 0.0 or f0 * f1 > 0.0:
                continue
            result = optimize.root_scalar(
                lambda eta: self._signed_tieline_distance(target, eta),
                bracket=(float(eta_grid[i]), float(eta_grid[i + 1])),
                method="brentq",
                xtol=xtol,
                maxiter=maxiter,
            )
            if result.converged:
                add_candidate(result.root)

        if not candidates:
            finite_values = signed_values[np.isfinite(signed_values)]
            best_distance = None if finite_values.size == 0 else float(np.min(np.abs(finite_values)))
            raise ValueError(
                "Could not locate a surrogate tie-line containing global composition "
                f"{target.tolist()} within eta range [{eta_lower}, {eta_upper}]. "
                f"Best sampled signed distance was {best_distance}."
            )

        candidates.sort()
        valid = [c for c in candidates if c[0] <= tolerance and -tolerance <= c[3] <= 1.0 + tolerance]
        residual_norm, _, eta, phase_fraction, left, right, residual = (valid or candidates)[0]
        if residual_norm > tolerance:
            raise ValueError(
                "No surrogate tie-line matched global composition "
                f"{target.tolist()} within tolerance {tolerance}; best residual was {residual_norm} at eta={eta}."
            )
        if phase_fraction < -tolerance or phase_fraction > 1.0 + tolerance:
            raise ValueError(
                "Global composition lies on a surrogate tie-line extension, not inside the two-phase segment: "
                f"composition={target.tolist()}, eta={eta}, phase_fraction={phase_fraction}."
            )

        phase_fraction = float(np.clip(phase_fraction, 0.0, 1.0))
        left, right = left.copy(), right.copy()
        if not returnMeta:
            return left, right
        metadata = {
            "eta": eta,
            "phase_fraction": phase_fraction,
            "phase_fraction_phase": self.tieline_phases[1],
            "endpoint_phases": self.tieline_phases,
            "global_composition": target.copy(),
            "residual": residual.copy(),
            "residual_norm": residual_norm,
            "endpoints": (
                {"phase": self.tieline_phases[0], "composition": left.copy()},
                {"phase": self.tieline_phases[1], "composition": right.copy()},
            ),
        }
        return left, right, metadata

    def getInterfacialComposition(self, eta, T=None, returnMeta=False, **kwargs):
        """Returns surrogate tie-line compositions for thermodynamics-like APIs."""
        self._validate_temperature(T)
        left, right = self.interface_compositions(float(eta))
        if not returnMeta:
            return left, right
        metadata = {
            "endpoint_phases": self.tieline_phases,
            "endpoints": (
                {"phase": self.tieline_phases[0], "composition": left.copy()},
                {"phase": self.tieline_phases[1], "composition": right.copy()},
            ),
        }
        return left, right, metadata

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        """
        Returns surrogate 2x2 interdiffusivity matrices.

        ``query_context='interface'`` uses tie-line endpoint diffusivity data.
        All other contexts use the general/bulk diffusivity source. In the
        default ``nearest`` mode this preserves nearest-sampled behavior. In
        ``continuous_grid`` mode interface values are 1D eta splines and bulk
        values are tensor-product splines over the configured rectangular
        composition grid. In ``simplex_linear`` mode interface values use the
        same 1D eta splines and bulk values use scattered linear interpolation
        over simplex-valid training points.
        """
        self._validate_temperature(T)
        phase = self._phase_index(phase)
        context = "interface" if query_context == "interface" else "general"
        values = np.asarray(x, dtype=np.float64)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        if values.shape[1] != 2:
            raise ValueError("getInterdiffusivity expects independent ternary compositions with shape (n, 2).")
        if self.diffusivityInterpolation in {
            _DIFFUSIVITY_INTERPOLATION_CONTINUOUS_GRID,
            _DIFFUSIVITY_INTERPOLATION_SIMPLEX_LINEAR,
        }:
            if context == "interface":
                out = self._interfaceDiffusivityInterpolators[phase].evaluate(values)
            else:
                out = self._bulkDiffusivityInterpolators[phase].evaluate(values)
            return out[0].copy() if single else out.copy()

        samples_x = self.diffusivity_compositions[context][phase]
        samples_d = self.diffusivities[context][phase]
        deltas = values[:, np.newaxis, :] - samples_x[np.newaxis, :, :]
        indices = np.argmin(np.sum(deltas * deltas, axis=2), axis=1)
        out = samples_d[indices]
        return out[0].copy() if single else out.copy()

    def validate_diffusivity_matrices(
        self,
        *,
        phases=None,
        matrix_interface_eta_count=201,
        matrix_bulk_grid_counts=(101, 101),
        matrix_bulk_axes=None,
        eigen_imag_tol=1e-12,
        eigen_real_min=1e-14,
        raise_on_invalid=False,
    ):
        """
        Densely validates surrogate-generated interdiffusivity matrices.

        Interface samples are evaluated at phase endpoint compositions with
        ``query_context='interface'``. Bulk samples are evaluated over a dense
        independent-composition grid with ``query_context='general'``. This
        method is intentionally opt-in because dense validation can be expensive.
        """
        phases = self._validation_phases(phases)
        interface_samples = self._interface_validation_samples(matrix_interface_eta_count)
        bulk_samples = self._bulk_validation_samples(matrix_bulk_grid_counts, matrix_bulk_axes)
        report = {
            "interface": self._validate_diffusivity_regime(
                "interface",
                phases,
                interface_samples,
                eigen_imag_tol=eigen_imag_tol,
                eigen_real_min=eigen_real_min,
            ),
            "bulk": self._validate_diffusivity_regime(
                "general",
                phases,
                bulk_samples,
                eigen_imag_tol=eigen_imag_tol,
                eigen_real_min=eigen_real_min,
            ),
        }
        report["summary"] = self._combine_validation_summaries(report)
        if raise_on_invalid and not report["summary"]["ok"]:
            raise ValueError(
                "Surrogate diffusivity validation failed: "
                f"{report['summary']['invalid_count']} invalid matrices across {report['summary']['sample_count']} samples."
            )
        return report

    def compare_diffusivity_to_ground_truth(
        self,
        *,
        thermodynamics=None,
        database=None,
        thermodynamics_kwargs=None,
        phases=None,
        error_interface_eta_count=201,
        error_bulk_grid_counts=(101, 101),
        error_bulk_axes=None,
        relative_error_floor=1e-300,
        eigen_imag_tol=1e-12,
        eigen_real_min=1e-14,
        raise_on_error=False,
    ):
        """
        Compares surrogate diffusivity matrices to ground-truth thermodynamics.

        Ground truth is sampled at the same interface endpoint and bulk
        composition points used for surrogate queries. If ``thermodynamics`` is
        omitted, the method rebuilds a ``MulticomponentThermodynamics`` object
        from stored validation metadata or an explicit ``database`` argument.
        """
        phases = self._validation_phases(phases)
        floor = float(relative_error_floor)
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("relative_error_floor must be positive and finite.")
        truth = self._validation_thermodynamics(
            thermodynamics=thermodynamics,
            database=database,
            thermodynamics_kwargs=thermodynamics_kwargs,
        )
        interface_samples = self._interface_validation_samples(error_interface_eta_count)
        bulk_samples = self._bulk_validation_samples(error_bulk_grid_counts, error_bulk_axes)
        report = {
            "interface": self._compare_diffusivity_regime(
                "interface",
                phases,
                interface_samples,
                truth,
                relative_error_floor=floor,
                eigen_imag_tol=eigen_imag_tol,
                eigen_real_min=eigen_real_min,
            ),
            "bulk": self._compare_diffusivity_regime(
                "general",
                phases,
                bulk_samples,
                truth,
                relative_error_floor=floor,
                eigen_imag_tol=eigen_imag_tol,
                eigen_real_min=eigen_real_min,
            ),
        }
        report["summary"] = self._combine_comparison_summaries(report)
        if raise_on_error and not report["summary"]["ok"]:
            raise ValueError("Surrogate diffusivity ground-truth comparison produced non-finite values.")
        return report

    def _validation_phases(self, phases):
        if phases is None:
            return self.tieline_phases
        out = tuple(str(phase) for phase in phases)
        for phase in out:
            self._phase_index(phase)
        return out

    def _interface_validation_samples(self, eta_count):
        eta_count = _coerce_positive_int(eta_count, "interface eta count")
        eta = np.linspace(self.eta_bounds[0], self.eta_bounds[1], eta_count, dtype=np.float64)
        compositions = {}
        for phase in self.tieline_phases:
            compositions[phase] = np.asarray(
                [
                    np.interp(eta, self.eta_samples, self.tieline_compositions[phase][:, component])
                    for component in range(2)
                ],
                dtype=np.float64,
            ).T
        return {
            "eta": eta,
            "compositions": compositions,
            "axes": None,
        }

    def _bulk_validation_samples(self, grid_counts, axes):
        counts = _coerce_grid_counts(grid_counts, "bulk grid counts")
        if axes is None:
            axes = self._default_bulk_validation_axes(counts)
        else:
            axes = self._coerce_validation_axes(axes)
        points = _bulk_points_from_axes(axes)
        points = points[_valid_simplex_mask(points, self.min_composition)]
        if points.shape[0] == 0:
            raise ValueError("bulk validation sampling produced no valid ternary composition points.")
        return {
            "points": points,
            "axes": axes,
            "grid_counts": counts,
        }

    def _default_bulk_validation_axes(self, counts):
        if self.diffusivityBulkGridAxes is not None:
            bounds = np.asarray(
                [
                    [self.diffusivityBulkGridAxes[0][0], self.diffusivityBulkGridAxes[0][-1]],
                    [self.diffusivityBulkGridAxes[1][0], self.diffusivityBulkGridAxes[1][-1]],
                ],
                dtype=np.float64,
            )
        else:
            samples = np.concatenate(
                [self.diffusivity_compositions["general"][phase] for phase in self.tieline_phases],
                axis=0,
            )
            bounds = np.asarray([np.min(samples, axis=0), np.max(samples, axis=0)], dtype=np.float64).T
        return _densified_axes_from_bounds(bounds, counts)

    def _coerce_validation_axes(self, axes):
        axes = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in axes)
        if len(axes) != 2:
            raise ValueError("bulk validation axes must contain exactly two component axes.")
        for i, axis in enumerate(axes):
            if axis.size < 1:
                raise ValueError(f"bulk validation axis {i} must contain at least one value.")
            if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
                raise ValueError(f"bulk validation axis {i} must be finite and strictly increasing.")
        return tuple(axis.copy() for axis in axes)

    def _validate_diffusivity_regime(self, context, phases, samples, *, eigen_imag_tol, eigen_real_min):
        regime = "interface" if context == "interface" else "bulk"
        phase_reports = {}
        for phase in phases:
            compositions = samples["compositions"][phase] if context == "interface" else samples["points"]
            matrices = self.getInterdiffusivity(
                compositions,
                self.temperature,
                phase=phase,
                query_context="interface" if context == "interface" else "general",
            )
            diagnostics = _matrix_validity_diagnostics(
                matrices,
                eigen_imag_tol=eigen_imag_tol,
                eigen_real_min=eigen_real_min,
            )
            phase_reports[phase] = {
                "phase": phase,
                "regime": regime,
                "eta": samples.get("eta"),
                "compositions": compositions,
                "matrices": matrices,
                "valid": diagnostics["valid"],
                "finite": diagnostics["finite"],
                "nonzero": diagnostics["nonzero"],
                "eigenvalues": diagnostics["eigenvalues"],
                "scales": diagnostics["scales"],
                "nearest_training_distance": self._nearest_training_distances(compositions, context, phase),
                "summary": self._validity_summary(diagnostics["valid"]),
            }
        return {
            "regime": regime,
            "samples": {key: value for key, value in samples.items() if key != "compositions"},
            "phases": phase_reports,
            "summary": self._combine_phase_summaries(phase_reports),
        }

    def _compare_diffusivity_regime(
        self,
        context,
        phases,
        samples,
        thermodynamics,
        *,
        relative_error_floor,
        eigen_imag_tol,
        eigen_real_min,
    ):
        regime = "interface" if context == "interface" else "bulk"
        phase_reports = {}
        for phase in phases:
            compositions = samples["compositions"][phase] if context == "interface" else samples["points"]
            surrogate_matrices = self.getInterdiffusivity(
                compositions,
                self.temperature,
                phase=phase,
                query_context="interface" if context == "interface" else "general",
            )
            truth_matrices = self._ground_truth_diffusivity_matrices(thermodynamics, compositions, phase)
            absolute_error = np.abs(surrogate_matrices - truth_matrices)
            relative_error = absolute_error / np.maximum(np.abs(truth_matrices), float(relative_error_floor))
            surrogate_diagnostics = _matrix_validity_diagnostics(
                surrogate_matrices,
                eigen_imag_tol=eigen_imag_tol,
                eigen_real_min=eigen_real_min,
            )
            truth_diagnostics = _matrix_validity_diagnostics(
                truth_matrices,
                eigen_imag_tol=eigen_imag_tol,
                eigen_real_min=eigen_real_min,
            )
            finite = (
                np.all(np.isfinite(surrogate_matrices), axis=(1, 2))
                & np.all(np.isfinite(truth_matrices), axis=(1, 2))
                & np.all(np.isfinite(relative_error), axis=(1, 2))
            )
            phase_reports[phase] = {
                "phase": phase,
                "regime": regime,
                "eta": samples.get("eta"),
                "compositions": compositions,
                "surrogate_matrices": surrogate_matrices,
                "truth_matrices": truth_matrices,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "surrogate_valid": surrogate_diagnostics["valid"],
                "truth_valid": truth_diagnostics["valid"],
                "surrogate_eigenvalues": surrogate_diagnostics["eigenvalues"],
                "truth_eigenvalues": truth_diagnostics["eigenvalues"],
                "nearest_training_distance": self._nearest_training_distances(compositions, context, phase),
                "finite": finite,
                "summary": self._comparison_summary(finite, absolute_error, relative_error),
            }
        return {
            "regime": regime,
            "samples": {key: value for key, value in samples.items() if key != "compositions"},
            "phases": phase_reports,
            "summary": self._combine_phase_comparison_summaries(phase_reports),
        }

    def _ground_truth_diffusivity_matrices(self, thermodynamics, compositions, phase):
        matrices = []
        for i, composition in enumerate(np.asarray(compositions, dtype=np.float64)):
            matrices.append(
                _validate_2x2_matrix(
                    thermodynamics.getInterdiffusivity(composition, self.temperature, phase=phase),
                    f"ground-truth diffusivity for phase {phase} at validation sample {i}",
                )
            )
        return np.asarray(matrices, dtype=np.float64)

    def _validation_thermodynamics(self, *, thermodynamics=None, database=None, thermodynamics_kwargs=None):
        if thermodynamics is not None:
            return thermodynamics
        database = self.metadata.get("validation_database") if database is None else database
        if database is None:
            raise ValueError(
                "Ground-truth diffusivity validation requires a thermodynamics object, a database path, "
                "or stored validation_database metadata."
            )
        if thermodynamics_kwargs is None:
            thermodynamics_kwargs = self.metadata.get("validation_thermodynamics_kwargs", {})
        return MulticomponentThermodynamics(
            str(database),
            list(self.elements),
            list(self.phases),
            **dict(thermodynamics_kwargs or {}),
        )

    def _nearest_training_distances(self, compositions, context, phase, chunk_size=10000):
        compositions = np.asarray(compositions, dtype=np.float64)
        training = self.diffusivity_compositions[context][phase]
        distances = np.empty(compositions.shape[0], dtype=np.float64)
        for start in range(0, compositions.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), compositions.shape[0])
            deltas = compositions[start:stop, np.newaxis, :] - training[np.newaxis, :, :]
            distances[start:stop] = np.sqrt(np.min(np.sum(deltas * deltas, axis=2), axis=1))
        return distances

    def _validity_summary(self, valid):
        valid = np.asarray(valid, dtype=bool)
        invalid_count = int(np.count_nonzero(~valid))
        sample_count = int(valid.size)
        return {
            "ok": invalid_count == 0,
            "sample_count": sample_count,
            "valid_count": int(np.count_nonzero(valid)),
            "invalid_count": invalid_count,
            "invalid_fraction": 0.0 if sample_count == 0 else float(invalid_count / sample_count),
        }

    def _comparison_summary(self, finite, absolute_error, relative_error):
        finite = np.asarray(finite, dtype=bool)
        sample_count = int(finite.size)
        nonfinite_count = int(np.count_nonzero(~finite))
        return {
            "ok": nonfinite_count == 0,
            "sample_count": sample_count,
            "finite_count": int(np.count_nonzero(finite)),
            "nonfinite_count": nonfinite_count,
            "max_absolute_error": float(np.nanmax(absolute_error)) if absolute_error.size else np.nan,
            "mean_absolute_error": float(np.nanmean(absolute_error)) if absolute_error.size else np.nan,
            "max_absolute_error_by_component": np.nanmax(absolute_error, axis=0) if absolute_error.size else np.full((2, 2), np.nan),
            "mean_absolute_error_by_component": np.nanmean(absolute_error, axis=0) if absolute_error.size else np.full((2, 2), np.nan),
            "max_relative_error": float(np.nanmax(relative_error)) if relative_error.size else np.nan,
            "mean_relative_error": float(np.nanmean(relative_error)) if relative_error.size else np.nan,
            "max_relative_error_by_component": np.nanmax(relative_error, axis=0) if relative_error.size else np.full((2, 2), np.nan),
            "mean_relative_error_by_component": np.nanmean(relative_error, axis=0) if relative_error.size else np.full((2, 2), np.nan),
        }

    def _combine_phase_summaries(self, phase_reports):
        sample_count = sum(report["summary"]["sample_count"] for report in phase_reports.values())
        invalid_count = sum(report["summary"]["invalid_count"] for report in phase_reports.values())
        return {
            "ok": invalid_count == 0,
            "sample_count": int(sample_count),
            "valid_count": int(sample_count - invalid_count),
            "invalid_count": int(invalid_count),
            "invalid_fraction": 0.0 if sample_count == 0 else float(invalid_count / sample_count),
        }

    def _combine_phase_comparison_summaries(self, phase_reports):
        sample_count = sum(report["summary"]["sample_count"] for report in phase_reports.values())
        nonfinite_count = sum(report["summary"]["nonfinite_count"] for report in phase_reports.values())
        max_relative = [
            report["summary"]["max_relative_error"]
            for report in phase_reports.values()
            if np.isfinite(report["summary"]["max_relative_error"])
        ]
        max_absolute = [
            report["summary"]["max_absolute_error"]
            for report in phase_reports.values()
            if np.isfinite(report["summary"]["max_absolute_error"])
        ]
        return {
            "ok": nonfinite_count == 0,
            "sample_count": int(sample_count),
            "finite_count": int(sample_count - nonfinite_count),
            "nonfinite_count": int(nonfinite_count),
            "max_absolute_error": float(np.max(max_absolute)) if max_absolute else np.nan,
            "max_relative_error": float(np.max(max_relative)) if max_relative else np.nan,
        }

    def _combine_validation_summaries(self, report):
        sample_count = report["interface"]["summary"]["sample_count"] + report["bulk"]["summary"]["sample_count"]
        invalid_count = report["interface"]["summary"]["invalid_count"] + report["bulk"]["summary"]["invalid_count"]
        return {
            "ok": invalid_count == 0,
            "sample_count": int(sample_count),
            "valid_count": int(sample_count - invalid_count),
            "invalid_count": int(invalid_count),
            "invalid_fraction": 0.0 if sample_count == 0 else float(invalid_count / sample_count),
        }

    def _combine_comparison_summaries(self, report):
        sample_count = report["interface"]["summary"]["sample_count"] + report["bulk"]["summary"]["sample_count"]
        nonfinite_count = report["interface"]["summary"]["nonfinite_count"] + report["bulk"]["summary"]["nonfinite_count"]
        max_absolute = [
            report[regime]["summary"]["max_absolute_error"]
            for regime in ("interface", "bulk")
            if np.isfinite(report[regime]["summary"]["max_absolute_error"])
        ]
        max_relative = [
            report[regime]["summary"]["max_relative_error"]
            for regime in ("interface", "bulk")
            if np.isfinite(report[regime]["summary"]["max_relative_error"])
        ]
        return {
            "ok": nonfinite_count == 0,
            "sample_count": int(sample_count),
            "finite_count": int(sample_count - nonfinite_count),
            "nonfinite_count": int(nonfinite_count),
            "max_absolute_error": float(np.max(max_absolute)) if max_absolute else np.nan,
            "max_relative_error": float(np.max(max_relative)) if max_relative else np.nan,
        }

    def clearCache(self):
        """No-op compatibility method for thermodynamics-like objects."""
        return

    def save(self, path):
        """Saves surrogate arrays and metadata to a compressed NPZ archive."""
        path = _as_path_with_npz_suffix(path)
        arrays = {
            "elements": np.asarray(self.elements),
            "phases": np.asarray(self.phases),
            "tieline_phases": np.asarray(self.tieline_phases),
            "temperature": np.asarray(self.temperature, dtype=np.float64),
            "min_composition": np.asarray(self.min_composition, dtype=np.float64),
            "eta_samples": self.eta_samples,
            "tieline_compositions": np.asarray([self.tieline_compositions[p] for p in self.tieline_phases], dtype=np.float64),
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
            "diffusivity_interpolation": np.asarray(self.diffusivityInterpolation),
        }
        if self.diffusivityBulkGridAxes is not None:
            arrays["diffusivity_bulk_grid_axis_0"] = self.diffusivityBulkGridAxes[0]
            arrays["diffusivity_bulk_grid_axis_1"] = self.diffusivityBulkGridAxes[1]
        for context in ("interface", "general"):
            for i, phase in enumerate(self.tieline_phases):
                arrays[f"diffusivity_compositions_{context}_{i}"] = self.diffusivity_compositions[context][phase]
                arrays[f"diffusivities_{context}_{i}"] = self.diffusivities[context][phase]
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        """Loads a surrogate saved by :meth:`save`."""
        path = _as_path_with_npz_suffix(path)
        with np.load(path, allow_pickle=False) as data:
            elements = tuple(str(v) for v in data["elements"].tolist())
            phases = tuple(str(v) for v in data["phases"].tolist())
            tieline_phases = tuple(str(v) for v in data["tieline_phases"].tolist())
            tieline_array = np.asarray(data["tieline_compositions"], dtype=np.float64)
            tieline_compositions = {phase: tieline_array[i] for i, phase in enumerate(tieline_phases)}
            diffusivity_compositions = {"interface": {}, "general": {}}
            diffusivities = {"interface": {}, "general": {}}
            for context in ("interface", "general"):
                for i, phase in enumerate(tieline_phases):
                    diffusivity_compositions[context][phase] = np.asarray(data[f"diffusivity_compositions_{context}_{i}"], dtype=np.float64)
                    diffusivities[context][phase] = np.asarray(data[f"diffusivities_{context}_{i}"], dtype=np.float64)
            metadata = json.loads(str(data["metadata_json"].tolist()))
            diffusivity_interpolation = (
                _DIFFUSIVITY_INTERPOLATION_NEAREST
                if "diffusivity_interpolation" not in data
                else str(data["diffusivity_interpolation"].tolist())
            )
            diffusivity_bulk_grids = None
            if "diffusivity_bulk_grid_axis_0" in data and "diffusivity_bulk_grid_axis_1" in data:
                diffusivity_bulk_grids = (
                    np.asarray(data["diffusivity_bulk_grid_axis_0"], dtype=np.float64),
                    np.asarray(data["diffusivity_bulk_grid_axis_1"], dtype=np.float64),
                )
            return cls(
                elements=elements,
                phases=phases,
                tieline_phases=tieline_phases,
                temperature=float(data["temperature"]),
                eta_samples=np.asarray(data["eta_samples"], dtype=np.float64),
                tieline_compositions=tieline_compositions,
                diffusivity_compositions=diffusivity_compositions,
                diffusivities=diffusivities,
                min_composition=float(data["min_composition"]),
                metadata=metadata,
                diffusivity_interpolation=diffusivity_interpolation,
                diffusivity_bulk_grids=diffusivity_bulk_grids,
            )
