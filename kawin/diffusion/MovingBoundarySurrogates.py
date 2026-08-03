import json
from pathlib import Path

import numpy as np
from scipy import optimize
try:
    from scipy.interpolate import PchipInterpolator, RectBivariateSpline
except ImportError:  # pragma: no cover - SciPy is a package dependency.
    PchipInterpolator = None
    RectBivariateSpline = None

from kawin.thermo import MulticomponentThermodynamics


_DIFFUSIVITY_INTERPOLATION_NEAREST = "nearest"
_DIFFUSIVITY_INTERPOLATION_CONTINUOUS_GRID = "continuous_grid"


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
    if mode not in {_DIFFUSIVITY_INTERPOLATION_NEAREST, _DIFFUSIVITY_INTERPOLATION_CONTINUOUS_GRID}:
        raise ValueError(
            "diffusivity_interpolation must be 'nearest' or 'continuous_grid'."
        )
    return mode


def _signed_cuberoot(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.cbrt(np.abs(values))


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
    solves can use cheap runtime evaluations without matrix repair.
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
    ):
        """
        Samples tie-lines and diffusivities from a thermodynamics source.

        Every tie-line sample is accepted only if equilibrium metadata contains
        exactly the two explicitly requested ``tieline_phases``. Continuous
        bulk diffusivity interpolation requires explicit ``diffusivity_bulk_grids``
        so the rectangular sampling domain and array ordering are reproducible.
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

        eta_samples = np.linspace(0.0, 1.0, 11) if eta_samples is None else np.asarray(eta_samples, dtype=np.float64).reshape(-1)
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
            bulk_points = cls._merge_bulk_points(
                elements,
                min_composition,
                diffusivity_bulk_points,
                _bulk_points_from_grids(diffusivity_bulk_grids),
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
        composition grid.
        """
        self._validate_temperature(T)
        phase = self._phase_index(phase)
        context = "interface" if query_context == "interface" else "general"
        values = np.asarray(x, dtype=np.float64)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        if values.shape[1] != 2:
            raise ValueError("getInterdiffusivity expects independent ternary compositions with shape (n, 2).")
        if self.diffusivityInterpolation == _DIFFUSIVITY_INTERPOLATION_CONTINUOUS_GRID:
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
