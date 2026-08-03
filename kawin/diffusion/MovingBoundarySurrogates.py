import json
from pathlib import Path

import numpy as np
from scipy import optimize

from kawin.thermo import MulticomponentThermodynamics


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
    independent composition. Tie-lines are linearly interpolated in ``eta``;
    diffusivities are selected by nearest sampled composition. Because that
    nearest-neighbor diffusivity field is discontinuous, it is best suited to
    phase-uniform or lagged coefficient use. Implicit composition-dependent
    moving-boundary solves require a sufficiently smooth direct thermodynamic
    or surrogate diffusivity source for reliable Picard and finite-difference
    Jacobian convergence.
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
        min_composition=1e-10,
        thermodynamics_kwargs=None,
    ):
        """
        Samples tie-lines and diffusivities from a thermodynamics source.

        Every tie-line sample is accepted only if equilibrium metadata contains
        exactly the two explicitly requested ``tieline_phases``.
        """
        if tieline_phases is None:
            raise ValueError("tieline_phases must be provided explicitly.")
        tieline_phases = _validate_tieline_phases(tieline_phases)
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
        Returns nearest sampled 2x2 interdiffusivity matrices.

        ``query_context='interface'`` uses interface endpoint samples. All other
        contexts use the general sample pool, which includes interface samples
        plus any requested bulk samples. The returned nearest-neighbor mapping
        is discontinuous in composition, so continuous surrogate interpolation is
        deferred to a future implementation before relying on this source for
        implicit composition-dependent bulk diffusivity.
        """
        self._validate_temperature(T)
        phase = self._phase_index(phase)
        context = "interface" if query_context == "interface" else "general"
        samples_x = self.diffusivity_compositions[context][phase]
        samples_d = self.diffusivities[context][phase]
        values = np.asarray(x, dtype=np.float64)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        if values.shape[1] != 2:
            raise ValueError("getInterdiffusivity expects independent ternary compositions with shape (n, 2).")
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
        }
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
            )
