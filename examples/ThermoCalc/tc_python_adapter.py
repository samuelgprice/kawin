"""TC-Python thermodynamic and kinetic adapter for kawin examples.

This module intentionally lives under ``examples`` because it depends on an
external Thermo-Calc installation, databases, and license.  The public
``TCPythonThermodynamics`` class follows the subset of kawin's multicomponent
thermodynamics API used by the moving-boundary surrogate builder.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

import numpy as np

from kawin.thermo.utils import _getMatrixPhase, _getPrecipitatePhase, _process_xT_arrays

from examples.debugInPlace import debugInPlace

GAS_CONSTANT = 8.31446261815324
_MULTISTART_SEED_ATOL = 1.0e-10

def _is_qthiss_iteration_error(exc: Exception) -> bool:
    """Match only TC's QTHISS iteration-limit failure, allowing whitespace variation."""

    pattern = r"\bERROR\s+IN\s+QTHISS\s*:\s*TOO\s+MANY\s+ITERATIONS\b"
    return re.search(pattern, str(exc), re.IGNORECASE) is not None


def _is_missing_diffusion_quantity_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return (
        "ERROR IN DCHEMD" in message
        and (("NO SUCH GRADIENT ELEMENT" in message) or ("NO SUCH DIFFUSING ELEMENT" in message))
    )

class ThermoCalcError(RuntimeError):
    """Base class for example-local TC-Python adapter failures."""


class ThermoCalcInputError(ValueError):
    """Raised when a composition, temperature, phase, or configuration is invalid."""


class ThermoCalcBackendError(ThermoCalcError):
    """Raised when TC-Python cannot be imported, started, or licensed."""


class ThermoCalcCalculationError(ThermoCalcError):
    """Raised when a TC-Python calculation fails after the input is accepted."""


class ThermoCalcSolveError(ThermoCalcCalculationError):
    """Raised specifically when ``calculate()`` fails, including exhausted retries.

    Bulk sampling may drop this error without hiding failed quantity queries.
    """

    failed_during_calculate = True


class ThermoCalcDatabaseError(ThermoCalcError):
    """Raised when a database cannot support the requested TC-Python workflow."""


@dataclass(frozen=True)
class ThermoCalcConfig:
    """Configuration for the TC-Python example adapter.

    Elements are stored in kawin order with the reference element first.
    Public composition inputs use independent mole fractions in the order of
    ``independent_elements``; for the default Fe-Cr-Ni case this is
    ``[x_CR, x_NI]`` and ``x_FE`` is computed by closure. By default,
    equilibrium-like calculations keep Thermo-Calc's default phase selection so
    additional stable phases are not hidden by the adapter. Set
    ``use_default_phases=False`` to reproduce the older behavior where only
    ``phases`` are selected in the Thermo-Calc system. The optional
    ``global_minimization_max_grid_points`` value is applied through
    TC-Python's ``SingleEquilibriumOptions`` object. The two kinetics-only
    switches leave equilibrium and driving-force minimization unchanged; their
    defaults are reasserted after a forced kinetics calculation. The
    ``equilibrium_qthiss_retry_grid_points`` also applies to driving-force and
    kinetics calculations that already use global minimization. Local kinetics
    calculations are never retried. Later queries regain the configured grid.
    ``kinetics_constrain_single_composition_set`` suspends all composition sets,
    re-enters only the requested kinetics set, and uses local minimization.
    Global minimization can create new sets despite phase-status restrictions;
    the usual bulk-composition conditions remain, and site fractions are free.
    ``kinetics_multistart_mode='global_scout'`` instead uses a global calculation
    to find candidate sublattice constitutions, minimizes each candidate locally
    as one composition set, and returns the finite candidate with lowest phase
    Gibbs energy. The scout is never used directly for diffusivity, and local
    candidate calculators are fresh for every candidate because reused TC
    calculation objects can retain a path-dependent phase-Hessian state even
    after their conditions are reset. This heuristic does not prove that every
    local basin was found.
    ``kinetics_multistart_phases`` optionally restricts that strategy to named
    phases; other phases use the ordinary kinetics settings. The optional
    ``kinetics_session_restart_interval`` recycles the TC Java session between
    kinetics queries to bound long-run native memory growth while preserving
    already materialized adapter data and diagnostic callbacks.
    """

    thermodynamic_database: str = None #"TCFE9"
    kinetic_database: str | None = None #"MOBFE4"
    user_database_path: str | Path | None = None
    elements: tuple[str, ...] = None #("FE", "CR", "NI")
    phases: tuple[str, ...] = None #("BCC_A2", "FCC_A1")
    reference_element: str = None #"FE"
    pressure: float = 101325.0
    use_default_phases: bool = True
    global_minimization_max_grid_points: int | None = None
    equilibrium_qthiss_retry_grid_points: tuple[int, ...] = ()
    kinetics_disable_global_minimization: bool = False
    kinetics_disable_positive_definite_hessian: bool = False
    kinetics_constrain_single_composition_set: bool = False
    kinetics_multistart_mode: str = "off"
    kinetics_multistart_phases: tuple[str, ...] | None = None
    kinetics_session_restart_interval: int | None = None
    cache_dir: str | Path | None = Path("examples") / "ThermoCalc" / "outputs" / "tc_cache"
    timeout_seconds: float | None = 300.0
    calculation_version: int = 1

    def __post_init__(self):
        normalized_elements = tuple(str(element).upper() for element in self.elements)
        normalized_phases = tuple(str(phase).upper() for phase in self.phases)
        normalized_reference = str(self.reference_element).upper()
        object.__setattr__(self, "elements", normalized_elements)
        object.__setattr__(self, "phases", normalized_phases)
        object.__setattr__(self, "reference_element", normalized_reference)
        if normalized_reference not in normalized_elements:
            raise ThermoCalcInputError("reference_element must be present in elements.")
        if normalized_elements[0] != normalized_reference:
            raise ThermoCalcInputError("reference_element must be the first entry in elements for kawin compatibility.")
        if len(normalized_elements) < 2:
            raise ThermoCalcInputError("At least two elements are required.")
        if len(normalized_phases) < 2:
            raise ThermoCalcInputError("At least matrix and precipitate phases are required.")
        if self.user_database_path is not None and self.kinetic_database is not None:
            raise ThermoCalcInputError(
                "Use either a commercial thermodynamic/kinetic database pair or a single user_database_path."
            )
        if self.global_minimization_max_grid_points is not None:
            value = int(self.global_minimization_max_grid_points)
            if value < 1:
                raise ThermoCalcInputError("global_minimization_max_grid_points must be a positive integer when set.")
            object.__setattr__(self, "global_minimization_max_grid_points", value)
        try:
            retry_points = tuple(self.equilibrium_qthiss_retry_grid_points)
        except TypeError as exc:
            raise ThermoCalcInputError("equilibrium_qthiss_retry_grid_points must be a sequence of positive integers.") from exc
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1 for value in retry_points):
            raise ThermoCalcInputError("equilibrium_qthiss_retry_grid_points must contain positive integers.")
        if any(later <= earlier for earlier, later in zip(retry_points, retry_points[1:])):
            raise ThermoCalcInputError("equilibrium_qthiss_retry_grid_points must increase strictly.")
        if self.global_minimization_max_grid_points is not None and retry_points:
            if retry_points[0] <= self.global_minimization_max_grid_points:
                raise ThermoCalcInputError(
                    "equilibrium_qthiss_retry_grid_points must exceed global_minimization_max_grid_points."
                )
        object.__setattr__(self, "equilibrium_qthiss_retry_grid_points", tuple(int(value) for value in retry_points))
        multistart_mode = str(self.kinetics_multistart_mode).strip().lower()
        if multistart_mode not in {"off", "global_scout"}:
            raise ThermoCalcInputError("kinetics_multistart_mode must be 'off' or 'global_scout'.")
        if multistart_mode == "global_scout" and self.kinetics_disable_positive_definite_hessian:
            raise ThermoCalcInputError(
                "kinetics_multistart_mode='global_scout' requires the positive-definite phase Hessian."
            )
        object.__setattr__(self, "kinetics_multistart_mode", multistart_mode)
        if self.kinetics_multistart_phases is not None:
            try:
                multistart_phases = tuple(str(phase).upper() for phase in self.kinetics_multistart_phases)
            except TypeError as exc:
                raise ThermoCalcInputError("kinetics_multistart_phases must be a sequence of phase names.") from exc
            unknown = [
                phase for phase in multistart_phases
                if phase not in normalized_phases
                and base_phase_name(phase) not in {base_phase_name(item) for item in normalized_phases}
            ]
            if unknown:
                raise ThermoCalcInputError(
                    f"kinetics_multistart_phases contains phases outside configured phases: {unknown}."
                )
            object.__setattr__(self, "kinetics_multistart_phases", multistart_phases)
        if self.kinetics_session_restart_interval is not None:
            interval = self.kinetics_session_restart_interval
            if not isinstance(interval, (int, np.integer)) or isinstance(interval, (bool, np.bool_)) or interval < 1:
                raise ThermoCalcInputError(
                    "kinetics_session_restart_interval must be a positive integer when set."
                )
            object.__setattr__(self, "kinetics_session_restart_interval", int(interval))

    @property
    def independent_elements(self) -> tuple[str, ...]:
        """Elements represented by independent composition inputs."""

        return tuple(element for element in self.elements if element != self.reference_element)

    @property
    def matrix_phase(self) -> str:
        """Default matrix phase."""

        return self.phases[0]

    @property
    def precipitate_phase(self) -> str:
        """Default precipitate phase."""

        return self.phases[1]

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for dataset manifests."""

        metadata = asdict(self)
        metadata["cache_dir"] = None if self.cache_dir is None else str(self.cache_dir)
        metadata["user_database_path"] = None if self.user_database_path is None else str(self.user_database_path)
        return metadata


def tc_element_name(element: str) -> str:
    """Return the mixed-case element spelling expected by TC-Python quantities."""

    element = str(element).strip()
    return element[:1].upper() + element[1:].lower()


def base_phase_name(phase: str) -> str:
    """Strip TC-Python composition-set suffixes such as ``#1`` from phase names."""

    return str(phase).split("#", maxsplit=1)[0].upper()


def _site_fraction_quantity(phase: str, constituent: str, sublattice: int) -> str:
    """Return the Console Mode site-fraction quantity for one sublattice.

    TC-Python's single-equilibrium result accepts quantity strings. Its
    ``ThermodynamicQuantity`` factory does not provide site fractions, so use
    the same ``Y(phase,constituent#ordinal)`` syntax as ``ScheilQuantity``.
    """

    return f"Y({phase},{constituent}#{sublattice})"


def normalized_driving_force_to_j_per_mol(dgm: float, T: float) -> float:
    """Convert TC-Python's dimensionless ``DGM`` quantity to J/mol."""

    return float(dgm) * GAS_CONSTANT * float(T)


def validate_independent_composition(x: Any, config: ThermoCalcConfig) -> np.ndarray:
    """Validate and return one independent-composition vector.

    The returned vector has length ``len(config.elements) - 1``.  All mole
    fractions must be finite and non-negative, and the computed reference
    element fraction must be non-negative.
    """

    composition = np.asarray(x, dtype=np.float64)
    if composition.ndim != 1 or composition.size != len(config.independent_elements):
        raise ThermoCalcInputError(
            "Composition must be a 1-D independent mole-fraction vector in "
            f"{list(config.independent_elements)} order."
        )
    if not np.all(np.isfinite(composition)):
        raise ThermoCalcInputError("Composition contains non-finite values.")
    if np.any(composition < 0):
        raise ThermoCalcInputError("Composition contains negative mole fractions.")
    reference_fraction = 1.0 - float(np.sum(composition))
    if reference_fraction < -1e-12:
        raise ThermoCalcInputError("Independent mole fractions sum to more than one.")
    return composition


def independent_to_full_composition(x: Any, config: ThermoCalcConfig) -> np.ndarray:
    """Convert independent mole fractions to full ``config.elements`` order."""

    independent = validate_independent_composition(x, config)
    reference_fraction = max(0.0, 1.0 - float(np.sum(independent)))
    return np.concatenate(([reference_fraction], independent))


def full_to_independent_composition(x: Any, config: ThermoCalcConfig) -> np.ndarray:
    """Convert a full composition in ``config.elements`` order to independent form."""

    composition = np.asarray(x, dtype=np.float64)
    if composition.shape != (len(config.elements),):
        raise ThermoCalcInputError(f"Full composition must have shape ({len(config.elements)},).")
    return composition[1:]


class _TCPythonBackend:
    """Thin wrapper around the installed TC-Python API.

    The backend keeps one TC-Python server session and a small set of reusable
    single-equilibrium calculations. Minimization settings are reasserted when
    switching between kinetics and unrestricted calculations because the shared
    engine can retain them across calculation objects. TC-Python itself does not
    expose a Python timeout per calculation here; ``timeout_seconds`` is recorded
    in metadata so callers can enforce process-level limits around long runs.
    """

    def __init__(self):
        self._tc_python = None
        self._session = None
        self._setup = None
        self._system = None
        self._systems: dict[bool, Any] = {}
        self._config: ThermoCalcConfig | None = None
        self._calculations: dict[tuple[Any, ...], Any] = {}
        self._last_calculation_kind: str | None = None
        self._retry_calculation: Any | None = None
        self._retry_settings_dirty = False
        self._site_fraction_capture_callback = None
        self._site_fraction_capture_index = 0
        self.totalNumCalcs = 0
        self.totalNumCaches = 0
        self.totalNumQueries = 0
        self.total_kind_lst = []
        self.total_x_lst = []
        self._kinetics_queries_since_restart = 0
        self._automatic_session_restart_count = 0

    def start(self, config: ThermoCalcConfig, *, reset_counters: bool = True):
        """Start TC-Python and build the selected system.

        ``reset_counters=False`` is reserved for automatic memory-management
        restarts, which must preserve run-level progress and diagnostics.
        """

        if self._session is not None:
            return
        try:
            import tc_python
            from tc_python import TCPython
        except ImportError as exc:
            raise ThermoCalcBackendError(
                "TC-Python is not importable. Install Thermo-Calc with TC-Python support "
                "and run this example in an environment where tc_python is available."
            ) from exc

        self._tc_python = tc_python
        self._config = config
        if reset_counters:
            self.totalNumCalcs=0
            self.totalNumCaches=0
            self.totalNumQueries=0
            self.total_kind_lst=[]
            self.total_x_lst=[]
            self._automatic_session_restart_count = 0
        self._kinetics_queries_since_restart = 0
        self._last_calculation_kind = None
        self._retry_calculation = None
        self._retry_settings_dirty = False
        try:
            self._session = TCPython()
            self._setup = self._session.__enter__()
            if config.cache_dir is not None:
                cache_dir = Path(config.cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                self._setup.set_cache_folder(str(cache_dir))
            self._system = self._get_system(config.use_default_phases)
        except Exception as exc:
            self.close(report_totals=False)
            self._raise_backend_error(exc)

    def close(self, *, report_totals: bool = True):
        """Close the TC-Python session, optionally reporting cumulative counters."""

        if self._session is not None:
            if report_totals:
                print("\n")
                print(f"Total number of calcs: {self.totalNumCalcs}")
                print(f"Total number of caches: {self.totalNumCaches}")
                print(f"Total number of queries: {self.totalNumQueries}")
            try:
                self._session.__exit__(None, None, None)
            finally:
                self._session = None
                self._setup = None
                self._system = None
                self._systems = {}
                self._calculations = {}
                self._last_calculation_kind = None
                self._retry_calculation = None
                self._retry_settings_dirty = False

    def restart(self):
        """Restart the TC-Python session using the current configuration."""

        if self._config is None:
            return
        config = self._config
        self.close()
        self.start(config)

    def _maybe_restart_before_kinetics(self):
        """Recycle the TC session at a safe query boundary to release Java memory."""
        config = self._require_config()
        interval = config.kinetics_session_restart_interval
        if interval is None or self._kinetics_queries_since_restart < interval:
            return
        self.close(report_totals=False)
        self.start(config, reset_counters=False)
        self._automatic_session_restart_count += 1

    def get_runtime_version(self) -> str | None:
        """Return the installed TC-Python package version when available."""

        if self._tc_python is None:
            try:
                import tc_python
            except ImportError:
                return None
            return getattr(tc_python, "__version__", None)
        return getattr(self._tc_python, "__version__", None)

    def preflight(self, config: ThermoCalcConfig, x: np.ndarray, T: float) -> dict[str, Any]:
        """Run a one-point compatibility check for the configured databases."""

        self.start(config)
        try:
            equilibrium = self.calculate_equilibrium(x, T)
            driving_force = self.calculate_driving_force(x, T, config.precipitate_phase)
            kinetics = self.calculate_kinetics(x, T, config.matrix_phase)
        except Exception as exc:
            self._raise_database_error(exc)
        return {
            "ok": True,
            "stable_phases": equilibrium["stable_phases"],
            "driving_force": driving_force["driving_force"],
            "interdiffusivity_shape": tuple(np.shape(kinetics["interdiffusivity"])),
            "tracer_diffusivity_shape": tuple(np.shape(kinetics["tracer_diffusivity"])),
        }

    def calculate_equilibrium(self, x: np.ndarray, T: float) -> dict[str, Any]:
        """Calculate equilibrium phases, phase amounts, compositions, and chemical potentials.

        Stable phases are reported by base phase name. When Thermo-Calc creates
        multiple composition sets such as ``BCC_B2#2``, amounts are summed by
        base phase and the representative composition is taken from the largest
        composition set for that base phase. Diffusivities are intentionally
        excluded because the surrogate obtains them from forced single-phase
        kinetics calculations rather than an unrestricted equilibrium result.
        """

        config = self._require_config()
        result = self._calculate("equilibrium", None, x, T)
        raw_stable_phases = [str(phase).upper() for phase in result.get_stable_phases()]
        stable_phases = [phase for phase in raw_stable_phases] # currently stable_phases is the same as raw_stable_phases but still using stable_phases as this may change in the future
        if len(set(stable_phases))!=len(stable_phases):
            raise ValueError("More than one phase with the same base phase name")
        stable_phase_names = list(dict.fromkeys(stable_phases))
        phase_amounts = {phase: 0.0 for phase in config.phases}
        phase_compositions = {}
        representative_amounts = {}
        for raw_phase, base_phase in zip(raw_stable_phases, stable_phases):
            amount = self._safe_value(result, self._tq().mole_fraction_of_a_phase(raw_phase), default=0.0)
            phase_amounts[base_phase] = phase_amounts.get(base_phase, 0.0) + amount
            if amount >= representative_amounts.get(base_phase, -np.inf):
                representative_amounts[base_phase] = amount
                phase_compositions[base_phase] = self._phase_composition(result, raw_phase)
        reported_phases = tuple(dict.fromkeys((*config.phases, *stable_phases)))

        return {
            "stable_phases": stable_phase_names,
            "phase_amounts": {
                phase: phase_amounts.get(phase, 0.0)
                for phase in reported_phases
            },
            "phase_compositions": {
                phase: phase_compositions[phase]
                for phase in reported_phases
                if phase in phase_compositions
            },
            "chemical_potentials": {
                element: self._value(result, self._tq().chemical_potential_of_component(tc_element_name(element)))
                for element in config.elements
            },
        }

    def calculate_driving_force(self, x: np.ndarray, T: float, precipitate_phase: str) -> dict[str, Any]:
        """Calculate TC-Python DGM and dormant precipitate composition."""

        config = self._require_config()
        precipitate_phase = precipitate_phase.upper()
        result = self._calculate("driving_force", precipitate_phase, x, T)
        dgm = self._value(result, self._tq().normalized_driving_force_of_a_phase(precipitate_phase))
        return {
            "driving_force": normalized_driving_force_to_j_per_mol(dgm, T),
            "normalized_driving_force": dgm,
            "precipitate_composition": self._phase_composition(result, precipitate_phase),
            "precipitate_phase": precipitate_phase,
            "units": "J/mol",
            "elements": config.elements,
        }

    def calculate_kinetics(self, x: np.ndarray, T: float, phase: str, collect_diagnostics: bool = False) -> dict[str, Any]:
        """Calculate single-phase diffusivities and optionally inspect that result.

        During capture, failed tracer or phase-state queries are recorded as
        optional diagnostic errors without discarding a chemical matrix. In
        global-scout multistart mode, every scout set is locally minimized and
        the lowest-Gibbs-energy successful single-set candidate is returned.
        """

        self._maybe_restart_before_kinetics()
        phase = phase.upper()
        try:
            if self._uses_kinetics_multistart(phase):
                return self._calculate_kinetics_multistart(x, T, phase, collect_diagnostics)
            result = self._calculate("kinetics", phase, x, T)
            try:
                output = self._kinetics_output_from_result(result, phase, collect_diagnostics)
            finally:
                self._invalidate_result(result)
            if collect_diagnostics:
                output["diagnostics"]["kinetics_strategy"] = "single_start"
                output["diagnostics"]["automatic_session_restart_count"] = (
                    self._automatic_session_restart_count
                )
            return output
        finally:
            self._kinetics_queries_since_restart += 1

    def _uses_kinetics_multistart(self, phase: str) -> bool:
        """Return whether global-scout multistart applies to this phase."""
        config = self._require_config()
        if config.kinetics_multistart_mode != "global_scout":
            return False
        selected = config.kinetics_multistart_phases
        if selected is None:
            return True
        return phase in selected or base_phase_name(phase) in {
            base_phase_name(candidate) for candidate in selected
        }

    def _kinetics_output_from_result(
        self, result: Any, phase: str, collect_diagnostics: bool
    ) -> dict[str, Any]:
        """Materialize kinetics quantities while a TC temporary result is valid.

        TC-Python invalidates a temporary result when its calculation state is
        changed. Multistart therefore converts every successful candidate to
        ordinary Python and NumPy data before starting the next candidate.
        """

        config = self._require_config()
        tq = self._tq()
        independent = config.independent_elements
        reference = tc_element_name(config.reference_element)
        interdiffusivity = np.array(
            [
                [
                    self._value(
                        result,
                        tq.chemical_diffusion_coefficient(
                            phase,
                            tc_element_name(diffusing),
                            tc_element_name(gradient),
                            reference,
                        ),
                    )
                    for gradient in independent
                ]
                for diffusing in independent
            ],
            dtype=np.float64,
        )
        tracer_errors = {}
        if collect_diagnostics:
            tracer = []
            for element in config.elements:
                try:
                    value = float(result.get_value_of(
                        tq.tracer_diffusion_coefficient(phase, tc_element_name(element))
                    ))
                    if not np.isfinite(value):
                        raise ValueError("non-finite tracer diffusivity")
                    tracer.append(value)
                except Exception as exc:
                    tracer.append(None)
                    tracer_errors[f"tracer_diffusivity[{element}]"] = str(exc)
        else:
            tracer = np.array(
                [
                    self._value(result, tq.tracer_diffusion_coefficient(phase, tc_element_name(element)))
                    for element in config.elements
                ],
                dtype=np.float64,
            )
        output = {
            "interdiffusivity": interdiffusivity,
            "tracer_diffusivity": tracer,
            "phase": phase,
            "independent_elements": independent,
            "elements": config.elements,
            "units": "m^2/s",
        }
        if collect_diagnostics:
            output["diagnostics"] = self._kinetics_diagnostics(result, phase)
            output["diagnostics"]["errors"].update(tracer_errors)
        return output

    def _calculate_kinetics_multistart(
        self, x: np.ndarray, T: float, phase: str, collect_diagnostics: bool = False
    ):
        """Select materialized data from the lowest-energy local candidate.

        The scout may contain several composition sets of the requested base
        phase. Their site fractions are used only as local start values; every
        accepted candidate must retain the requested set at the requested
        overall composition. Candidate failures are isolated unless none
        survives. When every failure occurred inside ``calculate()``, the
        aggregate remains a :class:`ThermoCalcSolveError` so bulk samplers can
        honor their configured failed-point policy. Candidate quantities are
        materialized immediately because later solves invalidate TC temporary
        result objects.
        """

        # Mark the shared engine as kinetics-owned before any attempt so a
        # later unrestricted calculation restores global/Hessian settings even
        # when the scout or every candidate fails.
        self._last_calculation_kind = "kinetics"
        scout = self._run_multistart_scout(x, T, phase)
        try:
            seeds, ledger = self._extract_multistart_seeds(scout, phase)
        finally:
            self._invalidate_result(scout)
        if not seeds:
            details = "; ".join(entry["error"] for entry in ledger if entry.get("error"))
            suffix = f" ({details})" if details else ""
            raise ThermoCalcCalculationError(
                f"Global kinetics scout found no valid {base_phase_name(phase)} site-fraction seeds{suffix}."
            )

        successful = []
        candidate_errors = []
        for seed in seeds:
            entry = seed["ledger_entry"]
            result = None
            try:
                result, gibbs_energy = self._run_multistart_candidate(x, T, phase, seed)
                output = self._kinetics_output_from_result(result, phase, collect_diagnostics)
                entry.update(status="converged", gibbs_energy=float(gibbs_energy), error=None)
                successful.append((float(gibbs_energy), seed["seed_index"], output, entry))
            except Exception as exc:
                entry.update(status="failed", gibbs_energy=None, error=str(exc))
                candidate_errors.append(exc)
            finally:
                self._invalidate_result(result)

        if not successful:
            summary = "; ".join(
                f"seed {entry['seed_index']} ({entry['source_composition_set']}): {entry.get('error')}"
                for entry in ledger if entry.get("status") == "failed"
            )
            error_type = ThermoCalcSolveError if all(
                getattr(error, "failed_during_calculate", False) for error in candidate_errors
            ) else ThermoCalcCalculationError
            raise error_type(f"All multistart kinetics candidates failed for {phase}. {summary}")

        gibbs_energy, selected_index, selected_output, selected_entry = min(
            successful, key=lambda candidate: (candidate[0], candidate[1])
        )
        selected_entry["selected"] = True
        for entry in ledger:
            entry.setdefault("selected", False)
        multistart_diagnostics = {
            "kinetics_strategy": "multistart_global_scout",
            "scout_seed_count": len(ledger),
            "unique_seed_count": len(seeds),
            "converged_candidate_count": len(successful),
            "selected_seed_index": selected_index,
            "selected_source_composition_set": selected_entry["source_composition_set"],
            "selected_gibbs_energy": gibbs_energy,
            "selected_gibbs_energy_units": "J",
            "multistart_candidates": ledger,
            "automatic_session_restart_count": self._automatic_session_restart_count,
        }
        if collect_diagnostics:
            selected_output["diagnostics"].update(multistart_diagnostics)
        return selected_output

    def _extract_multistart_seeds(self, result: Any, phase: str):
        """Extract complete, finite, distinct site-fraction seeds from a scout.

        Seed identity uses every fraction. Initialization uses exactly the
        system's composition degrees of freedom as ordering markers, selected
        round-robin from nontrivial sublattices. This distinguishes ordered
        basins without overconstraining the temporary seed equilibrium.
        """

        try:
            stable = [str(name).upper() for name in result.get_stable_phases()]
        except Exception as exc:
            raise ThermoCalcCalculationError(f"Could not read global-scout composition sets: {exc}") from exc
        matching = [name for name in stable if base_phase_name(name) == base_phase_name(phase)]
        seeds = []
        ledger = []
        signatures = []
        for seed_index, source_phase in enumerate(matching):
            entry = {
                "seed_index": seed_index,
                "source_composition_set": source_phase,
                "status": "seeded",
                "gibbs_energy": None,
                "selected": False,
                "error": None,
            }
            ledger.append(entry)
            try:
                state = self._site_fraction_phase_state(result, source_phase, True, "kinetics_scout")
                site_fractions = state.get("site_fractions")
                if not site_fractions:
                    raise ValueError("site fractions are unavailable")
                phase_composition = np.asarray(state.get("phase_composition"), dtype=np.float64)
                if phase_composition.shape != (len(self._require_config().elements),):
                    raise ValueError("phase composition is unavailable")
                if not np.all(np.isfinite(phase_composition)):
                    raise ValueError("phase composition contains non-finite values")
                values = []
                labels = []
                command_candidates = []
                for sublattice in site_fractions:
                    index = int(sublattice["sublattice"])
                    constituents = sublattice.get("constituents")
                    if not constituents:
                        raise ValueError(f"sublattice {index} constituents are unavailable")
                    sublattice_values = []
                    for constituent, value in sorted(constituents.items()):
                        value = float(value)
                        if not np.isfinite(value):
                            raise ValueError(f"non-finite Y({source_phase},{constituent}#{index})")
                        labels.append((index, str(constituent).upper()))
                        values.append(value)
                        sublattice_values.append((str(constituent), value))
                    if len(sublattice_values) > 1:
                        command_candidates.append([
                            (index, constituent, value)
                            for constituent, value in sorted(
                                sublattice_values,
                                key=lambda item: (item[0].upper() == "VA", -item[1], item[0]),
                            )
                        ])
                commands = []
                required_markers = len(self._require_config().independent_elements)
                rank = 0
                while len(commands) < required_markers:
                    added = False
                    for candidates in command_candidates:
                        if rank < len(candidates):
                            commands.append(candidates[rank])
                            added = True
                            if len(commands) == required_markers:
                                break
                    if not added:
                        break
                    rank += 1
                if len(commands) != required_markers:
                    raise ValueError(
                        f"only {len(commands)} independent ordering markers are available; "
                        f"expected {required_markers}"
                    )
                signature = np.asarray(values, dtype=np.float64)
                duplicate = any(
                    labels == prior_labels
                    and np.allclose(signature, prior_values, rtol=0.0, atol=_MULTISTART_SEED_ATOL)
                    for prior_labels, prior_values in signatures
                )
                if duplicate:
                    entry["status"] = "duplicate_seed"
                    continue
                signatures.append((labels, signature))
                seeds.append({
                    "seed_index": seed_index,
                    "source_composition_set": source_phase,
                    "phase_composition": phase_composition,
                    "site_fractions": commands,
                    "ledger_entry": entry,
                })
            except Exception as exc:
                entry.update(status="invalid_seed", error=str(exc))
        return seeds, ledger

    def _run_multistart_scout(self, x: np.ndarray, T: float, phase: str):
        """Run a reusable global phase-restricted scout with QTHISS retries.

        Calculator reuse follows TC-Python's loop guidance and prevents one
        Java-side calculation object from accumulating per sampled point.
        A calculator that raises is evicted before a later query reuses it.
        """

        config = self._require_config()
        attempts = (config.global_minimization_max_grid_points,) + config.equilibrium_qthiss_retry_grid_points
        last_error = None
        for attempt_index, grid_points in enumerate(attempts):
            key = ("kinetics_scout", phase, False, grid_points)
            calc = self._calculations.get(key)
            if calc is None:
                calc = self._get_system(False).with_single_equilibrium_calculation()
            calc = self._configure_global_minimization(
                calc,
                config,
                "kinetics_scout",
                restore_shared_settings=True,
                grid_points_override=grid_points,
            )
            self._calculations[key] = calc
            calc.set_phase_to_suspended("*")
            calc.set_phase_to_entered(phase)
            calc.remove_all_conditions()
            self._set_conditions(calc, x, T)
            self._record_calculation_attempt("kinetics_scout", x)
            try:
                result = calc.calculate()
            except Exception as exc:
                self._emit_site_fraction_calculation(
                    "kinetics_scout", phase, x, T, error=exc, grid_points=grid_points
                )
                self._calculations.pop(key, None)
                last_error = exc
                if not _is_qthiss_iteration_error(exc) or attempt_index == len(attempts) - 1:
                    break
            else:
                self._emit_site_fraction_calculation(
                    "kinetics_scout", phase, x, T, result=result, grid_points=grid_points
                )
                return result
        raise ThermoCalcSolveError(f"TC-Python kinetics scout failed: {last_error}") from last_error

    def _run_multistart_candidate(self, x: np.ndarray, T: float, phase: str, seed: dict[str, Any]):
        """Run one local single-set candidate from a scout constitution.

        A temporarily constrained local solve installs ordering markers from
        the scout constitution. The final local solve removes those constraints
        and moves that state to the requested composition. Calculators are
        always fresh per query to prevent composition-path history from
        changing the derivative/Hessian state used for interdiffusivity even
        when the converged site fractions are nearly identical.
        """

        config = self._require_config()
        calc = self._get_system(False).with_single_equilibrium_calculation()
        calc = self._configure_global_minimization(
            calc, config, "kinetics_candidate", restore_shared_settings=True
        )
        self._force_kinetics_phase(calc, phase, force_single=True)
        seed_x = full_to_independent_composition(seed["phase_composition"], config)
        calc.remove_all_conditions()
        self._set_intensive_conditions(calc, T)
        self._apply_multistart_seed_conditions(calc, phase, seed["site_fractions"])
        self._record_calculation_attempt("kinetics_candidate_seed", seed_x)
        try:
            seed_result = calc.calculate()
        except Exception as exc:
            self._emit_site_fraction_calculation(
                "kinetics_candidate_seed", phase, seed_x, T, error=exc, grid_points=None,
                candidate_index=seed["seed_index"], source_phase=seed["source_composition_set"],
            )
            raise ThermoCalcSolveError(
                f"TC-Python kinetics candidate {seed['seed_index']} seed initialization failed: {exc}"
            ) from exc
        try:
            self._emit_site_fraction_calculation(
                "kinetics_candidate_seed", phase, seed_x, T, result=seed_result, grid_points=None,
                candidate_index=seed["seed_index"], source_phase=seed["source_composition_set"],
            )
            self._validate_single_kinetics_phase(
                seed_result, phase, seed_x, force_single=True, check_composition=False
            )
        except Exception:
            raise
        finally:
            self._invalidate_result(seed_result)
        calc.remove_all_conditions()
        self._set_conditions(calc, x, T)
        self._record_calculation_attempt("kinetics_candidate", x)
        try:
            result = calc.calculate()
        except Exception as exc:
            self._emit_site_fraction_calculation(
                "kinetics_candidate", phase, x, T, error=exc, grid_points=None,
                candidate_index=seed["seed_index"], source_phase=seed["source_composition_set"],
            )
            raise ThermoCalcSolveError(
                f"TC-Python kinetics candidate {seed['seed_index']} failed: {exc}"
            ) from exc
        try:
            self._emit_site_fraction_calculation(
                "kinetics_candidate", phase, x, T, result=result, grid_points=None,
                candidate_index=seed["seed_index"], source_phase=seed["source_composition_set"],
            )
            self._validate_single_kinetics_phase(result, phase, x, force_single=True)
            gibbs_energy = self._value(result, self._tq().gibbs_energy_of_a_phase(phase))
            if not np.isfinite(gibbs_energy):
                raise ThermoCalcCalculationError(
                    f"Kinetics candidate {seed['seed_index']} returned non-finite Gibbs energy."
                )
        except Exception:
            raise
        return result, float(gibbs_energy)

    @staticmethod
    def _apply_multistart_seed_conditions(calc: Any, phase: str, site_fractions):
        """Temporarily constrain independent site fractions to initialize a basin.

        The caller removes all conditions before the final local minimization,
        so no site fraction remains fixed in the diffusivity-producing result.
        """

        for sublattice, constituent, value in site_fractions:
            quantity = _site_fraction_quantity(phase, constituent, sublattice)
            calc.set_condition(quantity, float(value))

    def _record_calculation_attempt(self, kind: str, x: np.ndarray):
        """Update calculation counters for an actual TC-Python calculate call."""

        self.totalNumCalcs += 1
        self.total_kind_lst.append(kind)
        self.total_x_lst.append(x.copy())

    @staticmethod
    def _invalidate_result(result: Any):
        """Release TC temporary result data after all required values are copied.

        TC-Python keeps result data in the Java backend until ``invalidate`` is
        called or another state change invalidates it implicitly. Explicit
        invalidation is essential for long multistart sampling loops. Cleanup
        errors are intentionally ignored because all useful values have already
        been materialized and must not be discarded due to cleanup failure.
        """
        if result is None:
            return
        invalidate = getattr(result, "invalidate", None)
        if callable(invalidate):
            try:
                invalidate()
            except Exception:
                pass

    def _kinetics_diagnostics(self, result: Any, phase: str) -> dict[str, Any]:
        """Read optional kinetic and phase-state quantities from one result.

        All values come from the forced single-phase kinetics equilibrium. A
        failed factor, mobility, composition, or site-fraction quantity is
        represented by ``None`` and named in ``errors`` without invalidating a
        usable interdiffusivity matrix.
        """
        config = self._require_config()
        tq = self._tq()
        dq = self._dq()
        errors = {}

        def optional(name, query):
            try:
                value = query()
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    raise ValueError(f"{name} is non-finite")
                return value
            except Exception as exc:
                errors[name] = str(exc)
                return None

        stable_phases = optional("stable_phases", lambda: [str(name).upper() for name in result.get_stable_phases()])
        factor = [
            [
                optional(
                    f"thermodynamic_factor[{diffusing},{gradient}]",
                    lambda d=diffusing, g=gradient: float(result.get_value_of(
                        dq.thermodynamic_factor(
                            phase, tc_element_name(d), tc_element_name(g), tc_element_name(config.reference_element)
                        )
                    )),
                )
                for gradient in config.independent_elements
            ]
            for diffusing in config.independent_elements
        ]
        mobilities = [
            optional(
                f"mobility[{element}]",
                lambda e=element: float(result.get_value_of(
                    dq.mobility_of_component_in_phase(phase, tc_element_name(e))
                )),
            )
            for element in config.elements
        ]
        composition = [
            optional(
                f"phase_composition[{element}]",
                lambda e=element: float(result.get_value_of(
                    tq.composition_of_phase_as_mole_fraction(phase, tc_element_name(e))
                )),
            )
            for element in config.elements
        ]
        site_fractions = optional(
            "site_fractions.phase",
            lambda: self._get_system(False).get_phase_object(phase).get_sublattices(),
        )
        if site_fractions is not None:
            sublattices = site_fractions
            site_fractions = []
            for index, sublattice in enumerate(sublattices, start=1):
                names = optional(
                    f"site_fractions[{index}].constituents",
                    lambda s=sublattice: sorted(species.get_name() for species in s.get_constituents()),
                )
                constituents = None if names is None else {
                    name: optional(
                        f"site_fractions[{index},{name}]",
                        lambda n=name, i=index: float(result.get_value_of(
                            _site_fraction_quantity(phase, n, i)
                        )),
                    )
                    for name in names
                }
                site_fractions.append({"sublattice": index, "constituents": constituents})
        return {
            "thermodynamic_factors": factor,
            "mobilities": mobilities,
            "stable_composition_sets": stable_phases,
            "phase_composition": composition,
            "site_fractions": site_fractions,
            "errors": errors,
        }

    def _build_system(self, config: ThermoCalcConfig, include_default_phases: bool):
        """Build a TC-Python system with either default or phase-restricted selection."""
        elements = [tc_element_name(element) for element in config.elements]
        if config.user_database_path is None and config.kinetic_database is None:
            builder = self._setup.select_database_and_elements(config.thermodynamic_database, elements)
        elif config.user_database_path is None:
            builder = self._setup.select_thermodynamic_and_kinetic_databases_with_elements(
                config.thermodynamic_database,
                config.kinetic_database,
                elements,
            )
        else:
            builder = self._setup.select_user_database_and_elements(str(config.user_database_path), elements)

        if not include_default_phases:
            builder = builder.without_default_phases()
            for phase in config.phases:
                builder = builder.select_phase(phase)
        return builder.get_system()

    def _get_calculation(self, kind: str, phase: str | None):
        config = self._require_config()
        include_default_phases = config.use_default_phases if not kind.startswith("kinetics") else False
        key = (kind, phase, include_default_phases)
        if key in self._calculations:
            return self._calculations[key]

        system = self._get_system(include_default_phases)
        calc = system.with_single_equilibrium_calculation()
        calc = self._configure_global_minimization(calc, config, kind)
        if kind == "driving_force":
            calc.set_phase_to_dormant(phase)
        elif kind == "kinetics":
            self._force_kinetics_phase(calc, phase)
        self._calculations[key] = calc
        return calc

    def _force_kinetics_phase(self, calc: Any, phase: str, *, force_single: bool = False):
        """Select one kinetics composition set while leaving its sites free.

        In constrained mode, suspend all selected and database-supplied sets
        before re-entering the requested one. Local minimization is separately
        enforced so Thermo-Calc does not create a new set during the solve.
        """
        constrain_single = force_single or self._require_config().kinetics_constrain_single_composition_set
        if constrain_single:
            calc.set_phase_to_suspended("*")
            calc.set_phase_to_entered(phase, 0.0)
            return
        for candidate in self._require_config().phases:
            if candidate == phase:
                calc.set_phase_to_entered(candidate)
            else:
                calc.set_phase_to_suspended(candidate)

    def _validate_single_kinetics_phase(
        self, result: Any, phase: str, x: np.ndarray, *, force_single: bool = False,
        check_composition: bool = True,
    ):
        """Reject a constrained result if TC split, relabeled, or drifted.

        A local solve with other sets suspended should retain just the entered
        set. Check the actual result because shared TC state may still change.
        Temporary multistart seed equilibria can omit the composition check;
        their ordering markers select a basin rather than reproduce the scout
        composition exactly. Final kinetics results always check it.
        """
        if not (force_single or self._require_config().kinetics_constrain_single_composition_set):
            return
        stable = [str(name).upper() for name in result.get_stable_phases()]
        if stable != [phase]:
            raise ThermoCalcCalculationError(
                f"Constrained kinetics for {phase} returned stable composition sets {stable}; expected only {phase}."
            )
        if not check_composition:
            return
        actual = self._phase_composition(result, phase)
        expected = independent_to_full_composition(x, self._require_config())
        if not np.allclose(actual, expected, rtol=0, atol=1e-6):
            raise ThermoCalcCalculationError(
                f"Constrained kinetics for {phase} returned composition {actual.tolist()}; "
                f"expected {expected.tolist()}."
            )

    def _emit_site_fraction_calculation(
        self, kind, phase, x, T, *, result=None, error=None, grid_points=None,
        candidate_index=None, source_phase=None,
    ):
        """Record one actual calculate attempt without changing its result.

        Every stable composition set is queried from that same result. Missing
        optional quantities are recorded per phase; failed calculate attempts
        retain their error and have no phase-state values.
        """
        callback = self._site_fraction_capture_callback
        if callback is None:
            return
        config = self._require_config()
        record = {
            "record_type": "calculation_site_fractions",
            "calculation_index": self._site_fraction_capture_index,
            "kind": kind,
            "requested_phase": phase,
            "temperature": float(T),
            "input_composition": np.asarray(x, dtype=np.float64).tolist(),
            "input_full_composition": independent_to_full_composition(x, config).tolist(),
            "grid_points": grid_points,
            "candidate_index": candidate_index,
            "source_composition_set": source_phase,
            "status": "calculate_error" if error is not None else "ok",
            "error": None if error is None else str(error),
            "stable_composition_sets": None,
            "phases": [],
            "diagnostic_errors": {},
        }
        if result is not None:
            try:
                stable = [str(name).upper() for name in result.get_stable_phases()]
                record["stable_composition_sets"] = stable
            except Exception as exc:
                stable = []
                record["diagnostic_errors"]["stable_composition_sets"] = str(exc)
            names = list(dict.fromkeys(stable + ([phase] if phase is not None else [])))
            for name in names:
                try:
                    record["phases"].append(self._site_fraction_phase_state(result, name, name in stable, kind))
                except Exception as exc:
                    record["diagnostic_errors"][f"phase[{name}]"] = str(exc)
        callback(record)
        self._site_fraction_capture_index += 1

    def _site_fraction_phase_state(self, result, phase, stable, kind):
        """Read one composition set's composition and labeled sublattice sites."""
        config = self._require_config()
        tq = self._tq()
        errors = {}

        def optional(label, query):
            try:
                value = query()
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    raise ValueError("non-finite value")
                return value
            except Exception as exc:
                errors[label] = str(exc)
                return None

        composition = [
            optional(f"composition[{element}]", lambda e=element: float(result.get_value_of(
                tq.composition_of_phase_as_mole_fraction(phase, tc_element_name(e))
            )))
            for element in config.elements
        ]
        amount = optional("phase_amount", lambda: float(result.get_value_of(tq.mole_fraction_of_a_phase(phase)))) if stable else None
        include_default_phases = config.use_default_phases if not kind.startswith("kinetics") else False
        system = self._get_system(include_default_phases)
        phase_object = optional("phase_object", lambda: system.get_phase_object(phase))
        if phase_object is None and base_phase_name(phase) != phase:
            phase_object = optional("base_phase_object", lambda: system.get_phase_object(base_phase_name(phase)))
            if phase_object is not None:
                errors.pop("phase_object", None)
        sublattices = None if phase_object is None else optional("sublattices", phase_object.get_sublattices)
        site_fractions = None if sublattices is None else []
        if sublattices is not None:
            for index, sublattice in enumerate(sublattices, start=1):
                ratio = optional(f"sublattice[{index}].site_ratio", lambda s=sublattice: float(s.get_nr_of_sites()))
                names = optional(
                    f"sublattice[{index}].constituents",
                    lambda s=sublattice: sorted(str(species.get_name()) for species in s.get_constituents()),
                )
                constituents = None if names is None else {
                    name: optional(
                        f"sublattice[{index}].{name}",
                        lambda n=name, i=index: float(result.get_value_of(
                            _site_fraction_quantity(phase, n, i)
                        )),
                    )
                    for name in names
                }
                site_fractions.append({"sublattice": index, "site_ratio": ratio, "constituents": constituents})
        return {
            "phase": phase,
            "stable": stable,
            "phase_amount": amount,
            "phase_composition": composition,
            "site_fractions": site_fractions,
            "diagnostic_errors": errors,
        }

    def _calculate(self, kind: str, phase: str | None, x: np.ndarray, T: float):
        """Calculate with the requested settings after any kinetics mode switch.

        The TC-Python engine can retain minimization settings across calculation
        objects in one session, so a transition reasserts the relevant options.
        Only QTHISS iteration failures from a ``calculate()`` call already
        using global minimization trigger higher-grid retries. An active site-
        fraction capture emits a record for each actual attempt, including
        failed attempts and retries, before the result can be replaced.
        """
        calc = self._get_calculation(kind, phase)
        config = self._require_config()
        mode_switch = (
            self._last_calculation_kind is not None
            and kind.startswith("kinetics") != self._last_calculation_kind.startswith("kinetics")
            and (config.kinetics_disable_global_minimization or config.kinetics_disable_positive_definite_hessian
                 or config.kinetics_constrain_single_composition_set
                 or config.kinetics_multistart_mode != "off")
        )
        if mode_switch or self._retry_settings_dirty:
            calc = self._configure_global_minimization(
                calc,
                config,
                kind,
                restore_shared_settings=kind != "kinetics" or self._retry_settings_dirty,
            )
            include_default_phases = config.use_default_phases if not kind.startswith("kinetics") else False
            self._calculations[(kind, phase, include_default_phases)] = calc
            self._retry_settings_dirty = False
        try:
            self._retry_calculation = None
            calc.remove_all_conditions()
            self._set_conditions(calc, x, T)
            self.totalNumCalcs += 1
            self.total_kind_lst.append(kind)
            self.total_x_lst.append(x.copy())
            try:
                result = calc.calculate()
            except Exception as exc:
                self._emit_site_fraction_calculation(
                    kind, phase, x, T, error=exc,
                    grid_points=None if kind == "kinetics" and (
                        config.kinetics_disable_global_minimization or config.kinetics_constrain_single_composition_set)
                    else config.global_minimization_max_grid_points,
                )
                uses_global_minimization = kind != "kinetics" or not (
                    config.kinetics_disable_global_minimization or config.kinetics_constrain_single_composition_set)
                if (uses_global_minimization and config.equilibrium_qthiss_retry_grid_points
                        and _is_qthiss_iteration_error(exc)):
                    return self._retry_after_qthiss(
                        kind, phase, x, T, exc, config.equilibrium_qthiss_retry_grid_points,
                    )
                raise ThermoCalcSolveError(f"TC-Python {kind} calculation failed: {exc}") from exc
            self._emit_site_fraction_calculation(
                kind, phase, x, T, result=result,
                grid_points=None if kind == "kinetics" and (
                    config.kinetics_disable_global_minimization or config.kinetics_constrain_single_composition_set)
                else config.global_minimization_max_grid_points,
            )
            if kind == "kinetics":
                self._validate_single_kinetics_phase(result, phase, x)
            return result
        except ThermoCalcSolveError:
            raise
        except ThermoCalcCalculationError:
            raise
        except Exception as exc:
            print(kind, phase, x, T)
            if not _is_qthiss_iteration_error(exc):
                debugInPlace()
            raise ThermoCalcCalculationError(f"TC-Python {kind} calculation failed: {exc}") from exc
        finally:
            self._last_calculation_kind = kind

    def _retry_after_qthiss(
        self, kind: str, phase: str | None, x: np.ndarray, T: float,
        initial_error: Exception, retry_points: tuple[int, ...],
    ):
        """Retry QTHISS failures on fresh calculators with larger global grids.

        Retries preserve phase status and minimization mode. The normal
        calculator is evicted and shared settings are restored before the
        next query uses the configured grid limit.
        """
        config = self._require_config()
        include_default_phases = config.use_default_phases if not kind.startswith("kinetics") else False
        key = (kind, phase, include_default_phases)
        last_error = initial_error
        try:
            system = self._get_system(include_default_phases)
            for grid_points in retry_points:
                print(f"Retrying {kind} {phase} at {(x, T)} with {grid_points} grid points", flush=True)
                calc = system.with_single_equilibrium_calculation()
                calc = self._configure_global_minimization(
                    calc,
                    config,
                    kind,
                    restore_shared_settings=True,
                    grid_points_override=grid_points,
                )
                if kind == "kinetics":
                    self._force_kinetics_phase(calc, phase)
                elif kind == "driving_force":
                    calc.set_phase_to_dormant(phase)
                calc.remove_all_conditions()
                self._set_conditions(calc, x, T)
                self.totalNumCalcs += 1
                self.total_kind_lst.append(kind)
                self.total_x_lst.append(x.copy())
                try:
                    result = calc.calculate()
                except Exception as exc:
                    self._emit_site_fraction_calculation(kind, phase, x, T, error=exc, grid_points=grid_points)
                    if not _is_qthiss_iteration_error(exc):
                        raise ThermoCalcSolveError(f"TC-Python {kind} calculation failed: {exc}") from exc
                    last_error = exc
                else:
                    self._emit_site_fraction_calculation(kind, phase, x, T, result=result, grid_points=grid_points)
                    if kind == "kinetics":
                        self._validate_single_kinetics_phase(result, phase, x)
                    self._retry_calculation = calc
                    return result
            raise ThermoCalcSolveError(f"TC-Python {kind} calculation failed: {last_error}") from last_error
        finally:
            self._calculations.pop(key, None)
            self._retry_settings_dirty = True

    def _configure_global_minimization(
        self,
        calc: Any,
        config: ThermoCalcConfig,
        kind: str,
        *,
        restore_shared_settings: bool = False,
        grid_points_override: int | None = None,
    ):
        """Apply minimization settings, restoring defaults after shared-state changes.

        A forced local solve or high-grid retry can change the shared Thermo-Calc
        core state even when calculation objects differ. Reasserting settings
        applies normal defaults before kinetics-specific overrides. The grid
        override is used only by temporary QTHISS retries; a kinetics retry
        retains its configured Hessian mode.
        """
        local_kinetics = kind == "kinetics_candidate" or (kind == "kinetics" and (
            config.kinetics_disable_global_minimization or config.kinetics_constrain_single_composition_set))
        if local_kinetics:
            configured_calc = calc.disable_global_minimization()
            if configured_calc is not None:
                calc = configured_calc
        elif restore_shared_settings:
            configured_calc = calc.enable_global_minimization()
            if configured_calc is not None:
                calc = configured_calc
        max_grid_points = None if local_kinetics else (
            config.global_minimization_max_grid_points if grid_points_override is None else grid_points_override
        )
        disable_hessian = kind == "kinetics" and config.kinetics_disable_positive_definite_hessian
        if max_grid_points is None and not disable_hessian and not restore_shared_settings:
            return calc
        if self._tc_python is None:
            raise ThermoCalcBackendError("TC-Python module is not available for SingleEquilibriumOptions.")

        options_factory = getattr(self._tc_python, "SingleEquilibriumOptions", None)
        if options_factory is None:
            raise ThermoCalcBackendError("This TC-Python version does not expose SingleEquilibriumOptions.")

        options = options_factory()
        if restore_shared_settings:
            setter = getattr(options, "enable_force_positive_definite_phase_hessian", None)
            if setter is None:
                raise ThermoCalcBackendError(
                    "This TC-Python SingleEquilibriumOptions object does not support "
                    "enable_force_positive_definite_phase_hessian()."
                )
            configured_options = setter()
            if configured_options is not None:
                options = configured_options
        if max_grid_points is not None:
            setter = getattr(options, "set_global_minimization_max_grid_points", None)
            if setter is None:
                raise ThermoCalcBackendError(
                    "This TC-Python SingleEquilibriumOptions object does not support "
                    "set_global_minimization_max_grid_points()."
                )
            configured_options = setter(int(max_grid_points))
            if configured_options is not None:
                options = configured_options
        if disable_hessian:
            setter = getattr(options, "disable_force_positive_definite_phase_hessian", None)
            if setter is None:
                raise ThermoCalcBackendError(
                    "This TC-Python SingleEquilibriumOptions object does not support "
                    "disable_force_positive_definite_phase_hessian()."
                )
            configured_options = setter()
            if configured_options is not None:
                options = configured_options

        with_options = getattr(calc, "with_options", None)
        if with_options is None:
            raise ThermoCalcBackendError("This TC-Python calculation object does not support with_options().")
        configured_calc = with_options(options)
        return calc if configured_calc is None else configured_calc

    def _get_system(self, include_default_phases: bool):
        """Return a cached system for the requested phase-selection mode."""
        config = self._require_config()
        include_default_phases = bool(include_default_phases)
        if include_default_phases not in self._systems:
            self._systems[include_default_phases] = self._build_system(config, include_default_phases)
        return self._systems[include_default_phases]

    def _set_conditions(self, calc: Any, x: np.ndarray, T: float):
        """Set intensive variables, system size, and independent composition."""

        config = self._require_config()
        self._set_intensive_conditions(calc, T)
        tq = self._tq()
        for element, value in zip(config.independent_elements, x):
            calc.set_condition(tq.mole_fraction_of_a_component(tc_element_name(element)), float(value))

    def _set_intensive_conditions(self, calc: Any, T: float):
        """Set temperature, pressure, and unit system size without composition."""

        config = self._require_config()
        tq = self._tq()
        calc.set_condition(tq.temperature(), float(T))
        calc.set_condition(tq.pressure(), float(config.pressure))
        calc.set_condition(tq.system_size(), 1.0)

    def _phase_composition(self, result: Any, phase: str) -> np.ndarray:
        config = self._require_config()
        tq = self._tq()
        return np.array(
            [
                self._value(result, tq.composition_of_phase_as_mole_fraction(phase, tc_element_name(element)))
                for element in config.elements
            ],
            dtype=np.float64,
        )

    def _safe_value(self, result: Any, quantity: Any, default: float) -> float:
        try:
            return self._value(result, quantity)
        except ThermoCalcCalculationError:
            return default

    def _value(self, result: Any, quantity: Any) -> float:
        try:
            return float(result.get_value_of(quantity))
        except Exception as exc:
            # from examples.debugInPlace import debugInPlace
            print(exc)
            # debugInPlace()
            raise ThermoCalcCalculationError(f"Unable to query {quantity}: {exc}") from exc

    def _tq(self):
        return self._tc_python.ThermodynamicQuantity

    def _dq(self):
        """Return TC-Python's diffusion-quantity factory.

        Thermodynamic factors and component mobilities are exposed by
        ``DiffusionQuantity`` in TC-Python 2026a, while phase compositions and
        Gibbs energies remain on ``ThermodynamicQuantity``.
        """
        return self._tc_python.DiffusionQuantity

    def _require_config(self) -> ThermoCalcConfig:
        if self._config is None or self._setup is None:
            raise ThermoCalcBackendError("TC-Python backend has not been started.")
        return self._config

    def _raise_backend_error(self, exc: Exception):
        message = str(exc)
        if "license" in message.lower():
            raise ThermoCalcBackendError(f"Thermo-Calc license is not available: {message}") from exc
        self._raise_database_error(exc)

    def _raise_database_error(self, exc: Exception):
        message = str(exc)
        if "QPFIND : NO SUCH INTENSIVE VARIABLE" in message:
            raise ThermoCalcDatabaseError(
                "TC-Python rejected this database during preflight with "
                "'QPFIND : NO SUCH INTENSIVE VARIABLE'. The checked-in Lee Fe-Cr-Ni "
                "TDB is pycalphad-oriented and needs a separate TC-compatible "
                "database conversion/repair step."
            ) from exc
        raise ThermoCalcDatabaseError(f"TC-Python database preflight failed: {message}") from exc


class TCPythonThermodynamics:
    """Kawin-style thermodynamics facade backed by TC-Python.

    The object is lazy: importing and constructing it does not start
    Thermo-Calc.  Enter it as a context manager for long data runs so the same
    TC-Python server session can be reused across many points.
    """

    def __init__(
        self,
        config: ThermoCalcConfig | None = None,
        backend: Any | None = None,
        default_remove_cache: bool = True,
    ):
        """Create an adapter with an optional per-instance cache policy.

        Parameters
        ----------
        config : ThermoCalcConfig, optional
            Thermo-Calc system configuration.
        backend : optional
            Backend used to execute TC-Python calculations. Primarily useful
            for testing.
        default_remove_cache : bool, optional
            Cache policy for public calculation calls that omit
            ``removeCache``. ``True`` recalculates by default; ``False``
            reuses adapter-side results for identical queries. An explicit
            ``removeCache`` argument on an individual call takes precedence.
        """
        self.config = ThermoCalcConfig() if config is None else config
        self.elements = list(self.config.elements)
        self.phases = list(self.config.phases)
        self.numElements = len(self.elements)
        self.independent_elements = list(self.config.independent_elements)
        self._backend = _TCPythonBackend() if backend is None else backend
        self._started = False
        self._cache: dict[tuple[Any, ...], Any] = {}
        self.default_remove_cache = bool(default_remove_cache)
        self._kinetics_diagnostic_callback = None
        self._kinetics_diagnostic_index = 0

    def __enter__(self):
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def close(self):
        """Close the underlying TC-Python session."""

        self._backend.close()
        self._started = False

    def restartSession(self):
        """Restart the TC-Python session and preserve adapter-side cache state."""

        self._backend.restart()
        self._started = True

    def clearCache(self):
        """Clear adapter-side cached values without restarting TC-Python."""

        self._cache.clear()

    @contextmanager
    def captureKineticsDiagnostics(self, callback):
        """Emit ordered records for interdiffusivity queries within this scope.

        Cached queries also emit records. Optional tracer diffusivities,
        mobilities, thermodynamic factors, and phase-state values come from the
        same forced kinetics equilibrium as the returned diffusivity; a cached
        result missing current diagnostic fields is refreshed on first use.
        """
        if not callable(callback):
            raise TypeError("callback must be callable.")
        previous_callback = self._kinetics_diagnostic_callback
        previous_index = self._kinetics_diagnostic_index
        self._kinetics_diagnostic_callback = callback
        self._kinetics_diagnostic_index = 0
        try:
            yield self
        finally:
            self._kinetics_diagnostic_callback = previous_callback
            self._kinetics_diagnostic_index = previous_index

    @contextmanager
    def captureCalculationSiteFractions(self, callback):
        """Emit one record for each backend ``calculate()`` attempt in scope.

        Successful equilibrium, driving-force, and kinetics attempts include
        the stable composition sets and their phase compositions and labeled
        site fractions. Failed attempts, including higher-grid retries, retain
        the calculation error. Adapter cache hits make no backend attempt and
        therefore emit no record. Optional quantity failures do not discard a
        successful calculation.
        """
        if not callable(callback):
            raise TypeError("callback must be callable.")
        backend = self._backend
        previous_callback = getattr(backend, "_site_fraction_capture_callback", None)
        previous_index = getattr(backend, "_site_fraction_capture_index", 0)
        backend._site_fraction_capture_callback = callback
        backend._site_fraction_capture_index = 0
        try:
            yield self
        finally:
            backend._site_fraction_capture_callback = previous_callback
            backend._site_fraction_capture_index = previous_index

    def getRuntimeVersion(self) -> str | None:
        """Return the TC-Python package version when it can be determined."""

        return self._backend.get_runtime_version()

    def preflight(self, x=(0.38, 0.001), T=1373.0) -> dict[str, Any]:
        """Run a lightweight one-point database/license compatibility check."""

        composition = validate_independent_composition(x, self.config)
        return self._backend.preflight(self.config, composition, float(T))

    def getDrivingForce(self, x, T, precPhase=None, removeCache=None):
        """Return driving force in J/mol and precipitate composition.

        ``x`` uses independent mole fractions in ``independent_elements`` order.
        The precipitate composition returned to kawin also uses independent
        order.  TC-Python's dimensionless ``DGM`` is converted as
        ``DGM * R * T``. When ``removeCache`` is omitted, the instance's
        ``default_remove_cache`` policy is used.
        """

        removeCache = self._resolve_remove_cache(removeCache)
        precPhase = _getPrecipitatePhase(self.phases, precPhase).upper()
        x_array, T_array = self._process_xT(x, T)
        values = [self._get_driving_force_single(xi, Ti, precPhase, removeCache) for xi, Ti in zip(x_array, T_array)]
        driving_force, precipitate_composition = zip(*values)
        return np.squeeze(np.array(driving_force, dtype=np.float64)), np.squeeze(
            np.array(precipitate_composition, dtype=np.float64)
        )

    def getInterdiffusivity(self, x, T, phase=None, removeCache=None):
        """Return the Fe-reference chemical interdiffusivity matrix.

        Rows and columns follow ``independent_elements`` order.  For the
        default Fe-Cr-Ni configuration, the result is a 2x2 matrix in
        ``[CR, NI]`` order. When ``removeCache`` is omitted, the instance's
        ``default_remove_cache`` policy is used. An active kinetics diagnostic
        scope emits one record per query, including cache hits.
        """

        removeCache = self._resolve_remove_cache(removeCache)
        phase = _getMatrixPhase(self.phases, phase).upper()
        x_array, T_array = self._process_xT(x, T)
        values = []
        for xi, Ti in zip(x_array, T_array):
            kinetics, cache_hit = self._get_kinetics_single(
                xi, Ti, phase, removeCache, diagnostics=self._kinetics_diagnostic_callback is not None
            )
            values.append(kinetics["interdiffusivity"])
            if self._kinetics_diagnostic_callback is not None:
                diagnostics = kinetics.get("diagnostics", {})
                record = {
                    "record_type": "kinetics",
                    "query_index": self._kinetics_diagnostic_index,
                    "cache_hit": cache_hit,
                    "requested_phase": phase,
                    "temperature": float(Ti),
                    "input_composition": xi.tolist(),
                    "interdiffusivity": np.asarray(kinetics["interdiffusivity"], dtype=np.float64).tolist(),
                    "tracer_diffusivities": [None if value is None else float(value) for value in kinetics["tracer_diffusivity"]],
                    "thermodynamic_factors": diagnostics.get("thermodynamic_factors"),
                    "mobilities": diagnostics.get("mobilities"),
                    "stable_composition_sets": diagnostics.get("stable_composition_sets"),
                    "phase_composition": diagnostics.get("phase_composition"),
                    "site_fractions": diagnostics.get("site_fractions"),
                    "kinetics_strategy": diagnostics.get("kinetics_strategy"),
                    "scout_seed_count": diagnostics.get("scout_seed_count"),
                    "unique_seed_count": diagnostics.get("unique_seed_count"),
                    "converged_candidate_count": diagnostics.get("converged_candidate_count"),
                    "selected_seed_index": diagnostics.get("selected_seed_index"),
                    "selected_source_composition_set": diagnostics.get("selected_source_composition_set"),
                    "selected_gibbs_energy": diagnostics.get("selected_gibbs_energy"),
                    "selected_gibbs_energy_units": diagnostics.get("selected_gibbs_energy_units"),
                    "automatic_session_restart_count": diagnostics.get(
                        "automatic_session_restart_count"
                    ),
                    "multistart_candidates": diagnostics.get("multistart_candidates"),
                    "diagnostic_errors": dict(diagnostics.get("errors", {})),
                }
                if not diagnostics:
                    record["diagnostic_errors"]["diagnostics"] = "Backend did not return optional diagnostics."
                self._kinetics_diagnostic_callback(record)
                self._kinetics_diagnostic_index += 1
        return np.squeeze(np.array(values, dtype=np.float64))

    def getTracerDiffusivity(self, x, T, phase=None, removeCache=None):
        """Return tracer diffusivities in full ``elements`` order.

        When ``removeCache`` is omitted, the instance's
        ``default_remove_cache`` policy is used. Missing tracer values from a
        diagnostic capture raise an error rather than returning NaNs.
        """

        removeCache = self._resolve_remove_cache(removeCache)
        phase = _getMatrixPhase(self.phases, phase).upper()
        x_array, T_array = self._process_xT(x, T)
        values = [self._get_kinetics_single(xi, Ti, phase, removeCache)[0]["tracer_diffusivity"] for xi, Ti in zip(x_array, T_array)]
        if any(any(value is None for value in tracer) for tracer in values):
            raise ThermoCalcCalculationError(f"Tracer diffusivity is unavailable for phase {phase}.")
        return np.squeeze(np.array(values, dtype=np.float64))

    def getEquilibriumData(self, x, T, removeCache=None):
        """Return stable phases, phase amounts, compositions, and chemical potentials.

        Diffusivity quantities are queried through the dedicated kinetics API.
        When ``removeCache`` is omitted, the instance's
        ``default_remove_cache`` policy is used.
        """

        removeCache = self._resolve_remove_cache(removeCache)
        x_array, T_array = self._process_xT(x, T)
        values = [self._get_equilibrium_single(xi, Ti, removeCache) for xi, Ti in zip(x_array, T_array)]
        return values[0] if len(values) == 1 else values

    def getInterfacialComposition(self, x, T, gExtra=0, precPhase=None, returnMeta=False, removeCache=None):
        """Return planar BCC/FCC tie-line endpoints from equilibrium.

        Nonzero ``gExtra`` is intentionally unsupported in this example because
        Gibbs-Thomson-corrected multicomponent curvature is outside the first
        TC-Python integration scope. When ``removeCache`` is omitted, the
        instance's ``default_remove_cache`` policy is used. If the expected
        phases are absent, metadata includes the actual stable-phase names.
        """

        if np.any(np.asarray(gExtra, dtype=np.float64) != 0):
            raise NotImplementedError("TC-Python example adapter currently supports only planar gExtra=0 endpoints.")

        precPhase = _getPrecipitatePhase(self.phases, precPhase).upper()
        matrix_phase = self.config.matrix_phase
        equilibrium = self.getEquilibriumData(x, T, removeCache=removeCache)
        phase_compositions = equilibrium["phase_compositions"]
        if set([matrix_phase, precPhase])==set(phase_compositions.keys()):
            x_alpha = full_to_independent_composition(phase_compositions[matrix_phase], self.config)
            x_beta = full_to_independent_composition(phase_compositions[precPhase], self.config)
            metadata = {
                "endpoint_phases": (matrix_phase, precPhase),
                "endpoints": (
                    {"phase": matrix_phase, "composition": x_alpha},
                    {"phase": precPhase, "composition": x_beta},
                ),
            }
        else:
            # from examples.debugInPlace import debugInPlace
            print(x)
            # debugInPlace()
            x_alpha = -1.0 * np.ones(len(self.independent_elements), dtype=np.float64)
            x_beta = -1.0 * np.ones(len(self.independent_elements), dtype=np.float64)
            metadata = {
                "endpoint_phases": (None, None),
                "stable_phases": tuple(equilibrium["stable_phases"]),
                "endpoints": (
                    {"phase": None, "composition": x_alpha},
                    {"phase": None, "composition": x_beta},
                ),
            }
        if returnMeta:
            return np.squeeze(x_alpha), np.squeeze(x_beta), metadata
        return np.squeeze(x_alpha), np.squeeze(x_beta)

    def _process_xT(self, x, T) -> tuple[np.ndarray, np.ndarray]:
        x_array, T_array = _process_xT_arrays(x, T, isBinary=False)
        for xi, Ti in zip(x_array, T_array):
            validate_independent_composition(xi, self.config)
            if not np.isfinite(Ti) or Ti <= 0:
                raise ThermoCalcInputError("Temperature must be a positive finite value in K.")
        return x_array.astype(np.float64), T_array.astype(np.float64)

    def _resolve_remove_cache(self, removeCache: bool | None) -> bool:
        """Return an explicit cache policy or this instance's default policy."""

        if removeCache is None:
            return self.default_remove_cache
        return bool(removeCache)

    def _get_equilibrium_single(self, x: np.ndarray, T: float, removeCache: bool):
        key = ("equilibrium", tuple(np.asarray(x, dtype=float)), float(T))
        return self._cached_or_calculate(key, removeCache, lambda: self._backend.calculate_equilibrium(x, T))

    def _get_driving_force_single(self, x: np.ndarray, T: float, precPhase: str, removeCache: bool):
        key = ("driving_force", precPhase, tuple(np.asarray(x, dtype=float)), float(T))
        result = self._cached_or_calculate(
            key,
            removeCache,
            lambda: self._backend.calculate_driving_force(x, T, precPhase),
        )
        return result["driving_force"], full_to_independent_composition(result["precipitate_composition"], self.config)

    def _get_kinetics_single(self, x: np.ndarray, T: float, phase: str, removeCache: bool, *, diagnostics: bool = False):
        """Return kinetics and cache status, refreshing incomplete diagnostics.

        Diagnostic fields can grow between adapter versions. Cached kinetics
        results that predate a required field are recalculated so captures do
        not silently emit a mixture of old and current record schemas.
        """
        key = ("kinetics", phase, tuple(np.asarray(x, dtype=float)), float(T))
        if diagnostics and not removeCache and key in self._cache:
            cached_diagnostics = self._cache[key].get("diagnostics")
            if not isinstance(cached_diagnostics, dict) or "mobilities" not in cached_diagnostics:
                del self._cache[key]
        cache_hit = not removeCache and key in self._cache
        if diagnostics:
            callback = lambda: self._backend.calculate_kinetics(x, T, phase, collect_diagnostics=True)
        else:
            callback = lambda: self._backend.calculate_kinetics(x, T, phase)
        return self._cached_or_calculate(key, removeCache, callback), cache_hit

    def _cached_or_calculate(self, key: tuple[Any, ...], removeCache: bool, callback):
        self._backend.totalNumQueries += 1
        if not removeCache and key in self._cache:
            self._backend.totalNumCaches += 1
            return self._cache[key]
        self._ensure_started()
        result = callback()
        if not removeCache:
            self._cache[key] = result
        return result

    def _ensure_started(self):
        if not self._started:
            self._backend.start(self.config)
            self._started = True
