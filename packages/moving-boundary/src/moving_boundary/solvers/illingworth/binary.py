# Relocated from samuelgprice/kawin 25bf6cd60732ed34ed6f70b94716a515c76b714c (R06 correction on top of main 5f4ad57).
# Source: kawin/diffusion/MovingBoundaryIllingworthFDM.py; import changes only. See NOTICE and docs/LEDGERS.md.
import warnings
from dataclasses import dataclass

import numpy as np

from moving_boundary.compatibility import GenericModel
from moving_boundary.compatibility import DiffusionModel
from moving_boundary.compatibility.recording import (
    _advance_history_rows,
    _finalize_history_rows,
    _history_row_target,
)
from moving_boundary.solvers.illingworth._interstitials import (
    has_interstitials,
    record_initial_u_fractions,
    resolve_diffusivity_provider,
    validate_composition_basis,
    validate_interstitial_element_order,
    x_to_u_independent,
)
from moving_boundary.solvers.illingworth.shared import (
    _BULK_DIFFUSIVITY_IMPLICIT,
    _BULK_DIFFUSIVITY_PHASE_UNIFORM,
    _PendingImplicitStep,
    _check_trial_dt_against_outer_minimum,
    _coerce_bulk_diffusivity_mode,
    _magnitude,
    _outer_timestep_limits,
    _pending_step_mismatches,
    _validate_outer_dt_fractions,
)
from moving_boundary.mesh import CartesianFD1D, MixedBoundary1D, PeriodicBoundary1D
from moving_boundary.solvers.illingworth.numerics.binary import (
    flatten_1d_coordinates,
    integrate_planar_transformed_profile,
    reconstruct_planar_transformed_profile,
    solve_illingworth_tridiagonal,
)
from kawin.solver import explicitEulerIterator

def _loge_arange(start, stop, log_step):
    """Returns exponentially spaced target times with fixed natural-log spacing."""
    logs = np.arange(np.log(start), np.log(stop), log_step)
    return np.exp(logs)


@dataclass(frozen=True)
class _BinaryBulkSolveResult:
    """
    One binary transformed-phase solve and its interface-face diffusivity.

    ``interface_D`` is the diffusivity on the face adjacent to the interface
    (a scalar in every mode), which the next interface update uses to form the
    phase-side diffusive flux. ``interface_deviation`` is the solved deviation
    of the interface-adjacent node from the interface composition (``w[-2]``
    on the left, ``w[1]`` on the right). The interface update must use it
    rather than recomputing ``profile - c_interface``: that difference loses
    the digits below the rounding unit of ``c_interface`` (ledger R06).
    """

    profile: np.ndarray
    interface_D: float
    inner_iterations: int = 0
    inner_update_norm: float = 0.0
    interface_deviation: float = np.nan


def _validate_binary_diffusivity(values, phase, context="binary Illingworth diffusivity"):
    """Returns ``values`` as float64 after checking every entry is finite and positive."""
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{context} for phase {phase} must be finite.")
    if np.any(values <= 0.0):
        raise ValueError(f"{context} for phase {phase} must be positive.")
    return values


_DEFAULT_BULK_PICARD_RTOL = 1.0e-10
_ZERO_INVENTORY_SCALE_FACTOR = 64.0 * np.finfo(float).eps


def _require_positive_finite(value, name):
    """Returns ``float(value)`` after checking it is positive and finite, else raises ``ValueError``."""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return value


def _binary_residual_scale(inventory, domain_length, composition_floor):
    """
    Returns the normalization of the binary interface residual for one step (ledger R08).

    ``scale = max(|inventory|, domain_length * composition_floor)``, where
    ``inventory`` is the transformed solute inventory of the accepted old state
    and ``composition_floor`` is a composition (``minComposition`` or the
    explicit ``interface_residual_composition_scale``). The residual check
    ``|r| <= residual_tolerance * scale`` therefore bounds the relative
    inventory change per step while ``|inventory| >= R * floor``, and the change
    of the mean composition per step by ``residual_tolerance * floor`` below
    that.

    Only when ``inventory`` and ``composition_floor`` are both exactly zero (a
    test on the inputs, possible only with ``minComposition = 0``) is the
    scale ``64 eps * domain_length``: an absolute, round-off-level bound,
    because a relative bound is undefined at zero inventory. A positive floor
    whose product with the length underflows to zero is not that case.

    Raises ``ValueError`` naming the quantity for a non-finite inventory, a
    non-finite or non-positive length, a non-finite or negative floor, and a
    scale that overflows or underflows to a non-positive value. Nothing is
    replaced by a substitute value.
    """
    inventory = float(inventory)
    domain_length = float(domain_length)
    composition_floor = float(composition_floor)
    if not np.isfinite(inventory):
        raise ValueError(f"Illingworth residual normalization: the old inventory {inventory!r} is not finite.")
    if not np.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError(f"Illingworth residual normalization: the domain length {domain_length!r} must be positive and finite.")
    if not np.isfinite(composition_floor) or composition_floor < 0.0:
        raise ValueError(
            f"Illingworth residual normalization: the composition floor {composition_floor!r} must be non-negative and finite."
        )
    if inventory == 0.0 and composition_floor == 0.0:
        scale = domain_length * _ZERO_INVENTORY_SCALE_FACTOR
    else:
        scale = max(abs(inventory), domain_length * composition_floor)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"Illingworth residual normalization: the scale {scale!r} (inventory {inventory!r}, domain length "
            f"{domain_length!r}, composition floor {composition_floor!r}) must be positive and finite."
        )
    return float(scale)

class _ScalarHistory:
    """
    Scalar time history of the binary Illingworth model (interface, inventory).

    ``record`` follows the shared row rule (``compatibility.recording``,
    LEDGERS R09): with ``record=k`` the initial row and one row every ``k``
    accepted steps (counted from ``reset`` across solves) are appended, a
    forced row is appended without consuming a cadence slot, and nothing is
    overwritten. ``record=False`` keeps one row holding the current value.
    ``N`` is the last stored row, ``currentIndex`` the cadence counter,
    ``_rowCount`` the number of stored rows and ``_lastRecordStored`` whether
    the latest ``record`` call stored a row.
    """
    def __init__(self, record: bool | int = False):
        if isinstance(record, bool):
            self.recordInterval = 1 if record else -1
        else:
            self.recordInterval = int(record)
        self.batchSize = 1000
        self.reset()

    def reset(self):
        self._y = np.zeros(self.batchSize, dtype=np.float64)
        self._time = np.zeros(self.batchSize, dtype=np.float64)
        self.currentIndex = 0
        self.currentY = 0.0
        self.currentTime = 0.0
        self.N = 0
        self._rowCount = 0
        self._lastRecordStored = False

    def preallocate(self, capacity):
        """Preallocates scalar history storage without changing recorded values."""
        capacity = int(capacity)
        if capacity <= self._time.shape[0]:
            return
        y_new = np.zeros(capacity, dtype=self._y.dtype)
        time_new = np.zeros(capacity, dtype=self._time.dtype)
        y_new[: self._y.shape[0]] = self._y
        time_new[: self._time.shape[0]] = self._time
        self._y = y_new
        self._time = time_new

    def record(self, time, y, force: bool = False):
        """Stores ``y`` under the shared row rule; ``currentY``/``currentTime`` always take this call's values."""
        target = _history_row_target(self, force)
        if target is not None:
            if target >= self._time.shape[0]:
                self._y = np.pad(self._y, (0, self.batchSize))
                self._time = np.pad(self._time, (0, self.batchSize))
            self._y[target] = y
            self._time[target] = time
        _advance_history_rows(self, target, force)

        self.currentY = float(y)
        self.currentTime = float(time)

    def finalize(self):
        """Appends the current value unless the latest record stored a row, then trims (flag-based, no time tolerance)."""
        _finalize_history_rows(self)

    def y(self, time=None):
        """
        Returns the current value without ``time``; otherwise a stored-row lookup.

        A timed request interpolates linearly between stored rows and clamps to
        the first/last stored row, so the current time of an unstored step
        returns the last stored row, not the live value.
        """
        if time is None:
            return float(self.currentY)
        if self.recordInterval > 0:
            if time <= self._time[0]:
                return float(self._y[0])
            if time >= self._time[self.N]:
                return float(self._y[self.N])
            uind = np.argmax(self._time > time)
            lind = uind - 1
            uy, utime = self._y[uind], self._time[uind]
            ly, ltime = self._y[lind], self._time[lind]
            return float((uy - ly) * (time - ltime) / (utime - ltime) + ly)
        return float(self._y[0])

class _VectorHistory:
    """
    Vector time history (fixed length) of the binary transformed profiles ``p``/``q``.

    Recording follows the same shared row rule and bookkeeping as
    ``_ScalarHistory``. Timed lookups need an exactly recorded time.
    """
    def __init__(self, n_components: int, record: bool | int = False):
        if isinstance(record, bool):
            self.recordInterval = 1 if record else -1
        else:
            self.recordInterval = int(record)
        self.n_components = int(n_components)
        self.batchSize = 1000
        self.reset()

    def reset(self):
        '''
        Resets arrays for a vector-valued history.
        '''
        self._y = np.zeros((self.batchSize, self.n_components), dtype=np.float64)
        self._time = np.zeros(self.batchSize, dtype=np.float64)
        self.currentIndex = 0
        self.currentY = np.zeros(self.n_components, dtype=np.float64)
        self.currentTime = 0.0
        self.N = 0
        self._rowCount = 0
        self._lastRecordStored = False

    def preallocate(self, capacity):
        """Preallocates vector history storage without changing recorded values."""
        capacity = int(capacity)
        if capacity <= self._time.shape[0]:
            return
        y_new = np.zeros((capacity, self.n_components), dtype=self._y.dtype)
        time_new = np.zeros(capacity, dtype=self._time.dtype)
        y_new[: self._y.shape[0]] = self._y
        time_new[: self._time.shape[0]] = self._time
        self._y = y_new
        self._time = time_new

    def record(self, time, y, force: bool = False):
        '''
        Stores current state of time and vector variable under the shared row rule.
        '''
        values = np.asarray(y, dtype=np.float64).reshape(-1)
        if values.size != self.n_components:
            raise ValueError(f"Expected {self.n_components} components, got {values.size}.")
        target = _history_row_target(self, force)
        if target is not None:
            if target >= self._time.shape[0]:
                self._y = np.pad(self._y, ((0, self.batchSize), (0, 0)))
                self._time = np.pad(self._time, (0, self.batchSize))
            self._y[target] = values
            self._time[target] = time
        _advance_history_rows(self, target, force)
        self.currentY = values.copy()
        self.currentTime = float(time)

    def finalize(self):
        '''
        Appends the current value unless the latest record stored a row, then
        removes extra padding (flag-based, no time tolerance).
        '''
        _finalize_history_rows(self)

    def y(self, time=None):
        '''
        Returns a copy of the current vector without ``time``; otherwise the
        value at an exactly recorded time (``atol=1e-14``, last match), or
        ``ValueError``. The live value is never substituted for an unstored time.
        '''
        if time is None:
            return np.array(self.currentY, dtype=np.float64)
        recorded_time = self._time[: self.N + 1]
        matches = np.where(np.isclose(recorded_time, float(time), atol=1e-14, rtol=0.0))[0]
        if len(matches) == 0:
            raise ValueError(
                f"Requested time {float(time):.6g} was not found in exact recorded interface composition times."
            )
        return self._y[matches[-1]].copy()

class MovingBoundaryIllingworthFD1DModel(DiffusionModel):
    """
    Binary planar moving-boundary model following Illingworth and Golosnoy.

    This implementation ports the planar implicit front-fixing algorithm from
    the authors' MAP C++ code. Each step solves a fixed-point problem for the
    future interface position and the two future transformed concentration
    profiles; the phase solves are tri-diagonal and the interface equation uses
    the paper's conservative planar balance.

    Diffusivity is selected with ``bulk_diffusivity_mode``, mirroring the
    ternary Illingworth model:

    - ``'phase_uniform'`` (default) evaluates one diffusivity per phase at that
      phase's (fixed) interface composition. For constant-diffusivity
      thermodynamics this is the MAP-verified recurrence (solved in deviation
      form, see below, so not bitwise identical to the MAP solve). When
      the temperature is constant these values cannot change, so they are
      evaluated once in ``setup()`` and reused; otherwise they are re-evaluated
      each step at the interface temperature.
    - ``'composition_dependent_lagged'`` evaluates one diffusivity per
      finite-volume face from the accepted old-time profile (face ``j`` uses the
      arithmetic mean of nodes ``j`` and ``j + 1``) and freezes those values for
      the whole step. The lag makes the D(c) coupling first-order in ``dt``.
    - ``'composition_dependent_implicit'`` re-evaluates the face values from a
      Picard iterate inside each phase solve, seeded from the old profile with
      the interface Dirichlet value imposed, until the update norm falls below
      ``bulk_picard_atol + rtol * max|profile|``. ``rtol`` is
      ``bulk_picard_rtol``, or ``1e-10`` when it is ``None``; it no longer
      follows ``tolerance`` (ledger R08).

    In every mode the interface balance uses the diffusivity on the
    interface-adjacent face returned by the most recent phase solve, so the
    interface flux is consistent with the bulk flux of that face and the
    transformed inventory remains conserved. A failed implicit step (fixed-point
    or Picard non-convergence, zero pivot, interface leaving the domain) is
    retried with ``dt *= retry_factor`` up to ``max_step_retries`` total
    attempts; the default of one attempt keeps the original raise-on-failure
    behavior.

    Each phase is solved for its deviation from the interface composition and
    the interface balance uses the solved interface-adjacent deviations. This
    is algebraically the MAP scheme, but it keeps the interface fluxes
    accurate when the near-interface profile is close to the interface
    composition: forming ``c_interface - p[-2]`` from absolute compositions
    leaves round-off of order ``eps * |c|``, which the flux amplifies by
    ``D dt / dz`` into noise in the interface position. With small interface
    cells and long semi-log steps that noise exceeded the position tolerance
    and the fixed point stalled (issue R06 in the moving-boundary package's
    ``docs/LEDGERS.md``). Stored profiles remain absolute compositions, so
    each step's old deviation is rounded once, to the unit of
    ``c_interface``.

    The fixed-point iteration on the interface position accepts a trial
    position ``s*`` (never the pass-zero iterate) only when both checks hold
    (ledger R08):

    - ``|s* - s_previous| <= tolerance * R``. ``tolerance`` is dimensionless
      (relative to the domain length ``R``), default ``1e-12``. Before R08 it
      was an absolute length with default ``1e-8``; an old absolute value
      ``t`` corresponds to ``t / R``.
    - ``|r| <= residual_tolerance * scale`` (default ``1e-13``). ``r`` is the
      conservative interface residual ``lhs * (s* - s) - rhs`` of the planar
      balance, formed with the same operations as the interface update from
      the deviations and interface-face diffusivities of the bulk solves at
      ``s*``; it equals the step's change of the transformed inventory up to
      round-off. ``scale = max(|I_old|, R * c_floor)`` is fixed once per step
      from the accepted old inventory ``I_old``. ``c_floor`` is
      ``interface_residual_composition_scale`` when given, otherwise
      ``constraints.minComposition`` read at the start of the step. (The
      ternary models use the same argument name, but there the default is a
      composition scale of 1, so their default residual scale is ``R``; see
      ledger R13.)

    The residual check bounds the change of the transformed inventory per step
    by ``residual_tolerance * max(|I_old|, R * c_floor)`` up to round-off, so
    by ``N * residual_tolerance`` relative over ``N`` steps while
    ``|I_old| >= R * c_floor``. The floor can bind for valid data: interface
    compositions are not clamped, and a profile near a zero interface
    composition can integrate to less than ``R * minComposition``. The bound is
    then absolute, at most ``residual_tolerance * c_floor`` of mean
    composition per step. If the old inventory and the floor are both exactly
    zero (only possible with ``minComposition = 0``) the scale is
    ``64 eps * R``, an absolute round-off-level bound. Any other non-finite or
    non-positive scale raises ``ValueError``.
    The transformed profile histories ``pData`` and ``qData`` are recorded by
    default, but may be disabled with ``record_pq_data=False`` to reduce
    memory use in long runs; ``getTransformedState()`` without a time still
    returns the accepted ``p``/``q``. With ``record=k`` every history stores the
    initial row, one row every ``k`` accepted steps and a final row per solve;
    getters without a time always return the current accepted state (LEDGERS
    R09). Recording arrays can also be preallocated once the
    timestep schedule is known; this avoids repeated padding but may allocate
    large full-profile histories up front. Custom transformed grids may be
    supplied with ``transformed_u_grid`` and ``transformed_v_grid`` for planar
    validation cases that use nonuniform Landau-coordinate spacing; grids must
    be finite, strictly increasing, and span exactly from 0 to 1.

    Interstitial solutes (``kawin.thermo.Mobility.interstitials``, e.g. Fe-C)
    are supported through u-fractions, ``u = x / (1 - x_interstitial)``. The
    transformed state, the recorded profiles and the conserved inventory are
    then u-fractions, and diffusion is in the volume-fixed frame where only the
    substitutional element carries volume, with a constant molar volume per mole
    of substitutional atoms that is equal in both phases. Thermodynamics objects
    are still queried with mole fractions (the conversion happens at the query
    boundary), and their diffusivity must be the u-basis volume-fixed value
    returned by ``kawin.thermo.Mobility.interdiffusivity`` and Thermo-Calc's
    ``DC``. ``interface_compositions`` are interpreted in
    ``interface_composition_basis`` (``'x'`` by default, or ``'u'``) and stored
    in ``interfaceCompositions`` as u-fractions; the caller's values are kept in
    ``interfaceCompositionsInput``. ``getCompositions()`` returns mole
    fractions. For substitutional systems ``u == x`` and results are unchanged.

    Notes
    -----
    Limitations:

    - **Geometry:** planar only. Cylindrical and spherical equations are
      intentionally excluded until their coefficient paths are separately
      tested.
    - **Outer boundaries:** homogeneous zero flux only (closed system).
      Periodic, fixed-composition and nonzero-flux boundary conditions raise
      ``NotImplementedError``, as in the ternary models.
    - **Interface:** local equilibrium with fixed interface compositions. There
      is no interface mobility and no para-equilibrium.
    - **Molar volume:** constant and equal in both phases (per mole of
      substitutional atoms with interstitials), volume-fixed frame, no
      Kirkendall or bulk flow. Interstitial u-fractions have no site-fraction
      upper cap.
    - **Phases:** exactly two, one interface. A vanishing phase is not handled;
      the step fails once the interface would leave the domain.
    - **Temperature:** may vary; the diffusivities follow it.
    - **Output:** ``toDict``/``save`` write recorded histories only. Restart
      is not supported: ``fromDict`` and ``load`` raise ``NotImplementedError``.
    """

    def __init__(
        self,
        mesh,
        elements,
        phases,
        thermodynamics,
        temperature,
        interfacePosition,
        interface_compositions: tuple[float, float],
        time_step: float,
        dt_mode: str = "fixed",
        semiLog_dt: float | None = None,
        semiLogT0: float | None = None,
        geometry: str = "planar",
        phase_a_nodes: int | None = None,
        phase_b_nodes: int | None = None,
        tolerance: float = 1e-12,
        max_iterations: int = 100,
        bulk_diffusivity_mode: str = _BULK_DIFFUSIVITY_PHASE_UNIFORM,
        bulk_picard_rtol: float | None = None,
        bulk_picard_atol: float = 1e-12,
        bulk_picard_max_iterations: int = 25,
        bulk_picard_relaxation: float = 1.0,
        max_step_retries: int = 1,
        retry_factor: float = 0.5,
        constraints=None,
        record=False,
        record_pq_data: bool = True,
        preallocate_recordings: bool = False,
        transformed_u_grid=None,
        transformed_v_grid=None,
        interface_composition_basis: str = "x",
        residual_tolerance: float = 1e-13,
        interface_residual_composition_scale: float | None = None,
    ):
        # Validated before the host constructor builds any state (ledger R08).
        self.tolerance = _require_positive_finite(tolerance, "tolerance")
        self.residualTolerance = _require_positive_finite(residual_tolerance, "residual_tolerance")
        self.interfaceResidualCompositionScale = (
            None
            if interface_residual_composition_scale is None
            else _require_positive_finite(interface_residual_composition_scale, "interface_residual_composition_scale")
        )
        self.initialInterfacePosition = float(interfacePosition)
        self.interfaceCompositionBasis = validate_composition_basis(interface_composition_basis, "interface_composition_basis")
        self.interfaceCompositionsInput = tuple(float(v) for v in interface_compositions)
        self.interfaceCompositions = self.interfaceCompositionsInput
        self.timeStep = float(time_step)
        self.dtMode = str(dt_mode)
        self.semiLog_dt = None if semiLog_dt is None else float(semiLog_dt)
        self.semiLogT0 = None if semiLogT0 is None else float(semiLogT0)
        self.geometry = str(geometry)
        self.phaseANodes = None if phase_a_nodes is None else int(phase_a_nodes)
        self.phaseBNodes = None if phase_b_nodes is None else int(phase_b_nodes)
        self.maxIterations = int(max_iterations)
        self.bulkDiffusivityMode = _coerce_bulk_diffusivity_mode(bulk_diffusivity_mode)
        self.bulkPicardRtol = None if bulk_picard_rtol is None else float(bulk_picard_rtol)
        self.bulkPicardAtol = float(bulk_picard_atol)
        self.bulkPicardMaxIterations = int(bulk_picard_max_iterations)
        self.bulkPicardRelaxation = float(bulk_picard_relaxation)
        self.maxStepRetries = int(max_step_retries)
        self.retryFactor = float(retry_factor)
        self.recordPqData = bool(record_pq_data)
        self.preallocateRecordings = bool(preallocate_recordings)
        self._inputUGrid = self._validate_transformed_grid(transformed_u_grid, "transformed_u_grid")
        self._inputVGrid = self._validate_transformed_grid(transformed_v_grid, "transformed_v_grid")
        if self._inputUGrid is not None:
            if self.phaseANodes is not None and self.phaseANodes != len(self._inputUGrid):
                raise ValueError("phase_a_nodes must match the length of transformed_u_grid.")
            self.phaseANodes = len(self._inputUGrid)
        if self._inputVGrid is not None:
            if self.phaseBNodes is not None and self.phaseBNodes != len(self._inputVGrid):
                raise ValueError("phase_b_nodes must match the length of transformed_v_grid.")
            self.phaseBNodes = len(self._inputVGrid)

        self.interfaceData = _ScalarHistory(record)
        self.concData = _ScalarHistory(record)
        self.pData = None
        self.qData = None
        self._currdt = np.inf
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
        self._lastImplicitIterations = 0
        self._lastImplicitError = np.nan
        self._lastImplicitResidual = np.nan
        self._lastImplicitResidualScale = np.nan
        self._lastStepRetries = 0
        self._lastLeftPicardIterations = 0
        self._lastRightPicardIterations = 0
        self._nearFinalNoop = False
        self._outerMinTimeStep = 0.0
        self._outerMaxTimeStep = np.inf
        self._solveMinDtFrac = 1e-8
        self._solveMaxDtFrac = 1.0
        self._pendingCandidate = None
        self._awaitingFreshCandidate = False
        self._initialInventory = None

        self._z = None
        self._R = None
        self._u_grid = None
        self._v_grid = None
        self._p_curr = None
        self._q_curr = None
        self._s_curr = None
        self._s_old = None
        self._D_left = None
        self._D_right = None
        self._phaseUniformCache = None

        super().__init__(
            mesh=mesh,
            elements=elements,
            phases=phases,
            thermodynamics=thermodynamics,
            temperature=temperature,
            constraints=constraints,
            record=record,
        )
        self._validateModelConfiguration()
        self.interfaceCompositions = self._interface_compositions_as_u()
        self.interfaceData.currentY = self.initialInterfacePosition
        self.interfaceData._y[0] = self.initialInterfacePosition

    def _interface_compositions_as_u(self):
        """
        Returns the fixed interface compositions as u-fractions.

        Mole-fraction inputs are converted with ``u = x / (1 - x)`` when the
        solute is interstitial; otherwise (substitutional solute, or
        ``interface_composition_basis='u'``) the input values are returned
        unchanged.
        """
        if self.interfaceCompositionBasis == "u" or not has_interstitials(self.allElements):
            return self.interfaceCompositionsInput
        return tuple(float(x_to_u_independent(v, self.allElements)) for v in self.interfaceCompositionsInput)

    def _diffusivity_source(self):
        """
        Returns the object queried for interdiffusivities.

        With interstitials this wraps ``self.therm`` so u-fraction compositions
        are converted to mole fractions before each query; otherwise it is
        ``self.therm`` itself.
        """
        return resolve_diffusivity_provider(self.therm, self.allElements)

    def _validate_transformed_grid(self, grid, name):
        """Validates an optional planar Landau-coordinate grid."""
        if grid is None:
            return None
        values = np.asarray(grid, dtype=np.float64).reshape(-1)
        if len(values) < 3:
            raise ValueError(f"{name} must contain at least three nodes.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        if not np.isclose(values[0], 0.0, rtol=0.0, atol=1e-14) or not np.isclose(values[-1], 1.0, rtol=0.0, atol=1e-14):
            raise ValueError(f"{name} must start at 0 and end at 1.")
        if not np.all(np.diff(values) > 0.0):
            raise ValueError(f"{name} must be strictly increasing.")
        values[0] = 0.0
        values[-1] = 1.0
        return values

    def _validateModelConfiguration(self):
        if not isinstance(self.mesh, CartesianFD1D):
            raise TypeError("MovingBoundaryIllingworthFD1DModel requires a CartesianFD1D mesh.")
        if len(self.allElements) != 2 or self.mesh.numResponses != 1:
            raise ValueError("MovingBoundaryIllingworthFD1DModel currently supports only binary systems.")
        validate_interstitial_element_order(self.allElements)
        if len(self.phases) != 2:
            raise ValueError("MovingBoundaryIllingworthFD1DModel requires exactly two explicit phases.")
        self._validate_external_boundary_conditions()
        if self.geometry != "planar":
            raise NotImplementedError("MovingBoundaryIllingworthFD1DModel currently implements only planar geometry.")
        if not np.isfinite(self.timeStep) or self.timeStep <= 0:
            raise ValueError("time_step must be a positive finite value.")
        if self.dtMode not in {"fixed", "semi_log"}:
            raise ValueError("dt_mode must be 'fixed' or 'semi_log'.")
        if self.dtMode == "semi_log" and ((self.semiLog_dt is None) or (self.semiLogT0 is None)):
            raise ValueError("semiLog_dt and semiLogT0 must be specified when dt_mode is 'semi_log'.")
        if self.semiLog_dt is not None and (not np.isfinite(self.semiLog_dt) or self.semiLog_dt <= 0):
            raise ValueError("semiLog_dt must be a positive finite value.")
        if self.semiLogT0 is not None and (not np.isfinite(self.semiLogT0) or self.semiLogT0 <= 0):
            raise ValueError("semiLogT0 must be a positive finite value.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be a positive finite value.")
        _require_positive_finite(self.residualTolerance, "residual_tolerance")
        if self.interfaceResidualCompositionScale is not None:
            _require_positive_finite(self.interfaceResidualCompositionScale, "interface_residual_composition_scale")
        if self.maxIterations < 2:
            raise ValueError("max_iterations must be at least 2.")
        self.bulkDiffusivityMode = _coerce_bulk_diffusivity_mode(self.bulkDiffusivityMode)
        if self.bulkPicardMaxIterations < 1:
            raise ValueError("bulk_picard_max_iterations must be at least 1.")
        if not np.isfinite(self.bulkPicardRelaxation) or not (0.0 < self.bulkPicardRelaxation <= 1.0):
            raise ValueError("bulk_picard_relaxation must lie in (0, 1].")
        if not np.isfinite(self.bulkPicardAtol) or self.bulkPicardAtol < 0:
            raise ValueError("bulk_picard_atol must be a non-negative finite value.")
        if self.bulkPicardRtol is not None and (not np.isfinite(self.bulkPicardRtol) or self.bulkPicardRtol < 0):
            raise ValueError("bulk_picard_rtol must be a non-negative finite value.")
        if self.maxStepRetries < 1:
            raise ValueError("max_step_retries must be at least 1.")
        if not np.isfinite(self.retryFactor) or not (0.0 < self.retryFactor < 1.0):
            raise ValueError("retry_factor must lie in (0, 1).")
        if self.phaseANodes is not None and self.phaseANodes < 3:
            raise ValueError("phase_a_nodes must be at least 3 when specified.")
        if self.phaseBNodes is not None and self.phaseBNodes < 3:
            raise ValueError("phase_b_nodes must be at least 3 when specified.")
        self.initialInterfacePosition = self._clipInterfacePosition(self.initialInterfacePosition, strict=True)

    def _validate_external_boundary_conditions(self):
        """
        Validates that the fixed external boundaries are homogeneous zero flux.

        As in the ternary Illingworth models, the discretization is conservative
        for a closed planar domain: both outer rows are zero-flux rows. Nonzero
        Neumann fluxes, fixed-composition boundaries, periodic wrapping, or
        other boundary objects would need extra terms that are not part of this
        solver, so they are rejected before setup instead of being ignored.
        """
        bc = getattr(self.mesh, "boundaryConditions", None)
        if bc is None:
            return
        if isinstance(bc, PeriodicBoundary1D):
            raise NotImplementedError("Binary Illingworth currently supports only homogeneous zero-flux external boundaries; periodic boundaries are unsupported.")
        if not isinstance(bc, MixedBoundary1D):
            raise NotImplementedError("Binary Illingworth currently supports only MixedBoundary1D homogeneous zero-flux external boundaries.")

        expected_shape = (self.mesh.numResponses,)
        for attr in ("LBCtype", "RBCtype", "LBCvalue", "RBCvalue"):
            values = np.asarray(getattr(bc, attr, None))
            if values.shape != expected_shape:
                raise NotImplementedError("Boundary-condition arrays must match the independent-component count for binary Illingworth.")

        left_type = np.asarray(bc.LBCtype)
        right_type = np.asarray(bc.RBCtype)
        if not np.all(left_type == MixedBoundary1D.NEUMANN) or not np.all(right_type == MixedBoundary1D.NEUMANN):
            raise NotImplementedError("Binary Illingworth currently supports only Neumann zero-flux external boundaries; fixed-composition or mixed boundaries are unsupported.")

        left_value = np.asarray(bc.LBCvalue, dtype=np.float64)
        right_value = np.asarray(bc.RBCvalue, dtype=np.float64)
        if not np.all(np.isfinite(left_value)) or not np.all(np.isfinite(right_value)):
            raise NotImplementedError("Binary Illingworth boundary flux values must be finite zero constants.")
        if not np.all(left_value == 0.0) or not np.all(right_value == 0.0):
            raise NotImplementedError("Binary Illingworth currently supports only homogeneous zero-flux external boundaries; nonzero fluxes are unsupported.")

    def _clipInterfacePosition(self, interface_position: float, strict: bool = True) -> float:
        z = flatten_1d_coordinates(self.mesh.z)
        eps = max(float(z[-1] - z[0]) * 1e-14, 1e-14)
        lower = float(z[0] + eps)
        upper = float(z[-1] - eps)
        if strict and not (lower < interface_position < upper):
            raise ValueError("Interface position must lie strictly inside the FD domain.")
        return float(np.clip(interface_position, lower, upper))

    def _getBoundaryConditions(self):
        bc = getattr(self.mesh, "boundaryConditions", None)
        if bc is None:
            bc = MixedBoundary1D(self.mesh.responses)
            self.mesh.boundaryConditions = bc
        return bc

    def reset(self):
        super().reset()
        if has_interstitials(self.allElements):
            record_initial_u_fractions(self)
        self.interfaceData.reset()
        self.interfaceData.record(0, self.initialInterfacePosition)
        self.concData.reset()
        self.pData = None
        self.qData = None
        self._currdt = np.inf
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
        self._lastImplicitIterations = 0
        self._lastImplicitError = np.nan
        self._lastImplicitResidual = np.nan
        self._lastImplicitResidualScale = np.nan
        self._lastStepRetries = 0
        self._lastLeftPicardIterations = 0
        self._lastRightPicardIterations = 0
        self._nearFinalNoop = False
        self._outerMinTimeStep = 0.0
        self._outerMaxTimeStep = np.inf
        self._solveMinDtFrac = 1e-8
        self._solveMaxDtFrac = 1.0
        self._pendingCandidate = None
        self._awaitingFreshCandidate = False
        self._initialInventory = None
        self._z = None
        self._R = None
        self._u_grid = None
        self._v_grid = None
        self._p_curr = None
        self._q_curr = None
        self._s_curr = None
        self._s_old = None
        self._D_left = None
        self._D_right = None
        self._phaseUniformCache = None
        if hasattr(self, "mesh") and self.mesh is not None:
            self._validateModelConfiguration()

    def setup(self):
        """
        Initializes the transformed two-phase state once; later calls are no-ops.

        ``solve`` calls ``setup`` on every call. Setup builds the transformed
        state by interpolating the stored physical profile and then replaces
        that profile with its reconstruction, and it records the reference
        inventory and starts the p/q histories. Re-running it would therefore
        re-interpolate an already-reconstructed profile (changing the state),
        restart the p/q histories and reset the reference inventory on every
        continued ``solve``. Later calls are no-ops until ``reset`` clears
        ``isSetup``; call ``reset`` after changing settings that setup consumes.
        """
        if self.isSetup:
            return
        super().setup()
        self._validateModelConfiguration()
        self._z = flatten_1d_coordinates(self.mesh.z).astype(np.float64)
        if not np.isclose(self._z[0], 0.0):
            raise ValueError("MovingBoundaryIllingworthFD1DModel expects a 1D domain starting at 0.")
        self._R = float(self._z[-1] - self._z[0])

        c0 = np.asarray(self.data.currentY, dtype=np.float64).reshape(-1)
        s0 = float(self.interfaceData.currentY)
        n_left = self.phaseANodes if self.phaseANodes is not None else max(3, int(np.searchsorted(self._z, s0, side="right")))
        n_right = self.phaseBNodes if self.phaseBNodes is not None else max(3, len(self._z) - int(np.searchsorted(self._z, s0, side="left")))

        self._u_grid = (
            self._inputUGrid.copy()
            if self._inputUGrid is not None
            else np.linspace(0.0, 1.0, int(n_left), dtype=np.float64)
        )
        self._v_grid = (
            self._inputVGrid.copy()
            if self._inputVGrid is not None
            else np.linspace(0.0, 1.0, int(n_right), dtype=np.float64)
        )
        n_left = len(self._u_grid)
        n_right = len(self._v_grid)
        if self.recordPqData:
            self.pData = _VectorHistory(int(n_left), self.interfaceData.recordInterval)
            self.qData = _VectorHistory(int(n_right), self.interfaceData.recordInterval)
        self._p_curr, self._q_curr = self._initialize_transformed_state(c0, s0)
        self._s_curr = s0
        self._s_old = s0
        self._phaseUniformCache = None
        D_left, D_right = self._step_diffusivities(self._p_curr, self._q_curr, s0, 0.0)
        if self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_PHASE_UNIFORM and self._temperature_is_constant():
            self._phaseUniformCache = (D_left, D_right)
        self._D_left = self._left_interface_D(D_left)
        self._D_right = self._right_interface_D(D_right)
        if self.recordPqData:
            self.pData.record(0, self._p_curr)
            self.qData.record(0, self._q_curr)

        physical = self._reconstruct_physical_profile(self._p_curr, self._q_curr, s0)[:, np.newaxis]
        self.data.currentY = physical
        self._initialInventory = self.getTotalInventoryFromState(self._p_curr, self._q_curr, self._s_curr)
        self.concData.record(0, self.checkMassIntegral(self._p_curr, self._q_curr, self._s_curr))

    def setTimeInfo(self, currTime, simTime):
        """
        Stores solve-time bounds and prepares optional semi-log target times.

        In ``dt_mode='semi_log'``, the implicit Illingworth recurrence still
        advances one complete step at a time; only the target output/step times
        are nonuniform. ``semiLogT0`` is the first relative target time and
        ``semiLog_dt`` is the spacing in natural-log time.

        Also stores the outer wrapper's minimum and maximum timestep for this
        solve (from the ``minDtFrac``/``maxDtFrac`` passed to ``solve``) and
        drops any uncommitted step candidate.
        """
        super().setTimeInfo(currTime, simTime)
        self._currdt = np.inf
        self._nearFinalNoop = False
        self._outerMinTimeStep, self._outerMaxTimeStep = _outer_timestep_limits(
            self._solveMinDtFrac, self._solveMaxDtFrac, self.initialTime, self.finalTime
        )
        self._pendingCandidate = None
        if self.dtMode != "semi_log" or simTime <= 0:
            self._semiLogTimes = None
            self._semiLogNextIndex = 0
            self._preallocateRecordingHistories(simTime)
            return

        t0_rel = max(float(self.semiLogT0), 1e-15)
        sim_time = float(simTime)
        if sim_time <= t0_rel:
            rel_times = np.asarray([sim_time], dtype=np.float64)
        else:
            rel_times = _loge_arange(t0_rel, sim_time, float(self.semiLog_dt))
            rel_times = rel_times[(rel_times > 0.0) & (rel_times < sim_time)]
            rel_times = np.append(rel_times, sim_time)
        self._semiLogTimes = float(currTime) + np.asarray(rel_times, dtype=np.float64)
        self._semiLogNextIndex = 0
        self._preallocateRecordingHistories(simTime)

    def _estimateStepCount(self, simTime):
        """Returns the number of implicit steps expected for the current schedule."""
        sim_time = float(simTime)
        if sim_time <= 0:
            return 0
        if self.dtMode == "semi_log" and self._semiLogTimes is not None:
            return int(len(self._semiLogTimes))
        return int(np.ceil(sim_time / self.timeStep))

    def _recordCapacityForSteps(self, record_interval, step_count):
        """
        Estimates rows needed by the history record/finalize convention.

        For a solve started after ``reset`` this bounds the rows stored: the
        initial row, one regular row every ``record_interval`` steps and the
        appended forced final row (which uses no cadence slot); the extra rows
        also cover a roundoff-driven near-final no-op step. Continued solves may
        need more, which the normal ``record`` growth path supplies; capacity
        never changes the stored values.
        """
        if record_interval <= 0:
            return 1
        return int(np.ceil((int(step_count) + 2) / int(record_interval))) + 2

    def _preallocateRecordingHistories(self, simTime):
        """Preallocates enabled history arrays when the run length is predictable."""
        if not self.preallocateRecordings:
            return
        step_count = self._estimateStepCount(simTime)
        histories = [self.data, self.interfaceData, self.concData]
        if self.recordPqData:
            histories.extend([self.pData, self.qData])
        for history in histories:
            if history is None or not hasattr(history, "preallocate"):
                continue
            capacity = self._recordCapacityForSteps(history.recordInterval, step_count)
            history.preallocate(capacity)

    def _initialize_transformed_state(self, composition, interface_position):
        c = np.asarray(composition, dtype=np.float64).reshape(-1)
        c_left_int, c_right_int = self.interfaceCompositions

        z_left = interface_position * self._u_grid
        z_right = interface_position + (self._R - interface_position) * self._v_grid
        left_mask = self._z <= interface_position
        right_mask = self._z >= interface_position

        if not np.any(left_mask) or not np.any(right_mask):
            raise ValueError("Initial interface leaves an empty phase.")

        p = np.interp(z_left, self._z[left_mask], c[left_mask])
        q = np.interp(z_right, self._z[right_mask], c[right_mask])
        p[-1] = c_left_int
        q[0] = c_right_int
        return p.astype(np.float64), q.astype(np.float64)

    def _temperatures_at_positions(self, positions, time):
        """Returns finite temperatures at physical positions, broadcasting a scalar field."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1)
        T = np.asarray(
            self.temperatureParameters(positions.reshape(-1, 1), float(time)),
            dtype=np.float64,
        ).reshape(-1)
        if T.size == 1 and positions.size != 1:
            T = np.full(positions.shape, float(T[0]), dtype=np.float64)
        if T.size != positions.size or not np.all(np.isfinite(T)):
            raise ValueError("Illingworth temperatures must be finite and match the queried positions.")
        return T

    def _phase_diffusivity(self, phase, c_interface, time, position):
        """
        Returns the scalar ``'phase_uniform'`` diffusivity of ``phase``.

        The diffusivity is evaluated at the phase's fixed interface composition
        and at the temperature of the interface position, and is reused for
        every face of that phase's bulk solve. Providers that accept
        ``query_context`` are asked for the ``'interface'`` context.
        ``c_interface`` is a u-fraction; the provider receives the mole fraction.
        """
        temperature = float(self._temperatures_at_positions([position], time)[0])
        source = self._diffusivity_source()
        try:
            D = source.getInterdiffusivity(float(c_interface), temperature, phase=phase, query_context="interface")
        except TypeError:
            D = source.getInterdiffusivity(float(c_interface), temperature, phase=phase)
        D = np.asarray(D, dtype=np.float64).reshape(-1)
        if D.size != 1:
            raise ValueError("Phase-uniform diffusivity query must return a single value.")
        return float(_validate_binary_diffusivity(D, phase, context="phase-uniform diffusivity")[0])

    def _bulk_face_diffusivities(self, face_compositions, phase, time, face_positions):
        """
        Returns validated diffusivities, shape ``(n_faces,)``, at face compositions.

        The provider is queried vectorized with ``query_context='general'``,
        then vectorized without ``query_context`` (for ``Thermodynamics``-style
        signatures), and finally one face at a time if the vectorized result
        is unusable. ``face_compositions`` are u-fractions; the provider
        receives mole fractions.
        """
        source = self._diffusivity_source()
        face_compositions = np.asarray(face_compositions, dtype=np.float64).reshape(-1)
        temperatures = self._temperatures_at_positions(face_positions, time)
        if temperatures.size != face_compositions.size:
            raise ValueError("face_positions must match the face composition count.")
        n_faces = face_compositions.size

        def vectorized_call(**kwargs):
            values = source.getInterdiffusivity(face_compositions, temperatures, phase=phase, **kwargs)
            values = np.asarray(values, dtype=np.float64).reshape(-1)
            if values.size == 1 and n_faces != 1:
                values = np.full(n_faces, float(values[0]), dtype=np.float64)
            if values.size != n_faces:
                raise ValueError("bulk diffusivity query returned an unexpected shape.")
            return values

        values = None
        for kwargs in ({"query_context": "general"}, {}):
            try:
                values = vectorized_call(**kwargs)
                break
            except TypeError:
                continue
            except ValueError:
                break
        if values is None:
            values = np.empty(n_faces, dtype=np.float64)
            for i, (composition, temperature) in enumerate(zip(face_compositions, temperatures)):
                try:
                    D = source.getInterdiffusivity(float(composition), float(temperature), phase=phase, query_context="general")
                except TypeError:
                    D = source.getInterdiffusivity(float(composition), float(temperature), phase=phase)
                values[i] = float(np.asarray(D, dtype=np.float64).reshape(-1)[0])
        return _validate_binary_diffusivity(values, phase, context="bulk face diffusivity")

    def _left_face_compositions(self, p):
        """Face ``j`` composition ``0.5*(p[j] + p[j+1])``; the interface face is the last."""
        p = np.asarray(p, dtype=np.float64)
        return 0.5 * (p[:-1] + p[1:])

    def _right_face_compositions(self, q):
        """Face ``j`` composition ``0.5*(q[j] + q[j+1])``; the interface face is the first."""
        q = np.asarray(q, dtype=np.float64)
        return 0.5 * (q[:-1] + q[1:])

    def _left_face_positions(self, future_s):
        u_faces = 0.5 * (self._u_grid[:-1] + self._u_grid[1:])
        return float(future_s) * u_faces

    def _right_face_positions(self, future_s):
        v_faces = 0.5 * (self._v_grid[:-1] + self._v_grid[1:])
        return float(future_s) + (self._R - float(future_s)) * v_faces

    def _left_face_diffusivities(self, p, future_s, time):
        """Returns left-phase face diffusivities evaluated from profile ``p``."""
        return self._bulk_face_diffusivities(
            self._left_face_compositions(p), self.phases[0], time, self._left_face_positions(future_s)
        )

    def _right_face_diffusivities(self, q, future_s, time):
        """Returns right-phase face diffusivities evaluated from profile ``q``."""
        return self._bulk_face_diffusivities(
            self._right_face_compositions(q), self.phases[1], time, self._right_face_positions(future_s)
        )

    def _temperature_is_constant(self):
        """
        Returns whether the temperature field is demonstrably constant in space and time.

        True for an isothermal scalar temperature and for a temperature array
        whose values are all identical; False for callables, varying arrays, or
        unrecognized specifications. Used only to decide whether the
        ``'phase_uniform'`` diffusivities can be cached: with fixed interface
        compositions and constant temperature they cannot change during a run.
        """
        params = getattr(self.temperatureParameters, "Tparameters", None)
        if params is None or callable(params):
            return False
        if isinstance(params, tuple):
            if len(params) != 2:
                return False
            values = np.asarray(params[1], dtype=np.float64).reshape(-1)
            return values.size > 0 and bool(np.all(np.isfinite(values))) and bool(np.all(values == values[0]))
        try:
            values = np.asarray(params, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return False
        return values.size == 1 and bool(np.isfinite(values[0]))

    def _step_diffusivities(self, p, q, s, time):
        """
        Returns the per-step left/right diffusivity inputs for the bulk solves.

        ``'phase_uniform'`` returns two scalars, taken from the cache built in
        ``setup()`` when the temperature is constant (see
        ``_temperature_is_constant``). ``'composition_dependent_lagged'``
        returns face arrays from the accepted old-time profiles, frozen for the
        whole step (face temperatures are sampled at the old interface
        position). ``'composition_dependent_implicit'`` also returns these lagged
        arrays; they only seed the first interface update, while the bulk solves
        evaluate their own Picard face values.
        """
        if self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_PHASE_UNIFORM:
            if self._phaseUniformCache is not None:
                return self._phaseUniformCache
            c_left, c_right = self.interfaceCompositions
            return (
                self._phase_diffusivity(self.phases[0], c_left, time, s),
                self._phase_diffusivity(self.phases[1], c_right, time, s),
            )
        return self._left_face_diffusivities(p, s, time), self._right_face_diffusivities(q, s, time)

    @staticmethod
    def _left_interface_D(D_left):
        """Returns the left interface-face diffusivity (last face) from a scalar or face array."""
        D_left = np.asarray(D_left, dtype=np.float64)
        return float(D_left) if D_left.ndim == 0 else float(D_left[-1])

    @staticmethod
    def _right_interface_D(D_right):
        """Returns the right interface-face diffusivity (first face) from a scalar or face array."""
        D_right = np.asarray(D_right, dtype=np.float64)
        return float(D_right) if D_right.ndim == 0 else float(D_right[0])

    def solve(self, simTime, iterator=explicitEulerIterator, verbose=False, vIt=10, minDtFrac=1e-8, maxDtFrac=1):
        """
        Solves the implicit recurrence using a single-stage Euler wrapper.

        Multi-stage iterators are rejected because the implicit Illingworth step
        is not a continuously valid right-hand side for RK intermediate states.

        Each step is solved at ``min(schedule, remaining time, maxDtFrac *
        simTime)``, so the wrapper never shortens a solved step. A retry-reduced
        step below ``minDtFrac * simTime`` raises ``ValueError`` (pass a smaller
        ``minDtFrac`` when ``max_step_retries > 1``); only the exact final
        remainder may be shorter. ``postProcess`` refuses to commit an outer
        update that does not reproduce the converged candidate.

        ``simTime`` must be finite and nonnegative and is checked before setup;
        ``simTime=0`` runs setup and takes no step.
        """
        if iterator is not explicitEulerIterator:
            raise ValueError("MovingBoundaryIllingworthFD1DModel supports only explicitEulerIterator.")
        if not np.isfinite(simTime) or simTime < 0.0:
            raise ValueError("simTime must be finite and nonnegative.")
        _validate_outer_dt_fractions(minDtFrac, maxDtFrac, "MovingBoundaryIllingworthFD1DModel")
        self._solveMinDtFrac = float(minDtFrac)
        self._solveMaxDtFrac = float(maxDtFrac)
        return super().solve(simTime, iterator=iterator, verbose=verbose, vIt=vIt, minDtFrac=minDtFrac, maxDtFrac=maxDtFrac)

    def getCurrentX(self):
        return [self._p_curr.copy(), self._q_curr.copy(), float(self._s_curr)]

    def flattenX(self, X):
        return np.concatenate((np.asarray(X[0], dtype=np.float64), np.asarray(X[1], dtype=np.float64), [float(X[2])]))

    def unflattenX(self, X_flat, X_ref):
        n_p = len(np.asarray(X_ref[0], dtype=np.float64))
        n_q = len(np.asarray(X_ref[1], dtype=np.float64))
        p = np.asarray(X_flat[:n_p], dtype=np.float64)
        q = np.asarray(X_flat[n_p : n_p + n_q], dtype=np.float64)
        s = float(X_flat[n_p + n_q])
        return [p, q, s]

    def _compute_dt(self, t):
        """
        Returns the trial dt: the schedule, limited by the remaining time and the outer maximum.

        Including the outer maximum here means the implicit step is solved at
        the dt the wrapper will commit (LEDGERS R07). The near-final no-op
        threshold still uses the unclamped schedule step.
        """
        remaining = getattr(self, "finalTime", np.inf) - float(t)
        step_scale = self.timeStep
        if self.dtMode == "semi_log":
            scheduled_dt = self._computeSemiLogDt(t)
            if np.isfinite(scheduled_dt) and scheduled_dt > 0:
                step_scale = scheduled_dt
            dt = min(scheduled_dt, remaining, self._outerMaxTimeStep)
        else:
            dt = min(self.timeStep, remaining, self._outerMaxTimeStep)
        self._nearFinalNoop = bool(np.isfinite(remaining) and 0 < remaining <= max(step_scale, 1e-15) * 1e-10)
        if self._nearFinalNoop:
            self._currdt = max(step_scale, 1e-15)
            return self._currdt
        if not np.isfinite(dt) or dt <= 0:
            dt = self.timeStep
        self._currdt = float(dt)
        return float(dt)

    def _updateSemiLogIndex(self, t):
        if self._semiLogTimes is None:
            return
        while self._semiLogNextIndex < len(self._semiLogTimes):
            if self._semiLogTimes[self._semiLogNextIndex] > float(t) + 1e-15:
                break
            self._semiLogNextIndex += 1

    def _computeSemiLogDt(self, t):
        """Returns the step needed to reach the next semi-log target time."""
        if self.dtMode != "semi_log" or self._semiLogTimes is None:
            return np.inf
        self._updateSemiLogIndex(t)
        if self._semiLogNextIndex >= len(self._semiLogTimes):
            return np.inf
        return max(1e-15, float(self._semiLogTimes[self._semiLogNextIndex] - float(t)))

    def _new_interface_planar(self, left_deviation, right_deviation, s, old_s, future_s, dt, pass_number, D_left_int, D_right_int):
        """
        Returns the next fixed-point estimate of the future interface position.

        This is the MAP/paper planar conservative interface balance, written in
        terms of the interface-adjacent deviations ``left_deviation = p[-2] -
        c_left`` and ``right_deviation = q[1] - c_right`` (see
        ``_BinaryBulkSolveResult.interface_deviation``). Algebraically it is the
        MAP expression; using the solved deviations avoids the cancellation in
        ``c_left - p[-2]``, which ``D dt / dz`` amplifies into round-off noise
        in the interface position (ledger R06).
        ``D_left_int`` and ``D_right_int`` are the scalar diffusivities on the
        interface-adjacent faces (left face ``-1``, right face ``0``) used by
        the most recent bulk solves, so the interface flux matches the bulk
        flux through those faces. On pass zero the velocity sign comes from the
        previous step, and afterwards from the current estimate.
        """
        if pass_number == 0:
            velocity_sign_probe = s - old_s
        else:
            velocity_sign_probe = future_s - s

        lhs, rhs = self._interface_balance_terms(
            left_deviation, right_deviation, future_s, dt, D_left_int, D_right_int, velocity_sign_probe
        )
        if abs(lhs) <= 1e-300:
            raise ZeroDivisionError("Illingworth interface update encountered a zero denominator.")
        return float(s + rhs / lhs)

    def _interface_balance_terms(self, left_deviation, right_deviation, future_s, dt, D_left_int, D_right_int, velocity_sign_probe):
        """
        Returns ``(lhs, rhs)`` of the planar conservative interface balance ``lhs * (s_next - s) = rhs``.

        ``rhs`` is ``dt`` times the net diffusive flux into the interface cells,
        formed from the interface-adjacent deviations and interface-face
        diffusivities at the trial position ``future_s``; ``lhs`` is the
        tie-line jump corrected by the upwind interface-cell deviation of the
        branch selected by ``velocity_sign_probe >= 0``. Both the interface
        update and the R08 residual use this one function, so they share the
        same operations in the same order.
        """
        c_left_int, c_right_int = self.interfaceCompositions
        diff_l = -left_deviation / (1.0 - self._u_grid[-2])
        diff_l = diff_l * D_left_int / future_s
        diff_r = right_deviation / self._v_grid[1]
        diff_r = diff_r * D_right_int / (self._R - future_s)
        rhs = (diff_r - diff_l) * dt

        if velocity_sign_probe >= 0:
            lhs = (c_left_int - c_right_int) - right_deviation * (1.0 - self._v_grid[1] / 2.0)
        else:
            lhs = (c_left_int - c_right_int) + left_deviation * (0.5 + self._u_grid[-2] / 2.0)
        return lhs, rhs

    def _interface_residual(self, left_deviation, right_deviation, s, future_s, dt, D_left_int, D_right_int):
        """
        Returns the conservative interface residual ``lhs * (future_s - s) - rhs`` at a trial position (ledger R08).

        The deviations and interface-face diffusivities must come from the bulk
        solves at ``future_s``, and the motion branch is that of the candidate's
        own displacement ``future_s - s`` (not the pass-zero probe). Forming it
        from the displacement and the deviation fluxes, never as
        ``future_s - s_next``, keeps R06's cancellation-resistant arithmetic.
        The residual equals the step's change of the transformed inventory up
        to round-off.
        """
        displacement = future_s - s
        lhs, rhs = self._interface_balance_terms(
            left_deviation, right_deviation, future_s, dt, D_left_int, D_right_int, displacement
        )
        return float(lhs * displacement - rhs)

    def _interface_residual_scale(self, p, q, s):
        """
        Returns the residual normalization ``max(|I_old|, R * c_floor)`` for one step (ledger R08).

        ``I_old`` is the transformed inventory of the accepted old state, with
        the same trapezoid quadrature as the conservation check. ``c_floor`` is
        ``interface_residual_composition_scale`` when given, otherwise
        ``constraints.minComposition`` read now, so it is resolved once per
        step. See ``_binary_residual_scale`` for the zero-inventory safeguard
        and the ``ValueError`` cases.
        """
        inventory = float(self.getTotalInventoryFromState(p, q, s))
        if self.interfaceResidualCompositionScale is not None:
            floor = float(self.interfaceResidualCompositionScale)
        else:
            floor = float(self.constraints.minComposition)
        return _binary_residual_scale(inventory, self._R, floor)

    def _new_concentration_left_planar(self, p, s, future_s, dt, D_left, return_deviation=False):
        """
        Solves the left transformed concentration profile for one implicit step.

        Coefficients are the planar Illingworth finite-volume coefficients from
        the MAP implementation. Interior rows are assembled with vectorized
        slices, but the boundary rows retain the original one-sided forms.

        The system is solved for the deviation ``w = p_new - c_left`` from the
        interface composition. Every non-Dirichlet row sums to its old-volume
        coefficient (``-s * width / 2``), so the deviation system has the same
        matrix and the same right-hand side with ``p - c_left`` in place of
        ``p`` and a zero Dirichlet value: an exact identity, assembled directly
        rather than as ``rhs - A c_left``. Near-interface deviations then carry
        relative rather than absolute precision; the discretization is
        unchanged but the results differ from the MAP absolute-composition
        solve by amplified round-off (ledger R06). Returns ``w + c_left``, or
        ``(w + c_left, w)`` when ``return_deviation`` is true.

        ``D_left`` is either a scalar phase-uniform diffusivity, assembled with
        the original MAP expressions, or a face array of shape
        ``(len(p) - 1,)`` where face ``j`` lies between nodes ``j`` and
        ``j + 1`` and the interface face is ``D_left[-1]``. The upwind and
        cell-volume terms do not depend on D.
        """
        c_left = self.interfaceCompositions[0]
        w_old = np.asarray(p, dtype=np.float64) - c_left
        n = len(p)
        lo = np.zeros(n, dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)
        D_values = np.asarray(D_left, dtype=np.float64)
        phase_uniform = D_values.ndim == 0
        if not phase_uniform and D_values.shape != (n - 1,):
            raise ValueError(f"Left face diffusivities must have shape ({n - 1},); received {D_values.shape}.")
        tmpB = future_s - s
        u = self._u_grid
        left_diff = u[1:-1] - u[:-2]
        left_sum = u[1:-1] + u[:-2]
        right_diff = u[2:] - u[1:-1]
        right_sum = u[2:] + u[1:-1]
        cell_width = right_sum - left_sum
        if phase_uniform:
            tmpA = float(D_values) * dt / future_s
            A_first = tmpA
            A_lo = tmpA / left_diff
            A_up = tmpA / right_diff
            A_diag = -tmpA * (1.0 / left_diff + 1.0 / right_diff)
        else:
            A_faces = D_values * dt / future_s
            A_first = A_faces[0]
            A_lo = A_faces[:-1] / left_diff
            A_up = A_faces[1:] / right_diff
            A_diag = -(A_lo + A_up)

        if future_s >= s:
            diag[0] = -A_first / u[1] - future_s * u[1] / 2.0
            up[0] = A_first / u[1] + tmpB * u[1] / 2.0
            rhs[0] = -w_old[0] * s * u[1] / 2.0
            lo[1:-1] = A_lo
            diag[1:-1] = A_diag - tmpB * left_sum / 2.0
            diag[1:-1] = diag[1:-1] - future_s * cell_width / 2.0
            up[1:-1] = A_up + tmpB * right_sum / 2.0
            rhs[1:-1] = -s * w_old[1:-1] * cell_width / 2.0
        else:
            diag[0] = -A_first / u[1] + tmpB * u[1] / 2.0 - future_s * u[1] / 2.0
            up[0] = A_first / u[1]
            rhs[0] = -w_old[0] * s * u[1] / 2.0
            lo[1:-1] = A_lo - tmpB * left_sum / 2.0
            diag[1:-1] = A_diag + tmpB * right_sum / 2.0
            diag[1:-1] = diag[1:-1] - future_s * cell_width / 2.0
            up[1:-1] = A_up
            rhs[1:-1] = -s * w_old[1:-1] * cell_width / 2.0

        diag[-1] = -1.0
        rhs[-1] = 0.0
        w = solve_illingworth_tridiagonal(lo, diag, up, rhs)
        profile = w + c_left
        return (profile, w) if return_deviation else profile

    def _new_concentration_right_planar(self, q, s, future_s, dt, D_right, return_deviation=False):
        """
        Solves the right transformed concentration profile for one implicit step.

        This is the large phase for the Figure-3 setup, so interior coefficient
        assembly is vectorized while preserving the MAP code's boundary rows and
        tridiagonal sign convention.

        As in ``_new_concentration_left_planar`` the system is solved for the
        deviation ``w = q_new - c_right``: every non-Dirichlet row (including the
        zero-flux far-boundary row) sums to its old-volume coefficient, so the
        right-hand side uses ``q - c_right`` and the Dirichlet value is zero.
        Returns ``w + c_right``, or ``(w + c_right, w)`` when
        ``return_deviation`` is true.

        ``D_right`` is either a scalar phase-uniform diffusivity (original MAP
        expressions) or a face array of shape ``(len(q) - 1,)`` where the
        interface face is ``D_right[0]`` and the zero-flux far-boundary row uses
        ``D_right[-1]``.
        """
        c_right = self.interfaceCompositions[1]
        w_old = np.asarray(q, dtype=np.float64) - c_right
        n = len(q)
        lo = np.zeros(n, dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)
        D_values = np.asarray(D_right, dtype=np.float64)
        phase_uniform = D_values.ndim == 0
        if not phase_uniform and D_values.shape != (n - 1,):
            raise ValueError(f"Right face diffusivities must have shape ({n - 1},); received {D_values.shape}.")
        tmpB = future_s - s
        span = self._R - future_s
        v = self._v_grid
        left_diff = v[1:-1] - v[:-2]
        left_sum = v[1:-1] + v[:-2]
        right_diff = v[2:] - v[1:-1]
        right_sum = v[2:] + v[1:-1]
        cell_width = right_sum - left_sum
        if phase_uniform:
            tmpA = float(D_values) * dt / (self._R - future_s)
            A_last = tmpA
            A_lo = tmpA / left_diff
            A_up = tmpA / right_diff
            A_diag = -tmpA * (1.0 / right_diff + 1.0 / left_diff)
        else:
            A_faces = D_values * dt / (self._R - future_s)
            A_last = A_faces[-1]
            A_lo = A_faces[:-1] / left_diff
            A_up = A_faces[1:] / right_diff
            A_diag = -(A_up + A_lo)

        diag[0] = -1.0
        rhs[0] = 0.0

        if future_s >= s:
            lo[1:-1] = A_lo
            diag[1:-1] = A_diag - tmpB * (1.0 - left_sum / 2.0)
            diag[1:-1] = diag[1:-1] - span * cell_width / 2.0
            up[1:-1] = A_up + tmpB * (1.0 - right_sum / 2.0)
            rhs[1:-1] = -(self._R - s) * w_old[1:-1] * cell_width / 2.0

            tmp = v[-2]
            lo[-1] = A_last / (1.0 - tmp)
            diag[-1] = -A_last / (1.0 - tmp) - tmpB * (1.0 - (1.0 + tmp) / 2.0)
            diag[-1] = diag[-1] - span * (1.0 - tmp) / 2.0
            rhs[-1] = -w_old[-1] * (self._R - s) * (1.0 - tmp) / 2.0
        else:
            lo[1:-1] = A_lo - tmpB * (1.0 - left_sum / 2.0)
            diag[1:-1] = A_diag + tmpB * (1.0 - right_sum / 2.0)
            diag[1:-1] = diag[1:-1] - span * cell_width / 2.0
            up[1:-1] = A_up
            rhs[1:-1] = -(self._R - s) * w_old[1:-1] * cell_width / 2.0

            tmp = v[-2]
            lo[-1] = A_last / (1.0 - tmp) - tmpB * (1.0 - (1.0 + tmp) / 2.0)
            diag[-1] = -A_last / (1.0 - tmp) - span * (1.0 - tmp) / 2.0
            rhs[-1] = -w_old[-1] * (self._R - s) * (1.0 - tmp) / 2.0

        w = solve_illingworth_tridiagonal(lo, diag, up, rhs)
        profile = w + c_right
        return (profile, w) if return_deviation else profile

    def _bulk_picard_relative_tolerance(self):
        """
        Returns the effective relative tolerance for implicit bulk Picard solves.

        ``bulk_picard_rtol`` when given, otherwise the fixed default ``1e-10``.
        Since R08 it does not follow the (now dimensionless) interface
        ``tolerance``.
        """
        return float(_DEFAULT_BULK_PICARD_RTOL if self.bulkPicardRtol is None else self.bulkPicardRtol)

    def _bulk_picard_update_has_converged(self, update_norm, profile):
        """
        Tests Picard convergence against ``atol + rtol * max|profile|``.

        The update norm is the maximum absolute change from the current
        coefficient iterate to the unrelaxed linear solve; relaxation only forms
        the next coefficient iterate.
        """
        profile_scale = max(1.0e-12, float(np.max(np.abs(np.asarray(profile, dtype=np.float64)))))
        threshold = self.bulkPicardAtol + self._bulk_picard_relative_tolerance() * profile_scale
        return float(update_norm) <= threshold

    def _solve_bulk_picard(self, old_profile, s, future_s, dt, phase_side):
        """
        Solves one phase with composition-dependent face diffusivities by Picard iteration.

        The iterate is seeded from the accepted old profile with the fixed
        interface Dirichlet value imposed. Each iteration evaluates face
        diffusivities from face averages of the iterate (temperatures at the
        trial face positions for ``future_s``), performs the linear tridiagonal
        solve, and checks the update norm. With ``bulk_picard_relaxation == 1``
        the loop also stops when the re-evaluated face values are unchanged.
        Diffusivities and the update norm use physical (absolute) compositions;
        only the linear solve works in deviation form, and the returned result
        carries the final solve's interface-adjacent deviation.
        Non-convergence raises ``RuntimeError`` so the step-retry logic can
        reduce ``dt``.
        """
        if phase_side == "left":
            deviation_index = -2
            interface_index = -1
            c_interface = self.interfaceCompositions[0]
            face_diffusivities = self._left_face_diffusivities
            linear_solve = self._new_concentration_left_planar
            interface_D = self._left_interface_D
        else:
            deviation_index = 1
            interface_index = 0
            c_interface = self.interfaceCompositions[1]
            face_diffusivities = self._right_face_diffusivities
            linear_solve = self._new_concentration_right_planar
            interface_D = self._right_interface_D

        iterate = np.asarray(old_profile, dtype=np.float64).copy()
        iterate[interface_index] = c_interface
        update_norm = np.inf
        D_faces = face_diffusivities(iterate, future_s, self.currentTime)
        for iteration in range(1, self.bulkPicardMaxIterations + 1):
            profile, deviation = linear_solve(old_profile, s, future_s, dt, D_faces, return_deviation=True)
            update_norm = float(np.max(np.abs(profile - iterate)))
            if self._bulk_picard_update_has_converged(update_norm, profile):
                return _BinaryBulkSolveResult(profile, interface_D(D_faces), iteration, update_norm, float(deviation[deviation_index]))
            next_iterate = iterate + self.bulkPicardRelaxation * (profile - iterate)
            next_iterate[interface_index] = c_interface
            D_next = face_diffusivities(next_iterate, future_s, self.currentTime)
            if self.bulkPicardRelaxation == 1.0 and np.array_equal(D_next, D_faces):
                return _BinaryBulkSolveResult(profile, interface_D(D_faces), iteration, 0.0, float(deviation[deviation_index]))
            iterate = next_iterate
            D_faces = D_next

        raise RuntimeError(
            f"{phase_side} bulk Picard solve failed to converge after "
            f"{self.bulkPicardMaxIterations} iterations; final update norm was {update_norm:.3e}."
        )

    def _solve_bulk_phases(self, p, q, s, future_s, dt, D_left, D_right):
        """
        Solves both transformed phases for one interface trial.

        In ``'phase_uniform'`` and ``'composition_dependent_lagged'`` modes the
        supplied per-step ``D_left`` and ``D_right`` (scalars or frozen face
        arrays) are used directly; in ``'composition_dependent_implicit'`` mode
        each phase runs its own Picard iteration and they are ignored.
        """
        if self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_IMPLICIT:
            return (
                self._solve_bulk_picard(p, s, future_s, dt, "left"),
                self._solve_bulk_picard(q, s, future_s, dt, "right"),
            )
        p_future, w_left = self._new_concentration_left_planar(p, s, future_s, dt, D_left, return_deviation=True)
        q_future, w_right = self._new_concentration_right_planar(q, s, future_s, dt, D_right, return_deviation=True)
        return (
            _BinaryBulkSolveResult(p_future, self._left_interface_D(D_left), interface_deviation=float(w_left[-2])),
            _BinaryBulkSolveResult(q_future, self._right_interface_D(D_right), interface_deviation=float(w_right[1])),
        )

    def _take_implicit_step_planar(self, p, q, s, old_s, dt):
        """
        Performs one implicit Illingworth step by fixed-point iteration on ``s``.

        Each pass updates the interface from the latest profiles and
        interface-face diffusivities, then re-solves both phases at that trial
        position. Convergence requires at least two passes (the pass-zero
        iterate is never accepted), ``|delta s| <= tolerance * R`` and
        ``|r| <= residual_tolerance * scale`` (ledger R08), where ``r`` is the
        conservative interface residual at the trial position (see
        ``_interface_residual``) and ``scale`` is fixed once per step from the
        old state (see ``_interface_residual_scale``). On acceptance
        ``_lastImplicitError`` holds ``|delta s|`` (a length),
        ``_lastImplicitResidual`` the residual and
        ``_lastImplicitResidualScale`` the scale. Per-step diffusivities are evaluated once at
        the old state (see ``_step_diffusivities``); in implicit mode they only
        provide the pass-zero interface-face values. Pass zero takes the
        interface-adjacent deviations from the old profiles; later passes use
        the deviations returned by the bulk solves at the current trial
        position, so they always belong to the candidate being iterated.

        Returns ``(p_future, q_future, s_future, D_left_int, D_right_int)``.
        """
        p_future = np.asarray(p, dtype=np.float64).copy()
        q_future = np.asarray(q, dtype=np.float64).copy()
        c_left_int, c_right_int = self.interfaceCompositions
        left_deviation = float(p_future[-2] - c_left_int)
        right_deviation = float(q_future[1] - c_right_int)
        future_s = float(s)
        error = np.inf
        residual = np.nan
        residual_scale = self._interface_residual_scale(p, q, s)
        position_bound = self.tolerance * self._R
        residual_bound = self.residualTolerance * residual_scale
        D_left, D_right = self._step_diffusivities(p, q, s, self.currentTime)
        D_left_int = self._left_interface_D(D_left)
        D_right_int = self._right_interface_D(D_right)

        for count in range(self.maxIterations):
            previous_future_s = future_s
            future_s = self._new_interface_planar(left_deviation, right_deviation, s, old_s, future_s, dt, count, D_left_int, D_right_int)
            if not np.isfinite(future_s):
                raise ValueError("Illingworth interface update produced a non-finite position.")
            left_result, right_result = self._solve_bulk_phases(p, q, s, future_s, dt, D_left, D_right)
            p_future = left_result.profile
            q_future = right_result.profile
            left_deviation = left_result.interface_deviation
            right_deviation = right_result.interface_deviation
            D_left_int = left_result.interface_D
            D_right_int = right_result.interface_D
            error = abs(previous_future_s - future_s)
            residual = self._interface_residual(left_deviation, right_deviation, s, future_s, dt, D_left_int, D_right_int)
            if count >= 1 and error <= position_bound and abs(residual) <= residual_bound:
                self._lastImplicitIterations = count + 1
                self._lastImplicitError = float(error)
                self._lastImplicitResidual = float(residual)
                self._lastImplicitResidualScale = float(residual_scale)
                self._lastLeftPicardIterations = int(left_result.inner_iterations)
                self._lastRightPicardIterations = int(right_result.inner_iterations)
                return p_future, q_future, self._clipInterfacePosition(future_s, strict=True), D_left_int, D_right_int

        raise RuntimeError(
            "Illingworth implicit step failed to converge within "
            f"{self.maxIterations} iterations; final interface error was {error:.3e} "
            f"(bound {position_bound:.3e}) and final residual {residual:.3e} (bound {residual_bound:.3e})."
        )

    def _reconstruct_physical_profile(self, p, q, s):
        return reconstruct_planar_transformed_profile(
            z=self._z,
            p=p,
            q=q,
            s=s,
            domain_length=self._R,
            u=self._u_grid,
            v=self._v_grid,
        )

    def _try_step_retries(self, p, q, s, dt):
        """
        Attempts the implicit step, shrinking ``dt`` by ``retry_factor`` after each failure.

        Up to ``max_step_retries`` attempts are made. ``RuntimeError``,
        ``ValueError`` and ``ZeroDivisionError`` trigger a retry; the last one
        is re-raised once attempts are exhausted. On success ``_currdt`` is set
        to the accepted step so ``getDt`` advances time consistently.
        """
        trial_dt = float(dt)
        last_error = None
        for retry in range(self.maxStepRetries):
            try:
                p_new, q_new, s_new, D_left_int, D_right_int = self._take_implicit_step_planar(p, q, s, self._s_old, trial_dt)
            except (RuntimeError, ValueError, ZeroDivisionError) as exc:
                last_error = exc
                trial_dt *= self.retryFactor
                continue
            self._currdt = float(trial_dt)
            self._lastStepRetries = int(retry)
            self._D_left = D_left_int
            self._D_right = D_right_int
            return p_new, q_new, s_new, trial_dt
        raise last_error

    def getdXdt(self, t, xCurr):
        """
        Returns the implicit-step increment divided by the accepted ``dt``.

        The complete implicit step (with optional ``dt`` retries) is taken here,
        so ``getDt`` must be used to advance time by the accepted step. A
        converged step below the outer minimum raises ``ValueError`` unless it
        is the exact final remainder. The converged state is staged in
        ``_pendingCandidate`` for ``postProcess`` to verify.

        A proposal counts as rejected from its start until it succeeds
        (``_awaitingFreshCandidate``), so if this raises, ``postProcess``
        refuses every commit until a later call succeeds.
        """
        self._pendingCandidate = None
        self._awaitingFreshCandidate = True
        p = np.asarray(xCurr[0], dtype=np.float64).reshape(-1)
        q = np.asarray(xCurr[1], dtype=np.float64).reshape(-1)
        s = self._clipInterfacePosition(float(xCurr[2]), strict=True)
        p[-1] = self.interfaceCompositions[0]
        q[0] = self.interfaceCompositions[1]
        dt = self._compute_dt(t)
        if self._nearFinalNoop:
            self._awaitingFreshCandidate = False
            return [np.zeros_like(p), np.zeros_like(q), 0.0]
        p_new, q_new, s_new, dt = self._try_step_retries(p, q, s, dt)
        _check_trial_dt_against_outer_minimum(dt, getattr(self, "finalTime", np.inf) - float(t), self._outerMinTimeStep)
        self._stage_pending_candidate(t, dt, (p, q, s), (p_new, q_new, s_new))
        self._awaitingFreshCandidate = False
        return [(p_new - p) / dt, (q_new - q) / dt, (s_new - s) / dt]

    def _committed_state(self, x):
        """Returns ``(p, q, s)`` as ``postProcess`` commits them, without validating ``s``."""
        p = np.asarray(x[0], dtype=np.float64).reshape(-1).copy()
        q = np.asarray(x[1], dtype=np.float64).reshape(-1).copy()
        p[-1] = self.interfaceCompositions[0]
        q[0] = self.interfaceCompositions[1]
        return p, q, float(x[2])

    def _stage_pending_candidate(self, t, dt, old, new):
        """
        Stages the converged step for verification in ``postProcess``.

        Round-off allowances scale with the compositions (at least 1) and, for
        the interface, with the domain length, so the check is independent of
        the length unit.
        """
        p_old, q_old, s_old = old
        self._pendingCandidate = _PendingImplicitStep(
            start_time=float(t),
            dt=float(dt),
            state=self._committed_state(new),
            scales=(
                _magnitude(p_old, new[0]),
                _magnitude(q_old, new[1]),
                _magnitude(s_old, new[2], floor=self._R),
            ),
        )

    def _verify_pending_candidate(self, time, x):
        """
        Refuses an outer update that does not reproduce the converged candidate.

        The wrapper commits ``old + dXdt * dt``; if its ``time`` or state differ
        from the candidate solved in ``getdXdt`` (for example after an outer
        timestep clamp), the implicit equations were not solved for that state,
        so this raises ``RuntimeError`` before any accepted state or history
        changes. See ``_pending_step_mismatches`` for the tolerances.

        Rejection lifecycle: a mismatch, or an exception while checking,
        discards the candidate and sets ``_awaitingFreshCandidate``. While it
        is set (also after a failed ``getdXdt`` or an interrupted commit),
        every commit is refused, including a resubmission and the originally
        correct update, until ``getdXdt`` solves a new candidate. A model that
        has never proposed a step has no candidate and is not awaiting one, so
        a direct ``postProcess`` commits unchecked, as before.
        """
        if self._awaitingFreshCandidate:
            self._pendingCandidate = None
            raise RuntimeError(
                "Cannot commit a binary state: the last proposed step was rejected or failed and no new "
                "candidate has been solved since; call getdXdt again. Accepted state was not advanced."
            )
        pending = self._pendingCandidate
        if pending is None:
            return
        try:
            mismatches = _pending_step_mismatches(
                pending, self.currentTime, time, self._committed_state(x), ("left profile", "right profile", "interface")
            )
        except Exception:
            self._reject_pending_proposal()
            raise
        if mismatches:
            self._reject_pending_proposal()
            raise RuntimeError(
                "Outer solver update does not match the pending converged binary candidate "
                f"({', '.join(mismatches)}); accepted state was not advanced."
            )

    def _reject_pending_proposal(self):
        """Discards the staged candidate and refuses commits until a new one is solved."""
        self._pendingCandidate = None
        self._awaitingFreshCandidate = True

    def getDt(self, dXdt):
        if np.isfinite(self._currdt) and self._currdt > 0:
            return self._currdt
        return self.timeStep

    def getTransformedState(self, time=None):
        """
        Returns the transformed left/right phase state ``(p, q)``.

        Without ``time`` this is a copy of the current accepted state, also when
        the p/q histories are disabled (``record_pq_data=False``). A timed
        request needs the histories and an exactly recorded time.
        """
        return self.getTransformedStateLeft(time), self.getTransformedStateRight(time)

    def getTransformedStateLeft(self, time=None):
        """
        Returns the left transformed composition vector ``p`` (current, or recorded at ``time``).
        """
        if time is None and self._p_curr is not None:
            return self._p_curr.copy()
        if self.pData is None:
            raise ValueError("Transformed left-state history is not available; set record_pq_data=True.")
        return self.pData.y(time)

    def getTransformedStateRight(self, time=None):
        """
        Returns the right transformed composition vector ``q`` (current, or recorded at ``time``).
        """
        if time is None and self._q_curr is not None:
            return self._q_curr.copy()
        if self.qData is None:
            raise ValueError("Transformed right-state history is not available; set record_pq_data=True.")
        return self.qData.y(time)

    def postProcess(self, time, x):
        """
        Commits the wrapper's update after checking it against the pending candidate.

        See ``_verify_pending_candidate``; a near-final no-op step only advances
        time and stops the solve. Once checked, the candidate is consumed and
        ``_awaitingFreshCandidate`` stays set until the commit completes, so an
        update whose commit raised part-way cannot be resubmitted unchecked.
        """
        self._verify_pending_candidate(time, x)
        if self._nearFinalNoop:
            self.currentTime = time
            self._nearFinalNoop = False
            self._pendingCandidate = None
            return [self._p_curr.copy(), self._q_curr.copy(), self._s_curr], True
        self._pendingCandidate = None
        self._awaitingFreshCandidate = True
        GenericModel.postProcess(self, time, x)
        p = np.asarray(x[0], dtype=np.float64).reshape(-1).copy()
        q = np.asarray(x[1], dtype=np.float64).reshape(-1).copy()
        s = self._clipInterfacePosition(float(x[2]), strict=True)
        p[-1] = self.interfaceCompositions[0]
        q[0] = self.interfaceCompositions[1]

        physical = self._reconstruct_physical_profile(p, q, s)[:, np.newaxis]
        self.data.record(time, physical)
        self.interfaceData.record(time, s)
        if self.recordPqData:
            self.pData.record(time, p)
            self.qData.record(time, q)

        self.concData.record(time, self.checkMassIntegral(p, q, s))

        self._s_old = float(self._s_curr)
        self._p_curr = p
        self._q_curr = q
        self._s_curr = float(s)
        self.updateCoupledModels()
        self._awaitingFreshCandidate = False
        return [self._p_curr.copy(), self._q_curr.copy(), self._s_curr], False

    def postSolve(self):
        """
        Finalizes every history after a solve (shared row rule, LEDGERS R09).

        Every history, ``data`` included, appends the final accepted state only
        if its latest record stored no row; a repeated call adds nothing (R09c
        removed the interval-1 duplicate ``data`` row). All histories share their
        recorded times; ``data`` row 0 still holds the profile before the
        transformed-state reconstruction of ``setup``.
        """
        self.data.finalize()
        self.interfaceData.finalize()
        self.concData.finalize()
        if self.pData is not None:
            self.pData.finalize()
        if self.qData is not None:
            self.qData.finalize()

    def getInterfacePosition(self, time=None):
        """Returns the current accepted interface position, or the recorded one at ``time`` (interpolated)."""
        return self.interfaceData.y(time)

    def getTotalInventoryFromState(self, p, q, s):
        return integrate_planar_transformed_profile(
            p=p,
            q=q,
            s=s,
            domain_length=self._R,
            u=self._u_grid,
            v=self._v_grid,
        )

    def getTotalInventory(self, time=None):
        if time is None:
            return self.getTotalInventoryFromState(self._p_curr, self._q_curr, self._s_curr)
        return self.concData.y(time) * self._R

    def getTotalMass(self, time=None):
        return self.getTotalInventory(time=time)

    def checkConservation(self, tolerance: float, time=None):
        """
        Checks absolute transformed-inventory drift against the initial value.

        Returns
        -------
        drift : float
            Absolute inventory difference from the stored initial inventory.
        """
        if self._initialInventory is None:
            raise ValueError("Model must be setup before conservation checks.")
        drift = abs(self.getTotalInventory(time=time) - self._initialInventory)
        if drift > float(tolerance):
            warnings.warn(
                f"Illingworth inventory drift {drift:.3e} exceeded tolerance {float(tolerance):.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return drift

    @classmethod
    def fromDict(cls, data):
        """
        Rejects restart because output dictionaries do not hold the accepted state.

        ``toDict`` and ``save`` write the recorded physical-profile history
        only. The interface position, transformed profiles, previous-step state
        and reference inventory are not stored, so the inherited loader left a
        model reporting the loaded time with its old interface (LEDGERS R11).
        The inherited ``load`` calls this method, so it also raises before any
        state changes. Read saved output with ``numpy.load`` instead.
        """
        raise NotImplementedError(
            "Binary Illingworth output files cannot be restarted because deserialization "
            "does not reconstruct the accepted state (interface position, transformed profiles, "
            "previous step and reference inventory); read saved output with numpy.load."
        )

    def checkMassIntegral(self, p, q, s):
        """
        Returns the transformed-coordinate average composition.

        The trapezoidal integration supports both the standard uniform Landau
        grids and custom nonuniform grids used for validation against the MAP
        implementation.
        """
        return float(self.getTotalInventoryFromState(p, q, s) / self._R)
