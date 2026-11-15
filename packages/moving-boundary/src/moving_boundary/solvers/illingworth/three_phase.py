# Relocated from samuelgprice/kawin fae6149ccc6ebfda2100c4ed5a9b99b962e77620 (R06 three-phase correction on top of main 720d122).
# Source: kawin/diffusion/MovingBoundaryIllingworthTernaryThreePhaseFDM.py; import changes only. See NOTICE and docs/LEDGERS.md.
import warnings
from dataclasses import dataclass

import numpy as np
from moving_boundary.compatibility import DiffusionModel
from moving_boundary.solvers.illingworth._interstitials import (
    composition_bound_violation,
    full_mole_fractions,
    has_interstitials,
    independent_interstitial_mask,
    resolve_diffusivity_provider,
    resolve_interface_equilibrium,
    validate_interstitial_element_order,
)
from moving_boundary.equilibrium import CallableTernaryInterfaceEquilibrium
from moving_boundary.solvers.illingworth.shared import (
    _ArrayHistory,
    _ScalarHistory,
    _BULK_DIFFUSIVITY_IMPLICIT,
    _BULK_DIFFUSIVITY_LAGGED,
    _BULK_DIFFUSIVITY_PHASE_UNIFORM,
    _COMPONENTWISE_ROUNDOFF_FACTOR,
    _INTERFACE_ETA_FD_STEP,
    _SUPPORTED_BULK_DIFFUSIVITY_MODES,
    _FloorProjection,
    _all_or_nothing,
    _bounded_finite_difference_perturbation,
    _finalize_histories_atomically,
    _record_histories_atomically,
    _restore_history_snapshots,
    _coerce_bulk_diffusivity_mode,
    _loge_arange,
    _no_floor_projection,
    _profile_within_floor_bounds,
    _project_shift_roundoff_to_floor,
    _validate_eta_bounds,
    _componentwise_residual_scale,
    _explicit_residual_composition_scale,
    _validate_default_residual_floor,
    _validate_ternary_numerical_controls,
    _validate_ternary_diffusivity_matrix,
    _validate_ternary_diffusivity_stack,
)
from moving_boundary.mesh import CartesianFD1D, MixedBoundary1D, PeriodicBoundary1D
from moving_boundary.solvers.illingworth.numerics.ternary import (
    # Compiled single-RHS block elimination (with the same validation and
    # pivot-failure fallbacks as the Python solver); the two-phase model uses it too.
    _solve_illingworth_block_tridiagonal_two_phase as solve_illingworth_block_tridiagonal,
    _fill_three_phase_interval_system,
    flatten_1d_coordinates,
    integrate_planar_transformed_molar_inventories,
    integrate_planar_transformed_profile_sequence,
    reconstruct_planar_transformed_profile_sequence,
    validate_ternary_profile,
)
from kawin.solver import explicitEulerIterator


@dataclass(frozen=True, slots=True)
class _ThreePhaseBulkResult:
    """Result from one transformed interval solve in the three-phase model.

    ``left_transfer`` and ``right_transfer`` are the total ALE plus diffusive transfers
    over the timestep on the interface-adjacent internal faces. They use the
    same orientation as the interval grid, from smaller to larger transformed
    coordinate. They are evaluated in deviation form from the solved deviation
    ``w = profile - reference`` (ledger R06; see ``_finish_interval_solve``),
    never from differences of the stored absolute ``profile``.

    ``projection`` records the round-off floor projection applied to
    ``profile`` (see ``_project_floor_roundoff``); its ``inventory_change``
    uses this phase's new cell widths and belongs to the returned profile.
    ``projection_split`` is a ``(2, 2)`` array whose rows are the parts of
    that change attributed to the phase's left and right interface residual
    (they add up to ``inventory_change``). Both are required, so that every
    return path, including the Picard exit, must forward them.
    """

    profile: np.ndarray
    left_transfer: np.ndarray
    right_transfer: np.ndarray
    left_face_matrix: np.ndarray | None
    right_face_matrix: np.ndarray | None
    projection: _FloorProjection
    projection_split: np.ndarray
    inner_iterations: int = 0
    inner_update_norm: float = 0.0
    diffusivity_evaluations: int = 0


@dataclass(frozen=True, slots=True)
class _ThreePhaseStepKinematics:
    """Pure old-to-trial geometry and phase-frame displacements for one step."""

    old_right_boundary: float
    new_right_boundary: float
    interface_displacements: np.ndarray
    phase_displacements: np.ndarray


@dataclass(frozen=True, slots=True)
class _ThreePhaseCandidate:
    """Mutually consistent profiles, geometry, etas, and residuals for one trial."""

    x_hat: np.ndarray
    profiles: tuple[np.ndarray, np.ndarray, np.ndarray]
    interfaces: np.ndarray
    etas: np.ndarray
    interface_compositions: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    residual: np.ndarray
    scaled_residual: np.ndarray
    scaled_norm: float
    physical_norm: float
    bulk_results: tuple[_ThreePhaseBulkResult, _ThreePhaseBulkResult, _ThreePhaseBulkResult]
    kinematics: _ThreePhaseStepKinematics | None = None


@dataclass(frozen=True, slots=True)
class _ThreePhaseAcceptedState:
    """Fully validated outputs prepared before an atomic accepted-state commit."""

    profiles: tuple[np.ndarray, np.ndarray, np.ndarray]
    interfaces: np.ndarray
    etas: np.ndarray
    right_boundary: float
    interface_compositions: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    fixed_mesh_profile: np.ndarray
    legacy_inventory: np.ndarray
    molar_inventories: np.ndarray
    total_moles: float
    dependent_closure_error: float


@dataclass(frozen=True, slots=True)
class _ThreePhasePendingState:
    """Exact converged candidate awaiting verification and accepted-state commit.

    ``projection`` stages the candidate's floor-projection totals
    ``(inventory_change, absolute_change, projected_nodes)``; ``postProcess``
    adds them to the cumulative totals only when the step is committed.
    """

    profiles: tuple[np.ndarray, np.ndarray, np.ndarray]
    interfaces: np.ndarray
    etas: np.ndarray
    right_boundary: float
    start_time: float
    dt: float
    projection: tuple | None = None


@dataclass(frozen=True, slots=True)
class ThreePhaseInitialEtaEstimate:
    """
    Diagnostics from initial tie-line selection for the three-phase model.

    ``etas`` are the selected A|B and B|C tie-line coordinates. ``velocities``
    are instantaneous discrete conservative-balance estimates evaluated with
    initial adjacent-node gradients; they are used only to seed the finite-step
    discrete nonlinear solve. ``residual_scale`` is the balance-term magnitude
    that the relative convergence tolerance was applied to. With
    ``initial_eta_componentwise_scaling``, ``component_scales`` holds the four
    per-row scales at the returned iterate (otherwise ``None``).
    """

    etas: np.ndarray
    velocities: np.ndarray
    residual_norm: float
    residual: np.ndarray
    flux_delta: np.ndarray
    interface_compositions: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    method: str
    solver: str
    brackets: tuple[tuple[float, float], tuple[float, float]]
    converged: bool
    iterations: int
    function_calls: int
    residual_scale: float = np.nan
    component_scales: np.ndarray | None = None


class MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(DiffusionModel):
    """
    Three-phase sequential ternary Illingworth moving-boundary model.

    The model tracks a fixed planar phase order ``A | B | C`` with two moving
    sharp interfaces. Each phase is solved on its own Landau grid with two
    independent substitutional ternary components. The two interfaces are
    advanced simultaneously from four nonlinear residual equations: two
    component balances at ``A|B`` and two at ``B|C``; each step takes at least
    one Newton update so tiny timesteps cannot leave the interfaces frozen.
    Startup etas converge relative to the balance-term magnitude
    (``initial_eta_root_rtol``; ``initial_eta_root_xtol`` is an optional
    absolute floor), so they do not depend on length units. Phase
    disappearance is not handled; when exactly one phase is thinner than
    ``terminal_thin_phase_width`` after the normal retry budget is exhausted,
    an optional terminal retry path can keep reducing ``dt`` until one final
    converged step is found and then stop the solve early.

    The geometry is planar, the left boundary is fixed, and the right boundary
    is a stress-free material boundary. ``phase_molar_volumes=(VmA, VmB, VmC)``
    supplies constant phase molar volumes in m^3/mol when the length coordinate
    is in metres; differing phase values may produce one-dimensional expansion
    or contraction, and physical inventory APIs then return mol/m^2. The right
    position is algebraically dependent on the two interfaces. The model is an
    all-substitutional ternary formulation and treats each supplied chemical/
    interdiffusion matrix as phase-local and volume-fixed. It does not transform
    or rescale that matrix for differing molar volumes and does not model
    composition-dependent or species-dependent partial molar volumes,
    Kirkendall markers, vacancies, lattice drift, or mechanics.

    Trial geometry remains candidate-local and the accepted boundary advances
    atomically with the corresponding profiles, interface state, diagnostics,
    and histories. Legacy ``data``/``getTotalInventory`` output remains fixed-
    mesh and length-weighted. Physical conservation uses the explicit molar
    APIs, while ``getPhysicalPhaseProfiles`` is authoritative for the complete
    moving specimen and preserves both sides of interface discontinuities.

    Interstitial solutes (``kawin.thermo.Mobility.interstitials``) are
    supported as in the two-phase model: the two independent components are
    stored as u-fractions (``u_k = x_k / sum(substitutional x)``), the dependent
    (first) element must be substitutional, diffusivities are the volume-fixed
    u-basis matrices, thermodynamics objects are queried with mole fractions,
    and tie-line closures return mole fractions unless they declare
    ``composition_basis = "u"``. ``phase_molar_volumes`` are then volumes per
    mole of substitutional atoms; the molar APIs return moles of every
    component, and only the substitutional entries add up to
    ``getTotalSubstitutionalMoles``. Composition bounds apply to the
    substitutional u-fractions, with interstitial u-fractions bounded below
    only. ``getCompositions`` and ``getPhysicalPhaseProfiles`` return mole
    fractions. For substitutional systems ``u == x`` and results are unchanged.

    ``interface_newton_start`` chooses where each step's interface Newton
    iteration starts: ``'extrapolated'`` (default), the linear extrapolation of
    the last accepted step to the new ``dt``, or ``'old_state'``, the accepted
    ``(interfaces, etas)``. As in the two-phase model, the extrapolated start
    leaves the converged step unchanged but avoids most retries on sharp
    moving fronts. Interval systems use the compiled block-tridiagonal solver
    shared with the two-phase model, and face diffusivity stacks are
    validated in one batch. In ``composition_dependent_lagged`` mode the face
    matrices depend only on the accepted profiles (the model is isothermal),
    so, as in the two-phase model, they are evaluated once per step attempt
    and shared by all Newton candidates. Within an attempt the validated
    tie-line compositions are also cached per interface and eta, and with
    Numba the interval systems are assembled by a compiled kernel
    (``_fill_three_phase_interval_system``) that is bitwise identical to the
    NumPy assembly used otherwise.

    The per-step interface residual rows are normalized per component (see
    ``_residual_scale``; ledger R13). By default the two rows of component
    ``k`` are divided by ``max(|I~_k|, L * constraints.minComposition)``,
    with ``I~_k`` its density-weighted inventory and ``L`` the normalized
    specimen length, so the tolerance bounds each row, and the change of the
    component's inventory per step by twice that, relative to the component's
    own inventory, however dilute. ``interface_residual_composition_scale``
    (scalar or one value per component; default ``None``) instead selects the
    earlier expression ``max(|I~_k|, L * s_k, 1e-300)``, unchanged bit for
    bit; ``1.0`` reproduces the default used before R13 (every row divided by
    ``L``) while ``|I~_k| <= L``. ``initial_eta_componentwise_scaling`` scales
    each row of the startup eta balance separately (see
    ``_estimate_initial_etas``); its default leaves results unchanged.

    **Deviation form (ledger R06).** Each interval is solved for the deviation
    ``w = C - r`` from a reference profile ``r``: the interface composition for
    the outer phases A and C (as in the two-phase model) and, for the middle
    phase B, the linear profile ``c_L + xi (c_R - c_L)`` between its two
    interface compositions. Every free row of an interval system sums to its
    old cell width times ``I``, so the matrix is unchanged; only the
    right-hand side changes (``_solve_interval_planar``). The
    interface-adjacent transfers are formed as ``d_f c + (face coefficient) @
    (deviation terms)`` with the face displacement ``d_f``
    (``_finish_interval_solve``), so the residual no longer contains
    ``-A x0 + A x1`` with absolute compositions, whose round-off
    ``eps |A| |c|`` (``A = dt D / (L dxi)``) exceeded the residual tolerance on
    fine interface cells or long steps. Measured on the asymmetric similarity
    case with 201 nodes per phase and 1e-12-scale interface cells: the scaled
    residual's float error fell from 5e-11..3e-10 to below 1e-18, and the run
    no longer retries. The error that remains comes from the solve itself
    (conditioning, about ``eps |A| |w|``) and from the rounding of the input
    geometry. Diffusivity queries, Picard iterates and norms, stored profiles
    and the endpoint inventory terms stay in absolute compositions.

    **Precision of stored compositions.** A stored profile ``r + w`` resolves
    each component only to about ``eps (|r| + |w|)`` in absolute terms. A
    dilute value just above ``minComposition`` beside a large interface
    composition can therefore land a few ulps below the floor. Such
    violations within the policy threshold
    ``_FLOOR_PROJECTION_ULPS * eps * (|r_i| + |w_i|)`` (a fixed policy, not an
    error bound) are moved onto the floor by ``_project_floor_roundoff``, with
    the Dirichlet nodes never touched; larger ones are rejected by the
    composition validator as before. The signed inventory change is added to
    the interface residuals with the phase's density weight (the middle
    phase's change is split node by node as ``(1 - xi_i, xi_i)`` between A|B
    and B|C), so a converged residual describes the stored profiles. Totals
    are reported per accepted step (``_lastStepProjection*``) and accumulated
    over committed steps (``_cumulativeProjection*``,
    ``_cumulativeProjectedNodes``; ``_cumulativeProjectedSolves`` also counts
    discarded solves).

    ``print_diagnostics`` (default ``False``; also settable afterwards as
    ``printDiagnostics``) prints the last timestep retry of an attempt and the
    state at a step that fails after all retries.

    Limitations (the first three are enforced and raise):

    - **Geometry:** planar only. The left boundary is fixed and the right
      boundary is a stress-free material boundary.
    - **Temperature:** isothermal; a scalar temperature only.
    - **Outer boundaries:** homogeneous zero flux only.
    - **Phase order:** fixed ``A | B | C``. A vanishing phase is not handled
      beyond the optional terminal thin-phase stop; there is no hand-off to
      the two-phase model.
    - **Interface:** local equilibrium through one-parameter tie-line closures.
      Neighbouring tie-lines must not cross (see the two-phase model). There
      is no interface mobility and no para-equilibrium.
    - **Molar volume:** constant per phase (per mole of substitutional atoms
      with interstitials), with no composition dependence or Kirkendall flow.
      Interstitial u-fractions have no site-fraction upper cap.
    - **Accuracy:** interface positions converge at first order in the grid
      spacing, as measured against the exact asymmetric similarity solution
      (``kawin/tests/test_illingworth_ternary_three_phase_similarity.py``).
      Use fine interface grids or Richardson extrapolation for benchmark
      numbers.
    - **Numerical controls:** ``time_step``, ``tolerance`` and the effective
      ``residual_tolerance`` must be positive and finite, as must
      ``semiLog_dt`` and ``semiLogT0`` when ``dt_mode='semi_log'`` (they are
      ignored in fixed mode). They are checked before the host constructor
      runs and again at the start of every ``solve``.
    """

    def __init__(
        self,
        mesh,
        elements,
        phases,
        thermodynamics,
        temperature,
        interfacePositions,
        time_step: float,
        interface_equilibria,
        initial_eta_guess=None,
        initial_eta_method: str = "instantaneous_balance",
        initial_eta_brackets=None,
        initial_eta_root_xtol: float | None = None,
        initial_eta_root_rtol: float = 1e-10,
        initial_eta_root_maxiter: int = 100,
        initial_velocity_guess=None,
        bulk_diffusivity_mode: str = _BULK_DIFFUSIVITY_PHASE_UNIFORM,
        bulk_picard_rtol: float | None = None,
        bulk_picard_atol: float = 1e-12,
        bulk_picard_max_iterations: int = 25,
        bulk_picard_relaxation: float = 1.0,
        dt_mode: str = "fixed",
        semiLog_dt: float | None = None,
        semiLogT0: float | None = None,
        geometry: str = "planar",
        phase_nodes=None,
        tolerance: float = 1e-8,
        residual_tolerance: float | None = None,
        max_iterations: int = 25,
        max_step_retries: int = 8,
        retry_factor: float = 0.5,
        min_middle_width_fraction: float = 1e-10,
        terminal_thin_phase_width: float | None = 1e-9,
        terminal_thin_phase_extra_retries: int = 20,
        terminal_thin_phase_policy: str = "prompt",
        constraints=None,
        record=False,
        record_pq_data: bool = True,
        transformed_grids=None,
        phase_molar_volumes=None,
        interface_newton_start: str = "extrapolated",
        interface_residual_composition_scale=None,
        initial_eta_componentwise_scaling: bool = False,
        print_diagnostics: bool = False,
    ):
        self.initialInterfacePositions = np.asarray(interfacePositions, dtype=np.float64).reshape(-1)
        self.printDiagnostics = bool(print_diagnostics)
        self.interfaceResidualCompositionScale = (
            None
            if interface_residual_composition_scale is None
            else np.broadcast_to(np.asarray(interface_residual_composition_scale, dtype=np.float64), (2,)).copy()
        )
        self.initialEtaComponentwiseScaling = bool(initial_eta_componentwise_scaling)
        self.timeStep = float(time_step)
        if interface_newton_start not in {"old_state", "extrapolated"}:
            raise ValueError("interface_newton_start must be 'old_state' or 'extrapolated'.")
        self.interfaceNewtonStart = str(interface_newton_start)
        self.interfaceEquilibria = self._coerce_interface_equilibria(interface_equilibria)
        self.initialEtaGuess = None if initial_eta_guess is None else np.asarray(initial_eta_guess, dtype=np.float64).reshape(-1)
        self.initialEtaMethod = str(initial_eta_method)
        self.initialEtaBrackets = initial_eta_brackets
        self.initialEtaRootXtol = None if initial_eta_root_xtol is None else float(initial_eta_root_xtol)
        self.initialEtaRootRtol = float(initial_eta_root_rtol)
        self.initialEtaRootMaxiter = int(initial_eta_root_maxiter)
        self.initialVelocityGuess = None if initial_velocity_guess is None else np.asarray(initial_velocity_guess, dtype=np.float64).reshape(-1)
        self.initialEtaEstimate = None
        self.bulkDiffusivityMode = _coerce_bulk_diffusivity_mode(bulk_diffusivity_mode)
        self.bulkPicardRtol = None if bulk_picard_rtol is None else float(bulk_picard_rtol)
        self.bulkPicardAtol = float(bulk_picard_atol)
        self.bulkPicardMaxIterations = int(bulk_picard_max_iterations)
        self.bulkPicardRelaxation = float(bulk_picard_relaxation)
        self.dtMode = str(dt_mode)
        self.semiLog_dt = None if semiLog_dt is None else float(semiLog_dt)
        self.semiLogT0 = None if semiLogT0 is None else float(semiLogT0)
        self.geometry = str(geometry)
        self._hasExplicitPhaseMolarVolumes = phase_molar_volumes is not None
        molar_volumes = np.ones(3, dtype=np.float64) if phase_molar_volumes is None else np.asarray(phase_molar_volumes, dtype=np.float64).reshape(-1)
        if molar_volumes.shape != (3,) or not np.all(np.isfinite(molar_volumes)) or np.any(molar_volumes <= 0.0):
            raise ValueError("phase_molar_volumes must contain three positive finite values.")
        self._phaseMolarVolumes = molar_volumes.copy()
        self._phaseMolarVolumes.setflags(write=False)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            molar_densities = 1.0 / molar_volumes
            normalized_densities = molar_densities / molar_densities[0]
        if not np.all(np.isfinite(molar_densities)) or not np.all(np.isfinite(normalized_densities)):
            raise ValueError("phase_molar_volumes produce non-finite molar densities.")
        self._phaseMolarDensities = molar_densities.copy()
        self._phaseMolarDensities.setflags(write=False)
        self._normalizedPhaseMolarDensities = normalized_densities.copy()
        self._normalizedPhaseMolarDensities.setflags(write=False)
        self._equalPhaseMolarVolumes = bool(np.all(molar_volumes == molar_volumes[0]))
        self.phaseNodes = None if phase_nodes is None else tuple(int(v) for v in np.asarray(phase_nodes, dtype=np.int64).reshape(-1))
        self.tolerance = float(tolerance)
        self.residualTolerance = float(tolerance if residual_tolerance is None else residual_tolerance)
        self.maxIterations = int(max_iterations)
        self.maxStepRetries = int(max_step_retries)
        self.retryFactor = float(retry_factor)
        self.minMiddleWidthFraction = float(min_middle_width_fraction)
        self.terminalThinPhaseWidth = None if terminal_thin_phase_width is None else float(terminal_thin_phase_width)
        self.terminalThinPhaseExtraRetries = int(terminal_thin_phase_extra_retries)
        self.terminalThinPhasePolicy = str(terminal_thin_phase_policy)
        self.recordPqData = bool(record_pq_data)
        self._inputGrids = None if transformed_grids is None else tuple(self._validate_transformed_grid(g, f"transformed_grids[{i}]") for i, g in enumerate(transformed_grids))
        if self._inputGrids is not None:
            if len(self._inputGrids) != 3:
                raise ValueError("transformed_grids must contain exactly three grids.")
            if self.phaseNodes is not None and tuple(len(g) for g in self._inputGrids) != self.phaseNodes:
                raise ValueError("phase_nodes must match transformed_grids lengths when both are specified.")
            self.phaseNodes = tuple(len(g) for g in self._inputGrids)

        self.interfaceData = _ArrayHistory((2,), record)
        self.etaData = _ArrayHistory((2,), record)
        self.inventoryData = _ArrayHistory((2,), record)
        self.rightBoundaryData = _ScalarHistory(record)
        self.molarInventoryData = _ArrayHistory((3,), record)
        self.totalMolesData = _ScalarHistory(record)
        self.dependentMoleClosureErrorData = _ScalarHistory(record)
        self.profileData = None

        self._currdt = np.inf
        self._outerMinTimeStep = 0.0
        self._outerMaxTimeStep = np.inf
        self._solveMinDtFrac = 1e-8
        self._solveMaxDtFrac = 1.0
        self._pendingCandidate = None
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
        self._lastStepRetries = 0
        self._lastImplicitIterations = 0
        self._lastImplicitResidual = np.nan
        self._lastImplicitPhysicalResidual = np.nan
        self._lastImplicitConverged = False
        self._lastImplicitForcedUpdateSkipped = False
        self._lastImplicitFailureReason = None
        self._lastInterfaceCompositions = None
        self._terminalThinPhaseStop = False
        self._terminalThinPhaseInfo = None
        self._initialInventory = None
        self._initialMolarInventories = None
        self._initialTotalMoles = None

        self._z = None
        self._R = None
        self._R0 = None
        self._initialTotalAmount = None
        self._grids = None
        self._profiles_curr = None
        self._interfaces_curr = None
        self._interfaces_old = None
        self._etas_curr = None
        self._etas_old = None
        self._lastAcceptedDt = None
        self._laggedFaceDiffusivityCache = None
        self._interfaceCompositionCache = None
        self._reset_projection_diagnostics()
        self._interstitialMaskCache = None
        # Reference seam: False forces the NumPy interval assembly (tests compare both).
        self._useCompiledIntervalAssembly = True

        _validate_ternary_numerical_controls(self)
        super().__init__(
            mesh=mesh,
            elements=elements,
            phases=phases,
            thermodynamics=thermodynamics,
            temperature=temperature,
            constraints=constraints,
            record=record,
        )
        if not self._equalPhaseMolarVolumes:
            self.data._timeInterpolationEnabled = False
            self.data._timeInterpolationError = (
                "Fixed-mesh interpolation is unavailable for a moving unequal-volume domain; "
                "request an exact recorded time or use getPhysicalPhaseProfiles()."
            )
        self._validateModelConfiguration()
        self.interfaceData.currentY = self.initialInterfacePositions.copy()
        self.interfaceData._y[0] = self.initialInterfacePositions.copy()

    def _coerce_interface_equilibria(self, interface_equilibria):
        if interface_equilibria is None:
            raise ValueError("interface_equilibria must provide closures for A|B and B|C.")
        if len(interface_equilibria) != 2:
            raise ValueError("interface_equilibria must contain exactly two closures.")
        closures = []
        for i, closure in enumerate(interface_equilibria):
            if hasattr(closure, "interface_compositions"):
                closures.append(closure)
            elif callable(closure):
                closures.append(CallableTernaryInterfaceEquilibrium(closure))
            else:
                raise TypeError(f"interface_equilibria[{i}] must be an eta-capable closure or callable.")
        return tuple(closures)

    def _validate_transformed_grid(self, grid, name):
        """Validates a supplied phase Landau-coordinate grid."""
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
            raise TypeError("MovingBoundaryIllingworthTernaryThreePhaseFD1DModel requires a CartesianFD1D mesh.")
        if len(self.allElements) != 3 or self.mesh.numResponses != 2:
            raise ValueError("MovingBoundaryIllingworthTernaryThreePhaseFD1DModel requires ternary systems with two independent responses.")
        if len(self.phases) != 3:
            raise ValueError("MovingBoundaryIllingworthTernaryThreePhaseFD1DModel requires exactly three explicit phases.")
        validate_interstitial_element_order(self.allElements)
        if self.geometry != "planar":
            raise NotImplementedError("MovingBoundaryIllingworthTernaryThreePhaseFD1DModel currently implements only planar geometry.")
        _validate_ternary_numerical_controls(self)
        _validate_default_residual_floor(self)
        if self.initialEtaMethod != "instantaneous_balance":
            raise ValueError("initial_eta_method must be 'instantaneous_balance'.")
        if self.initialEtaRootXtol is not None and not (np.isfinite(self.initialEtaRootXtol) and self.initialEtaRootXtol > 0.0):
            raise ValueError("initial_eta_root_xtol must be positive.")
        if not (np.isfinite(self.initialEtaRootRtol) and self.initialEtaRootRtol > 0.0):
            raise ValueError("initial_eta_root_rtol must be positive and finite.")
        if self.initialEtaRootMaxiter < 1:
            raise ValueError("initial_eta_root_maxiter must be at least 1.")
        if self.initialVelocityGuess is not None and self.initialVelocityGuess.shape != (2,):
            raise ValueError("initial_velocity_guess must contain two interface velocities.")
        if self.initialInterfacePositions.shape != (2,) or not np.all(np.isfinite(self.initialInterfacePositions)):
            raise ValueError("interfacePositions must contain two finite values.")
        if self.maxIterations < 2:
            raise ValueError("max_iterations must be at least 2.")
        if self.maxStepRetries < 1:
            raise ValueError("max_step_retries must be at least 1.")
        if not (0.0 < self.retryFactor < 1.0):
            raise ValueError("retry_factor must be between 0 and 1.")
        if not np.isfinite(self.minMiddleWidthFraction) or self.minMiddleWidthFraction <= 0.0:
            raise ValueError("min_middle_width_fraction must be positive and finite.")
        if self.terminalThinPhaseWidth is not None and (not np.isfinite(self.terminalThinPhaseWidth) or self.terminalThinPhaseWidth <= 0.0):
            raise ValueError("terminal_thin_phase_width must be positive and finite when specified.")
        if self.terminalThinPhaseExtraRetries < 0:
            raise ValueError("terminal_thin_phase_extra_retries must be non-negative.")
        if self.terminalThinPhasePolicy not in {"prompt", "continue", "raise"}:
            raise ValueError("terminal_thin_phase_policy must be 'prompt', 'continue', or 'raise'.")
        if self.bulkPicardRtol is not None and (not np.isfinite(self.bulkPicardRtol) or self.bulkPicardRtol <= 0.0):
            raise ValueError("bulk_picard_rtol must be positive when specified.")
        if not np.isfinite(self.bulkPicardAtol) or self.bulkPicardAtol <= 0.0:
            raise ValueError("bulk_picard_atol must be positive and finite.")
        if self.bulkPicardMaxIterations < 1:
            raise ValueError("bulk_picard_max_iterations must be at least 1.")
        if not np.isfinite(self.bulkPicardRelaxation) or not (0.0 < self.bulkPicardRelaxation <= 1.0):
            raise ValueError("bulk_picard_relaxation must be in the interval (0, 1].")
        if self.phaseNodes is not None and (len(self.phaseNodes) != 3 or any(n < 3 for n in self.phaseNodes)):
            raise ValueError("phase_nodes must contain three integers of at least 3.")
        for closure in self.interfaceEquilibria:
            _validate_eta_bounds(closure)
        self._validate_external_boundary_conditions()
        self._validate_isothermal_temperature()

    def _validate_external_boundary_conditions(self):
        """Rejects unsupported external boundary conditions for the closed planar solver."""
        bc = getattr(self.mesh, "boundaryConditions", None)
        if bc is None:
            return
        if isinstance(bc, PeriodicBoundary1D):
            raise NotImplementedError("Three-phase Illingworth supports only homogeneous zero-flux external boundaries.")
        if not isinstance(bc, MixedBoundary1D):
            raise NotImplementedError("Three-phase Illingworth supports only MixedBoundary1D zero-flux external boundaries.")
        expected_shape = (self.mesh.numResponses,)
        for attr in ("LBCtype", "RBCtype", "LBCvalue", "RBCvalue"):
            values = np.asarray(getattr(bc, attr, None))
            if values.shape != expected_shape:
                raise NotImplementedError("Boundary-condition arrays must match the independent-component count.")
        if not np.all(np.asarray(bc.LBCtype) == MixedBoundary1D.NEUMANN) or not np.all(np.asarray(bc.RBCtype) == MixedBoundary1D.NEUMANN):
            raise NotImplementedError("Three-phase Illingworth supports only Neumann zero-flux external boundaries.")
        if not np.all(np.asarray(bc.LBCvalue, dtype=np.float64) == 0.0) or not np.all(np.asarray(bc.RBCvalue, dtype=np.float64) == 0.0):
            raise NotImplementedError("Three-phase Illingworth supports only homogeneous zero-flux external boundaries.")

    def _validate_isothermal_temperature(self):
        """Validates the scalar isothermal temperature assumption used by v1."""
        params = getattr(self.temperatureParameters, "Tparameters", None)
        if isinstance(params, tuple) and len(params) == 2:
            values = np.asarray(params[1], dtype=np.float64).reshape(-1)
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError("Temperature values must be finite.")
            if not np.all(values == values[0]):
                raise NotImplementedError("Three-phase Illingworth currently supports only isothermal temperature.")
            return
        if callable(params):
            raise NotImplementedError("Three-phase Illingworth currently supports only isothermal temperature.")
        values = np.asarray(params, dtype=np.float64).reshape(-1)
        if values.size != 1 or not np.isfinite(values[0]):
            raise ValueError("Three-phase Illingworth requires a finite scalar isothermal temperature.")

    def _getBoundaryConditions(self):
        bc = getattr(self.mesh, "boundaryConditions", None)
        if bc is None:
            bc = MixedBoundary1D(self.mesh.responses)
            self.mesh.boundaryConditions = bc
        return bc

    def _minimum_middle_width(self, right_boundary=None):
        domain_length = float(self._R if right_boundary is None else right_boundary)
        return max(domain_length * self.minMiddleWidthFraction, 1e-14)

    def _phase_widths(self, interfaces, right_boundary=None):
        """Returns the physical widths of the ``A``, ``B``, and ``C`` regions."""
        s_ab, s_bc = np.asarray(interfaces, dtype=np.float64).reshape(2)
        domain_length = float(self._R if right_boundary is None else right_boundary)
        return np.asarray([s_ab, s_bc - s_ab, domain_length - s_bc], dtype=np.float64)

    def _single_terminal_thin_phase(self, interfaces, right_boundary=None):
        """Returns ``(index, width)`` only when exactly one phase is below the terminal threshold."""
        if self.terminalThinPhaseWidth is None:
            return None
        widths = self._phase_widths(interfaces, right_boundary=right_boundary)
        mask = np.isfinite(widths) & (widths < self.terminalThinPhaseWidth)
        if int(np.count_nonzero(mask)) != 1:
            return None
        index = int(np.flatnonzero(mask)[0])
        return index, float(widths[index])

    def _validate_interfaces(self, interfaces, strict=True, right_boundary=None):
        """Validates interfaces against an explicit accepted or trial boundary."""
        values = np.asarray(interfaces, dtype=np.float64).reshape(2)
        if right_boundary is not None:
            domain_length = float(right_boundary)
        elif self._R is None:
            z = flatten_1d_coordinates(self.mesh.z)
            domain_length = float(z[-1] - z[0])
        else:
            domain_length = float(self._R)
        if not np.isfinite(domain_length) or domain_length <= 0.0:
            if strict:
                raise ValueError("Three-phase trial geometry requires a positive finite right boundary.")
            return values
        eps = max(domain_length * 1e-14, 1e-14)
        min_width = max(domain_length * self.minMiddleWidthFraction, eps)
        widths = np.asarray([values[0], values[1] - values[0], domain_length - values[1]], dtype=np.float64)
        valid = np.all(np.isfinite(widths)) and np.all(widths > eps) and widths[1] > min_width
        if strict and not valid:
            raise ValueError("Three-phase trial geometry must satisfy 0 < s_AB < s_BC < R_trial with a nonzero middle phase and valid phase widths.")
        values[0] = np.clip(values[0], eps, domain_length - eps)
        values[1] = np.clip(values[1], eps, domain_length - eps)
        if values[1] - values[0] < min_width:
            center = 0.5 * (values[0] + values[1])
            values[0] = center - 0.5 * min_width
            values[1] = center + 0.5 * min_width
        return values

    def _right_boundary_from_interfaces(self, interfaces):
        """
        Returns the algebraic right boundary required by total-mole conservation.

        The result is trial-local and this helper never mutates accepted state.
        Equal phase molar volumes take an exact shortcut to ``_R0``.
        """
        if self._R0 is None or self._initialTotalAmount is None:
            raise ValueError("Model must be setup before evaluating trial right-boundary kinematics.")
        values = np.asarray(interfaces, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(values)):
            raise ValueError("Trial interfaces must contain finite values.")
        if self._equalPhaseMolarVolumes:
            return float(self._R0)
        s_ab, s_bc = values
        rho_a, rho_b, rho_c = self._phaseMolarDensities
        return float(s_bc + (self._initialTotalAmount - rho_a * s_ab - rho_b * (s_bc - s_ab)) / rho_c)

    def _phase_frame_displacements(self, old_interfaces, new_interfaces, old_right_boundary, new_right_boundary):
        """
        Returns finite phase-frame displacements ``[DeltaU_A, DeltaU_B, DeltaU_C]``.

        The local interface kinematics are algebraically equivalent to the
        total-mole boundary relation. Phase C is anchored to the moving right
        material boundary, so ``DeltaU_C`` is exactly ``DeltaR``.
        """
        if self._equalPhaseMolarVolumes:
            return np.zeros(3, dtype=np.float64)
        old_interfaces = np.asarray(old_interfaces, dtype=np.float64).reshape(2)
        new_interfaces = np.asarray(new_interfaces, dtype=np.float64).reshape(2)
        delta_s_ab, delta_s_bc = new_interfaces - old_interfaces
        vm_a, vm_b, vm_c = self._phaseMolarVolumes
        delta_u_b = (1.0 - vm_b / vm_a) * delta_s_ab
        local_delta_u_c = (vm_c / vm_b) * delta_u_b + (1.0 - vm_c / vm_b) * delta_s_bc
        delta_r = float(new_right_boundary) - float(old_right_boundary)
        scale = max(abs(delta_r), abs(local_delta_u_c), abs(float(self._R0)), 1.0)
        if not np.isclose(local_delta_u_c, delta_r, rtol=5e-13, atol=5e-15 * scale):
            raise RuntimeError("Algebraic total-mole boundary motion disagrees with local interface kinematics.")
        return np.asarray([0.0, delta_u_b, delta_r], dtype=np.float64)

    def _phase_frame_velocities(self, interface_velocities):
        """Returns the differential limit of the finite phase-frame displacements."""
        if self._equalPhaseMolarVolumes:
            return np.zeros(3, dtype=np.float64)
        v_ab, v_bc = np.asarray(interface_velocities, dtype=np.float64).reshape(2)
        vm_a, vm_b, vm_c = self._phaseMolarVolumes
        velocity_b = (1.0 - vm_b / vm_a) * v_ab
        velocity_c = (vm_c / vm_b) * velocity_b + (1.0 - vm_c / vm_b) * v_bc
        return np.asarray([0.0, velocity_b, velocity_c], dtype=np.float64)

    def _step_kinematics(self, old_interfaces, new_interfaces):
        """Builds and validates immutable old-to-trial kinematics without state mutation."""
        old_interfaces = np.asarray(old_interfaces, dtype=np.float64).reshape(2)
        new_interfaces = np.asarray(new_interfaces, dtype=np.float64).reshape(2)
        old_right = self._right_boundary_from_interfaces(old_interfaces)
        new_right = self._right_boundary_from_interfaces(new_interfaces)
        self._validate_interfaces(old_interfaces, strict=True, right_boundary=old_right)
        self._validate_interfaces(new_interfaces, strict=True, right_boundary=new_right)
        interface_displacements = (new_interfaces - old_interfaces).copy()
        phase_displacements = self._phase_frame_displacements(old_interfaces, new_interfaces, old_right, new_right)
        interface_displacements.setflags(write=False)
        phase_displacements.setflags(write=False)
        return _ThreePhaseStepKinematics(old_right, new_right, interface_displacements, phase_displacements)

    def _eta_bounds(self):
        return tuple(_validate_eta_bounds(closure) for closure in self.interfaceEquilibria)

    def _initial_eta_brackets(self):
        bounds = self._eta_bounds()
        if self.initialEtaBrackets is None:
            return bounds
        values = tuple(np.asarray(self.initialEtaBrackets, dtype=np.float64).reshape(2, 2))
        brackets = []
        for i, bracket in enumerate(values):
            lower, upper = float(bracket[0]), float(bracket[1])
            eta_lower, eta_upper = bounds[i]
            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or upper <= lower
                or lower < eta_lower
                or upper > eta_upper
            ):
                raise ValueError("initial_eta_brackets must lie within each interface eta bound and be increasing.")
            brackets.append((lower, upper))
        return tuple(brackets)

    def _initial_eta_start(self, brackets):
        if self.initialEtaGuess is None:
            return np.asarray([0.5 * (lower + upper) for lower, upper in brackets], dtype=np.float64)
        if self.initialEtaGuess.shape != (2,):
            raise ValueError("initial_eta_guess must contain two eta values.")
        etas = self.initialEtaGuess.astype(np.float64).copy()
        for i, (lower, upper) in enumerate(brackets):
            if etas[i] < lower or etas[i] > upper:
                raise ValueError("initial_eta_guess must lie within each initial eta bracket.")
        return etas

    def _initial_adjacent_compositions(self, composition, interfaces):
        """Interpolates initial bulk compositions adjacent to both interfaces."""
        c = np.asarray(composition, dtype=np.float64)
        s_ab, s_bc = np.asarray(interfaces, dtype=np.float64).reshape(2)
        boundaries = np.asarray([0.0, s_ab, s_bc, self._R], dtype=np.float64)
        adjacent = []
        for phase_index, grid in enumerate(self._grids):
            left, right = boundaries[phase_index], boundaries[phase_index + 1]
            mask = (self._z >= left) & (self._z <= right)
            if not np.any(mask):
                raise ValueError("Initial interfaces leave an empty phase.")
            if phase_index == 0:
                query = left + (right - left) * grid[-2]
                values = [np.interp(query, self._z[mask], c[mask, component]) for component in range(2)]
                adjacent.append(np.asarray(values, dtype=np.float64))
            elif phase_index == 1:
                query_left = left + (right - left) * grid[1]
                query_right = left + (right - left) * grid[-2]
                values_left = [np.interp(query_left, self._z[mask], c[mask, component]) for component in range(2)]
                values_right = [np.interp(query_right, self._z[mask], c[mask, component]) for component in range(2)]
                adjacent.append((np.asarray(values_left, dtype=np.float64), np.asarray(values_right, dtype=np.float64)))
            else:
                query = left + (right - left) * grid[1]
                values = [np.interp(query, self._z[mask], c[mask, component]) for component in range(2)]
                adjacent.append(np.asarray(values, dtype=np.float64))
        return adjacent[0], adjacent[1][0], adjacent[1][1], adjacent[2]

    def _diffusivity_source(self):
        """
        Returns the object queried for interdiffusivity matrices.

        With interstitials this wraps ``self.therm`` so u-fraction compositions
        are converted to mole fractions before each query; otherwise it is
        ``self.therm`` itself.
        """
        return resolve_diffusivity_provider(self.therm, self.allElements)

    def _interface_face_diffusivity(self, composition, phase_index, position, context):
        """Returns one validated diffusivity matrix for an initial interface-face estimate."""
        T = np.asarray(self.temperatureParameters(np.asarray([[float(position)]], dtype=np.float64), 0.0), dtype=np.float64).reshape(-1)
        source = self._diffusivity_source()
        try:
            D = source.getInterdiffusivity(composition, float(T[0]), phase=self.phases[phase_index], query_context=context)
        except TypeError:
            D = source.getInterdiffusivity(composition, float(T[0]), phase=self.phases[phase_index])
        return _validate_ternary_diffusivity_matrix(D, self.phases[phase_index], context=context)

    def _initial_face_diffusivity_matrices(self, interfaces, interface_compositions, adjacent):
        """Returns diffusivity matrices for the four initial interface-adjacent faces."""
        s_ab, s_bc = np.asarray(interfaces, dtype=np.float64).reshape(2)
        p_a, p_b_left, p_b_right, p_c = adjacent
        c_a_ab, c_b_ab = interface_compositions[0]
        c_b_bc, c_c_bc = interface_compositions[1]

        if self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_PHASE_UNIFORM:
            D_a = self._interface_face_diffusivity(c_a_ab, 0, s_ab, "initial-eta interface")
            D_b = self._interface_face_diffusivity(0.5 * (c_b_ab + c_b_bc), 1, 0.5 * (s_ab + s_bc), "initial-eta interface")
            D_c = self._interface_face_diffusivity(c_c_bc, 2, s_bc, "initial-eta interface")
            D_b_left = D_b
            D_b_right = D_b
        else:
            D_a = self._interface_face_diffusivity(0.5 * (p_a + c_a_ab), 0, s_ab, "initial-eta bulk")
            D_b_left = self._interface_face_diffusivity(0.5 * (c_b_ab + p_b_left), 1, s_ab, "initial-eta bulk")
            D_b_right = self._interface_face_diffusivity(0.5 * (p_b_right + c_b_bc), 1, s_bc, "initial-eta bulk")
            D_c = self._interface_face_diffusivity(0.5 * (c_c_bc + p_c), 2, s_bc, "initial-eta bulk")
        return D_a, D_b_left, D_b_right, D_c

    def _initial_face_transfer_rate(self, relative_face_velocity, D_face, length, dxi, left_value, right_value):
        """
        Returns the instantaneous conservative ALE plus diffusive face transfer.

        The sign convention matches the transient interval transfer ``H``:
        Positive grid velocity relative to the phase frame uses the right
        transformed node as the ALE donor because transformed advection has
        the opposite sign.
        """
        diffusive = np.matmul(np.asarray(D_face, dtype=np.float64), (np.asarray(right_value, dtype=np.float64) - np.asarray(left_value, dtype=np.float64)) / (float(length) * float(dxi)))
        relative_face_velocity = float(relative_face_velocity)
        if relative_face_velocity > 0.0:
            return relative_face_velocity * np.asarray(right_value, dtype=np.float64) + diffusive
        if relative_face_velocity < 0.0:
            return relative_face_velocity * np.asarray(left_value, dtype=np.float64) + diffusive
        return diffusive

    def _initial_discrete_interface_terms(self, interfaces, interface_compositions, adjacent, velocities, *, return_zero_velocity=False, return_face_transfers=False):
        """
        Evaluates the discrete conservative ALE instantaneous interface balance.

        Initial non-interface profile nodes and phase widths are held fixed.
        ``velocities`` contains ``[V_AB, V_BC]``; eta affects only the
        equilibrium endpoint compositions supplied in ``interface_compositions``.
        When requested, the zero-velocity residual is returned from the same
        face diffusivity evaluation so diagnostics do not repeat thermodynamic
        work during the nonlinear initial-eta solve. With
        ``return_face_transfers`` (requires ``return_zero_velocity``), the four
        density-weighted zero-velocity face transfers (rows ``A|B`` left/right,
        ``B|C`` left/right) are also returned; they scale the relative
        initial-eta convergence test.
        """
        s_ab, s_bc = np.asarray(interfaces, dtype=np.float64).reshape(2)
        v_ab, v_bc = np.asarray(velocities, dtype=np.float64).reshape(2)
        phase_velocities = self._phase_frame_velocities(velocities)
        _, velocity_b, velocity_c = phase_velocities
        p_a, p_b_left, p_b_right, p_c = adjacent
        c_a_ab, c_b_ab = interface_compositions[0]
        c_b_bc, c_c_bc = interface_compositions[1]
        length_a = s_ab
        length_b = s_bc - s_ab
        length_c = self._R0 - s_bc
        u_a = self._grids[0]
        u_b = self._grids[1]
        u_c = self._grids[2]
        D_a, D_b_left, D_b_right, D_c = self._initial_face_diffusivity_matrices(interfaces, interface_compositions, adjacent)

        xi_a_right = 0.5 * (u_a[-2] + 1.0)
        xi_b_left = 0.5 * u_b[1]
        xi_b_right = 0.5 * (u_b[-2] + 1.0)
        xi_c_left = 0.5 * u_c[1]
        w_a_right = xi_a_right * v_ab
        w_b_left = (1.0 - xi_b_left) * v_ab + xi_b_left * v_bc - velocity_b
        w_b_right = (1.0 - xi_b_right) * v_ab + xi_b_right * v_bc - velocity_b
        w_c_left = (1.0 - xi_c_left) * v_bc + xi_c_left * velocity_c - velocity_c

        transfer_a_right = self._initial_face_transfer_rate(w_a_right, D_a, length_a, 1.0 - u_a[-2], p_a, c_a_ab)
        transfer_b_left = self._initial_face_transfer_rate(w_b_left, D_b_left, length_b, u_b[1], c_b_ab, p_b_left)
        transfer_b_right = self._initial_face_transfer_rate(w_b_right, D_b_right, length_b, 1.0 - u_b[-2], p_b_right, c_b_bc)
        transfer_c_left = self._initial_face_transfer_rate(w_c_left, D_c, length_c, u_c[1], c_c_bc, p_c)

        length_rate_a = v_ab
        length_rate_b = v_bc - v_ab
        length_rate_c = velocity_c - v_bc
        dm_a_right = 0.5 * (1.0 - u_a[-2]) * length_rate_a * c_a_ab
        dm_b_left = 0.5 * u_b[1] * length_rate_b * c_b_ab
        dm_b_right = 0.5 * (1.0 - u_b[-2]) * length_rate_b * c_b_bc
        dm_c_left = 0.5 * u_c[1] * length_rate_c * c_c_bc

        density_a, density_b, density_c = self._normalizedPhaseMolarDensities
        residual_ab = density_a * (dm_a_right + transfer_a_right)
        residual_ab += density_b * (dm_b_left - transfer_b_left)
        residual_bc = density_b * (dm_b_right + transfer_b_right)
        residual_bc += density_c * (dm_c_left - transfer_c_left)
        residual = np.concatenate((residual_ab, residual_bc))
        if not return_zero_velocity:
            return residual, None

        zero_a_right = self._initial_face_transfer_rate(0.0, D_a, length_a, 1.0 - u_a[-2], p_a, c_a_ab)
        zero_b_left = self._initial_face_transfer_rate(0.0, D_b_left, length_b, u_b[1], c_b_ab, p_b_left)
        zero_b_right = self._initial_face_transfer_rate(0.0, D_b_right, length_b, 1.0 - u_b[-2], p_b_right, c_b_bc)
        zero_c_left = self._initial_face_transfer_rate(0.0, D_c, length_c, u_c[1], c_c_bc, p_c)
        zero_ab = density_a * zero_a_right - density_b * zero_b_left
        zero_bc = density_b * zero_b_right - density_c * zero_c_left
        zero_velocity_residual = np.concatenate((zero_ab, zero_bc))
        if return_face_transfers:
            face_transfers = np.vstack(
                (density_a * zero_a_right, density_b * zero_b_left, density_b * zero_b_right, density_c * zero_c_left)
            )
            return residual, zero_velocity_residual, face_transfers
        return residual, zero_velocity_residual

    def _initial_discrete_interface_residuals(self, interfaces, interface_compositions, adjacent, velocities):
        """Returns the instantaneous residual from ``_initial_discrete_interface_terms``."""
        residual, _ = self._initial_discrete_interface_terms(interfaces, interface_compositions, adjacent, velocities)
        return residual

    def _initial_eta_tolerance(self, residual_scale):
        """
        Returns the initial-eta residual threshold ``max(xtol, rtol * residual_scale)``.

        ``initial_eta_root_xtol=None`` disables the absolute floor, so the
        test is invariant to length units and diffusivity magnitude.
        """
        absolute = 0.0 if self.initialEtaRootXtol is None else self.initialEtaRootXtol
        return max(absolute, self.initialEtaRootRtol * float(residual_scale))

    def _estimate_initial_etas(self, composition, interfaces):
        """
        Solves the discrete instantaneous two-interface balance for startup etas.

        The unknowns are two scaled interface velocities and one eta for each
        interface. Initial phase widths and non-interface transformed profile
        nodes are held fixed, and the residual is the ``dt -> 0`` limit of the
        conservative ALE finite-step interface balance.

        Convergence is tested relative to the balance-term magnitude at the
        current iterate (see ``_initial_eta_tolerance``): the largest
        density-weighted individual face transfer, the velocity-proportional
        part of the residual, or the change of the face transfers across the
        eta brackets (which keeps the scale non-zero at a stationary
        equilibrium root). In SI units these terms can be ``~1e-11``, so an
        absolute test would make the converged etas depend on the starting
        guess and the length unit.

        With ``initial_eta_componentwise_scaling`` each of the four residual
        rows (two components at each interface) is divided by its own scale:
        the same three terms taken per row, floored per interface at
        ``64 * eps / initial_eta_root_rtol`` times that interface's largest
        row scale, and frozen at the current Newton iterate for its Jacobian
        and line search. Convergence then requires
        ``max|scaled residual| <= initial_eta_root_rtol``. Use it when the two
        components have very different diffusivities: the shared max-norm scale
        is set by the fast component and leaves the slow component's balance
        effectively unconverged. ``residual_scale`` on the estimate is then the
        largest row scale.
        """
        brackets = self._initial_eta_brackets()
        eta0 = self._initial_eta_start(brackets)
        adjacent = self._initial_adjacent_compositions(composition, interfaces)

        def face_transfers_at(etas):
            _, _, face_transfers = self._initial_discrete_interface_terms(
                interfaces,
                self._interface_compositions(etas),
                adjacent,
                np.zeros(2, dtype=np.float64),
                return_zero_velocity=True,
                return_face_transfers=True,
            )
            return face_transfers

        sensitivity_scale = None

        def eta_sensitivity_scale():
            # Change of the face transfers across the eta brackets. It does not
            # vanish at a stationary equilibrium root, where the transfers and
            # velocities all go to zero. Evaluated lazily by ``converged_at``.
            try:
                return float(
                    np.max(np.abs(face_transfers_at([brackets[0][1], brackets[1][1]]) - face_transfers_at([brackets[0][0], brackets[1][0]])))
                )
            except ValueError:
                return 0.0

        def converged_at(norm, scale):
            nonlocal sensitivity_scale
            if norm <= self._initial_eta_tolerance(scale):
                return True
            if sensitivity_scale is None:
                sensitivity_scale = eta_sensitivity_scale()
            return norm <= self._initial_eta_tolerance(sensitivity_scale)

        componentwise = self.initialEtaComponentwiseScaling
        component_scale_floor = min(1.0, _COMPONENTWISE_ROUNDOFF_FACTOR * float(np.finfo(np.float64).eps) / self.initialEtaRootRtol)

        def interface_rows(face_values):
            # Face rows are (A right, B left, B right, C left); residual rows are
            # (A|B component 1, A|B component 2, B|C component 1, B|C component 2).
            face_values = np.abs(np.asarray(face_values, dtype=np.float64))
            return np.concatenate((np.max(face_values[:2], axis=0), np.max(face_values[2:], axis=0)))

        component_sensitivity = None
        if componentwise:
            # Per-row version of ``eta_sensitivity_scale``.
            try:
                component_sensitivity = interface_rows(
                    face_transfers_at([brackets[0][1], brackets[1][1]]) - face_transfers_at([brackets[0][0], brackets[1][0]])
                )
            except ValueError:
                component_sensitivity = np.zeros(4, dtype=np.float64)

        def evaluate(velocities, etas):
            interface_compositions = self._interface_compositions(etas)
            residual, zero_velocity_residual, face_transfers = self._initial_discrete_interface_terms(
                interfaces,
                interface_compositions,
                adjacent,
                velocities,
                return_zero_velocity=True,
                return_face_transfers=True,
            )
            flux_delta = -zero_velocity_residual
            # Relative convergence scale: individual face transfers, the
            # velocity-proportional (swept-inventory) part of the residual, and
            # the eta sensitivity of the transfers.
            scale = max(
                float(np.max(np.abs(face_transfers))),
                float(np.max(np.abs(residual - zero_velocity_residual))),
                0.0 if sensitivity_scale is None else sensitivity_scale,
                1e-300,
            )
            component_scales = None
            if componentwise:
                # The same terms per residual row, floored per interface at
                # round-off of that interface's largest term (as in the
                # two-phase estimator).
                component_scales = np.maximum.reduce(
                    [interface_rows(face_transfers), np.abs(residual - zero_velocity_residual), component_sensitivity, np.full(4, 1e-300)]
                )
                for block in (slice(0, 2), slice(2, 4)):
                    component_scales[block] = np.maximum(component_scales[block], component_scale_floor * float(np.max(component_scales[block])))
            return interface_compositions, residual, flux_delta, scale, component_scales

        _, residual_at_zero, _, _, _ = evaluate(np.zeros(2, dtype=np.float64), eta0)
        velocity_columns = np.column_stack(
            [
                evaluate(np.asarray([1.0, 0.0], dtype=np.float64), eta0)[1] - residual_at_zero,
                evaluate(np.asarray([0.0, 1.0], dtype=np.float64), eta0)[1] - residual_at_zero,
            ]
        )
        velocity_scales = np.empty(2, dtype=np.float64)
        for i in range(2):
            scale = float(np.linalg.norm(residual_at_zero) / max(float(np.linalg.norm(velocity_columns[:, i])), 1e-300))
            velocity_scales[i] = scale if np.isfinite(scale) and scale > 0.0 else 1.0
        if self.initialVelocityGuess is None:
            try:
                velocity0_physical = np.linalg.lstsq(velocity_columns, -residual_at_zero, rcond=None)[0]
            except np.linalg.LinAlgError:
                velocity0_physical = np.zeros(2, dtype=np.float64)
        else:
            velocity0_physical = np.asarray(self.initialVelocityGuess, dtype=np.float64).reshape(2)
        velocity0 = velocity0_physical / velocity_scales

        lower = np.asarray([-np.inf, -np.inf, brackets[0][0], brackets[1][0]], dtype=np.float64)
        upper = np.asarray([np.inf, np.inf, brackets[0][1], brackets[1][1]], dtype=np.float64)
        x = np.asarray([velocity0[0], velocity0[1], eta0[0], eta0[1]], dtype=np.float64)
        x = np.clip(x, lower, upper)
        best = None
        success = False
        nfev = 0

        def residual_unknowns(values):
            velocities = np.asarray(values[:2], dtype=np.float64) * velocity_scales
            etas = np.asarray(values[2:], dtype=np.float64)
            interface_compositions, residual, flux_delta, scale, component_scales = evaluate(velocities, etas)
            if not np.all(np.isfinite(residual)):
                raise ValueError("Initial eta residual is non-finite.")
            return residual, flux_delta, interface_compositions, scale, component_scales

        for iteration in range(1, self.initialEtaRootMaxiter + 1):
            residual, flux_delta, interface_compositions, residual_scale, component_scales = residual_unknowns(x)
            nfev += 1
            # Componentwise scaling freezes this iterate's row scales for the
            # Jacobian and line search; re-evaluating them at perturbed points
            # would make the scaled residual saturate as |V| grows. Without it
            # the weights are exactly 1, which leaves the arithmetic unchanged.
            weights = component_scales if componentwise else np.ones(4, dtype=np.float64)
            scaled_residual = residual / weights
            norm_current = float(np.max(np.abs(scaled_residual)))
            if best is None or norm_current < best[0]:
                best = (norm_current, x.copy(), residual.copy(), flux_delta.copy(), interface_compositions, residual_scale, component_scales)
            if (norm_current <= self.initialEtaRootRtol) if componentwise else converged_at(norm_current, residual_scale):
                success = True
                break

            jacobian = np.zeros((4, 4), dtype=np.float64)
            for variable in range(4):
                step, x_perturbed, direction = _bounded_finite_difference_perturbation(x, lower, upper, variable)
                residual_perturbed = residual_unknowns(x_perturbed)[0] / weights
                nfev += 1
                if direction == "forward":
                    jacobian[:, variable] = (residual_perturbed - scaled_residual) / step
                else:
                    jacobian[:, variable] = (scaled_residual - residual_perturbed) / step

            try:
                step = np.linalg.solve(jacobian, -scaled_residual)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jacobian, -scaled_residual, rcond=None)[0]

            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = np.clip(x + scale * step, lower, upper)
                residual_trial, flux_delta_trial, interface_compositions_trial, residual_scale_trial, component_scales_trial = residual_unknowns(trial)
                nfev += 1
                norm_trial = float(np.max(np.abs(residual_trial / weights)))
                if np.isfinite(norm_trial) and norm_trial < norm_current:
                    x = trial
                    accepted = True
                    if best is None or norm_trial < best[0]:
                        best = (
                            norm_trial,
                            trial.copy(),
                            residual_trial.copy(),
                            flux_delta_trial.copy(),
                            interface_compositions_trial,
                            residual_scale_trial,
                            component_scales_trial,
                        )
                    break
            if not accepted:
                break

        if best is None:
            raise ValueError("Instantaneous initial eta solve did not produce a finite residual.")
        _, x_best, residual, flux_delta, interface_compositions, residual_scale, component_scales = best
        residual_norm = float(np.max(np.abs(residual)))
        if componentwise:
            residual_scale = float(np.max(component_scales))
            converged = float(np.max(np.abs(residual / component_scales))) <= self.initialEtaRootRtol
        else:
            if sensitivity_scale is not None:
                residual_scale = max(residual_scale, sensitivity_scale)
            converged = converged_at(residual_norm, residual_scale)
        if not success and not converged:
            raise ValueError("Instantaneous three-phase initial eta solve failed to converge.")
        estimate = ThreePhaseInitialEtaEstimate(
            etas=np.asarray(x_best[2:], dtype=np.float64).copy(),
            velocities=np.asarray(x_best[:2], dtype=np.float64) * velocity_scales,
            residual_norm=float(residual_norm),
            residual=residual.copy(),
            flux_delta=flux_delta.copy(),
            interface_compositions=interface_compositions,
            method="instantaneous_balance",
            solver="damped_newton_4x4",
            brackets=brackets,
            converged=bool(success),
            iterations=int(iteration),
            function_calls=int(nfev),
            residual_scale=float(residual_scale),
            component_scales=None if component_scales is None else np.asarray(component_scales, dtype=np.float64).copy(),
        )
        self.initialEtaEstimate = estimate
        return estimate.etas.copy()

    def _interface_compositions(self, etas):
        """
        Returns validated ``((A_AB, B_AB), (B_BC, C_BC))`` in the state basis for two etas.

        Inside a step attempt (``_try_step_retries``) results are cached per
        interface by the exact eta value: Newton's finite-difference columns
        perturb one variable at a time, so most candidates repeat an eta that
        was already evaluated. Closures are taken to be functions of eta only
        (the model is isothermal); cached vectors are returned as copies.
        Outside an attempt every call evaluates the closures afresh.
        """
        etas = np.asarray(etas, dtype=np.float64).reshape(2)
        cache = self._interfaceCompositionCache
        out = []
        for i, (closure, eta) in enumerate(zip(self.interfaceEquilibria, etas)):
            key = (i, float(eta))
            if cache is not None and key in cache:
                left, right = cache[key]
                out.append((left.copy(), right.copy()))
                continue
            left, right = resolve_interface_equilibrium(closure, self.allElements).interface_compositions(float(eta))
            left = self._validate_composition_vector(left, f"interface {i} left composition")
            right = self._validate_composition_vector(right, f"interface {i} right composition")
            if cache is not None:
                cache[key] = (left.copy(), right.copy())
            out.append((left, right))
        return tuple(out)

    def _interstitial_mask(self):
        """Returns the (fixed) independent-component interstitial mask, computed once."""
        if self._interstitialMaskCache is None:
            self._interstitialMaskCache = independent_interstitial_mask(self.allElements)
        return self._interstitialMaskCache

    def _validate_composition_vector(self, values, name):
        """Rejects a non-finite interface composition or one outside ``_profile_composition_violation`` bounds."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite two-component vector.")
        if composition_bound_violation(values.reshape(1, 2), self._interstitial_mask(), self.constraints.minComposition) is not None:
            raise ValueError(f"{name} violates ternary composition bounds.")
        return values.copy()

    def _validate_profile_compositions(self, profile, name):
        profile = validate_ternary_profile(profile, name)
        violation = self._profile_composition_violation(profile)
        if violation is not None:
            raise ValueError(f"{name} violates ternary composition bounds: {violation}.")

    def _profile_composition_violation(self, profile):
        """
        Returns a short diagnostic for the first simplex violation in a profile.

        Interval solves are linear solves and are not positivity preserving for
        all coupled diffusivity matrices, timestep sizes, and moving-boundary
        guesses. Candidate profiles must therefore be rejected before their
        interface residuals are considered converged. With interstitials the
        simplex applies to the substitutional u-fractions only and interstitial
        u-fractions are bounded below.
        """
        return composition_bound_violation(profile, self._interstitial_mask(), self.constraints.minComposition)

    def _validate_candidate_profiles(self, profiles):
        """Rejects nonlinear trial states whose transformed profiles leave the ternary simplex."""
        for i, profile in enumerate(profiles):
            profile = validate_ternary_profile(profile, f"candidate transformed profile {i}")
            violation = self._profile_composition_violation(profile)
            if violation is not None:
                raise ValueError(f"candidate transformed profile {i} violates ternary composition bounds: {violation}.")

    def _is_infeasible_candidate_error(self, exc):
        """Returns True for candidates rejected by geometry or composition bounds."""
        message = str(exc)
        return isinstance(exc, ValueError) and (
            "candidate transformed profile" in message
            or "violates ternary composition bounds" in message
            or "Three-phase trial geometry" in message
        )

    @_all_or_nothing(skip_when_setup=True)
    def setup(self):
        """
        Initializes the three-phase state once and preserves accepted state on continuation.

        All-or-nothing (LEDGERS R10a): if setup raises (an invalid setting, a
        failing provider or tie-line closure), the exact pre-setup state is
        restored, including ``isSetup=False``, and the exception re-raised.
        External state (provider caches, a created ``mesh.boundaryConditions``)
        is not restored; ``_interfaceCompositionCache`` is model-owned and is
        restored with the other bindings.
        """
        if self.isSetup:
            return
        super().setup()
        self._pendingCandidate = None
        self._validateModelConfiguration()
        self._getBoundaryConditions()
        self._z = flatten_1d_coordinates(self.mesh.z).astype(np.float64)
        if not np.isclose(self._z[0], 0.0):
            raise ValueError("MovingBoundaryIllingworthTernaryThreePhaseFD1DModel expects a 1D domain starting at 0.")
        self._R = float(self._z[-1] - self._z[0])
        self._R0 = float(self._R)
        interfaces0 = self._validate_interfaces(self.initialInterfacePositions, strict=True, right_boundary=self._R0)
        self.initialInterfacePositions = interfaces0.copy()
        self._initialTotalAmount = float(np.dot(self._phaseMolarDensities, self._phase_widths(interfaces0, self._R0)))
        self._grids = self._build_transformed_grids(interfaces0)
        self.profileData = [_ArrayHistory((len(grid), 2), self.interfaceData.recordInterval) for grid in self._grids] if self.recordPqData else None

        c0 = np.asarray(self.data.currentY, dtype=np.float64)
        etas0 = self._estimate_initial_etas(c0, interfaces0)
        profiles = self._initialize_transformed_state(c0, interfaces0, etas0)
        self._profiles_curr = tuple(profile.copy() for profile in profiles)
        self._interfaces_curr = interfaces0.copy()
        self._interfaces_old = interfaces0.copy()
        self._etas_curr = etas0.copy()
        self._etas_old = etas0.copy()
        self._lastAcceptedDt = None
        self._lastInterfaceCompositions = self._interface_compositions(etas0)

        self.interfaceData.reset()
        self.interfaceData.record(0, interfaces0)
        self.etaData.reset()
        self.etaData.record(0, etas0)
        self.rightBoundaryData.reset()
        self.rightBoundaryData.record(0, self._R0)
        if self.recordPqData:
            for history, profile in zip(self.profileData, self._profiles_curr):
                history.record(0, profile)

        physical = self._reconstruct_physical_profile(self._profiles_curr, interfaces0, right_boundary=self._R0)
        self.data.currentY = physical
        self.data._y[0] = physical
        self._initialInventory = self.getTotalInventoryFromState(self._profiles_curr, interfaces0, right_boundary=self._R0)
        self.inventoryData.reset()
        self.inventoryData.record(0, self._initialInventory)
        self.molarInventoryData.reset()
        self.totalMolesData.reset()
        self.dependentMoleClosureErrorData.reset()
        if self._hasExplicitPhaseMolarVolumes:
            self._initialMolarInventories = self._get_all_component_moles_from_state(
                self._profiles_curr, interfaces0, self._R0
            )
            self._initialTotalMoles = self._get_total_substitutional_moles_from_geometry(interfaces0, self._R0)
            closure_error = self._dependent_mole_closure_error(
                self._initialMolarInventories, self._initialTotalMoles
            )
        else:
            self._initialMolarInventories = None
            self._initialTotalMoles = None
            closure_error = np.nan
        self.molarInventoryData.record(
            0, np.full(3, np.nan) if self._initialMolarInventories is None else self._initialMolarInventories
        )
        self.totalMolesData.record(0, np.nan if self._initialTotalMoles is None else self._initialTotalMoles)
        self.dependentMoleClosureErrorData.record(0, closure_error)

    def _build_transformed_grids(self, interfaces):
        if self._inputGrids is not None:
            return tuple(grid.copy() for grid in self._inputGrids)
        if self.phaseNodes is not None:
            counts = self.phaseNodes
        else:
            s_ab, s_bc = interfaces
            counts = (
                max(3, int(np.searchsorted(self._z, s_ab, side="right"))),
                max(3, int(np.searchsorted(self._z, s_bc, side="right") - np.searchsorted(self._z, s_ab, side="left") + 1)),
                max(3, len(self._z) - int(np.searchsorted(self._z, s_bc, side="left"))),
            )
        return tuple(np.linspace(0.0, 1.0, int(count), dtype=np.float64) for count in counts)

    def _initialize_transformed_state(self, composition, interfaces, etas):
        c = np.asarray(composition, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError("Initial ternary composition must have shape (n_nodes, 2).")
        comps = self._interface_compositions(etas)
        boundaries = np.concatenate(([0.0], interfaces, [self._R]))
        profiles = []
        for i, grid in enumerate(self._grids):
            left, right = boundaries[i], boundaries[i + 1]
            z_phase = left + (right - left) * grid
            mask = (self._z >= left) & (self._z <= right)
            if not np.any(mask):
                raise ValueError("Initial interfaces leave an empty phase.")
            profile = np.empty((len(grid), 2), dtype=np.float64)
            for component in range(2):
                profile[:, component] = np.interp(z_phase, self._z[mask], c[mask, component])
            profiles.append(profile)
        profiles[0][-1] = comps[0][0]
        profiles[1][0] = comps[0][1]
        profiles[1][-1] = comps[1][0]
        profiles[2][0] = comps[1][1]
        return tuple(profiles)

    def _reconstruct_physical_profile(self, profiles, interfaces, right_boundary=None):
        """
        Reconstructs on the immutable fixed mesh for an explicit geometry.

        Fixed nodes beyond a contracted right boundary are returned as NaN.
        Expanded material beyond ``_R0`` is intentionally absent here and is
        available from :meth:`getPhysicalPhaseProfiles`.
        """
        domain_length = float(self._R if right_boundary is None else right_boundary)
        self._validate_interfaces(interfaces, strict=True, right_boundary=domain_length)
        inside = self._z <= domain_length
        physical = np.full((len(self._z), 2), np.nan, dtype=np.float64)
        if np.any(inside):
            physical[inside] = reconstruct_planar_transformed_profile_sequence(
                z=self._z[inside],
                profiles=profiles,
                interfaces=interfaces,
                domain_length=domain_length,
                grids=self._grids,
            )
        return physical

    def setTimeInfo(self, currTime, simTime):
        """Stores solve bounds, including the outer wrapper's maximum timestep."""
        super().setTimeInfo(currTime, simTime)
        self._currdt = np.inf
        self._outerMinTimeStep = float(self._solveMinDtFrac) * float(simTime)
        self._outerMaxTimeStep = float(self._solveMaxDtFrac) * float(simTime)
        self._pendingCandidate = None
        self._terminalThinPhaseStop = False
        self._terminalThinPhaseInfo = None
        if self.dtMode != "semi_log" or simTime <= 0:
            self._semiLogTimes = None
            self._semiLogNextIndex = 0
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

    def _updateSemiLogIndex(self, t):
        if self._semiLogTimes is None:
            return
        while self._semiLogNextIndex < len(self._semiLogTimes):
            if self._semiLogTimes[self._semiLogNextIndex] > float(t) + 1e-15:
                break
            self._semiLogNextIndex += 1

    def _computeSemiLogDt(self, t):
        if self.dtMode != "semi_log" or self._semiLogTimes is None:
            return np.inf
        self._updateSemiLogIndex(t)
        if self._semiLogNextIndex >= len(self._semiLogTimes):
            return np.inf
        return max(1e-15, float(self._semiLogTimes[self._semiLogNextIndex] - float(t)))

    def _compute_dt(self, t):
        remaining = getattr(self, "finalTime", np.inf) - float(t)
        if self.dtMode == "semi_log":
            scheduled_dt = self._computeSemiLogDt(t)
            dt = min(scheduled_dt, remaining, self._outerMaxTimeStep)
        else:
            dt = min(self.timeStep, remaining, self._outerMaxTimeStep)
        if not np.isfinite(dt) or dt <= 0:
            dt = self.timeStep
        self._currdt = float(dt)
        return float(dt)

    def _is_final_remainder_below_outer_minimum(self, t, trial_dt):
        """Returns whether ``trial_dt`` is the exact final remainder allowed by the outer solver."""
        remaining = float(getattr(self, "finalTime", np.inf)) - float(t)
        if not np.isfinite(remaining) or remaining <= 0.0 or remaining >= self._outerMinTimeStep:
            return False
        scale = max(1.0, abs(float(t)), abs(float(trial_dt)), abs(remaining))
        atol = 64.0 * np.finfo(float).eps * scale
        return bool(np.isclose(float(trial_dt), remaining, rtol=2e-13, atol=atol))

    def _validate_pending_timestep_against_outer_minimum(self, t, trial_dt):
        """Rejects a subminimum retry only when the outer wrapper would enlarge its timestep."""
        if trial_dt < self._outerMinTimeStep and not self._is_final_remainder_below_outer_minimum(t, trial_dt):
            self._pendingCandidate = None
            raise ValueError(
                f"Converged nonlinear timestep {trial_dt} is smaller than the outer solver minimum "
                f"{self._outerMinTimeStep}; reduce minDtFrac."
            )

    def solve(self, simTime, iterator=explicitEulerIterator, verbose=False, vIt=10, minDtFrac=1e-8, maxDtFrac=1):
        """
        Advances by ``simTime`` through an Euler wrapper without reinitializing accepted state.

        ``simTime`` must be positive and finite. The duration, the timestep
        fractions and the stored numerical controls
        (``_validate_ternary_numerical_controls``) are checked before any
        bookkeeping, setup or schedule allocation, so a control changed after
        construction is refused without modifying the model.
        """
        if iterator is not explicitEulerIterator:
            raise ValueError("MovingBoundaryIllingworthTernaryThreePhaseFD1DModel supports only explicitEulerIterator.")
        if not np.isfinite(simTime) or simTime <= 0.0:
            raise ValueError("simTime must be positive and finite.")
        if not np.isfinite(minDtFrac) or not np.isfinite(maxDtFrac) or minDtFrac < 0.0 or maxDtFrac <= 0.0:
            raise ValueError("minDtFrac must be nonnegative and maxDtFrac must be positive and finite.")
        if minDtFrac > maxDtFrac:
            raise ValueError("minDtFrac cannot exceed maxDtFrac for the implicit three-phase solver.")
        _validate_ternary_numerical_controls(self)
        _validate_default_residual_floor(self)
        self._solveMinDtFrac = float(minDtFrac)
        self._solveMaxDtFrac = float(maxDtFrac)
        return super().solve(simTime, iterator=iterator, verbose=verbose, vIt=vIt, minDtFrac=minDtFrac, maxDtFrac=maxDtFrac)

    def getCurrentX(self):
        return [*(profile.copy() for profile in self._profiles_curr), self._interfaces_curr.copy(), self._etas_curr.copy()]

    def flattenX(self, X):
        return np.concatenate((
            np.asarray(X[0], dtype=np.float64).reshape(-1),
            np.asarray(X[1], dtype=np.float64).reshape(-1),
            np.asarray(X[2], dtype=np.float64).reshape(-1),
            np.asarray(X[3], dtype=np.float64).reshape(2),
            np.asarray(X[4], dtype=np.float64).reshape(2),
        ))

    def unflattenX(self, X_flat, X_ref):
        sizes = [np.asarray(X_ref[i], dtype=np.float64).size for i in range(3)]
        cursor = 0
        profiles = []
        for i, size in enumerate(sizes):
            profiles.append(np.asarray(X_flat[cursor : cursor + size], dtype=np.float64).reshape(np.asarray(X_ref[i]).shape))
            cursor += size
        interfaces = np.asarray(X_flat[cursor : cursor + 2], dtype=np.float64)
        cursor += 2
        etas = np.asarray(X_flat[cursor : cursor + 2], dtype=np.float64)
        return [profiles[0], profiles[1], profiles[2], interfaces, etas]

    def _identity(self):
        return np.eye(2, dtype=np.float64)

    def _phase_face_diffusivity_matrices(self, values, n_faces, phase, context="transient face diffusivity"):
        values = np.asarray(values)
        if values.shape == (2, 2):
            matrix = _validate_ternary_diffusivity_matrix(values, phase, context=context)
            return np.broadcast_to(matrix, (n_faces, 2, 2)).copy()
        if values.shape != (n_faces, 2, 2):
            raise ValueError(f"{context} for phase {phase} must have shape (2, 2) or ({n_faces}, 2, 2); received {values.shape}.")
        out = np.empty(values.shape, dtype=np.float64)
        for i, matrix in enumerate(values):
            out[i] = _validate_ternary_diffusivity_matrix(matrix, phase, context=f"{context} face {i}")
        return out

    def _temperatures_at_positions(self, positions, time):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1)
        T = np.asarray(self.temperatureParameters(positions.reshape(-1, 1), float(time)), dtype=np.float64).reshape(-1)
        if T.size == 1 and positions.size != 1:
            T = np.full(positions.shape, float(T[0]), dtype=np.float64)
        if T.size != positions.size or not np.all(np.isfinite(T)):
            raise ValueError("Bulk face temperatures must be finite and match the face count.")
        return T

    def _bulk_face_diffusivity_matrices(self, face_compositions, phase, time, physical_face_positions):
        """
        Returns validated bulk diffusivity matrices for interval faces.

        ``face_compositions`` are in the state basis (u-fractions with
        interstitials); the provider receives mole fractions. Vectorized
        provider results are validated in one batch
        (``_validate_ternary_diffusivity_stack``); scalar-only providers keep
        the face-by-face query and validation.
        """
        face_compositions = np.asarray(face_compositions, dtype=np.float64)
        source = self._diffusivity_source()
        if face_compositions.ndim != 2 or face_compositions.shape[1] != 2:
            raise ValueError("face_compositions must have shape (n_faces, 2).")
        temperatures = self._temperatures_at_positions(physical_face_positions, time)

        def validate_stack(values):
            matrices = np.asarray(values)
            if matrices.shape == (2, 2) and face_compositions.shape[0] == 1:
                matrices = matrices.reshape(1, 2, 2)
            if matrices.shape != (face_compositions.shape[0], 2, 2):
                raise ValueError("bulk diffusivity query returned an unexpected matrix shape.")
            # Batched spectral checks; exceptional stacks fall back to the
            # face-by-face validator, so accepted matrices and errors are unchanged.
            return _validate_ternary_diffusivity_stack(matrices, phase)

        try:
            values = source.getInterdiffusivity(face_compositions, temperatures, phase=phase, query_context="general")
        except (TypeError, ValueError):
            pass
        else:
            return validate_stack(values)

        out = np.empty((face_compositions.shape[0], 2, 2), dtype=np.float64)
        for i, (composition, temperature) in enumerate(zip(face_compositions, temperatures)):
            try:
                D = source.getInterdiffusivity(composition, float(temperature), phase=phase, query_context="general")
            except TypeError:
                D = source.getInterdiffusivity(composition, float(temperature), phase=phase)
            out[i] = _validate_ternary_diffusivity_matrix(D, phase, context=f"bulk face diffusivity face {i}")
        return out

    def _phase_uniform_diffusivity(self, phase_index, interface_compositions, interfaces):
        if phase_index == 0:
            composition = interface_compositions[0][0]
            position = interfaces[0]
        elif phase_index == 1:
            composition = 0.5 * (interface_compositions[0][1] + interface_compositions[1][0])
            position = 0.5 * (interfaces[0] + interfaces[1])
        else:
            composition = interface_compositions[1][1]
            position = interfaces[1]
        T = np.asarray(self.temperatureParameters(np.asarray([[float(position)]], dtype=np.float64), self.currentTime), dtype=np.float64).reshape(-1)
        source = self._diffusivity_source()
        try:
            D = source.getInterdiffusivity(composition, float(T[0]), phase=self.phases[phase_index], query_context="interface")
        except TypeError:
            D = source.getInterdiffusivity(composition, float(T[0]), phase=self.phases[phase_index])
        return _validate_ternary_diffusivity_matrix(D, self.phases[phase_index], context="three-phase phase-uniform diffusivity")

    def _face_positions(self, grid, bounds):
        face_xi = 0.5 * (grid[:-1] + grid[1:])
        return float(bounds[0]) + (float(bounds[1]) - float(bounds[0])) * face_xi

    def _lagged_face_compositions(self, profile):
        return 0.5 * (np.asarray(profile, dtype=np.float64)[:-1] + np.asarray(profile, dtype=np.float64)[1:])

    def _interval_face_diffusivity(self, profile, grid, phase_index, new_bounds):
        return self._bulk_face_diffusivity_matrices(
            self._lagged_face_compositions(profile),
            self.phases[phase_index],
            self.currentTime,
            self._face_positions(grid, new_bounds),
        )

    def _lagged_interval_face_diffusivity(self, profile, phase_index, new_bounds):
        """
        Returns one phase's lagged face matrices, reused within a step attempt.

        In ``composition_dependent_lagged`` mode the matrices depend only on the
        accepted old profile: the model is isothermal, so the trial face
        positions (``new_bounds``) do not change the temperature.
        ``_try_step_retries`` opens a cache for each attempt so that all Newton
        candidates share one evaluation per phase; an entry is reused only for
        the same profile object. Without an open cache every call evaluates
        afresh.
        """
        cache = self._laggedFaceDiffusivityCache
        if cache is not None:
            entry = cache.get(phase_index)
            if entry is not None and entry[0] is profile:
                return entry[1]
        D = self._interval_face_diffusivity(profile, self._grids[phase_index], phase_index, new_bounds)
        if cache is not None:
            cache[phase_index] = (profile, D)
        return D

    def _internal_face_displacements(self, grid, old_bounds, new_bounds, phase_frame_displacement=0.0):
        """Returns each internal grid-face displacement relative to its phase frame."""
        grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        face_xi = 0.5 * (grid[:-1] + grid[1:])
        old_left, old_right = float(old_bounds[0]), float(old_bounds[1])
        new_left, new_right = float(new_bounds[0]), float(new_bounds[1])
        grid_displacement = (1.0 - face_xi) * (new_left - old_left) + face_xi * (new_right - old_right)
        return grid_displacement - float(phase_frame_displacement)

    def _face_transfer_coefficients(self, delta_x, D_face, dt, length, dxi):
        """
        Returns node coefficients for one conservative ALE plus diffusive face.

        The returned pair ``(left_coeff, right_coeff)`` satisfies
        ``H_f = left_coeff @ C_j + right_coeff @ C_{j+1}`` for a face between
        transformed nodes ``j`` and ``j + 1``. Positive face displacement uses
        the right node as the ALE donor because the transformed advection
        velocity has the opposite sign to physical grid motion.
        """
        I = self._identity()
        A = float(dt) * np.asarray(D_face, dtype=np.float64) / (float(length) * float(dxi))
        left_coeff = -A
        right_coeff = A.copy()
        delta_x = float(delta_x)
        if delta_x > 0.0:
            right_coeff = right_coeff + delta_x * I
        elif delta_x < 0.0:
            left_coeff = left_coeff + delta_x * I
        return left_coeff, right_coeff

    def _evaluate_face_transfer(self, left_coeff, right_coeff, left_value, right_value):
        """Evaluates a preassembled total face transfer from its node coefficients."""
        return np.matmul(left_coeff, np.asarray(left_value, dtype=np.float64)) + np.matmul(right_coeff, np.asarray(right_value, dtype=np.float64))

    def _solve_interval_planar(
        self,
        profile,
        grid,
        old_bounds,
        new_bounds,
        phase_index,
        left_value,
        right_value,
        D_faces,
        validate_diffusivity=True,
        phase_frame_displacement=0.0,
    ):
        """
        Solves one transformed planar interval on a moving Landau grid.

        The finite-volume equation is assembled face by face as
        ``L_new*DeltaXi*C_new - L_old*DeltaXi*C_old = H_right - H_left``.
        ``H`` contains both the face-local upwind ALE transfer, based on grid
        displacement relative to the phase frame, and the implicit diffusive
        transfer. Endpoint values are exact Dirichlet constraints when supplied;
        ``None`` means the material external boundary has homogeneous zero
        relative transfer.

        Deviation form (ledger R06): the system is solved for
        ``w = C_new - r`` with the reference profile ``r`` of
        ``_interval_reference`` (the interface composition for an outer phase,
        linear in ``xi`` between the two interface compositions for the middle
        phase). Every free row sums to ``L_old*DeltaXi*I`` (the face
        displacements telescope and the diffusive terms cancel), so the matrix
        is unchanged and free rows get
        ``rhs_i = L_old*DeltaXi_i*(C_old_i - r_i) - lower_i@dr_{i-1} - upper_i@dr_{i+1}``
        with ``dr_{i+-1} = (xi_{i+-1} - xi_i)*(c_R - c_L)`` formed from grid
        differences (zero for the outer phases); Dirichlet rows get ``0``. The
        right-hand side is assembled in that form, never as ``b - M r``. In
        exact arithmetic the solution is the same as solving for ``C_new``;
        in floating point the interface transfers no longer cancel absolute
        compositions (see ``_finish_interval_solve``). Diffusivities passed in
        ``D_faces`` are evaluated by the callers from absolute compositions.

        With Numba the system is assembled by the compiled
        ``_fill_three_phase_interval_system`` (same operations in the same
        order, so results are bitwise identical); otherwise by the vectorized
        NumPy code below. Both receive ``C_old - r`` and zero fixed values; the
        middle-phase ``dr`` correction is applied afterwards by the same NumPy
        code on either path.
        """
        profile = validate_ternary_profile(profile, "interval profile")
        grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        n = len(profile)
        reference, delta = self._interval_reference(grid, n, left_value, right_value)
        shifted = profile - reference
        zero = np.zeros(2, dtype=np.float64)
        reference_info = (reference, delta, grid, old_bounds, new_bounds, phase_frame_displacement)
        D_values = np.asarray(D_faces)
        D_faces = self._phase_face_diffusivity_matrices(D_values, n - 1, self.phases[phase_index]) if validate_diffusivity else np.asarray(D_values, dtype=np.float64)
        if D_faces.shape == (2, 2):
            D_faces = np.broadcast_to(D_faces, (n - 1, 2, 2)).copy()

        old_left, old_right = float(old_bounds[0]), float(old_bounds[1])
        new_left, new_right = float(new_bounds[0]), float(new_bounds[1])
        old_length = old_right - old_left
        new_length = new_right - new_left
        if old_length <= 0.0 or new_length <= 0.0:
            raise ValueError("Transformed interval length must remain positive.")
        if _fill_three_phase_interval_system is not None and self._useCompiledIntervalAssembly:
            lower = np.zeros((n, 2, 2), dtype=np.float64)
            diagonal = np.zeros((n, 2, 2), dtype=np.float64)
            upper = np.zeros((n, 2, 2), dtype=np.float64)
            rhs = np.zeros((n, 2), dtype=np.float64)
            end_coefficients = np.empty((2, 2, 2, 2), dtype=np.float64)
            _fill_three_phase_interval_system(
                lower, diagonal, upper, rhs, end_coefficients, shifted, grid,
                old_left, old_right, new_left, new_right, float(phase_frame_displacement),
                float(self._currdt), np.ascontiguousarray(D_faces, dtype=np.float64),
                left_value is not None, zero,
                right_value is not None, zero,
            )
            face_coefficients = ((end_coefficients[0, 0], end_coefficients[0, 1]), (end_coefficients[1, 0], end_coefficients[1, 1]))
            self._apply_reference_slope_correction(lower, upper, rhs, grid, delta)
            return self._finish_interval_solve(lower, diagonal, upper, rhs, face_coefficients, D_faces, left_value, right_value, reference_info)
        face_xi = np.empty(n + 1, dtype=np.float64)
        face_xi[0] = 0.0
        face_xi[-1] = 1.0
        face_xi[1:-1] = 0.5 * (grid[:-1] + grid[1:])
        old_widths = old_length * (face_xi[1:] - face_xi[:-1])
        new_widths = new_length * (face_xi[1:] - face_xi[:-1])

        lower = np.zeros((n, 2, 2), dtype=np.float64)
        diagonal = np.zeros((n, 2, 2), dtype=np.float64)
        upper = np.zeros((n, 2, 2), dtype=np.float64)
        rhs = np.zeros((n, 2), dtype=np.float64)
        I = self._identity()
        fixed_rows = np.zeros(n, dtype=bool)
        if left_value is not None:
            fixed_rows[0] = True
        if right_value is not None:
            fixed_rows[-1] = True

        free_rows = ~fixed_rows
        diagonal[free_rows] = new_widths[free_rows, None, None] * I
        rhs[free_rows] = old_widths[free_rows, None] * shifted[free_rows]
        if right_value is not None:
            diagonal[-1] = I
            rhs[-1] = zero
        if left_value is not None:
            diagonal[0] = I
            rhs[0] = zero

        face_displacements = self._internal_face_displacements(
            grid,
            old_bounds,
            new_bounds,
            phase_frame_displacement=phase_frame_displacement,
        )
        # Vectorized form of ``_face_transfer_coefficients`` for every face, with
        # the same per-element arithmetic. Row ``i`` receives face ``i - 1``
        # (as its right row) before face ``i`` (as its left row), matching the
        # face-ordered accumulation of the scalar loop.
        A = float(self._currdt) * np.asarray(D_faces, dtype=np.float64) / (float(new_length) * np.diff(grid)[:, None, None])
        left_coeffs = -A
        right_coeffs = A.copy()
        face_displacements = np.asarray(face_displacements, dtype=np.float64)
        positive = face_displacements > 0.0
        negative = face_displacements < 0.0
        right_coeffs[positive] = right_coeffs[positive] + face_displacements[positive, None, None] * I
        left_coeffs[negative] = left_coeffs[negative] + face_displacements[negative, None, None] * I
        free_right = ~fixed_rows[1:]
        free_left = ~fixed_rows[:-1]
        lower[1:][free_right] += left_coeffs[free_right]
        diagonal[1:][free_right] += right_coeffs[free_right]
        diagonal[:-1][free_left] -= left_coeffs[free_left]
        upper[:-1][free_left] -= right_coeffs[free_left]
        face_coefficients = ((left_coeffs[0], right_coeffs[0]), (left_coeffs[-1], right_coeffs[-1]))
        self._apply_reference_slope_correction(lower, upper, rhs, grid, delta)
        return self._finish_interval_solve(lower, diagonal, upper, rhs, face_coefficients, D_faces, left_value, right_value, reference_info)

    def _interval_reference(self, grid, n, left_value, right_value):
        """
        Returns the deviation-form reference profile ``r`` and slope ``c_R - c_L``.

        An outer phase (one Dirichlet end) uses its interface composition at
        every node, returned as a single 2-vector that broadcasts over the
        nodes, and ``delta=None``. The middle phase uses the ``(n, 2)`` profile
        ``r_i = c_L + xi_i*(c_R - c_L)`` with the end values set to ``c_L`` and
        ``c_R`` exactly, so ``r + 0`` reproduces both Dirichlet values bitwise.
        Interior ``r_i`` are rounded; the solve never uses differences of them
        (see ``_apply_reference_slope_correction``). Without Dirichlet ends the
        reference is zero (absolute form).
        """
        if left_value is not None and right_value is not None:
            c_left = np.asarray(left_value, dtype=np.float64).reshape(2)
            c_right = np.asarray(right_value, dtype=np.float64).reshape(2)
            delta = c_right - c_left
            reference = c_left + grid[:, None] * delta
            reference[0] = c_left
            reference[-1] = c_right
            return reference, delta
        if right_value is not None:
            value = right_value
        elif left_value is not None:
            value = left_value
        else:
            return np.zeros(2, dtype=np.float64), None
        return np.asarray(value, dtype=np.float64).reshape(2), None

    def _apply_reference_slope_correction(self, lower, upper, rhs, grid, delta):
        """
        Adds the middle-phase reference-slope terms to the deviation right-hand side.

        For free rows ``i = 1 .. n-2`` this subtracts
        ``lower_i@dr_{i-1} + upper_i@dr_{i+1}`` with
        ``dr_{i-1} = -(xi_i - xi_{i-1})*delta`` and
        ``dr_{i+1} = (xi_{i+1} - xi_i)*delta``, using the assembled
        off-diagonal blocks. The ``dr`` come from exact-input grid differences,
        not from differences of the rounded reference values. Nothing is done
        for an outer phase (``delta is None``), whose reference is constant.
        """
        if delta is None or len(grid) < 3:
            return
        dxi = grid[1:] - grid[:-1]
        dr_lower = -(dxi[:-1, None] * delta)
        dr_upper = dxi[1:, None] * delta
        # Explicit 2x2 block-vector products (batched matmul is slow for 2x2 stacks).
        lo, up = lower[1:-1], upper[1:-1]
        rhs[1:-1] -= lo[:, :, 0] * dr_lower[:, 0, None] + lo[:, :, 1] * dr_lower[:, 1, None]
        rhs[1:-1] -= up[:, :, 0] * dr_upper[:, 0, None] + up[:, :, 1] * dr_upper[:, 1, None]

    def _finish_interval_solve(self, lower, diagonal, upper, rhs, face_coefficients, D_faces, left_value, right_value, reference_info):
        """
        Solves an assembled deviation system and evaluates its interface transfers.

        ``reference_info = (r, delta, grid, old_bounds, new_bounds, frame)``
        describes the reference of ``_interval_reference`` and the geometry.
        The block solve returns ``w``; the retained profile is ``r + w``
        (Dirichlet nodes exactly ``r``), followed by the floor projection of
        ``_project_floor_roundoff``.

        The end transfers use ``(L_f + R_f) = d_f I`` exactly, with the face
        displacement ``d_f`` evaluated as in ``_internal_face_displacements`` (not
        ``L_f + R_f`` recomputed in floating point), so they contain no
        difference of absolute compositions:
        ``H_left = d_0 c_L + R_0@(dr_1 + w_1)`` and
        ``H_right = d_f c_R + L_f@(w_{n-2} - dr_{n-1})`` with
        ``dr_1 = (xi_1 - xi_0)*delta`` and ``dr_{n-1} = (xi_{n-1} - xi_{n-2})*delta``
        (zero for the outer phases, giving ``d c + R@w`` and ``d c + L@w``).
        Their round-off is about ``eps*|H|`` instead of ``eps*|A|*|c|`` with
        ``A = dt D / (L dxi)``.
        """
        reference, delta, grid, old_bounds, new_bounds, frame = reference_info
        deviation = solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)
        n = deviation.shape[0]
        fixed_nodes = []
        if left_value is not None:
            deviation[0] = 0.0
            fixed_nodes.append(0)
        if right_value is not None:
            deviation[-1] = 0.0
            fixed_nodes.append(n - 1)
        profile = reference + deviation
        left_matrix = None if left_value is None else D_faces[0]
        right_matrix = None if right_value is None else D_faces[-1]
        # End-face displacements with the same scalar operations as
        # ``_internal_face_displacements``.
        old_left, old_right = float(old_bounds[0]), float(old_bounds[1])
        new_left, new_right = float(new_bounds[0]), float(new_bounds[1])
        d_left, d_right, frame = new_left - old_left, new_right - old_right, float(frame)
        if left_value is None:
            left_transfer = np.zeros(2, dtype=np.float64)
        else:
            face_xi = 0.5 * (float(grid[0]) + float(grid[1]))
            displacement = (1.0 - face_xi) * d_left + face_xi * d_right - frame
            step = deviation[1] if delta is None else (grid[1] - grid[0]) * delta + deviation[1]
            left_transfer = displacement * np.asarray(left_value, dtype=np.float64) + np.matmul(face_coefficients[0][1], step)
        if right_value is None:
            right_transfer = np.zeros(2, dtype=np.float64)
        else:
            face_xi = 0.5 * (float(grid[-2]) + float(grid[-1]))
            displacement = (1.0 - face_xi) * d_left + face_xi * d_right - frame
            step = deviation[-2] if delta is None else deviation[-2] - (grid[-1] - grid[-2]) * delta
            right_transfer = displacement * np.asarray(right_value, dtype=np.float64) + np.matmul(face_coefficients[-1][0], step)
        new_length = new_right - new_left
        profile, projection, split = self._project_floor_roundoff(profile, reference, deviation, fixed_nodes, grid, new_length, left_value is not None, right_value is not None)
        return _ThreePhaseBulkResult(profile, left_transfer, right_transfer, left_matrix, right_matrix, projection, split)

    def _interval_cell_widths(self, grid, length):
        """Returns the new-geometry cell widths ``L*DeltaXi_i`` used by the interval assembly."""
        face_xi = np.empty(len(grid) + 1, dtype=np.float64)
        face_xi[0] = 0.0
        face_xi[-1] = 1.0
        face_xi[1:-1] = 0.5 * (grid[:-1] + grid[1:])
        return float(length) * (face_xi[1:] - face_xi[:-1])

    def _project_floor_roundoff(self, profile, reference, deviation, fixed_nodes, grid, length, has_left_interface, has_right_interface):
        """
        Applies the round-off floor projection to one deviation-form interval profile.

        Uses the two-phase ``_project_shift_roundoff_to_floor`` with the per-node
        reference ``r`` as shift (threshold
        ``_FLOOR_PROJECTION_ULPS * eps * (|r_i| + |w_i|)``, a fixed policy, not
        an error bound), never touching the Dirichlet nodes, and with this
        model's cell widths as inventory weights, so the change matches the
        cell equations. Larger violations are left for the composition
        validator. Returns ``(profile, projection, split)``; ``split`` rows are
        the parts of ``projection.inventory_change`` attributed to the phase's
        left and right interface residual: all of it to the single interface
        of an outer phase, and node by node ``(1 - xi_i, xi_i)`` for the middle
        phase. Every projecting solve is counted at once in
        ``_currentStepProjectedSolves`` and ``_cumulativeProjectedSolves``
        (intermediate Picard solves and discarded candidates included).
        """
        no_split = np.zeros((2, 2), dtype=np.float64)
        substitutional = ~np.asarray(self._interstitial_mask(), dtype=bool)
        floor = float(self.constraints.minComposition)
        if _profile_within_floor_bounds(profile, substitutional, floor):
            return profile, _no_floor_projection(), no_split
        weights = self._interval_cell_widths(grid, length)
        projected, projection = _project_shift_roundoff_to_floor(
            profile, reference, deviation, self._interstitial_mask(), floor, fixed_nodes, grid, length, weights=weights,
        )
        if projection.mask is None:
            return projected, projection, no_split
        self._currentStepProjectedSolves = getattr(self, "_currentStepProjectedSolves", 0) + 1
        self._cumulativeProjectedSolves = getattr(self, "_cumulativeProjectedSolves", 0) + 1
        if has_left_interface and has_right_interface:
            left_fraction = 1.0 - grid
        elif has_left_interface:
            left_fraction = np.ones_like(grid)
        else:
            left_fraction = np.zeros_like(grid)
        split = np.empty((2, 2), dtype=np.float64)
        split[0] = (weights * left_fraction) @ (projected - profile)
        split[1] = projection.inventory_change - split[0]
        return projected, projection, split

    def _solve_interval_picard(self, profile, grid, old_bounds, new_bounds, phase_index, left_value, right_value, phase_frame_displacement=0.0):
        """
        Solves one interval with composition-dependent Picard face matrices.

        Face diffusivities are queried from, and the update norm is measured
        on, absolute compositions; each linear solve is in deviation form. The
        converged result forwards the final linear solve's deviation-form
        transfers, profile, ``projection`` and ``projection_split``.
        """
        iterate = np.asarray(profile, dtype=np.float64).copy()
        if left_value is not None:
            iterate[0] = left_value
        if right_value is not None:
            iterate[-1] = right_value
        update_norm = np.inf
        for iteration in range(1, self.bulkPicardMaxIterations + 1):
            D_faces = self._interval_face_diffusivity(iterate, grid, phase_index, new_bounds)
            linear = self._solve_interval_planar(
                profile,
                grid,
                old_bounds,
                new_bounds,
                phase_index,
                left_value,
                right_value,
                D_faces,
                validate_diffusivity=False,
                phase_frame_displacement=phase_frame_displacement,
            )
            update = linear.profile - iterate
            update_norm = float(np.max(np.abs(update)))
            scale = max(1e-12, float(np.max(np.abs(linear.profile))))
            threshold = self.bulkPicardAtol + float(self.tolerance if self.bulkPicardRtol is None else self.bulkPicardRtol) * scale
            if update_norm <= threshold:
                return _ThreePhaseBulkResult(
                    linear.profile,
                    linear.left_transfer,
                    linear.right_transfer,
                    linear.left_face_matrix,
                    linear.right_face_matrix,
                    linear.projection,
                    linear.projection_split,
                    inner_iterations=iteration,
                    inner_update_norm=update_norm,
                    diffusivity_evaluations=len(profile) - 1,
                )
            iterate = iterate + self.bulkPicardRelaxation * update
        raise RuntimeError(f"bulk Picard solve failed to converge for phase {self.phases[phase_index]} after {self.bulkPicardMaxIterations} iterations")

    def _solve_bulk_profiles(self, profiles, old_interfaces, new_interfaces, interface_compositions, kinematics=None):
        """
        Solves all phases after validating complete old and trial geometries.

        The geometry is validated by ``_step_kinematics`` (both interface pairs,
        strictly, against their right boundaries); a supplied ``kinematics``
        must come from it for the same interfaces.
        """
        kinematics = self._step_kinematics(old_interfaces, new_interfaces) if kinematics is None else kinematics
        old_bounds = np.asarray(
            [[0.0, old_interfaces[0]], [old_interfaces[0], old_interfaces[1]], [old_interfaces[1], kinematics.old_right_boundary]],
            dtype=np.float64,
        )
        new_bounds = np.asarray(
            [[0.0, new_interfaces[0]], [new_interfaces[0], new_interfaces[1]], [new_interfaces[1], kinematics.new_right_boundary]],
            dtype=np.float64,
        )
        boundary_values = (
            (None, interface_compositions[0][0]),
            (interface_compositions[0][1], interface_compositions[1][0]),
            (interface_compositions[1][1], None),
        )
        results = []
        for phase_index in range(3):
            left_value, right_value = boundary_values[phase_index]
            phase_displacement = kinematics.phase_displacements[phase_index]
            if self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_PHASE_UNIFORM:
                D = self._phase_uniform_diffusivity(phase_index, interface_compositions, new_interfaces)
                result = self._solve_interval_planar(
                    profiles[phase_index], self._grids[phase_index], old_bounds[phase_index], new_bounds[phase_index],
                    phase_index, left_value, right_value, D, phase_frame_displacement=phase_displacement,
                )
            elif self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_LAGGED:
                D = self._lagged_interval_face_diffusivity(profiles[phase_index], phase_index, new_bounds[phase_index])
                result = self._solve_interval_planar(
                    profiles[phase_index], self._grids[phase_index], old_bounds[phase_index], new_bounds[phase_index],
                    phase_index, left_value, right_value, D, validate_diffusivity=False,
                    phase_frame_displacement=phase_displacement,
                )
            elif self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_IMPLICIT:
                result = self._solve_interval_picard(
                    profiles[phase_index], self._grids[phase_index], old_bounds[phase_index], new_bounds[phase_index],
                    phase_index, left_value, right_value, phase_frame_displacement=phase_displacement,
                )
            else:
                raise ValueError(f"Unsupported bulk diffusivity mode {self.bulkDiffusivityMode}.")
            results.append(result)
        return tuple(results)

    def _interface_residuals(self, old_profiles, old_interfaces, new_interfaces, interface_compositions, bulk_results, kinematics=None):
        """
        Returns interface residuals from the same discrete transfers as the bulk.

        Endpoint inventory is represented by the trapezoidal endpoint half-cell
        weights. Each phase block is weighted by ``rho_alpha / rho_A`` before
        combining the exact interface-adjacent transfers returned by the bulk
        solve. Thus the two residual blocks telescope exactly with the
        density-weighted conservative bulk cell equations. The transfers come
        from the deviation-form solves (ledger R06), so they carry no
        ``eps*|A|*|c|`` cancellation error.

        Each phase's round-off floor projection (``projection_split`` of its
        bulk result) is added with the phase's density weight: an outer
        phase's change to its single interface, the middle phase's change
        split node by node as ``(1 - xi_i, xi_i)`` between A|B and B|C. The
        converged residual therefore describes the stored profiles, and the
        two blocks still add up to the total density-weighted inventory change.
        """
        kinematics = self._step_kinematics(old_interfaces, new_interfaces) if kinematics is None else kinematics
        c_a_ab, c_b_ab = interface_compositions[0]
        c_b_bc, c_c_bc = interface_compositions[1]
        old_lengths = np.asarray(
            [
                float(old_interfaces[0]),
                float(old_interfaces[1] - old_interfaces[0]),
                float(kinematics.old_right_boundary - old_interfaces[1]),
            ],
            dtype=np.float64,
        )
        new_lengths = np.asarray(
            [
                float(new_interfaces[0]),
                float(new_interfaces[1] - new_interfaces[0]),
                float(kinematics.new_right_boundary - new_interfaces[1]),
            ],
            dtype=np.float64,
        )
        endpoint_a_right = self._endpoint_inventory_change(self._grids[0], old_profiles[0], c_a_ab, old_lengths[0], new_lengths[0], "right")
        endpoint_b_left = self._endpoint_inventory_change(self._grids[1], old_profiles[1], c_b_ab, old_lengths[1], new_lengths[1], "left")
        endpoint_b_right = self._endpoint_inventory_change(self._grids[1], old_profiles[1], c_b_bc, old_lengths[1], new_lengths[1], "right")
        endpoint_c_left = self._endpoint_inventory_change(self._grids[2], old_profiles[2], c_c_bc, old_lengths[2], new_lengths[2], "left")
        density_a, density_b, density_c = self._normalizedPhaseMolarDensities
        residual_ab = density_a * (endpoint_a_right + bulk_results[0].right_transfer)
        residual_ab += density_b * (endpoint_b_left - bulk_results[1].left_transfer)
        residual_bc = density_b * (endpoint_b_right + bulk_results[1].right_transfer)
        residual_bc += density_c * (endpoint_c_left - bulk_results[2].left_transfer)
        residual_ab += density_a * bulk_results[0].projection_split[1] + density_b * bulk_results[1].projection_split[0]
        residual_bc += density_b * bulk_results[1].projection_split[1] + density_c * bulk_results[2].projection_split[0]
        return np.concatenate((residual_ab, residual_bc))

    def _endpoint_inventory_change(self, grid, old_profile, new_value, old_length, new_length, side):
        """Returns the transformed trapezoidal endpoint inventory change."""
        grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        if side == "left":
            weight = 0.5 * grid[1]
            old_value = np.asarray(old_profile[0], dtype=np.float64)
        elif side == "right":
            weight = 0.5 * (1.0 - grid[-2])
            old_value = np.asarray(old_profile[-1], dtype=np.float64)
        else:
            raise ValueError("endpoint side must be 'left' or 'right'.")
        return weight * (float(new_length) * np.asarray(new_value, dtype=np.float64) - float(old_length) * old_value)

    def _residual_scale(self, profiles, interfaces):
        """
        Returns the four interface-residual scales for one implicit step.

        The residual is a density-weighted inventory defect (composition times
        length). With ``I~_k = sum_alpha (rho_alpha / rho_A) w_alpha <c_k>_alpha``
        the accepted old inventory of component ``k`` (physical moles are
        ``rho_A * I~_k``) and ``L`` the normalized specimen length (``R0`` with
        equal molar volumes), the two rows of component ``k`` sum to the
        step's change of ``I~_k`` up to round-off. Both rows use the same
        ``scale_k``, so each row is bounded by ``residualTolerance * scale_k``
        and the component's inventory change per step by twice that. The rows
        can cancel in the sum while each misplaces solute between the
        interfaces, which is why both are tested and not only their sum. The
        dependent component has no row; its error is minus the sum of the
        substitutional independent ones.

        Default (no ``interface_residual_composition_scale``; ledger R13):
        ``scale_k = max(|I~_k|, L * constraints.minComposition)``
        (``_componentwise_residual_scale``, the binary model's rule per
        component), so the bounds are relative to the component's own
        inventory, however dilute. Where the floor binds, the bound on the
        component's total is an absolute change of mean composition of
        ``2 * residualTolerance * minComposition``. ``minComposition`` is read
        here, once per step; an exactly zero inventory with a zero floor gives
        ``64 eps L`` and non-finite or non-positive results raise
        ``ValueError``.

        With ``interface_residual_composition_scale`` (``s_k`` per component)
        the earlier expression is kept bit for bit:
        ``scale_k = max(|I~_k|, L * s_k, 1e-300)``. ``s_k = 1`` reproduces the
        default used before R13 (every row divided by ``L``, a bound on the
        change of mean composition of ``2 * residualTolerance``) while
        ``|I~_k| <= L``. The attribute may hold a scalar, list, tuple or array
        (also when assigned after construction); the arithmetic uses a local
        validated ``float64`` pair (``_explicit_residual_composition_scale``)
        and never changes it. The scale does not depend on ``dt``.
        """
        normalized_total = float(self._R0) if self._equalPhaseMolarVolumes else float(self._initialTotalAmount / self._phaseMolarDensities[0])
        widths = self._phase_widths(interfaces, right_boundary=self._right_boundary_from_interfaces(interfaces))
        inventory = np.zeros(2, dtype=np.float64)
        for density, width, profile, grid in zip(self._normalizedPhaseMolarDensities, widths, profiles, self._grids):
            inventory += density * width * np.trapezoid(np.asarray(profile, dtype=np.float64), grid, axis=0)
        if self.interfaceResidualCompositionScale is None:
            return np.tile(_componentwise_residual_scale(inventory, normalized_total, self.constraints.minComposition), 2)
        composition_scale = _explicit_residual_composition_scale(self.interfaceResidualCompositionScale)
        component = np.maximum(np.maximum(np.abs(inventory), normalized_total * composition_scale), 1e-300)
        return np.tile(component, 2)

    def _scaled_bounds(self):
        eps = 1e-14
        if self._equalPhaseMolarVolumes:
            position_upper = 1.0 - eps
        else:
            maximum_length = float(self._initialTotalAmount / np.min(self._phaseMolarDensities))
            position_upper = maximum_length / float(self._R0) - eps
        return np.asarray([eps, eps, 0.0, 0.0], dtype=np.float64), np.asarray([position_upper, position_upper, 1.0, 1.0], dtype=np.float64)

    def _physical_to_scaled(self, interfaces, etas):
        bounds = self._eta_bounds()
        return np.asarray(
            [
                interfaces[0] / self._R0,
                interfaces[1] / self._R0,
                (etas[0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]),
                (etas[1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]),
            ],
            dtype=np.float64,
        )

    def _scaled_to_physical(self, x_hat):
        x_hat = np.asarray(x_hat, dtype=np.float64).reshape(4)
        bounds = self._eta_bounds()
        interfaces = np.asarray([x_hat[0] * self._R0, x_hat[1] * self._R0], dtype=np.float64)
        etas = np.asarray(
            [
                bounds[0][0] + x_hat[2] * (bounds[0][1] - bounds[0][0]),
                bounds[1][0] + x_hat[3] * (bounds[1][1] - bounds[1][0]),
            ],
            dtype=np.float64,
        )
        return interfaces, etas

    def _scaled_variables_in_bounds(self, x_hat, lower, upper):
        if np.any(np.asarray(x_hat) < lower) or np.any(np.asarray(x_hat) > upper):
            return False
        interfaces, _ = self._scaled_to_physical(x_hat)
        try:
            right_boundary = self._right_boundary_from_interfaces(interfaces)
            self._validate_interfaces(interfaces, strict=True, right_boundary=right_boundary)
        except ValueError:
            return False
        return True

    def _bounded_newton_step(self, x_hat, step, lower, upper):
        step = np.asarray(step, dtype=np.float64).copy()
        if not np.all(np.isfinite(step)):
            raise RuntimeError("Three-phase interface Newton step is non-finite.")
        active_tol = 10.0 * np.finfo(float).eps
        for i in range(step.size):
            if x_hat[i] <= lower[i] + active_tol and step[i] < 0.0:
                step[i] = 0.0
            elif x_hat[i] >= upper[i] - active_tol and step[i] > 0.0:
                step[i] = 0.0
        alpha_max = 1.0
        for i in range(step.size):
            if step[i] > 0.0:
                alpha_max = min(alpha_max, float((upper[i] - x_hat[i]) / step[i]))
            elif step[i] < 0.0:
                alpha_max = min(alpha_max, float((lower[i] - x_hat[i]) / step[i]))
        if step[0] - step[1] > 0.0:
            gap_hat = (x_hat[1] - x_hat[0]) - self._minimum_middle_width(self._R0) / self._R0
            alpha_max = min(alpha_max, max(0.0, float(gap_hat / (step[0] - step[1]))))
        return step, max(0.0, alpha_max * (1.0 - 1e-12))

    def _extrapolated_interface_start(self, interfaces, etas, dt, lower, upper):
        """
        Returns the extrapolated scaled Newton start ``[s_AB, s_BC, eta_AB, eta_BC]``, or ``None``.

        With ``interface_newton_start='extrapolated'`` both interfaces and both
        tie-line coordinates are extrapolated linearly from the last accepted
        step, ``x + (x - x_old) * dt / dt_last``, and clipped to the solve
        bounds. Only the Newton starting point changes; ``None`` (start from the
        unmoved old state) is returned when the option is off, before the first
        accepted step, or when the extrapolation does not move.
        """
        if self.interfaceNewtonStart != "extrapolated" or self._lastAcceptedDt is None or self._etas_old is None or self._interfaces_old is None:
            return None
        ratio = float(dt) / float(self._lastAcceptedDt)
        if not np.isfinite(ratio) or ratio <= 0.0:
            return None
        interfaces = np.asarray(interfaces, dtype=np.float64)
        etas = np.asarray(etas, dtype=np.float64)
        future_interfaces = interfaces + (interfaces - np.asarray(self._interfaces_old, dtype=np.float64)) * ratio
        future_etas = etas + (etas - np.asarray(self._etas_old, dtype=np.float64)) * ratio
        if np.array_equal(future_interfaces, interfaces) and np.array_equal(future_etas, etas):
            return None
        return np.clip(self._physical_to_scaled(future_interfaces, future_etas), lower, upper)

    def _evaluate_interface_candidate(self, profiles, old_interfaces, x_hat, residual_scale, dt):
        interfaces, etas = self._scaled_to_physical(x_hat)
        kinematics = self._step_kinematics(old_interfaces, interfaces)
        interfaces = self._validate_interfaces(interfaces, strict=True, right_boundary=kinematics.new_right_boundary)
        interface_compositions = self._interface_compositions(etas)
        bulk_results = self._solve_bulk_profiles(profiles, old_interfaces, interfaces, interface_compositions, kinematics=kinematics)
        candidate_profiles = tuple(result.profile for result in bulk_results)
        self._validate_candidate_profiles(candidate_profiles)
        residual = self._interface_residuals(
            profiles,
            old_interfaces,
            interfaces,
            interface_compositions,
            bulk_results,
            kinematics=kinematics,
        )
        scaled = residual / residual_scale
        return _ThreePhaseCandidate(
            x_hat=np.asarray(x_hat, dtype=np.float64).copy(),
            profiles=candidate_profiles,
            interfaces=interfaces.copy(),
            etas=etas.copy(),
            interface_compositions=interface_compositions,
            residual=residual,
            scaled_residual=scaled,
            scaled_norm=float(np.max(np.abs(scaled))),
            physical_norm=float(np.max(np.abs(residual))),
            bulk_results=bulk_results,
            kinematics=kinematics,
        )

    def _solve_interface_planar(self, profiles, interfaces, etas, dt):
        """
        Solves the coupled two-interface step by damped finite-difference Newton.

        Newton starts from the accepted old ``(interfaces, etas)``, or with
        ``interface_newton_start='extrapolated'`` (the default) from the linear
        extrapolation of the last accepted step (``_extrapolated_interface_start``);
        a start that cannot be evaluated falls back to the old state. The eta
        Jacobian columns use ``_INTERFACE_ETA_FD_STEP``.

        The residual scale (``_residual_scale``) does not depend on ``dt``, so the
        unmodified old state can meet the tolerance when ``flux * dt`` is tiny,
        which would accept a step with both interfaces frozen and leave the
        untransferred solute as an inventory defect. A candidate is therefore
        accepted only after at least one Newton update. While the start already
        meets the tolerance, a line-search trial is accepted if it improves the
        residual or also meets the tolerance. If that mandatory update cannot be
        carried out (zero step, failed finite-difference perturbation, or no
        acceptable trial), the converged start is accepted and
        ``_lastImplicitForcedUpdateSkipped`` is set.
        """
        lower, upper = self._scaled_bounds()
        x_hat = self._physical_to_scaled(interfaces, etas)
        if not self._scaled_variables_in_bounds(x_hat, lower, upper):
            raise ValueError("Initial three-phase nonlinear iterate lies outside scaled solve bounds.")
        residual_scale = self._residual_scale(profiles, interfaces)
        best = None
        failure_reason = "maximum iterations reached"
        converged_start = None
        newton_updates = 0
        iteration = 0
        # Optional extrapolated start; a start that cannot be evaluated falls
        # back to the old state.
        pending = None
        extrapolated = self._extrapolated_interface_start(interfaces, etas, dt, lower, upper)
        if extrapolated is not None and self._scaled_variables_in_bounds(extrapolated, lower, upper):
            try:
                pending = self._evaluate_interface_candidate(profiles, interfaces, extrapolated, residual_scale, dt)
                x_hat = extrapolated
            except (RuntimeError, ValueError, ZeroDivisionError):
                pending = None

        def accept(candidate, forced_update_skipped=False):
            self._lastImplicitIterations = iteration
            self._lastImplicitResidual = candidate.scaled_norm
            self._lastImplicitPhysicalResidual = candidate.physical_norm
            self._lastImplicitConverged = True
            self._lastImplicitForcedUpdateSkipped = bool(forced_update_skipped)
            self._lastImplicitFailureReason = None
            return candidate

        for iteration in range(1, self.maxIterations + 1):
            if pending is not None:
                candidate, pending = pending, None
            else:
                candidate = self._evaluate_interface_candidate(profiles, interfaces, x_hat, residual_scale, dt)
            if best is None or candidate.scaled_norm < best.scaled_norm:
                best = candidate
            if candidate.scaled_norm <= self.residualTolerance:
                if newton_updates > 0:
                    return accept(candidate)
                converged_start = candidate

            jacobian = np.zeros((4, 4), dtype=np.float64)
            for variable in range(4):
                # Variables 2 and 3 are the scaled tie-line coordinates; see
                # ``_INTERFACE_ETA_FD_STEP`` for why they use a larger step.
                step_size, x_perturbed, direction = _bounded_finite_difference_perturbation(
                    x_hat, lower, upper, variable, step=_INTERFACE_ETA_FD_STEP if variable >= 2 else None,
                )
                if not self._scaled_variables_in_bounds(x_perturbed, lower, upper):
                    x_perturbed = x_hat.copy()
                    x_perturbed[variable] -= step_size
                    direction = "backward"
                try:
                    perturbed = self._evaluate_interface_candidate(profiles, interfaces, x_perturbed, residual_scale, dt)
                except ValueError as exc:
                    if not self._is_infeasible_candidate_error(exc):
                        raise
                    x_perturbed = x_hat.copy()
                    if direction == "forward":
                        x_perturbed[variable] -= step_size
                        direction = "backward"
                    else:
                        x_perturbed[variable] += step_size
                        direction = "forward"
                    if not self._scaled_variables_in_bounds(x_perturbed, lower, upper):
                        failure_reason = f"finite-difference perturbation left admissible composition bounds for variable {variable}"
                        break
                    try:
                        perturbed = self._evaluate_interface_candidate(profiles, interfaces, x_perturbed, residual_scale, dt)
                    except ValueError as exc2:
                        if self._is_infeasible_candidate_error(exc2):
                            failure_reason = f"finite-difference perturbation left admissible composition bounds for variable {variable}"
                            break
                        raise
                if direction == "forward":
                    jacobian[:, variable] = (perturbed.scaled_residual - candidate.scaled_residual) / step_size
                else:
                    jacobian[:, variable] = (candidate.scaled_residual - perturbed.scaled_residual) / step_size
            else:
                try:
                    step = np.linalg.solve(jacobian, -candidate.scaled_residual)
                except np.linalg.LinAlgError:
                    step = np.linalg.lstsq(jacobian, -candidate.scaled_residual, rcond=None)[0]
                step, alpha_start = self._bounded_newton_step(x_hat, step, lower, upper)
                accepted = False
                for scale in (alpha_start, 0.5 * alpha_start, 0.25 * alpha_start, 0.125 * alpha_start, 0.0625 * alpha_start):
                    if scale <= 0.0:
                        continue
                    trial = x_hat + scale * step
                    if not self._scaled_variables_in_bounds(trial, lower, upper):
                        continue
                    try:
                        trial_candidate = self._evaluate_interface_candidate(profiles, interfaces, trial, residual_scale, dt)
                    except ValueError as exc:
                        if self._is_infeasible_candidate_error(exc):
                            continue
                        raise
                    improves = np.isfinite(trial_candidate.scaled_norm) and trial_candidate.scaled_norm < candidate.scaled_norm
                    if improves or (converged_start is not None and trial_candidate.scaled_norm <= self.residualTolerance):
                        x_hat = trial
                        newton_updates += 1
                        accepted = True
                        break
                if not accepted:
                    failure_reason = "line search failed"
                    break
                continue

            break
        if converged_start is not None:
            return accept(converged_start, forced_update_skipped=True)
        self._lastImplicitIterations = self.maxIterations
        self._lastImplicitResidual = np.inf if best is None else best.scaled_norm
        self._lastImplicitPhysicalResidual = np.inf if best is None else best.physical_norm
        self._lastImplicitConverged = False
        self._lastImplicitForcedUpdateSkipped = False
        self._lastImplicitFailureReason = failure_reason
        self._lastBest = best
        raise RuntimeError(f"Three-phase Illingworth interface solve failed to converge; best residual was {self._lastImplicitResidual:.3e}.")

    def _store_pending_candidate(self, candidate, start_time, trial_dt):
        """Stores an immutable-by-convention copy of a converged nonlinear candidate."""
        right_boundary = (
            candidate.kinematics.new_right_boundary
            if candidate.kinematics is not None
            else self._right_boundary_from_interfaces(candidate.interfaces)
        )
        self._pendingCandidate = _ThreePhasePendingState(
            profiles=tuple(profile.copy() for profile in candidate.profiles),
            interfaces=candidate.interfaces.copy(),
            etas=candidate.etas.copy(),
            right_boundary=float(right_boundary),
            start_time=float(start_time),
            dt=float(trial_dt),
            projection=(
                self._lastStepProjectionInventoryChange.copy(),
                self._lastStepProjectionAbsoluteChange.copy(),
                int(self._lastStepProjectedNodes),
            ),
        )

    def _record_step_projection(self, candidate):
        """
        Sets the last-step round-off floor-projection diagnostics from an accepted candidate.

        ``_lastStepProjectionInventoryChange`` and
        ``_lastStepProjectionAbsoluteChange`` are the normalized-density-weighted
        (``rho_alpha / rho_A``) sums over the three phases of the signed and
        absolute componentwise inventory changes (the units of the interface
        residual), and ``_lastStepProjectedNodes`` the projected node count.
        ``_lastStepProjectedSolves`` counts every projecting linear solve of
        this step attempt (rejected trials and Picard iterations included).
        They reach the cumulative totals only when ``postProcess`` commits the
        step.
        """
        inventory = np.zeros(2, dtype=np.float64)
        absolute = np.zeros(2, dtype=np.float64)
        nodes = 0
        for density, result in zip(self._normalizedPhaseMolarDensities, candidate.bulk_results):
            projection = getattr(result, "projection", None)
            if projection is None or projection.mask is None:
                continue
            inventory = inventory + density * projection.inventory_change
            absolute = absolute + density * projection.absolute_change
            nodes += projection.projected_nodes
        self._lastStepProjectionInventoryChange = inventory
        self._lastStepProjectionAbsoluteChange = absolute
        self._lastStepProjectedNodes = int(nodes)
        self._lastStepProjectedSolves = int(getattr(self, "_currentStepProjectedSolves", 0))

    def _commit_step_projection(self, pending):
        """Adds a committed step's staged floor-projection totals to the cumulative ones."""
        if pending is None or pending.projection is None:
            return
        inventory_change, absolute_change, nodes = pending.projection
        self._cumulativeProjectionInventoryChange = self._cumulativeProjectionInventoryChange + inventory_change
        self._cumulativeProjectionAbsoluteChange = self._cumulativeProjectionAbsoluteChange + absolute_change
        self._cumulativeProjectedNodes += int(nodes)

    def _accept_step_candidate(self, profiles, interfaces, etas, candidate, trial_dt, retry, start_time=None):
        """Stores the exact pending root and returns its Euler-wrapper derivatives."""
        self._currdt = float(trial_dt)
        self._lastStepRetries = int(retry)
        self._record_step_projection(candidate)
        self._store_pending_candidate(
            candidate,
            self.currentTime if start_time is None else start_time,
            trial_dt,
        )
        return [
            (candidate.profiles[0] - profiles[0]) / trial_dt,
            (candidate.profiles[1] - profiles[1]) / trial_dt,
            (candidate.profiles[2] - profiles[2]) / trial_dt,
            (candidate.interfaces - interfaces) / trial_dt,
            (candidate.etas - etas) / trial_dt,
        ]

    def _try_step_retries(self, profiles, interfaces, etas, trial_dt, retry_count, retry_offset=0, start_time=None):
        """Attempts implicit solves while shrinking ``trial_dt`` after each failed trial."""
        last_error = None
        for retry in range(int(retry_count)):
            if retry==(int(retry_count)-1):
                self._print_diagnostic(f"last timestep retry ({retry}) at dt={trial_dt:.6g}")
            self._currdt = trial_dt
            # One lagged face-diffusivity evaluation per attempt, shared by its
            # Newton candidates (see ``_lagged_interval_face_diffusivity``), and
            # a per-eta interface-composition cache (see ``_interface_compositions``).
            self._laggedFaceDiffusivityCache = {}
            self._interfaceCompositionCache = {}
            try:
                candidate = self._solve_interface_planar(profiles, interfaces, etas, trial_dt)
                return self._accept_step_candidate(
                    profiles,
                    interfaces,
                    etas,
                    candidate,
                    trial_dt,
                    retry_offset + retry,
                    start_time=start_time,
                ), trial_dt, None
            except (RuntimeError, ValueError, ZeroDivisionError) as exc:
                last_error = exc
                trial_dt *= self.retryFactor
            finally:
                self._laggedFaceDiffusivityCache = None
                self._interfaceCompositionCache = None
        return None, trial_dt, last_error

    def _confirm_terminal_thin_phase_retries(self, phase_index, width, t, trial_dt):
        """
        Warns about a near-disappearing phase and returns whether terminal retries should run.

        ``terminal_thin_phase_policy='prompt'`` asks the user directly.
        ``'continue'`` is intended for batch runs that should always try to
        produce a final valid state, while ``'raise'`` preserves the hard-error
        behavior.
        """
        phase = self.phases[int(phase_index)]
        message = (
            f"Three-phase Illingworth phase {phase!r} width {float(width):.6g} is below "
            f"terminal_thin_phase_width={float(self.terminalThinPhaseWidth):.6g} after normal timestep retries at t={float(t):.6g}. "
            f"The solver can keep shrinking trial_dt from {float(trial_dt):.6g} for up to "
            f"{self.terminalThinPhaseExtraRetries} extra retries, then stop after the next converged step."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        if self.terminalThinPhasePolicy == "continue":
            return True
        if self.terminalThinPhasePolicy == "raise":
            return False
        try:
            answer = input(f"{message} Continue? [y/N] ")
        except (EOFError, KeyboardInterrupt, OSError) as exc:
            raise RuntimeError(
                "terminal_thin_phase_policy='prompt' could not read a response from stdin. "
                "Set terminal_thin_phase_policy='continue' to allow terminal thin-phase retries in non-interactive runs, "
                "or set terminal_thin_phase_policy='raise' to keep the hard-error behavior."
            ) from exc
        return answer.strip().lower() in {"y", "yes"}

    def getdXdt(self, t, xCurr):
        """Attempt an implicit step, reducing its timestep on nonlinear failure.

        Retry exhaustion raises RuntimeError chained from the last failure,
        without importing research utilities or pausing for a debugger.
        The explicit terminal-thin-phase prompt policy is unchanged.
        """
        self._pendingCandidate = None
        self._terminalThinPhaseStop = False
        self._terminalThinPhaseInfo = None
        self._currentStepProjectedSolves = 0
        profiles = tuple(np.asarray(xCurr[i], dtype=np.float64).reshape((-1, 2)).copy() for i in range(3))
        interfaces = self._validate_interfaces(np.asarray(xCurr[3], dtype=np.float64), strict=True)
        etas = np.asarray(xCurr[4], dtype=np.float64).reshape(2)
        interface_compositions = self._interface_compositions(etas)
        profiles[0][-1] = interface_compositions[0][0]
        profiles[1][0] = interface_compositions[0][1]
        profiles[1][-1] = interface_compositions[1][0]
        profiles[2][0] = interface_compositions[1][1]
        dt = self._compute_dt(t)
        trial_dt = float(dt)
        dXdt, trial_dt, last_error = self._try_step_retries(
            profiles, interfaces, etas, trial_dt, self.maxStepRetries, start_time=t
        )
        if dXdt is not None:
            self._validate_pending_timestep_against_outer_minimum(t, trial_dt)
            return dXdt
        thin_phase = self._single_terminal_thin_phase(interfaces.copy())
        if thin_phase is not None:
            phase_index, width = thin_phase
            if self._confirm_terminal_thin_phase_retries(phase_index, width, t, trial_dt):
                dXdt, trial_dt, extra_error = self._try_step_retries(
                    profiles,
                    interfaces,
                    etas,
                    trial_dt,
                    self.terminalThinPhaseExtraRetries,
                    retry_offset=self.maxStepRetries,
                    start_time=t,
                )
                if dXdt is not None:
                    try:
                        self._validate_pending_timestep_against_outer_minimum(t, trial_dt)
                    except ValueError:
                        self._terminalThinPhaseStop = False
                        self._terminalThinPhaseInfo = None
                        raise
                    future_interfaces = xCurr[3] + dXdt[3] * trial_dt
                    future_right = self._right_boundary_from_interfaces(future_interfaces)
                    if not (self._single_terminal_thin_phase(future_interfaces, right_boundary=future_right) is not None):
                        raise ValueError(f"Expecting next interfaces to also satisify _single_terminal_thin_phase() but got: {(xCurr[3]+dXdt[3]*trial_dt).tolist()}")
                    self._terminalThinPhaseStop = True
                    self._terminalThinPhaseInfo = {
                        "phase": self.phases[int(phase_index)],
                        "phase_index": int(phase_index),
                        "width": float(width),
                        "threshold": float(self.terminalThinPhaseWidth),
                        "time": float(t),
                        "dt": float(self._currdt),
                    }
                    return dXdt
                if extra_error is not None:
                    last_error = extra_error
        self._print_diagnostic(f"t: {t}")
        self._print_diagnostic(f"etas: {etas}")
        self._print_diagnostic(f"interfaces: {interfaces}")
        self._print_diagnostic(f"self._lastImplicitFailureReason: {self._lastImplicitFailureReason}")
        raise RuntimeError("Three-phase Illingworth step failed after timestep retries.") from last_error

    def _print_diagnostic(self, message):
        """Prints a solver diagnostic only when ``printDiagnostics`` is enabled."""
        if self.printDiagnostics:
            print(message)

    def getDt(self, dXdt):
        if np.isfinite(self._currdt) and self._currdt > 0.0:
            return self._currdt
        return self.timeStep

    def _prepare_accepted_state(self, x):
        """Constructs and validates every derived output before accepted-state mutation."""
        profiles = tuple(np.asarray(x[i], dtype=np.float64).reshape((-1, 2)).copy() for i in range(3))
        interfaces = np.asarray(x[3], dtype=np.float64).reshape(2).copy()
        right_boundary = self._right_boundary_from_interfaces(interfaces)
        interfaces = self._validate_interfaces(interfaces, strict=True, right_boundary=right_boundary)
        etas = np.asarray(x[4], dtype=np.float64).reshape(2).copy()
        interface_compositions = self._interface_compositions(etas)
        profiles[0][-1] = interface_compositions[0][0]
        profiles[1][0] = interface_compositions[0][1]
        profiles[1][-1] = interface_compositions[1][0]
        profiles[2][0] = interface_compositions[1][1]
        for i, profile in enumerate(profiles):
            self._validate_profile_compositions(profile, f"transformed profile {i}")
        fixed_mesh_profile = self._reconstruct_physical_profile(
            profiles, interfaces, right_boundary=right_boundary
        )
        legacy_inventory = self.getTotalInventoryFromState(
            profiles, interfaces, right_boundary=right_boundary
        )
        if self._hasExplicitPhaseMolarVolumes:
            molar_inventories = self._get_all_component_moles_from_state(
                profiles, interfaces, right_boundary
            )
            total_moles = self._get_total_substitutional_moles_from_geometry(
                interfaces, right_boundary
            )
            dependent_closure_error = self._dependent_mole_closure_error(
                molar_inventories, total_moles
            )
            scale = max(abs(total_moles), 1.0)
            if not np.isclose(self._substitutional_mole_sum(molar_inventories), total_moles, rtol=5e-13, atol=5e-15 * scale):
                raise RuntimeError("Direct component moles do not sum to the geometric substitutional total.")
        else:
            molar_inventories = np.full(3, np.nan, dtype=np.float64)
            total_moles = np.nan
            dependent_closure_error = np.nan
        return _ThreePhaseAcceptedState(
            profiles=profiles,
            interfaces=interfaces,
            etas=etas,
            right_boundary=right_boundary,
            interface_compositions=interface_compositions,
            fixed_mesh_profile=fixed_mesh_profile,
            legacy_inventory=legacy_inventory,
            molar_inventories=molar_inventories,
            total_moles=total_moles,
            dependent_closure_error=dependent_closure_error,
        )

    def _commit_accepted_state(self, time, accepted):
        """
        Commits one prepared state and all synchronized histories as one transaction.

        Uses the shared atomic history helpers (``_record_histories_atomically``,
        ``_restore_history_snapshots``); any exception, including an
        interruption such as ``KeyboardInterrupt``, in recording, in the state
        assignment or in ``updateCoupledModels`` restores this model's
        histories and accepted state and is re-raised unchanged. Coupled models
        updated before the failing one keep their update (LEDGERS R10, D6).
        """
        records = [
            (self.data, accepted.fixed_mesh_profile),
            (self.interfaceData, accepted.interfaces),
            (self.etaData, accepted.etas),
            (self.rightBoundaryData, accepted.right_boundary),
            (self.inventoryData, accepted.legacy_inventory),
            (self.molarInventoryData, accepted.molar_inventories),
            (self.totalMolesData, accepted.total_moles),
            (self.dependentMoleClosureErrorData, accepted.dependent_closure_error),
        ]
        if self.recordPqData:
            records.extend(zip(self.profileData, accepted.profiles))
        previous_state = (
            self._profiles_curr,
            self._interfaces_old,
            self._interfaces_curr,
            self._etas_old,
            self._etas_curr,
            self._lastAcceptedDt,
            self._R,
            self._lastInterfaceCompositions,
            self.currentTime,
        )
        snapshots = _record_histories_atomically(time, records)
        try:
            previous_interfaces = self._interfaces_curr.copy()
            accepted_dt = float(time) - float(self.currentTime)
            self._profiles_curr = tuple(profile.copy() for profile in accepted.profiles)
            self._interfaces_old = previous_interfaces
            self._interfaces_curr = accepted.interfaces.copy()
            self._etas_old = self._etas_curr.copy()
            self._etas_curr = accepted.etas.copy()
            self._lastAcceptedDt = accepted_dt if accepted_dt > 0.0 else self._lastAcceptedDt
            self._R = float(accepted.right_boundary)
            self._lastInterfaceCompositions = accepted.interface_compositions
            self.currentTime = float(time)
            self.updateCoupledModels()
        except BaseException:
            _restore_history_snapshots(snapshots)
            (
                self._profiles_curr,
                self._interfaces_old,
                self._interfaces_curr,
                self._etas_old,
                self._etas_curr,
                self._lastAcceptedDt,
                self._R,
                self._lastInterfaceCompositions,
                self.currentTime,
            ) = previous_state
            raise

    def _verify_pending_candidate(self, time, x):
        """
        Verifies that the outer Euler update reproduces the exact pending root.

        The nonlinear candidate remains separate from accepted model state until
        this check succeeds. This prevents an outer timestep clamp or correction
        from committing a state for which the interface equations were not solved.
        """
        pending = self._pendingCandidate
        if pending is None:
            raise RuntimeError(
                "Cannot commit a three-phase state without a pending converged nonlinear candidate."
            )
        expected_time = pending.start_time + pending.dt
        time_scale = max(1.0, abs(float(time)), abs(expected_time), abs(pending.start_time))
        time_atol = 64.0 * np.finfo(float).eps * time_scale
        mismatches = []
        if not np.isclose(self.currentTime, pending.start_time, rtol=2e-13, atol=time_atol):
            mismatches.append("accepted start time")
        if not np.isclose(float(time), expected_time, rtol=2e-13, atol=time_atol):
            mismatches.append("timestep")

        try:
            profiles = tuple(np.asarray(x[i], dtype=np.float64).reshape((-1, 2)) for i in range(3))
            interfaces = np.asarray(x[3], dtype=np.float64).reshape(2)
            etas = np.asarray(x[4], dtype=np.float64).reshape(2)
        except (IndexError, TypeError, ValueError) as exc:
            self._pendingCandidate = None
            self._terminalThinPhaseStop = False
            self._terminalThinPhaseInfo = None
            raise RuntimeError(
                "Outer solver state does not have the shape of the pending three-phase nonlinear candidate."
            ) from exc

        state_atol = 64.0 * np.finfo(float).eps
        for phase, actual, expected in zip(self.phases, profiles, pending.profiles):
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=2e-13, atol=state_atol
            ):
                mismatches.append(f"{phase} profile")
        if not np.allclose(interfaces, pending.interfaces, rtol=2e-13, atol=state_atol):
            mismatches.append("interfaces")
        if not np.allclose(etas, pending.etas, rtol=2e-13, atol=state_atol):
            mismatches.append("etas")
        wrapper_right = self._right_boundary_from_interfaces(interfaces)
        if not np.isclose(wrapper_right, pending.right_boundary, rtol=2e-13, atol=state_atol):
            mismatches.append("right boundary")
        if mismatches:
            self._pendingCandidate = None
            self._terminalThinPhaseStop = False
            self._terminalThinPhaseInfo = None
            raise RuntimeError(
                "Outer solver update does not match the pending converged three-phase candidate "
                f"({', '.join(mismatches)}); accepted state was not advanced."
            )
        return pending

    def postProcess(self, time, x):
        """
        Verifies the wrapper's update against the pending candidate, then prepares and commits it.

        Preparation (``_prepare_accepted_state``) touches no model state; the
        commit (``_commit_accepted_state``) is atomic. The candidate is consumed
        whether or not the commit succeeds, so after any exception, including
        an interruption such as ``KeyboardInterrupt`` (re-raised unchanged),
        accepted state and histories are unchanged, a staged terminal
        thin-phase stop is cleared, a resubmission is refused and only a newly
        solved candidate can be committed. Coupled models updated before a
        failing one keep their update (LEDGERS R10, D6).
        """
        pending = self._verify_pending_candidate(time, x)
        try:
            accepted = self._prepare_accepted_state(x)
            self._commit_accepted_state(time, accepted)
            self._commit_step_projection(pending)
        except BaseException:
            self._terminalThinPhaseStop = False
            self._terminalThinPhaseInfo = None
            raise
        finally:
            self._pendingCandidate = None
        if self._terminalThinPhaseStop:
            self.finalTime = time
            self._terminalThinPhaseStop = False
            return self.getCurrentX(), True
        return self.getCurrentX(), False

    def postSolve(self):
        """
        Finalizes every history, ``data`` included, with the shared flag rule (LEDGERS R09).

        A history appends the final accepted state only if its latest record
        stored no row (an off-cadence final step); times are not compared, and a
        repeated call adds nothing, and all histories stay synchronized (the same
        rule as the compatibility ``MeshData.finalize``).

        All histories are finalized as one unit (``_finalize_histories_atomically``,
        LEDGERS R10b): if one raises, every history is restored to its state
        before the call and the exception re-raised; calling ``postSolve`` again
        is the retry. A continued solve without that retry records as if the
        first solve had never been finalized.
        """
        histories = [
            self.data,
            self.interfaceData,
            self.etaData,
            self.rightBoundaryData,
            self.inventoryData,
            self.molarInventoryData,
            self.totalMolesData,
            self.dependentMoleClosureErrorData,
        ]
        if self.profileData is not None:
            histories.extend(self.profileData)
        _finalize_histories_atomically(histories)

    @_all_or_nothing()
    def reset(self):
        """
        Returns the model to its initial state (time 0, initial records, ``isSetup=False``).

        The initial profile is recorded with the constructor's semantics: the
        host reset records the raw mole fractions of ``mesh.y``, which are then
        replaced by the constructor's u-fraction conversion
        (``_recordInitialMeshComposition``, the recording step of
        ``_validateMeshComposition``). ``mesh.y`` already holds the constructor's
        clamped values, so it is neither clamped nor modified again, and repeated
        resets convert the same ``mesh.y``. Reset + solve therefore equals a
        freshly constructed model bit for bit, with or without interstitials.

        All-or-nothing (LEDGERS R10a): if any part raises (the host reset and
        provider ``clearCache``, or re-recording the initial profile), every
        model-owned field is restored and the exception re-raised, so the
        previous simulation is kept. Provider caches and other external state
        are not restored.
        """
        super().reset()
        self._recordInitialMeshComposition(self.mesh.flattenResponse(self.mesh.y))
        self.interfaceData.reset()
        self.interfaceData.record(0, self.initialInterfacePositions)
        self.etaData.reset()
        self.inventoryData.reset()
        self.rightBoundaryData.reset()
        self.molarInventoryData.reset()
        self.totalMolesData.reset()
        self.dependentMoleClosureErrorData.reset()
        self.profileData = None
        self._currdt = np.inf
        self._outerMinTimeStep = 0.0
        self._outerMaxTimeStep = np.inf
        self._solveMinDtFrac = 1e-8
        self._solveMaxDtFrac = 1.0
        self._pendingCandidate = None
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
        self._lastStepRetries = 0
        self._lastImplicitIterations = 0
        self._lastImplicitResidual = np.nan
        self._lastImplicitPhysicalResidual = np.nan
        self._lastImplicitConverged = False
        self._lastImplicitForcedUpdateSkipped = False
        self._lastImplicitFailureReason = None
        self._lastInterfaceCompositions = None
        self._terminalThinPhaseStop = False
        self._terminalThinPhaseInfo = None
        self._initialInventory = None
        self._initialMolarInventories = None
        self._initialTotalMoles = None
        self._z = None
        self._R = None
        self._R0 = None
        self._initialTotalAmount = None
        self._grids = None
        self._profiles_curr = None
        self._interfaces_curr = None
        self._interfaces_old = None
        self._etas_curr = None
        self._etas_old = None
        self._lastAcceptedDt = None
        self._laggedFaceDiffusivityCache = None
        self._interfaceCompositionCache = None
        self._reset_projection_diagnostics()

    def _reset_projection_diagnostics(self):
        """
        Clears the round-off floor-projection diagnostics.

        Last accepted step (see ``_record_step_projection``), and cumulative
        totals over committed steps; ``_cumulativeProjectedSolves`` counts
        every projecting linear solve, discarded ones included.
        """
        self._lastStepProjectionInventoryChange = np.zeros(2, dtype=np.float64)
        self._lastStepProjectionAbsoluteChange = np.zeros(2, dtype=np.float64)
        self._lastStepProjectedNodes = 0
        self._lastStepProjectedSolves = 0
        self._currentStepProjectedSolves = 0
        self._cumulativeProjectionInventoryChange = np.zeros(2, dtype=np.float64)
        self._cumulativeProjectionAbsoluteChange = np.zeros(2, dtype=np.float64)
        self._cumulativeProjectedNodes = 0
        self._cumulativeProjectedSolves = 0

    def getInterfacePositions(self, time=None):
        """Returns the current accepted interface positions ``[s_AB, s_BC]``, or the recorded ones at an exact ``time``."""
        return self.interfaceData.y(time)

    def getInterfaceEtas(self, time=None):
        """Returns the current accepted tie-line coordinates ``[eta_AB, eta_BC]``, or the recorded ones at an exact ``time``."""
        return self.etaData.y(time)

    def getInterfaceCompositions(self, time=None):
        """Returns ``((A_AB, B_AB), (B_BC, C_BC))`` interface compositions."""
        return self._interface_compositions(self.getInterfaceEtas(time))

    def getTransformedState(self, time=None):
        """Returns the three transformed phase profiles: current (copies) without ``time``, else recorded at an exact ``time``."""
        if time is None and self._profiles_curr is not None:
            return tuple(profile.copy() for profile in self._profiles_curr)
        if self.profileData is None:
            raise ValueError("Transformed profile history is not available; set record_pq_data=True.")
        return tuple(history.y(time) for history in self.profileData)

    def getRightBoundary(self, time=None):
        """Returns the accepted right material-boundary position."""
        if time is None:
            return float(self._R)
        return self.rightBoundaryData.y(time)

    def getTotalInventoryFromState(self, profiles, interfaces, right_boundary=None):
        """
        Returns the legacy length-weighted independent-component inventory.

        ``right_boundary`` permits pure candidate evaluation without changing
        the accepted ``_R``. No phase molar-density weighting is applied.
        """
        domain_length = float(self._R if right_boundary is None else right_boundary)
        return integrate_planar_transformed_profile_sequence(profiles, interfaces, domain_length, self._grids)

    def getTotalInventory(self, time=None):
        """Returns the legacy inventory without applying phase molar densities."""
        if time is None:
            return self.getTotalInventoryFromState(self._profiles_curr, self._interfaces_curr)
        return self.inventoryData.y(time)

    def getTotalMass(self, time=None):
        """Returns the legacy composition-length inventory."""
        return self.getTotalInventory(time=time)

    def _require_explicit_phase_molar_volumes(self):
        """Rejects physical-mole queries when the molar-volume scale is arbitrary."""
        if not self._hasExplicitPhaseMolarVolumes:
            raise ValueError("Physical mole APIs require explicit phase_molar_volumes.")

    def _get_all_component_moles_from_state(self, profiles, interfaces, right_boundary):
        """Directly integrates dependent and independent component moles for explicit geometry."""
        self._require_explicit_phase_molar_volumes()
        return integrate_planar_transformed_molar_inventories(
            profiles,
            interfaces,
            float(right_boundary),
            self._grids,
            self._phaseMolarVolumes,
            substitutional_mask=self._substitutional_mask(),
        )

    def _substitutional_mask(self):
        """Returns the independent-component substitutional mask, or ``None`` without interstitials."""
        if not has_interstitials(self.allElements):
            return None
        return ~independent_interstitial_mask(self.allElements)

    def _substitutional_mole_sum(self, molar_inventories):
        """Returns the summed moles of the substitutional components (all three without interstitials)."""
        values = np.asarray(molar_inventories, dtype=np.float64).reshape(3)
        mask = self._substitutional_mask()
        if mask is None:
            return np.sum(molar_inventories)
        return float(values[0] + np.sum(values[1:][mask]))

    def _get_total_substitutional_moles_from_geometry(self, interfaces, right_boundary):
        """Returns total substitutional moles per unit area from phase widths and densities."""
        self._require_explicit_phase_molar_volumes()
        widths = self._phase_widths(interfaces, right_boundary=right_boundary)
        return float(np.dot(self._phaseMolarDensities, widths))

    def _dependent_mole_closure_error(self, molar_inventories, total_moles):
        """
        Returns direct dependent moles minus ``Ntot - N1 - N2``.

        With interstitials only the substitutional independent components are
        subtracted, because interstitial moles are not part of ``Ntot``.
        """
        values = np.asarray(molar_inventories, dtype=np.float64).reshape(3)
        mask = self._substitutional_mask()
        if mask is not None:
            return float(values[0] - (float(total_moles) - np.sum(values[1:][mask])))
        return float(values[0] - (float(total_moles) - values[1] - values[2]))

    def getAllComponentMoles(self, time=None):
        """
        Returns directly integrated ``[N0, N1, N2]`` in mol/m^2.

        With interstitial components the profiles are u-fractions and
        ``phase_molar_volumes`` are volumes per mole of substitutional atoms,
        so these are still moles of each component; only the substitutional
        entries add up to ``getTotalSubstitutionalMoles``.
        """
        self._require_explicit_phase_molar_volumes()
        if time is None:
            return self._get_all_component_moles_from_state(
                self._profiles_curr, self._interfaces_curr, self._R
            )
        return self.molarInventoryData.y(time)

    def getMolarInventories(self, time=None):
        """Returns all directly integrated component inventories in mol/m^2."""
        return self.getAllComponentMoles(time=time)

    def getIndependentComponentMoles(self, time=None):
        """Returns directly integrated ``[N1, N2]`` in mol/m^2."""
        return self.getAllComponentMoles(time=time)[1:]

    def getTotalSubstitutionalMoles(self, time=None):
        """
        Returns geometric total substitutional inventory in mol/m^2.

        This is the phase widths divided by the phase molar volumes; it counts
        substitutional atoms only, so interstitial moles are not included.
        """
        self._require_explicit_phase_molar_volumes()
        if time is None:
            return self._get_total_substitutional_moles_from_geometry(
                self._interfaces_curr, self._R
            )
        return self.totalMolesData.y(time)

    def getDependentMoleClosureError(self, time=None):
        """
        Returns ``N0_direct - (Ntot - N1 - N2)`` in mol/m^2.

        ``Ntot`` is the substitutional total from ``getTotalSubstitutionalMoles``.
        With interstitial components only the substitutional independent
        components are subtracted (e.g. ``N0 - (Ntot - N_Mn)`` for Fe-C-Mn),
        because interstitial moles are not part of ``Ntot``.
        """
        self._require_explicit_phase_molar_volumes()
        if time is None:
            return self._dependent_mole_closure_error(
                self.getAllComponentMoles(), self.getTotalSubstitutionalMoles()
            )
        return self.dependentMoleClosureErrorData.y(time)

    def getMolarConservationDiagnostics(self, time=None):
        """
        Returns signed and absolute physical-mole conservation diagnostics.

        The three component inventories are independently integrated from
        their composition fields: mole fractions for substitutional systems,
        or u-fractions per mole of substitutional atoms when interstitials are
        present (the results are moles of each component in both cases).
        ``total_substitutional_moles`` and the dependent-component closure
        involve substitutional components only. Physical units are available
        only when ``phase_molar_volumes`` was explicitly supplied.
        """
        self._require_explicit_phase_molar_volumes()
        if self._initialMolarInventories is None or self._initialTotalMoles is None:
            raise ValueError("Model must be setup before molar conservation diagnostics.")
        component_moles = self.getAllComponentMoles(time=time)
        component_drift = component_moles - self._initialMolarInventories
        total_moles = self.getTotalSubstitutionalMoles(time=time)
        total_drift = float(total_moles - self._initialTotalMoles)
        closure_error = float(self.getDependentMoleClosureError(time=time))
        diagnostic_time = float(self.currentTime if time is None else time)
        return {
            "time": diagnostic_time,
            "component_moles": component_moles.copy(),
            "initial_component_moles": self._initialMolarInventories.copy(),
            "component_drift": component_drift.copy(),
            "absolute_component_drift": np.abs(component_drift),
            "N0_drift": float(component_drift[0]),
            "N1_drift": float(component_drift[1]),
            "N2_drift": float(component_drift[2]),
            "absolute_N0_drift": float(abs(component_drift[0])),
            "absolute_N1_drift": float(abs(component_drift[1])),
            "absolute_N2_drift": float(abs(component_drift[2])),
            "total_substitutional_moles": float(total_moles),
            "initial_total_substitutional_moles": float(self._initialTotalMoles),
            "total_substitutional_mole_drift": total_drift,
            "absolute_total_substitutional_mole_drift": abs(total_drift),
            "dependent_component_closure_error": closure_error,
            "absolute_dependent_component_closure_error": abs(closure_error),
        }

    def getPhysicalCoordinates(self, time=None):
        """Returns authoritative phase-wise physical coordinates spanning ``[0, R(t)]``."""
        if time is None:
            interfaces = self._interfaces_curr
            right_boundary = self._R
        else:
            interfaces = self.getInterfacePositions(time)
            right_boundary = self.getRightBoundary(time)
        boundaries = np.concatenate(([0.0], np.asarray(interfaces, dtype=np.float64), [right_boundary]))
        return tuple(
            float(left) + (float(right) - float(left)) * grid
            for grid, left, right in zip(self._grids, boundaries[:-1], boundaries[1:])
        )

    def getPhysicalPhaseProfiles(self, time=None):
        """
        Returns ``(coordinates, compositions)`` for each complete moving phase.

        Each phase retains its own interface endpoint, so discontinuous left
        and right equilibrium compositions are not averaged together.
        Historical requests require an exactly recorded transformed state.
        Compositions are full mole fractions (stored u-fractions are converted
        when interstitials are present).
        """
        coordinates = self.getPhysicalCoordinates(time=time)
        profiles = self.getTransformedState(time=time)
        out = []
        for phase_coordinates, independent in zip(coordinates, profiles):
            if has_interstitials(self.allElements):
                out.append((phase_coordinates.copy(), full_mole_fractions(independent, self.allElements)))
                continue
            dependent = 1.0 - np.sum(independent, axis=1)
            out.append((phase_coordinates.copy(), np.column_stack((dependent, independent))))
        return tuple(out)

    def getFixedPhysicalCoordinates(self):
        """Returns the immutable coordinates paired with the legacy fixed-mesh data snapshots."""
        return self._z.copy()

    def _fixed_mesh_profile(self, time=None):
        """Returns a safe fixed-mesh snapshot, guarding NaN interpolation after contraction."""
        if time is None:
            return np.asarray(self.data.currentY, dtype=np.float64).copy()
        recorded_time = self.data._time[: self.data.N + 1]
        matches = np.flatnonzero(np.isclose(recorded_time, float(time), rtol=0.0, atol=1e-14))
        if len(matches):
            return np.asarray(self.data._y[matches[-1]], dtype=np.float64).copy()
        if not self._equalPhaseMolarVolumes:
            raise ValueError(
                "Fixed-mesh interpolation is unavailable for a moving unequal-volume domain; "
                "request an exact recorded time or use getPhysicalPhaseProfiles()."
            )
        if float(time) <= recorded_time[0] or self.data.N == 0:
            return np.asarray(self.data._y[0], dtype=np.float64).copy()
        if float(time) >= recorded_time[-1]:
            return np.asarray(self.data._y[self.data.N], dtype=np.float64).copy()
        upper = int(np.searchsorted(recorded_time, float(time), side="right"))
        lower = upper - 1
        fraction = (float(time) - recorded_time[lower]) / (recorded_time[upper] - recorded_time[lower])
        return self.data._y[lower] + fraction * (self.data._y[upper] - self.data._y[lower])

    def getCompositions(self, time=None):
        """
        Returns full ternary mole fractions paired with the immutable fixed mesh.

        The stored response profile contains the two independent components
        (u-fractions when interstitials are present, converted here to mole
        fractions). Contracted-out nodes are NaN. For moving unequal-volume
        domains, non-recorded-time interpolation is rejected because a later
        NaN snapshot cannot be safely interpolated at a formerly valid node.
        """
        independent = self._fixed_mesh_profile(time=time)
        if has_interstitials(self.allElements):
            return full_mole_fractions(independent, self.allElements)
        dependent = 1.0 - np.sum(independent, axis=1)
        return np.column_stack((dependent, independent))

    def getFluxes(self, t=None, xCurr=None):
        """Rejects the inherited fixed-mesh flux API for the transformed ALE formulation."""
        raise NotImplementedError(
            "getFluxes is not defined for the transformed three-phase ALE solver; "
            "the conservative face transfers are candidate-local numerical quantities."
        )

    def checkConservation(self, tolerance: float, time=None):
        """
        Checks legacy componentwise composition-length drift from the initial value.

        The return value is the absolute drift vector for the two independent
        components. A warning is emitted when any component exceeds
        ``tolerance``. This legacy quantity is not the physical conservation
        diagnostic for unequal phase molar volumes.
        """
        if self._initialInventory is None:
            raise ValueError("Model must be setup before conservation checks.")
        drift = np.abs(self.getTotalInventory(time=time) - self._initialInventory)
        if np.any(drift > float(tolerance)):
            warnings.warn(
                f"Three-phase ternary Illingworth inventory drift {drift} exceeded tolerance {float(tolerance):.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return drift

    def checkMolarConservation(self, tolerance: float, time=None):
        """Checks direct all-component physical-mole drift from the initial state."""
        diagnostics = self.getMolarConservationDiagnostics(time=time)
        drift = diagnostics["absolute_component_drift"]
        total_drift = diagnostics["absolute_total_substitutional_mole_drift"]
        closure_error = diagnostics["absolute_dependent_component_closure_error"]
        if np.any(drift > float(tolerance)) or total_drift > float(tolerance) or closure_error > float(tolerance):
            warnings.warn(
                "Three-phase physical molar conservation tolerance was exceeded: "
                f"component drift={drift}, total drift={total_drift:.3e}, closure error={closure_error:.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return drift

    def toDict(self):
        """
        Converts solved histories to an output dictionary with moving-domain metadata.

        The dictionary is sufficient to interpret recorded geometry and molar
        diagnostics, but it is not a complete restart representation.
        """
        data = super().toDict()
        data.update(
            {
                "interface_positions": self.interfaceData._y,
                "interface_etas": self.etaData._y,
                "inventory": self.inventoryData._y,
                "interface_interval": self.interfaceData.recordInterval,
                "interface_index": self.interfaceData.N,
                "moving_domain_output_version": np.asarray(1, dtype=np.int64),
                "moving_domain_restart_supported": np.asarray(False),
                "phase_molar_volumes": self._phaseMolarVolumes.copy(),
                "phase_molar_volumes_explicit": np.asarray(self._hasExplicitPhaseMolarVolumes),
                "initial_right_boundary": np.asarray(self._R0, dtype=np.float64),
                "right_boundary_history": self.rightBoundaryData._y[: self.rightBoundaryData.N + 1].copy(),
                "right_boundary_time": self.rightBoundaryData._time[: self.rightBoundaryData.N + 1].copy(),
                "molar_inventory_history": self.molarInventoryData._y[: self.molarInventoryData.N + 1].copy(),
                "molar_inventory_time": self.molarInventoryData._time[: self.molarInventoryData.N + 1].copy(),
                "total_substitutional_moles_history": self.totalMolesData._y[: self.totalMolesData.N + 1].copy(),
                "dependent_mole_closure_error_history": self.dependentMoleClosureErrorData._y[
                    : self.dependentMoleClosureErrorData.N + 1
                ].copy(),
                "fixed_mesh_coordinate_semantics": np.asarray("laboratory_sampling_coordinates"),
                "moving_profile_coordinate_semantics": np.asarray("phasewise_physical_coordinates"),
            }
        )
        if self._grids is not None:
            for phase_index, grid in enumerate(self._grids):
                data[f"transformed_grid_{phase_index}"] = np.asarray(grid, dtype=np.float64).copy()
        if self.profileData is not None:
            data["profiles"] = np.asarray([history._y for history in self.profileData], dtype=object)
        return data

    @classmethod
    def fromDict(cls, data):
        """Rejects restart because output dictionaries do not restore the full accepted state."""
        raise NotImplementedError(
            "Three-phase Illingworth output files cannot be restarted because deserialization "
            "does not reconstruct the complete nonlinear accepted state."
        )


    ''' Plotting functions for diagnostics '''
    def confirmLastRecordedIndex(self, data):
        if not ((data._time[data.N]!=0).all() and (data._time[data.N+1]==0).all()):
            raise ValueError(f"{data._time[data.N]}==0 or {data._time[data.N+1]}!=0")
        if not ((data._y[data.N]!=0).all() and (data._y[data.N+1]==0).all()):
            raise ValueError(f"{data._y[data.N]}==0 or {data._y[data.N+1]}!=0")

    def plot_latestCompProfile(self):
        """Plots independent components on complete phase-wise physical coordinates."""
        import matplotlib.pyplot as plt
        phase_profiles = self.getPhysicalPhaseProfiles()
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, element in enumerate(self.elements):
            for phase_index, (coordinates, compositions) in enumerate(phase_profiles):
                ax.plot(
                    coordinates * 1.0e6,
                    compositions[:, i + 1],
                    label=f"X({element})" if phase_index == 0 else None,
                )
        for position in self.getInterfacePositions():
            ax.axvline(position * 1.0e6, color="0.35", linestyle="--", linewidth=1)
        ax.set_xlabel("Distance (um)")
        ax.set_ylabel("Mole fraction")
        ax.legend()
        fig.tight_layout()
        plt.show(block=False)
        return fig, ax

    def plot_phaseWidths_vs_time(self):
        """Plots recorded phase widths using the historical moving right boundary."""
        import matplotlib.pyplot as plt
        count = self.interfaceData.N + 1
        y = self.interfaceData._y[:count]
        right = self.rightBoundaryData._y[:count]
        time = self.interfaceData._time[:count]
        if self.rightBoundaryData.N + 1 != count or not np.array_equal(
            self.rightBoundaryData._time[:count], time
        ):
            raise RuntimeError("Interface and right-boundary histories are not synchronized.")
        widths = np.column_stack((y[:, 0], y[:, 1] - y[:, 0], right - y[:, 1]))
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, phase in enumerate(self.phases):
            ax.plot(time, widths[:, i], label=f"{phase} width")
        ax.legend()
        plt.show(block=False)
        return fig, ax

    def plot_etas_vs_time(self):
        import matplotlib.pyplot as plt
        self.confirmLastRecordedIndex(self.etaData)
        times = self.etaData._time[: self.etaData.N + 1]
        etas = self.etaData._y[: self.etaData.N + 1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(times, etas[:, 0], label="A|B interface eta")
        ax.plot(times, etas[:, 1], label="B|C interface eta")
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Eta")
        ax.legend()
        plt.show(block=False)
        return fig, ax
