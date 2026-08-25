import warnings
from dataclasses import dataclass

import numpy as np

from kawin.GenericModel import GenericModel
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.MovingBoundaryEquilibrium import CallableTernaryInterfaceEquilibrium
from kawin.diffusion.MovingBoundaryIllingworthTernaryFDM import (
    _ArrayHistory,
    _BULK_DIFFUSIVITY_IMPLICIT,
    _BULK_DIFFUSIVITY_LAGGED,
    _BULK_DIFFUSIVITY_PHASE_UNIFORM,
    _SUPPORTED_BULK_DIFFUSIVITY_MODES,
    _bounded_finite_difference_perturbation,
    _coerce_bulk_diffusivity_mode,
    _loge_arange,
    _validate_eta_bounds,
    _validate_ternary_diffusivity_matrix,
)
from kawin.diffusion.mesh import CartesianFD1D, MixedBoundary1D, PeriodicBoundary1D
from kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D import (
    flatten_1d_coordinates,
    integrate_planar_transformed_profile_sequence,
    reconstruct_planar_transformed_profile_sequence,
    solve_illingworth_block_tridiagonal,
    validate_ternary_profile,
)
from kawin.solver import explicitEulerIterator
from kawin.thermo.Mobility import interstitials


@dataclass(frozen=True, slots=True)
class _ThreePhaseBulkResult:
    """Result from one transformed interval solve in the three-phase model.

    ``left_transfer`` and ``right_transfer`` are the total ALE plus diffusive transfers
    over the timestep on the interface-adjacent internal faces. They use the
    same orientation as the interval grid, from smaller to larger transformed
    coordinate.
    """

    profile: np.ndarray
    left_transfer: np.ndarray
    right_transfer: np.ndarray
    left_face_matrix: np.ndarray | None
    right_face_matrix: np.ndarray | None
    inner_iterations: int = 0
    inner_update_norm: float = 0.0
    diffusivity_evaluations: int = 0


@dataclass(frozen=True, slots=True)
class _ThreePhaseCandidate:
    """Mutually consistent profiles, interfaces, etas, and residuals for one trial."""

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


@dataclass(frozen=True, slots=True)
class ThreePhaseInitialEtaEstimate:
    """
    Diagnostics from initial tie-line selection for the three-phase model.

    ``etas`` are the selected A|B and B|C tie-line coordinates. ``velocities``
    are instantaneous discrete conservative-balance estimates evaluated with
    initial adjacent-node gradients; they are used only to seed the finite-step
    discrete nonlinear solve.
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


class MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(DiffusionModel):
    """
    Three-phase sequential ternary Illingworth moving-boundary model.

    The model tracks a fixed planar phase order ``A | B | C`` with two moving
    sharp interfaces. Each phase is solved on its own Landau grid with two
    independent substitutional ternary components. The two interfaces are
    advanced simultaneously from four nonlinear residual equations: two
    component balances at ``A|B`` and two at ``B|C``. Phase disappearance is
    not handled; when exactly one phase is thinner than
    ``terminal_thin_phase_width`` after the normal retry budget is exhausted,
    an optional terminal retry path can keep reducing ``dt`` until one final
    converged step is found and then stop the solve early.
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
        initial_eta_root_xtol: float = 1e-12,
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
    ):
        self.initialInterfacePositions = np.asarray(interfacePositions, dtype=np.float64).reshape(-1)
        self.timeStep = float(time_step)
        self.interfaceEquilibria = self._coerce_interface_equilibria(interface_equilibria)
        self.initialEtaGuess = None if initial_eta_guess is None else np.asarray(initial_eta_guess, dtype=np.float64).reshape(-1)
        self.initialEtaMethod = str(initial_eta_method)
        self.initialEtaBrackets = initial_eta_brackets
        self.initialEtaRootXtol = float(initial_eta_root_xtol)
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
        self.profileData = None

        self._currdt = np.inf
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
        self._nearFinalNoop = False
        self._lastStepRetries = 0
        self._lastImplicitIterations = 0
        self._lastImplicitResidual = np.nan
        self._lastImplicitPhysicalResidual = np.nan
        self._lastImplicitConverged = False
        self._lastImplicitFailureReason = None
        self._lastInterfaceCompositions = None
        self._terminalThinPhaseStop = False
        self._terminalThinPhaseInfo = None
        self._initialInventory = None

        self._z = None
        self._R = None
        self._grids = None
        self._profiles_curr = None
        self._interfaces_curr = None
        self._interfaces_old = None
        self._etas_curr = None

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
        if any(e in interstitials for e in self.allElements):
            raise ValueError("MovingBoundaryIllingworthTernaryThreePhaseFD1DModel supports only substitutional systems.")
        if self.geometry != "planar":
            raise NotImplementedError("MovingBoundaryIllingworthTernaryThreePhaseFD1DModel currently implements only planar geometry.")
        if not np.isfinite(self.timeStep) or self.timeStep <= 0.0:
            raise ValueError("time_step must be a positive finite value.")
        if self.dtMode not in {"fixed", "semi_log"}:
            raise ValueError("dt_mode must be 'fixed' or 'semi_log'.")
        if self.dtMode == "semi_log" and ((self.semiLog_dt is None) or (self.semiLogT0 is None)):
            raise ValueError("semiLog_dt and semiLogT0 must be specified when dt_mode is 'semi_log'.")
        if self.initialEtaMethod != "instantaneous_balance":
            raise ValueError("initial_eta_method must be 'instantaneous_balance'.")
        if self.initialEtaRootXtol <= 0.0:
            raise ValueError("initial_eta_root_xtol must be positive.")
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

    def _minimum_middle_width(self):
        return max(float(self._R) * self.minMiddleWidthFraction, 1e-14)

    def _phase_widths(self, interfaces):
        """Returns the physical widths of the ``A``, ``B``, and ``C`` regions."""
        s_ab, s_bc = np.asarray(interfaces, dtype=np.float64).reshape(2)
        return np.asarray([s_ab, s_bc - s_ab, float(self._R) - s_bc], dtype=np.float64)

    def _single_terminal_thin_phase(self, interfaces):
        """Returns ``(index, width)`` only when exactly one phase is below the terminal threshold."""
        if self.terminalThinPhaseWidth is None:
            return None
        widths = self._phase_widths(interfaces)
        mask = np.isfinite(widths) & (widths < self.terminalThinPhaseWidth)
        if int(np.count_nonzero(mask)) != 1:
            return None
        index = int(np.flatnonzero(mask)[0])
        return index, float(widths[index])

    def _validate_interfaces(self, interfaces, strict=True):
        values = np.asarray(interfaces, dtype=np.float64).reshape(2)
        if self._R is None:
            z = flatten_1d_coordinates(self.mesh.z)
            domain_length = float(z[-1] - z[0])
        else:
            domain_length = float(self._R)
        eps = max(domain_length * 1e-14, 1e-14)
        min_width = max(domain_length * self.minMiddleWidthFraction, eps)
        valid = eps < values[0] and values[0] + min_width < values[1] and values[1] < domain_length - eps
        if strict and not valid:
            raise ValueError("Three-phase interface positions must satisfy 0 < s_AB < s_BC < R with a nonzero middle phase.")
        values[0] = np.clip(values[0], eps, domain_length - eps)
        values[1] = np.clip(values[1], eps, domain_length - eps)
        if values[1] - values[0] < min_width:
            center = 0.5 * (values[0] + values[1])
            values[0] = center - 0.5 * min_width
            values[1] = center + 0.5 * min_width
        return values

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

    def _interface_face_diffusivity(self, composition, phase_index, position, context):
        """Returns one validated diffusivity matrix for an initial interface-face estimate."""
        T = np.asarray(self.temperatureParameters(np.asarray([[float(position)]], dtype=np.float64), 0.0), dtype=np.float64).reshape(-1)
        try:
            D = self.therm.getInterdiffusivity(composition, float(T[0]), phase=self.phases[phase_index], query_context=context)
        except TypeError:
            D = self.therm.getInterdiffusivity(composition, float(T[0]), phase=self.phases[phase_index])
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

    def _initial_face_transfer_rate(self, face_velocity, D_face, length, dxi, left_value, right_value):
        """
        Returns the instantaneous conservative ALE plus diffusive face transfer.

        The sign convention matches the transient interval transfer ``H``:
        positive physical face velocity uses the right transformed node as the
        ALE donor because the transformed advection velocity has the opposite
        sign.
        """
        diffusive = np.matmul(np.asarray(D_face, dtype=np.float64), (np.asarray(right_value, dtype=np.float64) - np.asarray(left_value, dtype=np.float64)) / (float(length) * float(dxi)))
        face_velocity = float(face_velocity)
        if face_velocity > 0.0:
            return face_velocity * np.asarray(right_value, dtype=np.float64) + diffusive
        if face_velocity < 0.0:
            return face_velocity * np.asarray(left_value, dtype=np.float64) + diffusive
        return diffusive

    def _initial_discrete_interface_terms(self, interfaces, interface_compositions, adjacent, velocities, *, return_zero_velocity=False):
        """
        Evaluates the discrete conservative ALE instantaneous interface balance.

        Initial non-interface profile nodes and phase widths are held fixed.
        ``velocities`` contains ``[V_AB, V_BC]``; eta affects only the
        equilibrium endpoint compositions supplied in ``interface_compositions``.
        When requested, the zero-velocity residual is returned from the same
        face diffusivity evaluation so diagnostics do not repeat thermodynamic
        work during the nonlinear initial-eta solve.
        """
        s_ab, s_bc = np.asarray(interfaces, dtype=np.float64).reshape(2)
        v_ab, v_bc = np.asarray(velocities, dtype=np.float64).reshape(2)
        p_a, p_b_left, p_b_right, p_c = adjacent
        c_a_ab, c_b_ab = interface_compositions[0]
        c_b_bc, c_c_bc = interface_compositions[1]
        length_a = s_ab
        length_b = s_bc - s_ab
        length_c = self._R - s_bc
        u_a = self._grids[0]
        u_b = self._grids[1]
        u_c = self._grids[2]
        D_a, D_b_left, D_b_right, D_c = self._initial_face_diffusivity_matrices(interfaces, interface_compositions, adjacent)

        xi_a_right = 0.5 * (u_a[-2] + 1.0)
        xi_b_left = 0.5 * u_b[1]
        xi_b_right = 0.5 * (u_b[-2] + 1.0)
        xi_c_left = 0.5 * u_c[1]
        w_a_right = xi_a_right * v_ab
        w_b_left = (1.0 - xi_b_left) * v_ab + xi_b_left * v_bc
        w_b_right = (1.0 - xi_b_right) * v_ab + xi_b_right * v_bc
        w_c_left = (1.0 - xi_c_left) * v_bc

        transfer_a_right = self._initial_face_transfer_rate(w_a_right, D_a, length_a, 1.0 - u_a[-2], p_a, c_a_ab)
        transfer_b_left = self._initial_face_transfer_rate(w_b_left, D_b_left, length_b, u_b[1], c_b_ab, p_b_left)
        transfer_b_right = self._initial_face_transfer_rate(w_b_right, D_b_right, length_b, 1.0 - u_b[-2], p_b_right, c_b_bc)
        transfer_c_left = self._initial_face_transfer_rate(w_c_left, D_c, length_c, u_c[1], c_c_bc, p_c)

        length_rate_a = v_ab
        length_rate_b = v_bc - v_ab
        length_rate_c = -v_bc
        dm_a_right = 0.5 * (1.0 - u_a[-2]) * length_rate_a * c_a_ab
        dm_b_left = 0.5 * u_b[1] * length_rate_b * c_b_ab
        dm_b_right = 0.5 * (1.0 - u_b[-2]) * length_rate_b * c_b_bc
        dm_c_left = 0.5 * u_c[1] * length_rate_c * c_c_bc

        residual_ab = dm_a_right + dm_b_left + transfer_a_right - transfer_b_left
        residual_bc = dm_b_right + dm_c_left + transfer_b_right - transfer_c_left
        residual = np.concatenate((residual_ab, residual_bc))
        if not return_zero_velocity:
            return residual, None

        zero_a_right = self._initial_face_transfer_rate(0.0, D_a, length_a, 1.0 - u_a[-2], p_a, c_a_ab)
        zero_b_left = self._initial_face_transfer_rate(0.0, D_b_left, length_b, u_b[1], c_b_ab, p_b_left)
        zero_b_right = self._initial_face_transfer_rate(0.0, D_b_right, length_b, 1.0 - u_b[-2], p_b_right, c_b_bc)
        zero_c_left = self._initial_face_transfer_rate(0.0, D_c, length_c, u_c[1], c_c_bc, p_c)
        zero_velocity_residual = np.concatenate((zero_a_right - zero_b_left, zero_b_right - zero_c_left))
        return residual, zero_velocity_residual

    def _initial_discrete_interface_residuals(self, interfaces, interface_compositions, adjacent, velocities):
        """Returns the instantaneous residual from ``_initial_discrete_interface_terms``."""
        residual, _ = self._initial_discrete_interface_terms(interfaces, interface_compositions, adjacent, velocities)
        return residual

    def _estimate_initial_etas(self, composition, interfaces):
        """
        Solves the discrete instantaneous two-interface balance for startup etas.

        The unknowns are two scaled interface velocities and one eta for each
        interface. Initial phase widths and non-interface transformed profile
        nodes are held fixed, and the residual is the ``dt -> 0`` limit of the
        conservative ALE finite-step interface balance.
        """
        brackets = self._initial_eta_brackets()
        eta0 = self._initial_eta_start(brackets)
        adjacent = self._initial_adjacent_compositions(composition, interfaces)

        def evaluate(velocities, etas):
            interface_compositions = self._interface_compositions(etas)
            residual, zero_velocity_residual = self._initial_discrete_interface_terms(
                interfaces,
                interface_compositions,
                adjacent,
                velocities,
                return_zero_velocity=True,
            )
            flux_delta = -zero_velocity_residual
            return interface_compositions, residual, flux_delta

        _, residual_at_zero, _ = evaluate(np.zeros(2, dtype=np.float64), eta0)
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
            interface_compositions, residual, flux_delta = evaluate(velocities, etas)
            if not np.all(np.isfinite(residual)):
                raise ValueError("Initial eta residual is non-finite.")
            return residual, flux_delta, interface_compositions

        for iteration in range(1, self.initialEtaRootMaxiter + 1):
            residual, flux_delta, interface_compositions = residual_unknowns(x)
            nfev += 1
            norm_current = float(np.max(np.abs(residual)))
            if best is None or norm_current < best[0]:
                best = (norm_current, x.copy(), residual.copy(), flux_delta.copy(), interface_compositions)
            if norm_current <= self.initialEtaRootXtol:
                success = True
                break

            jacobian = np.zeros((4, 4), dtype=np.float64)
            for variable in range(4):
                step, x_perturbed, direction = _bounded_finite_difference_perturbation(x, lower, upper, variable)
                residual_perturbed, _, _ = residual_unknowns(x_perturbed)
                nfev += 1
                if direction == "forward":
                    jacobian[:, variable] = (residual_perturbed - residual) / step
                else:
                    jacobian[:, variable] = (residual - residual_perturbed) / step

            try:
                step = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]

            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = np.clip(x + scale * step, lower, upper)
                residual_trial, flux_delta_trial, interface_compositions_trial = residual_unknowns(trial)
                nfev += 1
                norm_trial = float(np.max(np.abs(residual_trial)))
                if np.isfinite(norm_trial) and norm_trial < norm_current:
                    x = trial
                    accepted = True
                    if best is None or norm_trial < best[0]:
                        best = (norm_trial, trial.copy(), residual_trial.copy(), flux_delta_trial.copy(), interface_compositions_trial)
                    break
            if not accepted:
                break

        if best is None:
            raise ValueError("Instantaneous initial eta solve did not produce a finite residual.")
        residual_norm, x_best, residual, flux_delta, interface_compositions = best
        if not success and residual_norm > self.initialEtaRootXtol:
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
        )
        self.initialEtaEstimate = estimate
        return estimate.etas.copy()

    def _interface_compositions(self, etas):
        etas = np.asarray(etas, dtype=np.float64).reshape(2)
        out = []
        for i, (closure, eta) in enumerate(zip(self.interfaceEquilibria, etas)):
            left, right = closure.interface_compositions(float(eta))
            left = self._validate_composition_vector(left, f"interface {i} left composition")
            right = self._validate_composition_vector(right, f"interface {i} right composition")
            out.append((left, right))
        return tuple(out)

    def _validate_composition_vector(self, values, name):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        min_comp = float(self.constraints.minComposition)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite two-component vector.")
        dependent = 1.0 - float(np.sum(values))
        if np.any(values < min_comp) or dependent < min_comp:
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
        interface residuals are considered converged.
        """
        profile = np.asarray(profile, dtype=np.float64)
        min_comp = float(self.constraints.minComposition)
        dependent = 1.0 - np.sum(profile, axis=1)
        component_bad = np.argwhere(profile < min_comp)
        if component_bad.size:
            node, component = component_bad[0]
            return f"node {int(node)} component {int(component)} = {profile[node, component]:.6g}"
        dependent_bad = np.flatnonzero(dependent < min_comp)
        if dependent_bad.size:
            node = int(dependent_bad[0])
            return f"node {node} dependent component = {dependent[node]:.6g}"
        return None

    def _validate_candidate_profiles(self, profiles):
        """Rejects nonlinear trial states whose transformed profiles leave the ternary simplex."""
        for i, profile in enumerate(profiles):
            profile = validate_ternary_profile(profile, f"candidate transformed profile {i}")
            violation = self._profile_composition_violation(profile)
            if violation is not None:
                raise ValueError(f"candidate transformed profile {i} violates ternary composition bounds: {violation}.")

    def _is_infeasible_candidate_error(self, exc):
        """Returns True for candidate states rejected by admissible-composition bounds."""
        message = str(exc)
        return isinstance(exc, ValueError) and (
            "candidate transformed profile" in message
            or "violates ternary composition bounds" in message
        )

    def setup(self):
        super().setup()
        self._validateModelConfiguration()
        self._getBoundaryConditions()
        self._z = flatten_1d_coordinates(self.mesh.z).astype(np.float64)
        if not np.isclose(self._z[0], 0.0):
            raise ValueError("MovingBoundaryIllingworthTernaryThreePhaseFD1DModel expects a 1D domain starting at 0.")
        self._R = float(self._z[-1] - self._z[0])
        interfaces0 = self._validate_interfaces(self.initialInterfacePositions, strict=True)
        self.initialInterfacePositions = interfaces0.copy()
        self._grids = self._build_transformed_grids(interfaces0)
        self.profileData = [_ArrayHistory((len(grid), 2), self.interfaceData.recordInterval) for grid in self._grids] if self.recordPqData else None

        c0 = np.asarray(self.data.currentY, dtype=np.float64)
        etas0 = self._estimate_initial_etas(c0, interfaces0)
        profiles = self._initialize_transformed_state(c0, interfaces0, etas0)
        self._profiles_curr = tuple(profile.copy() for profile in profiles)
        self._interfaces_curr = interfaces0.copy()
        self._interfaces_old = interfaces0.copy()
        self._etas_curr = etas0.copy()
        self._lastInterfaceCompositions = self._interface_compositions(etas0)

        self.interfaceData.reset()
        self.interfaceData.record(0, interfaces0)
        self.etaData.reset()
        self.etaData.record(0, etas0)
        if self.recordPqData:
            for history, profile in zip(self.profileData, self._profiles_curr):
                history.record(0, profile)

        physical = self._reconstruct_physical_profile(self._profiles_curr, interfaces0)
        self.data.currentY = physical
        self.data._y[0] = physical
        self._initialInventory = self.getTotalInventoryFromState(self._profiles_curr, interfaces0)
        self.inventoryData.reset()
        self.inventoryData.record(0, self._initialInventory)

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

    def _reconstruct_physical_profile(self, profiles, interfaces):
        return reconstruct_planar_transformed_profile_sequence(
            z=self._z,
            profiles=profiles,
            interfaces=interfaces,
            domain_length=self._R,
            grids=self._grids,
        )

    def setTimeInfo(self, currTime, simTime):
        """Stores solve-time bounds and prepares optional semi-log target times."""
        super().setTimeInfo(currTime, simTime)
        self._currdt = np.inf
        self._nearFinalNoop = False
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
        step_scale = self.timeStep
        if self.dtMode == "semi_log":
            scheduled_dt = self._computeSemiLogDt(t)
            if np.isfinite(scheduled_dt) and scheduled_dt > 0:
                step_scale = scheduled_dt
            dt = min(scheduled_dt, remaining)
        else:
            dt = min(self.timeStep, remaining)
        self._nearFinalNoop = bool(np.isfinite(remaining) and 0 < remaining <= max(step_scale, 1e-15) * 1e-10)
        if self._nearFinalNoop:
            self._currdt = max(step_scale, 1e-15)
            return self._currdt
        if not np.isfinite(dt) or dt <= 0:
            dt = self.timeStep
        self._currdt = float(dt)
        return float(dt)

    def solve(self, simTime, iterator=explicitEulerIterator, verbose=False, vIt=10, minDtFrac=1e-8, maxDtFrac=1):
        """Solves the implicit three-phase recurrence through an Euler wrapper."""
        if iterator is not explicitEulerIterator:
            raise ValueError("MovingBoundaryIllingworthTernaryThreePhaseFD1DModel supports only explicitEulerIterator.")
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
        """Returns validated bulk diffusivity matrices for interval faces."""
        face_compositions = np.asarray(face_compositions, dtype=np.float64)
        if face_compositions.ndim != 2 or face_compositions.shape[1] != 2:
            raise ValueError("face_compositions must have shape (n_faces, 2).")
        temperatures = self._temperatures_at_positions(physical_face_positions, time)

        def validate_stack(values):
            matrices = np.asarray(values)
            if matrices.shape == (2, 2) and face_compositions.shape[0] == 1:
                matrices = matrices.reshape(1, 2, 2)
            if matrices.shape != (face_compositions.shape[0], 2, 2):
                raise ValueError("bulk diffusivity query returned an unexpected matrix shape.")
            out = np.empty(matrices.shape, dtype=np.float64)
            for i, matrix in enumerate(matrices):
                out[i] = _validate_ternary_diffusivity_matrix(matrix, phase, context=f"bulk face diffusivity face {i}")
            return out

        try:
            values = self.therm.getInterdiffusivity(face_compositions, temperatures, phase=phase, query_context="general")
        except (TypeError, ValueError):
            pass
        else:
            return validate_stack(values)

        out = np.empty((face_compositions.shape[0], 2, 2), dtype=np.float64)
        for i, (composition, temperature) in enumerate(zip(face_compositions, temperatures)):
            try:
                D = self.therm.getInterdiffusivity(composition, float(temperature), phase=phase, query_context="general")
            except TypeError:
                D = self.therm.getInterdiffusivity(composition, float(temperature), phase=phase)
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
        try:
            D = self.therm.getInterdiffusivity(composition, float(T[0]), phase=self.phases[phase_index], query_context="interface")
        except TypeError:
            D = self.therm.getInterdiffusivity(composition, float(T[0]), phase=self.phases[phase_index])
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

    def _internal_face_displacements(self, grid, old_bounds, new_bounds):
        """Returns physical displacement of each internal transformed face."""
        grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        face_xi = 0.5 * (grid[:-1] + grid[1:])
        old_left, old_right = float(old_bounds[0]), float(old_bounds[1])
        new_left, new_right = float(new_bounds[0]), float(new_bounds[1])
        return (1.0 - face_xi) * (new_left - old_left) + face_xi * (new_right - old_right)

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

    def _solve_interval_planar(self, profile, grid, old_bounds, new_bounds, phase_index, left_value, right_value, D_faces, validate_diffusivity=True):
        """
        Solves one transformed planar interval on a moving Landau grid.

        The finite-volume equation is assembled face by face as
        ``L_new*DeltaXi*C_new - L_old*DeltaXi*C_old = H_right - H_left``.
        ``H`` contains both the face-local upwind ALE transfer and the implicit
        diffusive transfer. Endpoint values are exact Dirichlet constraints when
        supplied; ``None`` means the fixed external boundary has homogeneous
        zero transfer.
        """
        profile = validate_ternary_profile(profile, "interval profile")
        grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        n = len(profile)
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

        for i in range(n):
            if fixed_rows[i]:
                diagonal[i] = I
                rhs[i] = np.asarray(left_value if i == 0 else right_value, dtype=np.float64)
                continue
            diagonal[i] = new_widths[i] * I
            rhs[i] = old_widths[i] * profile[i]

        face_displacements = self._internal_face_displacements(grid, old_bounds, new_bounds)
        face_coefficients = []
        for face in range(n - 1):
            left_coeff, right_coeff = self._face_transfer_coefficients(
                face_displacements[face],
                D_faces[face],
                float(self._currdt),
                new_length,
                grid[face + 1] - grid[face],
            )
            face_coefficients.append((left_coeff, right_coeff))
            left_row = face
            right_row = face + 1
            if not fixed_rows[left_row]:
                diagonal[left_row] -= left_coeff
                upper[left_row] -= right_coeff
            if not fixed_rows[right_row]:
                lower[right_row] += left_coeff
                diagonal[right_row] += right_coeff

        solved = solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)
        left_matrix = None if left_value is None else D_faces[0]
        right_matrix = None if right_value is None else D_faces[-1]
        zero = np.zeros(2, dtype=np.float64)
        if left_value is None:
            left_transfer = zero.copy()
        else:
            left_transfer = self._evaluate_face_transfer(*face_coefficients[0], solved[0], solved[1])
        if right_value is None:
            right_transfer = zero.copy()
        else:
            right_transfer = self._evaluate_face_transfer(*face_coefficients[-1], solved[-2], solved[-1])
        return _ThreePhaseBulkResult(solved, left_transfer, right_transfer, left_matrix, right_matrix)

    def _solve_interval_picard(self, profile, grid, old_bounds, new_bounds, phase_index, left_value, right_value):
        """Solves one interval with composition-dependent Picard face matrices."""
        iterate = np.asarray(profile, dtype=np.float64).copy()
        if left_value is not None:
            iterate[0] = left_value
        if right_value is not None:
            iterate[-1] = right_value
        update_norm = np.inf
        for iteration in range(1, self.bulkPicardMaxIterations + 1):
            D_faces = self._interval_face_diffusivity(iterate, grid, phase_index, new_bounds)
            linear = self._solve_interval_planar(profile, grid, old_bounds, new_bounds, phase_index, left_value, right_value, D_faces, validate_diffusivity=False)
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
                    inner_iterations=iteration,
                    inner_update_norm=update_norm,
                    diffusivity_evaluations=len(profile) - 1,
                )
            iterate = iterate + self.bulkPicardRelaxation * update
        raise RuntimeError(f"bulk Picard solve failed to converge for phase {self.phases[phase_index]} after {self.bulkPicardMaxIterations} iterations")

    def _solve_bulk_profiles(self, profiles, old_interfaces, new_interfaces, interface_compositions):
        old_bounds = np.asarray([[0.0, old_interfaces[0]], [old_interfaces[0], old_interfaces[1]], [old_interfaces[1], self._R]], dtype=np.float64)
        new_bounds = np.asarray([[0.0, new_interfaces[0]], [new_interfaces[0], new_interfaces[1]], [new_interfaces[1], self._R]], dtype=np.float64)
        boundary_values = (
            (None, interface_compositions[0][0]),
            (interface_compositions[0][1], interface_compositions[1][0]),
            (interface_compositions[1][1], None),
        )
        results = []
        for phase_index in range(3):
            left_value, right_value = boundary_values[phase_index]
            if self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_PHASE_UNIFORM:
                D = self._phase_uniform_diffusivity(phase_index, interface_compositions, new_interfaces)
                result = self._solve_interval_planar(profiles[phase_index], self._grids[phase_index], old_bounds[phase_index], new_bounds[phase_index], phase_index, left_value, right_value, D)
            elif self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_LAGGED:
                D = self._interval_face_diffusivity(profiles[phase_index], self._grids[phase_index], phase_index, new_bounds[phase_index])
                result = self._solve_interval_planar(profiles[phase_index], self._grids[phase_index], old_bounds[phase_index], new_bounds[phase_index], phase_index, left_value, right_value, D, validate_diffusivity=False)
            elif self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_IMPLICIT:
                result = self._solve_interval_picard(profiles[phase_index], self._grids[phase_index], old_bounds[phase_index], new_bounds[phase_index], phase_index, left_value, right_value)
            else:
                raise ValueError(f"Unsupported bulk diffusivity mode {self.bulkDiffusivityMode}.")
            results.append(result)
        return tuple(results)

    def _interface_residuals(self, old_profiles, old_interfaces, new_interfaces, interface_compositions, bulk_results):
        """
        Returns interface residuals from the same discrete transfers as the bulk.

        Endpoint inventory is represented by the trapezoidal endpoint half-cell
        weights. Combining those endpoint inventory changes with the
        interface-adjacent total face transfers makes the residual telescope
        exactly with the conservative bulk cell equations.
        """
        c_a_ab, c_b_ab = interface_compositions[0]
        c_b_bc, c_c_bc = interface_compositions[1]
        old_lengths = np.asarray(
            [
                float(old_interfaces[0]),
                float(old_interfaces[1] - old_interfaces[0]),
                float(self._R - old_interfaces[1]),
            ],
            dtype=np.float64,
        )
        new_lengths = np.asarray(
            [
                float(new_interfaces[0]),
                float(new_interfaces[1] - new_interfaces[0]),
                float(self._R - new_interfaces[1]),
            ],
            dtype=np.float64,
        )
        endpoint_ab = self._endpoint_inventory_change(self._grids[0], old_profiles[0], c_a_ab, old_lengths[0], new_lengths[0], "right")
        endpoint_ab += self._endpoint_inventory_change(self._grids[1], old_profiles[1], c_b_ab, old_lengths[1], new_lengths[1], "left")
        endpoint_bc = self._endpoint_inventory_change(self._grids[1], old_profiles[1], c_b_bc, old_lengths[1], new_lengths[1], "right")
        endpoint_bc += self._endpoint_inventory_change(self._grids[2], old_profiles[2], c_c_bc, old_lengths[2], new_lengths[2], "left")
        residual_ab = endpoint_ab + bulk_results[0].right_transfer - bulk_results[1].left_transfer
        residual_bc = endpoint_bc + bulk_results[1].right_transfer - bulk_results[2].left_transfer
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
        inventory = self.getTotalInventoryFromState(profiles, interfaces)
        return np.maximum(np.maximum(np.repeat(np.maximum(np.abs(inventory), self._R), 2), 1e-300), 1e-300)

    def _scaled_bounds(self):
        eps = 1e-14
        return np.asarray([eps, eps, 0.0, 0.0], dtype=np.float64), np.asarray([1.0 - eps, 1.0 - eps, 1.0, 1.0], dtype=np.float64)

    def _physical_to_scaled(self, interfaces, etas):
        bounds = self._eta_bounds()
        return np.asarray(
            [
                interfaces[0] / self._R,
                interfaces[1] / self._R,
                (etas[0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]),
                (etas[1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]),
            ],
            dtype=np.float64,
        )

    def _scaled_to_physical(self, x_hat):
        x_hat = np.asarray(x_hat, dtype=np.float64).reshape(4)
        bounds = self._eta_bounds()
        interfaces = np.asarray([x_hat[0] * self._R, x_hat[1] * self._R], dtype=np.float64)
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
        return interfaces[1] - interfaces[0] >= self._minimum_middle_width()

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
            gap_hat = (x_hat[1] - x_hat[0]) - self._minimum_middle_width() / self._R
            alpha_max = min(alpha_max, max(0.0, float(gap_hat / (step[0] - step[1]))))
        return step, max(0.0, alpha_max * (1.0 - 1e-12))

    def _evaluate_interface_candidate(self, profiles, old_interfaces, x_hat, residual_scale, dt):
        interfaces, etas = self._scaled_to_physical(x_hat)
        interfaces = self._validate_interfaces(interfaces, strict=True)
        interface_compositions = self._interface_compositions(etas)
        bulk_results = self._solve_bulk_profiles(profiles, old_interfaces, interfaces, interface_compositions)
        candidate_profiles = tuple(result.profile for result in bulk_results)
        self._validate_candidate_profiles(candidate_profiles)
        residual = self._interface_residuals(profiles, old_interfaces, interfaces, interface_compositions, bulk_results)
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
        )

    def _solve_interface_planar(self, profiles, interfaces, etas, dt):
        lower, upper = self._scaled_bounds()
        x_hat = self._physical_to_scaled(interfaces, etas)
        if not self._scaled_variables_in_bounds(x_hat, lower, upper):
            raise ValueError("Initial three-phase nonlinear iterate lies outside scaled solve bounds.")
        residual_scale = self._residual_scale(profiles, interfaces)
        best = None
        failure_reason = "maximum iterations reached"
        for iteration in range(1, self.maxIterations + 1):
            candidate = self._evaluate_interface_candidate(profiles, interfaces, x_hat, residual_scale, dt)
            if best is None or candidate.scaled_norm < best.scaled_norm:
                best = candidate
            if candidate.scaled_norm <= self.residualTolerance:
                self._lastImplicitIterations = iteration
                self._lastImplicitResidual = candidate.scaled_norm
                self._lastImplicitPhysicalResidual = candidate.physical_norm
                self._lastImplicitConverged = True
                self._lastImplicitFailureReason = None
                return candidate

            jacobian = np.zeros((4, 4), dtype=np.float64)
            for variable in range(4):
                step_size, x_perturbed, direction = _bounded_finite_difference_perturbation(x_hat, lower, upper, variable)
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
                    if np.isfinite(trial_candidate.scaled_norm) and trial_candidate.scaled_norm < candidate.scaled_norm:
                        x_hat = trial
                        accepted = True
                        break
                if not accepted:
                    failure_reason = "line search failed"
                    break
                continue

            break
        self._lastImplicitIterations = self.maxIterations
        self._lastImplicitResidual = np.inf if best is None else best.scaled_norm
        self._lastImplicitPhysicalResidual = np.inf if best is None else best.physical_norm
        self._lastImplicitConverged = False
        self._lastImplicitFailureReason = failure_reason
        raise RuntimeError(f"Three-phase Illingworth interface solve failed to converge; best residual was {self._lastImplicitResidual:.3e}.")

    def _accept_step_candidate(self, profiles, interfaces, etas, candidate, trial_dt, retry):
        """Stores diagnostics and returns derivatives for an accepted implicit trial."""
        self._currdt = float(trial_dt)
        self._lastStepRetries = int(retry)
        self._lastInterfaceCompositions = candidate.interface_compositions
        return [
            (candidate.profiles[0] - profiles[0]) / trial_dt,
            (candidate.profiles[1] - profiles[1]) / trial_dt,
            (candidate.profiles[2] - profiles[2]) / trial_dt,
            (candidate.interfaces - interfaces) / trial_dt,
            (candidate.etas - etas) / trial_dt,
        ]

    def _try_step_retries(self, profiles, interfaces, etas, trial_dt, retry_count, retry_offset=0):
        """Attempts implicit solves while shrinking ``trial_dt`` after each failed trial."""
        last_error = None
        for retry in range(int(retry_count)):
            self._currdt = trial_dt
            try:
                candidate = self._solve_interface_planar(profiles, interfaces, etas, trial_dt)
                return self._accept_step_candidate(profiles, interfaces, etas, candidate, trial_dt, retry_offset + retry), trial_dt, None
            except (RuntimeError, ValueError, ZeroDivisionError) as exc:
                last_error = exc
                trial_dt *= self.retryFactor
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
        profiles = tuple(np.asarray(xCurr[i], dtype=np.float64).reshape((-1, 2)).copy() for i in range(3))
        interfaces = self._validate_interfaces(np.asarray(xCurr[3], dtype=np.float64), strict=True)
        etas = np.asarray(xCurr[4], dtype=np.float64).reshape(2)
        interface_compositions = self._interface_compositions(etas)
        profiles[0][-1] = interface_compositions[0][0]
        profiles[1][0] = interface_compositions[0][1]
        profiles[1][-1] = interface_compositions[1][0]
        profiles[2][0] = interface_compositions[1][1]
        dt = self._compute_dt(t)
        if self._nearFinalNoop:
            return [np.zeros_like(profiles[0]), np.zeros_like(profiles[1]), np.zeros_like(profiles[2]), np.zeros(2), np.zeros(2)]
        trial_dt = float(dt)
        dXdt, trial_dt, last_error = self._try_step_retries(profiles, interfaces, etas, trial_dt, self.maxStepRetries)
        if dXdt is not None:
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
                )
                if dXdt is not None:
                    if not( self._single_terminal_thin_phase(xCurr[3]+dXdt[3]*trial_dt) is not None):
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
        print(f"t: {t}")
        print(f"etas: {etas}")
        print(f"interfaces: {interfaces}")
        print(f"self._lastImplicitFailureReason: {self._lastImplicitFailureReason}")
        raise RuntimeError("Three-phase Illingworth step failed after timestep retries.") from last_error

    def getDt(self, dXdt):
        if np.isfinite(self._currdt) and self._currdt > 0.0:
            return self._currdt
        return self.timeStep

    def postProcess(self, time, x):
        if self._nearFinalNoop:
            self.currentTime = time
            self._nearFinalNoop = False
            return self.getCurrentX(), True
        GenericModel.postProcess(self, time, x)
        profiles = tuple(np.asarray(x[i], dtype=np.float64).reshape((-1, 2)).copy() for i in range(3))
        interfaces = self._validate_interfaces(np.asarray(x[3], dtype=np.float64), strict=True)
        etas = np.asarray(x[4], dtype=np.float64).reshape(2)
        interface_compositions = self._interface_compositions(etas)
        profiles[0][-1] = interface_compositions[0][0]
        profiles[1][0] = interface_compositions[0][1]
        profiles[1][-1] = interface_compositions[1][0]
        profiles[2][0] = interface_compositions[1][1]
        for i, profile in enumerate(profiles):
            self._validate_profile_compositions(profile, f"transformed profile {i}")
        physical = self._reconstruct_physical_profile(profiles, interfaces)
        self.data.record(time, physical)
        self.interfaceData.record(time, interfaces)
        self.etaData.record(time, etas)
        if self.recordPqData:
            for history, profile in zip(self.profileData, profiles):
                history.record(time, profile)
        self.inventoryData.record(time, self.getTotalInventoryFromState(profiles, interfaces))

        self._interfaces_old = self._interfaces_curr.copy()
        self._profiles_curr = tuple(profile.copy() for profile in profiles)
        self._interfaces_curr = interfaces.copy()
        self._etas_curr = etas.copy()
        self.updateCoupledModels()
        if self._terminalThinPhaseStop:
            self.finalTime = time
            self._terminalThinPhaseStop = False
            return self.getCurrentX(), True
        return self.getCurrentX(), False

    def postSolve(self):
        self.data.finalize()
        self.interfaceData.finalize()
        self.etaData.finalize()
        self.inventoryData.finalize()
        if self.profileData is not None:
            for history in self.profileData:
                history.finalize()

    def reset(self):
        super().reset()
        self.interfaceData.reset()
        self.interfaceData.record(0, self.initialInterfacePositions)
        self.etaData.reset()
        self.inventoryData.reset()
        self.profileData = None
        self._currdt = np.inf
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
        self._nearFinalNoop = False
        self._lastStepRetries = 0
        self._lastImplicitIterations = 0
        self._lastImplicitResidual = np.nan
        self._lastImplicitPhysicalResidual = np.nan
        self._lastImplicitConverged = False
        self._lastImplicitFailureReason = None
        self._lastInterfaceCompositions = None
        self._terminalThinPhaseStop = False
        self._terminalThinPhaseInfo = None
        self._initialInventory = None
        self._z = None
        self._R = None
        self._grids = None
        self._profiles_curr = None
        self._interfaces_curr = None
        self._interfaces_old = None
        self._etas_curr = None

    def getInterfacePositions(self, time=None):
        """Returns the two recorded interface positions ``[s_AB, s_BC]``."""
        return self.interfaceData.y(time)

    def getInterfaceEtas(self, time=None):
        """Returns the two recorded tie-line coordinates ``[eta_AB, eta_BC]``."""
        return self.etaData.y(time)

    def getInterfaceCompositions(self, time=None):
        """Returns ``((A_AB, B_AB), (B_BC, C_BC))`` interface compositions."""
        return self._interface_compositions(self.getInterfaceEtas(time))

    def getTransformedState(self, time=None):
        """Returns the three recorded transformed phase profiles."""
        if self.profileData is None:
            raise ValueError("Transformed profile history is not available; set record_pq_data=True.")
        return tuple(history.y(time) for history in self.profileData)

    def getTotalInventoryFromState(self, profiles, interfaces):
        """Returns componentwise inventory for the three transformed intervals."""
        return integrate_planar_transformed_profile_sequence(profiles, interfaces, self._R, self._grids)

    def getTotalInventory(self, time=None):
        if time is None:
            return self.getTotalInventoryFromState(self._profiles_curr, self._interfaces_curr)
        return self.inventoryData.y(time)

    def getTotalMass(self, time=None):
        return self.getTotalInventory(time=time)

    def getCompositions(self, time=None):
        """
        Returns full ternary mole fractions on the physical mesh.

        The stored response profile contains the two independent substitutional
        components; the dependent component is reconstructed from the simplex
        constraint.
        """
        independent = np.asarray(self.data.y(time), dtype=np.float64)
        dependent = 1.0 - np.sum(independent, axis=1)
        return np.column_stack((dependent, independent))

    def checkConservation(self, tolerance: float, time=None):
        """
        Checks componentwise transformed-inventory drift from the initial value.

        The return value is the absolute drift vector for the two independent
        components. A warning is emitted when any component exceeds
        ``tolerance``.
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

    def toDict(self):
        """Converts solved three-phase Illingworth histories to a restart dictionary."""
        data = super().toDict()
        data.update(
            {
                "interface_positions": self.interfaceData._y,
                "interface_etas": self.etaData._y,
                "inventory": self.inventoryData._y,
                "interface_interval": self.interfaceData.recordInterval,
                "interface_index": self.interfaceData.N,
            }
        )
        if self.profileData is not None:
            data["profiles"] = np.asarray([history._y for history in self.profileData], dtype=object)
        return data


    ''' Plotting functions for diagnostics '''
    def confirmLastRecordedIndex(self, data):
        if not ((data._time[data.N]!=0).all() and (data._time[data.N+1]==0).all()):
            raise ValueError(f"{data._time[data.N]}==0 or {data._time[data.N+1]}!=0")
        if not ((data._y[data.N]!=0).all() and (data._y[data.N+1]==0).all()):
            raise ValueError(f"{data._y[data.N]}==0 or {data._y[data.N+1]}!=0")

    def plot_latestCompProfile(self):
        import matplotlib.pyplot as plt
        self.confirmLastRecordedIndex(self.data)
        self.confirmLastRecordedIndex(self.interfaceData)
        y = self.data._y[self.interfaceData.N]
        z_um = self._z * 1.0e6
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, element in enumerate(self.elements):
            ax.plot(z_um, y[:, i], label=f"X({element})")
        for position in self.interfaceData._y[self.interfaceData.N]:
            ax.axvline(position * 1.0e6, color="0.35", linestyle="--", linewidth=1)
        ax.set_xlabel("Distance (um)")
        ax.set_ylabel("Mole fraction")
        plt.show(block=False)
        return fig, ax

    def plot_phaseWidths_vs_time(self):
        import matplotlib.pyplot as plt
        self.confirmLastRecordedIndex(self.interfaceData)
        y = self.interfaceData._y[:self.interfaceData.N+1]
        widths = np.column_stack((y[:, 0], y[:, 1] - y[:, 0], self._R - y[:, 1]))
        time=self.interfaceData._time[:self.interfaceData.N+1]
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