import warnings
from dataclasses import dataclass

import numpy as np

from kawin.GenericModel import GenericModel
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.DiffusionParameters import TemperatureParameters
from kawin.diffusion.MovingBoundaryEquilibrium import (
    CallableTernaryInterfaceEquilibrium,
    ThermodynamicTernaryInterfaceEquilibrium,
)
from kawin.diffusion.mesh import CartesianFD1D, MixedBoundary1D, PeriodicBoundary1D
from kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D import (
    flatten_1d_coordinates,
    integrate_planar_transformed_profile_components,
    reconstruct_planar_transformed_profile_components,
    solve_illingworth_block_tridiagonal,
)
from kawin.solver import explicitEulerIterator
from kawin.thermo.Mobility import interstitials

_BULK_DIFFUSIVITY_PHASE_UNIFORM = "phase_uniform"
_BULK_DIFFUSIVITY_LAGGED = "composition_dependent_lagged"
_BULK_DIFFUSIVITY_IMPLICIT = "composition_dependent_implicit"
_SUPPORTED_BULK_DIFFUSIVITY_MODES = {
    _BULK_DIFFUSIVITY_PHASE_UNIFORM,
    _BULK_DIFFUSIVITY_LAGGED,
    _BULK_DIFFUSIVITY_IMPLICIT,
}

def debugInPlace():
    try:
        import debugpy
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('break on this line')
    except:
        pass

def _loge_arange(start, stop, log_step):
    """Returns exponentially spaced target times with fixed natural-log spacing."""
    logs = np.arange(np.log(start), np.log(stop), log_step)
    return np.exp(logs)


def _matvec_2x2(matrix, vector):
    """Multiplies a 2x2 matrix by a length-2 vector without BLAS dispatch."""
    matrix = np.asarray(matrix, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    return np.asarray(
        [
            matrix[0, 0] * vector[0] + matrix[0, 1] * vector[1],
            matrix[1, 0] * vector[0] + matrix[1, 1] * vector[1],
        ],
        dtype=np.float64,
    )


def _bounded_finite_difference_perturbation(x, lower, upper, variable):
    """
    Returns the baseline one-sided finite-difference perturbation for one variable.

    The step-size formula and forward/backward bound choice are shared by the
    initial-eta and finite-step interface Newton loops.
    """
    x = np.asarray(x, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(x[variable]))
    if np.isfinite(upper[variable] - lower[variable]):
        step = min(step, 0.25 * max(upper[variable] - lower[variable], 1e-15))
    x_perturbed = x.copy()
    if x[variable] + step <= upper[variable]:
        x_perturbed[variable] += step
        return step, x_perturbed, "forward"
    x_perturbed[variable] -= step
    return step, x_perturbed, "backward"


def _newton_step_2x2(jacobian, residual):
    """Returns the exact 2x2 Newton step used by the local ternary solves."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64).reshape(2)
    a = float(jacobian[0, 0])
    b = float(jacobian[0, 1])
    c = float(jacobian[1, 0])
    d = float(jacobian[1, 1])
    determinant = a * d - b * c
    if abs(determinant) <= 1e-300:
        raise RuntimeError("Ternary Newton Jacobian is singular.")
    r0 = -float(residual[0])
    r1 = -float(residual[1])
    return np.asarray([(d * r0 - b * r1) / determinant, (-c * r0 + a * r1) / determinant], dtype=np.float64)


def _validate_motion_branch(motion_branch):
    """Validates the two fixed upwind branch labels used by the interface solve."""
    if motion_branch not in {"positive", "negative"}:
        raise ValueError("motion_branch must be 'positive' or 'negative'.")


def _allocate_ternary_block_system(n):
    """Allocates zeroed 2x2 block-tridiagonal arrays for one ternary phase solve."""
    lower = np.zeros((n, 2, 2), dtype=np.float64)
    diagonal = np.zeros((n, 2, 2), dtype=np.float64)
    upper = np.zeros((n, 2, 2), dtype=np.float64)
    rhs = np.zeros((n, 2), dtype=np.float64)
    return lower, diagonal, upper, rhs


def _select_interface_motion_branch(s, old_s, future_s, atol=1e-15):
    """
    Selects the conservative interface upwind branch for one residual evaluation.

    The branch is determined first from ``future_s - s``. If that displacement
    is effectively zero, the previous accepted displacement ``s - old_s`` is
    used as a deterministic fallback; if both are effectively zero the positive
    branch is chosen. The returned string must be passed unchanged to the two
    phase solves and the interface residual for face-flux cancellation.
    """
    delta_s = float(future_s) - float(s)
    if delta_s > atol:
        return "positive"
    if delta_s < -atol:
        return "negative"
    previous_delta_s = float(s) - float(old_s)
    if previous_delta_s < -atol:
        return "negative"
    return "positive"


@dataclass(frozen=True)
class InitialEtaEstimate:
    """
    Diagnostics from initial tie-line selection.

    ``eta`` is the selected tie-line coordinate. ``velocity`` is the scalar
    interface velocity returned by the initialization strategy. ``method``
    identifies the strategy that produced the estimate. ``branch`` is the
    swept-inventory branch used by branch-aware estimators.
    """

    eta: float
    residual_norm: float
    velocity: float
    residual: np.ndarray
    flux_delta: np.ndarray
    left_interface_composition: np.ndarray
    right_interface_composition: np.ndarray
    method: str
    solver: str
    bracket: tuple[float, float]
    converged: bool
    iterations: int
    function_calls: int
    branch: str | None = None


@dataclass(frozen=True, slots=True)
class _InterfaceCandidate:
    """
    Mutually consistent state from one fixed-branch interface trial.

    The concentration profiles, interface compositions, diffusivity matrices,
    branch, and residuals all come from the same trial ``x_hat``. Keeping them
    grouped avoids mixing values from different nonlinear iterations.
    ``frozen=True`` is shallow: fields cannot be rebound, but NumPy array
    contents remain mutable.
    """

    x_hat: np.ndarray
    future_s: float
    future_eta: float
    motion_branch: str
    c_left: np.ndarray
    c_right: np.ndarray
    D_left: np.ndarray
    D_right: np.ndarray
    p_future: np.ndarray
    q_future: np.ndarray
    residual: np.ndarray
    scaled_residual: np.ndarray
    scaled_norm: float
    physical_norm: float
    left_inner_iterations: int = 0
    right_inner_iterations: int = 0
    left_inner_update_norm: float = 0.0
    right_inner_update_norm: float = 0.0
    bulk_diffusivity_evaluations: int = 0


@dataclass(frozen=True, slots=True)
class _BulkPhaseSolveResult:
    """
    Result from one transformed ternary bulk phase solve.

    ``interface_flux`` is the physical diffusive flux at the face adjacent to
    the moving interface, computed from the same face matrix used in the linear
    finite-volume assembly.
    """

    profile: np.ndarray
    interface_flux: np.ndarray
    interface_face_matrix: np.ndarray
    inner_iterations: int = 0
    inner_update_norm: float = 0.0
    converged: bool = True
    failure_reason: str | None = None
    diffusivity_evaluations: int = 0
    diffusivity_provider_calls: int = 0
    face_matrices_evaluated: int = 0


def _validate_ternary_diffusivity_matrix(D, phase, context="ternary Illingworth diffusivity"):
    """
    Validates a ternary 2x2 diffusion matrix with scale-invariant eigen tests.

    Eigenvalues are computed from ``D / ||D||_inf`` so acceptance does not
    depend on diffusivity units. ``imaginary_tol`` and ``positive_tol`` are
    dimensionless tolerances on those scaled eigenvalues; values at or below the
    positivity tolerance are treated as nonpositive. The original unscaled
    matrix is returned after validation.
    """
    values = np.asarray(D)
    label = f"{context} for phase {phase}"
    if values.shape != (2, 2):
        raise ValueError(f"{label} must have shape (2, 2); received {values.shape}.")
    if np.iscomplexobj(values) and np.any(np.imag(values) != 0.0):
        raise ValueError(f"{label} must be real-valued.")
    values = np.asarray(np.real(values), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values.")

    matrix_norm = float(np.linalg.norm(values, ord=np.inf))
    if not np.isfinite(matrix_norm) or matrix_norm <= 0.0:
        raise ValueError(f"{label} must have a positive finite matrix norm.")
    scaled_eigenvalues = np.linalg.eigvals(values / matrix_norm)
    imaginary_tol = 1.0e-12
    positive_tol = 1.0e-14
    if np.any(np.abs(np.imag(scaled_eigenvalues)) > imaginary_tol):
        raise ValueError(f"{label} must have real positive eigenvalues; scaled eigenvalues={scaled_eigenvalues}.")
    real_eigenvalues = np.real(scaled_eigenvalues)
    if np.any(real_eigenvalues <= positive_tol):
        raise ValueError(f"{label} must have strictly positive real eigenvalues; scaled eigenvalues={scaled_eigenvalues}.")
    return values


def _validate_eta_bounds(interface_equilibrium):
    eta_bounds = getattr(interface_equilibrium, "eta_bounds", None)
    if eta_bounds is None or len(eta_bounds) != 2:
        raise ValueError("interface_equilibrium must expose finite eta_bounds for automatic initial tie-line selection.")
    lower, upper = tuple(float(v) for v in eta_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("interface_equilibrium eta_bounds must be finite and non-collapsed.")
    return lower, upper


def _coerce_initial_eta_bracket(eta_bracket, eta_bounds):
    lower, upper = eta_bounds
    if eta_bracket is None:
        return (lower, upper)
    values = np.asarray(eta_bracket, dtype=np.float64).reshape(-1)
    if values.size != 2:
        raise ValueError("initial eta bracket must contain exactly two values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("initial eta bracket must be finite.")
    bracket_lower, bracket_upper = tuple(float(v) for v in values)
    if bracket_upper <= bracket_lower:
        raise ValueError("initial eta bracket must be strictly increasing.")
    tol = 1e-12 * max(1.0, abs(lower), abs(upper))
    if bracket_lower < lower - tol or bracket_upper > upper + tol:
        raise ValueError("initial eta bracket must lie within interface_equilibrium eta_bounds.")
    return (float(np.clip(bracket_lower, lower, upper)), float(np.clip(bracket_upper, lower, upper)))


def _temperature_at_interface(temperature, interface_position):
    temperature_parameters = TemperatureParameters(temperature)
    values = np.asarray(
        temperature_parameters(np.asarray([float(interface_position)], dtype=np.float64), 0.0),
        dtype=np.float64,
    ).reshape(-1)
    if values.size == 0 or not np.isfinite(values[0]):
        raise ValueError("temperature must evaluate to a finite value at the initial interface.")
    return float(values[0])


def _get_stefan_interdiffusivity(thermodynamics, composition, temperature, phase):
    try:
        D = thermodynamics.getInterdiffusivity(composition, temperature, phase=phase, query_context="interface")
    except TypeError:
        D = thermodynamics.getInterdiffusivity(composition, temperature, phase=phase)
    return _validate_ternary_diffusivity_matrix(D, phase, context="initial-eta diffusivity")


def _get_bulk_interdiffusivity(thermodynamics, composition, temperature, phase, context="bulk diffusivity"):
    """Returns a validated general/bulk ternary interdiffusivity matrix."""
    try:
        D = thermodynamics.getInterdiffusivity(composition, temperature, phase=phase, query_context="general")
    except TypeError:
        D = thermodynamics.getInterdiffusivity(composition, temperature, phase=phase)
    return _validate_ternary_diffusivity_matrix(D, phase, context=context)


def _coerce_bulk_diffusivity_mode(mode):
    """Normalizes and validates the ternary Illingworth bulk diffusivity mode."""
    value = str(mode)
    if value not in _SUPPORTED_BULK_DIFFUSIVITY_MODES:
        raise ValueError(
            "bulk_diffusivity_mode must be 'phase_uniform', "
            "'composition_dependent_lagged', or 'composition_dependent_implicit'."
        )
    return value


def estimate_initial_eta_from_instantaneous_balance(
    composition,
    z,
    interface_position,
    phases,
    thermodynamics,
    temperature,
    interface_equilibrium,
    transformed_u_grid,
    transformed_v_grid,
    eta_bracket=None,
    eta_guess=None,
    velocity_guess=None,
    root_xtol=1e-12,
    root_maxiter=100,
    bulk_diffusivity_mode=_BULK_DIFFUSIVITY_PHASE_UNIFORM,
):
    """
    Estimates initial eta by solving the instantaneous discrete balance.

    This method keeps the initial geometry and non-interface transformed
    concentrations fixed at ``s0`` and solves for ``(V0, eta0)`` in
    ``V0 * L(eta) - (G_B(eta; s0) - G_A(eta; s0)) = 0``. The branch-specific
    swept-inventory coefficient ``L`` matches the positive- and negative-motion
    branches used by the planar ternary residual in the finite-step solver.
    """
    composition = np.asarray(composition, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    u_grid = np.asarray(transformed_u_grid, dtype=np.float64).reshape(-1)
    v_grid = np.asarray(transformed_v_grid, dtype=np.float64).reshape(-1)
    if composition.ndim != 2 or composition.shape[1] != 2 or composition.shape[0] != z.size:
        raise ValueError("composition must have shape (n_nodes, 2) matching z.")
    if z.size < 3 or not np.all(np.diff(z) > 0.0):
        raise ValueError("z must be a strictly increasing 1D grid with at least three nodes.")
    if not np.isclose(z[0], 0.0):
        raise ValueError("estimate_initial_eta_from_instantaneous_balance expects a 1D domain starting at 0.")
    if u_grid.size < 3 or v_grid.size < 3:
        raise ValueError("transformed grids must each contain at least three nodes.")
    if not np.isclose(u_grid[0], 0.0) or not np.isclose(u_grid[-1], 1.0) or not np.all(np.diff(u_grid) > 0.0):
        raise ValueError("transformed_u_grid must be strictly increasing from 0 to 1.")
    if not np.isclose(v_grid[0], 0.0) or not np.isclose(v_grid[-1], 1.0) or not np.all(np.diff(v_grid) > 0.0):
        raise ValueError("transformed_v_grid must be strictly increasing from 0 to 1.")
    if len(phases) != 2:
        raise ValueError("phases must contain exactly the left and right phases.")
    if thermodynamics is None or not hasattr(thermodynamics, "getInterdiffusivity"):
        raise TypeError("thermodynamics must provide getInterdiffusivity for initial eta estimation.")
    bulk_diffusivity_mode = _coerce_bulk_diffusivity_mode(bulk_diffusivity_mode)

    s = float(interface_position)
    domain_length = float(z[-1] - z[0])
    if not (0.0 < s < domain_length):
        raise ValueError("interface_position must lie strictly inside the domain.")
    bracket = _coerce_initial_eta_bracket(eta_bracket, _validate_eta_bounds(interface_equilibrium))
    if eta_guess is None:
        eta0 = 0.5 * (bracket[0] + bracket[1])
    else:
        eta0 = float(eta_guess)
        if eta0 < bracket[0] or eta0 > bracket[1]:
            raise ValueError("eta_guess must lie within the initial eta bracket.")
    temperature_value = _temperature_at_interface(temperature, s)

    left_mask = z <= s
    right_mask = z >= s
    if not np.any(left_mask) or not np.any(right_mask):
        raise ValueError("Initial interface leaves an empty phase.")
    u_adjacent = float(u_grid[-2])
    v_adjacent = float(v_grid[1])
    z_left_adjacent = s * u_adjacent
    z_right_adjacent = s + (domain_length - s) * v_adjacent
    p_adjacent = np.asarray(
        [np.interp(z_left_adjacent, z[left_mask], composition[left_mask, component]) for component in range(2)],
        dtype=np.float64,
    )
    q_adjacent = np.asarray(
        [np.interp(z_right_adjacent, z[right_mask], composition[right_mask, component]) for component in range(2)],
        dtype=np.float64,
    )

    def evaluate_terms(eta, branch):
        c_left, c_right = interface_equilibrium.interface_compositions(float(eta))
        c_left = np.asarray(c_left, dtype=np.float64).reshape(2)
        c_right = np.asarray(c_right, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(c_left)) or not np.all(np.isfinite(c_right)):
            raise ValueError("interface compositions are non-finite at the queried eta.")
        if bulk_diffusivity_mode == _BULK_DIFFUSIVITY_PHASE_UNIFORM:
            D_left = _get_stefan_interdiffusivity(thermodynamics, c_left, temperature_value, phases[0])
            D_right = _get_stefan_interdiffusivity(thermodynamics, c_right, temperature_value, phases[1])
        else:
            D_left = _get_bulk_interdiffusivity(
                thermodynamics,
                0.5 * (p_adjacent + c_left),
                temperature_value,
                phases[0],
                context="initial-eta left interface-face diffusivity",
            )
            D_right = _get_bulk_interdiffusivity(
                thermodynamics,
                0.5 * (c_right + q_adjacent),
                temperature_value,
                phases[1],
                context="initial-eta right interface-face diffusivity",
            )
        G_left = _matvec_2x2(D_left, (c_left - p_adjacent) / (s * (1.0 - u_adjacent)))
        G_right = _matvec_2x2(D_right, (q_adjacent - c_right) / ((domain_length - s) * v_adjacent))
        flux_delta = G_right - G_left
        if branch == "positive":
            swept_inventory = c_left - q_adjacent * (1.0 - v_adjacent / 2.0) - c_right * v_adjacent / 2.0
        elif branch == "negative":
            swept_inventory = p_adjacent * ((1.0 + u_adjacent) / 2.0) + c_left * ((1.0 - u_adjacent) / 2.0) - c_right
        else:
            raise ValueError("branch must be 'positive' or 'negative'.")
        if not np.all(np.isfinite(flux_delta)) or not np.all(np.isfinite(swept_inventory)):
            raise ValueError("instantaneous balance terms are non-finite at the queried eta.")
        return swept_inventory, flux_delta, c_left.copy(), c_right.copy()

    def velocity_scale_for_branch(branch):
        swept_inventory, flux_delta, _, _ = evaluate_terms(eta0, branch)
        scale = float(np.linalg.norm(flux_delta) / max(float(np.linalg.norm(swept_inventory)), 1e-300))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return scale

    branch_results = []
    for branch in ("positive", "negative"):
        velocity_scale = velocity_scale_for_branch(branch)
        if velocity_guess is None:
            swept_inventory, flux_delta, _, _ = evaluate_terms(eta0, branch)
            scaled_velocity0 = float(np.dot(swept_inventory, flux_delta) / max(float(np.dot(swept_inventory, swept_inventory)), 1e-300))
            scaled_velocity0 /= velocity_scale
            if branch == "positive":
                scaled_velocity0 = abs(scaled_velocity0)
            else:
                scaled_velocity0 = -abs(scaled_velocity0)
        else:
            scaled_velocity0 = float(velocity_guess) / velocity_scale
            if branch == "positive" and scaled_velocity0 < 0.0:
                scaled_velocity0 = abs(scaled_velocity0)
            elif branch == "negative" and scaled_velocity0 > 0.0:
                scaled_velocity0 = -abs(scaled_velocity0)
        velocity_bounds = (0.0, np.inf) if branch == "positive" else (-np.inf, 0.0)

        def residual_unknowns(unknowns):
            velocity = float(unknowns[0]) * velocity_scale
            eta = float(unknowns[1])
            swept_inventory, flux_delta, _, _ = evaluate_terms(eta, branch)
            return velocity * swept_inventory - flux_delta

        lower = np.asarray([velocity_bounds[0], bracket[0]], dtype=np.float64)
        upper = np.asarray([velocity_bounds[1], bracket[1]], dtype=np.float64)
        x = np.clip(np.asarray([scaled_velocity0, eta0], dtype=np.float64), lower, upper)
        best = None
        success = False
        nfev = 0
        for _ in range(int(root_maxiter)):
            residual_current = residual_unknowns(x)
            nfev += 1
            norm_current = float(np.max(np.abs(residual_current)))
            if best is None or norm_current < best[0]:
                best = (norm_current, x.copy(), residual_current.copy())
            if norm_current <= float(root_xtol):
                success = True
                break

            jacobian = np.zeros((2, 2), dtype=np.float64)
            for variable in range(2):
                step, x_perturbed, difference_direction = _bounded_finite_difference_perturbation(x, lower, upper, variable)
                if difference_direction == "forward":
                    residual_perturbed = residual_unknowns(x_perturbed)
                    jacobian[:, variable] = (residual_perturbed - residual_current) / step
                else:
                    residual_perturbed = residual_unknowns(x_perturbed)
                    jacobian[:, variable] = (residual_current - residual_perturbed) / step
                nfev += 1

            try:
                step = _newton_step_2x2(jacobian, residual_current)
            except RuntimeError:
                break

            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = np.clip(x + scale * step, lower, upper)
                residual_trial = residual_unknowns(trial)
                nfev += 1
                norm_trial = float(np.max(np.abs(residual_trial)))
                if np.isfinite(norm_trial) and norm_trial < norm_current:
                    x = trial
                    accepted = True
                    break
            if not accepted:
                break
        if best is None:
            continue
        velocity = float(best[1][0]) * velocity_scale
        eta = float(best[1][1])
        swept_inventory, flux_delta, c_left, c_right = evaluate_terms(eta, branch)
        residual = velocity * swept_inventory - flux_delta
        branch_results.append(
            (
                float(np.max(np.abs(residual))),
                branch,
                success,
                nfev,
                velocity,
                eta,
                residual.copy(),
                flux_delta.copy(),
                c_left.copy(),
                c_right.copy(),
            )
        )

    branch_results = [record for record in branch_results if np.all(np.isfinite(record[5]))]
    if len(branch_results) == 0:
        raise ValueError("Instantaneous initial eta solve did not produce a finite residual.")
    best = min(branch_results, key=lambda record: record[0])
    residual_norm, branch, success, nfev, velocity, eta, residual, flux_delta, c_left, c_right = best
    if not success:
        raise ValueError("Instantaneous initial eta solve failed to converge.")
    return InitialEtaEstimate(
        eta=eta,
        residual_norm=residual_norm,
        velocity=velocity,
        residual=residual,
        flux_delta=flux_delta,
        left_interface_composition=c_left,
        right_interface_composition=c_right,
        method="instantaneous_balance",
        solver="damped_newton_2x2",
        bracket=bracket,
        converged=bool(success),
        iterations=int(nfev),
        function_calls=int(nfev),
        branch=branch,
    )


class _ScalarHistory:
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

    def record(self, time, y, force: bool = False):
        if self.recordInterval > 0:
            if self.currentIndex % self.recordInterval == 0 or force:
                self.N = int(self.currentIndex / self.recordInterval)
                if self.N >= self._time.shape[0]:
                    self._y = np.pad(self._y, (0, self.batchSize))
                    self._time = np.pad(self._time, (0, self.batchSize))
                self._y[self.N] = y
                self._time[self.N] = time
            self.currentIndex += 1
        else:
            self._y[self.N] = y
            self._time[self.N] = time
        self.currentY = float(y)
        self.currentTime = float(time)

    def finalize(self):
        if self.recordInterval > 0 and np.isclose(self._time[self.N], self.currentTime, rtol=0.0, atol=1e-14):
            self._y = self._y[: self.N + 1]
            self._time = self._time[: self.N + 1]
            return
        self.record(self.currentTime, self.currentY, force=True)
        self._y = self._y[: self.N + 1]
        self._time = self._time[: self.N + 1]

    def y(self, time=None):
        if time is None:
            return float(self._y[self.N])
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


class _ArrayHistory:
    def __init__(self, shape, record: bool | int = False):
        if isinstance(record, bool):
            self.recordInterval = 1 if record else -1
        else:
            self.recordInterval = int(record)
        self.shape = tuple(int(v) for v in shape)
        self.batchSize = 1000
        self.reset()

    def reset(self):
        self._y = np.zeros((self.batchSize, *self.shape), dtype=np.float64)
        self._time = np.zeros(self.batchSize, dtype=np.float64)
        self.currentIndex = 0
        self.currentY = np.zeros(self.shape, dtype=np.float64)
        self.currentTime = 0.0
        self.N = 0

    def record(self, time, y, force: bool = False):
        values = np.asarray(y, dtype=np.float64)
        if values.shape != self.shape:
            raise ValueError(f"Expected history value with shape {self.shape}, got {values.shape}.")
        if self.recordInterval > 0:
            if self.currentIndex % self.recordInterval == 0 or force:
                self.N = int(self.currentIndex / self.recordInterval)
                if self.N >= self._time.shape[0]:
                    self._y = np.pad(self._y, ((0, self.batchSize), *[(0, 0) for _ in self.shape]))
                    self._time = np.pad(self._time, (0, self.batchSize))
                self._y[self.N] = values
                self._time[self.N] = time
            self.currentIndex += 1
        else:
            self._y[self.N] = values
            self._time[self.N] = time
        self.currentY = values.copy()
        self.currentTime = float(time)

    def finalize(self):
        if self.recordInterval > 0 and np.isclose(self._time[self.N], self.currentTime, rtol=0.0, atol=1e-14):
            self._y = self._y[: self.N + 1]
            self._time = self._time[: self.N + 1]
            return
        self.record(self.currentTime, self.currentY, force=True)
        self._y = self._y[: self.N + 1]
        self._time = self._time[: self.N + 1]

    def y(self, time=None):
        if time is None:
            return self._y[self.N].copy()
        recorded_time = self._time[: self.N + 1]
        matches = np.where(np.isclose(recorded_time, float(time), atol=1e-14, rtol=0.0))[0]
        if len(matches) == 0:
            raise ValueError(f"Requested exact history time {float(time):.6g} was not recorded.")
        return self._y[matches[-1]].copy()


class MovingBoundaryIllingworthTernaryFD1DModel(DiffusionModel):
    """
    Ternary planar moving-boundary model using an Illingworth front-fixing form.

    The model is additive relative to the binary Illingworth implementation. It
    stores two independent substitutional components per transformed grid node,
    solves full 2-by-2 block-tridiagonal phase systems, and advances the
    interface through a conservative two-component residual. The initial
    tie-line coordinate is estimated from the instantaneous discrete balance
    that matches the finite-step interface residual, so callers must provide an
    eta-capable interface equilibrium. Only planar Cartesian finite-difference
    meshes are supported.

    By default, ``bulk_diffusivity_mode='phase_uniform'`` preserves the legacy
    behavior: one 2-by-2 interdiffusivity matrix is evaluated for each phase at
    that phase's interface composition and reused throughout the transformed
    bulk solve. ``bulk_diffusivity_mode='composition_dependent_lagged'`` instead
    evaluates one matrix per finite-volume face from the accepted old-time
    profile and freezes those matrices during each candidate linear solve. The
    ``'composition_dependent_implicit'`` mode evaluates those face matrices from
    a Picard iterate inside each candidate phase solve. The Picard iterate is
    initialized deterministically from the accepted old-time profile with the
    trial interface boundary imposed. The interface residual consumes the exact
    interface-adjacent diffusive flux returned by each bulk solve. When exactly
    one phase width is below ``terminal_thin_phase_width`` after the normal
    retry budget is exhausted, an optional terminal retry path can keep
    reducing ``dt`` until one final converged step is found and then stop the
    solve early.
    """

    def __init__(
        self,
        mesh,
        elements,
        phases,
        thermodynamics,
        temperature,
        interfacePosition,
        time_step: float,
        interface_equilibrium=None,
        interface_compositions=None,
        eta_bounds: tuple[float, float] = (0.0, 1.0),
        initial_eta_method: str = "instantaneous_balance",
        initial_eta_bracket=None,
        initial_eta_guess: float | None = None,
        initial_velocity_guess: float | None = None,
        initial_eta_root_xtol: float = 1e-12,
        initial_eta_root_maxiter: int = 100,
        bulk_diffusivity_mode: str = _BULK_DIFFUSIVITY_PHASE_UNIFORM,
        bulk_picard_rtol: float | None = None,
        bulk_picard_atol: float = 1e-12,
        bulk_picard_max_iterations: int = 25,
        bulk_picard_relaxation: float = 1.0,
        dt_mode: str = "fixed",
        semiLog_dt: float | None = None,
        semiLogT0: float | None = None,
        geometry: str = "planar",
        phase_a_nodes: int | None = None,
        phase_b_nodes: int | None = None,
        tolerance: float = 1e-8,
        residual_tolerance: float | None = None,
        max_iterations: int = 25,
        max_step_retries: int = 8,
        retry_factor: float = 0.5,
        terminal_thin_phase_width: float | None = 1e-9,
        terminal_thin_phase_extra_retries: int = 20,
        terminal_thin_phase_policy: str = "prompt",
        constraints=None,
        record=False,
        record_pq_data: bool = True,
        transformed_u_grid=None,
        transformed_v_grid=None,
    ):
        self.initialInterfacePosition = float(interfacePosition)
        self.timeStep = float(time_step)
        self.initialEta = np.nan
        self.initialEtaMethod = str(initial_eta_method)
        self.initialEtaBracket = initial_eta_bracket
        self.initialEtaGuess = None if initial_eta_guess is None else float(initial_eta_guess)
        self.initialVelocityGuess = None if initial_velocity_guess is None else float(initial_velocity_guess)
        self.initialEtaRootXtol = float(initial_eta_root_xtol)
        self.initialEtaRootMaxiter = int(initial_eta_root_maxiter)
        self.bulkDiffusivityMode = _coerce_bulk_diffusivity_mode(bulk_diffusivity_mode)
        self.bulkPicardRtol = None if bulk_picard_rtol is None else float(bulk_picard_rtol)
        self.bulkPicardAtol = float(bulk_picard_atol)
        self.bulkPicardMaxIterations = int(bulk_picard_max_iterations)
        self.bulkPicardRelaxation = float(bulk_picard_relaxation)
        self.dtMode = str(dt_mode)
        self.semiLog_dt = None if semiLog_dt is None else float(semiLog_dt)
        self.semiLogT0 = None if semiLogT0 is None else float(semiLogT0)
        self.geometry = str(geometry)
        self.phaseANodes = None if phase_a_nodes is None else int(phase_a_nodes)
        self.phaseBNodes = None if phase_b_nodes is None else int(phase_b_nodes)
        self.tolerance = float(tolerance)
        self.residualTolerance = float(tolerance if residual_tolerance is None else residual_tolerance)
        self.maxIterations = int(max_iterations)
        self.maxStepRetries = int(max_step_retries)
        self.retryFactor = float(retry_factor)
        self.terminalThinPhaseWidth = None if terminal_thin_phase_width is None else float(terminal_thin_phase_width)
        self.terminalThinPhaseExtraRetries = int(terminal_thin_phase_extra_retries)
        self.terminalThinPhasePolicy = str(terminal_thin_phase_policy)
        self.recordPqData = bool(record_pq_data)
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

        self.interfaceEquilibrium = self._coerce_interface_equilibrium(
            interface_equilibrium,
            interface_compositions,
            eta_bounds,
        )
        self.interfaceData = _ScalarHistory(record)
        self.etaData = _ScalarHistory(record)
        self.inventoryData = _ArrayHistory((2,), record)
        self.pData = None
        self.qData = None

        self._currdt = np.inf
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
        self._nearFinalNoop = False
        self._reset_implicit_diagnostics()
        self._lastStepRetries = 0
        self._lastInterfaceCompositions = None
        self._terminalThinPhaseStop = False
        self._terminalThinPhaseInfo = None
        self.initialEtaEstimate = None
        self._initialInventory = None

        self._z = None
        self._R = None
        self._u_grid = None
        self._v_grid = None
        self._p_curr = None
        self._q_curr = None
        self._s_curr = None
        self._s_old = None
        self._eta_curr = None
        self._D_left = None
        self._D_right = None

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
        self.interfaceData.currentY = self.initialInterfacePosition
        self.interfaceData._y[0] = self.initialInterfacePosition
        self.etaData.currentY = self.initialEta
        self.etaData._y[0] = self.initialEta

    def _coerce_interface_equilibrium(self, closure, interface_compositions, eta_bounds):
        if closure is not None and interface_compositions is not None:
            raise ValueError("Specify either interface_equilibrium or interface_compositions, not both.")
        if interface_compositions is not None:
            raise ValueError("MovingBoundaryIllingworthTernaryFD1DModel now requires an eta-capable interface_equilibrium.")
        if closure is None:
            return ThermodynamicTernaryInterfaceEquilibrium(None, eta_bounds=eta_bounds)
        if hasattr(closure, "interface_compositions"):
            return closure
        if callable(closure):
            return CallableTernaryInterfaceEquilibrium(closure, eta_bounds=eta_bounds)
        raise TypeError("interface_equilibrium must be an eta-capable closure object or callable.")

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
            raise TypeError("MovingBoundaryIllingworthTernaryFD1DModel requires a CartesianFD1D mesh.")
        if len(self.allElements) != 3 or self.mesh.numResponses != 2:
            raise ValueError("MovingBoundaryIllingworthTernaryFD1DModel requires ternary systems with two independent responses.")
        self._validate_external_boundary_conditions()
        self._validate_isothermal_temperature()
        if any(e in interstitials for e in self.allElements):
            raise ValueError("MovingBoundaryIllingworthTernaryFD1DModel supports only substitutional systems.")
        if len(self.phases) != 2:
            raise ValueError("MovingBoundaryIllingworthTernaryFD1DModel requires exactly two explicit phases.")
        if self.geometry != "planar":
            raise NotImplementedError("MovingBoundaryIllingworthTernaryFD1DModel currently implements only planar geometry.")
        if not np.isfinite(self.timeStep) or self.timeStep <= 0:
            raise ValueError("time_step must be a positive finite value.")
        if self.dtMode not in {"fixed", "semi_log"}:
            raise ValueError("dt_mode must be 'fixed' or 'semi_log'.")
        self.bulkDiffusivityMode = _coerce_bulk_diffusivity_mode(self.bulkDiffusivityMode)
        if self.bulkPicardRtol is not None and (not np.isfinite(self.bulkPicardRtol) or self.bulkPicardRtol <= 0.0):
            raise ValueError("bulk_picard_rtol must be positive when specified.")
        if not np.isfinite(self.bulkPicardAtol) or self.bulkPicardAtol <= 0.0:
            raise ValueError("bulk_picard_atol must be a positive finite value.")
        if self.bulkPicardMaxIterations < 1:
            raise ValueError("bulk_picard_max_iterations must be at least 1.")
        if not np.isfinite(self.bulkPicardRelaxation) or not (0.0 < self.bulkPicardRelaxation <= 1.0):
            raise ValueError("bulk_picard_relaxation must be in the interval (0, 1].")
        if self.dtMode == "semi_log" and ((self.semiLog_dt is None) or (self.semiLogT0 is None)):
            raise ValueError("semiLog_dt and semiLogT0 must be specified when dt_mode is 'semi_log'.")
        _validate_eta_bounds(self.interfaceEquilibrium)
        if self.initialEtaMethod != "instantaneous_balance":
            raise ValueError("initial_eta_method must be 'instantaneous_balance'.")
        if self.initialEtaRootXtol <= 0.0:
            raise ValueError("initial eta root tolerance must be positive.")
        if self.initialEtaRootMaxiter < 1:
            raise ValueError("initial_eta_root_maxiter must be at least 1.")
        if self.maxIterations < 2:
            raise ValueError("max_iterations must be at least 2.")
        if self.maxStepRetries < 1:
            raise ValueError("max_step_retries must be at least 1.")
        if not (0.0 < self.retryFactor < 1.0):
            raise ValueError("retry_factor must be between 0 and 1.")
        if self.terminalThinPhaseWidth is not None and (not np.isfinite(self.terminalThinPhaseWidth) or self.terminalThinPhaseWidth <= 0.0):
            raise ValueError("terminal_thin_phase_width must be positive and finite when specified.")
        if self.terminalThinPhaseExtraRetries < 0:
            raise ValueError("terminal_thin_phase_extra_retries must be non-negative.")
        if self.terminalThinPhasePolicy not in {"prompt", "continue", "raise"}:
            raise ValueError("terminal_thin_phase_policy must be 'prompt', 'continue', or 'raise'.")
        if self.phaseANodes is not None and self.phaseANodes < 3:
            raise ValueError("phase_a_nodes must be at least 3 when specified.")
        if self.phaseBNodes is not None and self.phaseBNodes < 3:
            raise ValueError("phase_b_nodes must be at least 3 when specified.")
        self.initialInterfacePosition = self._clipInterfacePosition(self.initialInterfacePosition, strict=True)

    def _validate_external_boundary_conditions(self):
        """
        Validates that the fixed external boundaries are homogeneous zero flux.

        The ternary Illingworth discretization implemented here is conservative
        for a closed planar domain. Nonzero Neumann fluxes, fixed-composition
        boundaries, periodic wrapping, mixed/Robin-like objects, or
        time-dependent boundary objects require extra terms that are not part of
        this solver and are rejected before setup.
        """
        bc = getattr(self.mesh, "boundaryConditions", None)
        if bc is None:
            return
        if isinstance(bc, PeriodicBoundary1D):
            raise NotImplementedError("Ternary Illingworth currently supports only homogeneous zero-flux external boundaries; periodic boundaries are unsupported.")
        if not isinstance(bc, MixedBoundary1D):
            raise NotImplementedError("Ternary Illingworth currently supports only MixedBoundary1D homogeneous zero-flux external boundaries.")

        expected_shape = (self.mesh.numResponses,)
        for attr in ("LBCtype", "RBCtype", "LBCvalue", "RBCvalue"):
            values = np.asarray(getattr(bc, attr, None))
            if values.shape != expected_shape:
                raise NotImplementedError("Boundary-condition arrays must match the independent-component count for ternary Illingworth.")

        left_type = np.asarray(bc.LBCtype)
        right_type = np.asarray(bc.RBCtype)
        if not np.all(left_type == MixedBoundary1D.NEUMANN) or not np.all(right_type == MixedBoundary1D.NEUMANN):
            raise NotImplementedError("Ternary Illingworth currently supports only Neumann zero-flux external boundaries; fixed-composition or mixed boundaries are unsupported.")

        left_value = np.asarray(bc.LBCvalue, dtype=np.float64)
        right_value = np.asarray(bc.RBCvalue, dtype=np.float64)
        if not np.all(np.isfinite(left_value)) or not np.all(np.isfinite(right_value)):
            raise NotImplementedError("Ternary Illingworth boundary flux values must be finite zero constants.")
        if not np.all(left_value == 0.0) or not np.all(right_value == 0.0):
            raise NotImplementedError("Ternary Illingworth currently supports only homogeneous zero-flux external boundaries; nonzero fluxes are unsupported.")

    def _validate_isothermal_temperature(self):
        """
        Validates that the configured temperature is demonstrably constant.

        The current ternary interface-equilibrium and diffusivity path is
        isothermal. Scalar temperatures and temperature-array wrappers with
        identical finite values are accepted; callable or varying temperature
        specifications are rejected instead of being sampled once and treated as
        constant.
        """
        params = getattr(self.temperatureParameters, "Tparameters", None)
        if isinstance(params, tuple) and len(params) == 2:
            temperatures = np.asarray(params[1], dtype=np.float64).reshape(-1)
            if temperatures.size == 0 or not np.all(np.isfinite(temperatures)):
                raise ValueError("Temperature values must be finite.")
            if not np.all(temperatures == temperatures[0]):
                raise NotImplementedError("Ternary Illingworth currently supports only isothermal temperature; time-dependent temperature arrays are unsupported.")
            return
        if callable(params):
            raise NotImplementedError("Ternary Illingworth currently supports only isothermal temperature; callable temperature functions are unsupported.")

        values = np.asarray(params, dtype=np.float64).reshape(-1)
        if values.size != 1 or not np.isfinite(values[0]):
            raise ValueError("Ternary Illingworth requires a finite scalar isothermal temperature.")

    def _clipInterfacePosition(self, interface_position: float, strict: bool = True) -> float:
        z = flatten_1d_coordinates(self.mesh.z)
        eps = max(float(z[-1] - z[0]) * 1e-14, 1e-14)
        lower = float(z[0] + eps)
        upper = float(z[-1] - eps)
        if strict and not (lower < interface_position < upper):
            raise ValueError("Interface position must lie strictly inside the FD domain.")
        return float(np.clip(interface_position, lower, upper))

    def _phase_widths(self, interface_position):
        """Returns the physical widths of the left and right phase regions."""
        s = float(interface_position)
        return np.asarray([s, float(self._R) - s], dtype=np.float64)

    def _single_terminal_thin_phase(self, interface_position):
        """Returns ``(index, width)`` only when exactly one phase is below the terminal threshold."""
        if self.terminalThinPhaseWidth is None:
            return None
        widths = self._phase_widths(interface_position)
        mask = np.isfinite(widths) & (widths < self.terminalThinPhaseWidth)
        if int(np.count_nonzero(mask)) != 1:
            return None
        index = int(np.flatnonzero(mask)[0])
        return index, float(widths[index])

    def _getBoundaryConditions(self):
        bc = getattr(self.mesh, "boundaryConditions", None)
        if bc is None:
            bc = MixedBoundary1D(self.mesh.responses)
            self.mesh.boundaryConditions = bc
        return bc

    def _reset_implicit_diagnostics(self):
        """
        Resets diagnostics for the most recent nonlinear interface solve.

        ``_lastImplicitCandidateEvaluations`` counts attempted complete
        interface-candidate builds. ``_lastImplicitFunctionEvaluations`` is
        retained as a private compatibility alias with the same value.
        """
        self._lastImplicitIterations = 0
        self._lastImplicitResidual = np.nan
        self._lastImplicitPhysicalResidual = np.nan
        self._lastImplicitCandidateEvaluations = 0
        self._lastImplicitFunctionEvaluations = 0
        self._lastImplicitJacobianEvaluations = 0
        self._lastImplicitMotionBranch = None
        self._lastImplicitConverged = False
        self._lastImplicitFailureReason = None
        self._lastBulkLeftPicardIterations = 0
        self._lastBulkRightPicardIterations = 0
        self._lastBulkLeftUpdateNorm = np.nan
        self._lastBulkRightUpdateNorm = np.nan
        self._lastBulkDiffusivityEvaluations = 0
        self._lastBulkDiffusivityProviderCalls = 0
        self._lastBulkFaceMatricesEvaluated = 0
        self._lastBulkConverged = True
        self._lastBulkFailureReason = None
        self._currentBulkDiffusivityProviderCalls = 0
        self._currentBulkFaceMatricesEvaluated = 0
        self._bulkDiffusivityCountingActive = False

    def _record_implicit_success(self, iterations, candidate, candidate_evaluations, jacobian_evaluations):
        """
        Records diagnostics for a converged nonlinear interface solve.

        Candidate evaluations count attempted full interface-candidate builds.
        One Jacobian evaluation means one complete finite-difference Jacobian
        construction, not one perturbed candidate evaluation.
        """
        self._lastImplicitIterations = int(iterations)
        self._lastImplicitResidual = candidate.scaled_norm
        self._lastImplicitPhysicalResidual = candidate.physical_norm
        self._lastImplicitCandidateEvaluations = int(candidate_evaluations)
        self._lastImplicitFunctionEvaluations = int(candidate_evaluations)
        self._lastImplicitJacobianEvaluations = int(jacobian_evaluations)
        self._lastImplicitMotionBranch = candidate.motion_branch
        self._lastImplicitConverged = True
        self._lastImplicitFailureReason = None
        self._lastBulkLeftPicardIterations = int(candidate.left_inner_iterations)
        self._lastBulkRightPicardIterations = int(candidate.right_inner_iterations)
        self._lastBulkLeftUpdateNorm = float(candidate.left_inner_update_norm)
        self._lastBulkRightUpdateNorm = float(candidate.right_inner_update_norm)
        self._lastBulkDiffusivityProviderCalls = int(self._currentBulkDiffusivityProviderCalls)
        self._lastBulkFaceMatricesEvaluated = int(self._currentBulkFaceMatricesEvaluated)
        self._lastBulkDiffusivityEvaluations = self._lastBulkFaceMatricesEvaluated
        self._lastBulkConverged = True
        self._lastBulkFailureReason = None
        self._bulkDiffusivityCountingActive = False

    def _record_implicit_failure(self, iterations, best_scaled_norm, best_physical_norm, best_motion_branch, candidate_evaluations, jacobian_evaluations, reason):
        """Records diagnostics for a failed nonlinear interface solve."""
        self._lastImplicitIterations = int(iterations)
        self._lastImplicitResidual = float(best_scaled_norm)
        self._lastImplicitPhysicalResidual = float(best_physical_norm)
        self._lastImplicitCandidateEvaluations = int(candidate_evaluations)
        self._lastImplicitFunctionEvaluations = int(candidate_evaluations)
        self._lastImplicitJacobianEvaluations = int(jacobian_evaluations)
        self._lastImplicitMotionBranch = best_motion_branch
        self._lastImplicitConverged = False
        self._lastImplicitFailureReason = reason
        self._lastBulkDiffusivityProviderCalls = int(self._currentBulkDiffusivityProviderCalls)
        self._lastBulkFaceMatricesEvaluated = int(self._currentBulkFaceMatricesEvaluated)
        self._lastBulkDiffusivityEvaluations = self._lastBulkFaceMatricesEvaluated
        self._bulkDiffusivityCountingActive = False

    def _record_completed_bulk_candidate(self, candidate):
        """Records bulk diagnostics from the latest fully evaluated interface candidate."""
        self._lastBulkLeftPicardIterations = int(candidate.left_inner_iterations)
        self._lastBulkRightPicardIterations = int(candidate.right_inner_iterations)
        self._lastBulkLeftUpdateNorm = float(candidate.left_inner_update_norm)
        self._lastBulkRightUpdateNorm = float(candidate.right_inner_update_norm)
        self._lastBulkConverged = True
        self._lastBulkFailureReason = None

    def _reset_candidate_bulk_diagnostics(self):
        """Initializes phase-bulk diagnostics for one interface-candidate attempt."""
        self._lastBulkLeftPicardIterations = 0
        self._lastBulkRightPicardIterations = 0
        self._lastBulkLeftUpdateNorm = np.inf
        self._lastBulkRightUpdateNorm = np.inf
        self._lastBulkConverged = None
        self._lastBulkFailureReason = "not evaluated"

    def _record_bulk_phase_success(self, phase_label, result):
        """Records one successfully completed phase solve for the current candidate."""
        if phase_label == "left":
            self._lastBulkLeftPicardIterations = int(result.inner_iterations)
            self._lastBulkLeftUpdateNorm = float(result.inner_update_norm)
        elif phase_label == "right":
            self._lastBulkRightPicardIterations = int(result.inner_iterations)
            self._lastBulkRightUpdateNorm = float(result.inner_update_norm)
        if self._lastBulkConverged is not False:
            self._lastBulkConverged = True
            self._lastBulkFailureReason = None

    def _record_bulk_phase_failure(self, phase_label, iterations, update_norm, reason):
        """Records diagnostics for a genuine phase-bulk failure."""
        if phase_label == "left":
            self._lastBulkLeftPicardIterations = int(iterations)
            self._lastBulkLeftUpdateNorm = float(update_norm)
        elif phase_label == "right":
            self._lastBulkRightPicardIterations = int(iterations)
            self._lastBulkRightUpdateNorm = float(update_norm)
        self._lastBulkConverged = False
        self._lastBulkFailureReason = str(reason)

    def _record_bulk_diffusivity_provider_call(self):
        """Counts one attempted bulk/general diffusivity provider call."""
        if getattr(self, "_bulkDiffusivityCountingActive", False):
            self._currentBulkDiffusivityProviderCalls += 1

    def _record_bulk_face_matrices_evaluated(self, count):
        """Counts validated face matrices returned by bulk diffusivity queries."""
        if getattr(self, "_bulkDiffusivityCountingActive", False):
            self._currentBulkFaceMatricesEvaluated += int(count)

    def reset(self):
        super().reset()
        self.interfaceData.reset()
        self.interfaceData.record(0, self.initialInterfacePosition)
        self.etaData.reset()
        self.etaData.record(0, self.initialEta)
        self.inventoryData.reset()
        self.pData = None
        self.qData = None
        self._currdt = np.inf
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
        self._nearFinalNoop = False
        self._reset_implicit_diagnostics()
        self._lastStepRetries = 0
        self._lastInterfaceCompositions = None
        self._terminalThinPhaseStop = False
        self._terminalThinPhaseInfo = None
        self.initialEtaEstimate = None
        self._initialInventory = None
        self._z = None
        self._R = None
        self._u_grid = None
        self._v_grid = None
        self._p_curr = None
        self._q_curr = None
        self._s_curr = None
        self._s_old = None
        self._eta_curr = None
        self._D_left = None
        self._D_right = None
        if hasattr(self, "mesh") and self.mesh is not None:
            self._validateModelConfiguration()

    def setup(self):
        super().setup()
        self._validateModelConfiguration()
        self._getBoundaryConditions()
        self._z = flatten_1d_coordinates(self.mesh.z).astype(np.float64)
        if not np.isclose(self._z[0], 0.0):
            raise ValueError("MovingBoundaryIllingworthTernaryFD1DModel expects a 1D domain starting at 0.")
        self._R = float(self._z[-1] - self._z[0])

        c0 = np.asarray(self.data.currentY, dtype=np.float64)
        s0 = float(self.interfaceData.currentY)
        n_left = self.phaseANodes if self.phaseANodes is not None else max(3, int(np.searchsorted(self._z, s0, side="right")))
        n_right = self.phaseBNodes if self.phaseBNodes is not None else max(3, len(self._z) - int(np.searchsorted(self._z, s0, side="left")))
        self._u_grid = self._inputUGrid.copy() if self._inputUGrid is not None else np.linspace(0.0, 1.0, int(n_left), dtype=np.float64)
        self._v_grid = self._inputVGrid.copy() if self._inputVGrid is not None else np.linspace(0.0, 1.0, int(n_right), dtype=np.float64)

        if self.recordPqData:
            self.pData = _ArrayHistory((len(self._u_grid), 2), self.interfaceData.recordInterval)
            self.qData = _ArrayHistory((len(self._v_grid), 2), self.interfaceData.recordInterval)
        if isinstance(self.interfaceEquilibrium, ThermodynamicTernaryInterfaceEquilibrium) and self.interfaceEquilibrium.thermodynamics is None:
            self.interfaceEquilibrium.thermodynamics = self.therm
            self.interfaceEquilibrium.phases = self.phases
        estimate_kwargs = {
            "composition": c0,
            "z": self._z,
            "interface_position": s0,
            "phases": self.phases,
            "thermodynamics": self.therm,
            "temperature": self.temperatureParameters,
            "interface_equilibrium": self.interfaceEquilibrium,
            "transformed_u_grid": self._u_grid,
            "transformed_v_grid": self._v_grid,
            "eta_bracket": self.initialEtaBracket,
            "root_xtol": self.initialEtaRootXtol,
            "root_maxiter": self.initialEtaRootMaxiter,
            "bulk_diffusivity_mode": self.bulkDiffusivityMode,
        }
        self.initialEtaEstimate = estimate_initial_eta_from_instantaneous_balance(
            **estimate_kwargs,
            eta_guess=self.initialEtaGuess,
            velocity_guess=self.initialVelocityGuess,
        )
        print(f"initialEtaEstimate: {self.initialEtaEstimate.eta}")
        eta0 = float(self.initialEtaEstimate.eta)
        self.initialEta = eta0
        self.etaData.reset()
        self.etaData.record(0, eta0)
        self._p_curr, self._q_curr = self._initialize_transformed_state(c0, s0, eta0)
        self._s_curr = s0
        self._s_old = s0
        self._eta_curr = eta0
        c_left, c_right = self._interface_compositions(eta0)
        self._lastInterfaceCompositions = (c_left.copy(), c_right.copy())
        if self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_PHASE_UNIFORM:
            self._D_left = self._phase_diffusivity_matrix(c_left, self.phases[0], 0.0, s0)
            self._D_right = self._phase_diffusivity_matrix(c_right, self.phases[1], 0.0, s0)
        elif self.bulkDiffusivityMode in {_BULK_DIFFUSIVITY_LAGGED, _BULK_DIFFUSIVITY_IMPLICIT}:
            self._D_left = self._left_lagged_face_diffusivity_matrices(self._p_curr, s0, 0.0)[-1]
            self._D_right = self._right_lagged_face_diffusivity_matrices(self._q_curr, s0, 0.0)[0]
        else:
            raise ValueError(
                "bulkDiffusivityMode must be 'phase_uniform', "
                "'composition_dependent_lagged', or 'composition_dependent_implicit'."
            )

        if self.recordPqData:
            self.pData.record(0, self._p_curr)
            self.qData.record(0, self._q_curr)

        physical = self._reconstruct_physical_profile(self._p_curr, self._q_curr, s0)
        self.data.currentY = physical
        self.data._y[0] = physical
        self._initialInventory = self.getTotalInventoryFromState(self._p_curr, self._q_curr, self._s_curr)
        self.inventoryData.record(0, self._initialInventory)

    def _interface_compositions(self, eta):
        if isinstance(self.interfaceEquilibrium, ThermodynamicTernaryInterfaceEquilibrium) and self.interfaceEquilibrium.thermodynamics is None:
            self.interfaceEquilibrium.thermodynamics = self.therm
            self.interfaceEquilibrium.phases = self.phases
        c_left, c_right = self.interfaceEquilibrium.interface_compositions(float(eta))
        self._validate_composition_vector(c_left, "left interface composition")
        self._validate_composition_vector(c_right, "right interface composition")
        return c_left, c_right

    def _validate_composition_vector(self, values, name):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        min_comp = float(self.constraints.minComposition)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite two-component vector.")
        dependent = 1.0 - float(np.sum(values))
        if np.any(values < min_comp) or dependent < min_comp:
            raise ValueError(f"{name} violates the ternary composition bounds.")

    def _initialize_transformed_state(self, composition, interface_position, eta):
        c = np.asarray(composition, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError("Initial ternary composition must have shape (n_nodes, 2).")
        c_left_int, c_right_int = self._interface_compositions(eta)

        z_left = float(interface_position) * self._u_grid
        z_right = float(interface_position) + (self._R - float(interface_position)) * self._v_grid
        left_mask = self._z <= float(interface_position)
        right_mask = self._z >= float(interface_position)
        if not np.any(left_mask) or not np.any(right_mask):
            raise ValueError("Initial interface leaves an empty phase.")

        p = np.empty((len(self._u_grid), 2), dtype=np.float64)
        q = np.empty((len(self._v_grid), 2), dtype=np.float64)
        for component in range(2):
            p[:, component] = np.interp(z_left, self._z[left_mask], c[left_mask, component])
            q[:, component] = np.interp(z_right, self._z[right_mask], c[right_mask, component])
        p[-1] = c_left_int
        q[0] = c_right_int
        return p, q

    def _phase_diffusivity_matrix(self, composition, phase, time, position):
        """
        Returns the phase-uniform 2x2 matrix used by one transformed bulk solve.

        ``composition`` is expected to be the interface composition for
        ``phase``. The resulting matrix may depend on that interface state,
        temperature, and time, but it is not re-evaluated at bulk nodes or
        faces inside the implicit phase solve.
        """
        composition = np.asarray(composition, dtype=np.float64).reshape(2)
        T = np.asarray(self.temperatureParameters(np.asarray([[float(position)]], dtype=np.float64), float(time)), dtype=np.float64).reshape(-1)
        temperature = float(T[0])
        try:
            self._record_bulk_diffusivity_provider_call()
            D = self.therm.getInterdiffusivity(composition, temperature, phase=phase, query_context="interface")
        except TypeError:
            self._record_bulk_diffusivity_provider_call()
            D = self.therm.getInterdiffusivity(composition, temperature, phase=phase)
        return _validate_ternary_diffusivity_matrix(D, phase, context="transient diffusivity")

    def _temperatures_at_positions(self, positions, time):
        """Returns finite temperatures at physical face positions."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1)
        T = np.asarray(
            self.temperatureParameters(positions.reshape(-1, 1), float(time)),
            dtype=np.float64,
        ).reshape(-1)
        if T.size == 1 and positions.size != 1:
            T = np.full(positions.shape, float(T[0]), dtype=np.float64)
        if T.size != positions.size or not np.all(np.isfinite(T)):
            raise ValueError("Bulk face temperatures must be finite and match the face count.")
        return T

    def _bulk_face_diffusivity_matrices(self, face_compositions, phase, time, physical_face_positions):
        """
        Returns validated general/bulk diffusivity matrices at face compositions.

        Vectorized thermodynamic queries are used when supported. Providers that
        only accept one composition at a time fall back to a deterministic loop.
        """
        face_compositions = np.asarray(face_compositions, dtype=np.float64)
        if face_compositions.ndim != 2 or face_compositions.shape[1] != 2:
            raise ValueError("face_compositions must have shape (n_faces, 2).")
        temperatures = self._temperatures_at_positions(physical_face_positions, time)
        if temperatures.size != face_compositions.shape[0]:
            raise ValueError("physical_face_positions must match the face composition count.")

        def validate_stack(values):
            matrices = np.asarray(values)
            if matrices.shape == (2, 2) and face_compositions.shape[0] == 1:
                matrices = matrices.reshape(1, 2, 2)
            if matrices.shape != (face_compositions.shape[0], 2, 2):
                raise ValueError("bulk diffusivity query returned an unexpected matrix shape.")
            out = np.empty(matrices.shape, dtype=np.float64)
            for i, matrix in enumerate(matrices):
                out[i] = _validate_ternary_diffusivity_matrix(matrix, phase, context=f"bulk face diffusivity face {i}")
            self._record_bulk_face_matrices_evaluated(out.shape[0])
            return out

        try:
            self._record_bulk_diffusivity_provider_call()
            values = self.therm.getInterdiffusivity(face_compositions, temperatures, phase=phase, query_context="general")
        except TypeError:
            pass
        except ValueError:
            pass
        else:
            return validate_stack(values)

        out = np.empty((face_compositions.shape[0], 2, 2), dtype=np.float64)
        for i, (composition, temperature) in enumerate(zip(face_compositions, temperatures)):
            try:
                self._record_bulk_diffusivity_provider_call()
                D = self.therm.getInterdiffusivity(composition, float(temperature), phase=phase, query_context="general")
            except TypeError:
                self._record_bulk_diffusivity_provider_call()
                D = self.therm.getInterdiffusivity(composition, float(temperature), phase=phase)
            out[i] = _validate_ternary_diffusivity_matrix(D, phase, context=f"bulk face diffusivity face {i}")
        self._record_bulk_face_matrices_evaluated(out.shape[0])
        return out

    def _left_lagged_face_compositions(self, p):
        """
        Builds fully lagged left-phase face compositions.

        Face ``j`` is between old accepted nodes ``p[j]`` and ``p[j + 1]``; the
        interface-adjacent face therefore uses ``0.5 * (p[-2] + p[-1])`` where
        ``p[-1]`` is the old accepted left interface composition.
        """
        p = np.asarray(p, dtype=np.float64)
        return 0.5 * (p[:-1] + p[1:])

    def _right_lagged_face_compositions(self, q):
        """
        Builds fully lagged right-phase face compositions.

        Face ``j`` is between old accepted nodes ``q[j]`` and ``q[j + 1]``; the
        interface-adjacent face therefore uses ``0.5 * (q[0] + q[1])`` where
        ``q[0]`` is the old accepted right interface composition.
        """
        q = np.asarray(q, dtype=np.float64)
        return 0.5 * (q[:-1] + q[1:])

    def _left_face_positions(self, future_s):
        u_faces = 0.5 * (self._u_grid[:-1] + self._u_grid[1:])
        return float(future_s) * u_faces

    def _right_face_positions(self, future_s):
        v_faces = 0.5 * (self._v_grid[:-1] + self._v_grid[1:])
        return float(future_s) + (self._R - float(future_s)) * v_faces

    def _left_lagged_face_diffusivity_matrices(self, p, future_s, time):
        """Returns left-phase lagged bulk face diffusivity matrices."""
        return self._bulk_face_diffusivity_matrices(
            self._left_lagged_face_compositions(p),
            self.phases[0],
            time,
            self._left_face_positions(future_s),
        )

    def _right_lagged_face_diffusivity_matrices(self, q, future_s, time):
        """Returns right-phase lagged bulk face diffusivity matrices."""
        return self._bulk_face_diffusivity_matrices(
            self._right_lagged_face_compositions(q),
            self.phases[1],
            time,
            self._right_face_positions(future_s),
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
        """
        Solves the implicit ternary recurrence using a single-stage Euler wrapper.

        Multi-stage iterators are rejected because each right-hand side evaluation
        performs a complete implicit moving-interface step for a specific time
        increment.
        """
        if iterator is not explicitEulerIterator:
            raise ValueError("MovingBoundaryIllingworthTernaryFD1DModel supports only explicitEulerIterator.")
        return super().solve(simTime, iterator=iterator, verbose=verbose, vIt=vIt, minDtFrac=minDtFrac, maxDtFrac=maxDtFrac)

    def getCurrentX(self):
        return [self._p_curr.copy(), self._q_curr.copy(), float(self._s_curr), float(self._eta_curr)]

    def flattenX(self, X):
        return np.concatenate((
            np.asarray(X[0], dtype=np.float64).reshape(-1),
            np.asarray(X[1], dtype=np.float64).reshape(-1),
            [float(X[2]), float(X[3])],
        ))

    def unflattenX(self, X_flat, X_ref):
        n_p = np.asarray(X_ref[0], dtype=np.float64).size
        n_q = np.asarray(X_ref[1], dtype=np.float64).size
        p = np.asarray(X_flat[:n_p], dtype=np.float64).reshape(np.asarray(X_ref[0]).shape)
        q = np.asarray(X_flat[n_p : n_p + n_q], dtype=np.float64).reshape(np.asarray(X_ref[1]).shape)
        s = float(X_flat[n_p + n_q])
        eta = float(X_flat[n_p + n_q + 1])
        return [p, q, s, eta]

    def _identity(self):
        return np.eye(2, dtype=np.float64)

    def _phase_face_diffusivity_matrices(self, D, n_faces, phase, context="transient face diffusivity"):
        """
        Returns validated face diffusivity matrices for one phase.

        A single already-validated 2-by-2 phase-uniform matrix is exposed as a
        broadcast view over all finite-volume faces. A face-specific input must
        already have shape ``(n_faces, 2, 2)`` and is validated face by face.
        """
        n_faces = int(n_faces)
        if n_faces < 1:
            raise ValueError("A transformed phase solve requires at least one face.")
        values = np.asarray(D)
        if values.shape == (2, 2):
            matrix = _validate_ternary_diffusivity_matrix(values, phase, context=context)
            return np.broadcast_to(matrix, (n_faces, 2, 2))
        if values.shape != (n_faces, 2, 2):
            raise ValueError(f"{context} for phase {phase} must have shape (2, 2) or ({n_faces}, 2, 2); received {values.shape}.")
        out = np.empty(values.shape, dtype=np.float64)
        for i, matrix in enumerate(values):
            out[i] = _validate_ternary_diffusivity_matrix(matrix, phase, context=f"{context} face {i}")
        return out

    def _left_interface_diffusive_flux(self, p_future, future_s, c_left, interface_face_matrix):
        """
        Returns the physical left-phase diffusive flux at the interface face.

        The interface-adjacent face is between ``p_future[-2]`` and the
        Dirichlet endpoint ``c_left == p_future[-1]``.
        """
        diff_l = _matvec_2x2(interface_face_matrix, (np.asarray(c_left, dtype=np.float64) - p_future[-2]) / (1.0 - self._u_grid[-2]))
        return diff_l / float(future_s)

    def _right_interface_diffusive_flux(self, q_future, future_s, c_right, interface_face_matrix):
        """
        Returns the physical right-phase diffusive flux at the interface face.

        The interface-adjacent face is between the Dirichlet endpoint
        ``c_right == q_future[0]`` and ``q_future[1]``.
        """
        diff_r = _matvec_2x2(interface_face_matrix, (q_future[1] - np.asarray(c_right, dtype=np.float64)) / self._v_grid[1])
        return diff_r / (self._R - float(future_s))

    def _solve_concentration_left_planar(self, p, s, future_s, dt, c_left, D_left_faces, motion_branch, validate_diffusivity=True):
        """
        Solves the left transformed bulk system using face diffusivity matrices.

        ``D_left_faces[j]`` is the 2-by-2 matrix on the face between
        transformed nodes ``j`` and ``j + 1``. The interface-adjacent face is
        ``D_left_faces[-1]``. Internal callers pass prevalidated matrices from
        the phase-uniform or face-diffusivity providers; external diagnostic
        calls validate by default.
        """
        _validate_motion_branch(motion_branch)
        n = len(p)
        D_left_values = np.asarray(D_left_faces)
        phase_uniform = D_left_values.shape == (2, 2)
        if phase_uniform:
            D_left_uniform = (
                _validate_ternary_diffusivity_matrix(D_left_values, self.phases[0], context="left transient diffusivity")
                if validate_diffusivity
                else np.asarray(D_left_values, dtype=np.float64)
            )
        else:
            D_left_faces = (
                self._phase_face_diffusivity_matrices(D_left_values, n - 1, self.phases[0])
                if validate_diffusivity
                else np.asarray(D_left_values, dtype=np.float64)
            )
        lower, diagonal, upper, rhs = _allocate_ternary_block_system(n)
        I = self._identity()
        tmpA_scale = float(dt) / float(future_s)
        tmpA_uniform = D_left_uniform * tmpA_scale if phase_uniform else None
        tmpA_faces = None if phase_uniform else D_left_faces * tmpA_scale
        tmpB = float(future_s) - float(s)
        u = self._u_grid

        if motion_branch == "positive":
            A_right = tmpA_uniform if phase_uniform else tmpA_faces[0]
            diagonal[0] = -A_right / u[1] - I * (future_s * u[1] / 2.0)
            upper[0] = A_right / u[1] + I * (tmpB * u[1] / 2.0)
            rhs[0] = -p[0] * s * u[1] / 2.0
            for i in range(1, n - 1):
                left_diff = u[i] - u[i - 1]
                right_diff = u[i + 1] - u[i]
                left_sum = u[i] + u[i - 1]
                right_sum = u[i + 1] + u[i]
                cell_width = right_sum - left_sum
                A_left = tmpA_uniform if phase_uniform else tmpA_faces[i - 1]
                A_right = tmpA_uniform if phase_uniform else tmpA_faces[i]
                lower[i] = A_left / left_diff
                if phase_uniform:
                    diagonal[i] = -tmpA_uniform * (1.0 / left_diff + 1.0 / right_diff)
                else:
                    diagonal[i] = -A_left / left_diff - A_right / right_diff
                diagonal[i] += -I * (tmpB * left_sum / 2.0 + future_s * cell_width / 2.0)
                upper[i] = A_right / right_diff + I * (tmpB * right_sum / 2.0)
                rhs[i] = -s * p[i] * cell_width / 2.0
        else:
            A_right = tmpA_uniform if phase_uniform else tmpA_faces[0]
            diagonal[0] = -A_right / u[1] + I * (tmpB * u[1] / 2.0 - future_s * u[1] / 2.0)
            upper[0] = A_right / u[1]
            rhs[0] = -p[0] * s * u[1] / 2.0
            for i in range(1, n - 1):
                left_diff = u[i] - u[i - 1]
                right_diff = u[i + 1] - u[i]
                left_sum = u[i] + u[i - 1]
                right_sum = u[i + 1] + u[i]
                cell_width = right_sum - left_sum
                A_left = tmpA_uniform if phase_uniform else tmpA_faces[i - 1]
                A_right = tmpA_uniform if phase_uniform else tmpA_faces[i]
                lower[i] = A_left / left_diff - I * (tmpB * left_sum / 2.0)
                if phase_uniform:
                    diagonal[i] = -tmpA_uniform * (1.0 / left_diff + 1.0 / right_diff)
                else:
                    diagonal[i] = -A_left / left_diff - A_right / right_diff
                diagonal[i] += I * (tmpB * right_sum / 2.0 - future_s * cell_width / 2.0)
                upper[i] = A_right / right_diff
                rhs[i] = -s * p[i] * cell_width / 2.0

        diagonal[-1] = -I
        rhs[-1] = -np.asarray(c_left, dtype=np.float64)
        profile = solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)
        interface_face_matrix = D_left_uniform if phase_uniform else np.asarray(D_left_faces[-1], dtype=np.float64)
        return _BulkPhaseSolveResult(
            profile=profile,
            interface_flux=self._left_interface_diffusive_flux(profile, future_s, c_left, interface_face_matrix),
            interface_face_matrix=interface_face_matrix,
        )

    def _new_concentration_left_planar(self, p, s, future_s, dt, c_left, D_left, motion_branch):
        """
        Compatibility wrapper returning only the solved left concentration.

        ``D_left`` may be either a single phase-uniform matrix or a face array
        with shape ``(len(p) - 1, 2, 2)``.
        """
        return self._solve_concentration_left_planar(p, s, future_s, dt, c_left, D_left, motion_branch).profile

    def _solve_concentration_right_planar(self, q, s, future_s, dt, c_right, D_right_faces, motion_branch, validate_diffusivity=True):
        """
        Solves the right transformed bulk system using face diffusivity matrices.

        ``D_right_faces[j]`` is the 2-by-2 matrix on the face between
        transformed nodes ``j`` and ``j + 1``. The interface-adjacent face is
        ``D_right_faces[0]``. Internal callers pass prevalidated matrices from
        the phase-uniform or face-diffusivity providers; external diagnostic
        calls validate by default.
        """
        _validate_motion_branch(motion_branch)
        n = len(q)
        D_right_values = np.asarray(D_right_faces)
        phase_uniform = D_right_values.shape == (2, 2)
        if phase_uniform:
            D_right_uniform = (
                _validate_ternary_diffusivity_matrix(D_right_values, self.phases[1], context="right transient diffusivity")
                if validate_diffusivity
                else np.asarray(D_right_values, dtype=np.float64)
            )
        else:
            D_right_faces = (
                self._phase_face_diffusivity_matrices(D_right_values, n - 1, self.phases[1])
                if validate_diffusivity
                else np.asarray(D_right_values, dtype=np.float64)
            )
        lower, diagonal, upper, rhs = _allocate_ternary_block_system(n)
        I = self._identity()
        tmpA_scale = float(dt) / (self._R - float(future_s))
        tmpA_uniform = D_right_uniform * tmpA_scale if phase_uniform else None
        tmpA_faces = None if phase_uniform else D_right_faces * tmpA_scale
        tmpB = float(future_s) - float(s)
        span = self._R - float(future_s)
        v = self._v_grid

        diagonal[0] = -I
        rhs[0] = -np.asarray(c_right, dtype=np.float64)
        if motion_branch == "positive":
            for i in range(1, n - 1):
                left_diff = v[i] - v[i - 1]
                right_diff = v[i + 1] - v[i]
                left_sum = v[i] + v[i - 1]
                right_sum = v[i + 1] + v[i]
                cell_width = right_sum - left_sum
                A_left = tmpA_uniform if phase_uniform else tmpA_faces[i - 1]
                A_right = tmpA_uniform if phase_uniform else tmpA_faces[i]
                lower[i] = A_left / left_diff
                if phase_uniform:
                    diagonal[i] = -tmpA_uniform * (1.0 / right_diff + 1.0 / left_diff)
                else:
                    diagonal[i] = -A_right / right_diff - A_left / left_diff
                diagonal[i] += -I * (tmpB * (1.0 - left_sum / 2.0) + span * cell_width / 2.0)
                upper[i] = A_right / right_diff + I * (tmpB * (1.0 - right_sum / 2.0))
                rhs[i] = -(self._R - s) * q[i] * cell_width / 2.0

            tmp = v[-2]
            A_left = tmpA_uniform if phase_uniform else tmpA_faces[-1]
            lower[-1] = A_left / (1.0 - tmp)
            diagonal[-1] = -A_left / (1.0 - tmp)
            diagonal[-1] += -I * (tmpB * (1.0 - (1.0 + tmp) / 2.0) + span * (1.0 - tmp) / 2.0)
            rhs[-1] = -q[-1] * (self._R - s) * (1.0 - tmp) / 2.0
        else:
            for i in range(1, n - 1):
                left_diff = v[i] - v[i - 1]
                right_diff = v[i + 1] - v[i]
                left_sum = v[i] + v[i - 1]
                right_sum = v[i + 1] + v[i]
                cell_width = right_sum - left_sum
                A_left = tmpA_uniform if phase_uniform else tmpA_faces[i - 1]
                A_right = tmpA_uniform if phase_uniform else tmpA_faces[i]
                lower[i] = A_left / left_diff - I * (tmpB * (1.0 - left_sum / 2.0))
                if phase_uniform:
                    diagonal[i] = -tmpA_uniform * (1.0 / right_diff + 1.0 / left_diff)
                else:
                    diagonal[i] = -A_right / right_diff - A_left / left_diff
                diagonal[i] += I * (tmpB * (1.0 - right_sum / 2.0) - span * cell_width / 2.0)
                upper[i] = A_right / right_diff
                rhs[i] = -(self._R - s) * q[i] * cell_width / 2.0

            tmp = v[-2]
            A_left = tmpA_uniform if phase_uniform else tmpA_faces[-1]
            lower[-1] = A_left / (1.0 - tmp) - I * (tmpB * (1.0 - (1.0 + tmp) / 2.0))
            diagonal[-1] = -A_left / (1.0 - tmp) - I * (span * (1.0 - tmp) / 2.0)
            rhs[-1] = -q[-1] * (self._R - s) * (1.0 - tmp) / 2.0

        profile = solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)
        interface_face_matrix = D_right_uniform if phase_uniform else np.asarray(D_right_faces[0], dtype=np.float64)
        return _BulkPhaseSolveResult(
            profile=profile,
            interface_flux=self._right_interface_diffusive_flux(profile, future_s, c_right, interface_face_matrix),
            interface_face_matrix=interface_face_matrix,
        )

    def _new_concentration_right_planar(self, q, s, future_s, dt, c_right, D_right, motion_branch):
        """
        Compatibility wrapper returning only the solved right concentration.

        ``D_right`` may be either a single phase-uniform matrix or a face array
        with shape ``(len(q) - 1, 2, 2)``.
        """
        return self._solve_concentration_right_planar(q, s, future_s, dt, c_right, D_right, motion_branch).profile

    def _bulk_picard_relative_tolerance(self):
        """Returns the effective relative tolerance for implicit bulk Picard solves."""
        return float(self.tolerance if self.bulkPicardRtol is None else self.bulkPicardRtol)

    def _bulk_picard_update_has_converged(self, update_norm, profile):
        """
        Tests Picard convergence against an absolute-plus-relative profile norm.

        The update norm is the maximum absolute component change from the
        current coefficient iterate to the unrelaxed linear solve. Relaxation is
        used only to form the next coefficient iterate when the update is still
        too large.
        """
        profile_scale = max(1.0e-12, float(np.max(np.abs(np.asarray(profile, dtype=np.float64)))))
        threshold = self.bulkPicardAtol + self._bulk_picard_relative_tolerance() * profile_scale
        return float(update_norm) <= threshold

    def _left_candidate_initial_profile(self, p, c_left):
        """Builds the deterministic left Picard seed for one interface trial."""
        profile = np.asarray(p, dtype=np.float64).copy()
        profile[-1] = np.asarray(c_left, dtype=np.float64)
        return profile

    def _right_candidate_initial_profile(self, q, c_right):
        """Builds the deterministic right Picard seed for one interface trial."""
        profile = np.asarray(q, dtype=np.float64).copy()
        profile[0] = np.asarray(c_right, dtype=np.float64)
        return profile

    def _solve_concentration_left_picard(self, p, s, future_s, dt, c_left, motion_branch):
        """
        Solves the left bulk recurrence with composition-dependent Picard matrices.

        Face matrices are evaluated from the current Picard profile at face
        averages, the existing linear finite-volume block solve is applied, and
        an optional relaxation forms the next coefficient iterate. Nonconvergence
        raises ``RuntimeError`` so the surrounding timestep retry logic can
        reduce ``dt`` without changing the outer interface Newton algorithm.
        """
        iterate = self._left_candidate_initial_profile(p, c_left)
        update_norm = np.inf
        D_faces = self._left_lagged_face_diffusivity_matrices(iterate, future_s, self.currentTime)
        for iteration in range(1, self.bulkPicardMaxIterations + 1):
            linear = self._solve_concentration_left_planar(p, s, future_s, dt, c_left, D_faces, motion_branch, validate_diffusivity=False)
            update_norm = float(np.max(np.abs(linear.profile - iterate)))
            if self._bulk_picard_update_has_converged(update_norm, linear.profile):
                return _BulkPhaseSolveResult(
                    profile=linear.profile,
                    interface_flux=linear.interface_flux,
                    interface_face_matrix=linear.interface_face_matrix,
                    inner_iterations=iteration,
                    inner_update_norm=update_norm,
                )
            next_iterate = iterate + self.bulkPicardRelaxation * (linear.profile - iterate)
            next_iterate[-1] = np.asarray(c_left, dtype=np.float64)
            D_next = self._left_lagged_face_diffusivity_matrices(next_iterate, future_s, self.currentTime)
            if self.bulkPicardRelaxation == 1.0 and np.array_equal(D_next, D_faces):
                return _BulkPhaseSolveResult(
                    profile=linear.profile,
                    interface_flux=linear.interface_flux,
                    interface_face_matrix=linear.interface_face_matrix,
                    inner_iterations=iteration,
                    inner_update_norm=0.0,
                )
            iterate = next_iterate
            D_faces = D_next

        reason = f"left bulk Picard solve failed to converge after {self.bulkPicardMaxIterations} iterations"
        self._record_bulk_phase_failure("left", self.bulkPicardMaxIterations, update_norm, reason)
        raise RuntimeError(reason)

    def _solve_concentration_right_picard(self, q, s, future_s, dt, c_right, motion_branch):
        """
        Solves the right bulk recurrence with composition-dependent Picard matrices.

        The iteration convention mirrors the left phase, but the interface face
        is face zero and the trial Dirichlet composition is imposed at node zero.
        """
        iterate = self._right_candidate_initial_profile(q, c_right)
        update_norm = np.inf
        D_faces = self._right_lagged_face_diffusivity_matrices(iterate, future_s, self.currentTime)
        for iteration in range(1, self.bulkPicardMaxIterations + 1):
            linear = self._solve_concentration_right_planar(q, s, future_s, dt, c_right, D_faces, motion_branch, validate_diffusivity=False)
            update_norm = float(np.max(np.abs(linear.profile - iterate)))
            if self._bulk_picard_update_has_converged(update_norm, linear.profile):
                return _BulkPhaseSolveResult(
                    profile=linear.profile,
                    interface_flux=linear.interface_flux,
                    interface_face_matrix=linear.interface_face_matrix,
                    inner_iterations=iteration,
                    inner_update_norm=update_norm,
                )
            next_iterate = iterate + self.bulkPicardRelaxation * (linear.profile - iterate)
            next_iterate[0] = np.asarray(c_right, dtype=np.float64)
            D_next = self._right_lagged_face_diffusivity_matrices(next_iterate, future_s, self.currentTime)
            if self.bulkPicardRelaxation == 1.0 and np.array_equal(D_next, D_faces):
                return _BulkPhaseSolveResult(
                    profile=linear.profile,
                    interface_flux=linear.interface_flux,
                    interface_face_matrix=linear.interface_face_matrix,
                    inner_iterations=iteration,
                    inner_update_norm=0.0,
                )
            iterate = next_iterate
            D_faces = D_next

        reason = f"right bulk Picard solve failed to converge after {self.bulkPicardMaxIterations} iterations"
        self._record_bulk_phase_failure("right", self.bulkPicardMaxIterations, update_norm, reason)
        raise RuntimeError(reason)

    def _interface_residual(
        self,
        p_future,
        q_future,
        s,
        old_s,
        future_s,
        dt,
        c_left,
        c_right,
        c_left_old,
        c_right_old,
        left_interface_flux,
        right_interface_flux,
        motion_branch,
    ):
        """
        Returns the two-component planar interface inventory residual.

        The original planar Illingworth residual assumes fixed phase-side
        interface compositions. In the ternary eta formulation those endpoint
        compositions may change during a step, and the trapezoidal transformed
        inventory stores those endpoint values in the interface-adjacent
        half-cells. The endpoint correction below accounts for that inventory
        change using the accepted old geometry ``s`` and the old discrete
        endpoint values from ``p[-1]`` and ``q[0]``. ``motion_branch`` must be
        the same branch used by both phase bulk solves for this residual.
        """
        _validate_motion_branch(motion_branch)
        diff_l = np.asarray(left_interface_flux, dtype=np.float64).reshape(2)
        diff_r = np.asarray(right_interface_flux, dtype=np.float64).reshape(2)
        rhs = (diff_r - diff_l) * dt

        if motion_branch == "positive":
            lhs = c_left - q_future[1] * (1.0 - self._v_grid[1] / 2.0) - c_right * self._v_grid[1] / 2.0
        else:
            lhs = p_future[-2] * (0.5 + self._u_grid[-2] / 2.0) + c_left * (0.5 - self._u_grid[-2] / 2.0) - c_right
        residual = (future_s - s) * lhs - rhs
        left_endpoint_change = (
            float(s)
            * 0.5
            * (1.0 - self._u_grid[-2])
            * (np.asarray(c_left, dtype=np.float64) - np.asarray(c_left_old, dtype=np.float64))
        )
        right_endpoint_change = (
            (self._R - float(s))
            * 0.5
            * self._v_grid[1]
            * (np.asarray(c_right, dtype=np.float64) - np.asarray(c_right_old, dtype=np.float64))
        )
        return residual + left_endpoint_change + right_endpoint_change

    def _active_interface_variables(self):
        eta_lower, eta_upper = tuple(float(v) for v in self.interfaceEquilibrium.eta_bounds)
        eta_active = not np.isclose(eta_lower, eta_upper, rtol=0.0, atol=1e-14)
        return eta_lower, eta_upper, eta_active

    def _eta_scaling_bounds(self):
        """
        Returns finite non-degenerate eta bounds for scaled nonlinear variables.

        The ternary Illingworth interface solve treats eta as a dimensionless
        nonlinear unknown on ``[0, 1]`` after affine scaling from the physical
        tie-line coordinate. Collapsed or invalid eta intervals cannot be scaled
        robustly and are rejected explicitly.
        """
        eta_bounds = getattr(self.interfaceEquilibrium, "eta_bounds", None)
        if eta_bounds is None or len(eta_bounds) != 2:
            raise ValueError("interface_equilibrium must expose two finite eta_bounds.")
        eta_lower, eta_upper = tuple(float(v) for v in eta_bounds)
        if not np.isfinite(eta_lower) or not np.isfinite(eta_upper) or eta_upper <= eta_lower:
            raise ValueError("interface_equilibrium eta_bounds must be finite and non-degenerate for scaled solving.")
        return eta_lower, eta_upper, eta_upper - eta_lower

    def _interface_scaled_bounds(self):
        """
        Returns scaled nonlinear-variable bounds for ``[s_hat, eta_hat]``.

        ``s_hat`` is kept strictly inside the planar domain by a dimensionless
        margin. ``eta_hat`` is allowed to reach either endpoint of the
        eta interval.
        """
        s_hat_eps = 1e-14
        return (
            np.asarray([s_hat_eps, 0.0], dtype=np.float64),
            np.asarray([1.0 - s_hat_eps, 1.0], dtype=np.float64),
        )

    def _interface_physical_to_scaled(self, future_s, future_eta, eta_lower, eta_span):
        """Converts physical nonlinear variables to ``[s_hat, eta_hat]``."""
        if not np.isfinite(self._R) or self._R <= 0.0:
            raise ValueError("Domain length must be positive and finite for scaled interface solving.")
        return np.asarray(
            [float(future_s) / self._R, (float(future_eta) - eta_lower) / eta_span],
            dtype=np.float64,
        )

    def _interface_scaled_to_physical(self, x_hat, eta_lower, eta_span):
        """Converts scaled nonlinear variables ``[s_hat, eta_hat]`` to physical values."""
        x_hat = np.asarray(x_hat, dtype=np.float64).reshape(2)
        return float(x_hat[0] * self._R), float(eta_lower + x_hat[1] * eta_span)

    def _interface_residual_scale(self, p, q, s):
        """
        Builds fixed componentwise residual scales for one implicit timestep.

        The physical residual has units of composition times length. Scaling by
        the accepted old inventory or by ``R`` times an O(1) composition scale
        gives a dimensionless residual norm whose tolerance is independent of
        the chosen length units.
        """
        old_inventory = integrate_planar_transformed_profile_components(p, q, s, self._R, self._u_grid, self._v_grid)
        composition_scale = 1.0
        floor = 1e-300
        return np.maximum(np.maximum(np.abs(old_inventory), self._R * composition_scale), floor)

    def _scaled_interface_variables_in_bounds(self, x_hat, lower, upper):
        """Returns whether scaled interface variables satisfy the solver bounds."""
        x_hat = np.asarray(x_hat, dtype=np.float64)
        return not (np.any(x_hat < lower) or np.any(x_hat > upper))

    def _interface_candidate_has_converged(self, candidate, lower, upper):
        """Applies the final nonlinear convergence check without changing tolerance semantics."""
        return self._scaled_interface_variables_in_bounds(candidate.x_hat, lower, upper) and candidate.scaled_norm <= self.residualTolerance

    def _interface_candidate_improves(self, candidate, current_norm):
        """Returns whether a backtracking trial improves the current scaled residual norm."""
        return np.isfinite(candidate.scaled_norm) and candidate.scaled_norm < current_norm

    def _bounded_scaled_newton_step(self, x_hat, newton_step, lower, upper):
        """
        Returns an active-bound-aware Newton direction and feasible first alpha.

        Variables already at a bound have outward Newton components zeroed so
        they do not block feasible motion in other components. The first line
        search length is the largest alpha satisfying the scaled bounds, with a
        small fraction-to-boundary safety factor only when a nominally interior
        variable would otherwise land exactly on a limiting bound.
        """
        x_hat = np.asarray(x_hat, dtype=np.float64)
        step = np.asarray(newton_step, dtype=np.float64).copy()
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        if not np.all(np.isfinite(step)):
            raise RuntimeError("Interface Newton step is non-finite.")
        active_tol = 10.0 * np.finfo(float).eps
        for i in range(step.size):
            if x_hat[i] <= lower[i] + active_tol and step[i] < 0.0:
                step[i] = 0.0
            elif x_hat[i] >= upper[i] - active_tol and step[i] > 0.0:
                step[i] = 0.0
        if not np.any(step):
            return step, 0.0

        alpha_max = 1.0
        limited_by_interior_variable = False
        for i in range(step.size):
            if step[i] > 0.0:
                limit = (upper[i] - x_hat[i]) / step[i]
            elif step[i] < 0.0:
                limit = (lower[i] - x_hat[i]) / step[i]
            else:
                continue
            if limit < alpha_max:
                alpha_max = float(limit)
                limited_by_interior_variable = lower[i] + active_tol < x_hat[i] < upper[i] - active_tol

        alpha_max = max(0.0, alpha_max)
        if alpha_max < 1.0 and limited_by_interior_variable:
            alpha_max *= 1.0 - 1e-12
        return step, alpha_max

    def _evaluate_interface_candidate(
        self,
        p,
        q,
        s,
        old_s,
        dt,
        eta_lower,
        eta_span,
        residual_scale,
        c_left_old,
        c_right_old,
        x_hat,
        motion_branch,
    ):
        """
        Evaluates one fixed-branch nonlinear interface trial.

        The caller owns branch selection, Newton convergence, and line search.
        This helper only builds the mutually consistent physical candidate and
        its scaled residual for the supplied trial variables.
        """
        self._reset_candidate_bulk_diagnostics()
        future_s, future_eta = self._interface_scaled_to_physical(x_hat, eta_lower, eta_span)
        c_left, c_right = self._interface_compositions(future_eta)
        if self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_PHASE_UNIFORM:
            try:
                D_left = self._phase_diffusivity_matrix(c_left, self.phases[0], self.currentTime, future_s)
                left_result = self._solve_concentration_left_planar(p, s, future_s, dt, c_left, D_left, motion_branch, validate_diffusivity=False)
            except Exception as exc:
                self._record_bulk_phase_failure("left", 0, np.inf, f"left bulk solve failed: {exc}")
                raise
            self._record_bulk_phase_success("left", left_result)
            try:
                D_right = self._phase_diffusivity_matrix(c_right, self.phases[1], self.currentTime, future_s)
                right_result = self._solve_concentration_right_planar(q, s, future_s, dt, c_right, D_right, motion_branch, validate_diffusivity=False)
            except Exception as exc:
                self._record_bulk_phase_failure("right", 0, np.inf, f"right bulk solve failed: {exc}")
                raise
            self._record_bulk_phase_success("right", right_result)
        elif self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_LAGGED:
            try:
                D_left = self._left_lagged_face_diffusivity_matrices(p, future_s, self.currentTime)
                left_result = self._solve_concentration_left_planar(p, s, future_s, dt, c_left, D_left, motion_branch, validate_diffusivity=False)
            except Exception as exc:
                self._record_bulk_phase_failure("left", 0, np.inf, f"left bulk solve failed: {exc}")
                raise
            self._record_bulk_phase_success("left", left_result)
            try:
                D_right = self._right_lagged_face_diffusivity_matrices(q, future_s, self.currentTime)
                right_result = self._solve_concentration_right_planar(q, s, future_s, dt, c_right, D_right, motion_branch, validate_diffusivity=False)
            except Exception as exc:
                self._record_bulk_phase_failure("right", 0, np.inf, f"right bulk solve failed: {exc}")
                raise
            self._record_bulk_phase_success("right", right_result)
        elif self.bulkDiffusivityMode == _BULK_DIFFUSIVITY_IMPLICIT:
            try:
                left_result = self._solve_concentration_left_picard(p, s, future_s, dt, c_left, motion_branch)
            except Exception as exc:
                if self._lastBulkFailureReason is None or self._lastBulkFailureReason == "not evaluated":
                    self._record_bulk_phase_failure("left", 0, np.inf, f"left bulk solve failed: {exc}")
                raise
            self._record_bulk_phase_success("left", left_result)
            try:
                right_result = self._solve_concentration_right_picard(q, s, future_s, dt, c_right, motion_branch)
            except Exception as exc:
                if self._lastBulkFailureReason is None or self._lastBulkFailureReason == "not evaluated":
                    self._record_bulk_phase_failure("right", 0, np.inf, f"right bulk solve failed: {exc}")
                raise
            self._record_bulk_phase_success("right", right_result)
        else:
            raise ValueError(
                "bulkDiffusivityMode must be 'phase_uniform', "
                "'composition_dependent_lagged', or 'composition_dependent_implicit'."
            )
        residual = self._interface_residual(
            left_result.profile,
            right_result.profile,
            s,
            old_s,
            future_s,
            dt,
            c_left,
            c_right,
            c_left_old,
            c_right_old,
            left_result.interface_flux,
            right_result.interface_flux,
            motion_branch,
        )
        scaled_residual = residual / residual_scale
        return _InterfaceCandidate(
            x_hat=np.asarray(x_hat, dtype=np.float64).copy(),
            future_s=future_s,
            future_eta=future_eta,
            motion_branch=motion_branch,
            c_left=c_left,
            c_right=c_right,
            D_left=left_result.interface_face_matrix,
            D_right=right_result.interface_face_matrix,
            p_future=left_result.profile,
            q_future=right_result.profile,
            residual=residual,
            scaled_residual=scaled_residual,
            scaled_norm=float(np.max(np.abs(scaled_residual))),
            physical_norm=float(np.max(np.abs(residual))),
            left_inner_iterations=left_result.inner_iterations,
            right_inner_iterations=right_result.inner_iterations,
            left_inner_update_norm=left_result.inner_update_norm,
            right_inner_update_norm=right_result.inner_update_norm,
            bulk_diffusivity_evaluations=left_result.diffusivity_evaluations + right_result.diffusivity_evaluations,
        )

    def _solve_interface_planar(self, p, q, s, old_s, eta, dt):
        self._reset_implicit_diagnostics()
        self._currentBulkDiffusivityProviderCalls = 0
        self._currentBulkFaceMatricesEvaluated = 0
        self._bulkDiffusivityCountingActive = True
        eta_lower, _, eta_span = self._eta_scaling_bounds()
        lower, upper = self._interface_scaled_bounds()
        x_hat = self._interface_physical_to_scaled(s, eta, eta_lower, eta_span)
        if not self._scaled_interface_variables_in_bounds(x_hat, lower, upper):
            self._bulkDiffusivityCountingActive = False
            raise ValueError("Initial nonlinear interface iterate lies outside scaled solve bounds.")
        residual_scale = self._interface_residual_scale(p, q, s)
        c_left_old = np.asarray(p[-1], dtype=np.float64).copy()
        c_right_old = np.asarray(q[0], dtype=np.float64).copy()
        best_scaled_norm = np.inf
        best_physical_norm = np.inf
        best_motion_branch = None
        candidate_evaluations = 0
        jacobian_evaluations = 0
        iterations_attempted = 0
        failure_reason = "maximum iterations reached"

        def record_failure(reason):
            self._record_implicit_failure(
                iterations_attempted,
                best_scaled_norm,
                best_physical_norm,
                best_motion_branch,
                candidate_evaluations,
                jacobian_evaluations,
                reason,
            )

        def evaluate_candidate(trial_x_hat, motion_branch):
            nonlocal candidate_evaluations
            candidate_evaluations += 1
            candidate = self._evaluate_interface_candidate(
                p=p,
                q=q,
                s=s,
                old_s=old_s,
                dt=dt,
                eta_lower=eta_lower,
                eta_span=eta_span,
                residual_scale=residual_scale,
                c_left_old=c_left_old,
                c_right_old=c_right_old,
                x_hat=trial_x_hat,
                motion_branch=motion_branch,
            )
            self._record_completed_bulk_candidate(candidate)
            return candidate

        for count in range(self.maxIterations):
            iterations_attempted = count + 1
            future_s, _ = self._interface_scaled_to_physical(x_hat, eta_lower, eta_span)
            motion_branch = _select_interface_motion_branch(s, old_s, future_s)
            try:
                candidate = evaluate_candidate(x_hat, motion_branch)
            except Exception:
                if best_motion_branch is None:
                    best_motion_branch = motion_branch
                record_failure("candidate evaluation failed")
                raise
            scaled_residual = candidate.scaled_residual
            norm = candidate.scaled_norm
            physical_norm = candidate.physical_norm
            if norm < best_scaled_norm:
                best_scaled_norm = norm
                best_physical_norm = physical_norm
                best_motion_branch = motion_branch
            if self._interface_candidate_has_converged(candidate, lower, upper):
                self._record_implicit_success(iterations_attempted, candidate, candidate_evaluations, jacobian_evaluations)
                return (
                    candidate.p_future,
                    candidate.q_future,
                    candidate.future_s,
                    candidate.future_eta,
                    candidate.c_left,
                    candidate.c_right,
                    candidate.D_left,
                    candidate.D_right,
                )

            jacobian = np.zeros((2, len(x_hat)), dtype=np.float64)
            for variable in range(len(x_hat)):
                step, x_perturbed, difference_direction = _bounded_finite_difference_perturbation(x_hat, lower, upper, variable)
                if difference_direction == "forward":
                    try:
                        residual_perturbed = evaluate_candidate(x_perturbed, motion_branch).scaled_residual
                    except Exception:
                        if best_motion_branch is None:
                            best_motion_branch = motion_branch
                        record_failure("candidate evaluation failed")
                        raise
                    jacobian[:, variable] = (residual_perturbed - scaled_residual) / step
                else:
                    try:
                        residual_perturbed = evaluate_candidate(x_perturbed, motion_branch).scaled_residual
                    except Exception:
                        if best_motion_branch is None:
                            best_motion_branch = motion_branch
                        record_failure("candidate evaluation failed")
                        raise
                    jacobian[:, variable] = (scaled_residual - residual_perturbed) / step
            jacobian_evaluations += 1

            try:
                step = self._least_squares_step_2xN(jacobian, scaled_residual)
                step, alpha_start = self._bounded_scaled_newton_step(x_hat, step, lower, upper)
            except RuntimeError:
                record_failure("singular or unusable Jacobian/Newton step")
                raise
            accepted = False
            for scale in (alpha_start, 0.5 * alpha_start, 0.25 * alpha_start, 0.125 * alpha_start, 0.0625 * alpha_start):
                if scale <= 0.0:
                    continue
                trial = x_hat + scale * step
                if not self._scaled_interface_variables_in_bounds(trial, lower, upper):
                    continue
                try:
                    trial_candidate = evaluate_candidate(trial, motion_branch)
                except Exception:
                    if best_motion_branch is None:
                        best_motion_branch = motion_branch
                    record_failure("candidate evaluation failed")
                    raise
                trial_norm = trial_candidate.scaled_norm
                if self._interface_candidate_improves(trial_candidate, norm):
                    x_hat = trial
                    accepted = True
                    break
            if not accepted:
                failure_reason = "line search failed"
                break

        record_failure(failure_reason)
        raise RuntimeError(
            "Ternary Illingworth interface solve failed to converge; "
            f"best residual was {best_scaled_norm:.3e}."
        )

    def _least_squares_step_2xN(self, jacobian, residual):
        """
        Returns the Gauss-Newton update for a 2-residual, one- or two-variable solve.

        This avoids calling ``np.linalg`` for tiny systems, which also makes the
        interface nonlinear solve independent of the platform LAPACK behavior.
        """
        jacobian = np.asarray(jacobian, dtype=np.float64)
        residual = np.asarray(residual, dtype=np.float64).reshape(2)
        if jacobian.shape[1] == 1:
            j = jacobian[:, 0]
            denom = float(j[0] * j[0] + j[1] * j[1])
            if denom <= 1e-300:
                raise RuntimeError("Ternary Newton Jacobian is singular.")
            return np.asarray([-float(j[0] * residual[0] + j[1] * residual[1]) / denom], dtype=np.float64)
        return _newton_step_2x2(jacobian, residual)

    def _take_implicit_step_planar(self, p, q, s, old_s, eta, dt):
        return self._solve_interface_planar(p, q, s, old_s, eta, dt)

    def _reconstruct_physical_profile(self, p, q, s):
        return reconstruct_planar_transformed_profile_components(
            z=self._z,
            p=p,
            q=q,
            s=s,
            domain_length=self._R,
            u=self._u_grid,
            v=self._v_grid,
        )

    def _accept_step_result(self, p, q, s, eta, result, trial_dt, retry):
        """Stores diagnostics and returns derivatives for an accepted implicit trial."""
        p_new, q_new, s_new, eta_new, c_left_new, c_right_new, D_left, D_right = result
        self._currdt = float(trial_dt)
        self._lastStepRetries = int(retry)
        self._lastInterfaceCompositions = (c_left_new.copy(), c_right_new.copy())
        self._D_left = D_left
        self._D_right = D_right
        return [
            (p_new - p) / trial_dt,
            (q_new - q) / trial_dt,
            (s_new - s) / trial_dt,
            (eta_new - eta) / trial_dt,
        ]

    def _try_step_retries(self, p, q, s, eta, trial_dt, retry_count, retry_offset=0):
        """Attempts implicit solves while shrinking ``trial_dt`` after each failed trial."""
        last_error = None
        for retry in range(int(retry_count)):
            try:
                result = self._take_implicit_step_planar(
                    p,
                    q,
                    s,
                    self._s_old,
                    eta,
                    trial_dt,
                )
                return self._accept_step_result(p, q, s, eta, result, trial_dt, retry_offset + retry), trial_dt, None
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
            f"Ternary Illingworth phase {phase!r} width {float(width):.6g} is below "
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
        p = np.asarray(xCurr[0], dtype=np.float64).reshape((-1, 2))
        q = np.asarray(xCurr[1], dtype=np.float64).reshape((-1, 2))
        s = self._clipInterfacePosition(float(xCurr[2]), strict=True)
        eta = float(xCurr[3])
        c_left, c_right = self._interface_compositions(eta)
        p = p.copy()
        q = q.copy()
        p[-1] = c_left
        q[0] = c_right
        dt = self._compute_dt(t)
        if self._nearFinalNoop:
            return [np.zeros_like(p), np.zeros_like(q), 0.0, 0.0]

        trial_dt = float(dt)
        dXdt, trial_dt, last_error = self._try_step_retries(p, q, s, eta, trial_dt, self.maxStepRetries)
        if dXdt is not None:
            return dXdt

        thin_phase = self._single_terminal_thin_phase(s)
        if thin_phase is not None:
            phase_index, width = thin_phase
            if self._confirm_terminal_thin_phase_retries(phase_index, width, t, trial_dt):
                dXdt, trial_dt, extra_error = self._try_step_retries(
                    p,
                    q,
                    s,
                    eta,
                    trial_dt,
                    self.terminalThinPhaseExtraRetries,
                    retry_offset=self.maxStepRetries,
                )
                if dXdt is not None:
                    if self._single_terminal_thin_phase(s + dXdt[2] * trial_dt) is None:
                        raise ValueError(f"Expecting next interface to also satisfy _single_terminal_thin_phase() but got: {s + dXdt[2] * trial_dt}")
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
        print(f"eta: {eta}")
        print(f"s: {s}")
        print(f"R-s: {self._R-s}")
        from examples.debugInPlace import debugInPlace
        debugInPlace()
        raise RuntimeError("Ternary Illingworth step failed after timestep retries.") from last_error

    def getDt(self, dXdt):
        if np.isfinite(self._currdt) and self._currdt > 0:
            return self._currdt
        return self.timeStep

    def postProcess(self, time, x):
        if self._nearFinalNoop:
            self.currentTime = time
            self._nearFinalNoop = False
            return [self._p_curr.copy(), self._q_curr.copy(), self._s_curr, self._eta_curr], True
        GenericModel.postProcess(self, time, x)
        p = np.asarray(x[0], dtype=np.float64).reshape((-1, 2)).copy()
        q = np.asarray(x[1], dtype=np.float64).reshape((-1, 2)).copy()
        s = self._clipInterfacePosition(float(x[2]), strict=True)
        eta = float(x[3])
        c_left, c_right = self._interface_compositions(eta)
        p[-1] = c_left
        q[0] = c_right

        self._validate_profile_compositions(p, "left transformed profile")
        self._validate_profile_compositions(q, "right transformed profile")
        physical = self._reconstruct_physical_profile(p, q, s)
        self.data.record(time, physical)
        self.interfaceData.record(time, s)
        self.etaData.record(time, eta)
        if self.recordPqData:
            self.pData.record(time, p)
            self.qData.record(time, q)
        self.inventoryData.record(time, self.getTotalInventoryFromState(p, q, s))

        self._s_old = float(self._s_curr)
        self._p_curr = p
        self._q_curr = q
        self._s_curr = float(s)
        self._eta_curr = float(eta)
        self.updateCoupledModels()
        if self._terminalThinPhaseStop:
            self.finalTime = time
            self._terminalThinPhaseStop = False
            return [self._p_curr.copy(), self._q_curr.copy(), self._s_curr, self._eta_curr], True
        return [self._p_curr.copy(), self._q_curr.copy(), self._s_curr, self._eta_curr], False

    def _validate_profile_compositions(self, profile, name):
        profile = np.asarray(profile, dtype=np.float64)
        min_comp = float(self.constraints.minComposition)
        dependent = 1.0 - np.sum(profile, axis=1)
        if not np.all(np.isfinite(profile)):
            raise ValueError(f"{name} contains non-finite values.")
        if np.any(profile < min_comp) or np.any(dependent < min_comp):
            raise ValueError(f"{name} violates ternary composition bounds.")

    def postSolve(self):
        self.data.finalize()
        self.interfaceData.finalize()
        self.etaData.finalize()
        self.inventoryData.finalize()
        if self.pData is not None:
            self.pData.finalize()
        if self.qData is not None:
            self.qData.finalize()

    def getInterfacePosition(self, time=None):
        return self.interfaceData.y(time)

    def getInterfaceEta(self, time=None):
        """Returns the recorded scalar tie-line coordinate."""
        return self.etaData.y(time)

    def getInterfaceCompositions(self, time=None):
        """Returns the interface composition vectors at the current or recorded eta."""
        return self._interface_compositions(self.getInterfaceEta(time))

    def getTransformedState(self, time=None):
        """Returns the recorded transformed left and right ternary profiles."""
        return self.getTransformedStateLeft(time), self.getTransformedStateRight(time)

    def getTransformedStateLeft(self, time=None):
        """Returns the recorded left transformed composition matrix ``p``."""
        if self.pData is None:
            raise ValueError("Transformed left-state history is not available; set record_pq_data=True.")
        return self.pData.y(time)

    def getTransformedStateRight(self, time=None):
        """Returns the recorded right transformed composition matrix ``q``."""
        if self.qData is None:
            raise ValueError("Transformed right-state history is not available; set record_pq_data=True.")
        return self.qData.y(time)

    def getTotalInventoryFromState(self, p, q, s):
        return integrate_planar_transformed_profile_components(
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
        return self.inventoryData.y(time)

    def getTotalMass(self, time=None):
        return self.getTotalInventory(time=time)

    def getCompositions(self, time=None):
        """
        Returns full ternary mole fractions on the physical mesh.

        This model supports only substitutional ternaries, so the stored
        independent responses are already mole fractions and the dependent
        component is reconstructed as ``1 - x_1 - x_2``.
        """
        independent = np.asarray(self.data.y(time), dtype=np.float64)
        dependent = 1.0 - np.sum(independent, axis=1)
        return np.column_stack((dependent, independent))

    def checkConservation(self, tolerance: float, time=None):
        """
        Checks componentwise transformed-inventory drift from the initial value.

        The return value is the absolute drift vector for the two independent
        components. A warning is emitted when any component exceeds ``tolerance``.
        """
        if self._initialInventory is None:
            raise ValueError("Model must be setup before conservation checks.")
        drift = np.abs(self.getTotalInventory(time=time) - self._initialInventory)
        if np.any(drift > float(tolerance)):
            warnings.warn(
                f"Ternary Illingworth inventory drift {drift} exceeded tolerance {float(tolerance):.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return drift

    def toDict(self):
        """Converts solved ternary Illingworth histories to a restart dictionary."""
        data = super().toDict()
        data.update(
            {
                "interface_position": self.interfaceData._y,
                "interface_eta": self.etaData._y,
                "inventory": self.inventoryData._y,
                "interface_interval": self.interfaceData.recordInterval,
                "interface_index": self.interfaceData.N,
            }
        )
        if self.pData is not None:
            data["p"] = self.pData._y
            data["q"] = self.qData._y
        return data
