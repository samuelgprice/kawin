import warnings
from dataclasses import dataclass

import numpy as np
from scipy import optimize

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


@dataclass(frozen=True)
class InitialEtaEstimate:
    """
    Diagnostics from held-profile Stefan residual initial tie-line selection.

    ``eta`` is the selected tie-line coordinate. ``velocity`` is the scalar
    least-squares interface velocity that best aligns the two-component flux
    imbalance with the interface composition jump for the selected eta.
    ``method`` identifies the initialization strategy that produced the
    estimate. ``branch`` is ``"positive"``, ``"negative"``, or ``None`` for
    methods without a swept-inventory branch choice.
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


def _validate_stefan_diffusivity_matrix(D, phase):
    """Validates a ternary 2x2 diffusivity matrix used by the Stefan estimator."""
    D = np.asarray(D, dtype=np.float64)
    if D.shape != (2, 2) or not np.all(np.isfinite(D)):
        raise ValueError(f"Diffusivity for phase {phase} must be a finite 2x2 matrix.")
    trace = float(D[0, 0] + D[1, 1])
    determinant = float(D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0])
    discriminant = trace * trace - 4.0 * determinant
    scale = max(trace * trace, abs(determinant), 1.0)
    if discriminant < -1e-12 * scale:
        raise ValueError(f"Diffusivity for phase {phase} must have positive real eigenvalues.")
    root = float(np.sqrt(max(discriminant, 0.0)))
    eigenvalues = (0.5 * (trace + root), 0.5 * (trace - root))
    if eigenvalues[0] <= 0.0 or eigenvalues[1] <= 0.0:
        raise ValueError(f"Diffusivity for phase {phase} must have positive real eigenvalues.")
    if abs(determinant) <= 1e-300:
        raise ValueError(f"Diffusivity for phase {phase} is singular.")
    inverse = np.asarray([[D[1, 1], -D[0, 1]], [-D[1, 0], D[0, 0]]], dtype=np.float64) / determinant
    condition_estimate = np.max(np.sum(np.abs(D), axis=1)) * np.max(np.sum(np.abs(inverse), axis=1))
    if condition_estimate > 1e12:
        raise ValueError(f"Diffusivity for phase {phase} is too ill-conditioned for the ternary Illingworth solve.")
    return D.astype(np.float64)


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
    return _validate_stefan_diffusivity_matrix(D, phase)


def estimate_initial_eta_from_stefan_residual(
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
    root_xtol=1e-12,
    root_rtol=1e-12,
    root_maxiter=100,
):
    """
    Estimates the initial ternary tie-line coordinate from a Stefan residual.

    The estimator keeps the initial transformed profile fixed and solves a
    scalar equal-velocity condition with ``scipy.optimize.root_scalar`` using
    ``method='brentq'``. The root condition is the 2D cross product between the
    interface composition jump and the two-component flux imbalance. At zero,
    the flux imbalance is parallel to the composition jump, so both independent
    components imply the same scalar interface velocity. This local diagnostic
    is used only to choose the initial tie-line; it does not run the implicit
    Illingworth update and intentionally does not call ``_interface_residual``.
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
        raise ValueError("estimate_initial_eta_from_stefan_residual expects a 1D domain starting at 0.")
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

    s = float(interface_position)
    domain_length = float(z[-1] - z[0])
    if not (0.0 < s < domain_length):
        raise ValueError("interface_position must lie strictly inside the domain.")
    eta_bounds = _validate_eta_bounds(interface_equilibrium)
    bracket = _coerce_initial_eta_bracket(eta_bracket, eta_bounds)
    temperature_value = _temperature_at_interface(temperature, s)

    left_mask = z <= s
    right_mask = z >= s
    if not np.any(left_mask) or not np.any(right_mask):
        raise ValueError("Initial interface leaves an empty phase.")
    z_left_adjacent = s * float(u_grid[-2])
    z_right_adjacent = s + (domain_length - s) * float(v_grid[1])
    p_adjacent = np.asarray(
        [np.interp(z_left_adjacent, z[left_mask], composition[left_mask, component]) for component in range(2)],
        dtype=np.float64,
    )
    q_adjacent = np.asarray(
        [np.interp(z_right_adjacent, z[right_mask], composition[right_mask, component]) for component in range(2)],
        dtype=np.float64,
    )

    def evaluate_eta(eta):
        c_left, c_right = interface_equilibrium.interface_compositions(float(eta))
        c_left = np.asarray(c_left, dtype=np.float64).reshape(2)
        c_right = np.asarray(c_right, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(c_left)) or not np.all(np.isfinite(c_right)):
            raise ValueError("interface compositions are non-finite at the queried eta.")
        jump = c_left - c_right
        jump_norm_sq = float(np.dot(jump, jump))
        if jump_norm_sq <= 1e-300:
            raise ValueError("tie-line has a degenerate interface composition jump at the queried eta.")
        D_left = _get_stefan_interdiffusivity(thermodynamics, c_left, temperature_value, phases[0])
        D_right = _get_stefan_interdiffusivity(thermodynamics, c_right, temperature_value, phases[1])
        left_gradient = (c_left - p_adjacent) / (s * (1.0 - float(u_grid[-2])))
        right_gradient = (q_adjacent - c_right) / ((domain_length - s) * float(v_grid[1]))
        flux_delta = _matvec_2x2(D_right, right_gradient) - _matvec_2x2(D_left, left_gradient)
        velocity = float(np.dot(jump, flux_delta) / jump_norm_sq)
        residual = velocity * jump - flux_delta
        root_value = float(jump[0] * flux_delta[1] - jump[1] * flux_delta[0])
        if not np.isfinite(root_value) or not np.all(np.isfinite(residual)):
            raise ValueError("Stefan residual is non-finite at the queried eta.")
        return root_value, velocity, residual.copy(), flux_delta.copy(), c_left.copy(), c_right.copy()

    f_lower = evaluate_eta(bracket[0])[0]
    f_upper = evaluate_eta(bracket[1])[0]
    endpoint_atol = max(1e-14 * max(abs(f_lower), abs(f_upper), 1.0), 1e-300)
    lower_is_root = abs(f_lower) <= endpoint_atol
    upper_is_root = abs(f_upper) <= endpoint_atol
    if not (lower_is_root or upper_is_root) and f_lower * f_upper > 0.0:
        raise ValueError(
            "Initial eta bracket does not contain a sign change for the held-profile Stefan root; "
            f"f({bracket[0]:.6g})={f_lower:.6g}, f({bracket[1]:.6g})={f_upper:.6g}."
        )

    def root_function(eta):
        eta = float(eta)
        if lower_is_root and np.isclose(eta, bracket[0], rtol=0.0, atol=0.0):
            return 0.0
        if upper_is_root and np.isclose(eta, bracket[1], rtol=0.0, atol=0.0):
            return 0.0
        return evaluate_eta(eta)[0]

    solution = optimize.root_scalar(
        root_function,
        bracket=bracket,
        method="brentq",
        xtol=float(root_xtol),
        rtol=float(root_rtol),
        maxiter=int(root_maxiter),
    )
    if not solution.converged:
        raise ValueError("Initial eta Brent root solve failed to converge.")

    eta = float(solution.root)
    _, velocity, residual, flux_delta, c_left, c_right = evaluate_eta(eta)
    return InitialEtaEstimate(
        eta=eta,
        residual_norm=float(np.max(np.abs(residual))),
        velocity=velocity,
        residual=residual,
        flux_delta=flux_delta,
        left_interface_composition=c_left,
        right_interface_composition=c_right,
        method="stefan_cross_brentq",
        solver="root_scalar(brentq)",
        bracket=bracket,
        converged=bool(solution.converged),
        iterations=int(solution.iterations),
        function_calls=int(solution.function_calls),
    )


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
        D_left = _get_stefan_interdiffusivity(thermodynamics, c_left, temperature_value, phases[0])
        D_right = _get_stefan_interdiffusivity(thermodynamics, c_right, temperature_value, phases[1])
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
                step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(x[variable]))
                if np.isfinite(upper[variable] - lower[variable]):
                    step = min(step, 0.25 * max(upper[variable] - lower[variable], 1e-15))
                x_perturbed = x.copy()
                if x[variable] + step <= upper[variable]:
                    x_perturbed[variable] += step
                    residual_perturbed = residual_unknowns(x_perturbed)
                    jacobian[:, variable] = (residual_perturbed - residual_current) / step
                else:
                    x_perturbed[variable] -= step
                    residual_perturbed = residual_unknowns(x_perturbed)
                    jacobian[:, variable] = (residual_current - residual_perturbed) / step
                nfev += 1

            a = float(jacobian[0, 0])
            b = float(jacobian[0, 1])
            c = float(jacobian[1, 0])
            d = float(jacobian[1, 1])
            determinant = a * d - b * c
            if abs(determinant) <= 1e-300:
                break
            r0 = -float(residual_current[0])
            r1 = -float(residual_current[1])
            step = np.asarray([(d * r0 - b * r1) / determinant, (-c * r0 + a * r1) / determinant], dtype=np.float64)

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
    tie-line coordinate is estimated from the initial held-profile Stefan
    residual, so callers must provide an eta-capable interface equilibrium.
    Only planar Cartesian finite-difference meshes are supported.
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
        initial_eta_method: str = "stefan_cross_brentq",
        initial_eta_bracket=None,
        initial_eta_guess: float | None = None,
        initial_velocity_guess: float | None = None,
        initial_eta_root_xtol: float = 1e-12,
        initial_eta_root_rtol: float = 1e-12,
        initial_eta_root_maxiter: int = 100,
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
        self.initialEtaRootRtol = float(initial_eta_root_rtol)
        self.initialEtaRootMaxiter = int(initial_eta_root_maxiter)
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
        self._lastImplicitIterations = 0
        self._lastImplicitResidual = np.nan
        self._lastStepRetries = 0
        self._lastInterfaceCompositions = None
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
        if any(e in interstitials for e in self.allElements):
            raise ValueError("MovingBoundaryIllingworthTernaryFD1DModel supports only substitutional systems.")
        if len(self.phases) != 2:
            raise ValueError("MovingBoundaryIllingworthTernaryFD1DModel requires exactly two explicit phases.")
        if isinstance(getattr(self.mesh, "boundaryConditions", None), PeriodicBoundary1D):
            raise ValueError("Periodic boundary conditions are not supported.")
        if self.geometry != "planar":
            raise NotImplementedError("MovingBoundaryIllingworthTernaryFD1DModel currently implements only planar geometry.")
        if not np.isfinite(self.timeStep) or self.timeStep <= 0:
            raise ValueError("time_step must be a positive finite value.")
        if self.dtMode not in {"fixed", "semi_log"}:
            raise ValueError("dt_mode must be 'fixed' or 'semi_log'.")
        if self.dtMode == "semi_log" and ((self.semiLog_dt is None) or (self.semiLogT0 is None)):
            raise ValueError("semiLog_dt and semiLogT0 must be specified when dt_mode is 'semi_log'.")
        _validate_eta_bounds(self.interfaceEquilibrium)
        if self.initialEtaMethod not in {"stefan_cross_brentq", "instantaneous_balance"}:
            raise ValueError("initial_eta_method must be 'stefan_cross_brentq' or 'instantaneous_balance'.")
        if self.initialEtaRootXtol <= 0.0 or self.initialEtaRootRtol <= 0.0:
            raise ValueError("initial eta root tolerances must be positive.")
        if self.initialEtaRootMaxiter < 1:
            raise ValueError("initial_eta_root_maxiter must be at least 1.")
        if self.maxIterations < 2:
            raise ValueError("max_iterations must be at least 2.")
        if self.maxStepRetries < 1:
            raise ValueError("max_step_retries must be at least 1.")
        if not (0.0 < self.retryFactor < 1.0):
            raise ValueError("retry_factor must be between 0 and 1.")
        if self.phaseANodes is not None and self.phaseANodes < 3:
            raise ValueError("phase_a_nodes must be at least 3 when specified.")
        if self.phaseBNodes is not None and self.phaseBNodes < 3:
            raise ValueError("phase_b_nodes must be at least 3 when specified.")
        self.initialInterfacePosition = self._clipInterfacePosition(self.initialInterfacePosition, strict=True)

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
        self._lastImplicitIterations = 0
        self._lastImplicitResidual = np.nan
        self._lastStepRetries = 0
        self._lastInterfaceCompositions = None
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
        }
        if self.initialEtaMethod == "stefan_cross_brentq":
            self.initialEtaEstimate = estimate_initial_eta_from_stefan_residual(
                **estimate_kwargs,
                root_rtol=self.initialEtaRootRtol,
            )
        else:
            self.initialEtaEstimate = estimate_initial_eta_from_instantaneous_balance(
                **estimate_kwargs,
                eta_guess=self.initialEtaGuess,
                velocity_guess=self.initialVelocityGuess,
            )
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
        self._D_left = self._phase_diffusivity_matrix(c_left, self.phases[0], 0.0, s0)
        self._D_right = self._phase_diffusivity_matrix(c_right, self.phases[1], 0.0, s0)

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
        composition = np.asarray(composition, dtype=np.float64).reshape(2)
        T = np.asarray(self.temperatureParameters(np.asarray([[float(position)]], dtype=np.float64), float(time)), dtype=np.float64).reshape(-1)
        temperature = float(T[0])
        try:
            D = self.therm.getInterdiffusivity(composition, temperature, phase=phase, query_context="interface")
        except TypeError:
            D = self.therm.getInterdiffusivity(composition, temperature, phase=phase)
        D = np.asarray(D, dtype=np.float64).reshape(2, 2)
        return self._validate_diffusivity_matrix(D, phase)

    def _validate_diffusivity_matrix(self, D, phase):
        if D.shape != (2, 2) or not np.all(np.isfinite(D)):
            raise ValueError(f"Diffusivity for phase {phase} must be a finite 2x2 matrix.")
        trace = float(D[0, 0] + D[1, 1])
        determinant = float(D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0])
        discriminant = trace * trace - 4.0 * determinant
        scale = max(trace * trace, abs(determinant), 1.0)
        if discriminant < -1e-12 * scale:
            raise ValueError(f"Diffusivity for phase {phase} must have positive real eigenvalues.")
        root = float(np.sqrt(max(discriminant, 0.0)))
        eigenvalues = (0.5 * (trace + root), 0.5 * (trace - root))
        if eigenvalues[0] <= 0.0 or eigenvalues[1] <= 0.0:
            raise ValueError(f"Diffusivity for phase {phase} must have positive real eigenvalues.")
        if abs(determinant) <= 1e-300:
            raise ValueError(f"Diffusivity for phase {phase} is singular.")
        inverse = np.asarray([[D[1, 1], -D[0, 1]], [-D[1, 0], D[0, 0]]], dtype=np.float64) / determinant
        condition_estimate = np.max(np.sum(np.abs(D), axis=1)) * np.max(np.sum(np.abs(inverse), axis=1))
        if condition_estimate > 1e12:
            raise ValueError(f"Diffusivity for phase {phase} is too ill-conditioned for the ternary Illingworth solve.")
        return D.astype(np.float64)

    def setTimeInfo(self, currTime, simTime):
        """Stores solve-time bounds and prepares optional semi-log target times."""
        super().setTimeInfo(currTime, simTime)
        self._currdt = np.inf
        self._nearFinalNoop = False
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

    def _new_concentration_left_planar(self, p, s, future_s, dt, c_left, D_left):
        n = len(p)
        lower = np.zeros((n, 2, 2), dtype=np.float64)
        diagonal = np.zeros((n, 2, 2), dtype=np.float64)
        upper = np.zeros((n, 2, 2), dtype=np.float64)
        rhs = np.zeros((n, 2), dtype=np.float64)
        I = self._identity()
        tmpA = D_left * (float(dt) / float(future_s))
        tmpB = float(future_s) - float(s)
        u = self._u_grid

        if future_s >= s:
            diagonal[0] = -tmpA / u[1] - I * (future_s * u[1] / 2.0)
            upper[0] = tmpA / u[1] + I * (tmpB * u[1] / 2.0)
            rhs[0] = -p[0] * s * u[1] / 2.0
            for i in range(1, n - 1):
                left_diff = u[i] - u[i - 1]
                right_diff = u[i + 1] - u[i]
                left_sum = u[i] + u[i - 1]
                right_sum = u[i + 1] + u[i]
                cell_width = right_sum - left_sum
                lower[i] = tmpA / left_diff
                diagonal[i] = -tmpA * (1.0 / left_diff + 1.0 / right_diff)
                diagonal[i] += -I * (tmpB * left_sum / 2.0 + future_s * cell_width / 2.0)
                upper[i] = tmpA / right_diff + I * (tmpB * right_sum / 2.0)
                rhs[i] = -s * p[i] * cell_width / 2.0
        else:
            diagonal[0] = -tmpA / u[1] + I * (tmpB * u[1] / 2.0 - future_s * u[1] / 2.0)
            upper[0] = tmpA / u[1]
            rhs[0] = -p[0] * s * u[1] / 2.0
            for i in range(1, n - 1):
                left_diff = u[i] - u[i - 1]
                right_diff = u[i + 1] - u[i]
                left_sum = u[i] + u[i - 1]
                right_sum = u[i + 1] + u[i]
                cell_width = right_sum - left_sum
                lower[i] = tmpA / left_diff - I * (tmpB * left_sum / 2.0)
                diagonal[i] = -tmpA * (1.0 / left_diff + 1.0 / right_diff)
                diagonal[i] += I * (tmpB * right_sum / 2.0 - future_s * cell_width / 2.0)
                upper[i] = tmpA / right_diff
                rhs[i] = -s * p[i] * cell_width / 2.0

        diagonal[-1] = -I
        rhs[-1] = -np.asarray(c_left, dtype=np.float64)
        return solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)

    def _new_concentration_right_planar(self, q, s, future_s, dt, c_right, D_right):
        n = len(q)
        lower = np.zeros((n, 2, 2), dtype=np.float64)
        diagonal = np.zeros((n, 2, 2), dtype=np.float64)
        upper = np.zeros((n, 2, 2), dtype=np.float64)
        rhs = np.zeros((n, 2), dtype=np.float64)
        I = self._identity()
        tmpA = D_right * (float(dt) / (self._R - float(future_s)))
        tmpB = float(future_s) - float(s)
        span = self._R - float(future_s)
        v = self._v_grid

        diagonal[0] = -I
        rhs[0] = -np.asarray(c_right, dtype=np.float64)
        if future_s >= s:
            for i in range(1, n - 1):
                left_diff = v[i] - v[i - 1]
                right_diff = v[i + 1] - v[i]
                left_sum = v[i] + v[i - 1]
                right_sum = v[i + 1] + v[i]
                cell_width = right_sum - left_sum
                lower[i] = tmpA / left_diff
                diagonal[i] = -tmpA * (1.0 / right_diff + 1.0 / left_diff)
                diagonal[i] += -I * (tmpB * (1.0 - left_sum / 2.0) + span * cell_width / 2.0)
                upper[i] = tmpA / right_diff + I * (tmpB * (1.0 - right_sum / 2.0))
                rhs[i] = -(self._R - s) * q[i] * cell_width / 2.0

            tmp = v[-2]
            lower[-1] = tmpA / (1.0 - tmp)
            diagonal[-1] = -tmpA / (1.0 - tmp)
            diagonal[-1] += -I * (tmpB * (1.0 - (1.0 + tmp) / 2.0) + span * (1.0 - tmp) / 2.0)
            rhs[-1] = -q[-1] * (self._R - s) * (1.0 - tmp) / 2.0
        else:
            for i in range(1, n - 1):
                left_diff = v[i] - v[i - 1]
                right_diff = v[i + 1] - v[i]
                left_sum = v[i] + v[i - 1]
                right_sum = v[i + 1] + v[i]
                cell_width = right_sum - left_sum
                lower[i] = tmpA / left_diff - I * (tmpB * (1.0 - left_sum / 2.0))
                diagonal[i] = -tmpA * (1.0 / right_diff + 1.0 / left_diff)
                diagonal[i] += I * (tmpB * (1.0 - right_sum / 2.0) - span * cell_width / 2.0)
                upper[i] = tmpA / right_diff
                rhs[i] = -(self._R - s) * q[i] * cell_width / 2.0

            tmp = v[-2]
            lower[-1] = tmpA / (1.0 - tmp) - I * (tmpB * (1.0 - (1.0 + tmp) / 2.0))
            diagonal[-1] = -tmpA / (1.0 - tmp) - I * (span * (1.0 - tmp) / 2.0)
            rhs[-1] = -q[-1] * (self._R - s) * (1.0 - tmp) / 2.0

        return solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)

    def _interface_residual(self, p_future, q_future, s, old_s, future_s, dt, c_left, c_right, c_left_old, c_right_old, D_left, D_right):
        """
        Returns the two-component planar interface inventory residual.

        The original planar Illingworth residual assumes fixed phase-side
        interface compositions. In the ternary eta formulation those endpoint
        compositions may change during a step, and the trapezoidal transformed
        inventory stores those endpoint values in the interface-adjacent
        half-cells. The endpoint correction below accounts for that inventory
        change using the accepted old geometry ``s`` and the old discrete
        endpoint values from ``p[-1]`` and ``q[0]``.
        """
        velocity_probe = future_s - s
        if abs(velocity_probe) <= 1e-15:
            velocity_probe = s - old_s
        diff_l = _matvec_2x2(D_left, (c_left - p_future[-2]) / (1.0 - self._u_grid[-2]))
        diff_l = diff_l / future_s
        diff_r = _matvec_2x2(D_right, (q_future[1] - c_right) / self._v_grid[1])
        diff_r = diff_r / (self._R - future_s)
        rhs = (diff_r - diff_l) * dt

        if velocity_probe >= 0:
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

    def _solve_interface_planar(self, p, q, s, old_s, eta, dt):
        z_eps = max(self._R * 1e-14, 1e-14)
        eta_lower, eta_upper, eta_active = self._active_interface_variables()
        lower = np.asarray([z_eps, eta_lower] if eta_active else [z_eps], dtype=np.float64)
        upper = np.asarray([self._R - z_eps, eta_upper] if eta_active else [self._R - z_eps], dtype=np.float64)
        x = np.asarray([s, eta] if eta_active else [s], dtype=np.float64)
        x = np.clip(x, lower, upper)
        c_left_old = np.asarray(p[-1], dtype=np.float64).copy()
        c_right_old = np.asarray(q[0], dtype=np.float64).copy()
        best = None

        def evaluate(params):
            future_s = float(params[0])
            future_eta = float(params[1]) if eta_active else float(eta_lower)
            c_left, c_right = self._interface_compositions(future_eta)
            D_left = self._phase_diffusivity_matrix(c_left, self.phases[0], self.currentTime, future_s)
            D_right = self._phase_diffusivity_matrix(c_right, self.phases[1], self.currentTime, future_s)
            p_future = self._new_concentration_left_planar(p, s, future_s, dt, c_left, D_left)
            q_future = self._new_concentration_right_planar(q, s, future_s, dt, c_right, D_right)
            residual = self._interface_residual(p_future, q_future, s, old_s, future_s, dt, c_left, c_right, c_left_old, c_right_old, D_left, D_right)
            return residual, p_future, q_future, c_left, c_right, D_left, D_right

        for count in range(self.maxIterations):
            residual, p_future, q_future, c_left, c_right, D_left, D_right = evaluate(x)
            norm = float(np.max(np.abs(residual)))
            if best is None or norm < best[0]:
                best = (norm, x.copy(), p_future.copy(), q_future.copy(), c_left.copy(), c_right.copy(), D_left.copy(), D_right.copy())
            if norm <= self.residualTolerance:
                self._lastImplicitIterations = count + 1
                self._lastImplicitResidual = norm
                future_s = float(x[0])
                future_eta = float(x[1]) if eta_active else float(eta_lower)
                return p_future, q_future, future_s, future_eta, c_left, c_right, D_left, D_right

            jacobian = np.zeros((2, len(x)), dtype=np.float64)
            for variable in range(len(x)):
                step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(x[variable]))
                step = min(step, 0.25 * max(upper[variable] - lower[variable], 1e-15))
                x_perturbed = x.copy()
                if x[variable] + step <= upper[variable]:
                    x_perturbed[variable] += step
                    residual_perturbed = evaluate(x_perturbed)[0]
                    jacobian[:, variable] = (residual_perturbed - residual) / step
                else:
                    x_perturbed[variable] -= step
                    residual_perturbed = evaluate(x_perturbed)[0]
                    jacobian[:, variable] = (residual - residual_perturbed) / step

            step = self._least_squares_step_2xN(jacobian, residual)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = np.clip(x + scale * step, lower, upper)
                trial_residual = evaluate(trial)[0]
                trial_norm = float(np.max(np.abs(trial_residual)))
                if np.isfinite(trial_norm) and trial_norm < norm:
                    x = trial
                    accepted = True
                    break
            if not accepted:
                break

        if best is not None:
            self._lastImplicitIterations = self.maxIterations
            self._lastImplicitResidual = best[0]
        raise RuntimeError(
            "Ternary Illingworth interface solve failed to converge; "
            f"best residual was {np.inf if best is None else best[0]:.3e}."
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
                raise RuntimeError("Interface Jacobian is singular.")
            return np.asarray([-float(j[0] * residual[0] + j[1] * residual[1]) / denom], dtype=np.float64)

        a = float(jacobian[0, 0])
        b = float(jacobian[0, 1])
        c = float(jacobian[1, 0])
        d = float(jacobian[1, 1])
        determinant = a * d - b * c
        if abs(determinant) <= 1e-300:
            raise RuntimeError("Interface Jacobian is singular.")
        r0 = -float(residual[0])
        r1 = -float(residual[1])
        return np.asarray([(d * r0 - b * r1) / determinant, (-c * r0 + a * r1) / determinant], dtype=np.float64)

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

        last_error = None
        trial_dt = float(dt)
        for retry in range(self.maxStepRetries):
            try:
                p_new, q_new, s_new, eta_new, c_left_new, c_right_new, D_left, D_right = self._take_implicit_step_planar(
                    p,
                    q,
                    s,
                    self._s_old,
                    eta,
                    trial_dt,
                )
                self._currdt = trial_dt
                self._lastStepRetries = retry
                self._lastInterfaceCompositions = (c_left_new.copy(), c_right_new.copy())
                self._D_left = D_left
                self._D_right = D_right
                return [
                    (p_new - p) / trial_dt,
                    (q_new - q) / trial_dt,
                    (s_new - s) / trial_dt,
                    (eta_new - eta) / trial_dt,
                ]
            except (RuntimeError, ValueError, ZeroDivisionError) as exc:
                last_error = exc
                trial_dt *= self.retryFactor
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
