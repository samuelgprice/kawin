from types import SimpleNamespace

import numpy as np
import pytest

from kawin.diffusion import MovingBoundaryIllingworthTernaryThreePhaseFD1DModel
from kawin.diffusion import MovingBoundaryIllingworthTernaryFD1DModel
from kawin.diffusion import estimate_initial_eta_from_instantaneous_balance
from kawin.diffusion.MovingBoundaryIllingworthTernaryThreePhaseFDM import (
    _BULK_DIFFUSIVITY_IMPLICIT,
    _BULK_DIFFUSIVITY_LAGGED,
    _BULK_DIFFUSIVITY_PHASE_UNIFORM,
    _ThreePhaseCandidate,
)
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D import (
    integrate_planar_transformed_profile_sequence,
    reconstruct_planar_transformed_profile_sequence,
)


class _ConstantPairEquilibrium:
    eta_bounds = (0.0, 1.0)

    def __init__(self, left, right):
        self.left = np.asarray(left, dtype=np.float64)
        self.right = np.asarray(right, dtype=np.float64)

    def interface_compositions(self, eta):
        return self.left.copy(), self.right.copy()


class _LinearPairEquilibrium:
    eta_bounds = (0.0, 1.0)

    def __init__(self, left_base, left_slope, right_base, right_slope):
        self.left_base = np.asarray(left_base, dtype=np.float64)
        self.left_slope = np.asarray(left_slope, dtype=np.float64)
        self.right_base = np.asarray(right_base, dtype=np.float64)
        self.right_slope = np.asarray(right_slope, dtype=np.float64)

    def interface_compositions(self, eta):
        eta = float(eta)
        return self.left_base + eta * self.left_slope, self.right_base + eta * self.right_slope


class _ThreePhaseStepProfile:
    def __init__(self, interfaces, values):
        self.interfaces = tuple(float(v) for v in interfaces)
        self.values = tuple(np.asarray(v, dtype=np.float64) for v in values)

    def __call__(self, z):
        x = np.asarray(z, dtype=np.float64).reshape((-1, 1))[:, 0]
        out = np.empty((len(x), 2), dtype=np.float64)
        out[x < self.interfaces[0]] = self.values[0]
        middle = (x >= self.interfaces[0]) & (x < self.interfaces[1])
        out[middle] = self.values[1]
        out[x >= self.interfaces[1]] = self.values[2]
        return out


class _RecordingThreePhaseThermodynamics:
    def __init__(self):
        self.calls = []

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        self.calls.append(
            {
                "phase": phase,
                "composition": values.copy(),
                "query_context": kwargs.get("query_context"),
            }
        )
        if values.ndim == 2:
            return np.broadcast_to(np.eye(2, dtype=np.float64), (values.shape[0], 2, 2)).copy()
        return np.eye(2, dtype=np.float64)


def _make_stationary_model(*, mode=_BULK_DIFFUSIVITY_PHASE_UNIFORM, record=True, interfaces=(0.35, 0.7), **model_kwargs):
    phase_values = (
        np.asarray([0.20, 0.10], dtype=np.float64),
        np.asarray([0.30, 0.15], dtype=np.float64),
        np.asarray([0.25, 0.20], dtype=np.float64),
    )
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 31)
    mesh.setResponseProfile(ProfileBuilder([(_ThreePhaseStepProfile(interfaces, phase_values), ["X", "Y"])]))
    therm = _RecordingThreePhaseThermodynamics()
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["A", "B", "C"],
        thermodynamics=therm,
        temperature=1000.0,
        interfacePositions=interfaces,
        interface_equilibria=(
            _ConstantPairEquilibrium(phase_values[0], phase_values[1]),
            _ConstantPairEquilibrium(phase_values[1], phase_values[2]),
        ),
        initial_eta_guess=(0.25, 0.75),
        bulk_diffusivity_mode=mode,
        phase_nodes=(5, 5, 5),
        time_step=1e-4,
        record=record,
        tolerance=1e-10,
        residual_tolerance=1e-10,
        **model_kwargs,
    ), therm


def _make_eta_varying_three_phase_model(*, mode=_BULK_DIFFUSIVITY_PHASE_UNIFORM, record=True):
    target_etas = np.asarray([0.2, 0.8], dtype=np.float64)
    eq_ab = _LinearPairEquilibrium([0.20, 0.10], [0.10, 0.0], [0.30, 0.15], [0.10, 0.0])
    eq_bc = _LinearPairEquilibrium([0.24, 0.15], [0.10, 0.0], [0.18, 0.20], [0.10, 0.0])
    interfaces = (0.35, 0.7)
    phase_values = (
        eq_ab.interface_compositions(target_etas[0])[0],
        eq_ab.interface_compositions(target_etas[0])[1],
        eq_bc.interface_compositions(target_etas[1])[1],
    )
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 31)
    mesh.setResponseProfile(ProfileBuilder([(_ThreePhaseStepProfile(interfaces, phase_values), ["X", "Y"])]))
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["A", "B", "C"],
        thermodynamics=_RecordingThreePhaseThermodynamics(),
        temperature=1000.0,
        interfacePositions=interfaces,
        interface_equilibria=(eq_ab, eq_bc),
        initial_eta_guess=(0.5, 0.5),
        bulk_diffusivity_mode=mode,
        bulk_picard_max_iterations=20,
        phase_nodes=(7, 7, 7),
        time_step=1e-5,
        record=record,
        tolerance=1e-10,
        residual_tolerance=1e-10,
        max_iterations=50,
    )


def _flatten_adjacent(adjacent):
    return np.concatenate([np.asarray(values, dtype=np.float64).reshape(2) for values in adjacent])


def _unflatten_adjacent(values):
    values = np.asarray(values, dtype=np.float64).reshape(4, 2)
    return tuple(values[i].copy() for i in range(4))


def _simplex_admissible(values):
    values = np.asarray(values, dtype=np.float64)
    return bool(np.all(values > 0.0) and np.all(np.sum(values, axis=-1) < 1.0))


def _construct_balanced_initial_adjacent(model, interfaces, target_etas, target_velocities, base_adjacent=None, locked_components=()):
    interface_compositions = model._interface_compositions(target_etas)
    if base_adjacent is None:
        c_a_ab, c_b_ab = interface_compositions[0]
        c_b_bc, c_c_bc = interface_compositions[1]
        base_adjacent = (c_a_ab, c_b_ab, c_b_bc, c_c_bc)

    base = _flatten_adjacent(base_adjacent)
    locked = set(locked_components)
    free = np.asarray([index for index in range(base.size) if index not in locked], dtype=int)

    def residual_from_flat(flat):
        adjacent = _unflatten_adjacent(flat)
        return model._initial_discrete_interface_residuals(interfaces, interface_compositions, adjacent, target_velocities)

    residual0 = residual_from_flat(base)
    jacobian = np.zeros((4, len(free)), dtype=np.float64)
    for column, index in enumerate(free):
        perturbed = base.copy()
        perturbed[index] += 1.0e-7
        jacobian[:, column] = (residual_from_flat(perturbed) - residual0) / 1.0e-7

    correction = np.linalg.lstsq(jacobian, -residual0, rcond=None)[0]
    balanced = base.copy()
    balanced[free] += correction
    adjacent = _unflatten_adjacent(balanced)
    final_residual = residual_from_flat(balanced)

    assert _simplex_admissible(balanced.reshape(4, 2))
    assert np.allclose(final_residual, 0.0, rtol=0.0, atol=2.0e-11)
    return adjacent


def _estimate_from_controlled_adjacent(model, interfaces, adjacent, eta_guess, velocity_guess):
    model.initialEtaGuess = np.asarray(eta_guess, dtype=np.float64)
    model.initialVelocityGuess = None if velocity_guess is None else np.asarray(velocity_guess, dtype=np.float64)
    model._initial_adjacent_compositions = lambda composition, queried_interfaces: adjacent
    returned_etas = model._estimate_initial_etas(model.data.currentY, interfaces)
    assert np.allclose(returned_etas, model.initialEtaEstimate.etas, rtol=0.0, atol=0.0)
    return model.initialEtaEstimate


def _construct_two_phase_known_profile(equilibrium, target_eta, target_velocity, interface_position, domain_length, u_grid, v_grid):
    c_left, c_right = equilibrium.interface_compositions(target_eta)
    u_adjacent = float(u_grid[-2])
    v_adjacent = float(v_grid[1])
    base = np.concatenate((c_left, c_right))

    def residual_from_flat(flat):
        p_adjacent = flat[:2]
        q_adjacent = flat[2:]
        g_left = (c_left - p_adjacent) / (interface_position * (1.0 - u_adjacent))
        g_right = (q_adjacent - c_right) / ((domain_length - interface_position) * v_adjacent)
        flux_delta = g_right - g_left
        if target_velocity > 0.0:
            swept = c_left - q_adjacent * (1.0 - v_adjacent / 2.0) - c_right * v_adjacent / 2.0
        else:
            swept = p_adjacent * ((1.0 + u_adjacent) / 2.0) + c_left * ((1.0 - u_adjacent) / 2.0) - c_right
        return target_velocity * swept - flux_delta

    balanced = base.copy()
    for _ in range(2):
        residual0 = residual_from_flat(balanced)
        jacobian = np.zeros((2, 4), dtype=np.float64)
        for index in range(4):
            perturbed = balanced.copy()
            perturbed[index] += 1.0e-7
            jacobian[:, index] = (residual_from_flat(perturbed) - residual0) / 1.0e-7
        balanced += np.linalg.lstsq(jacobian, -residual0, rcond=None)[0]
    assert _simplex_admissible(balanced.reshape(2, 2))
    assert np.allclose(residual_from_flat(balanced), 0.0, rtol=0.0, atol=2.0e-13)

    p_adjacent = balanced[:2]
    q_adjacent = balanced[2:]
    z_left = interface_position * u_adjacent
    z_right = interface_position + (domain_length - interface_position) * v_adjacent
    z = np.asarray([0.0, z_left, interface_position, z_right, domain_length], dtype=np.float64)
    composition = np.vstack((p_adjacent, p_adjacent, c_left, q_adjacent, q_adjacent))
    return z, composition, p_adjacent, q_adjacent


def _nonuniform_moving_candidate_state(model):
    state = model.getCurrentX()
    state[0][0:3] -= [0.03, 0.01]
    state[0][3:6] += [0.02, 0.005]
    state[1][1:3] += [0.04, 0.01]
    state[1][3:6] -= [0.03, 0.015]
    state[2][1:4] += [0.03, -0.005]
    state[2][4:] -= [0.02, 0.0]
    return state


def _interval_fv_residual(old_profile, new_profile, grid, old_bounds, new_bounds, D_faces, dt, left_value=None, right_value=None):
    old_profile = np.asarray(old_profile, dtype=np.float64)
    new_profile = np.asarray(new_profile, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    D_faces = np.asarray(D_faces, dtype=np.float64)
    old_left, old_right = tuple(float(v) for v in old_bounds)
    new_left, new_right = tuple(float(v) for v in new_bounds)
    old_length = old_right - old_left
    new_length = new_right - new_left
    n = len(grid)
    face_xi = np.empty(n + 1, dtype=np.float64)
    face_xi[0] = 0.0
    face_xi[-1] = 1.0
    face_xi[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    old_widths = old_length * (face_xi[1:] - face_xi[:-1])
    new_widths = new_length * (face_xi[1:] - face_xi[:-1])
    internal_H = []
    for face in range(n - 1):
        xi = 0.5 * (grid[face] + grid[face + 1])
        delta_x = (1.0 - xi) * (new_left - old_left) + xi * (new_right - old_right)
        diffusive = float(dt) * np.matmul(D_faces[face], new_profile[face + 1] - new_profile[face]) / (new_length * (grid[face + 1] - grid[face]))
        if delta_x > 0.0:
            ale = delta_x * new_profile[face + 1]
        elif delta_x < 0.0:
            ale = delta_x * new_profile[face]
        else:
            ale = np.zeros(2, dtype=np.float64)
        internal_H.append(ale + diffusive)
    residual = np.empty_like(new_profile)
    for i in range(n):
        H_left = np.zeros(2, dtype=np.float64) if i == 0 else internal_H[i - 1]
        H_right = np.zeros(2, dtype=np.float64) if i == n - 1 else internal_H[i]
        residual[i] = new_widths[i] * new_profile[i] - old_widths[i] * old_profile[i] - (H_right - H_left)
    if left_value is not None:
        residual[0] = new_profile[0] - np.asarray(left_value, dtype=np.float64)
    if right_value is not None:
        residual[-1] = new_profile[-1] - np.asarray(right_value, dtype=np.float64)
    return residual


def _interval_fv_residual_with_uniform_ale_donor(old_profile, new_profile, grid, old_bounds, new_bounds, D_faces, dt, donor):
    old_profile = np.asarray(old_profile, dtype=np.float64)
    new_profile = np.asarray(new_profile, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    old_left, old_right = tuple(float(v) for v in old_bounds)
    new_left, new_right = tuple(float(v) for v in new_bounds)
    old_length = old_right - old_left
    new_length = new_right - new_left
    n = len(grid)
    face_xi = np.empty(n + 1, dtype=np.float64)
    face_xi[0] = 0.0
    face_xi[-1] = 1.0
    face_xi[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    old_widths = old_length * (face_xi[1:] - face_xi[:-1])
    new_widths = new_length * (face_xi[1:] - face_xi[:-1])
    internal_H = []
    for face in range(n - 1):
        xi = 0.5 * (grid[face] + grid[face + 1])
        delta_x = (1.0 - xi) * (new_left - old_left) + xi * (new_right - old_right)
        donor_value = new_profile[face + 1] if donor == "right" else new_profile[face]
        diffusive = float(dt) * np.matmul(D_faces[face], new_profile[face + 1] - new_profile[face]) / (new_length * (grid[face + 1] - grid[face]))
        internal_H.append(delta_x * donor_value + diffusive)
    residual = []
    for i in range(1, n - 1):
        residual.append(new_widths[i] * new_profile[i] - old_widths[i] * old_profile[i] - (internal_H[i] - internal_H[i - 1]))
    return np.asarray(residual, dtype=np.float64)


def _manual_initial_ale_residual(model, interfaces, interface_compositions, adjacent, velocities):
    s_ab, s_bc = np.asarray(interfaces, dtype=np.float64).reshape(2)
    v_ab, v_bc = np.asarray(velocities, dtype=np.float64).reshape(2)
    p_a, p_b_left, p_b_right, p_c = adjacent
    c_a_ab, c_b_ab = interface_compositions[0]
    c_b_bc, c_c_bc = interface_compositions[1]
    u_a, u_b, u_c = model._grids
    xi_a_right = 0.5 * (u_a[-2] + 1.0)
    xi_b_left = 0.5 * u_b[1]
    xi_b_right = 0.5 * (u_b[-2] + 1.0)
    xi_c_left = 0.5 * u_c[1]
    w_a_right = xi_a_right * v_ab
    w_b_left = (1.0 - xi_b_left) * v_ab + xi_b_left * v_bc
    w_b_right = (1.0 - xi_b_right) * v_ab + xi_b_right * v_bc
    w_c_left = (1.0 - xi_c_left) * v_bc

    def donor_term(w, left, right):
        if w > 0.0:
            return w * right
        if w < 0.0:
            return w * left
        return np.zeros(2, dtype=np.float64)

    length_rate_b = v_bc - v_ab
    endpoint_ab = 0.5 * (1.0 - u_a[-2]) * v_ab * c_a_ab + 0.5 * u_b[1] * length_rate_b * c_b_ab
    endpoint_bc = 0.5 * (1.0 - u_b[-2]) * length_rate_b * c_b_bc - 0.5 * u_c[1] * v_bc * c_c_bc
    residual_ab = endpoint_ab + donor_term(w_a_right, p_a, c_a_ab) - donor_term(w_b_left, c_b_ab, p_b_left)
    residual_bc = endpoint_bc + donor_term(w_b_right, p_b_right, c_b_bc) - donor_term(w_c_left, c_c_bc, p_c)
    return np.concatenate((residual_ab, residual_bc))


def test_three_phase_sequence_inventory_and_reconstruction_helpers():
    grids = tuple(np.linspace(0.0, 1.0, 5) for _ in range(3))
    profiles = (
        np.full((5, 2), [0.20, 0.10], dtype=np.float64),
        np.full((5, 2), [0.30, 0.15], dtype=np.float64),
        np.full((5, 2), [0.25, 0.20], dtype=np.float64),
    )
    interfaces = np.asarray([0.25, 0.75], dtype=np.float64)

    inventory = integrate_planar_transformed_profile_sequence(profiles, interfaces, 1.0, grids)
    reconstructed = reconstruct_planar_transformed_profile_sequence(
        np.asarray([0.0, 0.2, 0.25, 0.5, 0.75, 0.9, 1.0]),
        profiles,
        interfaces,
        1.0,
        grids,
    )

    assert np.allclose(inventory, 0.25 * profiles[0][0] + 0.5 * profiles[1][0] + 0.25 * profiles[2][0], rtol=0.0, atol=1.0e-15)
    assert np.allclose(reconstructed[0], profiles[0][0], rtol=0.0, atol=1.0e-15)
    assert np.allclose(reconstructed[2], profiles[0][0], rtol=0.0, atol=1.0e-15)
    assert np.allclose(reconstructed[3], profiles[1][0], rtol=0.0, atol=1.0e-15)
    assert np.allclose(reconstructed[-1], profiles[2][0], rtol=0.0, atol=1.0e-15)


def test_three_phase_interval_translation_satisfies_ale_finite_volume_equation():
    model, _ = _make_stationary_model(record=False)
    model.setup()
    grid = np.asarray([0.0, 0.15, 0.45, 0.75, 1.0], dtype=np.float64)
    old_profile = np.asarray(
        [
            [0.18, 0.08],
            [0.24, 0.11],
            [0.31, 0.09],
            [0.28, 0.16],
            [0.22, 0.14],
        ],
        dtype=np.float64,
    )
    old_bounds = np.asarray([0.20, 0.70], dtype=np.float64)
    new_bounds = np.asarray([0.25, 0.75], dtype=np.float64)
    D_faces = np.zeros((len(grid) - 1, 2, 2), dtype=np.float64)
    model._currdt = 0.01

    result = model._solve_interval_planar(
        old_profile,
        grid,
        old_bounds,
        new_bounds,
        1,
        old_profile[0],
        old_profile[-1],
        D_faces,
        validate_diffusivity=False,
    )
    residual = _interval_fv_residual(old_profile, result.profile, grid, old_bounds, new_bounds, D_faces, model._currdt, old_profile[0], old_profile[-1])

    assert np.max(np.abs(result.profile[1:-1] - old_profile[1:-1])) > 1.0e-4
    assert np.allclose(residual, 0.0, rtol=0.0, atol=2e-15)


def test_three_phase_middle_interval_uses_face_local_ale_upwinding():
    model, _ = _make_stationary_model(record=False)
    model.setup()
    grid = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    old_profile = np.asarray(
        [
            [0.18, 0.08],
            [0.23, 0.10],
            [0.32, 0.12],
            [0.27, 0.17],
            [0.21, 0.13],
            [0.16, 0.09],
        ],
        dtype=np.float64,
    )
    old_bounds = np.asarray([0.30, 0.80], dtype=np.float64)
    new_bounds = np.asarray([0.40, 0.70], dtype=np.float64)
    D_faces = np.zeros((len(grid) - 1, 2, 2), dtype=np.float64)
    model._currdt = 0.01
    displacements = model._internal_face_displacements(grid, old_bounds, new_bounds)

    result = model._solve_interval_planar(
        old_profile,
        grid,
        old_bounds,
        new_bounds,
        1,
        old_profile[0],
        old_profile[-1],
        D_faces,
        validate_diffusivity=False,
    )
    residual = _interval_fv_residual(old_profile, result.profile, grid, old_bounds, new_bounds, D_faces, model._currdt, old_profile[0], old_profile[-1])
    all_right_residual = _interval_fv_residual_with_uniform_ale_donor(old_profile, result.profile, grid, old_bounds, new_bounds, D_faces, model._currdt, "right")
    all_left_residual = _interval_fv_residual_with_uniform_ale_donor(old_profile, result.profile, grid, old_bounds, new_bounds, D_faces, model._currdt, "left")

    assert np.any(displacements > 0.0)
    assert np.any(displacements < 0.0)
    assert np.allclose(residual, 0.0, rtol=0.0, atol=2e-15)
    assert np.max(np.abs(all_right_residual)) > 1e-4
    assert np.max(np.abs(all_left_residual)) > 1e-4


@pytest.mark.parametrize(
    "interval, old_s, new_s",
    [
        ("left", 0.45, 0.52),
        ("left", 0.45, 0.38),
        ("right", 0.45, 0.52),
        ("right", 0.45, 0.38),
    ],
)
def test_three_phase_outer_interval_reduces_to_two_phase_moving_grid_solve(interval, old_s, new_s):
    c_left = np.asarray([0.26, 0.11], dtype=np.float64)
    c_right = np.asarray([0.34, 0.16], dtype=np.float64)
    grid = np.asarray([0.0, 0.2, 0.5, 0.75, 1.0], dtype=np.float64)
    if interval == "left":
        profile = np.asarray(
            [
                [0.18, 0.06],
                [0.21, 0.09],
                [0.25, 0.10],
                [0.28, 0.12],
                c_left,
            ],
            dtype=np.float64,
        )
    else:
        profile = np.asarray(
            [
                c_right,
                [0.35, 0.18],
                [0.32, 0.17],
                [0.30, 0.15],
                [0.28, 0.14],
            ],
            dtype=np.float64,
        )
    D = np.asarray([[1.3, 0.08], [0.04, 0.9]], dtype=np.float64)
    dt = 0.015
    three_phase, _ = _make_stationary_model(record=False)
    three_phase.setup()
    three_phase._currdt = dt
    two_mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    two_mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(old_s, c_left, c_right), ["X", "Y"])]))
    two_phase = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=two_mesh,
        elements=["Z", "X", "Y"],
        phases=["A", "B"],
        thermodynamics=_RecordingThreePhaseThermodynamics(),
        temperature=1000.0,
        interfacePosition=old_s,
        interface_equilibrium=_ConstantPairEquilibrium(c_left, c_right),
        initial_eta_guess=0.0,
        transformed_u_grid=grid,
        transformed_v_grid=grid,
        time_step=dt,
        record=False,
    )
    two_phase.setup()

    branch = "positive" if new_s > old_s else "negative"
    if interval == "left":
        generic = three_phase._solve_interval_planar(profile, grid, [0.0, old_s], [0.0, new_s], 0, None, c_left, D)
        reference = two_phase._solve_concentration_left_planar(profile, old_s, new_s, dt, c_left, D, branch)
    else:
        generic = three_phase._solve_interval_planar(profile, grid, [old_s, 1.0], [new_s, 1.0], 2, c_right, None, D)
        reference = two_phase._solve_concentration_right_planar(profile, old_s, new_s, dt, c_right, D, branch)

    assert np.allclose(generic.profile, reference.profile, rtol=1.0e-13, atol=1e-13)


def test_three_phase_constant_profile_preserved_by_geometric_motion_with_zero_diffusion():
    model, _ = _make_stationary_model(record=False)
    model.setup()
    grid = np.asarray([0.0, 0.17, 0.48, 0.79, 1.0], dtype=np.float64)
    value = np.asarray([0.27, 0.13], dtype=np.float64)
    profile = np.broadcast_to(value, (len(grid), 2)).copy()
    D_faces = np.zeros((len(grid) - 1, 2, 2), dtype=np.float64)
    model._currdt = 0.025

    result = model._solve_interval_planar(
        profile,
        grid,
        old_bounds=[0.24, 0.82],
        new_bounds=[0.36, 0.68],
        phase_index=1,
        left_value=value,
        right_value=value,
        D_faces=D_faces,
        validate_diffusivity=False,
    )

    assert np.allclose(result.profile, profile, rtol=0.0, atol=2.0e-16)


def test_three_phase_model_rejects_non_three_phase_input():
    model, _ = _make_stationary_model()
    with pytest.raises(ValueError, match="exactly three"):
        MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
            mesh=model.mesh,
            elements=["Z", "X", "Y"],
            phases=["A", "B"],
            thermodynamics=_RecordingThreePhaseThermodynamics(),
            temperature=1000.0,
            interfacePositions=(0.35, 0.7),
            interface_equilibria=model.interfaceEquilibria,
            time_step=1e-4,
        )


def test_three_phase_model_rejects_interface_collision_on_setup():
    model, _ = _make_stationary_model(interfaces=(0.5, 0.5 + 1e-15))
    with pytest.raises(ValueError, match="nonzero middle phase"):
        model.setup()


def test_three_phase_stationary_profile_stays_stationary_and_records_histories():
    model, _ = _make_stationary_model(record=True)

    model.solve(1e-4)

    assert np.allclose(model.getInterfacePositions(), [0.35, 0.7], rtol=0.0, atol=1e-12)
    assert np.allclose(model.getInterfaceEtas(), [0.25, 0.75], rtol=0.0, atol=1e-12)
    assert np.allclose(model.checkConservation(1e-10), [0.0, 0.0], rtol=0.0, atol=1e-12)
    profiles = model.getTransformedState()
    assert len(profiles) == 3
    assert model.interfaceData._y[: model.interfaceData.N + 1].shape[1] == 2
    assert model.etaData._y[: model.etaData.N + 1].shape[1] == 2


def test_three_phase_single_thin_phase_extra_retry_stops_after_converged_step(monkeypatch):
    model, _ = _make_stationary_model(
        record=True,
        max_step_retries=2,
        retry_factor=0.5,
        terminal_thin_phase_width=1e-9,
        terminal_thin_phase_extra_retries=3,
        terminal_thin_phase_policy="continue",
    )
    model.setup()
    x_curr = model.getCurrentX()
    x_curr[3] = np.asarray([5e-10, 0.7], dtype=np.float64)
    model._interfaces_curr = x_curr[3].copy()
    calls = []

    def fake_solve(profiles, interfaces, etas, dt):
        calls.append(float(dt))
        if len(calls) < 4:
            raise RuntimeError("forced retry")
        return _ThreePhaseCandidate(
            x_hat=np.zeros(4, dtype=np.float64),
            profiles=tuple(profile.copy() for profile in profiles),
            interfaces=np.asarray(interfaces, dtype=np.float64).copy(),
            etas=np.asarray(etas, dtype=np.float64).copy(),
            interface_compositions=model._interface_compositions(etas),
            residual=np.zeros(4, dtype=np.float64),
            scaled_residual=np.zeros(4, dtype=np.float64),
            scaled_norm=0.0,
            physical_norm=0.0,
            bulk_results=(None, None, None),
        )

    monkeypatch.setattr(model, "_solve_interface_planar", fake_solve)

    with pytest.warns(RuntimeWarning, match="below terminal_thin_phase_width"):
        dXdt = model.getdXdt(0.0, x_curr)

    dt = model.getDt(dXdt)
    x_next = [np.asarray(value) + np.asarray(derivative) * dt for value, derivative in zip(x_curr, dXdt)]
    _, stop = model.postProcess(dt, x_next)

    assert stop is True
    assert np.allclose(calls, [1e-4, 5e-5, 2.5e-5, 1.25e-5], rtol=0.0, atol=1e-18)
    assert model.currentTime == pytest.approx(1.25e-5)
    assert model.finalTime == pytest.approx(1.25e-5)
    assert model._terminalThinPhaseInfo["phase"] == "A"
    assert model._terminalThinPhaseInfo["width"] == pytest.approx(5e-10)


def test_three_phase_terminal_retry_requires_exactly_one_thin_phase(monkeypatch):
    model, _ = _make_stationary_model(
        record=False,
        max_step_retries=2,
        retry_factor=0.5,
        terminal_thin_phase_width=1e-9,
        terminal_thin_phase_extra_retries=3,
        terminal_thin_phase_policy="continue",
    )
    model.setup()
    x_curr = model.getCurrentX()
    x_curr[3] = np.asarray([5e-10, 1.0 - 5e-10], dtype=np.float64)
    calls = []

    def fake_solve(profiles, interfaces, etas, dt):
        calls.append(float(dt))
        raise RuntimeError("forced retry")

    monkeypatch.setattr(model, "_solve_interface_planar", fake_solve)

    with pytest.raises(RuntimeError, match="Three-phase Illingworth step failed after timestep retries"):
        model.getdXdt(0.0, x_curr)

    assert np.allclose(calls, [1e-4, 5e-5], rtol=0.0, atol=1e-18)


def test_three_phase_prompt_policy_requires_interactive_stdin(monkeypatch):
    model, _ = _make_stationary_model(
        record=False,
        max_step_retries=2,
        retry_factor=0.5,
        terminal_thin_phase_width=1e-9,
        terminal_thin_phase_extra_retries=3,
        terminal_thin_phase_policy="prompt",
    )
    model.setup()
    x_curr = model.getCurrentX()
    x_curr[3] = np.asarray([5e-10, 0.7], dtype=np.float64)

    class NonInteractiveStdin:
        def isatty(self):
            return False

    def fake_solve(profiles, interfaces, etas, dt):
        raise RuntimeError("forced retry")

    monkeypatch.setattr(model, "_solve_interface_planar", fake_solve)
    monkeypatch.setattr("sys.stdin", NonInteractiveStdin())

    with pytest.warns(RuntimeWarning, match="below terminal_thin_phase_width"):
        with pytest.raises(RuntimeError, match="stdin is not interactive"):
            model.getdXdt(0.0, x_curr)


def test_three_phase_initial_etas_are_estimated_from_instantaneous_balance():
    target_etas = np.asarray([0.2, 0.8], dtype=np.float64)
    eq_ab = _LinearPairEquilibrium([0.20, 0.10], [0.10, 0.0], [0.30, 0.15], [0.10, 0.0])
    eq_bc = _LinearPairEquilibrium([0.24, 0.15], [0.10, 0.0], [0.18, 0.20], [0.10, 0.0])
    phase_values = (
        eq_ab.interface_compositions(target_etas[0])[0],
        eq_ab.interface_compositions(target_etas[0])[1],
        eq_bc.interface_compositions(target_etas[1])[1],
    )
    interfaces = (0.35, 0.7)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 31)
    mesh.setResponseProfile(ProfileBuilder([(_ThreePhaseStepProfile(interfaces, phase_values), ["X", "Y"])]))
    model = MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["A", "B", "C"],
        thermodynamics=_RecordingThreePhaseThermodynamics(),
        temperature=1000.0,
        interfacePositions=interfaces,
        interface_equilibria=(eq_ab, eq_bc),
        initial_eta_guess=(0.5, 0.5),
        bulk_diffusivity_mode=_BULK_DIFFUSIVITY_PHASE_UNIFORM,
        phase_nodes=(5, 5, 5),
        time_step=1e-4,
        record=False,
        tolerance=1e-10,
        residual_tolerance=1e-10,
    )

    model.setup()

    assert np.allclose(model.getInterfaceEtas(), target_etas, rtol=0.0, atol=1e-8)
    assert np.allclose(model.initialEtaEstimate.etas, target_etas, rtol=0.0, atol=1e-8)
    assert model.initialEtaEstimate.residual_norm <= model.initialEtaRootXtol


@pytest.mark.parametrize(
    "target_velocities, velocity_guess",
    [
        (np.asarray([0.08, 0.05], dtype=np.float64), np.asarray([0.02, 0.01], dtype=np.float64)),
        (np.asarray([-0.08, -0.05], dtype=np.float64), np.asarray([-0.02, -0.01], dtype=np.float64)),
        (np.asarray([0.08, -0.05], dtype=np.float64), np.asarray([0.02, -0.01], dtype=np.float64)),
        (np.asarray([-0.08, 0.05], dtype=np.float64), np.asarray([-0.02, 0.01], dtype=np.float64)),
    ],
    ids=["both-positive", "both-negative", "ab-positive-bc-negative", "ab-negative-bc-positive"],
)
@pytest.mark.parametrize(
    "use_automatic_velocity_guess",
    [False, True],
    ids=["explicit-velocity-guess", "automatic-velocity-guess"],
)
def test_three_phase_initial_eta_solver_converges_to_moving_known_solution(use_automatic_velocity_guess, target_velocities, velocity_guess):
    model = _make_eta_varying_three_phase_model(record=False)
    model.setup()
    interfaces = np.asarray([0.35, 0.7], dtype=np.float64)
    target_etas = np.asarray([0.2, 0.8], dtype=np.float64)
    adjacent = _construct_balanced_initial_adjacent(model, interfaces, target_etas, target_velocities)

    estimate = _estimate_from_controlled_adjacent(
        model,
        interfaces,
        adjacent,
        eta_guess=np.asarray([0.35, 0.65], dtype=np.float64),
        velocity_guess=None if use_automatic_velocity_guess else velocity_guess,
    )

    assert estimate.converged
    assert estimate.residual_norm <= model.initialEtaRootXtol
    assert np.allclose(estimate.etas, target_etas, rtol=0.0, atol=5.0e-10)
    assert np.allclose(estimate.velocities, target_velocities, rtol=2.0e-10, atol=5.0e-12)


def test_three_phase_initial_eta_solver_crosses_ale_donor_switch():
    model = _make_eta_varying_three_phase_model(record=False)
    model.setup()
    interfaces = np.asarray([0.35, 0.7], dtype=np.float64)
    target_etas = np.asarray([0.2, 0.8], dtype=np.float64)
    target_velocities = np.asarray([0.08, -0.05], dtype=np.float64)
    initial_velocity_guess = np.asarray([-0.02, -0.05], dtype=np.float64)
    adjacent = _construct_balanced_initial_adjacent(model, interfaces, target_etas, target_velocities)
    xi_a_right = 0.5 * (model._grids[0][-2] + 1.0)
    xi_b_left = 0.5 * model._grids[1][1]

    estimate = _estimate_from_controlled_adjacent(
        model,
        interfaces,
        adjacent,
        eta_guess=np.asarray([0.35, 0.65], dtype=np.float64),
        velocity_guess=initial_velocity_guess,
    )

    w_b_left_initial = (1.0 - xi_b_left) * initial_velocity_guess[0] + xi_b_left * initial_velocity_guess[1]
    w_b_left_final = (1.0 - xi_b_left) * estimate.velocities[0] + xi_b_left * estimate.velocities[1]
    assert xi_a_right * initial_velocity_guess[0] < 0.0
    assert xi_a_right * estimate.velocities[0] > 0.0
    assert w_b_left_initial < 0.0
    assert w_b_left_final > 0.0
    assert estimate.converged
    assert estimate.residual_norm <= model.initialEtaRootXtol
    assert np.allclose(estimate.etas, target_etas, rtol=0.0, atol=5.0e-10)
    assert np.allclose(estimate.velocities, target_velocities, rtol=2.0e-10, atol=5.0e-12)


def test_three_phase_initial_residual_is_finite_step_residual_limit():
    model = _make_eta_varying_three_phase_model(record=False)
    model.setup()
    old_interfaces = np.asarray([0.35, 0.7], dtype=np.float64)
    velocities = np.asarray([1.4, -0.9], dtype=np.float64)
    etas = np.asarray([0.63, 0.27], dtype=np.float64)
    interface_compositions = model._interface_compositions(etas)
    c_a_ab, c_b_ab = interface_compositions[0]
    c_b_bc, c_c_bc = interface_compositions[1]
    old_profiles = (
        np.linspace(np.asarray([0.18, 0.075], dtype=np.float64), c_a_ab, len(model._grids[0])),
        np.linspace(c_b_ab, c_b_bc, len(model._grids[1])),
        np.linspace(c_c_bc, np.asarray([0.28, 0.18], dtype=np.float64), len(model._grids[2])),
    )
    old_profiles[0][1:-1] += np.asarray([0.006, -0.003], dtype=np.float64)
    old_profiles[1][1:-1] += np.asarray([-0.004, 0.005], dtype=np.float64)
    old_profiles[2][1:-1] += np.asarray([0.005, -0.004], dtype=np.float64)
    adjacent = (old_profiles[0][-2], old_profiles[1][1], old_profiles[1][-2], old_profiles[2][1])
    instantaneous = model._initial_discrete_interface_residuals(old_interfaces, interface_compositions, adjacent, velocities)

    errors = []
    for epsilon in (8.0e-7, 4.0e-7, 2.0e-7):
        model._currdt = epsilon
        new_interfaces = old_interfaces + epsilon * velocities
        bulk_results = model._solve_bulk_profiles(old_profiles, old_interfaces, new_interfaces, interface_compositions)
        finite_step = model._interface_residuals(old_profiles, old_interfaces, new_interfaces, interface_compositions, bulk_results)
        errors.append(float(np.max(np.abs(finite_step / epsilon - instantaneous))))

    assert errors[-1] < errors[0]
    assert errors[-1] < 5.0e-6


def test_three_phase_initial_balance_uses_face_local_b_upwinding():
    model = _make_eta_varying_three_phase_model(record=False)
    model.setup()
    interfaces = np.asarray([0.35, 0.7], dtype=np.float64)
    interface_compositions = model._interface_compositions(np.asarray([0.4, 0.6], dtype=np.float64))
    adjacent = (
        np.asarray([0.19, 0.09], dtype=np.float64),
        np.asarray([0.34, 0.17], dtype=np.float64),
        np.asarray([0.27, 0.16], dtype=np.float64),
        np.asarray([0.24, 0.19], dtype=np.float64),
    )
    residual_zero = model._initial_discrete_interface_residuals(interfaces, interface_compositions, adjacent, [0.0, 0.0])
    positive_b_left = np.asarray([1.0, -10.0], dtype=np.float64)
    negative_b_left = np.asarray([1.0, -14.0], dtype=np.float64)

    for velocities in (positive_b_left, negative_b_left):
        actual_ale = model._initial_discrete_interface_residuals(interfaces, interface_compositions, adjacent, velocities) - residual_zero
        expected_ale = _manual_initial_ale_residual(model, interfaces, interface_compositions, adjacent, velocities)
        assert np.allclose(actual_ale, expected_ale, rtol=0.0, atol=1.0e-15)

    xi_b_left = 0.5 * model._grids[1][1]
    assert (1.0 - xi_b_left) * positive_b_left[0] + xi_b_left * positive_b_left[1] > 0.0
    assert (1.0 - xi_b_left) * negative_b_left[0] + xi_b_left * negative_b_left[1] < 0.0


@pytest.mark.parametrize(
    "interface, velocity",
    [
        ("ab", 1.25),
        ("ab", -1.25),
        ("bc", 1.25),
        ("bc", -1.25),
    ],
)
def test_three_phase_initial_balance_reduces_to_two_phase_swept_inventory(interface, velocity):
    model = _make_eta_varying_three_phase_model(record=False)
    model.setup()
    interfaces = np.asarray([0.35, 0.7], dtype=np.float64)
    interface_compositions = model._interface_compositions(np.asarray([0.4, 0.6], dtype=np.float64))
    c_a_ab, c_b_ab = interface_compositions[0]
    c_b_bc, c_c_bc = interface_compositions[1]
    adjacent = (
        np.asarray([0.19, 0.09], dtype=np.float64),
        np.asarray([0.34, 0.17], dtype=np.float64),
        np.asarray([0.27, 0.16], dtype=np.float64),
        np.asarray([0.24, 0.19], dtype=np.float64),
    )
    residual_zero = model._initial_discrete_interface_residuals(interfaces, interface_compositions, adjacent, [0.0, 0.0])
    velocities = np.asarray([velocity, 0.0] if interface == "ab" else [0.0, velocity], dtype=np.float64)
    coefficient = (model._initial_discrete_interface_residuals(interfaces, interface_compositions, adjacent, velocities) - residual_zero) / velocity

    if interface == "ab" and velocity > 0.0:
        expected = c_a_ab - adjacent[1] * (1.0 - model._grids[1][1] / 2.0) - c_b_ab * model._grids[1][1] / 2.0
        actual = coefficient[:2]
    elif interface == "ab":
        expected = adjacent[0] * ((1.0 + model._grids[0][-2]) / 2.0) + c_a_ab * ((1.0 - model._grids[0][-2]) / 2.0) - c_b_ab
        actual = coefficient[:2]
    elif velocity > 0.0:
        expected = c_b_bc - adjacent[3] * (1.0 - model._grids[2][1] / 2.0) - c_c_bc * model._grids[2][1] / 2.0
        actual = coefficient[2:]
    else:
        expected = adjacent[2] * ((1.0 + model._grids[1][-2]) / 2.0) + c_b_bc * ((1.0 - model._grids[1][-2]) / 2.0) - c_c_bc
        actual = coefficient[2:]
    assert np.allclose(actual, expected, rtol=0.0, atol=1.0e-14)


@pytest.mark.parametrize("target_velocity", [0.08, -0.08])
def test_three_phase_initial_eta_ab_limit_matches_two_phase_initializer(target_velocity):
    model = _make_eta_varying_three_phase_model(record=False)
    model.setup()
    interfaces = np.asarray([0.35, 0.7], dtype=np.float64)
    target_eta_ab = 0.2
    target_eta_bc = 0.8
    domain_length = float(interfaces[1])
    z, composition, p_adjacent, q_adjacent = _construct_two_phase_known_profile(
        model.interfaceEquilibria[0],
        target_eta_ab,
        target_velocity,
        interface_position=float(interfaces[0]),
        domain_length=domain_length,
        u_grid=model._grids[0],
        v_grid=model._grids[1],
    )

    two_phase = estimate_initial_eta_from_instantaneous_balance(
        composition=composition,
        z=z,
        interface_position=float(interfaces[0]),
        phases=["A", "B"],
        thermodynamics=_RecordingThreePhaseThermodynamics(),
        temperature=1000.0,
        interface_equilibrium=model.interfaceEquilibria[0],
        transformed_u_grid=model._grids[0],
        transformed_v_grid=model._grids[1],
        eta_bracket=(0.0, 1.0),
        eta_guess=0.35,
        velocity_guess=0.25 * target_velocity,
        root_xtol=1.0e-12,
        bulk_diffusivity_mode=_BULK_DIFFUSIVITY_PHASE_UNIFORM,
    )
    assert two_phase.converged
    assert np.isclose(two_phase.eta, target_eta_ab, rtol=0.0, atol=2.0e-10)
    assert np.isclose(two_phase.velocity, target_velocity, rtol=2.0e-10, atol=5.0e-12)

    c_b_bc, c_c_bc = model._interface_compositions([target_eta_ab, target_eta_bc])[1]
    target_etas = np.asarray([two_phase.eta, target_eta_bc], dtype=np.float64)
    target_velocities = np.asarray([two_phase.velocity, 0.0], dtype=np.float64)
    adjacent = _construct_balanced_initial_adjacent(
        model,
        interfaces,
        target_etas,
        target_velocities,
        base_adjacent=(p_adjacent, q_adjacent, c_b_bc, c_c_bc),
        locked_components=(0, 1, 2, 3),
    )

    estimate = _estimate_from_controlled_adjacent(
        model,
        interfaces,
        adjacent,
        eta_guess=np.asarray([0.35, 0.65], dtype=np.float64),
        velocity_guess=np.asarray([0.25 * target_velocity, 0.0], dtype=np.float64),
    )

    assert estimate.converged
    assert np.allclose(adjacent[0], p_adjacent, rtol=0.0, atol=0.0)
    assert np.allclose(adjacent[1], q_adjacent, rtol=0.0, atol=0.0)
    assert np.isclose(estimate.etas[0], two_phase.eta, rtol=0.0, atol=5.0e-10)
    assert np.isclose(estimate.velocities[0], two_phase.velocity, rtol=2.0e-10, atol=5.0e-12)
    assert np.isclose(estimate.velocities[1], 0.0, rtol=0.0, atol=5.0e-12)


def test_three_phase_nonconverged_candidate_residuals_telescope_total_inventory():
    model = _make_eta_varying_three_phase_model(record=False)
    model.setup()
    old_interfaces = np.asarray([0.35, 0.7], dtype=np.float64)
    old_etas = np.asarray([0.2, 0.8], dtype=np.float64)
    old_interface_compositions = model._interface_compositions(old_etas)
    c_a_ab_old, c_b_ab_old = old_interface_compositions[0]
    c_b_bc_old, c_c_bc_old = old_interface_compositions[1]
    old_profiles = (
        np.linspace(np.asarray([0.17, 0.07], dtype=np.float64), c_a_ab_old, len(model._grids[0])),
        np.linspace(c_b_ab_old, c_b_bc_old, len(model._grids[1])),
        np.linspace(c_c_bc_old, np.asarray([0.31, 0.17], dtype=np.float64), len(model._grids[2])),
    )
    old_profiles[0][1:-1] += np.asarray([0.01, -0.004], dtype=np.float64)
    old_profiles[1][1:-1] += np.asarray([-0.006, 0.008], dtype=np.float64)
    old_profiles[2][1:-1] += np.asarray([0.008, -0.005], dtype=np.float64)
    new_interfaces = np.asarray([0.41, 0.66], dtype=np.float64)
    new_interface_compositions = model._interface_compositions(np.asarray([0.73, 0.41], dtype=np.float64))
    model._currdt = 1e-5

    bulk_results = model._solve_bulk_profiles(old_profiles, old_interfaces, new_interfaces, new_interface_compositions)
    residual = model._interface_residuals(old_profiles, old_interfaces, new_interfaces, new_interface_compositions, bulk_results)
    old_inventory = model.getTotalInventoryFromState(old_profiles, old_interfaces)
    new_inventory = model.getTotalInventoryFromState(tuple(result.profile for result in bulk_results), new_interfaces)
    residual_sum = residual[:2] + residual[2:]

    assert np.max(np.abs(residual_sum)) > 1.0e-4
    assert np.allclose(new_inventory - old_inventory, residual_sum, rtol=0.0, atol=5.0e-16)


@pytest.mark.parametrize("mode", [_BULK_DIFFUSIVITY_PHASE_UNIFORM, _BULK_DIFFUSIVITY_LAGGED, _BULK_DIFFUSIVITY_IMPLICIT])
def test_three_phase_moving_raw_candidate_conserves_inventory_without_correction(mode):
    model = _make_eta_varying_three_phase_model(mode=mode, record=True)
    model.setup()
    state = _nonuniform_moving_candidate_state(model)
    old_inventory = model.getTotalInventoryFromState(tuple(state[:3]), state[3])

    dXdt = model.getdXdt(model.currentTime, state)
    raw_profiles = tuple(state[i] + model._currdt * dXdt[i] for i in range(3))
    raw_interfaces = state[3] + model._currdt * dXdt[3]
    raw_inventory = model.getTotalInventoryFromState(raw_profiles, raw_interfaces)
    drift_bound = 2.0 * model._lastImplicitPhysicalResidual + 5.0e-15

    assert np.max(np.abs(dXdt[3])) > 0.0
    assert np.max(np.abs(dXdt[4])) > 0.0
    assert model._lastImplicitResidual <= model.residualTolerance
    assert np.all(np.abs(raw_inventory - old_inventory) <= drift_bound)


def test_three_phase_interface_candidate_rejects_invalid_bulk_profile():
    model, _ = _make_stationary_model(record=False)
    model.setup()
    invalid_profiles = [profile.copy() for profile in model.getTransformedState()]
    invalid_profiles[1][2] = [0.51, 0.51]
    zero_flux = np.zeros(2, dtype=np.float64)

    def invalid_bulk_profiles(*args, **kwargs):
        return tuple(
            SimpleNamespace(
                profile=profile,
                left_transfer=zero_flux.copy(),
                right_transfer=zero_flux.copy(),
                left_face_matrix=None,
                right_face_matrix=None,
            )
            for profile in invalid_profiles
        )

    model._solve_bulk_profiles = invalid_bulk_profiles
    x_hat = model._physical_to_scaled(model.getInterfacePositions(), model.getInterfaceEtas())
    residual_scale = model._residual_scale(model.getTransformedState(), model.getInterfacePositions())

    with pytest.raises(ValueError, match="candidate transformed profile 1 violates ternary composition bounds"):
        model._evaluate_interface_candidate(
            model.getTransformedState(),
            model.getInterfacePositions(),
            x_hat,
            residual_scale,
            model.timeStep,
        )


@pytest.mark.parametrize(
    "mode, expected_context",
    [
        (_BULK_DIFFUSIVITY_PHASE_UNIFORM, "interface"),
        (_BULK_DIFFUSIVITY_LAGGED, "general"),
        (_BULK_DIFFUSIVITY_IMPLICIT, "general"),
    ],
)
def test_three_phase_diffusivity_modes_query_all_phases(mode, expected_context):
    model, therm = _make_stationary_model(mode=mode, record=False)

    model.solve(1e-4)

    phases = {call["phase"] for call in therm.calls}
    contexts = {call["query_context"] for call in therm.calls}
    assert phases == {"A", "B", "C"}
    assert expected_context in contexts
