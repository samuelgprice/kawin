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
    integrate_planar_transformed_molar_inventories,
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


class _ThreePhaseLinearProfile:
    def __init__(self, interfaces, endpoint_values, middle_bump):
        self.interfaces = tuple(float(v) for v in interfaces)
        self.endpoint_values = tuple(np.asarray(v, dtype=np.float64) for v in endpoint_values)
        self.middle_bump = np.asarray(middle_bump, dtype=np.float64)

    def __call__(self, z):
        x = np.asarray(z, dtype=np.float64).reshape(-1)
        s_ab, s_bc = self.interfaces
        left_a, right_a, left_b, right_b, left_c, right_c = self.endpoint_values
        out = np.empty((len(x), 2), dtype=np.float64)
        in_a = x < s_ab
        xi_a = x[in_a] / s_ab
        out[in_a] = left_a + xi_a[:, None] * (right_a - left_a)
        in_b = (x >= s_ab) & (x < s_bc)
        xi_b = (x[in_b] - s_ab) / (s_bc - s_ab)
        out[in_b] = left_b + xi_b[:, None] * (right_b - left_b)
        out[in_b] += (4.0 * xi_b * (1.0 - xi_b))[:, None] * self.middle_bump
        in_c = x >= s_bc
        xi_c = (x[in_c] - s_bc) / (1.0 - s_bc)
        out[in_c] = left_c + xi_c[:, None] * (right_c - left_c)
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


def _make_eta_varying_three_phase_model(*, mode=_BULK_DIFFUSIVITY_PHASE_UNIFORM, record=True, phase_molar_volumes=None):
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
        phase_molar_volumes=phase_molar_volumes,
        bulk_picard_max_iterations=20,
        phase_nodes=(7, 7, 7),
        time_step=1e-5,
        record=record,
        tolerance=1e-10,
        residual_tolerance=1e-10,
        max_iterations=50,
    )


def _make_nonstationary_unequal_volume_model(*, record=True, phase_molar_volumes=(2.0, 3.0, 1.5)):
    target_etas = np.asarray([0.2, 0.8], dtype=np.float64)
    eq_ab = _LinearPairEquilibrium([0.20, 0.10], [0.10, 0.0], [0.30, 0.15], [0.10, 0.0])
    eq_bc = _LinearPairEquilibrium([0.24, 0.15], [0.10, 0.0], [0.18, 0.20], [0.10, 0.0])
    interfaces = (0.35, 0.7)
    c_a_ab, c_b_ab = eq_ab.interface_compositions(target_etas[0])
    c_b_bc, c_c_bc = eq_bc.interface_compositions(target_etas[1])
    profile = _ThreePhaseLinearProfile(
        interfaces,
        (
            [0.17, 0.07],
            c_a_ab,
            c_b_ab,
            c_b_bc,
            c_c_bc,
            [0.31, 0.17],
        ),
        middle_bump=[0.006, -0.004],
    )
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 61)
    mesh.setResponseProfile(ProfileBuilder([(profile, ["X", "Y"])]))
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["A", "B", "C"],
        thermodynamics=_RecordingThreePhaseThermodynamics(),
        temperature=1000.0,
        interfacePositions=interfaces,
        interface_equilibria=(eq_ab, eq_bc),
        initial_eta_guess=target_etas,
        phase_molar_volumes=phase_molar_volumes,
        phase_nodes=(9, 9, 9),
        time_step=2e-6,
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


def _interval_fv_residual(
    old_profile,
    new_profile,
    grid,
    old_bounds,
    new_bounds,
    D_faces,
    dt,
    left_value=None,
    right_value=None,
    phase_frame_displacement=0.0,
):
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
        delta_x = (1.0 - xi) * (new_left - old_left) + xi * (new_right - old_right) - float(phase_frame_displacement)
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


@pytest.mark.parametrize(
    "phase_molar_volumes",
    [(), (1.0, 2.0), (1.0, 2.0, 0.0), (1.0, np.nan, 2.0), (1.0, np.inf, 2.0)],
)
def test_three_phase_rejects_invalid_phase_molar_volumes(phase_molar_volumes):
    with pytest.raises(ValueError, match="three positive finite"):
        _make_stationary_model(record=False, phase_molar_volumes=phase_molar_volumes)


def test_three_phase_rejects_molar_volumes_with_nonfinite_reciprocals():
    with pytest.raises(ValueError, match="non-finite molar densities"):
        _make_stationary_model(record=False, phase_molar_volumes=(1e-320, 1.0, 1.0))


def test_three_phase_molar_volume_state_is_immutable_and_equal_volume_kinematics_are_exact():
    model, _ = _make_stationary_model(record=False, phase_molar_volumes=(7.2e-6, 7.2e-6, 7.2e-6))
    model.setup()
    old_interfaces = np.asarray([0.35, 0.70], dtype=np.float64)
    new_interfaces = np.asarray([0.41, 0.66], dtype=np.float64)

    kinematics = model._step_kinematics(old_interfaces, new_interfaces)

    assert kinematics.old_right_boundary == model._R0
    assert kinematics.new_right_boundary == model._R0
    assert kinematics.new_right_boundary - kinematics.old_right_boundary == 0.0
    assert np.array_equal(kinematics.phase_displacements, np.zeros(3, dtype=np.float64))
    assert np.array_equal(
        model._residual_scale(model.getTransformedState(), model.getInterfacePositions()),
        np.full(4, model._R0),
    )
    with pytest.raises(ValueError):
        model._phaseMolarVolumes[0] = 1.0
    with pytest.raises(ValueError):
        model._phaseMolarDensities[0] = 1.0


def test_three_phase_legacy_inventory_semantics_do_not_depend_on_molar_volumes():
    legacy, _ = _make_stationary_model(record=False)
    unequal, _ = _make_stationary_model(record=False, phase_molar_volumes=(2.0, 3.0, 1.5))
    legacy.setup()
    unequal.setup()

    assert np.array_equal(legacy.getTotalInventory(), unequal.getTotalInventory())


@pytest.mark.parametrize("new_interfaces", [(0.41, 0.66), (0.30, 0.75), (0.38, 0.76)])
def test_three_phase_unequal_volume_algebraic_and_local_kinematics_agree(new_interfaces):
    molar_volumes = np.asarray([2.0, 3.0, 1.5], dtype=np.float64)
    model, _ = _make_stationary_model(record=False, phase_molar_volumes=molar_volumes)
    model.setup()
    old_interfaces = np.asarray([0.35, 0.70], dtype=np.float64)
    new_interfaces = np.asarray(new_interfaces, dtype=np.float64)

    kinematics = model._step_kinematics(old_interfaces, new_interfaces)
    old_widths = model._phase_widths(old_interfaces, kinematics.old_right_boundary)
    new_widths = model._phase_widths(new_interfaces, kinematics.new_right_boundary)
    delta_s_ab, delta_s_bc = new_interfaces - old_interfaces
    local_delta_u_b = (1.0 - molar_volumes[1] / molar_volumes[0]) * delta_s_ab
    local_delta_u_c = (molar_volumes[2] / molar_volumes[1]) * local_delta_u_b
    local_delta_u_c += (1.0 - molar_volumes[2] / molar_volumes[1]) * delta_s_bc

    assert np.isclose(np.dot(model._phaseMolarDensities, old_widths), model._initialTotalAmount, rtol=0.0, atol=1e-16)
    assert np.isclose(np.dot(model._phaseMolarDensities, new_widths), model._initialTotalAmount, rtol=0.0, atol=1e-16)
    assert np.isclose(kinematics.new_right_boundary - kinematics.old_right_boundary, local_delta_u_c, rtol=0.0, atol=1e-16)
    assert kinematics.phase_displacements[2] == kinematics.new_right_boundary - kinematics.old_right_boundary
    assert kinematics.phase_displacements[0] == 0.0
    assert np.isclose(
        (delta_s_ab - kinematics.phase_displacements[0]) / molar_volumes[0],
        (delta_s_ab - kinematics.phase_displacements[1]) / molar_volumes[1],
        rtol=0.0,
        atol=1e-16,
    )
    assert np.isclose(
        (delta_s_bc - kinematics.phase_displacements[1]) / molar_volumes[1],
        (delta_s_bc - kinematics.phase_displacements[2]) / molar_volumes[2],
        rtol=0.0,
        atol=1e-16,
    )
    assert np.isclose(0.0 - kinematics.phase_displacements[0], 0.0, rtol=0.0, atol=0.0)
    assert np.isclose(
        kinematics.new_right_boundary - kinematics.old_right_boundary - kinematics.phase_displacements[2],
        0.0,
        rtol=0.0,
        atol=0.0,
    )


def test_direct_all_component_molar_inventory_integrates_dependent_component():
    grids = (
        np.asarray([0.0, 0.1, 0.45, 0.8, 1.0]),
        np.asarray([0.0, 0.25, 0.55, 0.9, 1.0]),
        np.asarray([0.0, 0.2, 0.6, 0.75, 1.0]),
    )
    profiles = (
        np.column_stack((0.18 + 0.05 * grids[0], 0.08 + 0.02 * grids[0])),
        np.column_stack((0.31 - 0.04 * grids[1], 0.12 + 0.03 * grids[1])),
        np.column_stack((0.24 + 0.02 * grids[2], 0.19 - 0.05 * grids[2])),
    )
    interfaces = np.asarray([0.23, 0.71], dtype=np.float64)
    domain_length = 1.08
    molar_volumes = np.asarray([2.0, 3.0, 1.5], dtype=np.float64)

    inventory = integrate_planar_transformed_molar_inventories(
        profiles, interfaces, domain_length, grids, molar_volumes
    )
    widths = np.diff(np.concatenate(([0.0], interfaces, [domain_length])))
    total_direct = np.sum(widths / molar_volumes)
    dependent_direct = np.sum(
        [
            width * np.trapezoid(1.0 - np.sum(profile, axis=1), grid) / molar_volume
            for profile, grid, width, molar_volume in zip(profiles, grids, widths, molar_volumes)
        ]
    )

    assert np.isclose(inventory[0], dependent_direct, rtol=0.0, atol=1e-16)
    assert np.isclose(np.sum(inventory), total_direct, rtol=0.0, atol=2e-16)
    assert np.isclose(inventory[0], total_direct - inventory[1] - inventory[2], rtol=0.0, atol=2e-16)


def test_direct_dependent_inventory_is_conserved_when_total_and_independent_moles_are_conserved():
    grids = tuple(np.asarray([0.0, 0.1, 0.45, 0.8, 1.0]) for _ in range(3))
    profiles = (
        np.full((5, 2), [0.20, 0.10], dtype=np.float64),
        np.full((5, 2), [0.30, 0.15], dtype=np.float64),
        np.full((5, 2), [0.25, 0.20], dtype=np.float64),
    )
    perturbed = tuple(profile.copy() for profile in profiles)
    zero_integral_shape = grids[1] - 0.5
    perturbed[1][:, 0] += 0.04 * zero_integral_shape
    perturbed[1][:, 1] -= 0.03 * zero_integral_shape
    interfaces = np.asarray([0.25, 0.75], dtype=np.float64)
    molar_volumes = np.asarray([2.0, 3.0, 1.5], dtype=np.float64)

    old_inventory = integrate_planar_transformed_molar_inventories(profiles, interfaces, 1.0, grids, molar_volumes)
    new_inventory = integrate_planar_transformed_molar_inventories(perturbed, interfaces, 1.0, grids, molar_volumes)

    assert np.allclose(new_inventory[1:], old_inventory[1:], rtol=0.0, atol=2e-17)
    assert np.isclose(np.sum(new_inventory), np.sum(old_inventory), rtol=0.0, atol=1e-16)
    assert np.isclose(new_inventory[0], old_inventory[0], rtol=0.0, atol=1e-16)


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


@pytest.mark.parametrize(
    "old_bounds,new_bounds,phase_displacement,expected_signs",
    [
        ([0.20, 0.70], [0.31, 0.81], 0.11, (False, False)),
        ([0.20, 0.70], [0.27, 0.85], 0.04, (True, False)),
        ([0.20, 0.70], [0.15, 0.61], -0.02, (False, True)),
        ([0.20, 0.80], [0.30, 0.70], 0.00, (True, True)),
    ],
    ids=("rigid-translation", "expansion-positive", "contraction-negative", "sign-changing"),
)
def test_three_phase_relative_ale_operator_isolated_cases(old_bounds, new_bounds, phase_displacement, expected_signs):
    model, _ = _make_stationary_model(record=False)
    model.setup()
    grid = np.asarray([0.0, 0.08, 0.31, 0.67, 0.91, 1.0], dtype=np.float64)
    value = np.asarray([0.27, 0.13], dtype=np.float64)
    profile = np.broadcast_to(value, (len(grid), 2)).copy()
    D_faces = np.zeros((len(grid) - 1, 2, 2), dtype=np.float64)
    model._currdt = 0.025

    relative_displacements = model._internal_face_displacements(
        grid, old_bounds, new_bounds, phase_frame_displacement=phase_displacement
    )
    result = model._solve_interval_planar(
        profile,
        grid,
        old_bounds,
        new_bounds,
        phase_index=1,
        left_value=value,
        right_value=value,
        D_faces=D_faces,
        validate_diffusivity=False,
        phase_frame_displacement=phase_displacement,
    )
    residual = _interval_fv_residual(
        profile,
        result.profile,
        grid,
        old_bounds,
        new_bounds,
        D_faces,
        model._currdt,
        left_value=value,
        right_value=value,
        phase_frame_displacement=phase_displacement,
    )

    has_positive, has_negative = expected_signs
    assert bool(np.any(relative_displacements > 1e-15)) is has_positive
    assert bool(np.any(relative_displacements < -1e-15)) is has_negative
    assert np.allclose(result.profile, profile, rtol=0.0, atol=3e-16)
    assert np.allclose(residual, 0.0, rtol=0.0, atol=3e-16)


def test_three_phase_relative_ale_donor_follows_relative_displacement_sign():
    model, _ = _make_stationary_model(record=False)
    model.setup()
    left = np.asarray([0.18, 0.07], dtype=np.float64)
    right = np.asarray([0.31, 0.16], dtype=np.float64)
    zero_diffusivity = np.zeros((2, 2), dtype=np.float64)

    for displacement, expected in ((0.04, 0.04 * right), (-0.03, -0.03 * left), (0.0, np.zeros(2))):
        coefficients = model._face_transfer_coefficients(displacement, zero_diffusivity, 0.2, 0.5, 0.1)
        transfer = model._evaluate_face_transfer(*coefficients, left, right)
        assert np.allclose(transfer, expected, rtol=0.0, atol=1e-17)


def test_three_phase_explicit_equal_volumes_match_legacy_bulk_path_exactly():
    legacy, _ = _make_stationary_model(record=False)
    equal_volume, _ = _make_stationary_model(record=False, phase_molar_volumes=(7.2e-6, 7.2e-6, 7.2e-6))
    legacy.setup()
    equal_volume.setup()
    old_profiles = legacy.getTransformedState()
    old_interfaces = np.asarray([0.35, 0.70], dtype=np.float64)
    new_interfaces = np.asarray([0.39, 0.67], dtype=np.float64)
    interface_compositions = legacy._interface_compositions(np.asarray([0.25, 0.75]))
    legacy._currdt = equal_volume._currdt = 1e-5

    legacy_results = legacy._solve_bulk_profiles(old_profiles, old_interfaces, new_interfaces, interface_compositions)
    equal_results = equal_volume._solve_bulk_profiles(old_profiles, old_interfaces, new_interfaces, interface_compositions)

    for legacy_result, equal_result in zip(legacy_results, equal_results):
        assert np.array_equal(legacy_result.profile, equal_result.profile)
        assert np.array_equal(legacy_result.left_transfer, equal_result.left_transfer)
        assert np.array_equal(legacy_result.right_transfer, equal_result.right_transfer)


def test_three_phase_explicit_equal_volumes_match_legacy_nonlinear_step_exactly():
    legacy = _make_eta_varying_three_phase_model(record=False)
    equal_volume = _make_eta_varying_three_phase_model(
        record=False,
        phase_molar_volumes=(7.2e-6, 7.2e-6, 7.2e-6),
    )
    legacy.setup()
    equal_volume.setup()
    legacy_state = _nonuniform_moving_candidate_state(legacy)
    equal_state = _nonuniform_moving_candidate_state(equal_volume)

    legacy_derivative = legacy.getdXdt(legacy.currentTime, legacy_state)
    equal_derivative = equal_volume.getdXdt(equal_volume.currentTime, equal_state)

    for legacy_value, equal_value in zip(legacy_derivative, equal_derivative):
        assert np.array_equal(legacy_value, equal_value)


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


def test_three_phase_invalid_trial_geometry_is_rejected_before_diffusivity(monkeypatch):
    model, therm = _make_stationary_model(record=False, phase_molar_volumes=(2.0, 3.0, 1.5))
    model.setup()
    therm.calls.clear()

    def unexpected_diffusivity(*args, **kwargs):
        raise AssertionError("diffusivity must not be evaluated for invalid trial geometry")

    monkeypatch.setattr(model, "_phase_uniform_diffusivity", unexpected_diffusivity)
    with pytest.raises(ValueError, match="R_trial"):
        model._solve_bulk_profiles(
            model.getTransformedState(),
            np.asarray([0.35, 0.70]),
            np.asarray([1.10, 1.15]),
            model._interface_compositions(np.asarray([0.25, 0.75])),
        )
    assert therm.calls == []


def test_three_phase_unequal_volume_candidate_keeps_trial_boundary_local_and_d_unchanged():
    model, _ = _make_stationary_model(record=False, phase_molar_volumes=(2.0, 3.0, 1.5))
    model.setup()
    accepted_profiles = tuple(profile.copy() for profile in model._profiles_curr)
    accepted_interfaces = model._interfaces_curr.copy()
    accepted_right_boundary = model._R
    new_interfaces = np.asarray([0.39, 0.67], dtype=np.float64)
    x_hat = model._physical_to_scaled(new_interfaces, model._etas_curr)
    model._currdt = model.timeStep

    candidate = model._evaluate_interface_candidate(
        accepted_profiles,
        accepted_interfaces,
        x_hat,
        model._residual_scale(accepted_profiles, accepted_interfaces),
        model.timeStep,
    )

    assert candidate.kinematics.new_right_boundary != accepted_right_boundary
    assert model._R == accepted_right_boundary
    assert np.array_equal(model._interfaces_curr, accepted_interfaces)
    for actual, expected in zip(model._profiles_curr, accepted_profiles):
        assert np.array_equal(actual, expected)
    assert np.array_equal(candidate.bulk_results[0].right_face_matrix, np.eye(2))
    assert np.array_equal(candidate.bulk_results[1].left_face_matrix, np.eye(2))
    assert np.array_equal(candidate.bulk_results[1].right_face_matrix, np.eye(2))
    assert np.array_equal(candidate.bulk_results[2].left_face_matrix, np.eye(2))


def test_three_phase_unequal_volume_full_solve_commits_synchronized_stationary_state():
    model, _ = _make_stationary_model(record=False, phase_molar_volumes=(2.0, 3.0, 1.5))

    model.solve(model.timeStep)

    assert model.getRightBoundary() == model._R0
    assert np.allclose(model.checkMolarConservation(1e-10), 0.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("mismatch", ["state", "timestep"])
def test_three_phase_pending_candidate_rejects_outer_mismatch_without_commit(mismatch):
    model, _ = _make_stationary_model(record=True, phase_molar_volumes=(2.0, 3.0, 1.5))
    model.setup()
    model.setTimeInfo(0.0, model.timeStep)
    state = model.getCurrentX()
    derivative = model.getdXdt(0.0, state)
    dt = model.getDt(derivative)
    wrapper_state = [
        np.asarray(value) + np.asarray(rate) * dt
        for value, rate in zip(state, derivative)
    ]
    wrapper_time = dt
    if mismatch == "state":
        wrapper_state[3] = wrapper_state[3].copy()
        wrapper_state[3][0] += 1e-8
    else:
        wrapper_time = 0.5 * dt
    accepted_right = model._R
    accepted_time = model.currentTime
    history_state = [
        (history.N, history.currentIndex, history.currentTime)
        for history in (model.data, model.interfaceData, model.etaData, model.rightBoundaryData)
    ]

    with pytest.raises(RuntimeError, match="does not match the pending converged"):
        model.postProcess(wrapper_time, wrapper_state)

    assert model._pendingCandidate is None
    assert model._R == accepted_right
    assert model.currentTime == accepted_time
    assert np.array_equal(model._interfaces_curr, state[3])
    assert [
        (history.N, history.currentIndex, history.currentTime)
        for history in (model.data, model.interfaceData, model.etaData, model.rightBoundaryData)
    ] == history_state


def test_three_phase_nonlinear_timestep_honors_outer_maximum_fraction():
    model, _ = _make_stationary_model(record=True, phase_molar_volumes=(2.0, 3.0, 1.5))

    model.solve(2.0e-4, maxDtFrac=0.125)

    recorded_times = model.rightBoundaryData._time[: model.rightBoundaryData.N + 1]
    assert len(recorded_times) == 9
    assert np.allclose(np.diff(recorded_times), 2.5e-5, rtol=0.0, atol=2e-18)
    assert model._pendingCandidate is None


def test_three_phase_retry_below_outer_minimum_is_rejected_before_wrapper(monkeypatch):
    model, _ = _make_stationary_model(
        record=False,
        phase_molar_volumes=(2.0, 3.0, 1.5),
        max_step_retries=2,
        retry_factor=0.5,
    )
    model.setup()
    model._solveMinDtFrac = 0.75
    model.setTimeInfo(0.0, model.timeStep)
    calls = []

    def retry_once(profiles, interfaces, etas, dt):
        calls.append(dt)
        if len(calls) == 1:
            raise RuntimeError("forced retry")
        return _ThreePhaseCandidate(
            x_hat=np.zeros(4),
            profiles=tuple(profile.copy() for profile in profiles),
            interfaces=np.asarray(interfaces).copy(),
            etas=np.asarray(etas).copy(),
            interface_compositions=model._interface_compositions(etas),
            residual=np.zeros(4),
            scaled_residual=np.zeros(4),
            scaled_norm=0.0,
            physical_norm=0.0,
            bulk_results=(None, None, None),
        )

    monkeypatch.setattr(model, "_solve_interface_planar", retry_once)

    with pytest.raises(ValueError, match="smaller than the outer solver minimum"):
        model.getdXdt(0.0, model.getCurrentX())

    assert np.allclose(calls, [model.timeStep, 0.5 * model.timeStep], rtol=0.0, atol=1e-18)
    assert model._pendingCandidate is None


def test_three_phase_final_remainder_below_outer_minimum_is_solved_and_accepted(monkeypatch):
    model, _ = _make_stationary_model(
        record=True,
        phase_molar_volumes=(6.8e-6, 8.1e-6, 7.4e-6),
    )
    solved_dts = []
    solve_candidate = model._solve_interface_planar

    def record_trial_dt(profiles, interfaces, etas, dt):
        solved_dts.append(float(dt))
        return solve_candidate(profiles, interfaces, etas, dt)

    monkeypatch.setattr(model, "_solve_interface_planar", record_trial_dt)
    model.solve(1.2e-4, minDtFrac=0.25)

    assert np.allclose(solved_dts, [1.0e-4, 2.0e-5], rtol=0.0, atol=2e-18)
    assert model.currentTime == pytest.approx(1.2e-4, rel=0.0, abs=2e-18)
    assert np.array_equal(
        model.rightBoundaryData._time[: model.rightBoundaryData.N + 1],
        np.asarray([0.0, 1.0e-4, 1.2e-4]),
    )
    assert model._pendingCandidate is None
    assert np.allclose(model.checkMolarConservation(2e-10), 0.0, rtol=0.0, atol=2e-10)


def test_three_phase_tiny_final_remainder_commits_all_histories_and_allows_continuation(monkeypatch):
    model, _ = _make_stationary_model(
        record=True,
        phase_molar_volumes=(6.8e-6, 8.1e-6, 7.4e-6),
    )
    tiny_remainder = 5.0e-15
    first_duration = model.timeStep + tiny_remainder
    solved_dts = []
    solve_candidate = model._solve_interface_planar

    def record_trial_dt(profiles, interfaces, etas, dt):
        solved_dts.append(float(dt))
        return solve_candidate(profiles, interfaces, etas, dt)

    monkeypatch.setattr(model, "_solve_interface_planar", record_trial_dt)
    model.solve(first_duration, minDtFrac=0.25)

    expected_first_times = np.asarray([0.0, model.timeStep, first_duration])
    histories = [
        model.data,
        model.interfaceData,
        model.etaData,
        model.rightBoundaryData,
        model.inventoryData,
        model.molarInventoryData,
        model.totalMolesData,
        model.dependentMoleClosureErrorData,
        *model.profileData,
    ]
    assert len(solved_dts) == 2
    assert solved_dts[1] == pytest.approx(first_duration - model.timeStep, rel=0.0, abs=1e-20)
    assert solved_dts[1] < model.timeStep * 1e-10
    for history in histories:
        assert np.array_equal(history._time[: history.N + 1], expected_first_times)
    assert model.currentTime == pytest.approx(first_duration, rel=0.0, abs=1e-20)
    assert model._pendingCandidate is None

    model.solve(model.timeStep, minDtFrac=0.25)

    expected_final_time = first_duration + model.timeStep
    expected_times = np.append(expected_first_times, expected_final_time)
    assert len(solved_dts) == 3
    assert solved_dts[-1] == pytest.approx(model.timeStep, rel=0.0, abs=2e-20)
    for history in histories:
        assert np.array_equal(history._time[: history.N + 1], expected_times)
    assert model.currentTime == pytest.approx(expected_final_time, rel=0.0, abs=1e-20)
    assert model._pendingCandidate is None
    assert np.allclose(model.checkMolarConservation(2e-10), 0.0, rtol=0.0, atol=2e-10)


def test_three_phase_structured_molar_conservation_diagnostics_are_direct_and_signed():
    model, _ = _make_stationary_model(record=False, phase_molar_volumes=(2.0, 3.0, 1.5))
    model.solve(model.timeStep)

    diagnostics = model.getMolarConservationDiagnostics()

    assert diagnostics["time"] == model.currentTime
    assert np.array_equal(diagnostics["component_moles"], model.getAllComponentMoles())
    assert np.allclose(diagnostics["component_drift"], 0.0, rtol=0.0, atol=3e-17)
    assert np.allclose(diagnostics["absolute_component_drift"], 0.0, rtol=0.0, atol=3e-17)
    assert diagnostics["N0_drift"] == pytest.approx(0.0, abs=3e-17)
    assert diagnostics["N1_drift"] == pytest.approx(0.0, abs=3e-17)
    assert diagnostics["N2_drift"] == pytest.approx(0.0, abs=3e-17)
    assert diagnostics["total_substitutional_mole_drift"] == pytest.approx(0.0, abs=3e-17)
    assert diagnostics["dependent_component_closure_error"] == pytest.approx(0.0, abs=2e-16)

    normalized, _ = _make_stationary_model(record=False)
    normalized.setup()
    with pytest.raises(ValueError, match="explicit phase_molar_volumes"):
        normalized.getMolarConservationDiagnostics()


def test_three_phase_public_nonstationary_unequal_volume_solve_commits_moving_geometry():
    model = _make_nonstationary_unequal_volume_model(record=True)

    model.solve(2e-6)

    times = model.interfaceData._time[: model.interfaceData.N + 1]
    interfaces = model.interfaceData._y[: model.interfaceData.N + 1]
    right_boundaries = model.rightBoundaryData._y[: model.rightBoundaryData.N + 1]
    assert len(times) == 2
    assert not np.isclose(interfaces[-1, 0], interfaces[0, 0], rtol=0.0, atol=1e-13)
    assert not np.isclose(interfaces[-1, 1], interfaces[0, 1], rtol=0.0, atol=1e-13)
    assert not np.isclose(right_boundaries[-1], right_boundaries[0], rtol=0.0, atol=1e-13)
    predicted_right_velocity = model._phase_frame_velocities(model.initialEtaEstimate.velocities)[2]
    assert np.sign(right_boundaries[-1] - right_boundaries[0]) == np.sign(predicted_right_velocity)

    for old_interfaces, new_interfaces, old_right, new_right in zip(
        interfaces[:-1], interfaces[1:], right_boundaries[:-1], right_boundaries[1:]
    ):
        kinematics = model._step_kinematics(old_interfaces, new_interfaces)
        assert np.isclose(new_right - old_right, kinematics.phase_displacements[2], rtol=0.0, atol=2e-16)
        assert 0.0 < new_interfaces[0] < new_interfaces[1] < new_right

    directly_integrated = []
    for time, step_interfaces, right_boundary in zip(times, interfaces, right_boundaries):
        directly_integrated.append(
            integrate_planar_transformed_molar_inventories(
                model.getTransformedState(time=time),
                step_interfaces,
                right_boundary,
                model._grids,
                model._phaseMolarVolumes,
            )
        )
    directly_integrated = np.asarray(directly_integrated)
    assert np.allclose(directly_integrated[-1], directly_integrated[0], rtol=0.0, atol=2e-11)
    assert np.allclose(model.getAllComponentMoles(), directly_integrated[-1], rtol=0.0, atol=2e-16)
    assert np.isclose(model.getTotalSubstitutionalMoles(), model._initialTotalMoles, rtol=0.0, atol=2e-16)
    diagnostics = model.getMolarConservationDiagnostics()
    assert np.all(diagnostics["absolute_component_drift"] < 2e-11)
    assert diagnostics["absolute_total_substitutional_mole_drift"] < 2e-16

    histories = (
        model.data,
        model.interfaceData,
        model.etaData,
        model.rightBoundaryData,
        model.inventoryData,
        model.molarInventoryData,
        model.totalMolesData,
        model.dependentMoleClosureErrorData,
        *model.profileData,
    )
    for history in histories:
        assert history.N == len(times) - 1
        assert np.array_equal(history._time[: history.N + 1], times)


def test_three_phase_physical_mole_apis_require_explicit_volumes():
    model, _ = _make_stationary_model(record=False)
    model.setup()

    with pytest.raises(ValueError, match="explicit phase_molar_volumes"):
        model.getAllComponentMoles()
    with pytest.raises(ValueError, match="explicit phase_molar_volumes"):
        model.getTotalSubstitutionalMoles()
    assert np.all(np.isfinite(model.getTotalInventory()))


def test_three_phase_trial_outputs_are_pure_until_atomic_accepted_commit():
    model, _ = _make_stationary_model(record=True, phase_molar_volumes=(2.0, 3.0, 1.5))
    model.setup()
    accepted_right = model._R
    accepted_interfaces = model._interfaces_curr.copy()
    new_interfaces = np.asarray([0.41, 0.66], dtype=np.float64)
    trial_right = model._right_boundary_from_interfaces(new_interfaces)
    profiles = model.getTransformedState()

    legacy = model.getTotalInventoryFromState(profiles, new_interfaces, right_boundary=trial_right)
    moles = model._get_all_component_moles_from_state(profiles, new_interfaces, trial_right)
    fixed = model._reconstruct_physical_profile(profiles, new_interfaces, right_boundary=trial_right)
    prepared = model._prepare_accepted_state([*profiles, new_interfaces, model._etas_curr])

    assert np.all(np.isfinite(legacy))
    assert np.all(np.isfinite(moles))
    assert np.any(np.isnan(fixed))
    assert model._R == accepted_right
    assert np.array_equal(model._interfaces_curr, accepted_interfaces)
    assert model.rightBoundaryData.N == 0

    model._commit_accepted_state(model.timeStep, prepared)

    assert model._R == trial_right
    assert model.getRightBoundary() == prepared.right_boundary
    assert np.array_equal(model._interfaces_curr, new_interfaces)


def test_three_phase_failed_retries_do_not_mutate_accepted_boundary_or_histories(monkeypatch):
    model, _ = _make_stationary_model(record=True, phase_molar_volumes=(2.0, 3.0, 1.5))
    model.setup()
    state = model.getCurrentX()
    accepted_right = model._R
    histories = (
        model.data,
        model.interfaceData,
        model.etaData,
        model.rightBoundaryData,
        model.inventoryData,
        model.molarInventoryData,
        model.totalMolesData,
        model.dependentMoleClosureErrorData,
        *model.profileData,
    )
    before = [(history.N, history.currentIndex, history.currentTime) for history in histories]

    def fail_candidate(*args, **kwargs):
        raise RuntimeError("synthetic nonlinear failure")

    monkeypatch.setattr(model, "_solve_interface_planar", fail_candidate)
    derivative, _, error = model._try_step_retries(
        tuple(state[:3]), state[3], state[4], model.timeStep, retry_count=3
    )

    assert derivative is None
    assert isinstance(error, RuntimeError)
    assert model._R == accepted_right
    assert [(history.N, history.currentIndex, history.currentTime) for history in histories] == before


def test_three_phase_atomic_history_failure_rolls_back_every_history(monkeypatch):
    model, _ = _make_stationary_model(record=True, phase_molar_volumes=(2.0, 3.0, 1.5))
    model.setup()
    profiles = model.getTransformedState()
    new_interfaces = np.asarray([0.41, 0.66], dtype=np.float64)
    accepted = model._prepare_accepted_state([*profiles, new_interfaces, model._etas_curr])
    histories = (
        model.data,
        model.interfaceData,
        model.etaData,
        model.rightBoundaryData,
        model.inventoryData,
        model.molarInventoryData,
        model.totalMolesData,
        model.dependentMoleClosureErrorData,
        *model.profileData,
    )
    for history in histories:
        history.batchSize = 1
        history._time = history._time[:1].copy()
        history._y = history._y[:1].copy()
    before = [
        (
            history.N,
            history.currentIndex,
            history.currentTime,
            np.asarray(history.currentY).copy(),
            history._time.copy(),
            history._y.copy(),
        )
        for history in histories
    ]
    accepted_right = model._R

    def fail_record(*args, **kwargs):
        raise RuntimeError("synthetic history failure")

    monkeypatch.setattr(model.etaData, "record", fail_record)
    with pytest.raises(RuntimeError, match="synthetic history failure"):
        model._commit_accepted_state(model.timeStep, accepted)

    assert model._R == accepted_right
    for history, snapshot in zip(histories, before):
        index, current_index, current_time, current_y, times, values = snapshot
        assert history.N == index
        assert history.currentIndex == current_index
        assert history.currentTime == current_time
        assert np.array_equal(np.asarray(history.currentY), current_y, equal_nan=True)
        assert np.array_equal(history._time, times)
        assert np.array_equal(history._y, values, equal_nan=True)


@pytest.mark.parametrize(
    "new_interfaces,direction",
    [((0.30, 0.75), "expansion"), ((0.41, 0.66), "contraction")],
)
def test_three_phase_unequal_accepted_geometry_histories_and_physical_profiles(new_interfaces, direction):
    model, _ = _make_stationary_model(record=True, phase_molar_volumes=(2.0, 3.0, 1.5))
    model.setup()
    profiles = model.getTransformedState()
    new_interfaces = np.asarray(new_interfaces, dtype=np.float64)
    accepted_time = model.timeStep

    model._commit_accepted_state(
        accepted_time,
        model._prepare_accepted_state([*profiles, new_interfaces, model._etas_curr]),
    )

    if direction == "expansion":
        assert model._R > model._R0
    else:
        assert model._R < model._R0
    histories = (
        model.data,
        model.interfaceData,
        model.etaData,
        model.rightBoundaryData,
        model.inventoryData,
        model.molarInventoryData,
        model.totalMolesData,
        model.dependentMoleClosureErrorData,
        *model.profileData,
    )
    expected_times = np.asarray([0.0, accepted_time])
    for history in histories:
        assert np.array_equal(history._time[: history.N + 1], expected_times)

    all_moles = model.getAllComponentMoles()
    total_moles = model.getTotalSubstitutionalMoles()
    assert np.isclose(np.sum(all_moles), total_moles, rtol=0.0, atol=2e-16)
    assert np.isclose(all_moles[0], total_moles - all_moles[1] - all_moles[2], rtol=0.0, atol=2e-16)
    assert np.isclose(model.getDependentMoleClosureError(), 0.0, rtol=0.0, atol=2e-16)
    assert np.array_equal(model.getAllComponentMoles(time=accepted_time), model.molarInventoryData.y(accepted_time))

    phase_profiles = model.getPhysicalPhaseProfiles(time=accepted_time)
    coordinates = tuple(item[0] for item in phase_profiles)
    assert coordinates[0][0] == 0.0
    assert coordinates[-1][-1] == model._R
    assert coordinates[0][-1] == new_interfaces[0]
    assert coordinates[1][0] == new_interfaces[0]
    assert coordinates[1][-1] == new_interfaces[1]
    assert coordinates[2][0] == new_interfaces[1]
    assert not np.array_equal(phase_profiles[0][1][-1], phase_profiles[1][1][0])
    assert not np.array_equal(phase_profiles[1][1][-1], phase_profiles[2][1][0])

    fixed_coordinates = model.getFixedPhysicalCoordinates()
    fixed_compositions = model.getCompositions(time=accepted_time)
    assert np.array_equal(fixed_coordinates, model._z)
    assert fixed_compositions.shape == (len(fixed_coordinates), 3)
    if direction == "contraction":
        outside = fixed_coordinates > model._R
        assert np.any(outside)
        assert np.all(np.isnan(fixed_compositions[outside]))
        assert np.all(np.isfinite(model.getCompositions(time=0.0)))
    else:
        assert coordinates[-1][-1] > fixed_coordinates[-1]
        assert np.all(np.isfinite(fixed_compositions))
    with pytest.raises(ValueError, match="Fixed-mesh interpolation is unavailable"):
        model.getCompositions(time=0.5 * accepted_time)
    with pytest.raises(ValueError, match="Fixed-mesh interpolation is unavailable"):
        model.data.y(time=0.5 * accepted_time)
    with pytest.raises(ValueError, match="exact history time"):
        model.getPhysicalPhaseProfiles(time=0.5 * accepted_time)
    model.postSolve()
    for history in histories:
        assert np.array_equal(history._time, expected_times)


def test_three_phase_equal_volume_accepted_state_keeps_right_boundary_exactly_fixed():
    model, _ = _make_stationary_model(
        record=True,
        phase_molar_volumes=(7.2e-6, 7.2e-6, 7.2e-6),
    )
    model.setup()
    profiles = model.getTransformedState()

    accepted = model._prepare_accepted_state(
        [*profiles, np.asarray([0.39, 0.67]), model._etas_curr]
    )
    model._commit_accepted_state(model.timeStep, accepted)

    assert model._R == model._R0
    assert model.getRightBoundary() == model._R0
    assert model.rightBoundaryData.y(model.timeStep) == model._R0
    assert np.all(np.isfinite(model.getCompositions(time=model.timeStep)))
    expected_midpoint = 0.5 * (model.data._y[0] + model.data._y[1])
    assert np.allclose(model.data.y(0.5 * model.timeStep), expected_midpoint, rtol=0.0, atol=1e-16)


@pytest.mark.parametrize(
    "new_interfaces,direction",
    [((0.30, 0.75), "expansion"), ((0.41, 0.66), "contraction")],
)
def test_three_phase_matplotlib_uses_complete_moving_domain(new_interfaces, direction):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    model, _ = _make_stationary_model(record=True, phase_molar_volumes=(2.0, 3.0, 1.5))
    model.setup()
    profiles = model.getTransformedState()
    accepted = model._prepare_accepted_state(
        [*profiles, np.asarray(new_interfaces, dtype=np.float64), model._etas_curr]
    )
    model._commit_accepted_state(model.timeStep, accepted)

    profile_figure, profile_axis = model.plot_latestCompProfile()
    width_figure, width_axis = model.plot_phaseWidths_vs_time()

    composition_lines = profile_axis.lines[: 3 * len(model.elements)]
    plotted_maximum = max(float(np.max(line.get_xdata())) for line in composition_lines)
    assert plotted_maximum == pytest.approx(model.getRightBoundary() * 1.0e6)
    assert all(np.all(np.isfinite(line.get_ydata())) for line in composition_lines)
    if direction == "expansion":
        assert plotted_maximum > model._R0 * 1.0e6
    else:
        assert plotted_maximum < model._R0 * 1.0e6
    assert width_axis.lines[2].get_ydata()[-1] == pytest.approx(
        model.getRightBoundary() - model.getInterfacePositions()[1]
    )
    plt.close(profile_figure)
    plt.close(width_figure)


@pytest.mark.parametrize("phase_molar_volumes", [(1.0, 1.0, 1.0), (2.0, 3.0, 1.5)])
def test_three_phase_output_metadata_is_complete_but_not_restartable(phase_molar_volumes):
    model = _make_nonstationary_unequal_volume_model(
        record=True,
        phase_molar_volumes=phase_molar_volumes,
    )
    model.solve(model.timeStep)

    data = model.toDict()

    assert int(data["moving_domain_output_version"]) == 1
    assert not bool(data["moving_domain_restart_supported"])
    assert bool(data["phase_molar_volumes_explicit"])
    assert np.array_equal(data["phase_molar_volumes"], model._phaseMolarVolumes)
    assert float(data["initial_right_boundary"]) == model._R0
    assert np.array_equal(data["right_boundary_history"], model.rightBoundaryData._y)
    assert np.array_equal(data["right_boundary_time"], model.rightBoundaryData._time)
    assert np.array_equal(data["molar_inventory_history"], model.molarInventoryData._y)
    assert np.array_equal(data["total_substitutional_moles_history"], model.totalMolesData._y)
    assert np.array_equal(
        data["dependent_mole_closure_error_history"],
        model.dependentMoleClosureErrorData._y,
    )
    for phase_index, grid in enumerate(model._grids):
        assert np.array_equal(data[f"transformed_grid_{phase_index}"], grid)

    restart_model = _make_nonstationary_unequal_volume_model(
        record=True,
        phase_molar_volumes=phase_molar_volumes,
    )
    with pytest.raises(NotImplementedError, match="cannot be restarted"):
        restart_model.fromDict(data)


@pytest.mark.parametrize(
    "phase_molar_volumes",
    [None, (7.2e-6, 7.2e-6, 7.2e-6), (6.8e-6, 8.1e-6, 7.4e-6)],
    ids=("legacy-equal-volume", "explicit-equal-volume", "unequal-realistic-volume"),
)
def test_three_phase_repeated_solve_matches_one_shot_without_reinitializing(phase_molar_volumes):
    one_shot = _make_nonstationary_unequal_volume_model(
        record=True,
        phase_molar_volumes=phase_molar_volumes,
    )
    split = _make_nonstationary_unequal_volume_model(
        record=True,
        phase_molar_volumes=phase_molar_volumes,
    )

    one_shot.solve(4.0e-6)
    split.solve(2.0e-6)
    initial_right_boundary = split._R0
    initial_total_amount = split._initialTotalAmount
    initial_inventory = split._initialInventory.copy()
    initial_molar_inventories = (
        None if split._initialMolarInventories is None else split._initialMolarInventories.copy()
    )
    initial_total_moles = split._initialTotalMoles
    first_history_times = split.interfaceData._time[: split.interfaceData.N + 1].copy()
    split.solve(2.0e-6)

    for actual, expected in zip(split.getTransformedState(), one_shot.getTransformedState()):
        assert np.allclose(actual, expected, rtol=2e-12, atol=2e-13)
    assert np.allclose(split.getInterfacePositions(), one_shot.getInterfacePositions(), rtol=2e-12, atol=2e-13)
    assert np.allclose(split.getInterfaceEtas(), one_shot.getInterfaceEtas(), rtol=2e-12, atol=2e-13)
    assert split.getRightBoundary() == pytest.approx(one_shot.getRightBoundary(), rel=2e-12, abs=2e-13)
    assert split.currentTime == pytest.approx(one_shot.currentTime, rel=0.0, abs=1e-18)
    assert split.currentTime == pytest.approx(4.0e-6, rel=0.0, abs=1e-18)
    assert split._R0 == initial_right_boundary
    assert split._initialTotalAmount == initial_total_amount
    assert np.array_equal(split._initialInventory, initial_inventory)
    assert np.array_equal(split.interfaceData._time[: len(first_history_times)], first_history_times)
    assert np.array_equal(
        split.interfaceData._time[: split.interfaceData.N + 1],
        np.asarray([0.0, 2.0e-6, 4.0e-6]),
    )
    if phase_molar_volumes is None:
        assert split._initialMolarInventories is None
        assert split._initialTotalMoles is None
        assert np.allclose(split.getTotalInventory(), one_shot.getTotalInventory(), rtol=2e-12, atol=2e-13)
    else:
        assert np.array_equal(split._initialMolarInventories, initial_molar_inventories)
        assert split._initialTotalMoles == initial_total_moles
        assert np.allclose(split.getAllComponentMoles(), one_shot.getAllComponentMoles(), rtol=2e-12, atol=2e-8)
        assert split.getTotalSubstitutionalMoles() == pytest.approx(
            one_shot.getTotalSubstitutionalMoles(), rel=2e-12, abs=2e-8
        )
        molar_drift = split.checkMolarConservation(2e-6)
        assert np.all(molar_drift <= 2e-6)
        assert np.all(
            molar_drift / np.maximum(np.abs(split._initialMolarInventories), 1.0)
            <= 6e-11
        )


def test_three_phase_reset_explicitly_reinitializes_after_continuation():
    model = _make_nonstationary_unequal_volume_model(
        record=True,
        phase_molar_volumes=(6.8e-6, 8.1e-6, 7.4e-6),
    )
    reference = _make_nonstationary_unequal_volume_model(
        record=True,
        phase_molar_volumes=(6.8e-6, 8.1e-6, 7.4e-6),
    )
    model.solve(4.0e-6)

    model.reset()
    assert not model.isSetup
    assert model.currentTime == 0.0
    assert model._R0 is None
    assert model._R is None
    assert model._profiles_curr is None
    model.solve(2.0e-6)
    reference.solve(2.0e-6)

    assert model.isSetup
    for actual, expected in zip(model.getTransformedState(), reference.getTransformedState()):
        assert np.allclose(actual, expected, rtol=2e-12, atol=2e-13)
    assert np.allclose(model.getInterfacePositions(), reference.getInterfacePositions(), rtol=2e-12, atol=2e-13)
    assert np.allclose(model.getInterfaceEtas(), reference.getInterfaceEtas(), rtol=2e-12, atol=2e-13)
    assert model.getRightBoundary() == pytest.approx(reference.getRightBoundary(), rel=2e-12, abs=2e-13)
    assert model.currentTime == pytest.approx(2.0e-6, rel=0.0, abs=1e-18)


def test_three_phase_inherited_fixed_mesh_flux_api_is_explicitly_unsupported():
    model, _ = _make_stationary_model(record=False)
    model.setup()

    with pytest.raises(NotImplementedError, match="transformed three-phase ALE"):
        model.getFluxes()


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

    class DebugStub:
        @staticmethod
        def debugInPlace():
            pass

    monkeypatch.setattr(model, "_solve_interface_planar", fake_solve)
    monkeypatch.setitem(__import__("sys").modules, "examples.debugInPlace", DebugStub)

    with pytest.raises(RuntimeError, match="Three-phase Illingworth step failed after timestep retries"):
        model.getdXdt(0.0, x_curr)

    assert np.allclose(calls, [1e-4, 5e-5], rtol=0.0, atol=1e-18)


def test_three_phase_prompt_policy_reports_input_read_failure(monkeypatch):
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

    def fake_solve(profiles, interfaces, etas, dt):
        raise RuntimeError("forced retry")

    def fake_input(prompt):
        raise OSError("input unavailable")

    monkeypatch.setattr(model, "_solve_interface_planar", fake_solve)
    monkeypatch.setattr("builtins.input", fake_input)

    with pytest.warns(RuntimeWarning, match="below terminal_thin_phase_width"):
        with pytest.raises(RuntimeError, match="could not read a response from stdin"):
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


@pytest.mark.parametrize("phase_molar_volumes", [None, (2.0, 3.0, 1.5)], ids=("equal-volume", "unequal-volume"))
def test_three_phase_initial_residual_is_finite_step_residual_limit(phase_molar_volumes):
    model = _make_eta_varying_three_phase_model(record=False, phase_molar_volumes=phase_molar_volumes)
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
    for epsilon in (4.0e-7, 2.0e-7, 1.0e-7):
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


def test_three_phase_unequal_volume_arbitrary_candidate_satisfies_discrete_molar_telescope():
    molar_volumes = np.asarray([2.0, 3.0, 1.5], dtype=np.float64)
    model = _make_eta_varying_three_phase_model(
        record=False,
        phase_molar_volumes=molar_volumes,
    )
    model.setup()
    old_interfaces = np.asarray([0.35, 0.70], dtype=np.float64)
    old_interface_compositions = model._interface_compositions(np.asarray([0.2, 0.8]))
    c_a_ab_old, c_b_ab_old = old_interface_compositions[0]
    c_b_bc_old, c_c_bc_old = old_interface_compositions[1]
    old_profiles = (
        np.linspace(np.asarray([0.17, 0.07]), c_a_ab_old, len(model._grids[0])),
        np.linspace(c_b_ab_old, c_b_bc_old, len(model._grids[1])),
        np.linspace(c_c_bc_old, np.asarray([0.31, 0.17]), len(model._grids[2])),
    )
    old_profiles[0][1:-1] += np.asarray([0.01, -0.004])
    old_profiles[1][1:-1] += np.asarray([-0.006, 0.008])
    old_profiles[2][1:-1] += np.asarray([0.008, -0.005])
    new_interfaces = np.asarray([0.41, 0.66], dtype=np.float64)
    new_interface_compositions = model._interface_compositions(np.asarray([0.73, 0.41]))
    model._currdt = 1e-5

    kinematics = model._step_kinematics(old_interfaces, new_interfaces)
    bulk_results = model._solve_bulk_profiles(
        old_profiles,
        old_interfaces,
        new_interfaces,
        new_interface_compositions,
        kinematics=kinematics,
    )
    new_profiles = tuple(result.profile for result in bulk_results)
    residual = model._interface_residuals(
        old_profiles,
        old_interfaces,
        new_interfaces,
        new_interface_compositions,
        bulk_results,
        kinematics=kinematics,
    )
    old_bounds = (
        (0.0, old_interfaces[0]),
        (old_interfaces[0], old_interfaces[1]),
        (old_interfaces[1], kinematics.old_right_boundary),
    )
    new_bounds = (
        (0.0, new_interfaces[0]),
        (new_interfaces[0], new_interfaces[1]),
        (new_interfaces[1], kinematics.new_right_boundary),
    )
    boundary_values = (
        (None, new_interface_compositions[0][0]),
        (new_interface_compositions[0][1], new_interface_compositions[1][0]),
        (new_interface_compositions[1][1], None),
    )

    for phase_index in range(3):
        D_faces = np.broadcast_to(np.eye(2), (len(model._grids[phase_index]) - 1, 2, 2))
        cell_residual = _interval_fv_residual(
            old_profiles[phase_index],
            new_profiles[phase_index],
            model._grids[phase_index],
            old_bounds[phase_index],
            new_bounds[phase_index],
            D_faces,
            model._currdt,
            left_value=boundary_values[phase_index][0],
            right_value=boundary_values[phase_index][1],
            phase_frame_displacement=kinematics.phase_displacements[phase_index],
        )
        assert np.allclose(cell_residual, 0.0, rtol=0.0, atol=8e-16)

    old_lengths = np.asarray([right - left for left, right in old_bounds])
    new_lengths = np.asarray([right - left for left, right in new_bounds])
    c_a_ab, c_b_ab = new_interface_compositions[0]
    c_b_bc, c_c_bc = new_interface_compositions[1]
    endpoints = (
        (
            np.zeros(2),
            model._endpoint_inventory_change(model._grids[0], old_profiles[0], c_a_ab, old_lengths[0], new_lengths[0], "right"),
        ),
        (
            model._endpoint_inventory_change(model._grids[1], old_profiles[1], c_b_ab, old_lengths[1], new_lengths[1], "left"),
            model._endpoint_inventory_change(model._grids[1], old_profiles[1], c_b_bc, old_lengths[1], new_lengths[1], "right"),
        ),
        (
            model._endpoint_inventory_change(model._grids[2], old_profiles[2], c_c_bc, old_lengths[2], new_lengths[2], "left"),
            np.zeros(2),
        ),
    )
    external_left = np.zeros(2)
    external_right = np.zeros(2)
    phase_transfer_balances = (
        endpoints[0][1] + bulk_results[0].right_transfer - external_left,
        endpoints[1][0] + endpoints[1][1] + bulk_results[1].right_transfer - bulk_results[1].left_transfer,
        endpoints[2][0] + external_right - bulk_results[2].left_transfer,
    )
    normalized_phase_changes = []
    for phase_index in range(3):
        direct_change = new_lengths[phase_index] * np.trapezoid(new_profiles[phase_index], model._grids[phase_index], axis=0)
        direct_change -= old_lengths[phase_index] * np.trapezoid(old_profiles[phase_index], model._grids[phase_index], axis=0)
        normalized_phase_changes.append(model._normalizedPhaseMolarDensities[phase_index] * direct_change)
        assert np.allclose(
            normalized_phase_changes[-1],
            model._normalizedPhaseMolarDensities[phase_index] * phase_transfer_balances[phase_index],
            rtol=0.0,
            atol=7e-16,
        )

    reconstructed_ab = model._normalizedPhaseMolarDensities[0] * (endpoints[0][1] + bulk_results[0].right_transfer)
    reconstructed_ab += model._normalizedPhaseMolarDensities[1] * (endpoints[1][0] - bulk_results[1].left_transfer)
    reconstructed_bc = model._normalizedPhaseMolarDensities[1] * (endpoints[1][1] + bulk_results[1].right_transfer)
    reconstructed_bc += model._normalizedPhaseMolarDensities[2] * (endpoints[2][0] - bulk_results[2].left_transfer)
    assert np.allclose(residual[:2], reconstructed_ab, rtol=0.0, atol=0.0)
    assert np.allclose(residual[2:], reconstructed_bc, rtol=0.0, atol=0.0)

    old_moles = integrate_planar_transformed_molar_inventories(
        old_profiles, old_interfaces, kinematics.old_right_boundary, model._grids, molar_volumes
    )
    new_moles = integrate_planar_transformed_molar_inventories(
        new_profiles, new_interfaces, kinematics.new_right_boundary, model._grids, molar_volumes
    )
    physical_independent_change = new_moles[1:] - old_moles[1:]
    normalized_global_change = np.sum(np.asarray(normalized_phase_changes), axis=0)

    assert np.max(np.abs(residual[:2] + residual[2:])) > 1e-5
    assert np.allclose(normalized_global_change, residual[:2] + residual[2:], rtol=0.0, atol=7e-16)
    assert np.allclose(
        physical_independent_change,
        model._phaseMolarDensities[0] * (residual[:2] + residual[2:]),
        rtol=0.0,
        atol=4e-16,
    )
    assert np.isclose(np.sum(new_moles), np.sum(old_moles), rtol=0.0, atol=5e-16)
    assert np.isclose(new_moles[0] - old_moles[0], -np.sum(physical_independent_change), rtol=0.0, atol=5e-16)


def test_three_phase_unequal_volume_nonlinear_step_converges_and_conserves_direct_moles():
    molar_volumes = np.asarray([2.0, 3.0, 1.5], dtype=np.float64)
    model = _make_eta_varying_three_phase_model(
        record=False,
        phase_molar_volumes=molar_volumes,
    )
    model.setup()
    state = _nonuniform_moving_candidate_state(model)
    old_profiles = tuple(np.asarray(profile, dtype=np.float64) for profile in state[:3])
    old_interfaces = np.asarray(state[3], dtype=np.float64)
    old_etas = np.asarray(state[4], dtype=np.float64)
    model._currdt = model.timeStep

    candidate = model._solve_interface_planar(old_profiles, old_interfaces, old_etas, model.timeStep)
    old_right = model._right_boundary_from_interfaces(old_interfaces)
    old_moles = integrate_planar_transformed_molar_inventories(
        old_profiles, old_interfaces, old_right, model._grids, molar_volumes
    )
    new_moles = integrate_planar_transformed_molar_inventories(
        candidate.profiles,
        candidate.interfaces,
        candidate.kinematics.new_right_boundary,
        model._grids,
        molar_volumes,
    )

    assert candidate.scaled_norm <= model.residualTolerance
    assert np.max(np.abs(candidate.residual[:2])) <= model.residualTolerance * model._residual_scale(old_profiles, old_interfaces)[0]
    assert np.max(np.abs(candidate.residual[2:])) <= model.residualTolerance * model._residual_scale(old_profiles, old_interfaces)[0]
    assert np.allclose(new_moles[1:], old_moles[1:], rtol=0.0, atol=8e-11)
    assert np.isclose(new_moles[0], old_moles[0], rtol=0.0, atol=8e-11)
    assert np.isclose(np.sum(new_moles), np.sum(old_moles), rtol=0.0, atol=5e-16)


@pytest.mark.parametrize("mode", [_BULK_DIFFUSIVITY_LAGGED, _BULK_DIFFUSIVITY_IMPLICIT])
def test_three_phase_unequal_volume_variable_diffusivity_modes_use_relative_displacement(mode):
    model = _make_eta_varying_three_phase_model(
        mode=mode,
        record=False,
        phase_molar_volumes=(2.0, 3.0, 1.5),
    )
    model.setup()
    state = _nonuniform_moving_candidate_state(model)
    old_profiles = tuple(np.asarray(profile, dtype=np.float64) for profile in state[:3])
    old_interfaces = np.asarray(state[3], dtype=np.float64)
    new_interfaces = np.asarray([0.39, 0.67], dtype=np.float64)
    interface_compositions = model._interface_compositions(np.asarray([0.63, 0.27]))
    model._currdt = model.timeStep

    kinematics = model._step_kinematics(old_interfaces, new_interfaces)
    results = model._solve_bulk_profiles(
        old_profiles,
        old_interfaces,
        new_interfaces,
        interface_compositions,
        kinematics=kinematics,
    )
    old_bounds = (
        (0.0, old_interfaces[0]),
        (old_interfaces[0], old_interfaces[1]),
        (old_interfaces[1], kinematics.old_right_boundary),
    )
    new_bounds = (
        (0.0, new_interfaces[0]),
        (new_interfaces[0], new_interfaces[1]),
        (new_interfaces[1], kinematics.new_right_boundary),
    )
    boundary_values = (
        (None, interface_compositions[0][0]),
        (interface_compositions[0][1], interface_compositions[1][0]),
        (interface_compositions[1][1], None),
    )
    for phase_index, result in enumerate(results):
        D_faces = np.broadcast_to(np.eye(2), (len(model._grids[phase_index]) - 1, 2, 2))
        residual = _interval_fv_residual(
            old_profiles[phase_index],
            result.profile,
            model._grids[phase_index],
            old_bounds[phase_index],
            new_bounds[phase_index],
            D_faces,
            model._currdt,
            left_value=boundary_values[phase_index][0],
            right_value=boundary_values[phase_index][1],
            phase_frame_displacement=kinematics.phase_displacements[phase_index],
        )
        assert np.allclose(residual, 0.0, rtol=0.0, atol=8e-16)


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
