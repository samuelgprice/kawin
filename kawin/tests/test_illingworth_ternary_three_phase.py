from types import SimpleNamespace

import numpy as np
import pytest

from kawin.diffusion import MovingBoundaryIllingworthTernaryThreePhaseFD1DModel
from kawin.diffusion import MovingBoundaryIllingworthTernaryFD1DModel
from kawin.diffusion.MovingBoundaryIllingworthTernaryThreePhaseFDM import _BULK_DIFFUSIVITY_IMPLICIT, _BULK_DIFFUSIVITY_LAGGED, _BULK_DIFFUSIVITY_PHASE_UNIFORM
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


def _make_stationary_model(*, mode=_BULK_DIFFUSIVITY_PHASE_UNIFORM, record=True, interfaces=(0.35, 0.7)):
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

    assert np.allclose(inventory, 0.25 * profiles[0][0] + 0.5 * profiles[1][0] + 0.25 * profiles[2][0])
    assert np.allclose(reconstructed[0], profiles[0][0])
    assert np.allclose(reconstructed[2], profiles[0][0])
    assert np.allclose(reconstructed[3], profiles[1][0])
    assert np.allclose(reconstructed[-1], profiles[2][0])


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

    assert not np.allclose(result.profile[1:-1], old_profile[1:-1])
    assert np.allclose(residual, 0.0, atol=2e-15)


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
    assert np.allclose(residual, 0.0, atol=2e-15)
    assert np.max(np.abs(all_right_residual)) > 1e-4
    assert np.max(np.abs(all_left_residual)) > 1e-4


def test_three_phase_left_interval_reduces_to_two_phase_left_moving_grid_solve():
    c_left = np.asarray([0.26, 0.11], dtype=np.float64)
    c_right = np.asarray([0.34, 0.16], dtype=np.float64)
    grid = np.asarray([0.0, 0.2, 0.5, 0.75, 1.0], dtype=np.float64)
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
    D = np.asarray([[1.3, 0.08], [0.04, 0.9]], dtype=np.float64)
    old_s = 0.45
    new_s = 0.52
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

    generic = three_phase._solve_interval_planar(profile, grid, [0.0, old_s], [0.0, new_s], 0, None, c_left, D)
    reference = two_phase._solve_concentration_left_planar(profile, old_s, new_s, dt, c_left, D, "positive")

    assert np.allclose(generic.profile, reference.profile, atol=1e-13)


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

    assert np.allclose(model.getInterfacePositions(), [0.35, 0.7], atol=1e-12)
    assert np.allclose(model.getInterfaceEtas(), [0.25, 0.75], atol=1e-12)
    assert np.allclose(model.checkConservation(1e-10), [0.0, 0.0], atol=1e-12)
    profiles = model.getTransformedState()
    assert len(profiles) == 3
    assert model.interfaceData._y[: model.interfaceData.N + 1].shape[1] == 2
    assert model.etaData._y[: model.etaData.N + 1].shape[1] == 2


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

    assert np.allclose(model.getInterfaceEtas(), target_etas, atol=1e-8)
    assert np.allclose(model.initialEtaEstimate.etas, target_etas, atol=1e-8)
    assert model.initialEtaEstimate.residual_norm <= model.initialEtaRootXtol


@pytest.mark.parametrize("mode", [_BULK_DIFFUSIVITY_PHASE_UNIFORM, _BULK_DIFFUSIVITY_LAGGED, _BULK_DIFFUSIVITY_IMPLICIT])
def test_three_phase_moving_raw_candidate_conserves_inventory_without_correction(mode):
    model = _make_eta_varying_three_phase_model(mode=mode, record=True)
    model.setup()
    state = _nonuniform_moving_candidate_state(model)
    old_inventory = model.getTotalInventoryFromState(tuple(state[:3]), state[3])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("raw candidate profiles should not be inventory-corrected")

    model._correct_candidate_inventory = fail_if_called
    dXdt = model.getdXdt(model.currentTime, state)
    raw_profiles = tuple(state[i] + model._currdt * dXdt[i] for i in range(3))
    raw_interfaces = state[3] + model._currdt * dXdt[3]
    raw_inventory = model.getTotalInventoryFromState(raw_profiles, raw_interfaces)

    assert np.max(np.abs(dXdt[3])) > 0.0
    assert np.max(np.abs(dXdt[4])) > 0.0
    assert model._lastImplicitResidual <= model.residualTolerance
    assert np.allclose(raw_inventory, old_inventory, atol=2e-15)


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
                left_flux=zero_flux.copy(),
                right_flux=zero_flux.copy(),
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
