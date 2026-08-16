from types import SimpleNamespace

import numpy as np
import pytest

from kawin.diffusion import MovingBoundaryIllingworthTernaryThreePhaseFD1DModel
from kawin.diffusion.MovingBoundaryIllingworthTernaryThreePhaseFDM import _BULK_DIFFUSIVITY_IMPLICIT, _BULK_DIFFUSIVITY_LAGGED, _BULK_DIFFUSIVITY_PHASE_UNIFORM
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder
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
