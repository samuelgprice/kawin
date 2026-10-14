"""Interstitial (u-fraction) support in the ternary three-phase Illingworth model."""

import numpy as np
import pytest

from kawin.diffusion import MovingBoundaryIllingworthTernaryThreePhaseFD1DModel
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder
from kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D import integrate_planar_transformed_molar_inventories

INTERFACES = (0.35, 0.7)
PHASE_U = (
    np.asarray([0.21, 0.09], dtype=np.float64),
    np.asarray([0.31, 0.16], dtype=np.float64),
    np.asarray([0.20, 0.21], dtype=np.float64),
)


class _LinearPairEquilibrium:
    eta_bounds = (0.0, 1.0)
    composition_basis = "u"

    def __init__(self, left_base, left_slope, right_base, right_slope):
        self.left_base = np.asarray(left_base, dtype=np.float64)
        self.left_slope = np.asarray(left_slope, dtype=np.float64)
        self.right_base = np.asarray(right_base, dtype=np.float64)
        self.right_slope = np.asarray(right_slope, dtype=np.float64)

    def interface_compositions(self, eta):
        eta = float(eta)
        return self.left_base + eta * self.left_slope, self.right_base + eta * self.right_slope


class _ThreePhaseStepProfile:
    """Piecewise-constant mole-fraction start built from u-fraction plateaus (first component is C)."""

    def __call__(self, z):
        x = np.asarray(z, dtype=np.float64).reshape((-1, 1))[:, 0]
        out = np.empty((len(x), 2), dtype=np.float64)
        for mask, u in (
            (x < INTERFACES[0], PHASE_U[0]),
            ((x >= INTERFACES[0]) & (x < INTERFACES[1]), PHASE_U[1]),
            (x >= INTERFACES[1], PHASE_U[2]),
        ):
            out[mask] = u / (1.0 + u[0])
        return out


class _SpyThreePhaseThermodynamics:
    """Constant u-basis matrices with a fast first (interstitial) component."""

    def __init__(self):
        self.received = []

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        self.received.append(values.copy())
        scale = {"A": 1.0, "B": 0.7, "C": 1.3}[phase]
        matrix = scale * np.asarray([[1.0, 0.02], [0.0, 0.05]], dtype=np.float64)
        if values.ndim == 2:
            return np.broadcast_to(matrix, (values.shape[0], 2, 2)).copy()
        return matrix


def _model(mode="phase_uniform"):
    eq_ab = _LinearPairEquilibrium([0.20, 0.10], [0.10, 0.0], [0.30, 0.15], [0.10, 0.0])
    eq_bc = _LinearPairEquilibrium([0.24, 0.15], [0.10, 0.0], [0.18, 0.20], [0.10, 0.0])
    mesh = CartesianFD1D(["C", "MN"], [0.0, 1.0], 31)
    mesh.setResponseProfile(ProfileBuilder([(_ThreePhaseStepProfile(), ["C", "MN"])]))
    therm = _SpyThreePhaseThermodynamics()
    model = MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh,
        elements=["FE", "C", "MN"],
        phases=["A", "B", "C"],
        thermodynamics=therm,
        temperature=1000.0,
        interfacePositions=INTERFACES,
        interface_equilibria=(eq_ab, eq_bc),
        initial_eta_guess=(0.5, 0.5),
        bulk_diffusivity_mode=mode,
        phase_molar_volumes=(2.0, 3.0, 1.5),
        phase_nodes=(9, 9, 9),
        time_step=1e-4,
        record=True,
        tolerance=1e-10,
        residual_tolerance=1e-10,
        max_iterations=50,
    )
    return model, therm


def test_masked_molar_inventory_matches_hand_integration():
    grids = (np.linspace(0, 1, 5),) * 2
    profiles = (np.tile([0.5, 0.2], (5, 1)), np.tile([0.1, 0.3], (5, 1)))
    unmasked = integrate_planar_transformed_molar_inventories(profiles, [0.4], 1.0, grids, (2.0, 4.0))
    masked = integrate_planar_transformed_molar_inventories(profiles, [0.4], 1.0, grids, (2.0, 4.0), substitutional_mask=[False, True])
    np.testing.assert_allclose(unmasked, [0.4 * 0.3 / 2 + 0.6 * 0.6 / 4, 0.4 * 0.5 / 2 + 0.6 * 0.1 / 4, 0.4 * 0.2 / 2 + 0.6 * 0.3 / 4])
    # u_C may exceed 1 - u_Mn: the dependent u is 1 - u_Mn only.
    np.testing.assert_allclose(masked, [0.4 * 0.8 / 2 + 0.6 * 0.7 / 4, 0.4 * 0.5 / 2 + 0.6 * 0.1 / 4, 0.4 * 0.2 / 2 + 0.6 * 0.3 / 4])
    widths_over_v = 0.4 / 2 + 0.6 / 4
    assert masked[0] + masked[2] == pytest.approx(widths_over_v, rel=1e-14)


@pytest.mark.parametrize("mode", ["phase_uniform", "composition_dependent_lagged"])
def test_three_phase_interstitial_state_queries_and_conservation(mode):
    model, therm = _model(mode)
    z = np.ravel(model.mesh.z)
    expected = np.where((z < INTERFACES[0])[:, None], PHASE_U[0], np.where((z < INTERFACES[1])[:, None], PHASE_U[1], PHASE_U[2]))
    np.testing.assert_allclose(model.data.currentY, expected, rtol=1e-14)
    model.reset()
    np.testing.assert_allclose(model.data.currentY, expected, rtol=1e-14)

    model.solve(2.0e-3, minDtFrac=1.0e-10)
    assert model.interfaceData.N > 3

    # Every query received mole fractions: x = u / (1 + u_C) for the accepted interface states.
    therm.received.clear()
    comps = model._interface_compositions(model.getInterfaceEtas())
    model._interface_face_diffusivity(comps[0][1], 1, float(model.getInterfacePositions()[0]), "interface")
    np.testing.assert_allclose(therm.received[-1], comps[0][1] / (1.0 + comps[0][1][0]), rtol=1e-14)

    # Component moles are conserved (the legacy length-weighted inventory is not,
    # because unequal phase molar volumes move the right boundary).
    moles = model.molarInventoryData._y[: model.molarInventoryData.N + 1]
    np.testing.assert_allclose(moles - moles[0], 0.0, atol=1e-10)
    # Substitutional moles (Fe + Mn) close against the geometric total; carbon is extra.
    totals = model.totalMolesData._y[: model.totalMolesData.N + 1]
    np.testing.assert_allclose(moles[:, 0] + moles[:, 2], totals, rtol=1e-12)
    closure = model.dependentMoleClosureErrorData._y[: model.dependentMoleClosureErrorData.N + 1]
    np.testing.assert_allclose(closure, 0.0, atol=1e-12)

    # Outputs are full mole fractions.
    for _, phase_comps in model.getPhysicalPhaseProfiles():
        np.testing.assert_allclose(np.sum(phase_comps, axis=1), 1.0, atol=1e-14)
    full = model.getCompositions()
    valid = np.all(np.isfinite(full), axis=1)
    np.testing.assert_allclose(np.sum(full[valid], axis=1), 1.0, atol=1e-14)
    u = model.data.y()[valid]
    np.testing.assert_allclose(full[valid][:, 1:], u / (1.0 + u[:, :1]), rtol=1e-13)
