from pathlib import Path

import numpy as np
import pytest

from kawin.diffusion import (
    MovingBoundaryIllingworthTernaryFD1DModel,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    estimate_initial_eta_from_instantaneous_balance,
    estimate_initial_eta_from_stefan_residual,
)
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D import (
    integrate_planar_transformed_profile_components,
    solve_illingworth_block_tridiagonal,
)


class _ConstantTernaryThermodynamics:
    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        if phase == "ALPHA":
            return np.asarray([[1.0e-15, 1.0e-16], [2.0e-16, 0.8e-15]], dtype=np.float64)
        return np.asarray([[0.7e-15, -0.5e-16], [0.1e-16, 1.2e-15]], dtype=np.float64)


class _IdentityTernaryThermodynamics:
    def __init__(self, invalid=False):
        self.invalid = invalid

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        if self.invalid:
            return np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        return np.eye(2, dtype=np.float64)


class _CoupledTernaryThermodynamics:
    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        if phase == "ALPHA":
            return np.asarray([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]], dtype=np.float64)
        return np.asarray([[7.0e-4, -1.0e-4], [2.0e-4, 1.1e-3]], dtype=np.float64)


class _LinearInterfaceEquilibrium:
    eta_bounds = (0.0, 1.0)

    def __init__(self, mode="valid"):
        self.mode = mode

    def interface_compositions(self, eta):
        eta = float(eta)
        left = np.asarray([0.20 + 0.10 * eta, 0.10], dtype=np.float64)
        right = np.asarray([0.30 + 0.10 * eta, 0.15], dtype=np.float64)
        if self.mode == "nonfinite":
            left[0] = np.nan
        elif self.mode == "degenerate":
            right = left.copy()
        return left, right


class _TieLineSamplingThermodynamics:
    def __init__(self, endpoint_mode="valid"):
        self.elements = ["Z", "X", "Y"]
        self.phases = ["ALPHA", "BETA"]
        self.endpoint_mode = endpoint_mode

    def clearCache(self):
        pass

    def getInterfacialComposition(self, x, T, gExtra=0, precPhase=None, returnMeta=False):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        eta = float((x[0] - 0.20) / 0.20)
        left = np.asarray([0.20 + 0.10 * eta, 0.10], dtype=np.float64)
        right = np.asarray([0.30 + 0.10 * eta, 0.15], dtype=np.float64)
        endpoints = [
            {"phase": "ALPHA", "composition": left},
            {"phase": "BETA", "composition": right},
        ]
        if self.endpoint_mode == "missing":
            endpoints = [{"phase": "ALPHA", "composition": left}]
        elif self.endpoint_mode == "extra":
            endpoints.append({"phase": "GAMMA", "composition": np.asarray([0.25, 0.05])})
        elif self.endpoint_mode == "duplicate":
            endpoints = [
                {"phase": "ALPHA", "composition": left},
                {"phase": "ALPHA", "composition": left + 0.01},
            ]
        elif self.endpoint_mode == "unlabeled":
            endpoints = [
                {"composition": left},
                {"phase": "BETA", "composition": right},
            ]
        if not returnMeta:
            return left, right
        return left, right, {"endpoint_phases": tuple(e.get("phase") for e in endpoints), "endpoints": tuple(endpoints)}

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        composition = np.asarray(composition, dtype=np.float64).reshape(2)
        if phase == "ALPHA":
            base = 1.0 + composition[0]
            return np.asarray([[base, composition[1]], [0.25 * composition[1], base + 0.5]], dtype=np.float64)
        base = 2.0 + composition[0]
        return np.asarray([[base, -0.5 * composition[1]], [0.1 * composition[1], base + 0.25]], dtype=np.float64)


def _block_times_vector(block, vector):
    return np.asarray(
        [
            block[0, 0] * vector[0] + block[0, 1] * vector[1],
            block[1, 0] * vector[0] + block[1, 1] * vector[1],
        ],
        dtype=np.float64,
    )


def _legacy_interface_residual(model, p_future, q_future, s, old_s, future_s, dt, c_left, c_right, D_left, D_right):
    velocity_probe = future_s - s
    if abs(velocity_probe) <= 1e-15:
        velocity_probe = s - old_s
    diff_l = _block_times_vector(D_left, (c_left - p_future[-2]) / (1.0 - model._u_grid[-2])) / future_s
    diff_r = _block_times_vector(D_right, (q_future[1] - c_right) / model._v_grid[1]) / (model._R - future_s)
    rhs = (diff_r - diff_l) * dt
    if velocity_probe >= 0:
        lhs = c_left - q_future[1] * (1.0 - model._v_grid[1] / 2.0) - c_right * model._v_grid[1] / 2.0
    else:
        lhs = p_future[-2] * (0.5 + model._u_grid[-2] / 2.0) + c_left * (0.5 - model._u_grid[-2] / 2.0) - c_right
    return (future_s - s) * lhs - rhs


def _make_residual_identity_state():
    left, right = _LinearInterfaceEquilibrium().interface_compositions(0.2)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 31)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.45, left, right), ["X", "Y"])]))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=_IdentityTernaryThermodynamics(),
        temperature=1000.0,
        interfacePosition=0.45,
        interface_equilibrium=_LinearInterfaceEquilibrium(),
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(0.0, 1.0),
        time_step=1.0e-4,
        tolerance=1e-12,
        record=True,
    )
    model.setup()
    p = model._p_curr.copy()
    q = model._q_curr.copy()
    p[:-1] += np.asarray([0.005, -0.003], dtype=np.float64) * np.linspace(0.0, 1.0, len(p) - 1)[:, None]
    q[1:] += np.asarray([-0.004, 0.002], dtype=np.float64) * np.linspace(0.0, 1.0, len(q) - 1)[:, None]
    return model, p, q


def test_ternary_block_solve_preserves_component_coupling():
    diagonal_block = np.asarray([[2.0, 0.5], [0.25, 3.0]], dtype=np.float64)
    expected = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    lower = np.zeros((3, 2, 2), dtype=np.float64)
    diagonal = np.repeat(diagonal_block[np.newaxis, :, :], 3, axis=0)
    upper = np.zeros((3, 2, 2), dtype=np.float64)
    rhs = np.asarray([_block_times_vector(diagonal_block, row) for row in expected], dtype=np.float64)

    actual = solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)

    assert np.allclose(actual, expected)


@pytest.mark.parametrize("future_s", [0.47, 0.43])
def test_ternary_interface_residual_matches_inventory_change_when_interface_compositions_change(future_s):
    model, p, q = _make_residual_identity_state()
    s = float(model._s_curr)
    dt = 1.0e-4
    c_left, c_right = model._interface_compositions(0.8)
    c_left_old = p[-1].copy()
    c_right_old = q[0].copy()
    D_left = _CoupledTernaryThermodynamics().getInterdiffusivity(c_left, 1000.0, phase="ALPHA")
    D_right = _CoupledTernaryThermodynamics().getInterdiffusivity(c_right, 1000.0, phase="BETA")
    p_future = model._new_concentration_left_planar(p, s, future_s, dt, c_left, D_left)
    q_future = model._new_concentration_right_planar(q, s, future_s, dt, c_right, D_right)

    residual = model._interface_residual(
        p_future,
        q_future,
        s,
        s,
        future_s,
        dt,
        c_left,
        c_right,
        c_left_old,
        c_right_old,
        D_left,
        D_right,
    )
    old_inventory = integrate_planar_transformed_profile_components(p, q, s, model._R, model._u_grid, model._v_grid)
    future_inventory = integrate_planar_transformed_profile_components(
        p_future,
        q_future,
        future_s,
        model._R,
        model._u_grid,
        model._v_grid,
    )

    assert not np.allclose(c_left, c_left_old)
    assert not np.allclose(c_right, c_right_old)
    assert np.allclose(residual, future_inventory - old_inventory, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("future_s", [0.47, 0.43])
def test_ternary_interface_residual_reduces_to_legacy_formula_when_interface_compositions_do_not_change(future_s):
    model, p, q = _make_residual_identity_state()
    s = float(model._s_curr)
    dt = 1.0e-4
    c_left = p[-1].copy()
    c_right = q[0].copy()
    D_left = _CoupledTernaryThermodynamics().getInterdiffusivity(c_left, 1000.0, phase="ALPHA")
    D_right = _CoupledTernaryThermodynamics().getInterdiffusivity(c_right, 1000.0, phase="BETA")
    p_future = model._new_concentration_left_planar(p, s, future_s, dt, c_left, D_left)
    q_future = model._new_concentration_right_planar(q, s, future_s, dt, c_right, D_right)

    residual = model._interface_residual(
        p_future,
        q_future,
        s,
        s,
        future_s,
        dt,
        c_left,
        c_right,
        p[-1],
        q[0],
        D_left,
        D_right,
    )
    legacy = _legacy_interface_residual(model, p_future, q_future, s, s, future_s, dt, c_left, c_right, D_left, D_right)
    endpoint_change = s * 0.5 * (1.0 - model._u_grid[-2]) * (c_left - p[-1])
    endpoint_change += (model._R - s) * 0.5 * model._v_grid[1] * (c_right - q[0])

    assert np.allclose(endpoint_change, np.zeros(2))
    assert np.allclose(residual, legacy, rtol=1e-13, atol=1e-15)


def test_initial_eta_estimator_selects_known_stefan_minimum():
    eta_true = 0.5
    left = np.asarray([0.20 + 0.10 * eta_true, 0.10], dtype=np.float64)
    right = np.asarray([0.30 + 0.10 * eta_true, 0.15], dtype=np.float64)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))

    estimate = estimate_initial_eta_from_stefan_residual(
        composition=mesh.y,
        z=mesh.z,
        interface_position=0.5,
        phases=["ALPHA", "BETA"],
        thermodynamics=_IdentityTernaryThermodynamics(),
        temperature=1000.0,
        interface_equilibrium=_LinearInterfaceEquilibrium(),
        transformed_u_grid=np.linspace(0.0, 1.0, 5),
        transformed_v_grid=np.linspace(0.0, 1.0, 5),
        eta_bracket=(0.0, 1.0),
    )

    assert np.isclose(estimate.eta, eta_true)
    assert np.isclose(estimate.residual_norm, 0.0)
    assert estimate.residual.shape == (2,)
    assert estimate.flux_delta.shape == (2,)
    assert np.allclose(estimate.left_interface_composition, left)
    assert np.allclose(estimate.right_interface_composition, right)
    assert estimate.converged
    assert estimate.bracket == (0.0, 1.0)


def test_initial_eta_estimator_uses_transformed_adjacent_nodes_not_interface_endpoint():
    eta_true = 0.5
    eta_bad_endpoint = 0.0
    z = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    equilibrium = _LinearInterfaceEquilibrium()
    left_true, right_true = equilibrium.interface_compositions(eta_true)
    left_bad, _ = equilibrium.interface_compositions(eta_bad_endpoint)
    composition = np.asarray(
        [
            left_true,
            left_true,
            left_bad,
            right_true,
            right_true,
        ],
        dtype=np.float64,
    )

    estimate = estimate_initial_eta_from_stefan_residual(
        composition=composition,
        z=z,
        interface_position=0.5,
        phases=["ALPHA", "BETA"],
        thermodynamics=_IdentityTernaryThermodynamics(),
        temperature=1000.0,
        interface_equilibrium=equilibrium,
        transformed_u_grid=np.asarray([0.0, 0.5, 1.0]),
        transformed_v_grid=np.asarray([0.0, 0.5, 1.0]),
        eta_bracket=(0.0, 1.0),
    )

    assert np.isclose(estimate.eta, eta_true)


def test_instantaneous_balance_estimator_solves_velocity_and_eta():
    eta_true = 0.25
    left, right = _LinearInterfaceEquilibrium().interface_compositions(eta_true)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))

    estimate = estimate_initial_eta_from_instantaneous_balance(
        composition=mesh.y,
        z=mesh.z,
        interface_position=0.5,
        phases=["ALPHA", "BETA"],
        thermodynamics=_IdentityTernaryThermodynamics(),
        temperature=1000.0,
        interface_equilibrium=_LinearInterfaceEquilibrium(),
        transformed_u_grid=np.linspace(0.0, 1.0, 5),
        transformed_v_grid=np.linspace(0.0, 1.0, 5),
        eta_bracket=(0.0, 1.0),
    )

    assert np.isclose(estimate.eta, eta_true, atol=1e-10)
    assert np.isclose(estimate.residual_norm, 0.0, atol=1e-10)
    assert np.isclose(estimate.velocity, 0.0, atol=1e-10)
    assert estimate.method == "instantaneous_balance"
    assert estimate.branch in {"positive", "negative"}
    assert estimate.converged


def test_ternary_illingworth_rejects_fixed_interface_compositions():
    left = np.asarray([0.20, 0.10], dtype=np.float64)
    right = np.asarray([0.30, 0.15], dtype=np.float64)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))

    with pytest.raises(ValueError, match="eta-capable"):
        MovingBoundaryIllingworthTernaryFD1DModel(
            mesh=mesh,
            elements=["Z", "X", "Y"],
            phases=["ALPHA", "BETA"],
            thermodynamics=_ConstantTernaryThermodynamics(),
            temperature=1000.0,
            interfacePosition=0.5,
            interface_compositions=(left, right),
            time_step=1.0,
            record=True,
        )


def test_initial_eta_estimator_rejects_collapsed_bounds():
    class _CollapsedEquilibrium(_LinearInterfaceEquilibrium):
        eta_bounds = (0.0, 0.0)

    with pytest.raises(ValueError, match="eta_bounds"):
        estimate_initial_eta_from_stefan_residual(
            composition=np.asarray([[0.2, 0.1], [0.3, 0.15], [0.3, 0.15]], dtype=np.float64),
            z=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
            interface_position=0.5,
            phases=["ALPHA", "BETA"],
            thermodynamics=_IdentityTernaryThermodynamics(),
            temperature=1000.0,
            interface_equilibrium=_CollapsedEquilibrium(),
            transformed_u_grid=np.asarray([0.0, 0.5, 1.0]),
            transformed_v_grid=np.asarray([0.0, 0.5, 1.0]),
        )


@pytest.mark.parametrize(
    "equilibrium, thermodynamics, match",
    [
        (_LinearInterfaceEquilibrium(mode="nonfinite"), _IdentityTernaryThermodynamics(), "non-finite"),
        (_LinearInterfaceEquilibrium(mode="degenerate"), _IdentityTernaryThermodynamics(), "degenerate"),
        (_LinearInterfaceEquilibrium(), _IdentityTernaryThermodynamics(invalid=True), "positive real eigenvalues"),
    ],
)
def test_initial_eta_estimator_rejects_invalid_candidates(equilibrium, thermodynamics, match):
    with pytest.raises(ValueError, match=match):
        estimate_initial_eta_from_stefan_residual(
            composition=np.asarray([[0.2, 0.1], [0.25, 0.1], [0.35, 0.15], [0.35, 0.15]], dtype=np.float64),
            z=np.asarray([0.0, 0.25, 0.75, 1.0], dtype=np.float64),
            interface_position=0.5,
            phases=["ALPHA", "BETA"],
            thermodynamics=thermodynamics,
            temperature=1000.0,
            interface_equilibrium=equilibrium,
            transformed_u_grid=np.asarray([0.0, 0.5, 1.0]),
            transformed_v_grid=np.asarray([0.0, 0.5, 1.0]),
            eta_bracket=(0.0, 1.0),
        )


def test_ternary_illingworth_setup_estimates_and_records_initial_eta():
    eta_true = 0.5
    left, right = _LinearInterfaceEquilibrium().interface_compositions(eta_true)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=_IdentityTernaryThermodynamics(),
        temperature=1000.0,
        interfacePosition=0.5,
        interface_equilibrium=_LinearInterfaceEquilibrium(),
        initial_eta_bracket=(0.0, 1.0),
        time_step=1.0,
        record=True,
    )

    model.setup()

    assert np.isclose(model.getInterfacePosition(), 0.5)
    assert np.isclose(model.initialEta, eta_true)
    assert np.isclose(model.getInterfaceEta(), eta_true)
    assert np.isclose(model.etaData._y[0], eta_true)
    assert np.allclose(model.getInterfaceCompositions()[0], left)
    assert np.allclose(model.getInterfaceCompositions()[1], right)
    assert model.initialEtaEstimate is not None


def test_ternary_illingworth_can_use_instantaneous_initial_eta_method():
    eta_true = 0.25
    left, right = _LinearInterfaceEquilibrium().interface_compositions(eta_true)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=_IdentityTernaryThermodynamics(),
        temperature=1000.0,
        interfacePosition=0.5,
        interface_equilibrium=_LinearInterfaceEquilibrium(),
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(0.0, 1.0),
        time_step=1.0,
        record=True,
    )

    model.setup()

    assert np.isclose(model.initialEta, eta_true, atol=1e-10)
    assert model.initialEtaEstimate.method == "instantaneous_balance"


def test_ternary_illingworth_conserves_inventory_with_eta_dependent_interface_compositions():
    left = np.asarray([0.25, 0.12], dtype=np.float64)
    right = np.asarray([0.33, 0.15], dtype=np.float64)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.45, left, right), ["X", "Y"])]))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=_CoupledTernaryThermodynamics(),
        temperature=1000.0,
        interfacePosition=0.45,
        interface_equilibrium=_LinearInterfaceEquilibrium(),
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(0.0, 1.0),
        time_step=1.0e-4,
        tolerance=1.0e-10,
        max_iterations=40,
        record=True,
    )

    model.solve(1.0e-2, minDtFrac=1.0e-10)

    inventory = np.asarray(model.inventoryData._y[: model.inventoryData.N + 1], dtype=np.float64)
    eta = np.asarray(model.etaData._y[: model.etaData.N + 1], dtype=np.float64)
    assert np.ptp(eta) > 1.0e-8
    assert np.allclose(inventory, inventory[0], rtol=0.0, atol=1.0e-11)


def _build_surrogate(**kwargs):
    params = {
        "thermodynamics": _TieLineSamplingThermodynamics(),
        "elements": ["Z", "X", "Y"],
        "phases": ["ALPHA", "BETA"],
        "tieline_phases": ("ALPHA", "BETA"),
        "temperature": 1000.0,
        "probe_start": np.asarray([0.20, 0.10], dtype=np.float64),
        "probe_end": np.asarray([0.40, 0.20], dtype=np.float64),
        "eta_samples": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    }
    params.update(kwargs)
    return TernaryMovingBoundaryThermodynamicsSurrogate.from_database(**params)


def test_ternary_surrogate_requires_explicit_tieline_phases():
    with pytest.raises(ValueError, match="tieline_phases"):
        _build_surrogate(tieline_phases=None)


@pytest.mark.parametrize("endpoint_mode", ["missing", "extra", "duplicate", "unlabeled"])
def test_ternary_surrogate_rejects_unexpected_tieline_phase_metadata(endpoint_mode):
    with pytest.raises(ValueError):
        _build_surrogate(thermodynamics=_TieLineSamplingThermodynamics(endpoint_mode=endpoint_mode))


def test_ternary_surrogate_interpolates_tielines_and_returns_metadata():
    surrogate = _build_surrogate()

    left, right = surrogate.interface_compositions(0.25)
    left_meta, right_meta, meta = surrogate.getInterfacialComposition(0.25, T=1000.0, returnMeta=True)

    assert np.allclose(left, np.asarray([0.225, 0.10]))
    assert np.allclose(right, np.asarray([0.325, 0.15]))
    assert np.allclose(left_meta, left)
    assert np.allclose(right_meta, right)
    assert meta["endpoint_phases"] == ("ALPHA", "BETA")
    assert meta["endpoints"][0]["phase"] == "ALPHA"
    assert meta["endpoints"][1]["phase"] == "BETA"


def test_ternary_surrogate_finds_tieline_for_global_composition():
    surrogate = _build_surrogate()
    expected_left, expected_right = surrogate.interface_compositions(0.25)
    global_comp = expected_left + 0.4 * (expected_right - expected_left)

    left, right, meta = surrogate.getTielineOfGlobalComposition(global_comp, T=1000.0, returnMeta=True)

    assert np.allclose(left, expected_left)
    assert np.allclose(right, expected_right)
    assert np.isclose(meta["eta"], 0.25)
    assert np.isclose(meta["phase_fraction"], 0.4)
    assert meta["phase_fraction_phase"] == "BETA"
    assert meta["endpoint_phases"] == ("ALPHA", "BETA")
    assert np.allclose(meta["global_composition"], global_comp)
    assert np.allclose(meta["residual"], np.zeros(2), atol=1.0e-10)


def test_ternary_surrogate_finds_tieline_for_full_ternary_global_composition():
    surrogate = _build_surrogate()
    expected_left, expected_right = surrogate.interface_compositions(0.75)
    independent_global = expected_left + 0.25 * (expected_right - expected_left)
    full_global = np.asarray([1.0 - np.sum(independent_global), *independent_global], dtype=np.float64)

    left, right = surrogate.getTielineOfGlobalComposition(full_global, T=1000.0)

    assert np.allclose(left, expected_left)
    assert np.allclose(right, expected_right)


def test_ternary_surrogate_rejects_global_composition_on_tieline_extension():
    surrogate = _build_surrogate()
    left, right = surrogate.interface_compositions(0.25)
    global_comp = left - 0.2 * (right - left)

    with pytest.raises(ValueError, match="tie-line extension"):
        surrogate.getTielineOfGlobalComposition(global_comp, T=1000.0)


def test_ternary_surrogate_returns_nearest_interface_and_general_diffusivities():
    bulk_point = np.asarray([[0.45, 0.05]], dtype=np.float64)
    surrogate = _build_surrogate(diffusivity_bulk_points=bulk_point)

    interface_matrix = surrogate.getInterdiffusivity([0.249, 0.10], 1000.0, phase="ALPHA", query_context="interface")
    expected_interface = _TieLineSamplingThermodynamics().getInterdiffusivity([0.25, 0.10], 1000.0, phase="ALPHA")
    general_matrix = surrogate.getInterdiffusivity([0.451, 0.05], 1000.0, phase="ALPHA", query_context="general")
    expected_general = _TieLineSamplingThermodynamics().getInterdiffusivity([0.45, 0.05], 1000.0, phase="ALPHA")

    assert np.allclose(interface_matrix, expected_interface)
    assert np.allclose(general_matrix, expected_general)


def test_ternary_surrogate_rejects_nonmatching_temperature():
    surrogate = _build_surrogate()

    with pytest.raises(ValueError, match="isothermal"):
        surrogate.getInterdiffusivity([0.25, 0.10], 1100.0, phase="ALPHA")


def test_ternary_surrogate_save_load_preserves_phase_order_and_predictions():
    surrogate = _build_surrogate(diffusivity_bulk_points=np.asarray([[0.45, 0.05]], dtype=np.float64))
    path = Path.cwd() / "ternary_surrogate_test_roundtrip.npz"
    try:
        surrogate.save(path)
        loaded = TernaryMovingBoundaryThermodynamicsSurrogate.load(path)
    finally:
        path.unlink(missing_ok=True)

    assert loaded.tieline_phases == ("ALPHA", "BETA")
    assert np.allclose(loaded.interface_compositions(0.25)[0], surrogate.interface_compositions(0.25)[0])
    assert np.allclose(
        loaded.getInterdiffusivity([0.451, 0.05], 1000.0, phase="ALPHA"),
        surrogate.getInterdiffusivity([0.451, 0.05], 1000.0, phase="ALPHA"),
    )


def test_ternary_illingworth_accepts_surrogate_for_thermo_and_equilibrium():
    surrogate = _build_surrogate()
    left, right = surrogate.interface_compositions(0.0)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=surrogate,
        temperature=1000.0,
        interfacePosition=0.5,
        interface_equilibrium=surrogate,
        initial_eta_bracket=(0.0, 1.0),
        time_step=1.0,
        record=True,
    )

    model.solve(1.0)

    assert np.isclose(model.getInterfacePosition(), 0.5)
    assert np.isclose(model.getInterfaceEta(), 0.0)
    assert np.allclose(model.checkConservation(1e-12), np.zeros(2))
