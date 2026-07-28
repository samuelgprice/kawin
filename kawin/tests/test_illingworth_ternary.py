from pathlib import Path

import numpy as np
import pytest

from kawin.diffusion import (
    MovingBoundaryIllingworthTernaryFD1DModel,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    estimate_initial_eta_from_instantaneous_balance,
    estimate_initial_eta_from_stefan_residual,
)
from kawin.diffusion.mesh import CartesianFD1D, MixedBoundary1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.MovingBoundaryIllingworthTernaryFDM import _select_interface_motion_branch
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


class _EtaVaryingInterfaceEquilibrium:
    eta_bounds = (0.0, 1.0)

    def interface_compositions(self, eta):
        eta = float(eta)
        left = np.asarray([0.20 + 0.08 * eta, 0.08 + 0.03 * eta], dtype=np.float64)
        right = np.asarray([0.34 + 0.04 * eta, 0.16 - 0.02 * eta], dtype=np.float64)
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


def _legacy_interface_residual(model, p_future, q_future, s, future_s, dt, c_left, c_right, D_left, D_right, motion_branch):
    diff_l = _block_times_vector(D_left, (c_left - p_future[-2]) / (1.0 - model._u_grid[-2])) / future_s
    diff_r = _block_times_vector(D_right, (q_future[1] - c_right) / model._v_grid[1]) / (model._R - future_s)
    rhs = (diff_r - diff_l) * dt
    if motion_branch == "positive":
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


def _record_solve_interface_branches(previous_delta_s, max_iterations=1):
    model, p, q = _make_residual_identity_state()
    s = float(model._s_curr)
    old_s = s - float(previous_delta_s)
    records = []
    left_original = model._new_concentration_left_planar
    right_original = model._new_concentration_right_planar
    residual_original = model._interface_residual

    def left_spy(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch):
        records.append(("left", float(future_s_arg), motion_branch))
        return left_original(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch)

    def right_spy(q_arg, s_arg, future_s_arg, dt_arg, c_right_arg, D_right_arg, motion_branch):
        records.append(("right", float(future_s_arg), motion_branch))
        return right_original(q_arg, s_arg, future_s_arg, dt_arg, c_right_arg, D_right_arg, motion_branch)

    def residual_spy(*args):
        records.append(("residual", float(args[4]), args[-1]))
        return residual_original(*args)

    model._new_concentration_left_planar = left_spy
    model._new_concentration_right_planar = right_spy
    model._interface_residual = residual_spy
    model.maxIterations = int(max_iterations)
    model.residualTolerance = -1.0
    with pytest.raises(RuntimeError, match="failed to converge"):
        model._solve_interface_planar(p, q, s, old_s, 0.2, 1.0e-4)
    return s, records


def _recorded_planar_inventories(model):
    """Recomputes recorded ternary Illingworth inventories with the production helper."""
    return np.asarray(
        [
            integrate_planar_transformed_profile_components(
                np.asarray(model.pData._y[i], dtype=np.float64),
                np.asarray(model.qData._y[i], dtype=np.float64),
                float(model.interfaceData._y[i]),
                model._R,
                model._u_grid,
                model._v_grid,
            )
            for i in range(model.pData.N + 1)
        ],
        dtype=np.float64,
    )


def test_ternary_block_solve_preserves_component_coupling():
    diagonal_block = np.asarray([[2.0, 0.5], [0.25, 3.0]], dtype=np.float64)
    expected = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    lower = np.zeros((3, 2, 2), dtype=np.float64)
    diagonal = np.repeat(diagonal_block[np.newaxis, :, :], 3, axis=0)
    upper = np.zeros((3, 2, 2), dtype=np.float64)
    rhs = np.asarray([_block_times_vector(diagonal_block, row) for row in expected], dtype=np.float64)

    actual = solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)

    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "previous_delta_s, expected_branch",
    [
        (-1.0e-3, "negative"),
        (1.0e-3, "positive"),
        (-0.5e-15, "positive"),
    ],
)
def test_ternary_solver_uses_consistent_fallback_branch_when_future_position_is_unchanged(previous_delta_s, expected_branch):
    s, records = _record_solve_interface_branches(previous_delta_s)
    first_residual_index = next(i for i, item in enumerate(records) if item[0] == "residual")
    first_evaluation = records[: first_residual_index + 1]

    assert _select_interface_motion_branch(s, s - previous_delta_s, s) == expected_branch
    assert [name for name, _, _ in first_evaluation] == ["left", "right", "residual"]
    assert all(np.isclose(future_s, s) for _, future_s, _ in first_evaluation)
    assert {branch for _, _, branch in first_evaluation} == {expected_branch}


def test_ternary_solver_freezes_upwind_branch_for_finite_difference_jacobian_near_zero_motion():
    s, records = _record_solve_interface_branches(-1.0e-3)
    perturbed_records = [
        item for item in records if item[0] in {"left", "right", "residual"} and item[1] > s + 1.0e-10
    ]

    assert perturbed_records
    assert {branch for _, _, branch in perturbed_records} == {"negative"}


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
    motion_branch = _select_interface_motion_branch(s, s, future_s)
    p_future = model._new_concentration_left_planar(p, s, future_s, dt, c_left, D_left, motion_branch)
    q_future = model._new_concentration_right_planar(q, s, future_s, dt, c_right, D_right, motion_branch)

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
        motion_branch,
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
    motion_branch = _select_interface_motion_branch(s, s, future_s)
    p_future = model._new_concentration_left_planar(p, s, future_s, dt, c_left, D_left, motion_branch)
    q_future = model._new_concentration_right_planar(q, s, future_s, dt, c_right, D_right, motion_branch)

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
        motion_branch,
    )
    legacy = _legacy_interface_residual(model, p_future, q_future, s, future_s, dt, c_left, c_right, D_left, D_right, motion_branch)
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


@pytest.mark.parametrize(
    "case_name, left_bulk, right_bulk, direction",
    [
        ("moves_right", np.asarray([0.35, 0.10], dtype=np.float64), np.asarray([0.25, 0.16], dtype=np.float64), 1.0),
        ("moves_left", np.asarray([0.22, 0.10], dtype=np.float64), np.asarray([0.37, 0.15], dtype=np.float64), -1.0),
    ],
)
def test_ternary_illingworth_conserves_inventory_for_moving_eta_dependent_tielines(case_name, left_bulk, right_bulk, direction):
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(
        ProfileBuilder([(StepProfile1D(0.45, left_bulk, right_bulk), ["X", "Y"])]),
        boundaryConditions=MixedBoundary1D(2),
    )
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=_CoupledTernaryThermodynamics(),
        temperature=1000.0,
        interfacePosition=0.45,
        interface_equilibrium=_EtaVaryingInterfaceEquilibrium(),
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(0.0, 1.0),
        time_step=1.0e-4,
        tolerance=1.0e-11,
        max_iterations=50,
        record=True,
    )

    model.solve(2.0e-2, minDtFrac=1.0e-10)

    motion_tol = 1.0e-9
    eta_tol = 1.0e-9
    composition_tol = 1.0e-10
    total_motion_tol = 1.0e-7
    total_eta_tol = 1.0e-7
    total_composition_tol = 1.0e-8
    inventory_tol = 5.0e-11
    simplex_tol = 1.0e-12
    positions = np.asarray(model.interfaceData._y[: model.interfaceData.N + 1], dtype=np.float64)
    etas = np.asarray(model.etaData._y[: model.etaData.N + 1], dtype=np.float64)
    p_history = np.asarray(model.pData._y[: model.pData.N + 1], dtype=np.float64)
    q_history = np.asarray(model.qData._y[: model.qData.N + 1], dtype=np.float64)
    inventories = _recorded_planar_inventories(model)
    inventory_drift = np.max(np.abs(inventories - inventories[0]), axis=0)
    assert positions.size == etas.size == p_history.shape[0] == q_history.shape[0] == inventories.shape[0]

    interface_history = np.asarray(
        [model.interfaceEquilibrium.interface_compositions(float(eta)) for eta in etas],
        dtype=np.float64,
    )
    left_history = interface_history[:, 0, :]
    right_history = interface_history[:, 1, :]
    delta_s = np.diff(positions)
    delta_eta = np.diff(etas)
    delta_c_left = np.max(np.abs(np.diff(left_history, axis=0)), axis=1)
    delta_c_right = np.max(np.abs(np.diff(right_history, axis=0)), axis=1)
    simultaneous = (
        (direction * delta_s > motion_tol)
        & (np.abs(delta_eta) > eta_tol)
        & (delta_c_left > composition_tol)
        & (delta_c_right > composition_tol)
    )
    all_compositions = np.concatenate((p_history.reshape(-1, 2), q_history.reshape(-1, 2)), axis=0)
    dependent_compositions = 1.0 - np.sum(all_compositions, axis=1)

    assert case_name in {"moves_right", "moves_left"}
    assert model.interfaceData.N >= 5
    assert np.any(simultaneous)
    assert direction * (positions[-1] - positions[0]) > total_motion_tol
    assert abs(etas[-1] - etas[0]) > total_eta_tol
    assert np.linalg.norm(left_history[-1] - left_history[0], ord=np.inf) > total_composition_tol
    assert np.linalg.norm(right_history[-1] - right_history[0], ord=np.inf) > total_composition_tol
    assert np.all(inventory_drift <= inventory_tol)
    assert np.all(np.isfinite(all_compositions))
    assert np.all(all_compositions >= -simplex_tol)
    assert np.all(dependent_compositions >= -simplex_tol)


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
