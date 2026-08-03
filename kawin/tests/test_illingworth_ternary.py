from pathlib import Path

import numpy as np
import pytest

import kawin.diffusion.MovingBoundaryIllingworthTernaryFDM as ternary_fdm
from kawin.diffusion import (
    MovingBoundaryIllingworthTernaryFD1DModel,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    estimate_initial_eta_from_instantaneous_balance,
    estimate_initial_eta_from_stefan_residual,
)
from kawin.diffusion.DiffusionParameters import TemperatureParameters
from kawin.diffusion.mesh import CartesianFD1D, MixedBoundary1D, PeriodicBoundary1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.MovingBoundaryIllingworthTernaryFDM import (
    _get_stefan_interdiffusivity,
    _select_interface_motion_branch,
    _validate_ternary_diffusivity_matrix,
)
from kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D import (
    _BLOCK_PIVOT_RCOND_LIMIT,
    _BLOCK_PIVOT_STATUS_NONFINITE,
    _BLOCK_PIVOT_STATUS_SINGULAR,
    _BLOCK_PIVOT_STATUS_VALID,
    _check_block_pivot,
    _estimate_2x2_rcond,
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


class _LengthScaledCoupledTernaryThermodynamics:
    def __init__(self, length_scale):
        self.length_scale = float(length_scale)
        self.base = _CoupledTernaryThermodynamics()

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        return self.base.getInterdiffusivity(composition, temperature, phase=phase, **kwargs) * self.length_scale * self.length_scale


class _RecordingCompositionDependentThermodynamics:
    def __init__(self):
        self.calls = []

    def clearCache(self):
        pass

    def reset(self):
        self.calls.clear()

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        composition = np.asarray(composition, dtype=np.float64).reshape(2)
        self.calls.append(
            {
                "phase": phase,
                "composition": composition.copy(),
                "temperature": float(temperature),
                "query_context": kwargs.get("query_context"),
            }
        )
        if phase == "ALPHA":
            return np.asarray(
                [
                    [1.0 + 0.10 * composition[0], 0.02 + 0.01 * composition[1]],
                    [0.03 + 0.02 * composition[0], 1.2 + 0.10 * composition[1]],
                ],
                dtype=np.float64,
            )
        return np.asarray(
            [
                [1.4 + 0.10 * composition[0], -0.02 + 0.01 * composition[1]],
                [0.04 + 0.01 * composition[0], 1.1 + 0.10 * composition[1]],
            ],
            dtype=np.float64,
        )


class _SmoothBulkTernaryThermodynamics:
    def __init__(self, vectorized=True, constant=False):
        self.vectorized = bool(vectorized)
        self.constant = bool(constant)
        self.calls = []

    def clearCache(self):
        pass

    def reset(self):
        self.calls.clear()

    def _matrix_for(self, composition, phase):
        x, y = np.asarray(composition, dtype=np.float64)
        if self.constant:
            if phase == "ALPHA":
                return np.asarray([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]], dtype=np.float64)
            return np.asarray([[7.0e-4, -1.0e-4], [2.0e-4, 1.1e-3]], dtype=np.float64)
        if phase == "ALPHA":
            return np.asarray([[1.0e-3 + 1.0e-4 * x, 2.0e-4 + 2.0e-5 * y], [1.0e-4 + 1.0e-5 * x, 8.0e-4 + 8.0e-5 * y]], dtype=np.float64)
        return np.asarray([[7.0e-4 + 8.0e-5 * x, -1.0e-4 + 1.0e-5 * y], [2.0e-4 + 1.0e-5 * x, 1.1e-3 + 7.0e-5 * y]], dtype=np.float64)

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        single = values.ndim == 1
        if not single and not self.vectorized:
            raise ValueError("scalar-only thermodynamics")
        values = np.atleast_2d(values)
        self.calls.append(
            {
                "phase": phase,
                "composition": values.copy(),
                "temperature": np.asarray(temperature, dtype=np.float64).copy(),
                "query_context": kwargs.get("query_context"),
            }
        )
        matrices = np.asarray([self._matrix_for(row, phase) for row in values], dtype=np.float64)
        return matrices[0] if single else matrices


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


def _curved_eta_varying_interface_compositions(eta):
    """
    Return curved, non-crossing eta-dependent ternary interface endpoints.

    A curved centerline is constructed first. Each tie-line is then oriented
    by tilting the centerline normal toward its tangent. Mildly varying
    half-widths control the two phase boundaries.

    The constants below are chosen so that, for eta in [0, 1], the endpoints
    remain inside the ternary simplex and the tie-lines remain ordered.
    """
    eta = float(eta)

    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1]; got {eta}.")

    # Curved centerline and its derivative.
    bend = eta * (1.0 - eta)

    center = np.asarray(
        [
            0.24 + 0.25 * eta,
            0.15 - 0.015 * eta + 0.22 * bend,
        ],
        dtype=np.float64,
    )

    center_derivative = np.asarray(
        [
            0.25,
            -0.015 + 0.22 * (1.0 - 2.0 * eta),
        ],
        dtype=np.float64,
    )

    tangent = center_derivative / np.linalg.norm(center_derivative)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)

    # Angle measured from the normal toward the tangent.
    tilt = np.deg2rad(55.0 - 30.0 * eta)
    tie_direction = np.sin(tilt) * tangent + np.cos(tilt) * normal

    # Mildly asymmetric and eta-dependent boundary distances.
    left_half_width = 0.042 * (1.0 + 0.12 * (2.0 * eta - 1.0))
    right_half_width = 0.055 * (1.0 - 0.10 * (2.0 * eta - 1.0))

    left = center - left_half_width * tie_direction
    right = center + right_half_width * tie_direction

    return left, right


class _EtaVaryingInterfaceEquilibrium:
    eta_bounds = (0.0, 1.0)

    def interface_compositions(self, eta):
        return _curved_eta_varying_interface_compositions(eta)


class _ShiftedEtaVaryingInterfaceEquilibrium:
    eta_bounds = (2.0, 5.0)

    def interface_compositions(self, eta):
        eta_hat = (float(eta) - self.eta_bounds[0]) / (self.eta_bounds[1] - self.eta_bounds[0])
        return _curved_eta_varying_interface_compositions(eta_hat)


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


class _QuasiBinaryCuZnDummyEquilibrium:
    """Fixed Cu-Zn endpoints with eta-pinned dummy interface compositions."""

    eta_bounds = (0.0, 1.0)

    def __init__(self, zn_left, zn_right, dummy):
        self.zn_left = float(zn_left)
        self.zn_right = float(zn_right)
        self.dummy = float(dummy)
        assert dummy<(1-zn_left)
        assert dummy<(1-zn_right)

    def interface_compositions(self, eta):
        return (
            np.asarray([self.zn_left, self.dummy*eta], dtype=np.float64),
            np.asarray([self.zn_right, self.dummy*eta], dtype=np.float64),
        )


class _QuasiBinaryTernaryThermodynamics:
    """Diagonal ternary diffusivity closure for quasi-binary diagnostics."""

    def __init__(self, diffusivities, dummy_diffusivity):
        self.diffusivities = {phase: float(value) for phase, value in diffusivities.items()}
        self.dummy_diffusivity = float(dummy_diffusivity)

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        return np.asarray(
            [
                [self.diffusivities[phase], 0.0],
                [0.0, self.dummy_diffusivity],
            ],
            dtype=np.float64,
        )


def _block_times_vector(block, vector):
    return np.asarray(
        [
            block[0, 0] * vector[0] + block[0, 1] * vector[1],
            block[1, 0] * vector[0] + block[1, 1] * vector[1],
        ],
        dtype=np.float64,
    )


def _assemble_test_block_tridiagonal(lower, diagonal, upper):
    n_nodes = diagonal.shape[0]
    matrix = np.zeros((2 * n_nodes, 2 * n_nodes), dtype=np.float64)
    for node in range(n_nodes):
        rows = slice(2 * node, 2 * node + 2)
        matrix[rows, rows] = diagonal[node]
        if node > 0:
            matrix[rows, slice(2 * (node - 1), 2 * node)] = lower[node]
        if node < n_nodes - 1:
            matrix[rows, slice(2 * (node + 1), 2 * (node + 2))] = upper[node]
    return matrix


def _representative_block_tridiagonal_system(n_nodes=4):
    lower = np.zeros((n_nodes, 2, 2), dtype=np.float64)
    diagonal = np.zeros((n_nodes, 2, 2), dtype=np.float64)
    upper = np.zeros((n_nodes, 2, 2), dtype=np.float64)
    for node in range(n_nodes):
        diagonal[node] = np.asarray(
            [[4.0 + 0.25 * node, 0.35], [-0.20, 3.5 + 0.15 * node]],
            dtype=np.float64,
        )
        if node > 0:
            lower[node] = np.asarray([[-0.30, 0.08], [0.04, -0.25]], dtype=np.float64)
        if node < n_nodes - 1:
            upper[node] = np.asarray([[-0.18, -0.05], [0.06, -0.22]], dtype=np.float64)
    return lower, diagonal, upper


def _reference_2x2_rcond(matrix):
    scale = float(np.max(np.abs(matrix)))
    if not np.isfinite(scale) or scale <= 0.0:
        return 0.0
    scaled = matrix / scale
    a = float(scaled[0, 0])
    b = float(scaled[0, 1])
    c = float(scaled[1, 0])
    d = float(scaled[1, 1])
    determinant = a * d - b * c
    matrix_norm = max(abs(a) + abs(b), abs(c) + abs(d))
    inverse_adjugate_norm = max(abs(d) + abs(b), abs(c) + abs(a))
    denominator = matrix_norm * inverse_adjugate_norm
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0
    rcond = abs(determinant) / denominator
    if not np.isfinite(rcond):
        return 0.0
    return float(rcond)


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
    left_original = model._solve_concentration_left_planar
    right_original = model._solve_concentration_right_planar
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

    model._solve_concentration_left_planar = left_spy
    model._solve_concentration_right_planar = right_spy
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


def _make_length_scaled_illingworth_model(domain_length, interface_equilibrium=None):
    domain_length = float(domain_length)
    interface_equilibrium = _EtaVaryingInterfaceEquilibrium() if interface_equilibrium is None else interface_equilibrium
    mesh = CartesianFD1D(["X", "Y"], [0.0, domain_length], 21)
    mesh.setResponseProfile(
        ProfileBuilder(
            [
                (
                    StepProfile1D(
                        0.45 * domain_length,
                        np.asarray([0.16, 0.06], dtype=np.float64),
                        np.asarray([0.3261538461538461, 0.193], dtype=np.float64),
                    ),
                    ["X", "Y"],
                )
            ]
        ),
        boundaryConditions=MixedBoundary1D(2),
    )
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=_LengthScaledCoupledTernaryThermodynamics(domain_length),
        temperature=1000.0,
        interfacePosition=0.45 * domain_length,
        interface_equilibrium=interface_equilibrium,
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=interface_equilibrium.eta_bounds,
        time_step=1.0e-4,
        tolerance=1.0e-11,
        max_iterations=50,
        record=True,
    )


def _make_scope_validation_model(boundary_conditions=None, temperature=1000.0):
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    profile = ProfileBuilder([(StepProfile1D(0.5, np.asarray([0.25, 0.10]), np.asarray([0.35, 0.15])), ["X", "Y"])])
    if boundary_conditions is None:
        mesh.setResponseProfile(profile)
    else:
        mesh.setResponseProfile(profile, boundaryConditions=boundary_conditions)
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=_IdentityTernaryThermodynamics(),
        temperature=temperature,
        interfacePosition=0.5,
        interface_equilibrium=_LinearInterfaceEquilibrium(),
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(0.0, 1.0),
        time_step=1.0e-4,
        tolerance=1.0e-10,
        max_iterations=25,
        record=True,
    )


def _nonzero_flux_boundary_conditions():
    bc = MixedBoundary1D(2)
    bc.setLBC(0, "flux", 1.0e-6)
    return bc


def _fixed_composition_boundary_conditions():
    bc = MixedBoundary1D(2)
    bc.setLBC(0, "composition", 0.2)
    return bc


def _mixed_boundary_type_conditions():
    bc = MixedBoundary1D(2)
    bc.setLBC(0, "flux", 0.0)
    bc.setRBC(0, "composition", 0.3)
    return bc


def _record_scaled_jacobian_perturbations(domain_length=1.0e-6):
    model = _make_length_scaled_illingworth_model(
        domain_length,
        interface_equilibrium=_ShiftedEtaVaryingInterfaceEquilibrium(),
    )
    model.setup()
    p = model._p_curr.copy()
    q = model._q_curr.copy()
    s = float(model._s_curr)
    eta = float(model._eta_curr)
    future_s_calls = []
    eta_calls = []
    left_original = model._solve_concentration_left_planar
    interface_original = model._interface_compositions

    def left_spy(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch):
        future_s_calls.append(float(future_s_arg))
        return left_original(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch)

    def interface_spy(eta_arg):
        eta_calls.append(float(eta_arg))
        return interface_original(eta_arg)

    def stop_after_jacobian(jacobian, residual):
        raise RuntimeError("stop after scaled Jacobian probes")

    model._solve_concentration_left_planar = left_spy
    model._interface_compositions = interface_spy
    model._least_squares_step_2xN = stop_after_jacobian
    model.residualTolerance = -1.0
    with pytest.raises(RuntimeError, match="stop after scaled Jacobian probes"):
        model._solve_interface_planar(p, q, s, model._s_old, eta, 1.0e-4)
    return model, np.asarray(future_s_calls, dtype=np.float64), np.asarray(eta_calls, dtype=np.float64)


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
    "matrix",
    [
        np.asarray([[4.0, 0.35], [-0.20, 3.5]], dtype=np.float64),
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.asarray([[1.0, 1.0], [1.0, 1.0 + 1.0e-10]], dtype=np.float64),
        np.asarray([[1.0, 0.0], [0.0, 2.0e-12]], dtype=np.float64),
        np.asarray([[0.0, 2.0], [-3.0, 0.5]], dtype=np.float64),
    ],
)
def test_ternary_block_rcond_matches_reference_expression(matrix):
    actual, status = _estimate_2x2_rcond(matrix)
    expected = _reference_2x2_rcond(matrix)

    assert status == _BLOCK_PIVOT_STATUS_VALID
    assert actual == pytest.approx(expected, rel=1.0e-15, abs=0.0)


@pytest.mark.parametrize("scale", [1.0e-250, 1.0e-125, 1.0, 1.0e125, 1.0e250])
def test_ternary_block_rcond_is_invariant_to_global_scaling(scale):
    matrix = np.asarray([[1.0, -3.0], [2.0, 4.0]], dtype=np.float64)
    scaled_rcond, scaled_status = _estimate_2x2_rcond(scale * matrix)
    reference_rcond, reference_status = _estimate_2x2_rcond(matrix)

    assert scaled_status == reference_status == _BLOCK_PIVOT_STATUS_VALID
    assert scaled_rcond == pytest.approx(reference_rcond)


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((2, 2), dtype=np.float64),
        np.asarray([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.asarray([[np.inf, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.asarray([[1.0, 0.0], [0.0, -np.inf]], dtype=np.float64),
    ],
)
def test_ternary_block_rcond_rejects_zero_and_nonfinite_matrices(matrix):
    rcond, status = _estimate_2x2_rcond(matrix)

    assert rcond == 0.0
    assert status in (_BLOCK_PIVOT_STATUS_NONFINITE, _BLOCK_PIVOT_STATUS_SINGULAR)


def test_ternary_block_rcond_reports_finite_exactly_singular_matrix_as_singular():
    matrix = np.asarray([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)

    rcond, status = _estimate_2x2_rcond(matrix)

    assert rcond == 0.0
    assert status == _BLOCK_PIVOT_STATUS_SINGULAR


@pytest.mark.parametrize(
    "matrix, match, expected_status",
    [
        (np.asarray([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float64), "nonfinite", _BLOCK_PIVOT_STATUS_NONFINITE),
        (np.asarray([[np.inf, 0.0], [0.0, 1.0]], dtype=np.float64), "nonfinite", _BLOCK_PIVOT_STATUS_NONFINITE),
        (np.asarray([[-np.inf, 0.0], [0.0, 1.0]], dtype=np.float64), "nonfinite", _BLOCK_PIVOT_STATUS_NONFINITE),
        (np.zeros((2, 2), dtype=np.float64), "singular", _BLOCK_PIVOT_STATUS_SINGULAR),
    ],
)
def test_ternary_block_pivot_messages_distinguish_nonfinite_and_singular(matrix, match, expected_status):
    rcond, status = _estimate_2x2_rcond(matrix)

    assert rcond == 0.0
    assert status == expected_status
    with pytest.raises(np.linalg.LinAlgError, match=match):
        _check_block_pivot(matrix, "message probe")


def test_ternary_block_rcond_threshold_decisions_are_preserved():
    accepted = np.asarray([[1.0, 0.0], [0.0, 1.1 * _BLOCK_PIVOT_RCOND_LIMIT]], dtype=np.float64)
    rejected = np.asarray([[1.0, 0.0], [0.0, 0.9 * _BLOCK_PIVOT_RCOND_LIMIT]], dtype=np.float64)
    nearly_singular = np.asarray([[1.0, 0.0], [0.0, 0.5 * _BLOCK_PIVOT_RCOND_LIMIT]], dtype=np.float64)
    accepted_rcond, accepted_status = _estimate_2x2_rcond(accepted)
    rejected_rcond, rejected_status = _estimate_2x2_rcond(rejected)
    nearly_singular_rcond, nearly_singular_status = _estimate_2x2_rcond(nearly_singular)

    assert accepted_status == _BLOCK_PIVOT_STATUS_VALID
    assert accepted_rcond > _BLOCK_PIVOT_RCOND_LIMIT
    _check_block_pivot(accepted, "accepted threshold probe")
    assert rejected_status == _BLOCK_PIVOT_STATUS_VALID
    assert rejected_rcond < _BLOCK_PIVOT_RCOND_LIMIT
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        _check_block_pivot(rejected, "rejected threshold probe")
    assert nearly_singular_status == _BLOCK_PIVOT_STATUS_VALID
    assert 0.0 < nearly_singular_rcond < _BLOCK_PIVOT_RCOND_LIMIT


def test_ternary_block_pivot_success_path_avoids_numpy_reductions_and_linalg(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pivot validation should use scalar arithmetic in the hot path.")

    monkeypatch.setattr(np, "max", forbidden)
    monkeypatch.setattr(np, "abs", forbidden)
    monkeypatch.setattr(np, "all", forbidden)
    monkeypatch.setattr(np, "isfinite", forbidden)
    monkeypatch.setattr(np.linalg, "cond", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    matrix = np.asarray([[4.0, 0.35], [-0.20, 3.5]], dtype=np.float64)
    rcond, status = _estimate_2x2_rcond(matrix)

    assert status == _BLOCK_PIVOT_STATUS_VALID
    assert rcond > _BLOCK_PIVOT_RCOND_LIMIT
    _check_block_pivot(matrix, "hot path probe")


def test_ternary_block_solve_matches_dense_nonsymmetric_system():
    lower, diagonal, upper = _representative_block_tridiagonal_system()
    matrix = _assemble_test_block_tridiagonal(lower, diagonal, upper)
    rhs = np.asarray([0.2, -0.1, 0.4, 0.6, -0.3, 0.7, 0.9, -0.5], dtype=np.float64)
    expected = np.linalg.solve(matrix, rhs).reshape(4, 2)

    actual = solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs.reshape(4, 2))

    assert np.allclose(actual, expected)


def test_ternary_block_solve_supports_multiple_right_hand_sides():
    lower, diagonal, upper = _representative_block_tridiagonal_system()
    matrix = _assemble_test_block_tridiagonal(lower, diagonal, upper)
    rhs = np.asarray(
        [
            [0.20, 0.40, -0.10],
            [-0.10, 0.30, 0.50],
            [0.40, -0.20, 0.10],
            [0.60, 0.25, -0.30],
            [-0.30, 0.75, 0.20],
            [0.70, -0.15, 0.45],
            [0.90, 0.10, -0.40],
            [-0.50, 0.55, 0.35],
        ],
        dtype=np.float64,
    )
    expected = np.linalg.solve(matrix, rhs).reshape(4, 2, 3)

    actual = solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs.reshape(4, 2, 3))

    assert np.allclose(actual, expected)


@pytest.mark.parametrize("scale", [1.0e-200, 1.0e-120, 1.0, 1.0e120, 1.0e200])
def test_ternary_block_solve_is_invariant_to_global_equation_scaling(scale):
    lower, diagonal, upper = _representative_block_tridiagonal_system()
    matrix = _assemble_test_block_tridiagonal(lower, diagonal, upper)
    rhs = np.asarray([0.2, -0.1, 0.4, 0.6, -0.3, 0.7, 0.9, -0.5], dtype=np.float64)
    expected = np.linalg.solve(matrix, rhs).reshape(4, 2)

    actual = solve_illingworth_block_tridiagonal(scale * lower, scale * diagonal, scale * upper, scale * rhs.reshape(4, 2))

    assert np.allclose(actual, expected)


def test_ternary_block_solve_does_not_call_dense_condition_check_on_success(monkeypatch):
    lower, diagonal, upper = _representative_block_tridiagonal_system()
    rhs = np.asarray([0.2, -0.1, 0.4, 0.6, -0.3, 0.7, 0.9, -0.5], dtype=np.float64).reshape(4, 2)

    def fail_cond(matrix):
        raise AssertionError("np.linalg.cond should not be called on the block Thomas success path.")

    monkeypatch.setattr(np.linalg, "cond", fail_cond)

    actual = solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)

    assert np.all(np.isfinite(actual))


def test_ternary_block_solve_falls_back_when_intermediate_thomas_pivot_is_singular_but_full_system_is_solvable():
    identity = np.eye(2, dtype=np.float64)
    lower = np.zeros((3, 2, 2), dtype=np.float64)
    diagonal = np.repeat(identity[np.newaxis, :, :], 3, axis=0)
    upper = np.zeros((3, 2, 2), dtype=np.float64)
    upper[0] = identity
    upper[1] = identity
    lower[1] = identity
    lower[2] = identity
    matrix = _assemble_test_block_tridiagonal(lower, diagonal, upper)
    rhs = np.asarray([0.2, -0.1, 0.4, 0.6, -0.3, 0.7], dtype=np.float64)
    expected = np.linalg.solve(matrix, rhs).reshape(3, 2)

    actual = solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs.reshape(3, 2))

    assert np.allclose(actual, expected)


def test_ternary_block_solve_raises_for_genuinely_singular_system():
    lower = np.zeros((2, 2, 2), dtype=np.float64)
    diagonal = np.zeros((2, 2, 2), dtype=np.float64)
    upper = np.zeros((2, 2, 2), dtype=np.float64)
    rhs = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(np.linalg.LinAlgError, match="full block-tridiagonal system is singular"):
        solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)


def test_ternary_block_solve_rejects_nonfinite_coefficients_and_rhs():
    lower, diagonal, upper = _representative_block_tridiagonal_system()
    rhs = np.ones((4, 2), dtype=np.float64)
    bad_diagonal = diagonal.copy()
    bad_diagonal[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="coefficients must be finite"):
        solve_illingworth_block_tridiagonal(lower, bad_diagonal, upper, rhs)

    bad_rhs = rhs.copy()
    bad_rhs[0, 0] = np.inf
    with pytest.raises(ValueError, match="rhs must be finite"):
        solve_illingworth_block_tridiagonal(lower, diagonal, upper, bad_rhs)


def test_ternary_block_solve_validates_input_shapes():
    lower, diagonal, upper = _representative_block_tridiagonal_system()

    with pytest.raises(ValueError, match="matching shapes"):
        solve_illingworth_block_tridiagonal(lower[:-1], diagonal, upper, np.ones((4, 2)))

    with pytest.raises(ValueError, match="rhs must have shape"):
        solve_illingworth_block_tridiagonal(lower, diagonal, upper, np.ones((4, 3)))


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64),
        np.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=np.float64),
        np.asarray([[2.0, 1.0], [0.0, 2.0]], dtype=np.float64),
    ],
)
def test_ternary_diffusivity_validation_accepts_positive_real_eigenvalues(matrix):
    actual = _validate_ternary_diffusivity_matrix(matrix, "ALPHA", context="test diffusivity")

    assert np.allclose(actual, matrix)


@pytest.mark.parametrize("scale", [1.0e-300, 1.0e-150, 1.0, 1.0e150, 1.0e300])
def test_ternary_diffusivity_validation_is_invariant_to_positive_unit_scaling(scale):
    matrix = scale * np.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=np.float64)

    actual = _validate_ternary_diffusivity_matrix(matrix, "ALPHA", context="scaled test diffusivity")

    assert np.allclose(actual, matrix)


@pytest.mark.parametrize(
    "matrix, match",
    [
        (np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64), "scaled eigenvalues"),
        (np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64), "scaled eigenvalues"),
        (np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64), "scaled eigenvalues"),
        (np.asarray([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64), "finite"),
        (np.asarray([1.0, 2.0], dtype=np.float64), "shape"),
    ],
)
def test_ternary_diffusivity_validation_rejects_invalid_matrices(matrix, match):
    with pytest.raises(ValueError, match=match):
        _validate_ternary_diffusivity_matrix(matrix, "ALPHA", context="invalid test diffusivity")


def test_ternary_diffusivity_validation_rejects_complex_valued_matrix():
    matrix = np.asarray([[1.0 + 1.0e-8j, 0.0], [0.0, 1.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="real-valued"):
        _validate_ternary_diffusivity_matrix(matrix, "ALPHA", context="complex test diffusivity")


def test_ternary_diffusivity_validation_call_sites_share_helper(monkeypatch):
    model = _make_scope_validation_model()
    model.setup()
    matrix = np.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=np.float64)
    diffusivity_calls = []
    validation_calls = []

    class _SharedValidationThermodynamics:
        def clearCache(self):
            pass

        def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
            diffusivity_calls.append((phase, kwargs.get("query_context")))
            return matrix

    def validation_spy(D, phase, context="ternary Illingworth diffusivity"):
        validation_calls.append((phase, context))
        return np.asarray(D, dtype=np.float64)

    monkeypatch.setattr(ternary_fdm, "_validate_ternary_diffusivity_matrix", validation_spy)
    thermodynamics = _SharedValidationThermodynamics()
    stefan = _get_stefan_interdiffusivity(thermodynamics, [0.2, 0.1], 1000.0, "ALPHA")
    model.therm = thermodynamics
    transient = model._phase_diffusivity_matrix([0.2, 0.1], "ALPHA", 0.0, 0.5)

    assert np.allclose(stefan, matrix)
    assert np.allclose(transient, matrix)
    assert diffusivity_calls == [("ALPHA", "interface"), ("ALPHA", "interface")]
    assert validation_calls == [("ALPHA", "initial-eta diffusivity"), ("ALPHA", "transient diffusivity")]
    assert not hasattr(model, "_validate_diffusivity_matrix")


def test_ternary_bulk_diffusivity_is_phase_uniform_and_evaluated_at_interface_compositions():
    thermodynamics = _RecordingCompositionDependentThermodynamics()
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(
        ProfileBuilder(
            [
                (
                    StepProfile1D(
                        0.45,
                        np.asarray([0.25, 0.10], dtype=np.float64),
                        np.asarray([0.35, 0.15], dtype=np.float64),
                    ),
                    ["X", "Y"],
                )
            ]
        )
    )
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=thermodynamics,
        temperature=1000.0,
        interfacePosition=0.45,
        interface_equilibrium=_LinearInterfaceEquilibrium(),
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(0.0, 1.0),
        time_step=1.0e-4,
        tolerance=1.0e-10,
        max_iterations=25,
        record=True,
    )
    model.setup()
    p = model._p_curr.copy()
    q = model._q_curr.copy()
    p[:-1] = np.linspace([0.05, 0.05], [0.18, 0.20], len(p) - 1)
    q[1:] = np.linspace([0.45, 0.05], [0.10, 0.35], len(q) - 1)
    eta = float(model._eta_curr)
    c_left, c_right = model._interface_compositions(eta)
    expected_left = thermodynamics.getInterdiffusivity(c_left, 1000.0, phase="ALPHA", query_context="interface")
    expected_right = thermodynamics.getInterdiffusivity(c_right, 1000.0, phase="BETA", query_context="interface")
    thermodynamics.reset()
    bulk_calls = []
    left_original = model._solve_concentration_left_planar
    right_original = model._solve_concentration_right_planar

    def left_spy(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch):
        bulk_calls.append(("ALPHA", np.asarray(D_left_arg, dtype=np.float64).copy()))
        return left_original(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch)

    def right_spy(q_arg, s_arg, future_s_arg, dt_arg, c_right_arg, D_right_arg, motion_branch):
        bulk_calls.append(("BETA", np.asarray(D_right_arg, dtype=np.float64).copy()))
        return right_original(q_arg, s_arg, future_s_arg, dt_arg, c_right_arg, D_right_arg, motion_branch)

    model._solve_concentration_left_planar = left_spy
    model._solve_concentration_right_planar = right_spy
    model.residualTolerance = np.inf

    model._solve_interface_planar(p, q, model._s_curr, model._s_old, eta, 1.0e-4)

    assert [(call["phase"], call["query_context"]) for call in thermodynamics.calls] == [
        ("ALPHA", "interface"),
        ("BETA", "interface"),
    ]
    assert np.allclose(thermodynamics.calls[0]["composition"], c_left)
    assert np.allclose(thermodynamics.calls[1]["composition"], c_right)
    assert not any(np.allclose(call["composition"], p[0]) for call in thermodynamics.calls)
    assert not any(np.allclose(call["composition"], q[-1]) for call in thermodynamics.calls)
    assert len(bulk_calls) == 2
    assert bulk_calls[0][0] == "ALPHA"
    assert bulk_calls[0][1].shape == (2, 2)
    assert np.allclose(bulk_calls[0][1], expected_left)
    assert bulk_calls[1][0] == "BETA"
    assert bulk_calls[1][1].shape == (2, 2)
    assert np.allclose(bulk_calls[1][1], expected_right)


def test_ternary_left_face_diffusivity_indexing_and_cross_terms(monkeypatch):
    model, p, _ = _make_residual_identity_state()
    s = float(model._s_curr)
    future_s = s + 2.0e-3
    dt = 1.0e-4
    c_left = p[-1].copy()
    D_faces = np.asarray(
        [
            [[1.0e-3 + i * 1.0e-5, 2.0e-4 + i * 1.0e-5], [1.0e-4 + i * 5.0e-6, 8.0e-4 + i * 1.0e-5]]
            for i in range(len(p) - 1)
        ],
        dtype=np.float64,
    )
    captured = {}

    def capture_solver(lower, diagonal, upper, rhs):
        captured["lower"] = lower.copy()
        captured["diagonal"] = diagonal.copy()
        captured["upper"] = upper.copy()
        out = p.copy()
        out[-1] = c_left
        return out

    monkeypatch.setattr(ternary_fdm, "solve_illingworth_block_tridiagonal", capture_solver)
    result = model._solve_concentration_left_planar(p, s, future_s, dt, c_left, D_faces, "positive")

    i = 2
    u = model._u_grid
    A_left = D_faces[i - 1] * (dt / future_s)
    A_right = D_faces[i] * (dt / future_s)
    left_diff = u[i] - u[i - 1]
    right_diff = u[i + 1] - u[i]
    expected_diagonal = -A_left / left_diff - A_right / right_diff
    expected_diagonal += -np.eye(2) * ((future_s - s) * (u[i] + u[i - 1]) / 2.0 + future_s * (u[i + 1] - u[i - 1]) / 2.0)

    assert np.allclose(captured["lower"][i], A_left / left_diff)
    assert np.allclose(captured["diagonal"][i], expected_diagonal)
    assert np.allclose(captured["upper"][i][0, 1], (A_right / right_diff)[0, 1])
    assert np.allclose(captured["upper"][i][1, 0], (A_right / right_diff)[1, 0])
    assert np.allclose(captured["upper"][i][0, 1], captured["lower"][i + 1][0, 1])
    assert np.allclose(captured["upper"][i][1, 0], captured["lower"][i + 1][1, 0])
    assert np.allclose(result.interface_face_matrix, D_faces[-1])
    expected_flux = _block_times_vector(D_faces[-1], (c_left - p[-2]) / (1.0 - model._u_grid[-2])) / future_s
    assert np.allclose(result.interface_flux, expected_flux)


def test_ternary_right_face_diffusivity_indexing_and_interface_flux(monkeypatch):
    model, _, q = _make_residual_identity_state()
    s = float(model._s_curr)
    future_s = s - 2.0e-3
    dt = 1.0e-4
    c_right = q[0].copy()
    D_faces = np.asarray(
        [
            [[7.0e-4 + i * 1.0e-5, -1.0e-4 + i * 2.0e-6], [2.0e-4 + i * 4.0e-6, 1.1e-3 + i * 1.0e-5]]
            for i in range(len(q) - 1)
        ],
        dtype=np.float64,
    )
    captured = {}

    def capture_solver(lower, diagonal, upper, rhs):
        captured["lower"] = lower.copy()
        captured["diagonal"] = diagonal.copy()
        captured["upper"] = upper.copy()
        out = q.copy()
        out[0] = c_right
        return out

    monkeypatch.setattr(ternary_fdm, "solve_illingworth_block_tridiagonal", capture_solver)
    result = model._solve_concentration_right_planar(q, s, future_s, dt, c_right, D_faces, "negative")

    i = 2
    v = model._v_grid
    A_left = D_faces[i - 1] * (dt / (model._R - future_s))
    A_right = D_faces[i] * (dt / (model._R - future_s))
    left_diff = v[i] - v[i - 1]
    right_diff = v[i + 1] - v[i]
    expected_diagonal = -A_right / right_diff - A_left / left_diff
    expected_diagonal += np.eye(2) * ((future_s - s) * (1.0 - (v[i + 1] + v[i]) / 2.0) - (model._R - future_s) * (v[i + 1] - v[i - 1]) / 2.0)

    assert np.allclose(captured["lower"][i][0, 1], (A_left / left_diff)[0, 1])
    assert np.allclose(captured["lower"][i][1, 0], (A_left / left_diff)[1, 0])
    assert np.allclose(captured["diagonal"][i], expected_diagonal)
    assert np.allclose(captured["upper"][i], A_right / right_diff)
    assert np.allclose(captured["upper"][i][0, 1], captured["lower"][i + 1][0, 1])
    assert np.allclose(captured["upper"][i][1, 0], captured["lower"][i + 1][1, 0])
    assert np.allclose(result.interface_face_matrix, D_faces[0])
    expected_flux = _block_times_vector(D_faces[0], (q[1] - c_right) / model._v_grid[1]) / (model._R - future_s)
    assert np.allclose(result.interface_flux, expected_flux)


def test_ternary_uniform_face_array_matches_phase_uniform_solve():
    model, p, q = _make_residual_identity_state()
    s = float(model._s_curr)
    future_s = s + 1.0e-3
    dt = 1.0e-4
    c_left, c_right = model._interface_compositions(0.4)
    D_left = _CoupledTernaryThermodynamics().getInterdiffusivity(c_left, 1000.0, phase="ALPHA")
    D_right = _CoupledTernaryThermodynamics().getInterdiffusivity(c_right, 1000.0, phase="BETA")
    motion_branch = _select_interface_motion_branch(s, s, future_s)

    left_uniform = model._solve_concentration_left_planar(p, s, future_s, dt, c_left, D_left, motion_branch)
    left_faces = np.broadcast_to(D_left, (len(p) - 1, 2, 2))
    left_face_array = model._solve_concentration_left_planar(p, s, future_s, dt, c_left, left_faces, motion_branch)
    right_uniform = model._solve_concentration_right_planar(q, s, future_s, dt, c_right, D_right, motion_branch)
    right_faces = np.broadcast_to(D_right, (len(q) - 1, 2, 2))
    right_face_array = model._solve_concentration_right_planar(q, s, future_s, dt, c_right, right_faces, motion_branch)

    assert np.allclose(left_face_array.profile, left_uniform.profile, rtol=1.0e-14, atol=1.0e-15)
    assert np.allclose(left_face_array.interface_flux, left_uniform.interface_flux, rtol=1.0e-14, atol=1.0e-15)
    assert np.allclose(right_face_array.profile, right_uniform.profile, rtol=1.0e-14, atol=1.0e-15)
    assert np.allclose(right_face_array.interface_flux, right_uniform.interface_flux, rtol=1.0e-14, atol=1.0e-15)


def test_ternary_bulk_diffusivity_mode_defaults_to_phase_uniform():
    model = _make_scope_validation_model()

    assert model.bulkDiffusivityMode == "phase_uniform"


def test_ternary_lagged_mode_queries_spatially_varying_bulk_face_matrices():
    thermodynamics = _SmoothBulkTernaryThermodynamics()
    model = _make_scope_validation_model()
    model.therm = thermodynamics
    model.bulkDiffusivityMode = "composition_dependent_lagged"
    model.setup()
    thermodynamics.reset()
    p = model._p_curr.copy()
    p[:-1] = np.linspace([0.08, 0.04], [0.28, 0.16], len(p) - 1)

    matrices = model._left_lagged_face_diffusivity_matrices(p, model._s_curr, model.currentTime)
    expected_faces = 0.5 * (p[:-1] + p[1:])

    assert matrices.shape == (len(p) - 1, 2, 2)
    assert not np.allclose(matrices[0], matrices[-1])
    assert len(thermodynamics.calls) == 1
    assert thermodynamics.calls[0]["phase"] == "ALPHA"
    assert thermodynamics.calls[0]["query_context"] == "general"
    assert thermodynamics.calls[0]["composition"].shape == expected_faces.shape
    assert np.allclose(thermodynamics.calls[0]["composition"], expected_faces)


def test_ternary_lagged_mode_uses_scalar_fallback_and_initial_eta_face_convention():
    thermodynamics = _SmoothBulkTernaryThermodynamics(vectorized=False)
    model = _make_scope_validation_model()
    model.therm = thermodynamics
    model.bulkDiffusivityMode = "composition_dependent_lagged"

    model.setup()

    first_left_call = next(call for call in thermodynamics.calls if call["phase"] == "ALPHA" and call["query_context"] == "general")
    first_right_call = next(call for call in thermodynamics.calls if call["phase"] == "BETA" and call["query_context"] == "general")
    c_left, c_right = model.interfaceEquilibrium.interface_compositions(model.initialEta)
    z_left_adjacent = model.initialInterfacePosition * model._u_grid[-2]
    z_right_adjacent = model.initialInterfacePosition + (model._R - model.initialInterfacePosition) * model._v_grid[1]
    c0 = np.asarray(model.data._y[0], dtype=np.float64)
    left_mask = model._z <= model.initialInterfacePosition
    right_mask = model._z >= model.initialInterfacePosition
    p_adjacent = np.asarray([np.interp(z_left_adjacent, model._z[left_mask], c0[left_mask, component]) for component in range(2)])
    q_adjacent = np.asarray([np.interp(z_right_adjacent, model._z[right_mask], c0[right_mask, component]) for component in range(2)])

    assert first_left_call["composition"].shape == (1, 2)
    assert first_right_call["composition"].shape == (1, 2)
    assert np.allclose(first_left_call["composition"][0], 0.5 * (p_adjacent + c_left))
    assert np.allclose(first_right_call["composition"][0], 0.5 * (c_right + q_adjacent))


def test_ternary_lagged_constant_diffusivity_reproduces_phase_uniform_step():
    thermodynamics = _SmoothBulkTernaryThermodynamics(constant=True)
    uniform = _make_scope_validation_model()
    uniform.therm = thermodynamics
    uniform.setup()
    lagged = _make_scope_validation_model()
    lagged.therm = _SmoothBulkTernaryThermodynamics(constant=True)
    lagged.bulkDiffusivityMode = "composition_dependent_lagged"
    lagged.setup()

    x_uniform = uniform.getCurrentX()
    x_lagged = lagged.getCurrentX()
    d_uniform = uniform.getdXdt(uniform.currentTime, x_uniform)
    d_lagged = lagged.getdXdt(lagged.currentTime, x_lagged)

    for actual, expected in zip(d_lagged, d_uniform):
        assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)
    assert lagged._lastImplicitIterations == uniform._lastImplicitIterations
    assert lagged._lastImplicitCandidateEvaluations == uniform._lastImplicitCandidateEvaluations
    assert lagged._lastStepRetries == uniform._lastStepRetries


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_ternary_lagged_mode_conserves_inventory_for_opposite_motion(direction):
    left_bulk = np.asarray([0.16, 0.06], dtype=np.float64)
    right_bulk = np.asarray([0.3261538461538461, 0.193], dtype=np.float64)
    if direction < 0.0:
        left_bulk, right_bulk = np.asarray([0.16, 0.12], dtype=np.float64), np.asarray([0.56, 0.25], dtype=np.float64)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 31)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.45, left_bulk, right_bulk), ["X", "Y"])]), boundaryConditions=MixedBoundary1D(2))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=_SmoothBulkTernaryThermodynamics(),
        temperature=1000.0,
        interfacePosition=0.45,
        interface_equilibrium=_EtaVaryingInterfaceEquilibrium(),
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(0.0, 1.0),
        bulk_diffusivity_mode="composition_dependent_lagged",
        time_step=5.0e-5,
        tolerance=1.0e-10,
        max_iterations=50,
        record=True,
    )
    model.solve(5.0e-4, minDtFrac=1.0e-12)

    inventories = np.asarray(model.inventoryData._y[: model.inventoryData.N + 1], dtype=np.float64)
    assert model._lastImplicitConverged is True
    assert model._lastStepRetries == 0
    assert np.allclose(inventories, inventories[0], rtol=0.0, atol=5.0e-11)
    assert np.all((model.interfaceData._y[: model.interfaceData.N + 1] > 0.0) & (model.interfaceData._y[: model.interfaceData.N + 1] < model._R))
    assert np.all((model.etaData._y[: model.etaData.N + 1] >= 0.0) & (model.etaData._y[: model.etaData.N + 1] <= 1.0))


def test_ternary_scope_validation_accepts_default_zero_flux_boundaries():
    model = _make_scope_validation_model()

    model.setup()

    assert isinstance(model.mesh.boundaryConditions, MixedBoundary1D)


def test_ternary_scope_validation_accepts_explicit_zero_flux_boundaries():
    boundary_conditions = MixedBoundary1D(2)
    boundary_conditions.setLBC(0, "flux", 0.0)
    boundary_conditions.setRBC(0, "flux", 0.0)
    boundary_conditions.setLBC(1, MixedBoundary1D.NEUMANN, 0.0)
    boundary_conditions.setRBC(1, MixedBoundary1D.NEUMANN, 0.0)
    model = _make_scope_validation_model(boundary_conditions=boundary_conditions)

    model.setup()

    assert np.all(boundary_conditions.LBCvalue == 0.0)
    assert np.all(boundary_conditions.RBCvalue == 0.0)


@pytest.mark.parametrize(
    "boundary_conditions, match",
    [
        pytest.param(_nonzero_flux_boundary_conditions(), "nonzero fluxes", id="nonzero_flux"),
        pytest.param(_fixed_composition_boundary_conditions(), "fixed-composition", id="fixed_composition"),
        pytest.param(_mixed_boundary_type_conditions(), "fixed-composition or mixed", id="mixed_boundary_types"),
        pytest.param(PeriodicBoundary1D(), "periodic", id="periodic"),
    ],
)
def test_ternary_scope_validation_rejects_unsupported_boundaries_before_solve(boundary_conditions, match):
    with pytest.raises(NotImplementedError, match=match):
        _make_scope_validation_model(boundary_conditions=boundary_conditions)


@pytest.mark.parametrize(
    "temperature",
    [
        1000.0,
        TemperatureParameters(1000.0),
        TemperatureParameters([0.0, 1.0, 2.0], [1000.0, 1000.0, 1000.0]),
    ],
)
def test_ternary_scope_validation_accepts_constant_temperatures(temperature):
    model = _make_scope_validation_model(temperature=temperature)

    model.setup()

    assert np.isfinite(model.temperatureParameters(np.asarray([[0.5]], dtype=np.float64), 0.0)[0])


@pytest.mark.parametrize(
    "temperature, match",
    [
        (TemperatureParameters([0.0, 1.0], [1000.0, 1100.0]), "time-dependent"),
        (lambda z, t: 1000.0 + np.asarray(z, dtype=np.float64).reshape(-1), "callable"),
    ],
)
def test_ternary_scope_validation_rejects_state_dependent_temperature_before_solve(temperature, match):
    with pytest.raises(NotImplementedError, match=match):
        _make_scope_validation_model(temperature=temperature)


def test_ternary_scaled_interface_solve_is_invariant_to_length_units():
    reference = _make_length_scaled_illingworth_model(1.0)
    rescaled = _make_length_scaled_illingworth_model(1000.0)

    reference.solve(2.0e-3, minDtFrac=1.0e-10)
    rescaled.solve(2.0e-3, minDtFrac=1.0e-10)

    reference_s_hat = np.asarray(reference.interfaceData._y[: reference.interfaceData.N + 1], dtype=np.float64) / reference._R
    rescaled_s_hat = np.asarray(rescaled.interfaceData._y[: rescaled.interfaceData.N + 1], dtype=np.float64) / rescaled._R
    reference_eta = np.asarray(reference.etaData._y[: reference.etaData.N + 1], dtype=np.float64)
    rescaled_eta = np.asarray(rescaled.etaData._y[: rescaled.etaData.N + 1], dtype=np.float64)

    assert reference_s_hat.shape == rescaled_s_hat.shape
    assert np.allclose(rescaled_s_hat, reference_s_hat, rtol=0.0, atol=1.0e-12)
    assert np.allclose(rescaled_eta, reference_eta, rtol=0.0, atol=1.0e-12)
    assert np.isclose(rescaled._lastImplicitResidual, reference._lastImplicitResidual, rtol=1.0e-8, atol=1.0e-15)


def test_ternary_scaled_interface_jacobian_perturbs_position_with_domain_length():
    domain_length = 1.0e-6
    model, future_s_calls, _ = _record_scaled_jacobian_perturbations(domain_length)
    base_s = float(model._s_curr)
    positive_perturbations = future_s_calls[future_s_calls > base_s] - base_s
    expected_physical_step = domain_length * np.sqrt(np.finfo(float).eps)

    assert positive_perturbations.size > 0
    assert np.isclose(np.min(positive_perturbations), expected_physical_step, rtol=1.0e-6, atol=1.0e-20)
    assert np.min(positive_perturbations) < 1.0e-12


def test_ternary_scaled_interface_jacobian_perturbs_eta_with_eta_interval():
    model, _, eta_calls = _record_scaled_jacobian_perturbations()
    base_eta = float(model._eta_curr)
    eta_span = model.interfaceEquilibrium.eta_bounds[1] - model.interfaceEquilibrium.eta_bounds[0]
    positive_perturbations = eta_calls[eta_calls > base_eta] - base_eta
    expected_eta_step = eta_span * np.sqrt(np.finfo(float).eps) * max(1.0, abs((base_eta - model.interfaceEquilibrium.eta_bounds[0]) / eta_span))

    assert positive_perturbations.size > 0
    assert np.isclose(np.min(positive_perturbations), expected_eta_step, rtol=1.0e-6, atol=1.0e-14)


def test_ternary_interface_scaled_bounds_preserve_minimum_phase_widths():
    model = _make_length_scaled_illingworth_model(1.0)
    model.setup()
    lower, upper = model._interface_scaled_bounds()
    eta_lower, _, eta_span = model._eta_scaling_bounds()
    lower_s, lower_eta = model._interface_scaled_to_physical([lower[0], 0.5], eta_lower, eta_span)
    upper_s, upper_eta = model._interface_scaled_to_physical([upper[0], 0.5], eta_lower, eta_span)

    assert 0.0 < lower_s < upper_s < model._R
    assert np.isclose(lower_s, model._R * 1.0e-14, rtol=1.0e-12, atol=1.0e-20)
    assert np.isclose(model._R - upper_s, model._R * 1.0e-14, rtol=0.0, atol=1.0e-17)
    assert lower_eta == upper_eta == 0.5


@pytest.mark.parametrize(
    "eta_hat, outward_eta_step",
    [
        (1.0e-16, -1.0),
        (1.0 - 1.0e-16, 1.0),
    ],
)
def test_ternary_scaled_line_search_removes_outward_eta_component_at_active_bounds(eta_hat, outward_eta_step):
    model = _make_length_scaled_illingworth_model(1.0)
    model.setup()
    lower, upper = model._interface_scaled_bounds()
    x_hat = np.asarray([0.45, eta_hat], dtype=np.float64)
    newton_step = np.asarray([1.0e-3, outward_eta_step], dtype=np.float64)

    bounded_step, alpha = model._bounded_scaled_newton_step(x_hat, newton_step, lower, upper)
    trial = x_hat + alpha * bounded_step

    assert bounded_step[0] == newton_step[0]
    assert bounded_step[1] == 0.0
    assert alpha > 0.0
    assert trial[0] > x_hat[0]
    assert lower[1] <= trial[1] <= upper[1]
    assert np.all(trial >= lower)
    assert np.all(trial <= upper)


@pytest.mark.parametrize(
    "eta_hat, inward_eta_step",
    [
        (0.0, 0.25),
        (1.0, -0.25),
    ],
)
def test_ternary_scaled_line_search_allows_inward_eta_motion_at_active_bounds(eta_hat, inward_eta_step):
    model = _make_length_scaled_illingworth_model(1.0)
    model.setup()
    lower, upper = model._interface_scaled_bounds()
    x_hat = np.asarray([0.45, eta_hat], dtype=np.float64)
    newton_step = np.asarray([1.0e-3, inward_eta_step], dtype=np.float64)

    bounded_step, alpha = model._bounded_scaled_newton_step(x_hat, newton_step, lower, upper)
    trial = x_hat + alpha * bounded_step

    assert np.allclose(bounded_step, newton_step)
    assert alpha > 0.0
    assert lower[1] <= trial[1] <= upper[1]
    assert abs(trial[1] - eta_hat) > 0.0


def test_ternary_scaled_line_search_limits_alpha_to_eta_upper_bound_without_clipping():
    model = _make_length_scaled_illingworth_model(1.0)
    model.setup()
    lower, upper = model._interface_scaled_bounds()
    x_hat = np.asarray([0.45, 0.80], dtype=np.float64)
    newton_step = np.asarray([1.0e-3, 1.0], dtype=np.float64)

    bounded_step, alpha = model._bounded_scaled_newton_step(x_hat, newton_step, lower, upper)
    trial = x_hat + alpha * bounded_step

    assert np.allclose(bounded_step, newton_step)
    assert 0.0 < alpha < 0.20
    assert trial[1] < upper[1]
    assert np.all(trial >= lower)
    assert np.all(trial <= upper)


def test_ternary_scaled_line_search_keeps_forced_outward_eta_trials_feasible():
    model = _make_length_scaled_illingworth_model(1.0)
    model.setup()
    lower, upper = model._interface_scaled_bounds()
    eta_lower, _, eta_span = model._eta_scaling_bounds()
    p = model._p_curr.copy()
    q = model._q_curr.copy()
    s = float(model._s_curr)
    eta = eta_lower
    p[-1], q[0] = model.interfaceEquilibrium.interface_compositions(eta)
    evaluated = []
    left_original = model._solve_concentration_left_planar
    interface_original = model._interface_compositions

    def left_spy(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch):
        evaluated.append(("s", float(future_s_arg) / model._R))
        return left_original(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch)

    def interface_spy(eta_arg):
        evaluated.append(("eta", (float(eta_arg) - eta_lower) / eta_span))
        return interface_original(eta_arg)

    def forced_step(jacobian, residual):
        return np.asarray([1.0e-4, -1.0], dtype=np.float64)

    model._solve_concentration_left_planar = left_spy
    model._interface_compositions = interface_spy
    model._least_squares_step_2xN = forced_step
    model.residualTolerance = -1.0

    with pytest.raises(RuntimeError, match="failed to converge"):
        model._solve_interface_planar(p, q, s, model._s_old, eta, 1.0e-4)

    s_trials = np.asarray([value for kind, value in evaluated if kind == "s"], dtype=np.float64)
    eta_trials = np.asarray([value for kind, value in evaluated if kind == "eta"], dtype=np.float64)

    assert np.any(s_trials > s / model._R)
    assert np.all((s_trials >= lower[0]) & (s_trials <= upper[0]))
    assert np.all((eta_trials >= lower[1]) & (eta_trials <= upper[1]))


def test_ternary_interface_success_diagnostics_count_candidate_and_jacobian_work():
    model = _make_length_scaled_illingworth_model(1.0)
    model.setup()
    x = model.getCurrentX()

    model.getdXdt(model.currentTime, x)

    assert model._lastImplicitConverged is True
    assert model._lastImplicitIterations == 2
    assert model._lastImplicitCandidateEvaluations == 5
    assert model._lastImplicitFunctionEvaluations == model._lastImplicitCandidateEvaluations
    assert model._lastImplicitJacobianEvaluations == 1
    assert model._lastImplicitMotionBranch == "positive"
    assert model._lastImplicitFailureReason is None
    assert np.isclose(model._lastImplicitResidual, 4.499427273822066e-15, rtol=0.0, atol=1.0e-27)


def test_ternary_interface_failure_diagnostics_record_line_search_failure():
    model = _make_length_scaled_illingworth_model(1.0)
    model.setup()
    p = model._p_curr.copy()
    q = model._q_curr.copy()
    s = float(model._s_curr)
    eta = float(model._eta_curr)

    def zero_step(_jacobian, _residual):
        return np.zeros(2, dtype=np.float64)

    model._least_squares_step_2xN = zero_step
    model.residualTolerance = -1.0

    with pytest.raises(RuntimeError, match="failed to converge"):
        model._solve_interface_planar(p, q, s, model._s_old, eta, 1.0e-4)

    assert model._lastImplicitConverged is False
    assert model._lastImplicitFailureReason == "line search failed"
    assert model._lastImplicitIterations == 1
    assert model._lastImplicitCandidateEvaluations == 3
    assert model._lastImplicitJacobianEvaluations == 1
    assert model._lastImplicitMotionBranch == "positive"
    assert np.isfinite(model._lastImplicitResidual)
    assert np.isfinite(model._lastImplicitPhysicalResidual)


def test_ternary_interface_failure_diagnostics_record_maximum_iterations():
    model = _make_length_scaled_illingworth_model(1.0)
    model.setup()
    p = model._p_curr.copy()
    q = model._q_curr.copy()
    s = float(model._s_curr)
    eta = float(model._eta_curr)
    model.maxIterations = 1
    model.residualTolerance = -1.0

    with pytest.raises(RuntimeError, match="failed to converge"):
        model._solve_interface_planar(p, q, s, model._s_old, eta, 1.0e-4)

    assert model._lastImplicitConverged is False
    assert model._lastImplicitFailureReason == "maximum iterations reached"
    assert model._lastImplicitIterations == 1
    assert model._lastImplicitCandidateEvaluations == 4
    assert model._lastImplicitJacobianEvaluations == 1
    assert model._lastImplicitMotionBranch == "positive"


def test_ternary_interface_diagnostics_reset_before_candidate_evaluation_failure():
    model = _make_length_scaled_illingworth_model(1.0)
    model.setup()
    x = model.getCurrentX()
    model.getdXdt(model.currentTime, x)
    previous_residual = float(model._lastImplicitResidual)

    def failing_interface_compositions(_eta):
        raise ValueError("forced interface failure")

    model._interface_compositions = failing_interface_compositions
    p = model._p_curr.copy()
    q = model._q_curr.copy()

    with pytest.raises(ValueError, match="forced interface failure"):
        model._solve_interface_planar(p, q, float(model._s_curr), float(model._s_old), float(model._eta_curr), 1.0e-4)

    assert np.isfinite(previous_residual)
    assert model._lastImplicitConverged is False
    assert model._lastImplicitFailureReason == "candidate evaluation failed"
    assert model._lastImplicitIterations == 1
    assert model._lastImplicitCandidateEvaluations == 1
    assert model._lastImplicitJacobianEvaluations == 0
    assert model._lastImplicitMotionBranch == "positive"
    assert np.isinf(model._lastImplicitResidual)
    assert np.isinf(model._lastImplicitPhysicalResidual)


def test_ternary_getdxdt_reduces_timestep_after_failed_interface_solve():
    model = _make_length_scaled_illingworth_model(1.0)
    model.setup()
    original_step = model._take_implicit_step_planar
    calls = []

    def fail_once_then_solve(p, q, s, old_s, eta, dt):
        calls.append(float(dt))
        if len(calls) == 1:
            raise RuntimeError("forced first-step failure")
        return original_step(p, q, s, old_s, eta, dt)

    model._take_implicit_step_planar = fail_once_then_solve
    x = model.getCurrentX()
    model.getdXdt(model.currentTime, x)

    assert calls == [1.0e-4, 5.0e-5]
    assert model._lastStepRetries == 1
    assert model._currdt == 5.0e-5
    assert model._lastImplicitConverged is True


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
    left_result = model._solve_concentration_left_planar(p, s, future_s, dt, c_left, D_left, motion_branch)
    right_result = model._solve_concentration_right_planar(q, s, future_s, dt, c_right, D_right, motion_branch)
    p_future = left_result.profile
    q_future = right_result.profile

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
        left_result.interface_flux,
        right_result.interface_flux,
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
    left_result = model._solve_concentration_left_planar(p, s, future_s, dt, c_left, D_left, motion_branch)
    right_result = model._solve_concentration_right_planar(q, s, future_s, dt, c_right, D_right, motion_branch)
    p_future = left_result.profile
    q_future = right_result.profile

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
        left_result.interface_flux,
        right_result.interface_flux,
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


def test_ternary_illingworth_stationary_when_bulk_equals_phase_interface_compositions():
    equilibrium = _EtaVaryingInterfaceEquilibrium()
    eta0 = 0.4
    interface_position = 0.45
    left, right = equilibrium.interface_compositions(eta0)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(
        ProfileBuilder([(StepProfile1D(interface_position, left, right), ["X", "Y"])]),
        boundaryConditions=MixedBoundary1D(2),
    )
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=_CoupledTernaryThermodynamics(),
        temperature=1000.0,
        interfacePosition=interface_position,
        interface_equilibrium=equilibrium,
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(0.0, 1.0),
        time_step=1.0e-4,
        tolerance=1.0e-12,
        max_iterations=50,
        record=True,
    )

    model.solve(1.0e-3, minDtFrac=1.0e-10)

    n_records = model.interfaceData.N + 1
    positions = np.asarray(model.interfaceData._y[:n_records], dtype=np.float64)
    etas = np.asarray(model.etaData._y[:n_records], dtype=np.float64)
    p_history = np.asarray(model.pData._y[:n_records], dtype=np.float64)
    q_history = np.asarray(model.qData._y[:n_records], dtype=np.float64)
    composition_history = np.asarray(model.data._y[:n_records], dtype=np.float64)

    assert np.isclose(model.initialEta, eta0, rtol=0.0, atol=1.0e-10)
    assert n_records > 1
    assert np.allclose(positions, interface_position, rtol=0.0, atol=1.0e-14)
    assert np.allclose(etas, model.initialEta, rtol=0.0, atol=1.0e-12)
    assert np.allclose(p_history, p_history[0], rtol=0.0, atol=1.0e-13)
    assert np.allclose(q_history, q_history[0], rtol=0.0, atol=1.0e-13)
    assert np.allclose(composition_history, composition_history[0], rtol=0.0, atol=1.0e-13)
    assert np.allclose(model.checkConservation(1.0e-12), np.zeros(2), rtol=0.0, atol=1.0e-12)


def test_ternary_illingworth_quasi_binary_cu_zn_dummy_matches_binary_brass_case():
    from examples.Illingworth2005.compare_illingworth2005_planar import (
        build_fig3_present_work_model,
        build_fig6_illingworth_case_params,
    )

    dummy_x = 0.02
    params = build_fig6_illingworth_case_params(
        "thin",
        1.4e-8,
        {
            "fig6_dt_mode": ["fixed", "semi_log"][1],
            "fig6_time_step_s": 0.5,
            "fig6_semiLog_dt": 0.025, #0.00025,
            "fig6_semiLogT0": 1e-5,
            "fig6_t_end_s": 1e4, #1e3,
            "fig6_n_phase_a_nodes": 12,
            "fig6_n_phase_b_nodes": 16,
            "fig6_record": 1,
            "fig6_record_pq_data": True,
            "fig6_preallocate_recordings": False,
            "fig6_check_against_authors_cpp": False,
            "tolerance":1.0e-15,
        },
    )
    binary = build_fig3_present_work_model(params, record=True)
    binary.solve(params["t_end_s"], minDtFrac=1.0e-14, verbose=True, vIt=10000)

    left_bulk = np.asarray([params["c_liquid0_atpct"] / 100.0, dummy_x], dtype=np.float64)
    right_bulk = np.asarray([params["c_solid0_atpct"] / 100.0, dummy_x], dtype=np.float64)
    profile = ProfileBuilder([(StepProfile1D(params["s0_um"], left_bulk, right_bulk), ["ZN", "DUMMY"])])
    mesh = CartesianFD1D(["ZN", "DUMMY"], [0.0, params["R_um"]], params["n_alpha"] + params["n_beta"] - 1)
    mesh.setResponseProfile(profile, boundaryConditions=MixedBoundary1D(2))
    equilibrium = _QuasiBinaryCuZnDummyEquilibrium(
        zn_left=params["c_liquid_int_atpct"] / 100.0,
        zn_right=params["c_solid_int_atpct"] / 100.0,
        dummy=dummy_x,
    )
    thermodynamics = _QuasiBinaryTernaryThermodynamics(
        diffusivities={
            params["phase_a_name"]: params["D_liquid_um2_s"],
            params["phase_b_name"]: params["D_solid_um2_s"],
        },
        dummy_diffusivity=0.5,
    )

    for phase in (params["phase_a_name"], params["phase_b_name"]):
        D = thermodynamics.getInterdiffusivity(left_bulk, 1000.0, phase=phase)
        assert D.shape == (2, 2)
        assert D[0, 1] == 0.0
        assert D[1, 0] == 0.0
        assert D[1, 1] > 0.0

    ternary = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["CU", "ZN", "DUMMY"],
        phases=[params["phase_a_name"], params["phase_b_name"]],
        thermodynamics=thermodynamics,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=params["s0_um"],
        interface_equilibrium=equilibrium,
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(0.0, 1.0),
        initial_eta_guess=0.5,
        time_step=params["time_step_s"],
        dt_mode=params["dt_mode"],
        semiLog_dt=params["semiLog_dt"],
        semiLogT0=params["semiLogT0"],
        phase_a_nodes=params["n_alpha"],
        phase_b_nodes=params["n_beta"],
        tolerance=5.0e-16, #1.0e-17,
        max_iterations=25,
        record=True,
    )
    ternary.solve(params["t_end_s"], minDtFrac=1.0e-14, verbose=True, vIt=10000)

    binary_n = binary.interfaceData.N + 1
    ternary_n = ternary.interfaceData.N + 1
    assert ternary_n == binary_n
    assert np.allclose(ternary.interfaceData._time[:ternary_n], binary.interfaceData._time[:binary_n], rtol=0.0, atol=0.0)
    assert np.allclose(ternary.interfaceData._y[:ternary_n], binary.interfaceData._y[:binary_n], rtol=0.0, atol=1.0e-10)

    binary_p, binary_q = binary.getTransformedState()
    ternary_p, ternary_q = ternary.getTransformedState()
    assert np.allclose(ternary_p[:, 0], binary_p, rtol=0.0, atol=1.0e-10)
    assert np.allclose(ternary_q[:, 0], binary_q, rtol=0.0, atol=1.0e-10)
    assert np.allclose(ternary.data.y()[:, 0], binary.data.y()[:, 0], rtol=0.0, atol=1.0e-10)

    assert np.allclose(ternary_p[:, 1], dummy_x, rtol=0.0, atol=1.0e-13)
    assert np.allclose(ternary_q[:, 1], dummy_x, rtol=0.0, atol=1.0e-13)
    assert np.allclose(ternary.data.y()[:, 1], dummy_x, rtol=0.0, atol=1.0e-13)
    assert np.isclose(ternary.getTotalInventory()[1], dummy_x * params["R_um"], rtol=0.0, atol=1.0e-10)
    assert np.allclose(ternary.checkConservation(1.0e-10), np.zeros(2), rtol=0.0, atol=1.0e-10)


@pytest.mark.parametrize(
    "case_name, left_bulk, right_bulk, direction",
    [
        ("moves_right", np.asarray([0.16, 0.06], dtype=np.float64), np.asarray([0.3261538461538461, 0.193], dtype=np.float64), 1.0),
        ("moves_left", np.asarray([0.16, 0.12], dtype=np.float64), np.asarray([0.56, 0.25], dtype=np.float64), -1.0),
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
