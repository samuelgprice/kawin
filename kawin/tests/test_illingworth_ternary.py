from pathlib import Path

import numpy as np
import pytest

import kawin.diffusion.MovingBoundarySurrogates as surrogate_module
import kawin.diffusion.MovingBoundaryIllingworthTernaryFDM as ternary_fdm
from kawin.diffusion import (
    MovingBoundaryIllingworthTernaryFD1DModel,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    estimate_initial_eta_from_instantaneous_balance,
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


class _NoContextScalarBulkThermodynamics:
    def __init__(self):
        self.calls = []

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None):
        values = np.asarray(composition, dtype=np.float64)
        if values.ndim != 1:
            raise TypeError("scalar-only thermodynamics")
        self.calls.append({"phase": phase, "composition": values.copy(), "temperature": float(temperature)})
        x, y = values
        return np.asarray(
            [
                [1.0e-3 + 1.0e-4 * x, 2.0e-4 + 1.0e-5 * y],
                [1.0e-4 + 1.0e-5 * x, 9.0e-4 + 1.0e-4 * y],
            ],
            dtype=np.float64,
        )


class _PhaseSelectiveDifficultBulkThermodynamics:
    def __init__(self, difficult_phases):
        self.difficult_phases = {str(phase) for phase in difficult_phases}
        self.calls = []

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        self.calls.append(
            {
                "phase": phase,
                "composition": values.copy(),
                "query_context": kwargs.get("query_context"),
            }
        )
        matrices = []
        for x, y in values:
            if phase in self.difficult_phases and kwargs.get("query_context") == "general":
                scale = 1.0 + 4.0 * x + 3.0 * y
                matrices.append(
                    np.asarray(
                        [
                            [1.0e-3 * scale, 1.0e-4 * (1.0 + y)],
                            [5.0e-5 * (1.0 + x), 8.0e-4 * (1.0 + 2.0 * x + 2.0 * y)],
                        ],
                        dtype=np.float64,
                    )
                )
            elif phase == "ALPHA":
                matrices.append(np.asarray([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]], dtype=np.float64))
            else:
                matrices.append(np.asarray([[7.0e-4, -1.0e-4], [2.0e-4, 1.1e-3]], dtype=np.float64))
        matrices = np.asarray(matrices, dtype=np.float64)
        return matrices[0] if single else matrices


class _PiecewiseBulkTernaryThermodynamics:
    def __init__(self, threshold, direction, component=0, low=None, high=None):
        self.threshold = float(threshold)
        self.direction = float(direction)
        self.component = int(component)
        self.low = np.asarray(
            [[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]] if low is None else low,
            dtype=np.float64,
        )
        self.high = np.asarray(
            [[1.9e-3, 2.5e-4], [1.2e-4, 1.4e-3]] if high is None else high,
            dtype=np.float64,
        )
        self.calls = []

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        self.calls.append(
            {
                "phase": phase,
                "composition": values.copy(),
                "query_context": kwargs.get("query_context"),
            }
        )
        if phase != "ALPHA" or kwargs.get("query_context") != "general":
            matrices = np.broadcast_to(self.low, (values.shape[0], 2, 2)).copy()
        else:
            projected = self.direction * values[:, self.component]
            matrices = np.asarray([self.high if value > self.threshold else self.low for value in projected], dtype=np.float64)
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


class _SeedScanThermodynamics:
    def __init__(
        self,
        *,
        seed_y=0.20,
        negative_extent=0.03,
        positive_extent=0.07,
        half_tieline_length=0.04,
        endpoint_mode="valid",
    ):
        self.elements = ["Z", "X", "Y"]
        self.phases = ["ALPHA", "BETA"]
        self.seed_y = float(seed_y)
        self.negative_extent = float(negative_extent)
        self.positive_extent = float(positive_extent)
        self.half_tieline_length = float(half_tieline_length)
        self.endpoint_mode = endpoint_mode

    def clearCache(self):
        pass

    def _inside_two_phase_region(self, composition):
        y = float(np.asarray(composition, dtype=np.float64).reshape(2)[1])
        return self.seed_y - self.negative_extent <= y <= self.seed_y + self.positive_extent

    def getInterfacialComposition(self, x, T, gExtra=0, precPhase=None, returnMeta=False):
        x = np.asarray(x, dtype=np.float64).reshape(2)
        left = np.asarray([x[0] - self.half_tieline_length, x[1]], dtype=np.float64)
        right = np.asarray([x[0] + self.half_tieline_length, x[1]], dtype=np.float64)
        endpoints = [
            {"phase": "ALPHA", "composition": left},
            {"phase": "BETA", "composition": right},
        ]
        if self.endpoint_mode == "extra" or not self._inside_two_phase_region(x):
            endpoints = [
                {"phase": "ALPHA", "composition": left},
                {"phase": "GAMMA", "composition": right},
            ]
        if not returnMeta:
            return left, right
        return left, right, {"endpoint_phases": tuple(e.get("phase") for e in endpoints), "endpoints": tuple(endpoints)}

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        return _test_continuous_matrix(composition, phase)


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

    def left_spy(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch, validate_diffusivity=True):
        records.append(("left", float(future_s_arg), motion_branch))
        return left_original(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch, validate_diffusivity=validate_diffusivity)

    def right_spy(q_arg, s_arg, future_s_arg, dt_arg, c_right_arg, D_right_arg, motion_branch, validate_diffusivity=True):
        records.append(("right", float(future_s_arg), motion_branch))
        return right_original(q_arg, s_arg, future_s_arg, dt_arg, c_right_arg, D_right_arg, motion_branch, validate_diffusivity=validate_diffusivity)

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

    def left_spy(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch, validate_diffusivity=True):
        future_s_calls.append(float(future_s_arg))
        return left_original(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch, validate_diffusivity=validate_diffusivity)

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


def test_ternary_phase_uniform_solve_rejects_complex_diffusivity_before_real_conversion():
    model, p, _ = _make_residual_identity_state()
    s = float(model._s_curr)
    c_left = np.asarray([0.25, 0.10], dtype=np.float64)
    D_left = np.asarray([[1.0e-3 + 1.0e-8j, 0.0], [0.0, 8.0e-4]], dtype=np.complex128)

    with pytest.raises(ValueError, match="real-valued"):
        model._solve_concentration_left_planar(p, s, s + 1.0e-4, 1.0e-4, c_left, D_left, "positive")


def test_ternary_face_array_solve_rejects_single_complex_face_before_real_conversion():
    model, p, _ = _make_residual_identity_state()
    s = float(model._s_curr)
    c_left = np.asarray([0.25, 0.10], dtype=np.float64)
    D_faces = np.broadcast_to(np.asarray([[1.0e-3, 1.0e-4], [5.0e-5, 8.0e-4]], dtype=np.float64), (len(p) - 1, 2, 2)).astype(np.complex128)
    D_faces[1, 0, 1] += 1.0e-8j

    with pytest.raises(ValueError, match="real-valued"):
        model._solve_concentration_left_planar(p, s, s + 1.0e-4, 1.0e-4, c_left, D_faces, "positive")


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

    def left_spy(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch, validate_diffusivity=True):
        bulk_calls.append(("ALPHA", np.asarray(D_left_arg, dtype=np.float64).copy()))
        return left_original(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch, validate_diffusivity=validate_diffusivity)

    def right_spy(q_arg, s_arg, future_s_arg, dt_arg, c_right_arg, D_right_arg, motion_branch, validate_diffusivity=True):
        bulk_calls.append(("BETA", np.asarray(D_right_arg, dtype=np.float64).copy()))
        return right_original(q_arg, s_arg, future_s_arg, dt_arg, c_right_arg, D_right_arg, motion_branch, validate_diffusivity=validate_diffusivity)

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


def test_ternary_bulk_face_diffusivity_falls_back_to_no_context_scalar_api():
    model = _make_scope_validation_model()
    model.setup()
    thermodynamics = _NoContextScalarBulkThermodynamics()
    model.therm = thermodynamics
    face_compositions = np.asarray([[0.18, 0.08], [0.24, 0.12], [0.30, 0.16]], dtype=np.float64)
    face_positions = np.asarray([0.2, 0.4, 0.6], dtype=np.float64)

    matrices = model._bulk_face_diffusivity_matrices(face_compositions, "ALPHA", model.currentTime, face_positions)

    assert matrices.shape == (3, 2, 2)
    assert len(thermodynamics.calls) == 3
    assert [call["phase"] for call in thermodynamics.calls] == ["ALPHA", "ALPHA", "ALPHA"]
    assert np.allclose([call["composition"] for call in thermodynamics.calls], face_compositions)


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


def test_ternary_implicit_constant_diffusivity_reproduces_phase_uniform_step_in_one_picard_cycle():
    uniform = _make_scope_validation_model()
    uniform.therm = _SmoothBulkTernaryThermodynamics(constant=True)
    uniform.setup()
    implicit = _make_scope_validation_model()
    implicit.therm = _SmoothBulkTernaryThermodynamics(constant=True)
    implicit.bulkDiffusivityMode = "composition_dependent_implicit"
    implicit.setup()

    d_uniform = uniform.getdXdt(uniform.currentTime, uniform.getCurrentX())
    d_implicit = implicit.getdXdt(implicit.currentTime, implicit.getCurrentX())

    for actual, expected in zip(d_implicit, d_uniform):
        assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)
    assert implicit._lastImplicitConverged is True
    assert implicit._lastBulkConverged is True
    assert implicit._lastBulkLeftPicardIterations == 1
    assert implicit._lastBulkRightPicardIterations == 1
    assert implicit._lastBulkDiffusivityEvaluations > 0
    assert implicit._lastStepRetries == uniform._lastStepRetries


def test_ternary_implicit_candidate_evaluation_is_deterministic_after_other_trials():
    model = _make_scope_validation_model()
    model.therm = _SmoothBulkTernaryThermodynamics()
    model.bulkDiffusivityMode = "composition_dependent_implicit"
    model.setup()
    p, q, s, eta = model.getCurrentX()
    dt = model._compute_dt(model.currentTime)
    eta_lower, _, eta_span = model._eta_scaling_bounds()
    residual_scale = model._interface_residual_scale(p, q, s)
    c_left_old = p[-1].copy()
    c_right_old = q[0].copy()
    x_hat = model._interface_physical_to_scaled(s, eta, eta_lower, eta_span)
    x_other = x_hat + np.asarray([0.01, -0.01], dtype=np.float64)
    motion_branch = ternary_fdm._select_interface_motion_branch(s, s, s)

    first = model._evaluate_interface_candidate(
        p,
        q,
        s,
        s,
        dt,
        eta_lower,
        eta_span,
        residual_scale,
        c_left_old,
        c_right_old,
        x_hat,
        motion_branch,
    )
    model._evaluate_interface_candidate(
        p,
        q,
        s,
        s,
        dt,
        eta_lower,
        eta_span,
        residual_scale,
        c_left_old,
        c_right_old,
        x_other,
        motion_branch,
    )
    repeated = model._evaluate_interface_candidate(
        p,
        q,
        s,
        s,
        dt,
        eta_lower,
        eta_span,
        residual_scale,
        c_left_old,
        c_right_old,
        x_hat,
        motion_branch,
    )

    assert np.array_equal(repeated.p_future, first.p_future)
    assert np.array_equal(repeated.q_future, first.q_future)
    assert np.array_equal(repeated.residual, first.residual)
    assert repeated.left_inner_iterations == first.left_inner_iterations
    assert repeated.right_inner_iterations == first.right_inner_iterations


def test_ternary_invalid_bulk_diffusivity_mode_after_setup_fails_explicitly():
    model = _make_scope_validation_model()
    model.setup()
    p, q, s, eta = model.getCurrentX()
    dt = model._compute_dt(model.currentTime)
    eta_lower, _, eta_span = model._eta_scaling_bounds()
    residual_scale = model._interface_residual_scale(p, q, s)
    x_hat = model._interface_physical_to_scaled(s, eta, eta_lower, eta_span)
    model.bulkDiffusivityMode = "definitely_not_a_mode"

    with pytest.raises(ValueError, match="bulkDiffusivityMode"):
        model._evaluate_interface_candidate(
            p,
            q,
            s,
            s,
            dt,
            eta_lower,
            eta_span,
            residual_scale,
            p[-1].copy(),
            q[0].copy(),
            x_hat,
            ternary_fdm._select_interface_motion_branch(s, s, s),
        )


def test_ternary_implicit_picard_reuses_next_coefficient_query():
    model = _make_scope_validation_model()
    model.therm = _SmoothBulkTernaryThermodynamics()
    model.bulkDiffusivityMode = "composition_dependent_implicit"
    model.bulkPicardRtol = 1.0e-13
    model.bulkPicardAtol = 1.0e-15
    model.bulkPicardMaxIterations = 20
    model.setup()
    p, _, s, eta = model.getCurrentX()
    c_left, _ = model._interface_compositions(eta + 1.0e-3)
    calls = {"count": 0}
    original = model._left_lagged_face_diffusivity_matrices

    def count_left_face_query(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    model._left_lagged_face_diffusivity_matrices = count_left_face_query

    result = model._solve_concentration_left_picard(p, s, s + 1.0e-4, 1.0e-4, c_left, "positive")

    assert result.inner_iterations > 1
    assert calls["count"] == result.inner_iterations


def test_ternary_under_relaxed_piecewise_coefficients_do_not_accept_unrelaxed_shortcut():
    model = _make_scope_validation_model()
    model.therm = _SmoothBulkTernaryThermodynamics(constant=True)
    model.bulkDiffusivityMode = "composition_dependent_implicit"
    model.bulkPicardRelaxation = 0.25
    model.bulkPicardRtol = 1.0e-12
    model.bulkPicardAtol = 1.0e-15
    model.bulkPicardMaxIterations = 80
    model.setup()
    p, _, s, eta = model.getCurrentX()
    c_left, _ = model._interface_compositions(eta + 0.05)
    future_s = s + 1.0e-4
    D_low = np.asarray([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]], dtype=np.float64)
    D_high = np.asarray([[1.9e-3, 2.5e-4], [1.2e-4, 1.4e-3]], dtype=np.float64)
    initial = model._left_candidate_initial_profile(p, c_left)
    first_linear = model._solve_concentration_left_planar(
        p,
        s,
        future_s,
        model.timeStep,
        c_left,
        np.broadcast_to(D_low, (len(p) - 1, 2, 2)),
        "positive",
    )
    initial_faces = model._left_lagged_face_compositions(initial)
    linear_faces = model._left_lagged_face_compositions(first_linear.profile)
    deltas = linear_faces - initial_faces
    flat_index = int(np.argmax(np.abs(deltas)))
    face_index, component = np.unravel_index(flat_index, deltas.shape)
    direction = 1.0 if deltas[face_index, component] > 0.0 else -1.0
    threshold = direction * (initial_faces[face_index, component] + 0.5 * deltas[face_index, component])
    relaxed_first = initial + model.bulkPicardRelaxation * (first_linear.profile - initial)
    relaxed_faces = model._left_lagged_face_compositions(relaxed_first)
    assert direction * relaxed_faces[face_index, component] < threshold
    assert direction * linear_faces[face_index, component] > threshold
    model.therm = _PiecewiseBulkTernaryThermodynamics(
        threshold=threshold,
        direction=direction,
        component=component,
        low=D_low,
        high=D_high,
    )

    first = model._solve_concentration_left_picard(p, s, future_s, model.timeStep, c_left, "positive")
    second = model._solve_concentration_left_picard(p, s, future_s, model.timeStep, c_left, "positive")
    final_faces = model._left_lagged_face_diffusivity_matrices(first.profile, future_s, model.currentTime)

    assert first.inner_iterations > 1
    assert first.inner_update_norm <= model.bulkPicardAtol + model.bulkPicardRtol * np.max(np.abs(first.profile))
    assert np.array_equal(first.profile, second.profile)
    assert np.array_equal(first.interface_face_matrix, final_faces[-1])
    assert np.allclose(first.interface_flux, model._left_interface_diffusive_flux(first.profile, future_s, c_left, final_faces[-1]))
    assert not np.array_equal(first.profile, first_linear.profile)


def test_ternary_implicit_under_relaxation_converges_to_same_step():
    direct = _make_scope_validation_model()
    direct.therm = _SmoothBulkTernaryThermodynamics()
    direct.bulkDiffusivityMode = "composition_dependent_implicit"
    direct.bulkPicardRtol = 1.0e-11
    direct.setup()
    relaxed = _make_scope_validation_model()
    relaxed.therm = _SmoothBulkTernaryThermodynamics()
    relaxed.bulkDiffusivityMode = "composition_dependent_implicit"
    relaxed.bulkPicardRtol = 1.0e-11
    relaxed.bulkPicardRelaxation = 0.5
    relaxed.bulkPicardMaxIterations = 60
    relaxed.setup()

    d_direct = direct.getdXdt(direct.currentTime, direct.getCurrentX())
    d_relaxed = relaxed.getdXdt(relaxed.currentTime, relaxed.getCurrentX())

    for actual, expected in zip(d_relaxed, d_direct):
        assert np.allclose(actual, expected, rtol=1.0e-8, atol=1.0e-12)
    assert relaxed._lastBulkLeftPicardIterations >= direct._lastBulkLeftPicardIterations
    assert relaxed._lastBulkRightPicardIterations >= direct._lastBulkRightPicardIterations
    assert relaxed._lastBulkConverged is True


def test_ternary_implicit_inner_failure_propagates_to_timestep_retry(monkeypatch):
    model = _make_scope_validation_model()
    model.therm = _SmoothBulkTernaryThermodynamics()
    model.bulkDiffusivityMode = "composition_dependent_implicit"
    model.setup()
    original = model._solve_concentration_left_picard
    calls = {"count": 0}

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            model._lastBulkFailureReason = "forced inner failure"
            raise RuntimeError("forced inner failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_solve_concentration_left_picard", fail_once)

    dXdt = model.getdXdt(model.currentTime, model.getCurrentX())

    assert calls["count"] > 1
    assert model._lastStepRetries == 1
    assert model._lastImplicitConverged is True
    assert model._lastBulkConverged is True
    assert all(np.all(np.isfinite(np.asarray(part, dtype=np.float64))) for part in dXdt)


def _interface_candidate_test_inputs(model, x_hat=None):
    p, q, s, eta = model.getCurrentX()
    dt = model._compute_dt(model.currentTime)
    eta_lower, _, eta_span = model._eta_scaling_bounds()
    residual_scale = model._interface_residual_scale(p, q, s)
    if x_hat is None:
        x_hat = model._interface_physical_to_scaled(s, eta, eta_lower, eta_span)
    return {
        "p": p,
        "q": q,
        "s": s,
        "old_s": s,
        "dt": dt,
        "eta_lower": eta_lower,
        "eta_span": eta_span,
        "residual_scale": residual_scale,
        "c_left_old": p[-1].copy(),
        "c_right_old": q[0].copy(),
        "x_hat": np.asarray(x_hat, dtype=np.float64),
        "motion_branch": ternary_fdm._select_interface_motion_branch(s, s, s),
    }


def test_ternary_candidate_left_failure_resets_previous_right_diagnostics(monkeypatch):
    model = _make_scope_validation_model()
    model.therm = _SmoothBulkTernaryThermodynamics(constant=True)
    model.bulkDiffusivityMode = "composition_dependent_implicit"
    model.setup()
    model._bulkDiffusivityCountingActive = True
    first_args = _interface_candidate_test_inputs(model)
    model._evaluate_interface_candidate(**first_args)
    previous_provider_calls = model._currentBulkDiffusivityProviderCalls
    assert model._lastBulkLeftPicardIterations == 1
    assert model._lastBulkRightPicardIterations == 1

    def fail_left(*args, **kwargs):
        raise RuntimeError("forced left candidate failure")

    monkeypatch.setattr(model, "_solve_concentration_left_picard", fail_left)
    second_args = _interface_candidate_test_inputs(model, first_args["x_hat"] + np.asarray([0.0, 1.0e-3]))

    with pytest.raises(RuntimeError, match="forced left candidate failure"):
        model._evaluate_interface_candidate(**second_args)

    assert model._lastBulkConverged is False
    assert "left bulk solve failed: forced left candidate failure" == model._lastBulkFailureReason
    assert model._lastBulkLeftPicardIterations == 0
    assert model._lastBulkRightPicardIterations == 0
    assert np.isinf(model._lastBulkRightUpdateNorm)
    assert model._currentBulkDiffusivityProviderCalls >= previous_provider_calls


def test_ternary_candidate_right_failure_keeps_same_candidate_left_diagnostics(monkeypatch):
    model = _make_scope_validation_model()
    model.therm = _SmoothBulkTernaryThermodynamics(constant=True)
    model.bulkDiffusivityMode = "composition_dependent_implicit"
    model.setup()

    def fail_right(*args, **kwargs):
        raise ValueError("forced right candidate failure")

    monkeypatch.setattr(model, "_solve_concentration_right_picard", fail_right)

    with pytest.raises(ValueError, match="forced right candidate failure"):
        model._evaluate_interface_candidate(**_interface_candidate_test_inputs(model))

    assert model._lastBulkConverged is False
    assert model._lastBulkLeftPicardIterations == 1
    assert model._lastBulkRightPicardIterations == 0
    assert model._lastBulkLeftUpdateNorm <= 1.0e-15
    assert np.isinf(model._lastBulkRightUpdateNorm)
    assert model._lastBulkFailureReason == "right bulk solve failed: forced right candidate failure"


def test_ternary_non_picard_phase_exception_is_classified_without_replacement(monkeypatch):
    model = _make_scope_validation_model()
    model.therm = _SmoothBulkTernaryThermodynamics(constant=True)
    model.bulkDiffusivityMode = "composition_dependent_implicit"
    model.setup()

    def fail_block_solve(*args, **kwargs):
        raise ArithmeticError("block solve blew up")

    monkeypatch.setattr(model, "_solve_concentration_left_planar", fail_block_solve)

    with pytest.raises(ArithmeticError, match="block solve blew up"):
        model._evaluate_interface_candidate(**_interface_candidate_test_inputs(model))

    assert model._lastBulkConverged is False
    assert model._lastBulkLeftPicardIterations == 0
    assert "Picard solve failed to converge" not in model._lastBulkFailureReason
    assert model._lastBulkFailureReason == "left bulk solve failed: block solve blew up"


def test_ternary_outer_line_search_failure_does_not_mark_bulk_failed(monkeypatch):
    model = _make_scope_validation_model()
    model.therm = _SmoothBulkTernaryThermodynamics(constant=True)
    model.bulkDiffusivityMode = "composition_dependent_implicit"
    model.setup()
    monkeypatch.setattr(model, "_interface_candidate_has_converged", lambda candidate, lower, upper: False)
    monkeypatch.setattr(model, "_interface_candidate_improves", lambda candidate, previous_norm: False)
    p, q, s, eta = model.getCurrentX()

    with pytest.raises(RuntimeError, match="failed to converge"):
        model._solve_interface_planar(p, q, s, s, eta, model.timeStep)

    assert model._lastImplicitConverged is False
    assert model._lastImplicitFailureReason == "line search failed"
    assert model._lastBulkConverged is True
    assert model._lastBulkFailureReason is None
    assert model._lastBulkLeftPicardIterations == 1
    assert model._lastBulkRightPicardIterations == 1


def test_ternary_outer_max_iteration_failure_does_not_mark_bulk_failed(monkeypatch):
    model = _make_scope_validation_model()
    model.therm = _SmoothBulkTernaryThermodynamics(constant=True)
    model.bulkDiffusivityMode = "composition_dependent_implicit"
    model.maxIterations = 2
    model.setup()
    monkeypatch.setattr(model, "_interface_candidate_has_converged", lambda candidate, lower, upper: False)
    monkeypatch.setattr(model, "_interface_candidate_improves", lambda candidate, previous_norm: True)
    p, q, s, eta = model.getCurrentX()

    with pytest.raises(RuntimeError, match="failed to converge"):
        model._solve_interface_planar(p, q, s, s, eta, model.timeStep)

    assert model._lastImplicitConverged is False
    assert model._lastImplicitFailureReason == "maximum iterations reached"
    assert model._lastBulkConverged is True
    assert model._lastBulkFailureReason is None
    assert model._lastBulkLeftPicardIterations == 1
    assert model._lastBulkRightPicardIterations == 1


@pytest.mark.parametrize(
    "difficult_phase, expected_reason",
    [
        ("ALPHA", "left bulk Picard solve failed"),
        ("BETA", "right bulk Picard solve failed"),
    ],
)
def test_ternary_genuine_phase_picard_failure_identifies_phase(difficult_phase, expected_reason):
    model = _make_scope_validation_model()
    model.therm = _PhaseSelectiveDifficultBulkThermodynamics({difficult_phase})
    model.bulkDiffusivityMode = "composition_dependent_implicit"
    model.bulkPicardRtol = 1.0e-14
    model.bulkPicardAtol = 1.0e-16
    model.bulkPicardMaxIterations = 1
    model.setup()
    p, q, s, eta = model.getCurrentX()
    c_left, c_right = model._interface_compositions(eta + 0.05)

    with pytest.raises(RuntimeError, match=expected_reason):
        if difficult_phase == "ALPHA":
            model._solve_concentration_left_picard(p, s, s + 1.0e-4, model.timeStep, c_left, "positive")
        else:
            model._solve_concentration_right_picard(q, s, s + 1.0e-4, model.timeStep, c_right, "positive")

    assert model._lastBulkConverged is False
    assert expected_reason in model._lastBulkFailureReason
    if difficult_phase == "ALPHA":
        assert model._lastBulkLeftPicardIterations == 1
        assert model._lastBulkRightPicardIterations == 0
    else:
        assert model._lastBulkLeftPicardIterations == 0
        assert model._lastBulkRightPicardIterations == 1


def test_ternary_real_picard_nonconvergence_reaches_retry_limit():
    model = _make_scope_validation_model()
    model.therm = _PhaseSelectiveDifficultBulkThermodynamics({"ALPHA"})
    model.bulkDiffusivityMode = "composition_dependent_implicit"
    model.bulkPicardRtol = 1.0e-14
    model.bulkPicardAtol = 1.0e-16
    model.bulkPicardMaxIterations = 1
    model.maxStepRetries = 2
    model.setup()
    x = model.getCurrentX()
    x[3] = float(x[3] + 0.05)

    with pytest.raises(RuntimeError, match="failed after timestep retries"):
        model.getdXdt(model.currentTime, x)

    assert model._lastImplicitConverged is False
    assert model._lastImplicitFailureReason == "candidate evaluation failed"
    assert model._lastBulkConverged is False
    assert "left bulk Picard solve failed" in model._lastBulkFailureReason


def test_ternary_implicit_tolerance_refinement_has_stable_limit():
    loose = _make_scope_validation_model()
    loose.therm = _SmoothBulkTernaryThermodynamics()
    loose.bulkDiffusivityMode = "composition_dependent_implicit"
    loose.bulkPicardRtol = 1.0e-8
    loose.bulkPicardAtol = 1.0e-12
    loose.setup()
    tight = _make_scope_validation_model()
    tight.therm = _SmoothBulkTernaryThermodynamics()
    tight.bulkDiffusivityMode = "composition_dependent_implicit"
    tight.bulkPicardRtol = 1.0e-12
    tight.bulkPicardAtol = 1.0e-15
    tight.bulkPicardMaxIterations = 40
    tight.setup()

    d_loose = loose.getdXdt(loose.currentTime, loose.getCurrentX())
    d_tight = tight.getdXdt(tight.currentTime, tight.getCurrentX())

    for actual, expected in zip(d_loose, d_tight):
        assert np.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-10)
    assert tight._lastBulkLeftUpdateNorm <= loose._lastBulkLeftUpdateNorm
    assert tight._lastBulkRightUpdateNorm <= loose._lastBulkRightUpdateNorm


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


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_ternary_implicit_mode_conserves_inventory_for_opposite_motion(direction):
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
        bulk_diffusivity_mode="composition_dependent_implicit",
        bulk_picard_rtol=1.0e-10,
        bulk_picard_atol=1.0e-13,
        bulk_picard_max_iterations=40,
        time_step=5.0e-5,
        tolerance=1.0e-10,
        max_iterations=50,
        record=True,
    )
    model.solve(2.0e-4, minDtFrac=1.0e-12)

    inventories = np.asarray(model.inventoryData._y[: model.inventoryData.N + 1], dtype=np.float64)
    assert model._lastImplicitConverged is True
    assert model._lastBulkConverged is True
    assert model._lastBulkLeftPicardIterations > 0
    assert model._lastBulkRightPicardIterations > 0
    assert model._lastBulkDiffusivityProviderCalls > 0
    assert model._lastBulkFaceMatricesEvaluated > 0
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

    def left_spy(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch, validate_diffusivity=True):
        evaluated.append(("s", float(future_s_arg) / model._R))
        return left_original(p_arg, s_arg, future_s_arg, dt_arg, c_left_arg, D_left_arg, motion_branch, validate_diffusivity=validate_diffusivity)

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
        estimate_initial_eta_from_instantaneous_balance(
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
        (_LinearInterfaceEquilibrium(), _IdentityTernaryThermodynamics(invalid=True), "positive real eigenvalues"),
    ],
)
def test_initial_eta_estimator_rejects_invalid_candidates(equilibrium, thermodynamics, match):
    with pytest.raises(ValueError, match=match):
        estimate_initial_eta_from_instantaneous_balance(
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


def test_ternary_illingworth_rejects_removed_stefan_initial_eta_method():
    left, right = _LinearInterfaceEquilibrium().interface_compositions(0.25)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))

    with pytest.raises(ValueError, match="initial_eta_method must be 'instantaneous_balance'"):
        MovingBoundaryIllingworthTernaryFD1DModel(
            mesh=mesh,
            elements=["Z", "X", "Y"],
            phases=["ALPHA", "BETA"],
            thermodynamics=_IdentityTernaryThermodynamics(),
            temperature=1000.0,
            interfacePosition=0.5,
            interface_equilibrium=_LinearInterfaceEquilibrium(),
            initial_eta_method="stefan_cross_brentq",
            time_step=1.0,
            record=True,
        )


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


def _build_seed_surrogate(**kwargs):
    params = {
        "thermodynamics": _SeedScanThermodynamics(),
        "elements": ["Z", "X", "Y"],
        "phases": ["ALPHA", "BETA"],
        "tieline_phases": ("ALPHA", "BETA"),
        "temperature": 1000.0,
        "probe_point": np.asarray([0.30, 0.20], dtype=np.float64),
        "probe_samples_per_side": 2,
        "probe_boundary_margin": 0.005,
        "probe_boundary_search_step": 0.02,
        "probe_boundary_xtol": 1.0e-6,
    }
    params.update(kwargs)
    return TernaryMovingBoundaryThermodynamicsSurrogate.from_database(**params)


def _continuous_bulk_grids():
    return (
        np.asarray([0.18, 0.24, 0.30, 0.36, 0.42], dtype=np.float64),
        np.asarray([0.06, 0.10, 0.14, 0.18, 0.22], dtype=np.float64),
    )


def _test_continuous_matrix(composition, phase, scale=1.0):
    x, y = np.asarray(composition, dtype=np.float64)
    if phase == "ALPHA":
        return scale * np.asarray([[1.0 + x + 0.2 * y, 0.05 * y], [0.02 * y, 1.4 + x + 0.1 * y]], dtype=np.float64)
    return scale * np.asarray([[1.5 + x + 0.1 * y, -0.03 * y], [0.01 * y, 1.8 + x + 0.2 * y]], dtype=np.float64)


def _direct_continuous_surrogate(*, bulk_scale=10.0, invalid_bulk=False):
    eta = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    tielines = {
        "ALPHA": np.asarray([[0.20, 0.10], [0.25, 0.10], [0.30, 0.10]], dtype=np.float64),
        "BETA": np.asarray([[0.30, 0.15], [0.35, 0.15], [0.40, 0.15]], dtype=np.float64),
    }
    grids = _continuous_bulk_grids()
    grid_points = np.asarray(np.meshgrid(*grids, indexing="ij"), dtype=np.float64).reshape(2, -1).T
    interface_diffusivities = {
        phase: np.asarray([_test_continuous_matrix(point, phase) for point in tielines[phase]], dtype=np.float64)
        for phase in ("ALPHA", "BETA")
    }
    if invalid_bulk:
        bulk_diffusivities = {
            phase: np.tile(np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64), (grid_points.shape[0], 1, 1))
            for phase in ("ALPHA", "BETA")
        }
    else:
        bulk_diffusivities = {
            phase: np.asarray([_test_continuous_matrix(point, phase, scale=bulk_scale) for point in grid_points], dtype=np.float64)
            for phase in ("ALPHA", "BETA")
        }
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        tieline_phases=("ALPHA", "BETA"),
        temperature=1000.0,
        eta_samples=eta,
        tieline_compositions=tielines,
        diffusivity_compositions={
            "interface": {phase: tielines[phase] for phase in ("ALPHA", "BETA")},
            "general": {phase: grid_points for phase in ("ALPHA", "BETA")},
        },
        diffusivities={"interface": interface_diffusivities, "general": bulk_diffusivities},
        diffusivity_interpolation="continuous_grid",
        diffusivity_bulk_grids=grids,
    )


def _direct_nearest_surrogate(*, scale=1.0, invalid=False):
    eta = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    tielines = {
        "ALPHA": np.asarray([[0.20, 0.10], [0.25, 0.10], [0.30, 0.10]], dtype=np.float64),
        "BETA": np.asarray([[0.30, 0.15], [0.35, 0.15], [0.40, 0.15]], dtype=np.float64),
    }
    general_points = np.asarray([[0.20, 0.10], [0.30, 0.10], [0.30, 0.15], [0.40, 0.15]], dtype=np.float64)
    if invalid:
        interface_diffusivities = {
            phase: np.tile(np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64), (eta.size, 1, 1))
            for phase in ("ALPHA", "BETA")
        }
        general_diffusivities = {
            phase: np.tile(np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64), (general_points.shape[0], 1, 1))
            for phase in ("ALPHA", "BETA")
        }
    else:
        interface_diffusivities = {
            phase: np.asarray([_test_continuous_matrix(point, phase, scale=scale) for point in tielines[phase]], dtype=np.float64)
            for phase in ("ALPHA", "BETA")
        }
        general_diffusivities = {
            phase: np.asarray([_test_continuous_matrix(point, phase, scale=scale) for point in general_points], dtype=np.float64)
            for phase in ("ALPHA", "BETA")
        }
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        tieline_phases=("ALPHA", "BETA"),
        temperature=1000.0,
        eta_samples=eta,
        tieline_compositions=tielines,
        diffusivity_compositions={
            "interface": {phase: tielines[phase] for phase in ("ALPHA", "BETA")},
            "general": {phase: general_points for phase in ("ALPHA", "BETA")},
        },
        diffusivities={"interface": interface_diffusivities, "general": general_diffusivities},
    )


class _ContinuousMatrixTruth:
    def __init__(self, scale=1.0):
        self.scale = float(scale)
        self.elements = ["Z", "X", "Y"]
        self.phases = ["ALPHA", "BETA"]

    def clearCache(self):
        return

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        return _test_continuous_matrix(composition, phase, scale=self.scale)


class _ZeroMatrixTruth(_ContinuousMatrixTruth):
    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        return np.zeros((2, 2), dtype=np.float64)


class _RecordingSurrogateDiffusivity:
    def __init__(self, surrogate):
        self.surrogate = surrogate
        self.calls = []

    def clearCache(self):
        self.surrogate.clearCache()

    def getInterdiffusivity(self, composition, temperature, phase=None, query_context=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        self.calls.append(
            {
                "phase": phase,
                "query_context": query_context,
                "composition_shape": values.shape,
            }
        )
        return self.surrogate.getInterdiffusivity(
            composition,
            temperature,
            phase=phase,
            query_context=query_context,
            **kwargs,
        )


def test_ternary_surrogate_requires_explicit_tieline_phases():
    with pytest.raises(ValueError, match="tieline_phases"):
        _build_surrogate(tieline_phases=None)


@pytest.mark.parametrize("endpoint_mode", ["missing", "extra", "duplicate", "unlabeled"])
def test_ternary_surrogate_rejects_unexpected_tieline_phase_metadata(endpoint_mode):
    with pytest.raises(ValueError):
        _build_surrogate(thermodynamics=_TieLineSamplingThermodynamics(endpoint_mode=endpoint_mode))


def test_ternary_surrogate_seed_point_builds_asymmetric_eta_samples():
    surrogate = _build_seed_surrogate()

    assert surrogate.metadata["source"] == "from_database_seed_point"
    assert surrogate.eta_samples.shape == (5,)
    assert np.isclose(surrogate.eta_samples[0], 0.0)
    assert np.isclose(surrogate.eta_samples[-1], 1.0)
    assert np.all(np.diff(surrogate.eta_samples) > 0.0)
    assert not np.isclose(surrogate.eta_samples[2], 0.5)
    assert np.allclose(surrogate.metadata["probe_scan_direction"], [0.0, 1.0])

    generated = np.asarray(surrogate.metadata["generated_probe_points"], dtype=np.float64)
    assert generated.shape == (5, 2)
    assert np.min(generated[:, 1]) >= 0.20 - 0.03 + 0.005 - 1.0e-6
    assert np.max(generated[:, 1]) <= 0.20 + 0.07 - 0.005 + 1.0e-6


def test_ternary_surrogate_seed_point_rejects_invalid_seed_metadata():
    with pytest.raises(ValueError, match="Unexpected tie-line phases"):
        _build_seed_surrogate(thermodynamics=_SeedScanThermodynamics(endpoint_mode="extra"))


def test_ternary_surrogate_seed_point_rejects_mixed_line_arguments():
    with pytest.raises(ValueError, match="probe_point seed mode cannot be combined"):
        _build_seed_surrogate(probe_start=np.asarray([0.20, 0.10], dtype=np.float64))
    with pytest.raises(ValueError, match="probe_point seed mode cannot be combined"):
        _build_seed_surrogate(eta_samples=np.asarray([0.0, 0.5, 1.0], dtype=np.float64))


def test_ternary_surrogate_line_mode_still_requires_line_arguments():
    with pytest.raises(ValueError, match="probe_start, probe_end, and eta_samples"):
        _build_surrogate(probe_start=None)


def test_ternary_surrogate_seed_point_margin_must_fit_detected_boundaries():
    with pytest.raises(ValueError, match="probe_boundary_margin"):
        _build_seed_surrogate(probe_boundary_margin=0.031)


def test_seed_scan_resample_rejects_generated_samples_outside_expected_region():
    thermodynamics = _SeedScanThermodynamics()
    seed_sample = surrogate_module._sample_expected_tieline(
        thermodynamics,
        [0.30, 0.20],
        1000.0,
        "BETA",
        ("ALPHA", "BETA"),
        ["Z", "X", "Y"],
        1.0e-10,
    )

    with pytest.raises(ValueError, match="generated probe left the expected two-phase region"):
        surrogate_module._resample_seed_scan_side(
            thermodynamics,
            seed_sample,
            1.0,
            np.asarray([0.0, 1.0], dtype=np.float64),
            1000.0,
            "BETA",
            ("ALPHA", "BETA"),
            ["Z", "X", "Y"],
            1.0e-10,
            0.08,
            0.0,
            1,
        )


@pytest.mark.parametrize("diffusivity_interpolation", ["nearest", "continuous_grid", "simplex_linear"])
def test_ternary_surrogate_seed_point_supports_diffusivity_interpolation_modes(diffusivity_interpolation):
    kwargs = {}
    if diffusivity_interpolation == "continuous_grid":
        kwargs["diffusivity_bulk_grids"] = _continuous_bulk_grids()
    elif diffusivity_interpolation == "simplex_linear":
        kwargs["diffusivity_bulk_grids"] = (
            np.asarray([0.18, 0.30, 0.42, 0.70], dtype=np.float64),
            np.asarray([0.06, 0.18, 0.30, 0.45], dtype=np.float64),
        )
    surrogate = _build_seed_surrogate(diffusivity_interpolation=diffusivity_interpolation, **kwargs)

    matrix = surrogate.getInterdiffusivity([0.30, 0.20], 1000.0, phase="ALPHA", query_context="general")

    assert surrogate.diffusivityInterpolation == diffusivity_interpolation
    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))


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


def test_ternary_continuous_surrogate_requires_regular_bulk_grid():
    with pytest.raises(ValueError, match="diffusivity_bulk_grids"):
        _build_surrogate(diffusivity_interpolation="continuous_grid")


def test_ternary_continuous_surrogate_rejects_grid_outside_simplex():
    with pytest.raises(ValueError, match="ternary composition bounds"):
        _build_surrogate(
            diffusivity_interpolation="continuous_grid",
            diffusivity_bulk_grids=(
                np.asarray([0.70, 0.80], dtype=np.float64),
                np.asarray([0.25, 0.30], dtype=np.float64),
            ),
        )


def test_ternary_simplex_linear_surrogate_accepts_grid_crossing_simplex_boundary():
    surrogate = _build_surrogate(
        diffusivity_interpolation="simplex_linear",
        diffusivity_bulk_grids=(
            np.asarray([0.18, 0.30, 0.42, 0.70], dtype=np.float64),
            np.asarray([0.06, 0.18, 0.30, 0.45], dtype=np.float64),
        ),
    )

    matrix = surrogate.getInterdiffusivity([0.36, 0.22], 1000.0, phase="ALPHA", query_context="general")
    nearby = surrogate.getInterdiffusivity([0.361, 0.221], 1000.0, phase="ALPHA", query_context="general")

    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))
    assert not np.array_equal(matrix, nearby)
    assert np.linalg.norm(matrix - nearby, ord=np.inf) < 1.0e-2


def test_ternary_continuous_surrogate_interface_uses_tieline_only():
    first = _direct_continuous_surrogate(bulk_scale=10.0)
    second = _direct_continuous_surrogate(bulk_scale=100.0)
    left = first.interface_compositions(0.5)[0]

    first_interface = first.getInterdiffusivity(left, 1000.0, phase="ALPHA", query_context="interface")
    second_interface = second.getInterdiffusivity(left, 1000.0, phase="ALPHA", query_context="interface")
    first_general = first.getInterdiffusivity(left, 1000.0, phase="ALPHA", query_context="general")
    expected_interface = _test_continuous_matrix(left, "ALPHA")

    assert np.allclose(first_interface, expected_interface)
    assert np.allclose(second_interface, expected_interface)
    assert not np.allclose(first_interface, first_general)


def test_ternary_continuous_surrogate_interface_varies_continuously_with_endpoint_composition():
    surrogate = _direct_continuous_surrogate()

    matrix = surrogate.getInterdiffusivity([0.2500, 0.10], 1000.0, phase="ALPHA", query_context="interface")
    nearby = surrogate.getInterdiffusivity([0.2505, 0.10], 1000.0, phase="ALPHA", query_context="interface")

    assert not np.array_equal(matrix, nearby)
    assert np.linalg.norm(matrix - nearby, ord=np.inf) < 2.0e-3


def test_ternary_continuous_surrogate_bulk_vectorized_matches_scalar_and_is_continuous():
    surrogate = _build_surrogate(
        diffusivity_interpolation="continuous_grid",
        diffusivity_bulk_grids=_continuous_bulk_grids(),
    )
    points = np.asarray([[0.255, 0.105], [0.256, 0.106]], dtype=np.float64)

    vectorized = surrogate.getInterdiffusivity(points, 1000.0, phase="ALPHA", query_context="general")
    scalar = surrogate.getInterdiffusivity(points[0], 1000.0, phase="ALPHA", query_context="general")

    assert np.allclose(vectorized[0], scalar)
    assert not np.array_equal(vectorized[0], vectorized[1])
    assert np.linalg.norm(vectorized[1] - vectorized[0], ord=np.inf) < 1.0e-2


def test_ternary_continuous_surrogate_rejects_invalid_dense_interpolation():
    with pytest.raises(ValueError, match="positive real eigenvalues"):
        _direct_continuous_surrogate(invalid_bulk=True)


def test_ternary_surrogate_dense_matrix_validation_reports_interface_and_bulk_counts():
    surrogate = _direct_continuous_surrogate()

    report = surrogate.validate_diffusivity_matrices(
        matrix_interface_eta_count=5,
        matrix_bulk_grid_counts=(3, 4),
        matrix_bulk_axes=(np.asarray([0.18, 0.30, 0.42]), np.asarray([0.06, 0.12, 0.18, 0.22])),
    )

    assert report["summary"]["ok"] is True
    assert report["interface"]["phases"]["ALPHA"]["matrices"].shape == (5, 2, 2)
    assert report["bulk"]["phases"]["ALPHA"]["matrices"].shape == (12, 2, 2)
    assert report["interface"]["phases"]["ALPHA"]["nearest_training_distance"].shape == (5,)
    assert report["bulk"]["phases"]["ALPHA"]["nearest_training_distance"].shape == (12,)


def test_ternary_surrogate_dense_matrix_validation_reports_invalid_without_first_failure():
    surrogate = _direct_nearest_surrogate(invalid=True)

    report = surrogate.validate_diffusivity_matrices(
        matrix_interface_eta_count=3,
        matrix_bulk_grid_counts=(2, 2),
        matrix_bulk_axes=(np.asarray([0.20, 0.30]), np.asarray([0.10, 0.15])),
    )

    assert report["summary"]["ok"] is False
    assert report["summary"]["invalid_count"] > 1
    assert np.any(~report["interface"]["phases"]["ALPHA"]["valid"])
    assert np.any(~report["bulk"]["phases"]["BETA"]["valid"])
    with pytest.raises(ValueError, match="validation failed"):
        surrogate.validate_diffusivity_matrices(
            matrix_interface_eta_count=3,
            matrix_bulk_grid_counts=(2, 2),
            matrix_bulk_axes=(np.asarray([0.20, 0.30]), np.asarray([0.10, 0.15])),
            raise_on_invalid=True,
        )


def test_ternary_surrogate_dense_validation_uses_interface_and_general_contexts(monkeypatch):
    surrogate = _direct_continuous_surrogate()
    calls = []
    original = surrogate.getInterdiffusivity

    def spy(composition, temperature=None, phase=None, query_context=None, **kwargs):
        calls.append((phase, query_context, np.asarray(composition, dtype=np.float64).shape))
        return original(composition, temperature, phase=phase, query_context=query_context, **kwargs)

    monkeypatch.setattr(surrogate, "getInterdiffusivity", spy)

    surrogate.validate_diffusivity_matrices(
        phases=("ALPHA",),
        matrix_interface_eta_count=4,
        matrix_bulk_grid_counts=(2, 2),
        matrix_bulk_axes=(np.asarray([0.18, 0.24]), np.asarray([0.06, 0.10])),
    )

    assert ("ALPHA", "interface", (4, 2)) in calls
    assert ("ALPHA", "general", (4, 2)) in calls


def test_ternary_surrogate_ground_truth_comparison_reports_zero_error_and_distances():
    surrogate = _direct_nearest_surrogate()

    report = surrogate.compare_diffusivity_to_ground_truth(
        thermodynamics=_ContinuousMatrixTruth(),
        error_interface_eta_count=3,
        error_bulk_grid_counts=(2, 1),
        error_bulk_axes=(np.asarray([0.20, 0.30]), np.asarray([0.10])),
    )

    assert report["summary"]["ok"] is True
    assert np.isclose(report["summary"]["max_relative_error"], 0.0)
    assert np.isclose(report["interface"]["phases"]["ALPHA"]["summary"]["max_relative_error"], 0.0)
    assert np.any(np.isclose(report["interface"]["phases"]["ALPHA"]["nearest_training_distance"], 0.0))
    assert np.any(np.isclose(report["bulk"]["phases"]["ALPHA"]["nearest_training_distance"], 0.0))


def test_ternary_surrogate_ground_truth_comparison_reports_nonzero_relative_error():
    surrogate = _direct_continuous_surrogate(bulk_scale=10.0)

    report = surrogate.compare_diffusivity_to_ground_truth(
        thermodynamics=_ContinuousMatrixTruth(scale=1.0),
        error_interface_eta_count=3,
        error_bulk_grid_counts=(2, 2),
        error_bulk_axes=(np.asarray([0.18, 0.42]), np.asarray([0.06, 0.22])),
    )

    assert report["summary"]["ok"] is True
    assert report["bulk"]["phases"]["ALPHA"]["summary"]["max_relative_error"] > 1.0
    assert report["bulk"]["phases"]["ALPHA"]["relative_error"].shape[-2:] == (2, 2)


def test_ternary_surrogate_ground_truth_comparison_uses_safe_relative_error_floor():
    surrogate = _direct_nearest_surrogate()

    report = surrogate.compare_diffusivity_to_ground_truth(
        thermodynamics=_ZeroMatrixTruth(),
        error_interface_eta_count=3,
        error_bulk_grid_counts=(1, 1),
        error_bulk_axes=(np.asarray([0.20]), np.asarray([0.10])),
        relative_error_floor=1.0e-6,
    )

    assert report["summary"]["ok"] is True
    assert np.all(np.isfinite(report["interface"]["phases"]["ALPHA"]["relative_error"]))
    assert report["interface"]["phases"]["ALPHA"]["summary"]["max_relative_error"] > 1.0e5


def test_ternary_surrogate_ground_truth_validation_requires_source():
    surrogate = _direct_nearest_surrogate()

    with pytest.raises(ValueError, match="Ground-truth diffusivity validation requires"):
        surrogate.compare_diffusivity_to_ground_truth(
            error_interface_eta_count=2,
            error_bulk_grid_counts=(1, 1),
            error_bulk_axes=(np.asarray([0.20]), np.asarray([0.10])),
        )


def test_ternary_surrogate_from_database_stores_explicit_validation_database_and_roundtrips():
    surrogate = _build_surrogate(
        validation_database="fake_validation_source.tdb",
        validation_thermodynamics_kwargs={"parameters": {"A": 1.0}},
    )
    path = Path.cwd() / "ternary_surrogate_validation_metadata_roundtrip.npz"

    try:
        surrogate.save(path)
        loaded = TernaryMovingBoundaryThermodynamicsSurrogate.load(path)
    finally:
        path.unlink(missing_ok=True)

    assert loaded.metadata["validation_database"] == "fake_validation_source.tdb"
    assert loaded.metadata["validation_thermodynamics_kwargs"] == {"parameters": {"A": 1.0}}


def test_ternary_surrogate_from_database_path_stores_validation_database(monkeypatch):
    class FakeMulticomponentThermodynamics(_TieLineSamplingThermodynamics):
        def __init__(self, database, elements, phases, **kwargs):
            super().__init__()
            self.database = database
            self.elements = elements
            self.phases = phases

    monkeypatch.setattr(surrogate_module, "MulticomponentThermodynamics", FakeMulticomponentThermodynamics)

    surrogate = TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
        database="stored_validation_source.tdb",
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        tieline_phases=("ALPHA", "BETA"),
        temperature=1000.0,
        probe_start=np.asarray([0.20, 0.10], dtype=np.float64),
        probe_end=np.asarray([0.40, 0.20], dtype=np.float64),
        eta_samples=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )

    assert surrogate.metadata["validation_database"] == "stored_validation_source.tdb"
    report = surrogate.compare_diffusivity_to_ground_truth(
        phases=("ALPHA",),
        error_interface_eta_count=3,
        error_bulk_grid_counts=(2, 1),
        error_bulk_axes=(np.asarray([0.20, 0.25]), np.asarray([0.10])),
    )
    assert report["summary"]["ok"] is True
    assert np.isclose(report["summary"]["max_relative_error"], 0.0)


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


def test_ternary_continuous_surrogate_save_load_preserves_predictions():
    surrogate = _build_surrogate(
        diffusivity_interpolation="continuous_grid",
        diffusivity_bulk_grids=_continuous_bulk_grids(),
    )
    path = Path.cwd() / "continuous_ternary_surrogate_test_roundtrip.npz"

    try:
        surrogate.save(path)
        loaded = TernaryMovingBoundaryThermodynamicsSurrogate.load(path)
    finally:
        path.unlink(missing_ok=True)

    assert loaded.diffusivityInterpolation == "continuous_grid"
    assert np.allclose(loaded.diffusivityBulkGridAxes[0], surrogate.diffusivityBulkGridAxes[0])
    assert np.allclose(loaded.diffusivityBulkGridAxes[1], surrogate.diffusivityBulkGridAxes[1])
    assert np.allclose(
        loaded.getInterdiffusivity([0.25, 0.10], 1000.0, phase="ALPHA", query_context="interface"),
        surrogate.getInterdiffusivity([0.25, 0.10], 1000.0, phase="ALPHA", query_context="interface"),
    )
    assert np.allclose(
        loaded.getInterdiffusivity([0.255, 0.105], 1000.0, phase="ALPHA", query_context="general"),
        surrogate.getInterdiffusivity([0.255, 0.105], 1000.0, phase="ALPHA", query_context="general"),
    )


@pytest.mark.parametrize("bulk_diffusivity_mode", ["composition_dependent_lagged", "composition_dependent_implicit"])
def test_ternary_illingworth_variable_modes_query_surrogate_general_bulk_context(bulk_diffusivity_mode):
    surrogate = _build_surrogate(
        diffusivity_bulk_points=np.asarray(
            [
                [0.20, 0.10],
                [0.25, 0.10],
                [0.30, 0.15],
                [0.35, 0.15],
            ],
            dtype=np.float64,
        )
    )
    thermodynamics = _RecordingSurrogateDiffusivity(surrogate)
    left, right = surrogate.interface_compositions(0.0)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=thermodynamics,
        temperature=1000.0,
        interfacePosition=0.5,
        interface_equilibrium=surrogate,
        initial_eta_bracket=(0.0, 1.0),
        bulk_diffusivity_mode=bulk_diffusivity_mode,
        bulk_picard_max_iterations=10,
        time_step=1.0,
        record=True,
    )
    model.setup()
    setup_contexts = [(call["phase"], call["query_context"]) for call in thermodynamics.calls]
    thermodynamics.calls.clear()

    model.getdXdt(model.currentTime, model.getCurrentX())

    solve_contexts = [(call["phase"], call["query_context"]) for call in thermodynamics.calls]
    assert ("ALPHA", "general") in setup_contexts
    assert ("BETA", "general") in setup_contexts
    assert ("ALPHA", "general") in solve_contexts
    assert ("BETA", "general") in solve_contexts
    assert any(call["composition_shape"] == (len(model._u_grid) - 1, 2) for call in thermodynamics.calls)
    assert any(call["composition_shape"] == (len(model._v_grid) - 1, 2) for call in thermodynamics.calls)


def test_ternary_illingworth_composition_dependent_implicit_converges_with_continuous_surrogate():
    surrogate = _build_surrogate(
        diffusivity_interpolation="continuous_grid",
        diffusivity_bulk_grids=_continuous_bulk_grids(),
    )
    thermodynamics = _RecordingSurrogateDiffusivity(surrogate)
    left, right = surrogate.interface_compositions(0.0)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=thermodynamics,
        temperature=1000.0,
        interfacePosition=0.5,
        interface_equilibrium=surrogate,
        initial_eta_bracket=(0.0, 1.0),
        bulk_diffusivity_mode="composition_dependent_implicit",
        bulk_picard_max_iterations=20,
        time_step=1.0,
        record=True,
    )

    model.solve(1.0)

    assert model._lastBulkConverged is True
    assert model._lastBulkDiffusivityProviderCalls > 0
    assert model._lastBulkFaceMatricesEvaluated > 0
    assert ("ALPHA", "general") in [(call["phase"], call["query_context"]) for call in thermodynamics.calls]
    assert ("BETA", "general") in [(call["phase"], call["query_context"]) for call in thermodynamics.calls]
    assert np.allclose(model.checkConservation(1.0e-12), np.zeros(2))


def test_ternary_illingworth_phase_uniform_surrogate_uses_interface_diffusivity_context():
    surrogate = _build_surrogate()
    thermodynamics = _RecordingSurrogateDiffusivity(surrogate)
    left, right = surrogate.interface_compositions(0.0)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=thermodynamics,
        temperature=1000.0,
        interfacePosition=0.5,
        interface_equilibrium=surrogate,
        initial_eta_bracket=(0.0, 1.0),
        time_step=1.0,
        record=True,
    )
    model.setup()
    thermodynamics.calls.clear()

    model.getdXdt(model.currentTime, model.getCurrentX())

    assert ("ALPHA", "interface") in [(call["phase"], call["query_context"]) for call in thermodynamics.calls]
    assert ("BETA", "interface") in [(call["phase"], call["query_context"]) for call in thermodynamics.calls]
    assert not any(call["query_context"] == "general" for call in thermodynamics.calls)


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
