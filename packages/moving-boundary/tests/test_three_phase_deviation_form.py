"""Deviation-form (ledger R06) tests for the three-phase ternary Illingworth model.

Each interval is solved for ``w = C - r`` with a reference ``r`` (the interface
composition for the outer phases, linear in ``xi`` between the two interface
compositions for the middle phase), and the interface-adjacent transfers are
formed without differences of absolute compositions. These tests check the
row-sum identity the shift relies on, preservation of reference states, single
steps against 50-digit arithmetic, the round-off floor projection, and a fine
interface grid on which the absolute form could not converge.

Ported from samuelgprice/kawin fae6149 (merged to research main as 7af24f4;
kawin/tests/test_illingworth_ternary_three_phase_deviation_form.py). Test bodies are
unchanged except for the package imports and the interstitial builder, which is
called directly instead of imported from a research test module. The helpers the
research file took from other research test modules (model builders, closures,
the similarity case and the interstitial three-phase model) are reproduced verbatim
below and exposed as ``tpt`` and ``sim``. To avoid name clashes in one module, the
interstitial ones carry an ``_Interstitial`` / ``_interstitial`` prefix and the
similarity cases dictionary is ``_SIMILARITY_CASES`` (research ``CASES``).
"""

from decimal import Decimal, getcontext
from types import SimpleNamespace

import numpy as np
import pytest

from moving_boundary.analytic import FixedMatrixTernaryDiffusivity, SimilarityInitialProfile, solve_three_phase_similarity_roots
from moving_boundary.equilibrium import CallableTernaryInterfaceEquilibrium
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder
from moving_boundary.mesh.TransformedGrids import landau_grid
from moving_boundary.solvers import MovingBoundaryIllingworthTernaryThreePhaseFD1DModel
from moving_boundary.solvers.illingworth.numerics.ternary import _fill_three_phase_interval_system
from moving_boundary.solvers.illingworth.shared import (
    _FLOOR_PROJECTION_ULPS,
    _project_shift_roundoff_to_floor,
)
from moving_boundary.solvers.illingworth.three_phase import (
    _BULK_DIFFUSIVITY_IMPLICIT,
    _BULK_DIFFUSIVITY_LAGGED,
    _BULK_DIFFUSIVITY_PHASE_UNIFORM,
    _ThreePhaseBulkResult,
)


# --------------------------------------------------------------------------
# Reproduced research helpers (see module docstring)
# --------------------------------------------------------------------------
# --- from kawin/tests/test_illingworth_ternary_three_phase.py ---


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


# --- from kawin/tests/test_illingworth_ternary_three_phase_similarity.py (CASES renamed _SIMILARITY_CASES) ---


D = {
    "A": np.array([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]]),
    "B": np.array([[3.0e-4, -0.5e-4], [1.0e-4, 5.0e-4]]),
    "C": np.array([[6.0e-4, 1.0e-4], [-0.5e-4, 2.0e-3]]),
}


_SIMILARITY_CASES = {
    "opposite": (np.array([0.28, 0.10]), np.array([0.26, 0.12])),
    "drift": (np.array([0.38, 0.03]), np.array([0.26, 0.12])),
}


S0 = 0.3 + 1.0e-12  # off the physical mesh nodes


DOMAIN_LENGTH = 0.6


T0 = 0.01


T_END = 1.0


def _linear_closure(left_0, left_1, right_0, right_1):
    left_0, left_1, right_0, right_1 = (np.asarray(v, dtype=np.float64) for v in (left_0, left_1, right_0, right_1))
    return CallableTernaryInterfaceEquilibrium(
        lambda eta: (left_0 + eta * (left_1 - left_0), right_0 + eta * (right_1 - right_0)),
        eta_bounds=(0.0, 1.0),
    )


CLOSURE_AB = _linear_closure([0.30, 0.03], [0.34, 0.07], [0.18, 0.10], [0.26, 0.16])


CLOSURE_BC = _linear_closure([0.20, 0.12], [0.24, 0.18], [0.36, 0.04], [0.39, 0.08])


def _solution(case):
    c_a_far, c_c_far = _SIMILARITY_CASES[case]
    roots = solve_three_phase_similarity_roots(D["A"], D["B"], D["C"], c_a_far, c_c_far, CLOSURE_AB, CLOSURE_BC, s0=S0)
    # 'opposite' has one root; 'drift' also has a thin layer drifting left. Use the
    # root with the largest k_ab, the right-moving one.
    return roots[-1]


def _build_model(solution, nodes, semi_log_dt=0.025, diffusivities=D, initial_eta_guess=True, **model_kwargs):
    mesh = CartesianFD1D(["X", "Y"], [0.0, DOMAIN_LENGTH], 6001)
    mesh.setResponseProfile(ProfileBuilder([(SimilarityInitialProfile(solution, T0), ["X", "Y"])]))
    grids = (
        landau_grid(nodes, "linear", phase="left", spacing_ratio=0.1),
        landau_grid(nodes, "linear", phase="middle", spacing_ratio=0.2),
        landau_grid(nodes, "linear", phase="right", spacing_ratio=0.1),
    )
    options = dict(
        tolerance=1.0e-11,
        max_iterations=50,
        terminal_thin_phase_policy="continue",
        record=True,
        record_pq_data=True,
    )
    options.update(model_kwargs)
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["A", "B", "C"],
        thermodynamics=FixedMatrixTernaryDiffusivity(diffusivities, ["A", "B", "C"]),
        temperature=1000.0,
        interfacePositions=tuple(solution.interface_positions(T0)),
        interface_equilibria=(CLOSURE_AB, CLOSURE_BC),
        initial_eta_guess=tuple(solution.etas) if initial_eta_guess else None,
        time_step=T_END - T0,
        dt_mode="semi_log",
        semiLog_dt=semi_log_dt,
        semiLogT0=1.0e-3 * T0,
        transformed_grids=grids,
        **options,
    )


# --- from kawin/tests/test_illingworth_interstitials_three_phase.py (names prefixed) ---


_INTERSTITIAL_INTERFACES = (0.35, 0.7)


_INTERSTITIAL_PHASE_U = (
    np.asarray([0.21, 0.09], dtype=np.float64),
    np.asarray([0.31, 0.16], dtype=np.float64),
    np.asarray([0.20, 0.21], dtype=np.float64),
)


class _InterstitialLinearPairEquilibrium:
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


class _InterstitialThreePhaseStepProfile:
    """Piecewise-constant mole-fraction start built from u-fraction plateaus (first component is C)."""

    def __call__(self, z):
        x = np.asarray(z, dtype=np.float64).reshape((-1, 1))[:, 0]
        out = np.empty((len(x), 2), dtype=np.float64)
        for mask, u in (
            (x < _INTERSTITIAL_INTERFACES[0], _INTERSTITIAL_PHASE_U[0]),
            ((x >= _INTERSTITIAL_INTERFACES[0]) & (x < _INTERSTITIAL_INTERFACES[1]), _INTERSTITIAL_PHASE_U[1]),
            (x >= _INTERSTITIAL_INTERFACES[1], _INTERSTITIAL_PHASE_U[2]),
        ):
            out[mask] = u / (1.0 + u[0])
        return out


class _InterstitialSpyThreePhaseThermodynamics:
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


def _interstitial_model(mode="phase_uniform"):
    eq_ab = _InterstitialLinearPairEquilibrium([0.20, 0.10], [0.10, 0.0], [0.30, 0.15], [0.10, 0.0])
    eq_bc = _InterstitialLinearPairEquilibrium([0.24, 0.15], [0.10, 0.0], [0.18, 0.20], [0.10, 0.0])
    mesh = CartesianFD1D(["C", "MN"], [0.0, 1.0], 31)
    mesh.setResponseProfile(ProfileBuilder([(_InterstitialThreePhaseStepProfile(), ["C", "MN"])]))
    therm = _InterstitialSpyThreePhaseThermodynamics()
    model = MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh,
        elements=["FE", "C", "MN"],
        phases=["A", "B", "C"],
        thermodynamics=therm,
        temperature=1000.0,
        interfacePositions=_INTERSTITIAL_INTERFACES,
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


tpt = SimpleNamespace(**{name: globals()[name] for name in (
    "_ConstantPairEquilibrium",
    "_LinearPairEquilibrium",
    "_ThreePhaseStepProfile",
    "_ThreePhaseLinearProfile",
    "_RecordingThreePhaseThermodynamics",
    "_make_stationary_model",
    "_make_eta_varying_three_phase_model",
    "_make_nonstationary_unequal_volume_model",

)})
sim = SimpleNamespace(**{name: globals()[name] for name in (
    "D",
    "S0",
    "DOMAIN_LENGTH",
    "T0",
    "T_END",
    "_linear_closure",
    "CLOSURE_AB",
    "CLOSURE_BC",
    "_solution",
    "_build_model",
    "SimilarityInitialProfile",
    "FixedMatrixTernaryDiffusivity",
)})


EPS = np.finfo(np.float64).eps
COMPILED_MODES = (True, False) if _fill_three_phase_interval_system is not None else (False,)
# (left_value, right_value) for the three phase kinds.
END_VALUES = {
    0: (None, np.array([0.21, 0.11])),
    1: (np.array([0.29, 0.07]), np.array([0.18, 0.26])),
    2: (np.array([0.33, 0.04]), None),
}
BOUNDS = {0: (0.0, 0.31), 1: (0.31, 0.58), 2: (0.58, 1.0)}


# --------------------------------------------------------------------------
# 50-digit reference of one interval solve (mirrors the NumPy assembly in
# ``_solve_interval_planar``: float inputs converted exactly, everything else
# evaluated in Decimal).
# --------------------------------------------------------------------------
getcontext().prec = 50
_D0, _D1, _DH = Decimal(0), Decimal(1), Decimal("0.5")


def _d(x):
    return Decimal(float(x))


def _mat(m):
    return [[_d(m[0, 0]), _d(m[0, 1])], [_d(m[1, 0]), _d(m[1, 1])]]


def _mm(a, b):
    return [[a[i][0] * b[0][j] + a[i][1] * b[1][j] for j in range(2)] for i in range(2)]


def _mv(a, v):
    return [a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1]]


def _madd(a, b, s=1):
    return [[a[i][j] + s * b[i][j] for j in range(2)] for i in range(2)]


def _inv(a):
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return [[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]]


def _eye(k):
    return [[k, _D0], [_D0, k]]


def _exact_interval(profile, grid, old_bounds, new_bounds, frame, dt, D_faces, left_value, right_value):
    """Returns (solution, left_transfer, right_transfer, old_widths, new_widths) in Decimal."""
    n = len(profile)
    g = [_d(x) for x in grid]
    old_len = _d(old_bounds[1]) - _d(old_bounds[0])
    new_len = _d(new_bounds[1]) - _d(new_bounds[0])
    face_xi = [_D0] + [(g[i] + g[i + 1]) * _DH for i in range(n - 1)] + [_D1]
    old_w = [old_len * (face_xi[i + 1] - face_xi[i]) for i in range(n)]
    new_w = [new_len * (face_xi[i + 1] - face_xi[i]) for i in range(n)]
    D_faces = np.broadcast_to(np.asarray(D_faces, dtype=np.float64), (n - 1, 2, 2))
    d_left = _d(new_bounds[0]) - _d(old_bounds[0])
    d_right = _d(new_bounds[1]) - _d(old_bounds[1])
    L, R = [], []
    for f in range(n - 1):
        xi = face_xi[f + 1]
        disp = (_D1 - xi) * d_left + xi * d_right - _d(frame)
        k = _d(dt) / (new_len * (g[f + 1] - g[f]))
        A = [[_d(D_faces[f][i, j]) * k for j in range(2)] for i in range(2)]
        lc, rc = [[-A[i][j] for j in range(2)] for i in range(2)], A
        if disp > 0:
            rc = _madd(rc, _eye(disp))
        elif disp < 0:
            lc = _madd(lc, _eye(disp))
        L.append(lc)
        R.append(rc)
    zero2 = [[_D0, _D0], [_D0, _D0]]
    lower, diag, upper, rhs = [zero2] * n, [None] * n, [zero2] * n, [None] * n
    for i in range(n):
        if (i == 0 and left_value is not None) or (i == n - 1 and right_value is not None):
            diag[i] = _eye(_D1)
            rhs[i] = [_d(v) for v in (left_value if i == 0 else right_value)]
            continue
        diag[i] = _eye(new_w[i])
        rhs[i] = [_d(profile[i][0]) * old_w[i], _d(profile[i][1]) * old_w[i]]
        if i > 0:
            lower[i] = L[i - 1]
            diag[i] = _madd(diag[i], R[i - 1])
        if i < n - 1:
            diag[i] = _madd(diag[i], L[i], -1)
            upper[i] = [[-R[i][a][b] for b in range(2)] for a in range(2)]
    cu, dr = [None] * n, [None] * n
    inv = _inv(diag[0])
    cu[0], dr[0] = _mm(inv, upper[0]), _mv(inv, rhs[0])
    for i in range(1, n):
        inv = _inv(_madd(diag[i], _mm(lower[i], cu[i - 1]), -1))
        cu[i] = _mm(inv, upper[i])
        t = _mv(lower[i], dr[i - 1])
        dr[i] = _mv(inv, [rhs[i][0] - t[0], rhs[i][1] - t[1]])
    x = [None] * n
    x[-1] = dr[-1]
    for i in range(n - 2, -1, -1):
        t = _mv(cu[i], x[i + 1])
        x[i] = [dr[i][0] - t[0], dr[i][1] - t[1]]

    def transfer(f, a, b):
        u, v = _mv(L[f], x[a]), _mv(R[f], x[b])
        return [u[0] + v[0], u[1] + v[1]]

    lt = [_D0, _D0] if left_value is None else transfer(0, 0, 1)
    rt = [_D0, _D0] if right_value is None else transfer(n - 2, n - 2, n - 1)
    return x, lt, rt, old_w, new_w


def _to_float(values):
    return np.array([[float(v) for v in row] for row in values]) if isinstance(values[0], list) else np.array([float(v) for v in values])


def _stationary_model():
    model, _ = tpt._make_stationary_model(record=False)
    model.setup()
    return model


def _random_case(phase, n, rng, *, near_reference, jump, stiffness, moving, coupled=True):
    """Builds one random interval problem for ``phase`` (0 = A, 1 = B, 2 = C)."""
    left_value, right_value = END_VALUES[phase]
    if phase == 1 and jump == "large":
        left_value, right_value = np.array([0.45, 0.02]), np.array([0.03, 0.47])
    grid = np.concatenate(([0.0], np.sort(rng.uniform(0.0, 1.0, n - 2)), [1.0]))
    # Fine cells at the interface ends (geometric grading).
    grid = 0.5 - 0.5 * np.cos(np.pi * grid)
    if phase == 1:
        reference = left_value + grid[:, None] * (right_value - left_value)
    else:
        reference = np.broadcast_to(left_value if right_value is None else right_value, (n, 2))
    if near_reference:
        profile = reference + rng.uniform(-1e-6, 1e-6, (n, 2))
    else:
        profile = np.column_stack((rng.uniform(0.05, 0.4, n), rng.uniform(0.05, 0.4, n)))
    if left_value is not None:
        profile[0] = left_value
    if right_value is not None:
        profile[-1] = right_value
    base = np.array([[1.0, 0.3], [0.1, 0.6]]) if coupled else np.array([[1.0, 0.0], [0.0, 0.6]])
    D = base * rng.uniform(0.5, 1.5, (n - 1, 1, 1))
    old_bounds = BOUNDS[phase]
    if moving:
        # The external boundary of an outer phase moves with its frame.
        new_bounds = (old_bounds[0] + (3e-3 if phase > 0 else 0.0), old_bounds[1] + (-2e-3 if phase < 2 else 1e-3))
        frame = 1e-3 if phase == 2 else 0.0
    else:
        new_bounds, frame = old_bounds, 0.0
    return profile, grid, old_bounds, new_bounds, frame, float(stiffness), D, left_value, right_value


# --------------------------------------------------------------------------
# Row-sum identity
# --------------------------------------------------------------------------
def _capture_systems(model):
    captured = []
    real = model._finish_interval_solve

    def finish(lower, diagonal, upper, rhs, face_coefficients, D_faces, left_value, right_value, reference_info):
        captured.append((lower.copy(), diagonal.copy(), upper.copy(), left_value, right_value, reference_info))
        return real(lower, diagonal, upper, rhs, face_coefficients, D_faces, left_value, right_value, reference_info)

    model._finish_interval_solve = finish
    return captured


def _assert_row_sums(captured):
    checked = 0
    for lower, diagonal, upper, left_value, right_value, info in captured:
        grid, old_bounds = info[2], info[3]
        n = len(grid)
        face_xi = np.concatenate(([0.0], 0.5 * (grid[:-1] + grid[1:]), [1.0]))
        old_w = (float(old_bounds[1]) - float(old_bounds[0])) * np.diff(face_xi)
        for i in range(n):
            if (i == 0 and left_value is not None) or (i == n - 1 and right_value is not None):
                continue
            total = lower[i] + diagonal[i] + upper[i]
            magnitude = np.max(np.abs(lower[i]) + np.abs(diagonal[i]) + np.abs(upper[i]))
            assert np.max(np.abs(total - old_w[i] * np.eye(2))) <= 16.0 * EPS * magnitude
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("compiled", COMPILED_MODES)
@pytest.mark.parametrize("phase", [0, 1, 2])
@pytest.mark.parametrize("motion", [-1.0, 1.0])
@pytest.mark.parametrize("face_d", [False, True])
def test_interval_free_rows_sum_to_old_width(compiled, phase, motion, face_d):
    """Every free row satisfies lower + diagonal + upper = old_width * I (both displacement signs)."""
    model = _stationary_model()
    model._useCompiledIntervalAssembly = compiled
    rng = np.random.default_rng(11 + phase)
    profile, grid, old_bounds, _, _, _, D, left_value, right_value = _random_case(
        phase, 23, rng, near_reference=False, jump="small", stiffness=1.0, moving=False
    )
    # Interfaces move by 2e-3 / 7e-3 (times ``motion``); the external boundary
    # of phase C moves with that phase's frame (material boundary), phase A's
    # left boundary is fixed with a fixed frame.
    new_bounds = (old_bounds[0] + motion * (2e-3 if phase > 0 else 0.0), old_bounds[1] + motion * (7e-3 if phase < 2 else 1e-3))
    frame = {0: 0.0, 1: motion * 1e-3, 2: motion * 1e-3}[phase]
    model._currdt = 0.37
    captured = _capture_systems(model)
    model._solve_interval_planar(
        profile, grid, old_bounds, new_bounds, phase, left_value, right_value, D if face_d else D[0],
        validate_diffusivity=False, phase_frame_displacement=frame,
    )
    _assert_row_sums(captured)


@pytest.mark.parametrize("compiled", COMPILED_MODES)
@pytest.mark.parametrize("builder", ["similarity", "unequal_volumes"])
@pytest.mark.parametrize("variable", [0, 1])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_candidate_free_rows_sum_to_old_width(compiled, builder, variable, sign):
    """Row-sum identity on full candidates, including unequal molar volumes with a moving right boundary."""
    if builder == "similarity":
        model = sim._build_model(sim._solution("opposite"), 21, semi_log_dt=0.05)
    else:
        model = tpt._make_nonstationary_unequal_volume_model(record=False)
    model.setup()
    model._useCompiledIntervalAssembly = compiled
    profiles = tuple(p.copy() for p in model.getTransformedState())
    interfaces = model.getInterfacePositions().copy()
    etas = model.getInterfaceEtas().copy()
    model._currdt = 1e-3 * float(interfaces[0])
    x_hat = model._physical_to_scaled(interfaces, etas)
    x_hat[variable] += sign * 1e-3 * x_hat[variable]
    captured = _capture_systems(model)
    model._evaluate_interface_candidate(profiles, interfaces, x_hat, model._residual_scale(profiles, interfaces), model._currdt)
    assert len(captured) == 3
    if builder == "unequal_volumes":
        info = captured[2][5]
        assert info[4][1] != info[3][1]  # right boundary moved
    _assert_row_sums(captured)


@pytest.mark.parametrize("phase", [0, 1, 2])
@pytest.mark.parametrize("moving", [False, True])
def test_compiled_and_numpy_deviation_paths_are_bitwise_equal(phase, moving):
    if _fill_three_phase_interval_system is None:
        pytest.skip("Numba is not available")
    model = _stationary_model()
    rng = np.random.default_rng(3 + phase)
    profile, grid, ob, nb, frame, dt, D, lv, rv = _random_case(phase, 19, rng, near_reference=False, jump="large", stiffness=50.0, moving=moving)
    model._currdt = dt
    results = []
    for compiled in (True, False):
        model._useCompiledIntervalAssembly = compiled
        results.append(model._solve_interval_planar(profile, grid, ob, nb, phase, lv, rv, D, validate_diffusivity=False, phase_frame_displacement=frame))
    for field in ("profile", "left_transfer", "right_transfer", "projection_split"):
        np.testing.assert_array_equal(getattr(results[0], field), getattr(results[1], field))


# --------------------------------------------------------------------------
# Preservation of reference states
# --------------------------------------------------------------------------
@pytest.mark.parametrize("compiled", COMPILED_MODES)
@pytest.mark.parametrize("phase", [0, 2])
@pytest.mark.parametrize("moving", [False, True])
def test_outer_phase_at_interface_composition_is_preserved_exactly(compiled, phase, moving):
    """A uniform outer phase at its interface composition stays bitwise uniform; transfers are d*c."""
    model = _stationary_model()
    model._useCompiledIntervalAssembly = compiled
    rng = np.random.default_rng(5)
    _, grid, ob, nb, frame, _, D, lv, rv = _random_case(phase, 31, rng, near_reference=True, jump="small", stiffness=1e6, moving=moving)
    c = rv if phase == 0 else lv
    profile = np.broadcast_to(c, (31, 2)).copy()
    model._currdt = 1e6
    result = model._solve_interval_planar(profile, grid, ob, nb, phase, lv, rv, D, validate_diffusivity=False, phase_frame_displacement=frame)
    np.testing.assert_array_equal(result.profile, profile)
    d = model._internal_face_displacements(grid, ob, nb, frame)
    if phase == 0:
        np.testing.assert_array_equal(result.right_transfer, d[-1] * c)
    else:
        np.testing.assert_array_equal(result.left_transfer, d[0] * c)


@pytest.mark.parametrize("compiled", COMPILED_MODES)
@pytest.mark.parametrize("dt", [1e-3, 1.0, 1e5])
def test_linear_middle_phase_is_preserved_to_roundoff(compiled, dt):
    """A linear middle phase (static geometry, uniform D) keeps w at round-off; transfers equal dt D dc / L."""
    model = _stationary_model()
    model._useCompiledIntervalAssembly = compiled
    n = 41
    grid = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n))
    lv, rv = END_VALUES[1]
    delta = rv - lv
    profile = lv + grid[:, None] * delta
    profile[-1] = rv
    D = np.array([[1.0, 0.3], [0.1, 0.6]])
    bounds = BOUNDS[1]
    model._currdt = dt
    result = model._solve_interval_planar(profile, grid, bounds, bounds, 1, lv, rv, D, validate_diffusivity=False)
    deviation = result.profile - profile
    assert np.max(np.abs(deviation)) <= 64.0 * EPS * np.max(np.abs(profile))
    flux = dt * D @ delta / (bounds[1] - bounds[0])
    scale = np.max(np.abs(flux))
    assert np.max(np.abs(result.left_transfer - flux)) <= 1e-12 * scale
    assert np.max(np.abs(result.right_transfer - flux)) <= 1e-12 * scale


# --------------------------------------------------------------------------
# One step against 50-digit arithmetic
# --------------------------------------------------------------------------
CASES = [
    dict(near_reference=True, jump="small", stiffness=1e-3, moving=False),
    dict(near_reference=True, jump="small", stiffness=1e5, moving=False),
    dict(near_reference=False, jump="small", stiffness=10.0, moving=True),
    dict(near_reference=False, jump="large", stiffness=1e5, moving=True),
    dict(near_reference=True, jump="large", stiffness=1e7, moving=True),
]


@pytest.mark.parametrize("phase", [0, 1, 2])
@pytest.mark.parametrize("case", range(len(CASES)))
def test_single_interval_step_matches_50_digit_arithmetic(phase, case):
    """Transfers, profile and the retained profile's free-row inventory defect are at round-off level."""
    model = _stationary_model()
    rng = np.random.default_rng(100 * phase + case)
    n = 81
    profile, grid, ob, nb, frame, dt, D, lv, rv = _random_case(phase, n, rng, **CASES[case])
    model._currdt = dt
    result = model._solve_interval_planar(profile, grid, ob, nb, phase, lv, rv, D, validate_diffusivity=False, phase_frame_displacement=frame)
    x, lt, rt, old_w, new_w = _exact_interval(profile, grid, ob, nb, frame, dt, D, lv, rv)
    exact_profile = _to_float(x)
    composition_scale = max(np.max(np.abs(profile)), np.max(np.abs(exact_profile)))
    # The profile error is set by the solve's conditioning (up to 9e-15 in the
    # stiff large-jump middle case, A ~ 1e13), not by the shift; the absolute
    # form gave 1e-14 to 4e-12 on these cases.
    assert np.max(np.abs(result.profile - exact_profile)) <= 128.0 * EPS * composition_scale

    # Error scale of the deviation form: eps * (|H| + (|d| + |bounds|) |c| + |A| (|dr| + max|w|)),
    # with A = dt |D| / (L dxi) on the end face. The absolute form's error was
    # eps * |A| * |c| instead, with |c| >> |w| in the near-reference cases.
    d = model._internal_face_displacements(grid, ob, nb, frame)
    new_length = nb[1] - nb[0]
    D_faces = np.broadcast_to(D, (n - 1, 2, 2))
    reference = profile.copy()
    if phase == 1:
        reference = lv + grid[:, None] * (rv - lv)
    else:
        reference[:] = lv if rv is None else rv
    w_max = np.max(np.abs(exact_profile - reference))
    defect_slack = 0.0
    for float_value, exact_value, face, c in ((result.left_transfer, lt, 0, lv), (result.right_transfer, rt, -1, rv)):
        if c is None:
            continue
        exact = _to_float(exact_value)
        dxi = grid[1] - grid[0] if face == 0 else grid[-1] - grid[-2]
        A = dt * np.max(np.abs(D_faces[face]).sum(axis=1)) / (new_length * dxi) + abs(d[face])
        dr = 0.0 if phase != 1 else dxi * np.max(np.abs(rv - lv))
        # The float face displacement itself rounds the bound differences
        # (about eps * |bounds|), which enters as eps * |bounds| * |c|.
        bounds_scale = max(abs(ob[0]), abs(ob[1]), abs(nb[0]), abs(nb[1]))
        scale = np.max(np.abs(exact)) + (abs(d[face]) + bounds_scale) * np.max(np.abs(c)) + A * (dr + w_max)
        defect_slack += A * (dr + w_max)
        assert np.max(np.abs(float_value - exact)) <= 64.0 * EPS * scale

    # Free-row inventory defect of the stored profile, evaluated exactly:
    # sum new_w p - sum old_w p_old - (H_right - H_left).
    free = [i for i in range(n) if not ((i == 0 and lv is not None) or (i == n - 1 and rv is not None))]
    for k in range(2):
        H_left = _d(result.left_transfer[k]) if lv is not None else _D0
        H_right = _d(result.right_transfer[k]) if rv is not None else _D0
        defect = sum(new_w[i] * _d(result.profile[i, k]) - old_w[i] * _d(profile[i, k]) for i in free) - (H_right - H_left)
        magnitude = sum(abs(new_w[i] * _d(result.profile[i, k])) for i in free) + abs(H_left) + abs(H_right)
        assert abs(float(defect)) <= 64.0 * EPS * (float(magnitude) + defect_slack)


def _capture_final_D(model):
    """Wraps ``_solve_interval_planar`` to record the validated D of each phase's last solve."""
    captured = {}
    real = model._solve_interval_planar

    def interval(profile, grid, old_bounds, new_bounds, phase_index, left_value, right_value, D_faces, validate_diffusivity=True, phase_frame_displacement=0.0):
        D = model._phase_face_diffusivity_matrices(np.asarray(D_faces), len(profile) - 1, model.phases[phase_index]) if validate_diffusivity else D_faces
        captured[phase_index] = np.asarray(D, dtype=np.float64).copy()
        return real(profile, grid, old_bounds, new_bounds, phase_index, left_value, right_value, D_faces,
                    validate_diffusivity=validate_diffusivity, phase_frame_displacement=phase_frame_displacement)

    model._solve_interval_planar = interval
    return captured


def _exact_residual(model, profiles, interfaces, candidate, D):
    kin = candidate.kinematics
    new = candidate.interfaces
    ob = [(0.0, interfaces[0]), (interfaces[0], interfaces[1]), (interfaces[1], kin.old_right_boundary)]
    nb = [(0.0, new[0]), (new[0], new[1]), (new[1], kin.new_right_boundary)]
    ic = candidate.interface_compositions
    values = ((None, ic[0][0]), (ic[0][1], ic[1][0]), (ic[1][1], None))
    transfers = [_exact_interval(profiles[k], model._grids[k], ob[k], nb[k], kin.phase_displacements[k], model._currdt, D[k], *values[k])[1:3] for k in range(3)]

    def endpoint(k, side, value):
        g = model._grids[k]
        old_len, new_len = _d(ob[k][1]) - _d(ob[k][0]), _d(nb[k][1]) - _d(nb[k][0])
        w = _DH * _d(g[1]) if side == "left" else _DH * (_D1 - _d(g[-2]))
        old = profiles[k][0] if side == "left" else profiles[k][-1]
        return [w * (_d(value[j]) * new_len - _d(old[j]) * old_len) for j in range(2)]

    rho = [_d(v) for v in model._normalizedPhaseMolarDensities]
    ea, ebl, ebr, ec = endpoint(0, "right", ic[0][0]), endpoint(1, "left", ic[0][1]), endpoint(1, "right", ic[1][0]), endpoint(2, "left", ic[1][1])
    r_ab = [rho[0] * (ea[j] + transfers[0][1][j]) + rho[1] * (ebl[j] - transfers[1][0][j]) for j in range(2)]
    r_bc = [rho[1] * (ebr[j] + transfers[1][1][j]) + rho[2] * (ec[j] - transfers[2][0][j]) for j in range(2)]
    return np.array([float(v) for v in r_ab + r_bc])


@pytest.mark.parametrize("mode", [_BULK_DIFFUSIVITY_PHASE_UNIFORM, _BULK_DIFFUSIVITY_LAGGED, _BULK_DIFFUSIVITY_IMPLICIT])
@pytest.mark.parametrize("builder", ["eta_varying", "unequal_volumes", "interstitial"])
def test_candidate_residual_matches_50_digit_arithmetic(mode, builder):
    """The coupled four-component residual of a moving candidate is at round-off of the exact discrete one."""
    if builder == "eta_varying":
        model = tpt._make_eta_varying_three_phase_model(mode=mode, record=False)
    elif builder == "unequal_volumes":
        model = tpt._make_nonstationary_unequal_volume_model(record=False)
        model.bulkDiffusivityMode = mode
    else:
        model, _ = _interstitial_model(mode)
    model.setup()
    profiles = tuple(p.copy() for p in model.getTransformedState())
    interfaces = model.getInterfacePositions().copy()
    etas = model.getInterfaceEtas().copy()
    model._currdt = 5e-3
    model._laggedFaceDiffusivityCache = {}
    model._interfaceCompositionCache = {}
    x_hat = model._physical_to_scaled(interfaces, etas)
    x_hat[0] *= 1.0 + 2e-3
    x_hat[1] *= 1.0 - 1e-3
    captured = _capture_final_D(model)
    scale = model._residual_scale(profiles, interfaces)
    candidate = model._evaluate_interface_candidate(profiles, interfaces, x_hat, scale, model._currdt)
    exact = _exact_residual(model, profiles, interfaces, candidate, captured)
    assert np.max(np.abs(candidate.residual - exact) / scale) <= 1e-14


# --------------------------------------------------------------------------
# Floor projection
# --------------------------------------------------------------------------
def test_floor_projection_accounting_uses_cell_widths_and_splits_middle_phase():
    model = _stationary_model()
    floor = float(model.constraints.minComposition)
    n = 9
    grid = np.linspace(0.0, 1.0, n)
    reference = np.column_stack((0.4 - 0.2 * grid, 0.3 + 0.1 * grid))
    deviation = np.zeros((n, 2))
    deviation[3, 1] = floor - reference[3, 1]
    deviation[6, 1] = floor - reference[6, 1]
    profile = reference + deviation
    profile[3, 1] = np.nextafter(floor, -np.inf)  # a few ulps below the floor
    profile[6, 1] = floor - 3.0 * np.spacing(floor)
    length = 0.27
    before = model._cumulativeProjectedSolves
    projected, projection, split = model._project_floor_roundoff(profile, reference, deviation, [0, n - 1], grid, length, True, True)
    assert projected[3, 1] == floor and projected[6, 1] == floor
    assert projection.projected_nodes == 2
    assert model._cumulativeProjectedSolves == before + 1
    widths = model._interval_cell_widths(grid, length)
    change = projected - profile
    np.testing.assert_array_equal(projection.inventory_change, widths @ change)
    np.testing.assert_array_equal(split[0], (widths * (1.0 - grid)) @ change)
    np.testing.assert_array_equal(split[0] + split[1], projection.inventory_change)


def test_floor_projection_never_touches_dirichlet_nodes_and_rejects_large_violations():
    model = _stationary_model()
    floor = float(model.constraints.minComposition)
    n = 5
    grid = np.linspace(0.0, 1.0, n)
    reference = np.full((n, 2), 0.3)
    deviation = np.zeros((n, 2))
    profile = reference.copy()
    profile[0, 1] = floor - 1e-18  # Dirichlet node: must be left alone
    profile[-1, 1] = floor - 1e-18
    profile[2, 1] = floor - 1e-6  # far above the round-off threshold
    projected, projection, split = model._project_floor_roundoff(profile, reference, deviation, [0, n - 1], grid, 1.0, True, True)
    np.testing.assert_array_equal(projected, profile)
    assert projection.mask is None
    np.testing.assert_array_equal(split, 0.0)
    assert model._profile_composition_violation(projected) is not None


def test_per_node_shift_matches_constant_shift_threshold():
    """A per-node shift equal to a constant one gives the bitwise same projection as the two-phase call."""
    n = 6
    grid = np.linspace(0.0, 1.0, n)
    floor = 1e-10
    c = np.array([0.4, 0.3])
    deviation = np.zeros((n, 2))
    deviation[:, 1] = floor - c[1]
    profile = c + deviation
    profile[2, 1] = np.nextafter(floor, -np.inf)
    mask = np.zeros(2, dtype=bool)
    a = _project_shift_roundoff_to_floor(profile, c, deviation, mask, floor, -1, grid, 0.5)
    b = _project_shift_roundoff_to_floor(profile, np.broadcast_to(c, (n, 2)), deviation, mask, floor, [n - 1], grid, 0.5)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1].inventory_change, b[1].inventory_change)
    assert a[1].projected_nodes == 1


def test_interface_residual_adds_density_weighted_projection_split():
    model = tpt._make_nonstationary_unequal_volume_model(record=False)
    model.setup()
    profiles = model.getTransformedState()
    interfaces = model.getInterfacePositions()
    etas = model.getInterfaceEtas()
    ic = model._interface_compositions(etas)
    zero = np.zeros(2)

    def bulk(splits):
        return tuple(
            SimpleNamespace(left_transfer=zero, right_transfer=zero, projection_split=np.asarray(s, dtype=np.float64))
            for s in splits
        )

    base = model._interface_residuals(profiles, interfaces, interfaces, ic, bulk([np.zeros((2, 2))] * 3))
    splits = [[[0.0, 0.0], [1e-15, 2e-15]], [[3e-15, 4e-15], [5e-15, 6e-15]], [[7e-15, 8e-15], [0.0, 0.0]]]
    shifted = model._interface_residuals(profiles, interfaces, interfaces, ic, bulk(splits))
    rho = model._normalizedPhaseMolarDensities
    expected = np.concatenate((
        rho[0] * np.array(splits[0][1]) + rho[1] * np.array(splits[1][0]),
        rho[1] * np.array(splits[1][1]) + rho[2] * np.array(splits[2][0]),
    ))
    np.testing.assert_allclose(shifted - base, expected, rtol=1e-12, atol=1e-30)


def test_bulk_result_requires_projection_fields():
    with pytest.raises(TypeError):
        _ThreePhaseBulkResult(np.zeros((2, 2)), np.zeros(2), np.zeros(2), None, None)


def _dilute_outer_interval(model):
    """Phase C interval with Y exactly at the floor beside an interface value of 0.3 (projects with n = 41)."""
    floor = float(model.constraints.minComposition)
    n = 41
    grid = np.linspace(0.0, 1.0, n)
    c = np.array([0.2, 0.3])
    profile = np.tile([0.2, floor], (n, 1))
    profile[0] = c
    return floor, grid, c, profile


def test_dilute_far_field_projection_keeps_picard_queries_in_bounds(monkeypatch):
    """Implicit Picard queries D only with projected, in-bounds iterates; without projection the floor is violated."""
    model = tpt._make_eta_varying_three_phase_model(mode=_BULK_DIFFUSIVITY_IMPLICIT, record=False)
    model.setup()
    floor, grid, c, profile = _dilute_outer_interval(model)
    minimum = [np.inf]
    real = model._interval_face_diffusivity

    def recording(p, g, phase_index, new_bounds):
        minimum[0] = min(minimum[0], float(np.min(p)))
        return real(p, g, phase_index, new_bounds)

    model._interval_face_diffusivity = recording
    model._currdt = 1e-9
    before = model._cumulativeProjectedSolves
    result = model._solve_interval_picard(profile, grid, (0.7, 1.0), (0.7, 1.0), 2, c, None)
    assert result.projection.projected_nodes > 0
    assert model._cumulativeProjectedSolves > before
    assert minimum[0] >= floor
    assert model._profile_composition_violation(result.profile) is None
    np.testing.assert_array_equal(result.projection_split[0], result.projection.inventory_change)  # C: all to B|C
    np.testing.assert_array_equal(result.projection_split[1], 0.0)
    # Each projected value moved by at most the policy threshold.
    assert np.max(np.abs(result.projection.absolute_change)) <= _FLOOR_PROJECTION_ULPS * EPS * 0.6

    import moving_boundary.solvers.illingworth.three_phase as module

    monkeypatch.setattr(module, "_project_shift_roundoff_to_floor", lambda profile, *args, **kwargs: (profile, module._no_floor_projection()))
    model._interval_face_diffusivity = real
    unprojected = model._solve_interval_planar(profile, grid, (0.7, 1.0), (0.7, 1.0), 2, c, None, np.eye(2), validate_diffusivity=False)
    assert model._profile_composition_violation(unprojected.profile) is not None


def test_step_projection_totals_are_staged_and_committed_once():
    """Projection totals reach the cumulative diagnostics only when postProcess commits the step."""
    model, _ = tpt._make_stationary_model(record=False)
    model.setup()
    staged = (np.array([1e-18, -2e-18]), np.array([3e-18, 4e-18]), 2)
    real = model._record_step_projection

    def fake(candidate):
        real(candidate)
        model._lastStepProjectionInventoryChange = staged[0].copy()
        model._lastStepProjectionAbsoluteChange = staged[1].copy()
        model._lastStepProjectedNodes = staged[2]

    model._record_step_projection = fake
    commits = []
    original = model.postProcess

    def record(time, x):
        result = original(time, x)
        commits.append(time)
        return result

    model.postProcess = record
    x = model.getCurrentX()
    model.getdXdt(model.currentTime, x)
    model.getdXdt(model.currentTime, x)  # proposals alone must not count
    np.testing.assert_array_equal(model._cumulativeProjectionInventoryChange, 0.0)
    model.solve(3 * model.timeStep)
    steps = len(commits)
    assert steps >= 1
    np.testing.assert_allclose(model._cumulativeProjectionInventoryChange, staged[0] * steps, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(model._cumulativeProjectionAbsoluteChange, staged[1] * steps, rtol=1e-15, atol=0.0)
    assert model._cumulativeProjectedNodes == staged[2] * steps


class _PhaseDiffusivity:
    """Phase-wise constant matrices; records the smallest composition component queried."""

    def __init__(self, matrices):
        self.matrices = matrices
        self.minimum = np.inf

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        self.minimum = min(self.minimum, float(values.min()))
        D = self.matrices[phase]
        if values.ndim == 2:
            return np.broadcast_to(D, (values.shape[0], 2, 2)).copy()
        return D.copy()


def test_full_run_with_middle_phase_floor_projection():
    """Moving full run in which phase B projects: no retries, split accounted, queries in bounds, molar drift at tolerance.

    Phase B's interior (0.15 < xi < 0.85) starts with Y exactly at the floor beside interface Y
    values of 0.15 and 0.20 and keeps it (D_YY^B = 1e-8); outer far fields slightly off the
    interface compositions move both interfaces. On 720d122 (absolute form) this run fails at
    its first step, and it does so here too with the projection disabled.
    """
    eq_ab = tpt._LinearPairEquilibrium([0.20, 0.10], [0.10, 0.0], [0.30, 0.15], [0.10, 0.0])
    eq_bc = tpt._LinearPairEquilibrium([0.24, 0.15], [0.10, 0.0], [0.18, 0.20], [0.10, 0.0])
    etas0, interfaces = (0.2, 0.8), (0.35, 0.7)
    c_a, c_b_ab = eq_ab.interface_compositions(etas0[0])
    c_b_bc, c_c = eq_bc.interface_compositions(etas0[1])
    linear = tpt._ThreePhaseLinearProfile(
        interfaces,
        (c_a + 0.1 * (np.array([0.17, 0.07]) - c_a), c_a, c_b_ab, c_b_bc, c_c, c_c + 0.1 * (np.array([0.31, 0.17]) - c_c)),
        middle_bump=[0.0, 0.0],
    )
    floor = 1e-10

    def profile(z):
        out = linear(z)
        xi = (np.asarray(z, dtype=np.float64).reshape(-1) - interfaces[0]) / (interfaces[1] - interfaces[0])
        out[(xi > 0.15) & (xi < 0.85), 1] = floor
        return out

    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 701)
    mesh.setResponseProfile(ProfileBuilder([(profile, ["X", "Y"])]))
    provider = _PhaseDiffusivity({"A": np.eye(2), "B": np.diag([1.0, 1e-8]), "C": np.eye(2)})
    model = MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["A", "B", "C"], thermodynamics=provider, temperature=1000.0,
        interfacePositions=interfaces, interface_equilibria=(eq_ab, eq_bc), initial_eta_guess=etas0,
        bulk_diffusivity_mode=_BULK_DIFFUSIVITY_LAGGED, phase_nodes=(21, 41, 21), time_step=1e-3, record=False,
        tolerance=1e-10, residual_tolerance=1e-10, max_iterations=50, terminal_thin_phase_policy="continue",
        phase_molar_volumes=(2.0, 3.0, 1.5),
    )
    b_nodes, b_total, b_split = [0], np.zeros(2), np.zeros((2, 2))
    real_record = model._record_step_projection

    def record_projection(candidate):
        real_record(candidate)
        result = candidate.bulk_results[1]
        if result.projection.mask is not None:
            b_nodes[0] += result.projection.projected_nodes
            b_total[:] += result.projection.inventory_change
            b_split[:] += result.projection_split

    model._record_step_projection = record_projection
    retries = []
    original = model.postProcess

    def post(time, x):
        out = original(time, x)
        retries.append(int(model._lastStepRetries))
        return out

    model.postProcess = post
    model.solve(0.1)
    assert len(retries) == 100 and sum(retries) == 0
    assert model.getInterfacePositions()[0] > 0.355 and model.getInterfacePositions()[1] > 0.71  # both moved
    assert b_nodes[0] > 0  # the middle phase projected on accepted steps
    np.testing.assert_allclose(b_split.sum(axis=0), b_total, rtol=1e-12, atol=1e-30)
    assert np.all(np.abs(model._cumulativeProjectionInventoryChange) <= _FLOOR_PROJECTION_ULPS * EPS * model._cumulativeProjectedNodes)
    assert provider.minimum >= floor
    for stored in model.getTransformedState():
        assert model._profile_composition_violation(stored) is None
    drift = np.abs(model.getMolarInventories() - model._initialMolarInventories)
    assert np.all(drift <= len(retries) * model.residualTolerance)


# --------------------------------------------------------------------------
# Fine interface grid (ledger R06 probe: 201 nodes, end spacing ratio 1e-8)
# --------------------------------------------------------------------------
def _fine_grid_model(t_end):
    solution = sim._solution("opposite")
    nodes = 201
    grids = tuple(landau_grid(nodes, "linear", phase=phase, spacing_ratio=1e-8, min_interval=1e-15) for phase in ("left", "middle", "right"))
    mesh = CartesianFD1D(["X", "Y"], [0.0, sim.DOMAIN_LENGTH], 6001)
    mesh.setResponseProfile(ProfileBuilder([(sim.SimilarityInitialProfile(solution, sim.T0), ["X", "Y"])]))
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["A", "B", "C"],
        thermodynamics=sim.FixedMatrixTernaryDiffusivity(sim.D, ["A", "B", "C"]), temperature=1000.0,
        interfacePositions=tuple(solution.interface_positions(sim.T0)), interface_equilibria=(sim.CLOSURE_AB, sim.CLOSURE_BC),
        initial_eta_guess=tuple(solution.etas), time_step=t_end - sim.T0, dt_mode="semi_log", semiLog_dt=0.025,
        semiLogT0=1.0e-3 * sim.T0, transformed_grids=grids, tolerance=1.0e-11, max_iterations=50,
        terminal_thin_phase_policy="continue", record=False,
    )


def test_fine_interface_grid_runs_without_retries_and_residual_is_smooth():
    """On 1e-12-scale interface cells the step converges without retries and the residual is smooth near the root."""
    t_end = 0.05
    model = _fine_grid_model(t_end)
    retries = []
    original = model.postProcess

    def record(time, x):
        result = original(time, x)
        retries.append(int(model._lastStepRetries))
        return result

    model.postProcess = record
    model.solve(t_end - sim.T0)
    assert len(retries) > 50
    assert sum(retries) == 0  # absolute form: retries from t ~ 0.008 on
    # Each accepted step leaves at most the residual tolerance (scaled by R0)
    # per component as inventory defect.
    drift = np.abs(model.getTotalInventory() - model._initialInventory)
    print(f"fine-grid drift {drift}, steps {len(retries)}")
    assert np.all(drift <= len(retries) * model.residualTolerance * float(model._R0))

    profiles = tuple(p.copy() for p in model.getTransformedState())
    interfaces = model.getInterfacePositions().copy()
    etas = model.getInterfaceEtas().copy()
    t_now = float(model.currentTime)
    dt = t_now * (np.exp(0.025) - 1.0)
    model._currdt = dt
    model._laggedFaceDiffusivityCache = {}
    model._interfaceCompositionCache = {}
    root = model._solve_interface_planar(profiles, interfaces, etas, dt)
    scale = model._residual_scale(profiles, interfaces)
    ks = np.arange(-5, 6)
    for variable in range(4):
        values = []
        for k in ks:
            x_hat = root.x_hat.copy()
            x_hat[variable] += k * 1e-12
            values.append(model._evaluate_interface_candidate(profiles, interfaces, x_hat, scale, dt).scaled_residual)
        values = np.array(values)
        for j in range(4):
            fit = np.polyval(np.polyfit(ks, values[:, j], 2), ks)
            assert np.max(np.abs(values[:, j] - fit)) <= 1e-15  # absolute form: 5e-11 to 3e-10
