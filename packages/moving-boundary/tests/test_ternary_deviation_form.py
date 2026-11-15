"""Deviation-form phase solves of the two-phase ternary Illingworth model (issue R06).

Each phase is solved for ``w = c - c_interface(eta)`` with the trial candidate's
interface compositions. That is exact only because every non-Dirichlet block
row sums to its old-volume coefficient times the identity; these tests check
that identity on every branch, that a state already at the interface
compositions is preserved exactly, that one step agrees with 50-digit
arithmetic far from or across the interface compositions (including changing
eta, face-dependent diffusivity and interstitials), and that the interface
residual near a fine-grid step is free of the round-off noise that made steps
retry (R06).

The reconstruction ``w + c`` resolves each composition only to about
``eps * (|c| + |w|)``, so a dilute value just above the composition floor can
round below it. ``_project_shift_roundoff_to_floor`` repairs such violations
within a fixed policy threshold and reports the signed inventory change, which
the interface residual includes. The floor tests check the repair through the
production composition validator, its accounting, provider queries and
conservation.

Ported from samuelgprice/kawin 2741135 (kawin/tests/test_illingworth_ternary_deviation_form.py).
Test bodies are unchanged except that ``_profile_within_floor_bounds`` and
``_no_floor_projection`` are read from ``moving_boundary.solvers.illingworth.shared``
(the package's home for the research module's pre-class helpers) and that the
fine-grid model takes its research helpers (``kawin/tests/test_illingworth_interstitials_ternary.py``:
``_closure``, ``_diffusivities``, ``_x_from_u`` and the case constants) from the copy
reproduced below. Imports are the package's.

LEDGERS R13 (hand edit, this file is not generated): ``test_interface_residual_is_smooth_near_a_fine_grid_step``
sets ``interfaceResidualCompositionScale`` to 1.0 on its model, i.e. the default residual normalization in
force when it was written, because its ``1e-15`` noise threshold is in units of that scale (residual over
the domain length). The other tests, including the no-retry run, use the current default.
"""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from types import SimpleNamespace

import moving_boundary.solvers.illingworth.shared as ternary_shared
import moving_boundary.solvers.illingworth.ternary as ternary_fdm
from moving_boundary.equilibrium import CallableTernaryInterfaceEquilibrium
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from moving_boundary.mesh.TransformedGrids import landau_grid, two_phase_landau_grids
from moving_boundary.solvers import MovingBoundaryIllingworthTernaryFD1DModel
from moving_boundary.solvers.illingworth._interstitials import composition_bound_violation
from moving_boundary.solvers.illingworth.numerics.ternary import integrate_planar_transformed_profile_components
from moving_boundary.solvers.illingworth.shared import (
    _FLOOR_PROJECTION_ULPS,
    _project_shift_roundoff_to_floor,
    _select_interface_motion_branch,
)

EPS = np.finfo(float).eps
MODES = ("phase_uniform", "composition_dependent_lagged", "composition_dependent_implicit")
SUBSTITUTIONAL = ["Z", "X", "Y"]
FE_C_MN = ["FE", "C", "MN"]


class _LinearClosure:
    """Linear, non-crossing tie-line family; ``left(eta)``, ``right(eta)`` are affine in eta."""

    eta_bounds = (0.0, 1.0)

    def __init__(self, left0, left1, right0, right1):
        self.left0, self.left1 = np.asarray(left0, dtype=np.float64), np.asarray(left1, dtype=np.float64)
        self.right0, self.right1 = np.asarray(right0, dtype=np.float64), np.asarray(right1, dtype=np.float64)

    def interface_compositions(self, eta):
        eta = float(eta)
        return self.left0 + eta * (self.left1 - self.left0), self.right0 + eta * (self.right1 - self.right0)


class _RecordingSmoothThermodynamics:
    """Composition-dependent coupled matrices (both phases) that record every queried composition."""

    def __init__(self, scale=1.0):
        self.scale = float(scale)
        self.compositions = []

    def clearCache(self):
        pass

    def _matrix(self, x, y, phase):
        if phase == "ALPHA":
            m = [[1.0e-3 + 1.0e-4 * x, 2.0e-4 + 2.0e-5 * y], [1.0e-4 + 1.0e-5 * x, 8.0e-4 + 8.0e-5 * y]]
        else:
            m = [[7.0e-4 + 8.0e-5 * x, -1.0e-4 + 1.0e-5 * y], [2.0e-4 + 1.0e-5 * x, 1.1e-3 + 7.0e-5 * y]]
        return self.scale * np.asarray(m, dtype=np.float64)

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        self.compositions.append(values.copy())
        out = np.asarray([self._matrix(x, y, phase) for x, y in values], dtype=np.float64)
        return out[0] if single else out


class _IsotropicThermodynamics:
    """
    ``D(x) * I`` in both phases (optionally composition dependent), recording queried compositions.

    With an isotropic matrix each independent component, and the dependent one, obeys a discrete
    maximum principle under the upwind ALE scheme, so a dilute component can only cross its floor
    through round-off. Coupled matrices can undershoot genuinely (in either solve form).
    """

    def __init__(self, D=1.0e-3, slope=0.0):
        self.D, self.slope = float(D), float(slope)
        self.compositions = []

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        self.compositions.append(values.copy())
        out = np.asarray([(self.D + self.slope * x) * np.eye(2) for x, _ in values], dtype=np.float64)
        return out[0] if single else out


def _model(closure, eta0=0.3, elements=SUBSTITUTIONAL, mode="phase_uniform", thermodynamics=None, nodes=(14, 18),
           s0=0.45, length=1.0, grids=None, tolerance=1.0e-12, min_composition=None, bulk=None):
    """
    Two-phase ternary model whose initial state is the step between ``closure`` compositions at ``eta0``.

    ``bulk`` optionally replaces those step values by ``(left, right)`` bulk compositions so that the
    startup eta balance has a root; tests that set their own profiles only need a valid setup.
    """
    left, right = closure.interface_compositions(eta0) if bulk is None else (np.asarray(bulk[0]), np.asarray(bulk[1]))
    if elements == FE_C_MN:  # u-fraction closure, mesh profile in mole fractions
        left, right = left / (1.0 + left[0]), right / (1.0 + right[0])
    mesh = CartesianFD1D(list(elements[1:]), [0.0, length], 101)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(s0 * length, left, right), list(elements[1:]))]))
    if grids is None:
        grids = two_phase_landau_grids(nodes[0], nodes[1], method="linear",
                                       left_kwargs={"spacing_ratio": 0.2}, right_kwargs={"spacing_ratio": 0.05})
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh, elements=list(elements), phases=["ALPHA", "BETA"],
        thermodynamics=_RecordingSmoothThermodynamics() if thermodynamics is None else thermodynamics,
        temperature=1000.0, interfacePosition=s0 * length, interface_equilibrium=closure,
        initial_eta_bracket=(0.0, 1.0), initial_eta_guess=eta0, bulk_diffusivity_mode=mode,
        time_step=1.0e-3, tolerance=tolerance, max_iterations=50, terminal_thin_phase_policy="continue",
        record=True, transformed_u_grid=grids[0], transformed_v_grid=grids[1],
    )
    if min_composition is not None:
        model.constraints.minComposition = float(min_composition)
    model.setup()
    return model


def _substitutional_closure():
    return _LinearClosure([0.20, 0.10], [0.30, 0.12], [0.30, 0.15], [0.40, 0.13])


def _u_closure(basis="u"):
    left0, left1 = [0.002, 0.010], [0.002, 0.030]
    right0, right1 = [0.030, 0.020], [0.020, 0.050]
    closure = _LinearClosure(left0, left1, right0, right1)
    return CallableTernaryInterfaceEquilibrium(closure.interface_compositions, eta_bounds=(0.0, 1.0), composition_basis=basis)


def _capture_block_systems(monkeypatch):
    systems = []
    real = ternary_fdm.solve_illingworth_block_tridiagonal

    def capture(lower, diagonal, upper, rhs):
        systems.append(tuple(np.array(a, dtype=np.float64) for a in (lower, diagonal, upper, rhs)))
        return real(lower, diagonal, upper, rhs)

    monkeypatch.setattr(ternary_fdm, "solve_illingworth_block_tridiagonal", capture)
    return systems


def _old_volumes(grid, side, length):
    """Old-volume weight per node: half-cell at the free end, cell widths inside (times the old phase length)."""
    g = np.asarray(grid, dtype=np.float64)
    volume = np.zeros_like(g)
    volume[1:-1] = (g[2:] - g[:-2]) / 2.0
    if side == "left":
        volume[0] = g[1] / 2.0
    else:
        volume[-1] = (1.0 - g[-2]) / 2.0
    return volume * length


def _smooth_profiles(model, left_shape, right_shape):
    """Profiles from callables of the Landau coordinate, with the interface endpoints at the current compositions."""
    p = np.asarray(left_shape(model._u_grid), dtype=np.float64)
    q = np.asarray(right_shape(model._v_grid), dtype=np.float64)
    c_left, c_right = model._interface_compositions(model._eta_curr)
    p[-1], q[0] = c_left, c_right
    return p, q


# --- Row-sum identity ------------------------------------------------------------------


@pytest.mark.parametrize("phase", ("left", "right"))
@pytest.mark.parametrize("branch", ("positive", "negative"))
@pytest.mark.parametrize("face_arrays", (False, True))
@pytest.mark.parametrize("compiled", (True, False))
def test_block_rows_sum_to_old_volume_so_deviation_rhs_is_exact(monkeypatch, phase, branch, face_arrays, compiled):
    helper = f"_fill_ternary_{phase}_planar_interior"
    if compiled and getattr(ternary_fdm, helper) is None:
        pytest.skip("Numba is unavailable")
    if not compiled:
        monkeypatch.setattr(ternary_fdm, helper, None)
    model = _model(_substitutional_closure())
    p, q = _smooth_profiles(model, lambda u: np.column_stack((0.25 + 0.05 * u**2, 0.11 - 0.01 * u)),
                            lambda v: np.column_stack((0.33 - 0.03 * np.sqrt(v), 0.14 + 0.02 * v)))
    c_left, c_right = model._interface_compositions(0.55)  # trial composition differs from the old endpoint
    s, dt = float(model._s_curr), 0.07
    future_s = s * (1.03 if branch == "positive" else 0.97)
    D = np.asarray([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]])
    n = len(p) if phase == "left" else len(q)
    faces = np.asarray([D * (1.0 + 0.05 * k) + 1.0e-5 * k for k in range(n - 1)]) if face_arrays else D
    systems = _capture_block_systems(monkeypatch)
    if phase == "left":
        model._solve_concentration_left_planar(p, s, future_s, dt, c_left, faces, branch)
        grid, old, c, length, rows, dirichlet = model._u_grid, p, c_left, s, slice(0, -1), -1
    else:
        model._solve_concentration_right_planar(q, s, future_s, dt, c_right, faces, branch)
        grid, old, c, length, rows, dirichlet = model._v_grid, q, c_right, model._R - s, slice(1, None), 0
    lower, diagonal, upper, rhs = systems[0]

    volume = _old_volumes(grid, phase, length)[rows]
    row_sum = (lower + diagonal + upper)[rows]
    magnitude = (np.abs(lower) + np.abs(diagonal) + np.abs(upper))[rows].max()
    expected = -volume[:, None, None] * np.eye(2)
    assert np.max(np.abs(row_sum - expected)) <= 1.0e-13 * magnitude
    np.testing.assert_allclose(rhs[rows], -volume[:, None] * (old[rows] - c), rtol=1.0e-14, atol=0.0)
    assert_array_equal(diagonal[dirichlet], -np.eye(2))
    assert_array_equal(rhs[dirichlet], np.zeros(2))


# --- Exact preservation of an interface-composition state --------------------------------


def _uniform_interface_state(model):
    c_left, c_right = model._interface_compositions(model._eta_curr)
    p = np.broadcast_to(c_left, model._p_curr.shape).copy()
    q = np.broadcast_to(c_right, model._q_curr.shape).copy()
    return p, q


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("elements", ("substitutional", "interstitial"))
def test_state_at_interface_compositions_is_preserved_exactly(mode, elements):
    if elements == "substitutional":
        model = _model(_substitutional_closure(), mode=mode)
    else:
        model = _model(_u_closure(), elements=FE_C_MN, mode=mode)
    p, q = _uniform_interface_state(model)
    s, eta = float(model._s_curr), float(model._eta_curr)
    old_s = s
    for dt in (1.0e-3, 1.0e-2, 0.1, 1.0, 10.0):
        result = model._solve_interface_planar(p, q, s, old_s, eta, dt)
        assert_array_equal(result[0], p)
        assert_array_equal(result[1], q)
        assert result[2] == s
        assert result[3] == eta
        assert model._lastStepProjectedNodes == 0


# --- One step against 50-digit arithmetic ---------------------------------------------


def _dec(x):
    return Decimal(float(x))


def _mat_dec(m):
    m = np.asarray(m, dtype=np.float64)
    return [[_dec(m[0, 0]), _dec(m[0, 1])], [_dec(m[1, 0]), _dec(m[1, 1])]]


def _madd(a, b):
    return [[a[i][j] + b[i][j] for j in range(2)] for i in range(2)]


def _mscale(a, k):
    return [[a[i][j] * k for j in range(2)] for i in range(2)]


def _eye(k):
    return [[k, Decimal(0)], [Decimal(0), k]]


def _mmul(a, b):
    return [[a[i][0] * b[0][j] + a[i][1] * b[1][j] for j in range(2)] for i in range(2)]


def _mv(a, v):
    return [a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1]]


def _minv(a):
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return [[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]]


def _block_thomas_dec(lower, diagonal, upper, rhs):
    n = len(diagonal)
    cu, dr = [None] * n, [None] * n
    inv = _minv(diagonal[0])
    cu[0], dr[0] = _mmul(inv, upper[0]), _mv(inv, rhs[0])
    for i in range(1, n):
        inv = _minv(_madd(diagonal[i], _mscale(_mmul(lower[i], cu[i - 1]), Decimal(-1))))
        r = _mv(lower[i], dr[i - 1])
        cu[i] = _mmul(inv, upper[i])
        dr[i] = _mv(inv, [rhs[i][0] - r[0], rhs[i][1] - r[1]])
    x = [None] * n
    x[-1] = dr[-1]
    for i in range(n - 2, -1, -1):
        t = _mv(cu[i], x[i + 1])
        x[i] = [dr[i][0] - t[0], dr[i][1] - t[1]]
    return x


def _faces_dec(D, n_faces):
    D = np.asarray(D, dtype=np.float64)
    return [_mat_dec(D)] * n_faces if D.shape == (2, 2) else [_mat_dec(D[k]) for k in range(n_faces)]


def _exact_left(p, u, s, fs, dt, c_left, D, branch):
    """Absolute-form left solve in 50-digit arithmetic (both branches, uniform or face matrices)."""
    n, zero = len(p), Decimal(0)
    p = [[_dec(a), _dec(b)] for a, b in np.asarray(p)]
    u = [_dec(x) for x in u]
    A = [_mscale(m, dt / fs) for m in _faces_dec(D, n - 1)]
    B = fs - s
    lo, dg, up, rhs = [_eye(zero)] * n, [_eye(zero)] * n, [_eye(zero)] * n, [[zero, zero]] * n
    if branch == "positive":
        dg[0] = _madd(_mscale(A[0], -1 / u[1]), _eye(-fs * u[1] / 2))
        up[0] = _madd(_mscale(A[0], 1 / u[1]), _eye(B * u[1] / 2))
    else:
        dg[0] = _madd(_mscale(A[0], -1 / u[1]), _eye(B * u[1] / 2 - fs * u[1] / 2))
        up[0] = _mscale(A[0], 1 / u[1])
    rhs[0] = [-p[0][k] * s * u[1] / 2 for k in range(2)]
    for i in range(1, n - 1):
        ld, rd, ls, rs = u[i] - u[i - 1], u[i + 1] - u[i], u[i] + u[i - 1], u[i + 1] + u[i]
        base = _madd(_mscale(A[i - 1], -1 / ld), _mscale(A[i], -1 / rd))
        if branch == "positive":
            lo[i] = _mscale(A[i - 1], 1 / ld)
            dg[i] = _madd(base, _eye(-(B * ls / 2 + fs * (rs - ls) / 2)))
            up[i] = _madd(_mscale(A[i], 1 / rd), _eye(B * rs / 2))
        else:
            lo[i] = _madd(_mscale(A[i - 1], 1 / ld), _eye(-B * ls / 2))
            dg[i] = _madd(base, _eye(B * rs / 2 - fs * (rs - ls) / 2))
            up[i] = _mscale(A[i], 1 / rd)
        rhs[i] = [-s * p[i][k] * (rs - ls) / 2 for k in range(2)]
    dg[-1] = _eye(Decimal(-1))
    rhs[-1] = [-_dec(x) for x in c_left]
    return _block_thomas_dec(lo, dg, up, rhs)


def _exact_right(q, v, R, s, fs, dt, c_right, D, branch):
    """Absolute-form right solve in 50-digit arithmetic (both branches, uniform or face matrices)."""
    n, zero = len(q), Decimal(0)
    q = [[_dec(a), _dec(b)] for a, b in np.asarray(q)]
    v = [_dec(x) for x in v]
    span, old_span, B = R - fs, R - s, fs - s
    A = [_mscale(m, dt / span) for m in _faces_dec(D, n - 1)]
    lo, dg, up, rhs = [_eye(zero)] * n, [_eye(zero)] * n, [_eye(zero)] * n, [[zero, zero]] * n
    dg[0] = _eye(Decimal(-1))
    rhs[0] = [-_dec(x) for x in c_right]
    for i in range(1, n - 1):
        ld, rd, ls, rs = v[i] - v[i - 1], v[i + 1] - v[i], v[i] + v[i - 1], v[i + 1] + v[i]
        base = _madd(_mscale(A[i], -1 / rd), _mscale(A[i - 1], -1 / ld))
        if branch == "positive":
            lo[i] = _mscale(A[i - 1], 1 / ld)
            dg[i] = _madd(base, _eye(-(B * (1 - ls / 2) + span * (rs - ls) / 2)))
            up[i] = _madd(_mscale(A[i], 1 / rd), _eye(B * (1 - rs / 2)))
        else:
            lo[i] = _madd(_mscale(A[i - 1], 1 / ld), _eye(-B * (1 - ls / 2)))
            dg[i] = _madd(base, _eye(B * (1 - rs / 2) - span * (rs - ls) / 2))
            up[i] = _mscale(A[i], 1 / rd)
        rhs[i] = [-old_span * q[i][k] * (rs - ls) / 2 for k in range(2)]
    t = v[-2]
    if branch == "positive":
        lo[-1] = _mscale(A[-1], 1 / (1 - t))
        dg[-1] = _madd(_mscale(A[-1], -1 / (1 - t)), _eye(-(B * (1 - (1 + t) / 2) + span * (1 - t) / 2)))
    else:
        lo[-1] = _madd(_mscale(A[-1], 1 / (1 - t)), _eye(-B * (1 - (1 + t) / 2)))
        dg[-1] = _madd(_mscale(A[-1], -1 / (1 - t)), _eye(-span * (1 - t) / 2))
    rhs[-1] = [-q[-1][k] * old_span * (1 - t) / 2 for k in range(2)]
    return _block_thomas_dec(lo, dg, up, rhs)


def _exact_candidate(model, p, q, s, fs, dt, c_left, c_right, c_left_old, c_right_old, DL, DR, branch):
    """Exact profiles, interface deviations and residual (plus a residual term scale) for float inputs."""
    sd, fsd, dtd, R = _dec(s), _dec(fs), _dec(dt), _dec(model._R)
    pn = _exact_left(p, model._u_grid, sd, fsd, dtd, c_left, DL, branch)
    qn = _exact_right(q, model._v_grid, R, sd, fsd, dtd, c_right, DR, branch)
    u2, v1 = _dec(model._u_grid[-2]), _dec(model._v_grid[1])
    cl, cr = [_dec(x) for x in c_left], [_dec(x) for x in c_right]
    clo, cro = [_dec(x) for x in c_left_old], [_dec(x) for x in c_right_old]
    wl = [pn[-2][k] - cl[k] for k in range(2)]
    wr = [qn[1][k] - cr[k] for k in range(2)]
    gl = _mv(_faces_dec(DL, len(p) - 1)[-1], [-wl[k] / (1 - u2) for k in range(2)])
    gr = _mv(_faces_dec(DR, len(q) - 1)[0], [wr[k] / v1 for k in range(2)])
    residual, term_scale = [], []
    for k in range(2):
        flux = (gr[k] / (R - fsd) - gl[k] / fsd) * dtd
        if branch == "positive":
            lhs = (cl[k] - cr[k]) - wr[k] * (1 - v1 / 2)
        else:
            lhs = (cl[k] - cr[k]) + wl[k] * (Decimal("0.5") + u2 / 2)
        end_l = sd * Decimal("0.5") * (1 - u2) * (cl[k] - clo[k])
        end_r = (R - sd) * Decimal("0.5") * v1 * (cr[k] - cro[k])
        residual.append((fsd - sd) * lhs - flux + end_l + end_r)
        term_scale.append(abs((fsd - sd) * lhs) + abs(gr[k] / (R - fsd) * dtd) + abs(gl[k] / fsd * dtd) + abs(end_l) + abs(end_r))
    return pn, qn, wl, wr, residual, term_scale


# (closure, left profile of u, right profile of v); the trial eta differs from the old one.
# "dilute": components near 0 next to larger interface values; "near_one": a dependent component
# near zero (X + Y near 1); "crossing": both phases cross their interface compositions.
FAR_STATES = {
    "dilute": (_LinearClosure([0.30, 0.20], [0.35, 0.22], [0.05, 0.40], [0.04, 0.45]),
               lambda u: np.column_stack((0.30 * u**4 + 1.0e-4, 0.20 * u**3 + 2.0e-4)),
               lambda v: np.column_stack((0.05 * np.exp(-6.0 * v) + 1.0e-4, 0.40 - 0.05 * v))),
    "near_one": (_LinearClosure([0.60, 0.385], [0.62, 0.37], [0.50, 0.40], [0.52, 0.38]),
                 lambda u: np.column_stack((0.60 + 0.05 * (1.0 - u), 0.39 - 0.05 * (1.0 - u) + 0.003 * u)),
                 lambda v: np.column_stack((0.50 - 0.2 * v, 0.40 + 0.2 * v - 1.0e-4))),
    "crossing": (_substitutional_closure(),
                 lambda u: np.column_stack((0.25 + 0.06 * np.sin(9.0 * u) * (1.0 - u), 0.11 + 0.02 * np.cos(7.0 * u))),
                 lambda v: np.column_stack((0.34 + 0.05 * np.sin(11.0 * v), 0.14 + 0.03 * np.sin(5.0 * v)))),
}


def _ulps(err, scale):
    return float(err) / (EPS * float(scale))


def _one_step_case(closure, left_shape, right_shape, elements, mode, branch, eta_trial):
    """One candidate at ``eta_trial`` (not the old eta) and its 50-digit evaluation from the same float inputs."""
    model = _model(closure, elements=elements, mode=mode)
    p, q = _smooth_profiles(model, left_shape, right_shape)
    s, dt = float(model._s_curr), 2.0
    model.currentTime = 1.0
    eta_lower, _, eta_span = model._eta_scaling_bounds()
    x_hat = model._interface_physical_to_scaled(s * (1.05 if branch == "positive" else 0.95), eta_trial, eta_lower, eta_span)
    fs, eta_used = model._interface_scaled_to_physical(x_hat, eta_lower, eta_span)
    candidate = model._evaluate_interface_candidate(p, q, s, s, dt, eta_lower, eta_span, model._interface_residual_scale(p, q, s),
                                                    p[-1].copy(), q[0].copy(), x_hat, branch)
    c_left, c_right = model._interface_compositions(eta_used)
    if mode == "phase_uniform":
        DL = model._phase_diffusivity_matrix(c_left, "ALPHA", model.currentTime, fs)
        DR = model._phase_diffusivity_matrix(c_right, "BETA", model.currentTime, fs)
    else:
        DL = model._left_lagged_face_diffusivity_matrices(p, fs, model.currentTime)
        DR = model._right_lagged_face_diffusivity_matrices(q, fs, model.currentTime)
    left = model._solve_concentration_left_planar(p, s, fs, dt, c_left, DL, branch)
    right = model._solve_concentration_right_planar(q, s, fs, dt, c_right, DR, branch)
    with localcontext() as ctx:
        ctx.prec = 50
        exact = _exact_candidate(model, p, q, s, fs, dt, c_left, c_right, p[-1], q[0], DL, DR, branch)
    old_c_left, _ = model._interface_compositions(model._eta_curr)
    return candidate, left, right, c_left, c_right, old_c_left, exact


def _check_against_exact(candidate, left, right, c_left, c_right, exact):
    """
    Deviation-form candidate against 50-digit arithmetic from the same float inputs.

    Profiles are compared in ulps of ``|c| + |w|`` (the resolution of ``w + c``), the solver's
    interface-adjacent deviations in ulps of their own size (relative precision is what R06
    restores) and the residual in ulps of the sum of its term magnitudes. The measured margins
    are evidence for, not a proof of, the floor-projection policy threshold.
    """
    pn, qn, wl, wr, residual, term_scale = exact
    worst = {}
    for name, flt, ex, shift in (("left", candidate.p_future, pn, c_left), ("right", candidate.q_future, qn, c_right)):
        worst[name] = max(
            _ulps(abs(flt[i, k] - float(ex[i][k])), abs(shift[k]) + abs(float(ex[i][k]) - shift[k]))
            for i in range(len(flt)) for k in range(2)
        )
    dev_err = [_ulps(abs(left.interface_deviation[k] - float(wl[k])), abs(float(wl[k]))) for k in range(2)]
    dev_err += [_ulps(abs(right.interface_deviation[k] - float(wr[k])), abs(float(wr[k]))) for k in range(2)]
    res_err = [_ulps(abs(candidate.residual[k] - float(residual[k])), term_scale[k]) for k in range(2)]
    assert_array_equal(left.profile, candidate.p_future)
    assert_array_equal(right.profile, candidate.q_future)
    assert worst["left"] <= 4.0 and worst["right"] <= 4.0, worst
    assert max(dev_err) <= 64.0, dev_err
    assert max(res_err) <= 16.0, res_err


@pytest.mark.parametrize("state", sorted(FAR_STATES))
@pytest.mark.parametrize("branch", ("positive", "negative"))
@pytest.mark.parametrize("mode", ("phase_uniform", "composition_dependent_lagged"))
def test_step_with_changing_eta_matches_exact_arithmetic(state, branch, mode):
    closure, left_shape, right_shape = FAR_STATES[state]
    candidate, left, right, c_left, c_right, old_c_left, exact = _one_step_case(
        closure, left_shape, right_shape, SUBSTITUTIONAL, mode, branch, 0.8)
    assert np.max(np.abs(c_left - old_c_left)) > 1.0e-3  # eta changes: the shift differs from p[-1]
    _check_against_exact(candidate, left, right, c_left, c_right, exact)


@pytest.mark.parametrize("branch", ("positive", "negative"))
@pytest.mark.parametrize("mode", ("phase_uniform", "composition_dependent_lagged"))
def test_interstitial_step_with_changing_eta_matches_exact_arithmetic(branch, mode):
    left_shape = lambda u: np.column_stack((0.002 + 0.0005 * (1.0 - u), 0.015 - 0.005 * u**2))
    right_shape = lambda v: np.column_stack((0.010 + 0.02 * np.exp(-5.0 * v), 0.015 + 0.01 * np.exp(-3.0 * v)))
    candidate, left, right, c_left, c_right, old_c_left, exact = _one_step_case(
        _u_closure(), left_shape, right_shape, FE_C_MN, mode, branch, 0.7)
    assert np.max(np.abs(c_left - old_c_left)) > 1.0e-4
    _check_against_exact(candidate, left, right, c_left, c_right, exact)


# --- Round-off floor projection: helper ------------------------------------------------


def _helper(profile, shift, deviation=None, mask=(False, False), floor=1.0e-10, fixed_node=-1):
    profile = np.asarray(profile, dtype=np.float64)
    deviation = profile - np.asarray(shift) if deviation is None else np.asarray(deviation, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, len(profile))
    return _project_shift_roundoff_to_floor(profile, shift, deviation, np.asarray(mask), floor, fixed_node, grid, 0.7)


def test_floor_projection_sets_roundoff_violation_of_independent_component_to_floor():
    profile = [[1.0e-10 - 5.0e-18, 0.2], [0.11, 0.2], [0.1, 0.2]]
    projected, record = _helper(profile, [0.1, 0.2])
    assert projected[0, 0] == 1.0e-10 and projected[0, 1] == 0.2
    assert_array_equal(projected[1:], np.asarray(profile)[1:])
    assert composition_bound_violation(projected, np.zeros(2, dtype=bool), 1.0e-10) is None
    weights = np.asarray([0.25, 0.5, 0.25]) * 0.7
    assert record.inventory_change[0] == pytest.approx(weights[0] * (projected[0, 0] - profile[0][0]), rel=1e-12)
    assert record.inventory_change[1] == 0.0
    assert record.projected_nodes == 1 and record.repairs == ()


def test_floor_projection_repairs_sub_half_ulp_dependent_deficit_with_nextafter():
    # Codex review case: subtracting the 5.3e-17 deficit from 0.6 leaves 0.6 unchanged.
    floor = 1.0e-7
    row = np.asarray([0.6, 0.3999999000000001])
    assert 1.0 - np.sum(row) < floor
    assert row[0] - (floor - (1.0 - np.sum(row))) == row[0]
    projected, record = _helper([row, [0.3, 0.3], [0.6, 0.39]], [0.6, 0.39], floor=floor)
    assert composition_bound_violation(projected, np.zeros(2, dtype=bool), floor) is None
    assert projected[0, 1] == row[1] and projected[0, 0] < row[0]
    node, component, steps = record.repairs[0]
    assert (node, component) == (0, 0) and steps >= 1
    threshold = _FLOOR_PROJECTION_ULPS * EPS * (0.6 + 0.39)
    assert abs(projected[0, 0] - row[0]) <= threshold
    assert record.mask[0].tolist() == [False, False, True]


def test_floor_projection_leaves_large_violations_and_the_dirichlet_node_alone():
    profile = np.asarray([[1.0e-10 - 1.0e-12, 0.2], [0.11, 0.2], [1.0e-10 - 5.0e-18, 0.2]])
    projected, record = _helper(profile, [0.1, 0.2], fixed_node=-1)
    assert_array_equal(projected, profile)
    assert record.mask is None and record.projected_nodes == 0
    assert np.all(record.inventory_change == 0.0)


def test_floor_projection_with_interstitial_component():
    # Component 0 is interstitial (lower bound only); component 1 is the only substitutional one.
    floor = 1.0e-10
    profile = [[1.0e-10 - 3.0e-18, 1.0e-10 - 4.0e-18], [0.03, 0.02], [0.03, 0.02]]
    projected, record = _helper(profile, [0.03, 0.05], mask=(True, False), floor=floor)
    assert projected[0].tolist() == [floor, floor]
    assert composition_bound_violation(projected, np.asarray([True, False]), floor) is None
    assert record.mask[0].tolist() == [True, True, False]


@pytest.mark.parametrize("mask", ((False, False), (True, False), (False, True), (True, True)))
def test_fast_floor_screen_agrees_exactly_with_the_validator(mask):
    """The no-violation fast path must decide exactly as ``composition_bound_violation`` (per row)."""
    rng = np.random.default_rng(7)
    mask = np.asarray(mask)
    floor = 1.0e-7
    rows = []
    for _ in range(4000):
        kind = rng.integers(4)
        if kind == 0:  # an independent component within a few ulps of the floor
            row = [floor * (1.0 + rng.integers(-4, 5) * EPS), rng.uniform(0.1, 0.5)]
        elif kind == 1:  # the dependent component within rounding of the floor
            a = rng.uniform(0.2, 0.8)
            row = [a, (1.0 - floor - a) + rng.integers(-4, 5) * EPS * 0.5]
        elif kind == 2:  # ordinary interior values
            row = rng.uniform(0.0, 0.45, size=2)
        else:  # clearly invalid
            row = [rng.uniform(-1e-3, 0.0), rng.uniform(0.6, 1.2)]
        rows.append(rng.permutation(row) if rng.integers(2) else row)
    rows = np.asarray(rows, dtype=np.float64)
    for row in rows:
        profile = row[np.newaxis, :]
        expected = composition_bound_violation(profile, mask, floor) is None
        assert ternary_shared._profile_within_floor_bounds(profile, ~mask, floor) == expected, row
    # Whole multi-row profiles as well (min/max reductions over nodes).
    for start in range(0, len(rows), 40):
        profile = rows[start:start + 40]
        expected = composition_bound_violation(profile, mask, floor) is None
        assert ternary_shared._profile_within_floor_bounds(profile, ~mask, floor) == expected


def test_shared_empty_projection_record_is_read_only():
    record = ternary_shared._no_floor_projection()
    assert record is ternary_shared._no_floor_projection()
    with pytest.raises(ValueError):
        record.inventory_change[0] = 1.0


# --- Round-off floor projection: model-level regressions ------------------------------


def _without_projection(monkeypatch):
    monkeypatch.setattr(ternary_fdm, "_project_shift_roundoff_to_floor",
                        lambda profile, *args: (np.asarray(profile, dtype=np.float64), ternary_shared._no_floor_projection()))


def _dec_inventory(profile, grid, length):
    """Trapezoidal phase inventory of a float profile in exact arithmetic."""
    with localcontext() as ctx:
        ctx.prec = 60
        g = [Decimal(float(x)) for x in grid]
        total = [Decimal(0), Decimal(0)]
        for i in range(len(g) - 1):
            for k in range(2):
                total[k] += (g[i + 1] - g[i]) * (Decimal(float(profile[i, k])) + Decimal(float(profile[i + 1, k]))) / 2
        return [x * Decimal(float(length)) for x in total]


def _floor_case(kind):
    """(model, p, q, floor) with a right-phase bulk component just above its floor beside a large interface value."""
    if kind == "independent":
        closure = _LinearClosure([0.30, 0.10], [0.32, 0.10], [0.10, 0.15], [0.12, 0.15])
        model = _model(closure, mode="phase_uniform", thermodynamics=_IsotropicThermodynamics())
        floor = model.constraints.minComposition
        p, q = _smooth_profiles(model, lambda u: np.column_stack((0.30 + 0 * u, 0.10 + 0 * u)),
                                lambda v: np.column_stack((1.00000001e-10 + 0 * v, 0.15 + 0 * v)))
    elif kind == "dependent":
        # The bulk dependent component sits exactly on the floor (the floor is set to its float
        # value), so any downward reconstruction round-off of X + Y crosses it. X = 0.6 beside
        # c_X = 0.25 is outside Sterbenz's factor-2 range, so p - c and w + c round.
        y = 1.0 - 1.0e-7 - 0.6
        floor = 1.0 - (0.6 + y)
        closure = _LinearClosure([0.50, 0.30], [0.50, 0.31], [0.25, 0.70], [0.26, 0.69])
        model = _model(closure, mode="phase_uniform", thermodynamics=_IsotropicThermodynamics(), min_composition=floor)
        p, q = _smooth_profiles(model, lambda u: np.column_stack((0.50 + 0 * u, 0.30 + 0 * u)),
                                lambda v: np.column_stack((0.6 + 0 * v, y + 0 * v)))
    else:  # interstitial: Mn (substitutional) and C (interstitial) just above the floor in austenite
        closure = CallableTernaryInterfaceEquilibrium(
            _LinearClosure([0.002, 0.010], [0.002, 0.012], [0.030, 0.050], [0.031, 0.052]).interface_compositions,
            eta_bounds=(0.0, 1.0), composition_basis="u")
        model = _model(closure, elements=FE_C_MN, mode="phase_uniform", thermodynamics=_IsotropicThermodynamics())
        floor = model.constraints.minComposition
        p, q = _smooth_profiles(model, lambda u: np.column_stack((0.002 + 0 * u, 0.010 + 0 * u)),
                                lambda v: np.column_stack((1.00000001e-10 + 0 * v, 1.00000001e-10 + 0 * v)))
    assert model._validate_candidate_profiles((p, q)) is None
    return model, p, q, floor


def _floor_candidate(model, p, q, dt, eta_trial=0.4, factor=1.0 + 1.0e-6):
    s = float(model._s_curr)
    eta_lower, _, eta_span = model._eta_scaling_bounds()
    x_hat = model._interface_physical_to_scaled(s * factor, eta_trial, eta_lower, eta_span)
    fs, _ = model._interface_scaled_to_physical(x_hat, eta_lower, eta_span)
    branch = _select_interface_motion_branch(s, s, fs)
    return model._evaluate_interface_candidate(p, q, s, s, dt, eta_lower, eta_span, model._interface_residual_scale(p, q, s),
                                               p[-1].copy(), q[0].copy(), x_hat, branch)


@pytest.mark.parametrize("kind", ("independent", "dependent", "interstitial"))
def test_floor_projection_keeps_candidates_valid_and_accounts_for_inventory(monkeypatch, kind):
    model, p, q, floor = _floor_case(kind)
    dts = (1.0e-6, 1.0e-3, 1.0, 1.0e3)
    projected = []
    for dt in dts:
        candidate = _floor_candidate(model, p, q, dt)  # raises ValueError if a profile leaves the bounds
        record = candidate.right_projection
        projected.append(record.mask is not None)
        if record.mask is None:
            continue
        # The residual's correction equals the exact inventory change from unprojected to stored profile.
        fs = candidate.future_s
        with monkeypatch.context() as m:
            _without_projection(m)
            unprojected = model._solve_concentration_right_planar(q, float(model._s_curr), fs, dt, candidate.c_right,
                                                                  candidate.D_right, candidate.motion_branch).profile
        exact_change = [a - b for a, b in zip(_dec_inventory(candidate.q_future, model._v_grid, model._R - fs),
                                              _dec_inventory(unprojected, model._v_grid, model._R - fs))]
        np.testing.assert_allclose(record.inventory_change, [float(x) for x in exact_change], rtol=1e-12, atol=0.0)
        assert np.all(record.absolute_change >= np.abs(record.inventory_change))
        change = np.abs(candidate.q_future - unprojected)
        assert change[0].max() == 0.0  # never the Dirichlet node
        threshold = _FLOOR_PROJECTION_ULPS * EPS * (np.abs(candidate.c_right) + np.abs(unprojected - candidate.c_right))
        assert np.all(change <= threshold.sum(axis=1)[:, None])
    assert any(projected), "the case must exercise the projection"

    # Without the projection the deviation-form reconstruction leaves the bounds for at least one dt.
    with monkeypatch.context() as m:
        _without_projection(m)
        failures = 0
        for dt in dts:
            try:
                _floor_candidate(model, p, q, dt)
            except ValueError as exc:
                assert "composition bounds" in str(exc)
                failures += 1
        assert failures > 0


def test_floor_projection_does_not_hide_violations_larger_than_the_threshold():
    model, p, q, floor = _floor_case("independent")
    q = q.copy()
    q[5:, 0] = floor - 1.0e-13  # an invalid old state, far beyond round-off of the shift
    with pytest.raises(ValueError, match="composition bounds"):
        _floor_candidate(model, p, q, 1.0e-3)


def test_floor_projection_keeps_provider_queries_inside_the_bounds():
    closure = _LinearClosure([0.30, 0.10], [0.32, 0.10], [0.10, 0.15], [0.12, 0.15])
    thermo = _IsotropicThermodynamics(D=1.0e-3, slope=1.0e-3)
    model = _model(closure, mode="composition_dependent_implicit", thermodynamics=thermo)
    floor = model.constraints.minComposition
    p, q = _smooth_profiles(model, lambda u: np.column_stack((0.30 + 0 * u, 0.10 + 0 * u)),
                            lambda v: np.column_stack((1.00000001e-10 + 0 * v, 0.15 + 0 * v)))
    thermo.compositions.clear()
    projected = 0
    for dt in (1.0e-6, 1.0e-3, 1.0, 1.0e3):
        projected += _floor_candidate(model, p, q, dt).right_projection.projected_nodes
    queried = np.concatenate(thermo.compositions)
    assert projected > 0
    assert np.all(queried >= floor)
    assert np.all(1.0 - queried.sum(axis=1) >= floor)


# --- Multi-step: far-field precision, bias and conservation --------------------------


def test_far_field_precision_bias_and_conservation_near_the_floor():
    """
    A uniform far field (one component just above the floor, one genuinely dilute) beside large
    interface compositions, marched with a moving interface and changing eta.

    Attainable precision, fixed before measuring: in deviation form each far-field value is
    resolved to about ``eps * (|c| + |w|)`` per step, so the acceptance budget is
    ``16 * eps * (|c| + |w|)`` per step times the number of steps, in absolute terms. It is a
    budget, not a proven bound. Conservation is checked directly as stored minus initial
    inventory; the projection corrections are already in the residual and are not subtracted.
    """
    closure = _substitutional_closure()  # c_right X = 0.30 ... 0.40, c_right Y = 0.15 ... 0.13
    grids = (np.linspace(0.0, 1.0, 21), np.linspace(0.0, 1.0, 41))
    model = _model(closure, mode="phase_uniform", thermodynamics=_IsotropicThermodynamics(D=1.0e-5),
                   grids=grids, tolerance=1.0e-13)
    c_left, c_right = model._interface_compositions(model._eta_curr)
    far_value = np.asarray([1.00000001e-10, 1.0e-4])
    ramp = lambda v: np.clip(1.0 - v / 0.2, 0.0, 1.0)[:, None]
    p, q = _smooth_profiles(model, lambda u: np.broadcast_to(c_left, (len(u), 2)).copy(),
                            lambda v: far_value + (c_right - far_value) * ramp(v))
    s, old_s, eta = float(model._s_curr), float(model._s_curr), float(model._eta_curr)
    initial = model.getTotalInventoryFromState(p, q, s)
    scale = model._interface_residual_scale(p, q, s)
    far = slice(len(q) // 2, None)
    q_far0 = q[far].copy()
    c_max = np.maximum(np.abs(closure.right0), np.abs(closure.right1))
    budget_per_step = _FLOOR_PROJECTION_ULPS * EPS * (c_max + np.abs(q_far0 - closure.right0).max(axis=0) + np.abs(closure.right1 - closure.right0))
    steps, dt, etas, accepted_projections = 40, 1.0e-2, [], 0
    for _ in range(steps):
        result = model._solve_interface_planar(p, q, s, old_s, eta, dt)
        model._accept_step_result(p, q, s, eta, result, dt, 0)
        model._commit_pending_step_projection()  # what postProcess does when it commits the step
        accepted_projections += model._lastStepProjectedNodes
        p, q, s_new, eta = result[0], result[1], result[2], result[3]
        p[-1], q[0] = result[4], result[5]
        old_s, s = s, s_new
        etas.append(eta)
        assert model._validate_candidate_profiles((p, q)) is None
    assert np.ptp(etas) > 1.0e-6  # eta, hence the shift, changes during the run

    drift = q[far] - q_far0
    assert np.all(np.abs(drift) <= steps * budget_per_step), (np.abs(drift).max(axis=0), steps * budget_per_step)
    assert np.all(q[far, 0] >= model.constraints.minComposition)
    assert np.all(q[far, 1] > 0.99e-4)  # the dilute component stays well above its floor, never projected
    stored_change = model.getTotalInventoryFromState(p, q, s) - initial
    assert np.all(np.abs(stored_change) <= steps * model.residualTolerance * scale), (stored_change, scale)
    assert accepted_projections > 0  # the run exercises the projection
    assert model._cumulativeProjectedNodes == accepted_projections
    assert model._cumulativeProjectedSolves >= 1


def test_projection_totals_reach_cumulative_diagnostics_only_when_the_step_is_committed():
    """``getdXdt`` proposes a step; only ``postProcess`` commits it and its projection totals."""
    model, p, q, _floor = _floor_case("independent")
    model._p_curr, model._q_curr = p.copy(), q.copy()
    model._s_old = float(model._s_curr)
    x = model.getCurrentX()
    t = float(model.currentTime)

    dXdt = model.getdXdt(t, x)
    staged = model._pendingStepProjection
    assert staged is not None and staged[2] > 0  # the proposed step projected some nodes
    for _ in range(2):  # re-evaluating the same, uncommitted state must not count again
        dXdt = model.getdXdt(t, x)
        assert model._cumulativeProjectedNodes == 0
        np.testing.assert_array_equal(model._cumulativeProjectionInventoryChange, np.zeros(2))
    staged = model._pendingStepProjection
    dt = model.getDt(dXdt)
    x_new = [np.asarray(x[i]) + np.asarray(dXdt[i]) * dt for i in range(4)]
    model.postProcess(t + dt, x_new)  # also checks that the committed state passes the validator
    assert model._pendingStepProjection is None
    assert model._cumulativeProjectedNodes == staged[2]
    np.testing.assert_array_equal(model._cumulativeProjectionInventoryChange, staged[0])
    np.testing.assert_array_equal(model._cumulativeProjectionAbsoluteChange, staged[1])
    model.postProcess(t + dt, x_new)  # nothing staged: a second commit adds nothing
    assert model._cumulativeProjectedNodes == staged[2]


def _count_projecting_solves(monkeypatch):
    calls = []
    real = ternary_fdm._project_shift_roundoff_to_floor

    def spy(*args):
        profile, record = real(*args)
        calls.append(record.mask is not None)
        return profile, record

    monkeypatch.setattr(ternary_fdm, "_project_shift_roundoff_to_floor", spy)
    return calls


def test_projected_solve_counter_includes_picard_and_failed_candidates(monkeypatch):
    closure = _LinearClosure([0.30, 0.10], [0.32, 0.10], [0.10, 0.15], [0.12, 0.15])
    model = _model(closure, mode="composition_dependent_implicit", thermodynamics=_IsotropicThermodynamics(D=1.0e-3, slope=1.0e-1))
    p, q = _smooth_profiles(model, lambda u: np.column_stack((0.30 + 0 * u, 0.10 + 0 * u)),
                            lambda v: np.column_stack((1.00000001e-10 + 0 * v, 0.15 + 0 * v)))
    calls = _count_projecting_solves(monkeypatch)

    before = model._cumulativeProjectedSolves
    candidate = _floor_candidate(model, p, q, 1.0e-3)
    assert candidate.right_inner_iterations > 1  # Picard used intermediate linear solves
    assert sum(calls) >= 2
    assert model._cumulativeProjectedSolves - before == sum(calls)

    # A candidate that fails after its phase solves (here in validation) still counts them.
    calls.clear()
    before = model._cumulativeProjectedSolves

    def fail(_profiles):
        raise ValueError("forced validation failure")

    monkeypatch.setattr(model, "_validate_candidate_profiles", fail)
    with pytest.raises(ValueError, match="forced validation failure"):
        _floor_candidate(model, p, q, 1.0e-3)
    assert sum(calls) >= 1
    assert model._cumulativeProjectedSolves - before == sum(calls)


# --- Fine interface cells: residual smoothness and a regression case -----------------


def _research_interstitial_helpers():
    """Helpers copied from research ``kawin/tests/test_illingworth_interstitials_ternary.py`` at 2741135."""

    def _closure(basis="u"):
        """NPLE-like linear tie-line family in u-fractions: ferrite (left) / austenite (right)."""

        def tie_line(eta):
            left = np.array([0.002, 0.010 + 0.02 * eta])
            right = np.array([0.030 - 0.01 * eta, 0.020 + 0.03 * eta])
            if basis == "x":
                return left / (1.0 + left[0]), right / (1.0 + right[0])
            return left, right

        return CallableTernaryInterfaceEquilibrium(tie_line, eta_bounds=(0.0, 1.0), composition_basis=basis)

    def _diffusivities(ratio):
        """u-basis (C, Mn) matrices: fast carbon, Mn eigenvalue ``ratio`` times smaller, weak C-Mn coupling."""
        return (
            1.0e-2 * np.array([[1.0, -0.02], [0.0, ratio]]),
            1.0e-3 * np.array([[1.0, -0.05], [0.0, ratio]]),
        )

    def _x_from_u(u):
        u = np.asarray(u, dtype=np.float64)
        return u / (1.0 + u[..., :1])

    return SimpleNamespace(
        S0=0.2 + 1.0e-12,
        DOMAIN_LENGTH=0.4,
        U_LEFT_FAR=np.array([0.002, 0.015]),
        U_RIGHT_FAR=np.array([0.010, 0.015]),
        _closure=_closure,
        _diffusivities=_diffusivities,
        _x_from_u=_x_from_u,
    )


def _fine_grid_interstitial_model(t_end, n_right=401, power=3.5, semi_log_dt=0.02):
    """Synthetic Fe-C-Mn-like case (fast C, slow Mn) with an austenite interface cell of ~1e-10."""
    from moving_boundary.analytic import FixedMatrixTernaryDiffusivity

    it = _research_interstitial_helpers()

    D_left, D_right = it._diffusivities(1.0e-3)
    mesh = CartesianFD1D(["C", "MN"], [0.0, it.DOMAIN_LENGTH], 400)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(it.S0, it._x_from_u(it.U_LEFT_FAR), it._x_from_u(it.U_RIGHT_FAR)), ["C", "MN"])]))
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh, elements=FE_C_MN, phases=["ALPHA", "BETA"],
        thermodynamics=FixedMatrixTernaryDiffusivity({"ALPHA": D_left, "BETA": D_right}, ["ALPHA", "BETA"]),
        temperature=1000.0, interfacePosition=it.S0, interface_equilibrium=it._closure(), initial_eta_bracket=(0.0, 1.0),
        time_step=t_end, dt_mode="semi_log", semiLog_dt=semi_log_dt, semiLogT0=1.0e-6, tolerance=1.0e-11,
        max_iterations=50, max_step_retries=12, terminal_thin_phase_policy="continue", record=True,
        transformed_u_grid=landau_grid(41, "power", phase="left", exponent=1.5),
        transformed_v_grid=landau_grid(n_right, "power", phase="right", exponent=power, min_interval=1.0e-14),
    )


def test_interface_residual_is_smooth_near_a_fine_grid_step():
    # Absolute-composition solves gave scaled carbon-residual noise of 1.7e-10 (detrended) at
    # this step, above the 1e-11 tolerance; deviation form gives ~1e-17 (measured 2026-11-09).
    from kawin.solver import explicitEulerIterator

    model = _fine_grid_interstitial_model(30.0)
    model.interfaceResidualCompositionScale = np.ones(2)  # LEDGERS R13: the threshold below is in this scale's units
    model.solve(30.0, iterator=explicitEulerIterator, minDtFrac=1.0e-14)
    p, q = model._p_curr.copy(), model._q_curr.copy()
    s, old_s, eta = float(model._s_curr), float(model._s_old), float(model._eta_curr)
    dt = 30.0 * (np.exp(0.02) - 1.0)
    eta_lower, _, eta_span = model._eta_scaling_bounds()
    scale = model._interface_residual_scale(p, q, s)
    converged = model._solve_interface_planar(p, q, s, old_s, eta, dt)
    x_star = model._interface_physical_to_scaled(converged[2], converged[3], eta_lower, eta_span)
    ks = np.arange(-5, 6)
    for variable in (0, 1):
        values, branches, projections = [], set(), set()
        for k in ks:
            x_hat = x_star.copy()
            x_hat[variable] += k * 1.0e-12
            fs, _ = model._interface_scaled_to_physical(x_hat, eta_lower, eta_span)
            branch = _select_interface_motion_branch(s, old_s, fs)
            candidate = model._evaluate_interface_candidate(p, q, s, old_s, dt, eta_lower, eta_span, scale,
                                                            p[-1].copy(), q[0].copy(), x_hat, branch)
            values.append(candidate.residual / scale)
            branches.add(branch)
            projections.add(repr([(None if r.mask is None else np.argwhere(r.mask).tolist(), r.repairs)
                                  for r in (candidate.left_projection, candidate.right_projection)]))
        assert len(branches) == 1 and len(projections) == 1  # one motion branch, one projection set (analytic closure)
        values = np.asarray(values)
        for component in range(2):
            fit = np.polyval(np.polyfit(ks, values[:, component], 2), ks)
            assert np.max(np.abs(values[:, component] - fit)) < 1.0e-15


def test_fine_grid_run_needs_no_retries_and_conserves():
    # Absolute-composition solves needed 128 retries here (all after t = 0.1 s) and drifted
    # -3.8e-9 in carbon inventory; deviation form: no retries, drift 9e-11 (measured 2026-11-09).
    from kawin.solver import explicitEulerIterator

    model = _fine_grid_interstitial_model(100.0)
    retries = []
    original = model.postProcess

    def record(time, x):
        out = original(time, x)
        retries.append(model._lastStepRetries)
        return out

    model.postProcess = record
    model.solve(100.0, iterator=explicitEulerIterator, minDtFrac=1.0e-14)
    assert model.currentTime == pytest.approx(100.0)
    assert sum(retries) == 0
    n = model.interfaceData.N
    drift = np.asarray(model.inventoryData._y[n]) - np.asarray(model.inventoryData._y[0])
    assert abs(drift[0]) < 1.0e-9
