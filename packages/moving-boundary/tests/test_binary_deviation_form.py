"""Deviation-form phase solves of the binary Illingworth model (issue R06).

Each phase is solved for ``w = c - c_interface``. That is exact only because every
non-Dirichlet row of the MAP systems sums to its old-volume coefficient; these tests
check that identity on every branch, that a state already at the interface
compositions is preserved exactly, that the interface map is free of the
round-off noise that made the fixed point stall on fine grids (R06), and that a
single step stays within a few ulps of 50-digit arithmetic when the profiles are
far from (or cross) their interface compositions, where the shift gains nothing.

Ported from samuelgprice/kawin 25bf6cd (kawin/tests/test_illingworth_binary_deviation_form.py).
Test bodies are unchanged; imports are the package's, and the research helpers from
kawin/tests/test_diffusion.py (constant/variable providers and the default builder) are
reproduced below. The binary solve's default iterator is the research tests' explicit Euler.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from kawin.diffusion.DiffusionParameters import TemperatureParameters
import moving_boundary.solvers.illingworth.binary as illingworth_module
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from moving_boundary.mesh.TransformedGrids import two_phase_landau_grids
from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel


class ConstantBinaryThermodynamics:
    """Phase-constant diffusivities (research ``kawin/tests/test_diffusion.py``)."""

    def __init__(self, phases, diffusivities, interface_compositions):
        self.phases = phases
        self.diffusivities = diffusivities
        self.interface_compositions = interface_compositions

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        values = np.atleast_1d(T).astype(np.float64)
        return np.squeeze(np.ones(values.shape, dtype=np.float64) * self.diffusivities[phase])


class VariableBinaryThermodynamics(ConstantBinaryThermodynamics):
    """``D = D0 (1 + 0.5 x)`` (research ``kawin/tests/test_diffusion.py``)."""

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        composition = np.asarray(x, dtype=np.float64)
        return np.squeeze(float(self.diffusivities[phase]) * (1.0 + 0.5 * composition))


def _build_illingworth_default_model(therm_class=ConstantBinaryThermodynamics, **kwargs):
    """The research default binary case (s0 = 1, R = 5, 100/100 nodes), with overrides."""
    params = {
        "s0": 1.0, "R": 5.0, "n_alpha": 100, "d_alpha": 1.0e-7, "initial_alpha": 0.8,
        "interface_alpha": 0.6, "n_beta": 100, "d_beta": 1.0e-5, "initial_beta": 0.4,
        "interface_beta": 0.0, "time_step": 0.1, "tolerance": 1.0e-8, "record": True,
        "transformed_u_grid": None, "transformed_v_grid": None, "model_kwargs": {},
    }
    params.update(kwargs)
    mesh = CartesianFD1D(["CR"], [0.0, params["R"]], 501)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(params["s0"], params["initial_alpha"], params["initial_beta"]), "CR")]))
    therm = therm_class(
        phases=["ALPHA", "BETA"],
        diffusivities={"ALPHA": params["d_alpha"], "BETA": params["d_beta"]},
        interface_compositions=(params["interface_alpha"], params["interface_beta"]),
    )
    return MovingBoundaryIllingworthFD1DModel(
        mesh, ["NI", "CR"], ["ALPHA", "BETA"], thermodynamics=therm,
        temperature=TemperatureParameters(1000.0), interfacePosition=params["s0"],
        interface_compositions=(params["interface_alpha"], params["interface_beta"]),
        time_step=params["time_step"], geometry="planar",
        phase_a_nodes=params["n_alpha"], phase_b_nodes=params["n_beta"],
        tolerance=params["tolerance"], record=params["record"],
        transformed_u_grid=params["transformed_u_grid"], transformed_v_grid=params["transformed_v_grid"],
        **params["model_kwargs"],
    )


MODES = ("phase_uniform", "composition_dependent_lagged", "composition_dependent_implicit")


def _capture_systems(monkeypatch):
    """Records every tridiagonal system passed to the solver."""
    systems = []
    real = illingworth_module.solve_illingworth_tridiagonal

    def capture(lo, diag, up, rhs):
        systems.append(tuple(np.array(a, dtype=np.float64) for a in (lo, diag, up, rhs)))
        return real(lo, diag, up, rhs)

    monkeypatch.setattr(illingworth_module, "solve_illingworth_tridiagonal", capture)
    return systems


def _old_volumes(grid, side):
    """Old-volume weights per node: half-cell at the free end, cell widths inside."""
    g = np.asarray(grid, dtype=np.float64)
    volume = np.zeros_like(g)
    volume[1:-1] = (g[2:] - g[:-2]) / 2.0
    if side == "left":
        volume[0] = g[1] / 2.0
    else:
        volume[-1] = (1.0 - g[-2]) / 2.0
    return volume


@pytest.mark.parametrize("direction", (1.0, -1.0))
@pytest.mark.parametrize("face_arrays", (False, True))
def test_rows_sum_to_old_volume_so_deviation_rhs_is_exact(monkeypatch, direction, face_arrays):
    model = _build_illingworth_default_model(n_alpha=30, n_beta=40, d_alpha=1e-3, d_beta=1e-2)
    model.solve(0.3, minDtFrac=1e-14)
    p, q, s = model._p_curr.copy(), model._q_curr.copy(), model._s_curr
    c_left, c_right = model.interfaceCompositions
    future_s, dt = s * (1.0 + 0.02 * direction), 0.07
    D_left = np.linspace(1e-3, 3e-3, len(p) - 1) if face_arrays else 1e-3
    D_right = np.linspace(2e-2, 1e-2, len(q) - 1) if face_arrays else 1e-2
    systems = _capture_systems(monkeypatch)
    model._new_concentration_left_planar(p, s, future_s, dt, D_left)
    model._new_concentration_right_planar(q, s, future_s, dt, D_right)
    (l_lo, l_diag, l_up, l_rhs), (r_lo, r_diag, r_up, r_rhs) = systems

    for lo, diag, up, rhs, grid, side, scale, old, c in (
        (l_lo, l_diag, l_up, l_rhs, model._u_grid, "left", s, p, c_left),
        (r_lo, r_diag, r_up, r_rhs, model._v_grid, "right", model._R - s, q, c_right),
    ):
        rows = slice(0, -1) if side == "left" else slice(1, None)
        volume = _old_volumes(grid, side)[rows]
        magnitude = (np.abs(lo) + np.abs(diag) + np.abs(up))[rows]
        assert_allclose((lo + diag + up)[rows], -scale * volume, rtol=0.0, atol=0.0 + 1e-13 * magnitude.max())
        assert_allclose(rhs[rows], -scale * volume * (old[rows] - c), rtol=1e-14, atol=0.0)
        dirichlet = -1 if side == "left" else 0
        assert diag[dirichlet] == -1.0 and rhs[dirichlet] == 0.0


@pytest.mark.parametrize("mode", MODES)
def test_state_at_interface_compositions_is_preserved_exactly(mode):
    model = _build_illingworth_default_model(
        therm_class=VariableBinaryThermodynamics,
        n_alpha=30, n_beta=40, d_alpha=1e-3, d_beta=1e-2,
        initial_alpha=0.6, interface_alpha=0.6, initial_beta=0.3, interface_beta=0.3,
        time_step=0.05, model_kwargs={"bulk_diffusivity_mode": mode},
    )
    model.setup()
    p0, q0, s0 = model._p_curr.copy(), model._q_curr.copy(), model._s_curr
    assert np.all(p0 == 0.6) and np.all(q0 == 0.3)
    model.solve(0.5, minDtFrac=1e-14)
    assert model.interfaceData.N >= 10
    assert_array_equal(model._p_curr, p0)
    assert_array_equal(model._q_curr, q0)
    assert model._s_curr == s0


# Ni-P transient liquid phase bonding (Illingworth and Golosnoy 2005, Fig. 3), um and s.
NIP = {"c_liquid0": 0.19, "c_solid0": 0.0, "c_liquid": 0.10223, "c_solid": 0.00166,
       "D_liquid": 500.0, "D_solid": 18.0, "s0": 12.5, "R": 3012.5}


def _ni_p_model(tolerance):
    p = NIP
    mesh = CartesianFD1D(["P"], [0.0, p["R"]], 3013)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(p["s0"], p["c_liquid0"], p["c_solid0"]), "P")]))
    u_grid, v_grid = two_phase_landau_grids(81, 321, method="linear",
                                            left_kwargs={"spacing_ratio": 0.2}, right_kwargs={"spacing_ratio": 0.01})
    therm = ConstantBinaryThermodynamics(phases=["LIQUID", "FCC_A1"],
                                         diffusivities={"LIQUID": p["D_liquid"], "FCC_A1": p["D_solid"]},
                                         interface_compositions=(p["c_liquid"], p["c_solid"]))
    return MovingBoundaryIllingworthFD1DModel(
        mesh, ["NI", "P"], ["LIQUID", "FCC_A1"], thermodynamics=therm,
        temperature=TemperatureParameters(1423.15), interfacePosition=p["s0"],
        interface_compositions=(p["c_liquid"], p["c_solid"]),
        time_step=1.0, dt_mode="semi_log", semiLog_dt=0.005, semiLogT0=1.0e-6,
        phase_a_nodes=81, phase_b_nodes=321, transformed_u_grid=u_grid, transformed_v_grid=v_grid,
        tolerance=tolerance, max_iterations=200, record=True, record_pq_data=False,
    )


def test_ni_p_fine_tolerance_completes_and_conserves():
    # Before the deviation form this run stalled at 4,767 s: the interface map carried
    # nanometre round-off noise, ten times the 1e-10 um tolerance (R06).
    model = _ni_p_model(tolerance=1.0e-10)
    model.solve(6.0e3, minDtFrac=1e-14)
    assert model.currentTime == pytest.approx(6.0e3)
    drift = abs(model.getTotalInventory() - model._initialInventory) / abs(model._initialInventory)
    assert drift < 1.0e-10
    assert np.max(model.interfaceData._y[: model.interfaceData.N + 1]) < NIP["s0"] * NIP["c_liquid0"] / NIP["c_liquid"]


def test_interface_map_is_smooth_near_equilibrium():
    # Late in solidification the liquid is uniform at its interface composition. The map
    # G(s_trial) must vary smoothly on a 1e-12 um scale; the absolute-composition solve
    # spread over about 3e-9 um here.
    model = _ni_p_model(tolerance=1.0e-8)
    model.solve(3.0e4, minDtFrac=1e-14)
    p, q, s, old_s = model._p_curr.copy(), model._q_curr.copy(), model._s_curr, model._s_old
    dt = 150.0
    D_left, D_right = model._step_diffusivities(p, q, s, model.currentTime)
    trial = s - 0.034
    values = []
    for k in range(-5, 6):
        left, right = model._solve_bulk_phases(p, q, s, trial + k * 1e-12, dt, D_left, D_right)
        values.append(model._new_interface_planar(left.interface_deviation, right.interface_deviation, s, old_s,
                                                  trial + k * 1e-12, dt, 1, left.interface_D, right.interface_D))
    values = np.asarray(values)
    slope = np.polyfit(np.arange(-5, 6) * 1e-12, values - values[5], 1)[0]
    residual = values - values[5] - slope * np.arange(-5, 6) * 1e-12
    assert abs(slope) < 0.1
    assert np.max(np.abs(residual)) < 1.0e-12


# --- Accuracy far from the interface compositions -------------------------------------
# The deviation form gains digits where a profile is close to its interface composition.
# Elsewhere it must be no worse than the absolute MAP solve: one step is compared with the
# same step in 50-digit arithmetic (absolute form, no rounding) from identical float inputs.

def _dec(values):
    from decimal import Decimal

    return [Decimal(float(x)) for x in np.ravel(values)]


def _thomas_dec(lo, diag, up, rhs):
    from decimal import Decimal

    n = len(diag)
    c, d = [Decimal(0)] * n, [Decimal(0)] * n
    c[0], d[0] = up[0] / diag[0], rhs[0] / diag[0]
    for i in range(1, n):
        den = diag[i] - lo[i] * c[i - 1]
        c[i] = up[i] / den if i < n - 1 else Decimal(0)
        d[i] = (rhs[i] - lo[i] * d[i - 1]) / den
    x = [Decimal(0)] * n
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _face_coefficients_dec(D, n, dt, length):
    """``A`` per face (``n - 1`` faces) for a scalar or face-array diffusivity."""
    D = _dec(np.broadcast_to(np.asarray(D, dtype=np.float64), (n - 1,)))
    return [x * dt / length for x in D]


def _exact_left(model, p, s, fs, dt, D):
    """Absolute-form MAP left solve (both motion branches) in 50-digit arithmetic."""
    from decimal import Decimal

    u, p = _dec(model._u_grid), _dec(p)
    n, B, grow = len(p), fs - s, fs >= s
    A = _face_coefficients_dec(D, n, dt, fs)
    lo, dg, up, r = ([Decimal(0)] * n for _ in range(4))
    if grow:
        dg[0], up[0] = -A[0] / u[1] - fs * u[1] / 2, A[0] / u[1] + B * u[1] / 2
    else:
        dg[0], up[0] = -A[0] / u[1] + B * u[1] / 2 - fs * u[1] / 2, A[0] / u[1]
    r[0] = -p[0] * s * u[1] / 2
    for i in range(1, n - 1):
        a_lo, a_up = A[i - 1] / (u[i] - u[i - 1]), A[i] / (u[i + 1] - u[i])
        ls, rs = u[i] + u[i - 1], u[i + 1] + u[i]
        if grow:
            lo[i], dg[i], up[i] = a_lo, -(a_lo + a_up) - B * ls / 2, a_up + B * rs / 2
        else:
            lo[i], dg[i], up[i] = a_lo - B * ls / 2, -(a_lo + a_up) + B * rs / 2, a_up
        dg[i] -= fs * (rs - ls) / 2
        r[i] = -s * p[i] * (rs - ls) / 2
    dg[-1], r[-1] = Decimal(-1), -_dec([model.interfaceCompositions[0]])[0]
    return _thomas_dec(lo, dg, up, r)


def _exact_right(model, q, s, fs, dt, D):
    """Absolute-form MAP right solve (both motion branches) in 50-digit arithmetic."""
    from decimal import Decimal

    v, q, R = _dec(model._v_grid), _dec(q), _dec([model._R])[0]
    n, B, span, grow = len(q), fs - s, R - fs, fs >= s
    A = _face_coefficients_dec(D, n, dt, R - fs)
    lo, dg, up, r = ([Decimal(0)] * n for _ in range(4))
    dg[0], r[0] = Decimal(-1), -_dec([model.interfaceCompositions[1]])[0]
    for i in range(1, n - 1):
        a_lo, a_up = A[i - 1] / (v[i] - v[i - 1]), A[i] / (v[i + 1] - v[i])
        ls, rs = v[i] + v[i - 1], v[i + 1] + v[i]
        if grow:
            lo[i], dg[i], up[i] = a_lo, -(a_lo + a_up) - B * (1 - ls / 2), a_up + B * (1 - rs / 2)
        else:
            lo[i], dg[i], up[i] = a_lo - B * (1 - ls / 2), -(a_lo + a_up) + B * (1 - rs / 2), a_up
        dg[i] -= span * (rs - ls) / 2
        r[i] = -(R - s) * q[i] * (rs - ls) / 2
    t = v[-2]
    if grow:
        lo[-1], dg[-1] = A[-1] / (1 - t), -A[-1] / (1 - t) - B * (1 - (1 + t) / 2)
    else:
        lo[-1], dg[-1] = A[-1] / (1 - t) - B * (1 - (1 + t) / 2), -A[-1] / (1 - t)
    dg[-1] -= span * (1 - t) / 2
    r[-1] = -q[-1] * (R - s) * (1 - t) / 2
    return _thomas_dec(lo, dg, up, r)


def _exact_interface(model, p_new, q_new, s, fs, dt, DL_int, DR_int):
    """Absolute-form MAP interface update (pass >= 1 branch choice) in 50-digit arithmetic."""
    from decimal import Decimal

    cl, cr = _dec(model.interfaceCompositions)
    u2, v1, R = _dec([model._u_grid[-2], model._v_grid[1], model._R])
    DL, DR = _dec([DL_int, DR_int])
    rhs = ((q_new[1] - cr) / v1 * DR / (R - fs) - (cl - p_new[-2]) / (1 - u2) * DL / fs) * dt
    if fs >= s:
        lhs = cl - q_new[1] * (1 - v1 / 2) - cr * v1 / 2
    else:
        lhs = p_new[-2] * (Decimal("0.5") + u2 / 2) + cl * (Decimal("0.5") - u2 / 2) - cr
    return s + rhs / lhs


# (interface compositions, left profile of u, right profile of v); profiles end at the
# interface composition. "zero_right": c_right = 0 (the shift is a no-op on the right);
# "near_one": c_left near 1 beside bulk near 0; "crossing": both phases cross c_interface.
FAR_STATES = {
    "zero_right": ((0.6, 0.0), lambda u: 0.05 + 0.55 * u**4, lambda v: 0.95 * np.sqrt(v)),
    "near_one": ((0.97, 0.03), lambda u: 0.01 + 0.96 * u**6, lambda v: 0.03 + 0.9 * (1.0 - np.exp(-8.0 * v))),
    "crossing": ((0.4, 0.1), lambda u: 0.4 + 0.3 * np.sin(9.0 * u) * (1.0 - u), lambda v: 0.1 + 0.08 * np.sin(11.0 * v)),
}


@pytest.mark.parametrize("state", sorted(FAR_STATES))
@pytest.mark.parametrize("direction", (1.0, -1.0))
@pytest.mark.parametrize("face_arrays", (False, True))
def test_step_far_from_interface_compositions_matches_exact_arithmetic(state, direction, face_arrays):
    from decimal import localcontext

    (c_left, c_right), left_profile, right_profile = FAR_STATES[state]
    u_grid, v_grid = two_phase_landau_grids(14, 18, method="linear",
                                            left_kwargs={"spacing_ratio": 0.2}, right_kwargs={"spacing_ratio": 0.05})
    model = _build_illingworth_default_model(n_alpha=14, n_beta=18, interface_alpha=c_left, interface_beta=c_right,
                                             transformed_u_grid=u_grid, transformed_v_grid=v_grid)
    model.setup()
    p, q = left_profile(model._u_grid), right_profile(model._v_grid)
    p[-1], q[0] = c_left, c_right
    s, dt = 1.0, 2.0
    fs = s * (1.0 + 0.05 * direction)
    D_left = np.linspace(5e-4, 3e-3, len(p) - 1) if face_arrays else 1e-3
    D_right = np.linspace(2e-2, 5e-3, len(q) - 1) if face_arrays else 1e-2

    p_new, w_left = model._new_concentration_left_planar(p, s, fs, dt, D_left, return_deviation=True)
    q_new, w_right = model._new_concentration_right_planar(q, s, fs, dt, D_right, return_deviation=True)
    DL_int, DR_int = model._left_interface_D(D_left), model._right_interface_D(D_right)
    g_float = model._new_interface_planar(w_left[-2], w_right[1], s, s, fs, dt, 1, DL_int, DR_int)

    with localcontext() as ctx:
        ctx.prec = 50
        s_d, fs_d, dt_d = _dec([s, fs, dt])
        p_exact = _exact_left(model, p, s_d, fs_d, dt_d, D_left)
        q_exact = _exact_right(model, q, s_d, fs_d, dt_d, D_right)
        g_exact = _exact_interface(model, p_exact, q_exact, s_d, fs_d, dt_d, DL_int, DR_int)
        cl_d, cr_d = _dec([c_left, c_right])
        p_err = max(abs(float(a) - float(b)) for a, b in zip(p_new, p_exact))
        q_err = max(abs(float(a) - float(b)) for a, b in zip(q_new, q_exact))
        wl_err = abs(float(w_left[-2]) - float(p_exact[-2] - cl_d))
        wr_err = abs(float(w_right[1]) - float(q_exact[1] - cr_d))
        g_err = abs(g_float - float(g_exact))
        displacement = abs(float(g_exact - s_d))

    # Bounds in ulps of each phase's composition scale; measured (2026-11-09) at most 1.6
    # for profiles and 0.13 for the interface deviations. The interface update was
    # correctly rounded in every case, so it is bounded in ulps of the position itself.
    eps = np.finfo(float).eps
    left_scale = max(np.max(np.abs(p_new)), c_left)
    right_scale = max(np.max(np.abs(q_new)), c_right)
    assert p_err <= 8 * eps * left_scale, p_err / (eps * left_scale)
    assert q_err <= 8 * eps * right_scale, q_err / (eps * right_scale)
    assert wl_err <= 8 * eps * left_scale, wl_err / (eps * left_scale)
    assert wr_err <= 8 * eps * right_scale, wr_err / (eps * right_scale)
    assert g_err <= 4 * eps * abs(g_float), g_err / (eps * abs(g_float))
    assert displacement > 1e-3  # the step moves the interface, so the check is not vacuous
