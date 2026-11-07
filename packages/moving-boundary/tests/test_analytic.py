"""Exact solutions, checked on their own and against the solver where bounds exist.

These checks need no private data. The self-consistency tests validate the exact
solutions independently of any solver; the rest port accuracy bounds from the
research test suite at the same settings they were established for.
"""
from types import SimpleNamespace
import math
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad

from moving_boundary.analytic import (
    FixedMatrixTernaryDiffusivity,
    VALIDATION_PARAMS,
    compare_model_to_similarity,
    compare_three_phase_model_to_similarity,
    planar_exact_interface,
    planar_exact_profile,
    recorded_history,
    semi_infinite_validity_time,
    solve_planar_zener_j,
    solve_similarity_roots,
    solve_three_phase_similarity_roots,
    tie_line_crossing_etas,
)
from moving_boundary.equilibrium import CallableTernaryInterfaceEquilibrium


def _linear_closure(left_0, left_1, right_0, right_1):
    left_0, left_1, right_0, right_1 = (np.asarray(v, dtype=np.float64) for v in (left_0, left_1, right_0, right_1))
    return CallableTernaryInterfaceEquilibrium(
        lambda eta: (left_0 + eta * (left_1 - left_0), right_0 + eta * (right_1 - right_0)), eta_bounds=(0.0, 1.0))


# Source: test_illingworth_ternary_similarity.py at 06dc7b4.
D_LEFT = np.array([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]])
D_RIGHT = np.array([[3.0e-4, -0.5e-4], [1.0e-4, 5.0e-4]])
C_LEFT_FAR, C_RIGHT_FAR = np.array([0.38, 0.02]), np.array([0.20, 0.15])
S0, DOMAIN_LENGTH, T_END, T_COMPARE_MIN = 0.2 + 1.0e-12, 0.4, 1.0, 0.1


def _two_phase_closure():
    return _linear_closure([0.30, 0.03], [0.34, 0.07], [0.18, 0.10], [0.26, 0.16])


def _two_phase_solution():
    roots = solve_similarity_roots(D_LEFT, D_RIGHT, C_LEFT_FAR, C_RIGHT_FAR, _two_phase_closure(), s0=S0)
    assert len(roots) == 1
    return roots[0]


def test_analytic_import_does_not_load_kawin():
    code = "import sys, moving_boundary.analytic; assert 'kawin' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_planar_solution_satisfies_its_defining_equation():
    j = solve_planar_zener_j()
    p = VALIDATION_PARAMS
    ratio = (p["c_b"] - p["c_inf"]) / (p["c_b"] - p["c_a"])
    assert 2.0 * j * 0.5 * math.sqrt(math.pi) * math.exp(j * j) * math.erfc(j) == pytest.approx(ratio)
    t = p["t_init"]
    s = float(planar_exact_interface(t, j=j))
    values = planar_exact_profile(np.array([0.0, s * (1.0 + 1.0e-8), p["R"]]), t, j=j)
    assert values[0] == pytest.approx(p["c_a"])
    assert values[1] == pytest.approx(p["c_b"], rel=0.0, abs=1e-8)
    assert values[2] == pytest.approx(p["c_inf"], rel=0.0, abs=1e-12)


def test_two_phase_solution_is_self_consistent():
    solution = _two_phase_solution()
    assert np.linalg.norm(solution.flux_residual()) < 1.0e-12
    t = 0.5
    s = float(solution.interface_position(t))
    c_left, c_right = solution.interface_compositions()
    np.testing.assert_allclose(solution.profile([s], t, phase="left")[0], c_left, atol=1.0e-13)
    np.testing.assert_allclose(solution.profile([s], t, phase="right")[0], c_right, atol=1.0e-13)
    for x, D, phase in ((s - 0.02, D_LEFT, "left"), (s + 0.01, D_RIGHT, "right")):
        h, dt = 1.0e-4, 1.0e-4
        c_t = (solution.profile([x], t + dt, phase)[0] - solution.profile([x], t - dt, phase)[0]) / (2.0 * dt)
        c_xx = (solution.profile([x + h], t, phase)[0] - 2.0 * solution.profile([x], t, phase)[0]
                + solution.profile([x - h], t, phase)[0]) / h**2
        np.testing.assert_allclose(c_t, D @ c_xx, rtol=1.0e-5, atol=1.0e-9)
    width = 20.0 * np.sqrt(max(np.linalg.eigvals(D_LEFT).real.max(), np.linalg.eigvals(D_RIGHT).real.max()) * t)
    for i in range(2):
        left = quad(lambda x: solution.profile([x], t, "left")[0, i] - C_LEFT_FAR[i], s - width, s, epsabs=1.0e-14)[0]
        right = quad(lambda x: solution.profile([x], t, "right")[0, i] - C_RIGHT_FAR[i], s, s + width, epsabs=1.0e-14)[0]
        assert abs(left + right + (C_LEFT_FAR[i] - C_RIGHT_FAR[i]) * (s - S0)) < 1.0e-12


D3 = {"A": np.array([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]]), "B": np.array([[3.0e-4, -0.5e-4], [1.0e-4, 5.0e-4]]),
      "C": np.array([[6.0e-4, 1.0e-4], [-0.5e-4, 2.0e-3]])}
CLOSURE_AB = _linear_closure([0.30, 0.03], [0.34, 0.07], [0.18, 0.10], [0.26, 0.16])
CLOSURE_BC = _linear_closure([0.20, 0.12], [0.24, 0.18], [0.36, 0.04], [0.39, 0.08])


@pytest.mark.parametrize("far", [(np.array([0.28, 0.10]), np.array([0.26, 0.12])), (np.array([0.38, 0.03]), np.array([0.26, 0.12]))],
                         ids=["opposite", "drift"])
def test_three_phase_solution_is_self_consistent(far):
    solution = solve_three_phase_similarity_roots(D3["A"], D3["B"], D3["C"], *far, CLOSURE_AB, CLOSURE_BC, s0=0.3 + 1.0e-12)[-1]
    assert np.max(np.abs(solution.flux_residual())) < 1.0e-14
    assert solution.k_ab < solution.k_bc
    t = 0.5
    s_ab, s_bc = solution.interface_positions(t)
    (c_a, c_b_ab), (c_b_bc, c_c) = solution.interface_compositions_ab, solution.interface_compositions_bc
    for x, phase, expected in ((s_ab, 0, c_a), (s_ab, 1, c_b_ab), (s_bc, 1, c_b_bc), (s_bc, 2, c_c)):
        np.testing.assert_allclose(solution.profile([x], t, phase=phase)[0], expected, atol=1.0e-13)


def test_crossing_tie_lines_are_rejected():
    crossing = _linear_closure([0.20, 0.12], [0.24, 0.18], [0.36, 0.04], [0.30, 0.08])
    assert tie_line_crossing_etas(crossing).size > 0
    assert tie_line_crossing_etas(CLOSURE_AB).size == 0
    with pytest.raises(ValueError, match="crossing tie lines"):
        solve_three_phase_similarity_roots(D3["A"], D3["B"], D3["C"], np.array([0.28, 0.10]), np.array([0.26, 0.12]), CLOSURE_AB, crossing)


def test_validity_estimate_is_the_documented_wall_tail_time():
    assert semi_infinite_validity_time(np.diag([1.0e-2, 1.0e-4]), 0.4) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# State-basis comparison (ledger R05): stub model holding the exact state.
# ---------------------------------------------------------------------------

U_LEFT_FAR, U_RIGHT_FAR = np.array([0.002, 0.015]), np.array([0.010, 0.015])


def _interstitial_solution():
    closure = CallableTernaryInterfaceEquilibrium(
        lambda eta: (np.array([0.002, 0.010 + 0.02 * eta]), np.array([0.030 - 0.01 * eta, 0.020 + 0.03 * eta])),
        eta_bounds=(0.0, 1.0), composition_basis="u")
    d_left = 1.0e-2 * np.array([[1.0, -0.02], [0.0, 1.0e-2]])
    d_right = 1.0e-3 * np.array([[1.0, -0.05], [0.0, 1.0e-2]])
    roots = solve_similarity_roots(d_left, d_right, U_LEFT_FAR, U_RIGHT_FAR, closure, s0=0.2, k_bounds=(1.0e-6, 0.3), n_k=2001)
    assert len(roots) == 1
    return roots[0]


def _history(times, values):
    return SimpleNamespace(N=len(times) - 1, _time=np.asarray(times, dtype=np.float64), _y=np.asarray(values, dtype=np.float64))


class _ExactStateModel:
    """Exposes the exact u state and its mole-fraction view, as the solver does."""

    def __init__(self, solution, times):
        self.solution = solution
        self.grid = np.linspace(0.0, 1.0, 21)
        self.interfaceData = _history(times, solution.interface_position(times))
        self.etaData = _history(times, np.full(len(times), solution.eta))
        self.inventoryData = _history(times, np.zeros((len(times), 2)))

    def _coordinates(self, time):
        s = float(self.solution.interface_position(time))
        return s * self.grid, s + (0.4 - s) * self.grid

    def getTransformedState(self, time):
        left, right = self._coordinates(time)
        return self.solution.profile(left, time, phase="left"), self.solution.profile(right, time, phase="right")

    def getPhysicalPhaseProfiles(self, time):
        left, right = self._coordinates(time)
        full = lambda u: np.column_stack((1.0 - np.sum(u / (1.0 + u[:, :1]), axis=1), u / (1.0 + u[:, :1])))
        p, q = self.getTransformedState(time)
        return (left, full(p)), (right, full(q))


def test_comparison_uses_state_basis_with_interstitials():
    solution = _interstitial_solution()
    model = _ExactStateModel(solution, np.geomspace(0.1, 1.0, 8))
    metrics = compare_model_to_similarity(model, solution)
    assert np.max(metrics["profile_linf"]) < 1.0e-14
    # Subtracting the mole-fraction view would leave u^2 / (1 + u) for carbon.
    time = float(metrics["profile_times"][-1])
    _, (x_right, c_right) = model.getPhysicalPhaseProfiles(time)
    exact = solution.profile(x_right, time, phase="right")
    np.testing.assert_allclose(exact[:, 0] - c_right[:, 1], exact[:, 0] ** 2 / (1.0 + exact[:, 0]), rtol=1.0e-12)
    assert np.max(np.abs(exact[:, 0] - c_right[:, 1])) > 1.0e-4


def test_recorded_history_returns_only_recorded_entries():
    history = SimpleNamespace(N=1, _time=np.array([0.0, 1.0, 9.0]), _y=np.array([2.0, 3.0, 9.0]))
    times, values = recorded_history(history)
    np.testing.assert_array_equal(times, [0.0, 1.0])
    np.testing.assert_array_equal(values, [2.0, 3.0])


# ---------------------------------------------------------------------------
# Accuracy bounds at the settings they were established for.
# ---------------------------------------------------------------------------

def test_two_phase_solver_meets_source_bounds():
    """Source: test_ternary_illingworth_matches_similarity_solution, same settings and window.

    Substitutional, so x equals u and the bounds carry over unchanged.
    """
    from moving_boundary.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
    from moving_boundary.mesh.TransformedGrids import two_phase_landau_grids
    from moving_boundary.solvers import MovingBoundaryIllingworthTernaryFD1DModel

    solution = _two_phase_solution()
    assert semi_infinite_validity_time(D_LEFT, S0) > T_END
    assert semi_infinite_validity_time(D_RIGHT, DOMAIN_LENGTH - S0) > T_END

    def solve(nodes, semi_log_dt):
        mesh = CartesianFD1D(["X", "Y"], [0.0, DOMAIN_LENGTH], 400)
        mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(S0, C_LEFT_FAR, C_RIGHT_FAR), ["X", "Y"])]))
        u_grid, v_grid = two_phase_landau_grids(nodes, nodes, method="linear", spacing_ratio=0.1)
        model = MovingBoundaryIllingworthTernaryFD1DModel(
            mesh=mesh, elements=["Z", "X", "Y"], phases=["ALPHA", "BETA"],
            thermodynamics=FixedMatrixTernaryDiffusivity({"ALPHA": D_LEFT, "BETA": D_RIGHT}, ["ALPHA", "BETA"]),
            temperature=1000.0, interfacePosition=S0, interface_equilibrium=_two_phase_closure(),
            initial_eta_bracket=(0.0, 1.0), time_step=1.0, dt_mode="semi_log", semiLog_dt=semi_log_dt, semiLogT0=1.0e-6,
            tolerance=1.0e-11, max_iterations=50, terminal_thin_phase_policy="raise", record=True, record_pq_data=True,
            transformed_u_grid=u_grid, transformed_v_grid=v_grid)
        model.solve(T_END, minDtFrac=1e-14)
        return compare_model_to_similarity(model, solution, t_min=T_COMPARE_MIN)

    coarse, fine = solve(21, 0.05), solve(41, 0.025)
    for metrics in (coarse, fine):
        assert metrics["k_rel_error"] < 1.0e-2
        assert metrics["interface_rel_error"] < 1.0e-2
        assert metrics["eta_max_abs_error"] < 5.0e-3
        assert np.max(metrics["profile_linf"]) < 1.0e-3
        np.testing.assert_allclose(metrics["inventory_drift"], 0.0, atol=1.0e-9)
    assert fine["k_rel_error"] < coarse["k_rel_error"]
