"""
Ternary Illingworth two-phase solver against the exact planar similarity solution.

Uses a synthetic linear tie-line closure and constant, coupled, non-symmetric
diffusivity matrices on a domain wide enough to stay semi-infinite, so the
analytic solution in ``examples/ternaryExamples/ternaryTwoPhaseSimilarity.py``
is exact for the continuous problem.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from examples.ternaryExamples.ternaryTwoPhaseSimilarity import (
    FixedMatrixTernaryDiffusivity,
    compare_model_to_similarity,
    semi_infinite_validity_time,
    solve_similarity_roots,
)
from kawin.diffusion import CallableTernaryInterfaceEquilibrium, MovingBoundaryIllingworthTernaryFD1DModel
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.TransformedGrids import two_phase_landau_grids
from kawin.solver import explicitEulerIterator

D_LEFT = np.array([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]])
D_RIGHT = np.array([[3.0e-4, -0.5e-4], [1.0e-4, 5.0e-4]])
C_LEFT_FAR = np.array([0.38, 0.02])
C_RIGHT_FAR = np.array([0.20, 0.15])
C_LEFT_0, C_LEFT_1 = np.array([0.30, 0.03]), np.array([0.34, 0.07])
C_RIGHT_0, C_RIGHT_1 = np.array([0.18, 0.10]), np.array([0.26, 0.16])
S0 = 0.2 + 1.0e-12  # off the physical mesh nodes
DOMAIN_LENGTH = 0.4
T_END = 1.0
T_COMPARE_MIN = 0.1


def _closure():
    return CallableTernaryInterfaceEquilibrium(
        lambda eta: (C_LEFT_0 + eta * (C_LEFT_1 - C_LEFT_0), C_RIGHT_0 + eta * (C_RIGHT_1 - C_RIGHT_0)),
        eta_bounds=(0.0, 1.0),
    )


def _solution():
    roots = solve_similarity_roots(D_LEFT, D_RIGHT, C_LEFT_FAR, C_RIGHT_FAR, _closure(), s0=S0)
    assert len(roots) == 1
    return roots[0]


def _solve_model(nodes, semi_log_dt):
    mesh = CartesianFD1D(["X", "Y"], [0.0, DOMAIN_LENGTH], 400)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(S0, C_LEFT_FAR, C_RIGHT_FAR), ["X", "Y"])]))
    u_grid, v_grid = two_phase_landau_grids(nodes, nodes, method="linear", spacing_ratio=0.1)
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=FixedMatrixTernaryDiffusivity({"ALPHA": D_LEFT, "BETA": D_RIGHT}, ["ALPHA", "BETA"]),
        temperature=1000.0,
        interfacePosition=S0,
        interface_equilibrium=_closure(),
        initial_eta_bracket=(0.0, 1.0),
        time_step=1.0,
        dt_mode="semi_log",
        semiLog_dt=semi_log_dt,
        semiLogT0=1.0e-6,
        tolerance=1.0e-11,
        max_iterations=50,
        terminal_thin_phase_policy="continue",
        record=True,
        record_pq_data=True,
        transformed_u_grid=u_grid,
        transformed_v_grid=v_grid,
    )
    model.solve(T_END, iterator=explicitEulerIterator)
    return model


def test_similarity_solution_self_consistency():
    solution = _solution()
    assert np.linalg.norm(solution.flux_residual()) < 1.0e-12

    t = 0.5
    s = float(solution.interface_position(t))
    c_left, c_right = solution.interface_compositions()
    np.testing.assert_allclose(solution.profile([s], t, phase="left")[0], c_left, atol=1.0e-13)
    np.testing.assert_allclose(solution.profile([s], t, phase="right")[0], c_right, atol=1.0e-13)

    # Diffusion equation in each phase (central differences).
    for x, D, phase in ((s - 0.02, D_LEFT, "left"), (s + 0.01, D_RIGHT, "right")):
        h, dt = 1.0e-4, 1.0e-4
        c_t = (solution.profile([x], t + dt, phase)[0] - solution.profile([x], t - dt, phase)[0]) / (2.0 * dt)
        c_xx = (solution.profile([x + h], t, phase)[0] - 2.0 * solution.profile([x], t, phase)[0] + solution.profile([x - h], t, phase)[0]) / h**2
        np.testing.assert_allclose(c_t, D @ c_xx, rtol=1.0e-5, atol=1.0e-9)

    # Solute conservation relative to the initial step (validates the Stefan sign).
    width = 20.0 * np.sqrt(max(np.linalg.eigvals(D_LEFT).real.max(), np.linalg.eigvals(D_RIGHT).real.max()) * t)
    for i in range(2):
        left = quad(lambda x: solution.profile([x], t, "left")[0, i] - C_LEFT_FAR[i], s - width, s, epsabs=1.0e-14)[0]
        right = quad(lambda x: solution.profile([x], t, "right")[0, i] - C_RIGHT_FAR[i], s, s + width, epsabs=1.0e-14)[0]
        excess = left + right + (C_LEFT_FAR[i] - C_RIGHT_FAR[i]) * (s - S0)
        assert abs(excess) < 1.0e-12


def test_ternary_illingworth_matches_similarity_solution():
    solution = _solution()
    assert semi_infinite_validity_time(D_LEFT, S0) > T_END
    assert semi_infinite_validity_time(D_RIGHT, DOMAIN_LENGTH - S0) > T_END

    coarse = compare_model_to_similarity(_solve_model(21, 0.05), solution, t_min=T_COMPARE_MIN)
    fine = compare_model_to_similarity(_solve_model(41, 0.025), solution, t_min=T_COMPARE_MIN)

    for metrics in (coarse, fine):
        assert metrics["k_rel_error"] < 1.0e-2
        assert metrics["interface_rel_error"] < 1.0e-2
        assert metrics["eta_max_abs_error"] < 5.0e-3
        assert np.max(metrics["profile_linf"]) < 1.0e-3
        np.testing.assert_allclose(metrics["inventory_drift"], 0.0, atol=1.0e-9)
    assert fine["k_rel_error"] < coarse["k_rel_error"]
