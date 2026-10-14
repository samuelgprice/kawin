"""Interstitial (u-fraction) support in the ternary two-phase Illingworth model.

Fe-C-Mn-like systems: carbon (interstitial) diffuses orders of magnitude faster
than manganese. The tie-line closures and constant u-basis matrices make the
planar similarity solution exact, which separates conditioning problems from
thermodynamics.
"""

import os

import numpy as np
import pytest

from examples.ternaryExamples.ternaryTwoPhaseSimilarity import (
    FixedMatrixTernaryDiffusivity,
    compare_model_to_similarity,
    solve_similarity_roots,
)
import kawin.diffusion.MovingBoundaryIllingworthTernaryFDM as ternary_fdm
from kawin.diffusion import CallableTernaryInterfaceEquilibrium, MovingBoundaryIllingworthTernaryFD1DModel
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.TransformedGrids import two_phase_landau_grids
from kawin.solver import explicitEulerIterator

FE_C_MN = ["FE", "C", "MN"]
S0 = 0.2 + 1.0e-12
DOMAIN_LENGTH = 0.4
U_LEFT_FAR = np.array([0.002, 0.015])
U_RIGHT_FAR = np.array([0.010, 0.015])


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


def _model(thermodynamics, closure, *, nodes=21, spacing_ratio=0.1, semi_log_dt=0.05, mode="phase_uniform"):
    mesh = CartesianFD1D(["C", "MN"], [0.0, DOMAIN_LENGTH], 400)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(S0, _x_from_u(U_LEFT_FAR), _x_from_u(U_RIGHT_FAR)), ["C", "MN"])]))
    u_grid, v_grid = two_phase_landau_grids(nodes, nodes, method="linear", spacing_ratio=spacing_ratio)
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=FE_C_MN,
        phases=["ALPHA", "BETA"],
        thermodynamics=thermodynamics,
        temperature=1000.0,
        interfacePosition=S0,
        interface_equilibrium=closure,
        initial_eta_bracket=(0.0, 1.0),
        bulk_diffusivity_mode=mode,
        time_step=1.0,
        dt_mode="semi_log",
        semiLog_dt=semi_log_dt,
        semiLogT0=1.0e-6,
        tolerance=1.0e-11,
        max_iterations=50,
        terminal_thin_phase_policy="continue",
        record=True,
        transformed_u_grid=u_grid,
        transformed_v_grid=v_grid,
        initial_eta_componentwise_scaling=True,
    )


class _SpyTernaryDiffusivity(FixedMatrixTernaryDiffusivity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received = []

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        self.received.append(np.array(x, dtype=np.float64, copy=True))
        return super().getInterdiffusivity(x, T, phase=phase, query_context=query_context, **kwargs)


@pytest.mark.parametrize("closure_basis", ["u", "x"])
@pytest.mark.parametrize("mode", ["phase_uniform", "composition_dependent_lagged"])
def test_ternary_interstitial_plumbing(closure_basis, mode):
    D_left, D_right = _diffusivities(1.0e-2)
    therm = _SpyTernaryDiffusivity({"ALPHA": D_left, "BETA": D_right}, ["ALPHA", "BETA"])
    model = _model(therm, _closure(closure_basis), mode=mode)

    # Initial state is u after construction and after reset().
    expected = np.where(np.ravel(model.mesh.z)[:, None] < S0, U_LEFT_FAR, U_RIGHT_FAR)
    np.testing.assert_allclose(model.data.currentY, expected, rtol=1e-14)
    model.reset()
    np.testing.assert_allclose(model.data.currentY, expected, rtol=1e-14)

    model.solve(0.05, iterator=explicitEulerIterator)
    # Interface compositions are u-fractions whichever basis the closure uses.
    c_left, c_right = model.getInterfaceCompositions()
    exact_left, exact_right = _closure("u").interface_compositions(model.getInterfaceEta())
    np.testing.assert_allclose(c_left, exact_left, rtol=1e-13)
    np.testing.assert_allclose(c_right, exact_right, rtol=1e-13)
    # Provider queries receive mole fractions.
    therm.received.clear()
    model._phase_diffusivity_matrix(c_right, "BETA", model.currentTime, model._s_curr)
    np.testing.assert_allclose(therm.received[-1], _x_from_u(c_right), rtol=1e-14)
    # Outputs are full mole fractions that sum to one.
    for _, comps in model.getPhysicalPhaseProfiles():
        np.testing.assert_allclose(np.sum(comps, axis=1), 1.0, atol=1e-14)
    full = model.getCompositions()
    np.testing.assert_allclose(full[:, 1:], _x_from_u(model.data.y()), rtol=1e-13)
    np.testing.assert_allclose(np.sum(full, axis=1), 1.0, atol=1e-14)
    # New interface diagnostics are populated.
    assert np.all(np.isfinite(model._lastInterfacePeclet))
    assert np.isfinite(model._lastImplicitJacobianCond)
    assert model._lastImplicitResidualComponents.shape == (2,)


def test_ternary_interstitial_bounds_allow_interstitial_above_substitutional_simplex():
    D_left, D_right = _diffusivities(1.0e-2)
    model = _model(FixedMatrixTernaryDiffusivity({"ALPHA": D_left, "BETA": D_right}, ["ALPHA", "BETA"]), _closure())
    assert model._profile_composition_violation(np.array([[1.2, 0.3]])) is None
    assert "dependent" in model._profile_composition_violation(np.array([[0.01, 1.0]]))
    with pytest.raises(ValueError, match="ternary composition bounds"):
        model._validate_composition_vector(np.array([-0.01, 0.3]), "test composition")


def test_ternary_interstitial_rejects_interstitial_dependent_element():
    D_left, D_right = _diffusivities(1.0e-2)
    mesh = CartesianFD1D(["FE", "MN"], [0.0, DOMAIN_LENGTH], 50)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(S0, [0.9, 0.01], [0.9, 0.01]), ["FE", "MN"])]))
    with pytest.raises(ValueError, match="dependent"):
        MovingBoundaryIllingworthTernaryFD1DModel(
            mesh=mesh,
            elements=["C", "FE", "MN"],
            phases=["ALPHA", "BETA"],
            thermodynamics=FixedMatrixTernaryDiffusivity({"ALPHA": D_left, "BETA": D_right}, ["ALPHA", "BETA"]),
            temperature=1000.0,
            interfacePosition=S0,
            interface_equilibrium=_closure(),
            time_step=1.0,
        )


# ratio: (nodes, linear spacing ratio at the interface, semi-log dt)
_DISPARATE_CASES = {
    1.0e-2: (41, 0.1, 0.05),
    1.0e-4: (161, 0.001, 0.0125),
    1.0e-6: (161, 0.001, 0.0125),
}


def _run_disparate_similarity(ratio, nodes, spacing_ratio, semi_log_dt):
    D_left, D_right = _diffusivities(ratio)
    roots = solve_similarity_roots(D_left, D_right, U_LEFT_FAR, U_RIGHT_FAR, _closure(), s0=S0, k_bounds=(1.0e-6, 0.3), n_k=2001)
    assert len(roots) == 1
    model = _model(
        FixedMatrixTernaryDiffusivity({"ALPHA": D_left, "BETA": D_right}, ["ALPHA", "BETA"]),
        _closure(),
        nodes=nodes,
        spacing_ratio=spacing_ratio,
        semi_log_dt=semi_log_dt,
    )
    model.solve(1.0, iterator=explicitEulerIterator)
    return model, roots[0], compare_model_to_similarity(model, roots[0], t_min=0.1)


@pytest.mark.parametrize("ratio", sorted(_DISPARATE_CASES, reverse=True))
def test_ternary_similarity_disparate_eigenvalues(ratio):
    """
    Fast interstitial and slow substitutional solute against the exact similarity solution.

    Graded transformed grids keep the slow component's interface Peclet number
    near or below a few. Point-wise Mn profiles are not compared: the Mn spike
    (width D_Mn / v) is narrower than the remaining interface offset, so only
    the carbon profile is checked. ``solve_similarity_roots`` needs explicit
    ``k_bounds`` because its default ``k`` range scales with the smallest
    eigenvalue.
    """
    model, solution, metrics = _run_disparate_similarity(ratio, *_DISPARATE_CASES[ratio])
    assert metrics["k_rel_error"] < 1.0e-2
    assert metrics["interface_rel_error"] < 1.0e-2
    assert metrics["eta_max_abs_error"] < 1.0e-3
    assert np.max(metrics["profile_linf"][:, 0]) < 1.0e-3
    np.testing.assert_allclose(metrics["inventory_drift"], 0.0, atol=5.0e-9 * DOMAIN_LENGTH)
    assert np.isfinite(model._lastImplicitJacobianCond)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("KAWIN_RUN_SLOW") != "1", reason="set KAWIN_RUN_SLOW=1 to run (several minutes)")
def test_ternary_similarity_ratio_1e8():
    """Eigenvalue ratio 1e-8 (Fe-C-Mn ferrite growth scale); needs a very fine interface cell."""
    model, solution, metrics = _run_disparate_similarity(1.0e-8, 321, 1.0e-5, 0.0125)
    assert metrics["k_rel_error"] < 1.0e-2
    assert metrics["eta_max_abs_error"] < 1.0e-3
    np.testing.assert_allclose(metrics["inventory_drift"], 0.0, atol=5.0e-9 * DOMAIN_LENGTH)


def _branch_record(branch, success, raw_norm, selection_norm, eta=0.25):
    """Synthetic initial-eta branch record in the layout used by ``_select_converged_branch``."""
    zeros = np.zeros(2)
    return (raw_norm, branch, success, 3, 1.0, eta, zeros, zeros, zeros, zeros, 1.0, selection_norm)


def test_initial_eta_prefers_converged_branch_over_smaller_raw_residual():
    """An unconverged branch with a smaller raw residual must not hide a converged branch."""
    converged = _branch_record("positive", True, raw_norm=1e-6, selection_norm=1e-12, eta=0.3)
    unconverged = _branch_record("negative", False, raw_norm=1e-9, selection_norm=1e-3, eta=0.7)
    assert ternary_fdm._select_converged_branch([unconverged, converged]) is converged
    assert ternary_fdm._select_converged_branch([converged, unconverged]) is converged


def test_initial_eta_selects_by_active_norm_and_raises_without_converged_branch():
    small_raw = _branch_record("positive", True, raw_norm=1e-12, selection_norm=1e-11)
    small_scaled = _branch_record("negative", True, raw_norm=1e-8, selection_norm=1e-13)
    assert ternary_fdm._select_converged_branch([small_raw, small_scaled]) is small_scaled
    with pytest.raises(ValueError, match="failed to converge"):
        ternary_fdm._select_converged_branch([_branch_record("positive", False, 1e-15, 1e-15)])
    with pytest.raises(ValueError, match="finite residual"):
        ternary_fdm._select_converged_branch([_branch_record("positive", True, 1e-15, 1e-15, eta=np.nan)])
