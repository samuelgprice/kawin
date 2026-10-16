"""
Three-phase ternary Illingworth solver against the exact asymmetric similarity solution.

A middle phase B grows between two semi-infinite outer phases A and C with
different constant, coupled diffusivity matrices, different far fields and two
independent linear tie-line closures, so the two interfaces move at different
parabolic rates (``examples/ternaryExamples/ternaryThreePhaseSimilarity.py``,
derivation in ``ternaryThreePhaseSimilarity.md``). Two cases:

* ``opposite``: the layer grows into both outer phases (``k_ab < 0 < k_bc``);
* ``drift``: both interfaces move right (``0 < k_ab < k_bc``), so A advances
  into B while B grows into C.

The model starts from the exact state at ``T0 > 0`` (a zero-width start is
singular) and is compared over ``[4 T0, T_END]``.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from examples.ternaryExamples.ternaryThreePhaseSimilarity import (
    SimilarityInitialProfile,
    compare_three_phase_model_to_similarity,
    solve_three_phase_similarity_roots,
    tie_line_crossing_etas,
)
from examples.ternaryExamples.ternaryTwoPhaseSimilarity import FixedMatrixTernaryDiffusivity, semi_infinite_validity_time
from kawin.diffusion import CallableTernaryInterfaceEquilibrium, MovingBoundaryIllingworthTernaryThreePhaseFD1DModel
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder
from kawin.diffusion.mesh.TransformedGrids import landau_grid
from kawin.solver import explicitEulerIterator

D = {
    "A": np.array([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]]),
    "B": np.array([[3.0e-4, -0.5e-4], [1.0e-4, 5.0e-4]]),
    "C": np.array([[6.0e-4, 1.0e-4], [-0.5e-4, 2.0e-3]]),
}
CASES = {
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


def _flipped(closure):
    return CallableTernaryInterfaceEquilibrium(lambda eta: tuple(reversed(closure.interface_compositions(eta))), eta_bounds=(0.0, 1.0))


CLOSURE_AB = _linear_closure([0.30, 0.03], [0.34, 0.07], [0.18, 0.10], [0.26, 0.16])
CLOSURE_BC = _linear_closure([0.20, 0.12], [0.24, 0.18], [0.36, 0.04], [0.39, 0.08])


def _solution(case):
    c_a_far, c_c_far = CASES[case]
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


def _solve_model(solution, nodes, semi_log_dt=0.025, **kwargs):
    model = _build_model(solution, nodes, semi_log_dt, **kwargs)
    model.solve(T_END - T0, iterator=explicitEulerIterator)
    return model


@pytest.mark.parametrize("case", sorted(CASES))
def test_three_phase_similarity_reference_self_consistency(case):
    solution = _solution(case)
    c_a_far, c_c_far = CASES[case]
    assert np.max(np.abs(solution.flux_residual())) < 1.0e-14
    assert solution.k_ab < solution.k_bc
    if case == "opposite":
        assert solution.k_ab < 0.0 < solution.k_bc
    else:
        assert 0.0 < solution.k_ab < solution.k_bc

    t = 0.5
    s_ab, s_bc = solution.interface_positions(t)
    (c_a, c_b_ab), (c_b_bc, c_c) = solution.interface_compositions_ab, solution.interface_compositions_bc
    for x, phase, expected in ((s_ab, 0, c_a), (s_ab, 1, c_b_ab), (s_bc, 1, c_b_bc), (s_bc, 2, c_c)):
        np.testing.assert_allclose(solution.profile([x], t, phase=phase)[0], expected, atol=1.0e-13)

    # Diffusion equation in each phase (central differences).
    for x, phase, Dp in ((s_ab - 0.02, 0, D["A"]), (0.5 * (s_ab + s_bc), 1, D["B"]), (s_bc + 0.02, 2, D["C"])):
        h, dt = 1.0e-4, 1.0e-4
        c_t = (solution.profile([x], t + dt, phase)[0] - solution.profile([x], t - dt, phase)[0]) / (2.0 * dt)
        c_xx = (
            solution.profile([x + h], t, phase)[0] - 2.0 * solution.profile([x], t, phase)[0] + solution.profile([x - h], t, phase)[0]
        ) / h**2
        np.testing.assert_allclose(c_t, Dp @ c_xx, rtol=1.0e-4, atol=1.0e-8)

    # Solute conservation against the initial step at s0 (checks both Stefan signs).
    width = 1.0
    for i in range(2):
        left = quad(lambda x: solution.profile([x], t, 0)[0, i], S0 - width, s_ab, epsabs=1.0e-14, limit=200)[0]
        middle = quad(lambda x: solution.profile([x], t, 1)[0, i], s_ab, s_bc, epsabs=1.0e-14)[0]
        right = quad(lambda x: solution.profile([x], t, 2)[0, i], s_bc, S0 + width, epsabs=1.0e-14, limit=200)[0]
        assert abs(left + middle + right - width * (c_a_far[i] + c_c_far[i])) < 1.0e-12

    # Mirror image x -> 2 s0 - x: swap A and C, flip and swap the closures.
    mirrored = solve_three_phase_similarity_roots(
        D["C"], D["B"], D["A"], c_c_far, c_a_far, _flipped(CLOSURE_BC), _flipped(CLOSURE_AB), s0=S0
    )
    match = min(mirrored, key=lambda root: np.max(np.abs(root.ks + solution.ks[::-1])))
    np.testing.assert_allclose(match.ks, -solution.ks[::-1], rtol=1.0e-9)
    np.testing.assert_allclose(match.etas, solution.etas[::-1], atol=1.0e-9)


def test_three_phase_similarity_rejects_crossing_tie_lines():
    crossing = _linear_closure([0.20, 0.12], [0.24, 0.18], [0.36, 0.04], [0.30, 0.08])
    assert tie_line_crossing_etas(crossing).size > 0
    assert tie_line_crossing_etas(CLOSURE_AB).size == 0
    assert tie_line_crossing_etas(CLOSURE_BC).size == 0
    with pytest.raises(ValueError, match="crossing tie lines"):
        solve_three_phase_similarity_roots(D["A"], D["B"], D["C"], *CASES["opposite"], CLOSURE_AB, crossing)


@pytest.mark.parametrize("case", sorted(CASES))
def test_three_phase_model_matches_asymmetric_similarity_solution(case):
    solution = _solution(case)
    s_ab_end, s_bc_end = solution.interface_positions(T_END)
    assert semi_infinite_validity_time(D["A"], s_ab_end) > T_END
    assert semi_infinite_validity_time(D["C"], DOMAIN_LENGTH - s_bc_end) > T_END

    metrics = [compare_three_phase_model_to_similarity(_solve_model(solution, nodes), solution, T0) for nodes in (21, 41, 81)]
    k_models = np.asarray([m["k_model"] for m in metrics])
    k_errors = np.abs(k_models - solution.ks) / np.abs(solution.ks)

    for m in metrics:
        np.testing.assert_allclose(m["inventory_drift"], 0.0, atol=1.0e-9)

    # The scheme converges at first order in the grid spacing, for both interfaces.
    for key in ("eta_max_abs_error", "profile_linf"):
        errors = [np.max(m[key], axis=0) for m in metrics]
        assert np.all(errors[1] < errors[0]) and np.all(errors[2] < errors[1])
    assert np.all(k_errors[1] < k_errors[0]) and np.all(k_errors[2] < k_errors[1])
    orders = np.log2(k_errors[:-1] / k_errors[1:])
    assert np.all((orders > 0.6) & (orders < 1.5))

    # First-order Richardson extrapolation recovers both rates.
    richardson = 2.0 * k_models[2] - k_models[1]
    np.testing.assert_allclose(richardson, solution.ks, rtol=3.0e-3)

    finest = metrics[-1]
    assert np.all(finest["k_rel_error"] < 2.5e-2)
    assert np.all(finest["eta_max_abs_error"] < 1.5e-2)
    assert np.all(finest["profile_linf"] < 1.0e-2)


# ---------------------------------------------------------------------------
# Residual-scaling options with a fast and a slow component
# ---------------------------------------------------------------------------


def _disparate_diffusivities(ratio):
    """Component Y diffuses ``ratio`` times slower than X in every phase (weak X-Y coupling)."""
    return {
        "A": 1.0e-3 * np.array([[1.0, -0.02], [0.0, ratio]]),
        "B": 3.0e-4 * np.array([[1.0, 0.05], [0.0, ratio]]),
        "C": 6.0e-4 * np.array([[1.0, -0.03], [0.0, ratio]]),
    }


def _disparate_solution(ratio):
    diffusivities = _disparate_diffusivities(ratio)
    roots = solve_three_phase_similarity_roots(
        diffusivities["A"], diffusivities["B"], diffusivities["C"], *CASES["opposite"], CLOSURE_AB, CLOSURE_BC,
        s0=S0, k_scale=np.sqrt(3.0e-4),
    )
    assert len(roots) == 1
    return diffusivities, roots[0]


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, (1.0, 0.0)])
def test_three_phase_interface_residual_composition_scale_must_be_positive(scale):
    with pytest.raises(ValueError, match="interface_residual_composition_scale"):
        _build_model(_solution("opposite"), 11, interface_residual_composition_scale=scale).setup()


def test_three_phase_interface_residual_composition_scale_rows():
    solution = _solution("opposite")
    default = _build_model(solution, 21)
    scaled = _build_model(solution, 21, interface_residual_composition_scale=(1.0, 1.0e-3))
    for model in (default, scaled):
        model.setup()
    profiles, interfaces = scaled._profiles_curr, scaled._interfaces_curr
    np.testing.assert_array_equal(default._residual_scale(profiles, interfaces), np.full(4, DOMAIN_LENGTH))
    # Equal molar volumes: the old inventory is the plain inventory, and L = R0.
    inventory = scaled.getTotalInventoryFromState(profiles, interfaces)
    expected = np.maximum(np.abs(inventory), DOMAIN_LENGTH * np.array([1.0, 1.0e-3]))
    np.testing.assert_allclose(scaled._residual_scale(profiles, interfaces), np.tile(expected, 2), rtol=1.0e-14)
    assert expected[1] == pytest.approx(abs(inventory[1]))  # Y is limited by its own inventory


def test_three_phase_unit_composition_scale_matches_default_bitwise():
    """A scale of 1.0 gives max(|inventory|, L) = L, i.e. exactly the default residual scale."""
    solution = _solution("drift")
    histories = []
    for kwargs in ({}, {"interface_residual_composition_scale": 1.0}):
        model = _build_model(solution, 21, semi_log_dt=0.05, **kwargs)
        model.solve(0.05, iterator=explicitEulerIterator)
        n = model.interfaceData.N + 1
        histories.append((model.interfaceData._y[:n].copy(), model.etaData._y[:n].copy()))
    for default, scaled in zip(*histories):
        np.testing.assert_array_equal(default, scaled)


def test_three_phase_componentwise_initial_eta_converges_every_row():
    """
    With a 1e-3 diffusivity ratio the shared max-norm scale is set by the fast
    component; componentwise scaling converges each of the four rows to its own scale.
    """
    diffusivities, solution = _disparate_solution(1.0e-3)
    estimates = []
    for componentwise in (False, True):
        model = _build_model(solution, 41, diffusivities=diffusivities, initial_eta_guess=False, initial_eta_componentwise_scaling=componentwise)
        model.setup()
        estimates.append((model.initialEtaEstimate, model.initialEtaRootRtol))
    (default, _), (componentwise, rtol) = estimates
    assert default.component_scales is None
    assert componentwise.converged and componentwise.component_scales.shape == (4,)
    assert np.all(np.abs(componentwise.residual) <= rtol * componentwise.component_scales)
    np.testing.assert_allclose(componentwise.etas, default.etas, atol=1.0e-8)
    # The default stops at the shared scale, leaving the rows looser than their own scales allow.
    assert np.max(np.abs(default.residual) / componentwise.component_scales) > rtol


def test_three_phase_scaling_options_match_disparate_similarity_solution():
    diffusivities, solution = _disparate_solution(1.0e-2)
    model = _solve_model(
        solution, 81, diffusivities=diffusivities, initial_eta_guess=False,
        interface_residual_composition_scale=(1.0, 1.0e-3), initial_eta_componentwise_scaling=True,
    )
    metrics = compare_three_phase_model_to_similarity(model, solution, T0)
    # The slow component's interface layer (width ~ D_Y / v) is barely resolved
    # here, so the A|B rate converges more slowly than in the equal-D cases
    # (4.9% at 41 nodes, 3.7% at 81), and the A|B eta error stays near 0.022.
    assert np.all(metrics["k_rel_error"] < 5.0e-2)
    assert np.all(metrics["eta_max_abs_error"] < 3.0e-2)
    np.testing.assert_allclose(metrics["inventory_drift"], 0.0, atol=1.0e-9)
