"""Package-owned C33/F33: two-phase ternary Illingworth solver against the similarity solution.

Frozen at Stage 0 (2026-11-15; evidence and approved revisions R1/R2 in the plan's "Stage 0
closure"). Constants are the package-owned problem of ``tests/test_analytic.py`` (source: research
``test_illingworth_ternary_similarity.py`` at ``06dc7b4``); nothing here reads the research checkout.

Problem: constant 2x2 interdiffusivity matrices per phase (equal to the analytic ones through
``FixedMatrixTernaryDiffusivity``), a linear tie-line closure ``eta -> (c_left, c_right)`` on
``[0, 1]``, far-field compositions as below, elements ``Z, X, Y`` (X and Y independent mole
fractions; no interstitials, so the state basis is the mole-fraction basis), equal constant
molar volumes. The domain ``[0, 0.6]`` stands in for the semi-infinite problem (Stage 0 screening
and matched-grid sensitivity; see the qualifications below).

Start: the exact similarity profile at similarity time ``T0 = 0.1`` (model time 0 corresponds to
``T0``) is supplied on a 2401-node physical mesh. The public ``setup`` then interpolates it within
each phase, *estimates eta itself* (``initial_eta_guess`` only seeds that estimator) and pins the
interface nodes to the estimated tie line, so the started state is not exactly analytic. Under the
approved revision R1 this setup estimate is part of the discretization under test; its errors stay
in every accuracy measurement.

Qualifications (Stage 0): time convergence is approximately first order; spatial errors decrease
with observed orders trending toward, but below, one at the finest tested pair; coarse-level
cancellation and pre-asymptotic flags are documented; profile orders carry an interpolation
qualification; the domain/grid study establishes a sensitivity (R2), not negligible boundary
effects or grid-independent accuracy. The fixed cases are regression gates against the analytic
solution; they do not detect every possible regression.

``build(inputs, case_id)`` resolves every package class through ``inputs`` (the worker's
``tools/definitions/_bindings.Context``, which records the solver provider and enforces the
package mesh), or through ``DirectInputs`` for scratch use outside the acceptance worker.
"""
import importlib

import numpy as np

D_LEFT = np.array([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]])
D_RIGHT = np.array([[3.0e-4, -0.5e-4], [1.0e-4, 5.0e-4]])
C_LEFT_FAR = np.array([0.38, 0.02])
C_RIGHT_FAR = np.array([0.20, 0.15])
CLOSURE = dict(left=([0.30, 0.03], [0.34, 0.07]), right=([0.18, 0.10], [0.26, 0.16]))

DOMAIN_LENGTH = 0.6
S0 = 0.3 + 1.0e-12               # similarity origin, off the physical nodes
T0 = 0.1                         # similarity time of the exact start state
END_TIMES = {"C33": 0.4, "F33": 1.0}
NODES = 31                       # transformed nodes per phase
SPACING_RATIO = 0.1
PHYSICAL_NODES = 2401
SEMI_LOG_DT = 0.05
SEMI_LOG_T0 = 1.0e-3 * T0
TOLERANCE = 1.0e-11
MAX_ITERATIONS = 50
K_FIT_START = 2.0 * T0           # k is fitted over similarity times t >= 2 T0

FROZEN = dict(domain_length=DOMAIN_LENGTH, similarity_origin=S0, start_time=T0, end_times=END_TIMES,
              transformed_nodes=NODES, spacing_ratio=SPACING_RATIO, physical_nodes=PHYSICAL_NODES,
              dt_mode="semi_log", semi_log_dt=SEMI_LOG_DT, semi_log_t0=SEMI_LOG_T0, tolerance=TOLERANCE,
              max_iterations=MAX_ITERATIONS, residual_scale="default", initial_eta_guess="analytic eta (seed only)",
              terminal_thin_phase_policy="raise", k_fit_start=K_FIT_START)


class DirectInputs:
    """Minimal ``inputs`` for scratch use: resolves the same names straight from ``moving_boundary``."""

    family = "ternary"
    _NAMES = {"CartesianFD1D": "moving_boundary.mesh", "ProfileBuilder": "moving_boundary.mesh",
              "two_phase_landau_grids": "moving_boundary.mesh.TransformedGrids",
              "CallableTernaryInterfaceEquilibrium": "moving_boundary.equilibrium"}

    def model(self, name):
        return getattr(importlib.import_module("moving_boundary.solvers"), name)

    def symbol(self, module, name):
        return getattr(importlib.import_module(self._NAMES[name]), name)

    def imported_module(self, module):
        return importlib.import_module(module)


class ExactStart:
    """Callable mesh profile: the exact similarity state at ``t0`` (phase chosen by ``x < s(t0)``)."""

    def __init__(self, solution, t0):
        self.solution, self.t0 = solution, float(t0)

    def __call__(self, z):
        return self.solution.profile(np.asarray(z, dtype=np.float64).reshape(-1), self.t0)


def closure(inputs):
    """The linear tie-line closure on ``eta in [0, 1]``."""
    equilibrium = inputs.symbol("kawin.diffusion.MovingBoundaryEquilibrium", "CallableTernaryInterfaceEquilibrium")
    (l0, l1), (r0, r1) = ((np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
                          for a, b in (CLOSURE["left"], CLOSURE["right"]))
    return equilibrium(lambda eta: (l0 + eta * (l1 - l0), r0 + eta * (r1 - r0)), eta_bounds=(0.0, 1.0))


def solution(inputs):
    """The unique similarity root ``(k, eta)`` for the frozen problem (asserted unique)."""
    analytic = inputs.imported_module("moving_boundary.analytic")
    roots = analytic.solve_similarity_roots(D_LEFT, D_RIGHT, C_LEFT_FAR, C_RIGHT_FAR, closure(inputs), s0=S0)
    if len(roots) != 1:
        raise AssertionError(f"expected one similarity root, found {len(roots)}")
    return roots[0]


def build(inputs, case_id):
    """Return ``(model, duration, analytic)`` for C33 or F33 with the frozen configuration.

    ``analytic = ("two_phase", solution, T0)``: ``acceptance_cases.collect`` evaluates the exact
    solution at similarity time ``model time + T0``. The constructor arguments are the frozen
    Stage 0 configuration (``FROZEN``); the default (R13) residual scale is used.
    """
    if case_id not in END_TIMES:
        raise KeyError(case_id)
    exact = solution(inputs)
    analytic_module = inputs.imported_module("moving_boundary.analytic")
    mesh = inputs.symbol("kawin.diffusion.mesh", "CartesianFD1D")(["X", "Y"], [0.0, DOMAIN_LENGTH], PHYSICAL_NODES)
    mesh.setResponseProfile(inputs.symbol("kawin.diffusion.mesh", "ProfileBuilder")([(ExactStart(exact, T0), ["X", "Y"])]))
    grids = inputs.symbol("kawin.diffusion.mesh.TransformedGrids", "two_phase_landau_grids")
    u_grid, v_grid = grids(NODES, NODES, method="linear", spacing_ratio=SPACING_RATIO)
    duration = END_TIMES[case_id] - T0
    model = inputs.model("MovingBoundaryIllingworthTernaryFD1DModel")(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["ALPHA", "BETA"],
        thermodynamics=analytic_module.FixedMatrixTernaryDiffusivity({"ALPHA": D_LEFT, "BETA": D_RIGHT}, ["ALPHA", "BETA"]),
        temperature=1000.0, interfacePosition=float(exact.interface_position(T0)), interface_equilibrium=closure(inputs),
        initial_eta_guess=float(exact.eta), time_step=duration, dt_mode="semi_log", semiLog_dt=SEMI_LOG_DT,
        semiLogT0=SEMI_LOG_T0, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS,
        terminal_thin_phase_policy="raise", record=True, record_pq_data=True,
        transformed_u_grid=u_grid, transformed_v_grid=v_grid)
    return model, duration, ("two_phase", exact, T0)


def exact_arrays(model, exact, start, interface_values, interface_times, eta_values):
    """Exact-solution error arrays for a solved C33/F33 model (``acceptance_cases.collect``).

    - ``exact/interface_error``: ``s(tm) - s_exact(tm + start)`` at every recorded time (m);
    - ``exact/eta_error``: ``eta(tm) - eta_exact`` at every recorded time (eta is constant exactly);
    - ``exact/profile_error_X``, ``exact/profile_error_Y``: final-time state-basis error per
      component, left-phase nodes then right-phase nodes, each phase against its *own* formula
      (``phase='left'/'right'``) so the composition jump never enters a difference; the interface
      nodes compare with the analytic tie-line values;
    - ``exact/k_rel_error`` (one element): ``(k_fit - k)/k`` with ``k_fit`` the least-squares fit of
      ``s - S0 = k sqrt(t)`` over similarity times ``t >= 2 start``.
    """
    times = np.asarray(interface_times, dtype=np.float64) + start
    s = np.asarray(interface_values, dtype=np.float64)
    arrays = {"exact/interface_error": s - exact.interface_position(times),
              "exact/eta_error": np.asarray(eta_values, dtype=np.float64) - float(exact.eta)}
    final = float(model.currentTime) + start
    (x_left, _), (x_right, _) = model.getPhysicalPhaseProfiles()
    state_left, state_right = model.getTransformedState()
    error = np.vstack([np.asarray(state_left) - exact.profile(x_left, final, phase="left"),
                       np.asarray(state_right) - exact.profile(x_right, final, phase="right")])
    arrays["exact/profile_error_X"] = error[:, 0].copy()
    arrays["exact/profile_error_Y"] = error[:, 1].copy()
    window = times >= 2.0 * start - 1e-15
    if np.count_nonzero(window) < 2:
        raise ValueError("k fit window holds fewer than two recorded times")
    root = np.sqrt(times[window])
    k_fit = float(np.dot(root, s[window] - exact.s0) / np.dot(root, root))
    arrays["exact/k_rel_error"] = np.array([(k_fit - float(exact.k)) / float(exact.k)])
    return arrays
