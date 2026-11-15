"""LEDGERS R13: interface-residual normalization of the two-phase ternary and three-phase models.

Without an explicit ``interface_residual_composition_scale`` the per-step interface residual of
component ``k`` is divided by ``max(|I_k|, L * constraints.minComposition)``: the component's own
(density-weighted) inventory, floored. An explicit scale keeps the pre-R13 expression
``max(|I_k|, L * s_k, 1e-300)`` bit for bit.

Every test states its class in its docstring:

- **Defect**: fails on the pre-R13 code for the stated reason (the old default divided by the
  length term, so the tolerance weakened with dilution).
- **Characterization**: pins values of the new default, so it differs before R13 by construction.
- **Preservation**: holds before and after R13 (identities, explicit scales, the attempted Newton
  update and its fallback, length units, robustness).

The measured before-R13 outcome of each test is recorded in LEDGERS R13. All inputs are open:
fixed diffusivity matrices and linear callable tie-lines. Dilution multiplies component ``k`` by
``g_k`` in far fields and tie-lines and maps ``D -> S D S^-1`` with ``S = diag(g)``, so the exact
``(s, eta)`` path is unchanged and only the inventories shrink.
"""
import numpy as np
import pytest

from kawin.solver import explicitEulerIterator
from moving_boundary import CallableTernaryInterfaceEquilibrium
from moving_boundary.analytic import (
    FixedMatrixTernaryDiffusivity,
    SimilarityInitialProfile,
    solve_three_phase_similarity_roots,
)
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from moving_boundary.mesh.TransformedGrids import landau_grid, two_phase_landau_grids
from moving_boundary.solvers import (
    MovingBoundaryIllingworthTernaryFD1DModel as TwoPhaseModel,
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel as ThreePhaseModel,
)
from moving_boundary.solvers.illingworth import binary, shared

ONE = np.ones(2)
TOL = 1.0e-11
EPS = np.finfo(float).eps


# Fixtures


def _similar(D, g):
    g = np.asarray(g, dtype=np.float64)
    return np.asarray(D, dtype=np.float64) * g[:, None] / g[None, :]


def _linear(left_0, left_1, right_0, right_1, g, basis=None):
    left_0, left_1, right_0, right_1 = (np.asarray(v, dtype=np.float64) * g for v in (left_0, left_1, right_0, right_1))
    kwargs = {} if basis is None else {"composition_basis": basis}
    return CallableTernaryInterfaceEquilibrium(
        lambda eta: (left_0 + eta * (left_1 - left_0), right_0 + eta * (right_1 - right_0)), eta_bounds=(0.0, 1.0), **kwargs
    )


D_LEFT = np.array([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]])
D_RIGHT = np.array([[3.0e-4, -0.5e-4], [1.0e-4, 5.0e-4]])
LENGTH_SCALE = 2.5e-3
TIME_SCALE = 6.25e5


def _two_phase(g=ONE, tolerance=TOL, *, length_unit=None, nodes=21, **overrides):
    """Substitutional similarity couple (mean compositions 0.29 and 0.085 at ``g = 1``).

    ``length_unit`` gives the same couple in SI lengths expressed in that unit (1.0: metres).
    """
    g = np.asarray(g, dtype=np.float64)
    length, factor = (1.0, 1.0) if length_unit is None else (LENGTH_SCALE / length_unit, LENGTH_SCALE**2 / TIME_SCALE / length_unit**2)
    s0 = (0.2 + 1.0e-12) * length
    mesh = CartesianFD1D(["X", "Y"], [0.0, 0.4 * length], 400)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(s0, np.array([0.38, 0.02]) * g, np.array([0.20, 0.15]) * g), ["X", "Y"])]))
    u_grid, v_grid = two_phase_landau_grids(nodes, nodes, method="linear", spacing_ratio=0.1)
    kwargs = dict(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["ALPHA", "BETA"],
        thermodynamics=FixedMatrixTernaryDiffusivity(
            {"ALPHA": _similar(D_LEFT, g) * factor, "BETA": _similar(D_RIGHT, g) * factor}, ["ALPHA", "BETA"]),
        temperature=1000.0, interfacePosition=s0,
        interface_equilibrium=_linear([0.30, 0.03], [0.34, 0.07], [0.18, 0.10], [0.26, 0.16], g),
        initial_eta_bracket=(0.0, 1.0), time_step=1.0, dt_mode="semi_log", semiLog_dt=0.05, semiLogT0=1.0e-6,
        tolerance=tolerance, residual_tolerance=tolerance, max_iterations=50, terminal_thin_phase_policy="raise",
        record=True, record_pq_data=False, transformed_u_grid=u_grid, transformed_v_grid=v_grid,
        initial_eta_componentwise_scaling=True,
    )
    kwargs.update(overrides)
    return TwoPhaseModel(**kwargs), 1.0


def _two_phase_interstitial(g=ONE, tolerance=TOL, *, nodes=21, **overrides):
    """Fe-C-Mn-like couple in u-fractions: a fast dilute interstitial beside a slow substitutional solute."""
    g = np.asarray(g, dtype=np.float64)
    u_left, u_right = np.array([0.002, 0.015]) * g, np.array([0.010, 0.015]) * g

    def x_from_u(u):
        return u / (1.0 + u[..., :1])

    mesh = CartesianFD1D(["C", "MN"], [0.0, 0.4], 400)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.2 + 1.0e-12, x_from_u(u_left), x_from_u(u_right)), ["C", "MN"])]))
    u_grid, v_grid = two_phase_landau_grids(nodes, nodes, method="linear", spacing_ratio=0.1)
    kwargs = dict(
        mesh=mesh, elements=["FE", "C", "MN"], phases=["ALPHA", "BETA"],
        thermodynamics=FixedMatrixTernaryDiffusivity(
            {"ALPHA": _similar(1.0e-2 * np.array([[1.0, -0.02], [0.0, 1.0e-2]]), g),
             "BETA": _similar(1.0e-3 * np.array([[1.0, -0.05], [0.0, 1.0e-2]]), g)}, ["ALPHA", "BETA"]),
        temperature=1000.0, interfacePosition=0.2 + 1.0e-12,
        interface_equilibrium=_linear([0.002, 0.010], [0.002, 0.030], [0.030, 0.020], [0.020, 0.050], g, basis="u"),
        initial_eta_bracket=(0.0, 1.0), time_step=1.0, dt_mode="semi_log", semiLog_dt=0.05, semiLogT0=1.0e-6,
        tolerance=tolerance, residual_tolerance=tolerance, max_iterations=50, record=True, record_pq_data=False,
        transformed_u_grid=u_grid, transformed_v_grid=v_grid, initial_eta_componentwise_scaling=True,
        terminal_thin_phase_policy="raise",
    )
    kwargs.update(overrides)
    return TwoPhaseModel(**kwargs), 1.0


D3 = {
    "A": np.array([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]]),
    "B": np.array([[3.0e-4, -0.5e-4], [1.0e-4, 5.0e-4]]),
    "C": np.array([[6.0e-4, 1.0e-4], [-0.5e-4, 2.0e-3]]),
}
AB = ([0.30, 0.03], [0.34, 0.07], [0.18, 0.10], [0.26, 0.16])
BC = ([0.20, 0.12], [0.24, 0.18], [0.36, 0.04], [0.39, 0.08])
T0, T_END = 0.01, 1.0
_SOLUTION = []


def _similarity_solution():
    if not _SOLUTION:
        roots = solve_three_phase_similarity_roots(
            D3["A"], D3["B"], D3["C"], np.array([0.28, 0.10]), np.array([0.26, 0.12]), _linear(*AB, ONE), _linear(*BC, ONE),
            s0=0.3 + 1.0e-12)
        _SOLUTION.append(roots[-1])
    return _SOLUTION[0]


def _three_phase(g=ONE, tolerance=TOL, *, volumes=None, **overrides):
    """Substitutional three-phase similarity run; ``volumes`` sets unequal phase molar volumes."""
    g = np.asarray(g, dtype=np.float64)
    solution = _similarity_solution()
    base = SimilarityInitialProfile(solution, T0)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 0.6], 6001)
    mesh.setResponseProfile(ProfileBuilder([(lambda z: np.asarray(base(z), dtype=np.float64) * g, ["X", "Y"])]))
    grids = (
        landau_grid(21, "linear", phase="left", spacing_ratio=0.1),
        landau_grid(21, "linear", phase="middle", spacing_ratio=0.2),
        landau_grid(21, "linear", phase="right", spacing_ratio=0.1),
    )
    kwargs = dict(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["A", "B", "C"],
        thermodynamics=FixedMatrixTernaryDiffusivity({k: _similar(v, g) for k, v in D3.items()}, ["A", "B", "C"]),
        temperature=1000.0, interfacePositions=tuple(solution.interface_positions(T0)),
        interface_equilibria=(_linear(*AB, g), _linear(*BC, g)), initial_eta_guess=tuple(solution.etas),
        time_step=T_END - T0, dt_mode="semi_log", semiLog_dt=0.025, semiLogT0=1.0e-3 * T0, transformed_grids=grids,
        tolerance=tolerance, residual_tolerance=tolerance, max_iterations=50, terminal_thin_phase_policy="raise",
        record=True, record_pq_data=False, phase_molar_volumes=volumes,
    )
    kwargs.update(overrides)
    return ThreePhaseModel(**kwargs), T_END - T0


def _three_phase_volumes(g=ONE, tolerance=TOL, **overrides):
    return _three_phase(g, tolerance, volumes=(2.0, 3.0, 1.5), **overrides)


class _UPair:
    eta_bounds = (0.0, 1.0)
    composition_basis = "u"

    def __init__(self, left_base, left_slope, right_base, right_slope, g):
        self.values = tuple(np.asarray(v, dtype=np.float64) * g for v in (left_base, left_slope, right_base, right_slope))

    def interface_compositions(self, eta):
        left_base, left_slope, right_base, right_slope = self.values
        return left_base + float(eta) * left_slope, right_base + float(eta) * right_slope


class _InterstitialThermodynamics:
    def __init__(self, g):
        self.g = g

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        matrix = _similar({"A": 1.0, "B": 0.7, "C": 1.3}[phase] * np.asarray([[1.0, 0.02], [0.0, 0.05]]), self.g)
        if values.ndim == 2:
            return np.broadcast_to(matrix, (values.shape[0], 2, 2)).copy()
        return matrix


def _three_phase_interstitial(g=ONE, tolerance=TOL, **overrides):
    """Three-phase interstitial step couple (u-fractions) with unequal phase molar volumes."""
    g = np.asarray(g, dtype=np.float64)
    interfaces = (0.35, 0.7)
    phase_u = tuple(np.asarray(v) * g for v in ([0.21, 0.09], [0.31, 0.16], [0.20, 0.21]))

    def profile(z):
        x = np.asarray(z, dtype=np.float64).reshape((-1, 1))[:, 0]
        out = np.empty((len(x), 2), dtype=np.float64)
        for mask, u in ((x < interfaces[0], phase_u[0]), ((x >= interfaces[0]) & (x < interfaces[1]), phase_u[1]),
                        (x >= interfaces[1], phase_u[2])):
            out[mask] = u / (1.0 + u[0])
        return out

    mesh = CartesianFD1D(["C", "MN"], [0.0, 1.0], 31)
    mesh.setResponseProfile(ProfileBuilder([(profile, ["C", "MN"])]))
    kwargs = dict(
        mesh=mesh, elements=["FE", "C", "MN"], phases=["A", "B", "C"], thermodynamics=_InterstitialThermodynamics(g),
        temperature=1000.0, interfacePositions=interfaces,
        interface_equilibria=(_UPair([0.20, 0.10], [0.10, 0.0], [0.30, 0.15], [0.10, 0.0], g),
                              _UPair([0.24, 0.15], [0.10, 0.0], [0.18, 0.20], [0.10, 0.0], g)),
        initial_eta_guess=(0.5, 0.5), phase_molar_volumes=(2.0, 3.0, 1.5), phase_nodes=(9, 9, 9), time_step=1e-4,
        record=True, record_pq_data=False, tolerance=tolerance, residual_tolerance=tolerance, max_iterations=50,
        terminal_thin_phase_policy="raise",
    )
    kwargs.update(overrides)
    return ThreePhaseModel(**kwargs), 2.0e-2


BUILDERS = {"two_phase": _two_phase, "two_phase_interstitial": _two_phase_interstitial, "three_phase": _three_phase,
            "three_phase_volumes": _three_phase_volumes, "three_phase_interstitial": _three_phase_interstitial}


# Helpers


def _is_two_phase(model):
    return isinstance(model, TwoPhaseModel)


def _length(model):
    """The length term ``L`` of the residual scale."""
    if _is_two_phase(model):
        return float(model._R)
    return float(model._R0) if model._equalPhaseMolarVolumes else float(model._initialTotalAmount / model._phaseMolarDensities[0])


def _state(model):
    if _is_two_phase(model):
        return model._p_curr, model._q_curr, model._s_curr
    return model._profiles_curr, model._interfaces_curr


def _inventory(model, *state):
    """The inventory the residual rows sum to: transformed (two-phase) or density-weighted (three-phase)."""
    if _is_two_phase(model):
        return np.asarray(model.getTotalInventoryFromState(*state), dtype=np.float64)
    profiles, interfaces = state
    widths = model._phase_widths(interfaces, right_boundary=model._right_boundary_from_interfaces(interfaces))
    inventory = np.zeros(2, dtype=np.float64)
    for density, width, profile, grid in zip(model._normalizedPhaseMolarDensities, widths, profiles, model._grids):
        inventory += density * width * np.trapezoid(np.asarray(profile, dtype=np.float64), grid, axis=0)
    return inventory


def _scale(model, *state):
    """The residual scale per component (the three-phase model repeats it for both interfaces)."""
    if _is_two_phase(model):
        return np.asarray(model._interface_residual_scale(*state), dtype=np.float64)
    full = np.asarray(model._residual_scale(*state), dtype=np.float64)
    assert full.shape == (4,) and np.array_equal(full[:2], full[2:])
    return full[:2]


def _legacy_scale(inventory, length, composition_scale):
    """The pre-R13 expression, written out independently."""
    return np.maximum(np.maximum(np.abs(inventory), length * np.asarray(composition_scale, dtype=np.float64)), 1e-300)


def _track(model):
    """Per accepted interface solve: scale, residual rows (physical), inventory before, iterations, flags."""
    log = dict(scales=[], residuals=[], inventories=[], iterations=[], skipped=[], retries=[])
    two_phase = _is_two_phase(model)
    scale_name = "_interface_residual_scale" if two_phase else "_residual_scale"
    scale, solve = getattr(model, scale_name), model._solve_interface_planar
    last = {}

    def scale_spy(*args):
        last["scale"] = np.array(scale(*args), dtype=np.float64)
        last["inventory"] = _inventory(model, *args)
        return last["scale"].copy()

    def solve_spy(*args, **kwargs):
        out = solve(*args, **kwargs)
        log["scales"].append(last["scale"][:2])
        log["inventories"].append(last["inventory"])
        log["iterations"].append(int(model._lastImplicitIterations))
        log["skipped"].append(bool(model._lastImplicitForcedUpdateSkipped))
        if two_phase:
            log["residuals"].append(np.asarray(model._lastImplicitResidualComponents, dtype=np.float64) * last["scale"])
        else:
            log["residuals"].append(np.asarray(out.residual, dtype=np.float64).copy())
        return out

    setattr(model, scale_name, scale_spy)
    model._solve_interface_planar = solve_spy
    return log


def _recorded(history):
    n = int(history.N) + 1
    return np.asarray(history._y[:n], dtype=np.float64)


def _recorded_inventory(model):
    if _is_two_phase(model):
        return _recorded(model.inventoryData)
    if model._hasExplicitPhaseMolarVolumes:
        return _recorded(model.molarInventoryData)[:, 1:] / model._phaseMolarDensities[0]
    return _recorded(model.inventoryData)


_RUNS = {}


def _run(name, g=(1.0, 1.0), tolerance=TOL):
    """One tracked run per (fixture, dilution, tolerance); cached, since several tests read it."""
    key = (name, tuple(g), tolerance)
    if key not in _RUNS:
        model, duration = BUILDERS[name](np.asarray(g, dtype=np.float64), tolerance)
        log = _track(model)
        if name == "three_phase_interstitial":
            model.solve(duration, minDtFrac=1.0e-10)
        else:
            model.solve(duration, iterator=explicitEulerIterator)
        if _is_two_phase(model):
            position, eta = np.array([model._s_curr / model._R]), np.array([model._eta_curr])
        else:
            position, eta = np.asarray(model._interfaces_curr) / model._R0, np.asarray(model._etas_curr)
        _RUNS[key] = dict(log=log, position=position, eta=eta, inventory=_recorded_inventory(model),
                          steps=len(log["iterations"]), iterations=int(np.sum(log["iterations"])))
    return _RUNS[key]


DILUTIONS = [
    ("two_phase", (1.0e-6, 1.0e-6)), ("two_phase", (1.0e-6, 1.0)), ("two_phase", (1.0, 1.0e-6)),
    ("two_phase_interstitial", (1.0e-6, 1.0e-6)), ("two_phase_interstitial", (1.0e-4, 1.0)),
    ("three_phase", (1.0e-6, 1.0e-6)), ("three_phase", (1.0, 1.0e-4)),
    ("three_phase_volumes", (1.0e-6, 1.0e-6)),
    ("three_phase_interstitial", (1.0e-6, 1.0e-6)),
]


# Defect tests: the default normalization


def test_componentwise_scale_helper_follows_the_binary_rule():
    """Defect (the helper did not exist): per component it equals ``binary._binary_residual_scale``, errors included."""
    helper = shared._componentwise_residual_scale
    table = [(0.3, 2.0, 1e-10), (-0.3, 2.0, 1e-10), (1e-13, 2.0, 1e-10), (0.0, 2.0, 1e-10), (0.0, 2.0, 0.0), (5.0, 2.0, 0.0),
             (1e-320, 1.0, 0.0), (3.0, 2.0, 1.0), (0.0, 1e-200, 1e-100)]
    for inventory, length, floor in table:
        expected = binary._binary_residual_scale(inventory, length, floor)
        got = helper(np.array([inventory, 0.25]), length, floor)
        assert got.shape == (2,) and got.dtype == np.float64
        assert got[0] == expected and got[1] == binary._binary_residual_scale(0.25, length, floor)
    assert helper(np.zeros(2), 3.0, 0.0)[0] == 64.0 * EPS * 3.0
    bad = [((np.nan, 0.1), 1.0, 1e-10, "inventory"), ((0.1, np.inf), 1.0, 1e-10, "inventory"),
           ((0.1, 0.1), 0.0, 1e-10, "domain length"), ((0.1, 0.1), np.nan, 1e-10, "domain length"),
           ((0.1, 0.1), -1.0, 1e-10, "domain length"), ((0.1, 0.1), 1.0, -1e-10, "composition floor"),
           ((0.1, 0.1), 1.0, np.nan, "composition floor"), ((0.0, 0.1), 1e-200, 1e-200, "scale"),
           ((0.1, 0.1), 1e300, 1e300, "scale")]
    for inventory, length, floor, name in bad:
        with pytest.raises(ValueError, match=name):
            helper(np.array(inventory), length, floor)
        with pytest.raises(ValueError):   # the binary function refuses the same input
            for value in inventory:
                binary._binary_residual_scale(value, length, floor)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_default_scale_is_the_component_inventory(name):
    """Defect (the old default was the length term): the default scale is ``max(|I_k|, L * minComposition)``."""
    model, _ = BUILDERS[name]((1.0, 1.0e-3))
    model.setup()
    state = _state(model)
    inventory, length = _inventory(model, *state), _length(model)
    floor = length * model.constraints.minComposition
    assert np.all(np.abs(inventory) > floor) and np.all(np.abs(inventory) < 0.5 * length)
    np.testing.assert_allclose(_scale(model, *state), np.abs(inventory), rtol=4 * EPS, atol=0.0)
    assert _scale(model, *state)[1] < 1.0e-3 * length   # the dilute component is not scaled by the length


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_default_scale_floor_binds_at_min_composition(name):
    """Defect (``minComposition`` was not used): below ``L * minComposition`` the floor is the scale."""
    model, _ = BUILDERS[name]((1.0, 1.0e-3))
    model.setup()
    state = _state(model)
    inventory, length = np.abs(_inventory(model, *state)), _length(model)
    model.constraints.minComposition = 0.5 * (inventory[0] + inventory[1]) / length   # between the two mean compositions
    floor = length * model.constraints.minComposition
    assert inventory[1] < floor < inventory[0]
    np.testing.assert_allclose(_scale(model, *state), [inventory[0], floor], rtol=4 * EPS, atol=0.0)


@pytest.mark.parametrize("name", ["two_phase", "three_phase", "three_phase_volumes"])
def test_default_scale_zero_inventory_and_invalid_values(name):
    """Defect (the old default returned the length): the binary zero rule and loud errors on the default path."""
    model, _ = BUILDERS[name]()
    model.setup()
    state = _state(model)
    if _is_two_phase(model):
        zero = (np.zeros_like(state[0]), np.zeros_like(state[1]), state[2])
        broken = (np.full_like(state[0], np.nan), state[1], state[2])
    else:
        zero = (tuple(np.zeros_like(profile) for profile in state[0]), state[1])
        broken = ((np.full_like(state[0][0], np.nan),) + tuple(state[0][1:]), state[1])
    length = _length(model)
    np.testing.assert_array_equal(_scale(model, *zero), np.full(2, length * model.constraints.minComposition))
    model.constraints.minComposition = 0.0
    np.testing.assert_array_equal(_scale(model, *zero), np.full(2, 64.0 * EPS * length))
    np.testing.assert_allclose(_scale(model, *state), np.abs(_inventory(model, *state)), rtol=4 * EPS, atol=0.0)
    # The two-phase inventory integral refuses a non-finite profile itself; the three-phase scale names the inventory.
    with pytest.raises(ValueError, match="finite"):
        _scale(model, *broken)
    for value in (np.nan, -1.0e-10, np.inf):
        model.constraints.minComposition = value
        with pytest.raises(ValueError, match="composition floor"):
            _scale(model, *state)


@pytest.mark.parametrize("name, g", DILUTIONS)
def test_default_tolerance_is_dilution_invariant(name, g):
    """Defect (old tolerance behaved as ``tol / g``): iterations, interface positions and eta do not depend on dilution."""
    base, dilute = _run(name), _run(name, g)
    assert dilute["steps"] == base["steps"]
    assert dilute["log"]["iterations"] == base["log"]["iterations"]
    np.testing.assert_allclose(dilute["position"], base["position"], rtol=1.0e-10, atol=0.0)
    np.testing.assert_allclose(dilute["eta"], base["eta"], rtol=1.0e-10, atol=1.0e-13)
    assert not any(dilute["log"]["skipped"]) and not any(base["log"]["skipped"])


@pytest.mark.parametrize("name, g", [(name, (1.0, 1.0)) for name in sorted(BUILDERS)] + DILUTIONS)
def test_per_step_inventory_change_is_bounded_relative_to_the_component_inventory(name, g):
    """Defect (old bound was relative to the length): rows ``<= tol * scale_k``; a component's total change
    ``<= tol * scale_k`` (two-phase) or ``2 * tol * scale_k`` (three-phase), with ``scale_k = |I_k|`` here."""
    run = _run(name, g)
    scales, rows, before = (np.asarray(run["log"][key]) for key in ("scales", "residuals", "inventories"))
    np.testing.assert_allclose(scales, np.abs(before), rtol=4 * EPS, atol=0.0)   # the floor never binds in these runs
    roundoff = 32 * EPS * np.abs(before)
    change = np.diff(run["inventory"], axis=0)
    assert len(change) == len(rows) == run["steps"]
    if rows.shape[1] == 2:
        assert np.all(np.abs(rows) <= TOL * scales)
        assert np.all(np.abs(change) <= TOL * scales + roundoff)
    else:
        assert np.all(np.abs(rows[:, :2]) <= TOL * scales) and np.all(np.abs(rows[:, 2:]) <= TOL * scales)
        assert np.all(np.abs(change) <= 2 * TOL * scales + roundoff)


# Defect tests: validation timing


@pytest.mark.parametrize("builder", [_two_phase, _three_phase])
@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf, (1.0, 0.0)])
def test_invalid_explicit_scale_is_rejected_before_the_host_constructor(builder, scale):
    """Defect (was checked only by the configuration validator, after the host constructor ran)."""
    with pytest.raises(ValueError, match="interface_residual_composition_scale"):
        builder(mesh="not a mesh", interface_residual_composition_scale=scale)


@pytest.mark.parametrize("builder", [_two_phase, _three_phase])
@pytest.mark.parametrize("scale", [np.array([1.0, -1.0]), np.array([1.0, np.nan]), np.ones(3), 0.0])
def test_invalid_explicit_scale_set_later_is_rejected_before_setup(builder, scale):
    """Defect (``solve`` reached ``setup`` first; a wrong shape was not checked at all)."""
    model, duration = builder()
    calls = []
    setup = model.setup
    model.setup = lambda: calls.append("setup") or setup()
    model.interfaceResidualCompositionScale = scale
    with pytest.raises(ValueError, match="interface_residual_composition_scale"):
        model.solve(duration, iterator=explicitEulerIterator)
    assert calls == [] and model.currentTime == 0.0


@pytest.mark.parametrize("builder", [_two_phase, _three_phase])
@pytest.mark.parametrize("value", [np.nan, -1.0e-10, np.inf])
def test_invalid_min_composition_is_rejected_before_setup_on_the_default_path(builder, value):
    """Defect (``minComposition`` was not part of the residual scale, so nothing checked it here)."""
    model, duration = builder()
    calls = []
    setup = model.setup
    model.setup = lambda: calls.append("setup") or setup()
    model.constraints.minComposition = value
    with pytest.raises(ValueError, match="minComposition"):
        model.solve(duration, iterator=explicitEulerIterator)
    assert calls == []


# Characterization of the new default


def test_two_phase_interface_solve_diagnostics_on_the_default_scale():
    """Characterization: the counts of the superseded research test are unchanged; the scaled residual is the
    same physical defect divided by the component inventory instead of the domain length."""
    from test_illingworth_ternary import _make_length_scaled_illingworth_model   # the ported research builder

    model = _make_length_scaled_illingworth_model(1.0)
    model.setup()
    x = model.getCurrentX()

    model.getdXdt(model.currentTime, x)

    assert model._lastImplicitConverged is True
    assert model._lastImplicitIterations == 2
    assert model._lastImplicitCandidateEvaluations == 4
    assert model._lastImplicitFunctionEvaluations == model._lastImplicitCandidateEvaluations
    assert model._lastImplicitJacobianEvaluations == 1
    assert model._lastImplicitMotionBranch == "positive"
    assert model._lastImplicitFailureReason is None
    assert np.isclose(model._lastImplicitResidual, 2.8133097385190837e-14, rtol=1.0e-6, atol=0.0)
    scale = model._interface_residual_scale(x[0], x[1], x[2])
    np.testing.assert_allclose(scale, np.abs(model.getTotalInventoryFromState(x[0], x[1], x[2])), rtol=4 * EPS)


# Preservation tests


def test_superseded_research_diagnostics_test_holds_with_the_unit_scale():
    """Preservation: research ``test_ternary_interface_success_diagnostics_count_candidate_and_jacobian_work``,
    assertions verbatim, with the pre-R13 default requested explicitly (scale 1.0)."""
    from test_illingworth_ternary import _make_length_scaled_illingworth_model

    model = _make_length_scaled_illingworth_model(1.0)
    model.interfaceResidualCompositionScale = np.ones(2)
    model.setup()
    x = model.getCurrentX()

    model.getdXdt(model.currentTime, x)

    assert model._lastImplicitConverged is True
    assert model._lastImplicitIterations == 2
    assert model._lastImplicitCandidateEvaluations == 4
    assert model._lastImplicitFunctionEvaluations == model._lastImplicitCandidateEvaluations
    assert model._lastImplicitJacobianEvaluations == 1
    assert model._lastImplicitMotionBranch == "positive"
    assert model._lastImplicitFailureReason is None
    assert np.isclose(model._lastImplicitResidual, 4.494315292342209e-15, rtol=0.0, atol=1.0e-21)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_residual_rows_equal_the_step_inventory_change(name):
    """Preservation: ``dI_k = r_k`` (two-phase) and ``dI~_k = r_ab,k + r_bc,k`` (three-phase) up to round-off."""
    run = _run(name)
    rows, before = np.asarray(run["log"]["residuals"]), np.asarray(run["log"]["inventories"])
    total = rows if rows.shape[1] == 2 else rows[:, :2] + rows[:, 2:]
    change = np.diff(run["inventory"], axis=0)
    np.testing.assert_allclose(run["inventory"][:-1], before, rtol=1.0e-13, atol=0.0)
    assert np.all(np.abs(change - total) <= 32 * EPS * np.abs(before))


SCALES = [0.5, (1.0, 1.0e-3), 1.0, (1.0, 1.0), 1.0e-200, (1.0e-310, 1.0), 1.0e-310]


@pytest.mark.parametrize("name", ["two_phase", "two_phase_interstitial", "three_phase", "three_phase_volumes",
                                  "three_phase_interstitial"])
@pytest.mark.parametrize("scale", SCALES)
def test_explicit_scale_keeps_the_legacy_expression_bit_for_bit(name, scale):
    """Preservation: ``max(|I_k|, L * s_k, 1e-300)``, for ordinary and zero inventories and very small scales."""
    model, _ = BUILDERS[name](interface_residual_composition_scale=scale)
    model.setup()
    state = _state(model)
    length = _length(model)
    values = np.broadcast_to(np.asarray(scale, dtype=np.float64), (2,))
    np.testing.assert_array_equal(_scale(model, *state), _legacy_scale(_inventory(model, *state), length, values))
    if _is_two_phase(model):
        zero = (np.zeros_like(state[0]), np.zeros_like(state[1]), state[2])
    else:
        zero = (tuple(np.zeros_like(profile) for profile in state[0]), state[1])
    expected = np.where(length * values < 1e-300, 1e-300, length * values)   # the legacy lower bound, else L * s_k
    np.testing.assert_array_equal(_scale(model, *zero), expected)
    model.constraints.minComposition = np.nan   # not read on the explicit path
    np.testing.assert_array_equal(_scale(model, *zero), expected)


@pytest.mark.parametrize("name", ["two_phase", "three_phase", "three_phase_volumes"])
def test_unit_scale_is_the_pre_r13_default_scale(name):
    """Preservation: ``1.0`` gives the old default (the length term) for inventories below the length."""
    model, _ = BUILDERS[name](interface_residual_composition_scale=1.0)
    model.setup()
    np.testing.assert_array_equal(_scale(model, *_state(model)), np.full(2, _length(model)))


# Final interface positions and eta of the pre-R13 default runs (tolerance 1e-11), recorded on the pre-R13 code.
PRE_R13_DEFAULT = {
    "two_phase": dict(iterations=606, position=[0.5190558774522362], eta=[0.6519670392029407]),
    "three_phase": dict(iterations=1095, position=[0.4539755236005258, 0.5887326585500066],
                        eta=[0.6821314793336524, 0.6375359443704468]),
}


@pytest.mark.parametrize("name", sorted(PRE_R13_DEFAULT))
def test_unit_scale_reproduces_the_pre_r13_default_run(name):
    """Preservation: an explicit ``1.0`` (scalar or per component) reproduces the recorded pre-R13 default run."""
    recorded = PRE_R13_DEFAULT[name]
    results = []
    for scale in (1.0, (1.0, 1.0)):
        model, duration = BUILDERS[name](interface_residual_composition_scale=scale)
        log = _track(model)
        model.solve(duration, iterator=explicitEulerIterator)
        if _is_two_phase(model):
            position, eta = np.array([model._s_curr / model._R]), np.array([model._eta_curr])
        else:
            position, eta = np.asarray(model._interfaces_curr) / model._R0, np.asarray(model._etas_curr)
        assert int(np.sum(log["iterations"])) == recorded["iterations"]
        np.testing.assert_allclose(position, recorded["position"], rtol=1.0e-13, atol=0.0)
        np.testing.assert_allclose(eta, recorded["eta"], rtol=1.0e-13, atol=0.0)
        results.append((position, eta, _recorded_inventory(model)))
    for scalar, vector in zip(*results):   # scalar and per-component forms are the same run bit for bit
        np.testing.assert_array_equal(scalar, vector)


def test_default_tolerance_does_not_depend_on_the_length_unit():
    """Preservation: the same couple in metres and in micrometres takes the same Newton path."""
    results = []
    for length_unit in (1.0, 1.0e-6):
        model, duration = _two_phase(length_unit=length_unit)
        log = _track(model)
        model.solve(duration, iterator=explicitEulerIterator)
        results.append((log["iterations"], model._s_curr / model._R, model._eta_curr))
    (metre_iterations, metre_position, metre_eta), (micro_iterations, micro_position, micro_eta) = results
    assert len(metre_iterations) > 200 and metre_iterations == micro_iterations
    np.testing.assert_allclose(micro_position, metre_position, rtol=1.0e-10, atol=0.0)
    np.testing.assert_allclose(micro_eta, metre_eta, rtol=1.0e-10, atol=0.0)


def _block_updates(model, error):
    """Lets only the first candidate of each interface solve be evaluated; later ones raise ``error``."""
    evaluate, solve = model._evaluate_interface_candidate, model._solve_interface_planar
    calls = {"count": 0}

    def blocked(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] > 1:
            raise error
        return evaluate(*args, **kwargs)

    def solve_spy(*args, **kwargs):
        calls["count"] = 0
        return solve(*args, **kwargs)

    model._evaluate_interface_candidate = blocked
    model._solve_interface_planar = solve_spy


@pytest.mark.parametrize("builder, error", [
    (_two_phase, RuntimeError("blocked update")),
    (_three_phase, ValueError("Three-phase trial geometry blocked for the test")),
])
def test_attempted_newton_update_and_its_fallback_are_unchanged(builder, error):
    """Preservation: a start that meets the tolerance is accepted only after an attempted update; when the
    update cannot be carried out the start is accepted with ``_lastImplicitForcedUpdateSkipped`` set."""
    model, _ = builder()
    model.setup()
    model.residualTolerance = 1.0e6   # every start meets the tolerance
    model.getdXdt(model.currentTime, model.getCurrentX())
    assert model._lastImplicitConverged is True and model._lastImplicitForcedUpdateSkipped is False
    assert model._lastImplicitIterations == 2   # the start, then the candidate after one Newton update

    blocked, _ = builder()
    blocked.setup()
    blocked.residualTolerance = 1.0e6
    _block_updates(blocked, error)
    blocked.getdXdt(blocked.currentTime, blocked.getCurrentX())
    assert blocked._lastImplicitConverged is True and blocked._lastImplicitForcedUpdateSkipped is True

    strict, _ = builder()
    strict.setup()
    _block_updates(strict, error)   # at the real tolerance the start does not converge, so nothing is accepted
    with pytest.raises((RuntimeError, ValueError)):
        strict.getdXdt(strict.currentTime, strict.getCurrentX())
    assert strict._lastImplicitForcedUpdateSkipped is False


_FINE = {}


def _fine_grid_run():
    """The 201-node interstitial couple at tolerance 1e-13 (one run for the two tests below)."""
    if not _FINE:
        model, duration = _two_phase_interstitial(tolerance=1.0e-13, nodes=201)
        log = _track(model)
        retries = []
        solve = model._solve_interface_planar

        def solve_spy(*args, **kwargs):
            out = solve(*args, **kwargs)
            retries.append(int(model._lastStepRetries))
            return out

        model._solve_interface_planar = solve_spy
        model.solve(duration, iterator=explicitEulerIterator)
        _FINE.update(log=log, retries=retries + [int(model._lastStepRetries)], inventory=_recorded_inventory(model))
    return _FINE


def test_fine_grid_interstitial_run_needs_no_retry_on_the_default_scale():
    """Preservation (robustness): 201-node grids at a tight tolerance converge on every step without a retry."""
    run = _fine_grid_run()
    assert len(run["log"]["iterations"]) > 200 and not any(run["log"]["skipped"])
    assert max(run["retries"]) == 0


def test_fine_grid_interstitial_run_drift_is_within_the_accumulated_bound():
    """Defect (old bound was relative to the length): total drift ``<= N * tol * |I_k|`` plus round-off."""
    run = _fine_grid_run()
    inventory, steps = run["inventory"], len(run["log"]["iterations"])
    assert np.all(np.abs(inventory[-1] - inventory[0]) <= steps * (1.0e-13 + 32 * EPS) * np.abs(inventory[0]))


# Explicit scales assigned after construction (LEDGERS R13 follow-up)


def _one_step(model):
    """One interface solve from the initial state; returns the scale used and the resulting state as arrays."""
    model.setup()
    state = _state(model)
    scale = _scale(model, *state)
    model.getdXdt(model.currentTime, model.getCurrentX())
    return scale, float(model._lastImplicitResidual), int(model._lastImplicitIterations)


@pytest.mark.parametrize("builder", [_two_phase, _two_phase_interstitial, _three_phase, _three_phase_volumes])
@pytest.mark.parametrize("assigned", [1.0, 0.5, [1.0, 1.0e-3], (1.0, 1.0e-3), np.array([1.0, 1.0e-3]), np.float64(0.5), [1],
                                      1.0e-310, (1.0e-310, 1.0)],
                         ids=["float", "float-half", "list", "tuple", "array", "numpy-scalar", "one-item-list", "subnormal",
                              "subnormal-tuple"])
def test_valid_explicit_scale_assigned_after_construction_equals_the_constructor_argument(builder, assigned):
    """Preservation of the explicit expression for every valid form: a scalar, list, tuple or array assigned to
    ``interfaceResidualCompositionScale`` gives the scale and the step of the same value passed to the constructor."""
    reference, _ = builder(interface_residual_composition_scale=assigned)
    expected = _one_step(reference)
    model, _ = builder()
    model.interfaceResidualCompositionScale = assigned
    got = _one_step(model)
    np.testing.assert_array_equal(got[0], expected[0])
    assert got[1:] == expected[1:]
    values = np.broadcast_to(np.asarray(assigned, dtype=np.float64), (2,))
    state = _state(model)
    np.testing.assert_array_equal(_scale(model, *state), _legacy_scale(_inventory(model, *state), _length(model), values))
    assert model.interfaceResidualCompositionScale is assigned   # validation and use never replace the attribute


@pytest.mark.parametrize("builder", [_two_phase, _three_phase])
@pytest.mark.parametrize("assigned", [[1.0, -1.0], (1.0, 0.0), [1.0, np.nan], (np.inf, 1.0), [1.0, 1.0, 1.0], (), [],
                                      [[1.0, 1.0], [1.0, 1.0]], "large", [1.0, None], -0.5],
                         ids=["negative-list", "zero-tuple", "nan-list", "inf-tuple", "three-values", "empty-tuple", "empty-list",
                              "matrix", "string", "none-item", "negative-float"])
def test_invalid_explicit_scale_forms_assigned_later_are_rejected_before_setup(builder, assigned):
    """An invalid list, tuple, shape or value is refused by ``solve`` before ``setup`` and before any step."""
    model, duration = builder()
    calls = []
    setup = model.setup
    model.setup = lambda: calls.append("setup") or setup()
    model.interfaceResidualCompositionScale = assigned
    with pytest.raises(ValueError, match="interface_residual_composition_scale"):
        model.solve(duration, iterator=explicitEulerIterator)
    assert calls == [] and model.currentTime == 0.0


# Acceptance case C32 on the new default (LEDGERS R13: accepted behavior change, user decision 2026-11-15)


def _c32_run(scale=None):
    """The C32 configuration (``tools/acceptance_cases.py``): one Newton iteration, one normal retry, terminal retries.

    With ``maxIterations = 1`` a step is accepted only when its unmodified start already meets
    ``residualTolerance * scale`` (through the forced-update fallback), so the timestep is halved until it does.
    Returns the model and the ``(dt, accepted, scaled residual)`` record of every interface solve.
    """
    from test_illingworth_ternary_three_phase import _make_nonstationary_unequal_volume_model   # the ported research builder

    model = _make_nonstationary_unequal_volume_model()
    model.terminalThinPhasePolicy = "continue"   # what the materialized definition injects
    model.interfaceResidualCompositionScale = scale
    model.setup()
    model.maxIterations = 1
    model.maxStepRetries = 1
    model.terminalThinPhaseWidth = .31
    model.terminalThinPhaseExtraRetries = 25
    attempts = []
    solve = model._solve_interface_planar

    def solve_spy(profiles, interfaces, etas, dt):
        try:
            out = solve(profiles, interfaces, etas, dt)
        except RuntimeError:
            attempts.append((float(dt), False, float(model._lastImplicitResidual)))
            raise
        attempts.append((float(dt), True, float(model._lastImplicitResidual)))
        return out

    model._solve_interface_planar = solve_spy
    return model, attempts


@pytest.mark.parametrize("scale, halvings", [(None, 15), (1.0, 13)], ids=["new-default", "earlier-default-explicit"])
def test_c32_terminal_thin_phase_run_halves_until_the_start_meets_the_tolerance(scale, halvings):
    """Characterization (new default: 15 halvings) and preservation (explicit 1.0: the earlier 13).

    The inventory-based scale is smaller than the length term (0.26 and 0.14 against 0.98 here), so the
    unmodified start meets ``tol * scale`` only two halvings later. The run still goes through the
    terminal thin-phase path, stops there, and leaves a committed, consistent state.
    """
    model, attempts = _c32_run(scale)
    with pytest.warns(RuntimeWarning, match="below terminal_thin_phase_width"):
        model.solve(2e-6, minDtFrac=1e-14)

    # Retry and timestep behavior: every attempt halves the previous one; only the last is accepted.
    assert [dt for dt, _, _ in attempts] == [2e-6 * 0.5**k for k in range(halvings + 1)]
    assert [ok for _, ok, _ in attempts] == [False] * halvings + [True]
    assert all(residual > model.residualTolerance for _, _, residual in attempts[:-1])
    assert attempts[-1][2] <= model.residualTolerance
    assert attempts[-1][2] > 0.5 * model.residualTolerance   # one halving earlier would not have passed
    final_dt = 2e-6 * 0.5**halvings
    assert model.currentTime == final_dt and model._currdt == final_dt
    assert model._lastStepRetries == halvings
    assert model._lastImplicitIterations == 1 and model._lastImplicitConverged is True
    assert model._lastImplicitForcedUpdateSkipped is True   # accepted at the start: the update could not be attempted

    # The terminal thin-phase path was taken (one normal attempt, then the extra retries) and stopped the solve.
    assert model.maxStepRetries == 1 and halvings > model.maxStepRetries
    info = model._terminalThinPhaseInfo
    assert info is not None and info["phase"] == "C" and info["phase_index"] == 2
    assert info["threshold"] == .31 and info["width"] < .31 and info["time"] == 0.0 and info["dt"] == final_dt
    assert model._terminalThinPhaseStop is False and model.finalTime == final_dt < 2e-6

    # Lifecycle: one committed step, no pending candidate, histories and live state agree, moles conserved.
    assert model._pendingCandidate is None
    times = np.asarray(model.interfaceData._time[: int(model.interfaceData.N) + 1], dtype=np.float64)
    assert times[0] == 0.0 and times[-1] == final_dt and np.all(np.diff(times) >= 0.0)
    np.testing.assert_array_equal(_recorded(model.interfaceData)[-1], model._interfaces_curr)
    np.testing.assert_array_equal(_recorded(model.etaData)[-1], model._etas_curr)
    molar = _recorded(model.molarInventoryData)
    scale_now = _scale(model, model._profiles_curr, model._interfaces_curr)
    drift = np.abs(molar[-1, 1:] - molar[0, 1:]) / model._phaseMolarDensities[0]
    assert np.all(drift <= 2 * model.residualTolerance * scale_now * (1 + 1e-6) + 32 * EPS * scale_now)

    # A fresh model repeats the run bit for bit, including the retry count. (``reset`` is not part of this
    # case: ``maxIterations = 1`` is below the validated minimum and can only be set after ``setup``, so a
    # reset model is refused by the configuration validator, which is checked here too.)
    repeat, repeat_attempts = _c32_run(scale)
    with pytest.warns(RuntimeWarning, match="below terminal_thin_phase_width"):
        repeat.solve(2e-6, minDtFrac=1e-14)
    assert repeat_attempts == attempts and repeat.currentTime == model.currentTime
    np.testing.assert_array_equal(repeat._interfaces_curr, model._interfaces_curr)
    np.testing.assert_array_equal(repeat._etas_curr, model._etas_curr)
    for profile, expected in zip(repeat._profiles_curr, model._profiles_curr):
        np.testing.assert_array_equal(profile, expected)
    model.reset()
    with pytest.raises(ValueError, match="max_iterations must be at least 2"):
        model.solve(2e-6, minDtFrac=1e-14)
    assert len(attempts) == halvings + 1 and model._terminalThinPhaseInfo is None and model._pendingCandidate is None


def test_c32_scales_explain_the_two_extra_halvings():
    """The physical start residual is the same; only its divisor changed, by more than 2 and less than 8."""
    default, _ = _c32_run(None)
    earlier, _ = _c32_run(1.0)
    state = (default._profiles_curr, default._interfaces_curr)
    new_scale, old_scale = _scale(default, *state), _scale(earlier, *state)
    np.testing.assert_array_equal(old_scale, np.full(2, _length(earlier)))
    np.testing.assert_allclose(new_scale, np.abs(_inventory(default, *state)), rtol=4 * EPS, atol=0.0)
    ratio = old_scale / new_scale
    assert np.all(ratio > 2.0) and np.max(ratio) < 8.0   # two, not three, extra halvings of a residual linear in dt
