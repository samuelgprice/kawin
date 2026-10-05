"""
Ternary Illingworth solvers at physical length and diffusivity scales.

The unit-scaled tests use ``D ~ 1e-3`` and ``R ~ 0.4``; here the same kind of
problems are rescaled to ``R ~ 1e-3 m`` and ``D ~ 1e-14 m^2/s`` with tiny
initial timesteps. These regimes exercise two failure modes that do not show
up at unit scale:

- interface steps accepted with the interface frozen, because ``flux * dt``
  is below the residual tolerance before any Newton update is taken;
- initial-eta roots that depend on the starting guess or on the length
  units, because the instantaneous-balance residual is tiny in SI units.
"""

import numpy as np
import pytest

from examples.ternaryExamples.ternaryTwoPhaseSimilarity import FixedMatrixTernaryDiffusivity, solve_similarity_roots
from kawin.diffusion import (
    CallableTernaryInterfaceEquilibrium,
    MovingBoundaryIllingworthTernaryFD1DModel,
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
    estimate_initial_eta_from_instantaneous_balance,
)
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.TransformedGrids import two_phase_landau_grids
from kawin.solver import explicitEulerIterator

# Unit-scaled similarity problem (same as test_illingworth_ternary_similarity.py),
# mapped to physical units by x -> LENGTH_SCALE * x and t -> TIME_SCALE * t.
D_LEFT_UNIT = np.array([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]])
D_RIGHT_UNIT = np.array([[3.0e-4, -0.5e-4], [1.0e-4, 5.0e-4]])
C_LEFT_FAR = np.array([0.38, 0.02])
C_RIGHT_FAR = np.array([0.20, 0.15])
C_LEFT_0, C_LEFT_1 = np.array([0.30, 0.03]), np.array([0.34, 0.07])
C_RIGHT_0, C_RIGHT_1 = np.array([0.18, 0.10]), np.array([0.26, 0.16])
LENGTH_SCALE = 2.5e-3  # R = 0.4 * 2.5e-3 = 1e-3 m
TIME_SCALE = 6.25e5  # D = D_unit * LENGTH_SCALE**2 / TIME_SCALE ~ 1e-14 m^2/s
DOMAIN_LENGTH = 0.4 * LENGTH_SCALE
S0 = (0.2 + 1.0e-12) * LENGTH_SCALE


def _diffusivities(length_unit=1.0):
    """Returns physical ``(D_left, D_right)`` with lengths expressed in ``length_unit`` metres."""
    factor = LENGTH_SCALE**2 / TIME_SCALE / length_unit**2
    return D_LEFT_UNIT * factor, D_RIGHT_UNIT * factor


def _closure():
    return CallableTernaryInterfaceEquilibrium(
        lambda eta: (C_LEFT_0 + eta * (C_LEFT_1 - C_LEFT_0), C_RIGHT_0 + eta * (C_RIGHT_1 - C_RIGHT_0)),
        eta_bounds=(0.0, 1.0),
    )


def _two_phase_model(length_unit=1.0, nodes=21, semi_log_dt=0.05, initial_eta_guess=None):
    """Builds the physically scaled two-phase couple with lengths in ``length_unit`` metres."""
    D_left, D_right = _diffusivities(length_unit)
    domain_length = DOMAIN_LENGTH / length_unit
    s0 = S0 / length_unit
    mesh = CartesianFD1D(["X", "Y"], [0.0, domain_length], 400)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(s0, C_LEFT_FAR, C_RIGHT_FAR), ["X", "Y"])]))
    u_grid, v_grid = two_phase_landau_grids(nodes, nodes, method="linear", spacing_ratio=0.1)
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=FixedMatrixTernaryDiffusivity({"ALPHA": D_left, "BETA": D_right}, ["ALPHA", "BETA"]),
        temperature=1000.0,
        interfacePosition=s0,
        interface_equilibrium=_closure(),
        initial_eta_bracket=(0.0, 1.0),
        initial_eta_guess=initial_eta_guess,
        time_step=1.0,
        dt_mode="semi_log",
        semiLog_dt=semi_log_dt,
        semiLogT0=1.0e-6,
        tolerance=1.0e-11,
        max_iterations=50,
        terminal_thin_phase_policy="continue",
        record=True,
        record_pq_data=False,
        transformed_u_grid=u_grid,
        transformed_v_grid=v_grid,
    )


def _three_phase_model(length_unit=1.0, initial_eta_guess=(0.5, 0.5), semi_log_dt=0.05):
    """Builds a physically scaled three-phase couple with lengths in ``length_unit`` metres."""
    factor = LENGTH_SCALE**2 / TIME_SCALE / length_unit**2
    matrices = {
        "A": D_LEFT_UNIT * factor,
        "B": np.array([[5.0e-4, 1.0e-4], [0.5e-4, 4.0e-4]]) * factor,
        "C": D_RIGHT_UNIT * factor,
    }
    eq_ab = CallableTernaryInterfaceEquilibrium(
        lambda eta: (np.array([0.30, 0.03]) + eta * np.array([0.04, 0.04]), np.array([0.26, 0.08]) + eta * np.array([0.04, 0.03])),
        eta_bounds=(0.0, 1.0),
    )
    eq_bc = CallableTernaryInterfaceEquilibrium(
        lambda eta: (np.array([0.24, 0.09]) + eta * np.array([0.03, 0.03]), np.array([0.18, 0.10]) + eta * np.array([0.08, 0.06])),
        eta_bounds=(0.0, 1.0),
    )
    domain_length = DOMAIN_LENGTH / length_unit
    interfaces = (0.4 * domain_length, 0.6 * domain_length)
    values = (C_LEFT_FAR, np.array([0.27, 0.10]), C_RIGHT_FAR)

    def profile(z):
        x = np.asarray(z, dtype=np.float64).reshape(-1)
        out = np.empty((len(x), 2), dtype=np.float64)
        out[x < interfaces[0]] = values[0]
        out[(x >= interfaces[0]) & (x < interfaces[1])] = values[1]
        out[x >= interfaces[1]] = values[2]
        return out

    mesh = CartesianFD1D(["X", "Y"], [0.0, domain_length], 401)
    mesh.setResponseProfile(ProfileBuilder([(profile, ["X", "Y"])]))
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["A", "B", "C"],
        thermodynamics=FixedMatrixTernaryDiffusivity(matrices, ["A", "B", "C"]),
        temperature=1000.0,
        interfacePositions=interfaces,
        interface_equilibria=(eq_ab, eq_bc),
        initial_eta_guess=initial_eta_guess,
        phase_nodes=(11, 11, 11),
        time_step=1.0,
        dt_mode="semi_log",
        semiLog_dt=semi_log_dt,
        semiLogT0=1.0e-6,
        tolerance=1.0e-11,
        max_iterations=50,
        terminal_thin_phase_policy="continue",
        record=True,
        record_pq_data=False,
    )


def _recorded(history):
    n = int(history.N) + 1
    return np.asarray(history._time[:n], dtype=np.float64), np.asarray(history._y[:n], dtype=np.float64)


def test_two_phase_interface_moves_on_every_step_at_physical_scale():
    """Tiny early timesteps must still advance the interface and conserve solute to round-off."""
    roots = solve_similarity_roots(D_LEFT_UNIT, D_RIGHT_UNIT, C_LEFT_FAR, C_RIGHT_FAR, _closure(), s0=0.2)
    assert len(roots) == 1
    k_sign = np.sign(roots[0].k)
    assert k_sign != 0.0

    # solve() runs setup() itself; a second setup() would restart from the
    # already-initialized profile, so the spy is installed on a fresh model.
    model = _two_phase_model()
    skipped = []
    original = model._record_implicit_success

    def spy(*args, **kwargs):
        original(*args, **kwargs)
        skipped.append(bool(getattr(model, "_lastImplicitForcedUpdateSkipped", False)))

    model._record_implicit_success = spy
    model.solve(1.0, iterator=explicitEulerIterator)

    # Per-step eta changes at the earliest steps are below eps * eta, so the
    # interface position and inventory are the meaningful frozen-step checks.
    _, s = _recorded(model.interfaceData)
    _, inventory = _recorded(model.inventoryData)
    assert len(s) > 200
    ds = np.diff(s)
    assert np.all(np.sign(ds) == k_sign), f"{np.count_nonzero(ds == 0.0)} of {ds.size} steps left the interface frozen"
    assert not any(skipped)
    rel_drift = np.abs(inventory[-1] - inventory[0]) / np.abs(inventory[0])
    assert np.all(rel_drift < 1.0e-13), rel_drift


def test_three_phase_interfaces_move_on_every_step_at_physical_scale():
    """Three-phase counterpart of the frozen-step regression."""
    model = _three_phase_model()
    model.solve(1.0e-1, iterator=explicitEulerIterator)

    _, interfaces = _recorded(model.interfaceData)
    _, inventory = _recorded(model.inventoryData)
    assert len(interfaces) > 100
    steps = np.diff(interfaces, axis=0)
    frozen = np.all(steps == 0.0, axis=1)
    assert not np.any(frozen), f"{np.count_nonzero(frozen)} of {len(steps)} steps left both interfaces frozen"
    rel_drift = np.abs(inventory[-1] - inventory[0]) / np.abs(inventory[0])
    assert np.all(rel_drift < 1.0e-13), rel_drift


def test_two_phase_initial_eta_is_independent_of_guess_and_length_units():
    """The instantaneous-balance root must not depend on the starting eta or on the length unit."""
    etas = {}
    for length_unit in (1.0, 1.0e-6):
        model = _two_phase_model(length_unit=length_unit)
        model.setup()
        etas[length_unit] = float(model.initialEta)

        guesses = []
        for guess in np.linspace(0.0, 1.0, 11):
            estimate = estimate_initial_eta_from_instantaneous_balance(
                composition=np.asarray(model.data.currentY, dtype=np.float64),
                z=model._z,
                interface_position=S0 / length_unit,
                phases=model.phases,
                thermodynamics=model.therm,
                temperature=model.temperatureParameters,
                interface_equilibrium=model.interfaceEquilibrium,
                transformed_u_grid=model._u_grid,
                transformed_v_grid=model._v_grid,
                eta_bracket=(0.0, 1.0),
                eta_guess=float(guess),
            )
            assert estimate.converged
            guesses.append(estimate.eta)
        assert np.ptp(guesses) < 1.0e-9, guesses

    assert abs(etas[1.0] - etas[1.0e-6]) < 1.0e-9, etas


def test_three_phase_initial_etas_are_independent_of_guess_and_length_units():
    results = []
    for length_unit in (1.0, 1.0e-6):
        for guess in ((0.5, 0.5), (0.1, 0.9), (0.9, 0.1)):
            model = _three_phase_model(length_unit=length_unit, initial_eta_guess=guess)
            model.setup()
            assert model.initialEtaEstimate.converged
            results.append(np.asarray(model.initialEtaEstimate.etas, dtype=np.float64))
    results = np.asarray(results)
    assert np.all(np.ptp(results, axis=0) < 1.0e-9), results
