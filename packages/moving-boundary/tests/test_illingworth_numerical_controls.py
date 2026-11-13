"""R12: numerical controls are validated consistently and before any state changes.

Both ternary families (two-phase and three-phase) require positive finite base
``tolerance`` and effective ``residualTolerance``, and positive finite
``semiLog_dt``/``semiLogT0`` while ``dt_mode='semi_log'`` (unused semi-log values
in fixed mode stay accepted). The controls-only check runs before the host
constructor and again at the start of ``solve`` (before any bookkeeping, setup or
schedule), so a control changed on an existing model is refused without touching
it. Binary and two-phase ternary ``solve`` reject negative, NaN and infinite
durations before setup; zero keeps its setup-only no-op behavior, and
three-phase keeps rejecting zero.

Defect tests fail before the R12 fix: the host constructor, ``setup`` or
``setTimeInfo`` sentinel is reached, or no ``ValueError`` is raised. Preservation
tests pass before and after it. Every invalid-duration test stubs ``setup`` and
``setTimeInfo`` with sentinels, so an unfixed infinite duration stops before the
solver loop instead of running unbounded.
"""

import dataclasses

import numpy as np
import pytest

from kawin.diffusion.DiffusionParameters import TemperatureParameters
from moving_boundary.compatibility import DiffusionModel
from moving_boundary.mesh import CartesianFD1D, MixedBoundary1D, ProfileBuilder, StepProfile1D
from moving_boundary.solvers import (
    MovingBoundaryIllingworthFD1DModel,
    MovingBoundaryIllingworthTernaryFD1DModel,
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
)
from ported_helpers.illingworth_baseline_cases import (
    _BinaryThermodynamics,
    _CoupledTernaryThermodynamics,
    _CurvedInterfaceEquilibrium,
    _LinearPairEquilibrium,
    _ThreePhaseStepProfile,
    _ThreePhaseThermodynamics,
)


def _binary(**kwargs):
    mesh = CartesianFD1D(["CR"], [0.0, 5.0], 501)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(1.0, 0.8, 0.4), "CR")]))
    options = dict(time_step=0.1, phase_a_nodes=100, phase_b_nodes=100, record=True)
    options.update(kwargs)
    return MovingBoundaryIllingworthFD1DModel(
        mesh, ["NI", "CR"], ["ALPHA", "BETA"], thermodynamics=_BinaryThermodynamics({"ALPHA": 1.0e-3, "BETA": 1.0e-1}),
        temperature=TemperatureParameters(1000.0), interfacePosition=1.0, interface_compositions=(0.6, 0.0), **options,
    )


def _ternary(**kwargs):
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(
        ProfileBuilder([(StepProfile1D(0.45, np.asarray([0.16, 0.06]), np.asarray([0.3261538461538461, 0.193])), ["X", "Y"])]),
        boundaryConditions=MixedBoundary1D(2),
    )
    options = dict(time_step=1.0e-2, tolerance=1.0e-11, max_iterations=50, record=True)
    options.update(kwargs)
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["ALPHA", "BETA"],
        thermodynamics=_CoupledTernaryThermodynamics(), temperature=1000.0, interfacePosition=0.45,
        interface_equilibrium=_CurvedInterfaceEquilibrium(), initial_eta_bracket=(0.0, 1.0), **options,
    )


def _three_phase(**kwargs):
    interfaces = (0.35, 0.7)
    values = ([0.21, 0.09], [0.31, 0.16], [0.20, 0.21])
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 31)
    mesh.setResponseProfile(ProfileBuilder([(_ThreePhaseStepProfile(interfaces, values), ["X", "Y"])]))
    options = dict(
        phase_molar_volumes=(2.0, 3.0, 1.5), phase_nodes=(9, 9, 9), time_step=1.0e-4, record=True,
        tolerance=1.0e-10, residual_tolerance=1.0e-10, max_iterations=50,
    )
    options.update(kwargs)
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["A", "B", "C"], thermodynamics=_ThreePhaseThermodynamics(),
        temperature=1000.0, interfacePositions=interfaces,
        interface_equilibria=(
            _LinearPairEquilibrium([0.20, 0.10], [0.10, 0.0], [0.30, 0.15], [0.10, 0.0]),
            _LinearPairEquilibrium([0.24, 0.15], [0.10, 0.0], [0.18, 0.20], [0.10, 0.0]),
        ),
        initial_eta_guess=(0.5, 0.5), **options,
    )


# family -> (builder, one scheduled step)
_ALL = {"binary": (_binary, 0.1), "ternary": (_ternary, 1.0e-2), "three_phase": (_three_phase, 1.0e-4)}
_TERNARY_FAMILIES = ("ternary", "three_phase")
_ZERO_DURATION_FAMILIES = ("binary", "ternary")
_BAD = (0.0, -1.0e-12, np.nan, np.inf)


class _Reached(Exception):
    """Raised by a sentinel that the code under test must not reach."""


def _sentinel(name):
    def fail(*args, **kwargs):
        raise _Reached(f"{name} ran before the invalid input was rejected")
    return fail


def _install_solve_sentinels(model):
    model.setup = _sentinel("setup")
    model.setTimeInfo = _sentinel("setTimeInfo")


def _remove_solve_sentinels(model):
    del model.setup
    del model.setTimeInfo


_STATE = (
    "currentTime", "isSetup", "_currdt", "_pendingCandidate", "_awaitingFreshCandidate",
    "_solveMinDtFrac", "_solveMaxDtFrac", "_outerMinTimeStep", "_outerMaxTimeStep",
    "_semiLogTimes", "_semiLogNextIndex", "_nearFinalNoop", "_lastAcceptedDt", "_initialInventory",
    "_s_curr", "_s_old", "_p_curr", "_q_curr", "_eta_curr", "_eta_old",
    "_profiles_curr", "_interfaces_curr", "_etas_curr", "_R", "_z",
    "_terminalThinPhaseStop", "_terminalThinPhaseInfo",
)
_HISTORIES = (
    "data", "interfaceData", "concData", "etaData", "inventoryData", "pData", "qData", "profileData",
    "rightBoundaryData", "molarInventoryData", "totalMolesData", "dependentMoleClosureErrorData",
)


def _freeze(value):
    """Returns a comparable, copy-independent image of ``value`` (arrays by dtype, shape and bytes)."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return ("object-array", value.shape, tuple(_freeze(v) for v in value.reshape(-1)))
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(k), _freeze(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_freeze(v) for v in value))
    if isinstance(value, (float, np.floating)):
        return ("float", np.float64(value).tobytes())
    if isinstance(value, (bool, int, str, np.integer, np.bool_)) or value is None:
        return value
    if dataclasses.is_dataclass(value):
        return (type(value).__name__, _freeze({f.name: getattr(value, f.name) for f in dataclasses.fields(value)}))
    return ("object", type(value).__name__, id(value))


def _history_image(history):
    if history is None:
        return None
    if isinstance(history, (list, tuple)):
        return tuple(_history_image(item) for item in history)
    fields = {
        key: value for key, value in vars(history).items()
        if isinstance(value, (np.ndarray, list, tuple, float, int, bool, np.number)) or value is None
    }
    return _freeze(fields)


def _state(model):
    """Snapshots time, setup flag, bookkeeping, schedule, accepted state and every history."""
    image = {name: _freeze(getattr(model, name, "<absent>")) for name in _STATE}
    image.update({name: _history_image(getattr(model, name, None)) for name in _HISTORIES})
    return image


# Defect tests: invalid controls at construction (fail before the R12 fix)


def _semi_log(**overrides):
    options = dict(dt_mode="semi_log", semiLog_dt=0.5, semiLogT0=1.0e-6)
    options.update(overrides)
    return options


_CONSTRUCTOR_CASES = (
    [(f"tolerance={v}", dict(tolerance=v, residual_tolerance=1.0e-10), r"^tolerance must") for v in _BAD]
    + [(f"residual_tolerance={v}", dict(tolerance=1.0e-10, residual_tolerance=v), r"^residual_tolerance must") for v in _BAD]
    + [(f"semiLog_dt={v}", _semi_log(semiLog_dt=v), r"^semiLog_dt must") for v in (0.0, -0.1, np.nan, np.inf)]
    + [(f"semiLogT0={v}", _semi_log(semiLogT0=v), r"^semiLogT0 must") for v in (0.0, -0.1, np.nan, np.inf)]
)


@pytest.mark.parametrize("label, options, match", _CONSTRUCTOR_CASES, ids=[case[0] for case in _CONSTRUCTOR_CASES])
@pytest.mark.parametrize("family", _TERNARY_FAMILIES)
def test_invalid_control_is_rejected_before_the_host_constructor(family, label, options, match, monkeypatch):
    """The controls check names the control and raises before ``DiffusionModel.__init__`` runs."""
    monkeypatch.setattr(DiffusionModel, "__init__", _sentinel("host constructor"))
    with pytest.raises(ValueError, match=match):
        _ALL[family][0](**options)


# Defect tests: invalid controls on fresh and already-set-up models (fail before the R12 fix)


_MUTATIONS = {
    "residualTolerance=inf": dict(residualTolerance=np.inf),
    "residualTolerance=nan": dict(residualTolerance=np.nan),
    "residualTolerance=0": dict(residualTolerance=0.0),
    "residualTolerance=-1e-12": dict(residualTolerance=-1.0e-12),
    "tolerance=nan": dict(tolerance=np.nan),
    "tolerance=-1e-12": dict(tolerance=-1.0e-12),
    "semi_log semiLog_dt=-1": dict(dtMode="semi_log", semiLog_dt=-1.0, semiLogT0=1.0e-6),
    "semi_log semiLog_dt=0": dict(dtMode="semi_log", semiLog_dt=0.0, semiLogT0=1.0e-6),
    "semi_log semiLogT0=0": dict(dtMode="semi_log", semiLog_dt=0.5, semiLogT0=0.0),
    "semi_log semiLogT0=nan": dict(dtMode="semi_log", semiLog_dt=0.5, semiLogT0=np.nan),
}
_MUTATION_MATCH = {"residualTolerance": r"^residual_tolerance must", "tolerance": r"^tolerance must"}


def _mutation_match(changes):
    if "semiLog_dt" in changes:
        return r"^semiLog_dt must" if changes["semiLog_dt"] <= 0.0 else r"^semiLogT0 must"
    return _MUTATION_MATCH[next(iter(changes))]


@pytest.mark.parametrize("mutation", sorted(_MUTATIONS))
@pytest.mark.parametrize("set_up", [False, True], ids=["fresh", "set_up"])
@pytest.mark.parametrize("family", _TERNARY_FAMILIES)
def test_changed_control_is_rejected_by_solve_without_changes(family, set_up, mutation):
    """A stored control changed after construction is refused at the start of ``solve``.

    Nothing runs or changes: no setup, no time bookkeeping, no schedule, no history row.
    After every changed setting (including ``dtMode``) is restored, continuing matches
    an untouched model bit for bit.
    """
    build, step = _ALL[family]
    model = build()
    control = build()
    if set_up:
        model.solve(step, minDtFrac=1e-14)
        control.solve(step, minDtFrac=1e-14)
    changes = _MUTATIONS[mutation]
    original = {name: getattr(model, name) for name in changes}
    before = _state(model)
    for name, value in changes.items():
        setattr(model, name, value)
    _install_solve_sentinels(model)

    with pytest.raises(ValueError, match=_mutation_match(changes)):
        model.solve(step, minDtFrac=1e-14)

    _remove_solve_sentinels(model)
    assert _state(model) == before
    for name, value in original.items():
        setattr(model, name, value)
    assert {name: getattr(model, name) for name in changes} == {name: getattr(control, name) for name in changes}
    model.solve(step, minDtFrac=1e-14)
    control.solve(step, minDtFrac=1e-14)
    assert _state(model) == _state(control)


# Defect tests: invalid durations (fail before the R12 fix)


@pytest.mark.parametrize("duration", [-1.0e-3, np.nan, np.inf], ids=["negative", "nan", "inf"])
@pytest.mark.parametrize("set_up", [False, True], ids=["fresh", "set_up"])
@pytest.mark.parametrize("family", _ZERO_DURATION_FAMILIES)
def test_invalid_duration_is_rejected_before_setup(family, set_up, duration):
    """Negative, NaN and infinite durations raise before setup, time bookkeeping or a schedule.

    The sentinels keep an unfixed infinite duration from reaching the unbounded solver loop.
    """
    build, step = _ALL[family]
    model = build()
    if set_up:
        model.solve(step, minDtFrac=1e-14)
    before = _state(model)
    _install_solve_sentinels(model)

    with pytest.raises(ValueError, match="simTime must be finite and nonnegative"):
        model.solve(duration, minDtFrac=1e-14)

    _remove_solve_sentinels(model)
    assert _state(model) == before


# Preservation tests (pass before and after the R12 fix)


@pytest.mark.parametrize("family", _TERNARY_FAMILIES)
def test_valid_controls_reach_the_host_constructor(family, monkeypatch):
    """Valid fixed and semi-log controls pass the early check (the sentinel is reached)."""
    monkeypatch.setattr(DiffusionModel, "__init__", _sentinel("host constructor"))
    for options in ({}, _semi_log(), dict(tolerance=1.0e-6, residual_tolerance=1.0e-14)):
        with pytest.raises(_Reached):
            _ALL[family][0](**options)


@pytest.mark.parametrize("family", _TERNARY_FAMILIES)
def test_unused_semi_log_values_are_accepted_in_fixed_mode(family):
    """Semi-log values are inactive in fixed mode and are not validated there."""
    model = _ALL[family][0](semiLog_dt=-1.0, semiLogT0=np.nan)
    assert model.dtMode == "fixed"


@pytest.mark.parametrize("family", _ZERO_DURATION_FAMILIES)
def test_zero_duration_solve_is_a_setup_only_no_op(family):
    """Binary and two-phase ternary ``solve(0)`` runs setup and takes no step.

    Time does not advance, no interface step is accepted and every recorded row is at
    t=0. Whether the forced final record adds a duplicate t=0 profile row is the
    recording contract of LEDGERS R09 and is deliberately not asserted here.
    """
    model = _ALL[family][0]()
    initial_interface = model.interfaceData._y[0]
    model.solve(0.0, minDtFrac=1e-14)

    assert model.isSetup
    assert model.currentTime == 0
    assert model.interfaceData.N == 0
    assert model.interfaceData._y[0] == initial_interface
    assert np.all(model.data._time[: model.data.N + 1] == 0.0)


def test_three_phase_still_rejects_zero_duration():
    model = _three_phase()
    _install_solve_sentinels(model)
    with pytest.raises(ValueError, match="simTime must be positive and finite"):
        model.solve(0.0)


@pytest.mark.parametrize(
    "fractions",
    [dict(minDtFrac=-1.0), dict(maxDtFrac=0.0), dict(minDtFrac=np.nan), dict(maxDtFrac=np.inf), dict(minDtFrac=0.5, maxDtFrac=0.1)],
    ids=["negative-min", "zero-max", "nan-min", "inf-max", "min-above-max"],
)
@pytest.mark.parametrize("family", sorted(_ALL))
def test_invalid_timestep_fractions_are_still_rejected_before_setup(family, fractions):
    model = _ALL[family][0]()
    before = _state(model)
    _install_solve_sentinels(model)
    with pytest.raises(ValueError, match="DtFrac"):
        model.solve(_ALL[family][1], **fractions)
    _remove_solve_sentinels(model)
    assert _state(model) == before


@pytest.mark.parametrize("family", _ZERO_DURATION_FAMILIES)
def test_final_partial_step_still_reaches_the_requested_time(family):
    """A duration that is not a whole number of steps ends with a shorter final step."""
    build, step = _ALL[family]
    model = build()
    model.solve(2.5 * step, minDtFrac=1e-14)

    times = model.interfaceData._time[: model.interfaceData.N + 1]
    assert model.interfaceData.N == 3
    assert np.diff(times)[-1] == pytest.approx(0.5 * step, rel=1e-12)
    assert model.currentTime == pytest.approx(2.5 * step, rel=1e-14)
