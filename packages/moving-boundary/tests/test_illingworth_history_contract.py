"""R09: Illingworth history recording and no-time getters (stage R09a).

Contract (LEDGERS R09):

- A no-time getter returns the current accepted state, whatever the recording
  interval (including sparse or disabled recording). A request with an explicit
  time keeps the historical lookup rules: scalar histories interpolate and clamp,
  array histories need an exactly recorded time, and ``MeshData`` interpolates.
- Row 0 (the initial record) is never overwritten. Regular rows are stored every
  ``k`` accepted steps, counted from ``reset`` across continued solves. A forced
  final record is appended as an extra row, never overwrites a regular one and
  does not use a cadence slot.
- Final deduplication is flag-based: ``finalize`` adds its row unless the most
  recent record call itself stored one. Times are not compared, so a genuinely
  new snapshot a few 1e-15 after the last stored row is kept.
- At interval 1 the binary and two-phase ternary ``data`` history still gains one
  legacy duplicate row per ``postSolve`` (removal is R09c). Recording-disabled
  histories keep a single row holding the current state.
- Binary and two-phase ternary ``getTransformedState()`` with no time returns the
  live transformed state even without a p/q history.

Defect tests fail before R09a for the stated reason (wrong rows, a stale value or
a missing row). Preservation tests pass before and after it.
"""

import dataclasses

import numpy as np
import pytest

from moving_boundary.compatibility import MeshData
from moving_boundary.mesh import CartesianFD1D, MixedBoundary1D, ProfileBuilder, StepProfile1D
from moving_boundary.solvers import (
    MovingBoundaryIllingworthFD1DModel,
    MovingBoundaryIllingworthTernaryFD1DModel,
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
)
from moving_boundary.solvers.illingworth import binary as binary_module
from moving_boundary.solvers.illingworth import shared as shared_module
from kawin.diffusion.DiffusionParameters import TemperatureParameters
from ported_helpers.illingworth_baseline_cases import (
    _BinaryThermodynamics,
    _CoupledTernaryThermodynamics,
    _CurvedInterfaceEquilibrium,
    _LinearPairEquilibrium,
    _ThreePhaseStepProfile,
    _ThreePhaseThermodynamics,
)


def _binary(record, record_pq_data=True):
    mesh = CartesianFD1D(["CR"], [0.0, 5.0], 501)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(1.0, 0.8, 0.4), "CR")]))
    return MovingBoundaryIllingworthFD1DModel(
        mesh, ["NI", "CR"], ["ALPHA", "BETA"], thermodynamics=_BinaryThermodynamics({"ALPHA": 1.0e-3, "BETA": 1.0e-1}),
        temperature=TemperatureParameters(1000.0), interfacePosition=1.0, interface_compositions=(0.6, 0.0),
        time_step=0.1, phase_a_nodes=100, phase_b_nodes=100, record=record, record_pq_data=record_pq_data,
    )


def _ternary(record, record_pq_data=True):
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(
        ProfileBuilder([(StepProfile1D(0.45, np.asarray([0.16, 0.06]), np.asarray([0.3261538461538461, 0.193])), ["X", "Y"])]),
        boundaryConditions=MixedBoundary1D(2),
    )
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["ALPHA", "BETA"],
        thermodynamics=_CoupledTernaryThermodynamics(), temperature=1000.0, interfacePosition=0.45,
        interface_equilibrium=_CurvedInterfaceEquilibrium(), initial_eta_bracket=(0.0, 1.0),
        time_step=1.0e-2, tolerance=1.0e-11, max_iterations=50, record=record, record_pq_data=record_pq_data,
    )


def _three_phase(record, record_pq_data=True):
    interfaces = (0.35, 0.7)
    values = ([0.21, 0.09], [0.31, 0.16], [0.20, 0.21])
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 31)
    mesh.setResponseProfile(ProfileBuilder([(_ThreePhaseStepProfile(interfaces, values), ["X", "Y"])]))
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["A", "B", "C"], thermodynamics=_ThreePhaseThermodynamics(),
        temperature=1000.0, interfacePositions=interfaces,
        interface_equilibria=(
            _LinearPairEquilibrium([0.20, 0.10], [0.10, 0.0], [0.30, 0.15], [0.10, 0.0]),
            _LinearPairEquilibrium([0.24, 0.15], [0.10, 0.0], [0.18, 0.20], [0.10, 0.0]),
        ),
        initial_eta_guess=(0.5, 0.5), phase_molar_volumes=(2.0, 3.0, 1.5), phase_nodes=(9, 9, 9),
        time_step=1.0e-4, tolerance=1.0e-10, residual_tolerance=1.0e-10, max_iterations=50,
        record=record, record_pq_data=record_pq_data,
    )


# family -> (builder, one scheduled step)
_FAMILIES = {"binary": (_binary, 0.1), "ternary": (_ternary, 1.0e-2), "three_phase": (_three_phase, 1.0e-4)}
_TWO_PHASE = ("binary", "ternary")
_ACCEPTED = (
    "currentTime", "_s_curr", "_s_old", "_p_curr", "_q_curr", "_eta_curr", "_eta_old",
    "_profiles_curr", "_interfaces_curr", "_interfaces_old", "_etas_curr", "_etas_old", "_R",
)


def _freeze(value):
    """Returns a comparable, copy-independent image of ``value`` (arrays by dtype, shape and bytes)."""
    if isinstance(value, np.ndarray):
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


def _histories(model):
    """Every recorded history of ``model`` by name (``None`` p/q or profile histories omitted)."""
    names = (
        "data", "interfaceData", "concData", "etaData", "inventoryData", "pData", "qData",
        "rightBoundaryData", "molarInventoryData", "totalMolesData", "dependentMoleClosureErrorData",
    )
    out = {name: getattr(model, name) for name in names if getattr(model, name, None) is not None}
    for index, history in enumerate(getattr(model, "profileData", None) or ()):
        out[f"profileData[{index}]"] = history
    return out


def _times(history):
    return history._time[: history.N + 1].copy()


def _row_times(model):
    return {name: _times(history) for name, history in _histories(model).items()}


def _track_commits(model):
    """Wraps ``postProcess`` so every accepted time (after the initial 0) is collected."""
    accepted = [0.0]
    original = model.postProcess

    def post_process(time, x):
        result = original(time, x)
        accepted.append(float(model.currentTime))
        return result

    model.postProcess = post_process
    return accepted


def _assert_rows(model, expected, skip=()):
    for name, times in _row_times(model).items():
        if name in skip:
            continue
        assert np.array_equal(times, expected), f"{name} recorded times {times.tolist()}, expected {list(expected)}"


def _live_getters(family, model):
    """Pairs of (no-time getter value, accepted live value) for every no-time getter covered by R09."""
    pairs = {"data": (model.data.y(), model.data.currentY)}
    if family == "binary":
        pairs["interface"] = (model.getInterfacePosition(), model._s_curr)
        pairs["transformed"] = (model.getTransformedState(), (model._p_curr, model._q_curr))
        pairs["inventory"] = (
            model.getTotalInventory(), model.getTotalInventoryFromState(model._p_curr, model._q_curr, model._s_curr)
        )
    elif family == "ternary":
        pairs["interface"] = (model.getInterfacePosition(), model._s_curr)
        pairs["eta"] = (model.getInterfaceEta(), model._eta_curr)
        pairs["interface_compositions"] = (
            model.getInterfaceCompositions(), model._interface_compositions(model._eta_curr)
        )
        pairs["transformed"] = (model.getTransformedState(), (model._p_curr, model._q_curr))
        pairs["inventory"] = (
            model.getTotalInventory(), model.getTotalInventoryFromState(model._p_curr, model._q_curr, model._s_curr)
        )
    else:
        pairs["interfaces"] = (model.getInterfacePositions(), model._interfaces_curr)
        pairs["etas"] = (model.getInterfaceEtas(), model._etas_curr)
        pairs["interface_compositions"] = (
            model.getInterfaceCompositions(), model._interface_compositions(model._etas_curr)
        )
        pairs["transformed"] = (model.getTransformedState(), model._profiles_curr)
        pairs["right_boundary"] = (model.getRightBoundary(), model._R)
    return pairs


def _stale_getters(family, model):
    """Names of no-time getters whose value differs (bitwise) from the accepted live state."""
    stale = []
    for name, (reported, live) in _live_getters(family, model).items():
        if _freeze(np.asarray(reported, dtype=np.float64) if np.isscalar(reported) else reported) != _freeze(
            np.asarray(live, dtype=np.float64) if np.isscalar(live) else live
        ):
            stale.append(name)
    return stale


def _accepted(model):
    return {name: _freeze(getattr(model, name, "<absent>")) for name in _ACCEPTED}


# Defect tests (fail before R09a)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_initial_row_is_kept_when_the_run_is_shorter_than_the_interval(family):
    """One step at interval 3: every history holds the initial row and the forced final row."""
    build, step = _FAMILIES[family]
    model = build(3)
    accepted = _track_commits(model)

    model.solve(step, minDtFrac=1e-14)

    assert len(accepted) == 2
    _assert_rows(model, np.asarray(accepted))


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_off_cadence_final_row_is_appended_after_the_regular_sample(family):
    """Four steps at interval 3 give synchronized [0, 3, 4] steps in every history, including data."""
    build, step = _FAMILIES[family]
    model = build(3)
    accepted = _track_commits(model)

    model.solve(4 * step, minDtFrac=1e-14)

    assert len(accepted) == 5
    _assert_rows(model, np.asarray(accepted)[[0, 3, 4]])


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_continued_solves_keep_the_regular_cadence(family):
    """Two + two steps at interval 3: [0, 2, 3, 4]; forced rows use no cadence slot.

    The regular rows (0 and 3) and the final row equal those of a single four-step run.
    """
    build, step = _FAMILIES[family]
    continued = build(3)
    accepted = _track_commits(continued)
    continued.solve(2 * step, minDtFrac=1e-14)
    continued.solve(2 * step, minDtFrac=1e-14)
    single = build(3)
    single.solve(4 * step, minDtFrac=1e-14)

    assert len(accepted) == 5
    _assert_rows(continued, np.asarray(accepted)[[0, 2, 3, 4]])
    for name, history in _histories(continued).items():
        reference = _histories(single)[name]
        for row, reference_row in ((0, 0), (2, 1), (3, 2)):
            assert _freeze(history._y[row]) == _freeze(reference._y[reference_row]), (name, row)
            assert history._time[row] == reference._time[reference_row], (name, row)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_no_time_getters_return_the_live_state_after_every_commit(family):
    """At interval 3 every no-time getter equals the accepted state after each committed step."""
    build, step = _FAMILIES[family]
    model = build(3)
    stale = {}
    original = model.postProcess

    def post_process(time, x):
        result = original(time, x)
        found = _stale_getters(family, model)
        if found:
            stale[float(model.currentTime)] = found
        return result

    model.postProcess = post_process
    model.setup()
    after_setup = _stale_getters(family, model)
    model.solve(4 * step, minDtFrac=1e-14)

    assert after_setup == [], f"stale after setup: {after_setup}"
    assert stale == {}, f"stale no-time getters by accepted time: {stale}"
    assert _stale_getters(family, model) == []


@pytest.mark.parametrize("reset_first", [False, True])
def test_binary_time_zero_profile_is_the_reconstruction_and_timed_lookup_is_historical(reset_first):
    """Binary at t = 0 with no committed step (contracts C25/C29 capture this state).

    The no-time profile is the reconstruction from the accepted transformed state
    that the solver starts from; a request at t = 0 still returns stored row 0,
    the profile before reconstruction.
    """
    model = _binary(True)
    if reset_first:
        model.setup()
        model.reset()
    model.setup()
    reconstructed = model._reconstruct_physical_profile(model._p_curr, model._q_curr, model._s_curr)[:, np.newaxis]
    row0 = model.data._y[0].copy()

    assert model.currentTime == 0.0 and model.data.N == 0
    assert _freeze(np.asarray(model.data.y(), dtype=np.float64)) == _freeze(reconstructed)
    assert _freeze(np.asarray(model.data.y(0.0), dtype=np.float64)) == _freeze(row0)
    assert not np.array_equal(reconstructed, row0)
    live = model.getCompositions()
    historical = model.getCompositions(0.0)
    assert np.array_equal(live[:, 1], reconstructed[:, 0])
    assert np.array_equal(historical[:, 1], row0[:, 0])


@pytest.mark.parametrize("family", _TWO_PHASE)
def test_solve_zero_adds_no_row_at_a_sparse_interval(family):
    """At interval 3, ``solve(0)`` after a one-step run adds no row to any history (data included)."""
    build, step = _FAMILIES[family]
    model = build(3)
    model.solve(step, minDtFrac=1e-14)
    before = _row_times(model)

    model.solve(0.0)

    after = _row_times(model)
    changed = {name: (before[name].tolist(), after[name].tolist()) for name in before if not np.array_equal(before[name], after[name])}
    assert changed == {}, f"solve(0) changed recorded rows: {changed}"


@pytest.mark.parametrize("family", _TWO_PHASE)
def test_repeated_post_solve_adds_no_data_row_at_a_sparse_interval(family):
    """At interval 3 a second ``postSolve`` leaves every history, data included, unchanged."""
    build, step = _FAMILIES[family]
    model = build(3)
    model.solve(4 * step, minDtFrac=1e-14)
    before = {name: _freeze((h._time, h._y, h.N)) for name, h in _histories(model).items()}

    model.postSolve()

    changed = [name for name, h in _histories(model).items() if _freeze((h._time, h._y, h.N)) != before[name]]
    assert changed == [], f"a repeated postSolve changed: {changed}"


@pytest.mark.parametrize("family", _TWO_PHASE)
def test_transformed_state_without_pq_history_returns_the_live_state(family):
    """With ``record_pq_data=False`` a no-time call returns the accepted p/q; a timed call still raises."""
    build, step = _FAMILIES[family]
    model = build(True, record_pq_data=False)
    model.solve(2 * step, minDtFrac=1e-14)

    p, q = model.getTransformedState()

    assert _freeze(p) == _freeze(model._p_curr)
    assert _freeze(q) == _freeze(model._q_curr)
    assert _freeze(model.getTransformedStateLeft()) == _freeze(model._p_curr)
    assert _freeze(model.getTransformedStateRight()) == _freeze(model._q_curr)
    p[...] = -1.0
    assert not np.any(model._p_curr == -1.0), "the returned state must be a copy"
    with pytest.raises(ValueError, match="record_pq_data=True"):
        model.getTransformedState(model.currentTime)


def _history_factories():
    """The five history implementations, each with a value maker for row ``k``."""
    class _Mesh:
        y = np.zeros((3, 1))

        def flattenResponse(self, y):
            return y

    return {
        "binary_scalar": (lambda record: binary_module._ScalarHistory(record), lambda k: float(k) + 0.5),
        "binary_vector": (lambda record: binary_module._VectorHistory(2, record), lambda k: np.asarray([k, -k], dtype=float)),
        "shared_scalar": (lambda record: shared_module._ScalarHistory(record), lambda k: float(k) + 0.5),
        "shared_array": (lambda record: shared_module._ArrayHistory((2, 2), record), lambda k: np.full((2, 2), float(k))),
        "mesh_data": (lambda record: MeshData(_Mesh(), record), lambda k: np.full((3, 1), float(k))),
    }


@pytest.mark.parametrize("kind", sorted(_history_factories()))
def test_finalize_keeps_a_new_snapshot_closely_after_the_last_stored_row(kind):
    """Interval 2: a final accepted snapshot 5e-15 after the last stored row is appended, not dropped.

    ``MeshData`` is checked at interval 2 because its interval-1 finalize keeps
    the legacy duplicate until R09c. Its case passes before R09a too (the host
    finalize never deduplicated); the four package histories failed there.
    """
    make, value = _history_factories()[kind]
    history = make(2)
    times = [0.0, 1.0, 2.0, 2.0 + 5.0e-15]
    assert times[3] != times[2]
    for k, time in enumerate(times):
        history.record(time, value(k))

    history.finalize()

    assert _times(history).tolist() == [0.0, 2.0, times[3]]
    assert _freeze(np.asarray(history._y[2], dtype=np.float64)) == _freeze(np.asarray(value(3), dtype=np.float64))
    history.finalize()
    assert _times(history).tolist() == [0.0, 2.0, times[3]], "a repeated finalize must add nothing"


@pytest.mark.parametrize("family", _TWO_PHASE)
def test_post_solve_keeps_a_direct_commit_closely_after_the_last_stored_row(family):
    """Direct commits at interval 2 ending 5e-15 after a stored row: every history keeps the final snapshot."""
    build, step = _FAMILIES[family]
    model = build(2)
    model.setup()
    times = [step, 2.0 * step, 2.0 * step + 5.0e-15]
    assert times[2] != times[1]
    for time in times:
        model.postProcess(time, model.getCurrentX())

    model.postSolve()

    _assert_rows(model, np.asarray([0.0, times[1], times[2]]))
    assert _stale_getters(family, model) == []


# Preservation tests (pass before and after R09a)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_interval_one_layout_including_the_legacy_data_duplicate(family):
    """Interval 1: binary/two-phase data repeats its final row once per postSolve (R09c); others do not.

    This test is deliberately replaced when R09c removes the duplicate.
    """
    build, step = _FAMILIES[family]
    model = build(True)
    accepted = _track_commits(model)
    model.solve(2 * step, minDtFrac=1e-14)
    first = np.asarray(accepted)
    model.solve(step, minDtFrac=1e-14)
    second = np.asarray(accepted)

    if family == "three_phase":
        _assert_rows(model, second)
        return
    _assert_rows(model, second, skip=("data",))
    expected_data = np.concatenate((first, first[-1:], second[-1:], second[-1:]))
    assert np.array_equal(_times(model.data), expected_data), _times(model.data).tolist()
    assert _freeze(model.data._y[3]) == _freeze(model.data._y[2])
    assert _freeze(model.data._y[5]) == _freeze(model.data._y[4])


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_repeated_post_solve_adds_no_package_history_row(family):
    """Package histories (and three-phase data) are unchanged by a repeated postSolve, at intervals 1 and 3.

    At interval 1 the binary/two-phase data history adds exactly one legacy duplicate per call.
    """
    build, step = _FAMILIES[family]
    for record in (True, 3):
        model = build(record)
        model.solve(4 * step, minDtFrac=1e-14)
        before = {name: _freeze((_times(h), h._y[: h.N + 1], h.N)) for name, h in _histories(model).items()}
        data_rows = model.data.N + 1

        model.postSolve()

        for name, history in _histories(model).items():
            if name == "data" and family in _TWO_PHASE:
                continue
            assert _freeze((_times(history), history._y[: history.N + 1], history.N)) == before[name], (record, name)
        if family in _TWO_PHASE and record is True:
            assert model.data.N + 1 == data_rows + 1
            assert _freeze(model.data._y[model.data.N]) == _freeze(model.data._y[model.data.N - 1])


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_disabled_recording_keeps_one_live_row(family):
    """``record=False``: every history holds one row with the final state, and getters are live."""
    build, step = _FAMILIES[family]
    model = build(False)
    model.solve(2 * step, minDtFrac=1e-14)

    for name, history in _histories(model).items():
        assert history._time.shape[0] == 1 and history.N == 0, name
        assert history._time[0] == model.currentTime, name
    assert _stale_getters(family, model) == []


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_on_cadence_final_step_adds_no_row(family):
    """Three steps at interval 3 give [0, 3] in every history: the final row is the regular sample."""
    build, step = _FAMILIES[family]
    model = build(3)
    accepted = _track_commits(model)

    model.solve(3 * step, minDtFrac=1e-14)

    _assert_rows(model, np.asarray(accepted)[[0, 3]])


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_final_accepted_state_is_independent_of_the_recording_interval(family):
    """Recording never changes the solution: the final accepted state is bitwise equal for 1, 3 and off."""
    build, step = _FAMILIES[family]
    images = []
    for record in (True, 3, False):
        model = build(record)
        model.solve(4 * step, minDtFrac=1e-14)
        images.append(_accepted(model))

    assert images[1] == images[0]
    assert images[2] == images[0]


def test_explicit_time_lookup_rules_are_unchanged():
    """Scalars interpolate and clamp; arrays need an exact recorded time; MeshData interpolates."""
    factories = _history_factories()
    for kind in ("binary_scalar", "shared_scalar"):
        history = factories[kind][0](True)
        history.record(0.0, 1.0)
        history.record(1.0, 3.0)
        history.finalize()
        assert history.y(0.5) == 2.0
        assert history.y(-1.0) == 1.0
        assert history.y(5.0) == 3.0
    for kind in ("binary_vector", "shared_array"):
        make, value = factories[kind]
        history = make(True)
        history.record(0.0, value(0))
        history.record(1.0, value(1))
        history.finalize()
        assert np.array_equal(history.y(1.0), value(1))
        assert np.array_equal(history.y(1.0 + 5.0e-15), value(1))
        with pytest.raises(ValueError):
            history.y(0.5)
    make, value = factories["mesh_data"]
    data = make(True)
    data.record(0.0, value(0))
    data.record(1.0, value(2))
    assert np.array_equal(data.y(0.5), value(1))


@pytest.mark.parametrize("family", _TWO_PHASE)
def test_explicit_current_time_mid_run_is_a_recorded_lookup(family):
    """D3: a timed request at the current, unrecorded time clamps to the last stored row, not the live value."""
    build, step = _FAMILIES[family]
    model = build(3)
    stored = []
    original = model.postProcess

    def post_process(time, x):
        result = original(time, x)
        stored.append(
            (model.getInterfacePosition(model.currentTime), model.interfaceData._y[model.interfaceData.N], model._s_curr)
        )
        return result

    model.postProcess = post_process
    model.solve(2 * step, minDtFrac=1e-14)

    assert len(stored) == 2
    assert all(reported == last_row for reported, last_row, _ in stored)
    assert stored[0][0] != stored[0][2], "the off-cadence step must not be reported live by a timed request"


def test_three_phase_zero_duration_is_still_rejected():
    """Three-phase keeps rejecting ``solve(0)`` before any history changes."""
    model = _three_phase(3)
    model.setup()
    before = _row_times(model)

    with pytest.raises(ValueError):
        model.solve(0.0)

    after = _row_times(model)
    assert all(np.array_equal(before[name], after[name]) for name in before)


@pytest.mark.parametrize("record", [True, 3])
def test_three_phase_failed_commit_restores_every_history_field(record):
    """Three-phase atomicity: a coupled-model failure after recording restores every history attribute."""
    model = _three_phase(record)
    model.solve(3 * model.timeStep, minDtFrac=1e-14)
    before = {name: _freeze(dict(vars(history))) for name, history in _histories(model).items()}

    def fail():
        raise RuntimeError("coupled update failed")

    model.updateCoupledModels = fail
    with pytest.raises(RuntimeError, match="coupled update failed"):
        model.solve(model.timeStep, minDtFrac=1e-14)

    after = {name: _freeze(dict(vars(history))) for name, history in _histories(model).items()}
    assert [name for name in before if before[name] != after[name]] == []
