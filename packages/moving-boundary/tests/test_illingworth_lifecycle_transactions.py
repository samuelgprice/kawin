"""R10: Illingworth lifecycle transactions (failure recovery).

Contract (LEDGERS R10):

- R10a: ``setup()`` and ``reset()`` are all-or-nothing in every family. On any
  exception every model-owned field (time, ``isSetup``, accepted state, every
  history's rows and bookkeeping, candidate, latch, diagnostics) is restored to
  its value before the call and the exception is re-raised. A failed setup
  leaves ``isSetup=False``; a validation failure at the end of ``reset()``
  keeps the previous simulation (D5, D-R1). After the cause is fixed, a
  continued solve equals an untouched model's continuation, a repeated reset
  and solve equals an untouched model doing the same successful reset and
  solve, and setup/solve after a failed setup equals an untouched model, all
  bit for bit.

A successful reset and solve equals a freshly constructed model for binary and
two-phase ternary. Three-phase does not, before or after R10 (a pre-existing
limitation outside R10, user decision 2026-11-13): the host ``reset`` records
the raw ``mesh.y`` while the constructor stores its u-fraction conversion,
which differs from it in the last bits at some nodes.

- R10b: binary and two-phase ternary ``postProcess`` prepare every output
  before changing model state and commit histories, accepted state, time,
  projection totals and ``_lastAcceptedDt`` as one unit (as three-phase). On a
  failure nothing changes except that the candidate stays consumed and the
  R07 latch stays set: a resubmission is refused and only a newly solved
  candidate moves on, reproducing an untouched model bit for bit. A
  coupled-model failure rolls back this model only (D6). ``postSolve``
  finalizes all of a family's histories as one unit; repeating it is the retry.
  Ternary ``_lastAcceptedDt`` is accepted state (D7): a rejected proposal does
  not change it or the next extrapolated Newton start.

External state is never restored (provider caches, closure bindings, mesh
boundary conditions, coupled models, printed diagnostics).

Defect tests fail before the stage that fixes them for the stated reason
(partial state, advanced or rewound time, wrong rows), not because of import
or setup errors. Every reset-failure test checks the rollback first, so its
recovery part runs only once that holds. Preservation tests pass before and
after.
"""

import numpy as np
import pytest

from moving_boundary.compatibility import recording as recording_module
from moving_boundary.solvers.illingworth import binary as binary_module
from moving_boundary.solvers.illingworth import shared as shared_module
from moving_boundary.solvers.illingworth import three_phase as three_phase_module
from test_illingworth_history_contract import (
    _FAMILIES,
    _TWO_PHASE,
    _accepted,
    _assert_rows,
    _freeze,
    _histories,
    _track_commits,
)


def _is_history(value):
    return hasattr(value, "_y") and hasattr(value, "_time") and hasattr(value, "recordInterval")


def _image(model):
    """Bytewise image of every model-owned field; histories by their full attribute set (rows, capacity, bookkeeping)."""
    out = {}
    for name, value in vars(model).items():
        if _is_history(value):
            out[name] = ("history", _freeze(dict(vars(value))))
        elif isinstance(value, (list, tuple)) and value and all(_is_history(v) for v in value):
            out[name] = ("histories", tuple(_freeze(dict(vars(v))) for v in value))
        else:
            out[name] = _freeze(value)
    return out


def _image_changes(before, after):
    return sorted(name for name in set(before) | set(after) if before.get(name, "<absent>") != after.get(name, "<absent>"))


def _result(model):
    """Model-independent image of a simulation: time, accepted state and every history's attributes."""
    return {
        "currentTime": _freeze(model.currentTime),
        "isSetup": model.isSetup,
        "accepted": _accepted(model),
        "histories": {name: _freeze(dict(vars(h))) for name, h in _histories(model).items()},
    }


def _assert_same_result(model, reference):
    got, expected = _result(model), _result(reference)
    assert got["currentTime"] == expected["currentTime"]
    assert got["isSetup"] == expected["isSetup"]
    assert [k for k in expected["accepted"] if got["accepted"][k] != expected["accepted"][k]] == []
    assert sorted(got["histories"]) == sorted(expected["histories"])
    assert [k for k in expected["histories"] if got["histories"][k] != expected["histories"][k]] == []


def _raise(message):
    def fail(*args, **kwargs):
        raise RuntimeError(message)

    return fail


def _two_step_model(family, record=3):
    build, step = _FAMILIES[family]
    model = build(record)
    model.solve(2 * step, minDtFrac=1e-14)
    return model


def _fail_reset_with_clear_cache(model):
    """Makes the provider's ``clearCache`` raise, calls ``reset`` and checks the rollback; then removes the fault."""
    before = _image(model)
    model.therm.clearCache = _raise("clearCache failed")
    with pytest.raises(RuntimeError, match="clearCache failed"):
        model.reset()
    changed = _image_changes(before, _image(model))
    del model.therm.clearCache
    assert changed == [], f"a failed reset changed {changed}"


def _fail_reset_with_invalid_setting(model):
    """Sets an invalid ``max_iterations``, checks that ``reset`` rolls back, then restores the setting."""
    original = model.maxIterations
    model.maxIterations = 1
    before = _image(model)
    with pytest.raises(ValueError, match="max_iterations"):
        model.reset()
    changed = _image_changes(before, _image(model))
    model.maxIterations = original
    assert changed == [], f"a reset failing its configuration validation changed {changed}"


# R10a defect tests (fail before R10a)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_reset_clear_cache_failure_restores_the_model(family):
    """A failing provider ``clearCache`` in ``reset`` leaves time, ``isSetup``, state and histories byte-identical."""
    model = _two_step_model(family)

    _fail_reset_with_clear_cache(model)

    assert model.isSetup is True and model.currentTime > 0.0


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_reset_clear_cache_failure_then_continued_solve_equals_an_untouched_continuation(family):
    _, step = _FAMILIES[family]
    model = _two_step_model(family)
    untouched = _two_step_model(family)

    _fail_reset_with_clear_cache(model)
    model.solve(2 * step, minDtFrac=1e-14)
    untouched.solve(2 * step, minDtFrac=1e-14)

    _assert_same_result(model, untouched)


def _assert_reset_recovery(family, model):
    """A repeated reset and solve equals an untouched model's reset and solve (and a fresh model where that holds)."""
    build, step = _FAMILIES[family]
    model.reset()
    model.solve(2 * step, minDtFrac=1e-14)
    untouched = _two_step_model(family)
    untouched.reset()
    untouched.solve(2 * step, minDtFrac=1e-14)

    _assert_same_result(model, untouched)
    if family in _TWO_PHASE:
        fresh = build(3)
        fresh.solve(2 * step, minDtFrac=1e-14)
        _assert_same_result(model, fresh)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_reset_clear_cache_failure_then_repeated_reset_and_solve_equals_an_untouched_reset(family):
    model = _two_step_model(family)

    _fail_reset_with_clear_cache(model)

    _assert_reset_recovery(family, model)


@pytest.mark.parametrize("family", _TWO_PHASE)
def test_reset_validation_failure_keeps_the_previous_simulation(family):
    """D-R1: a configuration error found at the end of ``reset`` restores the previous simulation."""
    model = _two_step_model(family)

    _fail_reset_with_invalid_setting(model)

    assert model.isSetup is True and model.currentTime > 0.0


@pytest.mark.parametrize("family", _TWO_PHASE)
def test_reset_validation_failure_then_continued_solve_equals_an_untouched_continuation(family):
    _, step = _FAMILIES[family]
    model = _two_step_model(family)
    untouched = _two_step_model(family)

    _fail_reset_with_invalid_setting(model)
    model.solve(2 * step, minDtFrac=1e-14)
    untouched.solve(2 * step, minDtFrac=1e-14)

    _assert_same_result(model, untouched)


@pytest.mark.parametrize("family", _TWO_PHASE)
def test_reset_validation_failure_then_repeated_reset_and_solve_equals_an_untouched_reset(family):
    model = _two_step_model(family)

    _fail_reset_with_invalid_setting(model)

    _assert_reset_recovery(family, model)


def _check_failed_setup(model, fault, error, match):
    """Applies ``fault``, checks that ``setup`` fails, leaves ``isSetup=False`` and changes nothing."""
    undo = fault(model)
    before = _image(model)
    assert model.isSetup is False
    with pytest.raises(error, match=match):
        model.setup()
    changed = _image_changes(before, _image(model))
    undo()
    assert model.isSetup is False, "a failed setup must leave isSetup=False"
    assert changed == [], f"a failed setup changed {changed}"


def _provider_fault(model):
    model.therm.getInterdiffusivity = _raise("provider failed")

    def undo():
        del model.therm.getInterdiffusivity

    return undo


def _setting_fault(model):
    original = model.maxIterations
    model.maxIterations = 1

    def undo():
        model.maxIterations = original

    return undo


def _closure_fault(model):
    closure = model.interfaceEquilibrium if hasattr(model, "interfaceEquilibrium") else model.interfaceEquilibria[0]
    closure.interface_compositions = _raise("closure failed")

    def undo():
        del closure.interface_compositions

    return undo


_SETUP_FAULTS = {
    "provider": (_provider_fault, RuntimeError, "provider failed", sorted(_FAMILIES)),
    "invalid_setting": (_setting_fault, ValueError, "max_iterations", sorted(_FAMILIES)),
    "closure": (_closure_fault, RuntimeError, "closure failed", ("ternary", "three_phase")),
}
_SETUP_CASES = [(fault, family) for fault, spec in sorted(_SETUP_FAULTS.items()) for family in spec[3]]


@pytest.mark.parametrize(("fault", "family"), _SETUP_CASES)
def test_failed_setup_restores_the_exact_pre_setup_state(fault, family):
    """Provider, invalid-setting (direct ``setup()``) and closure failures leave ``isSetup=False`` and change nothing."""
    build, _ = _FAMILIES[family]
    make_fault, error, match, _ = _SETUP_FAULTS[fault]
    model = build(3)

    _check_failed_setup(model, make_fault, error, match)


@pytest.mark.parametrize(("fault", "family"), _SETUP_CASES)
def test_recovery_after_a_failed_setup_equals_an_untouched_model(fault, family):
    """Once the cause is fixed, ``setup`` and ``solve`` reproduce an untouched model bit for bit."""
    build, step = _FAMILIES[family]
    make_fault, error, match, _ = _SETUP_FAULTS[fault]
    model = build(3)
    untouched = build(3)

    _check_failed_setup(model, make_fault, error, match)
    model.setup()
    untouched.setup()
    _assert_same_result(model, untouched)
    model.solve(2 * step, minDtFrac=1e-14)
    untouched.solve(2 * step, minDtFrac=1e-14)

    _assert_same_result(model, untouched)


# R10a preservation tests (pass before and after R10a)


@pytest.mark.parametrize("family", _TWO_PHASE)
def test_successful_reset_then_solve_equals_a_fresh_model(family):
    build, step = _FAMILIES[family]
    for record in (True, 3):
        model = build(record)
        model.solve(2 * step, minDtFrac=1e-14)
        assert model.reset() is None
        assert model.isSetup is False and model.currentTime == 0
        model.solve(3 * step, minDtFrac=1e-14)
        fresh = build(record)
        fresh.solve(3 * step, minDtFrac=1e-14)

        _assert_same_result(model, fresh)


def test_three_phase_successful_reset_records_the_raw_mesh_profile():
    """Pre-existing three-phase reset behavior, kept by R10: row 0 is the raw ``mesh.y``, not the constructor's conversion.

    The fixture's raw profile differs from its u-fraction conversion in the last
    bits at some nodes, so reset + solve is reproducible but not equal to a fresh model.
    """
    build, step = _FAMILIES["three_phase"]
    model = build(3)
    constructed = np.array(model.data.currentY, copy=True)
    model.solve(2 * step, minDtFrac=1e-14)

    assert model.reset() is None

    raw = np.asarray(model.mesh.flattenResponse(model.mesh.y), dtype=np.float64)
    assert model.isSetup is False and model.currentTime == 0
    assert _freeze(np.asarray(model.data.currentY, dtype=np.float64)) == _freeze(raw)
    assert _freeze(np.asarray(model.data._y[0], dtype=np.float64)) == _freeze(raw)
    assert not np.array_equal(raw, constructed)
    model.solve(2 * step, minDtFrac=1e-14)
    again = build(3)
    again.solve(2 * step, minDtFrac=1e-14)
    again.reset()
    again.solve(2 * step, minDtFrac=1e-14)
    _assert_same_result(model, again)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_successful_setup_records_as_before_and_repeats_as_a_no_op(family):
    build, _ = _FAMILIES[family]
    model = build(3)
    other = build(3)

    assert model.setup() is None
    other.setup()
    after_first = _image(model)
    model.setup()

    assert model.isSetup is True
    _assert_same_result(model, other)
    assert _image_changes(after_first, _image(model)) == []
    assert all(history.N == 0 and history._rowCount == 1 for history in _histories(model).values())


# R10b: manual outer steps (as the explicit-Euler wrapper drives them)

# Fields a failed commit may change: the candidate is consumed (with its staged
# projection totals) and the R07 latch stays set.
_CONSUMED = {"_pendingCandidate", "_awaitingFreshCandidate", "_pendingStepProjection"}


def _stepping_model(family, record=True, steps=0):
    """A set-up model with ``steps`` committed manual steps, for a run ending at t = 1."""
    build, _ = _FAMILIES[family]
    model = build(record)
    model.setup()
    model.setTimeInfo(0.0, 1.0)
    for _ in range(steps):
        _commit(model, _propose(model))
    return model


def _propose(model):
    """Solves one step from the accepted state; returns ``(start, x, dXdt, dt)``."""
    start = model.currentTime
    x = model.getCurrentX()
    dXdt = model.getdXdt(start, x)
    return start, x, dXdt, model.getDt(dXdt)


def _update(proposal, fraction=1.0):
    start, x, dXdt, dt = proposal
    return start + fraction * dt, [np.asarray(x[i], dtype=np.float64) + np.asarray(dXdt[i], dtype=np.float64) * (fraction * dt) for i in range(len(x))]


def _commit(model, proposal):
    time, x = _update(proposal)
    return model.postProcess(time, x)


def _assert_failed_commit_changed_nothing(model, before, proposal):
    """Only the consumed candidate and the latch differ, and every later commit of the update is refused."""
    changed = [name for name in _image_changes(before, _image(model)) if name not in _CONSUMED]
    assert changed == [], f"a failed commit changed {changed}"
    assert model._pendingCandidate is None and model._awaitingFreshCandidate is True
    time, x = _update(proposal)
    with pytest.raises(RuntimeError, match="no new candidate has been solved"):
        model.postProcess(time, x)
    assert [name for name in _image_changes(before, _image(model)) if name not in _CONSUMED] == []


def _assert_recovery_equals_untouched(model, family, record, steps_before, steps_after=2, prepare=None):
    """Fresh candidates commit normally and reproduce a model that never failed (given the same ``prepare``)."""
    untouched = _stepping_model(family, record, steps_before)
    if prepare is not None:
        prepare(untouched)
    for _ in range(steps_after):
        _commit(model, _propose(model))
        _commit(untouched, _propose(untouched))
    _assert_same_result(model, untouched)
    assert model._awaitingFreshCandidate is False


def _fail_reconstruction(model):
    model._reconstruct_physical_profile = _raise("reconstruction failed")

    def undo():
        del model._reconstruct_physical_profile

    return undo, "reconstruction failed"


def _fail_composition_check(model):
    model._validate_profile_compositions = _raise("composition check failed")

    def undo():
        del model._validate_profile_compositions

    return undo, "composition check failed"


_COMMIT_FAULTS = {
    "reconstruction": (_fail_reconstruction, _TWO_PHASE),
    "composition_check": (_fail_composition_check, ("ternary",)),
}
_COMMIT_CASES = [(fault, family) for fault, (_, families) in sorted(_COMMIT_FAULTS.items()) for family in families]


# R10b defect tests (fail before R10b)


@pytest.mark.parametrize(("fault", "family"), _COMMIT_CASES)
def test_failure_while_preparing_outputs_leaves_the_accepted_step_unchanged(fault, family):
    """Profile reconstruction or the ternary composition check fail: no time, state, history or total moves."""
    model = _stepping_model(family, steps=1)
    proposal = _propose(model)
    before = _image(model)
    undo, message = _COMMIT_FAULTS[fault][0](model)

    with pytest.raises(RuntimeError, match=message):
        _commit(model, proposal)
    undo()

    _assert_failed_commit_changed_nothing(model, before, proposal)
    _assert_recovery_equals_untouched(model, family, True, 1)


_LAST_HISTORY = {"binary": "concData", "ternary": "inventoryData"}


@pytest.mark.parametrize(
    ("record", "steps_before", "trim"),
    [(True, 2, True), (3, 2, True), (3, 1, False)],
    ids=["interval1-growth", "interval3-on-cadence-growth", "interval3-off-cadence"],
)
@pytest.mark.parametrize("family", _TWO_PHASE)
def test_history_record_failure_mid_commit_restores_every_history(family, record, steps_before, trim):
    """The last history's record raises after the others recorded (with capacity growth): all are restored.

    With ``trim`` every history has exactly ``N + 1`` rows of capacity, so the
    stored row of this step needs growth; the off-cadence case stores no row and
    only advances the cadence counters.
    """
    def trim_capacity(target):
        for history in _histories(target).values():
            history._time = history._time[: history.N + 1].copy()
            history._y = history._y[: history.N + 1].copy()

    prepare = trim_capacity if trim else None
    model = _stepping_model(family, record, steps_before)
    if prepare is not None:
        prepare(model)
    proposal = _propose(model)
    before = _image(model)
    failing = getattr(model, _LAST_HISTORY[family])
    failing.record = _raise("history record failed")

    with pytest.raises(RuntimeError, match="history record failed"):
        _commit(model, proposal)
    del failing.record

    _assert_failed_commit_changed_nothing(model, before, proposal)
    _assert_recovery_equals_untouched(model, family, record, steps_before, prepare=prepare)


class _Recorder:
    """A stateful coupled model: stores the time and state of every update it receives."""

    def __init__(self):
        self.entries = []

    def updateCoupledModel(self, model):
        self.entries.append(_freeze((float(model.currentTime), model.getCurrentX())))


class _FailingCoupledModel:
    def updateCoupledModel(self, model):
        raise RuntimeError("coupled model failed")


@pytest.mark.parametrize("caller_discards_the_entry", [True, False])
@pytest.mark.parametrize("family", _TWO_PHASE)
def test_coupled_failure_rolls_back_this_model_only(family, caller_discards_the_entry):
    """D6: the model rolls back; the coupled model updated before the failure keeps its entry.

    If the caller discards that entry, recovery equals an untouched model and
    recorder pair; otherwise the recorder holds a duplicate of the step.
    """
    model = _stepping_model(family, steps=1)
    recorder, failing = _Recorder(), _FailingCoupledModel()
    model.couplingModels = [recorder, failing]
    proposal = _propose(model)
    before = _image(model)

    with pytest.raises(RuntimeError, match="coupled model failed"):
        _commit(model, proposal)

    _assert_failed_commit_changed_nothing(model, before, proposal)
    assert len(recorder.entries) == 1, "the documented limitation: the external update is not undone"
    model.couplingModels = [recorder]
    if caller_discards_the_entry:
        recorder.entries.clear()
    untouched = _stepping_model(family, steps=1)
    untouched_recorder = _Recorder()
    untouched.couplingModels = [untouched_recorder]
    for _ in range(2):
        _commit(model, _propose(model))
        _commit(untouched, _propose(untouched))

    _assert_same_result(model, untouched)
    if caller_discards_the_entry:
        assert recorder.entries == untouched_recorder.entries
    else:
        assert recorder.entries[0] == recorder.entries[1]
        assert recorder.entries[1:] == untouched_recorder.entries


def test_ternary_rejected_retry_reduced_proposal_keeps_the_last_accepted_dt():
    """D7: ``_lastAcceptedDt`` is accepted state; a rejected proposal changes neither it nor the next Newton start."""
    model = _stepping_model("ternary", steps=1)
    untouched = _stepping_model("ternary", steps=1)
    assert model.interfaceNewtonStart == "extrapolated"
    accepted_dt = model._lastAcceptedDt
    original_step = model._take_implicit_step_planar
    calls = []

    def fail_first(*args):
        calls.append(float(args[-1]))
        if len(calls) == 1:
            raise RuntimeError("forced first-attempt failure")
        return original_step(*args)

    model._take_implicit_step_planar = fail_first
    proposal = _propose(model)
    del model._take_implicit_step_planar
    assert proposal[3] == calls[1] < calls[0], "the proposal must be retry-reduced"
    time, x = _update(proposal, 0.5)
    with pytest.raises(RuntimeError, match="does not match the pending converged"):
        model.postProcess(time, x)

    assert model._lastAcceptedDt == accepted_dt
    starts = {}
    for name, candidate in (("model", model), ("untouched", untouched)):
        original = candidate._extrapolated_interface_start
        seen = starts.setdefault(name, [])

        def capture(*args, _original=original, _seen=seen):
            result = _original(*args)
            _seen.append(_freeze(result))
            return result

        candidate._extrapolated_interface_start = capture
        _commit(candidate, _propose(candidate))
    assert starts["model"] and starts["model"] == starts["untouched"]
    _assert_same_result(model, untouched)
    assert model._lastAcceptedDt == untouched._lastAcceptedDt


_THIRD_FINALIZED = {"binary": "concData", "ternary": "etaData", "three_phase": "etaData"}


def _fail_third_finalize(monkeypatch):
    """Makes the third ``_finalize_history_rows`` call finalize its history and then raise (in every module)."""
    real = recording_module._finalize_history_rows
    calls = []

    def finalize(history):
        calls.append(history)
        real(history)
        if len(calls) == 3:
            raise RuntimeError("finalization failed")

    for module in (recording_module, binary_module, shared_module, three_phase_module):
        if hasattr(module, "_finalize_history_rows"):
            monkeypatch.setattr(module, "_finalize_history_rows", finalize)
    return calls


@pytest.mark.parametrize(("record", "expected_rows"), [(3, [0, 3, 4]), (True, [0, 1, 2, 3, 4])], ids=["interval3", "interval1"])
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_post_solve_finalizes_all_histories_as_one_unit(family, record, expected_rows, monkeypatch):
    """A failure in the third history's finalization restores every history; repeating postSolve is the retry."""
    build, step = _FAMILIES[family]
    model = build(record)
    accepted = _track_commits(model)
    model.postSolve = lambda: None
    model.solve(4 * step, minDtFrac=1e-14)
    del model.postSolve
    before = _image(model)
    calls = _fail_third_finalize(monkeypatch)

    with pytest.raises(RuntimeError, match="finalization failed"):
        model.postSolve()

    assert calls[2] is getattr(model, _THIRD_FINALIZED[family])
    changed = _image_changes(before, _image(model))
    monkeypatch.undo()
    assert changed == [], f"a failed postSolve changed {changed}"
    model.postSolve()
    _assert_rows(model, np.asarray(accepted)[expected_rows])
    finalized = {name: _freeze(dict(vars(h))) for name, h in _histories(model).items()}
    model.postSolve()
    assert {name: _freeze(dict(vars(h))) for name, h in _histories(model).items()} == finalized
