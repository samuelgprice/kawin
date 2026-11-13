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

from test_illingworth_history_contract import (
    _FAMILIES,
    _TWO_PHASE,
    _accepted,
    _freeze,
    _histories,
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
