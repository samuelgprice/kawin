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

A successful reset (before setup) and a reset followed by setup and solve equal
a freshly constructed model in every family. This holds since the R10
reset-parity follow-ups: three-phase and two-phase ternary reset record the
constructor's u-fraction conversion of ``mesh.y`` (before, they recorded the raw
``mesh.y``, or a different interstitial conversion, which differed from the
constructor's in the last bits at some nodes for some profiles), and two-phase
ternary reset restores the unestimated ``initialEta`` (NaN) instead of recording
the previous setup's estimate. Binary reset needed no change (identical bits).

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

import test_illingworth_interstitials as _fe_c_tests
import test_illingworth_interstitials_ternary as _fe_c_mn_tests
from moving_boundary.analytic import FixedMatrixTernaryDiffusivity
from moving_boundary.compatibility import recording as recording_module
from moving_boundary.mesh import CartesianFD1D, MixedBoundary1D, ProfileBuilder, StepProfile1D
from moving_boundary.mesh.TransformedGrids import two_phase_landau_grids
from moving_boundary.solvers import MovingBoundaryIllingworthTernaryFD1DModel
from moving_boundary.solvers.illingworth import binary as binary_module
from moving_boundary.solvers.illingworth import shared as shared_module
from moving_boundary.solvers.illingworth import three_phase as three_phase_module
from moving_boundary.solvers.illingworth._interstitials import has_interstitials, x_to_u_independent
from test_illingworth_history_contract import (
    _FAMILIES,
    _TWO_PHASE,
    _CoupledTernaryThermodynamics,
    _CurvedInterfaceEquilibrium,
    _accepted,
    _assert_rows,
    _freeze,
    _histories,
    _track_commits,
)
from test_illingworth_interstitials_three_phase import _model as _three_phase_interstitial_model


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
    """A repeated reset and solve equals an untouched model's reset and solve, and a freshly constructed model."""
    build, step = _FAMILIES[family]
    model.reset()
    model.solve(2 * step, minDtFrac=1e-14)
    untouched = _two_step_model(family)
    untouched.reset()
    untouched.solve(2 * step, minDtFrac=1e-14)

    _assert_same_result(model, untouched)
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


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_successful_reset_then_solve_equals_a_fresh_model(family):
    """Preservation for the shared fixtures (their conversion is exact for binary/two-phase; the inexact
    profiles are covered by the reset-parity tests below)."""
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


# R10 reset-parity follow-ups: reset has the constructor's initialization semantics.
#
# Profile: three-phase and two-phase ternary reset record the constructor's u-fraction
# conversion of ``mesh.y`` (``_recordInitialMeshComposition``). Before, they recorded the raw
# ``mesh.y`` (no interstitials) or ``record_initial_u_fractions`` (interstitials), which differ
# from the constructor's conversion in the last bits for some profiles (the "inexact" fixtures);
# such tests fail before the fix for the family at the initial-profile bytes. Binary reset is
# unchanged: with one independent component both paths give identical bits (``fl(1 - x) + x``
# rounds to exactly 1, and the interstitial divisor is ``1 - x`` either way), so its fixtures are
# preservation evidence, as are the ternary fixtures whose conversion is exact.
#
# Eta (two-phase ternary): setup stores its estimate in ``initialEta``; reset recorded that stale
# estimate as ``etaData`` row 0 instead of the constructor's unestimated NaN. Reset now restores
# ``initialEta = NaN`` first (configured initialization inputs are untouched).


def _ternary_substitutional_model(left):
    """The history-contract ternary fixture with left composition ``left`` (right as there)."""
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(
        ProfileBuilder([(StepProfile1D(0.45, np.asarray(left), np.asarray([0.3261538461538461, 0.193])), ["X", "Y"])]),
        boundaryConditions=MixedBoundary1D(2),
    )
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["ALPHA", "BETA"],
        thermodynamics=_CoupledTernaryThermodynamics(), temperature=1000.0, interfacePosition=0.45,
        interface_equilibrium=_CurvedInterfaceEquilibrium(), initial_eta_bracket=(0.0, 1.0),
        time_step=1.0e-2, tolerance=1.0e-11, max_iterations=50, record=True,
    )


def _ternary_interstitial_model(x_left):
    """Fe-C-Mn (C interstitial) two-phase model of the ported interstitial tests, with left mole fractions ``x_left``."""
    mesh = CartesianFD1D(["C", "MN"], [0.0, _fe_c_mn_tests.DOMAIN_LENGTH], 400)
    x_right = _fe_c_mn_tests._x_from_u(_fe_c_mn_tests.U_RIGHT_FAR)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(_fe_c_mn_tests.S0, np.asarray(x_left), x_right), ["C", "MN"])]))
    u_grid, v_grid = two_phase_landau_grids(21, 21, method="linear", spacing_ratio=0.1)
    D_left, D_right = _fe_c_mn_tests._diffusivities(1.0e-2)
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh, elements=_fe_c_mn_tests.FE_C_MN, phases=["ALPHA", "BETA"],
        thermodynamics=FixedMatrixTernaryDiffusivity({"ALPHA": D_left, "BETA": D_right}, ["ALPHA", "BETA"]),
        temperature=1000.0, interfacePosition=_fe_c_mn_tests.S0, interface_equilibrium=_fe_c_mn_tests._closure(),
        initial_eta_bracket=(0.0, 1.0), time_step=1.0e-3, tolerance=1.0e-11, max_iterations=50,
        terminal_thin_phase_policy="continue", record=True, transformed_u_grid=u_grid, transformed_v_grid=v_grid,
        initial_eta_componentwise_scaling=True,
    )


def _ternary_interstitial_left(offset):
    return _fe_c_mn_tests._x_from_u(_fe_c_mn_tests.U_LEFT_FAR) + np.asarray([0.0, offset])


def _binary_interstitial_model():
    """Fe-C (C interstitial) binary model of the ported interstitial tests."""
    therm = _fe_c_tests._ConstantBinaryThermodynamics({"ALPHA": 1.0, "BETA": 0.5})
    return _fe_c_tests._binary_model(["FE", "C"], therm, StepProfile1D(1.0, 0.0123, 0.0377), (0.001, 0.02))


# fixture -> (family, builder, one step, inexact conversion)
_RESET_PARITY_FIXTURES = {
    "binary-substitutional": ("binary", lambda: _FAMILIES["binary"][0](True), _FAMILIES["binary"][1], False),
    "binary-interstitial": ("binary", _binary_interstitial_model, 0.05, False),
    "ternary-substitutional": ("ternary", lambda: _FAMILIES["ternary"][0](True), _FAMILIES["ternary"][1], False),
    "ternary-substitutional-inexact": ("ternary", lambda: _ternary_substitutional_model([0.176, 0.076]), 1.0e-2, True),
    "ternary-interstitial": ("ternary", lambda: _ternary_interstitial_model(_ternary_interstitial_left(0.0)), 1.0e-3, False),
    "ternary-interstitial-inexact": (
        "ternary", lambda: _ternary_interstitial_model(_ternary_interstitial_left(1.0e-5)), 1.0e-3, True,
    ),
    "three_phase-substitutional": ("three_phase", lambda: _FAMILIES["three_phase"][0](True), _FAMILIES["three_phase"][1], True),
    "three_phase-interstitial": ("three_phase", lambda: _three_phase_interstitial_model()[0], 1.0e-4, True),
}
# Fixtures whose reset re-records the initial profile through ``data.setResponseAtN`` (binary
# without interstitials keeps the host reset's raw record, which equals the constructor's).
_RECORDING_FIXTURES = sorted(name for name in _RESET_PARITY_FIXTURES if name != "binary-substitutional")
_TERNARY_FIXTURES = sorted(name for name, spec in _RESET_PARITY_FIXTURES.items() if spec[0] == "ternary")


def _initial_profile(model):
    return {
        "currentY": _freeze(np.asarray(model.data.currentY, dtype=np.float64)),
        "row0": _freeze(np.asarray(model.data._y[0], dtype=np.float64)),
        "mesh.y": _freeze(np.asarray(model.mesh.y, dtype=np.float64)),
    }


def _previous_reset_record(model):
    """What reset recorded before the follow-ups: raw ``mesh.y`` or, with interstitials, ``x_to_u_independent``."""
    x = np.asarray(model.mesh.flattenResponse(model.mesh.y), dtype=np.float64)
    if has_interstitials(model.allElements):
        return x_to_u_independent(x, model.allElements)
    return x


@pytest.mark.parametrize("fixture", sorted(_RESET_PARITY_FIXTURES))
def test_reset_parity_fixture_conversion_is_exact_or_inexact_as_declared(fixture):
    """Guards the fixtures: "inexact" ones really have a constructor conversion that differs in the last bits."""
    family, build, _, inexact = _RESET_PARITY_FIXTURES[fixture]
    fresh = build()
    constructor = np.asarray(fresh.data.currentY, dtype=np.float64).reshape(fresh.mesh.flattenResponse(fresh.mesh.y).shape)
    previous = _previous_reset_record(fresh)

    differing = int(np.sum(np.any(constructor != previous, axis=1)))
    assert (differing > 0) is inexact, f"{differing} nodes differ"
    np.testing.assert_allclose(constructor, previous, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("fixture", sorted(_RESET_PARITY_FIXTURES))
def test_reset_records_the_constructor_initial_profile(fixture):
    """After a run, reset (once and again) records exactly the fresh initial profile and ``data`` history.

    ``mesh.y`` is not modified by reset, and repeated resets give the same profile
    (the conversion is applied once to ``mesh.y``, never to an already converted profile).
    """
    _, build, step, _ = _RESET_PARITY_FIXTURES[fixture]
    fresh = build()
    model = build()
    model.solve(2 * step, minDtFrac=1e-14)
    mesh_before = _freeze(np.asarray(model.mesh.y, dtype=np.float64))

    for _ in range(2):
        model.reset()

        assert _freeze(np.asarray(model.mesh.y, dtype=np.float64)) == mesh_before
        assert _initial_profile(model) == _initial_profile(fresh)
        assert _freeze(dict(vars(model.data))) == _freeze(dict(vars(fresh.data)))


@pytest.mark.parametrize("fixture", sorted(_RESET_PARITY_FIXTURES))
def test_reset_restores_the_constructor_state(fixture):
    """After a run, reset (once and again) gives exactly the fresh state: histories, accepted state, time, ``isSetup``."""
    _, build, step, _ = _RESET_PARITY_FIXTURES[fixture]
    fresh = build()
    model = build()
    model.solve(2 * step, minDtFrac=1e-14)

    for _ in range(2):
        model.reset()

        _assert_same_result(model, fresh)


@pytest.mark.parametrize("fixture", sorted(_RESET_PARITY_FIXTURES))
def test_reset_then_solve_equals_a_fresh_model(fixture):
    """Run, reset, setup and solve (and again after two consecutive resets) reproduce a fresh model bit for bit."""
    _, build, step, _ = _RESET_PARITY_FIXTURES[fixture]
    model = build()
    model.solve(2 * step, minDtFrac=1e-14)

    model.reset()
    model.setup()
    fresh = build()
    fresh.setup()
    _assert_same_result(model, fresh)
    model.solve(3 * step, minDtFrac=1e-14)
    fresh.solve(3 * step, minDtFrac=1e-14)
    _assert_same_result(model, fresh)

    model.reset()
    model.reset()
    model.solve(2 * step, minDtFrac=1e-14)
    again = build()
    again.solve(2 * step, minDtFrac=1e-14)
    _assert_same_result(model, again)


def _assert_failed_reset_keeps_the_run_and_recovers(fixture, model, untouched, step):
    """Continuation equals an untouched continuation; a successful retry equals the untouched reset and a fresh model."""
    build = _RESET_PARITY_FIXTURES[fixture][1]
    model.solve(step, minDtFrac=1e-14)
    untouched.solve(step, minDtFrac=1e-14)
    _assert_same_result(model, untouched)

    model.reset()
    untouched.reset()
    model.solve(2 * step, minDtFrac=1e-14)
    untouched.solve(2 * step, minDtFrac=1e-14)
    fresh = build()
    fresh.solve(2 * step, minDtFrac=1e-14)
    _assert_same_result(model, untouched)
    _assert_same_result(model, fresh)


@pytest.mark.parametrize("fixture", _RECORDING_FIXTURES)
def test_reset_failing_while_recording_the_initial_profile_keeps_the_simulation(fixture):
    """R10a guarantees for the re-recording step: a failure there restores the run, which then continues and recovers."""
    _, build, step, _ = _RESET_PARITY_FIXTURES[fixture]
    model = build()
    model.solve(2 * step, minDtFrac=1e-14)
    model.data.setResponseAtN = _raise("recording the initial profile failed")
    before = _image(model)

    with pytest.raises(RuntimeError, match="recording the initial profile failed"):
        model.reset()

    changed = _image_changes(before, _image(model))
    del model.data.setResponseAtN
    assert changed == [], f"a failed reset changed {changed}"
    untouched = build()
    untouched.solve(2 * step, minDtFrac=1e-14)
    _assert_failed_reset_keeps_the_run_and_recovers(fixture, model, untouched, step)


_ETA_INPUTS = (
    "initialEtaMethod", "initialEtaBracket", "initialEtaGuess", "initialVelocityGuess", "initialEtaRootXtol",
    "initialEtaRootRtol", "initialEtaRootMaxiter", "initialEtaComponentwiseScaling",
)


def _eta_state(model):
    return {
        "initialEta": _freeze(float(model.initialEta)),
        "etaData": _freeze(dict(vars(model.etaData))),
        "getInterfaceEta": _freeze(np.asarray(model.getInterfaceEta(), dtype=np.float64)),
    }


def _eta_inputs(model):
    return _freeze({name: getattr(model, name) for name in _ETA_INPUTS})


@pytest.mark.parametrize("fixture", _TERNARY_FIXTURES)
def test_ternary_reset_restores_the_unestimated_initial_eta(fixture):
    """Before setup, ``initialEta``, ``etaData`` and ``getInterfaceEta()`` are NaN after reset, as after construction.

    Repeated resets keep that state, configured initialization inputs (method,
    bracket, guesses, root controls) are unchanged, and setup/solve still equals
    a fresh model.
    """
    _, build, step, _ = _RESET_PARITY_FIXTURES[fixture]
    fresh = build()
    model = build()
    configured = _eta_inputs(model)
    assert np.isnan(fresh.initialEta) and np.isnan(fresh.getInterfaceEta())
    model.solve(2 * step, minDtFrac=1e-14)
    assert np.isfinite(model.initialEta)

    for _ in range(2):
        model.reset()

        assert _eta_state(model) == _eta_state(fresh)
        assert _eta_inputs(model) == configured == _eta_inputs(fresh)

    model.solve(3 * step, minDtFrac=1e-14)
    fresh.solve(3 * step, minDtFrac=1e-14)
    _assert_same_result(model, fresh)
    assert _freeze(float(model.initialEta)) == _freeze(float(fresh.initialEta))


@pytest.mark.parametrize("fixture", _TERNARY_FIXTURES)
def test_ternary_reset_failing_after_restoring_the_initial_eta_keeps_the_estimate(fixture):
    """A failure later in reset (``inventoryData.reset``) restores ``initialEta`` and ``etaData`` with everything else."""
    _, build, step, _ = _RESET_PARITY_FIXTURES[fixture]
    model = build()
    model.solve(2 * step, minDtFrac=1e-14)
    estimated = _eta_state(model)
    model.inventoryData.reset = _raise("resetting the inventory history failed")
    before = _image(model)

    with pytest.raises(RuntimeError, match="resetting the inventory history failed"):
        model.reset()

    changed = _image_changes(before, _image(model))
    del model.inventoryData.reset
    assert changed == [], f"a failed reset changed {changed}"
    assert _eta_state(model) == estimated and np.isfinite(model.initialEta)
    untouched = build()
    untouched.solve(2 * step, minDtFrac=1e-14)
    _assert_failed_reset_keeps_the_run_and_recovers(fixture, model, untouched, step)


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


# R10 follow-up: interruptions (``BaseException``) during a commit, every family.
# The three-phase cases failed before the follow-up (its commit rollback caught
# only ``Exception``); binary and two-phase ternary already rolled back.

_REFUSAL = {
    "binary": "no new candidate has been solved",
    "ternary": "no new candidate has been solved",
    "three_phase": "without a pending converged nonlinear candidate",
}


class _InterruptingCoupledModel:
    def updateCoupledModel(self, model):
        raise KeyboardInterrupt("interrupted during the coupled update")


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_interrupt_during_coupled_update_rolls_back_and_refuses_resubmission(family):
    """A ``KeyboardInterrupt`` from a coupled model restores this model, consumes the candidate and propagates.

    The coupled model updated before the interruption keeps its entry (D6).
    """
    model = _stepping_model(family, steps=1)
    recorder = _Recorder()
    model.couplingModels = [recorder, _InterruptingCoupledModel()]
    proposal = _propose(model)
    before = _image(model)

    with pytest.raises(KeyboardInterrupt, match="interrupted during the coupled update"):
        _commit(model, proposal)

    changed = [name for name in _image_changes(before, _image(model)) if name not in _CONSUMED]
    assert changed == [], f"an interrupted commit changed {changed}"
    assert model._pendingCandidate is None
    assert len(recorder.entries) == 1, "the documented limitation: the external update is not undone"
    time, x = _update(proposal)
    with pytest.raises(RuntimeError, match=_REFUSAL[family]):
        model.postProcess(time, x)
    assert [name for name in _image_changes(before, _image(model)) if name not in _CONSUMED] == []


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_recovery_after_an_interrupted_commit_equals_an_untouched_model(family):
    """After the interruption, freshly solved candidates reproduce a model that was never interrupted."""
    model = _stepping_model(family, steps=1)
    model.couplingModels = [_InterruptingCoupledModel()]
    with pytest.raises(KeyboardInterrupt):
        _commit(model, _propose(model))
    model.couplingModels = []
    untouched = _stepping_model(family, steps=1)

    for _ in range(2):
        _commit(model, _propose(model))
        _commit(untouched, _propose(untouched))

    _assert_same_result(model, untouched)
    assert model._pendingCandidate is None


@pytest.mark.parametrize("family", ["ternary", "three_phase"])
def test_interrupted_commit_clears_the_terminal_thin_phase_stop(family):
    """A terminal thin-phase stop staged with the interrupted candidate is consumed with it; the run is not ended."""
    model = _stepping_model(family, steps=1)
    model.couplingModels = [_InterruptingCoupledModel()]
    proposal = _propose(model)
    final_time = model.finalTime
    model._terminalThinPhaseStop = True
    model._terminalThinPhaseInfo = {"phase": model.phases[0]}

    with pytest.raises(KeyboardInterrupt):
        _commit(model, proposal)

    assert model._terminalThinPhaseStop is False
    assert model._terminalThinPhaseInfo is None
    assert model.finalTime == final_time


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_successful_commits_with_a_coupled_model_are_unchanged(family):
    """Preservation: each commit updates the coupled model once with the accepted state and returns no stop."""
    model = _stepping_model(family)
    recorder = _Recorder()
    model.couplingModels = [recorder]
    plain = _stepping_model(family)

    for step in range(3):
        _, stop = _commit(model, _propose(model))
        _commit(plain, _propose(plain))
        assert stop is False
        assert len(recorder.entries) == step + 1
        assert recorder.entries[-1] == _freeze((float(model.currentTime), model.getCurrentX()))

    _assert_same_result(model, plain)
    assert model._pendingCandidate is None
