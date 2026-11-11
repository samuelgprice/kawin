"""R07: binary and two-phase ternary steps honour the outer Euler wrapper's timestep limits.

Each Illingworth ``getdXdt`` solves a complete implicit step and returns
``(new - old) / dt``. If the outer wrapper then commits a different ``dt``, the
committed state was never solved. These tests check that the outer maximum is
applied before the nonlinear solve, that a retry below the outer minimum is
rejected before the wrapper (the exact final remainder excepted), and that
``postProcess`` refuses an outer update that does not reproduce the converged
candidate. Three-phase has the same contract in
``test_illingworth_ternary_three_phase.py``.
"""

import numpy as np
import pytest

from kawin.diffusion.DiffusionParameters import TemperatureParameters
from moving_boundary.mesh import CartesianFD1D, MixedBoundary1D, ProfileBuilder, StepProfile1D
from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel, MovingBoundaryIllingworthTernaryFD1DModel


class _BinaryConstantDiffusivity:
    """Phase-uniform scalar diffusivities of the LEDGERS R07 binary reproduction."""

    def clearCache(self):
        pass

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        return np.squeeze(np.ones(np.atleast_1d(T).shape) * {"ALPHA": 1.0e-3, "BETA": 1.0e-1}[phase])


class _CoupledTernaryThermodynamics:
    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        if phase == "ALPHA":
            return np.asarray([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]], dtype=np.float64)
        return np.asarray([[7.0e-4, -1.0e-4], [2.0e-4, 1.1e-3]], dtype=np.float64)


class _CurvedInterfaceEquilibrium:
    """Curved, non-crossing eta-dependent tie-lines (as in ``test_illingworth_ternary``)."""

    eta_bounds = (0.0, 1.0)

    def interface_compositions(self, eta):
        eta = float(eta)
        bend = eta * (1.0 - eta)
        center = np.asarray([0.24 + 0.25 * eta, 0.15 - 0.015 * eta + 0.22 * bend], dtype=np.float64)
        derivative = np.asarray([0.25, -0.015 + 0.22 * (1.0 - 2.0 * eta)], dtype=np.float64)
        tangent = derivative / np.linalg.norm(derivative)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        tilt = np.deg2rad(55.0 - 30.0 * eta)
        direction = np.sin(tilt) * tangent + np.cos(tilt) * normal
        left = center - 0.042 * (1.0 + 0.12 * (2.0 * eta - 1.0)) * direction
        right = center + 0.055 * (1.0 - 0.10 * (2.0 * eta - 1.0)) * direction
        return left, right


def _binary_model(time_step, **kwargs):
    mesh = CartesianFD1D(["CR"], [0.0, 5.0], 501)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(1.0, 0.8, 0.4), "CR")]))
    options = dict(
        time_step=time_step, phase_a_nodes=100, phase_b_nodes=100, tolerance=2.0e-11,  # 1e-10 / R (R08)
        max_step_retries=15, record=True, bulk_diffusivity_mode="phase_uniform",
    )
    options.update(kwargs)
    return MovingBoundaryIllingworthFD1DModel(
        mesh, ["NI", "CR"], ["ALPHA", "BETA"], thermodynamics=_BinaryConstantDiffusivity(),
        temperature=TemperatureParameters(1000.0), interfacePosition=1.0,
        interface_compositions=(0.6, 0.0), **options,
    )


def _ternary_model(time_step, **kwargs):
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(
        ProfileBuilder([(StepProfile1D(0.45, np.asarray([0.16, 0.06]), np.asarray([0.3261538461538461, 0.193])), ["X", "Y"])]),
        boundaryConditions=MixedBoundary1D(2),
    )
    equilibrium = _CurvedInterfaceEquilibrium()
    options = dict(time_step=time_step, tolerance=1.0e-11, max_iterations=50, max_step_retries=15, record=True)
    options.update(kwargs)
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["ALPHA", "BETA"],
        thermodynamics=_CoupledTernaryThermodynamics(), temperature=1000.0, interfacePosition=0.45,
        interface_equilibrium=equilibrium, initial_eta_method="instantaneous_balance",
        initial_eta_bracket=equilibrium.eta_bounds, **options,
    )


# (builder, simulated time, scheduled step, outer maximum fraction, conservation bound)
_FAMILIES = {
    "binary": (_binary_model, 0.1, 0.1, 0.1, 1.0e-10),
    "ternary": (_ternary_model, 1.0e-2, 1.0e-2, 0.1, 1.0e-10),
}


def _relative_inventory_drift(model):
    initial = np.asarray(model._initialInventory, dtype=np.float64)
    current = np.asarray(model.getTotalInventoryFromState(*model.getCurrentX()[:3]), dtype=np.float64)
    return float(np.max(np.abs(current / initial - 1.0)))


def _accepted_history(model):
    """Accepted times, interface positions and final transformed state."""
    history = model.interfaceData
    return (
        np.asarray(history._time[: history.N + 1], dtype=np.float64),
        np.asarray(history._y[: history.N + 1], dtype=np.float64),
        [np.asarray(part, dtype=np.float64) for part in model.getCurrentX()],
    )


def _assert_same_run(actual, expected):
    for got, want in zip(actual[:2], expected[:2]):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(actual[2], expected[2]):
        np.testing.assert_array_equal(got, want)


def _fail_first_large_attempt(model, threshold):
    """Fails the first nonlinear attempt of each step when its trial dt is at least ``threshold``."""
    original_step = model._take_implicit_step_planar
    original_getdxdt = model.getdXdt
    first = {"attempt": True}
    trials = []

    def getdXdt(t, x):
        first["attempt"] = True
        return original_getdxdt(t, x)

    def step(*args):
        trials.append(float(args[-1]))
        if first["attempt"] and args[-1] >= threshold:
            first["attempt"] = False
            raise RuntimeError("forced first-attempt failure")
        first["attempt"] = False
        return original_step(*args)

    model.getdXdt = getdXdt
    model._take_implicit_step_planar = step
    return trials


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_outer_maximum_matches_run_configured_with_that_step(family):
    """A ``maxDtFrac`` clamp must take exactly the steps of a run scheduled at that step."""
    build, sim_time, time_step, max_frac, drift_bound = _FAMILIES[family]
    outer_max = max_frac * (sim_time - 0.0)  # the host wrapper's expression

    clamped = build(time_step)
    clamped.solve(sim_time, minDtFrac=1.0e-14, maxDtFrac=max_frac)
    explicit = build(outer_max)
    explicit.solve(sim_time, minDtFrac=1.0e-14)

    _assert_same_run(_accepted_history(clamped), _accepted_history(explicit))
    assert clamped.currentTime == pytest.approx(sim_time, rel=0.0, abs=1.0e-15)
    assert np.all(np.diff(_accepted_history(clamped)[0]) <= outer_max * (1.0 + 1.0e-12))
    assert _relative_inventory_drift(clamped) <= drift_bound


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_outer_maximum_with_retry_reduced_steps_matches_configured_run(family):
    """Retries start from the clamped step, so they match a run scheduled at that step."""
    build, sim_time, time_step, max_frac, drift_bound = _FAMILIES[family]
    outer_max = max_frac * (sim_time - 0.0)

    clamped = build(time_step)
    clamped_trials = _fail_first_large_attempt(clamped, 0.75 * outer_max)
    clamped.solve(sim_time, minDtFrac=1.0e-14, maxDtFrac=max_frac)
    explicit = build(outer_max)
    explicit_trials = _fail_first_large_attempt(explicit, 0.75 * outer_max)
    explicit.solve(sim_time, minDtFrac=1.0e-14)

    assert clamped_trials == explicit_trials
    assert max(clamped_trials) == outer_max
    _assert_same_run(_accepted_history(clamped), _accepted_history(explicit))
    assert _relative_inventory_drift(clamped) <= drift_bound


def _replay_accepted_steps(build, times):
    """Re-solves each accepted step with one unclamped solve of exactly that duration.

    The scheduled step is set to each duration, so the step is neither split nor
    treated as a near-final no-op (which triggers below ``time_step * 1e-10``).
    """
    replay = build(float(times[1] - times[0]))
    states = []
    for start, end in zip(times[:-1], times[1:]):
        replay.timeStep = float(end - start)
        replay.solve(float(end - start), minDtFrac=1.0e-14)
        states.append([np.asarray(part, dtype=np.float64) for part in replay.getCurrentX()])
    return replay, states


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_semi_log_schedule_with_outer_maximum_commits_solved_steps(family):
    """Every committed semi-log step equals an implicit solve over the committed duration."""
    build, sim_time, _time_step, max_frac, drift_bound = _FAMILIES[family]
    outer_max = max_frac * sim_time
    semi_log = dict(dt_mode="semi_log", semiLogT0=0.01 * sim_time, semiLog_dt=0.5)

    model = build(sim_time, **semi_log)
    model.solve(sim_time, minDtFrac=1.0e-14, maxDtFrac=max_frac)
    times, interfaces, final_state = _accepted_history(model)

    steps = np.diff(times)
    assert np.max(steps) <= outer_max * (1.0 + 1.0e-12)
    # The clamp is active: the unclamped schedule has gaps longer than the outer maximum.
    unclamped = build(sim_time, **semi_log)
    unclamped.setTimeInfo(0.0, sim_time)
    assert np.max(np.diff(np.concatenate(([0.0], unclamped._semiLogTimes)))) > outer_max
    # Every semi-log target is still reached exactly.
    for target in unclamped._semiLogTimes:
        assert np.min(np.abs(times - target)) <= 1.0e-15

    _replay, states = _replay_accepted_steps(build, times)
    np.testing.assert_allclose([state[2] for state in states], interfaces[1:], rtol=1.0e-9, atol=0.0)
    for got, want in zip(final_state, states[-1]):
        np.testing.assert_allclose(got, want, rtol=1.0e-9, atol=1.0e-12)
    assert _relative_inventory_drift(model) <= drift_bound


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_retry_below_outer_minimum_is_rejected_before_wrapper(family):
    build, sim_time, time_step, _max_frac, _drift_bound = _FAMILIES[family]
    model = build(time_step)
    original_step = model._take_implicit_step_planar
    trials = []

    def fail_until_below_minimum(*args):
        trials.append(float(args[-1]))
        if len(trials) <= 3:
            raise RuntimeError("forced retry")
        return original_step(*args)

    model._take_implicit_step_planar = fail_until_below_minimum
    with pytest.raises(ValueError, match="smaller than the outer solver minimum"):
        model.solve(sim_time, minDtFrac=0.5)

    assert trials == [time_step, time_step / 2, time_step / 4, time_step / 8]
    assert model.currentTime == 0.0
    assert model.interfaceData.N == 0
    assert model._pendingCandidate is None
    if family == "ternary":
        assert model._pendingStepProjection is None


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_final_remainder_below_outer_minimum_is_solved_and_accepted(family):
    build, _sim_time, _time_step, _max_frac, drift_bound = _FAMILIES[family]
    step = 1.0e-4 if family == "binary" else 1.0e-5
    final_time = 1.2 * step

    model = build(step)
    model.solve(final_time, minDtFrac=0.25)
    times = _accepted_history(model)[0]
    assert times[1] == step
    assert times[-1] == pytest.approx(final_time, rel=1.0e-14)
    assert times[-1] - times[-2] < 0.25 * final_time  # the final remainder is below the outer minimum
    # The remainder was solved, not stretched: replaying it from the same state reproduces it.
    _replay, states = _replay_accepted_steps(build, times)
    np.testing.assert_allclose(model.getCurrentX()[2], states[-1][2], rtol=1.0e-12, atol=0.0)

    # Continuation after a tiny final remainder.
    model.solve(step, minDtFrac=1.0e-14)
    assert model.currentTime == pytest.approx(final_time + step, rel=1.0e-14)
    assert _relative_inventory_drift(model) <= drift_bound


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_pending_candidate_rejects_outer_mismatch_without_commit(family):
    build, sim_time, time_step, _max_frac, _drift_bound = _FAMILIES[family]
    model = build(time_step)
    model.setup()
    model.setTimeInfo(0.0, sim_time)
    x = model.getCurrentX()
    before = [np.asarray(part, dtype=np.float64).copy() for part in x]
    dXdt = model.getdXdt(0.0, x)
    dt = model.getDt(dXdt)
    half = 0.5 * dt
    x_half = [np.asarray(x[i], dtype=np.float64) + np.asarray(dXdt[i], dtype=np.float64) * half for i in range(len(x))]

    with pytest.raises(RuntimeError, match="does not match the pending converged"):
        model.postProcess(half, x_half)

    assert model.currentTime == 0.0
    assert model.interfaceData.N == 0
    assert model._pendingCandidate is None
    for got, want in zip(model.getCurrentX(), before):
        np.testing.assert_array_equal(np.asarray(got, dtype=np.float64), want)


# --- Commit-guard follow-ups: large time origins and the rejection lifecycle ----------------

_LARGE_TIME = 1.0e9


def _proposed_step(family, start_time=0.0, time_step=1.0e-4):
    """Returns ``(model, x, dXdt, dt)`` for one proposed (uncommitted) step starting at ``start_time``."""
    build = _FAMILIES[family][0]
    model = build(time_step)
    model.setup()
    model.setTimeInfo(start_time, 1.0)
    model.currentTime = start_time
    x = model.getCurrentX()
    dXdt = model.getdXdt(start_time, x)
    return model, x, dXdt, model.getDt(dXdt)


def _euler_update(x, dXdt, dt):
    return [np.asarray(x[i], dtype=np.float64) + np.asarray(dXdt[i], dtype=np.float64) * dt for i in range(len(x))]


def _accepted_snapshot(model):
    """Accepted time, state, history lengths and (ternary) cumulative projection totals."""
    snapshot = {
        "time": model.currentTime,
        "state": [np.asarray(part, dtype=np.float64).copy() for part in model.getCurrentX()],
        "interface_N": model.interfaceData.N,
        "data_N": model.data.N,
    }
    if hasattr(model, "_cumulativeProjectedNodes"):
        snapshot["projection"] = (
            model._cumulativeProjectionInventoryChange.copy(),
            model._cumulativeProjectionAbsoluteChange.copy(),
            model._cumulativeProjectedNodes,
        )
    return snapshot


def _assert_not_advanced(model, snapshot):
    after = _accepted_snapshot(model)
    assert after["time"] == snapshot["time"]
    assert after["interface_N"] == snapshot["interface_N"]
    assert after["data_N"] == snapshot["data_N"]
    for got, want in zip(after["state"], snapshot["state"]):
        np.testing.assert_array_equal(got, want)
    if "projection" in snapshot:
        np.testing.assert_array_equal(after["projection"][0], snapshot["projection"][0])
        np.testing.assert_array_equal(after["projection"][1], snapshot["projection"][1])
        assert after["projection"][2] == snapshot["projection"][2]
        assert model._pendingStepProjection is None
    assert model._pendingCandidate is None


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_valid_update_at_large_start_time_is_committed(family):
    model, x, dXdt, dt = _proposed_step(family, start_time=_LARGE_TIME)
    model.postProcess(_LARGE_TIME + dt, _euler_update(x, dXdt, dt))
    assert model.currentTime == _LARGE_TIME + dt
    assert model.interfaceData.N == 1


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_solve_from_large_start_time_passes_verification(family):
    build = _FAMILIES[family][0]
    model = build(1.0e-4)
    model.setup()
    model.currentTime = _LARGE_TIME
    model.solve(1.0e-3, minDtFrac=1.0e-14)
    assert model.currentTime == pytest.approx(_LARGE_TIME + 1.0e-3, rel=0.0, abs=4.0 * np.spacing(_LARGE_TIME))
    assert model.interfaceData.N >= 10


@pytest.mark.parametrize("fraction", [0.5, 0.99, 1.01])
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_wrong_timestamp_at_large_start_time_is_rejected(family, fraction):
    """The full candidate state with a wrong elapsed time fails, however large the time origin."""
    model, x, dXdt, dt = _proposed_step(family, start_time=_LARGE_TIME)
    before = _accepted_snapshot(model)
    with pytest.raises(RuntimeError, match=r"does not match the pending converged .*timestep"):
        model.postProcess(_LARGE_TIME + fraction * dt, _euler_update(x, dXdt, dt))
    _assert_not_advanced(model, before)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_rejected_update_cannot_be_resubmitted_until_a_new_candidate_is_solved(family):
    model, x, dXdt, dt = _proposed_step(family)
    before = _accepted_snapshot(model)
    half = _euler_update(x, dXdt, 0.5 * dt)

    with pytest.raises(RuntimeError, match="does not match the pending converged"):
        model.postProcess(0.5 * dt, half)
    _assert_not_advanced(model, before)
    for attempt_time, attempt in ((0.5 * dt, half), (0.5 * dt, half), (dt, _euler_update(x, dXdt, dt))):
        # Repeated rejection, including the originally correct update: its candidate was discarded.
        with pytest.raises(RuntimeError, match="no new candidate has been solved"):
            model.postProcess(attempt_time, attempt)
        _assert_not_advanced(model, before)

    # Recovery: a freshly solved candidate commits normally.
    x = model.getCurrentX()
    dXdt = model.getdXdt(0.0, x)
    dt = model.getDt(dXdt)
    model.postProcess(dt, _euler_update(x, dXdt, dt))
    assert model.currentTime == dt
    assert model.interfaceData.N == 1
    assert model._pendingCandidate is None


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_malformed_update_rejects_the_proposal(family):
    """An exception while checking the update also rejects the proposal."""
    model, x, dXdt, dt = _proposed_step(family)
    before = _accepted_snapshot(model)
    with pytest.raises(IndexError):
        model.postProcess(dt, _euler_update(x, dXdt, dt)[:2])
    _assert_not_advanced(model, before)
    with pytest.raises(RuntimeError, match="no new candidate has been solved"):
        model.postProcess(dt, _euler_update(x, dXdt, dt))
    _assert_not_advanced(model, before)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_failed_or_subminimum_proposal_blocks_direct_commit(family):
    build, _sim_time, time_step, _max_frac, _drift_bound = _FAMILIES[family]
    model = build(time_step)
    model.setup()
    model.setTimeInfo(0.0, 1.0)
    x = model.getCurrentX()
    before = _accepted_snapshot(model)
    original_step = model._take_implicit_step_planar

    def always_fail(*args):
        raise RuntimeError("forced failure")

    model._take_implicit_step_planar = always_fail
    with pytest.raises((RuntimeError, ValueError)):
        model.getdXdt(0.0, x)
    with pytest.raises(RuntimeError, match="no new candidate has been solved"):
        model.postProcess(time_step, model.getCurrentX())
    _assert_not_advanced(model, before)

    # A converged retry below the outer minimum is rejected the same way.
    model._take_implicit_step_planar = original_step
    model._solveMinDtFrac = 0.5
    model.setTimeInfo(0.0, 4.0 * time_step)
    calls = []

    def fail_twice(*args):
        calls.append(args[-1])
        if len(calls) <= 2:
            raise RuntimeError("forced retry")
        return original_step(*args)

    model._take_implicit_step_planar = fail_twice
    with pytest.raises(ValueError, match="smaller than the outer solver minimum"):
        model.getdXdt(0.0, model.getCurrentX())
    with pytest.raises(RuntimeError, match="no new candidate has been solved"):
        model.postProcess(time_step / 4, model.getCurrentX())
    _assert_not_advanced(model, before)

    # Recovery through a newly solved candidate.
    model._take_implicit_step_planar = original_step
    model._solveMinDtFrac = 1.0e-14
    model.setTimeInfo(0.0, 1.0)
    x = model.getCurrentX()
    dXdt = model.getdXdt(0.0, x)
    dt = model.getDt(dXdt)
    model.postProcess(dt, _euler_update(x, dXdt, dt))
    assert model.currentTime == dt


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_direct_post_process_without_any_proposal_is_still_accepted(family):
    """Compatibility: a model that never proposed a step may commit directly (nothing to verify)."""
    build = _FAMILIES[family][0]
    model = build(1.0e-4)
    model.setup()
    model.postProcess(1.0e-4, model.getCurrentX())
    assert model.currentTime == 1.0e-4
    model.postProcess(2.0e-4, model.getCurrentX())
    assert model.currentTime == 2.0e-4


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_interrupted_commit_cannot_be_resubmitted_unchecked(family):
    """A commit that raises after verification consumed the candidate leaves commits refused.

    Rolling back what the interrupted commit already changed is LEDGERS R10; this only
    checks that the same update cannot then pass through the no-candidate exemption.
    """
    model, x, dXdt, dt = _proposed_step(family)
    update = _euler_update(x, dXdt, dt)

    def coupled_failure():
        raise RuntimeError("forced coupled-model failure")

    model.updateCoupledModels = coupled_failure
    with pytest.raises(RuntimeError, match="forced coupled-model failure"):
        model.postProcess(dt, update)
    del model.updateCoupledModels
    with pytest.raises(RuntimeError, match="no new candidate has been solved"):
        model.postProcess(dt, update)

    x = model.getCurrentX()
    dXdt = model.getdXdt(model.currentTime, x)
    step = model.getDt(dXdt)
    start = model.currentTime
    model.postProcess(start + step, _euler_update(x, dXdt, step))
    assert model.currentTime == start + step
    assert model._pendingCandidate is None


@pytest.mark.parametrize("bad_time", [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_non_finite_committed_time_is_rejected(family, bad_time):
    """A non-finite timestamp would otherwise make the ulp allowance infinite (inf) or the comparison undefined (NaN)."""
    model, x, dXdt, dt = _proposed_step(family)
    before = _accepted_snapshot(model)
    update = _euler_update(x, dXdt, dt)

    with pytest.raises(RuntimeError, match="non-finite time"):
        model.postProcess(bad_time, update)
    _assert_not_advanced(model, before)
    # The rejection latch stays active, even for the correct update.
    with pytest.raises(RuntimeError, match="no new candidate has been solved"):
        model.postProcess(dt, update)
    _assert_not_advanced(model, before)

    x = model.getCurrentX()
    dXdt = model.getdXdt(0.0, x)
    dt = model.getDt(dXdt)
    model.postProcess(dt, _euler_update(x, dXdt, dt))
    assert model.currentTime == dt
    assert model.interfaceData.N == 1


@pytest.mark.parametrize("field", ["start_time", "accepted_time", "committed_time", "dt"])
@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_pending_step_mismatches_flags_each_non_finite_time(field, bad):
    from moving_boundary.solvers.illingworth.shared import _PendingImplicitStep, _pending_step_mismatches

    times = {"start_time": 1.0, "accepted_time": 1.0, "committed_time": 1.5, "dt": 0.5}
    times[field] = bad
    pending = _PendingImplicitStep(start_time=times["start_time"], dt=times["dt"], state=(np.zeros(2),), scales=(1.0,))
    mismatches = _pending_step_mismatches(pending, times["accepted_time"], times["committed_time"], (np.zeros(2),), ("profile",))
    assert mismatches == ["non-finite time"]


def test_pending_step_mismatches_keeps_finite_four_ulp_allowance():
    from moving_boundary.solvers.illingworth.shared import _PendingImplicitStep, _pending_step_mismatches

    start, dt = 1.0e9, 1.0e-4
    pending = _PendingImplicitStep(start_time=start, dt=dt, state=(np.zeros(1),), scales=(1.0,))
    allowance = 4.0 * np.spacing(start + dt)
    assert allowance == pytest.approx(4.77e-7, rel=1.0e-3)
    for committed, expected in (
        (start + dt, []),
        (np.nextafter(start + dt, np.inf), []),
        (start + dt + 3.0 * allowance, ["timestep"]),
    ):
        assert _pending_step_mismatches(pending, start, committed, (np.zeros(1),), ("profile",)) == expected
