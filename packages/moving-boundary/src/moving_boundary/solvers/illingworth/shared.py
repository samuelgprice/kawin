# Complete shared helper section from samuelgprice/kawin fae6149ccc6ebfda2100c4ed5a9b99b962e77620 (R06 three-phase correction on top of main 720d122), MovingBoundaryIllingworthTernaryFDM.py
import functools
import math
import warnings
from dataclasses import dataclass

import numpy as np
from moving_boundary._spectral_validation import TERNARY_DIFFUSIVITY_POSITIVE_EIGENVALUE_TOL, TERNARY_DIFFUSIVITY_REAL_SPECTRUM_TOL

from moving_boundary.compatibility import GenericModel
from moving_boundary.compatibility import DiffusionModel
from moving_boundary.compatibility.recording import (
    _advance_history_rows,
    _finalize_history_rows,
    _history_row_target,
)
from kawin.diffusion.DiffusionParameters import TemperatureParameters
from moving_boundary.solvers.illingworth._interstitials import (
    composition_bound_violation,
    full_mole_fractions,
    has_interstitials,
    independent_interstitial_mask,
    record_initial_u_fractions,
    resolve_diffusivity_provider,
    resolve_interface_equilibrium,
    validate_interstitial_element_order,
)
from moving_boundary.equilibrium import (
    CallableTernaryInterfaceEquilibrium,
    ThermodynamicTernaryInterfaceEquilibrium,
)
from moving_boundary.mesh import CartesianFD1D, MixedBoundary1D, PeriodicBoundary1D
from moving_boundary.solvers.illingworth.numerics.ternary import (
    _fill_ternary_left_planar_interior,
    _fill_ternary_right_planar_interior,
    _solve_illingworth_block_tridiagonal_two_phase as solve_illingworth_block_tridiagonal,
    flatten_1d_coordinates,
    integrate_planar_transformed_profile_components,
    reconstruct_planar_transformed_profile_components,
)
from kawin.solver import explicitEulerIterator

_BULK_DIFFUSIVITY_PHASE_UNIFORM = "phase_uniform"
_BULK_DIFFUSIVITY_LAGGED = "composition_dependent_lagged"
_BULK_DIFFUSIVITY_IMPLICIT = "composition_dependent_implicit"
_SUPPORTED_BULK_DIFFUSIVITY_MODES = {
    _BULK_DIFFUSIVITY_PHASE_UNIFORM,
    _BULK_DIFFUSIVITY_LAGGED,
    _BULK_DIFFUSIVITY_IMPLICIT,
}
# Multi-start initial-eta roots whose interface compositions differ by more
# than this are treated as distinct roots of the startup balance, not round-off.
_INITIAL_ETA_DISTINCT_ROOT_COMPOSITION_TOL = 2e-5
# With componentwise initial-eta scaling, each component scale is floored at
# ``_COMPONENTWISE_ROUNDOFF_FACTOR * eps / root_rtol`` times the largest one.
# A component whose balance terms all (nearly) vanish, e.g. a non-partitioning
# slow solute at t = 0, is then required to balance to round-off of the
# largest terms instead of to a relative tolerance of ~0, which is unattainable.
_COMPONENTWISE_ROUNDOFF_FACTOR = 64.0

def _loge_arange(start, stop, log_step):
    """Returns exponentially spaced target times with fixed natural-log spacing."""
    logs = np.arange(np.log(start), np.log(stop), log_step)
    return np.exp(logs)


# Outer-timestep contract (LEDGERS R07) for the binary and two-phase ternary
# models. Each ``getdXdt`` solves a complete implicit step for one dt and returns
# ``(new - old) / dt``; the explicit-Euler wrapper must commit exactly that dt.
# Three-phase implements the same contract in its own methods.
_OUTER_UPDATE_RTOL = 2e-13
_OUTER_UPDATE_ULPS = 64.0
# Timestamps: allowance, in units in the last place of the largest time involved,
# for the elapsed step recovered from the wrapper's ``start + dt``.
_OUTER_TIME_ULPS = 4.0


@dataclass(frozen=True)
class _PendingImplicitStep:
    """
    A converged implicit step that ``postProcess`` has not committed yet.

    ``state`` holds the candidate in ``getCurrentX`` order, after the same
    interface-node overrides that ``postProcess`` applies. ``scales`` gives,
    per entry, the magnitude used for the absolute round-off allowance of the
    outer Euler update ``old + ((new - old) / dt) * dt``.
    """

    start_time: float
    dt: float
    state: tuple
    scales: tuple


def _is_recorded_history(value):
    """True for the Illingworth/``MeshData`` history objects (they carry ``_y``, ``_time`` and ``recordInterval``)."""
    return hasattr(value, "_y") and hasattr(value, "_time") and hasattr(value, "recordInterval")


def _snapshot_model_for_rollback(model):
    """
    Captures everything ``_restore_model_after_failure`` needs to undo a failed ``setup``/``reset``.

    That is every attribute binding of ``model`` (``vars(model)``) and, for each
    history object reachable from it (directly or in a list/tuple such as the
    three-phase ``profileData``), its attribute bindings plus a copy of the
    contents of every array it holds. Copying contents rather than rebinding to
    copies keeps object identity and aliasing (e.g. a ``currentY`` that is a view
    of ``_y``) intact. Other referenced objects (mesh, provider, closures,
    temperature, coupled models) are external and only their bindings are kept.
    """
    bindings = dict(vars(model))
    histories = {}
    for value in bindings.values():
        for candidate in value if isinstance(value, (list, tuple)) else (value,):
            if _is_recorded_history(candidate) and id(candidate) not in histories:
                attributes = dict(vars(candidate))
                contents = [(array, array.copy()) for array in attributes.values() if isinstance(array, np.ndarray)]
                histories[id(candidate)] = (candidate, attributes, contents)
    return bindings, tuple(histories.values())


def _restore_model_after_failure(model, snapshot):
    """Restores the bindings and history contents captured by ``_snapshot_model_for_rollback``."""
    bindings, histories = snapshot
    for history, attributes, contents in histories:
        vars(history).clear()
        vars(history).update(attributes)
        for array, saved in contents:
            if array.flags.writeable:
                array[...] = saved
    vars(model).clear()
    vars(model).update(bindings)


def _all_or_nothing(skip_when_setup=False):
    """
    Makes a model lifecycle method (``setup``/``reset``) all-or-nothing (LEDGERS R10a).

    If the method raises (any exception, including from the host
    ``DiffusionModel`` part or a provider), every model-owned field is restored
    to its value before the call and the exception is re-raised: time,
    ``isSetup``, accepted state, every history (rows, capacity and row
    bookkeeping), the pending candidate and rejection latch, and diagnostics.
    The method body is unchanged, so a successful call computes exactly as
    before. External state is not restored: provider state and caches (e.g. a
    partially cleared cache), closure bindings assigned on first use,
    ``mesh.boundaryConditions`` created when absent, temperature parameters,
    coupled models and printed diagnostics.

    With ``skip_when_setup``, an already set-up model calls the method directly
    without a snapshot (``setup`` is then a no-op, and ``solve`` calls it on
    every call).
    """
    def decorate(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if skip_when_setup and getattr(self, "isSetup", False):
                return method(self, *args, **kwargs)
            snapshot = _snapshot_model_for_rollback(self)
            try:
                return method(self, *args, **kwargs)
            except BaseException:
                _restore_model_after_failure(self, snapshot)
                raise

        return wrapper

    return decorate


# Atomic history recording (LEDGERS R10b). Moved from the three-phase model
# with unchanged behavior and shared by all three families.


def _history_value_shape(history):
    """Shape of one recorded value: ``shape`` (shared arrays), ``yShape`` (``MeshData``), ``(n_components,)`` (binary p/q), else scalar."""
    if hasattr(history, "shape"):
        return tuple(history.shape)
    if hasattr(history, "yShape"):
        return tuple(history.yShape)
    if hasattr(history, "n_components"):
        return (int(history.n_components),)
    return ()


def _history_record_target(history):
    """Returns the row a regular history record will write (shared R09 row rule), or ``None``."""
    return _history_row_target(history)


def _ensure_history_record_capacity(history, target):
    """
    Ensures one history target row exists before an atomic record sequence.

    Grows by ``max(batchSize, missing rows)`` zero rows, which for an
    append-only target is exactly the growth ``record`` itself would apply, so
    recorded values and capacity are unchanged.
    """
    if target is None or target < history._time.shape[0]:
        return
    growth = max(int(getattr(history, "batchSize", 1000)), target + 1 - history._time.shape[0])
    history._time = np.pad(history._time, (0, growth))
    history._y = np.pad(history._y, ((0, growth), *[(0, 0) for _ in range(history._y.ndim - 1)]))


def _record_histories_atomically(time, records):
    """
    Records ``(history, value)`` pairs at ``time`` as one unit and returns the snapshots.

    Value shapes are checked before anything changes. Each history's
    bookkeeping (``N``, cadence counter, row counter, stored-row flag,
    ``currentY``/``currentTime``), the row it is about to write and its
    capacity are snapshotted; if growing capacity or any ``record`` raises,
    every history is restored (``_restore_history_snapshots``) and the
    exception re-raised. The caller passes the returned snapshots to
    ``_restore_history_snapshots`` if a later part of its commit fails.
    """
    prepared = []
    for history, value in records:
        values = np.asarray(value, dtype=np.float64)
        expected_shape = _history_value_shape(history)
        if values.shape != tuple(expected_shape):
            raise ValueError(f"Expected history value with shape {tuple(expected_shape)}, got {values.shape}.")
        target = _history_record_target(history)
        prepared.append((history, value, target))

    snapshots = []
    for history, _, target in prepared:
        old_capacity = int(history._time.shape[0])
        row = None if target is None or target >= old_capacity else history._y[target].copy()
        row_time = None if target is None or target >= old_capacity else float(history._time[target])
        current_y = np.asarray(history.currentY).copy() if np.ndim(history.currentY) else float(history.currentY)
        snapshots.append(
            (
                history,
                target,
                row,
                row_time,
                history.currentIndex,
                history.N,
                current_y,
                history.currentTime,
                old_capacity,
                history._rowCount,
                history._lastRecordStored,
            )
        )
    try:
        for history, _, target in prepared:
            _ensure_history_record_capacity(history, target)
        for history, value, _ in prepared:
            history.record(time, value)
    except BaseException:
        _restore_history_snapshots(snapshots)
        raise
    return snapshots


def _restore_history_snapshots(snapshots):
    """Restores metadata (including the row counter and stored-row flag), written rows, and capacity."""
    for (
        history, target, row, row_time, current_index, index, current_y, current_time, old_capacity,
        row_count, last_record_stored,
    ) in snapshots:
        if target is not None and target < old_capacity:
            history._y[target] = row
            history._time[target] = row_time
        if history._time.shape[0] != old_capacity:
            history._time = history._time[:old_capacity].copy()
            history._y = history._y[:old_capacity].copy()
        history.currentIndex = current_index
        history.N = index
        history.currentY = current_y
        history.currentTime = current_time
        history._rowCount = row_count
        history._lastRecordStored = last_record_stored


def _finalize_histories_atomically(histories):
    """
    Finalizes every history (``history.finalize()``, the shared R09 row rule) as one unit.

    Targeted snapshot per history, no whole-history copy: the original
    ``_y``/``_time`` arrays (finalize truncates by rebinding, so keeping them
    restores the capacity), a copy of the row a forced final record would
    write, ``N``, the cadence counter, the row counter, the stored-row flag and
    the ``currentY``/``currentTime`` bindings. If any finalization raises,
    every history is restored and the exception re-raised; calling the
    finalization again is the retry and is idempotent once it succeeds.
    """
    snapshots = []
    for history in histories:
        target = None
        if history.recordInterval <= 0 or not history._lastRecordStored:
            target = _history_row_target(history, force=True)
        y, times = history._y, history._time
        inside = target is not None and target < times.shape[0]
        snapshots.append(
            (
                history, y, times, target if inside else None,
                y[target].copy() if inside else None, float(times[target]) if inside else None,
                history.N, history.currentIndex, history._rowCount, history._lastRecordStored,
                history.currentY, history.currentTime,
            )
        )
    try:
        for history in histories:
            history.finalize()
    except BaseException:
        for (
            history, y, times, target, row, row_time, index, current_index, row_count, last_record_stored,
            current_y, current_time,
        ) in snapshots:
            if target is not None:
                y[target] = row
                times[target] = row_time
            history._y = y
            history._time = times
            history.N = index
            history.currentIndex = current_index
            history._rowCount = row_count
            history._lastRecordStored = last_record_stored
            history.currentY = current_y
            history.currentTime = current_time
        raise


def _validate_outer_dt_fractions(min_dt_frac, max_dt_frac, model_name):
    """Rejects outer timestep fractions that no implicit step can satisfy."""
    if not np.isfinite(min_dt_frac) or not np.isfinite(max_dt_frac) or min_dt_frac < 0.0 or max_dt_frac <= 0.0:
        raise ValueError("minDtFrac must be nonnegative and maxDtFrac must be positive and finite.")
    if min_dt_frac > max_dt_frac:
        raise ValueError(f"minDtFrac cannot exceed maxDtFrac for {model_name}.")


_ZERO_INVENTORY_RESIDUAL_SCALE_FACTOR = 64.0 * np.finfo(float).eps


def _componentwise_residual_scale(inventory, domain_length, composition_floor):
    """
    Returns the default normalization of a ternary interface residual, per component (ledger R13).

    ``scale_k = max(|inventory_k|, domain_length * composition_floor)``: the
    rule of the binary model (``binary._binary_residual_scale``, ledger R08)
    applied to each independent component. ``inventory`` is the accepted old
    inventory the residual rows sum to (transformed for the two-phase model,
    density-weighted for the three-phase one), ``domain_length`` the length
    term ``L`` in the same units, and ``composition_floor`` is
    ``constraints.minComposition``. The residual check
    ``|r_k| <= residual_tolerance * scale_k`` is therefore relative to the
    component's own inventory while ``|inventory_k| >= L * floor``, and an
    absolute change of mean composition of ``residual_tolerance * floor``
    below that.

    Only when a component's inventory and the floor are both exactly zero
    (possible only with ``minComposition = 0``) is its scale
    ``64 eps * domain_length``, an absolute round-off-level bound. A positive
    floor whose product with the length underflows to zero is not that case.

    Raises ``ValueError`` naming the quantity for a non-finite inventory, a
    non-finite or non-positive length, a non-finite or negative floor, and a
    scale that overflows or underflows to a non-positive value. Nothing is
    replaced by a substitute value; in particular there is no ``1e-300``
    clamp here (explicit ``interface_residual_composition_scale`` values keep
    their own legacy expression in the models).
    """
    inventory = np.asarray(inventory, dtype=np.float64)
    domain_length = float(domain_length)
    composition_floor = float(composition_floor)
    if not np.all(np.isfinite(inventory)):
        raise ValueError(f"Illingworth residual normalization: the old inventory {inventory!r} is not finite.")
    if not np.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError(f"Illingworth residual normalization: the domain length {domain_length!r} must be positive and finite.")
    if not np.isfinite(composition_floor) or composition_floor < 0.0:
        raise ValueError(
            f"Illingworth residual normalization: the composition floor {composition_floor!r} must be non-negative and finite."
        )
    scale = np.maximum(np.abs(inventory), domain_length * composition_floor)
    if composition_floor == 0.0:
        scale = np.where(inventory == 0.0, domain_length * _ZERO_INVENTORY_RESIDUAL_SCALE_FACTOR, scale)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(
            f"Illingworth residual normalization: the scale {scale!r} (inventory {inventory!r}, domain length "
            f"{domain_length!r}, composition floor {composition_floor!r}) must be positive and finite."
        )
    return scale


def _explicit_residual_composition_scale(value):
    """
    Returns an explicit ``interface_residual_composition_scale`` as two validated ``float64`` values.

    ``value`` may be a scalar or anything that broadcasts to two values (a
    list, tuple or array), all positive and finite; anything else raises
    ``ValueError`` naming the argument. The result is a local read-only view
    for the caller's arithmetic: the model attribute is neither replaced nor
    converted, so a value assigned after construction is used exactly as the
    same value passed to the constructor, and captured settings are unchanged.
    """
    try:
        scale = np.broadcast_to(np.asarray(value, dtype=np.float64), (2,))
    except (TypeError, ValueError):
        raise ValueError("interface_residual_composition_scale must be a scalar or two values.") from None
    if not (np.all(np.isfinite(scale)) and np.all(scale > 0.0)):
        raise ValueError("interface_residual_composition_scale must be positive and finite.")
    return scale


def _validate_default_residual_floor(model):
    """
    Rejects a ``constraints.minComposition`` the default residual normalization cannot use (ledger R13).

    Only when no explicit ``interface_residual_composition_scale`` is set does
    the per-step residual scale read ``constraints.minComposition``
    (``_componentwise_residual_scale``); it must then be finite and
    non-negative. ``constraints`` exists only after the host constructor, so
    the two ternary models call this from their configuration validators and
    at the start of ``solve``, not before construction. It changes nothing.
    """
    if model.interfaceResidualCompositionScale is not None:
        return
    floor = float(model.constraints.minComposition)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError(
            "constraints.minComposition must be non-negative and finite: it is the composition floor of the "
            "default interface residual normalization."
        )


def _validate_ternary_numerical_controls(model):
    """
    Rejects invalid timestep and convergence controls of a ternary Illingworth model (LEDGERS R12).

    Reads only ``timeStep``, ``dtMode``, ``semiLog_dt``, ``semiLogT0``,
    ``tolerance``, the stored effective ``residualTolerance`` and
    ``interfaceResidualCompositionScale``; it changes
    nothing, so the two-phase and three-phase models call it before the host
    constructor, inside their configuration validators and at the start of
    ``solve`` (before any bookkeeping, setup or schedule allocation).

    An explicit ``interface_residual_composition_scale`` must be a scalar or
    two values, all positive and finite (ledger R13: the accepted values are
    unchanged; they are now refused here, also when the attribute was changed
    after construction, instead of only by the configuration validator).

    ``time_step``, ``tolerance`` and ``residual_tolerance`` must always be
    positive and finite; ``tolerance`` is checked even when an explicit residual
    tolerance and Picard rtol leave it unused. The semi-log spacing and first
    target are checked only when ``dt_mode='semi_log'``: unused values in fixed
    mode are ignored. Picard tolerances keep their own checks.
    """
    if not np.isfinite(model.timeStep) or model.timeStep <= 0:
        raise ValueError("time_step must be a positive finite value.")
    if model.dtMode not in {"fixed", "semi_log"}:
        raise ValueError("dt_mode must be 'fixed' or 'semi_log'.")
    if not np.isfinite(model.tolerance) or model.tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite value.")
    if not np.isfinite(model.residualTolerance) or model.residualTolerance <= 0.0:
        raise ValueError("residual_tolerance must be a positive finite value.")
    if model.dtMode == "semi_log":
        if model.semiLog_dt is None or model.semiLogT0 is None:
            raise ValueError("semiLog_dt and semiLogT0 must be specified when dt_mode is 'semi_log'.")
        if not np.isfinite(model.semiLog_dt) or model.semiLog_dt <= 0.0:
            raise ValueError("semiLog_dt must be a positive finite value.")
        if not np.isfinite(model.semiLogT0) or model.semiLogT0 <= 0.0:
            raise ValueError("semiLogT0 must be a positive finite value.")
    if model.interfaceResidualCompositionScale is not None:
        _explicit_residual_composition_scale(model.interfaceResidualCompositionScale)


def _outer_timestep_limits(min_dt_frac, max_dt_frac, initial_time, final_time):
    """
    Returns the outer wrapper's ``(minimum, maximum)`` timestep for one ``solve``.

    Uses the host expression ``fraction * (finalTime - initialTime)``, so the
    model's pre-solve clamp equals the wrapper's to the last bit. At the default
    ``maxDtFrac=1`` the maximum equals the remaining time of the first step,
    which the schedule already applies, so default runs are unchanged.
    """
    span = float(final_time) - float(initial_time)
    return float(min_dt_frac) * span, float(max_dt_frac) * span


def _is_outer_final_remainder(trial_dt, remaining, outer_minimum):
    """
    Returns whether ``trial_dt`` is the exact final remainder below the outer minimum.

    Mirrors the exemption in ``compatibility.solver.DESolver._getdXdt``, so the
    model and the wrapper accept the same subminimum steps.
    """
    remaining = float(remaining)
    return bool(
        remaining < outer_minimum
        and math.isclose(
            float(trial_dt),
            remaining,
            rel_tol=2e-13,
            abs_tol=64.0 * math.ulp(1.0) * max(1.0, abs(remaining)),
        )
    )


def _check_trial_dt_against_outer_minimum(trial_dt, remaining, outer_minimum):
    """
    Rejects a converged step below the outer minimum unless it is the final remainder.

    The wrapper would refuse such a step anyway; rejecting it here happens
    before the step is staged for commit and names the remedy.
    """
    if trial_dt < outer_minimum and not _is_outer_final_remainder(trial_dt, remaining, outer_minimum):
        raise ValueError(
            f"Converged nonlinear timestep {trial_dt} is smaller than the outer solver minimum "
            f"{outer_minimum}; reduce minDtFrac."
        )


def _pending_step_mismatches(pending, accepted_time, committed_time, state, names):
    """
    Lists where the outer Euler update differs from the converged candidate.

    Time: the check is on the elapsed step, ``(committed_time - start) - dt``,
    not on the absolute timestamp, so a relative tolerance on a large time
    origin cannot hide a wrong step. The wrapper forms ``start + dt`` in
    floating point, which resolves the step only to about half an ulp of the
    result, and the subtraction adds at most another ulp. The allowance is
    therefore ``4 ulp`` of the largest of the start, committed and accepted
    times and ``dt``. It grows only with that resolution: about 4.77e-7 at
    t = 1e9, 3.55e-15 at t = 4, and 4 ulp of ``dt`` from t = 0. A step that
    cannot be resolved at all at the time origin also cannot be detected. The
    accepted start time must match the candidate's start to the same allowance.
    A non-finite start, accepted or committed time or ``dt`` is reported as
    ``"non-finite time"`` before the allowance is formed, because ``ulp(inf)``
    is infinite and would accept any timestamp; the time checks are then skipped.

    State: each entry ``i`` is compared with ``rtol=2e-13`` and
    ``atol = 64 eps * pending.scales[i]``, the round-off of
    ``old + ((new - old) / dt) * dt``.
    """
    eps = np.finfo(float).eps
    start = float(pending.start_time)
    committed = float(committed_time)
    accepted = float(accepted_time)
    mismatches = []
    if not all(math.isfinite(value) for value in (start, committed, accepted, float(pending.dt))):
        mismatches.append("non-finite time")
    else:
        time_atol = _OUTER_TIME_ULPS * max(
            math.ulp(abs(start)), math.ulp(abs(committed)), math.ulp(abs(accepted)), math.ulp(abs(pending.dt))
        )
        if not abs(accepted - start) <= time_atol:
            mismatches.append("accepted start time")
        if not abs((committed - start) - float(pending.dt)) <= time_atol:
            mismatches.append("timestep")
    for name, actual, expected, scale in zip(names, state, pending.state, pending.scales):
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=_OUTER_UPDATE_RTOL, atol=_OUTER_UPDATE_ULPS * eps * float(scale)
        ):
            mismatches.append(name)
    return mismatches


def _magnitude(*values, floor=1.0):
    """Returns ``max(floor, max |v|)`` over the given arrays or scalars."""
    return max([float(floor)] + [float(np.max(np.abs(np.asarray(v, dtype=np.float64)))) for v in values])


def _matvec_2x2(matrix, vector):
    """Multiplies a 2x2 matrix by a length-2 vector without BLAS dispatch."""
    matrix = np.asarray(matrix, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    return np.asarray(
        [
            matrix[0, 0] * vector[0] + matrix[0, 1] * vector[1],
            matrix[1, 0] * vector[0] + matrix[1, 1] * vector[1],
        ],
        dtype=np.float64,
    )


def _bounded_finite_difference_perturbation(x, lower, upper, variable, step=None):
    """
    Returns the baseline one-sided finite-difference perturbation for one variable.

    The step-size formula and forward/backward bound choice are shared by the
    initial-eta and finite-step interface Newton loops. ``step`` overrides the
    default ``sqrt(eps) * max(1, |x|)`` size; it is still limited to a quarter
    of the bound range.
    """
    x = np.asarray(x, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if step is None:
        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(x[variable]))
    if np.isfinite(upper[variable] - lower[variable]):
        step = min(step, 0.25 * max(upper[variable] - lower[variable], 1e-15))
    x_perturbed = x.copy()
    if x[variable] + step <= upper[variable]:
        x_perturbed[variable] += step
        return step, x_perturbed, "forward"
    x_perturbed[variable] -= step
    return step, x_perturbed, "backward"


def _newton_step_2x2(jacobian, residual):
    """Returns the exact 2x2 Newton step used by the local ternary solves. Essentially solves the equation -Jx=r where J is 2x2 jacobian, 2x1 unknown, and r is 2x1 residual"""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64).reshape(2)
    a = float(jacobian[0, 0])
    b = float(jacobian[0, 1])
    c = float(jacobian[1, 0])
    d = float(jacobian[1, 1])
    determinant = a * d - b * c
    if abs(determinant) <= 1e-300:
        raise RuntimeError("Ternary Newton Jacobian is singular.")
    r0 = -float(residual[0])
    r1 = -float(residual[1])
    return np.asarray([(d * r0 - b * r1) / determinant, (-c * r0 + a * r1) / determinant], dtype=np.float64)


def _condition_number_2x2(matrix):
    """
    Returns the 2-norm condition number of a 2x2 matrix in closed form.

    Used only for diagnostics, so it avoids LAPACK like the rest of the 2x2
    interface algebra. Singular or non-finite matrices give ``inf``.
    """
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])
    frob = a * a + b * b + c * c + d * d
    det = abs(a * d - b * c)
    if not np.isfinite(frob) or not np.isfinite(det) or det <= 0.0:
        return np.inf
    root = np.sqrt(max(frob * frob - 4.0 * det * det, 0.0))
    smax2 = 0.5 * (frob + root)
    smin2 = 0.5 * (frob - root)
    if smin2 <= 0.0:
        smin2 = det * det / smax2
    return float(np.sqrt(smax2 / smin2))


def _smallest_real_eigenvalue_2x2(matrix):
    """Returns the smallest real part of the eigenvalues of a 2x2 matrix (closed form)."""
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])
    half_trace = 0.5 * (a + d)
    disc = half_trace * half_trace - (a * d - b * c)
    if disc < 0.0:
        return half_trace
    return half_trace - np.sqrt(disc)


def _validate_motion_branch(motion_branch):
    """Validates the two fixed upwind branch labels used by the interface solve."""
    if motion_branch not in {"positive", "negative"}:
        raise ValueError("motion_branch must be 'positive' or 'negative'.")


def _allocate_ternary_block_system(n):
    """Allocates zeroed 2x2 block-tridiagonal arrays for one ternary phase solve."""
    lower = np.zeros((n, 2, 2), dtype=np.float64)
    diagonal = np.zeros((n, 2, 2), dtype=np.float64)
    upper = np.zeros((n, 2, 2), dtype=np.float64)
    rhs = np.zeros((n, 2), dtype=np.float64)
    return lower, diagonal, upper, rhs


_MOTION_BRANCH_ATOL = 1e-15

# Finite-difference step for the scaled tie-line coordinate (eta_hat in [0, 1])
# in the per-step interface Newton. The default sqrt(eps) step is too small
# when a residual row depends only weakly on eta and carries round-off noise:
# with interface cells of ~1e-17 m the carbon flux loses ~7 digits, and
# noise/step then swamps the true derivative (Fe-Mn-C, 3201-node austenite
# grid). 1e-6 keeps the step well inside one segment of a tabulated tie-line
# closure (hundreds of segments over eta_hat in [0, 1]), so the truncation error
# stays negligible. The s step is unchanged: it must stay small relative to the
# step displacement so it does not cross zero motion. The phase solves now work
# in deviation form (ledger R06), which removes most of that noise; the step is
# kept unchanged so the Jacobian is not retuned by the same change.
_INTERFACE_ETA_FD_STEP = 1.0e-6

# Policy threshold for treating a composition-floor violation of a
# deviation-form reconstruction ``profile = w + c`` as round-off of the shift
# (see ``_project_shift_roundoff_to_floor``). A component may be moved onto the
# floor only if it is short of it by at most
# ``_FLOOR_PROJECTION_ULPS * eps * (|c| + |w|)``. This is a fixed, conservative
# policy, not a proven bound on the solve error; larger violations are left
# for the composition validator to reject as before.
_FLOOR_PROJECTION_ULPS = 16.0
# Directed-rounding steps allowed to finish a dependent-component repair whose
# deficit is below half an ulp of the component that absorbs it.
_FLOOR_PROJECTION_MAX_NEXTAFTER_STEPS = 4


def _select_interface_motion_branch(s, old_s, future_s, atol=_MOTION_BRANCH_ATOL):
    """
    Selects the conservative interface upwind branch for one residual evaluation.

    The branch is determined first from ``future_s - s``. If that displacement
    is effectively zero, the previous accepted displacement ``s - old_s`` is
    used as a deterministic fallback; if both are effectively zero the positive
    branch is chosen. The returned string must be passed unchanged to the two
    phase solves and the interface residual for face-flux cancellation.
    """
    delta_s = float(future_s) - float(s)
    if delta_s > atol:
        return "positive"
    if delta_s < -atol:
        return "negative"
    previous_delta_s = float(s) - float(old_s)
    if previous_delta_s < -atol:
        return "negative"
    return "positive"


@dataclass(frozen=True)
class InitialEtaEstimate:
    """
    Diagnostics from initial tie-line selection.

    ``eta`` is the selected tie-line coordinate. ``velocity`` is the scalar
    interface velocity returned by the initialization strategy. ``method``
    identifies the strategy that produced the estimate. ``branch`` is the
    swept-inventory branch used by branch-aware estimators.
    ``residual_scale`` is the balance-term magnitude that the relative
    convergence tolerance was applied to (``nan`` when not provided).
    """

    eta: float
    residual_norm: float
    velocity: float
    residual: np.ndarray
    flux_delta: np.ndarray
    left_interface_composition: np.ndarray
    right_interface_composition: np.ndarray
    method: str
    solver: str
    bracket: tuple[float, float]
    converged: bool
    iterations: int
    function_calls: int
    branch: str | None = None
    residual_scale: float = np.nan


@dataclass(frozen=True, slots=True)
class _InterfaceCandidate:
    """
    Mutually consistent state from one fixed-branch interface trial.

    The concentration profiles, interface compositions, diffusivity matrices,
    branch, and residuals all come from the same trial ``x_hat``. Keeping them
    grouped avoids mixing values from different nonlinear iterations.
    ``frozen=True`` is shallow: fields cannot be rebound, but NumPy array
    contents remain mutable.
    """

    x_hat: np.ndarray
    future_s: float
    future_eta: float
    motion_branch: str
    c_left: np.ndarray
    c_right: np.ndarray
    D_left: np.ndarray
    D_right: np.ndarray
    p_future: np.ndarray
    q_future: np.ndarray
    residual: np.ndarray
    scaled_residual: np.ndarray
    scaled_norm: float
    physical_norm: float
    left_inner_iterations: int = 0
    right_inner_iterations: int = 0
    left_inner_update_norm: float = 0.0
    right_inner_update_norm: float = 0.0
    bulk_diffusivity_evaluations: int = 0
    left_projection: "_FloorProjection | None" = None
    right_projection: "_FloorProjection | None" = None


@dataclass(frozen=True, slots=True)
class _FloorProjection:
    """
    Record of the round-off floor projection applied to one phase solve.

    ``inventory_change`` is the signed componentwise change of the phase
    inventory, ``sum_i weight_i * (projected - unprojected)_i`` with the
    new-geometry trapezoidal weights; the interface residual adds it so that
    the residual describes the profile that is stored. ``absolute_change`` is
    the same sum of absolute changes (a diagnostic of numerical intervention).
    ``mask`` is ``None`` when nothing was projected, otherwise an
    ``(n_nodes, 3)`` boolean array marking projected independent components
    (columns 0, 1) and repaired dependent components (column 2). ``repairs``
    lists ``(node, absorbing_component, nextafter_steps)`` for each dependent
    repair.
    """

    inventory_change: np.ndarray
    absolute_change: np.ndarray
    mask: np.ndarray | None = None
    repairs: tuple = ()

    @property
    def projected_nodes(self):
        """Number of nodes with at least one projected component."""
        return 0 if self.mask is None else int(np.count_nonzero(np.any(self.mask, axis=1)))


def _read_only_zeros(n):
    values = np.zeros(n, dtype=np.float64)
    values.setflags(write=False)
    return values


# Shared record for the (usual) case of a phase solve with nothing to project;
# its arrays are read-only so the shared instance cannot be modified in place.
_NO_FLOOR_PROJECTION = _FloorProjection(_read_only_zeros(2), _read_only_zeros(2))


def _no_floor_projection():
    """Returns the (shared, read-only) record of a phase solve that needed no floor projection."""
    return _NO_FLOOR_PROJECTION


def _profile_within_floor_bounds(profile, substitutional, floor):
    """
    Fast exact screen: returns True when no node of ``profile`` violates the floor.

    Uses the same arithmetic as ``composition_bound_violation``: each component
    against ``floor``, and the dependent component ``1 - sum(substitutional)``.
    For two substitutional columns the validator's row sum of a length-2 row is
    exactly ``p0 + p1``; for one, ``min(1 - x) == 1 - max(x)`` because the
    rounded subtraction is monotone. A ``False`` result (including NaN input)
    only means the exact per-node path must run.
    """
    if not profile.min() >= floor:
        return False
    count = int(np.count_nonzero(substitutional))
    if count == 2:
        return bool((1.0 - (profile[:, 0] + profile[:, 1])).min() >= floor)
    if count == 1:
        return bool(1.0 - profile[:, int(np.flatnonzero(substitutional)[0])].max() >= floor)
    return 1.0 >= floor


def _project_shift_roundoff_to_floor(profile, shift, deviation, interstitial_mask, min_composition, fixed_node, grid, length, weights=None):
    """
    Moves composition-floor violations caused by deviation-form round-off onto the floor.

    A deviation-form phase solve reconstructs ``profile = deviation + shift``,
    so every component carries an absolute round-off of about
    ``eps * (|shift| + |deviation|)`` regardless of its own size. ``shift`` is
    either one 2-vector (a constant reference, as in the two-phase model) or a
    per-node ``(n_nodes, 2)`` reference profile (the three-phase middle phase);
    the threshold below then uses ``|shift_ik|`` node by node. A dilute
    value just above ``min_composition`` beside a large interface composition
    can therefore land just below the floor, where the composition validator
    (``composition_bound_violation``, an exact comparison) would reject it at
    every timestep. This helper repairs only violations within the policy
    threshold ``_FLOOR_PROJECTION_ULPS * eps * (|shift_k| + |deviation_k|)``:

    - an independent component below the floor is set to the floor exactly;
    - a dependent substitutional component ``1 - sum(substitutional u)`` short
      of the floor (threshold summed over the substitutional components) has
      its deficit subtracted from the largest substitutional independent
      component at that node, followed by at most
      ``_FLOOR_PROJECTION_MAX_NEXTAFTER_STEPS`` directed-rounding steps when the
      deficit is below half an ulp of that component.

    A node is committed only if it then passes ``composition_bound_violation``
    with the production arithmetic; otherwise it is left unchanged and the
    validator rejects the candidate as before. ``fixed_node`` (the Dirichlet
    interface node, or a sequence of such nodes) is never modified.
    Interstitial components are bounded from below only. ``grid`` and
    ``length`` give the new-geometry trapezoidal weights used for the returned
    ``_FloorProjection.inventory_change``; a caller whose cell equations use
    other (bitwise different) node weights passes them as ``weights``
    instead, so the recorded change matches its own inventory.

    Returns ``(profile, projection)``; ``profile`` is the input array itself
    when nothing was projected. This is a controlled perturbation, unlike the
    deviation transformation itself, which is algebraically exact. The common
    case (no violation anywhere) is decided by ``_profile_within_floor_bounds``
    with the validator's exact arithmetic, so it costs a few vector reductions.
    """
    profile = np.asarray(profile, dtype=np.float64)
    floor = float(min_composition)
    interstitial_mask = np.asarray(interstitial_mask, dtype=bool)
    substitutional = ~interstitial_mask
    if _profile_within_floor_bounds(profile, substitutional, floor):
        return profile, _NO_FLOOR_PROJECTION
    n = profile.shape[0]
    dependent = 1.0 - np.sum(profile[:, substitutional], axis=1)
    flagged = np.flatnonzero(np.any(profile < floor, axis=1) | (dependent < floor))
    flagged = flagged[~np.isin(flagged, np.atleast_1d(np.asarray(fixed_node, dtype=np.intp)) % n)]
    if flagged.size == 0:
        return profile, _no_floor_projection()

    eps = np.finfo(np.float64).eps
    shift = np.abs(np.broadcast_to(np.asarray(shift, dtype=np.float64), profile.shape))
    deviation = np.asarray(deviation, dtype=np.float64)
    substitutional_indices = np.flatnonzero(substitutional)
    projected = profile.copy()
    mask = np.zeros((n, 3), dtype=bool)
    repairs = []
    for i in flagged:
        scale = shift[i] + np.abs(deviation[i])
        threshold = _FLOOR_PROJECTION_ULPS * eps * scale
        row = profile[i].copy()
        node_mask = np.zeros(3, dtype=bool)
        repair = None
        acceptable = True
        for k in range(2):
            if row[k] < floor:
                if floor - row[k] <= threshold[k]:
                    row[k] = floor
                    node_mask[k] = True
                else:
                    acceptable = False
        if acceptable and 1.0 - np.sum(row[substitutional]) < floor:
            dependent_threshold = _FLOOR_PROJECTION_ULPS * eps * float(np.sum(scale[substitutional]))
            deficit = floor - (1.0 - np.sum(row[substitutional]))
            if substitutional_indices.size == 0 or deficit > dependent_threshold:
                acceptable = False
            else:
                j = int(substitutional_indices[np.argmax(row[substitutional_indices])])
                trial = row.copy()
                trial[j] = row[j] - deficit
                steps = 0
                while 1.0 - np.sum(trial[substitutional]) < floor and steps < _FLOOR_PROJECTION_MAX_NEXTAFTER_STEPS:
                    trial[j] = np.nextafter(trial[j], -np.inf)
                    steps += 1
                if 1.0 - np.sum(trial[substitutional]) >= floor and abs(trial[j] - row[j]) <= dependent_threshold:
                    row = trial
                    node_mask[2] = True
                    repair = (int(i), j, steps)
                else:
                    acceptable = False
        if acceptable and composition_bound_violation(row[np.newaxis, :], interstitial_mask, floor) is None:
            projected[i] = row
            mask[i] = node_mask
            if repair is not None:
                repairs.append(repair)

    if not np.any(mask):
        return profile, _no_floor_projection()
    if weights is None:
        grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        weights = np.zeros(n, dtype=np.float64)
        half_widths = 0.5 * np.diff(grid)
        weights[:-1] += half_widths
        weights[1:] += half_widths
        weights *= float(length)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(n)
    change = projected - profile
    return projected, _FloorProjection(
        inventory_change=weights @ change,
        absolute_change=weights @ np.abs(change),
        mask=mask,
        repairs=tuple(repairs),
    )


@dataclass(frozen=True, slots=True)
class _BulkPhaseSolveResult:
    """
    Result from one transformed ternary bulk phase solve.

    ``interface_flux`` is the physical diffusive flux at the face adjacent to
    the moving interface, computed from the same face matrix used in the linear
    finite-volume assembly. ``interface_deviation`` is the solved deviation of
    the interface-adjacent node from the trial interface composition
    (``w[-2]`` on the left, ``w[1]`` on the right); the flux and the interface
    residual must use it rather than recomputing ``profile - c``, which loses
    the digits below the rounding unit of ``c`` (ledger R06). ``projection``
    records the round-off floor projection applied to ``profile`` (see
    ``_project_shift_roundoff_to_floor``); its ``inventory_change`` belongs to
    this returned profile. Both are required so that every return path,
    including the Picard exits, must forward them.
    """

    profile: np.ndarray
    interface_flux: np.ndarray
    interface_face_matrix: np.ndarray
    interface_deviation: np.ndarray
    projection: _FloorProjection
    inner_iterations: int = 0
    inner_update_norm: float = 0.0
    converged: bool = True
    failure_reason: str | None = None
    diffusivity_evaluations: int = 0
    diffusivity_provider_calls: int = 0
    face_matrices_evaluated: int = 0


def _validate_ternary_diffusivity_matrix(D, phase, context="ternary Illingworth diffusivity"):
    """
    Validates a ternary 2x2 diffusion matrix with scale-invariant eigen tests.

    Eigenvalues are computed from ``D / ||D||_inf`` so acceptance does not
    depend on diffusivity units. ``imaginary_tol`` and ``positive_tol`` are
    dimensionless tolerances on those scaled eigenvalues; values at or below the
    positivity tolerance are treated as nonpositive. The original unscaled
    matrix is returned after validation.
    """
    values = np.asarray(D)
    label = f"{context} for phase {phase}"
    if values.shape != (2, 2):
        raise ValueError(f"{label} must have shape (2, 2); received {values.shape}.")
    if np.iscomplexobj(values) and np.any(np.imag(values) != 0.0):
        raise ValueError(f"{label} must be real-valued.")
    values = np.asarray(np.real(values), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values.")

    matrix_norm = float(np.linalg.norm(values, ord=np.inf))
    if not np.isfinite(matrix_norm) or matrix_norm <= 0.0:
        raise ValueError(f"{label} must have a positive finite matrix norm.")
    scaled_eigenvalues = np.linalg.eigvals(values / matrix_norm)
    imaginary_tol = TERNARY_DIFFUSIVITY_REAL_SPECTRUM_TOL
    positive_tol = TERNARY_DIFFUSIVITY_POSITIVE_EIGENVALUE_TOL
    if np.any(np.abs(np.imag(scaled_eigenvalues)) > imaginary_tol):
        raise ValueError(f"{label} must have real positive eigenvalues; scaled eigenvalues={scaled_eigenvalues}.")
    real_eigenvalues = np.real(scaled_eigenvalues)
    if np.any(real_eigenvalues <= positive_tol):
        raise ValueError(f"{label} must have strictly positive real eigenvalues; scaled eigenvalues={scaled_eigenvalues}.")
    return values


def _validate_ternary_diffusivity_stack(matrices, phase, context="bulk face diffusivity face"):
    """Validate 2x2 face matrices in a batch, using scalar checks for exceptional rows.

    The common path applies the scalar validator's scale-invariant spectral
    thresholds to a stack. A failed or threshold-adjacent batch is checked in
    face order by the scalar validator to preserve its first error and message.
    """
    values = np.asarray(matrices)
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("bulk diffusivity query returned an unexpected matrix shape.")
    if not len(values):
        return np.empty(values.shape, dtype=np.float64)

    def scalar_reference():
        out = np.empty(values.shape, dtype=np.float64)
        for i, matrix in enumerate(values):
            out[i] = _validate_ternary_diffusivity_matrix(matrix, phase, context=f"{context} {i}")
        return out

    if values.dtype.kind not in "biufc":
        return scalar_reference()
    if np.iscomplexobj(values) and np.any(np.imag(values) != 0.0):
        return scalar_reference()
    real = np.asarray(np.real(values), dtype=np.float64)
    if not np.all(np.isfinite(real)):
        return scalar_reference()
    norms = np.max(np.sum(np.abs(real), axis=2), axis=1)
    if not np.all(np.isfinite(norms) & (norms > 0.0)):
        return scalar_reference()
    try:
        eigenvalues = np.linalg.eigvals(real / norms[:, None, None])
    except np.linalg.LinAlgError:
        return scalar_reference()
    imaginary = np.abs(np.imag(eigenvalues))
    positive = np.real(eigenvalues)
    margin = 64.0 * np.finfo(np.float64).eps
    if (not np.all(np.isfinite(eigenvalues))
            or np.any(imaginary >= TERNARY_DIFFUSIVITY_REAL_SPECTRUM_TOL - margin)
            or np.any(positive <= TERNARY_DIFFUSIVITY_POSITIVE_EIGENVALUE_TOL + margin)):
        return scalar_reference()
    return real.copy()


def _select_converged_branch(branch_results):
    """
    Returns the best converged record from the initial-eta motion-branch solves.

    Records are ``(raw_norm, branch, success, nfev, velocity, eta, residual,
    flux_delta, c_left, c_right, residual_scale, selection_norm)``. Records
    with a non-finite eta are discarded, then only converged records are
    considered, and the one with the smallest ``selection_norm`` (the
    convergence metric of the active scaling mode) is returned. An
    unconverged branch is therefore never chosen over a converged one, even
    when its raw residual is smaller.
    """
    finite = [record for record in branch_results if np.all(np.isfinite(record[5]))]
    if len(finite) == 0:
        raise ValueError("Instantaneous initial eta solve did not produce a finite residual.")
    converged = [record for record in finite if record[2]]
    if len(converged) == 0:
        raise ValueError("Instantaneous initial eta solve failed to converge.")
    return min(converged, key=lambda record: record[11])


def _validate_eta_bounds(interface_equilibrium):
    eta_bounds = getattr(interface_equilibrium, "eta_bounds", None)
    if eta_bounds is None or len(eta_bounds) != 2:
        raise ValueError("interface_equilibrium must expose finite eta_bounds for automatic initial tie-line selection.")
    lower, upper = tuple(float(v) for v in eta_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("interface_equilibrium eta_bounds must be finite and non-collapsed.")
    return lower, upper


def _coerce_initial_eta_bracket(eta_bracket, eta_bounds):
    lower, upper = eta_bounds
    if eta_bracket is None:
        return (lower, upper)
    values = np.asarray(eta_bracket, dtype=np.float64).reshape(-1)
    if values.size != 2:
        raise ValueError("initial eta bracket must contain exactly two values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("initial eta bracket must be finite.")
    bracket_lower, bracket_upper = tuple(float(v) for v in values)
    if bracket_upper <= bracket_lower:
        raise ValueError("initial eta bracket must be strictly increasing.")
    tol = 1e-12 * max(1.0, abs(lower), abs(upper))
    if bracket_lower < lower - tol or bracket_upper > upper + tol:
        raise ValueError("initial eta bracket must lie within interface_equilibrium eta_bounds.")
    return (float(np.clip(bracket_lower, lower, upper)), float(np.clip(bracket_upper, lower, upper)))


def _temperature_at_interface(temperature, interface_position):
    temperature_parameters = TemperatureParameters(temperature)
    values = np.asarray(
        temperature_parameters(np.asarray([float(interface_position)], dtype=np.float64), 0.0),
        dtype=np.float64,
    ).reshape(-1)
    if values.size == 0 or not np.isfinite(values[0]):
        raise ValueError("temperature must evaluate to a finite value at the initial interface.")
    return float(values[0])


def _get_stefan_interdiffusivity(thermodynamics, composition, temperature, phase):
    try:
        D = thermodynamics.getInterdiffusivity(composition, temperature, phase=phase, query_context="interface")
    except TypeError:
        D = thermodynamics.getInterdiffusivity(composition, temperature, phase=phase)
    return _validate_ternary_diffusivity_matrix(D, phase, context="initial-eta diffusivity")


def _get_bulk_interdiffusivity(thermodynamics, composition, temperature, phase, context="bulk diffusivity"):
    """Returns a validated general/bulk ternary interdiffusivity matrix."""
    try:
        D = thermodynamics.getInterdiffusivity(composition, temperature, phase=phase, query_context="general")
    except TypeError:
        D = thermodynamics.getInterdiffusivity(composition, temperature, phase=phase)
    return _validate_ternary_diffusivity_matrix(D, phase, context=context)


def _coerce_bulk_diffusivity_mode(mode):
    """Normalizes and validates the ternary Illingworth bulk diffusivity mode."""
    value = str(mode)
    if value not in _SUPPORTED_BULK_DIFFUSIVITY_MODES:
        raise ValueError(
            "bulk_diffusivity_mode must be 'phase_uniform', "
            "'composition_dependent_lagged', or 'composition_dependent_implicit'."
        )
    return value


def estimate_initial_eta_from_instantaneous_balance(
    composition,
    z,
    interface_position,
    phases,
    thermodynamics,
    temperature,
    interface_equilibrium,
    transformed_u_grid,
    transformed_v_grid,
    eta_bracket=None,
    eta_guess=None,
    velocity_guess=None,
    root_xtol=None,
    root_maxiter=100,
    bulk_diffusivity_mode=_BULK_DIFFUSIVITY_PHASE_UNIFORM,
    root_rtol=1e-10,
    componentwise_scaling=False,
):
    """
    Estimates initial eta by solving the instantaneous discrete balance.

    This method keeps the initial geometry and non-interface transformed
    concentrations fixed at ``s0`` and solves for ``(V0, eta0)`` in
    ``V0 * L(eta) - (G_B(eta; s0) - G_A(eta; s0)) = 0``. The branch-specific
    swept-inventory coefficient ``L`` matches the positive- and negative-motion
    branches used by the planar ternary residual in the finite-step solver.

    Both motion branches are solved; the converged branch with the smallest
    residual in the active scaling's norm is returned (see
    ``_select_converged_branch``), and ``ValueError`` is raised only when no
    branch converges.

    The residual has units of composition times velocity and, for realistic
    diffusivities in SI units, can be as small as ``1e-11``. Convergence is
    therefore tested relative to the size of the balance terms at the current
    iterate: ``max|residual| <= max(root_xtol, root_rtol * scale)`` with
    ``scale = max(|G_A|, |G_B|, |V0| * |L|, |G(eta_hi) - G(eta_lo)|)``
    (componentwise max norms). The individual phase fluxes are used rather
    than their difference so the scale stays meaningful when the fluxes nearly
    balance, and the flux change across the eta bracket keeps it non-zero at a
    stationary equilibrium root, where the fluxes and ``V0`` all vanish. This
    makes the converged eta independent of the length unit, of the diffusivity
    magnitude, and of the starting guess. ``root_xtol`` is an optional absolute
    floor (``None`` disables it); ``residual_scale`` on the returned estimate
    records ``scale``.

    With ``componentwise_scaling=True`` each residual row is divided by its own
    scale (the same four terms taken per component, floored at
    ``64 * eps / root_rtol`` times the largest component scale, and frozen at
    the current Newton iterate for its Jacobian and line search) and
    convergence requires ``max|scaled residual| <= root_rtol``. The floor means
    a component is never asked to balance below round-off of the largest
    terms (about ``64 * eps`` of them), which keeps the test attainable when
    every balance term of one component nearly vanishes, e.g. a
    non-partitioning slow solute with a flat start. Use it when the two components have
    very different diffusivities (e.g. an interstitial and a substitutional
    solute): the shared max-norm scale is then set by the fast component and
    leaves the slow component's balance effectively unconverged.
    ``residual_scale`` on the returned estimate is then the largest
    component scale.

    ``composition`` and the closure's interface compositions must be in the
    same basis as the diffusion state (u-fractions when interstitials are
    present); ``thermodynamics`` is queried with that basis, so callers pass a
    u-fraction wrapper (see ``kawin.diffusion._illingworth_interstitials``)
    when the provider expects mole fractions.
    """
    composition = np.asarray(composition, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    u_grid = np.asarray(transformed_u_grid, dtype=np.float64).reshape(-1)
    v_grid = np.asarray(transformed_v_grid, dtype=np.float64).reshape(-1)
    if composition.ndim != 2 or composition.shape[1] != 2 or composition.shape[0] != z.size:
        raise ValueError("composition must have shape (n_nodes, 2) matching z.")
    if z.size < 3 or not np.all(np.diff(z) > 0.0):
        raise ValueError("z must be a strictly increasing 1D grid with at least three nodes.")
    if not np.isclose(z[0], 0.0):
        raise ValueError("estimate_initial_eta_from_instantaneous_balance expects a 1D domain starting at 0.")
    if u_grid.size < 3 or v_grid.size < 3:
        raise ValueError("transformed grids must each contain at least three nodes.")
    if not np.isclose(u_grid[0], 0.0) or not np.isclose(u_grid[-1], 1.0) or not np.all(np.diff(u_grid) > 0.0):
        raise ValueError("transformed_u_grid must be strictly increasing from 0 to 1.")
    if not np.isclose(v_grid[0], 0.0) or not np.isclose(v_grid[-1], 1.0) or not np.all(np.diff(v_grid) > 0.0):
        raise ValueError("transformed_v_grid must be strictly increasing from 0 to 1.")
    if len(phases) != 2:
        raise ValueError("phases must contain exactly the left and right phases.")
    if thermodynamics is None or not hasattr(thermodynamics, "getInterdiffusivity"):
        raise TypeError("thermodynamics must provide getInterdiffusivity for initial eta estimation.")
    bulk_diffusivity_mode = _coerce_bulk_diffusivity_mode(bulk_diffusivity_mode)
    relative_tolerance = float(root_rtol)
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("root_rtol must be positive and finite.")
    absolute_tolerance = 0.0 if root_xtol is None else float(root_xtol)
    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("root_xtol must be None or a non-negative finite value.")

    s = float(interface_position)
    domain_length = float(z[-1] - z[0])
    if not (0.0 < s < domain_length):
        raise ValueError("interface_position must lie strictly inside the domain.")
    bracket = _coerce_initial_eta_bracket(eta_bracket, _validate_eta_bounds(interface_equilibrium))
    if eta_guess is None:
        eta0 = 0.5 * (bracket[0] + bracket[1])
    else:
        eta0 = float(eta_guess)
        if eta0 < bracket[0] or eta0 > bracket[1]:
            raise ValueError("eta_guess must lie within the initial eta bracket.")
    temperature_value = _temperature_at_interface(temperature, s)

    left_mask = z <= s
    right_mask = z >= s
    if not np.any(left_mask) or not np.any(right_mask):
        raise ValueError("Initial interface leaves an empty phase.")
    u_adjacent = float(u_grid[-2])
    v_adjacent = float(v_grid[1])
    z_left_adjacent = s * u_adjacent
    z_right_adjacent = s + (domain_length - s) * v_adjacent
    p_adjacent = np.asarray(
        [np.interp(z_left_adjacent, z[left_mask], composition[left_mask, component]) for component in range(2)],
        dtype=np.float64,
    )
    q_adjacent = np.asarray(
        [np.interp(z_right_adjacent, z[right_mask], composition[right_mask, component]) for component in range(2)],
        dtype=np.float64,
    )

    def evaluate_terms(eta, branch):
        c_left, c_right = interface_equilibrium.interface_compositions(float(eta))
        c_left = np.asarray(c_left, dtype=np.float64).reshape(2)
        c_right = np.asarray(c_right, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(c_left)) or not np.all(np.isfinite(c_right)):
            raise ValueError("interface compositions are non-finite at the queried eta.")
        if bulk_diffusivity_mode == _BULK_DIFFUSIVITY_PHASE_UNIFORM:
            D_left = _get_stefan_interdiffusivity(thermodynamics, c_left, temperature_value, phases[0])
            D_right = _get_stefan_interdiffusivity(thermodynamics, c_right, temperature_value, phases[1])
        else:
            D_left = _get_bulk_interdiffusivity(
                thermodynamics,
                0.5 * (p_adjacent + c_left),
                temperature_value,
                phases[0],
                context="initial-eta left interface-face diffusivity",
            )
            D_right = _get_bulk_interdiffusivity(
                thermodynamics,
                0.5 * (c_right + q_adjacent),
                temperature_value,
                phases[1],
                context="initial-eta right interface-face diffusivity",
            )
        G_left = _matvec_2x2(D_left, (c_left - p_adjacent) / (s * (1.0 - u_adjacent)))
        G_right = _matvec_2x2(D_right, (q_adjacent - c_right) / ((domain_length - s) * v_adjacent))
        flux_delta = G_right - G_left
        if branch == "positive":
            swept_inventory = c_left - q_adjacent * (1.0 - v_adjacent / 2.0) - c_right * v_adjacent / 2.0
        elif branch == "negative":
            swept_inventory = p_adjacent * ((1.0 + u_adjacent) / 2.0) + c_left * ((1.0 - u_adjacent) / 2.0) - c_right
        else:
            raise ValueError("branch must be 'positive' or 'negative'.")
        if not np.all(np.isfinite(flux_delta)) or not np.all(np.isfinite(swept_inventory)):
            raise ValueError("instantaneous balance terms are non-finite at the queried eta.")
        # The individual phase fluxes (not their difference) feed the relative
        # convergence scale, so it stays non-degenerate when the fluxes balance.
        side_fluxes = np.vstack((G_left, G_right))
        return swept_inventory, flux_delta, c_left.copy(), c_right.copy(), side_fluxes

    def eta_sensitivity_scale():
        # Change of the phase fluxes across the eta bracket. It does not vanish
        # at a stationary equilibrium root (where the fluxes and V0 all go to
        # zero) and makes the relative test an eta accuracy of roughly
        # ``root_rtol`` times the bracket width.
        try:
            low = evaluate_terms(bracket[0], "positive")[4]
            high = evaluate_terms(bracket[1], "positive")[4]
        except ValueError:
            return 0.0
        return float(np.max(np.abs(high - low)))

    def eta_sensitivity_components():
        # Per-component version of ``eta_sensitivity_scale``.
        try:
            low = evaluate_terms(bracket[0], "positive")[4]
            high = evaluate_terms(bracket[1], "positive")[4]
        except ValueError:
            return np.zeros(2, dtype=np.float64)
        return np.max(np.abs(high - low), axis=0)

    def component_scales_of(velocity, swept_inventory, side_fluxes):
        scales = np.maximum.reduce(
            [
                np.max(np.abs(side_fluxes), axis=0),
                abs(float(velocity)) * np.abs(swept_inventory),
                component_sensitivity,
                np.full(2, 1e-300),
            ]
        )
        return np.maximum(scales, component_scale_floor * float(np.max(scales)))

    def residual_scale_of(velocity, swept_inventory, side_fluxes):
        return max(
            float(np.max(np.abs(side_fluxes))),
            abs(float(velocity)) * float(np.max(np.abs(swept_inventory))),
            0.0 if sensitivity_scale is None else sensitivity_scale,
            1e-300,
        )

    def converged_at(norm, scale):
        # The bracket-edge sensitivity is evaluated lazily, only when the
        # cheaper scales do not already accept the iterate; the criterion is
        # the same either way.
        nonlocal sensitivity_scale
        if norm <= max(absolute_tolerance, relative_tolerance * scale):
            return True
        if sensitivity_scale is None:
            sensitivity_scale = eta_sensitivity_scale()
        return norm <= relative_tolerance * sensitivity_scale

    sensitivity_scale = None
    component_sensitivity = eta_sensitivity_components() if componentwise_scaling else None
    component_scale_floor = min(1.0, _COMPONENTWISE_ROUNDOFF_FACTOR * float(np.finfo(np.float64).eps) / relative_tolerance)

    def velocity_scale_for_branch(branch):
        swept_inventory, flux_delta, _, _, _ = evaluate_terms(eta0, branch)
        scale = float(np.linalg.norm(flux_delta) / max(float(np.linalg.norm(swept_inventory)), 1e-300))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return scale

    branch_results = []
    for branch in ("positive", "negative"):
        velocity_scale = velocity_scale_for_branch(branch)
        if velocity_guess is None:
            swept_inventory, flux_delta, _, _, _ = evaluate_terms(eta0, branch)
            scaled_velocity0 = float(np.dot(swept_inventory, flux_delta) / max(float(np.dot(swept_inventory, swept_inventory)), 1e-300))
            scaled_velocity0 /= velocity_scale
            if branch == "positive":
                scaled_velocity0 = abs(scaled_velocity0)
            else:
                scaled_velocity0 = -abs(scaled_velocity0)
        else:
            scaled_velocity0 = float(velocity_guess) / velocity_scale
            if branch == "positive" and scaled_velocity0 < 0.0:
                scaled_velocity0 = abs(scaled_velocity0)
            elif branch == "negative" and scaled_velocity0 > 0.0:
                scaled_velocity0 = -abs(scaled_velocity0)
        velocity_bounds = (0.0, np.inf) if branch == "positive" else (-np.inf, 0.0)

        def residual_unknowns(unknowns):
            velocity = float(unknowns[0]) * velocity_scale
            eta = float(unknowns[1])
            swept_inventory, flux_delta, _, _, side_fluxes = evaluate_terms(eta, branch)
            residual = velocity * swept_inventory - flux_delta
            if componentwise_scaling:
                return residual, component_scales_of(velocity, swept_inventory, side_fluxes)
            return residual, residual_scale_of(velocity, swept_inventory, side_fluxes)

        lower = np.asarray([velocity_bounds[0], bracket[0]], dtype=np.float64)
        upper = np.asarray([velocity_bounds[1], bracket[1]], dtype=np.float64)
        x = np.clip(np.asarray([scaled_velocity0, eta0], dtype=np.float64), lower, upper)
        best = None
        success = False
        nfev = 0
        for _ in range(int(root_maxiter)):
            residual_current, scale_current = residual_unknowns(x)
            nfev += 1
            if componentwise_scaling:
                # Freeze this iterate's component scales for the Jacobian and
                # line search; re-evaluating them at perturbed points would
                # make the scaled residual saturate as |V| grows.
                weights = scale_current
                residual_current = residual_current / weights
            norm_current = float(np.max(np.abs(residual_current)))
            if best is None or norm_current < best[0]:
                best = (norm_current, x.copy(), residual_current.copy())
            if (norm_current <= relative_tolerance) if componentwise_scaling else converged_at(norm_current, scale_current):
                success = True
                break

            jacobian = np.zeros((2, 2), dtype=np.float64)
            for variable in range(2):
                step, x_perturbed, difference_direction = _bounded_finite_difference_perturbation(x, lower, upper, variable)
                if difference_direction == "forward":
                    residual_perturbed, _ = residual_unknowns(x_perturbed)
                    if componentwise_scaling:
                        residual_perturbed = residual_perturbed / weights
                    jacobian[:, variable] = (residual_perturbed - residual_current) / step
                else:
                    residual_perturbed, _ = residual_unknowns(x_perturbed)
                    if componentwise_scaling:
                        residual_perturbed = residual_perturbed / weights
                    jacobian[:, variable] = (residual_current - residual_perturbed) / step
                nfev += 1

            try:
                step = _newton_step_2x2(jacobian, residual_current)
            except RuntimeError:
                break

            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = np.clip(x + scale * step, lower, upper)
                residual_trial, _ = residual_unknowns(trial)
                nfev += 1
                if componentwise_scaling:
                    residual_trial = residual_trial / weights
                norm_trial = float(np.max(np.abs(residual_trial)))
                if np.isfinite(norm_trial) and norm_trial < norm_current:
                    x = trial
                    accepted = True
                    break
            if not accepted:
                break
        if best is None:
            continue
        velocity = float(best[1][0]) * velocity_scale
        eta = float(best[1][1])
        swept_inventory, flux_delta, c_left, c_right, side_fluxes = evaluate_terms(eta, branch)
        residual = velocity * swept_inventory - flux_delta
        if componentwise_scaling:
            component_scales = component_scales_of(velocity, swept_inventory, side_fluxes)
            final_scale = float(np.max(component_scales))
            selection_norm = float(np.max(np.abs(residual / component_scales)))
        else:
            final_scale = residual_scale_of(velocity, swept_inventory, side_fluxes)
            selection_norm = float(np.max(np.abs(residual)))
        branch_results.append(
            (
                float(np.max(np.abs(residual))),
                branch,
                success,
                nfev,
                velocity,
                eta,
                residual.copy(),
                flux_delta.copy(),
                c_left.copy(),
                c_right.copy(),
                final_scale,
                selection_norm,
            )
        )

    best = _select_converged_branch(branch_results)
    residual_norm, branch, success, nfev, velocity, eta, residual, flux_delta, c_left, c_right, residual_scale, _ = best
    return InitialEtaEstimate(
        eta=eta,
        residual_norm=residual_norm,
        velocity=velocity,
        residual=residual,
        flux_delta=flux_delta,
        left_interface_composition=c_left,
        right_interface_composition=c_right,
        method="instantaneous_balance",
        solver="damped_newton_2x2",
        bracket=bracket,
        converged=bool(success),
        iterations=int(nfev),
        function_calls=int(nfev),
        branch=branch,
        residual_scale=float(residual_scale),
    )


class _ScalarHistory:
    """
    Scalar time history of a ternary/three-phase Illingworth model (interface, eta, ...).

    ``record`` follows the shared row rule (``compatibility.recording``,
    LEDGERS R09): with ``record=k`` the initial row and one row every ``k``
    accepted steps (counted from ``reset`` across solves) are appended, a
    forced row is appended without consuming a cadence slot, and nothing is
    overwritten. ``record=False`` keeps one row holding the current value.
    ``N`` is the last stored row, ``currentIndex`` the cadence counter,
    ``_rowCount`` the number of stored rows and ``_lastRecordStored`` whether
    the latest ``record`` call stored a row.
    """
    def __init__(self, record: bool | int = False):
        if isinstance(record, bool):
            self.recordInterval = 1 if record else -1
        else:
            self.recordInterval = int(record)
        self.batchSize = 1000
        self.reset()

    def reset(self):
        self._y = np.zeros(self.batchSize, dtype=np.float64)
        self._time = np.zeros(self.batchSize, dtype=np.float64)
        self.currentIndex = 0
        self.currentY = 0.0
        self.currentTime = 0.0
        self.N = 0
        self._rowCount = 0
        self._lastRecordStored = False

    def record(self, time, y, force: bool = False):
        """Stores ``y`` under the shared row rule; ``currentY``/``currentTime`` always take this call's values."""
        target = _history_row_target(self, force)
        if target is not None:
            if target >= self._time.shape[0]:
                self._y = np.pad(self._y, (0, self.batchSize))
                self._time = np.pad(self._time, (0, self.batchSize))
            self._y[target] = y
            self._time[target] = time
        _advance_history_rows(self, target, force)
        self.currentY = float(y)
        self.currentTime = float(time)

    def finalize(self):
        """Appends the current value unless the latest record stored a row, then trims (flag-based, no time tolerance)."""
        _finalize_history_rows(self)

    def y(self, time=None):
        """
        Returns the current value without ``time``; otherwise a stored-row lookup.

        A timed request interpolates linearly between stored rows and clamps to
        the first/last stored row, so the current time of an unstored step
        returns the last stored row, not the live value.
        """
        if time is None:
            return float(self.currentY)
        if self.recordInterval > 0:
            if time <= self._time[0]:
                return float(self._y[0])
            if time >= self._time[self.N]:
                return float(self._y[self.N])
            uind = np.argmax(self._time > time)
            lind = uind - 1
            uy, utime = self._y[uind], self._time[uind]
            ly, ltime = self._y[lind], self._time[lind]
            return float((uy - ly) * (time - ltime) / (utime - ltime) + ly)
        return float(self._y[0])


class _ArrayHistory:
    """
    Array-valued time history (fixed ``shape``) of a ternary/three-phase Illingworth model.

    Recording follows the same shared row rule and bookkeeping as
    ``_ScalarHistory``. Timed lookups need an exactly recorded time.
    """
    def __init__(self, shape, record: bool | int = False):
        if isinstance(record, bool):
            self.recordInterval = 1 if record else -1
        else:
            self.recordInterval = int(record)
        self.shape = tuple(int(v) for v in shape)
        self.batchSize = 1000
        self.reset()

    def reset(self):
        self._y = np.zeros((self.batchSize, *self.shape), dtype=np.float64)
        self._time = np.zeros(self.batchSize, dtype=np.float64)
        self.currentIndex = 0
        self.currentY = np.zeros(self.shape, dtype=np.float64)
        self.currentTime = 0.0
        self.N = 0
        self._rowCount = 0
        self._lastRecordStored = False

    def record(self, time, y, force: bool = False):
        """Stores a copy of ``y`` under the shared row rule; ``currentY`` always takes this call's value."""
        values = np.asarray(y, dtype=np.float64)
        if values.shape != self.shape:
            raise ValueError(f"Expected history value with shape {self.shape}, got {values.shape}.")
        target = _history_row_target(self, force)
        if target is not None:
            if target >= self._time.shape[0]:
                self._y = np.pad(self._y, ((0, self.batchSize), *[(0, 0) for _ in self.shape]))
                self._time = np.pad(self._time, (0, self.batchSize))
            self._y[target] = values
            self._time[target] = time
        _advance_history_rows(self, target, force)
        self.currentY = values.copy()
        self.currentTime = float(time)

    def finalize(self):
        """Appends the current value unless the latest record stored a row, then trims (flag-based, no time tolerance)."""
        _finalize_history_rows(self)

    def y(self, time=None):
        """
        Returns a copy of the current value without ``time``; otherwise an exact-time lookup.

        A timed request must match a stored time within ``atol=1e-14`` (the last
        match wins) or ``ValueError`` is raised; the live value is never
        substituted for an unstored time.
        """
        if time is None:
            return np.array(self.currentY, dtype=np.float64)
        recorded_time = self._time[: self.N + 1]
        matches = np.where(np.isclose(recorded_time, float(time), atol=1e-14, rtol=0.0))[0]
        if len(matches) == 0:
            raise ValueError(f"Requested exact history time {float(time):.6g} was not recorded.")
        return self._y[matches[-1]].copy()
