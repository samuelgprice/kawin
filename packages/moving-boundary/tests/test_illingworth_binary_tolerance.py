"""R08: the binary interface iteration stops on dimensionless, conservative criteria.

Before R08 the binary Illingworth model accepted a fixed-point iterate when
``|delta s| <= tolerance`` with ``tolerance`` an absolute length (default
``1e-8``). The result then depended on the length unit: the LEDGERS R08 problem
in metres stopped after two passes with a relative inventory drift of 1.8e-3.
An accepted step must now satisfy both

- ``|delta s| <= tolerance * R`` (``tolerance`` dimensionless, default 1e-12);
- ``|r| <= residual_tolerance * scale`` (default 1e-13), where ``r`` is the
  conservative interface residual at the candidate position, which equals the
  step's change of the transformed inventory up to round-off, and
  ``scale = max(|I_old|, R * c_floor)`` with ``c_floor`` the explicit
  ``interface_residual_composition_scale`` or else ``constraints.minComposition``.

The tests check the unit and composition-scale invariance this buys, the
residual identity and both acceptance checks through the step diagnostics,
the normalization floor and its failure modes, the decoupled Picard default,
constructor validation, that the pass-zero iterate is never accepted, and that
fixed-point nonconvergence still raises.
"""

import numpy as np
import pytest

from kawin.diffusion.DiffusionParameters import TemperatureParameters
from moving_boundary.compatibility.diffusion import DiffusionConstraints
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel
from moving_boundary.solvers.illingworth import binary as binary_module

EPS = np.finfo(float).eps
DEFAULT_TOLERANCE = 1.0e-12
DEFAULT_RESIDUAL_TOLERANCE = 1.0e-13
DEFAULT_PICARD_RTOL = 1.0e-10
# |delta I - r| <= IDENTITY_FACTOR * eps * R * max|composition| (measured: below 4).
IDENTITY_FACTOR = 16.0

# (left bulk, right bulk, left interface, right interface) per unit composition scale.
# Neither direction has a composition below the host clamp threshold, so the
# requested profile is used unchanged and scaled twins are exact twins.
_DIRECTIONS = {
    "right": (0.8, 0.4, 0.6, 0.0),
    "left": (0.8, 0.1, 0.6, 0.3),
}


class _ScaledConstantDiffusivity:
    """Phase-uniform diffusivities of the LEDGERS R07/R08 problem, scaled by ``length**2``."""

    def __init__(self, length=1.0):
        self.values = {"ALPHA": 1.0e-3 * length * length, "BETA": 1.0e-1 * length * length}

    def clearCache(self):
        pass

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        return np.squeeze(np.ones(np.atleast_1d(T).shape) * self.values[phase])


class _LinearDiffusivity(_ScaledConstantDiffusivity):
    """Composition-dependent diffusivity ``D0 * (1 + 0.5 x)`` for the lagged and implicit modes."""

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        return np.squeeze(self.values[phase] * (1.0 + 0.5 * np.asarray(x, dtype=np.float64)))


def _constraints(min_composition):
    if min_composition is None:
        return None
    constraints = DiffusionConstraints()
    constraints.minComposition = float(min_composition)
    return constraints


def _model(
    length=1.0,
    direction="right",
    composition_scale=1.0,
    min_composition=None,
    compositions=None,
    thermodynamics=None,
    **kwargs,
):
    """The LEDGERS R08 problem with lengths scaled by ``length`` and compositions by ``composition_scale``."""
    k = float(composition_scale)
    left0, right0, c_left, c_right = compositions if compositions is not None else _DIRECTIONS[direction]
    L = float(length)
    mesh = CartesianFD1D(["CR"], [0.0, 5.0 * L], 501)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(1.0 * L, left0 * k, right0 * k), "CR")]))
    options = dict(
        time_step=0.1, phase_a_nodes=100, phase_b_nodes=100, max_step_retries=15,
        record=True, bulk_diffusivity_mode="phase_uniform",
    )
    options.update(kwargs)
    return MovingBoundaryIllingworthFD1DModel(
        mesh, ["NI", "CR"], ["ALPHA", "BETA"],
        thermodynamics=_ScaledConstantDiffusivity(L) if thermodynamics is None else thermodynamics,
        temperature=TemperatureParameters(1000.0), interfacePosition=1.0 * L,
        interface_compositions=(c_left * k, c_right * k), constraints=_constraints(min_composition), **options,
    )


def _record_accepted_steps(model):
    """Wraps the implicit step so every accepted step's inventories and diagnostics are recorded."""
    records = []
    original = model._take_implicit_step_planar

    def wrapped(p, q, s, old_s, dt):
        result = original(p, q, s, old_s, dt)
        p_new, q_new, s_new = result[:3]
        records.append(
            dict(
                I_old=float(model.getTotalInventoryFromState(p, q, s)),
                I_new=float(model.getTotalInventoryFromState(p_new, q_new, s_new)),
                ds=float(model._lastImplicitError),
                residual=float(getattr(model, "_lastImplicitResidual", np.nan)),
                scale=float(getattr(model, "_lastImplicitResidualScale", np.nan)),
                iterations=int(model._lastImplicitIterations),
                c_max=float(max(np.max(np.abs(p)), np.max(np.abs(q)), np.max(np.abs(p_new)), np.max(np.abs(q_new)))),
            )
        )
        return result

    model._take_implicit_step_planar = wrapped
    return records


def _solve(model, duration=0.5):
    records = _record_accepted_steps(model)
    model.solve(duration, minDtFrac=1e-14)
    return records


def _history(model):
    history = model.interfaceData
    times = np.asarray(history._time[: history.N + 1], dtype=np.float64)
    positions = np.asarray(history._y[: history.N + 1], dtype=np.float64)
    return times, positions / model._R


def _roundoff(model, record):
    return IDENTITY_FACTOR * EPS * model._R * record["c_max"]


def _assert_inventory_bounds(model, records, residual_tolerance, floor):
    """Per-step ``|delta I| <= rtol * max(|I_old|, R floor)`` and the cumulative ``N``-step bound, up to round-off."""
    assert records
    total_allowance = 0.0
    for record in records:
        allowance = residual_tolerance * max(abs(record["I_old"]), model._R * floor) + _roundoff(model, record)
        assert abs(record["I_new"] - record["I_old"]) <= allowance
        total_allowance += allowance
    assert abs(model.getTotalInventory() - model._initialInventory) <= total_allowance


@pytest.mark.parametrize("direction", sorted(_DIRECTIONS))
def test_length_units_give_the_same_dimensionless_history(direction):
    """The R08 pair (R = 5 and R = 5e-6 with D scaled by L^2) reaches the same s/R history with defaults."""
    unscaled = _model(1.0, direction)
    metres = _model(1.0e-6, direction)
    unscaled_records = _solve(unscaled)
    metre_records = _solve(metres)

    unscaled_times, unscaled_s = _history(unscaled)
    metre_times, metre_s = _history(metres)
    np.testing.assert_array_equal(metre_times, unscaled_times)
    np.testing.assert_allclose(metre_s, unscaled_s, rtol=1.0e-12, atol=0.0)
    for model, records in ((unscaled, unscaled_records), (metres, metre_records)):
        _assert_inventory_bounds(model, records, DEFAULT_RESIDUAL_TOLERANCE, model.constraints.minComposition)


@pytest.mark.parametrize("direction", sorted(_DIRECTIONS))
def test_dilute_compositions_give_the_same_dimensionless_history(direction):
    """Compositions scaled by 1e-4 give the same s/R history and relative inventory bound."""
    reference = _model(direction=direction)
    dilute = _model(direction=direction, composition_scale=1.0e-4)
    _solve(reference)
    dilute_records = _solve(dilute)

    reference_times, reference_s = _history(reference)
    dilute_times, dilute_s = _history(dilute)
    np.testing.assert_array_equal(dilute_times, reference_times)
    np.testing.assert_allclose(dilute_s, reference_s, rtol=1.0e-12, atol=0.0)
    _assert_inventory_bounds(dilute, dilute_records, DEFAULT_RESIDUAL_TOLERANCE, dilute.constraints.minComposition)


@pytest.mark.parametrize("direction", sorted(_DIRECTIONS))
def test_dilute_floor_resolves_from_constraints_and_keeps_the_relative_bound(direction):
    """At compositions x 1e-12 with ``minComposition = 1e-16`` the scale is the inventory and the bound stays relative."""
    reference = _model(direction=direction)
    dilute = _model(direction=direction, composition_scale=1.0e-12, min_composition=1.0e-16)
    _solve(reference)
    records = _solve(dilute)

    np.testing.assert_allclose(_history(dilute)[1], _history(reference)[1], rtol=1.0e-12, atol=0.0)
    for record in records:
        assert record["scale"] == max(abs(record["I_old"]), dilute._R * 1.0e-16)
        assert record["scale"] == abs(record["I_old"])
        assert abs(record["residual"]) <= DEFAULT_RESIDUAL_TOLERANCE * abs(record["I_old"])
    _assert_inventory_bounds(dilute, records, DEFAULT_RESIDUAL_TOLERANCE, 1.0e-16)


@pytest.mark.parametrize(
    "mode",
    ["phase_uniform", "composition_dependent_lagged", "composition_dependent_implicit"],
)
@pytest.mark.parametrize("direction", sorted(_DIRECTIONS))
def test_accepted_steps_satisfy_the_residual_identity_and_both_checks(mode, direction):
    """Each accepted step: ``|delta I - r| <= c eps R``, ``|r| <= rtol * scale`` and ``|delta s| <= tolerance * R``."""
    thermodynamics = None if mode == "phase_uniform" else _LinearDiffusivity()
    model = _model(direction=direction, bulk_diffusivity_mode=mode, thermodynamics=thermodynamics)
    records = _solve(model)

    assert model.tolerance == DEFAULT_TOLERANCE
    assert model.residualTolerance == DEFAULT_RESIDUAL_TOLERANCE
    assert records
    for record in records:
        delta_inventory = record["I_new"] - record["I_old"]
        assert abs(delta_inventory - record["residual"]) <= _roundoff(model, record)
        assert record["scale"] == max(abs(record["I_old"]), model._R * model.constraints.minComposition)
        assert abs(record["residual"]) <= model.residualTolerance * record["scale"]
        assert record["ds"] <= model.tolerance * model._R
        assert record["iterations"] >= 2


def test_explicit_composition_scale_overrides_min_composition():
    """An explicit ``interface_residual_composition_scale`` replaces ``constraints.minComposition`` as the floor."""
    explicit = _model(composition_scale=1.0e-4, interface_residual_composition_scale=1.0e-3)
    default = _model(composition_scale=1.0e-4)
    explicit_records = _solve(explicit, duration=0.1)
    default_records = _solve(default, duration=0.1)

    R = explicit._R
    assert abs(explicit_records[0]["I_old"]) < R * 1.0e-3
    assert explicit_records[0]["scale"] == R * 1.0e-3
    assert default_records[0]["scale"] == abs(default_records[0]["I_old"])


def test_omitted_composition_scale_reads_min_composition_when_the_step_starts():
    """With the scale omitted the floor is ``constraints.minComposition`` at the time of the step."""
    model = _model(composition_scale=1.0e-4)
    model.setup()
    p, q, s = model._p_curr, model._q_curr, model._s_curr
    inventory = abs(float(model.getTotalInventoryFromState(p, q, s)))
    assert model._interface_residual_scale(p, q, s) == max(inventory, model._R * 1.0e-10)
    model.constraints.minComposition = 1.0e-3
    assert model._interface_residual_scale(p, q, s) == model._R * 1.0e-3


def test_zero_inventory_safeguard_applies_only_to_a_zero_floor():
    """Exactly zero inventory and a zero floor give the absolute ``64 eps R`` scale; a positive floor gives ``R * floor``."""
    model = _model(min_composition=0.0)
    model.setup()
    p = np.zeros_like(model._p_curr)
    q = np.zeros_like(model._q_curr)
    s = model._s_curr
    assert float(model.getTotalInventoryFromState(p, q, s)) == 0.0
    assert model._interface_residual_scale(p, q, s) == model._R * 64.0 * EPS

    model.constraints.minComposition = 1.0e-10
    assert model._interface_residual_scale(p, q, s) == model._R * 1.0e-10


@pytest.mark.parametrize(
    ("inventory", "length", "floor", "expected"),
    [
        (2.0, 5.0, 1.0e-10, 2.0),
        (-3.0, 5.0, 1.0e-10, 3.0),
        (0.0, 5.0, 1.0e-10, 5.0e-10),
        (1.0e-12, 5.0, 0.0, 1.0e-12),
        (0.0, 5.0, 0.0, 5.0 * 64.0 * EPS),
        (0.0, 1.0e-200, 1.0e-100, 1.0e-200 * 1.0e-100),
    ],
)
def test_residual_scale_values(inventory, length, floor, expected):
    """``scale = max(|I_old|, R * floor)``, and ``64 eps R`` only when both inputs are exactly zero."""
    assert binary_module._binary_residual_scale(inventory, length, floor) == expected


@pytest.mark.parametrize(
    ("inventory", "length", "floor", "match"),
    [
        (np.nan, 5.0, 1.0e-10, "inventory"),
        (np.inf, 5.0, 1.0e-10, "inventory"),
        (-np.inf, 5.0, 1.0e-10, "inventory"),
        (1.0, 5.0, -1.0e-10, "floor"),
        (1.0, 5.0, np.nan, "floor"),
        (1.0, 5.0, np.inf, "floor"),
        (1.0, np.nan, 1.0e-10, "domain length"),
        (1.0, 0.0, 1.0e-10, "domain length"),
        (1.0, -5.0, 1.0e-10, "domain length"),
        (0.0, 1.0e300, 1.0e10, "scale"),  # R * floor overflows
        (0.0, 1.0e-200, 1.0e-200, "scale"),  # positive floor: R * floor underflows to 0, never the safeguard
        (0.0, 1.0e-310, 0.0, "scale"),  # the zero-inventory safeguard itself underflows to 0
    ],
)
def test_residual_scale_rejects_invalid_values(inventory, length, floor, match):
    """Non-finite or non-positive inputs and scales raise ``ValueError`` naming the quantity; nothing is substituted."""
    with pytest.raises(ValueError, match=match):
        binary_module._binary_residual_scale(inventory, length, floor)


def test_floor_binding_regime_bounds_the_mean_composition_change():
    """With valid data whose inventory lies below ``R * minComposition`` the per-step bound is absolute."""
    # Uniform 1e-10 (exactly minComposition, not clamped), left interface 1.5e-10, right interface 0:
    # the trapezoid inventory is just below R * minComposition.
    model = _model(compositions=(1.0e-10, 1.0e-10, 1.5e-10, 0.0))
    records = _solve(model)
    floor = model.constraints.minComposition

    assert records
    for record in records:
        assert abs(record["I_old"]) < model._R * floor
        assert record["scale"] == model._R * floor
        assert abs(record["residual"]) <= DEFAULT_RESIDUAL_TOLERANCE * model._R * floor
        allowance = DEFAULT_RESIDUAL_TOLERANCE * model._R * floor + _roundoff(model, record)
        assert abs(record["I_new"] - record["I_old"]) <= allowance


@pytest.mark.parametrize("tolerance", [1.0e-6, 1.0e-14])
def test_picard_default_does_not_follow_tolerance(tolerance):
    """``bulk_picard_rtol=None`` resolves to 1e-10 whatever ``tolerance`` is; an explicit value is kept."""
    default = _model(tolerance=tolerance, bulk_diffusivity_mode="composition_dependent_implicit", thermodynamics=_LinearDiffusivity())
    explicit = _model(
        tolerance=tolerance, bulk_diffusivity_mode="composition_dependent_implicit",
        thermodynamics=_LinearDiffusivity(), bulk_picard_rtol=1.0e-8,
    )
    assert default._bulk_picard_relative_tolerance() == DEFAULT_PICARD_RTOL
    assert explicit._bulk_picard_relative_tolerance() == 1.0e-8


def test_implicit_picard_solves_meet_the_default_threshold():
    """Every implicit-mode Picard solve stops below ``atol + 1e-10 * max|profile|`` with ``bulk_picard_rtol`` omitted."""
    model = _model(tolerance=1.0e-6, bulk_diffusivity_mode="composition_dependent_implicit", thermodynamics=_LinearDiffusivity())
    results = []
    original = model._solve_bulk_picard

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result

    model._solve_bulk_picard = wrapped
    model.solve(0.2, minDtFrac=1e-14)

    assert results
    for result in results:
        profile_scale = max(1.0e-12, float(np.max(np.abs(result.profile))))
        assert result.inner_update_norm <= model.bulkPicardAtol + DEFAULT_PICARD_RTOL * profile_scale


@pytest.mark.parametrize("name", ["tolerance", "residual_tolerance", "interface_residual_composition_scale"])
@pytest.mark.parametrize("value", [0.0, -1.0e-12, np.nan, np.inf])
def test_invalid_tolerances_are_rejected_at_construction(name, value):
    """Non-positive or non-finite tolerances and composition scales raise ``ValueError`` before setup."""
    mesh = CartesianFD1D(["CR"], [0.0, 5.0], 501)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(1.0, 0.8, 0.4), "CR")]))
    before = np.array(mesh.y, copy=True)
    with pytest.raises(ValueError, match=name):
        MovingBoundaryIllingworthFD1DModel(
            mesh, ["NI", "CR"], ["ALPHA", "BETA"], thermodynamics=_ScaledConstantDiffusivity(),
            temperature=TemperatureParameters(1000.0), interfacePosition=1.0,
            interface_compositions=(0.6, 0.0), time_step=0.1, **{name: value},
        )
    np.testing.assert_array_equal(mesh.y, before)


def test_pass_zero_iterate_is_never_accepted():
    """A state at its interface compositions passes both checks at pass zero, yet each step takes two passes."""
    model = _model(compositions=(0.6, 0.3, 0.6, 0.3))
    passes = []
    original = model._new_interface_planar

    def wrapped(left_deviation, right_deviation, s, old_s, future_s, dt, pass_number, *args):
        result = original(left_deviation, right_deviation, s, old_s, future_s, dt, pass_number, *args)
        passes.append((pass_number, s, result))
        return result

    model._new_interface_planar = wrapped
    records = _solve(model)

    assert records
    pass_zero = [(s, result) for pass_number, s, result in passes if pass_number == 0]
    assert len(pass_zero) == len(records)
    for (s, result), record in zip(pass_zero, records):
        assert abs(result - s) <= model.tolerance * model._R
        assert record["iterations"] >= 2


def test_fixed_point_nonconvergence_still_raises():
    """The clamped x 1e-12 problem enters a two-point cycle; that stays a ``RuntimeError``, never an accepted step.

    The requested 0.8e-12 / 0.4e-12 profile is clamped by the host to a uniform
    1e-10 (the default ``minComposition``), so the bulk deviations dwarf the
    6e-13 tie-line jump and the motion-dependent denominator changes sign
    between iterates. This establishes nonconvergence of the fixed-point
    iteration only, not that the implicit equations have no root.
    """
    model = _model(composition_scale=1.0e-12)
    assert np.all(np.ravel(model.data.currentY) == 1.0e-10)
    with pytest.raises(RuntimeError, match="failed to converge"):
        model.solve(0.5, minDtFrac=1e-14)
    assert model.interfaceData.N == 0
