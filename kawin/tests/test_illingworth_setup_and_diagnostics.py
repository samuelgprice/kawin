"""
``setup()`` idempotence and the ``print_diagnostics`` option of the Illingworth models.

``solve`` calls ``setup`` on every call, so a second ``setup`` (explicit, or from
a continued ``solve``) must not re-initialize the transformed state, restart the
histories or reset the reference inventory.
"""

import numpy as np
import pytest

from kawin.tests.illingworth_baseline_cases import (
    _binary_model,
    _CoupledTernaryThermodynamics,
    _ternary_model,
    _three_phase_model,
)

SOLVE_TIME = {"binary": 0.5, "ternary": 2.0e-3, "three_phase": 2.0e-3}


def _build(kind):
    if kind == "binary":
        return _binary_model("phase_uniform", False)
    if kind == "ternary":
        return _ternary_model("phase_uniform", _CoupledTernaryThermodynamics(), np.array([0.16, 0.06]), np.array([0.3261538461538461, 0.193]))
    return _three_phase_model("phase_uniform")


def _state(model):
    if getattr(model, "_profiles_curr", None) is not None:
        return [np.array(p) for p in model._profiles_curr] + [np.atleast_1d(model._interfaces_curr), np.atleast_1d(model._etas_curr)]
    state = [np.array(model._p_curr), np.array(model._q_curr), np.atleast_1d(model._s_curr)]
    if getattr(model, "_eta_curr", None) is not None:
        state.append(np.atleast_1d(model._eta_curr))
    return state


def _assert_same_state(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("kind", ["binary", "ternary", "three_phase"])
def test_setup_is_idempotent(kind):
    model = _build(kind)
    model.setup()
    first = _state(model)
    inventory = np.array(model._initialInventory, copy=True)
    model.setup()
    _assert_same_state(first, _state(model))
    np.testing.assert_array_equal(model._initialInventory, inventory)


@pytest.mark.parametrize("kind", ["binary", "ternary", "three_phase"])
def test_explicit_setup_then_solve_matches_solve(kind):
    plain = _build(kind)
    plain.solve(SOLVE_TIME[kind], minDtFrac=1.0e-10)
    explicit = _build(kind)
    explicit.setup()
    explicit.solve(SOLVE_TIME[kind], minDtFrac=1.0e-10)
    _assert_same_state(_state(plain), _state(explicit))


@pytest.mark.parametrize("kind", ["binary", "ternary", "three_phase"])
def test_continued_solve_keeps_reference_inventory_and_history(kind):
    single = _build(kind)
    single.solve(SOLVE_TIME[kind], minDtFrac=1.0e-10)
    continued = _build(kind)
    continued.solve(0.5 * SOLVE_TIME[kind], minDtFrac=1.0e-10)
    inventory = np.array(continued._initialInventory, copy=True)
    first_record = np.array(continued.interfaceData._y[0], copy=True)
    continued.solve(0.5 * SOLVE_TIME[kind], minDtFrac=1.0e-10)
    np.testing.assert_array_equal(continued._initialInventory, inventory)
    np.testing.assert_array_equal(continued._initialInventory, single._initialInventory)
    np.testing.assert_array_equal(continued.interfaceData._y[0], first_record)
    assert continued.currentTime == pytest.approx(single.currentTime)


@pytest.mark.parametrize("kind", ["ternary", "three_phase"])
def test_print_diagnostics_is_quiet_by_default_and_prints_when_enabled(kind, capsys):
    quiet = _build(kind)
    assert quiet.printDiagnostics is False
    quiet._print_diagnostic("should not appear")
    quiet.setup()
    assert capsys.readouterr().out == ""

    loud = _build(kind)
    loud.printDiagnostics = True
    loud._print_diagnostic("diagnostic line")
    assert "diagnostic line" in capsys.readouterr().out
    if kind == "ternary":
        loud.setup()
        assert "initialEtaEstimate" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Binary external boundary conditions (same rules as the ternary models)
# ---------------------------------------------------------------------------


def _binary_model_with_boundaries(boundary_conditions):
    from kawin.diffusion import MovingBoundaryIllingworthFD1DModel, TemperatureParameters
    from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
    from kawin.tests.illingworth_baseline_cases import _BinaryThermodynamics

    mesh = CartesianFD1D(["CR"], [0.0, 5.0], 51)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(1.0, 0.8, 0.4), "CR")]), boundaryConditions=boundary_conditions)
    return MovingBoundaryIllingworthFD1DModel(
        mesh,
        ["NI", "CR"],
        ["ALPHA", "BETA"],
        thermodynamics=_BinaryThermodynamics({"ALPHA": 1.0e-7, "BETA": 1.0e-5}, variable=False),
        temperature=TemperatureParameters(1000.0),
        interfacePosition=1.0,
        interface_compositions=(0.6, 0.0),
        time_step=0.1,
    )


def _binary_bc(left=None, right=None):
    from kawin.diffusion.mesh import MixedBoundary1D

    bc = MixedBoundary1D(["CR"])
    if left is not None:
        bc.setLBC("CR", *left)
    if right is not None:
        bc.setRBC("CR", *right)
    return bc


def test_binary_accepts_explicit_zero_flux_boundaries():
    model = _binary_model_with_boundaries(_binary_bc(left=("flux", 0.0), right=("flux", 0.0)))
    model.solve(0.2)
    assert model.currentTime == pytest.approx(0.2)


@pytest.mark.parametrize(
    "boundaries, match",
    [
        pytest.param(dict(left=("flux", 1.0e-6)), "nonzero fluxes", id="nonzero_flux"),
        pytest.param(dict(left=("composition", 0.8)), "fixed-composition", id="fixed_composition"),
        pytest.param(dict(left=("flux", 0.0), right=("composition", 0.4)), "fixed-composition or mixed", id="mixed_boundary_types"),
        pytest.param(None, "periodic", id="periodic"),
    ],
)
def test_binary_rejects_unsupported_boundaries_before_solve(boundaries, match):
    from kawin.diffusion.mesh import PeriodicBoundary1D

    bc = PeriodicBoundary1D() if boundaries is None else _binary_bc(**boundaries)
    with pytest.raises(NotImplementedError, match=match):
        _binary_model_with_boundaries(bc)
