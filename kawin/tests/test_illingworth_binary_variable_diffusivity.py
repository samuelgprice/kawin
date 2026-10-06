"""Variable-diffusivity modes of the binary Illingworth moving-boundary model."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kawin.solver import explicitEulerIterator
from kawin.tests.test_diffusion import (
    ConstantBinaryThermodynamics,
    VariableBinaryThermodynamics,
    _build_illingworth_default_model,
)

MODES = ("phase_uniform", "composition_dependent_lagged", "composition_dependent_implicit")

# Larger diffusivities than the default regression case so that D(c) changes
# the profiles over several transformed cells within a short run.
FAST = {"d_alpha": 1.0e-3, "d_beta": 1.0e-2, "n_alpha": 40, "n_beta": 60, "time_step": 0.05}


class StrongVariableBinaryThermodynamics(ConstantBinaryThermodynamics):
    """D = D0 * (1 + 20 x^2): strongly composition dependent across the 0-0.8 range."""

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        composition = np.asarray(x, dtype=np.float64)
        return np.squeeze(float(self.diffusivities[phase]) * (1.0 + 20.0 * composition**2))


class NoContextVariableThermodynamics(ConstantBinaryThermodynamics):
    """``Thermodynamics``-style signature without ``query_context``; records query sizes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_sizes = []

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None):
        composition = np.asarray(x, dtype=np.float64)
        self.query_sizes.append(composition.size)
        return np.squeeze(float(self.diffusivities[phase]) * (1.0 + 0.5 * composition))


class CountingVariableThermodynamics(ConstantBinaryThermodynamics):
    """Records every diffusivity query; D depends on composition and temperature."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        self.calls.append((phase, query_context, np.asarray(T, dtype=np.float64).copy()))
        composition = np.asarray(x, dtype=np.float64)
        return np.squeeze(float(self.diffusivities[phase]) * (1.0 + 0.5 * composition) * np.asarray(T, dtype=np.float64) / 1000.0)


def _model(mode, therm_class=VariableBinaryThermodynamics, **kwargs):
    params = dict(FAST)
    model_kwargs = {"bulk_diffusivity_mode": mode}
    model_kwargs.update(kwargs.pop("model_kwargs", {}))
    params.update(kwargs)
    return _build_illingworth_default_model(therm_class=therm_class, model_kwargs=model_kwargs, **params)


def _interface_history(model):
    return model.interfaceData._y[: model.interfaceData.N + 1].copy()


def test_phase_uniform_default_matches_explicit_mode_bitwise():
    default = _build_illingworth_default_model()
    explicit = _build_illingworth_default_model(model_kwargs={"bulk_diffusivity_mode": "phase_uniform"})
    default.solve(0.5, iterator=explicitEulerIterator, minDtFrac=1e-14)
    explicit.solve(0.5, iterator=explicitEulerIterator, minDtFrac=1e-14)
    assert_allclose(_interface_history(explicit), _interface_history(default), rtol=0.0, atol=0.0)
    assert_allclose(explicit._p_curr, default._p_curr, rtol=0.0, atol=0.0)
    assert_allclose(explicit._q_curr, default._q_curr, rtol=0.0, atol=0.0)


def test_isothermal_phase_uniform_diffusivities_are_evaluated_once_in_setup():
    model = _model("phase_uniform", therm_class=CountingVariableThermodynamics)
    model.solve(0.5, iterator=explicitEulerIterator, minDtFrac=1e-14)
    assert model.interfaceData.N >= 10
    assert [(phase, context) for phase, context, _ in model.therm.calls] == [("ALPHA", "interface"), ("BETA", "interface")]
    assert model._D_left == pytest.approx(1.0e-3 * (1.0 + 0.5 * 0.6))
    assert model._D_right == pytest.approx(1.0e-2)


def test_time_varying_temperature_re_evaluates_phase_uniform_diffusivities_each_step():
    from kawin.diffusion import TemperatureParameters

    cached = _model("phase_uniform", therm_class=CountingVariableThermodynamics)
    varying = _model("phase_uniform", therm_class=CountingVariableThermodynamics)
    # Temperature array times are in hours: 1000 K -> 1360 K over the first hour.
    varying.temperatureParameters = TemperatureParameters([0.0, 1.0], [1000.0, 1360.0])
    cached.solve(0.2, iterator=explicitEulerIterator, minDtFrac=1e-14)
    varying.solve(0.2, iterator=explicitEulerIterator, minDtFrac=1e-14)

    n_steps = round(0.2 / FAST["time_step"])
    assert len(cached.therm.calls) == 2
    assert len(varying.therm.calls) == 2 * (n_steps + 1)
    temperatures = np.asarray([float(T) for _, _, T in varying.therm.calls[::2]])
    # setup() and the first step both query at t = 0; later steps see a rising T.
    assert temperatures[1] == temperatures[0]
    assert np.all(np.diff(temperatures[1:]) > 0.0)
    assert varying._D_left > cached._D_left


@pytest.mark.parametrize("mode", MODES[1:])
def test_composition_dependent_modes_reduce_to_phase_uniform_for_constant_d(mode):
    reference = _model("phase_uniform", therm_class=ConstantBinaryThermodynamics)
    model = _model(mode, therm_class=ConstantBinaryThermodynamics)
    reference.solve(0.5, iterator=explicitEulerIterator, minDtFrac=1e-14)
    model.solve(0.5, iterator=explicitEulerIterator, minDtFrac=1e-14)
    assert_allclose(_interface_history(model), _interface_history(reference), rtol=0.0, atol=1e-12)
    assert_allclose(model._p_curr, reference._p_curr, rtol=0.0, atol=1e-12)
    assert_allclose(model._q_curr, reference._q_curr, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_variable_d_conserves_transformed_inventory(mode):
    # Inventory drift is bounded by the fixed-point |delta s| tolerance times the
    # interface jump per step, so a tight tolerance isolates scheme conservation.
    model = _model(
        mode,
        therm_class=StrongVariableBinaryThermodynamics,
        tolerance=1e-13,
        model_kwargs={"bulk_picard_rtol": 1e-14},
    )
    model.solve(1.0, iterator=explicitEulerIterator, minDtFrac=1e-14)
    assert model.checkConservation(1e-11) < 1e-11
    assert np.all(np.isfinite(model._p_curr)) and np.all(np.isfinite(model._q_curr))


def test_variable_d_modes_change_the_solution():
    s_final = {}
    for mode in MODES:
        model = _model(mode, therm_class=StrongVariableBinaryThermodynamics)
        model.solve(1.0, iterator=explicitEulerIterator, minDtFrac=1e-14)
        s_final[mode] = model._s_curr
    # Interface D is the same in all modes only for phase_uniform's interface
    # composition; bulk face values differ, so the solutions must differ.
    assert abs(s_final["phase_uniform"] - s_final["composition_dependent_implicit"]) > 1e-8
    assert abs(s_final["composition_dependent_lagged"] - s_final["composition_dependent_implicit"]) > 0.0


def test_implicit_step_satisfies_discrete_equations_at_converged_profile():
    model = _model(
        "composition_dependent_implicit",
        therm_class=StrongVariableBinaryThermodynamics,
        tolerance=1e-13,
        model_kwargs={"bulk_picard_rtol": 1e-13, "bulk_picard_atol": 1e-14},
    )
    model.setup()
    p, q, s, dt = model._p_curr.copy(), model._q_curr.copy(), model._s_curr, 0.05
    p_new, q_new, s_new, D_left_int, D_right_int = model._take_implicit_step_planar(p, q, s, s, dt)

    D_left = model._left_face_diffusivities(p_new, s_new, model.currentTime)
    D_right = model._right_face_diffusivities(q_new, s_new, model.currentTime)
    assert_allclose(model._new_concentration_left_planar(p, s, s_new, dt, D_left), p_new, rtol=0.0, atol=1e-11)
    assert_allclose(model._new_concentration_right_planar(q, s, s_new, dt, D_right), q_new, rtol=0.0, atol=1e-11)
    assert D_left_int == pytest.approx(D_left[-1], rel=1e-10)
    assert D_right_int == pytest.approx(D_right[0], rel=1e-10)
    # The interface balance is a fixed point with the interface-face values.
    s_again = model._new_interface_planar(p_new, q_new, s, s, s_new, dt, 1, D_left_int, D_right_int)
    assert s_again == pytest.approx(s_new, rel=0.0, abs=1e-11)


def test_lagged_minus_implicit_difference_is_first_order_in_dt():
    differences = []
    for dt in (0.1, 0.05, 0.025):
        finals = []
        for mode in MODES[1:]:
            model = _model(mode, therm_class=StrongVariableBinaryThermodynamics, time_step=dt)
            model.solve(0.4, iterator=explicitEulerIterator, minDtFrac=1e-14)
            finals.append(model._s_curr)
        differences.append(abs(finals[0] - finals[1]))
    assert differences[0] > 0.0
    ratios = np.asarray(differences[:-1]) / np.asarray(differences[1:])
    assert np.all(ratios > 1.6), (differences, ratios)


def test_thermodynamics_without_query_context_uses_vectorized_face_queries():
    model = _model("composition_dependent_lagged", therm_class=NoContextVariableThermodynamics)
    model.solve(0.2, iterator=explicitEulerIterator, minDtFrac=1e-14)
    n_left_faces = FAST["n_alpha"] - 1
    n_right_faces = FAST["n_beta"] - 1
    sizes = set(model.therm.query_sizes)
    assert sizes == {n_left_faces, n_right_faces}


def test_picard_nonconvergence_raises_without_retries():
    model = _model(
        "composition_dependent_implicit",
        therm_class=StrongVariableBinaryThermodynamics,
        model_kwargs={"bulk_picard_max_iterations": 1},
    )
    with pytest.raises(RuntimeError, match="Picard"):
        model.solve(0.2, iterator=explicitEulerIterator, minDtFrac=1e-14)


def test_failed_step_is_retried_with_smaller_dt():
    model = _model("composition_dependent_lagged", model_kwargs={"max_step_retries": 3, "retry_factor": 0.5})
    original_step = model._take_implicit_step_planar
    attempted_dts = []

    def fail_once_then_solve(p, q, s, old_s, dt):
        attempted_dts.append(dt)
        if len(attempted_dts) == 1:
            raise RuntimeError("forced failure")
        return original_step(p, q, s, old_s, dt)

    model._take_implicit_step_planar = fail_once_then_solve
    model.solve(0.2, iterator=explicitEulerIterator, minDtFrac=1e-14)
    assert attempted_dts[0] == pytest.approx(FAST["time_step"])
    assert attempted_dts[1] == pytest.approx(0.5 * FAST["time_step"])
    t = model.interfaceData._time[: model.interfaceData.N + 1]
    assert t[1] == pytest.approx(0.5 * FAST["time_step"])
    assert t[-1] == pytest.approx(0.2)
    assert model.checkConservation(1e-10) < 1e-10


def test_retry_exhaustion_reraises_last_error():
    model = _model("phase_uniform", model_kwargs={"max_step_retries": 2})

    def always_fail(p, q, s, old_s, dt):
        raise ZeroDivisionError("forced failure")

    model._take_implicit_step_planar = always_fail
    with pytest.raises(ZeroDivisionError, match="forced failure"):
        model.solve(0.2, iterator=explicitEulerIterator, minDtFrac=1e-14)


@pytest.mark.parametrize("bad_value", [0.0, -1.0e-5, np.nan])
@pytest.mark.parametrize("mode", MODES)
def test_invalid_diffusivity_raises(mode, bad_value):
    class BadThermodynamics(ConstantBinaryThermodynamics):
        def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
            return np.squeeze(np.full(np.shape(x), bad_value, dtype=np.float64))

    model = _model(mode, therm_class=BadThermodynamics)
    with pytest.raises(ValueError, match="diffusivity"):
        model.solve(0.1, iterator=explicitEulerIterator, minDtFrac=1e-14)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"bulk_picard_max_iterations": 0}, "bulk_picard_max_iterations"),
        ({"bulk_picard_relaxation": 0.0}, "bulk_picard_relaxation"),
        ({"bulk_picard_relaxation": 1.5}, "bulk_picard_relaxation"),
        ({"bulk_picard_atol": -1.0}, "bulk_picard_atol"),
        ({"bulk_picard_rtol": np.nan}, "bulk_picard_rtol"),
        ({"max_step_retries": 0}, "max_step_retries"),
        ({"retry_factor": 1.0}, "retry_factor"),
    ],
)
def test_configuration_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model("phase_uniform", model_kwargs=kwargs)
