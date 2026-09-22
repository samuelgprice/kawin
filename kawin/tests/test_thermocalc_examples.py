import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from examples.ThermoCalc.tc_python_adapter import (
    GAS_CONSTANT,
    TCPythonThermodynamics,
    ThermoCalcCalculationError,
    ThermoCalcConfig,
    ThermoCalcDatabaseError,
    ThermoCalcInputError,
    ThermoCalcSolveError,
    independent_to_full_composition,
    normalized_driving_force_to_j_per_mol,
    validate_independent_composition,
    _TCPythonBackend,
)
from examples.ThermoCalc.training_data import (
    build_moving_boundary_surrogate,
    load_training_dataset,
    make_fecrni_demo_grid,
    sample_training_data,
)


def _fecrni_config(**overrides):
    """Builds the explicit Fe-Cr-Ni configuration exercised by adapter tests."""
    values = {
        "thermodynamic_database": "TCFE9",
        "kinetic_database": "MOBFE4",
        "elements": ("FE", "CR", "NI"),
        "phases": ("BCC_A2", "FCC_A1"),
        "reference_element": "FE",
    }
    values.update(overrides)
    return ThermoCalcConfig(**values)


class FakeThermoCalcBackend:
    def __init__(self, fail_once_group=None):
        self.config = None
        self.started = False
        self.restart_count = 0
        self.fail_once_group = fail_once_group
        self.failed_groups = set()
        self.totalNumCalcs = 0
        self.totalNumCaches = 0
        self.totalNumQueries = 0
        self.equilibrium_calls = 0

    def start(self, config):
        self.config = config
        self.started = True

    def close(self):
        self.started = False

    def restart(self):
        self.restart_count += 1
        self.started = True

    def get_runtime_version(self):
        return "fake-2026.1"

    def preflight(self, config, x, T):
        self.start(config)
        if config.user_database_path is not None:
            raise ThermoCalcDatabaseError("QPFIND : NO SUCH INTENSIVE VARIABLE")
        return {"ok": True, "stable_phases": ["BCC_A2"], "driving_force": 1.0}

    def calculate_equilibrium(self, x, T):
        self.equilibrium_calls += 1
        self._fail_once("equilibrium")
        return {
            "stable_phases": ["BCC_A2", "FCC_A1"],
            "phase_amounts": {"BCC_A2": 0.7, "FCC_A1": 0.3},
            "phase_compositions": {
                "BCC_A2": np.array([1.0 - x[0] - x[1], x[0], x[1]]),
                "FCC_A1": np.array([0.8, 0.15, 0.05]),
            },
            "chemical_potentials": {"FE": -1.0, "CR": -2.0, "NI": -3.0},
            "phase_interdiffusivities": {
                "BCC_A2": np.array([[1.0, 0.1], [0.2, 2.0]]) * 1e-14,
                "FCC_A1": np.array([[2.0, 0.2], [0.4, 4.0]]) * 1e-14,
            },
            "phase_tracerdiffusivities": {
                "BCC_A2": np.array([3.0, 4.0, 5.0]) * 1e-14,
                "FCC_A1": np.array([6.0, 8.0, 10.0]) * 1e-14,
            },
        }

    def calculate_driving_force(self, x, T, precipitate_phase):
        self._fail_once("driving_force")
        return {
            "driving_force": normalized_driving_force_to_j_per_mol(0.5, T),
            "normalized_driving_force": 0.5,
            "precipitate_composition": np.array([0.8, 0.15, 0.05]),
        }

    def calculate_kinetics(self, x, T, phase):
        self._fail_once(f"kinetics:{phase}")
        scale = 1.0 if phase == "BCC_A2" else 2.0
        return {
            "interdiffusivity": scale * np.array([[1.0, 0.1], [0.2, 2.0]]) * 1e-14,
            "tracer_diffusivity": scale * np.array([3.0, 4.0, 5.0]) * 1e-14,
        }

    def _fail_once(self, group):
        if self.fail_once_group == group and group not in self.failed_groups:
            self.failed_groups.add(group)
            raise ThermoCalcCalculationError(f"transient {group} failure")


class FakeSystemBuilder:
    def __init__(self, calls):
        self.calls = calls
        self.selected_phases = []
        self.used_without_default_phases = False

    def without_default_phases(self):
        self.calls.append(("without_default_phases",))
        self.used_without_default_phases = True
        return self

    def select_phase(self, phase):
        self.calls.append(("select_phase", phase))
        self.selected_phases.append(phase)
        return self

    def get_system(self):
        self.calls.append(("get_system",))
        return FakeTCSystem(
            self.calls,
            used_without_default_phases=self.used_without_default_phases,
            selected_phases=tuple(self.selected_phases),
        )


class FakeTCCalculation:
    def __init__(self, calls):
        self.calls = calls

    def with_options(self, options):
        self.calls.append(("with_options", type(options).__name__))
        return self

    def disable_global_minimization(self):
        self.calls.append(("disable_global_minimization",))
        return self

    def enable_global_minimization(self):
        self.calls.append(("enable_global_minimization",))
        return self

    def remove_all_conditions(self):
        self.calls.append(("remove_all_conditions",))

    def calculate(self):
        self.calls.append(("calculate",))
        return object()

    def set_phase_to_dormant(self, phase):
        self.calls.append(("set_phase_to_dormant", phase))

    def set_phase_to_entered(self, phase):
        self.calls.append(("set_phase_to_entered", phase))

    def set_phase_to_suspended(self, phase):
        self.calls.append(("set_phase_to_suspended", phase))


class FakeTCSystem(dict):
    def __init__(self, calls, **kwargs):
        super().__init__(**kwargs)
        self.calls = calls

    def with_single_equilibrium_calculation(self):
        self.calls.append(("with_single_equilibrium_calculation",))
        return FakeTCCalculation(self.calls)


class FakeSingleEquilibriumOptions:
    def __init__(self, calls):
        self.calls = calls
        self.grid_points = None
        self.force_positive_definite_hessian = None

    def set_global_minimization_max_grid_points(self, value):
        self.calls.append(("options_set_global_minimization_max_grid_points", value))
        self.grid_points = value
        return self

    def disable_force_positive_definite_phase_hessian(self):
        self.calls.append(("options_disable_force_positive_definite_phase_hessian",))
        self.force_positive_definite_hessian = False
        return self

    def enable_force_positive_definite_phase_hessian(self):
        self.calls.append(("options_enable_force_positive_definite_phase_hessian",))
        self.force_positive_definite_hessian = True
        return self


class FakeTCPythonModule:
    def __init__(self, calls):
        self.calls = calls

    def SingleEquilibriumOptions(self):
        self.calls.append(("SingleEquilibriumOptions",))
        return FakeSingleEquilibriumOptions(self.calls)


class FakeTCPythonSetup:
    def __init__(self):
        self.calls = []

    def select_thermodynamic_and_kinetic_databases_with_elements(self, thermodynamic_database, kinetic_database, elements):
        self.calls.append(
            (
                "select_thermodynamic_and_kinetic_databases_with_elements",
                thermodynamic_database,
                kinetic_database,
                tuple(elements),
            )
        )
        return FakeSystemBuilder(self.calls)


def test_composition_closure_and_validation():
    config = _fecrni_config()

    assert_allclose(validate_independent_composition([0.3, 0.2], config), [0.3, 0.2])
    assert_allclose(independent_to_full_composition([0.3, 0.2], config), [0.5, 0.3, 0.2])

    with pytest.raises(ThermoCalcInputError):
        validate_independent_composition([0.8, 0.3], config)
    with pytest.raises(ThermoCalcInputError):
        validate_independent_composition([0.1, -0.1], config)
    with pytest.raises(ThermoCalcInputError):
        validate_independent_composition([0.1, 0.2, 0.3], config)


def test_phase_selection_defaults_to_thermocalc_default_phases():
    config = _fecrni_config()
    backend = _TCPythonBackend()
    backend._setup = FakeTCPythonSetup()
    backend._config = config

    default_system = backend._get_system(True)
    restricted_system = backend._get_system(False)

    assert config.use_default_phases is True
    assert not default_system["used_without_default_phases"]
    assert default_system["selected_phases"] == ()
    assert restricted_system["used_without_default_phases"]
    assert restricted_system["selected_phases"] == config.phases
    assert ("without_default_phases",) not in backend._setup.calls[:2]


def test_phase_selection_can_restrict_to_configured_phases():
    config = _fecrni_config(use_default_phases=False)
    backend = _TCPythonBackend()
    backend._setup = FakeTCPythonSetup()
    backend._config = config

    system = backend._get_system(config.use_default_phases)

    assert system["used_without_default_phases"]
    assert system["selected_phases"] == config.phases


def test_global_minimization_grid_points_are_applied_to_calculations():
    config = _fecrni_config(global_minimization_max_grid_points=2500)
    backend = _TCPythonBackend()
    backend._setup = FakeTCPythonSetup()
    backend._config = config
    backend._tc_python = FakeTCPythonModule(backend._setup.calls)

    backend._get_calculation("equilibrium", None)

    assert ("SingleEquilibriumOptions",) in backend._setup.calls
    assert ("options_set_global_minimization_max_grid_points", 2500) in backend._setup.calls
    assert ("with_options", "FakeSingleEquilibriumOptions") in backend._setup.calls


def test_global_minimization_grid_points_must_be_positive():
    with pytest.raises(ThermoCalcInputError, match="global_minimization_max_grid_points"):
        _fecrni_config(global_minimization_max_grid_points=0)


def _qthiss_retry_backend(
    outcomes, *, retry_points=(20_000, 200_000),
    normal_grid_points=2000, disable_global=False, disable_hessian=False,
):
    """Create a fake backend recording each retry's grid and kinetics settings."""
    config = _fecrni_config(
        use_default_phases=False,
        global_minimization_max_grid_points=normal_grid_points,
        equilibrium_qthiss_retry_grid_points=retry_points,
        kinetics_disable_global_minimization=disable_global,
        kinetics_disable_positive_definite_hessian=disable_hessian,
    )
    calls = []
    remaining = {points: list(errors) for points, errors in outcomes.items()}

    class RetryCalculation(FakeTCCalculation):
        def __init__(self):
            super().__init__(calls)
            self.grid_points = 2000
            self.global_enabled = True
            self.hessian_enabled = True
            self.phase_status = {}

        def with_options(self, options):
            super().with_options(options)
            if options.grid_points is not None:
                self.grid_points = options.grid_points
            if options.force_positive_definite_hessian is not None:
                self.hessian_enabled = options.force_positive_definite_hessian
            return self

        def disable_global_minimization(self):
            super().disable_global_minimization()
            self.global_enabled = False
            return self

        def enable_global_minimization(self):
            super().enable_global_minimization()
            self.global_enabled = True
            return self

        def set_phase_to_entered(self, phase):
            super().set_phase_to_entered(phase)
            self.phase_status[phase] = "entered"

        def set_phase_to_suspended(self, phase):
            super().set_phase_to_suspended(phase)
            self.phase_status[phase] = "suspended"

        def set_phase_to_dormant(self, phase):
            super().set_phase_to_dormant(phase)
            self.phase_status[phase] = "dormant"

        def calculate(self):
            calls.append(("attempt", self.grid_points))
            calls.append((
                "attempt_settings", self.grid_points, self.global_enabled,
                self.hessian_enabled, tuple(sorted(self.phase_status.items())),
            ))
            errors = remaining.get(self.grid_points, [])
            if errors:
                raise errors.pop(0)
            return object()

    class RetrySystem:
        def with_single_equilibrium_calculation(self):
            return RetryCalculation()

    backend = _TCPythonBackend()
    backend._config = config
    backend._setup = object()
    backend._systems[False] = RetrySystem()
    backend._tc_python = FakeTCPythonModule(calls)
    backend._set_conditions = lambda calc, x, T: None
    backend.totalNumCalcs = 0
    backend.total_kind_lst = []
    backend.total_x_lst = []
    return backend, calls

def test_equilibrium_qthiss_retries_restore_normal_grid_for_next_query():
    qthiss = "ERROR IN QTHISS : TOO MANY ITERATIONS"
    backend, calls = _qthiss_retry_backend(
        {2000: [RuntimeError(qthiss)], 20_000: [RuntimeError(qthiss)]}
    )
    x = np.array([0.3, 0.2])

    backend._calculate("equilibrium", None, x, 1373.0)
    assert backend._retry_calculation is not None
    backend._calculate("equilibrium", None, x, 1373.0)

    assert [call for call in calls if call[0] == "attempt"] == [
        ("attempt", 2000), ("attempt", 20_000), ("attempt", 200_000), ("attempt", 2000)
    ]
    assert backend._retry_calculation is None
    assert backend._retry_settings_dirty is False
    assert calls.count(("enable_global_minimization",)) == 3
    assert backend.totalNumCalcs == 4
    assert backend._config.to_metadata()["equilibrium_qthiss_retry_grid_points"] == (20_000, 200_000)

def test_equilibrium_qthiss_retries_exhaust_then_restore_normal_grid():
    qthiss = "ERROR IN QTHISS : TOO MANY ITERATIONS"
    backend, calls = _qthiss_retry_backend({
        2000: [RuntimeError(qthiss)],
        20_000: [RuntimeError(qthiss)],
        200_000: [RuntimeError(qthiss)],
    })
    x = np.array([0.3, 0.2])

    with pytest.raises(ThermoCalcCalculationError, match="QTHISS : TOO MANY ITERATIONS"):
        backend._calculate("equilibrium", None, x, 1373.0)
    backend._calculate("equilibrium", None, x, 1373.0)

    assert [call for call in calls if call[0] == "attempt"] == [
        ("attempt", 2000), ("attempt", 20_000), ("attempt", 200_000), ("attempt", 2000)
    ]
    assert calls.count(("enable_global_minimization",)) == 3

def test_equilibrium_qthiss_retry_restores_normal_grid_before_kinetics():
    backend, calls = _qthiss_retry_backend({
        2000: [RuntimeError("ERROR IN QTHISS : TOO MANY ITERATIONS")],
    })
    x = np.array([0.3, 0.2])

    backend._calculate("equilibrium", None, x, 1373.0)
    backend._calculate("kinetics", "BCC_A2", x, 1373.0)

    assert [call for call in calls if call[0] == "attempt"] == [
        ("attempt", 2000), ("attempt", 20_000), ("attempt", 2000)
    ]
    assert backend._retry_settings_dirty is False

def test_equilibrium_qthiss_retry_restores_tc_default_grid_when_unset():
    backend, calls = _qthiss_retry_backend(
        {2000: [RuntimeError("ERROR IN QTHISS : TOO MANY ITERATIONS")]},
        normal_grid_points=None,
    )
    x = np.array([0.3, 0.2])

    backend._calculate("equilibrium", None, x, 1373.0)
    backend._calculate("equilibrium", None, x, 1373.0)

    assert [call for call in calls if call[0] == "attempt"] == [
        ("attempt", 2000), ("attempt", 20_000), ("attempt", 2000)
    ]
    assert calls.count(("enable_global_minimization",)) == 2

def test_equilibrium_qthiss_retries_stop_on_a_different_error(monkeypatch):
    monkeypatch.setattr("examples.ThermoCalc.tc_python_adapter.debugInPlace", lambda: None)
    backend, calls = _qthiss_retry_backend({
        2000: [RuntimeError("ERROR IN QTHISS : TOO MANY ITERATIONS")],
        20_000: [RuntimeError("different calculation error")],
    })

    with pytest.raises(ThermoCalcCalculationError, match="different calculation error"):
        backend._calculate("equilibrium", None, np.array([0.3, 0.2]), 1373.0)

    assert [call for call in calls if call[0] == "attempt"] == [("attempt", 2000), ("attempt", 20_000)]

@pytest.mark.parametrize(
    "message, retry_points",
    [("different calculation error", (20_000, 200_000)), ("ERROR IN QTHISS : TOO MANY ITERATIONS", ())],
)
def test_equilibrium_retry_requires_both_matching_error_and_enabled_levels(message, retry_points, monkeypatch):
    monkeypatch.setattr("examples.ThermoCalc.tc_python_adapter.debugInPlace", lambda: None)
    backend, calls = _qthiss_retry_backend({2000: [RuntimeError(message)]}, retry_points=retry_points)

    with pytest.raises(ThermoCalcCalculationError, match=message):
        backend._calculate("equilibrium", None, np.array([0.3, 0.2]), 1373.0)

    assert [call for call in calls if call[0] == "attempt"] == [("attempt", 2000)]

@pytest.mark.parametrize("kind, phase", [
    ("equilibrium", None), ("kinetics", "BCC_A2"), ("driving_force", "FCC_A1"),
])
def test_qthiss_retry_does_not_run_for_condition_setup_failure(kind, phase):
    backend, calls = _qthiss_retry_backend({})

    def fail_conditions(calc, x, T):
        raise RuntimeError("ERROR IN QTHISS : TOO MANY ITERATIONS")

    backend._set_conditions = fail_conditions
    with pytest.raises(ThermoCalcCalculationError, match="QTHISS : TOO MANY ITERATIONS") as failure:
        backend._calculate(kind, phase, np.array([0.3, 0.2]), 1373.0)

    assert not any(call[0] == "attempt" for call in calls)
    assert not isinstance(failure.value, ThermoCalcSolveError)

def test_driving_force_qthiss_retries_and_preserves_dormant_phase():
    backend, calls = _qthiss_retry_backend({2000: [RuntimeError("ERROR IN QTHISS : TOO MANY ITERATIONS")]})
    x = np.array([0.3, 0.2])

    backend._calculate("driving_force", "FCC_A1", x, 1373.0)
    backend._calculate("driving_force", "FCC_A1", x, 1373.0)

    assert [call for call in calls if call[0] == "attempt_settings"] == [
        ("attempt_settings", 2000, True, True, (("FCC_A1", "dormant"),)),
        ("attempt_settings", 20_000, True, True, (("FCC_A1", "dormant"),)),
        ("attempt_settings", 2000, True, True, (("FCC_A1", "dormant"),)),
    ]

@pytest.mark.parametrize("disable_hessian", [False, True])
def test_global_kinetics_qthiss_retries_preserve_phase_and_restore_settings(disable_hessian):
    qthiss = "ERROR IN QTHISS : TOO MANY ITERATIONS"
    backend, calls = _qthiss_retry_backend(
        {2000: [RuntimeError(qthiss)], 20_000: [RuntimeError(qthiss)]},
        disable_hessian=disable_hessian,
    )
    x = np.array([0.3, 0.2])

    backend._calculate("kinetics", "BCC_A2", x, 1373.0)
    assert backend._retry_calculation is not None
    backend._calculate("kinetics", "BCC_A2", x, 1373.0)
    backend._calculate("equilibrium", None, x, 1373.0)

    settings = [call for call in calls if call[0] == "attempt_settings"]
    phase_status = (("BCC_A2", "entered"), ("FCC_A1", "suspended"))
    assert settings == [
        ("attempt_settings", 2000, True, not disable_hessian, phase_status),
        ("attempt_settings", 20_000, True, not disable_hessian, phase_status),
        ("attempt_settings", 200_000, True, not disable_hessian, phase_status),
        ("attempt_settings", 2000, True, not disable_hessian, phase_status),
        ("attempt_settings", 2000, True, True, ()),
    ]
    assert backend._retry_calculation is None
    assert backend._retry_settings_dirty is False
    assert backend._config.to_metadata()["equilibrium_qthiss_retry_grid_points"] == (20_000, 200_000)
    assert backend.total_kind_lst == ["kinetics"] * 4 + ["equilibrium"]

def test_global_kinetics_qthiss_retries_exhaust_then_restore_settings():
    qthiss = "ERROR IN QTHISS : TOO MANY ITERATIONS"
    backend, calls = _qthiss_retry_backend(
        {2000: [RuntimeError(qthiss)], 20_000: [RuntimeError(qthiss)], 200_000: [RuntimeError(qthiss)]},
        disable_hessian=True,
    )
    x = np.array([0.3, 0.2])

    with pytest.raises(ThermoCalcCalculationError, match="QTHISS : TOO MANY ITERATIONS"):
        backend._calculate("kinetics", "BCC_A2", x, 1373.0)
    backend._calculate("kinetics", "BCC_A2", x, 1373.0)

    settings = [call for call in calls if call[0] == "attempt_settings"]
    assert [(call[1], call[2]) for call in settings] == [
        (2000, True), (20_000, True), (200_000, True), (2000, True),
    ]
    assert backend._retry_settings_dirty is False

@pytest.mark.parametrize(
    "message, retry_points",
    [("different calculation error", (20_000, 200_000)), ("ERROR IN QTHISS : TOO MANY ITERATIONS", ())],
)
def test_kinetics_retry_requires_matching_error_and_enabled_levels(message, retry_points, monkeypatch):
    monkeypatch.setattr("examples.ThermoCalc.tc_python_adapter.debugInPlace", lambda: None)
    backend, calls = _qthiss_retry_backend({2000: [RuntimeError(message)]}, retry_points=retry_points)

    with pytest.raises(ThermoCalcCalculationError, match=message):
        backend._calculate("kinetics", "BCC_A2", np.array([0.3, 0.2]), 1373.0)

    assert [call for call in calls if call[0] == "attempt"] == [("attempt", 2000)]

def test_kinetics_qthiss_retries_stop_on_different_error(monkeypatch):
    monkeypatch.setattr("examples.ThermoCalc.tc_python_adapter.debugInPlace", lambda: None)
    backend, calls = _qthiss_retry_backend(
        {2000: [RuntimeError("ERROR IN QTHISS : TOO MANY ITERATIONS")], 20_000: [RuntimeError("different error")]},
    )

    with pytest.raises(ThermoCalcCalculationError, match="different error"):
        backend._calculate("kinetics", "BCC_A2", np.array([0.3, 0.2]), 1373.0)
    backend._calculate("kinetics", "BCC_A2", np.array([0.3, 0.2]), 1373.0)

    assert [call for call in calls if call[0] == "attempt"] == [
        ("attempt", 2000), ("attempt", 20_000), ("attempt", 2000),
    ]

def test_equilibrium_qthiss_retry_points_must_increase_above_normal_grid():
    with pytest.raises(ThermoCalcInputError, match="increase strictly"):
        _fecrni_config(equilibrium_qthiss_retry_grid_points=(20_000, 20_000))
    with pytest.raises(ThermoCalcInputError, match="must exceed"):
        _fecrni_config(global_minimization_max_grid_points=20_000, equilibrium_qthiss_retry_grid_points=(20_000, 200_000))

@pytest.mark.parametrize("disable_hessian", [False, True])
def test_local_kinetics_qthiss_does_not_retry_or_change_minimization(disable_hessian):
    backend, calls = _qthiss_retry_backend(
        {2000: [RuntimeError("ERROR IN QTHISS : TOO MANY ITERATIONS")]},
        disable_global=True, disable_hessian=disable_hessian,
    )
    x = np.array([0.3, 0.2])

    with pytest.raises(ThermoCalcCalculationError, match="QTHISS : TOO MANY ITERATIONS"):
        backend._calculate("kinetics", "BCC_A2", x, 1373.0)
    backend._calculate("kinetics", "BCC_A2", x, 1373.0)

    settings = [call for call in calls if call[0] == "attempt_settings"]
    phase_status = (("BCC_A2", "entered"), ("FCC_A1", "suspended"))
    assert settings == [
        ("attempt_settings", 2000, False, not disable_hessian, phase_status),
        ("attempt_settings", 2000, False, not disable_hessian, phase_status),
    ]
    assert backend._retry_calculation is None
    assert backend._retry_settings_dirty is False
    assert ("enable_global_minimization",) not in calls

@pytest.mark.parametrize("disable_global", [False, True])
@pytest.mark.parametrize("disable_hessian", [False, True])
def test_kinetics_minimization_switches_are_independent(disable_global, disable_hessian):
    config = _fecrni_config(
        global_minimization_max_grid_points=2500,
        kinetics_disable_global_minimization=disable_global,
        kinetics_disable_positive_definite_hessian=disable_hessian,
    )
    backend = _TCPythonBackend()
    backend._setup = FakeTCPythonSetup()
    backend._config = config
    backend._tc_python = FakeTCPythonModule(backend._setup.calls)

    backend._get_calculation("equilibrium", None)
    backend._get_calculation("driving_force", "FCC_A1")
    nonkinetics_calls = list(backend._setup.calls)
    backend._get_calculation("kinetics", "BCC_A2")
    kinetics_calls = backend._setup.calls[len(nonkinetics_calls):]

    assert ("disable_global_minimization",) not in nonkinetics_calls
    assert ("options_disable_force_positive_definite_phase_hessian",) not in nonkinetics_calls
    assert (("disable_global_minimization",) in kinetics_calls) == disable_global
    assert (("options_disable_force_positive_definite_phase_hessian",) in kinetics_calls) == disable_hessian
    assert (("options_set_global_minimization_max_grid_points", 2500) in kinetics_calls) != disable_global
    assert config.to_metadata()["kinetics_disable_global_minimization"] == disable_global
    assert config.to_metadata()["kinetics_disable_positive_definite_hessian"] == disable_hessian

@pytest.mark.parametrize("disable_global", [False, True])
@pytest.mark.parametrize("disable_hessian", [False, True])
def test_minimization_settings_are_restored_when_switching_calculation_kinds(disable_global, disable_hessian):
    config = _fecrni_config(
        use_default_phases=False,
        global_minimization_max_grid_points=2500,
        kinetics_disable_global_minimization=disable_global,
        kinetics_disable_positive_definite_hessian=disable_hessian,
    )
    backend = _TCPythonBackend()
    backend._setup = FakeTCPythonSetup()
    backend._config = config
    backend._tc_python = FakeTCPythonModule(backend._setup.calls)
    backend._set_conditions = lambda calc, x, T: None
    backend.totalNumCalcs = 0
    backend.total_kind_lst = []
    backend.total_x_lst = []
    x = np.array([0.3, 0.2])

    for kind, phase in (
        ("equilibrium", None),
        ("kinetics", "BCC_A2"),
        ("equilibrium", None),
        ("kinetics", "BCC_A2"),
    ):
        backend._calculate(kind, phase, x, 1373.0)

    calls = backend._setup.calls
    boundaries = [i for i, call in enumerate(calls) if call == ("calculate",)]
    restored_equilibrium = calls[boundaries[1] + 1:boundaries[2]]
    restored_kinetics = calls[boundaries[2] + 1:boundaries[3]]
    switched = disable_global or disable_hessian

    assert (("enable_global_minimization",) in restored_equilibrium) == switched
    assert (("options_enable_force_positive_definite_phase_hessian",) in restored_equilibrium) == switched
    assert (("options_set_global_minimization_max_grid_points", 2500) in restored_equilibrium) == switched
    assert (("disable_global_minimization",) in restored_kinetics) == (switched and disable_global)
    assert (("options_disable_force_positive_definite_phase_hessian",) in restored_kinetics) == (switched and disable_hessian)

def test_driving_force_reasserts_global_minimization_after_local_kinetics():
    config = _fecrni_config(
        use_default_phases=False,
        kinetics_disable_global_minimization=True,
        kinetics_disable_positive_definite_hessian=True,
    )
    backend = _TCPythonBackend()
    backend._setup = FakeTCPythonSetup()
    backend._config = config
    backend._tc_python = FakeTCPythonModule(backend._setup.calls)
    backend._set_conditions = lambda calc, x, T: None
    backend.totalNumCalcs = 0
    backend.total_kind_lst = []
    backend.total_x_lst = []
    x = np.array([0.3, 0.2])

    backend._calculate("kinetics", "BCC_A2", x, 1373.0)
    backend._calculate("driving_force", "FCC_A1", x, 1373.0)

    calls = backend._setup.calls
    first_calculate = calls.index(("calculate",))
    restored = calls[first_calculate + 1:]
    assert ("enable_global_minimization",) in restored
    assert ("options_enable_force_positive_definite_phase_hessian",) in restored


def test_driving_force_conversion_and_default_phase():
    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=FakeThermoCalcBackend())

    dg, xp = therm.getDrivingForce([0.3, 0.2], 1000.0)

    assert_allclose(dg, 0.5 * GAS_CONSTANT * 1000.0)
    assert_allclose(xp, [0.15, 0.05])


def test_diffusion_shapes_and_element_ordering():
    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=FakeThermoCalcBackend())

    dnkj = therm.getInterdiffusivity([[0.3, 0.2], [0.35, 0.1]], 1373.0, phase="BCC_A2")
    tracer = therm.getTracerDiffusivity([0.3, 0.2], 1373.0, phase="BCC_A2")

    assert dnkj.shape == (2, 2, 2)
    assert tracer.shape == (3,)
    assert_allclose(dnkj[0], [[1.0e-14, 0.1e-14], [0.2e-14, 2.0e-14]])
    assert_allclose(tracer, [3.0e-14, 4.0e-14, 5.0e-14])


def test_planar_tie_line_metadata_and_gextra_rejection():
    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=FakeThermoCalcBackend())

    x_alpha, x_beta, metadata = therm.getInterfacialComposition([0.3, 0.2], 1373.0, returnMeta=True)

    assert_allclose(x_alpha, [0.3, 0.2])
    assert_allclose(x_beta, [0.15, 0.05])
    assert metadata["endpoint_phases"] == ("BCC_A2", "FCC_A1")
    with pytest.raises(NotImplementedError):
        therm.getInterfacialComposition([0.3, 0.2], 1373.0, gExtra=1.0)


def test_missing_tie_line_metadata_reports_actual_stable_phases():
    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=FakeThermoCalcBackend())
    therm.getEquilibriumData = lambda *args, **kwargs: {
        "stable_phases": ["BCC_A2#2"],
        "phase_compositions": {"BCC_A2#2": np.array([0.5, 0.3, 0.2])},
    }

    _, _, metadata = therm.getInterfacialComposition([0.3, 0.2], 1373.0, returnMeta=True)

    assert metadata["endpoint_phases"] == (None, None)
    assert metadata["stable_phases"] == ("BCC_A2#2",)


def test_default_remove_cache_is_configurable_per_adapter():
    backend = FakeThermoCalcBackend()
    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=backend, default_remove_cache=False)

    first = therm.getEquilibriumData([0.3, 0.2], 1373.0)
    second = therm.getEquilibriumData([0.3, 0.2], 1373.0)
    refreshed = therm.getEquilibriumData([0.3, 0.2], 1373.0, removeCache=True)

    assert backend.equilibrium_calls == 2
    assert first is second
    assert refreshed is not second


def test_checkpoint_resume_and_manifest_roundtrip():
    backend = FakeThermoCalcBackend(fail_once_group="equilibrium")
    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=backend)
    prefix = Path("examples") / "ThermoCalc" / "outputs" / "test_tc_dataset"

    dataset = sample_training_data(
        therm,
        [[0.3, 0.2], [0.35, 0.1]],
        1373.0,
        output_prefix=prefix,
        checkpoint_every=1,
        resume=False,
    )
    loaded = load_training_dataset(prefix)

    assert backend.restart_count == 1
    assert dataset["manifest"]["complete"]
    assert loaded["manifest"]["complete"]
    assert loaded["manifest"]["tc_python_version"] == "fake-2026.1"
    assert_allclose(loaded["arrays"]["compositions"], [[0.3, 0.2], [0.35, 0.1]])
    assert np.all(loaded["arrays"]["equilibrium_valid"])
    assert np.all(loaded["arrays"]["bcc_a2_kinetics_valid"])
    assert np.all(loaded["arrays"]["fcc_a1_kinetics_valid"])
    assert json.loads(prefix.with_suffix(".json").read_text())["complete"]


def test_demo_grid_size_and_order():
    grid = make_fecrni_demo_grid(config=_fecrni_config())

    assert grid.shape == (25, 2)
    assert np.all(grid[:, 0] >= 0.05)
    assert np.all(grid[:, 1] >= 0.001)


def test_build_moving_boundary_surrogate_from_adapter():
    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=FakeThermoCalcBackend())
    path = Path("examples") / "ThermoCalc" / "outputs" / "test_mb_surrogate.npz"

    surrogate = build_moving_boundary_surrogate(
        therm,
        temperature=1373.0,
        probe_start=(0.25, 0.068),
        probe_end=(0.45, 0.242),
        eta_samples=[0.0, 0.5, 1.0],
        diffusivity_bulk_points=[[0.3, 0.2], [0.35, 0.1]],
        output_path=path,
    )
    left, right = surrogate.interface_compositions(0.5)
    bcc_D = surrogate.getInterdiffusivity(left, 1373.0, phase="BCC_A2")
    loaded = type(surrogate).load(path)

    assert path.exists()
    assert surrogate.tieline_phases == ("BCC_A2", "FCC_A1")
    assert_allclose(left, [0.35, 0.155])
    assert_allclose(right, [0.15, 0.05])
    assert_allclose(bcc_D, [[1.0e-14, 0.1e-14], [0.2e-14, 2.0e-14]])
    assert loaded.metadata["source"] == "tc_python_adapter"


def test_live_tc_python_one_point():
    if not bool(int(__import__("os").environ.get("KAWIN_TC_PYTHON_LIVE", "0"))):
        pytest.skip("Set KAWIN_TC_PYTHON_LIVE=1 to run Thermo-Calc license/database tests.")

    therm = TCPythonThermodynamics(config=_fecrni_config())
    with therm:
        dg, xp = therm.getDrivingForce([0.38, 0.001], 1373.0)
        dnkj = therm.getInterdiffusivity([0.38, 0.001], 1373.0, phase="BCC_A2")
        tracer = therm.getTracerDiffusivity([0.38, 0.001], 1373.0, phase="BCC_A2")
        fcc_dnkj = therm.getInterdiffusivity([0.38, 0.001], 1373.0, phase="FCC_A1")
        fcc_tracer = therm.getTracerDiffusivity([0.38, 0.001], 1373.0, phase="FCC_A1")

    assert np.isfinite(dg)
    assert xp.shape == (2,)
    assert dnkj.shape == (2, 2)
    assert tracer.shape == (3,)
    assert fcc_dnkj.shape == (2, 2)
    assert fcc_tracer.shape == (3,)
