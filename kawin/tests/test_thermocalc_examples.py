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
from kawin.diffusion import DiffusivityDomainError, DiffusivityDomainStatus


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

    def calculate_kinetics(self, x, T, phase, collect_diagnostics=False):
        self._fail_once(f"kinetics:{phase}")
        scale = 1.0 if phase == "BCC_A2" else 2.0
        output = {
            "interdiffusivity": scale * np.array([[1.0, 0.1], [0.2, 2.0]]) * 1e-14,
            "tracer_diffusivity": scale * np.array([3.0, 4.0, 5.0]) * 1e-14,
        }
        if collect_diagnostics:
            output["diagnostics"] = {
                "thermodynamic_factors": [[1.0, 0.1], [0.2, 2.0]],
                "stable_composition_sets": [phase],
                "phase_composition": [1.0 - x[0] - x[1], x[0], x[1]],
                "site_fractions": [{"sublattice": 1, "constituents": {"Fe": 1.0 - x[0], "Cr": x[0]}}],
                "errors": {},
            }
        return output

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

    def set_phase_to_entered(self, phase, amount=None):
        self.calls.append(("set_phase_to_entered", phase) if amount is None else
                          ("set_phase_to_entered", phase, amount))

    def set_phase_to_fixed(self, phase, amount):
        self.calls.append(("set_phase_to_fixed", phase, amount))

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
    normal_grid_points=2000, disable_global=False, disable_hessian=False, single_set=False,
):
    """Create a fake backend recording each retry's grid and kinetics settings."""
    config = _fecrni_config(
        use_default_phases=False,
        global_minimization_max_grid_points=normal_grid_points,
        equilibrium_qthiss_retry_grid_points=retry_points,
        kinetics_disable_global_minimization=disable_global,
        kinetics_disable_positive_definite_hessian=disable_hessian,
        kinetics_constrain_single_composition_set=single_set,
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

        def set_phase_to_entered(self, phase, amount=None):
            super().set_phase_to_entered(phase, amount)
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


def test_calculate_failure_is_marked_for_bulk_sampling():
    backend, _ = _qthiss_retry_backend(
        {2000: [RuntimeError("ERROR IN QTHISS : TOO MANY ITERATIONS")]}, retry_points=(),
    )

    with pytest.raises(ThermoCalcSolveError, match="QTHISS : TOO MANY ITERATIONS") as failure:
        backend._calculate("kinetics", "BCC_A2", np.array([0.3, 0.2]), 1373.0)

    assert failure.value.failed_during_calculate is True


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


def test_single_set_kinetics_qthiss_does_not_retry_with_global_minimization():
    backend, calls = _qthiss_retry_backend(
        {2000: [RuntimeError("ERROR IN QTHISS : TOO MANY ITERATIONS")]},
        disable_global=False, single_set=True,
    )

    with pytest.raises(ThermoCalcSolveError, match="QTHISS : TOO MANY ITERATIONS"):
        backend._calculate("kinetics", "BCC_A2", np.array([0.3, 0.2]), 1373.0)

    assert calls.count(("attempt", 2000)) == 1
    assert ("attempt", 20_000) not in calls
    assert ("disable_global_minimization",) in calls


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


def test_single_composition_set_constraint_applies_to_kinetics_only():
    config = _fecrni_config(kinetics_constrain_single_composition_set=True)
    backend = _TCPythonBackend()
    backend._setup = FakeTCPythonSetup()
    backend._config = config
    backend._tc_python = FakeTCPythonModule(backend._setup.calls)

    backend._get_calculation("equilibrium", None)
    backend._get_calculation("driving_force", "FCC_A1")
    nonkinetics_calls = list(backend._setup.calls)
    backend._get_calculation("kinetics", "BCC_A2")
    kinetics_calls = backend._setup.calls[len(nonkinetics_calls):]

    assert ("set_phase_to_suspended", "*") not in nonkinetics_calls
    assert ("set_phase_to_suspended", "*") in kinetics_calls
    assert ("set_phase_to_entered", "BCC_A2", 0.0) in kinetics_calls
    assert ("set_phase_to_entered", "BCC_A2") not in kinetics_calls
    assert ("disable_global_minimization",) in kinetics_calls
    assert ("set_phase_to_fixed", "BCC_A2", 1.0) not in kinetics_calls
    assert config.to_metadata()["kinetics_constrain_single_composition_set"] is True
    assert _fecrni_config().kinetics_constrain_single_composition_set is False


def test_single_composition_set_constraint_checks_actual_kinetics_result():
    config = _fecrni_config(kinetics_constrain_single_composition_set=True)
    backend = _TCPythonBackend()
    backend._setup = object()
    backend._config = config
    x = np.array([0.3, 0.2])

    class Result:
        def __init__(self, phases, composition):
            self.phases = phases
            self.composition = composition

        def get_stable_phases(self):
            return self.phases

    backend._phase_composition = lambda result, phase: np.array(result.composition)
    backend._validate_single_kinetics_phase(Result(["BCC_A2"], [0.5, 0.3, 0.2]), "BCC_A2", x)
    with pytest.raises(ThermoCalcCalculationError, match="stable composition sets"):
        backend._validate_single_kinetics_phase(Result(["BCC_A2", "BCC_A2#2"], [0.5, 0.3, 0.2]), "BCC_A2", x)
    with pytest.raises(ThermoCalcCalculationError, match="expected"):
        backend._validate_single_kinetics_phase(Result(["BCC_A2"], [0.4, 0.4, 0.2]), "BCC_A2", x)


def test_single_composition_set_constraint_restores_global_minimization():
    config = _fecrni_config(kinetics_constrain_single_composition_set=True)
    backend = _TCPythonBackend()
    backend._setup = FakeTCPythonSetup()
    backend._config = config
    backend._tc_python = FakeTCPythonModule(backend._setup.calls)
    backend._set_conditions = lambda calc, x, T: None
    backend._validate_single_kinetics_phase = lambda result, phase, x: None
    backend.totalNumCalcs = 0
    backend.total_kind_lst = []
    backend.total_x_lst = []

    backend._calculate("kinetics", "BCC_A2", np.array([0.3, 0.2]), 1373.0)
    before_equilibrium = len(backend._setup.calls)
    backend._calculate("equilibrium", None, np.array([0.3, 0.2]), 1373.0)

    assert ("enable_global_minimization",) in backend._setup.calls[before_equilibrium:]


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


def test_kinetics_capture_records_queries_and_refreshes_incomplete_cache():
    backend = FakeThermoCalcBackend()
    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=backend, default_remove_cache=False)
    point = [0.3, 0.2]
    therm.getInterdiffusivity(point, 1373.0, phase="BCC_A2")
    records = []

    with therm.captureKineticsDiagnostics(records.append):
        matrix = therm.getInterdiffusivity(point, 1373.0, phase="BCC_A2")
        therm.getInterdiffusivity(point, 1373.0, phase="BCC_A2")

    therm.getInterdiffusivity(point, 1373.0, phase="BCC_A2")
    assert len(records) == 2
    assert [record["query_index"] for record in records] == [0, 1]
    assert [record["cache_hit"] for record in records] == [False, True]
    assert_allclose(records[0]["interdiffusivity"], matrix)
    assert_allclose(records[0]["tracer_diffusivities"], [3e-14, 4e-14, 5e-14])
    assert_allclose(records[0]["thermodynamic_factors"], [[1.0, 0.1], [0.2, 2.0]])
    assert records[0]["stable_composition_sets"] == ["BCC_A2"]
    assert_allclose(records[0]["phase_composition"], [0.5, 0.3, 0.2])
    assert records[0]["site_fractions"][0]["constituents"]["Cr"] == 0.3
    assert records[0]["diagnostic_errors"] == {}


def test_optional_kinetics_quantity_failure_does_not_discard_matrix():
    class PartialDiagnosticBackend(FakeThermoCalcBackend):
        def calculate_kinetics(self, x, T, phase, collect_diagnostics=False):
            output = super().calculate_kinetics(x, T, phase, collect_diagnostics=collect_diagnostics)
            if collect_diagnostics:
                output["diagnostics"]["thermodynamic_factors"][0][1] = None
                output["diagnostics"]["errors"]["thermodynamic_factor[CR,NI]"] = "unavailable"
            return output

    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=PartialDiagnosticBackend())
    records = []
    with therm.captureKineticsDiagnostics(records.append):
        matrix = therm.getInterdiffusivity([0.3, 0.2], 1373.0, phase="BCC_A2")
    assert np.all(np.isfinite(matrix))
    assert records[0]["thermodynamic_factors"][0][1] is None
    assert records[0]["diagnostic_errors"]["thermodynamic_factor[CR,NI]"] == "unavailable"


def test_backend_diagnostics_use_same_result_and_named_sublattices():
    class Quantities:
        @staticmethod
        def chemical_diffusion_coefficient(phase, diffusing, gradient, reference):
            return ("chemical", phase, diffusing, gradient, reference)

        @staticmethod
        def tracer_diffusion_coefficient(phase, element):
            return ("tracer", phase, element)

        @staticmethod
        def thermodynamic_factor(phase, diffusing, gradient, reference):
            return ("factor", phase, diffusing, gradient, reference)

        @staticmethod
        def composition_of_phase_as_mole_fraction(phase, element):
            return ("composition", phase, element)

    class Species:
        def __init__(self, name):
            self.name = name

        def get_name(self):
            return self.name

    class Sublattice:
        def get_constituents(self):
            return {Species("W"), Species("Ti")}

    class Phase:
        def get_sublattices(self):
            return [Sublattice()]

    class System:
        def get_phase_object(self, phase):
            assert phase == "BCC_B2#1"
            return Phase()

    class Result:
        def get_stable_phases(self):
            return ["BCC_B2#1"]

        def get_value_of(self, quantity):
            if isinstance(quantity, str) and quantity.startswith("Y("):
                phase, site = quantity[2:-1].split(",")
                constituent, sublattice = site.rsplit("#", 1)
                quantity = ("site", phase, constituent, int(sublattice))
            kind = quantity[0]
            if kind == "chemical":
                return 1e-14 * (1 + (quantity[2] == "Fe") + 2 * (quantity[3] == "Fe"))
            if kind == "tracer":
                return {"W": 1e-14, "Ti": 2e-14, "Fe": 3e-14}[quantity[2]]
            if kind == "factor":
                if quantity[2:] == ("Ti", "Fe", "W"):
                    raise RuntimeError("factor unavailable")
                return 1 + (quantity[2] == "Fe") + 2 * (quantity[3] == "Fe")
            if kind == "composition":
                return {"W": 0.5, "Ti": 0.3, "Fe": 0.2}[quantity[2]]
            if kind == "site":
                return {"W": 0.7, "Ti": 0.3}[quantity[2]]
            raise AssertionError(quantity)

    config = ThermoCalcConfig(
        thermodynamic_database="TCHEA5", kinetic_database="MOBHEA4",
        elements=("W", "TI", "FE"), phases=("BCC_B2#1", "LIQUID#1"), reference_element="W",
    )
    backend = _TCPythonBackend()
    backend._config = config
    backend._setup = object()
    backend._systems[False] = System()
    backend._tc_python = type("TCPythonModule", (), {"ThermodynamicQuantity": Quantities})()
    backend._calculate = lambda kind, phase, x, T: Result()

    output = backend.calculate_kinetics(np.array([0.3, 0.2]), 1973.0, "BCC_B2#1", collect_diagnostics=True)
    assert_allclose(output["interdiffusivity"], [[1e-14, 3e-14], [2e-14, 4e-14]])
    assert_allclose(output["tracer_diffusivity"], [1e-14, 2e-14, 3e-14])
    diagnostics = output["diagnostics"]
    assert diagnostics["thermodynamic_factors"] == [[1, None], [2, 4]]
    assert diagnostics["errors"]["thermodynamic_factor[TI,FE]"] == "factor unavailable"
    assert diagnostics["stable_composition_sets"] == ["BCC_B2#1"]
    assert diagnostics["phase_composition"] == [0.5, 0.3, 0.2]
    assert diagnostics["site_fractions"] == [
        {"sublattice": 1, "constituents": {"Ti": 0.3, "W": 0.7}}
    ]

    class FailingTracerResult(Result):
        def get_value_of(self, quantity):
            if quantity == ("tracer", "BCC_B2#1", "Fe"):
                raise RuntimeError("tracer unavailable")
            return super().get_value_of(quantity)

    backend._calculate = lambda kind, phase, x, T: FailingTracerResult()
    partial = backend.calculate_kinetics(np.array([0.3, 0.2]), 1973.0, "BCC_B2#1", collect_diagnostics=True)
    assert_allclose(partial["interdiffusivity"], output["interdiffusivity"])
    assert partial["tracer_diffusivity"] == [1e-14, 2e-14, None]
    assert partial["diagnostics"]["errors"]["tracer_diffusivity[FE]"] == "tracer unavailable"


def test_calculation_site_fraction_capture_records_equilibrium_sets_and_kinetics():
    """Capture phase states from each actual fake TC result in call order."""
    class Quantities:
        @staticmethod
        def composition_of_phase_as_mole_fraction(phase, element):
            return ("composition", phase, element)

        @staticmethod
        def mole_fraction_of_a_phase(phase):
            return ("amount", phase)

    class Species:
        def __init__(self, name):
            self.name = name

        def get_name(self):
            return self.name

    class Sublattice:
        def get_nr_of_sites(self):
            return 1.0

        def get_constituents(self):
            return {Species(name) for name in ("W", "Ti", "Fe")}

    class Phase:
        def get_sublattices(self):
            return [Sublattice(), Sublattice()]

    class Result:
        def __init__(self, stable):
            self.stable = stable

        def get_stable_phases(self):
            return self.stable

        def get_value_of(self, quantity):
            if isinstance(quantity, str) and quantity.startswith("Y("):
                phase, site = quantity[2:-1].split(",")
                constituent, sublattice = site.rsplit("#", 1)
                quantity = ("site", phase, constituent, int(sublattice))
            kind, phase, *rest = quantity
            if kind == "composition":
                return {"W": 0.5, "Ti": 0.3, "Fe": 0.2}[rest[0]]
            if kind == "amount":
                return 1.0 / len(self.stable)
            if kind == "site":
                constituent, sublattice = rest
                if phase == "BCC_B2#3" and constituent == "Ti" and sublattice == 2:
                    raise RuntimeError("site fraction unavailable")
                return ({"W": 0.8, "Ti": 0.15, "Fe": 0.05} if sublattice == 1 else
                        {"W": 0.2, "Ti": 0.45, "Fe": 0.35})[constituent]
            raise AssertionError(quantity)

    class Calculation(FakeTCCalculation):
        def __init__(self, result):
            super().__init__([])
            self.result = result

        def set_condition(self, quantity, value):
            pass

        def calculate(self):
            return self.result

    class System:
        def __init__(self):
            self.results = [Result(["BCC_B2#1", "BCC_B2#3"]), Result(["BCC_B2#1"])]

        def with_single_equilibrium_calculation(self):
            return Calculation(self.results.pop(0))

        def get_phase_object(self, phase):
            return Phase()

    config = ThermoCalcConfig(
        thermodynamic_database="TCHEA5", kinetic_database="MOBHEA4",
        elements=("W", "TI", "FE"), phases=("BCC_B2#1", "LIQUID#1"),
        reference_element="W", use_default_phases=False,
    )
    backend = _TCPythonBackend()
    backend._config = config
    backend._setup = object()
    backend._systems[False] = System()
    backend._tc_python = type("TC", (), {"ThermodynamicQuantity": Quantities})()
    backend._set_conditions = lambda calc, x, T: None
    backend.totalNumCalcs = 0
    backend.total_kind_lst = []
    backend.total_x_lst = []
    records = []
    therm = TCPythonThermodynamics(config, backend=backend)
    with therm.captureCalculationSiteFractions(records.append):
        backend._calculate("equilibrium", None, np.array([0.3, 0.2]), 1973.0)
        backend._calculate("kinetics", "BCC_B2#1", np.array([0.3, 0.2]), 1973.0)

    assert [record["kind"] for record in records] == ["equilibrium", "kinetics"]
    assert [record["calculation_index"] for record in records] == [0, 1]
    assert records[0]["stable_composition_sets"] == ["BCC_B2#1", "BCC_B2#3"]
    assert [state["phase"] for state in records[0]["phases"]] == ["BCC_B2#1", "BCC_B2#3"]
    assert records[0]["phases"][1]["site_fractions"][1]["constituents"]["Ti"] is None
    assert records[0]["phases"][1]["diagnostic_errors"]["sublattice[2].Ti"] == "site fraction unavailable"
    assert records[1]["input_full_composition"] == [0.5, 0.3, 0.2]
    state = records[1]["phases"][0]
    assert state["phase_composition"] == [0.5, 0.3, 0.2]
    assert state["site_fractions"][0] == {
        "sublattice": 1, "site_ratio": 1.0,
        "constituents": {"Fe": 0.05, "Ti": 0.15, "W": 0.8},
    }
    assert state["diagnostic_errors"] == {}
    assert backend._site_fraction_capture_callback is None


def test_calculation_site_fraction_capture_includes_failed_retry_attempts():
    qthiss = "ERROR IN QTHISS : TOO MANY ITERATIONS"
    backend, _ = _qthiss_retry_backend({2000: [RuntimeError(qthiss)]})
    therm = TCPythonThermodynamics(backend._config, backend=backend)
    records = []
    with therm.captureCalculationSiteFractions(records.append):
        backend._calculate("equilibrium", None, np.array([0.3, 0.2]), 1373.0)
    assert [record["status"] for record in records] == ["calculate_error", "ok"]
    assert [record["grid_points"] for record in records] == [2000, 20_000]
    assert qthiss in records[0]["error"]
    assert records[1]["diagnostic_errors"]["stable_composition_sets"]


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


def test_saved_surrogate_matrices_match_captured_kinetics():
    therm = TCPythonThermodynamics(
        config=_fecrni_config(), backend=FakeThermoCalcBackend(), default_remove_cache=False
    )
    path = Path("examples") / "ThermoCalc" / "outputs" / "test_mb_surrogate_capture.npz"
    sidecar = path.with_suffix(".jsonl")
    with sidecar.open("w", encoding="utf-8") as output:
        output.write(json.dumps({"record_type": "metadata", "schema_version": 1}) + "\n")
        with therm.captureKineticsDiagnostics(lambda record: output.write(json.dumps(record) + "\n")):
            surrogate = build_moving_boundary_surrogate(
                therm,
                temperature=1373.0,
                probe_start=(0.25, 0.068),
                probe_end=(0.45, 0.242),
                eta_samples=[0.0, 0.5, 1.0],
                diffusivity_bulk_points=[[0.3, 0.2], [0.35, 0.1]],
                output_path=path,
            )
    loaded = type(surrogate).load(path)
    records = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()[1:]]
    assert records
    for phase in loaded.tieline_phases:
        for context in ("interface", "general"):
            for composition, matrix in zip(
                loaded.diffusivity_compositions[context][phase], loaded.diffusivities[context][phase]
            ):
                assert any(
                    record["requested_phase"] == phase
                    and np.allclose(record["input_composition"], composition, rtol=0, atol=1e-14)
                    and np.allclose(record["interdiffusivity"], matrix)
                    for record in records
                )


@pytest.mark.parametrize("interpolation", ["nearest", "simplex_linear", "continuous_grid"])
def test_bulk_calculate_failures_are_optional_and_recorded(interpolation):
    failed_point = np.array([0.35, 0.16])

    class FailingBulkBackend(FakeThermoCalcBackend):
        def calculate_kinetics(self, x, T, phase, collect_diagnostics=False):
            if phase == "BCC_A2" and np.allclose(x, failed_point, rtol=0, atol=1e-15):
                raise ThermoCalcSolveError("TC-Python kinetics calculation failed: sample failure")
            return super().calculate_kinetics(x, T, phase, collect_diagnostics=collect_diagnostics)

    grids = (np.array([0.25, 0.35, 0.45]), np.array([0.08, 0.16, 0.24]))
    points = np.array([[0.25, 0.08], [0.35, 0.16], [0.45, 0.24], [0.45, 0.08]])
    kwargs = (
        {"diffusivity_bulk_grids": grids} if interpolation == "continuous_grid"
        else {"diffusivity_bulk_points": points}
    )
    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=FailingBulkBackend())
    with pytest.raises(ThermoCalcSolveError, match="sample failure"):
        build_moving_boundary_surrogate(therm, diffusivity_interpolation=interpolation, **kwargs)

    path = Path("examples") / "ThermoCalc" / "outputs" / f"test_filtered_bulk_{interpolation}.npz"
    try:
        surrogate = build_moving_boundary_surrogate(
            therm, diffusivity_interpolation=interpolation,
            skip_failed_bulk_calculations=True, output_path=path, **kwargs,
        )
        loaded = type(surrogate).load(path)
    finally:
        path.unlink(missing_ok=True)
    failures = loaded.metadata["failed_bulk_points"]
    assert failures == [{
        "phase": "BCC_A2", "composition": failed_point.tolist(),
        "temperature": 1373.0, "bulk_point_index": 4 if interpolation == "continuous_grid" else 1,
            "error_type": "ThermoCalcSolveError",
            "error": "TC-Python kinetics calculation failed: sample failure",
            "domain_status": "UNKNOWN",
            "domain_reason": "source_calculation_failed",
        }]
    assert loaded.metadata["requested_diffusivity_interpolation"] == interpolation
    assert loaded.metadata["effective_diffusivity_interpolation"] == (
        "simplex_linear" if interpolation == "continuous_grid" else interpolation
    )
    assert not np.any(np.all(np.isclose(
        loaded.diffusivity_compositions["general"]["BCC_A2"], failed_point, rtol=0, atol=1e-15,
    ), axis=1))
    assert np.any(np.all(np.isclose(
        loaded.diffusivity_compositions["general"]["FCC_A1"], failed_point, rtol=0, atol=1e-15,
    ), axis=1))


def test_bulk_skip_keeps_non_calculate_failures_as_errors():
    class FailedQuantityBackend(FakeThermoCalcBackend):
        def calculate_kinetics(self, x, T, phase, collect_diagnostics=False):
            if phase == "BCC_A2" and np.allclose(x, [0.35, 0.16], rtol=0, atol=1e-15):
                raise ThermoCalcCalculationError("quantity query failed")
            return super().calculate_kinetics(x, T, phase, collect_diagnostics=collect_diagnostics)

    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=FailedQuantityBackend())
    with pytest.raises(ThermoCalcCalculationError, match="quantity query failed"):
        build_moving_boundary_surrogate(
            therm, diffusivity_bulk_points=[[0.35, 0.16]], skip_failed_bulk_calculations=True,
        )


def test_all_failed_bulk_samples_fall_back_to_nearest():
    class FailingBulkBackend(FakeThermoCalcBackend):
        def calculate_kinetics(self, x, T, phase, collect_diagnostics=False):
            if np.allclose(x, [0.35, 0.16], rtol=0, atol=1e-15):
                raise ThermoCalcSolveError("sample failure")
            return super().calculate_kinetics(x, T, phase, collect_diagnostics=collect_diagnostics)

    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=FailingBulkBackend())
    surrogate = build_moving_boundary_surrogate(
        therm, diffusivity_interpolation="simplex_linear",
        diffusivity_bulk_points=[[0.35, 0.16]], skip_failed_bulk_calculations=True,
    )
    assert surrogate.diffusivityInterpolation == "nearest"
    assert len(surrogate.metadata["failed_bulk_points"]) == 2


@pytest.mark.parametrize("interpolation", ["nearest", "simplex_linear", "continuous_grid"])
def test_invalid_bulk_matrices_are_optionally_dropped_and_recorded(interpolation, monkeypatch):
    import kawin.diffusion.MovingBoundarySurrogates as surrogate_module

    invalid_point = np.array([0.35, 0.16])
    points = np.array([[0.25, 0.08], [0.35, 0.16], [0.45, 0.24], [0.45, 0.08]])
    grids = (np.array([0.25, 0.35, 0.45]), np.array([0.08, 0.16, 0.24]))
    kwargs = (
        {"diffusivity_bulk_grids": grids} if interpolation == "continuous_grid"
        else {"diffusivity_bulk_points": points}
    )

    class InvalidBulkBackend(FakeThermoCalcBackend):
        def calculate_kinetics(self, x, T, phase, collect_diagnostics=False):
            output = super().calculate_kinetics(x, T, phase, collect_diagnostics=collect_diagnostics)
            if phase == "BCC_A2" and np.allclose(x, invalid_point, rtol=0, atol=1e-15):
                output["interdiffusivity"] = np.diag([1e-14, -1e-14])
            return output

    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=InvalidBulkBackend())
    if interpolation == "nearest":
        unchanged = build_moving_boundary_surrogate(therm, diffusivity_interpolation=interpolation, **kwargs)
        assert "invalid_bulk_points" not in unchanged.metadata
        assert np.any(np.all(np.isclose(
            unchanged.diffusivity_compositions["general"]["BCC_A2"], invalid_point, rtol=0, atol=1e-15,
        ), axis=1))

    monkeypatch.setattr(surrogate_module, "debugInPlace", lambda: pytest.fail("interactive debugger was invoked"))
    path = Path("examples") / "ThermoCalc" / "outputs" / f"test_invalid_bulk_{interpolation}.npz"
    try:
        filtered = build_moving_boundary_surrogate(
            therm, diffusivity_interpolation=interpolation,
            drop_invalid_bulk_matrices=True, output_path=path, **kwargs,
        )
        loaded = type(filtered).load(path)
    finally:
        path.unlink(missing_ok=True)

    dropped = loaded.metadata["invalid_bulk_points"]
    assert len(dropped) == 1
    assert dropped[0]["phase"] == "BCC_A2"
    assert dropped[0]["composition"] == invalid_point.tolist()
    assert dropped[0]["bulk_point_index"] == (4 if interpolation == "continuous_grid" else 1)
    assert dropped[0]["temperature"] == 1373.0
    assert dropped[0]["error_type"] == "ValueError"
    assert "positive real eigenvalues" in dropped[0]["error"]
    assert "-1.e-14" in dropped[0]["diffusivity_repr"]
    assert loaded.metadata["effective_diffusivity_interpolation"] == (
        "simplex_linear" if interpolation == "continuous_grid" else interpolation
    )
    assert not np.any(np.all(np.isclose(
        loaded.diffusivity_compositions["general"]["BCC_A2"], invalid_point, rtol=0, atol=1e-15,
    ), axis=1))
    assert np.any(np.all(np.isclose(
        loaded.diffusivity_compositions["general"]["FCC_A1"], invalid_point, rtol=0, atol=1e-15,
    ), axis=1))


def test_invalid_bulk_validation_waits_for_all_queries_and_drops_nonfinite(monkeypatch):
    import kawin.diffusion.MovingBoundarySurrogates as surrogate_module

    points = np.array([[0.25, 0.08], [0.35, 0.16], [0.45, 0.08]])
    sampled_bulk = []
    validation_started = []

    class NonfiniteBulkBackend(FakeThermoCalcBackend):
        def calculate_kinetics(self, x, T, phase, collect_diagnostics=False):
            if any(np.allclose(x, point, rtol=0, atol=1e-15) for point in points):
                sampled_bulk.append((phase, tuple(x)))
            output = super().calculate_kinetics(x, T, phase, collect_diagnostics=collect_diagnostics)
            if phase == "BCC_A2" and np.allclose(x, points[1], rtol=0, atol=1e-15):
                output["interdiffusivity"][0, 0] = np.nan
            return output

    original_validator = surrogate_module._validate_positive_2x2_matrix

    def record_validation(matrix, label, **kwargs):
        if label.startswith("bulk diffusivity") and not validation_started:
            validation_started.append(len(sampled_bulk))
        return original_validator(matrix, label, **kwargs)

    monkeypatch.setattr(surrogate_module, "_validate_positive_2x2_matrix", record_validation)
    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=NonfiniteBulkBackend())
    filtered = build_moving_boundary_surrogate(
        therm, diffusivity_bulk_points=points, drop_invalid_bulk_matrices=True,
    )

    assert validation_started == [len(points) * 2]
    assert len(filtered.metadata["invalid_bulk_points"]) == 1
    assert "finite 2x2 matrix" in filtered.metadata["invalid_bulk_points"][0]["error"]


def test_calculation_and_invalid_matrix_filters_are_independent(tmp_path):
    class MixedBulkBackend(FakeThermoCalcBackend):
        def calculate_kinetics(self, x, T, phase, collect_diagnostics=False):
            if phase == "BCC_A2" and np.allclose(x, [0.25, 0.08], rtol=0, atol=1e-15):
                raise ThermoCalcSolveError("calculate failed")
            output = super().calculate_kinetics(x, T, phase, collect_diagnostics=collect_diagnostics)
            if phase == "BCC_A2" and np.allclose(x, [0.35, 0.16], rtol=0, atol=1e-15):
                output["interdiffusivity"] = np.zeros((2, 2))
            return output

    therm = TCPythonThermodynamics(config=_fecrni_config(), backend=MixedBulkBackend())
    filtered = build_moving_boundary_surrogate(
        therm, diffusivity_bulk_points=[[0.25, 0.08], [0.35, 0.16], [0.45, 0.08]],
        skip_failed_bulk_calculations=True, drop_invalid_bulk_matrices=True,
    )
    assert len(filtered.metadata["failed_bulk_points"]) == 1
    assert len(filtered.metadata["invalid_bulk_points"]) == 1
    assert filtered.metadata["failed_bulk_points"][0]["composition"] == [0.25, 0.08]
    assert filtered.metadata["invalid_bulk_points"][0]["composition"] == [0.35, 0.16]
    domain = filtered.diffusivityValidity["general"]["BCC_A2"]
    for point, expected in (([0.25, 0.08], DiffusivityDomainStatus.UNKNOWN),
                            ([0.35, 0.16], DiffusivityDomainStatus.KNOWN_INVALID)):
        assert domain.classify(point)[0] is expected
        with pytest.raises(DiffusivityDomainError) as error:
            filtered.getInterdiffusivity(point, phase="BCC_A2")
        assert error.value.status is expected
    fit_points, _ = filtered._fit_diffusivity_samples("general", "BCC_A2")
    assert not any(np.allclose(point, [0.25, 0.08]) or np.allclose(point, [0.35, 0.16]) for point in fit_points)
    path = tmp_path / "both_drop_validity_roundtrip.npz"
    try:
        filtered.save(path)
        restored = type(filtered).load(path)
    finally:
        path.unlink(missing_ok=True)
    restored_domain = restored.diffusivityValidity["general"]["BCC_A2"]
    for point, expected, reason in (([0.25, 0.08], DiffusivityDomainStatus.UNKNOWN, "source_calculation_failed"),
                                    ([0.35, 0.16], DiffusivityDomainStatus.KNOWN_INVALID, "zero_matrix_norm")):
        exact = restored_domain._exact_index(point)
        assert restored_domain.classify(point)[0] is expected
        assert restored_domain.reasons[exact] == reason
        assert not restored_domain.fit_usable[exact]


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
