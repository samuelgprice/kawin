"""Licence-free contracts for the user-database vacancy (VA) system selection.

A user TDB that declares ``ELEMENT VA`` without ``DEFAULT_COMMAND DEF_SYS_ELEMENT VA``
leaves VA out of the Thermo-Calc system, which silently drops FCC_A1/BCC_A2. The adapter
then appends VA to Thermo-Calc's element selection only. These fakes model only the
system-builder inventory calls; they carry no thermodynamic data.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from moving_boundary.providers import thermocalc as tc


class FakeSystem:
    def __init__(self, database_elements, system_elements):
        self._database = list(database_elements)
        self._system = list(system_elements)

    def get_all_elements_in_databases(self):
        return list(self._database)

    def get_elements_in_system(self):
        return list(self._system)


class FakeBuilder:
    def __init__(self, setup, call):
        self.setup, self.call = setup, call

    def without_default_phases(self):
        self.call["without_default_phases"] = True
        return self

    def select_phase(self, phase):
        self.call.setdefault("phases", []).append(phase)
        return self

    def get_system(self):
        """Entered elements: the selection, plus VA when the database's default command adds it."""
        entered = list(self.call["elements"])
        if self.setup.default_va and "Va" in self.setup.database_elements:
            entered.append("Va")
        return FakeSystem(self.setup.database_elements, entered)


class FakeSetup:
    """Records every database/element selection; ``default_va`` models DEF_SYS_ELEMENT VA."""

    def __init__(self, database_elements=("Cr", "Fe", "Ni", "Va"), default_va=False):
        self.database_elements = list(database_elements)
        self.default_va = default_va
        self.calls = []

    def _record(self, kind, elements, **extra):
        call = dict(kind=kind, elements=list(elements), **extra)
        self.calls.append(call)
        return FakeBuilder(self, call)

    def select_user_database_and_elements(self, path, elements):
        return self._record("user", elements, path=path)

    def select_database_and_elements(self, database, elements):
        return self._record("thermodynamic", elements, database=database)

    def select_thermodynamic_and_kinetic_databases_with_elements(self, thermodynamic, kinetic, elements):
        return self._record("thermodynamic_kinetic", elements, database=(thermodynamic, kinetic))


def _config(**overrides):
    values = dict(user_database_path="open.tdb", elements=("FE", "CR", "NI"), phases=("BCC_A2", "FCC_A1"),
                  reference_element="FE", cache_dir=None)
    values.update(overrides)
    return tc.ThermoCalcConfig(**values)


def _backend(setup, config):
    backend = tc._TCPythonBackend()
    backend._setup, backend._config = setup, config
    return backend


@pytest.mark.parametrize("include_default_phases", [False, True])
def test_user_database_missing_vacancy_is_rebuilt_with_va_selected(include_default_phases):
    setup = FakeSetup()
    backend = _backend(setup, _config())
    system = backend._build_system(backend._config, include_default_phases)

    assert [c["elements"] for c in setup.calls] == [["Fe", "Cr", "Ni"], ["Fe", "Cr", "Ni", "VA"]]
    assert all(c["kind"] == "user" and c["path"] == "open.tdb" for c in setup.calls)
    for call in setup.calls:  # the phase-selection rule is reapplied identically
        if include_default_phases:
            assert "without_default_phases" not in call and "phases" not in call
        else:
            assert call["without_default_phases"] and call["phases"] == ["BCC_A2", "FCC_A1"]
    assert "VA" in {e.upper() for e in system.get_elements_in_system()}
    assert backend._system_selections[include_default_phases] == {
        "tc_selected_elements": ["Fe", "Cr", "Ni", "VA"], "user_database_vacancy_added": True}


@pytest.mark.parametrize("setup", [FakeSetup(default_va=True), FakeSetup(database_elements=("Cr", "Fe", "Ni"))],
                         ids=["vacancy_already_entered", "no_vacancy_declared"])
def test_user_database_selection_is_unchanged_when_vacancy_is_entered_or_absent(setup):
    backend = _backend(setup, _config())
    backend._build_system(backend._config, False)
    assert [c["elements"] for c in setup.calls] == [["Fe", "Cr", "Ni"]]
    assert backend._system_selections[False] == {
        "tc_selected_elements": ["Fe", "Cr", "Ni"], "user_database_vacancy_added": False}


@pytest.mark.parametrize("databases", [dict(thermodynamic_database="TCFE9"),
                                       dict(thermodynamic_database="TCFE9", kinetic_database="MOBFE4")],
                         ids=["thermodynamic", "thermodynamic_kinetic"])
@pytest.mark.parametrize("include_default_phases", [False, True])
def test_commercial_database_selection_is_untouched(databases, include_default_phases):
    setup = FakeSetup()
    config = _config(user_database_path=None, **databases)
    backend = _backend(setup, config)
    backend._build_system(config, include_default_phases)
    assert len(setup.calls) == 1 and setup.calls[0]["elements"] == ["Fe", "Cr", "Ni"]
    assert setup.calls[0]["kind"] == ("thermodynamic_kinetic" if "kinetic_database" in databases else "thermodynamic")
    assert backend._system_selections == {}
    assert backend.system_selection() == {}


def test_system_selection_covers_equilibrium_and_kinetics_systems():
    setup = FakeSetup()
    backend = _backend(setup, _config(use_default_phases=True))
    assert backend.system_selection() == {
        "default_phases": {"tc_selected_elements": ["Fe", "Cr", "Ni", "VA"], "user_database_vacancy_added": True},
        "selected_phases": {"tc_selected_elements": ["Fe", "Cr", "Ni", "VA"], "user_database_vacancy_added": True}}
    restricted = _backend(FakeSetup(), _config(use_default_phases=False))
    assert list(restricted.system_selection()) == ["selected_phases"]


def test_physical_component_contract_is_unchanged_by_internal_vacancy():
    """VA stays out of elements, independent inputs, conditions and phase compositions."""
    setup = FakeSetup()
    config = _config()
    facade = tc.TCPythonThermodynamics(config)
    backend = facade._backend
    backend._setup, backend._config = setup, config
    backend._build_system(config, False)

    assert config.elements == ("FE", "CR", "NI") and config.independent_elements == ("CR", "NI")
    assert facade.elements == ["FE", "CR", "NI"] and facade.independent_elements == ["CR", "NI"]
    assert facade.numElements == 3
    assert "VA" not in config.to_metadata()["elements"]

    conditions, queried = [], []
    fake_tq = SimpleNamespace(temperature=lambda: "T", pressure=lambda: "P", system_size=lambda: "N",
                              mole_fraction_of_a_component=lambda e: f"X({e})",
                              composition_of_phase_as_mole_fraction=lambda p, e: queried.append(e) or f"X({p},{e})")
    backend._tc_python = SimpleNamespace(ThermodynamicQuantity=fake_tq)
    calc = SimpleNamespace(set_condition=lambda q, v: conditions.append((q, v)))
    backend._set_conditions(calc, np.array([0.45, 0.21]), 1373.0)
    assert conditions == [("T", 1373.0), ("P", 101325.0), ("N", 1.0), ("X(Cr)", 0.45), ("X(Ni)", 0.21)]
    result = SimpleNamespace(get_value_of=lambda q: 1.0 / 3.0)
    composition = backend._phase_composition(result, "FCC_A1")
    assert composition.shape == (3,) and queried == ["Fe", "Cr", "Ni"]


def test_to_metadata_keys_are_unchanged_for_legacy_provenance():
    """The O06-normalized metadata document must not gain or lose configuration keys."""
    expected = {"thermodynamic_database", "kinetic_database", "user_database_path", "elements", "phases",
                "reference_element", "pressure", "use_default_phases", "global_minimization_max_grid_points",
                "equilibrium_qthiss_retry_grid_points", "kinetics_disable_global_minimization",
                "kinetics_disable_positive_definite_hessian", "kinetics_constrain_single_composition_set",
                "kinetics_multistart_mode", "kinetics_multistart_phases", "kinetics_session_restart_interval",
                "cache_dir", "timeout_seconds", "calculation_version"}
    assert set(_config().to_metadata()) == expected
    assert list(_config().to_metadata()) == list(tc.ThermoCalcConfig.__dataclass_fields__)


def test_explicit_vacancy_in_elements_is_not_duplicated():
    system = FakeSystem(["Cr", "Fe", "Ni", "Va"], ["Cr", "Fe", "Ni"])
    config = SimpleNamespace(elements=("FE", "CR", "NI", "VA"))
    assert tc._user_system_lacks_declared_vacancy(system, config) is False
    assert tc._user_system_lacks_declared_vacancy(system, SimpleNamespace(elements=("FE", "CR", "NI"))) is True


def test_facade_system_selection_starts_the_session_and_tolerates_minimal_backends():
    started = []
    backend = SimpleNamespace(start=lambda config: started.append(config))
    facade = tc.TCPythonThermodynamics(_config(), backend=backend)
    assert facade.getSystemSelection() == {}
    assert started == [facade.config]
