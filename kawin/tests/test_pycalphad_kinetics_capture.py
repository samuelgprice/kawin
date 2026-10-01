from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from pycalphad import variables as v

import kawin.thermo.Thermodynamics as thermodynamics_module
from kawin.thermo.Thermodynamics import GeneralThermodynamics
from kawin.thermo import MulticomponentThermodynamics


class _Species:
    def __init__(self, name):
        self.name = name


class _SiteVariable:
    def __init__(self, sublattice, species):
        self.sublattice_index = sublattice
        self.species = _Species(species)


class _PhaseRecord:
    nonvacant_elements = ("CR", "FE", "NI")
    state_variables = (v.T,)
    variables = (
        _SiteVariable(0, "CR"),
        _SiteVariable(0, "FE"),
        _SiteVariable(0, "NI"),
        _SiteVariable(1, "CR"),
        _SiteVariable(1, "NI"),
    )


class _CompositionSet:
    phase_record = _PhaseRecord()
    X = np.asarray([0.3, 0.5, 0.2])
    dof = np.asarray([1373.0, 0.3, 0.5, 0.2, 0.6, 0.4])


def _thermodynamics(monkeypatch, *, mobility_failure=False):
    thermodynamics = object.__new__(GeneralThermodynamics)
    thermodynamics.elements = ["FE", "NI", "CR", "VA"]
    thermodynamics.numElements = 3
    thermodynamics.phases = ["FCC_A1"]
    thermodynamics.mobCallables = {"FCC_A1": {"CR": object(), "FE": object(), "NI": object()}}
    thermodynamics.diffCallables = {"FCC_A1": None}
    thermodynamics.mobility_correction = {"CR": 1.0, "FE": 1.0, "NI": 1.0}
    thermodynamics.vacancyPoorInterstitialSublattice = {}
    thermodynamics._parameters = {}
    thermodynamics._diffusivity_cache = {}
    thermodynamics._kinetics_diagnostic_callback = None
    thermodynamics._kinetics_diagnostic_index = 0
    result = type("Result", (), {"chemical_potentials": np.asarray([1.0, 2.0, 3.0])})()
    thermodynamics.local_equilibrium_calls = 0

    def get_local_equilibrium(*args, **kwargs):
        thermodynamics.local_equilibrium_calls += 1
        return result, [_CompositionSet()]

    thermodynamics.getLocalEq = get_local_equilibrium

    base_matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    base_factor = np.asarray([[10.0, 20.0], [30.0, 40.0]])
    monkeypatch.setattr(
        thermodynamics_module,
        "inverseMobility",
        lambda *args, **kwargs: (base_matrix.copy(), base_factor.copy(), np.eye(2)),
    )
    monkeypatch.setattr(
        thermodynamics_module,
        "tracer_diffusivity",
        lambda *args, **kwargs: np.asarray([30.0, 50.0, 70.0]),
    )
    if mobility_failure:
        def fail(*args, **kwargs):
            raise RuntimeError("mobility unavailable")

        monkeypatch.setattr(thermodynamics_module, "mobility_from_composition_set", fail)
    else:
        monkeypatch.setattr(
            thermodynamics_module,
            "mobility_from_composition_set",
            lambda *args, **kwargs: np.asarray([3.0, 5.0, 7.0]),
        )
    return thermodynamics


def test_pycalphad_capture_uses_same_result_and_input_element_order(monkeypatch):
    thermodynamics = _thermodynamics(monkeypatch)
    records = []

    with thermodynamics.captureKineticsDiagnostics(records.append):
        matrix = thermodynamics.getInterdiffusivity(
            [0.2, 0.3], 1373.0, phase="FCC_A1"
        )

    assert_allclose(matrix, [[4.0, 3.0], [2.0, 1.0]])
    assert len(records) == 1
    record = records[0]
    assert_allclose(record["interdiffusivity"], matrix)
    assert_allclose(record["thermodynamic_factors"], [[40.0, 30.0], [20.0, 10.0]])
    assert_allclose(record["phase_composition"], [0.5, 0.2, 0.3])
    assert_allclose(record["tracer_diffusivities"], [50.0, 70.0, 30.0])
    assert_allclose(record["mobilities"], [5.0, 7.0, 3.0])
    assert record["site_fractions"] == [
        {"sublattice": 1, "constituents": {"CR": 0.3, "FE": 0.5, "NI": 0.2}},
        {"sublattice": 2, "constituents": {"CR": 0.6, "NI": 0.4}},
    ]
    assert record["kinetics_strategy"] == "pycalphad_local_single_phase"
    assert record["diagnostic_errors"] == {}
    assert thermodynamics.local_equilibrium_calls == 1


def test_pycalphad_capture_orders_batches_and_restores_nested_scopes(monkeypatch):
    thermodynamics = _thermodynamics(monkeypatch)
    outer = []
    inner = []

    with thermodynamics.captureKineticsDiagnostics(outer.append):
        thermodynamics.getInterdiffusivity([0.2, 0.3], 1373.0, phase="FCC_A1")
        with thermodynamics.captureKineticsDiagnostics(inner.append):
            thermodynamics.getInterdiffusivity(
                [[0.2, 0.3], [0.25, 0.2]], 1373.0, phase="FCC_A1"
            )
        thermodynamics.getInterdiffusivity([0.3, 0.1], 1373.0, phase="FCC_A1")

    assert [record["query_index"] for record in inner] == [0, 1]
    assert [record["query_index"] for record in outer] == [0, 1]
    assert [record["input_composition"] for record in inner] == [[0.2, 0.3], [0.25, 0.2]]


def test_optional_pycalphad_diagnostic_failure_keeps_matrix(monkeypatch):
    thermodynamics = _thermodynamics(monkeypatch, mobility_failure=True)
    records = []

    with thermodynamics.captureKineticsDiagnostics(records.append):
        matrix = thermodynamics.getInterdiffusivity([0.2, 0.3], 1373.0, phase="FCC_A1")

    assert_allclose(matrix, [[4.0, 3.0], [2.0, 1.0]])
    assert records[0]["mobilities"] is None
    assert "mobility unavailable" in records[0]["diagnostic_errors"]["mobilities"]


def test_diffusivity_parameter_capture_reports_equivalent_mobility(monkeypatch):
    thermodynamics = _thermodynamics(monkeypatch)
    thermodynamics.mobCallables["FCC_A1"] = None
    thermodynamics.diffCallables["FCC_A1"] = {"CR": object(), "FE": object(), "NI": object()}
    base_matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    base_factor = np.asarray([[10.0, 20.0], [30.0, 40.0]])
    tracer = np.asarray([30.0, 50.0, 70.0])
    monkeypatch.setattr(
        thermodynamics_module,
        "inverseMobility_from_diffusivity",
        lambda *args, **kwargs: (base_matrix.copy(), base_factor.copy(), np.eye(2)),
    )
    monkeypatch.setattr(
        thermodynamics_module,
        "tracer_diffusivity_from_diff",
        lambda *args, **kwargs: tracer.copy(),
    )
    records = []

    with thermodynamics.captureKineticsDiagnostics(records.append):
        thermodynamics.getInterdiffusivity([0.2, 0.3], 1373.0, phase="FCC_A1")

    assert_allclose(records[0]["tracer_diffusivities"], [50.0, 70.0, 30.0])
    assert_allclose(
        records[0]["mobilities"],
        np.asarray([50.0, 70.0, 30.0]) / (8.3145 * 1373.0),
    )


def test_real_fecrni_pycalphad_capture_smoke():
    database = Path("examples") / "FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb"
    thermodynamics = MulticomponentThermodynamics(
        str(database), ["FE", "CR", "NI"], ["FCC_A1", "BCC_A2"]
    )
    records = []

    with thermodynamics.captureKineticsDiagnostics(records.append):
        matrix = thermodynamics.getInterdiffusivity(
            [0.38, 0.001], 1373.0, phase="FCC_A1"
        )

    assert np.asarray(matrix).shape == (2, 2)
    assert np.all(np.isfinite(matrix))
    assert len(records) == 1
    assert_allclose(records[0]["interdiffusivity"], matrix)
    assert np.all(np.isfinite(records[0]["mobilities"]))
    assert np.all(np.isfinite(records[0]["thermodynamic_factors"]))
    assert records[0]["site_fractions"]
