"""Package extensions for stored samples and genuine optional calculation records."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from moving_boundary import TernaryMovingBoundaryThermodynamicsSurrogate
from moving_boundary.surrogate_diagnostics import plot_selected_diffusivity_calculations
from moving_boundary.compatibility import thermo as local


@pytest.fixture
def sampled():
    points = np.array([[.2, .1], [.3, .1], [.2, .2]])
    matrices = np.broadcast_to([[2., .2], [.1, 1.]], (3, 2, 2)).copy()
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=("FE", "CR", "NI"), phases=("ALPHA", "BETA"),
        tieline_phases=("ALPHA", "BETA"), temperature=1000.,
        eta_samples=np.array([0., .5, 1.]),
        tieline_compositions={"ALPHA": points, "BETA": points + .1},
        diffusivity_compositions={c: {p: points for p in ("ALPHA", "BETA")} for c in ("interface", "general")},
        diffusivities={c: {p: matrices for p in ("ALPHA", "BETA")} for c in ("interface", "general")},
        diffusivity_interpolation="nearest")


def test_sample_values_need_no_records_or_surrogate_queries(sampled, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Plot must not evaluate the surrogate")
    monkeypatch.setattr(sampled, "getInterdiffusivity", refuse)
    fig = plot_selected_diffusivity_calculations(sampled, renderer=None)
    assert len(fig.data) == 4
    assert "TC" not in fig.layout.title.text
    for trace in fig.data:
        np.testing.assert_array_equal(trace.marker.color, np.full(3, np.log10(np.linalg.norm([[2., .2], [.1, 1.]]))))
        np.testing.assert_allclose(trace.a, [.1, .1, .2])
        assert all("Calculation diagnostics: unavailable" in text for text in trace.text)
        assert all("[2, 0.2]" in text for text in trace.text)


@pytest.mark.parametrize("backend", ["pycalphad", "thermocalc"])
@pytest.mark.parametrize("from_file", [False, True])
def test_optional_backend_records_are_matched(sampled, backend, from_file, tmp_path):
    records = []
    for phase in sampled.tieline_phases:
        for i, (point, matrix) in enumerate(zip(sampled.diffusivity_compositions["interface"][phase], sampled.diffusivities["interface"][phase])):
            records.append({"record_type": "kinetics", "backend": backend,
                            "query_index": i, "requested_phase": phase,
                            "input_composition": point.tolist(), "interdiffusivity": matrix.tolist(),
                            "site_fractions": None, "mobilities": None})
    source = records
    if from_file:
        source = tmp_path / "records.jsonl"
        source.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
    figure = plot_selected_diffusivity_calculations(sampled, source, renderer=None)
    assert all(f"Backend: {backend}" in text for trace in figure.data for text in trace.text)
    assert all("sublattice site fractions: unavailable" in text for trace in figure.data for text in trace.text)
    records[0]["interdiffusivity"][0][0] = 10.
    with pytest.raises(ValueError, match="Could not match"):
        plot_selected_diffusivity_calculations(sampled, records, renderer=None)


@pytest.fixture
def fake_thermo(monkeypatch):
    """Stock and capture branches share a synthetic equilibrium and unsorted factors."""
    therm = local.MulticomponentThermodynamics.__new__(local.MulticomponentThermodynamics)
    therm.elements, therm.phases, therm.numElements = ["FE", "NI", "CR", "VA"], ["ALPHA"], 3
    therm.mobCallables, therm.diffCallables = {"ALPHA": object()}, {"ALPHA": object()}
    therm.mobility_correction, therm._parameters = {}, {}
    therm.vacancyPoorInterstitialSublattice, therm._diffusivity_cache = {}, {}
    variable = SimpleNamespace(sublattice_index=0, species=SimpleNamespace(name="FE"))
    phase_record = SimpleNamespace(nonvacant_elements=["CR", "FE", "NI"],
        state_variables=[local.v.T], variables=[variable], phase_name="ALPHA")
    cs = SimpleNamespace(phase_record=phase_record, dof=np.array([1000., .7]), X=np.array([.2, .7, .1]))
    calls = []
    def equilibrium(*args, **kwargs):
        calls.append(kwargs["composition_sets"])
        return SimpleNamespace(chemical_potentials=np.array([1., 2., 3.])), [cs]
    therm.getLocalEq = equilibrium
    matrix, factors = np.array([[2., .3], [.1, 1.]]), np.array([[10., 20.], [30., 40.]])
    def inverse(*args, **kwargs):
        return matrix.copy(), factors.copy(), None
    # Importing the dotted module through importlib avoids package export aliases.
    import importlib
    host_module = importlib.import_module("kawin.thermo.Thermodynamics")
    for module in (local, host_module):
        monkeypatch.setattr(module, "inverseMobility", inverse)
        monkeypatch.setattr(module, "inverseMobility_from_diffusivity", inverse)
    monkeypatch.setattr(local, "tracer_diffusivity", lambda *a, **k: np.array([1., 2., 3.]))
    monkeypatch.setattr(local, "tracer_diffusivity_from_diff", lambda *a, **k: np.array([1., 2., 3.]))
    monkeypatch.setattr(local, "mobility_from_composition_set", lambda *a, **k: np.array([4., 5., 6.]))
    return therm, calls


@pytest.mark.parametrize("diffusion_model", [False, True])
@pytest.mark.parametrize("keep_cache", [False, True])
def test_capture_preserves_matrix_cache_and_order(fake_thermo, diffusion_model, keep_cache):
    therm, calls = fake_thermo
    if diffusion_model:
        therm.mobCallables["ALPHA"] = None
    normal = therm.getInterdiffusivity([.1, .2], 1000., removeCache=not keep_cache)
    records = []
    with therm.captureKineticsDiagnostics(records.append):
        captured = therm.getInterdiffusivity([.1, .2], 1000., removeCache=not keep_cache)
    assert len(calls) == 2 and len(records) == 1
    np.testing.assert_array_equal(captured, normal)
    np.testing.assert_array_equal(records[0]["interdiffusivity"], normal)
    np.testing.assert_array_equal(records[0]["thermodynamic_factors"], [[40., 30.], [20., 10.]])
    np.testing.assert_array_equal(records[0]["phase_composition"], [.7, .1, .2])
    np.testing.assert_array_equal(records[0]["tracer_diffusivities"], [2., 3., 1.])
    assert records[0]["cache_seeded"] is keep_cache
    assert records[0]["cache_hit"] is False
    assert (therm._diffusivity_cache["ALPHA"] is not None) is keep_cache
    assert records[0]["backend"] == "pycalphad"
    assert records[0]["site_fractions"] == [{"sublattice": 1, "constituents": {"FE": .7}}]
    assert not records[0]["diagnostic_errors"]
    json.dumps(records, allow_nan=False)


def test_nested_capture_and_interrupt_restore_scope(fake_thermo):
    therm, _ = fake_thermo
    outer, inner = [], []
    with pytest.raises(TypeError, match="callable"):
        with therm.captureKineticsDiagnostics(None):
            pass
    with therm.captureKineticsDiagnostics(outer.append):
        therm.getInterdiffusivity([.1, .2], 1000.)
        with pytest.raises(KeyboardInterrupt):
            with therm.captureKineticsDiagnostics(inner.append):
                therm.getInterdiffusivity([.1, .2], 1000.)
                raise KeyboardInterrupt
        therm.getInterdiffusivity([.1, .2], 1000.)
    assert [record["query_index"] for record in outer] == [0, 1]
    assert [record["query_index"] for record in inner] == [0]
    assert not any(name.startswith("_kinetics_diagnostic_") for name in vars(therm))


def test_optional_metadata_failure_does_not_replace_matrix(fake_thermo, monkeypatch):
    therm, calls = fake_thermo
    def fail(*args, **kwargs):
        raise ValueError("metadata unavailable")
    monkeypatch.setattr(local, "mobility_from_composition_set", fail)
    records = []
    with therm.captureKineticsDiagnostics(records.append):
        result = therm.getInterdiffusivity([.1, .2], 1000.)
    np.testing.assert_array_equal(result, [[1., .1], [.3, 2.]])
    assert len(calls) == 1 and records[0]["mobilities"] is None
    assert "metadata unavailable" in records[0]["diagnostic_errors"]["mobilities"]


@pytest.mark.parametrize("defaults", [False, True])
def test_real_pycalphad_capture_performs_one_solve_per_query(defaults, monkeypatch):
    from pathlib import Path
    from moving_boundary.providers.pycalphad import create_pycalphad_thermodynamics_source
    database = Path(__file__).parents[1] / "examples/FeCrNi_MatCalc_open.tdb"
    source = create_pycalphad_thermodynamics_source(database, ["FE", "CR", "NI"], ["BCC_A2", "FCC_A1"],
                                                  use_default_phases=defaults, g_offset=0.)
    therm = source.thermodynamics if defaults else source
    solve = therm.getLocalEq
    count = []
    def counted(*args, **kwargs):
        count.append(1)
        return solve(*args, **kwargs)
    monkeypatch.setattr(therm, "getLocalEq", counted)
    normal = source.getInterdiffusivity([.39696, .0001], 1373., phase="BCC_A2")
    records = []
    with source.captureKineticsDiagnostics(records.append):
        captured = source.getInterdiffusivity([.39696, .0001], 1373., phase="BCC_A2")
    assert len(count) == 2 and len(records) == 1
    np.testing.assert_array_equal(captured, normal)
    np.testing.assert_array_equal(records[0]["interdiffusivity"], normal)
    assert len(records[0]["mobilities"]) == 3 and len(records[0]["thermodynamic_factors"]) == 2
    assert records[0]["site_fractions"] and records[0]["diagnostic_errors"] == {}
    json.dumps(records, allow_nan=False)
