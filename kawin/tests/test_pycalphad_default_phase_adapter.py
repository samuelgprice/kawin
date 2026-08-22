from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from examples.ternaryExamples.pycalphad_default_phase_adapter import (
    PycalphadDefaultPhaseThermodynamics,
    create_pycalphad_thermodynamics_source,
    order_pycalphad_phase_universe,
    resolve_pycalphad_equilibrium_phases,
)
from kawin.diffusion.MovingBoundarySurrogates import _sample_expected_tieline
from kawin.thermo import MulticomponentThermodynamics


TDB_PATH = Path("examples") / "FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb"
ELEMENTS = ("FE", "CR", "NI")


class _FakePhaseRecord:
    def __init__(self, phase_name):
        self.phase_name = phase_name


class _FakeCompositionSet:
    def __init__(self, phase_name, full_composition, amount=0.5):
        self.phase_record = _FakePhaseRecord(phase_name)
        self.X = np.asarray(full_composition, dtype=np.float64)[[1, 0, 2]]
        self.NP = float(amount)


class _FakeWorkspace:
    def __init__(self, composition_sets):
        self._composition_sets = tuple(composition_sets)
        self.eq = type("_Eq", (), {"MU": np.asarray([[-1.0, -2.0, -3.0]], dtype=np.float64)})()

    def get_composition_sets(self):
        return list(self._composition_sets)


class _FakeWrappedThermodynamics:
    def __init__(self, composition_sets):
        self.composition_sets = tuple(composition_sets)

    def clearCache(self):
        pass

    def getEq(self, x, T, gExtra=0, precPhase=None):
        return _FakeWorkspace(self.composition_sets)

    def getInterdiffusivity(self, x, T, phase=None, **kwargs):
        scale = 1.0 if phase == "FCC_A1" else 2.0
        return scale * np.asarray([[1.0, 0.1], [0.2, 1.5]], dtype=np.float64)


def _fake_adapter(composition_sets):
    adapter = object.__new__(PycalphadDefaultPhaseThermodynamics)
    adapter.interface_phases = ("FCC_A1", "LIQUID")
    adapter.equilibrium_phases = ("FCC_A1", "LIQUID", "BCC_A2", "SIGMA")
    adapter.thermodynamics = _FakeWrappedThermodynamics(composition_sets)
    adapter.elements = ["FE", "CR", "NI", "VA"]
    adapter.phases = list(adapter.equilibrium_phases)
    adapter.phase_amount_tolerance = 1.0e-12
    return adapter


def test_resolve_default_phases_for_fecrni_tdb_includes_sigma():
    phases = resolve_pycalphad_equilibrium_phases(TDB_PATH, ELEMENTS)

    assert "SIGMA" in phases
    assert {"FCC_A1", "LIQUID", "BCC_A2"}.issubset(phases)


def test_order_phase_universe_preserves_canonical_equilibrium_order():
    phases = order_pycalphad_phase_universe(("LIQUID", "BCC_A2"), ("BCC_A2", "FCC_A1", "LIQUID", "SIGMA"))

    assert phases == ("BCC_A2", "FCC_A1", "LIQUID", "SIGMA")


def test_explicit_equilibrium_phase_override_bypasses_database_filtering():
    phases = resolve_pycalphad_equilibrium_phases("does-not-exist.tdb", ELEMENTS, equilibrium_phases=("FCC_A1", "LIQUID"))

    assert phases == ("FCC_A1", "LIQUID")


def test_valid_two_phase_metadata_returns_ordered_independent_endpoints():
    adapter = _fake_adapter(
        [
            _FakeCompositionSet("LIQUID", [0.20, 0.43, 0.37]),
            _FakeCompositionSet("FCC_A1", [0.36, 0.30, 0.34]),
        ]
    )

    x_alpha, x_beta, metadata = adapter.getInterfacialComposition([0.40, 0.30], 1650.0, precPhase="LIQUID", returnMeta=True)

    assert metadata["endpoint_phases"] == ("FCC_A1", "LIQUID")
    assert [endpoint["phase"] for endpoint in metadata["endpoints"]] == ["FCC_A1", "LIQUID"]
    assert_allclose(x_alpha, [0.30, 0.34])
    assert_allclose(x_beta, [0.43, 0.37])


def test_extra_stable_phase_metadata_makes_surrogate_sampling_reject_probe():
    adapter = _fake_adapter(
        [
            _FakeCompositionSet("FCC_A1", [0.36, 0.30, 0.34]),
            _FakeCompositionSet("LIQUID", [0.20, 0.43, 0.37]),
            _FakeCompositionSet("SIGMA", [0.50, 0.40, 0.10]),
        ]
    )

    with pytest.raises(ValueError, match="SIGMA"):
        _sample_expected_tieline(adapter, [0.40, 0.30], 1650.0, "LIQUID", ("FCC_A1", "LIQUID"), ELEMENTS, 1.0e-10)


def test_factory_selects_default_or_restricted_pycalphad_phase_universe():
    default_source = create_pycalphad_thermodynamics_source(
        TDB_PATH,
        ELEMENTS,
        ("FCC_A1", "LIQUID"),
        use_default_phases=True,
    )
    restricted_source = create_pycalphad_thermodynamics_source(
        TDB_PATH,
        ELEMENTS,
        ("FCC_A1", "LIQUID"),
        use_default_phases=False,
    )

    assert isinstance(default_source, PycalphadDefaultPhaseThermodynamics)
    assert {"FCC_A1", "LIQUID"}.issubset(default_source.phases)
    assert "SIGMA" in default_source.phases
    assert default_source.thermodynamics.gOffset == 0.0
    assert isinstance(restricted_source, MulticomponentThermodynamics)
    assert restricted_source.phases == ["FCC_A1", "LIQUID"]


def test_default_phase_equilibrium_is_interface_order_independent():
    source_ab = create_pycalphad_thermodynamics_source(
        TDB_PATH,
        ELEMENTS,
        ("FCC_A1", "LIQUID"),
        use_default_phases=True,
    )
    source_bc = create_pycalphad_thermodynamics_source(
        TDB_PATH,
        ELEMENTS,
        ("LIQUID", "BCC_A2"),
        use_default_phases=True,
    )
    composition = np.array([0.425901015263025, 0.280023740661975], dtype=np.float64)
    temperature = 1650.0

    data_ab = source_ab.getEquilibriumData(composition, temperature)
    data_bc = source_bc.getEquilibriumData(composition, temperature)

    assert source_ab.equilibrium_phases == source_bc.equilibrium_phases
    assert set(data_ab["stable_phases"]) == {"FCC_A1", "LIQUID", "BCC_A2"}
    assert set(data_bc["stable_phases"]) == {"FCC_A1", "LIQUID", "BCC_A2"}
    for phase in ("FCC_A1", "LIQUID", "BCC_A2"):
        assert_allclose(data_ab["phase_compositions"][phase], data_bc["phase_compositions"][phase], atol=1.0e-11)
