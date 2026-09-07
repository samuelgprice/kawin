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

    def set_global_minimization_max_grid_points(self, value):
        self.calls.append(("options_set_global_minimization_max_grid_points", value))
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
    config = ThermoCalcConfig()

    assert_allclose(validate_independent_composition([0.3, 0.2], config), [0.3, 0.2])
    assert_allclose(independent_to_full_composition([0.3, 0.2], config), [0.5, 0.3, 0.2])

    with pytest.raises(ThermoCalcInputError):
        validate_independent_composition([0.8, 0.3], config)
    with pytest.raises(ThermoCalcInputError):
        validate_independent_composition([0.1, -0.1], config)
    with pytest.raises(ThermoCalcInputError):
        validate_independent_composition([0.1, 0.2, 0.3], config)


def test_phase_selection_defaults_to_thermocalc_default_phases():
    config = ThermoCalcConfig()
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
    config = ThermoCalcConfig(use_default_phases=False)
    backend = _TCPythonBackend()
    backend._setup = FakeTCPythonSetup()
    backend._config = config

    system = backend._get_system(config.use_default_phases)

    assert system["used_without_default_phases"]
    assert system["selected_phases"] == config.phases


def test_global_minimization_grid_points_are_applied_to_calculations():
    config = ThermoCalcConfig(global_minimization_max_grid_points=2500)
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
        ThermoCalcConfig(global_minimization_max_grid_points=0)


def test_driving_force_conversion_and_default_phase():
    therm = TCPythonThermodynamics(backend=FakeThermoCalcBackend())

    dg, xp = therm.getDrivingForce([0.3, 0.2], 1000.0)

    assert_allclose(dg, 0.5 * GAS_CONSTANT * 1000.0)
    assert_allclose(xp, [0.15, 0.05])


def test_diffusion_shapes_and_element_ordering():
    therm = TCPythonThermodynamics(backend=FakeThermoCalcBackend())

    dnkj = therm.getInterdiffusivity([[0.3, 0.2], [0.35, 0.1]], 1373.0, phase="BCC_A2")
    tracer = therm.getTracerDiffusivity([0.3, 0.2], 1373.0, phase="BCC_A2")

    assert dnkj.shape == (2, 2, 2)
    assert tracer.shape == (3,)
    assert_allclose(dnkj[0], [[1.0e-14, 0.1e-14], [0.2e-14, 2.0e-14]])
    assert_allclose(tracer, [3.0e-14, 4.0e-14, 5.0e-14])


def test_planar_tie_line_metadata_and_gextra_rejection():
    therm = TCPythonThermodynamics(backend=FakeThermoCalcBackend())

    x_alpha, x_beta, metadata = therm.getInterfacialComposition([0.3, 0.2], 1373.0, returnMeta=True)

    assert_allclose(x_alpha, [0.3, 0.2])
    assert_allclose(x_beta, [0.15, 0.05])
    assert metadata["endpoint_phases"] == ("BCC_A2", "FCC_A1")
    with pytest.raises(NotImplementedError):
        therm.getInterfacialComposition([0.3, 0.2], 1373.0, gExtra=1.0)


def test_default_remove_cache_is_configurable_per_adapter():
    backend = FakeThermoCalcBackend()
    therm = TCPythonThermodynamics(backend=backend, default_remove_cache=False)

    first = therm.getEquilibriumData([0.3, 0.2], 1373.0)
    second = therm.getEquilibriumData([0.3, 0.2], 1373.0)
    refreshed = therm.getEquilibriumData([0.3, 0.2], 1373.0, removeCache=True)

    assert backend.equilibrium_calls == 2
    assert first is second
    assert refreshed is not second


def test_checkpoint_resume_and_manifest_roundtrip():
    backend = FakeThermoCalcBackend(fail_once_group="equilibrium")
    therm = TCPythonThermodynamics(backend=backend)
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
    grid = make_fecrni_demo_grid()

    assert grid.shape == (25, 2)
    assert np.all(grid[:, 0] >= 0.05)
    assert np.all(grid[:, 1] >= 0.001)


def test_build_moving_boundary_surrogate_from_adapter():
    therm = TCPythonThermodynamics(backend=FakeThermoCalcBackend())
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

    therm = TCPythonThermodynamics()
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
