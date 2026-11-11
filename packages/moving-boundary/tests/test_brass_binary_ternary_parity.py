"""Binary/ternary parity on the quasi-binary Cu-Zn brass case (hand port of a deferred research test).

Ports ``kawin/tests/test_illingworth_ternary.py::test_ternary_illingworth_quasi_binary_cu_zn_dummy_matches_binary_brass_case``
from the frozen research checkout (7af24f4). The research test built its binary reference with
``build_fig6_illingworth_case_params`` and ``build_fig3_present_work_model`` from the research example
``examples/Illingworth2005/compare_illingworth2005_planar.py``, which is not ported. ``_brass_thin_case``
and ``_build_brass_binary_model`` reproduce exactly the inputs those builders produced for this test.

- The research test passed ``"tolerance": 1.0e-15`` in its config, but ``build_fig6_illingworth_case_params``
  ignores that key and always returns ``FIG6_BASE_PARAMS["tolerance"] = 1.0e-10``. The binary model
  therefore ran with ``tolerance=1.0e-10`` (micrometres). Since LEDGERS R08 the binary ``tolerance`` is
  relative to the domain length, so this port passes the same position bound as ``1.0e-10 / R_um``
  (about 1.77e-13); the general default (1e-12) is too loose for this parity.
- ``max_iterations=10000`` comes from ``FIG6_OLAYE_BRASS_CONFIG["fig6_max_iterations"]``.
- ``grid_type="constant"`` gives no explicit transformed grids.

The configuration and assertions are unchanged; the plotting workflow is not ported. The generated
``test_illingworth_ternary.py`` still lists the test as deferred (``tools/port_tests.py``), because that
file is regenerated from research and hash-checked against ``docs/test-port-provenance.json``.
"""

import numpy as np

from kawin.diffusion.DiffusionParameters import TemperatureParameters
from moving_boundary.mesh import CartesianFD1D, MixedBoundary1D, ProfileBuilder, StepProfile1D
from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel, MovingBoundaryIllingworthTernaryFD1DModel


class _ConstantBinaryThermodynamics:
    """Phase-constant scalar diffusivities (research ``ConstantBinaryThermodynamics``)."""

    def __init__(self, phases, diffusivities):
        self.phases = list(phases)
        self.diffusivities = dict(diffusivities)

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        values = np.atleast_1d(T).astype(np.float64)
        return np.squeeze(np.ones(values.shape, dtype=np.float64) * self.diffusivities[phase])


class _QuasiBinaryCuZnDummyEquilibrium:
    """Fixed Cu-Zn endpoints with eta-pinned dummy interface compositions."""

    eta_bounds = (0.0, 1.0)

    def __init__(self, zn_left, zn_right, dummy):
        self.zn_left = float(zn_left)
        self.zn_right = float(zn_right)
        self.dummy = float(dummy)
        assert dummy < (1 - zn_left)
        assert dummy < (1 - zn_right)

    def interface_compositions(self, eta):
        return (
            np.asarray([self.zn_left, self.dummy * eta], dtype=np.float64),
            np.asarray([self.zn_right, self.dummy * eta], dtype=np.float64),
        )


class _QuasiBinaryTernaryThermodynamics:
    """Diagonal ternary diffusivity closure for quasi-binary diagnostics."""

    def __init__(self, diffusivities, dummy_diffusivity):
        self.diffusivities = {phase: float(value) for phase, value in diffusivities.items()}
        self.dummy_diffusivity = float(dummy_diffusivity)

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        return np.asarray(
            [
                [self.diffusivities[phase], 0.0],
                [0.0, self.dummy_diffusivity],
            ],
            dtype=np.float64,
        )


def _brass_thin_case():
    """The ``build_fig6_illingworth_case_params("thin", 1.4e-8, config)`` values the research test used.

    Units: micrometres, seconds, atomic percent. Phase A is beta, phase B is alpha.
    """
    return {
        "R_um": 565.0,
        "s0_um": 190.5,
        "c_liquid0_atpct": 39.4,
        "c_solid0_atpct": 29.1,
        "c_liquid_int_atpct": 36.9,
        "c_solid_int_atpct": 32.5,
        "D_liquid_um2_s": 100.0,
        "D_solid_um2_s": float(1.4e-8) * 1e8,  # the research cm^2/s -> um^2/s conversion
        "n_alpha": 12,
        "n_beta": 16,
        "time_step_s": 0.5,
        "dt_mode": "semi_log",
        "semiLog_dt": 0.025,
        "semiLogT0": 1e-5,
        "t_end_s": 1e4,
        "record_pq_data": True,
        "preallocate_recordings": False,
        "phase_a_name": "BETA",
        "phase_b_name": "ALPHA",
        "max_iterations": 10000,
        "tolerance": 1.0e-10,
    }


def _build_brass_binary_model(params):
    """Reproduces ``build_fig3_present_work_model(params, record=True)`` for the brass case."""
    profile = ProfileBuilder(
        [(StepProfile1D(params["s0_um"], params["c_liquid0_atpct"] / 100.0, params["c_solid0_atpct"] / 100.0), "ZN")]
    )
    mesh = CartesianFD1D(["ZN"], [0.0, params["R_um"]], params["n_alpha"] + params["n_beta"] - 1)
    mesh.setResponseProfile(profile)
    thermodynamics = _ConstantBinaryThermodynamics(
        phases=[params["phase_a_name"], params["phase_b_name"]],
        diffusivities={params["phase_a_name"]: params["D_liquid_um2_s"], params["phase_b_name"]: params["D_solid_um2_s"]},
    )
    return MovingBoundaryIllingworthFD1DModel(
        mesh,
        ["CU", "ZN"],
        [params["phase_a_name"], params["phase_b_name"]],
        thermodynamics=thermodynamics,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=params["s0_um"],
        interface_compositions=(params["c_liquid_int_atpct"] / 100.0, params["c_solid_int_atpct"] / 100.0),
        time_step=params["time_step_s"],
        dt_mode=params["dt_mode"],
        semiLog_dt=params["semiLog_dt"],
        semiLogT0=params["semiLogT0"],
        phase_a_nodes=params["n_alpha"],
        phase_b_nodes=params["n_beta"],
        transformed_u_grid=None,
        transformed_v_grid=None,
        tolerance=params["tolerance"] / params["R_um"],  # research absolute um tolerance, as relative (R08)
        record=True,
        record_pq_data=params["record_pq_data"],
        preallocate_recordings=params["preallocate_recordings"],
        max_iterations=params["max_iterations"],
    )


def test_ternary_illingworth_quasi_binary_cu_zn_dummy_matches_binary_brass_case():
    dummy_x = 0.02
    params = _brass_thin_case()
    binary = _build_brass_binary_model(params)
    binary.solve(params["t_end_s"], minDtFrac=1.0e-14, verbose=True, vIt=10000)

    left_bulk = np.asarray([params["c_liquid0_atpct"] / 100.0, dummy_x], dtype=np.float64)
    right_bulk = np.asarray([params["c_solid0_atpct"] / 100.0, dummy_x], dtype=np.float64)
    profile = ProfileBuilder([(StepProfile1D(params["s0_um"], left_bulk, right_bulk), ["ZN", "DUMMY"])])
    mesh = CartesianFD1D(["ZN", "DUMMY"], [0.0, params["R_um"]], params["n_alpha"] + params["n_beta"] - 1)
    mesh.setResponseProfile(profile, boundaryConditions=MixedBoundary1D(2))
    equilibrium = _QuasiBinaryCuZnDummyEquilibrium(
        zn_left=params["c_liquid_int_atpct"] / 100.0,
        zn_right=params["c_solid_int_atpct"] / 100.0,
        dummy=dummy_x,
    )
    thermodynamics = _QuasiBinaryTernaryThermodynamics(
        diffusivities={
            params["phase_a_name"]: params["D_liquid_um2_s"],
            params["phase_b_name"]: params["D_solid_um2_s"],
        },
        dummy_diffusivity=0.5,
    )

    for phase in (params["phase_a_name"], params["phase_b_name"]):
        D = thermodynamics.getInterdiffusivity(left_bulk, 1000.0, phase=phase)
        assert D.shape == (2, 2)
        assert D[0, 1] == 0.0
        assert D[1, 0] == 0.0
        assert D[1, 1] > 0.0

    ternary = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["CU", "ZN", "DUMMY"],
        phases=[params["phase_a_name"], params["phase_b_name"]],
        thermodynamics=thermodynamics,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=params["s0_um"],
        interface_equilibrium=equilibrium,
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(0.0, 1.0),
        initial_eta_guess=0.5,
        time_step=params["time_step_s"],
        dt_mode=params["dt_mode"],
        semiLog_dt=params["semiLog_dt"],
        semiLogT0=params["semiLogT0"],
        phase_a_nodes=params["n_alpha"],
        phase_b_nodes=params["n_beta"],
        tolerance=5.0e-16,
        max_iterations=25,
        record=True,
    )
    ternary.solve(params["t_end_s"], minDtFrac=1.0e-14, verbose=True, vIt=10000)

    binary_n = binary.interfaceData.N + 1
    ternary_n = ternary.interfaceData.N + 1
    assert ternary_n == binary_n
    assert np.allclose(ternary.interfaceData._time[:ternary_n], binary.interfaceData._time[:binary_n], rtol=0.0, atol=0.0)
    # Both models solve in deviation form (issue R06). A 50-digit march of this case puts the
    # binary within 3.5e-11 um and the ternary within 4.1e-11 um of the exact discrete
    # solution; the measured gap between them is 4.9e-11 um (it was 1.7e-10 while only the
    # binary used the deviation form).
    assert np.allclose(ternary.interfaceData._y[:ternary_n], binary.interfaceData._y[:binary_n], rtol=0.0, atol=1.0e-10)

    binary_p, binary_q = binary.getTransformedState()
    ternary_p, ternary_q = ternary.getTransformedState()
    assert np.allclose(ternary_p[:, 0], binary_p, rtol=0.0, atol=1.0e-10)
    assert np.allclose(ternary_q[:, 0], binary_q, rtol=0.0, atol=1.0e-10)
    assert np.allclose(ternary.data.y()[:, 0], binary.data.y()[:, 0], rtol=0.0, atol=1.0e-10)

    assert np.allclose(ternary_p[:, 1], dummy_x, rtol=0.0, atol=1.0e-13)
    assert np.allclose(ternary_q[:, 1], dummy_x, rtol=0.0, atol=1.0e-13)
    assert np.allclose(ternary.data.y()[:, 1], dummy_x, rtol=0.0, atol=1.0e-13)
    assert np.isclose(ternary.getTotalInventory()[1], dummy_x * params["R_um"], rtol=0.0, atol=1.0e-10)
    assert np.allclose(ternary.checkConservation(1.0e-10), np.zeros(2), rtol=0.0, atol=1.0e-10)
