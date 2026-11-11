"""Ported from research ``kawin/tests/test_diffusion.py`` at ``7af24f4`` (LF-normalized sha256
``0f23bd76ea83b3dc0ccb0aca1e589cf2f284d640c05eec87ef6469b6e27b91db``) by
``tools/port_tests.py``; do not edit by hand. Imports resolve to the package as
``tools/definitions/_bindings.py`` does; the full transform list is in
``docs/test-port-provenance.json``. Excluded tests: none. Selected names (with what they need): ConstantBinaryThermodynamics, VariableBinaryThermodynamics, _build_illingworth_default_model.
"""
# --- kawin/tests/test_diffusion.py:8-8 (aliases used)
import numpy as np


# --- kawin/tests/test_diffusion.py:12-12 (aliases used)
from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel
from moving_boundary.solvers.illingworth.shared import TemperatureParameters


# --- kawin/tests/test_diffusion.py:13-13 (aliases used)
from moving_boundary.mesh import CartesianFD1D


# --- kawin/tests/test_diffusion.py:14-14 (aliases used)
from moving_boundary.mesh import ProfileBuilder, StepProfile1D


# --- kawin/tests/test_diffusion.py:21-21 (aliases used)
from kawin.diffusion.DiffusionParameters import TemperatureParameters


# --- kawin/tests/test_diffusion.py:59-76
class ConstantBinaryThermodynamics:
    def __init__(self, phases, diffusivities, interface_compositions):
        self.phases = phases
        self.diffusivities = diffusivities
        self.interface_compositions = interface_compositions

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        values = np.atleast_1d(T).astype(np.float64)
        return np.squeeze(np.ones(values.shape, dtype=np.float64) * self.diffusivities[phase])

    def getInterfacialComposition(self, T, gExtra=0, precPhase=None):
        values = np.atleast_1d(T).astype(np.float64)
        left = np.ones(values.shape, dtype=np.float64) * self.interface_compositions[0]
        right = np.ones(values.shape, dtype=np.float64) * self.interface_compositions[1]
        return np.squeeze(left), np.squeeze(right)


# --- kawin/tests/test_diffusion.py:79-83
class VariableBinaryThermodynamics(ConstantBinaryThermodynamics):
    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        composition = np.asarray(x, dtype=np.float64)
        base = float(self.diffusivities[phase])
        return np.squeeze(base * (1.0 + 0.5 * composition))


# --- kawin/tests/test_diffusion.py:3239-3285
def _build_illingworth_default_model(therm_class=ConstantBinaryThermodynamics, **kwargs):
    params = {
        "s0": 1.0,
        "R": 5.0,
        "n_alpha": 100,
        "d_alpha": 1.0e-7,
        "initial_alpha": 0.8,
        "interface_alpha": 0.6,
        "n_beta": 100,
        "d_beta": 1.0e-5,
        "initial_beta": 0.4,
        "interface_beta": 0.0,
        "time_step": 0.1,
        "tolerance": 1.0e-8,
        "geometry": "planar",
        "record": True,
        "transformed_u_grid": None,
        "transformed_v_grid": None,
        "model_kwargs": {},
    }
    params.update(kwargs)
    profile = ProfileBuilder([(StepProfile1D(params["s0"], params["initial_alpha"], params["initial_beta"]), "CR")])
    mesh = CartesianFD1D(["CR"], [0.0, params["R"]], 501)
    mesh.setResponseProfile(profile)
    therm = therm_class(
        phases=["ALPHA", "BETA"],
        diffusivities={"ALPHA": params["d_alpha"], "BETA": params["d_beta"]},
        interface_compositions=(params["interface_alpha"], params["interface_beta"]),
    )
    return MovingBoundaryIllingworthFD1DModel(
        mesh,
        ["NI", "CR"],
        ["ALPHA", "BETA"],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=params["s0"],
        interface_compositions=(params["interface_alpha"], params["interface_beta"]),
        time_step=params["time_step"],
        geometry=params["geometry"],
        phase_a_nodes=params["n_alpha"],
        phase_b_nodes=params["n_beta"],
        tolerance=params["tolerance"],
        record=params["record"],
        transformed_u_grid=params["transformed_u_grid"],
        transformed_v_grid=params["transformed_v_grid"],
        **params["model_kwargs"],
    )
