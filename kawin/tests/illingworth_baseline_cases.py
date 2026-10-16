"""Substitutional-only Illingworth cases used for the bitwise regression baseline.

The interstitial (u-fraction) support in the Illingworth solvers must leave
substitutional-only runs bit-identical. ``make_illingworth_substitutional_baseline.py``
records every history produced by :func:`run_all_cases` with the pre-change
code, and ``test_illingworth_substitutional_baseline.py`` re-runs the same
cases and compares them with ``np.array_equal(..., equal_nan=True)``. Keep these builders
self-contained so later edits to other test helpers cannot change the baseline.
"""

from pathlib import Path

import numpy as np

from kawin.diffusion import (
    MovingBoundaryIllingworthFD1DModel,
    MovingBoundaryIllingworthTernaryFD1DModel,
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
)
from kawin.diffusion.DiffusionParameters import TemperatureParameters
from kawin.diffusion.mesh import CartesianFD1D, MixedBoundary1D, ProfileBuilder, StepProfile1D
from kawin.solver import explicitEulerIterator

BASELINE_PATH = Path(__file__).with_name("data") / "illingworth_substitutional_baseline.npz"


class _BinaryThermodynamics:
    """Binary diffusivity stub; ``variable=True`` gives D = D0 * (1 + 0.5 x)."""

    def __init__(self, diffusivities, variable=False):
        self.diffusivities = diffusivities
        self.variable = bool(variable)

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        composition = np.asarray(x, dtype=np.float64)
        base = float(self.diffusivities[phase])
        if self.variable:
            return np.squeeze(base * (1.0 + 0.5 * composition))
        return np.squeeze(np.ones(np.atleast_1d(T).shape, dtype=np.float64) * base)


class _CoupledTernaryThermodynamics:
    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        if phase == "ALPHA":
            return np.asarray([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]], dtype=np.float64)
        return np.asarray([[7.0e-4, -1.0e-4], [2.0e-4, 1.1e-3]], dtype=np.float64)


class _SmoothBulkTernaryThermodynamics:
    """Composition-dependent ternary matrices that accept single or stacked queries."""

    def clearCache(self):
        pass

    def _matrix_for(self, composition, phase):
        x, y = np.asarray(composition, dtype=np.float64)
        if phase == "ALPHA":
            return np.asarray([[1.0e-3 + 1.0e-4 * x, 2.0e-4 + 2.0e-5 * y], [1.0e-4 + 1.0e-5 * x, 8.0e-4 + 8.0e-5 * y]], dtype=np.float64)
        return np.asarray([[7.0e-4 + 8.0e-5 * x, -1.0e-4 + 1.0e-5 * y], [2.0e-4 + 1.0e-5 * x, 1.1e-3 + 7.0e-5 * y]], dtype=np.float64)

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        if values.ndim == 2:
            return np.asarray([self._matrix_for(row, phase) for row in values], dtype=np.float64)
        return self._matrix_for(values, phase)


def _curved_interface_compositions(eta):
    eta = float(eta)
    bend = eta * (1.0 - eta)
    center = np.asarray([0.24 + 0.25 * eta, 0.15 - 0.015 * eta + 0.22 * bend], dtype=np.float64)
    center_derivative = np.asarray([0.25, -0.015 + 0.22 * (1.0 - 2.0 * eta)], dtype=np.float64)
    tangent = center_derivative / np.linalg.norm(center_derivative)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    tilt = np.deg2rad(55.0 - 30.0 * eta)
    tie_direction = np.sin(tilt) * tangent + np.cos(tilt) * normal
    left_half_width = 0.042 * (1.0 + 0.12 * (2.0 * eta - 1.0))
    right_half_width = 0.055 * (1.0 - 0.10 * (2.0 * eta - 1.0))
    return center - left_half_width * tie_direction, center + right_half_width * tie_direction


class _CurvedInterfaceEquilibrium:
    eta_bounds = (0.0, 1.0)

    def interface_compositions(self, eta):
        return _curved_interface_compositions(eta)


class _LinearPairEquilibrium:
    eta_bounds = (0.0, 1.0)

    def __init__(self, left_base, left_slope, right_base, right_slope):
        self.left_base = np.asarray(left_base, dtype=np.float64)
        self.left_slope = np.asarray(left_slope, dtype=np.float64)
        self.right_base = np.asarray(right_base, dtype=np.float64)
        self.right_slope = np.asarray(right_slope, dtype=np.float64)

    def interface_compositions(self, eta):
        eta = float(eta)
        return self.left_base + eta * self.left_slope, self.right_base + eta * self.right_slope


class _ThreePhaseStepProfile:
    def __init__(self, interfaces, values):
        self.interfaces = tuple(float(v) for v in interfaces)
        self.values = tuple(np.asarray(v, dtype=np.float64) for v in values)

    def __call__(self, z):
        x = np.asarray(z, dtype=np.float64).reshape((-1, 1))[:, 0]
        out = np.empty((len(x), 2), dtype=np.float64)
        out[x < self.interfaces[0]] = self.values[0]
        middle = (x >= self.interfaces[0]) & (x < self.interfaces[1])
        out[middle] = self.values[1]
        out[x >= self.interfaces[1]] = self.values[2]
        return out


class _ThreePhaseThermodynamics:
    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        values = np.asarray(composition, dtype=np.float64)
        scale = {"A": 1.0, "B": 0.7, "C": 1.3}[phase]
        matrix = scale * np.asarray([[1.0, 0.1], [0.05, 0.8]], dtype=np.float64)
        if values.ndim == 2:
            return np.broadcast_to(matrix, (values.shape[0], 2, 2)).copy()
        return matrix


def _binary_model(mode, variable):
    profile = ProfileBuilder([(StepProfile1D(1.0, 0.8, 0.4), "CR")])
    mesh = CartesianFD1D(["CR"], [0.0, 5.0], 501)
    mesh.setResponseProfile(profile)
    therm = _BinaryThermodynamics({"ALPHA": 1.0e-7, "BETA": 1.0e-5}, variable=variable)
    return MovingBoundaryIllingworthFD1DModel(
        mesh,
        ["NI", "CR"],
        ["ALPHA", "BETA"],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=1.0,
        interface_compositions=(0.6, 0.0),
        time_step=0.1,
        phase_a_nodes=100,
        phase_b_nodes=100,
        tolerance=1.0e-8,
        record=True,
        bulk_diffusivity_mode=mode,
    )


def _ternary_model(mode, thermodynamics, left_bulk, right_bulk):
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(
        ProfileBuilder([(StepProfile1D(0.45, left_bulk, right_bulk), ["X", "Y"])]),
        boundaryConditions=MixedBoundary1D(2),
    )
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=thermodynamics,
        temperature=1000.0,
        interfacePosition=0.45,
        interface_equilibrium=_CurvedInterfaceEquilibrium(),
        initial_eta_bracket=(0.0, 1.0),
        bulk_diffusivity_mode=mode,
        time_step=1.0e-4,
        tolerance=1.0e-11,
        max_iterations=50,
        record=True,
    )


def _three_phase_model(mode):
    eq_ab = _LinearPairEquilibrium([0.20, 0.10], [0.10, 0.0], [0.30, 0.15], [0.10, 0.0])
    eq_bc = _LinearPairEquilibrium([0.24, 0.15], [0.10, 0.0], [0.18, 0.20], [0.10, 0.0])
    interfaces = (0.35, 0.7)
    phase_values = (
        np.asarray([0.21, 0.09], dtype=np.float64),
        np.asarray([0.31, 0.16], dtype=np.float64),
        np.asarray([0.20, 0.21], dtype=np.float64),
    )
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 31)
    mesh.setResponseProfile(ProfileBuilder([(_ThreePhaseStepProfile(interfaces, phase_values), ["X", "Y"])]))
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["A", "B", "C"],
        thermodynamics=_ThreePhaseThermodynamics(),
        temperature=1000.0,
        interfacePositions=interfaces,
        interface_equilibria=(eq_ab, eq_bc),
        initial_eta_guess=(0.5, 0.5),
        bulk_diffusivity_mode=mode,
        phase_molar_volumes=(2.0, 3.0, 1.5),
        phase_nodes=(9, 9, 9),
        time_step=1e-4,
        record=True,
        tolerance=1e-10,
        residual_tolerance=1e-10,
        max_iterations=50,
    )


def _history(prefix, history, out):
    """Stores the recorded rows (``0..N``) of a kawin-style history object."""
    if history is None:
        return
    n = int(history.N) + 1
    out[f"{prefix}_time"] = np.array(history._time[:n], copy=True)
    out[f"{prefix}_y"] = np.array(history._y[:n], copy=True)


def _collect(name, model, extra_histories):
    out = {}
    _history(f"{name}__data", model.data, out)
    for attr in extra_histories:
        value = getattr(model, attr, None)
        if isinstance(value, list):
            for i, item in enumerate(value):
                _history(f"{name}__{attr}{i}", item, out)
        else:
            _history(f"{name}__{attr}", value, out)
    out[f"{name}__compositions"] = np.array(model.getCompositions(), copy=True)
    return out


def run_all_cases():
    """Runs every baseline case and returns a flat ``{key: array}`` dictionary."""
    results = {}
    binary_histories = ("interfaceData", "concData", "pData", "qData")
    for mode, variable in (
        ("phase_uniform", False),
        ("composition_dependent_lagged", True),
        ("composition_dependent_implicit", True),
    ):
        model = _binary_model(mode, variable)
        model.solve(0.5, iterator=explicitEulerIterator, minDtFrac=1e-14)
        results.update(_collect(f"binary_{mode}", model, binary_histories))

    ternary_histories = ("interfaceData", "etaData", "inventoryData", "pData", "qData")
    ternary_cases = (
        ("ternary_uniform_right", "phase_uniform", _CoupledTernaryThermodynamics(), [0.16, 0.06], [0.3261538461538461, 0.193]),
        ("ternary_uniform_left", "phase_uniform", _CoupledTernaryThermodynamics(), [0.16, 0.12], [0.56, 0.25]),
        ("ternary_lagged_right", "composition_dependent_lagged", _SmoothBulkTernaryThermodynamics(), [0.16, 0.06], [0.3261538461538461, 0.193]),
        ("ternary_implicit_right", "composition_dependent_implicit", _SmoothBulkTernaryThermodynamics(), [0.16, 0.06], [0.3261538461538461, 0.193]),
    )
    for name, mode, therm, left, right in ternary_cases:
        model = _ternary_model(mode, therm, np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
        model.solve(3.0e-3, minDtFrac=1.0e-10)
        results.update(_collect(name, model, ternary_histories))
        p, q = model.getPhysicalPhaseProfiles()
        results[f"{name}__phys_left"] = np.array(p[1], copy=True)
        results[f"{name}__phys_right"] = np.array(q[1], copy=True)

    three_phase_histories = (
        "interfaceData",
        "etaData",
        "inventoryData",
        "rightBoundaryData",
        "molarInventoryData",
        "totalMolesData",
        "dependentMoleClosureErrorData",
        "profileData",
    )
    for mode in ("phase_uniform", "composition_dependent_lagged"):
        model = _three_phase_model(mode)
        model.solve(2.0e-3, minDtFrac=1.0e-10)
        results.update(_collect(f"three_phase_{mode}", model, three_phase_histories))
        for i, (coords, comps) in enumerate(model.getPhysicalPhaseProfiles()):
            results[f"three_phase_{mode}__phys{i}_z"] = np.array(coords, copy=True)
            results[f"three_phase_{mode}__phys{i}_c"] = np.array(comps, copy=True)
    return results
