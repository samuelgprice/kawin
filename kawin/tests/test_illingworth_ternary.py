from pathlib import Path

import numpy as np
import pytest

from kawin.diffusion import MovingBoundaryIllingworthTernaryFD1DModel, TernaryMovingBoundaryThermodynamicsSurrogate
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D import solve_illingworth_block_tridiagonal


class _ConstantTernaryThermodynamics:
    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        if phase == "ALPHA":
            return np.asarray([[1.0e-15, 1.0e-16], [2.0e-16, 0.8e-15]], dtype=np.float64)
        return np.asarray([[0.7e-15, -0.5e-16], [0.1e-16, 1.2e-15]], dtype=np.float64)


class _TieLineSamplingThermodynamics:
    def __init__(self, endpoint_mode="valid"):
        self.elements = ["Z", "X", "Y"]
        self.phases = ["ALPHA", "BETA"]
        self.endpoint_mode = endpoint_mode

    def clearCache(self):
        pass

    def getInterfacialComposition(self, x, T, gExtra=0, precPhase=None, returnMeta=False):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        eta = float((x[0] - 0.20) / 0.20)
        left = np.asarray([0.20 + 0.10 * eta, 0.10], dtype=np.float64)
        right = np.asarray([0.30 + 0.10 * eta, 0.15], dtype=np.float64)
        endpoints = [
            {"phase": "ALPHA", "composition": left},
            {"phase": "BETA", "composition": right},
        ]
        if self.endpoint_mode == "missing":
            endpoints = [{"phase": "ALPHA", "composition": left}]
        elif self.endpoint_mode == "extra":
            endpoints.append({"phase": "GAMMA", "composition": np.asarray([0.25, 0.05])})
        elif self.endpoint_mode == "duplicate":
            endpoints = [
                {"phase": "ALPHA", "composition": left},
                {"phase": "ALPHA", "composition": left + 0.01},
            ]
        elif self.endpoint_mode == "unlabeled":
            endpoints = [
                {"composition": left},
                {"phase": "BETA", "composition": right},
            ]
        if not returnMeta:
            return left, right
        return left, right, {"endpoint_phases": tuple(e.get("phase") for e in endpoints), "endpoints": tuple(endpoints)}

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        composition = np.asarray(composition, dtype=np.float64).reshape(2)
        if phase == "ALPHA":
            base = 1.0 + composition[0]
            return np.asarray([[base, composition[1]], [0.25 * composition[1], base + 0.5]], dtype=np.float64)
        base = 2.0 + composition[0]
        return np.asarray([[base, -0.5 * composition[1]], [0.1 * composition[1], base + 0.25]], dtype=np.float64)


def _block_times_vector(block, vector):
    return np.asarray(
        [
            block[0, 0] * vector[0] + block[0, 1] * vector[1],
            block[1, 0] * vector[0] + block[1, 1] * vector[1],
        ],
        dtype=np.float64,
    )


def test_ternary_block_solve_preserves_component_coupling():
    diagonal_block = np.asarray([[2.0, 0.5], [0.25, 3.0]], dtype=np.float64)
    expected = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    lower = np.zeros((3, 2, 2), dtype=np.float64)
    diagonal = np.repeat(diagonal_block[np.newaxis, :, :], 3, axis=0)
    upper = np.zeros((3, 2, 2), dtype=np.float64)
    rhs = np.asarray([_block_times_vector(diagonal_block, row) for row in expected], dtype=np.float64)

    actual = solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)

    assert np.allclose(actual, expected)


def test_ternary_illingworth_fixed_interface_no_motion_step_conserves_inventory():
    left = np.asarray([0.20, 0.10], dtype=np.float64)
    right = np.asarray([0.30, 0.15], dtype=np.float64)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=_ConstantTernaryThermodynamics(),
        temperature=1000.0,
        interfacePosition=0.5,
        interface_compositions=(left, right),
        time_step=1.0,
        record=True,
    )

    model.solve(1.0)

    assert np.isclose(model.getInterfacePosition(), 0.5)
    assert np.isclose(model.getInterfaceEta(), 0.0)
    assert np.allclose(model.getTotalInventory(), np.asarray([0.25, 0.125]))
    assert np.allclose(model.checkConservation(1e-12), np.zeros(2))
    assert model.getCompositions().shape == (21, 3)


def _build_surrogate(**kwargs):
    params = {
        "thermodynamics": _TieLineSamplingThermodynamics(),
        "elements": ["Z", "X", "Y"],
        "phases": ["ALPHA", "BETA"],
        "tieline_phases": ("ALPHA", "BETA"),
        "temperature": 1000.0,
        "probe_start": np.asarray([0.20, 0.10], dtype=np.float64),
        "probe_end": np.asarray([0.40, 0.20], dtype=np.float64),
        "eta_samples": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    }
    params.update(kwargs)
    return TernaryMovingBoundaryThermodynamicsSurrogate.from_database(**params)


def test_ternary_surrogate_requires_explicit_tieline_phases():
    with pytest.raises(ValueError, match="tieline_phases"):
        _build_surrogate(tieline_phases=None)


@pytest.mark.parametrize("endpoint_mode", ["missing", "extra", "duplicate", "unlabeled"])
def test_ternary_surrogate_rejects_unexpected_tieline_phase_metadata(endpoint_mode):
    with pytest.raises(ValueError):
        _build_surrogate(thermodynamics=_TieLineSamplingThermodynamics(endpoint_mode=endpoint_mode))


def test_ternary_surrogate_interpolates_tielines_and_returns_metadata():
    surrogate = _build_surrogate()

    left, right = surrogate.interface_compositions(0.25)
    left_meta, right_meta, meta = surrogate.getInterfacialComposition(0.25, T=1000.0, returnMeta=True)

    assert np.allclose(left, np.asarray([0.225, 0.10]))
    assert np.allclose(right, np.asarray([0.325, 0.15]))
    assert np.allclose(left_meta, left)
    assert np.allclose(right_meta, right)
    assert meta["endpoint_phases"] == ("ALPHA", "BETA")
    assert meta["endpoints"][0]["phase"] == "ALPHA"
    assert meta["endpoints"][1]["phase"] == "BETA"


def test_ternary_surrogate_returns_nearest_interface_and_general_diffusivities():
    bulk_point = np.asarray([[0.45, 0.05]], dtype=np.float64)
    surrogate = _build_surrogate(diffusivity_bulk_points=bulk_point)

    interface_matrix = surrogate.getInterdiffusivity([0.249, 0.10], 1000.0, phase="ALPHA", query_context="interface")
    expected_interface = _TieLineSamplingThermodynamics().getInterdiffusivity([0.25, 0.10], 1000.0, phase="ALPHA")
    general_matrix = surrogate.getInterdiffusivity([0.451, 0.05], 1000.0, phase="ALPHA", query_context="general")
    expected_general = _TieLineSamplingThermodynamics().getInterdiffusivity([0.45, 0.05], 1000.0, phase="ALPHA")

    assert np.allclose(interface_matrix, expected_interface)
    assert np.allclose(general_matrix, expected_general)


def test_ternary_surrogate_rejects_nonmatching_temperature():
    surrogate = _build_surrogate()

    with pytest.raises(ValueError, match="isothermal"):
        surrogate.getInterdiffusivity([0.25, 0.10], 1100.0, phase="ALPHA")


def test_ternary_surrogate_save_load_preserves_phase_order_and_predictions():
    surrogate = _build_surrogate(diffusivity_bulk_points=np.asarray([[0.45, 0.05]], dtype=np.float64))
    path = Path.cwd() / "ternary_surrogate_test_roundtrip.npz"
    try:
        surrogate.save(path)
        loaded = TernaryMovingBoundaryThermodynamicsSurrogate.load(path)
    finally:
        path.unlink(missing_ok=True)

    assert loaded.tieline_phases == ("ALPHA", "BETA")
    assert np.allclose(loaded.interface_compositions(0.25)[0], surrogate.interface_compositions(0.25)[0])
    assert np.allclose(
        loaded.getInterdiffusivity([0.451, 0.05], 1000.0, phase="ALPHA"),
        surrogate.getInterdiffusivity([0.451, 0.05], 1000.0, phase="ALPHA"),
    )


def test_ternary_illingworth_accepts_surrogate_for_thermo_and_equilibrium():
    surrogate = _build_surrogate()
    left, right = surrogate.interface_compositions(0.0)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=surrogate,
        temperature=1000.0,
        interfacePosition=0.5,
        interface_equilibrium=surrogate,
        initial_eta=0.0,
        time_step=1.0,
        record=True,
    )

    model.solve(1.0)

    assert np.isclose(model.getInterfacePosition(), 0.5)
    assert np.isclose(model.getInterfaceEta(), 0.0)
    assert np.allclose(model.checkConservation(1e-12), np.zeros(2))
