"""R11: binary and two-phase ternary output dictionaries are not restart files.

``toDict``/``save`` write recorded histories only. The host loader restores the
physical snapshots and time but not the interface, eta, transformed profiles,
previous-step state or reference inventory, so a "restarted" model reported the
loaded time with its old interface. ``fromDict`` (and the inherited ``load``,
which calls it) must refuse before changing any state; output serialization
stays as it was. Three-phase has the same contract in
``test_illingworth_ternary_three_phase.py``.

Defect tests fail before the R11 fix (the loader accepted the dictionary).
Preservation tests pass before and after it.
"""

import dataclasses

import numpy as np
import pytest

from kawin.diffusion.DiffusionParameters import TemperatureParameters
from moving_boundary.mesh import CartesianFD1D, MixedBoundary1D, ProfileBuilder, StepProfile1D
from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel, MovingBoundaryIllingworthTernaryFD1DModel


class _BinaryConstantDiffusivity:
    def clearCache(self):
        pass

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        return np.squeeze(np.ones(np.atleast_1d(T).shape) * {"ALPHA": 1.0e-3, "BETA": 1.0e-1}[phase])


class _CoupledTernaryThermodynamics:
    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        if phase == "ALPHA":
            return np.asarray([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]], dtype=np.float64)
        return np.asarray([[7.0e-4, -1.0e-4], [2.0e-4, 1.1e-3]], dtype=np.float64)


class _CurvedInterfaceEquilibrium:
    eta_bounds = (0.0, 1.0)

    def interface_compositions(self, eta):
        eta = float(eta)
        bend = eta * (1.0 - eta)
        center = np.asarray([0.24 + 0.25 * eta, 0.15 - 0.015 * eta + 0.22 * bend], dtype=np.float64)
        derivative = np.asarray([0.25, -0.015 + 0.22 * (1.0 - 2.0 * eta)], dtype=np.float64)
        tangent = derivative / np.linalg.norm(derivative)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        tilt = np.deg2rad(55.0 - 30.0 * eta)
        direction = np.sin(tilt) * tangent + np.cos(tilt) * normal
        left = center - 0.042 * (1.0 + 0.12 * (2.0 * eta - 1.0)) * direction
        right = center + 0.055 * (1.0 - 0.10 * (2.0 * eta - 1.0)) * direction
        return left, right


def _binary_model():
    mesh = CartesianFD1D(["CR"], [0.0, 5.0], 501)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(1.0, 0.8, 0.4), "CR")]))
    return MovingBoundaryIllingworthFD1DModel(
        mesh, ["NI", "CR"], ["ALPHA", "BETA"], thermodynamics=_BinaryConstantDiffusivity(),
        temperature=TemperatureParameters(1000.0), interfacePosition=1.0, interface_compositions=(0.6, 0.0),
        time_step=0.1, phase_a_nodes=100, phase_b_nodes=100, record=True, record_pq_data=True,
    )


def _ternary_model():
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(
        ProfileBuilder([(StepProfile1D(0.45, np.asarray([0.16, 0.06]), np.asarray([0.3261538461538461, 0.193])), ["X", "Y"])]),
        boundaryConditions=MixedBoundary1D(2),
    )
    equilibrium = _CurvedInterfaceEquilibrium()
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh, elements=["Z", "X", "Y"], phases=["ALPHA", "BETA"],
        thermodynamics=_CoupledTernaryThermodynamics(), temperature=1000.0, interfacePosition=0.45,
        interface_equilibrium=equilibrium, initial_eta_method="instantaneous_balance",
        initial_eta_bracket=equilibrium.eta_bounds, time_step=1.0e-2, tolerance=1.0e-11,
        max_iterations=50, record=True, record_pq_data=True,
    )


# (class, builder, one scheduled step)
_FAMILIES = {
    "binary": (MovingBoundaryIllingworthFD1DModel, _binary_model, 0.1),
    "ternary": (MovingBoundaryIllingworthTernaryFD1DModel, _ternary_model, 1.0e-2),
}
_BINARY_KEYS = {"x", "time", "interval", "index"}
_TERNARY_KEYS = _BINARY_KEYS | {"interface_position", "interface_eta", "inventory", "interface_interval", "interface_index"}
_STATE = ("currentTime", "isSetup", "_s_curr", "_s_old", "_p_curr", "_q_curr", "_eta_curr", "_eta_old",
          "_initialInventory", "_R", "_z", "_currdt", "_pendingCandidate")
_HISTORIES = ("data", "interfaceData", "concData", "etaData", "inventoryData", "pData", "qData")


def _freeze(value):
    """Returns a comparable, copy-independent image of ``value`` (arrays by dtype, shape and bytes)."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return ("object-array", value.shape, tuple(_freeze(v) for v in value.reshape(-1)))
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(k), _freeze(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_freeze(v) for v in value))
    if isinstance(value, (float, np.floating)):
        return ("float", np.float64(value).tobytes())
    if isinstance(value, (bool, int, str, np.integer, np.bool_)) or value is None:
        return value
    if dataclasses.is_dataclass(value):
        return (type(value).__name__, _freeze({f.name: getattr(value, f.name) for f in dataclasses.fields(value)}))
    return ("object", type(value).__name__, id(value))


def _state(model):
    """Snapshots the accepted state, time and every recorded history of ``model``."""
    image = {name: _freeze(getattr(model, name, "<absent>")) for name in _STATE}
    for name in _HISTORIES:
        history = getattr(model, name, None)
        if history is None:
            image[name] = None
            continue
        fields = {
            key: value for key, value in vars(history).items()
            if isinstance(value, (np.ndarray, list, tuple, float, int, bool, np.number)) or value is None
        }
        image[name] = _freeze(fields)
    return image


def _solved(family, steps=1):
    _, build, step = _FAMILIES[family]
    model = build()
    model.solve(steps * step, minDtFrac=1e-14)
    return model


def _target(family, kind, source):
    if kind == "source":
        return source
    model = _FAMILIES[family][1]()
    if kind == "set_up":
        model.setup()
    return model


# Defect tests (fail before the R11 fix)


@pytest.mark.parametrize("kind", ["fresh", "set_up", "source"])
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_from_dict_is_rejected_without_mutation(family, kind):
    """Instance ``fromDict`` raises before touching the model or the dictionary."""
    source = _solved(family)
    data = source.toDict()
    data_before = _freeze(data)
    target = _target(family, kind, source)
    before = _state(target)

    with pytest.raises(NotImplementedError, match="cannot be restarted"):
        target.fromDict(data)

    assert _state(target) == before
    assert _freeze(data) == data_before


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_from_dict_called_on_the_class_is_rejected(family):
    """The class-level call raises the same explicit error, like three-phase."""
    model_class = _FAMILIES[family][0]
    data = _solved(family).toDict()

    with pytest.raises(NotImplementedError, match="cannot be restarted"):
        model_class.fromDict(data)


@pytest.mark.parametrize("kind", ["fresh", "set_up"])
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_inherited_load_through_an_instance_is_rejected_without_mutation(family, kind, tmp_path):
    """The host ``load`` reaches ``fromDict`` and raises; the model is unchanged."""
    source = _solved(family)
    path = tmp_path / "output.npz"
    source.save(path)
    target = _target(family, kind, source)
    before = _state(target)

    with pytest.raises(NotImplementedError, match="cannot be restarted"):
        target.load(path)

    assert _state(target) == before


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_rejected_restart_leaves_continued_solve_unchanged(family):
    """After a refused load, continuing the solve matches a model that never saw it, bit for bit."""
    step = _FAMILIES[family][2]
    later = _solved(family, steps=2).toDict()
    attempted = _solved(family)
    control = _solved(family)

    with pytest.raises(NotImplementedError, match="cannot be restarted"):
        attempted.fromDict(later)
    attempted.solve(step, minDtFrac=1e-14)
    control.solve(step, minDtFrac=1e-14)

    assert _state(attempted) == _state(control)


# Preservation tests (pass before and after the R11 fix)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_output_dictionary_contents_are_unchanged(family):
    """``toDict`` keeps its keys and returns the recorded histories themselves."""
    model = _solved(family)
    data = model.toDict()

    expected = set(_BINARY_KEYS if family == "binary" else _TERNARY_KEYS)
    if family == "ternary" and model.pData is not None:
        expected |= {"p", "q"}
    assert set(data) == expected
    assert data["x"] is model.data._y
    assert data["time"] is model.data._time
    assert data["index"] == model.data.N
    if family == "ternary":
        assert data["interface_position"] is model.interfaceData._y
        assert data["interface_eta"] is model.etaData._y
        assert data["inventory"] is model.inventoryData._y
        assert data["interface_index"] == model.interfaceData.N
        assert data["p"] is model.pData._y
        assert data["q"] is model.qData._y


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_save_writes_the_output_dictionary(family, tmp_path):
    """``save`` still writes every ``toDict`` array, readable with ``numpy.load``."""
    model = _solved(family)
    path = tmp_path / "output.npz"
    model.save(path)
    data = model.toDict()

    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(data)
        for key, value in data.items():
            assert np.array_equal(saved[key], np.asarray(value)), key
