import numpy as np

from kawin.diffusion.mesh import CartesianFD1D
from kawin.diffusion.mesh.MeshBase import MeshData


def test_mesh_data_interpolates_only_recorded_rows_with_unfilled_capacity():
    mesh = CartesianFD1D(["A"], [0.0, 1.0], 3)
    history = MeshData(mesh, record=True)
    first = np.asarray([[0.0], [1.0], [2.0]])
    second = np.asarray([[2.0], [3.0], [4.0]])

    history.record(0.0, first)
    history.record(2.0, second)

    assert history.N == 1
    assert history._time.shape[0] == history.batchSize
    assert np.array_equal(history.y(0.0), first)
    assert np.array_equal(history.y(2.0), second)
    assert np.array_equal(history.y(1.0), 0.5 * (first + second))


def test_mesh_data_default_time_lookup_clamps_outside_recorded_range():
    mesh = CartesianFD1D(["A"], [0.0, 1.0], 3)
    history = MeshData(mesh, record=True)
    first = np.asarray([[0.0], [1.0], [2.0]])
    second = np.asarray([[2.0], [3.0], [4.0]])

    history.record(3.0, first)
    history.record(5.0, second)

    assert np.array_equal(history.y(2.0), first)
    assert np.array_equal(history.y(6.0), second)


def test_mesh_data_disabled_interpolation_still_allows_exact_timestamp_lookup():
    mesh = CartesianFD1D(["A"], [0.0, 1.0], 3)
    history = MeshData(mesh, record=True)
    first = np.asarray([[0.0], [1.0], [2.0]])
    second = np.asarray([[2.0], [3.0], [4.0]])
    history.record(0.0, first)
    history.record(2.0, second)
    history._timeInterpolationEnabled = False
    history._timeInterpolationError = "exact timestamps only"

    assert np.array_equal(history.y(2.0), second)
    try:
        history.y(1.0)
    except ValueError as exc:
        assert "exact timestamps only" in str(exc)
    else:
        raise AssertionError("disabled MeshData interpolation accepted a non-recorded time")
