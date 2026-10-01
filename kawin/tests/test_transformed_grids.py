import numpy as np
import pytest

from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.TransformedGrids import (
    describe_landau_grid,
    landau_grid,
    linear_spacing_grid,
    power_law_grid,
    summarize_landau_grid,
    tanh_grid,
    three_phase_landau_grids,
    two_phase_landau_grids,
    uniform_grid,
    validate_landau_grid,
)
from kawin.diffusion.MovingBoundaryIllingworthTernaryFDM import estimate_initial_eta_from_instantaneous_balance
from kawin.tests.test_illingworth_ternary import _IdentityTernaryThermodynamics, _LinearInterfaceEquilibrium


def _assert_valid_grid(grid, n_nodes):
    assert grid.dtype == np.float64
    assert grid.shape == (n_nodes,)
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert np.all(np.diff(grid) > 0.0)


@pytest.mark.parametrize("n_nodes", [3, 4, 17])
@pytest.mark.parametrize(
    "method, phase, cluster, kwargs",
    [
        ("uniform", "left", "interface", {}),
        ("uniform", "middle", "far", {}),
        ("power", "left", "interface", {"exponent": 2.0}),
        ("power", "right", "interface", {"exponent": 1.5}),
        ("power", "left", "far", {"exponent": 0.7}),
        ("power", "right", "far", {"exponent": 3.0}),
        ("tanh", "left", "interface", {"beta": 2.0}),
        ("tanh", "right", "far", {"beta": 2.0}),
        ("tanh", "middle", "interface", {"beta": 1.5}),
        ("linear", "left", "interface", {"spacing_ratio": 0.2}),
        ("linear", "right", "far", {"spacing_ratio": 0.5}),
        ("linear", "right", "interface", {"spacing_ratio": 1.5}),
        ("linear", "middle", "interface", {"spacing_ratio": 0.3}),
    ],
)
def test_landau_grid_basic_properties(n_nodes, method, phase, cluster, kwargs):
    if method == "linear" and phase == "middle" and n_nodes == 3:
        with pytest.raises(ValueError, match="at least 4 nodes"):
            landau_grid(n_nodes, method, phase=phase, cluster=cluster, **kwargs)
        return
    grid = landau_grid(n_nodes, method, phase=phase, cluster=cluster, **kwargs)
    _assert_valid_grid(grid, n_nodes)


@pytest.mark.parametrize("cluster_at", [0.0, 1.0])
def test_uniform_limits_match_linspace(cluster_at):
    expected = np.linspace(0.0, 1.0, 9)
    assert np.allclose(power_law_grid(9, 1.0, cluster_at), expected, rtol=0.0, atol=1e-15)
    assert np.array_equal(tanh_grid(9, 0.0, cluster_at), expected)
    assert np.array_equal(tanh_grid(9, 0.0, "both"), expected)
    assert np.array_equal(uniform_grid(9), expected)


def test_linear_spacing_grid_matches_specification():
    grid = linear_spacing_grid(101, 0.2)
    d = np.diff(grid)
    assert len(d) == 100
    assert d[0] == pytest.approx(0.2 / 100, rel=1e-12)
    assert d[-1] == pytest.approx(1.8 / 100, rel=1e-12)
    assert np.allclose(np.diff(d), np.diff(d)[0], rtol=1e-9, atol=0.0)
    assert np.sum(d) == pytest.approx(1.0, rel=1e-14)
    assert np.allclose(linear_spacing_grid(11, 0.4, 1.0), 1.0 - linear_spacing_grid(11, 0.4, 0.0)[::-1], rtol=0.0, atol=1e-14)
    assert np.allclose(linear_spacing_grid(11, 1.0), np.linspace(0.0, 1.0, 11), rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("n_nodes", [10, 11])
def test_linear_spacing_grid_both_ends(n_nodes):
    grid = linear_spacing_grid(n_nodes, 0.25, "both")
    d = np.diff(grid)
    h = 1.0 / (n_nodes - 1)
    assert d[0] == pytest.approx(0.25 * h, rel=1e-12)
    assert d[-1] == pytest.approx(0.25 * h, rel=1e-12)
    assert np.allclose(d, d[::-1], rtol=0.0, atol=1e-14)
    assert np.max(d) > h


@pytest.mark.parametrize("method, kwargs", [("power", {"exponent": 2.0}), ("tanh", {"beta": 2.5}), ("linear", {"spacing_ratio": 0.3})])
def test_clustering_direction_follows_interface(method, kwargs):
    u, v = two_phase_landau_grids(12, 15, method=method, cluster="interface", **kwargs)
    assert np.diff(u)[-1] < np.diff(u)[0]
    assert np.diff(v)[0] < np.diff(v)[-1]

    u_far, v_far = two_phase_landau_grids(12, 15, method=method, cluster="far", **kwargs)
    assert np.diff(u_far)[0] < np.diff(u_far)[-1]
    assert np.diff(v_far)[-1] < np.diff(v_far)[0]


def test_middle_phase_clusters_at_both_ends():
    grid = landau_grid(21, "tanh", phase="middle", beta=2.0)
    d = np.diff(grid)
    center = d[len(d) // 2]
    assert d[0] < center and d[-1] < center
    assert np.allclose(d, d[::-1], rtol=0.0, atol=1e-14)


def test_mirror_symmetry():
    for gen, param in ((power_law_grid, 2.3), (tanh_grid, 1.7)):
        at_zero = gen(13, param, 0.0)
        at_one = gen(13, param, 1.0)
        assert np.allclose(at_one, 1.0 - at_zero[::-1], rtol=0.0, atol=1e-14)


def test_power_law_reproduces_existing_clustered_v_grid():
    x = np.linspace(0.0, 1.0, 10)
    assert np.allclose(power_law_grid(10, 2.0), x * x, rtol=0.0, atol=1e-15)


def test_per_phase_overrides():
    u, v = two_phase_landau_grids(8, 10, method="tanh", beta=2.0, right_kwargs={"method": "uniform"})
    assert np.allclose(v, np.linspace(0.0, 1.0, 10), rtol=0.0, atol=1e-15)
    assert np.allclose(u, landau_grid(8, "tanh", phase="left", beta=2.0))


def test_three_phase_grids():
    a, b, c = three_phase_landau_grids(7, 11, 9, method="tanh", beta=1.5)
    _assert_valid_grid(a, 7)
    _assert_valid_grid(b, 11)
    _assert_valid_grid(c, 9)
    assert np.diff(a)[-1] < np.diff(a)[0]
    assert np.diff(c)[0] < np.diff(c)[-1]


def test_describe_landau_grid():
    u = landau_grid(11, "power", phase="left", exponent=2.0)
    info = describe_landau_grid(u, phase="left")
    assert info["n_nodes"] == 11
    assert info["interface_interval"] == pytest.approx(np.diff(u)[-1])
    assert info["far_interval"] == pytest.approx(np.diff(u)[0])
    assert info["max_adjacent_ratio"] >= 1.0
    assert describe_landau_grid(np.linspace(0, 1, 5))["max_adjacent_ratio"] == pytest.approx(1.0)


def test_summarize_landau_grid_prints_spacing(capsys):
    v = landau_grid(9, "power", phase="right", exponent=2.0)
    summary, fig = summarize_landau_grid(v, phase="right", physical_length=2.0)
    out = capsys.readouterr().out
    d = np.diff(v)
    assert fig is None
    assert "min spacing" in out and "max spacing" in out and "physical min/max" in out
    assert summary["min_interval_index"] == 0
    assert summary["max_interval_index"] == len(d) - 1
    assert summary["min_interval"] == pytest.approx(d.min())
    assert summary["max_interval"] == pytest.approx(d.max())
    assert summary["uniform_interval"] == pytest.approx(1.0 / 8)
    assert "uniform reference" in out


@pytest.mark.parametrize("phase", [None, "left", "middle"])
def test_summarize_landau_grid_plot(phase):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = landau_grid(11, "tanh", phase=phase or "left", beta=2.0)
    _, fig = summarize_landau_grid(grid, phase=phase, plot=True)
    try:
        assert len(fig.axes) == 3
        assert fig.axes[2].get_xscale() == "linear"
        assert sum(patch.get_height() for patch in fig.axes[1].patches) == len(grid) - 1
        h = 1.0 / (len(grid) - 1)
        for ax in fig.axes[1:]:
            reference_lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
            assert len(reference_lines) == 1
            assert np.allclose(reference_lines[0].get_xdata(), h)
    finally:
        plt.close(fig)
    _, fig = summarize_landau_grid(grid, phase=phase, plot=True, cdf_log=True)
    try:
        assert fig.axes[2].get_xscale() == "log"
    finally:
        plt.close(fig)


def test_summarize_landau_grid_plot_uniform_grid():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = uniform_grid(100)
    _, fig = summarize_landau_grid(grid, phase="left", plot=True)
    try:
        ax_hist = fig.axes[1]
        heights = [patch.get_height() for patch in ax_hist.patches]
        assert sum(heights) == 99
        assert max(heights) == 99
        low, high = ax_hist.get_xlim()
        assert low < 1.0 / 99 < high
        assert (high - low) > 0.1 / 99
        fig.canvas.draw()
        assert ax_hist.xaxis.get_major_formatter().get_offset() == ""
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "call",
    [
        lambda: uniform_grid(2),
        lambda: uniform_grid(3.5),
        lambda: power_law_grid(5, 0.0),
        lambda: power_law_grid(5, -1.0),
        lambda: power_law_grid(5, 2.0, "both"),
        lambda: tanh_grid(5, -0.1),
        lambda: tanh_grid(5, 1.0, 0.5),
        lambda: linear_spacing_grid(5, 0.0),
        lambda: linear_spacing_grid(5, 2.0),
        lambda: linear_spacing_grid(5, 2.5, 1.0),
        lambda: linear_spacing_grid(11, 3.0, "both"),
        lambda: linear_spacing_grid(5, 0.5, 0.5),
        lambda: landau_grid(5, "geometric"),
        lambda: landau_grid(5, "power", phase="middle", exponent=2.0),
        lambda: landau_grid(5, "tanh", phase="middle", cluster="far", beta=1.0),
        lambda: landau_grid(5, "tanh", phase="top", beta=1.0),
        lambda: landau_grid(40, "power", phase="right", exponent=12.0),
        lambda: validate_landau_grid([0.0, 0.5, 0.5, 1.0]),
        lambda: validate_landau_grid([0.1, 0.5, 1.0]),
        lambda: validate_landau_grid([0.0, np.nan, 1.0]),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


def test_generated_grids_accepted_by_ternary_solver_estimator():
    eta_true = 0.25
    left, right = _LinearInterfaceEquilibrium().interface_compositions(eta_true)
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], 21)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5, left, right), ["X", "Y"])]))
    u_grid, v_grid = two_phase_landau_grids(7, 9, method="tanh", beta=2.0)

    estimate = estimate_initial_eta_from_instantaneous_balance(
        composition=mesh.y,
        z=mesh.z,
        interface_position=0.5,
        phases=["ALPHA", "BETA"],
        thermodynamics=_IdentityTernaryThermodynamics(),
        temperature=1000.0,
        interface_equilibrium=_LinearInterfaceEquilibrium(),
        transformed_u_grid=u_grid,
        transformed_v_grid=v_grid,
        eta_bracket=(0.0, 1.0),
    )

    assert estimate.converged
    assert np.isclose(estimate.eta, eta_true, atol=1e-10)
