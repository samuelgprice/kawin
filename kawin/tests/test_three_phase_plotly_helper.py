import numpy as np
import pytest

from kawin.diffusion import TernaryMovingBoundaryThermodynamicsSurrogate


ELEMENTS = ("Z", "X", "Y")


class _History:
    def __init__(self, time, y):
        self._time = np.asarray(time, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self.N = len(self._time) - 1


class _SyntheticThreePhaseModel:
    def __init__(self):
        self.allElements = ELEMENTS
        self.phases = ("BCC", "LIQUID", "BCC")
        self._R = 1.0
        self._grids = tuple(np.asarray([0.0, 0.5, 1.0], dtype=np.float64) for _ in range(3))
        self.interfaceData = _History([0.0, 1.0], [[0.30, 0.70], [0.40, 0.65]])
        self.profileData = object()
        self._profiles = {
            0.0: (
                np.asarray([[0.10, 0.20], [0.12, 0.22], [0.14, 0.24]], dtype=np.float64),
                np.asarray([[0.20, 0.30], [0.22, 0.32], [0.24, 0.34]], dtype=np.float64),
                np.asarray([[0.30, 0.20], [0.32, 0.22], [0.34, 0.24]], dtype=np.float64),
            ),
            1.0: (
                np.asarray([[0.11, 0.19], [0.13, 0.21], [0.15, 0.23]], dtype=np.float64),
                np.asarray([[0.21, 0.29], [0.23, 0.31], [0.25, 0.33]], dtype=np.float64),
                np.asarray([[0.31, 0.19], [0.33, 0.21], [0.35, 0.23]], dtype=np.float64),
            ),
        }

    def getInterfacePositions(self, time=None):
        index = 0 if float(time) == 0.0 else 1
        return self.interfaceData._y[index].copy()

    def getTransformedState(self, time=None):
        return tuple(profile.copy() for profile in self._profiles[float(time)])


def _matrices(count):
    return np.broadcast_to(np.eye(2, dtype=np.float64), (count, 2, 2)).copy()


def _surrogate(phases, left, right):
    eta = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    general = np.asarray([[0.10, 0.10], [0.20, 0.20], [0.30, 0.10]], dtype=np.float64)
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=ELEMENTS,
        phases=phases,
        tieline_phases=phases,
        temperature=1000.0,
        eta_samples=eta,
        tieline_compositions={phases[0]: left, phases[1]: right},
        diffusivity_compositions={
            "interface": {phases[0]: left, phases[1]: right},
            "general": {phases[0]: general, phases[1]: general},
        },
        diffusivities={
            "interface": {phases[0]: _matrices(len(eta)), phases[1]: _matrices(len(eta))},
            "general": {phases[0]: _matrices(len(general)), phases[1]: _matrices(len(general))},
        },
        diffusivity_interpolation="nearest",
        metadata={"source": "synthetic"},
    )


def _result():
    eta_left = np.asarray([[0.10, 0.20], [0.12, 0.22], [0.14, 0.24]], dtype=np.float64)
    eta_right = np.asarray([[0.20, 0.30], [0.22, 0.32], [0.24, 0.34]], dtype=np.float64)
    bc_right = np.asarray([[0.30, 0.20], [0.32, 0.22], [0.34, 0.24]], dtype=np.float64)
    return {
        "model": _SyntheticThreePhaseModel(),
        "surrogate_ab": _surrogate(("BCC", "LIQUID"), eta_left, eta_right),
        "surrogate_bc": _surrogate(("LIQUID", "BCC"), eta_right, bc_right),
    }


def test_three_phase_plotly_helper_builds_combined_slider_figure():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    fig = plot_three_phase_composition_profile(_result(), renderer=None, show_tielines=False)

    assert len(fig.layout.sliders) == 1
    assert len(fig.frames) == 2
    assert fig.data[0].type == "scatterternary"
    assert fig.data[3].type == "scatter"
    assert fig.layout.ternary.aaxis.title.text == ELEMENTS[2]
    assert fig.layout.ternary.baxis.title.text == ELEMENTS[0]
    assert fig.layout.ternary.caxis.title.text == ELEMENTS[1]


def test_three_phase_plotly_helper_uses_diagnostic_ternary_coordinates_and_distinct_phase_labels():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    fig = plot_three_phase_composition_profile(_result(), renderer=None, show_tielines=False)
    first_phase = _result()["model"].getTransformedState(0.0)[0]
    full = np.column_stack((1.0 - np.sum(first_phase, axis=1), first_phase))

    assert np.allclose(fig.data[0].a, full[:, 2])
    assert np.allclose(fig.data[0].b, full[:, 0])
    assert np.allclose(fig.data[0].c, full[:, 1])
    assert [fig.data[index].name for index in range(3)] == ["A: BCC", "B: LIQUID", "C: BCC"]


def test_three_phase_plotly_helper_tieline_overlay_can_be_enabled_or_disabled():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    with_tielines = plot_three_phase_composition_profile(
        _result(),
        renderer=None,
        show_tielines=True,
        tieline_eta_count=3,
        display_tieline_count=2,
    )
    without_tielines = plot_three_phase_composition_profile(_result(), renderer=None, show_tielines=False)

    tieline_traces = [trace for trace in with_tielines.data if "surrogate tie-lines" in trace.name]
    assert len(tieline_traces) == 4
    assert not [trace for trace in without_tielines.data if "surrogate tie-lines" in trace.name]


def test_three_phase_plotly_helper_frames_sync_interface_markers():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    fig = plot_three_phase_composition_profile(_result(), renderer=None, show_tielines=False)

    first_ab = fig.frames[0].data[-2].x
    second_ab = fig.frames[1].data[-2].x
    first_bc = fig.frames[0].data[-1].x
    second_bc = fig.frames[1].data[-1].x
    assert first_ab == (300000.0, 300000.0)
    assert second_ab == (400000.0, 400000.0)
    assert first_bc == (700000.0, 700000.0)
    assert second_bc == (650000.0, 650000.0)
