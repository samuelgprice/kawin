import numpy as np
import pytest

from kawin.diffusion import TernaryMovingBoundaryThermodynamicsSurrogate


ELEMENTS = ("Z", "X", "Y")


class _RecordingDiffusivityProvider:
    def __init__(self, *, scalar_only=False, negative_components=()):
        self.scalar_only = scalar_only
        self.negative_components = {tuple(component) for component in negative_components}
        self.calls = []

    def getInterdiffusivity(self, composition, temperature, phase=None, query_context=None):
        values = np.asarray(composition, dtype=np.float64)
        self.calls.append(
            {
                "phase": phase,
                "composition": values.copy(),
                "temperature": np.asarray(temperature, dtype=np.float64).copy(),
                "query_context": query_context,
            }
        )
        if self.scalar_only and values.ndim == 2 and len(values) > 1:
            raise TypeError("scalar-only diffusivity provider")
        points = values if values.ndim == 2 else values.reshape(1, 2)
        offset = {"BCC": 1.0, "LIQUID": 10.0}[phase]
        matrices = np.empty((len(points), 2, 2), dtype=np.float64)
        matrices[:, 0, 0] = offset + points[:, 0]
        matrices[:, 0, 1] = offset + 2.0 * points[:, 1]
        matrices[:, 1, 0] = offset + 3.0 * points[:, 0]
        matrices[:, 1, 1] = offset + 4.0 * points[:, 1]
        for row, column in self.negative_components:
            matrices[:, row, column] *= -1.0
        return matrices[0] if values.ndim != 2 else matrices


class _History:
    def __init__(self, time, y):
        self._time = np.asarray(time, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self.N = len(self._time) - 1


class _SyntheticThreePhaseModel:
    def __init__(self, *, constant_profiles=False):
        self.allElements = ELEMENTS
        self.phases = ("BCC", "LIQUID", "BCC")
        self.bulkDiffusivityMode = "phase_uniform"
        self.dtMode = "semi_log"
        self.semiLog_dt = 0.05
        self.tolerance = 1.0e-10
        self.temperatureParameters = lambda positions, time: np.full(len(np.asarray(positions)), 1000.0)
        self.therm = None
        self._R = 1.0
        self._grids = tuple(np.asarray([0.0, 0.5, 1.0], dtype=np.float64) for _ in range(3))
        self.interfaceData = _History([0.0, 1.0], [[0.30, 0.70], [0.40, 0.65]])
        self.etaData = _History([0.0, 1.0], [[0.25, 0.75], [0.50, 0.30]])
        self.profileData = object()
        if constant_profiles:
            self._profiles = {
                0.0: (
                    np.full((3, 2), [0.10, 0.20], dtype=np.float64),
                    np.full((3, 2), [0.20, 0.30], dtype=np.float64),
                    np.full((3, 2), [0.30, 0.20], dtype=np.float64),
                ),
                1.0: (
                    np.full((3, 2), [0.11, 0.19], dtype=np.float64),
                    np.full((3, 2), [0.21, 0.29], dtype=np.float64),
                    np.full((3, 2), [0.31, 0.19], dtype=np.float64),
                ),
            }
        else:
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

    def getInterfaceEtas(self, time=None):
        index = 0 if float(time) == 0.0 else 1
        return self.etaData._y[index].copy()

    def getInterfaceCompositions(self, time=None):
        return (
            (
                np.asarray([0.10, 0.20], dtype=np.float64),
                np.asarray([0.20, 0.30], dtype=np.float64),
            ),
            (
                np.asarray([0.20, 0.30], dtype=np.float64),
                np.asarray([0.30, 0.20], dtype=np.float64),
            ),
        )


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


def _result(*, constant_profiles=False, diffusivity_mode=None, therm=None):
    eta_left = np.asarray([[0.10, 0.20], [0.12, 0.22], [0.14, 0.24]], dtype=np.float64)
    eta_right = np.asarray([[0.20, 0.30], [0.22, 0.32], [0.24, 0.34]], dtype=np.float64)
    bc_right = np.asarray([[0.30, 0.20], [0.32, 0.22], [0.34, 0.24]], dtype=np.float64)
    model = _SyntheticThreePhaseModel(constant_profiles=constant_profiles)
    if diffusivity_mode is not None:
        model.bulkDiffusivityMode = diffusivity_mode
    model.therm = therm
    return {
        "model": model,
        "surrogate_ab": _surrogate(("BCC", "LIQUID"), eta_left, eta_right),
        "surrogate_bc": _surrogate(("LIQUID", "BCC"), eta_right, bc_right),
    }


def _base_options():
    return {
        "renderer": None,
        "show_tielines": False,
        "show_global_average": False,
        "show_starting_phase_compositions": False,
        "show_interface_etas": False,
    }


def _plot_output(*args, **kwargs):
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    return plot_three_phase_composition_profile(*args, **kwargs)


def _figure(*args, **kwargs):
    return _plot_output(*args, **kwargs)["fig"]


def test_three_phase_plotly_helper_builds_combined_slider_figure():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    output = _plot_output(_result(), **_base_options())
    fig = output["fig"]

    assert "compositions" in output["aux"]
    assert "diffusivities" not in output["aux"]
    assert len(fig.layout.sliders) == 1
    assert len(fig.frames) == 2
    assert fig.data[0].type == "scatterternary"
    assert fig.data[3].type == "scatter"
    assert not [trace for trace in fig.data if "D[" in trace.name]
    assert fig.layout.width == 1650
    assert fig.layout.height == 900
    assert fig.layout.title.font.size == 11
    assert fig.layout.ternary.aaxis.title.text == ELEMENTS[2]
    assert fig.layout.ternary.baxis.title.text == ELEMENTS[0]
    assert fig.layout.ternary.caxis.title.text == ELEMENTS[1]


def test_three_phase_plotly_helper_uses_diagnostic_ternary_coordinates_and_distinct_phase_labels():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    fig = _figure(_result(), **_base_options())
    first_phase = _result()["model"].getTransformedState(0.0)[0]
    full = np.column_stack((1.0 - np.sum(first_phase, axis=1), first_phase))

    assert np.allclose(fig.data[0].a, full[:, 2])
    assert np.allclose(fig.data[0].b, full[:, 0])
    assert np.allclose(fig.data[0].c, full[:, 1])
    assert [fig.data[index].name for index in range(3)] == ["A: BCC", "B: LIQUID", "C: BCC"]


def test_three_phase_plotly_helper_tieline_overlay_can_be_enabled_or_disabled():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    with_tielines = _figure(
        _result(),
        show_tielines=True,
        tieline_eta_count=3,
        display_tieline_count=2,
        renderer=None,
        show_global_average=False,
        show_starting_phase_compositions=False,
        show_interface_etas=False,
    )
    without_tielines = _figure(_result(), **_base_options())

    tieline_traces = [trace for trace in with_tielines.data if "surrogate tie-lines" in trace.name]
    assert len(tieline_traces) == 4
    assert not [trace for trace in without_tielines.data if "surrogate tie-lines" in trace.name]


def test_three_phase_plotly_helper_frames_sync_interface_markers():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    fig = _figure(_result(), **_base_options())

    first_ab = fig.frames[0].data[-2].x
    second_ab = fig.frames[1].data[-2].x
    first_bc = fig.frames[0].data[-1].x
    second_bc = fig.frames[1].data[-1].x
    assert first_ab == (300000.0, 300000.0)
    assert second_ab == (400000.0, 400000.0)
    assert first_bc == (700000.0, 700000.0)
    assert second_bc == (650000.0, 650000.0)


def test_three_phase_plotly_helper_default_overlays_show_average_starting_compositions_and_etas():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    fig = _figure(
        _result(constant_profiles=True),
        renderer=None,
        show_tielines=False,
    )

    names = [trace.name for trace in fig.data]
    assert "Global average composition" in names
    assert "Starting phase compositions" in names
    assert "Misc info" in names
    assert {trace.name for trace in fig.data if trace.name.startswith("Global average X(")} == {
        "Global average X(Z)",
        "Global average X(X)",
        "Global average X(Y)",
    }

    average = next(trace for trace in fig.data if trace.name == "Global average composition")
    assert average.a[0] == pytest.approx(0.24)
    assert average.b[0] == pytest.approx(0.56)
    assert average.c[0] == pytest.approx(0.20)
    starting = next(trace for trace in fig.data if trace.name == "Starting phase compositions")
    assert starting.marker.symbol == "x"

    first_info = next(trace for trace in fig.frames[0].data if trace.name == "Misc info")
    second_info = next(trace for trace in fig.frames[1].data if trace.name == "Misc info")
    first_labels, first_values = first_info.cells.values
    second_labels, second_values = second_info.cells.values
    misc_info = next(trace for trace in fig.data if trace.name == "Misc info")
    assert misc_info.columnwidth == (0.58, 0.42)
    assert fig.layout.legend.orientation == "v"
    assert fig.layout.legend.font.size == 8
    assert fig.layout.legend.x == pytest.approx(-0.05)
    assert all(not trace.showlegend for trace in fig.data if trace.name.startswith("Global average X("))
    assert fig.layout.ternary.domain.y == (0.0, 1.0)
    assert fig.layout.xaxis.domain[0] > fig.layout.ternary.domain.x[1]
    assert misc_info.domain.x[0] == pytest.approx(fig.layout.xaxis.domain[0])
    assert misc_info.domain.x[1] == pytest.approx(fig.layout.xaxis.domain[1])
    assert list(first_labels) == ["Time", "A|B eta", "B|C eta", "A: BCC width", "B: LIQUID width", "C: BCC width"]
    assert first_values[1] == "0.25"
    assert first_values[2] == "0.75"
    assert list(first_values[3:]) == ["300000 um", "400000 um", "300000 um"]
    assert second_values[1] == "0.5"
    assert second_values[2] == "0.3"
    assert list(second_labels) == list(first_labels)
    assert list(second_values[3:]) == ["400000 um", "250000 um", "350000 um"]


def test_three_phase_plotly_helper_rejects_nonconstant_starting_phase_marker():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    with pytest.raises(ValueError, match="not constant"):
        _figure(
            _result(),
            renderer=None,
            show_tielines=False,
            show_global_average=False,
            show_starting_phase_compositions=True,
            show_interface_etas=False,
        )


def test_three_phase_plotly_helper_plots_phase_uniform_face_diffusivities():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    provider = _RecordingDiffusivityProvider()
    fig = _figure(
        _result(diffusivity_mode="phase_uniform", therm=provider),
        renderer=None,
        show_tielines=False,
        show_global_average=False,
        show_starting_phase_compositions=False,
        show_diffusivities=True,
    )

    diffusivity_traces = [trace for trace in fig.data if "D[" in trace.name]
    assert len(diffusivity_traces) == 12
    assert np.allclose(next(trace for trace in diffusivity_traces if trace.name == "A: BCC D[0,0]").y, [1.1, 1.1])
    assert np.allclose(next(trace for trace in diffusivity_traces if trace.name == "B: LIQUID D[0,0]").y, [10.2, 10.2])
    assert np.allclose(next(trace for trace in diffusivity_traces if trace.name == "C: BCC D[0,0]").y, [1.3, 1.3])
    assert np.allclose(
        next(trace for trace in diffusivity_traces if trace.name == "A: BCC D[0,0]").x,
        [75000.0, 225000.0],
    )
    assert {call["query_context"] for call in provider.calls} == {"interface"}
    assert {annotation.text for annotation in fig.layout.annotations if annotation.text is not None} >= {
        "D[0,0]",
        "D[0,1]",
        "D[1,0]",
        "D[1,1]",
    }
    for matrix_component in ((0, 0), (0, 1), (1, 0), (1, 1)):
        row = matrix_component[0] + 1
        col = matrix_component[1] + 3
        assert fig.get_subplot(row, col).yaxis.type == "log"
    assert any(trace.name == "Misc info" for trace in fig.data)


def test_three_phase_plotly_helper_returns_diffusivities_in_aux_data():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    provider = _RecordingDiffusivityProvider()
    output = _plot_output(
        _result(diffusivity_mode="phase_uniform", therm=provider),
        renderer=None,
        show_tielines=False,
        show_global_average=False,
        show_starting_phase_compositions=False,
        show_diffusivities=True,
    )

    assert set(output) == {"fig", "aux"}
    diffusivities = output["aux"]["diffusivities"]
    assert np.allclose(diffusivities["times"], [0.0, 1.0])
    assert diffusivities["phase_labels"] == ("A: BCC", "B: LIQUID", "C: BCC")
    assert diffusivities["distance_unit"] == "um"
    assert diffusivities["diffusivity_unit"] == "m^2/s"
    assert diffusivities["matrix_components"] == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert len(diffusivities["frames"]) == 2
    assert len(diffusivities["frames"][0]["segments"]) == 3
    assert diffusivities["frames"][0]["segments"][0]["phase"] == "A: BCC"
    assert diffusivities["frames"][0]["segments"][0]["distance"].shape == (2,)
    assert diffusivities["frames"][0]["segments"][0]["matrices"].shape == (2, 2, 2)
    assert np.allclose(
        diffusivities["frames"][0]["segments"][0]["matrices"][:, 0, 0],
        [1.1, 1.1],
    )
    compositions = output["aux"]["compositions"]
    assert compositions["elements"] == ELEMENTS
    assert compositions["phase_labels"] == ("A: BCC", "B: LIQUID", "C: BCC")
    assert compositions["composition_unit"] == "mole_fraction"
    assert len(compositions["frames"]) == 2
    assert compositions["frames"][0]["interfaces"].shape == (2,)
    assert compositions["frames"][0]["phase_widths"].shape == (3,)
    assert compositions["frames"][0]["global_average"].shape == (3,)
    assert compositions["frames"][0]["segments"][0]["compositions"].shape == (3, 3)
    assert np.allclose(
        compositions["frames"][0]["segments"][0]["compositions"][0],
        [0.70, 0.10, 0.20],
    )


def test_three_phase_plotly_helper_displays_run_metadata_in_title():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    result = _result()
    result["surrogate_ab"].metadata["source"] = "from_database_seed_point"
    result["surrogate_bc"].metadata["source"] = "from_database_seed_point"
    fig = _figure(
        result,
        renderer=None,
        show_tielines=False,
        show_global_average=False,
        show_starting_phase_compositions=False,
        show_interface_etas=False,
        run_config={
            "THERM_ENGINE": "TC",
            "TC_USE_DEFAULT_PHASES": True,
            "PYCALPHAD_USE_DEFAULT_PHASES": False,
        },
    )

    title = fig.layout.title.text
    assert "TEMPERATURE=1000 K" in title
    assert "TIELINE_SURROGATE_BUILD_MODE=seed_point" in title
    assert "BULK_DIFFUSIVITY_MODE=phase_uniform" in title
    assert "DT_MODE=semi_log" in title
    assert "TOLERANCE=1e-10" in title
    assert "SEMI_LOG_DT=0.05" in title
    assert "PHASE_NODES=(3, 3, 3)" in title
    assert "THERM_ENGINE=TC" in title
    assert "TC_USE_DEFAULT_PHASES=True" in title
    assert "PYCALPHAD_USE_DEFAULT_PHASES=False" in title


def test_three_phase_plotly_helper_uses_symmetric_log_only_for_signed_entries():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    provider = _RecordingDiffusivityProvider(negative_components=((0, 1), (1, 0)))
    fig = _figure(
        _result(diffusivity_mode="phase_uniform", therm=provider),
        renderer=None,
        show_tielines=False,
        show_global_average=False,
        show_starting_phase_compositions=False,
        show_interface_etas=False,
        show_diffusivities=True,
        use_symmetric_log=True,
    )

    assert fig.get_subplot(1, 3).yaxis.type == "log"
    assert fig.get_subplot(1, 4).yaxis.type == "linear"
    assert fig.get_subplot(2, 3).yaxis.type == "linear"
    assert fig.get_subplot(2, 4).yaxis.type == "log"
    assert "symlog" in fig.get_subplot(1, 4).yaxis.title.text
    assert "symlog" in fig.get_subplot(2, 3).yaxis.title.text

    signed_trace = next(trace for trace in fig.data if trace.name == "A: BCC D[0,1]")
    original_values = np.asarray(signed_trace.customdata, dtype=np.float64)[:, 2]
    assert np.all(original_values < 0.0)
    assert np.all(signed_trace.y < 0.0)
    assert not np.allclose(signed_trace.y, original_values)


@pytest.mark.parametrize("mode", ["composition_dependent_lagged", "composition_dependent_implicit"])
def test_three_phase_plotly_helper_reconstructs_composition_dependent_face_diffusivities(mode):
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    provider = _RecordingDiffusivityProvider()
    fig = _figure(
        _result(diffusivity_mode=mode, therm=provider),
        renderer=None,
        show_tielines=False,
        show_global_average=False,
        show_starting_phase_compositions=False,
        show_interface_etas=False,
        show_diffusivities=True,
    )

    first_d00 = next(trace for trace in fig.frames[0].data if trace.name == "A: BCC D[0,0]")
    second_d00 = next(trace for trace in fig.frames[1].data if trace.name == "A: BCC D[0,0]")
    assert np.allclose(first_d00.x, [75000.0, 225000.0])
    assert np.allclose(first_d00.y, [1.11, 1.13])
    assert np.allclose(second_d00.y, [1.12, 1.14])
    assert {trace.name.rsplit(" ", 1)[-1] for trace in fig.data if "D[" in trace.name} == {
        "D[0,0]",
        "D[0,1]",
        "D[1,0]",
        "D[1,1]",
    }
    assert {call["query_context"] for call in provider.calls} == {"general"}
    assert np.allclose(provider.calls[0]["composition"], [[0.11, 0.21], [0.13, 0.23]])
    assert np.allclose(provider.calls[0]["temperature"], [1000.0, 1000.0])


def test_three_phase_plotly_helper_supports_scalar_only_diffusivity_providers():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    provider = _RecordingDiffusivityProvider(scalar_only=True)
    fig = _figure(
        _result(diffusivity_mode="composition_dependent_lagged", therm=provider),
        renderer=None,
        show_tielines=False,
        show_global_average=False,
        show_starting_phase_compositions=False,
        show_interface_etas=False,
        show_diffusivities=True,
    )

    assert np.allclose(
        next(trace for trace in fig.data if trace.name == "A: BCC D[0,0]").y,
        [1.11, 1.13],
    )
    assert any(call["composition"].ndim == 1 for call in provider.calls)


def test_three_phase_plotly_helper_reports_missing_diffusivity_provider():
    pytest.importorskip("plotly")
    from examples.ternaryExamples.IllingworthTernaryThreePhasePlotly import plot_three_phase_composition_profile

    with pytest.raises(ValueError, match="getInterdiffusivity"):
        _figure(
            _result(diffusivity_mode="phase_uniform"),
            renderer=None,
            show_tielines=False,
            show_global_average=False,
            show_starting_phase_compositions=False,
            show_interface_etas=False,
            show_diffusivities=True,
        )
