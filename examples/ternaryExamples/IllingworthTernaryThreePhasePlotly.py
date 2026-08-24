"""Plotly helpers for three-phase Illingworth ternary example results."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kawin.diffusion import evaluate_tieline_diagnostics


_PHASE_PREFIXES = ("A", "B", "C")
_PHASE_COLORS = ("#1f77b4", "#d62728", "#2ca02c")
_COMPONENT_COLORS = ("#4c78a8", "#f58518", "#54a24b")
_TIELINE_COLORS = ("rgba(80,80,80,0.32)", "rgba(110,110,110,0.32)")


def _require_plotly(renderer="browser"):
    """Import Plotly lazily and configure the requested default renderer."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Plotly three-phase Illingworth helpers require the optional dependency; "
            "install 'kawin[diagnostics]'."
        ) from exc
    if renderer is not None:
        pio.renderers.default = str(renderer)
    return go, make_subplots


def _history_times(model):
    history = getattr(model, "interfaceData", None)
    if history is None or not hasattr(history, "_time") or not hasattr(history, "N"):
        raise ValueError("result['model'] must expose recorded interfaceData history.")
    count = int(history.N) + 1
    if count <= 0:
        raise ValueError("result['model'] does not contain recorded times.")
    return np.asarray(history._time[:count], dtype=np.float64)


def _selected_time_indices(times, time_indices, max_frames):
    """Validate explicit time indices and optionally thin them for frames."""
    count = len(times)
    if time_indices is None:
        indices = np.arange(count, dtype=int)
    else:
        indices = np.asarray(time_indices, dtype=int).reshape(-1)
        if indices.size == 0:
            raise ValueError("time_indices must select at least one recorded time.")
        if np.any(indices < 0) or np.any(indices >= count):
            raise IndexError(f"time_indices must be between 0 and {count - 1}.")
        _, first_positions = np.unique(indices, return_index=True)
        indices = indices[np.sort(first_positions)]
    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError("max_frames must be positive when specified.")
        if indices.size > max_frames:
            keep = np.unique(np.linspace(0, indices.size - 1, max_frames).round().astype(int))
            indices = indices[keep]
    return indices


def _elements(model):
    values = getattr(model, "allElements", None)
    if values is None:
        values = getattr(model, "elements", None)
    values = tuple(str(value) for value in values)
    if len(values) != 3:
        raise ValueError("Three-phase ternary plotting requires exactly three elements.")
    return values


def _component_indices(elements, components):
    """Return selected full-composition component indices."""
    if components is None:
        return (0, 1, 2)
    if isinstance(components, (str, int, np.integer)):
        components = (components,)
    selected = []
    upper_lookup = {element.upper(): index for index, element in enumerate(elements)}
    for component in components:
        if isinstance(component, (int, np.integer)):
            index = int(component)
            if index < 0 or index >= len(elements):
                raise IndexError("component indices must be in [0, 2].")
        else:
            key = str(component).upper()
            if key.startswith("X(") and key.endswith(")"):
                key = key[2:-1].strip().upper()
            if key not in upper_lookup:
                raise ValueError(f"Unknown component '{component}'; expected one of {elements}.")
            index = upper_lookup[key]
        if index not in selected:
            selected.append(index)
    if not selected:
        raise ValueError("components must select at least one element.")
    return tuple(selected)


def _distance_scale(unit):
    unit_key = str(unit).lower()
    if unit_key in {"m", "meter", "meters"}:
        return 1.0, "m"
    if unit_key in {"um", "micron", "microns", "micrometer", "micrometers"}:
        return 1.0e6, "um"
    if unit_key in {"nm", "nanometer", "nanometers"}:
        return 1.0e9, "nm"
    if unit_key in {"mm", "millimeter", "millimeters"}:
        return 1.0e3, "mm"
    raise ValueError("distance_unit must be one of 'm', 'um', 'nm', or 'mm'.")


def _full_composition(independent):
    independent = np.asarray(independent, dtype=np.float64)
    if independent.ndim != 2 or independent.shape[1] != 2:
        raise ValueError("Transformed phase profiles must have shape (n_nodes, 2).")
    dependent = 1.0 - np.sum(independent, axis=1)
    return np.column_stack((dependent, independent))


def _ternary_coordinates(full_composition):
    """Map full reference-first ternary compositions to the diagnostics corner order."""
    full_composition = np.asarray(full_composition, dtype=np.float64)
    return {"a": full_composition[:, 2], "b": full_composition[:, 0], "c": full_composition[:, 1]}


def _phase_labels(model):
    phases = tuple(str(phase) for phase in getattr(model, "phases", ()))
    if len(phases) != 3:
        raise ValueError("Three-phase plotting requires exactly three model phases.")
    return tuple(f"{prefix}: {phase}" for prefix, phase in zip(_PHASE_PREFIXES, phases))


def _frame_profile(model, time, elements, component_indices, distance_scale):
    """Build all dynamic profile arrays for one recorded time."""
    interfaces = np.asarray(model.getInterfacePositions(float(time)), dtype=np.float64).reshape(2)
    profiles = tuple(np.asarray(profile, dtype=np.float64) for profile in model.getTransformedState(float(time)))
    grids = tuple(np.asarray(grid, dtype=np.float64).reshape(-1) for grid in getattr(model, "_grids", ()))
    domain_length = float(getattr(model, "_R", np.nan))
    if len(profiles) != 3 or len(grids) != 3:
        raise ValueError("The model must expose three transformed profiles and three grids.")
    if not np.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("The model must expose a positive domain length as _R.")

    boundaries = np.concatenate(([0.0], interfaces, [domain_length]))
    phase_segments = []
    for phase_index, (profile, grid, left, right) in enumerate(zip(profiles, grids, boundaries[:-1], boundaries[1:])):
        if len(grid) != len(profile):
            raise ValueError(f"grid/profile length mismatch for phase interval {phase_index}.")
        distance = (float(left) + grid * (float(right) - float(left))) * distance_scale
        full = _full_composition(profile)
        phase_segments.append({"distance": distance, "full": full})

    xy_x = []
    xy_y = {index: [] for index in component_indices}
    for segment in phase_segments:
        xy_x.extend(segment["distance"].tolist())
        xy_x.append(np.nan)
        for component_index in component_indices:
            xy_y[component_index].extend(segment["full"][:, component_index].tolist())
            xy_y[component_index].append(np.nan)
    if xy_x:
        xy_x.pop()
        for values in xy_y.values():
            values.pop()

    return {
        "time": float(time),
        "interfaces": interfaces * distance_scale,
        "phase_segments": phase_segments,
        "xy_x": np.asarray(xy_x, dtype=np.float64),
        "xy_y": {index: np.asarray(values, dtype=np.float64) for index, values in xy_y.items()},
        "elements": elements,
    }


def _phase_customdata(frame, phase_index, phase_label):
    segment = frame["phase_segments"][phase_index]
    count = len(segment["distance"])
    return np.column_stack(
        (
            np.full(count, frame["time"], dtype=np.float64),
            np.full(count, phase_label, dtype=object),
            segment["distance"],
            segment["full"],
        )
    )


def _component_customdata(frame, component_index):
    x = frame["xy_x"]
    y = frame["xy_y"][component_index]
    return np.column_stack((np.full(len(x), frame["time"], dtype=np.float64), x, y))


def _phase_trace(go, frame, phase_index, phase_label, elements, unit_label):
    segment = frame["phase_segments"][phase_index]
    return go.Scatterternary(
        **_ternary_coordinates(segment["full"]),
        mode="lines+markers",
        name=phase_label,
        legendgroup=f"profile-{phase_index}",
        line={"color": _PHASE_COLORS[phase_index % len(_PHASE_COLORS)], "width": 2},
        marker={"color": _PHASE_COLORS[phase_index % len(_PHASE_COLORS)], "size": 5},
        customdata=_phase_customdata(frame, phase_index, phase_label),
        hovertemplate=(
            "time=%{customdata[0]:.6g} s<br>"
            "phase=%{customdata[1]}<br>"
            f"distance=%{{customdata[2]:.6g}} {unit_label}<br>"
            f"X({elements[0]})=%{{customdata[3]:.6g}}<br>"
            f"X({elements[1]})=%{{customdata[4]:.6g}}<br>"
            f"X({elements[2]})=%{{customdata[5]:.6g}}<extra></extra>"
        ),
    )


def _component_trace(go, frame, component_index, elements, unit_label):
    return go.Scatter(
        x=frame["xy_x"],
        y=frame["xy_y"][component_index],
        mode="lines+markers",
        name=f"X({elements[component_index]})",
        legendgroup=f"component-{component_index}",
        line={"color": _COMPONENT_COLORS[component_index % len(_COMPONENT_COLORS)], "width": 2},
        marker={"color": _COMPONENT_COLORS[component_index % len(_COMPONENT_COLORS)], "size": 4},
        customdata=_component_customdata(frame, component_index),
        hovertemplate=(
            "time=%{customdata[0]:.6g} s<br>"
            f"distance=%{{customdata[1]:.6g}} {unit_label}<br>"
            f"X({elements[component_index]})=%{{customdata[2]:.6g}}<extra></extra>"
        ),
    )


def _interface_trace(go, frame, interface_index, unit_label):
    x = float(frame["interfaces"][interface_index])
    return go.Scatter(
        x=[x, x],
        y=[0.0, 1.0],
        mode="lines",
        name=("A|B interface" if interface_index == 0 else "B|C interface"),
        legendgroup="interfaces",
        showlegend=interface_index == 0,
        line={"color": "rgba(40,40,40,0.55)", "dash": "dash", "width": 1.5},
        hovertemplate=f"distance=%{{x:.6g}} {unit_label}<extra></extra>",
    )


def _sample_indices(count, display_count):
    display_count = int(display_count)
    if display_count <= 0:
        raise ValueError("display_tieline_count must be positive.")
    if count <= display_count:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, display_count).round().astype(int))


def _add_static_tielines(fig, go, result, eta_count, display_count):
    """Overlay sampled A|B and B|C surrogate tie-lines on the ternary subplot."""
    for interface_index, key in enumerate(("surrogate_ab", "surrogate_bc")):
        surrogate = result.get(key)
        if surrogate is None:
            continue
        report = evaluate_tieline_diagnostics(surrogate, eta_count=eta_count)
        phases = tuple(report["phases"])
        endpoints = report["endpoint_compositions"]
        eta = np.asarray(report["eta"], dtype=np.float64)
        for position, sample_index in enumerate(_sample_indices(len(eta), display_count)):
            left = endpoints[phases[0]][sample_index]
            right = endpoints[phases[1]][sample_index]
            full = np.column_stack(
                (
                    1.0 - np.sum(np.vstack((left, right)), axis=1),
                    np.vstack((left, right)),
                )
            )
            fig.add_trace(
                go.Scatterternary(
                    **_ternary_coordinates(full),
                    mode="lines",
                    name=f"{phases[0]} | {phases[1]} surrogate tie-lines",
                    legendgroup=f"tielines-{interface_index}",
                    showlegend=position == 0,
                    line={"color": _TIELINE_COLORS[interface_index % len(_TIELINE_COLORS)], "width": 1},
                    hovertemplate=f"eta={eta[sample_index]:.6g}<extra>{phases[0]} | {phases[1]}</extra>",
                ),
                row=1,
                col=1,
            )


def _frame_name(index, time):
    return f"t{index}_{float(time):.12g}"


def plot_three_phase_composition_profile(
    result,
    *,
    time_indices=None,
    max_frames=None,
    components=None,
    distance_unit="um",
    show_tielines=True,
    tieline_eta_count=101,
    display_tieline_count=21,
    renderer="browser",
):
    """
    Build a Plotly ternary-plus-xy profile figure for a three-phase run result.

    The input must be a mapping returned by
    ``IllingworthTernaryThreePhaseNiTiNb_TC.run_case`` with a solved or setup
    three-phase model stored under ``"model"``. Profiles are read from the
    recorded transformed phase histories so each moving interval is plotted in
    its own phase color and interface discontinuities are preserved.
    """
    if "model" not in result:
        raise KeyError("result must contain result['model'].")
    model = result["model"]
    if getattr(model, "profileData", None) is None:
        raise ValueError("Transformed profile history is required; build the model with record_pq_data=True.")

    go, make_subplots = _require_plotly(renderer)
    times = _history_times(model)
    selected_indices = _selected_time_indices(times, time_indices, max_frames)
    elements = _elements(model)
    component_indices = _component_indices(elements, components)
    distance_scale, unit_label = _distance_scale(distance_unit)
    phase_labels = _phase_labels(model)
    frames_data = [
        _frame_profile(model, times[index], elements, component_indices, distance_scale)
        for index in selected_indices
    ]
    initial = frames_data[0]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "ternary"}, {"type": "xy"}]],
        column_widths=(0.48, 0.52),
        horizontal_spacing=0.10,
        subplot_titles=("Composition path", "Composition profile"),
    )

    dynamic_trace_indices = []
    for phase_index, phase_label in enumerate(phase_labels):
        dynamic_trace_indices.append(len(fig.data))
        fig.add_trace(_phase_trace(go, initial, phase_index, phase_label, elements, unit_label), row=1, col=1)
    for component_index in component_indices:
        dynamic_trace_indices.append(len(fig.data))
        fig.add_trace(_component_trace(go, initial, component_index, elements, unit_label), row=1, col=2)
    for interface_index in range(2):
        dynamic_trace_indices.append(len(fig.data))
        fig.add_trace(_interface_trace(go, initial, interface_index, unit_label), row=1, col=2)

    if show_tielines:
        _add_static_tielines(fig, go, result, int(tieline_eta_count), int(display_tieline_count))

    frames = []
    for frame_number, frame in enumerate(frames_data):
        traces = [
            _phase_trace(go, frame, phase_index, phase_label, elements, unit_label)
            for phase_index, phase_label in enumerate(phase_labels)
        ]
        traces.extend(_component_trace(go, frame, component_index, elements, unit_label) for component_index in component_indices)
        traces.extend(_interface_trace(go, frame, interface_index, unit_label) for interface_index in range(2))
        frames.append(
            go.Frame(
                name=_frame_name(frame_number, frame["time"]),
                data=traces,
                traces=dynamic_trace_indices,
            )
        )
    fig.frames = frames

    slider_steps = [
        {
            "label": f"{frame['time']:.6g} s",
            "method": "animate",
            "args": [
                [_frame_name(frame_number, frame["time"])],
                {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}},
            ],
        }
        for frame_number, frame in enumerate(frames_data)
    ]
    fig.update_layout(
        template="plotly_white",
        width=1250,
        height=700,
        title=f"Three-phase Illingworth composition profile, t={initial['time']:.6g} s",
        margin={"t": 95, "b": 90},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "time: "},
                "pad": {"t": 45},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 250, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    },
                ],
            }
        ],
        ternary={
            "sum": 1,
            "aaxis": {"title": elements[2]},
            "baxis": {"title": elements[0]},
            "caxis": {"title": elements[1]},
        },
    )
    all_x = np.concatenate([frame["xy_x"][np.isfinite(frame["xy_x"])] for frame in frames_data])
    fig.update_xaxes(title_text=f"Distance ({unit_label})", range=[float(np.min(all_x)), float(np.max(all_x))], row=1, col=2)
    fig.update_yaxes(title_text="Mole fraction", range=[0.0, 1.0], row=1, col=2)
    return fig


__all__ = ["plot_three_phase_composition_profile"]
