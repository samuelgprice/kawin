from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MovingBoundaryGeometry:
    left_index: int
    right_index: int
    left_face: float
    right_face: float
    left_center: float
    right_center: float
    interface_position: float
    left_volume: float
    right_volume: float
    left_distance: float
    right_distance: float


def _flatten_1d_coordinates(z):
    z = np.asarray(z)
    if z.ndim == 2 and z.shape[1] == 1:
        return z[:, 0]
    return np.ravel(z)


def get_moving_boundary_geometry(mesh, interface_position: float) -> MovingBoundaryGeometry:
    z = _flatten_1d_coordinates(mesh.z)
    z_edge = _flatten_1d_coordinates(mesh.zEdge)
    if len(z) < 2:
        raise ValueError("Moving boundary model requires at least two grid points.")

    if interface_position <= z[0] or interface_position >= z[-1]:
        raise ValueError("Interface position must lie between the first and last cell centers.")

    right_index = int(np.searchsorted(z, interface_position, side="right"))
    left_index = right_index - 1
    if left_index < 0 or right_index >= len(z):
        raise ValueError("Interface position must lie between two adjacent cell centers.")

    left_center = float(z[left_index])
    right_center = float(z[right_index])
    if not (left_center < interface_position < right_center):
        raise ValueError("Interface position must lie strictly between two adjacent cell centers.")

    left_face = float(z_edge[left_index])
    right_face = float(z_edge[right_index + 1])
    left_volume = interface_position - left_face
    right_volume = right_face - interface_position
    left_distance = interface_position - left_center
    right_distance = right_center - interface_position
    return MovingBoundaryGeometry(
        left_index=left_index,
        right_index=right_index,
        left_face=left_face,
        right_face=right_face,
        left_center=left_center,
        right_center=right_center,
        interface_position=float(interface_position),
        left_volume=float(left_volume),
        right_volume=float(right_volume),
        left_distance=float(left_distance),
        right_distance=float(right_distance),
    )


def get_control_volume_widths(mesh, interface_position: float) -> np.ndarray:
    z_edge = _flatten_1d_coordinates(mesh.zEdge)
    widths = np.diff(z_edge).astype(np.float64)
    geom = get_moving_boundary_geometry(mesh, interface_position)
    widths[geom.left_index] = geom.left_volume
    widths[geom.right_index] = geom.right_volume
    return widths


def integrate_binary_profile(mesh, composition: np.ndarray, interface_position: float) -> float:
    composition = np.asarray(composition, dtype=np.float64).reshape(-1)
    widths = get_control_volume_widths(mesh, interface_position)
    if len(composition) != len(widths):
        raise ValueError("Composition array must align with the 1D moving-boundary mesh.")
    return float(np.sum(composition * widths))


def summarize_moving_boundary_state(
    mesh,
    composition: np.ndarray,
    interface_position: float,
    window: int = 2,
    precision: int = 9,
    distance_multiplier: float = 1.0,
) -> str:
    '''
    Returns a compact text summary of the moving-boundary mesh state.

    This is intended for quick inspection in the debugger, so it focuses on
    the interface-adjacent cells while still listing the underlying faces,
    centers, cut-cell widths and cell compositions.
    '''
    z = _flatten_1d_coordinates(mesh.z).astype(np.float64)
    z_edge = _flatten_1d_coordinates(mesh.zEdge).astype(np.float64)
    composition = np.asarray(composition, dtype=np.float64).reshape(-1)
    if len(composition) != len(z):
        raise ValueError("Composition array must align with the 1D moving-boundary mesh.")

    geom = get_moving_boundary_geometry(mesh, interface_position)
    widths = get_control_volume_widths(mesh, interface_position)
    scale = float(distance_multiplier)

    left = max(0, geom.left_index - int(window))
    right = min(len(z) - 1, geom.right_index + int(window))
    fmt = f".{int(precision)}g"

    lines = [
        "MovingBoundary1D state:",
        (
            f"  interface_position = {format(geom.interface_position * scale, fmt)} "
            f"between centers[{geom.left_index}] = {format(geom.left_center * scale, fmt)} "
            f"and centers[{geom.right_index}] = {format(geom.right_center * scale, fmt)}"
        ),
        (
            f"  cut-cell faces = [{format(geom.left_face * scale, fmt)}, "
            f"{format(geom.interface_position * scale, fmt)}, {format(geom.right_face * scale, fmt)}]"
        ),
        (
            f"  cut-cell widths = left {format(geom.left_volume * scale, fmt)}, "
            f"right {format(geom.right_volume * scale, fmt)}"
        ),
        (
            f"  center-to-interface distances = left {format(geom.left_distance * scale, fmt)}, "
            f"right {format(geom.right_distance * scale, fmt)}"
        ),
        "  local cells:",
    ]

    for i in range(left, right + 1):
        marker = ""
        if i == geom.left_index:
            marker = " <left of interface>"
        elif i == geom.right_index:
            marker = " <right of interface>"
        lines.append(
            "    "
            f"[{i}] face=({format(z_edge[i] * scale, fmt)}, {format(z_edge[i+1] * scale, fmt)}) "
            f"center={format(z[i] * scale, fmt)} width={format(widths[i] * scale, fmt)} "
            f"composition={format(composition[i], fmt)}{marker}"
        )

    return "\n".join(lines)


def _extract_moving_boundary_inputs(mesh, composition=None, interface_position=None):
    '''
    Extracts composition and interface position from a mesh-like object when possible.

    This is meant to support debugger-time inspection where the caller may only
    have access to a mesh object plus a few local variables.
    '''
    if composition is None:
        raise ValueError(
            "Composition must be supplied explicitly. "
            "The debugger helper does not fall back to mesh.y because mesh.y may be stale during solving."
        )

    if interface_position is None:
        for attr in ("interface_position", "interfacePosition"):
            if hasattr(mesh, attr):
                interface_position = getattr(mesh, attr)
                break

    if interface_position is None:
        raise ValueError(
            "Interface position was not supplied and could not be inferred from the mesh. "
            "Pass interface_position explicitly or attach mesh.interface_position."
        )

    return np.asarray(composition, dtype=np.float64).reshape(-1), float(interface_position)


def debug_moving_boundary_state(
    mesh,
    composition=None,
    interface_position=None,
    interface_compositions=None,
    *,
    window: int = 2,
    precision: int = 9,
    distance_multiplier: float = 1.0,
    plot: bool = True,
    ax=None,
    show: bool = True,
    annotate: bool = True,
    annotate_positions: bool = False,
    annotate_interface_distances: bool = False,
    annotate_interface_compositions: bool = False,
    annotationWindow: int = 2,
    maxAnnotatedCells: int = 20,
    zoom_cells=None,
    print_summary: bool = True,
):
    '''
    Prints and optionally plots a moving-boundary mesh state for debugger use.

    This accepts either:
    - an explicit mesh + composition + interface_position, or
    - an explicit mesh + composition, with interface position inferred from
      ``mesh.interface_position`` or ``mesh.interfacePosition``.
    '''
    composition, interface_position = _extract_moving_boundary_inputs(
        mesh, composition=composition, interface_position=interface_position
    )
    summary = summarize_moving_boundary_state(
        mesh,
        composition,
        interface_position,
        window=window,
        precision=precision,
        distance_multiplier=distance_multiplier,
    )
    if print_summary:
        print(summary)

    axis = None
    if plot:
        from kawin.diffusion.Plot import plotMovingBoundaryState

        axis = plotMovingBoundaryState(
            mesh,
            composition=composition,
            interface_position=interface_position,
            interface_compositions=interface_compositions,
            ax=ax,
            distance_multiplier=distance_multiplier,
            annotate=annotate,
            annotate_positions=annotate_positions,
            annotate_interface_distances=annotate_interface_distances,
            annotate_interface_compositions=annotate_interface_compositions,
            annotationWindow=annotationWindow,
            maxAnnotatedCells=maxAnnotatedCells,
            zoom_cells=zoom_cells,
        )
        if show:
            import matplotlib.pyplot as plt

            plt.show()

    return summary, axis
