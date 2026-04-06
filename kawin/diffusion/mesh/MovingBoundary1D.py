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
