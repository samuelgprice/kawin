from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OlayeFDGeometry:
    """
    Discrete interface geometry for the Olaye/Ojo fixed-grid binary FD model.

    The interface lies strictly between ``left_index`` and ``right_index``.
    For the current implementation these are adjacent node indices.
    """

    left_index: int
    right_index: int
    interface_position: float
    p: float
    left_distance: float
    right_distance: float
    left_width: float
    right_width: float


def _flatten_1d_coordinates(z):
    """Returns a 1D view of a mesh coordinate array."""
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return np.ravel(arr)


def get_olaye_fd_geometry(mesh, interface_position: float) -> OlayeFDGeometry:
    """
    Returns discrete geometry around the moving interface.

    The control-volume widths at interface-adjacent nodes are truncated by the
    current interface location.
    """
    z = _flatten_1d_coordinates(mesh.z)
    if len(z) < 4:
        raise ValueError("Olaye FD model requires at least four grid nodes.")
    if interface_position <= z[0] or interface_position >= z[-1]:
        raise ValueError("Interface position must lie strictly inside the 1D FD domain.")

    right_index = int(np.searchsorted(z, interface_position, side="right"))
    left_index = right_index - 1
    if right_index >= len(z) or left_index < 0:
        raise ValueError("Interface must lie between two interior nodes.")
    if right_index != left_index + 1:
        raise ValueError("Expected adjacent bracketing nodes for a uniform 1D FD mesh.")

    dz = float(z[1] - z[0])
    p = float((interface_position - z[left_index]) / dz)
    left_distance = float(interface_position - z[left_index])
    right_distance = float(z[right_index] - interface_position)

    # Node-centered control volumes with half-cells at boundaries.
    left_width = float(0.5 * dz + left_distance)
    right_width = float(0.5 * dz + right_distance)

    return OlayeFDGeometry(
        left_index=int(left_index),
        right_index=int(right_index),
        interface_position=float(interface_position),
        p=p,
        left_distance=left_distance,
        right_distance=right_distance,
        left_width=left_width,
        right_width=right_width,
    )


def augment_profile_with_interface_compositions(
    z,
    composition,
    interface_position: float,
    interface_compositions: tuple[float, float],
):
    """
    Inserts duplicated interface coordinates and left/right interface values.

    This is useful for integration and plotting in a sharp-interface model.
    """
    z = _flatten_1d_coordinates(z)
    c = np.asarray(composition, dtype=np.float64).reshape(-1)
    s_idx = int(np.searchsorted(z, interface_position))
    z_aug = np.concatenate((z[:s_idx], [interface_position, interface_position], z[s_idx:]))
    c_aug = np.concatenate(
        (
            c[:s_idx],
            np.asarray(interface_compositions, dtype=np.float64).reshape(2),
            c[s_idx:],
        )
    )
    return z_aug, c_aug


def integrate_binary_olaye_fd_profile(
    z,
    composition,
    interface_position: float,
    interface_compositions: tuple[float, float],
) -> float:
    """
    Integrates the binary profile across the full domain with a sharp interface.

    The integral uses trapezoidal quadrature over an augmented profile that
    contains duplicated interface coordinates and side-specific interface values.
    """
    z_aug, c_aug = augment_profile_with_interface_compositions(
        z=z,
        composition=composition,
        interface_position=interface_position,
        interface_compositions=interface_compositions,
    )
    return float(np.trapezoid(c_aug, z_aug))


def build_piecewise_diffusivity_nodes(
    therm,
    phases: tuple[str, str],
    composition,
    temperature_nodes,
    geometry: OlayeFDGeometry,
):
    """
    Returns node diffusivities by evaluating phase-A left of interface and
    phase-B right of interface.
    """
    c = np.asarray(composition, dtype=np.float64).reshape(-1)
    T = np.asarray(temperature_nodes, dtype=np.float64).reshape(-1)
    if len(c) != len(T):
        raise ValueError("composition and temperature_nodes must have matching lengths.")

    left_phase, right_phase = phases
    D_nodes = np.zeros_like(c, dtype=np.float64)

    left_vals = np.asarray(
        therm.getInterdiffusivity(c[: geometry.left_index + 1], T[: geometry.left_index + 1], phase=left_phase),
        dtype=np.float64,
    ).reshape(-1)
    right_vals = np.asarray(
        therm.getInterdiffusivity(c[geometry.right_index :], T[geometry.right_index :], phase=right_phase),
        dtype=np.float64,
    ).reshape(-1)

    D_nodes[: geometry.left_index + 1] = left_vals
    D_nodes[geometry.right_index :] = right_vals
    return D_nodes
