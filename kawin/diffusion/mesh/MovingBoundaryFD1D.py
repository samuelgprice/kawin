from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MovingBoundaryFDGeometry:
    """
    Discrete geometry description for a node-centered FDM moving interface.

    The interface lies between ``left_index`` and ``right_index`` with
    normalized local coordinate ``p`` measured from the left node. The
    ``ignored_index`` follows the Lee/Oh-style explicit treatment where one
    adjacent node is reconstructed rather than updated directly.
    """
    left_index: int
    right_index: int
    interface_position: float
    p: float
    ignored_index: int
    left_near_index: int
    right_near_index: int
    left_distance: float
    right_distance: float


def _flatten_1d_coordinates(z):
    """
    Returns a 1D view of mesh coordinates.

    The FDM mesh stores 1D coordinates as ``(N, 1)`` arrays, but several
    helper routines operate more naturally on flattened vectors.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 2 and z.shape[1] == 1:
        return z[:, 0]
    return np.ravel(z)


def _regularize_interface_position(z, interface_position: float) -> float:
    """
    Nudges the interface off exact node locations and domain edges.

    Exact coincidence with a node makes the Lagrange interpolation stencils
    degenerate, so this helper shifts the position by a very small amount while
    keeping it inside the physical domain.
    """
    z = _flatten_1d_coordinates(z)
    if len(z) < 2:
        return float(interface_position)
    dx = float(np.min(np.diff(z)))
    eps = max(dx * 1e-8, 1e-14)
    position = float(interface_position)
    if position <= z[0]:
        return float(z[0] + eps)
    if position >= z[-1]:
        return float(z[-1] - eps)

    matches = np.where(np.isclose(z, position, atol=eps, rtol=0.0))[0]
    if len(matches) > 0:
        idx = int(matches[0])
        if idx == 0:
            return float(z[0] + eps)
        if idx == len(z) - 1:
            return float(z[-1] - eps)
        return float(position + eps)
    return position


def _geometry_from_z(z, interface_position: float, pstar: float) -> MovingBoundaryFDGeometry:
    """
    Builds moving-boundary geometry directly from a 1D coordinate vector.

    This is the array-based implementation behind
    :func:`get_moving_boundary_fd_geometry` and is reused by the pure helper
    routines in this module.
    """
    z = _flatten_1d_coordinates(z)
    interface_position = _regularize_interface_position(z, interface_position)
    if len(z) < 4:
        raise ValueError("FDM moving boundary model requires at least four grid nodes.")
    if not (0 < pstar < 1):
        raise ValueError("pstar must lie strictly between 0 and 1.")
    if interface_position <= z[0] or interface_position >= z[-1]:
        raise ValueError("Interface position must lie strictly inside the FDM domain.")

    right_index = int(np.searchsorted(z, interface_position, side="right"))
    left_index = right_index - 1
    if left_index < 0 or right_index >= len(z):
        raise ValueError("Interface position must lie between two adjacent nodes.")

    dx = float(z[1] - z[0])
    p = float((interface_position - z[left_index]) / dx)
    p = float(np.clip(p, 0.0, 1.0))
    ignored_index = left_index if p < pstar else right_index
    left_near_index = right_index - 1 if p < pstar else left_index
    right_near_index = right_index if p < pstar else min(right_index + 1, len(z) - 1)
    return MovingBoundaryFDGeometry(
        left_index=left_index,
        right_index=right_index,
        interface_position=float(interface_position),
        p=p,
        ignored_index=int(ignored_index),
        left_near_index=int(left_near_index),
        right_near_index=int(right_near_index),
        left_distance=float(interface_position - z[left_index]),
        right_distance=float(z[right_index] - interface_position),
    )


def get_moving_boundary_fd_geometry(mesh, interface_position: float, pstar: float) -> MovingBoundaryFDGeometry:
    """
    Returns the discrete interface geometry for a 1D FDM mesh.

    Parameters
    ----------
    mesh : object
        Mesh-like object with a ``z`` coordinate array.
    interface_position : float
        Current physical interface position.
    pstar : float
        Switching threshold used to decide which interface-adjacent node is
        ignored and reconstructed.
    """
    return _geometry_from_z(mesh.z, interface_position, pstar)


def exact_lagrange(x, derivative_num: int, xi, yi):
    """
    Evaluates a quadratic Lagrange interpolant or one of its derivatives.

    Parameters
    ----------
    x : float
        Evaluation point.
    derivative_num : int
        ``0`` for the value, ``1`` for the first derivative, or ``2`` for the
        second derivative.
    xi, yi : array-like
        Three interpolation coordinates and corresponding values.
    """
    x1, x2, x3 = xi
    y1, y2, y3 = yi
    if derivative_num == 0:
        func = lambda x_: y1 + (x_ - x1) * (
            (y2 - y1) / (x2 - x1)
            + ((x_ - x2) * ((y1 - y2) / (x2 - x1) + (y3 - y2) / (x3 - x2))) / (x3 - x1)
        )
    elif derivative_num == 1:
        func = lambda x_: (y2 - y1) / (x2 - x1) + ((x_ - x1) + (x_ - x2)) * (
            ((y1 - y2) / (x2 - x1) + (y3 - y2) / (x3 - x2))
        ) / (x3 - x1)
    elif derivative_num == 2:
        func = lambda x_: 2 * (((y1 - y2) / (x2 - x1) + (y3 - y2) / (x3 - x2))) / (x3 - x1)
    else:
        raise ValueError("derivative_num must be 0, 1 or 2.")
    return float(func(x))


def quad_fit_derivs(x: np.ndarray, y: np.ndarray, x_eval: float):
    """
    Returns the first and second derivatives of a quadratic fit at ``x_eval``.
    """
    return (
        exact_lagrange(x_eval, derivative_num=1, xi=x, yi=y),
        exact_lagrange(x_eval, derivative_num=2, xi=x, yi=y),
    )


def interpolate_previous_ignored_composition(z, composition, s_old, p_old, s_new, pstar, interface_compositions):
    """
    Reconstructs the interface-adjacent node that is ignored by the explicit step.

    When the interface moves, the node treated as ignored by the Lee/Oh update
    can change. This helper fills that node using quadratic interpolation from
    the retained neighboring nodes and the imposed interface compositions.
    """
    z = _flatten_1d_coordinates(z)
    c = np.asarray(composition, dtype=np.float64).copy()
    geom_old = _geometry_from_z(z, s_old, pstar)
    geom_new = _geometry_from_z(z, s_new, pstar)
    index = geom_old.ignored_index

    if index <= geom_new.left_index:
        if geom_new.left_index - 2 >= 0:
            c[index] = exact_lagrange(
                z[index],
                derivative_num=0,
                xi=[z[geom_new.left_index - 1], z[geom_new.left_index], s_new],
                yi=[c[geom_new.left_index - 1], c[geom_new.left_index], interface_compositions[0]],
            )
        elif geom_new.left_index >= 0:
            c[index] = c[geom_new.left_index] + (interface_compositions[0] - c[geom_new.left_index]) / (geom_new.p + 1.0)
    else:
        if geom_new.right_index + 2 < len(z):
            c[index] = exact_lagrange(
                z[index],
                derivative_num=0,
                xi=[s_new, z[geom_new.right_index], z[geom_new.right_index + 1]],
                yi=[interface_compositions[1], c[geom_new.right_index], c[geom_new.right_index + 1]],
            )
        elif geom_new.right_index < len(z):
            c[index] = c[geom_new.right_index] - (c[geom_new.right_index] - interface_compositions[1]) / (2.0 - geom_new.p)
    return c


def augment_profile_with_interface_compositions(z, composition, interface_position, interface_compositions):
    """
    Inserts duplicated interface coordinates and the two interface compositions.

    The augmented profile is convenient for plotting and for mass integration
    schemes that treat the interface as a discontinuity with distinct left and
    right compositions.
    """
    z = _flatten_1d_coordinates(z)
    c = np.asarray(composition, dtype=np.float64).reshape(-1)
    s_idx = int(np.searchsorted(z, interface_position))
    z_aug = np.concatenate((z[:s_idx], [interface_position, interface_position], z[s_idx:]))
    c_aug = np.concatenate((c[:s_idx], np.asarray(interface_compositions, dtype=np.float64), c[s_idx:]))
    return z_aug, c_aug


def integrate_binary_fd_profile(
    z,
    composition,
    s_old,
    p_old,
    s_new,
    pstar,
    interface_compositions,
    integration_mode="weighted",
    s_for_interp="new",
):
    """
    Integrates a binary FDM composition profile that contains a sharp interface.

    Parameters
    ----------
    z, composition : array-like
        Node coordinates and node-centered composition values.
    s_old, p_old, s_new : float
        Old and new interface positions together with the old normalized
        interface coordinate.
    pstar : float
        Threshold used by the ignored-node logic.
    interface_compositions : tuple[float, float]
        Left and right interface compositions.
    integration_mode : str
        One of ``"ignore"``, ``"noIgnore"``, or ``"weighted"``.
    s_for_interp : str
        Whether reconstruction of the ignored node should use the old or new
        interface placement.
    """
    z = _flatten_1d_coordinates(z)
    c = np.asarray(composition, dtype=np.float64).reshape(-1).copy()
    geom_new = _geometry_from_z(z, s_new, pstar)

    if s_for_interp == "new":
        c = interpolate_previous_ignored_composition(z, c, s_old, p_old, s_new, pstar, interface_compositions)
    elif s_for_interp != "old":
        raise ValueError("s_for_interp must be 'new' or 'old'.")

    z_aug, c_aug = augment_profile_with_interface_compositions(z, c, s_new, interface_compositions)
    if integration_mode == "noIgnore":
        return float(np.trapezoid(c_aug, z_aug))

    s_idx = int(np.searchsorted(z, s_new))
    if geom_new.p < pstar:
        z_ignore = np.concatenate((z[: s_idx - 1], [s_new, s_new], z[s_idx:]))
        c_ignore = np.concatenate((c[: s_idx - 1], np.asarray(interface_compositions, dtype=np.float64), c[s_idx:]))
    else:
        z_ignore = np.concatenate((z[:s_idx], [s_new, s_new], z[s_idx + 1 :]))
        c_ignore = np.concatenate((c[:s_idx], np.asarray(interface_compositions, dtype=np.float64), c[s_idx + 1 :]))
    ignore_value = float(np.trapezoid(c_ignore, z_ignore))

    if integration_mode == "ignore":
        return ignore_value
    if integration_mode == "weighted":
        no_ignore_value = float(np.trapezoid(c_aug, z_aug))
        weight_no_ignore = geom_new.p / pstar if geom_new.p < pstar else (1.0 - geom_new.p) / (1.0 - pstar)
        weight_ignore = 1.0 - weight_no_ignore
        return float(weight_no_ignore * no_ignore_value + weight_ignore * ignore_value)
    raise ValueError("integration_mode must be one of ['ignore', 'noIgnore', 'weighted'].")
