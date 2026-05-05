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

    Returned geometry fields
    ------------------------
    left_index / right_index
        The two mesh nodes that bracket the interface position such that
        ``z[left_index] < interface_position < z[right_index]``.
    p
        The normalized interface position between the bracketing nodes, with
        ``p = 0`` at ``left_index`` and ``p = 1`` at ``right_index``.
    ignored_index
        The interface-adjacent node that is reconstructed rather than updated
        directly during the explicit Lee/Oh-style step.
    left_near_index / right_near_index
        The nodes immediately adjacent to the ignored node that receive the
        quadratic interface-aware stencil update on the left and right sides.
    left_distance / right_distance
        One-sided node-to-interface distances used in gradient and flux
        calculations at the interface.
    """
    z = _flatten_1d_coordinates(z)
    # interface_position = _regularize_interface_position(z, interface_position) ## Bypassing this for now as it is unlikely that the interface will need to nudged and this function is slow (mostly due to calling np.isclose() on entire array)
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

    if p < pstar:
        i1, i2 = left_index - 2, left_index - 1
        j1, j2 = right_index, right_index + 1
    else:
        i1, i2 = left_index - 1, left_index
        j1, j2 = right_index + 1, right_index + 2
    minIndx=0; maxIndx=len(z)-1
    validIndx = lambda i: i>=minIndx and i<=maxIndx
    # if (validIndx(left_index-2)!=True) or (validIndx(left_index+3)!=True):
    # if (i1<0) or (i2<0) or (j1>(len(z)-1)) or (j2>(len(z)-1)):
    if all([validIndx(i) for i in [i1, i2, j1, j2]])!=True:
        try:
            import debugpy
            # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
            debugpy.listen(5678)
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
            debugpy.breakpoint()
            print('break on this line')
        except:
            pass
        # raise ValueError("Interface position is too close to the domain boundary.")

    ignored_index = left_index if p < pstar else right_index
    left_near_index = left_index - 1 if p < pstar else left_index # left_near_index = max(0, left_index - 1) if p < pstar else left_index
    right_near_index = right_index if p < pstar else right_index + 1 # right_near_index = right_index if p < pstar else min(right_index + 1, len(z) - 1)
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
        if index!=geom_new.left_index:
            raise ValueError("Unexpected geometry change: ignored node moved more than one position to the left.")
        if geom_new.left_index - 2 >= 0:
            c[index] = exact_lagrange(
                z[index],
                derivative_num=0,
                xi=[z[geom_new.left_index - 2], z[geom_new.left_index - 1], s_new],
                yi=[c[geom_new.left_index - 2], c[geom_new.left_index - 1], interface_compositions[0]],
            )
        elif geom_new.left_index - 1 >= 0:
            c[index] = c[geom_new.left_index - 1] + (interface_compositions[0] - c[geom_new.left_index - 1]) / (geom_new.p + 1.0)
        else:
            raise ValueError(f"geom_new.left_index ({geom_new.left_index}) is too close to the left boundary to reconstruct the ignored node composition.")
    else:
        if index != geom_new.right_index:
            raise ValueError("Unexpected geometry change: ignored node moved more than one position to the right.")
        if geom_new.right_index + 2 < len(z):
            c[index] = exact_lagrange(
                z[index],
                derivative_num=0,
                xi=[s_new, z[geom_new.right_index + 1], z[geom_new.right_index + 2]],
                yi=[interface_compositions[1], c[geom_new.right_index + 1], c[geom_new.right_index + 2]],
            )
        elif geom_new.right_index + 1 < len(z):
            c[index] = c[geom_new.right_index + 1] - (c[geom_new.right_index + 1] - interface_compositions[1]) / (2.0 - geom_new.p)
        else:
            raise ValueError(f"geom_new.right_index ({geom_new.right_index}) is too close to the right boundary to reconstruct the ignored node composition.")
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
    s_for_interp='none',
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


def summarize_moving_boundary_fd_state(
    mesh,
    composition: np.ndarray,
    interface_position: float,
    pstar: float,
    window: int = 2,
    precision: int = 9,
    distance_multiplier: float = 1.0,
) -> str:
    '''
    Returns a compact text summary of the FDM moving-boundary mesh state.

    This is intended for quick inspection in the debugger, so it focuses on
    the interface-adjacent nodes while still listing the local node positions,
    the active ignored node and the one-sided interface distances.
    '''
    z = _flatten_1d_coordinates(mesh.z).astype(np.float64)
    composition = np.asarray(composition, dtype=np.float64).reshape(-1)
    if len(composition) != len(z):
        raise ValueError("Composition array must align with the 1D FDM moving-boundary mesh.")

    geom = get_moving_boundary_fd_geometry(mesh, interface_position, pstar)
    scale = float(distance_multiplier)
    left = max(0, geom.left_index - int(window))
    right = min(len(z) - 1, geom.right_index + int(window))
    fmt = f".{int(precision)}g"

    lines = [
        "MovingBoundaryFD1D state:",
        (
            f"  interface_position = {format(geom.interface_position * scale, fmt)} "
            f"between nodes[{geom.left_index}] = {format(z[geom.left_index] * scale, fmt)} "
            f"and nodes[{geom.right_index}] = {format(z[geom.right_index] * scale, fmt)}"
        ),
        (
            f"  p = {format(geom.p, fmt)}, pstar = {format(pstar, fmt)}, "
            f"ignored_index = {geom.ignored_index}"
        ),
        (
            f"  center-to-interface distances = left {format(geom.left_distance * scale, fmt)}, "
            f"right {format(geom.right_distance * scale, fmt)}"
        ),
        "  local nodes:",
    ]

    for i in range(left, right + 1):
        marker = ""
        if i == geom.left_index:
            marker = " <left of interface>"
        elif i == geom.right_index:
            marker = " <right of interface>"
        if i == geom.ignored_index:
            marker += " <ignored>"
        lines.append(
            "    "
            f"[{i}] position={format(z[i] * scale, fmt)} "
            f"composition={format(composition[i], fmt)}{marker}"
        )

    return "\n".join(lines)


def _extract_moving_boundary_fd_inputs(mesh, composition=None, interface_position=None, pstar=None):
    '''
    Extracts composition, interface position and pstar from a mesh-like object when possible.

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

    if pstar is None:
        for attr in ("pstar",):
            if hasattr(mesh, attr):
                pstar = getattr(mesh, attr)
                break

    if pstar is None:
        raise ValueError(
            "pstar was not supplied and could not be inferred from the mesh. "
            "Pass pstar explicitly or attach mesh.pstar."
        )

    return np.asarray(composition, dtype=np.float64).reshape(-1), float(interface_position), float(pstar)


def debug_moving_boundary_fd_state(
    mesh,
    composition=None,
    interface_position=None,
    pstar=None,
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
    Prints and optionally plots an FDM moving-boundary mesh state for debugger use.

    This accepts either:
    - an explicit mesh + composition + interface_position + pstar, or
    - an explicit mesh + composition, with interface position inferred from
      ``mesh.interface_position`` / ``mesh.interfacePosition`` and ``pstar``
      inferred from ``mesh.pstar`` when available.
    '''
    composition, interface_position, pstar = _extract_moving_boundary_fd_inputs(
        mesh,
        composition=composition,
        interface_position=interface_position,
        pstar=pstar,
    )
    summary = summarize_moving_boundary_fd_state(
        mesh,
        composition,
        interface_position,
        pstar,
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
            pstar=pstar,
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
