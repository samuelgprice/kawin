from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from kawin.PlotUtils import _get_axis, _adjust_kwargs
from kawin.thermo.Mobility import u_to_x_frac, expand_u_frac, expand_x_frac, interstitials
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.DiffusionParameters import computeMobility, HashTable
from kawin.diffusion.mesh import FiniteVolumeGrid, FiniteVolume1D, FiniteDifference1D, Cartesian2D
from kawin.diffusion.mesh.MovingBoundary1D import get_moving_boundary_geometry
from kawin.diffusion.mesh.MovingBoundaryFD1D import get_moving_boundary_fd_geometry

def _get_1D_mesh(model: DiffusionModel):
    mesh = model.mesh
    if not isinstance(mesh, (FiniteVolume1D, FiniteDifference1D)):
        raise ValueError('Diffusion mesh must be a supported 1D finite-volume or finite-difference mesh.')
    return mesh

def _set_1D_xlim(ax, mesh: FiniteVolume1D, zScale, zOffset):
    ax.set_xlim([(mesh.zEdge[0]+zOffset)/zScale, (mesh.zEdge[-1]+zOffset)/zScale])
    ax.set_xlabel(f'Distance*{zScale:.0e} (m)')

def plot1D(model: DiffusionModel, elements=None, zScale=1, zOffset=0, time=None, ax=None, plotUFrac=False, *args, **kwargs):
    '''
    Plots composition profile of 1D mesh

    Parameters
    ----------
    model: Diffusion model
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    elements: list[str]
        List of elements to plot (can include the dependent/reference element)
        If elements is None, then it will plot the independent elements in the diffusion model
    zScale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. zScale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so zScale represents the desired unit/meters
        Defaults to 0
    zOffset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    ax: matplotlib Axis (optional)
        Will be created if None

    Returns
    -------
    matplotlib Axis
    '''
    ax = _get_axis(ax)
    mesh = _get_1D_mesh(model)
    # make sure elements is a list[str], either from the diffusion model or from user input
    elements = model.elements if elements is None else elements
    if isinstance(elements, str):
        elements = [elements]

    # Convert y to full composition space and convert to composition if plotting
    y_full = expand_u_frac(model.data.y(time), model.allElements, interstitials)
    if not plotUFrac:
        y_full = u_to_x_frac(y_full, model.allElements, interstitials)
    for e in elements:
        y = y_full[:,model.allElements.index(e)]
        plot_kwargs = _adjust_kwargs(e, {'label': e}, kwargs)
        ax.plot((mesh.z+zOffset)/zScale, y, *args, **plot_kwargs)

    _set_1D_xlim(ax, mesh, zScale, zOffset)
    if len(elements) > 1:
        ax.legend()
    ax.set_ylim([0,1])
    ax.set_ylabel(f'Composition (at.)')
    return ax

def plot1DTwoAxis(model: DiffusionModel, elementsL, elementsR, zScale=1, zOffset=0, time=None, axL=None, axR=None, plotUFrac=False, *args, **kwargs):
    '''
    Plots composition profile of 1D mesh on left and right axes

    Parameters
    ----------
    model: Diffusion model
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    elementsL: list[str]
        List of elements to plot (can include the dependent/reference element)
    elementsR: list[str]
        List of elements to plot (can include the dependent/reference element)
    zScale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. zScale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so zScale represents the desired unit/meters
        Defaults to 0
    zOffset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    axL: matplotlib axis (optional)
        Left axis
        Will be created if None
    axR: matplotlib axis (optional)
        Right axis
        Will be created if None

    Returns
    -------
    matplotlib Axis for left axis
    matplotlib Axis for right axis
    '''
    axL = _get_axis(axL)
    if axR is None:
        axR = axL.twinx()
    mesh = _get_1D_mesh(model)
    if isinstance(elementsL, str):
        elementsL = [elementsL]
    if isinstance(elementsR, str):
        elementsR = [elementsR]

    # Convert y to full composition space and convert to composition if plotting
    y_full = expand_u_frac(model.data.y(time), model.allElements, interstitials)
    if not plotUFrac:
        y_full = u_to_x_frac(y_full, model.allElements, interstitials)

    # Plotting is the same for each axis, so we can just loop across the two
    i = 0
    for ax, elements in zip([axL, axR], [elementsL, elementsR]):
        for e in elements:
            y = y_full[:,model.allElements.index(e)]
            plot_kwargs = _adjust_kwargs(e, {'label': e, 'color': f'C{i}'}, kwargs)
            ax.plot((mesh.z+zOffset)/zScale, y, *args, **plot_kwargs)
            i += 1

        # If the list of elements is small, we can add them to the y label
        if len(elements) <= 3:
            elList = ', '.join(elements)
            elementLabel = f'[{elList}] '
        else:
            elementLabel = ''
        ax.set_ylabel(f'Composition {elementLabel}(at.)') 
        ax.set_ylim([0,1])

    _set_1D_xlim(ax, mesh, zScale, zOffset)
    linesL, labelsL = axL.get_legend_handles_labels()
    linesR, labelsR = axR.get_legend_handles_labels()
    axL.legend(linesL+linesR, labelsL+labelsR, framealpha=1)
    return axL, axR

def plot1DPhases(model: DiffusionModel, phases=None, zScale=1, zOffset=0, time=None, ax=None, *args, **kwargs):
    '''
    Plots phase fractions over z

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    phases : list[str]
        Plots phases. If None, all phases in model are plotted
    zScale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. zScale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so zScale represents the desired unit/meters
        Defaults to 0
    zOffset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    ax : matplotlib Axes object
        Axis to plot on

    Returns
    -------
    matplotlib Axis
    '''
    ax = _get_axis(ax)
    mesh = _get_1D_mesh(model)
    
    # Compute phase fraction
    T = model.temperatureParameters(mesh.z, model.currentTime)
    # Temporary hash table, since we don't want to interfere with the internal model hash
    hashTable = HashTable()
    # Convert mesh.y (u-fraction) to composition to compute phase fractions
    x_full = model.getCompositions(time)
    mob_data = computeMobility(model.therm, x_full[:,1:], T, hashTable)
    phases = model.phases if phases is None else phases
    if isinstance(phases, str):
        phases = [phases]

    # plot phase fraction
    for p in phases:
        pf = []
        for p_labels, p_fracs in zip(mob_data.phases, mob_data.phase_fractions):
            pf.append(np.sum(p_fracs[p_labels==p]))
        plot_kwargs = _adjust_kwargs(p, {'label': p}, kwargs)
        ax.plot((mesh.z+zOffset)/zScale, pf, *args, **plot_kwargs)

    _set_1D_xlim(ax, mesh, zScale, zOffset)
    if len(phases) > 1:
        ax.legend()
    ax.set_ylim([0,1])
    ax.set_ylabel(f'Phase Fraction')
    return ax

def plot1DFlux(model: DiffusionModel, elements=None, zScale=1, zOffset=0, time=None, ax=None, *args, **kwargs):
    '''
    Plots flux of 1D mesh

    Parameters
    ----------
    model: Diffusion model
        Mesh in model must be Cartesian1D, Cylindrical1D, Spherical1D or any subclass of FiniteVolume1D
    elements: list[str]
        List of elements to plot (can include the dependent/reference element)
        If elements is None, then it will plot the independent elements in the diffusion model
    zScale: float (optional)
        Scaling for z-axis. Z will be divided by this, i.e. zScale = 1e-3 -> z is scaled from 1 to 1000
        Note: z is always in meters, so zScale represents the desired unit/meters
        Defaults to 0
    zOffset: float (optional)
        Offset in meters to shift z axis (positive value will increase all z values)
        Defaults to 0
    ax: matplotlib Axis (optional)
        Will be created if None

    Returns
    -------
    matplotlib Axis
    '''
    ax = _get_axis(ax)
    mesh = _get_1D_mesh(model)
    # make sure elements is a list[str], either from the diffusion model or from user input
    elements = model.elements if elements is None else elements
    if isinstance(elements, str):
        elements = [elements]

    fluxes = model.getFluxes(model.currentTime, [model.data.y(time)])
    # Sum of fluxes for substitutional elements = 0
    fluxes_sub = [fluxes[:,i] for i,e in enumerate(model.elements) if e not in interstitials]
    # If all dependent elements are interstitial, then flux of dependent element (substitutional) is 0
    if len(fluxes_sub) == 0:
        fluxes_sum = np.zeros((len(fluxes), 1))
    else:
        fluxes_sum = 0-np.sum(fluxes_sub,axis=0)[:,np.newaxis]
    fluxes_full = np.concatenate((fluxes_sum, fluxes), axis=1)
    for e in elements:
        y = fluxes_full[:,model.allElements.index(e)]
        plot_kwargs = _adjust_kwargs(e, {'label': e}, kwargs)
        ax.plot((mesh.zEdge+zOffset)/zScale, y, *args, **plot_kwargs)

    _set_1D_xlim(ax, mesh, zScale, zOffset)
    if len(elements) > 1:
        ax.legend()
    ax.set_ylabel(f'$J/V_m$ ($m/s$)')
    return ax


def plotMovingBoundaryState(
    model_or_mesh,
    composition=None,
    interface_position=None,
    interface_compositions=None,
    distance_multiplier=1.0,
    zScale=1,
    zOffset=0,
    time=None,
    ax=None,
    annotate=True,
    annotate_positions=False,
    annotate_interface_distances=False,
    annotate_interface_compositions=False,
    annotationWindow=2,
    maxAnnotatedCells=20,
    zoom_cells=None,
    auto_composition_zoom=True,
    composition_padding=0.08,
    *args,
    **kwargs,
):
    '''
    Plots the current moving-boundary mesh state on a single axis.

    The composition is shown on the usual y-axis, while a small geometry band
    below y=0 marks cell faces, cell centers and the interface position.
    This keeps the plot easy to call from a debugger while still showing how
    the interface sits relative to the discrete control volumes.

    Parameters
    ----------
    model_or_mesh:
        Either a MovingBoundary1DModel-like object, or a mesh-like object.
        For the mesh path, composition and interface_position may be passed
        explicitly. The interface position may also be inferred from
        mesh.interface_position / mesh.interfacePosition when available.
    '''
    ax = _get_axis(ax)
    left_interface_composition = None
    right_interface_composition = None
    pstar = kwargs.pop('pstar', None)
    if hasattr(model_or_mesh, 'getInterfacePosition') and hasattr(model_or_mesh, 'data'):
        mesh = _get_1D_mesh(model_or_mesh)
        composition = np.asarray(model_or_mesh.data.y(time), dtype=np.float64).reshape(-1)
        interface_position = float(model_or_mesh.getInterfacePosition(time))
        if pstar is None and hasattr(model_or_mesh, 'pstar'):
            pstar = float(model_or_mesh.pstar)
        if annotate_interface_compositions:
            state_time = model_or_mesh.currentTime if time is None else time
            try:
                _, _, left_interface_composition, right_interface_composition, _, _ = model_or_mesh._getInterfaceState(
                    state_time,
                    composition,
                    interface_position,
                )
            except Exception:
                left_interface_composition = None
                right_interface_composition = None
    else:
        mesh = model_or_mesh
        if isinstance(mesh, FiniteDifference1D):
            from kawin.diffusion.mesh.MovingBoundaryFD1D import _extract_moving_boundary_fd_inputs

            composition, interface_position, pstar = _extract_moving_boundary_fd_inputs(
                mesh,
                composition=composition,
                interface_position=interface_position,
                pstar=pstar,
            )
        else:
            from kawin.diffusion.mesh.MovingBoundary1D import _extract_moving_boundary_inputs

            composition, interface_position = _extract_moving_boundary_inputs(
                mesh,
                composition=composition,
                interface_position=interface_position,
            )
        if interface_compositions is not None:
            left_interface_composition = float(interface_compositions[0])
            right_interface_composition = float(interface_compositions[1])

    z = np.ravel(mesh.z).astype(np.float64)
    display_scale = float(distance_multiplier) / zScale
    x_centers = (z + zOffset) * display_scale
    x_interface = (interface_position + zOffset) * display_scale
    is_fdm = isinstance(mesh, FiniteDifference1D)

    if is_fdm:
        geometry = get_moving_boundary_fd_geometry(mesh, interface_position, pstar)
        x_edges = None
        x_left_face = x_interface
        x_right_face = x_interface
    else:
        geometry = get_moving_boundary_geometry(mesh, interface_position)
        z_edge = np.ravel(mesh.zEdge).astype(np.float64)
        x_edges = (z_edge + zOffset) * display_scale
        x_left_face = (geometry.left_face + zOffset) * display_scale
        x_right_face = (geometry.right_face + zOffset) * display_scale

    plot_kwargs = _adjust_kwargs(
        'moving_boundary',
        {'color': 'C0', 'marker': 'o', 'markersize': 4, 'linewidth': 1.5, 'label': 'Composition'},
        kwargs,
    )
    ax.plot(x_centers, composition, *args, **plot_kwargs)

    if zoom_cells is None:
        visible_comp = composition
    else:
        pad = max(0, int(zoom_cells))
        left_i = max(0, geometry.left_index - pad)
        right_i = min(len(z) - 1, geometry.right_index + pad)
        visible_comp = composition[left_i:right_i + 1]

    comp_min = float(np.min(visible_comp))
    comp_max = float(np.max(visible_comp))
    comp_span = comp_max - comp_min
    if auto_composition_zoom:
        if comp_span <= 0:
            y_data_pad = max(0.02, abs(comp_max) * 0.05, 1e-6)
        else:
            y_data_pad = max(float(composition_padding) * comp_span, 1e-6)
        ymax = comp_max + y_data_pad
    else:
        ymax = max(1.02, comp_max + 0.12)

    ymin = min(-0.18, comp_min - max(0.04, 0.5 * (ymax - comp_min)))
    ax.set_ylim([ymin, ymax])

    # Geometry strip below the composition profile.
    y_face_low = ymin + 0.02
    y_face_high = -0.015
    y_center = 0.5 * (y_face_low + y_face_high)
    y_center_highlight = y_center + 0.015

    if is_fdm:
        for center in x_centers:
            ax.plot([center, center], [y_face_low, y_face_high], color='0.88', linewidth=0.8, zorder=0)
        ax.axvspan(x_centers[geometry.left_index], x_interface, ymin=0, ymax=1, color='C1', alpha=0.08, zorder=0)
        ax.axvspan(x_interface, x_centers[geometry.right_index], ymin=0, ymax=1, color='C2', alpha=0.08, zorder=0)
    else:
        for edge in x_edges:
            ax.plot([edge, edge], [y_face_low, y_face_high], color='0.85', linewidth=0.8, zorder=0)
        ax.axvspan(x_left_face, x_interface, ymin=0, ymax=1, color='C1', alpha=0.08, zorder=0)
        ax.axvspan(x_interface, x_right_face, ymin=0, ymax=1, color='C2', alpha=0.08, zorder=0)

    ax.scatter(x_centers, np.full_like(x_centers, y_center), s=18, color='0.35', zorder=3, clip_on=False)
    ax.scatter(
        [x_centers[geometry.left_index], x_centers[geometry.right_index]],
        [y_center_highlight, y_center_highlight],
        s=28,
        color=['C1', 'C2'],
        zorder=4,
        clip_on=False,
    )
    ax.axvline(x_interface, color='C3', linestyle='--', linewidth=1.5, label='Interface')

    if annotate_interface_compositions and left_interface_composition is not None and right_interface_composition is not None:
        ax.scatter(
            [x_interface, x_interface],
            [left_interface_composition, right_interface_composition],
            s=42,
            color=['C1', 'C2'],
            edgecolors='black',
            linewidths=0.6,
            zorder=5,
            clip_on=False,
            label='Interface compositions',
        )
        ax.text(
            x_interface,
            left_interface_composition,
            f' c_int,L={left_interface_composition:.6g}',
            color='C1',
            ha='left',
            va='bottom',
            fontsize=8,
        )
        ax.text(
            x_interface,
            right_interface_composition,
            f' c_int,R={right_interface_composition:.6g}',
            color='C2',
            ha='left',
            va='top',
            fontsize=8,
        )

    if annotate_interface_distances:
        arrow_y = y_face_high + 0.12 * (ymax - ymin)
        text_y = arrow_y + 0.035 * (ymax - ymin)

        ax.annotate(
            '',
            xy=(x_interface, arrow_y),
            xytext=(x_centers[geometry.left_index], arrow_y),
            arrowprops={'arrowstyle': '<->', 'color': 'C1', 'linewidth': 1.2},
        )
        ax.annotate(
            '',
            xy=(x_centers[geometry.right_index], arrow_y),
            xytext=(x_interface, arrow_y),
            arrowprops={'arrowstyle': '<->', 'color': 'C2', 'linewidth': 1.2},
        )

        ax.text(
            0.5 * (x_centers[geometry.left_index] + x_interface),
            text_y,
            f'left_distance = {((interface_position - z[geometry.left_index]) * display_scale):.9g}',
            color='C1',
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=90,
        )
        ax.text(
            0.5 * (x_interface + x_centers[geometry.right_index]),
            text_y,
            f'right_distance = {((z[geometry.right_index] - interface_position) * display_scale):.9g}',
            color='C2',
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=90,
        )

    if annotate:
        if len(composition) <= maxAnnotatedCells:
            indices = np.arange(len(composition))
        else:
            start = max(0, geometry.left_index - int(annotationWindow))
            stop = min(len(composition), geometry.right_index + int(annotationWindow) + 1)
            indices = np.arange(start, stop)

        for i in indices:
                ax.text(
                x_centers[i],
                composition[i] + 0.02 * np.sign(x_centers[i] - x_interface),
                f'[{i}] {composition[i]:.7g}',
                ha='center',
                va='bottom',
                fontsize=8,
            )

        # ax.text(x_interface, ymax - 0.03 * (ymax - ymin), 'interface', color='C3', ha='center', va='top')
        if is_fdm:
            ax.text(x_centers[geometry.left_index], y_face_low, f'N{geometry.left_index}', color='C1', ha='right', va='top', fontsize=8)
            ax.text(x_centers[geometry.right_index], y_face_low, f'N{geometry.right_index}', color='C2', ha='left', va='top', fontsize=8)
            ax.text(
                x_centers[geometry.ignored_index],
                y_face_high,
                f'ignored={geometry.ignored_index}',
                color='C3',
                ha='center',
                va='bottom',
                fontsize=8,
            )
        else:
            ax.text(x_left_face, y_face_low, 'Lf', color='C1', ha='right', va='top', fontsize=8)
            ax.text(x_right_face, y_face_low, 'Rf', color='C2', ha='left', va='top', fontsize=8)
            ax.text(x_centers[geometry.left_index], y_face_high, f'C{geometry.left_index}', color='C1', ha='right', va='bottom', fontsize=8)
            ax.text(x_centers[geometry.right_index], y_face_high, f'C{geometry.right_index}', color='C2', ha='left', va='bottom', fontsize=8)

    if annotate_positions:
        if len(composition) <= maxAnnotatedCells:
            pos_indices = np.arange(len(composition))
        else:
            start = max(0, geometry.left_index - int(annotationWindow))
            stop = min(len(composition), geometry.right_index + int(annotationWindow) + 1)
            pos_indices = np.arange(start, stop)

        center_position_y = ymin + 0.72 * (y_center - ymin)
        for i in pos_indices:
            ax.text(
                x_centers[i],
                center_position_y,
                f'{x_centers[i]:.9g}',
                ha='center',
                va='top',
                fontsize=7,
                color='0.25',
                rotation=90,
            )

        if not is_fdm:
            edge_indices = np.unique(np.concatenate((pos_indices, [pos_indices[-1] + 1]))).astype(int)
            edge_indices = edge_indices[(edge_indices >= 0) & (edge_indices < len(x_edges))]
            for i in edge_indices:
                ax.text(
                    x_edges[i],
                    y_face_high,
                    f'{x_edges[i]:.9g}',
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    color='0.5',
                    rotation=90,
                )

        ax.text(
            x_interface, # - 0.015 * (x_edges[-1] - x_edges[0]),
            (y_face_high + 0.12 * (ymax - ymin)) + 0.035 * (ymax - ymin), #y_face_high + 0.05 * (ymax - ymin),
            f'{x_interface:.9g}',
            ha='right',
            va='bottom',
            fontsize=8,
            color='C3',
            rotation=90,
        )

    if zoom_cells is None:
        if is_fdm:
            ax.set_xlim([x_centers[0], x_centers[-1]])
            ax.set_xlabel(f'Distance*{display_scale:.0e} (m)')
        else:
            _set_1D_xlim(ax, mesh, zScale, zOffset)
    else:
        if is_fdm:
            ax.set_xlim([x_centers[left_i], x_centers[right_i]])
        else:
            ax.set_xlim([x_edges[left_i], x_edges[right_i + 1]])
    ax.set_xlabel(f'Distance*{display_scale:.0e} (m)')
    ax.set_ylabel('Composition / mesh state')
    ax.legend()
    return ax

def plot2D(model: DiffusionModel, element, zScale=1, time=None, ax=None, plotUFrac=False, *args, **kwargs):
    '''
    Plots a composition profile on a 2D mesh

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian2D
    element: str
        Element to plot
    zScale: float | list[float]
        Z axis scaling
        If float, will apply to both x and y axis
        If list, then first element applies to x and second element applies to y
    ax: matplotlib Axis
        Will be generated if None

    Returns
    -------
    matplotlib Axis (either same as input axis, or generated if no axis)
    matplotlib ScalarMappable (mappable to add a colorbar with)
    '''
    ax = _get_axis(ax)
    mesh: Cartesian2D = model.mesh
    if not isinstance(mesh, Cartesian2D):
        raise ValueError('Diffusion mesh must be Cartesian2D')
    
    # make sure zScale has 2 elements (for x and y axis)
    zScale = np.atleast_1d(zScale)
    if zScale.shape[0] == 1:
        zScale = zScale[0]*np.ones(2)

    y_flat = model.data.y(time)
    y_full = expand_u_frac(y_flat, model.allElements, interstitials)
    if not plotUFrac:
        y_full = u_to_x_frac(y_full, model.allElements, interstitials)
    # Reshape composition to mesh shape (Cartesian2D.unflattenResponse will give (Nx, Ny, 1))
    y = mesh.unflattenResponse(y_full[:,model.allElements.index(element)], 1)[...,0]

    # plot element
    # TODO: there should be a way to do contour and contourf (maybe separate plot functions?)
    plot_kwargs = _adjust_kwargs(element, {'vmin': 0, 'vmax': 1}, kwargs)
    cm = ax.pcolormesh(mesh.z[...,0]/zScale[0], mesh.z[...,1]/zScale[1], y, *args, **plot_kwargs)
    ax.set_title(element)
    ax.set_xlabel(f'Distance x*{zScale[0]:.0e} (m)')
    ax.set_ylabel(f'Distance y*{zScale[1]:.0e} (m)')
    return ax, cm

def plot2DPhases(model: DiffusionModel, phase, zScale=1, time=None, ax=None, *args, **kwargs):
    '''
    Plots a composition profile on a 2D mesh

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian2D
    phase: str
        phase to plot
    zScale: float | list[float]
        Z axis scaling
        If float, will apply to both x and y axis
        If list, then first element applies to x and second element applies to y
    ax: matplotlib Axis
        Will be generated if None

    Returns
    -------
    matplotlib Axis (either same as input axis, or generated if no axis)
    matplotlib ScalarMappable (mappable to add a colorbar with)
    '''
    ax = _get_axis(ax)
    mesh: Cartesian2D = model.mesh
    if not isinstance(mesh, Cartesian2D):
        raise ValueError('Diffusion mesh must be Cartesian2D')
    # make sure zScale has 2 elements (for x and y axis)
    zScale = np.atleast_1d(zScale)
    if zScale.shape[0] == 1:
        zScale = zScale[0]*np.ones(2)

    # We want z and y to be in [N,d] and [N,e] to be compatible with TemperatureParameters and computeMobility
    flatZ = mesh.flattenSpatial(mesh.z)
    fullX = model.getCompositions(time)
    T = model.temperatureParameters(flatZ, model.currentTime)
    # Temporary hash table, since we don't want to interfere with the internal model hash
    hashTable = HashTable()
    mob_data = computeMobility(model.therm, fullX[:,1:], T, hashTable)
    pf = []
    for p_labels, p_fracs in zip(mob_data.phases, mob_data.phase_fractions):
        pf.append(np.sum(p_fracs[p_labels==phase]))

    # Reshape phase fraction to mesh shape (Cartesian2D.unflattenResponse will give (Nx, Ny, 1))
    pf = mesh.unflattenResponse(np.array(pf), 1)[...,0]
    plot_kwargs = _adjust_kwargs(phase, {'vmin': 0, 'vmax': 1}, kwargs)
    cm = ax.pcolormesh(mesh.z[...,0]/zScale[0], mesh.z[...,1]/zScale[1], pf, *args, **plot_kwargs)
    ax.set_title(phase)
    ax.set_xlabel(f'Distance x*{zScale[0]:.0e} (m)')
    ax.set_ylabel(f'Distance y*{zScale[1]:.0e} (m)')
    return ax, cm

def plot2DFluxes(model: DiffusionModel, element, direction, zScale=1, time=None, ax=None, *args, **kwargs):
    '''
    Plots flux in x or y direction of element

    Parameters
    ----------
    model: DiffusionModel
        Mesh in model must be Cartesian2D
    element: str
        Element to plot
    direction: str
        'x' or 'y'
    zScale: float | list[float]
        Z axis scaling
        If float, will apply to both x and y axis
        If list, then first element applies to x and second element applies to y
    ax: matplotlib Axis
        Will be generated if None

    Returns
    -------
    matplotlib Axis (either same as input axis, or generated if no axis)
    matplotlib ScalarMappable (mappable to add a colorbar with)
    '''
    ax = _get_axis(ax)
    mesh: Cartesian2D = model.mesh
    if not isinstance(mesh, Cartesian2D):
        raise ValueError('Diffusion mesh must be Cartesian2D')
    # make sure zScale has 2 elements (for x and y axis)
    zScale = np.atleast_1d(zScale)
    if zScale.shape[0] == 1:
        zScale = zScale[0]*np.ones(2)

    if direction != 'x' and direction != 'y':
        raise ValueError("direction must be \'x\' or \'y\'")

    fluxes = model.getFluxes(model.currentTime, [model.data.y(time)])
    if direction == 'x':
        fluxes = fluxes[0]
        z = (mesh.zCorner[:,:-1] + mesh.zCorner[:,1:]) / 2
    elif direction == 'y':
        fluxes = fluxes[1]
        z = (mesh.zCorner[:-1,:] + mesh.zCorner[1:,:]) / 2

    flux_shape = fluxes.shape
    fluxes_flat = np.reshape(fluxes, (flux_shape[0]*flux_shape[1], flux_shape[2]))
    fluxes_sub = [fluxes_flat[:,i] for i,e in enumerate(model.elements) if e not in interstitials]
    # If all dependent elements are interstitial, then flux of dependent element (substitutional) is 0
    if len(fluxes_sub) == 0:
        fluxes_sum = np.zeros((len(fluxes_flat), 1))
    else:
        fluxes_sum = 0-np.sum(fluxes_sub, axis=0)[:,np.newaxis]
    fluxes_flat = np.concatenate((fluxes_sum, fluxes_flat), axis=1)
    y = fluxes_flat[:,model.allElements.index(element)]
    y = np.reshape(y, (flux_shape[0], flux_shape[1]))
    cm = ax.pcolormesh(z[...,0]/zScale[0], z[...,1]/zScale[1], y, *args, **kwargs)
    ax.set_title(f'$J_x/V_m$ {element} ($m/s$)')
    ax.set_xlabel(f'Distance x*{zScale[0]:.0e} (m)')
    ax.set_ylabel(f'Distance y*{zScale[1]:.0e} (m)')
    return ax, cm


