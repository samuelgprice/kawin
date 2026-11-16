"""Optional Matplotlib views of recorded interfaces and physical mole fractions."""

import numpy as np

from .analytic import recorded_history


def _axes(ax):
    """Create axes lazily so importing this module needs no plotting backend."""
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    return ax


def plot_interface_history(model, *, ax=None, time_offset=0.0, time_scale=1.0,
                           position_scale=1.0, position_offset=0.0, **plot_kwargs):
    """Plot all recorded interfaces and return axes without showing a window.

    Coordinates are ``(time + time_offset) * time_scale`` and
    ``(position - position_offset) * position_scale``. Offsets use solver
    units, allowing absolute similarity times and interface displacements.
    Matplotlib keyword arguments control line appearance and labels.
    """
    ax = _axes(ax)
    times, positions = recorded_history(model.interfaceData)
    ax.plot((times + time_offset) * time_scale,
            (np.asarray(positions).reshape(len(times), -1) - position_offset) * position_scale,
            **plot_kwargs)
    ax.set(xlabel="time", ylabel="interface position")
    return ax


def plot_composition_profiles(model, *, ax=None, time=None, elements=None,
                              position_scale=1.0, **plot_kwargs):
    """Plot selected elements' physical mole fractions and return axes.

    ``elements`` defaults to the independent components in model order.
    Ternary and three-phase profiles retain both interface endpoints and convert
    interstitial u states through the model's public physical-profile API.
    Binary profiles use the public physical mesh history (whose interface is
    mesh-resolved). Historical ternary profiles need recorded transformed data.
    No exact-solution overlay or implicit display is added.
    """
    ax = _axes(ax)
    names = list(model.allElements)
    selected = names[1:] if elements is None else list(elements)
    if not selected or any(name not in names for name in selected):
        raise ValueError(f"elements must select from {names}.")
    if hasattr(model, "getPhysicalPhaseProfiles"):
        profiles = model.getPhysicalPhaseProfiles(time)
        labels = list(model.phases)
    else:
        solute = np.asarray(model.data.currentY if time is None else model.data.y(time)).reshape(-1, 1)
        profiles = [(np.ravel(model.mesh.z), np.column_stack((1.0 - solute[:, 0], solute[:, 0])))]
        labels = ["model"]
    for phase, (positions, fractions) in zip(labels, profiles):
        for element in selected:
            options = {"label": f"x({element}), {phase}", **plot_kwargs}
            ax.plot(np.asarray(positions) * position_scale, fractions[:, names.index(element)], **options)
    ax.set(xlabel="position", ylabel="mole fraction")
    return ax
