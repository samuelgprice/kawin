"""Advanced checks for ``binary_planar_growth``; see that example for assumptions and references.

The Case supplies explicit physical inputs and callbacks. No main script
is imported, and no solve or file operation runs during import.
"""

from dataclasses import dataclass
from moving_boundary.plotting import plot_interface_history
import argparse
import sys
import numpy as np
from moving_boundary.analytic import VALIDATION_PARAMS, planar_exact_interface, planar_exact_profile, recorded_history


# Numerical accuracy limits for DEMO only: 41/41 nodes, dt=1e-5, t from 1e-4 to 1e-2.
# interface_rel: max|s - s_exact| / max|s_exact - s(t_start)| over the whole run.
# profile_linf_away: max|c - c_exact| on the mesh, excluding 3 cells around the interface.
# inventory_abs: |final - initial| solute inventory, in composition x length.
DEMO_LIMITS = {"interface_rel": 1.0e-2, "profile_linf_away": 5.0e-3, "inventory_abs": 1.0e-8}


@dataclass(frozen=True)
class Case:
    """Inputs and callbacks supplied by the main example at call time."""

    DEMO: object
    DEMO_LIMITS: object
    FAST: object
    description: str


def evaluate(model, config, *, case):
    """Return accuracy metrics against the exact solution, defined at ``DEMO_LIMITS``."""
    times, interface = recorded_history(model.interfaceData)
    exact = planar_exact_interface(times + config.t_start)
    z = np.ravel(model.mesh.z)
    error = np.ravel(model.data.currentY) - np.ravel(planar_exact_profile(z, model.currentTime + config.t_start))
    away = np.abs(z - float(model.getInterfacePosition())) > 3.0 * (z[1] - z[0])
    initial_inventory = float(np.ravel(recorded_history(model.concData)[1])[0]) * VALIDATION_PARAMS["R"]
    # A screening estimate of when BETA stops looking semi-infinite, as in
    # semi_infinite_validity_time: a wall-tail amplitude, not an error bound.
    wall_distance = VALIDATION_PARAMS["R"] - float(exact[-1])
    return {
        "interface_rel": float(np.max(np.abs(np.ravel(interface) - exact)) / np.max(np.abs(exact - exact[0]))),
        "profile_linf_away": float(np.max(np.abs(error[away]))),
        "inventory_abs": abs(float(np.ravel(model.getTotalInventory())[0]) - initial_inventory),
        "validity_time_estimate": (wall_distance / 4.0) ** 2 / VALIDATION_PARAMS["D_b"],
        "steps": int(model.interfaceData.N),
    }

def report(result, *, case):
    """Print the metrics, with limits where they apply."""
    limits = case.DEMO_LIMITS if result.config == case.DEMO else {}
    if not limits:
        print("These settings have no accuracy limits; DEMO is the configuration to judge accuracy by.")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.4g}" + (f"  (limit {limits[name]:g})" if name in limits else ""))

def plot(result, *, case):
    """Plot the interface against the exact solution, and the final profile; returns the figure."""
    import matplotlib.pyplot as plt

    times, interface = recorded_history(result.model.interfaceData)
    absolute = times + result.config.t_start
    z = np.ravel(result.model.mesh.z)
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    plot_interface_history(result.model, ax=left, time_offset=result.config.t_start, label="model")
    left.plot(absolute, planar_exact_interface(absolute), "--", label="exact")
    left.set(xlabel="time (nondimensional)", ylabel="interface position", title="Interface")
    left.legend()
    right.plot(z, np.ravel(result.model.data.currentY), label="model")
    right.plot(z, planar_exact_profile(z, result.model.currentTime + result.config.t_start), "--", label="exact")
    right.set(xlabel="position", ylabel="x(CR)", title="Final profile")
    right.legend()
    figure.tight_layout()
    return figure

def settings_from_command_line(config, plot_figures, *, case):
    """Apply ``--fast`` and ``--plot`` from a terminal; inside a notebook kernel, change nothing.

    A Jupyter kernel's ``sys.argv`` holds the kernel's own arguments, so they are not parsed.
    """
    if "ipykernel" in sys.modules:
        return config, plot_figures
    parser = argparse.ArgumentParser(description=case.description.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--fast", action="store_true", help="use quick check settings")
    parser.add_argument("--plot", action="store_true", help="show plots")
    args = parser.parse_args()
    return (case.FAST if args.fast else config), args.plot
