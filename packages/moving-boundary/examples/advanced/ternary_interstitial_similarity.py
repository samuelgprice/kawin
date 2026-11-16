"""Advanced checks for ``ternary_interstitial_similarity``; see that example for assumptions and references.

The Case supplies explicit physical inputs and callbacks. No main script
is imported, and no solve or file operation runs during import.
"""

from dataclasses import dataclass
from moving_boundary.plotting import plot_interface_history
import argparse
import sys
import numpy as np
from moving_boundary.analytic import compare_model_to_similarity, recorded_history, semi_infinite_validity_time


# Numerical accuracy limits for DEMO only: 81/81 nodes, width 0.8 + 0.4, t to 1, window t >= 0.1.
# k_rel: |k - k_exact| / |k_exact|. interface_rel: max|s - s_exact| / max|s_exact - s0|.
# eta_abs: max|eta - eta_exact|. carbon_profile_linf: max|u_C - u_C,exact| over five recorded
# times. inventory_abs: largest component |final - initial| in u x length.
DEMO_LIMITS = {"k_rel": 1.0e-2, "interface_rel": 1.0e-2, "eta_abs": 1.0e-3,
               "carbon_profile_linf": 1.0e-3, "inventory_abs": 5.0e-9 * 1.2}


@dataclass(frozen=True)
class Case:
    """Inputs and callbacks supplied by the main example at call time."""

    DEMO: object
    DEMO_LIMITS: object
    FAST: object
    description: str
    diffusivities: object


def evaluate(model, config, exact, *, case):
    """Return accuracy metrics in the u basis, or only validity estimates for a short run."""
    d = case.diffusivities(config.ratio)
    final_s = float(model.getInterfacePosition())
    metrics = {
        "validity_time_alpha": semi_infinite_validity_time(d["ALPHA"], config.s0),
        "validity_time_beta": semi_infinite_validity_time(d["BETA"], config.left_width + config.right_width - final_s),
    }
    if config.duration <= config.compare_from:
        return metrics
    comparison = compare_model_to_similarity(model, exact, t_min=config.compare_from)
    metrics.update(
        k_rel=comparison["k_rel_error"],
        interface_rel=comparison["interface_rel_error"],
        eta_abs=comparison["eta_max_abs_error"],
        carbon_profile_linf=float(np.max(comparison["profile_linf"][:, 0])),
        manganese_profile_linf=float(np.max(comparison["profile_linf"][:, 1])),
        inventory_abs=float(np.max(np.abs(comparison["inventory_drift"]))),
    )
    return metrics

def report(result, *, case):
    """Print the final state and the metrics, with limits where they apply."""
    print(f"final interface {result.model.getInterfacePosition():.6f}, eta {result.model.getInterfaceEta():.6f}; "
          f"exact k {result.exact.k:.6f}, eta {result.exact.eta:.6f}")
    limits = case.DEMO_LIMITS if result.config == case.DEMO else {}
    if not limits:
        print("These settings have no accuracy limits; DEMO is the configuration to judge accuracy by.")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.4g}" + (f"  (limit {limits[name]:g})" if name in limits else ""))

def plot(result, *, case):
    """Plot the interface and the final carbon profile in u and x against the exact u; returns the figure."""
    import matplotlib.pyplot as plt

    model, exact = result.model, result.exact
    times, interface = recorded_history(model.interfaceData)
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    plot_interface_history(model, ax=left, position_offset=result.config.s0, label="model")
    left.plot(times, exact.interface_position(times) - result.config.s0, "--", label="exact")
    left.set(xlabel="time", ylabel="s - s0", title="Interface displacement")
    left.legend()
    time = float(times[-1])
    (x_left, c_left), (x_right, c_right) = model.getPhysicalPhaseProfiles(time)
    u_left, u_right = model.getTransformedState(time)
    for x, u, c, side, label in ((x_left, u_left, c_left, "left", "ALPHA"), (x_right, u_right, c_right, "right", "BETA")):
        right.plot(x, u[:, 0], label=f"u_C, model ({label})")
        right.plot(x, c[:, 1], ":", label=f"x_C, model ({label})")
        right.plot(x, exact.profile(x, time, phase=side)[:, 0], "--", label=f"u_C, exact ({label})")
    right.set(xlabel="position", ylabel="carbon", title=f"Carbon at t = {time:.3g}")
    right.legend(fontsize="small")
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
