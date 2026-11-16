"""Advanced checks for ``binary_tlp_bonding``; see that example for assumptions and references.

The Case supplies explicit physical inputs and callbacks. No main script
is imported, and no solve or file operation runs during import.
"""

from dataclasses import dataclass
import argparse
import sys
import numpy as np
from moving_boundary.analytic import recorded_history


# Limits for DEMO only (see the notes for each check). early_interface_rel:
# max|s - s_exact| / (s_exact(t_valid) - s0) for early_from <= t <= liquid_validity_time;
# DEMO gives 1.1e-3. peak_over_bound: maximum half-width / (s0 c0 / c_liquid), an exact
# limit; DEMO gives 0.9941. inventory_rel: |final - initial| / initial phosphorus inventory;
# DEMO gives about 2e-12 with the default interface tolerances (about 2e-9 before LEDGERS R08).
DEMO_LIMITS = {"early_interface_rel": 3.0e-3, "peak_over_bound": 1.0, "inventory_rel": 2.0e-8}


@dataclass(frozen=True)
class Case:
    """Inputs and callbacks supplied by the main example at call time."""

    DEMO: object
    DEMO_LIMITS: object
    FAST: object
    PARAMS: object
    description: str
    exact_interface: object
    grid_summary: object
    grids: object
    liquid_validity_time: object
    profiles_at: object
    similarity_growth_constant: object
    width_bound: object


def evaluate(model, config, p, *, case):
    """Return the exact-check metrics (defined at ``DEMO_LIMITS``) and descriptive values."""
    times, width = recorded_history(model.interfaceData)
    width = np.ravel(width)
    k = case.similarity_growth_constant(p)
    t_valid = case.liquid_validity_time(p)
    window = (times >= config.early_from) & (times <= t_valid)
    exact = case.exact_interface(times[window], p, k)
    _, inventory = recorded_history(model.concData)
    inventory = np.ravel(inventory)
    peak = int(np.argmax(width))
    return {
        "growth_constant_k": k,
        "liquid_validity_time": t_valid,
        "early_interface_rel": float(np.max(np.abs(width[window] - exact)) / (case.exact_interface(t_valid, p, k) - p["s0"])),
        "peak_width": float(width[peak]),
        "peak_time": float(times[peak]),
        "width_bound": case.width_bound(p),
        "peak_over_bound": float(width[peak] / case.width_bound(p)),
        "inventory_rel": float(abs(inventory[-1] - inventory[0]) / abs(inventory[0])),
        "final_time": float(times[-1]),
        "final_width": float(width[-1]),
        "steps": int(len(times) - 1),
    }

def grid_summary(config, p, *, case):
    """Interface spacing of each phase at the start, in um, beside what it must resolve.

    The solid's diffusion length ``sqrt(4 D t)`` at ``early_from`` is the scale its interface
    interval has to resolve early on; ``solid_uniform_interval`` is the spacing a uniform grid
    with the same node count would have. Intervals scale with each phase's width as the
    interface moves.
    """
    u_grid, v_grid = case.grids(config)
    return {
        "liquid_interface_interval": p["s0"] * (u_grid[-1] - u_grid[-2]),
        "solid_interface_interval": (p["R"] - p["s0"]) * (v_grid[1] - v_grid[0]),
        "solid_far_interval": (p["R"] - p["s0"]) * (v_grid[-1] - v_grid[-2]),
        "solid_uniform_interval": (p["R"] - p["s0"]) / (config.solid_nodes - 1),
        "solid_diffusion_length_early": np.sqrt(4.0 * p["D_solid"] * config.early_from),
    }

def report(result, *, case):
    """Print the metrics, with limits where they apply, then the grid spacing near the interface."""
    limits = case.DEMO_LIMITS if result.config == case.DEMO else {}
    if not limits:
        print("These settings have no limits; DEMO is the configuration to judge them by.")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.6g}" + (f"  (limit {limits[name]:g})" if name in limits else ""))
    grid = case.grid_summary(result.config)
    print(f"grid at the start: solid interface interval {grid['solid_interface_interval']:.3g} um "
          f"(far end {grid['solid_far_interval']:.3g} um; uniform would be {grid['solid_uniform_interval']:.3g} um) "
          f"against a solid diffusion length of {grid['solid_diffusion_length_early']:.3g} um at "
          f"t = {result.config.early_from:g} s; liquid interface interval {grid['liquid_interface_interval']:.3g} um")

def profiles_at(model, config, time, p, *, case):
    """Physical ``(x, c)`` for each phase at a recorded ``time``, from the transformed state."""
    u_grid, v_grid = case.grids(config)
    s = float(model.getInterfacePosition(time))
    liquid, solid = model.getTransformedState(time)
    return (s * u_grid, np.ravel(liquid)), (s + (p["R"] - s) * v_grid, np.ravel(solid))

def plot(result, *, case):
    """Plot four panels and return the figure: the half-width history with the exact checks; the
    early half-width against sqrt(t), where the exact solution is a straight line; and liquid and
    solid profiles at a few times on separate scales (the solid holds under 0.2 at.% P)."""
    import matplotlib.pyplot as plt

    model, config, metrics = result.model, result.config, result.metrics
    times, width = recorded_history(model.interfaceData)
    figure, ((left, parabolic), (middle, right)) = plt.subplots(2, 2, figsize=(12, 8.4))
    t_valid = metrics["liquid_validity_time"]
    t_max = min(50.0 * t_valid, times[-1])
    early_times = times <= t_max
    root_t = np.sqrt(times[early_times])
    # Steps are log-spaced in time, so pick markers evenly spaced in sqrt(t) instead.
    picked = np.unique(np.searchsorted(root_t, np.linspace(0.0, root_t[-1], 40)).clip(0, len(root_t) - 1))
    parabolic.plot(root_t[picked], np.ravel(width)[early_times][picked], "o", ms=4, label="model")
    line_root_t = np.linspace(0.0, np.sqrt(t_max), 2)
    parabolic.plot(line_root_t, case.exact_interface(line_root_t**2), "--", color="C1",
                   label=f"exact: s = s0 + {metrics['growth_constant_k']:.2f} sqrt(t)")
    parabolic.axvspan(np.sqrt(config.early_from), np.sqrt(t_valid), color="C1", alpha=0.12, label="early check window")
    parabolic.axhline(metrics["width_bound"], color="0.4", ls=":")
    parabolic.set(xlabel="sqrt(time) (s^1/2)", ylabel="liquid half-width (um)", xlim=(0.0, np.sqrt(t_max)),
                  title="Early widening against sqrt(t)")
    parabolic.legend(fontsize=8, loc="lower right")
    shown = times > 0
    left.semilogx(times[shown], np.ravel(width)[shown], label="model")
    early = np.geomspace(config.semi_log_t0, min(10.0 * metrics["liquid_validity_time"], times[-1]), 200)
    left.semilogx(early, case.exact_interface(early), "--", label="exact, both phases semi-infinite")
    left.axvspan(config.early_from, metrics["liquid_validity_time"], color="C1", alpha=0.12, label="early check window")
    left.axhline(metrics["width_bound"], color="0.4", ls=":", label=f"bound s0 c0 / c_liquid = {metrics['width_bound']:.2f} um")
    left.set(xlabel="time (s)", ylabel="liquid half-width (um)", title="Liquid layer", ylim=(0, 1.15 * metrics["width_bound"]))
    left.legend(fontsize=8, loc="lower left")
    targets = [t for t in (1.0e-2, 1.0, 1.0e2, 1.0e4, times[-1]) if t <= times[-1]]
    for target in targets:
        time = float(times[np.argmin(np.abs(times - target))])
        (x_l, c_l), (x_s, c_s) = case.profiles_at(model, config, time)
        # Markers are the solver's nodes, which move with the interface (Landau transformation).
        line, = middle.plot(x_l, 100.0 * c_l, ".-", ms=3, lw=1, label=f"t = {time:.3g} s")
        right.plot(x_s - x_s[0], 100.0 * c_s, ".-", ms=3, lw=1, color=line.get_color())
    middle.set(xlabel="position from joint centre (um)", ylabel="P (at.%)", title="Liquid (markers are nodes)", xlim=(0.0, None))
    middle.legend(fontsize=8)
    right.set_xscale("symlog", linthresh=1.0)
    right.set(xlabel="distance into the solid from the interface (um, linear below 1)", ylabel="P (at.%)",
              title="Solid nickel (markers are nodes)", xlim=(0.0, case.PARAMS["R"]))
    figure.tight_layout()
    return figure

def settings_from_command_line(config, plot_figures, *, case):
    """Apply ``--fast`` and ``--plot`` from a terminal; inside a notebook kernel, change nothing.

    A Jupyter kernel's ``sys.argv`` holds the kernel's own arguments, so they are not parsed.
    """
    if "ipykernel" in sys.modules:
        return config, plot_figures
    parser = argparse.ArgumentParser(description=case.description.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--fast", action="store_true", help="only the early window and the peak")
    parser.add_argument("--plot", action="store_true", help="show plots")
    args = parser.parse_args()
    return (case.FAST if args.fast else config), args.plot
