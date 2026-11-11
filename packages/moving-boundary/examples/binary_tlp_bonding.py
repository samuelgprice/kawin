# %% [markdown]
# # How does a liquid interlayer widen and then solidify when bonding nickel?
#
# Transient liquid phase bonding: a Ni-19 at.% P interlayer between nickel plates melts at the
# bonding temperature. The liquid first dissolves nickel and widens; phosphorus then diffuses
# into the solid nickel, and the liquid solidifies isothermally. This is the Ni-P case of Zhou
# and North [2], as reproduced in Fig. 3 of Illingworth and Golosnoy [1], whose
# solute-conserving algorithm this solver implements. The inputs are from [1] (p. 213), and
# match Table 1 of [2]:
# interface compositions 10.223 at.% P (liquid) and 0.166 at.% P (solid), an
# initial liquid half-width of 12.5 um at 19 at.% P, phosphorus-free nickel, D = 500 um^2/s in
# the liquid and 18 um^2/s in the solid, and a half-domain of 3012.5 um.
#
# **What can be checked exactly.** There is no exact solution for the whole process, but there
# are three exact checks:
#
# 1. *Early times.* While neither diffusion field has reached a boundary, the step start is the
#    two-sided similarity problem [3]: `s = s0 + k sqrt(t)`, with `k` from the interface flux
#    balance (`similarity_growth_constant`). The liquid field reaches the symmetry plane after
#    about 0.014 s (`liquid_validity_time`), so this window covers roughly the first 15% of
#    the widening. The solid stays effectively semi-infinite for about 10^4 s. The check
#    starts at `early_from` = 1e-3 s, after the start-up transient of the singular step start.
# 2. *A width bound.* The liquid can never be wider than if it held all the phosphorus at its
#    interface composition, `s0 c0 / c_liquid` = 23.23 um ([1], p. 213). That paper's point is that
#    non-conservative schemes overshoot this bound; a conservative one stays just below it,
#    because the solid takes up a little phosphorus during the widening.
# 3. *Conservation* of the total phosphorus. The scheme conserves it to rounding; what drift
#    remains comes from the interface iteration tolerance.
#
# The long solidification stage (hours) has no exact solution; compare it qualitatively with
# Fig. 3 of [1] ("present work"), which peaks just below the bound within a few seconds
# and is nearly solidified by 10^5 s. With `DEMO` the half-width peaks at about 23.10 um
# after about 2.7 s and is about 2.5 um at 7x10^4 s. A convergence study (linear and power
# grids, space and time refined) gives a converged peak of 23.110 um, so DEMO's peak is about
# 0.06% low. That error is almost all spatial and falls at first order with the node count.
#
# **Assumptions.** Planar, constant diffusivity in each phase, equal molar volumes in liquid
# and solid, local equilibrium at fixed interface compositions, and zero flux at both ends. The
# left end is the symmetry plane at the centre of the joint, so this is half of the joint.
# Elements are `[NI, P]` with P the solute; LIQUID lies left of FCC_A1. Compositions are mole
# fractions; lengths are in um and times in s (any consistent units work). The temperature is
# Zhou and North's bonding temperature, 1150 C [2] ([1] does not state it); the
# API requires one, but the constant diffusivities here ignore it.
#
# **Limitations.** The solver cannot remove a vanishing phase, so runs stop before the liquid
# is gone. These settings are chosen for speed and are not converged (see the peak above).
# Finer and more strongly graded grids run to the full `duration` (161/641 nodes with
# `semi_log_dt` = 0.01 used to stall at about 6.4x10^4 s, before the solver solved each
# phase relative to its interface composition).
#
# **Things to try.** Make `solid_spacing_ratio` 1.0 (a uniform solid grid): the peak rises
# about 0.05 um, overshooting the converged value, because a coarse solid grid under-resolves
# the phosphorus the solid takes up early on. The report's grid line compares the interface
# spacing with the solid's diffusion length, and the plot markers show where the nodes are.
# Double the node counts: the early-time error halves (first order in space), at little cost,
# since run time follows the number of time steps rather than nodes. Halving `semi_log_dt`
# barely changes it. Or change a diffusivity in `PARAMS`; the exact checks follow.
#
# **Running.** Run the cells in order in a Jupyter or VS Code Interactive window, editing the
# Settings cell first. From a terminal: `python binary_tlp_bonding.py [--fast] [--plot]`.
# `DEMO` takes about 2 s and `FAST` about 1 s, measured on one machine.
"""How does a liquid interlayer widen and then solidify when bonding nickel? See the notes above."""

# %%
from dataclasses import dataclass
import argparse
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfcinv, erfcx

from kawin.diffusion.DiffusionParameters import TemperatureParameters
from moving_boundary.analytic import recorded_history
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from moving_boundary.mesh.TransformedGrids import two_phase_landau_grids
from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel

# Illingworth and Golosnoy [1], p. 213. Compositions are mole fractions of P.
PARAMS = {
    "c_liquid0": 0.19,         # initial liquid
    "c_solid0": 0.0,           # initial solid nickel
    "c_liquid": 0.10223,       # liquid at the interface
    "c_solid": 0.00166,        # solid at the interface
    "D_liquid": 500.0,         # um^2/s
    "D_solid": 18.0,           # um^2/s
    "s0": 12.5,                # initial liquid half-width, um
    "R": 3012.5,               # half-domain, um
}
TEMPERATURE = 1423.15  # K: 1150 C, Zhou and North's bonding temperature [2]; unused by constant diffusivities


# %% [markdown]
# ## Configurations

# %%
@dataclass(frozen=True)
class Config:
    """Numerical settings. ``*_spacing_ratio`` is the interface interval relative to uniform
    spacing (1.0 is uniform). The early-time check uses ``early_from`` up to the liquid
    validity time; earlier times hold the start-up transient of the singular step start."""

    liquid_nodes: int = 81
    solid_nodes: int = 321
    liquid_spacing_ratio: float = 0.2
    solid_spacing_ratio: float = 0.01
    semi_log_dt: float = 0.005
    semi_log_t0: float = 1.0e-6
    duration: float = 7.0e4
    mesh_nodes: int = 3013
    early_from: float = 1.0e-3


@dataclass(frozen=True)
class ExampleResult:
    """A solved example: its settings, model, exact early-time solution and named metrics."""

    config: Config
    model: object
    exact: object
    metrics: dict


DEMO = Config()                      # widening and most of the solidification
FAST = Config(duration=3.0)          # the early window and the peak only

# Limits for DEMO only (see the notes for each check). early_interface_rel:
# max|s - s_exact| / (s_exact(t_valid) - s0) for early_from <= t <= liquid_validity_time;
# DEMO gives 1.1e-3. peak_over_bound: maximum half-width / (s0 c0 / c_liquid), an exact
# limit; DEMO gives 0.9941. inventory_rel: |final - initial| / initial phosphorus inventory;
# DEMO gives about 2e-12 with the default interface tolerances (about 2e-9 before LEDGERS R08).
DEMO_LIMITS = {"early_interface_rel": 3.0e-3, "peak_over_bound": 1.0, "inventory_rel": 2.0e-8}

# %% [markdown]
# ## Settings
#
# Pick a configuration (`DEMO`, `FAST` or your own `Config(...)`), and whether to plot.

# %%
CONFIG = DEMO
PLOT = True


# %% [markdown]
# ## Exact checks

# %%
def similarity_growth_constant(p=PARAMS):
    """``k`` in ``s = s0 + k sqrt(t)`` for a step start with both phases semi-infinite [3].

    Solves the interface balance ``(cL - cS) k/2 = J_liquid + J_solid`` with erfc profiles
    in each phase. ``erfcx`` keeps the ``exp/erfc`` ratios finite for any ``k``.
    """
    root_dl, root_ds = np.sqrt(p["D_liquid"]), np.sqrt(p["D_solid"])

    def balance(k):
        liquid = (p["c_liquid0"] - p["c_liquid"]) * np.sqrt(p["D_liquid"] / np.pi) / erfcx(-k / (2.0 * root_dl))
        solid = (p["c_solid0"] - p["c_solid"]) * np.sqrt(p["D_solid"] / np.pi) / erfcx(k / (2.0 * root_ds))
        return liquid + solid - (p["c_liquid"] - p["c_solid"]) * k / 2.0

    scale = 20.0 * max(root_dl, root_ds)
    return brentq(balance, -scale, scale, xtol=1.0e-14)


def exact_interface(t, p=PARAMS, k=None):
    """Early-time exact half-width ``s0 + k sqrt(t)``; valid only while both phases look semi-infinite."""
    k = similarity_growth_constant(p) if k is None else k
    return p["s0"] + k * np.sqrt(np.asarray(t, dtype=np.float64))


def liquid_validity_time(p=PARAMS, tail=1.0e-3):
    """Time at which the liquid's erfc tail at the symmetry plane reaches ``tail``.

    A screening estimate using the initial distance ``s0`` (the interface moves away, so
    it is slightly conservative), not an error bound.
    """
    return (p["s0"] / (2.0 * erfcinv(tail))) ** 2 / p["D_liquid"]


def width_bound(p=PARAMS):
    """Largest possible liquid half-width: all phosphorus in liquid at its interface composition ([1], p. 213)."""
    return (p["s0"] * p["c_liquid0"] + (p["R"] - p["s0"]) * p["c_solid0"]) / p["c_liquid"]


# %% [markdown]
# ## Model, solve and evaluation

# %%
class ConstantDiffusivity:
    """Supplies one constant diffusivity per phase, in place of a thermodynamic database."""

    def __init__(self, diffusivities):
        self.diffusivities = dict(diffusivities)

    def clearCache(self):
        """No cache is retained."""

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        """Return a scalar or temperature-shaped array for the requested phase."""
        return np.squeeze(np.ones(np.atleast_1d(T).astype(np.float64).shape) * self.diffusivities[phase])


def grids(config):
    """Transformed grids, graded toward the interface. The solid needs strong grading: its
    diffusion length grows from well under 1 um to about 1 mm over the run."""
    return two_phase_landau_grids(config.liquid_nodes, config.solid_nodes, method="linear",
                                  left_kwargs={"spacing_ratio": config.liquid_spacing_ratio},
                                  right_kwargs={"spacing_ratio": config.solid_spacing_ratio})


def build_model(config=DEMO, p=PARAMS):
    """Build the solver from the step start; nothing is solved yet."""
    mesh = CartesianFD1D(["P"], [0.0, p["R"]], config.mesh_nodes)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(p["s0"], p["c_liquid0"], p["c_solid0"]), "P")]))
    u_grid, v_grid = grids(config)
    return MovingBoundaryIllingworthFD1DModel(
        mesh, ["NI", "P"], ["LIQUID", "FCC_A1"],
        thermodynamics=ConstantDiffusivity({"LIQUID": p["D_liquid"], "FCC_A1": p["D_solid"]}),
        temperature=TemperatureParameters(TEMPERATURE),
        interfacePosition=p["s0"],
        interface_compositions=(p["c_liquid"], p["c_solid"]),
        time_step=1.0, dt_mode="semi_log", semiLog_dt=config.semi_log_dt, semiLogT0=config.semi_log_t0,
        phase_a_nodes=config.liquid_nodes, phase_b_nodes=config.solid_nodes,
        transformed_u_grid=u_grid, transformed_v_grid=v_grid,
        # Default interface tolerances (dimensionless position 1e-12, residual 1e-13); see the notes above.
        max_iterations=200, record=True, record_pq_data=True,
    )


def evaluate(model, config, p=PARAMS):
    """Return the exact-check metrics (defined at ``DEMO_LIMITS``) and descriptive values."""
    times, width = recorded_history(model.interfaceData)
    width = np.ravel(width)
    k = similarity_growth_constant(p)
    t_valid = liquid_validity_time(p)
    window = (times >= config.early_from) & (times <= t_valid)
    exact = exact_interface(times[window], p, k)
    _, inventory = recorded_history(model.concData)
    inventory = np.ravel(inventory)
    peak = int(np.argmax(width))
    return {
        "growth_constant_k": k,
        "liquid_validity_time": t_valid,
        "early_interface_rel": float(np.max(np.abs(width[window] - exact)) / (exact_interface(t_valid, p, k) - p["s0"])),
        "peak_width": float(width[peak]),
        "peak_time": float(times[peak]),
        "width_bound": width_bound(p),
        "peak_over_bound": float(width[peak] / width_bound(p)),
        "inventory_rel": float(abs(inventory[-1] - inventory[0]) / abs(inventory[0])),
        "final_time": float(times[-1]),
        "final_width": float(width[-1]),
        "steps": int(len(times) - 1),
    }


def run(config=DEMO):
    """Build, solve and evaluate one configuration."""
    model = build_model(config)
    model.solve(config.duration, minDtFrac=1e-14)
    return ExampleResult(config, model, exact_interface, evaluate(model, config))


def grid_summary(config, p=PARAMS):
    """Interface spacing of each phase at the start, in um, beside what it must resolve.

    The solid's diffusion length ``sqrt(4 D t)`` at ``early_from`` is the scale its interface
    interval has to resolve early on; ``solid_uniform_interval`` is the spacing a uniform grid
    with the same node count would have. Intervals scale with each phase's width as the
    interface moves.
    """
    u_grid, v_grid = grids(config)
    return {
        "liquid_interface_interval": p["s0"] * (u_grid[-1] - u_grid[-2]),
        "solid_interface_interval": (p["R"] - p["s0"]) * (v_grid[1] - v_grid[0]),
        "solid_far_interval": (p["R"] - p["s0"]) * (v_grid[-1] - v_grid[-2]),
        "solid_uniform_interval": (p["R"] - p["s0"]) / (config.solid_nodes - 1),
        "solid_diffusion_length_early": np.sqrt(4.0 * p["D_solid"] * config.early_from),
    }


def report(result):
    """Print the metrics, with limits where they apply, then the grid spacing near the interface."""
    limits = DEMO_LIMITS if result.config == DEMO else {}
    if not limits:
        print("These settings have no limits; DEMO is the configuration to judge them by.")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.6g}" + (f"  (limit {limits[name]:g})" if name in limits else ""))
    grid = grid_summary(result.config)
    print(f"grid at the start: solid interface interval {grid['solid_interface_interval']:.3g} um "
          f"(far end {grid['solid_far_interval']:.3g} um; uniform would be {grid['solid_uniform_interval']:.3g} um) "
          f"against a solid diffusion length of {grid['solid_diffusion_length_early']:.3g} um at "
          f"t = {result.config.early_from:g} s; liquid interface interval {grid['liquid_interface_interval']:.3g} um")


def profiles_at(model, config, time, p=PARAMS):
    """Physical ``(x, c)`` for each phase at a recorded ``time``, from the transformed state."""
    u_grid, v_grid = grids(config)
    s = float(model.getInterfacePosition(time))
    liquid, solid = model.getTransformedState(time)
    return (s * u_grid, np.ravel(liquid)), (s + (p["R"] - s) * v_grid, np.ravel(solid))


def plot(result):
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
    parabolic.plot(line_root_t, exact_interface(line_root_t**2), "--", color="C1",
                   label=f"exact: s = s0 + {metrics['growth_constant_k']:.2f} sqrt(t)")
    parabolic.axvspan(np.sqrt(config.early_from), np.sqrt(t_valid), color="C1", alpha=0.12, label="early check window")
    parabolic.axhline(metrics["width_bound"], color="0.4", ls=":")
    parabolic.set(xlabel="sqrt(time) (s^1/2)", ylabel="liquid half-width (um)", xlim=(0.0, np.sqrt(t_max)),
                  title="Early widening against sqrt(t)")
    parabolic.legend(fontsize=8, loc="lower right")
    shown = times > 0
    left.semilogx(times[shown], np.ravel(width)[shown], label="model")
    early = np.geomspace(config.semi_log_t0, min(10.0 * metrics["liquid_validity_time"], times[-1]), 200)
    left.semilogx(early, exact_interface(early), "--", label="exact, both phases semi-infinite")
    left.axvspan(config.early_from, metrics["liquid_validity_time"], color="C1", alpha=0.12, label="early check window")
    left.axhline(metrics["width_bound"], color="0.4", ls=":", label=f"bound s0 c0 / c_liquid = {metrics['width_bound']:.2f} um")
    left.set(xlabel="time (s)", ylabel="liquid half-width (um)", title="Liquid layer", ylim=(0, 1.15 * metrics["width_bound"]))
    left.legend(fontsize=8, loc="lower left")
    targets = [t for t in (1.0e-2, 1.0, 1.0e2, 1.0e4, times[-1]) if t <= times[-1]]
    for target in targets:
        time = float(times[np.argmin(np.abs(times - target))])
        (x_l, c_l), (x_s, c_s) = profiles_at(model, config, time)
        # Markers are the solver's nodes, which move with the interface (Landau transformation).
        line, = middle.plot(x_l, 100.0 * c_l, ".-", ms=3, lw=1, label=f"t = {time:.3g} s")
        right.plot(x_s - x_s[0], 100.0 * c_s, ".-", ms=3, lw=1, color=line.get_color())
    middle.set(xlabel="position from joint centre (um)", ylabel="P (at.%)", title="Liquid (markers are nodes)", xlim=(0.0, None))
    middle.legend(fontsize=8)
    right.set_xscale("symlog", linthresh=1.0)
    right.set(xlabel="distance into the solid from the interface (um, linear below 1)", ylabel="P (at.%)",
              title="Solid nickel (markers are nodes)", xlim=(0.0, PARAMS["R"]))
    figure.tight_layout()
    return figure


def settings_from_command_line(config, plot_figures):
    """Apply ``--fast`` and ``--plot`` from a terminal; inside a notebook kernel, change nothing.

    A Jupyter kernel's ``sys.argv`` holds the kernel's own arguments, so they are not parsed.
    """
    if "ipykernel" in sys.modules:
        return config, plot_figures
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--fast", action="store_true", help="only the early window and the peak")
    parser.add_argument("--plot", action="store_true", help="show plots")
    args = parser.parse_args()
    return (FAST if args.fast else config), args.plot


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    CONFIG, PLOT = settings_from_command_line(CONFIG, PLOT)
    result = run(CONFIG)
    report(result)

# %%
if __name__ == "__main__" and PLOT:
    import matplotlib.pyplot as plt

    plot(result)
    plt.show()

# %% [markdown]
# ## References
#
# 1. T. C. Illingworth, I. O. Golosnoy, "Numerical solutions of diffusion-controlled moving
#    boundary problems which conserve solute," J. Comput. Phys. 209 (2005) 207-225,
#    doi:10.1016/j.jcp.2005.02.031. All inputs (p. 213), the width bound (p. 213) and Fig. 3;
#    the solver implements this paper's algorithm.
# 2. Y. Zhou, T. H. North, "Kinetic modelling of diffusion-controlled, two-phase moving
#    interface problems," Modell. Simul. Mater. Sci. Eng. 1 (1993) 505-516. The original Ni-P
#    bonding case: TLP bonding of nickel with Ni-19 at.% P filler at 1150 C, and its input
#    parameters (Table 1).
# 3. J. S. Kirkaldy, "Diffusion in multicomponent metallic systems: III. The motion of planar
#    phase interfaces," Can. J. Phys. 36 (1958) 917-925, doi:10.1139/p58-098. Similarity
#    solutions for planar interfaces with constant diffusivities; the early-time check is the
#    binary, two-phase case.
