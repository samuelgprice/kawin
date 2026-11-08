# %% [markdown]
# # How fast does a planar interface grow under diffusion control?
#
# Grows phase ALPHA into BETA with fixed interface compositions and compares the result
# with the exact Zener solution [1], `s(t) = 2 j sqrt(D t)` (Eqs. (14)-(15) of [2]). This is
# the place to start: it shows meshes, phases, transformed grids and solving, with no
# thermodynamic database. The compositions, domain, diffusivity and start time
# (`VALIDATION_PARAMS`) are the analytical validation case of Illingworth and Golosnoy [2]
# (p. 219), the case behind their Fig. 4(a).
#
# **Assumptions.** Planar, constant equal diffusivities, equal molar volumes and a
# semi-infinite BETA. ALPHA stays flat at its interface composition. Quantities are
# nondimensional (domain length 1, D = 1); the temperature is required by the API but has
# no effect here. Elements are `[NI, CR]` with CR the solute, and ALPHA lies to the left;
# the labels carry no thermodynamics. Compositions are mole fractions.
#
# **Things to try.** Change the node counts, `time_step` or the compositions in
# `VALIDATION_PARAMS`. The exact solution starts from `t_start > 0` because a zero-width
# start is singular. Coarse uniform grids resolve the early profile poorly: 21 nodes in
# BETA gives a far larger interface error than 41.
#
# This checks numerical accuracy for constant diffusivity. It says nothing about real
# thermodynamics; see `ternary_fecrni_lee1996.py` for that.
#
# **Running.** Run the cells in order in a Jupyter or VS Code Interactive window, editing
# the Settings cell first. From a terminal: `python binary_planar_growth.py [--fast] [--plot]`
# (under a second).
"""How fast does a planar interface grow under diffusion control? See the notes above."""

# %%
from dataclasses import dataclass
import argparse
import sys

import numpy as np

from kawin.diffusion.DiffusionParameters import TemperatureParameters
from moving_boundary.analytic import (
    PlanarExactProfile,
    VALIDATION_PARAMS,
    planar_exact_interface,
    planar_exact_profile,
    recorded_history,
)
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder
from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel


# %% [markdown]
# ## Configurations
#
# Times are absolute similarity times, so `t_start` is the exact state the run starts from.

# %%
@dataclass(frozen=True)
class Config:
    """Numerical settings; times are absolute, so ``t_start`` is the exact start state."""

    nodes: int = 41
    time_step: float = 1.0e-5
    t_start: float = 1.0e-4
    t_end: float = 1.0e-2
    mesh_nodes: int = 401

    @property
    def duration(self):
        return self.t_end - self.t_start


@dataclass(frozen=True)
class ExampleResult:
    """A solved example: its settings, model, exact solution (if any) and named metrics."""

    config: Config
    model: object
    exact: object
    metrics: dict


FAST = Config(nodes=21, time_step=2.0e-5, t_end=2.0e-4)      # quick check settings
COARSE_LONG = Config(nodes=21, time_step=2.0e-5)             # shows a coarse grid's error
DEMO = Config()

# Numerical accuracy limits for DEMO only: 41/41 nodes, dt=1e-5, t from 1e-4 to 1e-2.
# interface_rel: max|s - s_exact| / max|s_exact - s(t_start)| over the whole run.
# profile_linf_away: max|c - c_exact| on the mesh, excluding 3 cells around the interface.
# inventory_abs: |final - initial| solute inventory, in composition x length.
DEMO_LIMITS = {"interface_rel": 1.0e-2, "profile_linf_away": 5.0e-3, "inventory_abs": 1.0e-8}

# %% [markdown]
# ## Settings
#
# Pick a configuration (`DEMO`, `FAST` or `COARSE_LONG`, or your own `Config(...)`), and
# whether to plot.

# %%
CONFIG = DEMO
PLOT = True


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


def build_model(config=DEMO):
    """Build the solver from the exact state at ``config.t_start``; nothing is solved yet."""
    p = VALIDATION_PARAMS  # the validation case of [2], p. 219
    mesh = CartesianFD1D(["CR"], [0.0, p["R"]], config.mesh_nodes)
    mesh.setResponseProfile(ProfileBuilder([(PlanarExactProfile(config.t_start), "CR")]))
    return MovingBoundaryIllingworthFD1DModel(
        mesh, ["NI", "CR"], ["ALPHA", "BETA"],
        thermodynamics=ConstantDiffusivity({"ALPHA": p["D_a"], "BETA": p["D_b"]}),
        temperature=TemperatureParameters(p["temperature"]),
        interfacePosition=float(planar_exact_interface(config.t_start)),
        interface_compositions=(p["c_a"], p["c_b"]),
        time_step=config.time_step,
        phase_a_nodes=config.nodes,
        phase_b_nodes=config.nodes,
        tolerance=p["tolerance"],
        max_iterations=100,
        record=1,
        record_pq_data=False,
        preallocate_recordings=True,
    )


def solve_model(model, config):
    """Advance to ``t_end``; ``minDtFrac`` lets the final partial step complete."""
    model.solve(config.duration, minDtFrac=1e-14)
    return model


def evaluate(model, config):
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


def run(config=DEMO):
    """Build, solve and evaluate one configuration."""
    model = solve_model(build_model(config), config)
    return ExampleResult(config, model, planar_exact_interface, evaluate(model, config))


def report(result):
    """Print the metrics, with limits where they apply."""
    limits = DEMO_LIMITS if result.config == DEMO else {}
    if not limits:
        print("These settings have no accuracy limits; DEMO is the configuration to judge accuracy by.")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.4g}" + (f"  (limit {limits[name]:g})" if name in limits else ""))


def plot(result):
    """Plot the interface against the exact solution, and the final profile; returns the figure."""
    import matplotlib.pyplot as plt

    times, interface = recorded_history(result.model.interfaceData)
    absolute = times + result.config.t_start
    z = np.ravel(result.model.mesh.z)
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(absolute, np.ravel(interface), label="model")
    left.plot(absolute, planar_exact_interface(absolute), "--", label="exact")
    left.set(xlabel="time (nondimensional)", ylabel="interface position", title="Interface")
    left.legend()
    right.plot(z, np.ravel(result.model.data.currentY), label="model")
    right.plot(z, planar_exact_profile(z, result.model.currentTime + result.config.t_start), "--", label="exact")
    right.set(xlabel="position", ylabel="x(CR)", title="Final profile")
    right.legend()
    figure.tight_layout()
    return figure


def settings_from_command_line(config, plot_figures):
    """Apply ``--fast`` and ``--plot`` from a terminal; inside a notebook kernel, change nothing.

    A Jupyter kernel's ``sys.argv`` holds the kernel's own arguments, so they are not parsed.
    """
    if "ipykernel" in sys.modules:
        return config, plot_figures
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--fast", action="store_true", help="use quick check settings")
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
# 1. C. Zener, "Theory of growth of spherical precipitates from solid solution," J. Appl.
#    Phys. 20 (1949) 950-953. The exact growth solution.
# 2. T. C. Illingworth, I. O. Golosnoy, "Numerical solutions of diffusion-controlled moving
#    boundary problems which conserve solute," J. Comput. Phys. 209 (2005) 207-225,
#    doi:10.1016/j.jcp.2005.02.031. The validation case (p. 219), Eqs. (14)-(15) and Fig. 4(a);
#    the solver implements this paper's algorithm.
