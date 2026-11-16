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
from pathlib import Path
import sys

# Run notebook cells with the extraction repository root as the working directory.
sys.path.append(str(Path("packages/moving-boundary/examples").resolve()))

from advanced import binary_planar_growth as _advanced
from dataclasses import dataclass
import argparse

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
from moving_boundary.providers.constant import ConstantDiffusivity
from moving_boundary.run_diagnostics import inventory_drift
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


# Configuration-specific checks are documented in the advanced companion.
DEMO_LIMITS = _advanced.DEMO_LIMITS

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
        max_iterations=100,
        record=1,
        record_pq_data=False,
        preallocate_recordings=True,
    )


# %% [markdown]
# ## Short workflow
#
# Solve once with basic checks. Use the original `run` below for full validation.

# %%
def run_core(config=DEMO):
    """Solve once and measure final inventory drift; detailed accuracy checks use ``run``."""
    model = build_model(config)
    model.solve(config.duration, minDtFrac=1e-14)
    absolute, relative = inventory_drift(
        np.asarray([model.getTotalInventory(0.0), model.getTotalInventory()]), reduction="final")
    return ExampleResult(config, model, planar_exact_interface, {"final_time": float(model.currentTime),
                                               "inventory_abs": absolute, "inventory_rel": relative})


def solve_model(model, config):
    """Advance to ``t_end``; ``minDtFrac`` lets the final partial step complete."""
    model.solve(config.duration, minDtFrac=1e-14)
    return model


def run(config=DEMO):
    """Build, solve and evaluate one configuration."""
    model = solve_model(build_model(config), config)
    return ExampleResult(config, model, planar_exact_interface, evaluate(model, config))


# %% [markdown]
# ## Advanced checks and compatibility entry points
#
# The introductory path above needs only `run_core`. These wrappers retain
# the original detailed workflow and command-line behavior.

# %%
def _advanced_case():
    """Supply current case inputs to the optional detailed workflow. See the advanced companion."""
    return _advanced.Case(
        DEMO=DEMO,
        DEMO_LIMITS=DEMO_LIMITS,
        FAST=FAST,
        description=__doc__,
    )


def evaluate(model, config):
    """Return accuracy metrics against the exact solution, defined at ``DEMO_LIMITS``. See the advanced companion."""
    return _advanced.evaluate(model, config, case=_advanced_case())


def report(result):
    """Print the metrics, with limits where they apply. See the advanced companion."""
    return _advanced.report(result, case=_advanced_case())


def plot(result):
    """Plot the interface against the exact solution, and the final profile; returns the figure. See the advanced companion."""
    return _advanced.plot(result, case=_advanced_case())


def settings_from_command_line(config, plot_figures):
    """Apply ``--fast`` and ``--plot`` from a terminal; inside a notebook kernel, change nothing. See the advanced companion."""
    return _advanced.settings_from_command_line(config, plot_figures, case=_advanced_case())


# %% [markdown]
# ## Run
#
# For the introductory path use `result = run_core(CONFIG)` after the Settings
# and model cells. The compatibility entry point below also runs the detailed
# checks in `advanced/binary_planar_growth.py`.

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
