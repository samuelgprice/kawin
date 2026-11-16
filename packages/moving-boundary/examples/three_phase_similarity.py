# %% [markdown]
# # How do two independent interfaces move as a middle layer grows?
#
# A layer B grows between semi-infinite phases A and C. Different coupled matrices, far fields
# and two tie-line families give each interface its own parabolic rate, so neither mirrors the
# other. An exact similarity solution, following the constant-diffusivity multiphase treatment
# of Kirkaldy [1], exists for two cases:
#
# - `opposite`: the layer grows into both neighbours (`k_ab < 0 < k_bc`);
# - `drift`: both interfaces move right (`0 < k_ab < k_bc`).
#
# **Assumptions.** Planar, constant matrices, equal constant molar volumes, no bulk flow and
# semi-infinite outer phases. Units are nondimensional; elements are `[Z, X, Y]` with Z
# dependent, phases run `A | B | C` left to right, and compositions are mole fractions. A
# closure maps `eta -> (c_left, c_right)` at its interface.
#
# **Two things to note.** More than one similarity root can exist, as Maugis et al. showed for
# two-phase ternary couples [2]: `drift` also has a thin
# layer moving left, so this picks the root with the largest `k_ab`. And the model starts from
# the exact state at `T0 > 0`, because a zero-width start is singular; model time `t`
# corresponds to similarity time `T0 + t`.
#
# **Things to try.** Change node counts, `semi_log_dt`, the far fields or the closures
# (`tie_line_crossing_etas` flags closures that make the problem ill-posed). Errors fall about
# first-order with node count, so 21 nodes is only a quick check. A thin-phase stop would end
# a run; it is not a phase-disappearance transition.
#
# This checks numerical accuracy under those assumptions, not real materials.
#
# **Running.** Run the cells in order in a Jupyter or VS Code Interactive window, editing the
# Settings cell first. From a terminal:
# `python three_phase_similarity.py [--fast] [--case drift] [--plot]` (about 5 s).
"""How do two independent interfaces move as a middle layer grows? See the notes above."""

# %%
from pathlib import Path
import sys

# Run notebook cells with the extraction repository root as the working directory.
sys.path.append(str(Path("packages/moving-boundary/examples").resolve()))

from advanced import three_phase_similarity as _advanced
from dataclasses import dataclass
import argparse

import numpy as np

from moving_boundary.analytic import (
    FixedMatrixTernaryDiffusivity,
    SimilarityInitialProfile,
    compare_three_phase_model_to_similarity,
    recorded_history,
    semi_infinite_validity_time,
    solve_three_phase_similarity_roots,
)
from moving_boundary.equilibrium import CallableTernaryInterfaceEquilibrium
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder
from moving_boundary.mesh.TransformedGrids import landau_grid
from moving_boundary.run_diagnostics import inventory_drift
from moving_boundary.solvers import MovingBoundaryIllingworthTernaryThreePhaseFD1DModel

PHASES = ["A", "B", "C"]
DIFFUSIVITIES = {
    "A": np.array([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]]),
    "B": np.array([[3.0e-4, -0.5e-4], [1.0e-4, 5.0e-4]]),
    "C": np.array([[6.0e-4, 1.0e-4], [-0.5e-4, 2.0e-3]]),
}
FAR_FIELDS = {  # (far field of A, far field of C), as [X, Y]
    "opposite": (np.array([0.28, 0.10]), np.array([0.26, 0.12])),
    "drift": (np.array([0.38, 0.03]), np.array([0.26, 0.12])),
}
S0 = 0.3 + 1.0e-12  # off the physical mesh nodes
DOMAIN_LENGTH = 0.6
T0 = 0.01
T_FINAL = 1.0


def linear_closure(left_0, left_1, right_0, right_1):
    """A tie-line family linear in ``eta`` between two end tie lines."""
    left_0, left_1, right_0, right_1 = (np.asarray(v, dtype=np.float64) for v in (left_0, left_1, right_0, right_1))
    return CallableTernaryInterfaceEquilibrium(
        lambda eta: (left_0 + eta * (left_1 - left_0), right_0 + eta * (right_1 - right_0)), eta_bounds=(0.0, 1.0))


CLOSURE_AB = linear_closure([0.30, 0.03], [0.34, 0.07], [0.18, 0.10], [0.26, 0.16])
CLOSURE_BC = linear_closure([0.20, 0.12], [0.24, 0.18], [0.36, 0.04], [0.39, 0.08])


# %% [markdown]
# ## Configurations
#
# `t_end` is an absolute similarity time.

# %%
@dataclass(frozen=True)
class Config:
    """Case and numerical settings; ``t_end`` is an absolute similarity time."""

    case: str = "opposite"
    nodes: int = 81
    semi_log_dt: float = 0.025
    mesh_nodes: int = 6001
    t_end: float = T_FINAL

    @property
    def duration(self):
        return self.t_end - T0


@dataclass(frozen=True)
class ExampleResult:
    """A solved example: its settings, model, exact solution and named metrics."""

    config: Config
    model: object
    exact: object
    metrics: dict


FAST = Config(nodes=21, semi_log_dt=0.05, mesh_nodes=401, t_end=0.05)  # quick check settings
FAST_DRIFT = Config(case="drift", nodes=21, semi_log_dt=0.05, mesh_nodes=401, t_end=0.05)
COARSE_LONG = Config(nodes=21, semi_log_dt=0.05, mesh_nodes=401)
COARSE_LONG_DRIFT = Config(case="drift", nodes=21, semi_log_dt=0.05, mesh_nodes=401)
DEMO = Config()
DEMO_DRIFT = Config(case="drift")


# Configuration-specific checks are documented in the advanced companion.
DEMO_LIMITS = _advanced.DEMO_LIMITS

# %% [markdown]
# ## Settings
#
# Pick a configuration (`DEMO`, `DEMO_DRIFT`, `FAST`, `FAST_DRIFT`, or your own
# `Config(...)`), and whether to plot.

# %%
CONFIG = DEMO
PLOT = True


# %% [markdown]
# ## Exact solution and model

# %%
def exact_solution(config=DEMO):
    """The similarity root the model should follow: the one with the largest ``k_ab``."""
    roots = solve_three_phase_similarity_roots(
        DIFFUSIVITIES["A"], DIFFUSIVITIES["B"], DIFFUSIVITIES["C"], *FAR_FIELDS[config.case], CLOSURE_AB, CLOSURE_BC, s0=S0)
    return roots[-1]


def build_model(config=DEMO, exact=None):
    """Build the solver from the exact state at ``T0``; nothing is solved yet."""
    exact = exact_solution(config) if exact is None else exact
    mesh = CartesianFD1D(["X", "Y"], [0.0, DOMAIN_LENGTH], config.mesh_nodes)
    mesh.setResponseProfile(ProfileBuilder([(SimilarityInitialProfile(exact, T0), ["X", "Y"])]))
    grids = (landau_grid(config.nodes, "linear", phase="left", spacing_ratio=0.1),
             landau_grid(config.nodes, "linear", phase="middle", spacing_ratio=0.2),
             landau_grid(config.nodes, "linear", phase="right", spacing_ratio=0.1))
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=PHASES,
        thermodynamics=FixedMatrixTernaryDiffusivity(DIFFUSIVITIES, PHASES),
        temperature=1000.0,
        interfacePositions=tuple(exact.interface_positions(T0)),
        interface_equilibria=(CLOSURE_AB, CLOSURE_BC),
        initial_eta_guess=tuple(exact.etas),
        time_step=T_FINAL - T0,
        dt_mode="semi_log",
        semiLog_dt=config.semi_log_dt,
        semiLogT0=1.0e-3 * T0,
        transformed_grids=grids,
        tolerance=1.0e-11,
        max_iterations=50,
        terminal_thin_phase_policy="raise",
        record=True,
        record_pq_data=True,
    )


# %% [markdown]
# ## Short workflow
#
# Solve once with basic checks. Use the original `run` below for full validation.

# %%
def run_core(config=DEMO):
    """Solve once and measure final inventory drift; detailed accuracy checks use ``run``."""
    exact = exact_solution(config)
    model = build_model(config, exact)
    model.solve(config.duration, minDtFrac=1e-14)
    absolute, relative = inventory_drift(
        np.asarray([model.getTotalInventory(0.0), model.getTotalInventory()]), reduction="final")
    return ExampleResult(config, model, exact, {"final_time": float(model.currentTime),
                                               "inventory_abs": absolute, "inventory_rel": relative})


# %% [markdown]
# ## Evaluation and plots

# %%


def run(config=DEMO):
    """Build, solve and evaluate one configuration."""
    exact = exact_solution(config)
    model = build_model(config, exact)
    model.solve(config.duration, minDtFrac=1e-14)
    return ExampleResult(config, model, exact, evaluate(model, config, exact))


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
        DEMO_DRIFT=DEMO_DRIFT,
        DEMO_LIMITS=DEMO_LIMITS,
        DIFFUSIVITIES=DIFFUSIVITIES,
        DOMAIN_LENGTH=DOMAIN_LENGTH,
        FAR_FIELDS=FAR_FIELDS,
        FAST=FAST,
        FAST_DRIFT=FAST_DRIFT,
        PHASES=PHASES,
        T0=T0,
        T_FINAL=T_FINAL,
        description=__doc__,
    )


def evaluate(model, config, exact):
    """Return per-interface accuracy metrics; a short run has too few points in the window. See the advanced companion."""
    return _advanced.evaluate(model, config, exact, case=_advanced_case())


def report(result):
    """Print the final state and the metrics, with limits where they apply. See the advanced companion."""
    return _advanced.report(result, case=_advanced_case())


def plot(result):
    """Plot both interfaces against the exact solution, and the final profiles; returns the figure. See the advanced companion."""
    return _advanced.plot(result, case=_advanced_case())


def settings_from_command_line(config, plot_figures):
    """Apply ``--fast``, ``--case`` and ``--plot`` from a terminal; inside a notebook kernel, change nothing. See the advanced companion."""
    return _advanced.settings_from_command_line(config, plot_figures, case=_advanced_case())


# %% [markdown]
# ## Run
#
# For the introductory path use `result = run_core(CONFIG)` after the Settings
# and model cells. The compatibility entry point below also runs the detailed
# checks in `advanced/three_phase_similarity.py`.

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
# The numbers here are synthetic; the form of the exact solution, and the possibility of more
# than one root, come from the literature.
#
# 1. J. S. Kirkaldy, "Diffusion in multicomponent metallic systems: III. The motion of planar
#    phase interfaces," Can. J. Phys. 36 (1958) 917-925, doi:10.1139/p58-098. Unidimensional
#    diffusion in multiphase, multicomponent systems with constant diffusivities.
# 2. P. Maugis, W. D. Hopfe, J. E. Morral, J. S. Kirkaldy, "Multiple interface velocity
#    solutions for ternary biphase infinite diffusion couples," Acta Mater. 45 (1997)
#    1941-1954. More than one interface velocity can satisfy the same two-phase couple.
