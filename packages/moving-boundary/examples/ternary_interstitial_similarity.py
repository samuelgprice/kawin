# %% [markdown]
# # How does a fast interstitial and a slow substitutional solute move an interface?
#
# A Fe-C-Mn-like couple: carbon diffuses about 100 times faster than manganese. With constant
# matrices and a linear tie-line family the problem has an exact similarity solution [1, 2], so this
# checks numerics without any thermodynamic data. The numbers are synthetic and are not real
# Fe-C-Mn properties.
#
# **Composition basis is the main lesson.** Elements are `[FE, C, MN]` with FE dependent and
# interstitials listed after it. The solver's state is in u-fractions,
# `u_k = x_k / sum(substitutional x)` [3, Eq. (9)]. So here:
#
# - the diffusivity matrices, closure and exact solution are in u;
# - the closure declares `composition_basis="u"` (a closure in x would be converted);
# - the initial profile is given in mole fractions;
# - `getPhysicalPhaseProfiles` and `getCompositions` return mole fractions, while
#   `getTransformedState` returns the u state.
#
# Compare with the exact solution in u, as `compare_model_to_similarity` does. The plot shows
# why: the mole-fraction view of carbon differs from u by about u²/(1+u).
#
# **Assumptions.** Planar, constant matrices, constant volume per substitutional atom [3], no bulk
# flow, effectively semi-infinite phases. Units are nondimensional. ALPHA lies left and BETA
# right; the interface moves right, with parabolic `s = s0 + k sqrt(t)` and a constant tie
# line `eta`.
#
# **Things to try.** Change `ratio` (a smaller value needs a stronger graded grid), the spacing
# ratios or the far fields. Point-wise Mn profile errors are large because the Mn spike is
# narrower than the remaining interface offset, so only carbon has a profile limit.
# `terminal_thin_phase_policy` must be explicit for ternary models.
#
# This checks numerical accuracy under those assumptions, not Fe-C-Mn physics.
#
# **Running.** Run the cells in order in a Jupyter or VS Code Interactive window, editing the
# Settings cell first. From a terminal: `python ternary_interstitial_similarity.py [--fast] [--plot]`
# (about a second).
"""How do a fast interstitial and a slow substitutional solute move an interface? See the notes above."""

# %%
from pathlib import Path
import sys

# Run notebook cells with the extraction repository root as the working directory.
sys.path.append(str(Path("packages/moving-boundary/examples").resolve()))

from advanced import ternary_interstitial_similarity as _advanced
from dataclasses import dataclass
import argparse

import numpy as np

from moving_boundary.analytic import (
    FixedMatrixTernaryDiffusivity,
    compare_model_to_similarity,
    recorded_history,
    semi_infinite_validity_time,
    solve_similarity_roots,
)
from moving_boundary.equilibrium import CallableTernaryInterfaceEquilibrium
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from moving_boundary.mesh.TransformedGrids import two_phase_landau_grids
from moving_boundary.run_diagnostics import inventory_drift
from moving_boundary.solvers import MovingBoundaryIllingworthTernaryFD1DModel

ELEMENTS = ["FE", "C", "MN"]
PHASES = ["ALPHA", "BETA"]
U_LEFT_FAR = np.array([0.002, 0.015])   # far-field [u_C, u_MN] in ALPHA
U_RIGHT_FAR = np.array([0.010, 0.015])  # far-field [u_C, u_MN] in BETA


# %% [markdown]
# ## Configurations

# %%
@dataclass(frozen=True)
class Config:
    """Domain and numerical settings; ``ratio`` is the Mn/C eigenvalue ratio."""

    ratio: float = 1.0e-2
    left_width: float = 0.8
    right_width: float = 0.4
    left_nodes: int = 81
    right_nodes: int = 81
    left_spacing_ratio: float = 0.05
    right_spacing_ratio: float = 0.1
    semi_log_dt: float = 0.05
    mesh_nodes: int = 1198
    duration: float = 1.0
    compare_from: float = 0.1

    @property
    def s0(self):
        """Initial interface, nudged off the physical mesh nodes."""
        return self.left_width + 1.0e-12


@dataclass(frozen=True)
class ExampleResult:
    """A solved example: its settings, model, exact solution and named metrics."""

    config: Config
    model: object
    exact: object
    metrics: dict


# Quick check settings: short, and too brief for the t >= 0.1 accuracy window.
FAST = Config(left_width=0.2, right_width=0.2, left_nodes=21, right_nodes=21, left_spacing_ratio=0.1,
              mesh_nodes=400, duration=0.05)
# A wider ALPHA than BETA, because carbon's fast mode in ALPHA reaches farther.
DEMO = Config()


# Configuration-specific checks are documented in the advanced companion.
DEMO_LIMITS = _advanced.DEMO_LIMITS

# %% [markdown]
# ## Settings
#
# Pick a configuration (`DEMO` or `FAST`, or your own `Config(...)`), and whether to plot.

# %%
CONFIG = DEMO
PLOT = True


# %% [markdown]
# ## Problem definition and model

# %%
def closure():
    """An NPLE-like linear tie-line family in u-fractions: ``eta -> (u_ALPHA, u_BETA)``."""

    def tie_line(eta):
        return np.array([0.002, 0.010 + 0.02 * eta]), np.array([0.030 - 0.01 * eta, 0.020 + 0.03 * eta])

    return CallableTernaryInterfaceEquilibrium(tie_line, eta_bounds=(0.0, 1.0), composition_basis="u")


def diffusivities(ratio):
    """u-basis (C, Mn) matrices: fast carbon, slow Mn, weak C-Mn coupling."""
    return {"ALPHA": 1.0e-2 * np.array([[1.0, -0.02], [0.0, ratio]]),
            "BETA": 1.0e-3 * np.array([[1.0, -0.05], [0.0, ratio]])}


def x_from_u(u):
    """Convert ``[u_C, u_MN]`` to mole fractions, as the initial profile requires."""
    u = np.asarray(u, dtype=np.float64)
    return u / (1.0 + u[..., :1])


def transformed_grids(config):
    """Interface-clustered grids; ``spacing_ratio`` sets the interval next to the interface."""
    return two_phase_landau_grids(
        config.left_nodes, config.right_nodes, method="linear",
        left_kwargs={"spacing_ratio": config.left_spacing_ratio},
        right_kwargs={"spacing_ratio": config.right_spacing_ratio},
    )


def build_model(config=DEMO, *, grids=None):
    """Build the solver for a step in composition at ``config.s0``; nothing is solved yet.

    ``grids`` optionally supplies explicit ``(u_grid, v_grid)`` transformed grids.
    """
    mesh = CartesianFD1D(["C", "MN"], [0.0, config.left_width + config.right_width], config.mesh_nodes)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(config.s0, x_from_u(U_LEFT_FAR), x_from_u(U_RIGHT_FAR)), ["C", "MN"])]))
    u_grid, v_grid = transformed_grids(config) if grids is None else grids
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=ELEMENTS,
        phases=PHASES,
        thermodynamics=FixedMatrixTernaryDiffusivity(diffusivities(config.ratio), PHASES),
        temperature=1000.0,
        interfacePosition=config.s0,
        interface_equilibrium=closure(),
        initial_eta_bracket=(0.0, 1.0),
        bulk_diffusivity_mode="phase_uniform",
        time_step=1.0,
        dt_mode="semi_log",
        semiLog_dt=config.semi_log_dt,
        semiLogT0=1.0e-6,
        tolerance=1.0e-11,
        max_iterations=50,
        terminal_thin_phase_policy="raise",
        record=True,
        transformed_u_grid=u_grid,
        transformed_v_grid=v_grid,
        initial_eta_componentwise_scaling=True,
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
    return ExampleResult(config, model, None, {"final_time": float(model.currentTime),
                                               "inventory_abs": absolute, "inventory_rel": relative})


# %% [markdown]
# ## Exact solution, evaluation and plots

# %%
def exact_solution(config):
    """The similarity root, checked to be unique (several can exist in general [4]); ``k_bounds``
    is needed because the eigenvalues are far apart."""
    d = diffusivities(config.ratio)
    roots = solve_similarity_roots(d["ALPHA"], d["BETA"], U_LEFT_FAR, U_RIGHT_FAR, closure(), s0=config.s0,
                                   k_bounds=(1.0e-6, 0.3), n_k=2001)
    if len(roots) != 1:
        raise ValueError(f"Expected one similarity root, found {len(roots)}.")
    return roots[0]


def run(config=DEMO):
    """Build, solve and evaluate one configuration."""
    model = build_model(config)
    model.solve(config.duration, minDtFrac=1e-14)
    exact = exact_solution(config)
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
        DEMO_LIMITS=DEMO_LIMITS,
        FAST=FAST,
        description=__doc__,
        diffusivities=diffusivities,
    )


def evaluate(model, config, exact):
    """Return accuracy metrics in the u basis, or only validity estimates for a short run. See the advanced companion."""
    return _advanced.evaluate(model, config, exact, case=_advanced_case())


def report(result):
    """Print the final state and the metrics, with limits where they apply. See the advanced companion."""
    return _advanced.report(result, case=_advanced_case())


def plot(result):
    """Plot the interface and the final carbon profile in u and x against the exact u; returns the figure. See the advanced companion."""
    return _advanced.plot(result, case=_advanced_case())


def settings_from_command_line(config, plot_figures):
    """Apply ``--fast`` and ``--plot`` from a terminal; inside a notebook kernel, change nothing. See the advanced companion."""
    return _advanced.settings_from_command_line(config, plot_figures, case=_advanced_case())


# %% [markdown]
# ## Run
#
# For the introductory path use `result = run_core(CONFIG)` after the Settings
# and model cells. The compatibility entry point below also runs the detailed
# checks in `advanced/ternary_interstitial_similarity.py`.

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
# The numbers here are synthetic; the exact solution and the composition basis come from the
# literature.
#
# 1. J. S. Kirkaldy, "Diffusion in multicomponent metallic systems: III. The motion of planar
#    phase interfaces," Can. J. Phys. 36 (1958) 917-925, doi:10.1139/p58-098. Similarity
#    solutions for planar interfaces in multicomponent systems with constant diffusivities.
# 2. D. E. Coates, J. S. Kirkaldy, "Morphological stability of alpha-beta phase interfaces in
#    the Cu-Zn-Ni system at 775 C," Metall. Trans. 2 (1971) 3467-3477,
#    doi:10.1007/BF02811629. Applies the ternary two-phase similarity solution.
# 3. J.-O. Andersson, J. Agren, "Models for numerical treatment of multicomponent diffusion in
#    simple phases," J. Appl. Phys. 72 (1992) 1350-1355, doi:10.1063/1.351745. The u-fraction
#    definition (Eq. (9)), and the assumption that only substitutional elements take up volume
#    (Eqs. (4)-(5)).
# 4. P. Maugis, W. D. Hopfe, J. E. Morral, J. S. Kirkaldy, "Multiple interface velocity
#    solutions for ternary biphase infinite diffusion couples," Acta Mater. 45 (1997)
#    1941-1954. More than one interface velocity can satisfy the same couple.
