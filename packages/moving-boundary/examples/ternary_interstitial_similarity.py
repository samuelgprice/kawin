"""How does a fast interstitial and a slow substitutional solute move an interface?

A Fe-C-Mn-like couple: carbon diffuses about 100 times faster than manganese. With
constant matrices and a linear tie-line family the problem has an exact similarity
solution, so this example checks numerics without any thermodynamic data. The
numbers are synthetic and are not real Fe-C-Mn properties.

**Composition basis is the main lesson.** Elements are ``[FE, C, MN]`` with FE
dependent and interstitials listed after it. The solver's state is in u-fractions,
``u_k = x_k / sum(substitutional x)``. So here:

- the diffusivity matrices, closure and exact solution are in u;
- the closure declares ``composition_basis="u"`` (a closure in x would be converted);
- the initial profile is given in mole fractions;
- ``getPhysicalPhaseProfiles`` and ``getCompositions`` return mole fractions, while
  ``getTransformedState`` returns the u state.

Compare with the exact solution in u, as ``compare_model_to_similarity`` does.

Assumptions: planar, constant matrices, constant volume per substitutional atom, no
bulk flow, effectively semi-infinite phases. Units are nondimensional. ALPHA lies
left and BETA right; the interface moves right, with parabolic ``s = s0 + k sqrt(t)``
and a constant tie line ``eta``.

Try changing ``ratio`` (a smaller value needs a stronger graded grid), the spacing
ratios or the far fields. Point-wise Mn profile errors are large because the Mn
spike is narrower than the remaining interface offset, so only carbon has a profile
limit. ``terminal_thin_phase_policy`` must be explicit for ternary models.

This checks numerical accuracy under those assumptions, not Fe-C-Mn physics.

Run ``python ternary_interstitial_similarity.py`` (about a second), or add ``--fast``.
"""

from dataclasses import dataclass
import argparse

import numpy as np

from moving_boundary.analytic import (
    FixedMatrixTernaryDiffusivity,
    compare_model_to_similarity,
    semi_infinite_validity_time,
    solve_similarity_roots,
)
from moving_boundary.equilibrium import CallableTernaryInterfaceEquilibrium
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from moving_boundary.mesh.TransformedGrids import two_phase_landau_grids
from moving_boundary.solvers import MovingBoundaryIllingworthTernaryFD1DModel

ELEMENTS = ["FE", "C", "MN"]
PHASES = ["ALPHA", "BETA"]
U_LEFT_FAR = np.array([0.002, 0.015])   # far-field [u_C, u_MN] in ALPHA
U_RIGHT_FAR = np.array([0.010, 0.015])  # far-field [u_C, u_MN] in BETA


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

# Numerical accuracy limits for DEMO only: 81/81 nodes, width 0.8 + 0.4, t to 1, window t >= 0.1.
# k_rel: |k - k_exact| / |k_exact|. interface_rel: max|s - s_exact| / max|s_exact - s0|.
# eta_abs: max|eta - eta_exact|. carbon_profile_linf: max|u_C - u_C,exact| over five recorded
# times. inventory_abs: largest component |final - initial| in u x length.
DEMO_LIMITS = {"k_rel": 1.0e-2, "interface_rel": 1.0e-2, "eta_abs": 1.0e-3,
               "carbon_profile_linf": 1.0e-3, "inventory_abs": 5.0e-9 * 1.2}


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


def exact_solution(config):
    """The unique similarity root; ``k_bounds`` is needed because the eigenvalues are far apart."""
    d = diffusivities(config.ratio)
    roots = solve_similarity_roots(d["ALPHA"], d["BETA"], U_LEFT_FAR, U_RIGHT_FAR, closure(), s0=config.s0,
                                   k_bounds=(1.0e-6, 0.3), n_k=2001)
    if len(roots) != 1:
        raise ValueError(f"Expected one similarity root, found {len(roots)}.")
    return roots[0]


def evaluate(model, config, exact):
    """Return accuracy metrics in the u basis, or only validity estimates for a short run."""
    d = diffusivities(config.ratio)
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


def run(config=DEMO):
    """Build, solve and evaluate one configuration."""
    model = build_model(config)
    model.solve(config.duration, minDtFrac=1e-14)
    exact = exact_solution(config)
    return ExampleResult(config, model, exact, evaluate(model, config, exact))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="use quick check settings")
    args = parser.parse_args()
    result = run(FAST if args.fast else DEMO)
    print(f"final interface {result.model.getInterfacePosition():.6f}, eta {result.model.getInterfaceEta():.6f}; "
          f"exact k {result.exact.k:.6f}, eta {result.exact.eta:.6f}")
    for name, value in result.metrics.items():
        limit = DEMO_LIMITS.get(name) if not args.fast else None
        print(f"{name}: {value:.4g}" + (f"  (limit {limit:g})" if limit else ""))
