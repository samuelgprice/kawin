"""How fast does a planar interface grow under diffusion control?

Grows phase ALPHA into BETA with fixed interface compositions and compares the
result with the exact Zener solution, ``s(t) = 2 j sqrt(D t)``. This is the place
to start: it shows meshes, phases, transformed grids and solving, with no
thermodynamic database.

Assumptions: planar, constant equal diffusivities, equal molar volumes, a
semi-infinite BETA. ALPHA stays flat at its interface composition. Quantities are
nondimensional (domain length 1, D = 1); the temperature is required by the API
but has no effect here. Elements are ``[NI, CR]`` with CR the solute, and ALPHA
lies to the left; the labels themselves carry no thermodynamics. Compositions are
mole fractions.

Try changing the node counts, ``time_step`` or the compositions in
``VALIDATION_PARAMS``. The exact solution starts from ``t_start > 0`` because a
zero-width start is singular. Coarse uniform grids resolve the early profile
poorly: 21 nodes in BETA gives a far larger interface error than 41.

This checks numerical accuracy for constant diffusivity. It says nothing about real
thermodynamics; see ``ternary_fecrni_lee1996.py`` for that.

Run ``python binary_planar_growth.py`` (under a second), or add ``--fast`` or ``--plot``.
"""

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
from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel


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
    p = VALIDATION_PARAMS
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


def plot(result):
    """Plot the interface error against the exact solution."""
    import matplotlib.pyplot as plt

    times, interface = recorded_history(result.model.interfaceData)
    absolute = times + result.config.t_start
    plt.loglog(absolute[1:], np.abs(np.ravel(interface) - planar_exact_interface(absolute))[1:])
    plt.xlabel("time (nondimensional)")
    plt.ylabel("|s - s_exact|")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="use quick check settings")
    parser.add_argument("--plot", action="store_true", help="plot the interface error")
    args = parser.parse_args()
    result = run(FAST if args.fast else DEMO)
    if args.fast:
        print("Quick check settings: a coarse grid and a very short run, so the interface error is")
        print("large and has no limit. Run without --fast to judge accuracy.")
    for name, value in result.metrics.items():
        limit = DEMO_LIMITS.get(name) if not args.fast else None
        print(f"{name}: {value:.4g}" + (f"  (limit {limit:g})" if limit else ""))
    if args.plot:
        plot(result)
