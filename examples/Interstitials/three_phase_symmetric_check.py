"""Three-phase model on a sharp Fe-Mn-C front: symmetric austenite|ferrite|austenite against the two-phase model.

A ferrite seed of width ``2 * w`` sits in the middle of two 50 um austenite
layers of the Fe-Mn-C alloy (``fe_mn_c_ferrite`` case, cached Thermo-Calc
inputs; no license needed). By symmetry each half is exactly the two-phase
problem with a ``w``-wide ferrite seed, so the three-phase ferrite
half-thickness must reproduce the two-phase interface position, and both
interfaces must stay mirror images with equal tie-line coordinates. The two
models share the austenite grid (power law), the time-step schedule and the
tie-line and diffusivity tables; the three-phase middle (ferrite) grid is
linear-clustered at both ends.

Run (repo root, through the conda wrapper in AGENTS.md; ~2 min for 2 s):
    python -u -m examples.Interstitials.three_phase_symmetric_check
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time as wallclock

import numpy as np

from kawin.diffusion.MovingBoundaryIllingworthTernaryThreePhaseFDM import MovingBoundaryIllingworthTernaryThreePhaseFD1DModel
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.TransformedGrids import landau_grid
from kawin.solver import explicitEulerIterator
from examples.Interstitials import fe_mn_c_ferrite
from examples.Interstitials.ternary_ferrite_growth_le import OUTPUT_DIR, TabulatedTernaryDiffusivity, build_tc_inputs, run_kawin, tie_line_closure


class _AusteniteFerriteClosure:
    """gamma|alpha view of a (ferrite, austenite) closure, keeping eta and the composition basis."""

    def __init__(self, closure):
        self.closure = closure
        self.composition_basis = getattr(closure, "composition_basis", "x")

    @property
    def eta_bounds(self):
        return self.closure.eta_bounds

    def interface_compositions(self, eta):
        ferrite, austenite = self.closure.interface_compositions(eta)
        return austenite, ferrite


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--n-austenite", type=int, default=401)
    parser.add_argument("--power", type=float, default=3.0)
    parser.add_argument("--n-ferrite", type=int, default=81)
    parser.add_argument("--semi-log-dt", type=float, default=0.005)
    args = parser.parse_args()

    case = dataclasses.replace(fe_mn_c_ferrite.CASE, sim_time=args.t_end)
    data = build_tc_inputs(case)
    x_alloy = np.asarray(data["x_alloy"])
    closure = tie_line_closure(data)
    half_ferrite, L_g = case.ferrite_width, case.austenite_width
    R3 = 2.0 * L_g + 2.0 * half_ferrite
    mesh = CartesianFD1D(["C", case.solute], [0.0, R3], 4001)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.5 * R3, x_alloy, x_alloy), ["C", case.solute])]))
    grids = (
        landau_grid(args.n_austenite, "power", phase="left", exponent=args.power),
        landau_grid(args.n_ferrite, "linear", phase="middle", spacing_ratio=0.2),
        landau_grid(args.n_austenite, "power", phase="right", exponent=args.power),
    )
    model = MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh, elements=["FE", "C", case.solute], phases=["FCC_A1", "BCC_A2", "FCC_A1"],
        thermodynamics=TabulatedTernaryDiffusivity(data["tables"]), temperature=case.temperature,
        interfacePositions=(L_g, L_g + 2.0 * half_ferrite), interface_equilibria=(_AusteniteFerriteClosure(closure), closure),
        time_step=args.t_end, dt_mode="semi_log", semiLog_dt=args.semi_log_dt, semiLogT0=1.0e-7,
        bulk_diffusivity_mode="composition_dependent_lagged", tolerance=1.0e-10, max_iterations=50, max_step_retries=12,
        terminal_thin_phase_width=1.0e-12, terminal_thin_phase_policy="continue", record=True, transformed_grids=grids,
    )
    start = wallclock.perf_counter()
    model.solve(args.t_end, iterator=explicitEulerIterator, minDtFrac=1.0e-14)
    three_phase_wall = wallclock.perf_counter() - start

    n = model.interfaceData.N + 1
    t = np.asarray(model.interfaceData._time[:n])
    s = np.asarray(model.interfaceData._y[:n])
    half = 0.5 * (s[:, 1] - s[:, 0])
    asymmetry = s[:, 0] - (R3 - s[:, 1])

    start = wallclock.perf_counter()
    two_phase, diagnostics = run_kawin(case, data, n_right=args.n_austenite, power=args.power, semi_log_dt=args.semi_log_dt, semi_log_t0=1.0e-7)
    two_phase_wall = wallclock.perf_counter() - start

    times = [tt for tt in (0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0) if tt <= args.t_end]
    k3 = {f"{tt:g}": float(np.interp(tt, t, half) / np.sqrt(tt) * 1e6) for tt in times}
    k2 = {f"{tt:g}": float(two_phase.interfaceData.y(tt) / np.sqrt(tt) * 1e6) for tt in times}
    etas = np.asarray(model.etaData._y[model.etaData.N])
    report = {
        "settings": vars(args),
        "three_phase": {"wall_s": round(three_phase_wall, 1), "steps": int(model.interfaceData.N), "k_um_per_sqrt_s": k3},
        "two_phase": {"wall_s": round(two_phase_wall, 1), "steps": int(two_phase.interfaceData.N),
                      "retries": int(sum(d["retries"] for d in diagnostics)), "k_um_per_sqrt_s": k2},
        "k_rel_diff": {key: k3[key] / k2[key] - 1.0 for key in k3},
        "max_abs_asymmetry_m": float(np.max(np.abs(asymmetry))),
        "final_etas": etas.tolist(),
    }
    out = OUTPUT_DIR / "fe_mn_c_three_phase_symmetric_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
