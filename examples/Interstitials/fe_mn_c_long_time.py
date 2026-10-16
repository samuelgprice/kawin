"""Fe-Mn-C ferrite growth to full equilibrium, checked against the no-partitioning plateau and the lever rule.

Runs the ``fe_mn_c_ferrite`` case (cached Thermo-Calc inputs; no license
needed) far past the 100 s benchmark, through the two classic stages:

1. carbon-controlled no-partitioning growth, which stalls once carbon has
   spread through the austenite at the **no-partitioning plateau**: ferrite at
   the alloy Mn content and the austenite carbon of that tie-line;
2. slow Mn partitioning towards **full equilibrium**: the tie-line through the
   alloy composition.

Both reference ferrite fractions come from the cached tie-lines in
substitutional-mole units (u-fractions), which is what ``s / R`` measures under
kawin's equal-molar-volume convention. ``u`` is a projective map of ``x``, so
the tie-line through the alloy in ``x`` is the tie-line through the alloy in
``u``.

Grid: long runs need a mild austenite interface grid. Semi-log steps grow with
time, and on the strongly graded benchmark grids (power exponent 3 and above,
interface cells ~1e-12 m and below) the interface cells' own mass balance falls
below round-off at late times (see the ``MovingBoundaryIllingworthTernaryFD1DModel``
docstring). The default here is 401 nodes with exponent 1.5 (interface cell
~6e-9 m), which resolves the parabolic stage less finely but reaches
equilibrium cleanly.

Run (repo root, through the conda wrapper in AGENTS.md; ~1 min):
    python -u -m examples.Interstitials.fe_mn_c_long_time
    python -u -m examples.Interstitials.fe_mn_c_long_time --t-end 1e10 --n-right 401 --power 1.5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time as wallclock

import numpy as np

import kawin.diffusion.MovingBoundaryIllingworthTernaryFDM as ternary_fdm
from examples.Interstitials import fe_mn_c_ferrite
from examples.Interstitials.ternary_ferrite_growth_le import OUTPUT_DIR, build_tc_inputs, run_kawin


def _u(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 - x[..., :1])


def reference_fractions(data):
    """No-partitioning plateau and full-equilibrium ferrite fractions (substitutional moles) from the cached tie-lines."""
    u_alloy = _u(data["x_alloy"])
    eta = np.asarray(data["tie_lines"]["eta"])
    left, right = _u(data["tie_lines"]["left"]), _u(data["tie_lines"]["right"])

    def offset(i):
        d, a = right[i] - left[i], u_alloy - left[i]
        return (d[0] * a[1] - d[1] * a[0]) / np.linalg.norm(d)

    offsets = np.array([offset(i) for i in range(len(eta))])
    i = int(np.flatnonzero(np.diff(np.sign(offsets)) != 0)[0])
    w = offsets[i] / (offsets[i] - offsets[i + 1])
    ferrite, austenite = left[i] + w * (left[i + 1] - left[i]), right[i] + w * (right[i + 1] - right[i])
    f_eq = float(np.dot(u_alloy - austenite, ferrite - austenite) / np.dot(ferrite - austenite, ferrite - austenite))

    k = int(np.flatnonzero(np.diff(np.sign(left[:, 1] - u_alloy[1])) != 0)[0])
    wk = (u_alloy[1] - left[k, 1]) / (left[k + 1, 1] - left[k, 1])
    ferrite_np, austenite_np = left[k] + wk * (left[k + 1] - left[k]), right[k] + wk * (right[k + 1] - right[k])
    f_np = float((u_alloy[0] - austenite_np[0]) / (ferrite_np[0] - austenite_np[0]))
    return {
        "npl_plateau_fraction": f_np,
        "equilibrium_fraction": f_eq,
        "equilibrium_eta": float(eta[i] + w * (eta[i + 1] - eta[i])),
        "tc_equilibrium_fraction_atoms": float(data["equilibrium_ferrite_fraction"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--t-end", type=float, default=1.0e10)
    parser.add_argument("--n-right", type=int, default=401)
    parser.add_argument("--power", type=float, default=1.5)
    parser.add_argument("--semi-log-dt", type=float, default=0.005)
    args = parser.parse_args()

    case = dataclasses.replace(fe_mn_c_ferrite.CASE, sim_time=args.t_end)
    data = build_tc_inputs(case)
    refs = reference_fractions(data)
    R = case.ferrite_width + case.austenite_width

    # run_kawin uses minDtFrac=1e-14, too large relative to a 1e-7 s first step at
    # these end times; give the solver a minimum step well below the first step.
    original_solve = ternary_fdm.MovingBoundaryIllingworthTernaryFD1DModel.solve

    def solve(self, simTime, **kwargs):
        kwargs["minDtFrac"] = 1.0e-10 / float(simTime)
        return original_solve(self, simTime, **kwargs)

    ternary_fdm.MovingBoundaryIllingworthTernaryFD1DModel.solve = solve
    start = wallclock.perf_counter()
    try:
        model, diagnostics = run_kawin(case, data, n_right=args.n_right, power=args.power, semi_log_dt=args.semi_log_dt, semi_log_t0=1.0e-7)
    finally:
        ternary_fdm.MovingBoundaryIllingworthTernaryFD1DModel.solve = original_solve

    t = np.asarray(model.interfaceData._time[: model.interfaceData.N + 1])
    s = np.asarray(model.interfaceData._y[: model.interfaceData.N + 1])
    eta = np.asarray(model.etaData._y[: model.etaData.N + 1])
    decades = 10.0 ** np.arange(0.0, np.log10(args.t_end) + 0.1, 1.0)
    inventory = np.asarray(model.inventoryData._y[: model.inventoryData.N + 1])
    report = {
        "settings": vars(args),
        "references": refs,
        "wall_s": round(wallclock.perf_counter() - start, 1),
        "steps": int(model.interfaceData.N),
        "retries": int(sum(d["retries"] for d in diagnostics)),
        "inventory_drift": (inventory[-1] - inventory[0]).tolist(),
        "samples": {f"{tt:g}": {"ferrite_fraction": float(np.interp(tt, t, s) / R), "eta": float(np.interp(tt, t, eta))} for tt in decades},
        "final_fraction": float(s[-1] / R),
        "final_eta": float(eta[-1]),
    }
    out = OUTPUT_DIR / f"{case.name}_long_time_{args.n_right}_p{args.power:g}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
