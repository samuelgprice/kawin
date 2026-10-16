"""Time-step and grid convergence of the kawin interstitial benchmarks, with Richardson extrapolation.

Runs each benchmark's own ``run_kawin`` on a matrix of austenite grids and
semi-log time steps (each level halves the previous one), then removes the
two discretization errors one after the other:

1. **Time.** For each grid, ``k(dt -> 0) = 2 k(dt/2) - k(dt)`` from the two
   finest steps (backward Euler is first order in dt). The ratio of successive
   differences, ``(k1 - k2) / (k2 - k3)`` for three consecutive levels, is
   reported as the observed convergence ratio (2 for first order).
2. **Space.** From the dt-extrapolated values, the same first-order formula on
   the two finest grids, and a second estimate that uses the observed ratio
   ``r`` of the three finest grids, ``k3 + (k3 - k2) / (r - 1)``. The spread
   between the two is a measure of the extrapolation uncertainty.

The ternary grids are power laws ``v = (i/n)^p``. The first three change the
exponent with ``n`` (as used in the benchmarks); the last two share ``p = 3.7``,
so the final step is a clean doubling at a fixed mapping. 3201/p3.7 needs a
smallest relative interval below ``landau_grid``'s default 1e-12, so the grid
guard is lowered to 1e-14 for it. The binary grids (``v = (i/n)^2``) all share
one mapping.

For Fe-Mn-C the coarsest step (dlnt = 0.01) on the 1601 and 3201 grids reaches
the solver's resolution limit (interface cells of ~1e-16 m and below; see the
``MovingBoundaryIllingworthTernaryFD1DModel`` docstring): the carbon interface
residual's round-off noise approaches the 1e-10 tolerance, so many steps are
retried. Those runs only enter the reported dt ratios, not the extrapolation,
which uses the two finest steps.

The extrapolated growth constants ``k = s/sqrt(t)`` are compared with the
cached DICTRA references and, for the ternary cases, with the semi-analytical
NPLE reference from ``npl_similarity_reference.py`` when its JSON exists.

Cases (all use cached Thermo-Calc inputs and DICTRA references, so no
Thermo-Calc license is needed):

``fe-c``     binary exb1a (1050 K), to 1e9 s, grids 801/1601/3201/6401, dlnt 0.02 ... 0.0025
``fe-ni-c``  Fe-Ni-C exh2 geometry (973 K), to 2 s, grids 401/p3 ... 3201/p3.7, dlnt 0.01 ... 0.00125
``fe-mn-c``  Fe-Mn-C (1000 K), to 2 s, same grids and steps

Run (repo root, through the conda wrapper in AGENTS.md). The binary study takes
a few minutes; each ternary study about 10-15 minutes:
    python -u -m examples.Interstitials.convergence_study fe-ni-c
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time as wallclock

import numpy as np

from examples.Interstitials import fe_c_exb1a, fe_mn_c_ferrite, fe_ni_c_exh2_le
from examples.Interstitials.ternary_ferrite_growth_le import OUTPUT_DIR, build_tc_inputs, dictra_cache_path, run_kawin

TERNARY_GRIDS = ((401, 3.0), (801, 3.5), (1601, 3.7), (3201, 3.7))
TERNARY_MIN_INTERVAL = {(3201, 3.7): 1.0e-14}
TERNARY_DTS = (0.01, 0.005, 0.0025, 0.00125)
TERNARY_TIMES = (0.01, 0.1, 0.5, 1.0, 2.0)
BINARY_GRIDS = (801, 1601, 3201, 6401)
BINARY_DTS = (0.02, 0.01, 0.005, 0.0025)
BINARY_TIMES = (1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9)


def _ratio(a, b, c):
    """Observed convergence ratio (a - b) / (b - c) for three consecutive levels (2 for first order)."""
    return (a - b) / (b - c) if b != c else float("nan")


def _extrapolate(runs, grids, dts, times):
    """
    Richardson in dt per grid (two finest steps), then in grid spacing.

    Returns the dt-extrapolated value per grid, the observed dt ratios, the
    first-order and observed-order grid limits, and the observed grid ratios.
    """
    per_grid, dt_ratios = {}, {}
    for grid in grids:
        k = [runs[(grid, dt)]["k"] for dt in dts]
        per_grid[grid] = {t: 2.0 * k[-1][t] - k[-2][t] for t in times}
        dt_ratios[grid] = {t: [_ratio(k[i][t], k[i + 1][t], k[i + 2][t]) for i in range(len(dts) - 2)] for t in times}
    first_order, observed_order, grid_ratios = {}, {}, {}
    for t in times:
        values = [per_grid[g][t] for g in grids]
        ratios = [_ratio(values[i], values[i + 1], values[i + 2]) for i in range(len(grids) - 2)]
        grid_ratios[t] = ratios
        first_order[t] = 2.0 * values[-1] - values[-2]
        r = ratios[-1]
        observed_order[t] = values[-1] + (values[-1] - values[-2]) / (r - 1.0) if np.isfinite(r) and r > 1.0 else float("nan")
    return per_grid, dt_ratios, first_order, observed_order, grid_ratios


def _label(grid):
    return f"{grid[0]}/p{grid[1]:g}" if isinstance(grid, tuple) else str(grid)


def _times_dict(values, scale=1.0):
    return {f"{t:g}": (v * scale if np.isscalar(v) else v) for t, v in values.items()}


def study_ternary(name, case, npl_json):
    data = build_tc_inputs(case)
    dictra = json.loads(dictra_cache_path(case, 2.0).read_text())
    short = dataclasses.replace(case, sim_time=2.0)
    runs = {}
    for grid in TERNARY_GRIDS:
        for dt in TERNARY_DTS:
            start = wallclock.perf_counter()
            model, diagnostics = run_kawin(
                short, data, n_right=grid[0], power=grid[1], semi_log_dt=dt, semi_log_t0=1.0e-7,
                grid_min_interval=TERNARY_MIN_INTERVAL.get(grid),
            )
            run = {
                "k": {t: float(model.interfaceData.y(t)) / np.sqrt(t) for t in TERNARY_TIMES},
                "wall_s": wallclock.perf_counter() - start,
                "steps": int(model.interfaceData.N),
                "retries": int(sum(d["retries"] for d in diagnostics)),
                "inventory_drift": (np.asarray(model.getTotalInventory()) - np.asarray(model.inventoryData._y[0])).tolist(),
            }
            runs[(grid, dt)] = run
            print(f"{name} {_label(grid)} dt={dt}: k(2)={run['k'][2.0] * 1e6:.6f} um/sqrt(s), "
                  f"{run['steps']} steps, {run['retries']} retries, {run['wall_s']:.1f} s", flush=True)
    per_grid, dt_ratios, first_order, observed_order, grid_ratios = _extrapolate(runs, TERNARY_GRIDS, TERNARY_DTS, TERNARY_TIMES)
    dictra_k = {t: float(np.interp(t, dictra["time"], dictra["position"])) / np.sqrt(t) for t in TERNARY_TIMES}
    report = {
        "case": case.name,
        "grids": [_label(g) for g in TERNARY_GRIDS],
        "time_steps": TERNARY_DTS,
        "runs": {f"{_label(g)} dt={dt}": {**r, "k": _times_dict(r["k"], 1e6)} for (g, dt), r in runs.items()},
        "observed_dt_ratio": {_label(g): _times_dict(dt_ratios[g]) for g in TERNARY_GRIDS},
        "observed_grid_ratio": _times_dict(grid_ratios),
        "k_um_per_sqrt_s": {
            "dt_extrapolated": {_label(g): _times_dict(per_grid[g], 1e6) for g in TERNARY_GRIDS},
            "limit_first_order": _times_dict(first_order, 1e6),
            "limit_observed_order": _times_dict(observed_order, 1e6),
            "dictra": _times_dict(dictra_k, 1e6),
        },
        "limit_first_order_vs_dictra": {f"{t:g}": first_order[t] / dictra_k[t] - 1.0 for t in TERNARY_TIMES},
    }
    npl_path = OUTPUT_DIR / npl_json
    if npl_path.exists():
        npl = json.loads(npl_path.read_text())
        k_ref = float(npl["k_um_per_sqrt_s_iso_activity"])
        report["npl_reference_k_um_per_sqrt_s"] = {"iso_activity": k_ref, "cross_term": float(npl["k_um_per_sqrt_s_cross_term"])}
        report["vs_npl_iso_activity_at_2s"] = {
            "limit_first_order": first_order[2.0] * 1e6 / k_ref - 1.0,
            "limit_observed_order": observed_order[2.0] * 1e6 / k_ref - 1.0,
            "dictra": dictra_k[2.0] * 1e6 / k_ref - 1.0,
            **{f"kawin_{_label(g)}_dt0": per_grid[g][2.0] * 1e6 / k_ref - 1.0 for g in TERNARY_GRIDS},
        }
    return report


def study_fe_c():
    data = fe_c_exb1a.build_tc_inputs("exb1a", 100)
    sim_time = fe_c_exb1a.CASES["exb1a"]
    runs = {}
    for grid in BINARY_GRIDS:
        for dt in BINARY_DTS:
            start = wallclock.perf_counter()
            model = fe_c_exb1a.run_kawin(data, sim_time, n_right=grid, semi_log_dt=dt)
            run = {
                "k": {t: float(model.interfaceData.y(t)) / np.sqrt(t) for t in BINARY_TIMES},
                "wall_s": wallclock.perf_counter() - start,
                "steps": int(model.interfaceData.N),
            }
            runs[(grid, dt)] = run
            print(f"fe-c {grid} dt={dt}: s(1e3)={run['k'][1e3] * np.sqrt(1e3) * 1e6:.5f} um, s(1e9)={run['k'][1e9] * np.sqrt(1e9) * 1e3:.6f} mm, "
                  f"{run['steps']} steps, {run['wall_s']:.1f} s", flush=True)
    per_grid, dt_ratios, first_order, observed_order, grid_ratios = _extrapolate(runs, BINARY_GRIDS, BINARY_DTS, BINARY_TIMES)
    tight, default = data["dictra"], data["dictra_default"]
    thickness = {}
    for t in BINARY_TIMES:
        s_tight = float(np.interp(t, tight["time"], tight["position"]))
        thickness[f"{t:g}"] = {
            "dictra_tight_um": s_tight * 1e6,
            "dictra_default_um": float(np.interp(t, default["time"], default["position"])) * 1e6,
            "kawin_finest_dt0_um": per_grid[BINARY_GRIDS[-1]][t] * np.sqrt(t) * 1e6,
            "kawin_limit_first_order_um": first_order[t] * np.sqrt(t) * 1e6,
            "kawin_limit_observed_order_um": observed_order[t] * np.sqrt(t) * 1e6,
            "kawin_limit_first_order_vs_dictra_tight": first_order[t] * np.sqrt(t) / s_tight - 1.0,
        }
    return {
        "case": "fe_c_exb1a",
        "grids": list(BINARY_GRIDS),
        "time_steps": BINARY_DTS,
        "runs": {f"{g} dt={dt}": {**r, "k": _times_dict(r["k"])} for (g, dt), r in runs.items()},
        "observed_dt_ratio": {str(g): _times_dict(dt_ratios[g]) for g in BINARY_GRIDS},
        "observed_grid_ratio": _times_dict(grid_ratios),
        "thickness": thickness,
    }


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "fe-ni-c"
    if which == "fe-c":
        report = study_fe_c()
    elif which == "fe-ni-c":
        report = study_ternary(which, fe_ni_c_exh2_le.CASE, "fe_ni_c_exh2_le_npl_similarity_reference.json")
    elif which == "fe-mn-c":
        report = study_ternary(which, fe_mn_c_ferrite.CASE, "fe_mn_c_ferrite_le_npl_similarity_reference.json")
    else:
        raise SystemExit("usage: convergence_study {fe-c,fe-ni-c,fe-mn-c}")
    out = OUTPUT_DIR / f"{report['case']}_convergence.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
