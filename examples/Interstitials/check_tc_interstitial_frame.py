"""Checks the frame/basis of Thermo-Calc's chemical diffusion coefficient for C.

kawin's ``Mobility.interdiffusivity`` returns the volume-fixed reduced
diffusivity in the u-fraction basis used by DICTRA: for an interstitial k the
flux is ``J_k = -sum_j D_kj du_j/dz`` (per substitutional molar volume) and
``du_k/dt = -dJ_k/dz``. The Illingworth solvers rely on that convention.

This script tests whether TC's ``DC(FCC_A1,C,C,FE)`` uses the same convention.
It runs a single-phase austenite Fe-C diffusion problem in DICTRA at high carbon
(where a (1 - x_C) or u-vs-x frame mismatch would be several percent), starting
from a smooth cosine u_C profile entered point by point. The cosine has zero
slope at both closed ends and decays visibly within the run. A step start is
avoided because DICTRA's trapezoidal time integration rings at the
discontinuity; a linear start barely evolves away from the ends. It then
re-solves the same problem with a small, independent implicit finite-volume
solver, using TC's tabulated DC in two candidate frames:

``u_frame``  J = -D du/dz, du/dt = -dJ/dz    (kawin / expected DICTRA frame)
``x_frame``  J = -D dx/dz, du/dt = -dJ/dz    (gradient taken in mole fraction)

The frame that reproduces DICTRA to discretization accuracy is the one TC uses.
A secondary, database-mismatched sanity check compares TC's DC with kawin's
pycalphad diffusivity from the FCC-only Fe-C mobility in ``FECRC_DB``.

Requires TC-Python with the FEDEMO/MFEDEMO databases. Runtime is about 1-3 min.

Run (from the repo root, through the conda activation wrapper in AGENTS.md):
    python -m examples.Interstitials.check_tc_interstitial_frame
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("examples") / "Interstitials" / "outputs"
TEMPERATURE = 1373.0
WIDTH = 4.0e-4
U_LEFT = 0.08
U_RIGHT = 0.005
SIM_TIME = 1000.0
DICTRA_POINTS = 401
FV_CELLS = 800
FV_STEPS = 4000


def initial_u(z):
    """Cosine start between U_LEFT (z=0) and U_RIGHT (z=WIDTH), zero end slopes."""
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * (U_LEFT + U_RIGHT) + 0.5 * (U_LEFT - U_RIGHT) * np.cos(np.pi * z / WIDTH)


def u_from_x(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 - x)


def x_from_u(u):
    u = np.asarray(u, dtype=np.float64)
    return u / (1.0 + u)


def tabulate_tc_dc(system, x_values):
    """Returns DC(FCC_A1,C,C,FE) at each mole fraction in ``x_values``."""
    from tc_python import ALL_PHASES

    calc = (
        system.with_single_equilibrium_calculation()
        .set_condition("T", TEMPERATURE)
        .set_phase_to_suspended(ALL_PHASES)
        .set_phase_to_entered("FCC_A1", 1.0)
    )
    values = []
    for x in x_values:
        calc.set_condition("X(C)", float(x))
        result = calc.calculate()
        values.append(float(result.get_value_of("DC(FCC_A1,C,C,FE)")))
    return np.asarray(values, dtype=np.float64)


def run_dictra(system):
    """Runs the single-phase austenite problem and returns (z, u_C) at SIM_TIME."""
    from tc_python import GridPoint, PointByPointGrid, Region, Unit

    grid = PointByPointGrid(Unit.U_FRACTION)
    for z in np.linspace(0.0, WIDTH, DICTRA_POINTS):
        grid.add_point(GridPoint(float(z)).add_composition("C", float(initial_u(z))))
    calculator = (
        system.with_isothermal_diffusion_calculation()
        .set_temperature(TEMPERATURE)
        .set_simulation_time(SIM_TIME)
        .add_region(
            Region("Austenite")
            .with_point_by_point_grid_containing_compositions(grid)
            .add_phase("FCC_A1")
        )
    )
    results = calculator.calculate()
    z, x_c = results.get_mole_fraction_of_component_at_time("C", SIM_TIME)
    return np.asarray(z, dtype=np.float64), u_from_x(x_c)


def solve_fv(D_of_x, frame):
    """Implicit (lagged-D) planar finite volumes with zero-flux ends.

    The conserved variable is u_C. ``frame='u'`` drives the flux with du/dz,
    ``frame='x'`` with dx/dz = du/dz / (1 + u)^2 evaluated on each face.
    """
    dz = WIDTH / FV_CELLS
    z = (np.arange(FV_CELLS) + 0.5) * dz
    u = initial_u(z)
    dt = SIM_TIME / FV_STEPS
    for _ in range(FV_STEPS):
        u_face = 0.5 * (u[:-1] + u[1:])
        D_face = D_of_x(x_from_u(u_face))
        if frame == "x":
            D_face = D_face / (1.0 + u_face) ** 2
        coeff = D_face * dt / dz**2
        lower = np.zeros(FV_CELLS)
        upper = np.zeros(FV_CELLS)
        diag = np.ones(FV_CELLS)
        diag[:-1] += coeff
        diag[1:] += coeff
        upper[:-1] = -coeff
        lower[1:] = -coeff
        u = _thomas(lower, diag, upper, u.copy())
    return z, u


def _thomas(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    out = np.zeros(n)
    out[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        out[i] = dp[i] - cp[i] * out[i + 1]
    return out


def kawin_fecrc_dc(x_values):
    """kawin pycalphad D_CC for FCC Fe-C from FECRC_DB (different database)."""
    from kawin.tests.datasets import FECRC_DB
    from kawin.thermo import GeneralThermodynamics

    therm = GeneralThermodynamics(FECRC_DB, ["FE", "C"], ["FCC_A1"])
    return np.asarray([float(np.squeeze(therm.getInterdiffusivity(float(x), TEMPERATURE))) for x in x_values])


def main():
    from tc_python import TCPython

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_table = np.linspace(x_from_u(U_RIGHT) * 0.5, x_from_u(U_LEFT) * 1.05, 81)
    with TCPython() as session:
        system = (
            session.set_cache_folder(str(OUTPUT_DIR / "tc_cache"))
            .select_thermodynamic_and_kinetic_databases_with_elements("FEDEMO", "MFEDEMO", ["Fe", "C"])
            .get_system()
        )
        dc_table = tabulate_tc_dc(system, x_table)
        z_dictra, u_dictra = run_dictra(system)

    def D_of_x(x):
        return np.interp(x, x_table, dc_table)

    report = {"temperature_K": TEMPERATURE, "u_left": U_LEFT, "u_right": U_RIGHT, "time_s": SIM_TIME}
    for frame in ("u", "x"):
        z_fv, u_fv = solve_fv(D_of_x, frame)
        u_on_dictra = np.interp(z_dictra, z_fv, u_fv)
        err = u_on_dictra - u_dictra
        report[f"{frame}_frame_max_abs_error"] = float(np.max(np.abs(err)))
        report[f"{frame}_frame_rms_error"] = float(np.sqrt(np.mean(err**2)))
        report[f"{frame}_frame_max_rel_error_of_span"] = float(np.max(np.abs(err)) / (U_LEFT - U_RIGHT))
        report[f"{frame}_frame_argmax_z"] = float(z_dictra[int(np.argmax(np.abs(err)))])
        interior = (z_dictra > 0.05 * WIDTH) & (z_dictra < 0.95 * WIDTH)
        report[f"{frame}_frame_interior_max_abs_error"] = float(np.max(np.abs(err[interior])))

    # Discretization reference: the u-frame run on a 2x coarser grid/timestep.
    global FV_CELLS, FV_STEPS
    fine_cells, fine_steps = FV_CELLS, FV_STEPS
    FV_CELLS, FV_STEPS = fine_cells // 2, fine_steps // 2
    z_c, u_c = solve_fv(D_of_x, "u")
    FV_CELLS, FV_STEPS = fine_cells, fine_steps
    z_f, u_f = solve_fv(D_of_x, "u")
    report["u_frame_self_convergence_max_abs"] = float(np.max(np.abs(np.interp(z_f, z_c, u_c) - u_f)))

    try:
        kawin_dc = kawin_fecrc_dc(x_table[::10])
        report["secondary_kawin_FECRC_over_TC_DC"] = (kawin_dc / dc_table[::10]).tolist()
        report["secondary_x"] = x_table[::10].tolist()
    except Exception as exc:  # pragma: no cover - diagnostic only
        report["secondary_error"] = repr(exc)

    better = "u" if report["u_frame_max_abs_error"] < report["x_frame_max_abs_error"] else "x"
    report["closer_frame"] = better
    report["tc_dc_table"] = {"x": x_table.tolist(), "DC": dc_table.tolist()}
    report["dictra_profile"] = {"z": z_dictra.tolist(), "u_C": u_dictra.tolist()}
    out = OUTPUT_DIR / "tc_interstitial_frame_check.json"
    out.write_text(json.dumps(report, indent=2))
    summary = {k: v for k, v in report.items() if k not in ("tc_dc_table", "dictra_profile")}
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
