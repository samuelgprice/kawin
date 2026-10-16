"""Checks Thermo-Calc's ternary ``DC`` matrix (Fe-Ni-C) against kawin's u-fraction frame.

``check_tc_interstitial_frame.py`` verified the binary Fe-C coefficient
``DC(FCC_A1,C,C,FE)``. This script checks the full 2x2 matrix in Fe-Ni-C
austenite, in particular the C-Ni cross term ``DC(FCC_A1,C,NI,FE)`` that drives
uphill carbon diffusion across a Ni gradient and controls how carbon responds to
the Ni spike in the local-equilibrium ferrite-growth benchmark.

A single-phase austenite region (FEDEMO + MFEDEMO, 973 K, 10 um, closed) starts
with a cosine Ni profile (u_Ni 0.01-0.03) and a small cosine carbon profile.
Ni barely moves at 973 K, so carbon redistributes uphill through
``D_CNi du_Ni/dz`` by about ``(D_CNi/D_CC) * du_Ni ~ 6e-4`` in u_C. DICTRA's
result is compared with an independent implicit finite-volume solve that uses
TC's tabulated DC matrices (the same 2-D table approach as the benchmarks) in
four variants:

``u_frame``        J = -D du/dz                    (kawin's convention)
``x_frame``        J = -D dx/dz, x = x(u)
``no_cross``       u-frame with the off-diagonal terms set to zero
``flipped_cross``  u-frame with D_CNi sign-flipped

Only the correct frame should match DICTRA to discretization accuracy.
Runtime: about 2-4 min (DC table ~1 min, DICTRA ~1 min).

Run (repo root, through the conda activation wrapper in AGENTS.md):
    python -u -m examples.Interstitials.check_tc_interstitial_frame_ternary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_banded

OUTPUT_DIR = Path("examples") / "Interstitials" / "outputs"
TEMPERATURE = 973.0
WIDTH = 10.0e-6
SIM_TIME = 60.0
U_C_MEAN, U_C_AMP = 0.004, 0.001
U_NI_MEAN, U_NI_AMP = 0.02, 0.01
DICTRA_POINTS = 201
FV_CELLS = 400
FV_STEPS = 3000
ELEMENTS = ("C", "NI")


def initial_u(z):
    """Cosine starts for (u_C, u_Ni) with zero slope at both closed ends."""
    phase = np.cos(np.pi * np.asarray(z, dtype=np.float64) / WIDTH)
    return np.column_stack((U_C_MEAN + U_C_AMP * phase, U_NI_MEAN - U_NI_AMP * phase))


def x_from_u(u):
    """Mole fractions (x_C, x_Ni) from u-fractions; C is interstitial, Ni substitutional."""
    u = np.atleast_2d(u)
    return u / (1.0 + u[:, :1])


def u_from_x(x):
    x = np.atleast_2d(x)
    return x / (1.0 - x[:, :1])


def tc_dc_table(system, x_c, x_ni):
    """DC(FCC_A1,k,j,FE) on an (x_C, x_Ni) grid; rows/cols (C, NI)."""
    from tc_python import ALL_PHASES

    calc = (
        system.with_single_equilibrium_calculation()
        .set_condition("T", TEMPERATURE)
        .set_phase_to_suspended(ALL_PHASES)
        .set_phase_to_entered("FCC_A1", 1.0)
    )
    out = np.empty((len(x_c), len(x_ni), 2, 2))
    for i, xc in enumerate(x_c):
        for j, xn in enumerate(x_ni):
            calc.set_condition("X(C)", float(xc))
            calc.set_condition("X(NI)", float(xn))
            result = calc.calculate()
            for a, k in enumerate(ELEMENTS):
                for b, g in enumerate(ELEMENTS):
                    out[i, j, a, b] = float(result.get_value_of(f"DC(FCC_A1,{k},{g},FE)"))
    return out


def run_dictra(system):
    """Runs the single-phase austenite problem; returns (z, u) at SIM_TIME."""
    from tc_python import GridPoint, PointByPointGrid, Region, Unit

    grid = PointByPointGrid(Unit.U_FRACTION)
    for z in np.linspace(0.0, WIDTH, DICTRA_POINTS):
        u = initial_u(z)[0]
        grid.add_point(GridPoint(float(z)).add_composition("C", float(u[0])).add_composition("Ni", float(u[1])))
    calculator = (
        system.with_isothermal_diffusion_calculation()
        .set_temperature(TEMPERATURE)
        .set_simulation_time(SIM_TIME)
        .add_region(Region("Austenite").with_point_by_point_grid_containing_compositions(grid).add_phase("FCC_A1"))
    )
    result = calculator.calculate()
    z, x_c = result.get_mole_fraction_of_component_at_time("C", SIM_TIME)
    _, x_ni = result.get_mole_fraction_of_component_at_time("Ni", SIM_TIME)
    return np.asarray(z, dtype=np.float64), u_from_x(np.column_stack((x_c, x_ni)))


def solve_fv(D_of_x, variant):
    """
    Implicit (lagged-D) planar finite volumes for (u_C, u_Ni) with zero-flux ends.

    The conserved variables are u-fractions. ``u_frame`` uses J = -D du/dz;
    ``x_frame`` uses J = -D dx/dz with dx/dz = (dx/du) du/dz on each face.
    """
    dz = WIDTH / FV_CELLS
    z = (np.arange(FV_CELLS) + 0.5) * dz
    u = initial_u(z)
    dt = SIM_TIME / FV_STEPS
    n = 2 * FV_CELLS
    for _ in range(FV_STEPS):
        u_face = 0.5 * (u[:-1] + u[1:])
        D = D_of_x(x_from_u(u_face)).copy()
        if variant == "no_cross":
            D[:, 0, 1] = 0.0
            D[:, 1, 0] = 0.0
        elif variant == "flipped_cross":
            D[:, 0, 1] *= -1.0
        elif variant == "x_frame":
            uc, un = u_face[:, 0], u_face[:, 1]
            jac = np.zeros((len(u_face), 2, 2))
            jac[:, 0, 0] = 1.0 / (1.0 + uc) ** 2
            jac[:, 1, 0] = -un / (1.0 + uc) ** 2
            jac[:, 1, 1] = 1.0 / (1.0 + uc)
            D = np.einsum("fij,fjk->fik", D, jac)
        coeff = D * dt / dz**2  # (faces, 2, 2)
        # Banded matrix in interleaved ordering [c0, n0, c1, n1, ...]; bandwidth 3.
        ab = np.zeros((7, n))
        rhs = u.reshape(-1).copy()

        def add(row, col, value):
            ab[3 + row - col, col] += value

        for i in range(FV_CELLS):
            for a in range(2):
                add(2 * i + a, 2 * i + a, 1.0)
        for f in range(FV_CELLS - 1):
            left, right = f, f + 1
            for a in range(2):
                for b in range(2):
                    c = coeff[f, a, b]
                    add(2 * left + a, 2 * left + b, c)
                    add(2 * left + a, 2 * right + b, -c)
                    add(2 * right + a, 2 * right + b, c)
                    add(2 * right + a, 2 * left + b, -c)
        u = solve_banded((3, 3), ab, rhs).reshape(FV_CELLS, 2)
    return z, u


def main():
    from tc_python import TCPython

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_c = np.linspace(0.6 * U_C_MEAN, 1.6 * U_C_MEAN, 21)
    x_ni = np.linspace(0.7 * (U_NI_MEAN - U_NI_AMP), 1.15 * (U_NI_MEAN + U_NI_AMP), 21)
    with TCPython() as session:
        system = (
            session.set_cache_folder(str(OUTPUT_DIR / "tc_cache"))
            .select_thermodynamic_and_kinetic_databases_with_elements("FEDEMO", "MFEDEMO", ["Fe", "Ni", "C"])
            .without_default_phases()
            .select_phase("FCC_A1")
            .get_system()
        )
        table = tc_dc_table(system, x_c, x_ni)
        print("DC table done", flush=True)
        z_dictra, u_dictra = run_dictra(system)
        print("DICTRA done", flush=True)

    interpolator = RegularGridInterpolator((x_c, x_ni), table, bounds_error=False, fill_value=None)
    report = {
        "temperature_K": TEMPERATURE,
        "time_s": SIM_TIME,
        "D_at_mean_composition": interpolator(x_from_u([[U_C_MEAN, U_NI_MEAN]]))[0].tolist(),
        "dictra_uC_range": [float(u_dictra[:, 0].min()), float(u_dictra[:, 0].max())],
        "initial_uC_range": [U_C_MEAN - U_C_AMP, U_C_MEAN + U_C_AMP],
    }
    for variant in ("u_frame", "x_frame", "no_cross", "flipped_cross"):
        z_fv, u_fv = solve_fv(interpolator, variant)
        errors = {}
        for k, name in enumerate(("C", "Ni")):
            err = np.interp(z_dictra, z_fv, u_fv[:, k]) - u_dictra[:, k]
            errors[f"max_abs_error_u_{name}"] = float(np.max(np.abs(err)))
        report[variant] = errors
    out = OUTPUT_DIR / "tc_interstitial_frame_check_ternary.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
