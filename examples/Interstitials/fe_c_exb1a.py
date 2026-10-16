"""Fe-C ferrite growth: binary Illingworth (u-fractions) vs DICTRA exb1a / pyex_D_02.

Setup (DICTRA example exb1a, also TC-Python pyex_D_02): FEDEMO + MFEDEMO, Fe-C,
FCC_A1 and BCC_A2 only, 1050 K, closed planar system. A 1e-9 m ferrite region
(Fe-0.01 wt% C) is placed next to 2 mm of austenite (Fe-0.15 wt% C).
``--case exb1a`` runs 1e9 s; ``--case pyex_D_02`` runs 30 h.

kawin side
----------
``MovingBoundaryIllingworthFD1DModel`` with elements ``["FE", "C"]``, so the
state is u_C = x_C / (1 - x_C). The fixed local-equilibrium interface
compositions come from a Thermo-Calc FCC+BCC equilibrium. Diffusivities are
Thermo-Calc's ``DC(phase,C,C,FE)`` tabulated once against x_C per phase
(``composition_dependent_lagged`` face values interpolated from the table, so
no Thermo-Calc call happens inside the time loop). Thermo-Calc's DC uses the
same volume-fixed u-gradient frame as kawin
(``check_tc_interstitial_frame.py``).

DICTRA time stepping
--------------------
With DICTRA's default time-step control the exb1a run takes only ~44 steps up
to 1e4 s and under-predicts the parabolic rate constant by ~15%; refining the
austenite grid does not change that, tightening the time-step control does.
The reference run therefore uses ``TimestepControl`` with a relative error of
1e-3, a step-growth factor of 1.2, interface-position checking and a maximum
step of 1% of the simulation time (1e-4 / 1.05 gives the same answer). The
default-settings run is kept as ``dictra_default`` for comparison. Even the
converged DICTRA run keeps a slowly decaying start-up transient from the 1e-9 m
ferrite seed: its s/sqrt(t) rises from ~1.93e-6 at 100 s to ~1.99e-6 at 1e4 s,
where it meets kawin's converged ~1.98e-6.

Outputs land in ``examples/Interstitials/outputs``: the DICTRA and kawin
ferrite thickness histories, carbon profiles, a comparison JSON and a PNG.
Expected runtime: about 5-15 min on first use (the tight DICTRA run dominates),
under a minute afterwards; the Thermo-Calc inputs are cached in a JSON file.

Run (repo root, through the conda activation wrapper in AGENTS.md):
    python -m examples.Interstitials.fe_c_exb1a --case exb1a
"""

from __future__ import annotations

import argparse
import json
import time as wallclock
from pathlib import Path

import numpy as np

from kawin.diffusion import MovingBoundaryIllingworthFD1DModel
from kawin.diffusion.DiffusionParameters import TemperatureParameters
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.solver import explicitEulerIterator

OUTPUT_DIR = Path("examples") / "Interstitials" / "outputs"
TEMPERATURE = 1050.0
FERRITE_WIDTH = 1.0e-9
AUSTENITE_WIDTH = 2.0e-3
W_C_FERRITE = 0.01
W_C_AUSTENITE = 0.15
M_C, M_FE = 12.011, 55.845
CASES = {"exb1a": 1.0e9, "pyex_D_02": 30 * 3600.0}
PROFILE_TIMES = (1.0e3, 1.0e5, 1.0e9)


def x_from_wt(w_percent):
    n_c = w_percent / M_C
    return n_c / (n_c + (100.0 - w_percent) / M_FE)


def u_from_x(x):
    return np.asarray(x, dtype=np.float64) / (1.0 - np.asarray(x, dtype=np.float64))


class TabulatedBinaryDiffusivity:
    """``getInterdiffusivity`` from per-phase DC(x_C) tables (mole-fraction input)."""

    def __init__(self, tables):
        self.tables = {phase: (np.asarray(x), np.asarray(d)) for phase, (x, d) in tables.items()}

    def clearCache(self):
        pass

    def getInterdiffusivity(self, x, T, phase=None, query_context=None):
        x_table, d_table = self.tables[phase]
        x = np.asarray(x, dtype=np.float64)
        if np.any(x < x_table[0] - 1e-12) or np.any(x > x_table[-1] + 1e-12):
            raise ValueError(f"x_C={x} outside the {phase} diffusivity table [{x_table[0]}, {x_table[-1]}].")
        return np.squeeze(np.exp(np.interp(x, x_table, np.log(d_table))))


def _tc_system(session):
    return (
        session.set_cache_folder(str(OUTPUT_DIR / "tc_cache"))
        .select_thermodynamic_and_kinetic_databases_with_elements("FEDEMO", "MFEDEMO", ["Fe", "C"])
        .without_default_phases()
        .select_phase("FCC_A1")
        .select_phase("BCC_A2")
        .get_system()
    )


def tc_local_equilibrium(system):
    """Returns (x_C in BCC, x_C in FCC) at the two-phase local equilibrium."""
    calc = (
        system.with_single_equilibrium_calculation()
        .set_condition("T", TEMPERATURE)
        .set_condition("X(C)", 0.5 * (x_from_wt(0.02) + x_from_wt(0.6)))
    )
    result = calc.calculate()
    stable = [p.upper().split("#")[0] for p in result.get_stable_phases()]
    if not {"BCC_A2", "FCC_A1"} <= set(stable):
        raise RuntimeError(f"Expected BCC_A2+FCC_A1 at the tie-line query, got {stable}.")
    return (
        float(result.get_value_of("X(BCC_A2,C)")),
        float(result.get_value_of("X(FCC_A1,C)")),
    )


def tc_dc_table(system, phase, x_values):
    from tc_python import ALL_PHASES

    calc = (
        system.with_single_equilibrium_calculation()
        .set_condition("T", TEMPERATURE)
        .set_phase_to_suspended(ALL_PHASES)
        .set_phase_to_entered(phase, 1.0)
    )
    values = []
    for x in x_values:
        calc.set_condition("X(C)", float(x))
        values.append(float(calc.calculate().get_value_of(f"DC({phase},C,C,FE)")))
    return np.asarray(values, dtype=np.float64)


def run_dictra(system, sim_time, n_austenite, tight=True):
    """Runs the DICTRA reference; ``tight`` selects the converged time-step control."""
    from tc_python import CalculatedGrid, CompositionProfile, ElementProfile, Region, TimestepControl

    calculator = (
        system.with_isothermal_diffusion_calculation()
        .set_temperature(TEMPERATURE)
        .set_simulation_time(sim_time)
        .add_region(
            Region("Ferrite")
            .set_width(FERRITE_WIDTH)
            .with_grid(CalculatedGrid.linear().set_no_of_points(10))
            .with_composition_profile(CompositionProfile().add("C", ElementProfile.constant(W_C_FERRITE)))
            .add_phase("BCC_A2")
        )
        .add_region(
            Region("Austenite")
            .set_width(AUSTENITE_WIDTH)
            .with_grid(CalculatedGrid.geometric().set_no_of_points(n_austenite).set_geometrical_factor(1.05))
            .with_composition_profile(CompositionProfile().add("C", ElementProfile.constant(W_C_AUSTENITE)))
            .add_phase("FCC_A1")
        )
    )
    if tight:
        calculator = calculator.with_timestep_control(
            TimestepControl()
            .set_max_relative_error(1.0e-3)
            .set_max_absolute_error(1.0e-7)
            .set_max_timestep_increase_factor(1.2)
            .set_max_timestep_allowed_as_percent_of_simulation_time(1.0)
            .enable_check_interface_position()
        )
    result = calculator.calculate()
    times, positions = result.get_position_of_upper_boundary_of_region("Ferrite")
    profiles = {}
    for t in PROFILE_TIMES:
        if t <= sim_time:
            z, x_c = result.get_mole_fraction_of_component_at_time("C", float(t))
            profiles[str(t)] = {"z": list(map(float, z)), "u_C": u_from_x(np.asarray(x_c)).tolist()}
    return {"time": list(map(float, times)), "position": list(map(float, positions)), "profiles": profiles}


def build_tc_inputs(case, n_austenite):
    """Computes (or loads) the DICTRA reference, LE compositions and DC tables."""
    cache = OUTPUT_DIR / f"fe_c_{case}_tc_inputs_v2.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from tc_python import TCPython

    sim_time = CASES[case]
    with TCPython() as session:
        system = _tc_system(session)
        x_bcc, x_fcc = tc_local_equilibrium(system)
        x_ferrite0, x_austenite0 = x_from_wt(W_C_FERRITE), x_from_wt(W_C_AUSTENITE)
        bcc_x = np.linspace(min(x_ferrite0, x_bcc) * 0.25, max(x_ferrite0, x_bcc) * 1.5, 60)
        fcc_x = np.linspace(x_austenite0 * 0.8, x_fcc * 1.05, 120)
        tables = {
            "BCC_A2": (bcc_x.tolist(), tc_dc_table(system, "BCC_A2", bcc_x).tolist()),
            "FCC_A1": (fcc_x.tolist(), tc_dc_table(system, "FCC_A1", fcc_x).tolist()),
        }
        dictra = run_dictra(system, sim_time, n_austenite, tight=True)
        dictra_default = run_dictra(system, sim_time, n_austenite, tight=False)
    data = {"x_bcc_eq": x_bcc, "x_fcc_eq": x_fcc, "tables": tables, "dictra": dictra, "dictra_default": dictra_default}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def run_kawin(data, sim_time, *, n_left=41, n_right=1601, semi_log_dt=0.02, mode="composition_dependent_lagged"):
    R = FERRITE_WIDTH + AUSTENITE_WIDTH
    s0 = FERRITE_WIDTH
    x_ferrite0, x_austenite0 = x_from_wt(W_C_FERRITE), x_from_wt(W_C_AUSTENITE)
    mesh = CartesianFD1D(["C"], [0.0, R], 4001)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(s0, x_ferrite0, x_austenite0), "C")]))
    # Cluster austenite nodes at the interface where the carbon spike forms.
    v_grid = np.linspace(0.0, 1.0, n_right) ** 2
    therm = TabulatedBinaryDiffusivity({phase: tuple(v) for phase, v in data["tables"].items()})
    model = MovingBoundaryIllingworthFD1DModel(
        mesh,
        ["FE", "C"],
        ["BCC_A2", "FCC_A1"],
        thermodynamics=therm,
        temperature=TemperatureParameters(TEMPERATURE),
        interfacePosition=s0,
        interface_compositions=(data["x_bcc_eq"], data["x_fcc_eq"]),
        interface_composition_basis="x",
        time_step=sim_time,
        dt_mode="semi_log",
        semiLog_dt=semi_log_dt,
        semiLogT0=1.0e-6,
        phase_a_nodes=n_left,
        transformed_v_grid=v_grid,
        bulk_diffusivity_mode=mode,
        max_step_retries=8,
        tolerance=1.0e-14,
        record=True,
    )
    model.solve(sim_time, iterator=explicitEulerIterator, minDtFrac=1e-20)
    return model


def compare(model, data, sim_time):
    t_dictra = np.asarray(data["dictra"]["time"])
    s_dictra = np.asarray(data["dictra"]["position"])
    t_kawin = model.interfaceData._time[: model.interfaceData.N + 1]
    s_kawin = model.interfaceData._y[: model.interfaceData.N + 1]
    mask = (t_dictra >= 100.0) & (t_dictra <= sim_time)
    s_on_dictra = np.exp(np.interp(np.log(t_dictra[mask]), np.log(np.maximum(t_kawin, 1e-30)), np.log(s_kawin)))
    rel = (s_on_dictra - s_dictra[mask]) / s_dictra[mask]
    report = {
        "thickness_max_rel_error_t_ge_100s": float(np.max(np.abs(rel))),
        "thickness_rel_error_at_end": float(rel[-1]),
        "kawin_final_thickness": float(s_kawin[-1]),
        "dictra_final_thickness": float(s_dictra[mask][-1]),
        "kawin_inventory_drift": float(abs(model.getTotalInventory() - model.concData._y[0] * model._R)),
        "steps": int(model.interfaceData.N),
    }
    for t_probe in (1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9):
        if t_probe <= sim_time:
            report.setdefault("thickness_samples", {})[f"{t_probe:g}"] = {
                "dictra": float(np.interp(t_probe, t_dictra, s_dictra)),
                "dictra_default_timestep": float(
                    np.interp(t_probe, data["dictra_default"]["time"], data["dictra_default"]["position"])
                ),
                "kawin": float(model.interfaceData.y(t_probe)),
            }
    profile_errors = {}
    for key, ref in data["dictra"]["profiles"].items():
        t = float(key)
        z_ref = np.asarray(ref["z"])
        u_ref = np.asarray(ref["u_C"])
        u_kawin = np.ravel(model.data.y(t))
        z_kawin = np.ravel(model.mesh.z)
        # Skip the band between the two models' interfaces (plus a margin):
        # there one model is ferrite and the other austenite.
        s_kawin_t = model.interfaceData.y(t)
        s_dictra_t = float(np.interp(t, t_dictra, s_dictra))
        lo, hi = min(s_kawin_t, s_dictra_t), max(s_kawin_t, s_dictra_t)
        margin = 0.05 * hi
        away = (z_ref < lo - margin) | (z_ref > hi + margin)
        diff = np.interp(z_ref[away], z_kawin, u_kawin) - u_ref[away]
        profile_errors[key] = float(np.max(np.abs(diff)))
    report["profile_max_abs_error_u_away_from_interface"] = profile_errors
    histories = (t_dictra, s_dictra, t_kawin, s_kawin, np.asarray(data["dictra_default"]["time"]), np.asarray(data["dictra_default"]["position"]))
    return report, histories


def plot(histories, case):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_dictra, s_dictra, t_kawin, s_kawin, t_default, s_default = histories
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(t_dictra[t_dictra > 0], s_dictra[t_dictra > 0], "k-", label="DICTRA (tight time steps)")
    ax.loglog(t_default[t_default > 0], s_default[t_default > 0], color="0.6", linestyle=":", label="DICTRA (default time steps)")
    ax.loglog(t_kawin[1:], s_kawin[1:], "r--", label="kawin Illingworth (u)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ferrite thickness (m)")
    ax.set_title(f"Fe-C ferrite growth, {case}")
    ax.legend()
    fig.tight_layout()
    out = OUTPUT_DIR / f"fe_c_{case}_thickness.png"
    fig.savefig(out, dpi=150)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=sorted(CASES), default="exb1a")
    parser.add_argument("--mode", default="composition_dependent_lagged")
    args = parser.parse_args()
    sim_time = CASES[args.case]
    n_austenite = 100 if args.case == "exb1a" else 50
    data = build_tc_inputs(args.case, n_austenite)
    start = wallclock.perf_counter()
    model = run_kawin(data, sim_time, mode=args.mode)
    report, histories = compare(model, data, sim_time)
    report["kawin_runtime_s"] = wallclock.perf_counter() - start
    report["x_bcc_eq"], report["x_fcc_eq"] = data["x_bcc_eq"], data["x_fcc_eq"]
    report["plot"] = str(plot(histories, args.case))
    out = OUTPUT_DIR / f"fe_c_{args.case}_comparison.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
