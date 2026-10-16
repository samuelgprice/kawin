"""DICTRA exb5 (gamma/alpha/gamma Fe-Cr-Ni diffusion couple) against the ternary Illingworth models.

DICTRA example exb5 (Thermo-Calc ``Examples/Console-Mode/Diffusion Module -
DICTRA/exb5``; Kajihara, Lim and Kikuchi, ISIJ Int. 33 (1993) 498, couple A):
a thin alpha (BCC) slice of Fe-38Cr-0Ni (wt%) clamped between two thicker
gamma (FCC) slices of Fe-27Cr-28Ni, annealed at 1373 K for 1000 h. The macro
says "27%Cr, 20%Ni" in its header but enters ``ni lin .28 .28``; DICTRA runs
the macro, so 28 wt% Ni is used here. DICTRA simulates half of the symmetric
couple: alpha 93.45 um (half of l0 = 186.9 um, closed boundary at the
symmetry plane) next to 500 um of gamma, both with ``AUTO`` grids, FEDEMO +
MFEDEMO, default time-step control.

Compared here (all with FEDEMO/MFEDEMO data from Thermo-Calc):

* ``dictra``: exb5 exactly as written (``CalculatedGrid.automatic()``,
  default time-step control);
* ``dictra-10h`` and ``dictra-100h``: the same with the simulation stopped at
  10 h or 100 h. DICTRA's default time steps scale with the simulated time,
  and its 10 h thickness from the 1000 h run is 1.2% above that of the 10 h
  run, so these shorter horizons are the accurate DICTRA values at those
  times. Tight ``TimestepControl`` (``run_dictra(..., tight=True)``) did not
  finish in 50 min (1000 h) or 30 min (10 h);
* ``kawin2``: the two-phase ternary Illingworth model on the same half couple;
* ``kawin3``: the three-phase model on the full gamma|alpha|gamma couple;
* the experimental normalized thicknesses ``2 s / l0`` from ``exb5.exp``;
* an asymmetric variant (not a DICTRA example): the right gamma slice is
  Fe-25Cr-20Ni, simulated as a three-region DICTRA couple (``dictra-asym``,
  ``dictra-asym-tight-10h``) and with the three-phase model (``kawin3-asym``).

kawin uses BCC/FCC tie-lines from Thermo-Calc equilibria on a grid of global
compositions (sorted by ``eta`` = BCC x_Cr, which rises monotonically along
the field while BCC x_Ni peaks near x_Cr = 0.7; checked for crossing) and DC
matrices tabulated on (x_Cr, x_Ni) grids per phase, interpolated bilinearly.

Stages (``--stage``): ``tc-inputs``, one of the names above, or ``all``;
every result is cached in ``outputs/exb5/``.
Run from the repo root through the conda wrapper in AGENTS.md::

    python -u -m examples.ternaryExamples.exb5_gamma_alpha_gamma --stage all
"""

from __future__ import annotations

import argparse
import json
import time as wallclock
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from kawin.diffusion import (
    CallableTernaryInterfaceEquilibrium,
    MovingBoundaryIllingworthTernaryFD1DModel,
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
)
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.TransformedGrids import landau_grid
from kawin.solver import explicitEulerIterator

OUTPUT_DIR = Path("examples") / "ternaryExamples" / "outputs" / "exb5"
MOLAR_MASS = {"FE": 55.845, "CR": 51.996, "NI": 58.693}
TEMPERATURE = 1373.0
SIM_TIME = 3.6e6
L0 = 186.9e-6  # initial alpha thickness of the full couple
ALPHA_HALF = 93.45e-6
GAMMA_WIDTH = 500.0e-6
W_ALPHA = {"CR": 0.38, "NI": 1.0e-5}  # weight fractions, as in the macro
W_GAMMA = {"CR": 0.27, "NI": 0.28}
PROFILE_TIMES = (3600.0, 36000.0, 360000.0, 3.6e6)
HORIZONS = (36000.0, 360000.0)  # shorter DICTRA horizons (10 h, 100 h): default steps scale with the horizon
# exb5.exp dataset 7: log10(t / l0^2) and 2 s / l0 for couple A.
EXPERIMENT = {"log10_t_over_l0sq": [11.01, 12.01, 13.01, 14.01], "thickness_ratio": [1.11, 1.33, 1.44, 1.15]}


def _log(message):
    print(f"[{wallclock.strftime('%H:%M:%S')}] {message}", flush=True)


def mole_fractions(w):
    """(x_Cr, x_Ni) from weight fractions of Cr and Ni (balance Fe)."""
    n = {"CR": w["CR"] / MOLAR_MASS["CR"], "NI": w["NI"] / MOLAR_MASS["NI"], "FE": (1.0 - w["CR"] - w["NI"]) / MOLAR_MASS["FE"]}
    total = sum(n.values())
    return np.array([n["CR"] / total, n["NI"] / total])


def _tc_system(session):
    return (
        session.set_cache_folder(str(OUTPUT_DIR / "tc_cache"))
        .select_thermodynamic_and_kinetic_databases_with_elements("FEDEMO", "MFEDEMO", ["Fe", "Cr", "Ni"])
        .without_default_phases()
        .select_phase("FCC_A1")
        .select_phase("BCC_A2")
        .get_system()
    )


def tc_tie_lines(system, x_cr_values, x_ni_values, max_bcc_cr=0.65):
    """BCC/FCC tie-lines from two-phase equilibria on a global-composition grid, sorted by BCC x_Cr (<= ``max_bcc_cr``)."""
    calc = system.with_single_equilibrium_calculation().set_condition("T", TEMPERATURE)
    records = []
    for x_cr in x_cr_values:
        for x_ni in x_ni_values:
            if x_cr + x_ni >= 0.95:
                continue
            try:
                calc.set_condition("X(CR)", float(x_cr)).set_condition("X(NI)", float(x_ni))
                result = calc.calculate()
                stable = {p.upper().split("#")[0] for p in result.get_stable_phases()}
                if stable != {"BCC_A2", "FCC_A1"}:
                    continue
                bcc = [float(result.get_value_of(f"X(BCC_A2,{e})")) for e in ("CR", "NI")]
                fcc = [float(result.get_value_of(f"X(FCC_A1,{e})")) for e in ("CR", "NI")]
            except Exception as exc:  # noqa: BLE001 - a failed grid point is skipped
                _log(f"equilibrium at x_Cr={x_cr:.4g}, x_Ni={x_ni:.4g} failed: {exc}")
                continue
            if bcc[0] <= max_bcc_cr:
                records.append((bcc[0], bcc, fcc))
    records.sort(key=lambda r: r[0])
    etas, bcc, fcc = [], [], []
    for eta, b, f in records:
        if etas and eta - etas[-1] <= 1e-7:
            continue
        etas.append(eta)
        bcc.append(b)
        fcc.append(f)
    return np.asarray(etas), np.asarray(bcc), np.asarray(fcc)


def tc_dc_grid(system, phase, x_cr_values, x_ni_values):
    """DC matrices (rows/cols Cr, Ni; reference Fe) of ``phase`` on the (x_Cr, x_Ni) grid."""
    from tc_python import ALL_PHASES

    calc = (
        system.with_single_equilibrium_calculation()
        .set_condition("T", TEMPERATURE)
        .set_phase_to_suspended(ALL_PHASES)
        .set_phase_to_entered(phase, 1.0)
    )
    out = np.empty((len(x_cr_values), len(x_ni_values), 2, 2))
    for i, x_cr in enumerate(x_cr_values):
        for j, x_ni in enumerate(x_ni_values):
            calc.set_condition("X(CR)", float(x_cr)).set_condition("X(NI)", float(x_ni))
            result = calc.calculate()
            for a, k in enumerate(("CR", "NI")):
                for b, g in enumerate(("CR", "NI")):
                    out[i, j, a, b] = float(result.get_value_of(f"DC({phase},{k},{g},FE)"))
    return out


def build_tc_inputs():
    """Tie-lines and DC tables (cached)."""
    cache = OUTPUT_DIR / "tc_inputs.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from tc_python import TCPython

    with TCPython() as session:
        system = _tc_system(session)
        _log("sampling BCC/FCC tie-lines")
        etas, bcc, fcc = tc_tie_lines(system, np.linspace(0.15, 0.55, 33), np.linspace(0.0, 0.40, 33))
        _log(f"{len(etas)} tie-lines; BCC x_Cr in [{etas.min():.4g}, {etas.max():.4g}]")
        bcc_cr = np.linspace(0.25, 0.50, 11)
        bcc_ni = np.linspace(0.0, 0.12, 9)
        fcc_cr = np.linspace(0.15, 0.40, 11)
        fcc_ni = np.linspace(0.05, 0.35, 11)
        _log("tabulating BCC_A2 diffusivities")
        bcc_d = tc_dc_grid(system, "BCC_A2", bcc_cr, bcc_ni)
        _log("tabulating FCC_A1 diffusivities")
        fcc_d = tc_dc_grid(system, "FCC_A1", fcc_cr, fcc_ni)
    data = {
        "tie_lines": {"eta": etas.tolist(), "bcc": bcc.tolist(), "fcc": fcc.tolist()},
        "tables": {
            "BCC_A2": {"x_cr": bcc_cr.tolist(), "x_ni": bcc_ni.tolist(), "D": bcc_d.tolist()},
            "FCC_A1": {"x_cr": fcc_cr.tolist(), "x_ni": fcc_ni.tolist(), "D": fcc_d.tolist()},
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    _log(f"wrote {cache}")
    return data


def run_dictra(regions, *, tight, sim_time=SIM_TIME, profile_times=PROFILE_TIMES, grid_points=None):
    """
    Runs a planar DICTRA couple; ``regions`` is a list of ``(name, phase, width, w_cr, w_ni)`` (mass fractions).

    Every region uses ``CalculatedGrid.automatic()`` (the console ``AUTO``
    grid), or an n-point linear grid with ``grid_points=n`` (refinement
    checks). ``tight`` adds ``TimestepControl`` (relative error 1e-3, growth
    factor 1.2, at most 1% of the simulated time per step); otherwise
    DICTRA's default time-step control is used, as in the exb5 macro. Returns
    the lower/upper boundary positions of every region and mole-fraction
    profiles of Cr and Ni at ``profile_times``.
    """
    from tc_python import CalculatedGrid, CompositionProfile, ElementProfile, Region, TCPython, TimestepControl

    start = wallclock.perf_counter()
    with TCPython() as session:
        calculator = system = _tc_system(session)
        calculator = system.with_isothermal_diffusion_calculation().set_temperature(TEMPERATURE).set_simulation_time(sim_time)
        for name, phase, width, w_cr, w_ni in regions:
            grid = CalculatedGrid.automatic() if grid_points is None else CalculatedGrid.linear().set_no_of_points(int(grid_points))
            composition = CompositionProfile().add("Cr", ElementProfile.constant(100.0 * w_cr)).add("Ni", ElementProfile.constant(100.0 * w_ni))
            calculator = calculator.add_region(
                Region(name).set_width(width).with_grid(grid).with_composition_profile(composition).add_phase(phase)
            )
        if tight:
            calculator = calculator.with_timestep_control(
                TimestepControl()
                .set_max_relative_error(1.0e-3)
                .set_max_absolute_error(1.0e-7)
                .set_max_timestep_increase_factor(1.2)
                .set_max_timestep_allowed_as_percent_of_simulation_time(1.0)
            )
        result = calculator.calculate()
        out = {"regions": [list(r) for r in regions], "tight": bool(tight), "boundaries": {}, "profiles": {}}
        for name, *_ in regions:
            t_low, low = result.get_position_of_lower_boundary_of_region(name)
            t_up, up = result.get_position_of_upper_boundary_of_region(name)
            out["boundaries"][name] = {"time": list(map(float, t_low)), "lower": list(map(float, low)), "upper": list(map(float, up))}
        for t in profile_times:
            entry = {}
            for element in ("Cr", "Ni"):
                z, x = result.get_mole_fraction_of_component_at_time(element, float(t))
                entry["z"] = list(map(float, z))
                entry[f"x_{element.upper()}"] = list(map(float, x))
            out["profiles"][f"{t:g}"] = entry
    out["wall_s"] = round(wallclock.perf_counter() - start, 1)
    return out


EXB5_REGIONS = [
    ("alpha", "BCC_A2", ALPHA_HALF, W_ALPHA["CR"], W_ALPHA["NI"]),
    ("gamma", "FCC_A1", GAMMA_WIDTH, W_GAMMA["CR"], W_GAMMA["NI"]),
]
# Asymmetric variant (not a DICTRA example): the right gamma slice is
# Fe-25Cr-20Ni (single-phase FCC at 1373 K), so the two interfaces differ.
W_GAMMA_RIGHT_ASYM = {"CR": 0.25, "NI": 0.20}
ASYM_REGIONS = [
    ("gammaL", "FCC_A1", GAMMA_WIDTH, W_GAMMA["CR"], W_GAMMA["NI"]),
    ("alpha", "BCC_A2", L0, W_ALPHA["CR"], W_ALPHA["NI"]),
    ("gammaR", "FCC_A1", GAMMA_WIDTH, W_GAMMA_RIGHT_ASYM["CR"], W_GAMMA_RIGHT_ASYM["NI"]),
]


def cached(name, build):
    """Loads ``outputs/exb5/<name>.json`` or builds and writes it."""
    path = OUTPUT_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    _log(f"building {name}")
    value = build()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    _log(f"wrote {path}")
    return value


class TabulatedFeCrNiDiffusivity:
    """Bilinear (x_Cr, x_Ni) interpolation of Thermo-Calc DC matrices per phase; mole-fraction input."""

    def __init__(self, tables):
        self.interpolators = {
            phase: RegularGridInterpolator((np.asarray(t["x_cr"]), np.asarray(t["x_ni"])), np.asarray(t["D"]), bounds_error=False, fill_value=None)
            for phase, t in tables.items()
        }

    def clearCache(self):
        pass

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        points = np.atleast_2d(np.asarray(x, dtype=np.float64))
        out = self.interpolators[phase](points)
        return out[0] if np.asarray(x).ndim == 1 else out


def alpha_gamma_closure(data, flipped=False):
    """Linear interpolation in ``eta`` = BCC x_Cr; ``(bcc, fcc)``, or ``(fcc, bcc)`` when ``flipped``."""
    tie = data["tie_lines"]
    eta, bcc, fcc = (np.asarray(tie[k]) for k in ("eta", "bcc", "fcc"))

    def interface(e):
        b = np.array([np.interp(e, eta, bcc[:, k]) for k in range(2)])
        f = np.array([np.interp(e, eta, fcc[:, k]) for k in range(2)])
        return (f, b) if flipped else (b, f)

    return CallableTernaryInterfaceEquilibrium(interface, eta_bounds=(float(eta[0]), float(eta[-1])))


KAWIN_OPTIONS = dict(
    time_step=SIM_TIME,
    dt_mode="semi_log",
    bulk_diffusivity_mode="composition_dependent_lagged",
    tolerance=1.0e-10,
    max_iterations=50,
    max_step_retries=12,
    terminal_thin_phase_width=1.0e-12,
    terminal_thin_phase_policy="continue",
    record=True,
)


def run_kawin2(data, *, n_alpha=81, n_gamma=201, power=2.0, semi_log_dt=0.01, semi_log_t0=1.0e-2):
    """Two-phase model on DICTRA's half couple: alpha [0, 93.45 um] (closed at the symmetry plane) | gamma."""
    x_alpha, x_gamma = mole_fractions(W_ALPHA), mole_fractions(W_GAMMA)
    R = ALPHA_HALF + GAMMA_WIDTH
    mesh = CartesianFD1D(["CR", "NI"], [0.0, R], 4001)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(ALPHA_HALF, x_alpha, x_gamma), ["CR", "NI"])]))
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh, elements=["FE", "CR", "NI"], phases=["BCC_A2", "FCC_A1"],
        thermodynamics=TabulatedFeCrNiDiffusivity(data["tables"]), temperature=TEMPERATURE,
        interfacePosition=ALPHA_HALF, interface_equilibrium=alpha_gamma_closure(data),
        semiLog_dt=semi_log_dt, semiLogT0=semi_log_t0,
        transformed_u_grid=landau_grid(n_alpha, "power", phase="left", exponent=1.5),
        transformed_v_grid=landau_grid(n_gamma, "power", phase="right", exponent=power),
        **KAWIN_OPTIONS,
    )
    start = wallclock.perf_counter()
    model.solve(SIM_TIME, iterator=explicitEulerIterator, minDtFrac=1.0e-14)
    n = model.interfaceData.N + 1
    (z_a, c_a), (z_g, c_g) = model.getPhysicalPhaseProfiles()
    return {
        "time": model.interfaceData._time[:n].tolist(), "alpha_half": np.ravel(model.interfaceData._y[:n]).tolist(),
        "eta": np.ravel(model.etaData._y[:n]).tolist(), "wall_s": round(wallclock.perf_counter() - start, 1), "steps": n - 1,
        "final_profile": {"z": np.concatenate([z_a, z_g]).tolist(), "x_CR": np.concatenate([c_a[:, 1], c_g[:, 1]]).tolist(), "x_NI": np.concatenate([c_a[:, 2], c_g[:, 2]]).tolist()},
        "settings": {"n_alpha": n_alpha, "n_gamma": n_gamma, "power": power, "semi_log_dt": semi_log_dt, "semi_log_t0": semi_log_t0},
    }


def run_kawin3(data, *, widths=(GAMMA_WIDTH, L0, GAMMA_WIDTH), w_left=W_GAMMA, w_right=W_GAMMA, n_alpha=161, n_gamma=201, power=2.0, semi_log_dt=0.01, semi_log_t0=1.0e-2):
    """Three-phase model on a full gamma|alpha|gamma couple (exb5: both gamma slices equal)."""
    x_alpha, x_left, x_right = mole_fractions(W_ALPHA), mole_fractions(w_left), mole_fractions(w_right)
    s_ab, s_bc = widths[0], widths[0] + widths[1]
    R = float(sum(widths))
    mesh = CartesianFD1D(["CR", "NI"], [0.0, R], 8001)

    def initial(z):
        x = np.asarray(z, dtype=np.float64).reshape(-1)
        out = np.empty((len(x), 2))
        out[:] = x_alpha
        out[x < s_ab] = x_left
        out[x >= s_bc] = x_right
        return out

    mesh.setResponseProfile(ProfileBuilder([(initial, ["CR", "NI"])]))
    model = MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=mesh, elements=["FE", "CR", "NI"], phases=["FCC_A1", "BCC_A2", "FCC_A1"],
        thermodynamics=TabulatedFeCrNiDiffusivity(data["tables"]), temperature=TEMPERATURE,
        interfacePositions=(s_ab, s_bc), interface_equilibria=(alpha_gamma_closure(data, flipped=True), alpha_gamma_closure(data)),
        semiLog_dt=semi_log_dt, semiLogT0=semi_log_t0,
        transformed_grids=(
            landau_grid(n_gamma, "power", phase="left", exponent=power),
            landau_grid(n_alpha, "tanh", phase="middle", beta=3.0),
            landau_grid(n_gamma, "power", phase="right", exponent=power),
        ),
        **KAWIN_OPTIONS,
    )
    start = wallclock.perf_counter()
    model.solve(SIM_TIME, iterator=explicitEulerIterator, minDtFrac=1.0e-14)
    n = model.interfaceData.N + 1
    s = np.asarray(model.interfaceData._y[:n])
    profiles = model.getPhysicalPhaseProfiles()
    return {
        "time": model.interfaceData._time[:n].tolist(), "interfaces": s.tolist(), "etas": np.asarray(model.etaData._y[:n]).tolist(),
        "wall_s": round(wallclock.perf_counter() - start, 1), "steps": n - 1,
        "final_profile": {"z": np.concatenate([p[0] for p in profiles]).tolist(), "x_CR": np.concatenate([p[1][:, 1] for p in profiles]).tolist(), "x_NI": np.concatenate([p[1][:, 2] for p in profiles]).tolist()},
        "settings": {"widths": list(widths), "w_left": w_left, "w_right": w_right, "n_alpha": n_alpha, "n_gamma": n_gamma, "power": power, "semi_log_dt": semi_log_dt, "semi_log_t0": semi_log_t0},
    }


def asymmetric_report(results, times=(360.0, 3600.0, 36000.0, 360000.0, 3.6e6)):
    """Displacements (um) of the left (gammaL|alpha) and right (alpha|gammaR) interfaces from their start (NaN past a run's end)."""
    times = np.asarray(times)
    report = {"time_s": times.tolist()}
    for name, res in results.items():
        if "boundaries" in res:
            b = res["boundaries"]["alpha"]
            left, right = (np.interp(times, b["time"], np.asarray(b[k]) - b[k][0], right=np.nan) for k in ("lower", "upper"))
        else:
            s = np.asarray(res["interfaces"])
            left, right = (np.interp(times, res["time"], s[:, k] - s[0, k], right=np.nan) for k in (0, 1))
        report[name] = {"left_um": (1e6 * left).tolist(), "right_um": (1e6 * right).tolist(), "wall_s": res.get("wall_s")}
    return report


def thickness_report(results):
    """Normalized alpha thickness ``2 s / l0`` at the experimental times (NaN past a run's end)."""
    times = (10.0 ** np.asarray(EXPERIMENT["log10_t_over_l0sq"])) * L0**2
    report = {"time_s": times.tolist(), "experiment": EXPERIMENT["thickness_ratio"]}
    for name, res in results.items():
        if res is None or "asym" in name:
            continue
        if "boundaries" in res:  # DICTRA half couple: alpha upper boundary
            b = res["boundaries"]["alpha"]
            half = np.interp(times, b["time"], np.asarray(b["upper"]) - np.asarray(b["lower"]), right=np.nan)
        elif "alpha_half" in res:
            half = np.interp(times, res["time"], res["alpha_half"])
        else:
            s = np.asarray(res["interfaces"])
            half = 0.5 * np.interp(times, res["time"], s[:, 1] - s[:, 0])
        report[name] = (2.0 * half / L0).tolist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    stage_names = ["dictra", "dictra-10h", "dictra-100h", "kawin2", "kawin3", "dictra-asym", "dictra-asym-10h", "dictra-asym-100h", "kawin3-asym"]
    parser.add_argument("--stage", default="all", choices=["tc-inputs", "all"] + stage_names)
    args = parser.parse_args()
    data = build_tc_inputs()
    stages = stage_names if args.stage == "all" else [args.stage]
    builders = {
        "dictra": lambda: run_dictra(EXB5_REGIONS, tight=False),
        "dictra-10h": lambda: run_dictra(EXB5_REGIONS, tight=False, sim_time=HORIZONS[0], profile_times=(HORIZONS[0],)),
        "dictra-100h": lambda: run_dictra(EXB5_REGIONS, tight=False, sim_time=HORIZONS[1], profile_times=(HORIZONS[1],)),
        "kawin2": lambda: run_kawin2(data),
        "kawin3": lambda: run_kawin3(data),
        "dictra-asym": lambda: run_dictra(ASYM_REGIONS, tight=False),
        "dictra-asym-10h": lambda: run_dictra(ASYM_REGIONS, tight=False, sim_time=HORIZONS[0], profile_times=(HORIZONS[0],)),
        "dictra-asym-100h": lambda: run_dictra(ASYM_REGIONS, tight=False, sim_time=HORIZONS[1], profile_times=(HORIZONS[1],)),
        "kawin3-asym": lambda: run_kawin3(data, w_right=W_GAMMA_RIGHT_ASYM),
    }
    results = {}
    for stage in stages:
        if stage in builders:
            results[stage] = cached(stage, builders[stage])
            _log(f"{stage}: wall {results[stage].get('wall_s')} s")
    for name in builders:
        path = OUTPUT_DIR / f"{name}.json"
        if name not in results and path.exists():
            results[name] = json.loads(path.read_text())
    report = thickness_report(results)
    (OUTPUT_DIR / "thickness_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    asym = {name: res for name, res in results.items() if "asym" in name}
    if asym:
        report = asymmetric_report(asym)
        (OUTPUT_DIR / "asymmetric_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
