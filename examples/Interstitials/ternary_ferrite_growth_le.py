"""Fe-C-X ferrite growth under local equilibrium: ternary Illingworth (u-fractions) vs DICTRA.

Shared driver for the Fe-Ni-C (``fe_ni_c_exh2_le.py``) and Fe-Mn-C
(``fe_mn_c_ferrite.py``) benchmarks. A thin ferrite seed (BCC_A2) next to an
austenite layer (FCC_A1) of the same Fe-C-X composition is held at constant
temperature in a closed planar system, with **local equilibrium** at the
interface (FEDEMO + MFEDEMO databases).

kawin side
----------
``MovingBoundaryIllingworthTernaryFD1DModel`` with elements ``["FE", "C", X]``
(state: u_C, u_X). Tie-lines come from Thermo-Calc equilibria on a grid of
global compositions; each two-phase result gives one tie-line, and the family
is parameterised by ``eta = X(BCC_A2,X)`` (ferrite X content), sorted and
interpolated linearly in eta, returned in mole fractions. Diffusivities are
Thermo-Calc ``DC`` matrices tabulated on a small (x_C, x_X) grid per phase and
interpolated bilinearly, so the model's composition-dependent face matrices
never call Thermo-Calc inside the time loop. Thermo-Calc's DC uses kawin's
volume-fixed u-gradient frame (``check_tc_interstitial_frame.py``).

Numerics: under local equilibrium with negligible X partitioning a spike of X
of width about D_X / v (sub-nanometre here) forms in austenite at the
interface. The austenite transformed grid is a strong power law at the
interface and the interface residual can be tightened for X with
``--x-residual-scale``. The startup eta balance uses the default (max-norm)
scaling; ``--eta-componentwise`` switches to per-component scaling (both
converge to the same root on the grids used here). The t = 0 balance
starts from a step profile and depends on the first cell size (it can even pick
a negative-velocity root); the first steps re-solve eta, so growth after
~1e-5 s is unaffected.
Per-step diagnostics (eta, branch, Peclet number, Jacobian condition number,
per-component residual, retries) are saved with the results.

Stages (``--stage``): ``tc-inputs`` (tie-lines and DC tables, ~700 single
equilibria, well under a minute), ``dictra`` (the DICTRA local-equilibrium
reference) and ``kawin``; the first two are cached separately and ``all`` runs
whatever is missing. Progress lines are printed with timestamps.

DICTRA reference settings: a 100-point geometric austenite grid (factor 1.05)
with ``TimestepControl`` relative error 1e-3, step growth 1.2 and a maximum
step of 1% of the simulated time. DICTRA's default time-step control gave a
parabolic constant ~10% low in Fe-Ni-C (as in Fe-C, see ``fe_c_exb1a.py``);
with these settings DICTRA's step stays near 1e-5-4e-5 s, so ``--dictra-sim-time``
(default 2 s) keeps the reference to about 1-2 h (Fe-Ni-C 77 min, Fe-Mn-C 122 min). Growth is parabolic, so
the comparison uses ``s/sqrt(t)`` and the thickness inside that window. A
200-point grid (factor 1.03) changed k(1 s) by 1.6% at ~30x the cost.
"""

from __future__ import annotations

import argparse
import json
import time as wallclock
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from kawin.diffusion import CallableTernaryInterfaceEquilibrium, MovingBoundaryIllingworthTernaryFD1DModel
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.TransformedGrids import landau_grid
from kawin.solver import explicitEulerIterator

OUTPUT_DIR = Path("examples") / "Interstitials" / "outputs"
MOLAR_MASS = {"FE": 55.845, "C": 12.011, "NI": 58.693, "MN": 54.938}


@dataclass(frozen=True)
class FerriteGrowthCase:
    """One Fe-C-X ferrite-growth benchmark (compositions in wt%, SI otherwise)."""

    name: str
    solute: str  # substitutional element X, upper case
    temperature: float
    w_c: float
    w_x: float
    sim_time: float
    ferrite_width: float = 1.0e-9
    austenite_width: float = 50.0e-6
    profile_times: tuple = (5.0, 20.0)
    report_times: tuple = (0.1, 1.0, 5.0, 10.0, 20.0, 50.0)
    experiment: dict | None = None
    experiment_label: str = ""
    tie_line_x_c: tuple = (0.0015, 0.025, 16)
    tie_line_x_x_factors: tuple = (0.3, 1.6, 27)
    notes: dict = field(default_factory=dict)

    @property
    def independent(self):
        return ("C", self.solute)

    @property
    def tc_solute(self):
        return self.solute.capitalize()


def _log(message):
    print(f"[{wallclock.strftime('%H:%M:%S')}] {message}", flush=True)


def alloy_mole_fractions(case):
    n_c, n_x = case.w_c / MOLAR_MASS["C"], case.w_x / MOLAR_MASS[case.solute]
    n_fe = (100.0 - case.w_c - case.w_x) / MOLAR_MASS["FE"]
    total = n_c + n_x + n_fe
    return np.array([n_c / total, n_x / total])


def _tc_system(session, case):
    return (
        session.set_cache_folder(str(OUTPUT_DIR / "tc_cache"))
        .select_thermodynamic_and_kinetic_databases_with_elements("FEDEMO", "MFEDEMO", ["Fe", case.tc_solute, "C"])
        .without_default_phases()
        .select_phase("FCC_A1")
        .select_phase("BCC_A2")
        .get_system()
    )


def tc_alloy_equilibrium(system, case):
    """Returns the stable phases and BCC_A2 mole fraction of the alloy at the case temperature."""
    x_alloy = alloy_mole_fractions(case)
    calc = (
        system.with_single_equilibrium_calculation()
        .set_condition("T", case.temperature)
        .set_condition("X(C)", float(x_alloy[0]))
        .set_condition(f"X({case.solute})", float(x_alloy[1]))
    )
    result = calc.calculate()
    stable = sorted({p.upper().split("#")[0] for p in result.get_stable_phases()})
    ferrite = float(result.get_value_of("NP(BCC_A2)")) if "BCC_A2" in stable else 0.0
    return stable, ferrite


def tc_tie_lines(system, case, x_c_values, x_x_values):
    """
    Returns BCC/FCC tie-lines sorted by ``eta = x_X`` in ferrite.

    Ordinary equilibria are computed on a grid of global (x_C, x_X) and every
    two-phase result contributes its tie-line. Tie-lines of a ternary
    two-phase field do not cross, so sorting by the ferrite X content gives a
    one-parameter family; near-duplicate tie-lines are merged.
    """
    calc = system.with_single_equilibrium_calculation().set_condition("T", case.temperature)
    records = []
    for row, x_c in enumerate(x_c_values):
        print(f"tie-line row {row + 1}/{len(x_c_values)} (x_C={x_c:.4g}), {len(records)} tie-lines so far", flush=True)
        for x_x in x_x_values:
            try:
                calc.set_condition("X(C)", float(x_c))
                calc.set_condition(f"X({case.solute})", float(x_x))
                result = calc.calculate()
                stable = {p.upper().split("#")[0] for p in result.get_stable_phases()}
                if stable != {"BCC_A2", "FCC_A1"}:
                    continue
                left = [float(result.get_value_of(f"X(BCC_A2,{e})")) for e in case.independent]
                right = [float(result.get_value_of(f"X(FCC_A1,{e})")) for e in case.independent]
            except Exception as exc:
                print(f"equilibrium at x_C={x_c:.4g}, x_{case.solute}={x_x:.4g} failed: {exc}")
                continue
            records.append((left[1], left, right))
    records.sort(key=lambda r: r[0])
    etas, left, right = [], [], []
    for eta, l, r in records:
        if etas and eta - etas[-1] <= 1e-9 * max(1.0, abs(eta)):
            continue
        etas.append(eta)
        left.append(l)
        right.append(r)
    return np.asarray(etas), np.asarray(left), np.asarray(right)


def tc_dc_grid(system, case, phase, x_c_values, x_x_values):
    """Returns DC matrices (rows/cols C, X; reference FE) on the (x_C, x_X) grid."""
    from tc_python import ALL_PHASES

    calc = (
        system.with_single_equilibrium_calculation()
        .set_condition("T", case.temperature)
        .set_phase_to_suspended(ALL_PHASES)
        .set_phase_to_entered(phase, 1.0)
    )
    out = np.empty((len(x_c_values), len(x_x_values), 2, 2))
    for i, x_c in enumerate(x_c_values):
        for j, x_x in enumerate(x_x_values):
            calc.set_condition("X(C)", float(x_c))
            calc.set_condition(f"X({case.solute})", float(x_x))
            result = calc.calculate()
            for a, k in enumerate(case.independent):
                for b, g in enumerate(case.independent):
                    out[i, j, a, b] = float(result.get_value_of(f"DC({phase},{k},{g},FE)"))
    return out


def run_dictra(system, case, sim_time):
    """Runs the DICTRA local-equilibrium reference to ``sim_time`` (see the module docstring for settings)."""
    from tc_python import CalculatedGrid, CompositionProfile, ElementProfile, Region, TimestepControl

    def composition():
        return CompositionProfile().add("C", ElementProfile.constant(case.w_c)).add(case.tc_solute, ElementProfile.constant(case.w_x))

    calculator = (
        system.with_isothermal_diffusion_calculation()
        .set_temperature(case.temperature)
        .set_simulation_time(sim_time)
        .add_region(
            Region("Ferrite")
            .set_width(case.ferrite_width)
            .with_grid(CalculatedGrid.linear().set_no_of_points(10))
            .with_composition_profile(composition())
            .add_phase("BCC_A2")
        )
        .add_region(
            Region("Austenite")
            .set_width(case.austenite_width)
            .with_grid(CalculatedGrid.geometric().set_no_of_points(100).set_geometrical_factor(1.05))
            .with_composition_profile(composition())
            .add_phase("FCC_A1")
        )
        .with_timestep_control(
            TimestepControl()
            .set_max_relative_error(1.0e-3)
            .set_max_absolute_error(1.0e-7)
            .set_max_timestep_increase_factor(1.2)
            .set_max_timestep_allowed_as_percent_of_simulation_time(1.0)
        )
    )
    result = calculator.calculate()
    times, positions = result.get_position_of_upper_boundary_of_region("Ferrite")
    profiles = {}
    for t in sorted({t for t in tuple(case.profile_times) + (sim_time,) if t <= sim_time}):
        entry = {}
        for element in ("C", case.tc_solute):
            z, x = result.get_mole_fraction_of_component_at_time(element, float(t))
            entry["z"] = list(map(float, z))
            entry[f"x_{element.upper()}"] = list(map(float, x))
        profiles[str(t)] = entry
    return {"time": list(map(float, times)), "position": list(map(float, positions)), "profiles": profiles}


def build_tc_inputs(case):
    """Stage 1: tie-lines and DC tables (cached separately from the DICTRA run)."""
    cache = OUTPUT_DIR / f"{case.name}_tc_inputs.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from tc_python import TCPython

    x_alloy = alloy_mole_fractions(case)
    with TCPython() as session:
        system = _tc_system(session, case)
        stable, ferrite = tc_alloy_equilibrium(system, case)
        _log(f"alloy x(C, {case.solute}) = {x_alloy}; equilibrium at {case.temperature} K: {stable}, ferrite fraction {ferrite:.3f}")
        if stable != ["BCC_A2", "FCC_A1"]:
            raise RuntimeError(f"Alloy is not in the ferrite + austenite field at {case.temperature} K: {stable}.")
        _log("sampling tie-lines")
        c_lo, c_hi, c_n = case.tie_line_x_c
        f_lo, f_hi, f_n = case.tie_line_x_x_factors
        etas, left, right = tc_tie_lines(system, case, np.linspace(c_lo, c_hi, int(c_n)), x_alloy[1] * np.linspace(f_lo, f_hi, int(f_n)))
        if len(etas) < 5:
            raise RuntimeError(f"Only {len(etas)} BCC/FCC tie-lines found; widen the sampling grid.")
        _log(f"{len(etas)} tie-lines, eta (ferrite x_{case.solute}) in [{etas.min():.4g}, {etas.max():.4g}]")
        bcc_c = np.geomspace(1e-6, 1.2 * max(x_alloy[0], left[:, 0].max()), 14)
        bcc_x = np.linspace(0.5 * min(x_alloy[1], left[:, 1].min()), 1.2 * max(x_alloy[1], left[:, 1].max()), 8)
        fcc_c = np.linspace(0.8 * min(x_alloy[0], right[:, 0].min()), 1.05 * right[:, 0].max(), 16)
        fcc_x = np.linspace(0.7 * min(x_alloy[1], right[:, 1].min()), 1.1 * max(x_alloy[1], right[:, 1].max()), 10)
        _log("tabulating BCC_A2 diffusivities")
        bcc = tc_dc_grid(system, case, "BCC_A2", bcc_c, bcc_x)
        _log("tabulating FCC_A1 diffusivities")
        fcc = tc_dc_grid(system, case, "FCC_A1", fcc_c, fcc_x)
    data = {
        "x_alloy": x_alloy.tolist(),
        "equilibrium_ferrite_fraction": ferrite,
        "tie_lines": {"eta": etas.tolist(), "left": left.tolist(), "right": right.tolist()},
        "tables": {
            "BCC_A2": {"x_c": bcc_c.tolist(), "x_x": bcc_x.tolist(), "D": bcc.tolist()},
            "FCC_A1": {"x_c": fcc_c.tolist(), "x_x": fcc_x.tolist(), "D": fcc.tolist()},
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    _log(f"wrote {cache}")
    return data


def dictra_cache_path(case, sim_time):
    return OUTPUT_DIR / f"{case.name}_dictra_{sim_time:g}s.json"


def build_dictra_reference(case, sim_time):
    """Stage 2: DICTRA local-equilibrium reference to ``sim_time`` (cached per time)."""
    cache = dictra_cache_path(case, sim_time)
    if cache.exists():
        return json.loads(cache.read_text())
    from tc_python import TCPython

    with TCPython() as session:
        _log(f"running DICTRA (local equilibrium) to {sim_time:g} s")
        dictra = run_dictra(_tc_system(session, case), case, sim_time)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(dictra))
    _log(f"wrote {cache}")
    return dictra


class TabulatedTernaryDiffusivity:
    """Bilinear (x_C, x_X) interpolation of Thermo-Calc DC matrices; mole-fraction input."""

    def __init__(self, tables):
        self.interpolators = {}
        for phase, table in tables.items():
            x_axis = table["x_x"] if "x_x" in table else table["x_ni"]  # older Fe-Ni-C caches
            grid = (np.asarray(table["x_c"]), np.asarray(x_axis))
            self.interpolators[phase] = RegularGridInterpolator(grid, np.asarray(table["D"]), bounds_error=False, fill_value=None)

    def clearCache(self):
        pass

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        points = np.atleast_2d(np.asarray(x, dtype=np.float64))
        out = self.interpolators[phase](points)
        return out[0] if np.asarray(x).ndim == 1 else out


def tie_line_closure(data):
    tie = data["tie_lines"]
    eta = np.asarray(tie["eta"])
    left = np.asarray(tie["left"])
    right = np.asarray(tie["right"])

    def interface(e):
        return (
            np.array([np.interp(e, eta, left[:, k]) for k in range(2)]),
            np.array([np.interp(e, eta, right[:, k]) for k in range(2)]),
        )

    return CallableTernaryInterfaceEquilibrium(interface, eta_bounds=(float(eta[0]), float(eta[-1])), composition_basis="x")


def run_kawin(case, data, *, n_left=41, n_right=401, power=3.0, semi_log_dt=0.02, mode="composition_dependent_lagged", x_residual_scale=1.0, eta_componentwise=False, semi_log_t0=1.0e-7):
    x_alloy = np.asarray(data["x_alloy"])
    R = case.ferrite_width + case.austenite_width
    mesh = CartesianFD1D(["C", case.solute], [0.0, R], 2001)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(case.ferrite_width, x_alloy, x_alloy), ["C", case.solute])]))
    u_grid = landau_grid(n_left, "power", phase="left", exponent=1.5)
    v_grid = landau_grid(n_right, "power", phase="right", exponent=power)
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=["FE", "C", case.solute],
        phases=["BCC_A2", "FCC_A1"],
        thermodynamics=TabulatedTernaryDiffusivity(data["tables"]),
        temperature=case.temperature,
        interfacePosition=case.ferrite_width,
        interface_equilibrium=tie_line_closure(data),
        time_step=case.sim_time,
        dt_mode="semi_log",
        semiLog_dt=semi_log_dt,
        semiLogT0=semi_log_t0,
        bulk_diffusivity_mode=mode,
        tolerance=1.0e-10,
        max_iterations=50,
        max_step_retries=12,
        terminal_thin_phase_width=1.0e-12,
        terminal_thin_phase_policy="continue",
        record=True,
        transformed_u_grid=u_grid,
        transformed_v_grid=v_grid,
        interface_residual_composition_scale=(1.0, x_residual_scale),
        initial_eta_componentwise_scaling=eta_componentwise,
    )
    diagnostics = []
    original_post = model.postProcess

    def post_with_diagnostics(time, x):
        out = original_post(time, x)
        diagnostics.append(
            {
                "t": float(time),
                "s": float(model._s_curr),
                "eta": float(model._eta_curr),
                "branch": model._lastImplicitMotionBranch,
                "retries": int(model._lastStepRetries),
                "peclet": model._lastInterfacePeclet.tolist(),
                "jacobian_cond": float(model._lastImplicitJacobianCond),
                "residual_components": model._lastImplicitResidualComponents.tolist(),
            }
        )
        return out

    model.postProcess = post_with_diagnostics
    model.solve(case.sim_time, iterator=explicitEulerIterator, minDtFrac=1e-14)
    return model, diagnostics


def compare(case, model, data):
    t_k = model.interfaceData._time[: model.interfaceData.N + 1]
    s_k = model.interfaceData._y[: model.interfaceData.N + 1]
    report = {"thickness_samples": {}}
    dictra = data.get("dictra")
    dictra_end = float(dictra["time"][-1]) if dictra is not None else None
    times = sorted(set(case.report_times) | ({0.01, 0.1, 0.5, 1.0, dictra_end} if dictra is not None else set()))
    for t in times:
        if t > case.sim_time:
            continue
        s_t = float(model.interfaceData.y(t))
        sample = {"kawin_LE": s_t, "kawin_k": s_t / np.sqrt(t)}
        if dictra is not None and t <= dictra_end:
            s_dt = float(np.interp(t, dictra["time"], dictra["position"]))
            sample.update({"dictra_LE": s_dt, "dictra_k": s_dt / np.sqrt(t), "rel_diff": (s_t - s_dt) / s_dt})
        report["thickness_samples"][f"{t:g}"] = sample
    if dictra is not None:
        t_d = np.asarray(dictra["time"])
        s_d = np.asarray(dictra["position"])
        mask = (t_d >= 0.1 * dictra_end) & (t_d <= min(case.sim_time, dictra_end))
        rel = (np.interp(t_d[mask], t_k, s_k) - s_d[mask]) / s_d[mask]
        report["dictra_end_time"] = dictra_end
        report["thickness_max_rel_diff_last_decade"] = float(np.max(np.abs(rel)))
        report["thickness_rel_diff_at_dictra_end"] = float(rel[-1])
    report["inventory_drift"] = (np.asarray(model.getTotalInventory()) - np.asarray(model.inventoryData._y[0])).tolist()
    report["eta_initial"] = float(model.initialEta)
    report["eta_final"] = float(model._eta_curr)
    report[f"x_{case.solute.lower()}_alloy"] = float(data["x_alloy"][1])
    report["steps"] = int(model.interfaceData.N)
    return report, (t_k, s_k)


def plot(case, data, histories, suffix=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_k, s_k = histories
    fig, ax = plt.subplots(figsize=(6, 4))
    if data.get("dictra") is not None:
        ax.plot(data["dictra"]["time"], np.asarray(data["dictra"]["position"]) * 1e6, "k-", label="DICTRA, local equilibrium")
    ax.plot(t_k, s_k * 1e6, "r--", label="kawin Illingworth (u), local equilibrium")
    if case.experiment is not None:
        ax.plot(case.experiment["time"], np.asarray(case.experiment["half_thickness"]) * 1e6, "bs", label=case.experiment_label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ferrite thickness (um)")
    ax.set_xlim(0, case.sim_time)
    ax.set_title(f"Fe-{case.w_x:g}{case.tc_solute}-{case.w_c:g}C (wt%), {case.temperature:g} K")
    ax.legend()
    fig.tight_layout()
    out = OUTPUT_DIR / f"{case.name}{suffix}_thickness.png"
    fig.savefig(out, dpi=150)
    return out


def main(case, description=None):
    parser = argparse.ArgumentParser(description=description or __doc__.splitlines()[0])
    parser.add_argument("--mode", default="composition_dependent_lagged")
    parser.add_argument("--n-right", type=int, default=401)
    parser.add_argument("--power", type=float, default=3.0)
    parser.add_argument("--semi-log-dt", type=float, default=0.02)
    parser.add_argument("--x-residual-scale", type=float, default=1.0)
    parser.add_argument("--eta-componentwise", action="store_true", help="per-component scaling of the startup eta balance")
    parser.add_argument("--semi-log-t0", type=float, default=1.0e-7, help="first semi-log target time (s)")
    parser.add_argument("--dictra-sim-time", type=float, default=2.0, help="simulated time of the DICTRA reference (s)")
    parser.add_argument("--tag", default="", help="suffix for kawin output files (e.g. a grid label)")
    parser.add_argument(
        "--stage",
        choices=("tc-inputs", "dictra", "kawin", "kawin-only", "all"),
        default="all",
        help="tc-inputs and dictra are cached; kawin needs the DICTRA cache, kawin-only skips the comparison",
    )
    args = parser.parse_args()
    data = build_tc_inputs(case)
    if args.stage == "tc-inputs":
        return
    if args.stage in ("dictra", "all"):
        data["dictra"] = build_dictra_reference(case, args.dictra_sim_time)
        if args.stage == "dictra":
            return
    elif args.stage == "kawin":
        dictra_cache = dictra_cache_path(case, args.dictra_sim_time)
        if not dictra_cache.exists():
            raise RuntimeError("Run --stage dictra first (or use --stage kawin-only).")
        data["dictra"] = json.loads(dictra_cache.read_text())
    _log("running kawin")
    start = wallclock.perf_counter()
    model, diagnostics = run_kawin(
        case,
        data,
        n_right=args.n_right,
        power=args.power,
        semi_log_dt=args.semi_log_dt,
        mode=args.mode,
        x_residual_scale=args.x_residual_scale,
        eta_componentwise=args.eta_componentwise,
        semi_log_t0=args.semi_log_t0,
    )
    report, histories = compare(case, model, data)
    report["kawin_runtime_s"] = wallclock.perf_counter() - start
    report["settings"] = vars(args)
    suffix = ("_kawin_only" if args.stage == "kawin-only" else "") + (f"_{args.tag}" if args.tag else "")
    report["plot"] = str(plot(case, data, histories, suffix))
    (OUTPUT_DIR / f"{case.name}{suffix}_diagnostics.json").write_text(json.dumps(diagnostics))
    (OUTPUT_DIR / f"{case.name}{suffix}_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
