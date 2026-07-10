#%%

# %matplotlib inline
"""
Replicate Olaye & Ojo (2020) Figure 5 with the Olaye moving-boundary FD model.

This script runs the binary planar moving-interface solver and plots liquid
half-width versus time for a Ni-P TLP-style setup.

Notes
-----
- The model implementation currently supports planar geometry only.
- By default this script uses Table-2-style Ni-P parameters from the paper
  (converted from percent to mole-fraction-like units and micrometers to SI).
- If you have digitized Figure 5 experimental points, pass them with
  ``--exp-csv`` (columns: ``time_h`` and ``half_width_um``).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from kawin.diffusion import (
    MovingBoundaryOlayeFD1DModel,
    MovingBoundaryOlayeFD1DReworkModel,
    TemperatureParameters,
)
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.solver import explicitEulerIterator

# MODEL_VARIANT = "current"
MODEL_VARIANT = "rework"

def debugInPlace():
    try:
        import debugpy
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('break on this line')
    except:
        pass


from scipy import optimize
import math
def equation_a11(beta, c_a0, c_b0, c_a_eq, c_b_eq, d_a, d_b):
    return (
        ((c_a_eq - c_b_eq) * beta * np.sqrt(np.pi))
        - ((np.sqrt(d_a) * (c_a0 - c_a_eq)) / (1 + math.erf(beta / np.sqrt(d_a)))) * np.exp(-(beta**2) / d_a)
        + ((np.sqrt(d_b) * (c_b_eq - c_b0)) / (1 - math.erf(beta / np.sqrt(d_b)))) * np.exp(-(beta**2) / d_b)
    )

def solve_beta(c_a0, c_b0, c_a_eq, c_b_eq, d_a, d_b, left=-1, right=1):
    assert c_b0 < c_b_eq < c_a_eq < c_a0, "Expected: c_b0 < c_b_eq < c_a_eq < c_a0"
    if left<0 and right>0:
        grid = np.concatenate((-np.geomspace(1e-15, -left, 200)[::-1], np.geomspace(1e-15, right, 200)))
    else:
        grid = np.linspace(left, right, 4001)
    values = np.array([equation_a11(x, c_a0, c_b0, c_a_eq, c_b_eq, d_a, d_b) for x in grid], dtype=np.float64)
    indicesOfSignFlip = np.where(np.logical_and((np.diff(np.sign(values))!=0), ~np.isnan(np.diff(np.sign(values)))))[0]
    if indicesOfSignFlip.size != 1:
        raise ValueError("Could not bracket an analytic moving-boundary root.")
    else:
        x_left = grid[indicesOfSignFlip[0]]
        x_right = grid[indicesOfSignFlip[0]+1]
                        

        sol = optimize.root_scalar(
            equation_a11,
            bracket=[float(x_left), float(x_right)],
            method='brentq',
            args=(c_a0, c_b0, c_a_eq, c_b_eq, d_a, d_b),
            maxiter=100,
            rtol=1e-14,
            xtol=1e-14,
        )
        if sol.converged:
            return float(sol.root)
        else:
            raise ValueError("Root finding did not converge.")                


def calculateCFLConstants(model_input):
    ID_time_arr = model_input.interfaceData._time[:-1].copy()
    ID_s_arr = model_input.interfaceData._y[:-1].copy()
    ID_dt_arr = np.diff(ID_time_arr).copy()
    D_A = model_input.therm.diffusivities[model_input.therm.phases[0]]
    # D_B = model_input.therm.diffusivities[model_input.therm.phases[1]]
    mu_A_arr = ((ID_dt_arr*D_A) / (ID_s_arr[1:]*model_input._du)**2).copy()
    w_arr = (ID_s_arr[1:]/ID_s_arr[:-1]).copy()

    intermediateTime_arr = (ID_time_arr[:-1] + ID_time_arr[1:]) / 2
    return w_arr, mu_A_arr, intermediateTime_arr

class ConstantBinaryThermodynamics:
    """Minimal binary thermodynamics interface for fixed diffusivity runs."""

    def __init__(self, phases, diffusivities):
        self.phases = list(phases)
        self.diffusivities = dict(diffusivities)

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        values = np.atleast_1d(T).astype(np.float64)
        return np.squeeze(np.ones(values.shape, dtype=np.float64) * self.diffusivities[phase])


def _load_experimental_csv(path: pathlib.Path):
    """
    Loads an experimental overlay CSV.

    Expected columns (with header): ``time_h,half_width_um``.
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if "time_s" not in data.dtype.names or "half_width_um" not in data.dtype.names:
        raise ValueError("Experimental CSV must contain columns: time_s, half_width_um")
    return np.asarray(data["time_s"], dtype=np.float64), np.asarray(data["half_width_um"], dtype=np.float64)


FIG5_BASE_PARAMS = {
    "R_um": 3012.5,
    "s0_um": 12.5,
    "c_liquid0_pct": 19.0,
    "c_solid0_pct": 0.0,
    "c_liquid_int_pct": 10.223,
    "c_solid_int_pct": 0.166,
    "D_liquid_base": 500.0,
    "D_solid_base": 18.0,
    "D_scale": 1e-12,
    "n_nodes": 3013 + 1,
    "t_end_s": 1e1,
    "dt_mode": "semi_log_optional",
    "semiLogT0": 1e-5,
    "exp_csv": r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\Olaye2020\Olaye2020_fig5_PresentModel_curve.csv",
    "out": None,
    "show": True,
}

def _resolve_this_file():
    """
    Returns this script's path without trusting stale Interactive Window globals.

    In notebook/interactive execution, VS Code can leave ``__file__`` bound to
    another script that previously populated the shared kernel namespace. We
    therefore accept ``__file__`` only when it already points to this file and
    otherwise fall back to locating the script from the current working
    directory.
    """
    expected_parts = ("examples", "Olaye2020", "replicate_olaye2020_fig5.py")
    if "__file__" in globals():
        candidate = pathlib.Path(__file__).resolve()
        if tuple(candidate.parts[-3:]) == expected_parts:
            return candidate

    cwd = pathlib.Path.cwd().resolve()
    candidates = [cwd.joinpath(*expected_parts[-2:]), cwd.joinpath(*expected_parts)]
    candidates.extend(parent.joinpath(*expected_parts) for parent in cwd.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"Could not resolve {'/'.join(expected_parts)} from cwd={cwd}.")


THIS_FILE = _resolve_this_file()
SCRIPT_DIR = THIS_FILE.parent

OLAYE_NOTEBOOK_CONFIG = {
    "n_phase_a_nodes": 51,
    "n_phase_b_nodes": 5001,
    "semiLog_dt": 0.002763654842561367 / 1.0,
    "model_variant": MODEL_VARIANT,
    "out": None,
    "show": True,
    "save_run": True,
    "save_run_path": SCRIPT_DIR / "olaye2020_fig5_saved_run.npz",
    "label": None,
    "timeProfiling":True,
}


def compute_idealized_conc(params):
    """Returns the idealized average concentration for the Figure-5 setup."""
    return (
        params["s0_um"] * (params["c_liquid0_pct"] / 100.0)
        + (params["R_um"] - params["s0_um"]) * (params["c_solid0_pct"] / 100.0)
    ) / params["R_um"]


def theoretical_fig5_max_liquid_half_width_um(params):
    """
    Returns the idealized maximum liquid half-width for the Figure-5 setup.

    This assumes the initial liquid solute inventory is redistributed into
    liquid at the liquid-side interface composition.
    """
    return params["s0_um"] * params["c_liquid0_pct"] / params["c_liquid_int_pct"]


def _jsonable(value):
    """Converts nested notebook config values into JSON-serializable objects."""
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_olaye_run_payload(
    *,
    time_s,
    half_width_um,
    params,
    model_variant,
    label=None,
    mass_integral_initial=None,
    mass_integral_final=None,
):
    """
    Builds the saved-run payload for notebook-friendly Figure-5 comparisons.

    The saved arrays use seconds and micrometers so they can be overlaid
    directly with the Illingworth Figure-3 present-work results.
    """
    payload_params = {
        key: value
        for key, value in params.items()
        if key not in {"show", "out", "save_run", "save_run_path", "label"}
    }
    payload = {
        "time_s": np.asarray(time_s, dtype=np.float64),
        "half_width_um": np.asarray(half_width_um, dtype=np.float64),
        "label": np.array(label or f"Olaye Figure 5 ({model_variant})"),
        "source_script": np.array("examples/Olaye2020/replicate_olaye2020_fig5.py"),
        "model_family": np.array("olaye_fig5"),
        "model_variant": np.array(str(model_variant)),
        "theoretical_max_um": np.array([theoretical_fig5_max_liquid_half_width_um(params)], dtype=np.float64),
        "idealized_mass_integral": np.array([compute_idealized_conc(params)], dtype=np.float64),
        "params_json": np.array(json.dumps(_jsonable(payload_params), sort_keys=True)),
    }
    if mass_integral_initial is not None:
        payload["mass_integral_initial"] = np.array([float(mass_integral_initial)], dtype=np.float64)
    if mass_integral_final is not None:
        payload["mass_integral_final"] = np.array([float(mass_integral_final)], dtype=np.float64)
    return payload


def save_olaye_run_result(path, payload):
    """Saves an Olaye Figure-5 run payload to a compressed ``.npz`` file."""
    save_path = pathlib.Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **payload)
    return save_path


def run_case(
    *,
    R_um: float,
    s0_um: float,
    c_liquid0_pct: float,
    c_solid0_pct: float,
    c_liquid_int_pct: float,
    c_solid_int_pct: float,
    D_liquid_base: float,
    D_solid_base: float,
    D_scale: float,
    n_nodes: int,
    n_phase_a_nodes: int,
    n_phase_b_nodes: int,
    t_end_s: float,
    dt_mode: str,
    # semi_log_points: int,
    semiLog_dt: float,
    semiLogT0: float,
    model_variant: str,

):
    """Runs one Figure-5-style simulation and returns time and liquid half-width."""
    R_m = float(R_um) * 1e-6
    s0_m = float(s0_um) * 1e-6
    interface_position_m = s0_m

    # Percent inputs from the paper -> fraction-like internal values.
    c_liquid0 = float(c_liquid0_pct) / 100.0
    c_solid0 = float(c_solid0_pct) / 100.0
    c_liquid_int = float(c_liquid_int_pct) / 100.0
    c_solid_int = float(c_solid_int_pct) / 100.0

    # Match the paper convention directly:
    # left phase A = liquid, right phase B = solid base metal.
    profile = ProfileBuilder([(StepProfile1D(interface_position_m, c_liquid0, c_solid0), "P")])
    mesh = CartesianFD1D(["P"], [0.0, R_m], int(n_nodes))
    mesh.setResponseProfile(profile)

    therm = ConstantBinaryThermodynamics(
        phases=["LIQUID", "SOLID"],
        diffusivities={
            "LIQUID": float(D_liquid_base) * float(D_scale),
            "SOLID": float(D_solid_base) * float(D_scale),
        },
    )


    print(f"beta = {solve_beta(c_a0=c_liquid0, c_b0=c_solid0, c_a_eq=c_liquid_int, c_b_eq=c_solid_int, d_a=float(D_liquid_base) * float(D_scale), d_b=float(D_solid_base) * float(D_scale), left=-100.0, right=100.0)}")

    '''
    ## To plot in console use:
        
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # y_arr = (self.interfaceData._y[:self.interfaceData.currentIndex]-self.interfaceData._y[0])
    y_arr = (self.interfaceData._y[:self.interfaceData.currentIndex])
    x_arr = np.sqrt(self.interfaceData._time[:self.interfaceData.currentIndex])
    indexToPlotTo=-1
    ax.plot(x_arr[:indexToPlotTo], y_arr[:indexToPlotTo], 'o')
    beta = 7.207320366333016e-06
    indexToPlotTo_beta=indexToPlotTo
    ax.plot(x_arr[:indexToPlotTo_beta], 2*beta*x_arr[:indexToPlotTo_beta] + self.interfaceData._y[0], label=f'Analytic Solution, beta={beta}')

    import pandas as pd
    fig5_df = pd.read_csv("C:\\Users\\samth\\OneDrive - Northwestern University\\WS_DL\\Lab Data\\Price\\code\\kawin\\examples\\Olaye2020\\Olaye2020_fig5_PresentModel_curve.csv")
    fig5_df = fig5_df.sort_values(by=['time_s'])
    fig5_df['half_width_m'] = fig5_df['half_width_um']*1e-6
    if (np.sqrt(fig5_df['time_s'])<x_arr[indexToPlotTo]).any():
        digitizedIndexToPlot = np.max(np.where(np.sqrt(fig5_df['time_s'])<x_arr[indexToPlotTo])[0])
        ax.plot(np.sqrt(fig5_df['time_s'])[:digitizedIndexToPlot], fig5_df['half_width_m'][:digitizedIndexToPlot], 'o', color='tab:red',  label='Fig 5 "Present Model" (digitized)')
    
    
    LeeAndOh_NiP_results_loaded = np.loadz(r"C:\\Users\\samth\\Downloads\\LeeAndOh_NiP_results_100sec_N3013.npz")
    LeeAndOh_NiP_results_loaded['half_width_m'] = LeeAndOh_NiP_results_loaded['half_width_um'] * 1e-6
    if (np.sqrt(LeeAndOh_NiP_results_loaded['time_s'])<x_arr[indexToPlotTo]).any():
        LeeAndOhIndexToPlot = np.max(np.where(np.sqrt(LeeAndOh_NiP_results_loaded['time_s'])<x_arr[indexToPlotTo])[0])
        ax.plot(np.sqrt(LeeAndOh_NiP_results_loaded['time_s'])[:LeeAndOhIndexToPlot], LeeAndOh_NiP_results_loaded['half_width_m'][:LeeAndOhIndexToPlot], 'o', color='tab:red',  label='Fig 5 "Present Model" (digitized)')
    
    plt.legend()
    plt.show(block=True)

    print(f"max liquid width in sim:      {np.max(model.interfaceData._y)}")
    print(f"theoretical max liquid width: {(c_a0/c_a_eq) * l_a}")   ## (0.19/0.10223) * 12.5
    

    ## To plot after run use:
    
    # styleDict = {'marker':'o', 'markersize':2}
    styleDict = {'linewidth':2, 'alpha':0.75}
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # y_arr = (model.interfaceData._y[:model.interfaceData.currentIndex]-model.interfaceData._y[0])
    y_arr = (model.interfaceData._y[:model.interfaceData.currentIndex])
    x_arr = np.sqrt(model.interfaceData._time[:model.interfaceData.currentIndex])
    indexToPlotTo=-1
    ax.plot(x_arr[:indexToPlotTo], y_arr[:indexToPlotTo], label="Olaye model calculated results", zorder=3, **styleDict)
    beta = 7.207320366333016e-06
    indexToPlotTo_beta = np.max(np.where((2*beta*x_arr + model.interfaceData._y[0])<np.max(y_arr[:indexToPlotTo]))[0])
    # indexToPlotTo_beta=indexToPlotTo
    ax.plot(x_arr[:indexToPlotTo_beta], 2*beta*x_arr[:indexToPlotTo_beta] + model.interfaceData._y[0], label=f'Analytic Solution, beta={beta}')

    import pandas as pd
    fig5_df = pd.read_csv("C:\\Users\\samth\\OneDrive - Northwestern University\\WS_DL\\Lab Data\\Price\\code\\kawin\\examples\\Olaye2020\\Olaye2020_fig5_PresentModel_curve.csv")
    fig5_df = fig5_df.sort_values(by=['time_s'])
    fig5_df['half_width_m'] = fig5_df['half_width_um']*1e-6
    if (np.sqrt(fig5_df['time_s'])<x_arr[indexToPlotTo]).any():
        digitizedIndexToPlot = np.max(np.where(np.sqrt(fig5_df['time_s'])<x_arr[indexToPlotTo])[0])
        ax.plot(np.sqrt(fig5_df['time_s'])[:digitizedIndexToPlot], fig5_df['half_width_m'][:digitizedIndexToPlot], 'o', color='tab:red',  label='Fig 5 "Present Model" (digitized)')


    LeeAndOh_NiP_results_loaded = np.load("C:\\Users\\samth\\Downloads\\LeeAndOh_NiP_results_100sec_N3013.npz")
    LeeAndOh_NiP_results_loaded_df = pd.DataFrame.from_dict({item: LeeAndOh_NiP_results_loaded[item] for item in LeeAndOh_NiP_results_loaded.files})
    LeeAndOh_NiP_results_loaded_df['half_width_m'] = LeeAndOh_NiP_results_loaded_df['half_width_um'] * 1e-6
    if (np.sqrt(LeeAndOh_NiP_results_loaded_df['time_s'])<x_arr[indexToPlotTo]).any():
        LeeAndOhIndexToPlot = np.max(np.where(np.sqrt(LeeAndOh_NiP_results_loaded_df['time_s'])<x_arr[indexToPlotTo])[0])
        ax.plot(np.sqrt(LeeAndOh_NiP_results_loaded_df['time_s'])[:LeeAndOhIndexToPlot], LeeAndOh_NiP_results_loaded_df['half_width_m'][:LeeAndOhIndexToPlot], label='Lee and Oh calculated results', zorder=2, **styleDict)

    plt.legend()
    plt.show(block=True)

    print(f"max liquid width in sim:      {np.max(model.interfaceData._y)}")
    print(f"theoretical max liquid width: {(c_a0/c_a_eq) * l_a}")   ## (0.19/0.10223) * 12.5

    
    '''
    # debugInPlace()

    model_class = {
        "current": MovingBoundaryOlayeFD1DModel,
        "rework": MovingBoundaryOlayeFD1DReworkModel,
    }[str(model_variant)]

    model = model_class(
        mesh,
        ["NI", "P"],
        ["LIQUID", "SOLID"],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=interface_position_m,
        interface_compositions=(c_liquid_int, c_solid_int),
        first_step_mode="classical_explicit",
        main_step_mode="leapfrog_dufort_frankel",
        dt_mode=dt_mode,
        geometry="planar",
        phase_a_nodes=int(n_phase_a_nodes),
        phase_b_nodes=int(n_phase_b_nodes),
        # semi_log_points=semi_log_points,
        semiLog_dt=semiLog_dt,
        semiLogT0=semiLogT0,
        record= True,
    )

    print(f"Estimated total number of time steps: {int((np.log(t_end_s) - np.log(model.semiLogT0)) / model.semiLog_dt)}")
    model.solve(float(t_end_s), iterator=explicitEulerIterator, verbose=True, vIt=100, minDtFrac=1e-13)

    t_s = np.array(model.interfaceData._time[: model.interfaceData.N + 1], dtype=np.float64)
    s_m = np.array(model.interfaceData._y[: model.interfaceData.N + 1], dtype=np.float64)
    liquid_half_width_um = s_m * 1e6
    # t_h = t_s / 3600.0
    return t_s, liquid_half_width_um, model


def build_parser():
    """Builds the optional CLI parser retained for backward compatibility."""
    parser = argparse.ArgumentParser(description="Replicate Olaye & Ojo 2020 Figure 5 using the Olaye FD model.")
    parser.add_argument("--n-phase-a-nodes", type=int, default=OLAYE_NOTEBOOK_CONFIG["n_phase_a_nodes"])
    parser.add_argument("--n-phase-b-nodes", type=int, default=OLAYE_NOTEBOOK_CONFIG["n_phase_b_nodes"])
    parser.add_argument("--semiLog-dt", type=float, default=OLAYE_NOTEBOOK_CONFIG["semiLog_dt"])
    parser.add_argument("--model-variant", type=str, choices=["current", "rework"], default=OLAYE_NOTEBOOK_CONFIG["model_variant"])
    parser.add_argument("--out", type=str, default=None, help="Optional figure output path.")
    parser.add_argument("--save-run-path", type=str, default=None, help="Optional saved-run ``.npz`` output path.")
    parser.add_argument("--no-save-run", action="store_true", help="Disable saving the notebook-friendly run artifact.")
    parser.add_argument("--hide-plot", action="store_true", help="Do not show the matplotlib window.")
    return parser


def plot_olaye_fig5_notebook(config=None, ax=None):
    """
    Runs the Figure-5 example from editable in-file settings and plots the run.

    Parameters are taken from ``OLAYE_NOTEBOOK_CONFIG`` by default so the script
    can be run directly from a notebook cell or interactive window.
    """
    config = OLAYE_NOTEBOOK_CONFIG if config is None else {**OLAYE_NOTEBOOK_CONFIG, **dict(config)}
    args = {**FIG5_BASE_PARAMS, **config}

    out_path = (
        pathlib.Path(args["out"]).resolve()
        if args["out"] is not None
        else SCRIPT_DIR / "olaye2020_fig5_replication.png"
    )

    t_s, width_um, model = run_case(
        R_um=args["R_um"],
        s0_um=args["s0_um"],
        c_liquid0_pct=args["c_liquid0_pct"],
        c_solid0_pct=args["c_solid0_pct"],
        c_liquid_int_pct=args["c_liquid_int_pct"],
        c_solid_int_pct=args["c_solid_int_pct"],
        D_liquid_base=args["D_liquid_base"],
        D_solid_base=args["D_solid_base"],
        D_scale=args["D_scale"],
        n_nodes=args["n_nodes"],
        n_phase_a_nodes=args["n_phase_a_nodes"],
        n_phase_b_nodes=args["n_phase_b_nodes"],
        t_end_s=args["t_end_s"],
        dt_mode=args["dt_mode"],
        semiLog_dt=args["semiLog_dt"],
        semiLogT0=args["semiLogT0"],
        model_variant=args["model_variant"],
    )

    mask = np.isfinite(t_s) & np.isfinite(width_um)
    t_s_plot = t_s[mask]
    width_um_plot = width_um[mask]
    if t_s_plot.size < 2:
        raise ValueError("Simulation did not produce enough finite points to plot.")

    if args['timeProfiling']:
        conc_time_arr = np.asarray(model.concData._time, dtype=np.float64)
        conc_arr = np.asarray(model.concData._y, dtype=np.float64)
        
        payload = build_olaye_run_payload(
        time_s=t_s_plot,
        half_width_um=width_um_plot,
        params=args,
        model_variant=args["model_variant"],
        label=args["label"],
        mass_integral_initial=conc_arr[0],
        mass_integral_final=conc_arr[-1],
        )

        save_path = None
        if args["save_run"]:
            save_path = save_olaye_run_result(args["save_run_path"], payload)
            print(f"Saved run: {save_path}")
        
        return {
        "model": model,
        "figure": None,
        "axes": None,
        "time_s": t_s_plot,
        "half_width_um": width_um_plot,
        "payload": payload,
        "save_path": save_path,
        "params": args,
        }

    created_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)
    else:
        fig = ax.figure
    label = args["label"] or f"Olaye model ({args['model_variant']})"
    ax.plot(t_s_plot, width_um_plot, lw=2.0, color="tab:blue", label=label)

    exp_csv = args["exp_csv"]
    if exp_csv is not None:
        exp_t_s, exp_w_um = _load_experimental_csv(pathlib.Path(exp_csv))
        exp_mask = np.isfinite(exp_t_s) & np.isfinite(exp_w_um)
        ax.scatter(
            exp_t_s[exp_mask],
            exp_w_um[exp_mask],
            s=26,
            color="tab:red",
            marker="o",
            label="Experimental (digitized)",
        )

    alt_csv = SCRIPT_DIR / "Olaye2020_fig5_alt_PresentModelRough2_curve.csv"
    exp_t_s, exp_w_um = _load_experimental_csv(alt_csv)
    exp_mask = np.isfinite(exp_t_s) & np.isfinite(exp_w_um)
    ax.scatter(
        exp_t_s[exp_mask],
        exp_w_um[exp_mask],
        s=26,
        color="tab:pink",
        marker="o",
        facecolors="none",
        label="Experimental alt rough (digitized)",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Liquid half-width (um)")
    ax.set_title(
        f"n_phase_a_nodes:{args['n_phase_a_nodes']}, "
        f"n_phase_b_nodes:{args['n_phase_b_nodes']}, "
        f"semiLog_dt:{args['semiLog_dt']:.10f}    {args['model_variant']}",
        fontsize=10,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.00001, float(np.max(t_s_plot)))
    ax.set_ylim(12.5, 24)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax_twin = ax.twinx()
    ax_twin.set_ylabel("conc", color="green")
    conc_time_arr = np.asarray(model.concData._time, dtype=np.float64)
    conc_arr = np.asarray(model.concData._y, dtype=np.float64)
    ax_twin.plot(conc_time_arr, conc_arr, lw=1.0, color="tab:green", label="Conc")
    ax_twin.hlines(conc_arr[0], conc_time_arr[0], conc_time_arr[-1], lw=1.0, color="tab:green", linestyle="dashdot", label="Initial conc")
    idealized_conc = compute_idealized_conc(args)
    ax_twin.hlines(idealized_conc, conc_time_arr[0], conc_time_arr[-1], lw=1.0, color="tab:green", linestyle="dashed", label="Idealized conc")
    ax_twin.legend(loc="center right")

    w_arr_out, mu_A_arr_out, intermediate_time_arr_out = calculateCFLConstants(model)
    ax_twin_mu = ax.twinx()
    ax_twin_mu.set_ylabel("mu_A", color="orange")
    ax_twin_mu.plot(intermediate_time_arr_out, mu_A_arr_out, lw=1.0, color="tab:orange", label="Mu_A")
    ax_twin_mu.hlines(1, intermediate_time_arr_out[0], intermediate_time_arr_out[-1], lw=1.0, color="tab:orange", linestyle="dashdot", label="mu_A=1")
    ax_twin_mu.set_yscale("log")
    ax_twin_mu.spines["right"].set_position(("outward", 70))

    ax_twin_w = ax.twinx()
    ax_twin_w.set_ylabel("1/w", color="brown")
    ax_twin_w.plot(intermediate_time_arr_out, 1 / w_arr_out, lw=1.0, color="tab:brown", label="1/w")
    ax_twin_w.hlines(1, intermediate_time_arr_out[0], intermediate_time_arr_out[-1], lw=1.0, color="tab:brown", linestyle="dashdot", label="1/w=1")
    ax_twin_w.spines["right"].set_position(("outward", 120))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    payload = build_olaye_run_payload(
        time_s=t_s_plot,
        half_width_um=width_um_plot,
        params=args,
        model_variant=args["model_variant"],
        label=args["label"],
        mass_integral_initial=conc_arr[0],
        mass_integral_final=conc_arr[-1],
    )
    save_path = None
    if args["save_run"]:
        save_path = save_olaye_run_result(args["save_run_path"], payload)
        print(f"Saved run: {save_path}")

    if created_figure:
        if args["show"]:
            plt.show()
        else:
            plt.close(fig)

    return {
        "model": model,
        "figure": fig,
        "axes": ax,
        "time_s": t_s_plot,
        "half_width_um": width_um_plot,
        "payload": payload,
        "save_path": save_path,
        "params": args,
    }


def main(argv=None):
    """Runs the Figure-5 example using CLI overrides on top of notebook defaults."""
    args = build_parser().parse_args(argv)
    config = {
        "n_phase_a_nodes": args.n_phase_a_nodes,
        "n_phase_b_nodes": args.n_phase_b_nodes,
        "semiLog_dt": args.semiLog_dt,
        "model_variant": args.model_variant,
        "out": args.out,
        "save_run_path": args.save_run_path or OLAYE_NOTEBOOK_CONFIG["save_run_path"],
        "save_run": not args.no_save_run,
        "show": not args.hide_plot,
    }
    return plot_olaye_fig5_notebook(config)


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        plot_olaye_fig5_notebook()
    else:
        main()

r'''
Legacy exploratory notebook block intentionally disabled.

def deltat_to_deltax_ratio(modelDict_input):
    # return (modelDict_input["n_phase_nodes"] - 1) / modelDict_input["semi_log_points"]
    # return (modelDict_input["n_phase_nodes"] - 1) * modelDict_input["semiLog_dt"] ## not correct because dt changes
    return np.nan


def format_model_label(modelName_input, modelDict_input):
    ratio = deltat_to_deltax_ratio(modelDict_input)
    return f"{modelName_input} (dt/dx={ratio:.6g})"

for model_idx, (modelName, modelDict) in enumerate(models.items()):
    if 'error' in modelDict:
        print(f"Skipping {modelName} due to error: {modelDict['error']}")
        continue
    model = modelDict['model']
    y_arr = model.interfaceData._y[:model.interfaceData.currentIndex]
    x_arr = np.sqrt(model.interfaceData._time[:model.interfaceData.currentIndex])
    conc_time_arr = model.concData._time.copy()
    conc_x_arr = np.sqrt(conc_time_arr)
    conc_arr = model.concData._y.copy()
    modelLabel = format_model_label(modelName, modelDict)
    modelColor = model_colorway[model_idx % len(model_colorway)]
    plotly_fig.add_trace(
        go.Scatter(
            x=x_arr[:indexToPlotTo],
            y=y_arr[:indexToPlotTo],
            mode="lines",
            name=modelLabel,
            meta={
                "trace_kind": "model",
                "series_group": "interface",
                "model_name": modelName,
                "n_phase_nodes": modelDict["n_phase_nodes"],
                # "semi_log_points": modelDict["semi_log_points"],
                "semiLog_dt": modelDict["semiLog_dt"],
                "deltat_to_deltax_ratio": deltat_to_deltax_ratio(modelDict),
            },
            line={"width": styleDict["linewidth"], "color": modelColor},
            opacity=styleDict["alpha"],
            hovertemplate=(
                f"{modelLabel}<br>"
                + "sqrt(time): %{x}<br>"
                + "Liquid half-width: %{y}<extra></extra>"
            ),
        )
    )
    plotly_fig.add_trace(
        go.Scatter(
            x=conc_x_arr,
            y=conc_arr,
            mode="lines",
            name=f"{modelLabel} conc",
            meta={
                "trace_kind": "model",
                "series_group": "conc",
                "model_name": modelName,
                "n_phase_nodes": modelDict["n_phase_nodes"],
                "semiLog_dt": modelDict["semiLog_dt"],
                "deltat_to_deltax_ratio": deltat_to_deltax_ratio(modelDict),
            },
            yaxis="y2",
            line={"width": 1.0, "dash": "dash", "color": modelColor},
            opacity=0.9,
            hovertemplate=(
                f"{modelLabel} conc<br>"
                + "sqrt(time): %{x}<br>"
                + "Conc: %{y}<extra></extra>"
            ),
        )
    )

beta = 7.207320366333016e-06
indexToPlotTo_beta = np.max(np.where((2 * beta * x_arr + model.interfaceData._y[0]) < np.max(y_arr[:indexToPlotTo]))[0])
plotly_fig.add_trace(
    go.Scatter(
        x=x_arr[:indexToPlotTo_beta],
        y=2 * beta * x_arr[:indexToPlotTo_beta] + model.interfaceData._y[0],
        mode="lines",
        name=f"Analytic Solution, beta={beta}",
        meta={"trace_kind": "reference", "series_group": "interface"},
        line={"color": "gray", "dash": "dash"},
    )
)

fig5_df = pd.read_csv("C:\\Users\\samth\\OneDrive - Northwestern University\\WS_DL\\Lab Data\\Price\\code\\kawin\\examples\\Olaye2020\\Olaye2020_fig5_PresentModel_curve.csv")
fig5_df = fig5_df.sort_values(by=["time_s"])
fig5_df["half_width_m"] = fig5_df["half_width_um"] * 1e-6
if (np.sqrt(fig5_df["time_s"]) < x_arr[indexToPlotTo]).any():
    digitizedIndexToPlot = np.max(np.where(np.sqrt(fig5_df["time_s"]) < x_arr[indexToPlotTo])[0])
    plotly_fig.add_trace(
        go.Scatter(
            x=np.sqrt(fig5_df["time_s"])[:digitizedIndexToPlot],
            y=fig5_df["half_width_m"][:digitizedIndexToPlot],
            mode="markers",
            name='Fig 5 "Present Model" (digitized)',
            meta={"trace_kind": "reference", "series_group": "interface"},
            marker={"color": "gray", "symbol": "circle-open"},
        )
    )

fig5_alt_df = pd.read_csv(r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\Olaye2020\Olaye2020_fig5_alt_PresentModelRough_curve.csv")
fig5_alt_df = fig5_alt_df.sort_values(by=["time_s"])
fig5_alt_df["half_width_m"] = fig5_alt_df["half_width_um"] * 1e-6
if (np.sqrt(fig5_alt_df["time_s"]) < x_arr[indexToPlotTo]).any():
    digitizedAltRoughIndexToPlot = np.max(np.where(np.sqrt(fig5_alt_df["time_s"]) < x_arr[indexToPlotTo])[0])
    plotly_fig.add_trace(
        go.Scatter(
            x=np.sqrt(fig5_alt_df["time_s"])[:digitizedAltRoughIndexToPlot],
            y=fig5_alt_df["half_width_m"][:digitizedAltRoughIndexToPlot],
            mode="markers",
            name='Fig 5 "Present Model" alt rough (digitized)',
            meta={"trace_kind": "reference", "series_group": "interface"},
            marker={"color": "darkgray", "symbol": "circle-open"},
        )
    )

with np.load("C:\\Users\\samth\\Downloads\\LeeAndOh_NiP_results_100sec_N3013.npz") as LeeAndOh_NiP_results_loaded:
    LeeAndOh_NiP_results_loaded_df = pd.DataFrame.from_dict({item: LeeAndOh_NiP_results_loaded[item] for item in LeeAndOh_NiP_results_loaded.files}).copy()
LeeAndOh_NiP_results_loaded_df["half_width_m"] = LeeAndOh_NiP_results_loaded_df["half_width_um"] * 1e-6
if (np.sqrt(LeeAndOh_NiP_results_loaded_df["time_s"]) < x_arr[indexToPlotTo]).any():
    LeeAndOhIndexToPlot = np.max(np.where(np.sqrt(LeeAndOh_NiP_results_loaded_df["time_s"]) < x_arr[indexToPlotTo])[0])
    plotly_fig.add_trace(
        go.Scatter(
            x=np.sqrt(LeeAndOh_NiP_results_loaded_df["time_s"])[:LeeAndOhIndexToPlot],
            y=LeeAndOh_NiP_results_loaded_df["half_width_m"][:LeeAndOhIndexToPlot],
            mode="lines",
            name="Lee and Oh calculated results (N=3013)",
            meta={"trace_kind": "reference", "series_group": "interface"},
            line={"color": "black", "width": styleDict["linewidth"]},
            opacity=styleDict["alpha"],
        )
    )

with np.load("C:\\Users\\samth\\Downloads\\LeeAndOh_NiP_results_100sec_N4500.npz") as LeeAndOh_NiP_results_loaded:
    LeeAndOh_NiP_results_loaded_df = pd.DataFrame.from_dict({item: LeeAndOh_NiP_results_loaded[item] for item in LeeAndOh_NiP_results_loaded.files}).copy()
LeeAndOh_NiP_results_loaded_df["half_width_m"] = LeeAndOh_NiP_results_loaded_df["half_width_um"] * 1e-6
if (np.sqrt(LeeAndOh_NiP_results_loaded_df["time_s"]) < x_arr[indexToPlotTo]).any():
    LeeAndOhIndexToPlot = np.max(np.where(np.sqrt(LeeAndOh_NiP_results_loaded_df["time_s"]) < x_arr[indexToPlotTo])[0])
    plotly_fig.add_trace(
        go.Scatter(
            x=np.sqrt(LeeAndOh_NiP_results_loaded_df["time_s"])[:LeeAndOhIndexToPlot],
            y=LeeAndOh_NiP_results_loaded_df["half_width_m"][:LeeAndOhIndexToPlot],
            mode="lines",
            name="Lee and Oh calculated results (N=4500)",
            meta={"trace_kind": "reference", "series_group": "interface"},
            line={"color": "black", "width": styleDict["linewidth"], "dash": "dot"},
            opacity=styleDict["alpha"],
        )
    )

plotly_fig.add_hline(
    y=theoreticalMaxLiquidWidth,
    line_color="firebrick",
    line_dash="dashdot",
    annotation_text="Theoretical max liquid width",
    annotation_position="top left",
)

idealized_conc = compute_idealized_conc(FIG5_BASE_PARAMS)
# plotly_fig.add_trace(
#     go.Scatter(
#         x=[conc_x_arr[0], conc_x_arr[-1]],
#         y=[conc_arr[0], conc_arr[0]],
#         mode="lines",
#         name="Initial conc",
#         meta={"trace_kind": "reference"},
#         yaxis="y2",
#         line={"color": "#2ca02c", "width": 1.0, "dash": "dashdot"},
#         hovertemplate="Initial conc<br>sqrt(time): %{x}<br>Conc: %{y}<extra></extra>",
#     )
# )
plotly_fig.add_trace(
    go.Scatter(
        x=[conc_x_arr[0], conc_x_arr[-1]],
        y=[idealized_conc, idealized_conc],
        mode="lines",
        name="Idealized conc",
        meta={"trace_kind": "reference", "series_group": "conc"},
        yaxis="y2",
        line={"color": "#2ca02c", "width": 1.0, "dash": "dash"},
        hovertemplate="Idealized conc<br>sqrt(time): %{x}<br>Conc: %{y}<extra></extra>",
    )
)

plotly_fig.update_layout(
    width=1200,
    height=800,
    template="plotly_white",
    colorway=model_colorway,
    xaxis_title="sqrt(time) [s^0.5]",
    yaxis_title="Liquid half-width [m]",
    yaxis2={
        "title": "conc",
        "overlaying": "y",
        "side": "right",
        "showgrid": False,
    },
    hoverlabel={
        "font_size": 14,
        "namelength": -1,
    },
    title=f"MODEL_VARIANT: {MODEL_VARIANT}"
)

plotly_script_dir = pathlib.Path(__file__).resolve().parent
plotly_html_path = plotly_script_dir / "olaye2020_fig5_plotly_filters.html"
plotly_div_id = "olaye2020-fig5-plotly"
unique_n_phase_nodes = sorted({modelDict["n_phase_nodes"] for modelDict in models.values() if "error" not in modelDict})
# unique_semi_log_points = sorted({modelDict["semi_log_points"] for modelDict in models.values() if "error" not in modelDict})
unique_semiLog_dt = sorted({modelDict["semiLog_dt"] for modelDict in models.values() if "error" not in modelDict})
plotly_figure_html = plotly_fig.to_html(full_html=False, include_plotlyjs=True, div_id=plotly_div_id)

filter_page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Olaye 2020 Figure 5 Plotly Filters</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background: #f7f7f7;
      color: #222;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 16px;
      min-height: 100vh;
      padding: 16px;
      box-sizing: border-box;
    }}
    .controls {{
      background: #fff;
      border: 1px solid #d8d8d8;
      border-radius: 10px;
      padding: 16px;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    }}
    .control-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .control-group h3 {{
      margin: 0 0 8px 0;
      font-size: 16px;
    }}
    .control-actions {{
      display: flex;
      gap: 8px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}
    .control-actions button {{
      border: 1px solid #bbb;
      border-radius: 6px;
      background: #fafafa;
      padding: 4px 8px;
      cursor: pointer;
    }}
    .checkbox-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 70vh;
      overflow-y: auto;
      padding-right: 4px;
    }}
    .checkbox-list label {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
    }}
    .plot-card {{
      background: #fff;
      border: 1px solid #d8d8d8;
      border-radius: 10px;
      padding: 8px;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    }}
    .hint {{
      margin-top: 14px;
      font-size: 13px;
      color: #555;
      line-height: 1.4;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="controls">
      <div class="control-grid">
        <section class="control-group">
          <h3>`n_phase_nodes`</h3>
          <div class="control-actions">
            <button type="button" onclick="setAllChecked('n_phase_nodes', true)">All</button>
            <button type="button" onclick="setAllChecked('n_phase_nodes', false)">None</button>
          </div>
          <div class="checkbox-list" id="n_phase_nodes-options"></div>
        </section>
        <section class="control-group">
          <h3>`semiLog_dt`</h3>
          <div class="control-actions">
            <button type="button" onclick="setAllChecked('semiLog_dt', true)">All</button>
            <button type="button" onclick="setAllChecked('semiLog_dt', false)">None</button>
          </div>
          <div class="checkbox-list" id="semiLog_dt-options"></div>
        </section>
      </div>
      <section class="control-group" style="margin-top: 16px;">
        <h3>Trace Groups</h3>
        <div class="checkbox-list">
          <label><input type="checkbox" id="toggle-interface-traces" checked> Show interface-position traces</label>
          <label><input type="checkbox" id="toggle-conc-traces" checked> Show concentration traces</label>
        </div>
      </section>
      <div class="hint">
        Checked values remain visible. A model trace is shown only when both its `n_phase_nodes`
        and `semiLog_dt` values are currently selected. The trace-group toggles let you hide or
        show all interface-position curves and concentration curves independently.
      </div>
    </aside>
    <main class="plot-card">
      {plotly_figure_html}
    </main>
  </div>
  <script>
    const uniqueNPhaseNodes = {json.dumps(unique_n_phase_nodes)};
    const uniqueSemiLogDt = {json.dumps(unique_semiLog_dt)};
    const plotDiv = document.getElementById("{plotly_div_id}");

    function buildCheckboxes(containerId, filterName, values) {{
      const container = document.getElementById(containerId);
      values.forEach((value) => {{
        const label = document.createElement("label");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = true;
        checkbox.dataset.filterName = filterName;
        checkbox.value = String(value);
        checkbox.addEventListener("change", updateTraceVisibility);
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(String(value)));
        container.appendChild(label);
      }});
    }}

    function getSelectedValues(filterName) {{
      return new Set(
        Array.from(document.querySelectorAll(`input[data-filter-name="${{filterName}}"]:checked`))
          .map((input) => input.value)
      );
    }}

    function setAllChecked(filterName, checked) {{
      document.querySelectorAll(`input[data-filter-name="${{filterName}}"]`).forEach((input) => {{
        input.checked = checked;
      }});
      updateTraceVisibility();
    }}

    function updateTraceVisibility() {{
      const selectedNPhaseNodes = getSelectedValues("n_phase_nodes");
      const selectedSemiLogPoints = getSelectedValues("semiLog_dt");
      const showInterfaceTraces = document.getElementById("toggle-interface-traces").checked;
      const showConcTraces = document.getElementById("toggle-conc-traces").checked;
      const visibility = plotDiv.data.map((trace) => {{
        const meta = trace.meta || {{}};
        if (meta.series_group === "interface" && !showInterfaceTraces) {{
          return false;
        }}
        if (meta.series_group === "conc" && !showConcTraces) {{
          return false;
        }}
        if (meta.trace_kind !== "model") {{
          return true;
        }}
        return selectedNPhaseNodes.has(String(meta.n_phase_nodes))
          && selectedSemiLogPoints.has(String(meta.semiLog_dt));
      }});
      Plotly.restyle(plotDiv, {{visible: visibility}});
    }}

    buildCheckboxes("n_phase_nodes-options", "n_phase_nodes", uniqueNPhaseNodes);
    buildCheckboxes("semiLog_dt-options", "semiLog_dt", uniqueSemiLogDt);
    document.getElementById("toggle-interface-traces").addEventListener("change", updateTraceVisibility);
    document.getElementById("toggle-conc-traces").addEventListener("change", updateTraceVisibility);
    updateTraceVisibility();
  </script>
</body>
</html>

plotly_html_path.write_text(filter_page_html, encoding="utf-8")
print(f"Saved interactive Plotly filter page: {plotly_html_path}")
webbrowser.open(plotly_html_path.resolve().as_uri())

'''

# %%
