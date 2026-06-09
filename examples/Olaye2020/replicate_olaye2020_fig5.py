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
import pathlib

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from kawin.diffusion import MovingBoundaryOlayeFD1DModel, TemperatureParameters
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.solver import explicitEulerIterator

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
    n_phase_nodes: int,
    t_end_s: float,
    dt_mode: str,
    # semi_log_points: int,
    semiLog_dt: float,
    semiLogT0: float,

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

    model = MovingBoundaryOlayeFD1DModel(
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
        phase_a_nodes=int(n_phase_nodes),
        phase_b_nodes=int(n_phase_nodes),
        # semi_log_points=semi_log_points,
        semiLog_dt=semiLog_dt,
        semiLogT0=semiLogT0,
        record=True,
    )

    model.solve(float(t_end_s), iterator=explicitEulerIterator, verbose=True, vIt=100, minDtFrac=1e-13)

    t_s = np.array(model.interfaceData._time[: model.interfaceData.N + 1], dtype=np.float64)
    s_m = np.array(model.interfaceData._y[: model.interfaceData.N + 1], dtype=np.float64)
    liquid_half_width_um = s_m * 1e6
    # t_h = t_s / 3600.0
    return t_s, liquid_half_width_um, model


def build_parser():
    parser = argparse.ArgumentParser(description="Replicate Olaye & Ojo 2020 Figure 5 using the Olaye FD model.")
    parser.add_argument("--R-um", type=float, default=3012.5, help="Half-domain size R in micrometers.")
    parser.add_argument("--s0-um", type=float, default=12.5, help="Initial liquid half-width s0 in micrometers.")
    parser.add_argument("--c-liquid0-pct", type=float, default=19.0, help="Initial liquid composition (percent).")
    parser.add_argument("--c-solid0-pct", type=float, default=0.0, help="Initial solid composition (percent).")
    parser.add_argument("--c-liquid-int-pct", type=float, default=10.223, help="Liquid-side interface composition (percent).")
    parser.add_argument("--c-solid-int-pct", type=float, default=0.166, help="Solid-side interface composition (percent).")
    parser.add_argument("--D-liquid-base", type=float, default=500.0, help="Liquid diffusivity base value from table.")
    parser.add_argument("--D-solid-base", type=float, default=18.0, help="Solid diffusivity base value from table.")
    parser.add_argument("--D-scale", type=float, default=1e-12, help="Scale to convert table diffusivities to m^2/s.")
    parser.add_argument("--n-nodes", type=int, default=3013+1, help="Number of FD nodes.")
    parser.add_argument("--t-end-h", type=float, default=0.005, help="Simulation end time (hours).")
    # parser.add_argument("--semi_log_points", type=int, default=200, help="Number of semi-log points.")
    parser.add_argument("--semiLog_dt", type=float, default=1e-3, help="Time step for semi-log scaling.")
    parser.add_argument("--semiLogT0", type=float, default=1e-6, help="First dt")
    parser.add_argument(
        "--dt-mode",
        type=str,
        choices=["cfl", "semi_log_optional"],
        default="semi_log_optional",
        help="Time-step policy for Olaye model.",
    )
    parser.add_argument("--exp-csv", type=str, default=None, help="Optional CSV with experimental points: time_h,half_width_um.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output figure path. Default: examples/Olaye2020/olaye2020_fig5_replication.png",
    )
    parser.add_argument("--show", action="store_true", help="Show the matplotlib window.")
    return parser


def main(n_phase_nodes, semiLog_dt):
    # args = build_parser().parse_args()
    DEFAULT_FIG5_ARGS = {
    "R_um": 3012.5,
    "s0_um": 12.5,
    "c_liquid0_pct": 19.0,
    "c_solid0_pct": 0.0,
    "c_liquid_int_pct": 10.223,
    "c_solid_int_pct": 0.166,
    "D_liquid_base": 500.0,
    "D_solid_base": 18.0,
    "D_scale": 1e-12,
    "n_nodes": 3013+1,
    "n_phase_nodes": n_phase_nodes,
    "t_end_s": 1e4, # 0.005*3600
    "dt_mode": "semi_log_optional",
    # "semi_log_points": semi_log_points,
    "semiLog_dt": semiLog_dt,
    "semiLogT0": 1.8e-5,
    "exp_csv": r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\Olaye2020\Olaye2020_fig5_PresentModel_curve.csv",
    "out": None,
    "show": True,
    }
    args=DEFAULT_FIG5_ARGS.copy()

    script_dir = pathlib.Path(__file__).resolve().parent
    out_path = (
        pathlib.Path(args['out']).resolve()
        if args['out'] is not None
        # pathlib.Path(args.out).resolve()
        # if args.out is not None
        else script_dir / "olaye2020_fig5_replication.png"
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
        n_phase_nodes=args["n_phase_nodes"],
        t_end_s=args["t_end_s"],
        dt_mode=args["dt_mode"],
        # semi_log_points=args["semi_log_points"],
        semiLog_dt=args["semiLog_dt"],
        semiLogT0=args["semiLogT0"],
    )
    # t_h, width_um, model = run_case(
    #     R_um=args.R_um,
    #     s0_um=args.s0_um,
    #     c_liquid0_pct=args.c_liquid0_pct,
    #     c_solid0_pct=args.c_solid0_pct,
    #     c_liquid_int_pct=args.c_liquid_int_pct,
    #     c_solid_int_pct=args.c_solid_int_pct,
    #     D_liquid_base=args.D_liquid_base,
    #     D_solid_base=args.D_solid_base,
    #     D_scale=args.D_scale,
    #     n_nodes=args.n_nodes,
    #     n_phase_nodes=args.n_phase_nodes,
    #     t_end_s=args.t_end_s,
    #     dt_mode=args.dt_mode,
    # )

    mask = np.isfinite(t_s) & np.isfinite(width_um)
    t_s_plot = t_s[mask]
    width_um_plot = width_um[mask]
    if t_s_plot.size < 2:
        raise ValueError("Simulation did not produce enough finite points to plot.")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)
    ax.plot(t_s_plot, width_um_plot, lw=2.0, color="tab:blue", label="Olaye model (this work)")

    exp_csv = args["exp_csv"]
    # exp_csv = args.exp_csv
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

    # ax.set_xlabel("Time (h)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Liquid half-width (um)")
    ax.set_title("Olaye & Ojo (2020) Figure 5 Replication")

    x_min = float(np.min(t_s_plot))
    x_max = float(np.max(t_s_plot))
    y_min = float(np.min(width_um_plot))
    y_max = float(np.max(width_um_plot))
    if not np.isfinite(x_min + x_max + y_min + y_max):
        raise ValueError("Plot limits are not finite.")
    if x_max <= x_min:
        pad_x = max(1e-6, abs(x_min) * 1e-3 + 1e-6)
        x_min -= pad_x
        x_max += pad_x
    if y_max <= y_min:
        pad_y = max(1e-6, abs(y_min) * 1e-3 + 1e-6)
        y_min -= pad_y
        y_max += pad_y
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xscale('log')
    ax.set_xlim(0.1, x_max)

    ax.grid(True, alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved: {out_path}")

    # if args.show:
    if args["show"]:
        plt.show()
    else:
        plt.close(fig)

    return model

if __name__ == "__main__":
    models={}
    # semi_log_points_inputs = [2500]#, 5000, 10000, 20000, 50000]
    semiLog_dt_inputs = [0.002763654842561367, 0.002763654842561367/10]#, 5000, 10000, 20000, 50000]
    n_phase_nodes_inputs = [25+1, 50+1]#, 100+1, 200+1]

    conds_list=[]
    for semiLog_dt_input in semiLog_dt_inputs:
        for n_phase_nodes_input in n_phase_nodes_inputs:
            conds_list.append({'n_phase_nodes': n_phase_nodes_input, 'semiLog_dt': semiLog_dt_input})

    for cond in conds_list:
        t0=time.perf_counter()
        try:
            print(cond)
            model = main(**cond)
        except Exception as e:
            print(f"Error occurred while running case {cond}: {e}")
            models.update({f"model_{cond['n_phase_nodes']}_{cond['semiLog_dt']}": {**cond, 'error':str(e)}})
            continue
        t1=time.perf_counter()
        models.update({f"model_{cond['n_phase_nodes']}_{cond['semiLog_dt']}": {**cond, 'model': model, 'runtime':(t1-t0)}})

# %%

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# styleDict = {'marker':'o', 'markersize':2}
styleDict = {'linewidth':1, 'alpha':0.75}
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 8))
indexToPlotTo=-1
xMax = -np.inf
# for modelName, model in {"model_20000":model_20000, "model_10000":model_10000, "model_5000":model_5000}.items():
for modelName, modelDict in models.items():
    if 'error' in modelDict:
        print(f"Skipping {modelName} due to error: {modelDict['error']}")
        continue
    model=modelDict['model']
    # y_arr = (model.interfaceData._y[:model.interfaceData.currentIndex]-model.interfaceData._y[0])
    y_arr = (model.interfaceData._y[:model.interfaceData.currentIndex])
    x_arr = np.sqrt(model.interfaceData._time[:model.interfaceData.currentIndex])
    # ax.plot(x_arr[:indexToPlotTo], y_arr[:indexToPlotTo], label=f"Olaye model calculated results {modelName.split('_')[-1]} t-pts", zorder=3, **styleDict)
    ax.plot(x_arr[:indexToPlotTo], y_arr[:indexToPlotTo], label=f"Olaye model calculated results {modelName}", zorder=3, **styleDict)

    # if np.max(x_arr)>xMax
beta = 7.207320366333016e-06
indexToPlotTo_beta = np.max(np.where((2*beta*x_arr + model.interfaceData._y[0])<np.max(y_arr[:indexToPlotTo]))[0])
# indexToPlotTo_beta=indexToPlotTo
ax.plot(x_arr[:indexToPlotTo_beta], 2*beta*x_arr[:indexToPlotTo_beta] + model.interfaceData._y[0], label=f'Analytic Solution, beta={beta}', color='gray', linestyle='dashed', zorder=-1)

import pandas as pd
fig5_df = pd.read_csv("C:\\Users\\samth\\OneDrive - Northwestern University\\WS_DL\\Lab Data\\Price\\code\\kawin\\examples\\Olaye2020\\Olaye2020_fig5_PresentModel_curve.csv")
fig5_df = fig5_df.sort_values(by=['time_s'])
fig5_df['half_width_m'] = fig5_df['half_width_um']*1e-6
if (np.sqrt(fig5_df['time_s'])<x_arr[indexToPlotTo]).any():
    digitizedIndexToPlot = np.max(np.where(np.sqrt(fig5_df['time_s'])<x_arr[indexToPlotTo])[0])
    ax.plot(np.sqrt(fig5_df['time_s'])[:digitizedIndexToPlot], fig5_df['half_width_m'][:digitizedIndexToPlot], 'o', fillstyle='none', color='tab:gray',  label='Fig 5 "Present Model" (digitized)', zorder=-2)

with np.load("C:\\Users\\samth\\Downloads\\LeeAndOh_NiP_results_100sec_N3013.npz") as LeeAndOh_NiP_results_loaded:
    LeeAndOh_NiP_results_loaded_df = pd.DataFrame.from_dict({item: LeeAndOh_NiP_results_loaded[item] for item in LeeAndOh_NiP_results_loaded.files}).copy()
LeeAndOh_NiP_results_loaded_df['half_width_m'] = LeeAndOh_NiP_results_loaded_df['half_width_um'] * 1e-6
if (np.sqrt(LeeAndOh_NiP_results_loaded_df['time_s'])<x_arr[indexToPlotTo]).any():
    LeeAndOhIndexToPlot = np.max(np.where(np.sqrt(LeeAndOh_NiP_results_loaded_df['time_s'])<x_arr[indexToPlotTo])[0])
    ax.plot(np.sqrt(LeeAndOh_NiP_results_loaded_df['time_s'])[:LeeAndOhIndexToPlot], LeeAndOh_NiP_results_loaded_df['half_width_m'][:LeeAndOhIndexToPlot], label='Lee and Oh calculated results (N=3013)', zorder=2, color='k', **styleDict)

with np.load("C:\\Users\\samth\\Downloads\\LeeAndOh_NiP_results_100sec_N4500.npz") as LeeAndOh_NiP_results_loaded:
    LeeAndOh_NiP_results_loaded_df = pd.DataFrame.from_dict({item: LeeAndOh_NiP_results_loaded[item] for item in LeeAndOh_NiP_results_loaded.files}).copy()
LeeAndOh_NiP_results_loaded_df['half_width_m'] = LeeAndOh_NiP_results_loaded_df['half_width_um'] * 1e-6
if (np.sqrt(LeeAndOh_NiP_results_loaded_df['time_s'])<x_arr[indexToPlotTo]).any():
    LeeAndOhIndexToPlot = np.max(np.where(np.sqrt(LeeAndOh_NiP_results_loaded_df['time_s'])<x_arr[indexToPlotTo])[0])
    ax.plot(np.sqrt(LeeAndOh_NiP_results_loaded_df['time_s'])[:LeeAndOhIndexToPlot], LeeAndOh_NiP_results_loaded_df['half_width_m'][:LeeAndOhIndexToPlot], label='Lee and Oh calculated results (N=4500)', zorder=2, color='k', linestyle='dotted', **styleDict)

plt.legend()
plt.show(block=True)

theoreticalMaxLiquidWidth = (0.18999998/0.10223) * 12.5e-6
print(f"max liquid width in sim:      {np.max(model.interfaceData._y)}")
print(f"theoretical max liquid width: {(0.19/0.10223) * 12.5e-6}")   ## {(c_a0/c_a_eq) * l_a}
print(f"theoretical max liquid width: {theoreticalMaxLiquidWidth}")   ## {(c_a0/c_a_eq) * l_a}
print([f"({modelDict['n_phase_nodes']-1}^2 / {modelDict['semi_log_points']}): {((modelDict['n_phase_nodes']-1)**2)/modelDict['semi_log_points']}" for modelDict in models.values()])

# %%
import plotly.graph_objects as go
import json
import webbrowser

styleDict = {'linewidth':1, 'alpha':0.75}

plotly_fig = go.Figure()
indexToPlotTo = -1
theoreticalMaxLiquidWidth = (0.19 / 0.10223) * 12.5e-6

def deltat_to_deltax_ratio(modelDict_input):
    # return (modelDict_input["n_phase_nodes"] - 1) / modelDict_input["semi_log_points"]
    # return (modelDict_input["n_phase_nodes"] - 1) * modelDict_input["semiLog_dt"] ## not correct because dt changes
    return np.nan


def format_model_label(modelName_input, modelDict_input):
    ratio = deltat_to_deltax_ratio(modelDict_input)
    return f"{modelName_input} (dt/dx={ratio:.6g})"

for modelName, modelDict in models.items():
    if 'error' in modelDict:
        print(f"Skipping {modelName} due to error: {modelDict['error']}")
        continue
    model = modelDict['model']
    y_arr = model.interfaceData._y[:model.interfaceData.currentIndex]
    x_arr = np.sqrt(model.interfaceData._time[:model.interfaceData.currentIndex])
    modelLabel = format_model_label(modelName, modelDict)
    plotly_fig.add_trace(
        go.Scatter(
            x=x_arr[:indexToPlotTo],
            y=y_arr[:indexToPlotTo],
            mode="lines",
            name=modelLabel,
            meta={
                "trace_kind": "model",
                "model_name": modelName,
                "n_phase_nodes": modelDict["n_phase_nodes"],
                # "semi_log_points": modelDict["semi_log_points"],
                "semiLog_dt": modelDict["semiLog_dt"],
                "deltat_to_deltax_ratio": deltat_to_deltax_ratio(modelDict),
            },
            line={"width": styleDict["linewidth"]},
            opacity=styleDict["alpha"],
            hovertemplate=(
                f"{modelLabel}<br>"
                + "sqrt(time): %{x}<br>"
                + "Liquid half-width: %{y}<extra></extra>"
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
        meta={"trace_kind": "reference"},
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
            meta={"trace_kind": "reference"},
            marker={"color": "gray", "symbol": "circle-open"},
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
            meta={"trace_kind": "reference"},
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
            meta={"trace_kind": "reference"},
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

plotly_fig.update_layout(
    width=1200,
    height=800,
    template="plotly_white",
    colorway=[
        "#3366CC", "#DC3912", "#FF9900", "#109618", "#990099",
        "#0099C6", "#DD4477", "#66AA00", "#B82E2E", "#316395",
    ],
    xaxis_title="sqrt(time) [s^0.5]",
    yaxis_title="Liquid half-width [m]",
    hoverlabel={
        "font_size": 14,
        "namelength": -1,
    },
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
      <div class="hint">
        Checked values remain visible. A model trace is shown only when both its `n_phase_nodes`
        and `semiLog_dt` values are currently selected. Reference curves stay visible.
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
      const visibility = plotDiv.data.map((trace) => {{
        const meta = trace.meta || {{}};
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
    updateTraceVisibility();
  </script>
</body>
</html>
"""

plotly_html_path.write_text(filter_page_html, encoding="utf-8")
print(f"Saved interactive Plotly filter page: {plotly_html_path}")
webbrowser.open(plotly_html_path.resolve().as_uri())



# %%
