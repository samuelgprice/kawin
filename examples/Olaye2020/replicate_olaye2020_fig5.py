#%%

# %matplotlib inline
"""
Replicate selected Olaye & Ojo (2020) figures with the Olaye moving-boundary FD model.

This script runs the binary planar moving-interface solver and plots either
Figure 5 liquid half-width for a Ni-P TLP-style setup or Figure 6 interface
displacement for alpha-beta brass diffusion couples.

Notes
-----
- The model implementation currently supports planar geometry only.
- By default this script uses Figure 5 Table-2-style Ni-P parameters from the
  paper (converted from percent to mole-fraction-like units and micrometers to
  SI).
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
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
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
    "t_end_s": 8.765e4,
    "dt_mode": "semi_log_optional",
    "semiLogT0": 1e-6,
    "exp_csv": r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\Olaye2020\figureDataExtraction\Olaye2020_fig5_PresentModel_curve.csv",
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
    "figures": ("fig6",),
    "n_phase_a_nodes": 51,
    "n_phase_b_nodes": 228,
    "semiLog_dt": 0.0025 / 10.0, #0.002763654842561367 / 1.0,
    "fig6_layers": ("thick"),#, "thin"),
    "fig6_d_alpha_cm2_s": (2.5e-8,),# 2.5e-8,),
    "fig6_n_phase_a_nodes": 50,#51,
    "fig6_n_phase_b_nodes": 132, #68,
    "fig6_semiLog_dt": 0.0025 / 10.0,
    "fig6_t_end_s": 1e4, #7e5,
    "fig6_out": None,
    "fig6_plot_extracted_data": True,
    "fig6_plot_concentration_info": True,
    "fig6_plot_sqrt_time_analytical": True,
    "fig6_sqrt_time_out": None,
    "fig6_sqrt_time_plot_extracted_data": True,
    "fig6_sqrt_time_max_s": "auto",
    "fig6_sqrt_time_max_points_per_case": None,
    "fig6_sqrt_time_semi_infinite_erfc_argument_min": 1.05,
    "fig6_save_run_path": None,
    "model_variant": MODEL_VARIANT,
    "out": None,
    "show": True,
    "save_run": True,
    "save_run_path": SCRIPT_DIR / "olaye2020_fig5_saved_run.npz",
    "label": None,
    "timeProfiling":False,
    "record_pq_data": False,
    "preallocate_recordings": True,
}


FIG6_LAYER_PARAMS = {
    "thin": {
        "label": "Thin initial beta layer",
        "s0_um": 190.5,
        "R_um": 565.0,
    },
    "thick": {
        "label": "Thick initial beta layer",
        "s0_um": 381.0,
        "R_um": 755.5,
    },
}

FIG6_BASE_PARAMS = {
    "c_beta0_pct": 39.4,
    "c_alpha0_pct": 29.1,
    "c_beta_int_pct": 36.9,
    "c_alpha_int_pct": 32.5,
    "D_beta_um2_s": 100.0,
    "D_scale": 1e-12,
    "dt_mode": "semi_log_optional",
    "semiLogT0": 1e-5,
}

FIG6_VALID_D_ALPHA_CM2_S = (1.4e-8, 2.5e-8)

FIG6_EXTRACTED_DATA_FILES = {
    ("thin", 1.4e-8): {
        "filename": "Olaye2020_fig6_ThinBeta_D1pt4_DF_curve.csv",
        "label": "Digitized Fig. 6 thin beta, D_alpha=1.4E-8",
        "color": "tab:blue",
        "marker": "o",
    },
    ("thin", 2.5e-8): {
        "filename": "Olaye2020_fig6_ThinBeta_D2pt5_DF_curve.csv",
        "label": "Digitized Fig. 6 thin beta, D_alpha=2.5E-8",
        "color": "tab:red",
        "marker": "o",
    },
    ("thick", 1.4e-8): {
        "filename": "Olaye2020_fig6_ThickBeta_D1pt4_DF_curve.csv",
        "label": "Digitized Fig. 6 thick beta, D_alpha=1.4E-8",
        "color": "tab:orange",
        "marker": "D",
    },
    ("thick", 2.5e-8): {
        "filename": "Olaye2020_fig6_ThickBeta_D2pt5_DF_curve.csv",
        "label": "Digitized Fig. 6 thick beta, D_alpha=2.5E-8",
        "color": "tab:purple",
        "marker": "D",
    },
}


def _normalize_requested_items(value, valid_values, *, label):
    """Normalizes a string or sequence selector against allowed values."""
    valid = tuple(valid_values)
    if isinstance(value, str):
        requested = (value,)
    else:
        requested = tuple(value)
    normalized = []
    for item in requested:
        item_text = str(item).strip().lower()
        if item_text in {"all", "both"}:
            normalized.extend(valid)
            continue
        if all(str(v).startswith("fig") for v in valid):
            if item_text.startswith("fig"):
                item_text = f"fig{item_text[3:]}"
            elif item_text.isdigit():
                item_text = f"fig{item_text}"
        if item_text not in valid:
            raise ValueError(f"Unknown {label}: {item!r}. Expected one of {valid}.")
        normalized.append(item_text)
    return tuple(dict.fromkeys(normalized))


def _selected_figures(config):
    """Returns normalized figure names requested by the notebook config."""
    figures = config.get("figures", ("fig5",))
    normalized = _normalize_requested_items(figures, ("fig5", "fig6"), label="figure")
    return normalized


def _selected_fig6_layers(config):
    """Returns normalized Figure-6 beta-layer selections."""
    return _normalize_requested_items(config.get("fig6_layers", ("thin", "thick")), ("thin", "thick"), label="Figure 6 layer")


def _selected_fig6_d_alpha_cm2_s(config):
    """Returns normalized Figure-6 alpha diffusivity selections in ``cm^2/s``."""
    values = config.get("fig6_d_alpha_cm2_s", FIG6_VALID_D_ALPHA_CM2_S)
    if isinstance(values, str):
        if values.strip().lower() in {"all", "both"}:
            requested = FIG6_VALID_D_ALPHA_CM2_S
        else:
            requested = (float(values),)
    else:
        requested = tuple(float(value) for value in values)
    normalized = []
    for value in requested:
        matches = [valid for valid in FIG6_VALID_D_ALPHA_CM2_S if np.isclose(value, valid, rtol=1e-12, atol=0.0)]
        if not matches:
            raise ValueError(
                "Unknown Figure 6 alpha diffusivity "
                f"{value!r}. Expected one of {FIG6_VALID_D_ALPHA_CM2_S} cm^2/s."
            )
        normalized.append(matches[0])
    return tuple(dict.fromkeys(normalized))


def _cm2_s_to_um2_s(value):
    """Converts diffusivity from ``cm^2/s`` to ``um^2/s``."""
    return float(value) * 1e8


def _load_no_header_xy_csv(path_or_file):
    """
    Loads a no-header two-column digitized figure CSV.

    Returns finite ``x`` and ``y`` arrays in the units encoded by the file.
    """
    data = np.loadtxt(path_or_file, delimiter=",", dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"{path_or_file} must contain at least two columns.")
    mask = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    return data[mask, 0], data[mask, 1]


def build_fig6_case_params(layer, d_alpha_cm2_s, config=None):
    """
    Builds corrected Table-2 Figure-6 inputs for one brass validation case.

    The brass rows are interpreted with phase A as beta and phase B as alpha.
    The printed beta-layer thicknesses are corrected to the half-widths used as
    the initial planar interface positions.
    """
    cfg = OLAYE_NOTEBOOK_CONFIG if config is None else {**OLAYE_NOTEBOOK_CONFIG, **dict(config)}
    layer_key = _normalize_requested_items((layer,), ("thin", "thick"), label="Figure 6 layer")[0]
    d_alpha_value = _selected_fig6_d_alpha_cm2_s({"fig6_d_alpha_cm2_s": (d_alpha_cm2_s,)})[0]
    layer_params = FIG6_LAYER_PARAMS[layer_key]
    return {
        "figure": "fig6",
        "layer": layer_key,
        "d_alpha_cm2_s": d_alpha_value,
        "d_alpha_um2_s": _cm2_s_to_um2_s(d_alpha_value),
        "R_um": layer_params["R_um"],
        "s0_um": layer_params["s0_um"],
        "c_liquid0_pct": FIG6_BASE_PARAMS["c_beta0_pct"],
        "c_solid0_pct": FIG6_BASE_PARAMS["c_alpha0_pct"],
        "c_liquid_int_pct": FIG6_BASE_PARAMS["c_beta_int_pct"],
        "c_solid_int_pct": FIG6_BASE_PARAMS["c_alpha_int_pct"],
        "D_liquid_base": FIG6_BASE_PARAMS["D_beta_um2_s"],
        "D_solid_base": _cm2_s_to_um2_s(d_alpha_value),
        "D_scale": FIG6_BASE_PARAMS["D_scale"],
        "n_nodes": int(round(layer_params["R_um"])) + 1,
        "n_phase_a_nodes": int(cfg["fig6_n_phase_a_nodes"]),
        "n_phase_b_nodes": int(cfg["fig6_n_phase_b_nodes"]),
        "t_end_s": float(cfg["fig6_t_end_s"]),
        "dt_mode": FIG6_BASE_PARAMS["dt_mode"],
        "semiLog_dt": float(cfg["fig6_semiLog_dt"]),
        "semiLogT0": FIG6_BASE_PARAMS["semiLogT0"],
        "model_variant": cfg["model_variant"],
        "response_name": "ZN",
        "elements": ("CU", "ZN"),
        "phase_a_name": "BETA",
        "phase_b_name": "ALPHA",
        "label": f"{layer_params['label']}, D_alpha={d_alpha_value:.1E} cm^2/s",
        "record_pq_data": bool(cfg.get("record_pq_data", True)),
        "preallocate_recordings": bool(cfg.get("preallocate_recordings", False)),
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
    params,
    model_variant,
    half_width_um=None,
    interface_displacement_um=None,
    model_family="olaye_fig5",
    label=None,
    mass_integral_initial=None,
    mass_integral_final=None,
):
    """
    Builds the saved-run payload for notebook-friendly Olaye comparisons.

    The saved arrays use seconds and micrometers. Figure 5 stores
    ``half_width_um``; Figure 6 stores ``interface_displacement_um``.
    """
    if half_width_um is None and interface_displacement_um is None:
        raise ValueError("At least one plotted quantity must be supplied.")
    payload_params = {
        key: value
        for key, value in params.items()
        if key not in {"show", "out", "fig6_out", "save_run", "save_run_path", "fig6_save_run_path", "label"}
    }
    default_label = f"Olaye Figure 5 ({model_variant})" if str(model_family) == "olaye_fig5" else f"Olaye Figure 6 ({model_variant})"
    payload = {
        "time_s": np.asarray(time_s, dtype=np.float64),
        "label": np.array(label or default_label),
        "source_script": np.array("examples/Olaye2020/replicate_olaye2020_fig5.py"),
        "model_family": np.array(str(model_family)),
        "model_variant": np.array(str(model_variant)),
        "params_json": np.array(json.dumps(_jsonable(payload_params), sort_keys=True)),
    }
    if half_width_um is not None:
        payload["half_width_um"] = np.asarray(half_width_um, dtype=np.float64)
    if interface_displacement_um is not None:
        payload["interface_displacement_um"] = np.asarray(interface_displacement_um, dtype=np.float64)
    if str(model_family) == "olaye_fig5" and {"s0_um", "c_liquid0_pct", "c_liquid_int_pct"}.issubset(params):
        payload["theoretical_max_um"] = np.array([theoretical_fig5_max_liquid_half_width_um(params)], dtype=np.float64)
    if str(model_family) == "olaye_fig5" and {"s0_um", "R_um", "c_liquid0_pct", "c_solid0_pct"}.issubset(params):
        payload["idealized_mass_integral"] = np.array([compute_idealized_conc(params)], dtype=np.float64)
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
    record_pq_data: bool = True,
    preallocate_recordings: bool = False,
    response_name: str = "P",
    elements=("NI", "P"),
    phase_a_name: str = "LIQUID",
    phase_b_name: str = "SOLID",
    print_beta: bool = True,

):
    """Runs one planar Olaye simulation and returns time and phase-A width."""
    R_m = float(R_um) * 1e-6
    s0_m = float(s0_um) * 1e-6
    interface_position_m = s0_m

    # Percent inputs from the paper -> fraction-like internal values.
    c_liquid0 = float(c_liquid0_pct) / 100.0
    c_solid0 = float(c_solid0_pct) / 100.0
    c_liquid_int = float(c_liquid_int_pct) / 100.0
    c_solid_int = float(c_solid_int_pct) / 100.0

    # Match the paper convention directly: left side is phase A, right side is phase B.
    profile = ProfileBuilder([(StepProfile1D(interface_position_m, c_liquid0, c_solid0), response_name)])
    mesh = CartesianFD1D([response_name], [0.0, R_m], int(n_nodes))
    mesh.setResponseProfile(profile)

    therm = ConstantBinaryThermodynamics(
        phases=[phase_a_name, phase_b_name],
        diffusivities={
            phase_a_name: float(D_liquid_base) * float(D_scale),
            phase_b_name: float(D_solid_base) * float(D_scale),
        },
    )


    if print_beta:
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
    fig5_df = pd.read_csv("C:\\Users\\samth\\OneDrive - Northwestern University\\WS_DL\\Lab Data\\Price\\code\\kawin\\examples\\Olaye2020\\figureDataExtraction\\Olaye2020_fig5_PresentModel_curve.csv")
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
    fig5_df = pd.read_csv("C:\\Users\\samth\\OneDrive - Northwestern University\\WS_DL\\Lab Data\\Price\\code\\kawin\\examples\\Olaye2020\\figureDataExtraction\\Olaye2020_fig5_PresentModel_curve.csv")
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

    model_kwargs = {"record_pq_data": record_pq_data, "preallocate_recordings": preallocate_recordings}

    model = model_class(
        mesh,
        list(elements),
        [phase_a_name, phase_b_name],
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
        **model_kwargs,
    )
    print(f"Estimated total number of time steps: {int((np.log(t_end_s) - np.log(model.semiLogT0)) / model.semiLog_dt)}")
    print(f"Estimated total number of time steps: {len(np.arange(np.log(model.semiLogT0), np.log(t_end_s), model.semiLog_dt))}")
    model.solve(float(t_end_s), iterator=explicitEulerIterator, verbose=True, vIt=100, minDtFrac=1e-17)

    t_s = np.array(model.interfaceData._time[: model.interfaceData.N + 1], dtype=np.float64)
    s_m = np.array(model.interfaceData._y[: model.interfaceData.N + 1], dtype=np.float64)
    liquid_half_width_um = s_m * 1e6
    # t_h = t_s / 3600.0
    return t_s, liquid_half_width_um, model


def build_parser():
    """Builds the optional CLI parser retained for backward compatibility."""
    parser = argparse.ArgumentParser(description="Replicate Olaye & Ojo 2020 Figure 5 using the Olaye FD model.")
    parser.add_argument("--figures", nargs="+", default=OLAYE_NOTEBOOK_CONFIG["figures"], help="Figures to run: fig5, fig6, or all.")
    parser.add_argument("--n-phase-a-nodes", type=int, default=OLAYE_NOTEBOOK_CONFIG["n_phase_a_nodes"])
    parser.add_argument("--n-phase-b-nodes", type=int, default=OLAYE_NOTEBOOK_CONFIG["n_phase_b_nodes"])
    parser.add_argument("--semiLog-dt", type=float, default=OLAYE_NOTEBOOK_CONFIG["semiLog_dt"])
    parser.add_argument("--model-variant", type=str, choices=["current", "rework"], default=OLAYE_NOTEBOOK_CONFIG["model_variant"])
    parser.add_argument("--out", type=str, default=None, help="Optional figure output path.")
    parser.add_argument("--save-run-path", type=str, default=None, help="Optional saved-run ``.npz`` output path.")
    parser.add_argument("--no-save-run", action="store_true", help="Disable saving the notebook-friendly run artifact.")
    parser.add_argument("--hide-plot", action="store_true", help="Do not show the matplotlib window.")
    parser.add_argument("--no-record-pq-data", action="store_true", help="Disable Olaye pData/qData history recording.")
    parser.add_argument("--preallocate-recordings", action="store_true", help="Preallocate Olaye recording histories before solving.")
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
        record_pq_data=args.get("record_pq_data", True),
        preallocate_recordings=args.get("preallocate_recordings", False),
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

    alt_csv = SCRIPT_DIR / "figureDataExtraction\\Olaye2020_fig5_alt_PresentModelRough2_curve.csv"
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
    ax.set_xlim(0.00001, 1e5) #ax.set_xlim(0.00001, float(np.max(t_s_plot)))
    ax.set_ylim(0, 24) #ax.set_ylim(12.5, 24)
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

    initial_conc = conc_arr[0]
    print(f"Initial Conc:   {initial_conc}")
    print(f"Idealized Conc: {idealized_conc}")
    print(f"Initial vs Idealized Conc Frac Diff: {(initial_conc-idealized_conc)/idealized_conc}")
    conc_diffFromInitial_arr = conc_arr - initial_conc
    conc_diffFromIdealized_arr = conc_arr - idealized_conc
    print(f"Diff from Initial: {float(conc_diffFromInitial_arr.min()/initial_conc), float(conc_diffFromInitial_arr.max()/initial_conc)}")
    print(f"Diff from Idealized: {float(conc_diffFromIdealized_arr.min()/idealized_conc), float(conc_diffFromIdealized_arr.max()/idealized_conc)}")

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


def _fig6_style(layer, d_alpha_cm2_s):
    """Returns a stable Matplotlib style for one Figure-6 case."""
    color_lookup = {
        ("thin", 1.4e-8): "tab:cyan",
        ("thin", 2.5e-8): "tab:pink",
        ("thick", 1.4e-8): "tab:blue",
        ("thick", 2.5e-8): "black",
    }
    linestyle_lookup = {
        "thin": "--",
        "thick": "-.",
    }
    matched_d = _selected_fig6_d_alpha_cm2_s({"fig6_d_alpha_cm2_s": (d_alpha_cm2_s,)})[0]
    return {
        "color": color_lookup.get((layer, matched_d), None),
        "linestyle": linestyle_lookup.get(layer, "-"),
        "linewidth": 2.0,
    }


def _save_fig6_run_result(path, case_results):
    """Saves selected Figure-6 case arrays to one compressed ``.npz`` file."""
    payload = {"case_count": np.array([len(case_results)], dtype=np.int64)}
    for index, item in enumerate(case_results):
        prefix = f"case_{index}"
        payload[f"{prefix}_time_s"] = np.asarray(item["time_s"], dtype=np.float64)
        payload[f"{prefix}_interface_displacement_um"] = np.asarray(item["interface_displacement_um"], dtype=np.float64)
        payload[f"{prefix}_label"] = np.array(item["params"]["label"])
        payload[f"{prefix}_params_json"] = np.array(json.dumps(_jsonable(item["params"]), sort_keys=True))
    return save_olaye_run_result(path, payload)


def _plot_fig6_concentration_info(ax, case_results):
    """
    Adds a Figure-6 concentration-history twin axis.

    Each concentration curve reuses the interface-curve color for the same
    case, with lighter reference lines for the initial and idealized average
    concentrations.
    """
    ax_conc = ax.twinx()
    ax_conc.set_ylabel("conc", color="green")
    plotted_any = False
    for item in case_results:
        params = item["params"]
        style = _fig6_style(params["layer"], params["d_alpha_cm2_s"])
        color = style.get("color")
        model = item["model"]
        conc_time_arr = np.asarray(model.concData._time, dtype=np.float64)
        conc_arr = np.asarray(model.concData._y, dtype=np.float64)
        mask = np.isfinite(conc_time_arr) & np.isfinite(conc_arr)
        if np.count_nonzero(mask) < 2:
            continue
        conc_time_plot = conc_time_arr[mask]
        conc_plot = conc_arr[mask]
        ax_conc.plot(
            conc_time_plot,
            conc_plot,
            lw=1.0,
            color=color,
            linestyle=":",
            label=f"{params['label']} conc",
        )
        ax_conc.plot(
            [conc_time_plot[0], conc_time_plot[-1]],
            [conc_plot[0], conc_plot[0]],
            lw=0.8,
            color=color,
            linestyle="dashdot",
            alpha=0.45,
            label=f"{params['label']} initial conc",
        )
        idealized_conc = compute_idealized_conc(params)
        ax_conc.plot(
            [conc_time_plot[0], conc_time_plot[-1]],
            [idealized_conc, idealized_conc],
            lw=0.8,
            color=color,
            linestyle="dashed",
            alpha=0.45,
            label=f"{params['label']} idealized conc",
        )
        plotted_any = True
        
        print("\n")
        print(f"Conc Info for {params['label']}")
        initial_conc = conc_arr[0]
        print(f"Initial Conc:   {initial_conc}")
        print(f"Idealized Conc: {idealized_conc}")
        print(f"Initial vs Idealized Conc Frac Diff: {(initial_conc-idealized_conc)/idealized_conc}")
        conc_diffFromInitial_arr = conc_arr - initial_conc
        conc_diffFromIdealized_arr = conc_arr - idealized_conc
        print(f"Diff from Initial: {float(conc_diffFromInitial_arr.min()/initial_conc), float(conc_diffFromInitial_arr.max()/initial_conc)}")
        print(f"Diff from Idealized: {float(conc_diffFromIdealized_arr.min()/idealized_conc), float(conc_diffFromIdealized_arr.max()/idealized_conc)}")


    if plotted_any:
        ax_conc.legend(loc="center right", fontsize=7)
    return ax_conc


def _fig6_extracted_data_specs_for_cases(case_results):
    """Returns digitized Figure-6 overlay specs matching completed case results."""
    selected_cases = set()
    for item in case_results:
        params = item["params"]
        d_alpha = _selected_fig6_d_alpha_cm2_s({"fig6_d_alpha_cm2_s": (params["d_alpha_cm2_s"],)})[0]
        selected_cases.add((params["layer"], d_alpha))

    specs = []
    for key, spec in FIG6_EXTRACTED_DATA_FILES.items():
        if key not in selected_cases:
            continue
        specs.append(
            {
                **spec,
                "layer": key[0],
                "d_alpha_cm2_s": key[1],
                "path": SCRIPT_DIR / "figureDataExtraction" / spec["filename"],
            }
        )
    return specs


def _fig6_analytical_constants(params):
    """
    Returns the semi-infinite analytical constants for one Figure-6 case.

    Diffusivities are converted to ``um^2/s`` before solving for beta so the
    analytical displacement can be plotted directly as
    ``s(t) - s0 = 2*beta*sqrt(t)`` in micrometers.
    """
    d_scale_to_um2_s = float(params["D_scale"]) * 1e12
    d_a_um2_s = float(params["D_liquid_base"]) * d_scale_to_um2_s
    d_b_um2_s = float(params["D_solid_base"]) * d_scale_to_um2_s
    beta_um_sqrt_s = solve_beta(
        c_a0=float(params["c_liquid0_pct"]) / 100.0,
        c_b0=float(params["c_solid0_pct"]) / 100.0,
        c_a_eq=float(params["c_liquid_int_pct"]) / 100.0,
        c_b_eq=float(params["c_solid_int_pct"]) / 100.0,
        d_a=d_a_um2_s,
        d_b=d_b_um2_s,
        left=-100.0,
        right=100.0,
    )
    return {
        "d_a_um2_s": d_a_um2_s,
        "d_b_um2_s": d_b_um2_s,
        "s0_um": float(params["s0_um"]),
        "R_um": float(params["R_um"]),
        "beta_um_sqrt_s": beta_um_sqrt_s,
    }


def _fig6_semi_infinite_time_max_s(constants, erfc_argument_min):
    """
    Estimates a Figure-6 semi-infinite analytical comparison cutoff.

    This uses the smaller of the two finite-domain times at which the far
    boundary reaches ``eta = L/(2*sqrt(D*t)) == erfc_argument_min``.
    """
    erfc_argument_min = float(erfc_argument_min)
    if erfc_argument_min <= 0:
        raise ValueError("fig6_sqrt_time_semi_infinite_erfc_argument_min must be positive.")
    phase_a_width_um = constants["s0_um"]
    phase_b_width_um = constants["R_um"] - constants["s0_um"]
    if phase_a_width_um <= 0 or phase_b_width_um <= 0:
        raise ValueError("Figure 6 analytical comparison requires 0 < s0_um < R_um.")
    phase_a_time_s = (phase_a_width_um / (2.0 * erfc_argument_min)) ** 2 / constants["d_a_um2_s"]
    phase_b_time_s = (phase_b_width_um / (2.0 * erfc_argument_min)) ** 2 / constants["d_b_um2_s"]
    return min(phase_a_time_s, phase_b_time_s)


def _fig6_case_sqrt_time_max_s(cfg, constants):
    """Resolves the optional time limit for one Figure-6 sqrt-time overlay."""
    time_max_s = cfg.get("fig6_sqrt_time_max_s")
    if time_max_s is None:
        return None
    if isinstance(time_max_s, str):
        if time_max_s.lower() not in {"auto", "semi_infinite"}:
            raise ValueError('fig6_sqrt_time_max_s must be numeric, None, "auto", or "semi_infinite".')
        return _fig6_semi_infinite_time_max_s(
            constants,
            cfg.get("fig6_sqrt_time_semi_infinite_erfc_argument_min", 5.0),
        )
    time_max_s = float(time_max_s)
    if time_max_s <= 0:
        raise ValueError("fig6_sqrt_time_max_s must be positive when numeric.")
    return time_max_s


def _limit_plot_points(x, y, max_points):
    """Returns an evenly thinned plotting view while preserving endpoints."""
    if max_points is None or len(x) <= int(max_points):
        return x, y
    if int(max_points) < 2:
        raise ValueError("fig6_sqrt_time_max_points_per_case must be at least 2 or None.")
    indices = np.linspace(0, len(x) - 1, int(max_points), dtype=np.int64)
    return x[indices], y[indices]


def plot_olaye_fig6_sqrt_time_analytical(case_results, config=None, ax=None):
    """
    Plots Figure-6 results against ``sqrt(t)`` with analytical beta overlays.

    The analytical line is the semi-infinite planar moving-boundary solution
    ``s(t) - s0 = 2*beta*sqrt(t)``. It is intended as a separate diagnostic
    view of the same completed Figure-6 cases, not as a replacement for the
    log-time replication plot.
    """
    cfg = OLAYE_NOTEBOOK_CONFIG if config is None else {**OLAYE_NOTEBOOK_CONFIG, **dict(config)}
    if not case_results:
        raise ValueError("At least one Figure 6 case result is required for the sqrt-time analytical plot.")
    out_setting = cfg.get("fig6_sqrt_time_out")
    out_path = None if out_setting is False else (
        pathlib.Path(out_setting).resolve()
        if out_setting is not None
        else SCRIPT_DIR / "olaye2020_fig6_sqrt_time_analytical.png"
    )

    created_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)
    else:
        fig = ax.figure

    plotted_any = False
    for item in case_results:
        params = item["params"]
        constants = _fig6_analytical_constants(params)
        time_max_s = _fig6_case_sqrt_time_max_s(cfg, constants)
        time_s = np.asarray(item["time_s"], dtype=np.float64)
        displacement_um = np.asarray(item["interface_displacement_um"], dtype=np.float64)
        mask = np.isfinite(time_s) & np.isfinite(displacement_um) & (time_s >= 0)
        if time_max_s is not None:
            mask &= time_s <= time_max_s
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"{params['label']} has fewer than two finite points in the sqrt-time comparison window.")

        sqrt_time, displacement_plot_um = _limit_plot_points(
            np.sqrt(time_s[mask]),
            displacement_um[mask],
            cfg.get("fig6_sqrt_time_max_points_per_case"),
        )
        style = _fig6_style(params["layer"], params["d_alpha_cm2_s"])
        ax.plot(
            sqrt_time,
            displacement_plot_um,
            label=params["label"],
            **style,
        )

        analytical_sqrt_time = np.linspace(0.0, float(np.max(sqrt_time)), 250)
        ax.plot(
            analytical_sqrt_time,
            2.0 * constants["beta_um_sqrt_s"] * analytical_sqrt_time,
            color=style.get("color"),
            linestyle=":",
            linewidth=1.4,
            label=f"{params['label']} analytical, beta={constants['beta_um_sqrt_s']:.6g} um/sqrt(s)",
        )
        plotted_any = True

    plot_extracted = cfg.get("fig6_sqrt_time_plot_extracted_data", cfg.get("fig6_plot_extracted_data", True))
    if plot_extracted:
        non_extracted_xlim = ax.get_xlim()
        x_min, x_max = sorted(non_extracted_xlim)
        for extracted_spec in _fig6_extracted_data_specs_for_cases(case_results):
            if not extracted_spec["path"].exists():
                continue
            exp_t_s, exp_disp_um = _load_no_header_xy_csv(extracted_spec["path"])
            exp_sqrt_time = np.full_like(exp_t_s, np.nan, dtype=np.float64)
            nonnegative_time = np.isfinite(exp_t_s) & (exp_t_s >= 0)
            exp_sqrt_time[nonnegative_time] = np.sqrt(exp_t_s[nonnegative_time])
            mask = (
                np.isfinite(exp_sqrt_time)
                & np.isfinite(exp_disp_um)
                & (x_min <= exp_sqrt_time)
                & (exp_sqrt_time <= x_max)
            )
            if np.count_nonzero(mask) == 0:
                continue
            ax.scatter(
                exp_sqrt_time[mask],
                exp_disp_um[mask],
                s=20,
                color=extracted_spec["color"],
                marker=extracted_spec["marker"],
                facecolor='none',
                label=extracted_spec["label"],
                zorder=3,
            )
        ax.set_xlim(non_extracted_xlim)

    ax.set_xlabel("sqrt(time) (sqrt(s))")
    ax.set_ylabel("Interface displacement (um)")
    ax.set_title(
        f"Figure 6 Olaye sqrt-time analytical comparison, "
        f"nA:{cfg['fig6_n_phase_a_nodes']}, nB:{cfg['fig6_n_phase_b_nodes']}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    if plotted_any:
        ax.legend(fontsize=7)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved Figure 6 sqrt-time analytical figure: {out_path}")

    if created_figure:
        if cfg.get("show", True):
            plt.show()
        else:
            plt.close(fig)

    return {
        "figure": fig,
        "axes": ax,
        "cases": case_results,
        "params": cfg,
    }


def plot_olaye_fig6_notebook(config=None, ax=None):
    """
    Runs the corrected Figure-6 brass cases and plots interface displacement.

    Figure 6 uses phase A as beta and phase B as alpha. The plotted quantity is
    ``s(t) - s0`` in micrometers, matching the paper's interface displacement
    convention for the selected beta-layer half-width.
    """
    cfg = OLAYE_NOTEBOOK_CONFIG if config is None else {**OLAYE_NOTEBOOK_CONFIG, **dict(config)}
    out_path = (
        pathlib.Path(cfg["fig6_out"]).resolve()
        if cfg.get("fig6_out") is not None
        else SCRIPT_DIR / "olaye2020_fig6_replication.png"
    )

    created_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)
    else:
        fig = ax.figure

    case_results = []
    for layer in _selected_fig6_layers(cfg):
        for d_alpha_cm2_s in _selected_fig6_d_alpha_cm2_s(cfg):
            params = build_fig6_case_params(layer, d_alpha_cm2_s, cfg)
            t_s, phase_a_width_um, model = run_case(
                R_um=params["R_um"],
                s0_um=params["s0_um"],
                c_liquid0_pct=params["c_liquid0_pct"],
                c_solid0_pct=params["c_solid0_pct"],
                c_liquid_int_pct=params["c_liquid_int_pct"],
                c_solid_int_pct=params["c_solid_int_pct"],
                D_liquid_base=params["D_liquid_base"],
                D_solid_base=params["D_solid_base"],
                D_scale=params["D_scale"],
                n_nodes=params["n_nodes"],
                n_phase_a_nodes=params["n_phase_a_nodes"],
                n_phase_b_nodes=params["n_phase_b_nodes"],
                t_end_s=params["t_end_s"],
                dt_mode=params["dt_mode"],
                semiLog_dt=params["semiLog_dt"],
                semiLogT0=params["semiLogT0"],
                model_variant=params["model_variant"],
                record_pq_data=params["record_pq_data"],
                preallocate_recordings=params["preallocate_recordings"],
                response_name=params["response_name"],
                elements=params["elements"],
                phase_a_name=params["phase_a_name"],
                phase_b_name=params["phase_b_name"],
                print_beta=False,
            )

            displacement_um = phase_a_width_um - params["s0_um"]
            mask = np.isfinite(t_s) & np.isfinite(displacement_um)
            t_s_plot = t_s[mask]
            displacement_um_plot = displacement_um[mask]
            if t_s_plot.size < 2:
                raise ValueError(f"Figure 6 case {params['label']} did not produce enough finite points to plot.")

            ax.plot(t_s_plot, displacement_um_plot, label=params["label"], **_fig6_style(layer, d_alpha_cm2_s))
            payload = build_olaye_run_payload(
                time_s=t_s_plot,
                interface_displacement_um=displacement_um_plot,
                params=params,
                model_variant=params["model_variant"],
                model_family="olaye_fig6",
                label=params["label"],
            )
            case_results.append(
                {
                    "model": model,
                    "time_s": t_s_plot,
                    "phase_a_width_um": phase_a_width_um[mask],
                    "interface_displacement_um": displacement_um_plot,
                    "payload": payload,
                    "params": params,
                }
            )

    ax_conc = None
    if cfg.get("fig6_plot_concentration_info", True):
        ax_conc = _plot_fig6_concentration_info(ax, case_results)

    if cfg.get("fig6_plot_extracted_data", True):
        for extracted_spec in _fig6_extracted_data_specs_for_cases(case_results):
            if not extracted_spec["path"].exists():
                continue
            exp_t_s, exp_disp_um = _load_no_header_xy_csv(extracted_spec["path"])
            ax.scatter(
                exp_t_s,
                exp_disp_um,
                s=20,
                color=extracted_spec["color"],
                marker=extracted_spec["marker"],
                facecolor='none',
                label=extracted_spec["label"],
                zorder=3,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Interface displacement (um)")
    ax.set_title(
        f"Figure 6 Olaye Dufort-Frankel, "
        f"nA:{cfg['fig6_n_phase_a_nodes']}, nB:{cfg['fig6_n_phase_b_nodes']}, "
        f"semiLog_dt:{float(cfg['fig6_semiLog_dt']):.10f}",
        fontsize=10,
    )
    ax.set_xscale("log")
    ax.set_xlim(10, float(cfg["fig6_t_end_s"]))
    ax.set_ylim(-200, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    save_path = None
    if cfg.get("fig6_save_run_path") is not None:
        save_path = _save_fig6_run_result(cfg["fig6_save_run_path"], case_results)
        print(f"Saved Figure 6 run: {save_path}")

    sqrt_time_analytical = None
    if cfg.get("fig6_plot_sqrt_time_analytical", False):
        sqrt_time_analytical = plot_olaye_fig6_sqrt_time_analytical(case_results, cfg)

    if created_figure:
        if cfg["show"]:
            plt.show()
        else:
            plt.close(fig)

    return {
        "figure": fig,
        "axes": ax,
        "concentration_axes": ax_conc,
        "cases": case_results,
        "save_path": save_path,
        "sqrt_time_analytical": sqrt_time_analytical,
        "params": cfg,
    }


def run_and_plot(config=None):
    """Runs and plots each selected Olaye figure, returning results by figure."""
    cfg = OLAYE_NOTEBOOK_CONFIG if config is None else {**OLAYE_NOTEBOOK_CONFIG, **dict(config)}
    results = {}
    for figure in _selected_figures(cfg):
        if figure == "fig5":
            results[figure] = plot_olaye_fig5_notebook(cfg)
        elif figure == "fig6":
            results[figure] = plot_olaye_fig6_notebook(cfg)
        else:
            raise ValueError(f"Unsupported figure selection: {figure!r}")
    return results


def main(argv=None):
    """Runs the Figure-5 example using CLI overrides on top of notebook defaults."""
    args = build_parser().parse_args(argv)
    config = {
        "figures": tuple(args.figures),
        "n_phase_a_nodes": args.n_phase_a_nodes,
        "n_phase_b_nodes": args.n_phase_b_nodes,
        "semiLog_dt": args.semiLog_dt,
        "model_variant": args.model_variant,
        "out": args.out,
        "save_run_path": args.save_run_path or OLAYE_NOTEBOOK_CONFIG["save_run_path"],
        "save_run": not args.no_save_run,
        "show": not args.hide_plot,
        "record_pq_data": not args.no_record_pq_data,
        "preallocate_recordings": args.preallocate_recordings,
    }
    return run_and_plot(config)


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        results=run_and_plot()
    else:
        results=run_and_plot()
        # main()

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

fig5_df = pd.read_csv("C:\\Users\\samth\\OneDrive - Northwestern University\\WS_DL\\Lab Data\\Price\\code\\kawin\\examples\\Olaye2020\\figureDataExtraction\\Olaye2020_fig5_PresentModel_curve.csv")
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

fig5_alt_df = pd.read_csv(r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\Olaye2020\figureDataExtraction\Olaye2020_fig5_alt_PresentModelRough_curve.csv")
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

'''
Useful code

model = results['fig6']['cases'][0]['model']
finalConc = model.concData._y[0]
concDiffBasedOnPhaseFrac = lambda x, C_A, C_B: ((x*C_B) + (1-x)*C_A) - finalConc
finalPhaseFrac = lambda C_avg, C_A, C_B: (C_avg-C_A)/(C_B-C_A)
print(finalPhaseFrac(finalConc, 0.325, 0.369))
sol = optimize.root_scalar(
            concDiffBasedOnPhaseFrac,
            bracket=[0, 1],
            args=(0.325, 0.369),
            method='brentq',
            maxiter=100,
            rtol=1e-14,
            xtol=1e-14,
        )
sol

'''

# %%
