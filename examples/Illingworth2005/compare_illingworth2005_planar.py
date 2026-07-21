#%%
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np

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

def _find_repo_root(start):
    """Finds the repo root when ``__file__`` is unreliable in notebooks."""
    start = pathlib.Path(start).resolve()
    candidates = [start, *start.parents]
    candidates.extend([pathlib.Path.cwd().resolve(), *pathlib.Path.cwd().resolve().parents])
    for candidate in candidates:
        if (candidate / "kawin").is_dir() and (candidate / "examples").is_dir():
            return candidate
    raise RuntimeError("Could not locate the kawin repository root.")


def _resolve_this_file():
    """
    Returns this script's path without trusting stale Interactive Window globals.

    In notebook/interactive execution, VS Code can leave ``__file__`` bound to
    another script that previously populated the shared kernel namespace. We
    therefore accept ``__file__`` only when it already points to this file and
    otherwise fall back to locating the script from the current working
    directory.
    """
    expected_parts = ("examples", "Illingworth2005", "compare_illingworth2005_planar.py")
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


def _find_illingworth_source_dir(repo_root, script_dir):
    """Finds the authors' MAP C++ source directory in CLI and notebook runs."""
    candidates = [
        pathlib.Path(script_dir) / "illingworth_MAP_code",
        pathlib.Path(script_dir) / "Illingworth2005" / "illingworth_MAP_code",
        pathlib.Path(repo_root) / "examples" / "Illingworth2005" / "illingworth_MAP_code",
        pathlib.Path.cwd() / "examples" / "Illingworth2005" / "illingworth_MAP_code",
    ]
    for candidate in candidates:
        if (candidate / "ConservativeIFF.cpp").is_file():
            return candidate.resolve()
    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find illingworth_MAP_code. Searched:\n{searched}")


THIS_FILE = _resolve_this_file()
SCRIPT_DIR = THIS_FILE.parent
REPO_ROOT = _find_repo_root(THIS_FILE.parent)
AUTHOR_CPP_SOURCE_DIR = _find_illingworth_source_dir(REPO_ROOT, SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kawin.diffusion import MovingBoundaryIllingworthFD1DModel, TemperatureParameters
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.solver import explicitEulerIterator


AUTHOR_DEFAULT_PARAMS = {
    "s0": 1.0,
    "R": 5.0,
    "n_alpha": 13,
    "d_alpha": 1.0e-7,
    "initial_alpha": 0.8,
    "interface_alpha": 0.6,
    "n_beta": 3001,
    "d_beta": 1.0e-5,
    "initial_beta": 0.4,
    "interface_beta": 0.0,
    "time_step": 0.1,
    "n_time_steps": 10,
    "tolerance": 1.0e-8,
}


SCRIPT_CONFIG = {
    # Options: "cpp_comparison" or "fig3_present_work".
    "run_mode": "cpp_comparison",
    # Set this to a full path if PATH lookup fails or you want a specific compiler.
    "compiler": None,
    # Set this to keep the compiled executable and results.txt for inspection.
    # Leave as None to use a temporary directory.
    "build_dir": None,
    # If True, print per-time output in addition to the max-difference summary.
    "print_table": True,
    # Optional overrides for FIG3_PRESENT_WORK_PARAMS when run_mode is
    # "fig3_present_work".
    "fig3_overrides": {},
}

FIG3_NOTEBOOK_CONFIG = {
    "show": True,
    "out": None,
    "save_run": True,
    "save_run_path": SCRIPT_DIR / "illingworth2005_fig3_saved_run.npz",
    "label": None,
    "show_grid_plot": True,
    "grid_plot_out": None,
    # "dt_mode": "fixed",
    # "semiLog_dt": None,
    # "semiLogT0": None,
    # "checkAgainstAuthorsCPP": False,
    "cpp_compiler": None,
    "cpp_build_dir": None,
    "cpp_print_table": False,
}


FIG3_PRESENT_WORK_PARAMS = {
    # Parameters stated in the paragraph introducing Fig. 3.
    "R_um": 3012.5,
    "s0_um": 12.5,
    "c_liquid0_atpct": 19.0,
    "c_solid0_atpct": 0.0,
    "c_liquid_int_atpct": 10.223,
    "c_solid_int_atpct": 0.166,
    "D_liquid_um2_s": 500.0,
    "D_solid_um2_s": 18.0,
    # The paper notes a similar initial step size of 1 um for the comparison.
    # "spatial_step_um": 0.25,
    "n_alpha": 51,
    "n_beta": 228, #5001,
    # Options: "constant" or "geometric". For geometric grids,
    # geometric_ratio is the interval growth factor moving away from the
    # interface. Values > 1 cluster nodes near the interface; 0 < values < 1
    # cluster nodes toward the fixed far boundaries.
    "grid_type": "constant", #"constant",
    "geometric_ratio": 1.03,
    "min_transformed_interval": 1e-12,
    # The text mentions a 0.01 s time step for the comparison setup. That is
    # very expensive in pure Python out to 1e5 s, so the default here is a
    # runtime-friendly value. Set this to 0.01 for the literal paper timestep.
    "time_step_s": 0.005,
    "paper_time_step_s": 0.01,
    # Options: "fixed" or "semi_log". The authors' generated C++ comparison
    # is available only for fixed timesteps.
    "dt_mode": "semi_log",
    # Semi-log mode uses natural-log spacing. For example, semiLogT0=1e-4 and
    # semiLog_dt=0.1 generate targets exp(log(1e-4) + n*0.1), plus t_end_s.
    "semiLog_dt": 0.00025,
    "semiLogT0": 1e-6,
    "t_end_s": 8.5e4, #7e4,
    "record": 1,
    "record_pq_data": False,
    "preallocate_recordings": True,
    "plot_conc": True,
    "timeProfiling": False,
    "checkAgainstAuthorsCPP": False,
    "cpp_compiler": None,
    "cpp_build_dir": None,
    "cpp_print_table": False,
    "show": True,
    "out": None,
    "max_iterations":10000,
    "overlay_csvs": [
        {
            "path": REPO_ROOT / "examples" / "Olaye2020" / "Olaye2020_fig5_alt_PresentModelRough_curve.csv",
            "label": 'Olaye Fig. 5 "Present Model" rough digitization',
            "color": "tab:gray",
            "marker": "o",
        },
        {
            "path": REPO_ROOT / "examples" / "Olaye2020" / "Olaye2020_fig5_alt_PresentModelRough2_curve.csv",
            "label": 'Olaye Fig. 5 "Present Model" rough2 digitization',
            "color": "tab:pink",
            "marker": "o",
        },
    ],
}


class ConstantBinaryThermodynamics:
    """Minimal constant-diffusivity thermodynamics interface for comparison runs."""

    def __init__(self, phases, diffusivities):
        self.phases = list(phases)
        self.diffusivities = dict(diffusivities)

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        values = np.atleast_1d(T).astype(np.float64)
        return np.squeeze(np.ones(values.shape, dtype=np.float64) * self.diffusivities[phase])


def _load_time_half_width_csv(path):
    """
    Loads a digitized curve with ``time_s`` and ``half_width_um`` columns.

    The Olaye Figure 5 digitizations use the same units as the Illingworth
    Figure 3 plot here, so no conversion is applied.
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if "time_s" not in data.dtype.names or "half_width_um" not in data.dtype.names:
        raise ValueError(f"{path} must contain columns: time_s, half_width_um")
    return np.asarray(data["time_s"], dtype=np.float64), np.asarray(data["half_width_um"], dtype=np.float64)


def build_python_default_model(record=True):
    """Builds the Python Illingworth model using the MAP C++ default case."""
    p = AUTHOR_DEFAULT_PARAMS
    profile = ProfileBuilder([(StepProfile1D(p["s0"], p["initial_alpha"], p["initial_beta"]), "CR")])
    mesh = CartesianFD1D(["CR"], [0.0, p["R"]], 501)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=["ALPHA", "BETA"],
        diffusivities={"ALPHA": p["d_alpha"], "BETA": p["d_beta"]},
    )
    return MovingBoundaryIllingworthFD1DModel(
        mesh,
        ["NI", "CR"],
        ["ALPHA", "BETA"],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=p["s0"],
        interface_compositions=(p["interface_alpha"], p["interface_beta"]),
        time_step=p["time_step"],
        phase_a_nodes=p["n_alpha"],
        phase_b_nodes=p["n_beta"],
        tolerance=p["tolerance"],
        record=record,
    )


def run_python_default():
    """Runs the Python default comparison case and returns ``(time, interface)``."""
    p = AUTHOR_DEFAULT_PARAMS
    model = build_python_default_model(record=True)
    model.solve(p["n_time_steps"] * p["time_step"], iterator=explicitEulerIterator, minDtFrac=1e-15, verbose=True, vIt=100)
    n = model.interfaceData.N + 1
    return model.interfaceData._time[:n].copy(), model.interfaceData._y[:n].copy()


def theoretical_fig3_max_liquid_half_width_um(params=None):
    """
    Returns the paper's Figure-3 theoretical maximum liquid half-width.

    Illingworth and Golosnoy state this value as approximately 23.2 um. It
    follows from placing the initial liquid solute inventory in liquid at the
    liquid-side interface composition: ``s_max = s0*c_liquid0/c_liquid_int``.
    """
    p = FIG3_PRESENT_WORK_PARAMS if params is None else {**FIG3_PRESENT_WORK_PARAMS, **dict(params)}
    return p["s0_um"] * p["c_liquid0_atpct"] / p["c_liquid_int_atpct"]


def compute_fig3_idealized_conc(params=None):
    """Returns the initial domain-average composition for the Figure-3 setup."""
    p = FIG3_PRESENT_WORK_PARAMS if params is None else {**FIG3_PRESENT_WORK_PARAMS, **dict(params)}
    return (
        p["s0_um"] * (p["c_liquid0_atpct"] / 100.0)
        + (p["R_um"] - p["s0_um"]) * (p["c_solid0_atpct"] / 100.0)
    ) / p["R_um"]


def _fig3_phase_node_counts(params):
    """Returns the transformed phase-node counts used by the Figure-3 run."""
    if params.get("spatial_step_um") is not None:
        phase_a_nodes = int(round(params["s0_um"] / params["spatial_step_um"])) + 1
        phase_b_nodes = int(round((params["R_um"] - params["s0_um"]) / params["spatial_step_um"])) + 1
        n_mesh = int(round(params["R_um"] / params["spatial_step_um"])) + 1
    else:
        if params.get("n_alpha") is None or params.get("n_beta") is None:
            raise ValueError("Either spatial_step_um or both n_alpha and n_beta must be specified.")
        phase_a_nodes = int(params["n_alpha"])
        phase_b_nodes = int(params["n_beta"])
        n_mesh = phase_a_nodes + phase_b_nodes - 1
    return phase_a_nodes, phase_b_nodes, n_mesh


def _geometric_intervals_away_from_interface(n_nodes, ratio):
    """
    Builds normalized interval widths ordered from interface to far boundary.

    ``ratio`` is the geometric growth factor between neighboring intervals as
    distance from the interface increases. The widths are normalized to sum to
    one in transformed coordinates, so only the relative spacing is controlled.
    """
    n_intervals = int(n_nodes) - 1
    if n_intervals < 2:
        raise ValueError("Geometric transformed grids require at least three nodes per phase.")
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("geometric_ratio must be a positive finite number.")
    if np.isclose(ratio, 1.0, rtol=0.0, atol=1e-14):
        return np.full(n_intervals, 1.0 / n_intervals, dtype=np.float64)
    powers = np.arange(n_intervals, dtype=np.float64)
    log_widths = powers * np.log(ratio)
    widths = np.exp(log_widths - np.max(log_widths))
    return widths / np.sum(widths)


def build_symmetric_fig3_transformed_grids(params=None):
    """
    Builds optional transformed grids for the Illingworth Figure-3 run.

    ``grid_type="constant"`` returns ``None`` grids so the model follows its
    original uniform-grid setup. ``grid_type="geometric"`` constructs
    ratio-controlled interval widths that are mirrored about the interface:
    the last phase-A ``u`` interval and first phase-B ``v`` interval are the
    interface-adjacent intervals. A ratio greater than one clusters both
    phases at the interface; a ratio between zero and one clusters them toward
    the fixed far boundaries.
    """
    p = FIG3_PRESENT_WORK_PARAMS if params is None else {**FIG3_PRESENT_WORK_PARAMS, **dict(params)}
    phase_a_nodes, phase_b_nodes, n_mesh = _fig3_phase_node_counts(p)
    grid_type = str(p.get("grid_type", "constant")).lower()
    if grid_type not in {"constant", "geometric"}:
        raise ValueError("grid_type must be 'constant' or 'geometric'.")

    ratio = float(p.get("geometric_ratio", 1.0))
    min_interval = float(p.get("min_transformed_interval", 1e-12))
    if not np.isfinite(min_interval) or min_interval <= 0:
        raise ValueError("min_transformed_interval must be a positive finite number.")
    metadata = {
        "grid_type": grid_type,
        "geometric_ratio": ratio,
        "min_transformed_interval": min_interval,
        "grid_spacing_convention": (
            "geometric_ratio is the interval growth factor moving away from the interface; "
            "u intervals are mirrored against v intervals about the interface."
        ),
        "n_phase_a_nodes": phase_a_nodes,
        "n_phase_b_nodes": phase_b_nodes,
        "n_mesh": n_mesh,
    }
    if grid_type == "constant":
        metadata.update(
            {
                "phase_a_interface_interval": 1.0 / (phase_a_nodes - 1),
                "phase_b_interface_interval": 1.0 / (phase_b_nodes - 1),
                "phase_a_far_interval": 1.0 / (phase_a_nodes - 1),
                "phase_b_far_interval": 1.0 / (phase_b_nodes - 1),
            }
        )
        return None, None, metadata

    v_intervals = _geometric_intervals_away_from_interface(phase_b_nodes, ratio)
    u_intervals = _geometric_intervals_away_from_interface(phase_a_nodes, ratio)[::-1]
    u_grid = np.concatenate(([0.0], np.cumsum(u_intervals))).astype(np.float64)
    v_grid = np.concatenate(([0.0], np.cumsum(v_intervals))).astype(np.float64)
    u_grid[-1] = 1.0
    v_grid[-1] = 1.0
    u_diff = np.diff(u_grid)
    v_diff = np.diff(v_grid)
    min_actual_interval = float(min(np.min(u_diff), np.min(v_diff)))
    if min_actual_interval < min_interval:
        raise ValueError(
            "The requested geometric grid is too strongly clustered for the selected node count. "
            f"Smallest transformed interval is {min_actual_interval:.3e}, below "
            f"min_transformed_interval={min_interval:.3e}. "
            "Reduce geometric_ratio toward 1, reduce n_alpha/n_beta, or lower "
            "min_transformed_interval only if you intentionally want a very ill-conditioned grid."
        )
    metadata.update(
        {
            "phase_a_interface_interval": float(u_diff[-1]),
            "phase_b_interface_interval": float(v_diff[0]),
            "phase_a_far_interval": float(u_diff[0]),
            "phase_b_far_interval": float(v_diff[-1]),
            "min_actual_transformed_interval": min_actual_interval,
        }
    )
    
    u_mid = (u_grid[1:] + u_grid[:-1]) / 2
    v_mid = (v_grid[1:] + v_grid[:-1]) / 2
    du = np.diff(np.concatenate(([0], u_mid, [1])))
    dv = np.diff(np.concatenate(([0], v_mid, [1])))

    left_int_widthFraction = du[-1]
    right_int_widthFraction = dv[0]

    print(f"left_int_widthFraction: {left_int_widthFraction}")
    print(f"right_int_widthFraction: {right_int_widthFraction}")
    print(f"left_int_widthFraction/right_int_widthFraction: {left_int_widthFraction/right_int_widthFraction}")
    print("\n")
    def widthsFromIntWidthFrac(left_int_widthFraction, right_int_widthFraction):

        left_int_width = left_int_widthFraction * params['s0_um']
        left_bulk_width = (1-left_int_widthFraction) * params['s0_um']
        
        right_int_width = right_int_widthFraction * (params['R_um'] - params['s0_um'])
        right_bulk_width = (1-right_int_widthFraction) * (params['R_um'] - params['s0_um'])

        return left_int_width, left_bulk_width, right_int_width, right_bulk_width
    
    left_int_width, left_bulk_width, right_int_width, right_bulk_width = widthsFromIntWidthFrac(left_int_widthFraction, right_int_widthFraction)
    
    left_bulk = params['c_liquid0_atpct'] / 100.0
    right_bulk = params['c_solid0_atpct'] / 100.0
    left_int = params['c_liquid_int_atpct'] / 100.0
    right_int = params['c_solid_int_atpct'] / 100.0


    left_mass = left_int_width*left_int + left_bulk_width*left_bulk
    right_mass = right_int_width*right_int + right_bulk_width*right_bulk
    total_mass = left_mass + right_mass
    average_conc = total_mass / params['R_um']

    left_mass_idealized = left_bulk*params['s0_um']
    right_mass_idealized = right_bulk*(params['R_um']-params['s0_um'])
    idealized_conc = (left_mass_idealized + right_mass_idealized) / params['R_um']
    idealized_conc = compute_fig3_idealized_conc(params) # (left_bulk*params['s0_um'] + right_bulk*(params['R_um']-params['s0_um'])) / params['R_um']

    # left_int_widthFraction_cnst, right_int_widthFraction_cnst = 1/len(np.diff(u_grid)), 1/len(np.diff(v_grid))
    left_int_widthFraction_cnst, right_int_widthFraction_cnst = (1/len(np.diff(u_grid)))/2, (1/len(np.diff(v_grid)))/2
    print(f"left_int_widthFraction_cnst: {left_int_widthFraction_cnst}")
    print(f"right_int_widthFraction_cnst: {right_int_widthFraction_cnst}")
    print(f"left_int_widthFraction_cnst/right_int_widthFraction_cnst: {left_int_widthFraction_cnst/right_int_widthFraction_cnst}")
    print("\n")
    left_int_width_cnst, left_bulk_width_cnst, right_int_width_cnst, right_bulk_width_cnst = widthsFromIntWidthFrac(left_int_widthFraction_cnst, right_int_widthFraction_cnst)
    left_mass_cnst = left_int_width_cnst*left_int + left_bulk_width_cnst*left_bulk
    right_mass_cnst = right_int_width_cnst*right_int + right_bulk_width_cnst*right_bulk
    total_mass_cnst = left_mass_cnst + right_mass_cnst
    average_conc_cnst = total_mass_cnst / params['R_um']

    print(f"idealized_conc:    {idealized_conc}   ({left_mass_idealized} + {right_mass_idealized})")
    print(f"average_conc_geo:  {average_conc}   ({left_mass} + {right_mass})")
    print(f"average_conc_cnst: {average_conc_cnst}   ({left_mass_cnst} + {right_mass_cnst})")
    return u_grid, v_grid, metadata


def plot_fig3_transformed_grids(params=None, ax=None):
    """
    Plots the transformed phase grids used by a non-constant Figure-3 run.

    Phase A is shown as ``u - 1`` and phase B as ``v`` so the interface is at
    zero and mirrored spacing is visually checkable before the calculation.
    """
    p = FIG3_PRESENT_WORK_PARAMS if params is None else {**FIG3_PRESENT_WORK_PARAMS, **dict(params)}
    u_grid, v_grid, metadata = build_symmetric_fig3_transformed_grids(p)
    if metadata["grid_type"] == "constant":
        return None, None

    import matplotlib.pyplot as plt

    created_figure = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 2.4), dpi=140)

    phase_a_x = u_grid - 1.0
    phase_b_x = v_grid
    ax.plot(phase_a_x, np.zeros_like(phase_a_x), "|", markersize=16, color="tab:blue", label="phase A: u - 1")
    ax.plot(phase_b_x, np.ones_like(phase_b_x), "|", markersize=16, color="tab:orange", label="phase B: v")
    ax.axvline(0.0, color="0.25", linewidth=1.0, linestyle="--", label="interface")
    ax.set_yticks([0, 1], ["phase A", "phase B"])
    ax.set_xlabel("Transformed coordinate relative to interface")
    ax.set_title(
        f"Fig. 3 transformed grids: {metadata['grid_type']}, "
        f"geometric_ratio={metadata['geometric_ratio']:.6g}"
    )
    ax.set_xlim(-1.02, 1.02)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(fontsize=8, loc="upper center", ncol=3)

    out = p.get("grid_plot_out")
    if out:
        ax.figure.savefig(out, bbox_inches="tight")
        print(f"Saved transformed-grid preview: {out}")
    if created_figure and p.get("show_grid_plot", True):
        plt.show()
    elif created_figure:
        plt.close(ax.figure)
    return ax.figure, ax


def _jsonable(value):
    """Converts notebook config values into JSON-serializable objects."""
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_illingworth_run_payload(result, label=None):
    """
    Builds the saved-run payload for Figure-3 present-work comparisons.

    The payload uses seconds and micrometers so it can be overlaid directly
    with the saved Olaye Figure-5 output. The saved parameter JSON also stores
    ``n_phase_a_nodes`` and ``n_phase_b_nodes`` aliases so shared comparison
    utilities can display node counts across model families, along with the
    timestep controls that are active for the selected ``dt_mode``.
    """
    params = result["params"]
    payload_params = {
        key: value
        for key, value in params.items()
        if key not in {"show", "out", "save_run", "save_run_path", "label", "show_grid_plot", "grid_plot_out"}
    }
    if "n_alpha" in params and "n_phase_a_nodes" not in payload_params:
        payload_params["n_phase_a_nodes"] = params["n_alpha"]
    if "n_beta" in params and "n_phase_b_nodes" not in payload_params:
        payload_params["n_phase_b_nodes"] = params["n_beta"]
    dt_mode = str(params.get("dt_mode", "fixed"))
    payload_params["dt_mode"] = dt_mode
    payload_params["timestep_label"] = _format_fig3_timestep_label(params)
    if dt_mode == "fixed":
        if "time_step_s" in params:
            payload_params["active_time_step_s"] = params["time_step_s"]
    elif dt_mode == "semi_log":
        if "semiLogT0" in params:
            payload_params["active_semiLogT0"] = params["semiLogT0"]
        if "semiLog_dt" in params:
            payload_params["active_semiLog_dt"] = params["semiLog_dt"]
    grid_metadata = result.get("grid_metadata")
    if grid_metadata:
        payload_params["grid_metadata"] = grid_metadata
    payload = {
        "time_s": np.asarray(result["time_s"], dtype=np.float64),
        "half_width_um": np.asarray(result["liquid_half_width_um"], dtype=np.float64),
        "label": np.array(label or "Illingworth Figure 3 present work"),
        "source_script": np.array("examples/Illingworth2005/compare_illingworth2005_planar.py"),
        "model_family": np.array("illingworth_fig3_present_work"),
        "theoretical_max_um": np.array([result["theoretical_max_um"]], dtype=np.float64),
        "idealized_mass_integral": np.array([compute_fig3_idealized_conc(params)], dtype=np.float64),
        "mass_integral_initial": np.array([float(result["model"].concData._y[0])], dtype=np.float64),
        "mass_integral_final": np.array([float(result["model"].concData.y())], dtype=np.float64),
        "params_json": np.array(json.dumps(_jsonable(payload_params), sort_keys=True)),
    }
    if result.get("transformed_u_grid") is not None:
        payload["transformed_u_grid"] = np.asarray(result["transformed_u_grid"], dtype=np.float64)
    if result.get("transformed_v_grid") is not None:
        payload["transformed_v_grid"] = np.asarray(result["transformed_v_grid"], dtype=np.float64)
    return payload


def save_illingworth_run_result(path, payload):
    """Saves an Illingworth Figure-3 run payload to a compressed ``.npz`` file."""
    save_path = pathlib.Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **payload)
    return save_path


def _format_fig3_timestep_label(params):
    """Returns a compact timestep label for fixed and semi-log Figure-3 runs."""
    if params.get("dt_mode", "fixed") == "semi_log":
        return f"semi-log dt: T0={params['semiLogT0']} s, dln(t)={params['semiLog_dt']}"
    return f"dt={params['time_step_s']} s"


def build_fig3_present_work_model(params=None, record=None):
    """
    Builds the planar Illingworth model for the paper's Figure 3 present-work curve.

    Inputs are kept in the paper's units: micrometers, seconds, and atomic
    percent. The diffusion model is scale-consistent, so micrometers are used
    directly for length and ``um^2/s`` for diffusivity.
    """
    p = FIG3_PRESENT_WORK_PARAMS if params is None else {**FIG3_PRESENT_WORK_PARAMS, **dict(params)}
    record = p["record"] if record is None else record

    c_liquid0 = p["c_liquid0_atpct"] / 100.0
    c_solid0 = p["c_solid0_atpct"] / 100.0
    c_liquid_int = p["c_liquid_int_atpct"] / 100.0
    c_solid_int = p["c_solid_int_atpct"] / 100.0

    phase_a_nodes, phase_b_nodes, n_mesh = _fig3_phase_node_counts(p)
    transformed_u_grid, transformed_v_grid, _ = build_symmetric_fig3_transformed_grids(p)

    profile = ProfileBuilder([(StepProfile1D(p["s0_um"], c_liquid0, c_solid0), "P")])
    mesh = CartesianFD1D(["P"], [0.0, p["R_um"]], n_mesh)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=["LIQUID", "SOLID"],
        diffusivities={"LIQUID": p["D_liquid_um2_s"], "SOLID": p["D_solid_um2_s"]},
    )
    additionalInputParms = {}
    if p.get('max_iterations', None) is not None:
        additionalInputParms.update({'max_iterations':p["max_iterations"]})
    return MovingBoundaryIllingworthFD1DModel(
        mesh,
        ["NI", "P"],
        ["LIQUID", "SOLID"],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=p["s0_um"],
        interface_compositions=(c_liquid_int, c_solid_int),
        time_step=p["time_step_s"],
        dt_mode=p.get("dt_mode", "fixed"),
        semiLog_dt=p.get("semiLog_dt"),
        semiLogT0=p.get("semiLogT0"),
        phase_a_nodes=phase_a_nodes,
        phase_b_nodes=phase_b_nodes,
        transformed_u_grid=transformed_u_grid,
        transformed_v_grid=transformed_v_grid,
        tolerance=1.0e-8,
        record=record,
        record_pq_data=p.get("record_pq_data", True),
        preallocate_recordings=p.get("preallocate_recordings", False),
        **additionalInputParms,
    )


def run_fig3_present_work(params=None):
    """
    Runs and returns data for the Figure 3 present-work liquid half-width curve.

    Non-constant transformed grids are previewed before the solve when
    ``show_grid_plot`` is true, and their metadata is returned for saved-run
    provenance.
    """
    p = FIG3_PRESENT_WORK_PARAMS if params is None else {**FIG3_PRESENT_WORK_PARAMS, **dict(params)}
    transformed_u_grid, transformed_v_grid, grid_metadata = build_symmetric_fig3_transformed_grids(p)
    dt_mode = p.get("dt_mode", "fixed")
    if dt_mode == "fixed":
        n_steps = int(np.ceil(p["t_end_s"] / p["time_step_s"]))
    elif dt_mode == "semi_log":
        if p.get("semiLog_dt") is None or p.get("semiLogT0") is None:
            raise ValueError("semiLog_dt and semiLogT0 must be set when dt_mode is 'semi_log'.")
        if p["semiLog_dt"] <= 0 or p["semiLogT0"] <= 0:
            raise ValueError("semiLog_dt and semiLogT0 must be positive when dt_mode is 'semi_log'.")
        if p["t_end_s"] <= p["semiLogT0"]:
            n_steps = 1
        else:
            n_steps = int(np.ceil((np.log(p["t_end_s"]) - np.log(p["semiLogT0"])) / p["semiLog_dt"])) + 1
    else:
        raise ValueError("dt_mode must be 'fixed' or 'semi_log'.")
    if n_steps > 250_000:
        raise ValueError(
            f"Figure 3 run would require about {n_steps} Python implicit steps. "
            "Increase FIG3_PRESENT_WORK_PARAMS['time_step_s'] for fixed-step exploratory plotting, "
            "increase semiLog_dt for semi-log exploratory plotting, "
            "or run a shorter t_end_s."
        )
    print(f"Estimated number of time-steps: {n_steps}")
    if grid_metadata["grid_type"] != "constant" and p.get("show_grid_plot", True):
        plot_fig3_transformed_grids(p)
    model = build_fig3_present_work_model(p, record=p["record"])
    python_start = time.perf_counter()
    # debugInPlace()
    model.solve(p["t_end_s"], iterator=explicitEulerIterator, minDtFrac=1e-15, verbose=True, vIt=100)
    python_runtime_s = time.perf_counter() - python_start
    # debugInPlace()
    n = model.interfaceData.N + 1
    time_s = model.interfaceData._time[:n].copy()
    liquid_half_width_um = model.interfaceData._y[:n].copy()
    result = {
        "time_s": time_s,
        "liquid_half_width_um": liquid_half_width_um,
        "theoretical_max_um": theoretical_fig3_max_liquid_half_width_um(p),
        "model": model,
        "params": p,
        "python_runtime_s": python_runtime_s,
        "grid_metadata": grid_metadata,
        "transformed_u_grid": transformed_u_grid,
        "transformed_v_grid": transformed_v_grid,
    }
    if p.get("checkAgainstAuthorsCPP", False):
        if dt_mode != "fixed":
            result["cpp_comparison_skipped"] = (
                "Authors' generated C++ comparison is skipped for dt_mode='semi_log' "
                "because that driver mirrors only fixed timesteps."
            )
            print(result["cpp_comparison_skipped"])
        else:
            comparison = compare_fig3_result_to_authors_cpp(
                result,
                compiler=p.get("cpp_compiler"),
                build_dir=p.get("cpp_build_dir"),
            )
            result["cpp_comparison"] = comparison
            print_comparison_summary(comparison, print_table=p.get("cpp_print_table", False))
    return result


def plot_fig3_present_work(params=None, ax=None):
    """
    Plots the paper's Figure 3 present-work curve from the Python implementation.

    The generated plot uses the paper's log-time axis and includes the stated
    theoretical maximum liquid-layer half-width. When ``plot_conc`` is true,
    recorded ``concData`` values are plotted on a secondary y-axis.
    """
    import matplotlib.pyplot as plt

    notebook_params = FIG3_NOTEBOOK_CONFIG if params is None else {**FIG3_NOTEBOOK_CONFIG, **dict(params)}
    result = run_fig3_present_work(notebook_params)
    p = result["params"]
    created_figure = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)

    mask = result["time_s"] > 0
    ax.plot(
        result["time_s"][mask],
        result["liquid_half_width_um"][mask],
        label=p.get("label") or "Present work",
        linewidth=2.0,
        zorder=5
    )
    if p.get("checkAgainstAuthorsCPP", False) and "cpp_comparison" in result:
        comparison = result["cpp_comparison"]
        cpp_time = np.asarray(comparison.get("cpp_time", comparison["time"]), dtype=np.float64)
        cpp_interface = np.asarray(comparison.get("cpp_interface", comparison["cpp_s"]), dtype=np.float64)
        cpp_mask = np.isfinite(cpp_time) & np.isfinite(cpp_interface) & (cpp_time > 0)
        ax.plot(
            cpp_time[cpp_mask],
            cpp_interface[cpp_mask],
            color="tab:orange",
            linestyle="dotted",
            linewidth=1.4,
            label="Authors' C++ reference",
            zorder=6,
        )
    ax.plot(
        [0.1, p["t_end_s"]],
        [result["theoretical_max_um"], result["theoretical_max_um"]],
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label=f"Theoretical maximum = {result['theoretical_max_um']:.1f} um",
    )
    for overlay in p.get("overlay_csvs", []):
        csv_path = pathlib.Path(overlay["path"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Overlay CSV does not exist: {csv_path}")
        overlay_t_s, overlay_half_width_um = _load_time_half_width_csv(csv_path)
        overlay_mask = np.isfinite(overlay_t_s) & np.isfinite(overlay_half_width_um)
        ax.scatter(
            overlay_t_s[overlay_mask],
            overlay_half_width_um[overlay_mask],
            s=float(overlay.get("size", 26)),
            color=overlay.get("color", None),
            marker=overlay.get("marker", "o"),
            facecolors=overlay.get("facecolors", "none"),
            label=overlay.get("label", csv_path.stem),
            zorder=float(overlay.get("zorder", 3)),
        )
    ax.set_xscale("log")
    ax.set_xlim(0.00001, p["t_end_s"]) # ax.set_xlim(0.1, p["t_end_s"])
    ax.set_ylim(0, 24) #ax.set_ylim(12.5, 24) #ax.set_ylim(0.0, 30.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Interface position / liquid half-width (um)")
    if p.get("spatial_step_um") is not None:
        titleStr = f"{_format_fig3_timestep_label(p)}, transformed step~{p['spatial_step_um']} um"
    else:
        titleStr = f"{_format_fig3_timestep_label(p)}, n (alpha,beta)~{(p['n_alpha'], p['n_beta'])} um"
    grid_label = str(p.get("grid_type", "constant"))
    if grid_label == "geometric":
        titleStr += f", {grid_label} grid r={float(p.get('geometric_ratio', 1.0)):.6g}"
    else:
        titleStr += f", {grid_label} grid"
    ax.set_title(
        "Illingworth and Golosnoy 2005 Fig. 3 present-work curve\n"
        + titleStr
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="center left")

    if p.get("plot_conc", True):
        model = result["model"]
        n_conc = model.concData.N + 1
        conc_time_arr = model.concData._time[:n_conc].copy()
        conc_arr = model.concData._y[:n_conc].copy()
        conc_mask = np.isfinite(conc_time_arr) & np.isfinite(conc_arr) & (conc_time_arr > 0)
        if np.any(conc_mask):
            ax_twin = ax.twinx()
            ax_twin.set_ylabel("conc", color="tab:green")
            ax_twin.tick_params(axis="y", colors="tab:green")
            ax_twin.plot(
                conc_time_arr[conc_mask],
                conc_arr[conc_mask],
                lw=1.0,
                color="tab:green",
                label="Conc",
            )
            conc_t0 = conc_time_arr[conc_mask][0]
            conc_t1 = conc_time_arr[conc_mask][-1]
            ax_twin.plot(
                [conc_t0, conc_t1],
                [conc_arr[0], conc_arr[0]],
                lw=1.0,
                color="tab:green",
                linestyle="dashdot",
                label="Initial conc",
            )
            idealized_conc = compute_fig3_idealized_conc(p)
            print(f"idealized_conc: {idealized_conc}")
            print(f"(conc_arr[0], conc_arr[-1], conc_arr.min(), conc_arr.max()): {(conc_arr[0], conc_arr[-1], conc_arr.min(), conc_arr.max())}")
            ax_twin.plot(
                [conc_t0, conc_t1],
                [idealized_conc, idealized_conc],
                lw=1.0,
                color="tab:green",
                linestyle="dashed",
                label="Idealized conc",
            )
            ax_twin.legend(fontsize=8, loc="center right")

            initial_conc = conc_arr[0]
            print(f"Initial Conc:   {initial_conc}")
            print(f"Idealized Conc: {idealized_conc}")
            print(f"Initial vs Idealized Conc Frac Diff: {(initial_conc-idealized_conc)/idealized_conc}")
            conc_diffFromInitial_arr = conc_arr - initial_conc
            conc_diffFromIdealized_arr = conc_arr - idealized_conc
            print(f"Diff from Initial: {float(conc_diffFromInitial_arr.min()/initial_conc), float(conc_diffFromInitial_arr.max()/initial_conc)}")
            print(f"Diff from Idealized: {float(conc_diffFromIdealized_arr.min()/idealized_conc), float(conc_diffFromIdealized_arr.max()/idealized_conc)}")

    if p.get("timeProfiling"):
        return result, ax
    out = p.get("out")
    if out:
        ax.figure.savefig(out, bbox_inches="tight")
    payload = build_illingworth_run_payload(result, label=p.get("label"))
    save_path = None
    if p.get("save_run", False):
        save_path = save_illingworth_run_result(p["save_run_path"], payload)
        print(f"Saved run: {save_path}")
    if created_figure:
        if p.get("show", True):
            plt.show()
        else:
            plt.close(ax.figure)
    result["payload"] = payload
    result["save_path"] = save_path
    return result, ax


def parse_results(path):
    """Parses the authors' ``results.txt`` interface-position output."""
    data = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"No rows found in {path}.")
    data = np.atleast_2d(data)
    return np.asarray(data[:, 0], dtype=np.float64), np.asarray(data[:, 1], dtype=np.float64)


def _format_cpp_float(value):
    """Formats a Python float as a C++ double literal with round-trip precision."""
    return format(float(value), ".17g")


def fig3_params_to_author_cpp_params(params):
    """
    Converts Figure-3 Python parameters to the authors' planar C++ parameters.

    The authors' C++ driver requires an integer number of fixed timesteps. The
    Python comparison therefore requires ``t_end_s`` to be an integer multiple
    of ``time_step_s`` when running against generated C++ reference output.
    """
    p = FIG3_PRESENT_WORK_PARAMS if params is None else {**FIG3_PRESENT_WORK_PARAMS, **dict(params)}
    if p.get("dt_mode", "fixed") != "fixed":
        raise ValueError("Generated authors' C++ comparison currently supports only dt_mode='fixed'.")
    n_time_steps_float = float(p["t_end_s"]) / float(p["time_step_s"])
    n_time_steps = int(round(n_time_steps_float))
    if not np.isclose(n_time_steps_float, n_time_steps, rtol=0.0, atol=1e-9):
        raise ValueError(
            "C++ comparison requires t_end_s to be an integer multiple of time_step_s; "
            f"got t_end_s/time_step_s={n_time_steps_float}."
        )
    phase_a_nodes, phase_b_nodes, _ = _fig3_phase_node_counts(p)
    _, _, grid_metadata = build_symmetric_fig3_transformed_grids(p)
    return {
        "s0": float(p["s0_um"]),
        "R": float(p["R_um"]),
        "n_alpha": phase_a_nodes,
        "d_alpha": float(p["D_liquid_um2_s"]),
        "initial_alpha": float(p["c_liquid0_atpct"]) / 100.0,
        "interface_alpha": float(p["c_liquid_int_atpct"]) / 100.0,
        "n_beta": phase_b_nodes,
        "d_beta": float(p["D_solid_um2_s"]),
        "initial_beta": float(p["c_solid0_atpct"]) / 100.0,
        "interface_beta": float(p["c_solid_int_atpct"]) / 100.0,
        "time_step": float(p["time_step_s"]),
        "n_time_steps": n_time_steps,
        "tolerance": 1.0e-8,
        "grid_type": grid_metadata["grid_type"],
        "geometric_ratio": grid_metadata["geometric_ratio"],
    }


def _render_author_cpp_driver(params):
    """Renders a generated C++ driver that uses the authors' implementation files."""
    p = params
    return f"""#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "data_structures.h"
#include "subroutines.h"
#include "InOut.h"

static void fill_transformed_grid(double *u, int n, const char *side, const char *grid_type, double ratio)
{{
    int i;
    double total = 0.0;
    double cumulative = 0.0;
    double log_ratio;
    double max_log_width;
    if (n < 2)
    {{
        return;
    }}
    u[0] = 0.0;
    u[n - 1] = 1.0;
    if (grid_type[0] != 'g' || fabs(ratio - 1.0) < 1.0e-14)
    {{
        for (i = 1; i < n - 1; i++)
        {{
            u[i] = double(i) / double(n - 1);
        }}
        return;
    }}
    log_ratio = log(ratio);
    max_log_width = double(n - 2) * log_ratio;
    if (max_log_width < 0.0)
    {{
        max_log_width = 0.0;
    }}
    for (i = 0; i < n - 1; i++)
    {{
        total += exp(double(i) * log_ratio - max_log_width);
    }}
    for (i = 0; i < n - 2; i++)
    {{
        double exponent = (side[0] == 'l') ? double(n - 2 - i) : double(i);
        cumulative += exp(exponent * log_ratio - max_log_width) / total;
        u[i + 1] = cumulative;
    }}
}}

int main(void)
{{
    const double s_0 = {_format_cpp_float(p["s0"])};
    const double R = {_format_cpp_float(p["R"])};
    const int nAlpha = {int(p["n_alpha"])};
    const double dAlpha = {_format_cpp_float(p["d_alpha"])};
    const double initialAlpha = {_format_cpp_float(p["initial_alpha"])};
    const double interAlpha = {_format_cpp_float(p["interface_alpha"])};
    const int nBeta = {int(p["n_beta"])};
    const double dBeta = {_format_cpp_float(p["d_beta"])};
    const double initialBeta = {_format_cpp_float(p["initial_beta"])};
    const double interBeta = {_format_cpp_float(p["interface_beta"])};
    const double time_step = {_format_cpp_float(p["time_step"])};
    const int n_time_steps = {int(p["n_time_steps"])};
    const double tol = {_format_cpp_float(p["tolerance"])};
    const char *grid_type = "{p["grid_type"]}";
    const double geometric_ratio = {_format_cpp_float(p["geometric_ratio"])};

    int i;
    int tmp;
    two_phase *whole_system = (two_phase *) calloc(1, sizeof(two_phase));
    whole_system->s = s_0;
    whole_system->l = R;
    whole_system->old_s = whole_system->s;
    whole_system->future_s = whole_system->s;

    whole_system->left = (single_phase *) calloc(1, sizeof(single_phase));
    whole_system->left->n = nAlpha;
    whole_system->left->d_coeff = dAlpha;
    whole_system->left->c_boundary = interAlpha;
    whole_system->left->u = (double *) calloc(whole_system->left->n, sizeof(double));
    whole_system->left->c = (double *) calloc(whole_system->left->n, sizeof(double));
    whole_system->left->future_c = (double *) calloc(whole_system->left->n, sizeof(double));
    fill_transformed_grid(whole_system->left->u, whole_system->left->n, "left", grid_type, geometric_ratio);
    for (i = 0; i < whole_system->left->n; i++)
    {{
        whole_system->left->c[i] = initialAlpha;
        whole_system->left->future_c[i] = whole_system->left->c[i];
    }}
    whole_system->left->c[whole_system->left->n - 1] = whole_system->left->c_boundary;
    whole_system->left->future_c[whole_system->left->n - 1] = whole_system->left->c_boundary;

    whole_system->right = (single_phase *) calloc(1, sizeof(single_phase));
    whole_system->right->n = nBeta;
    whole_system->right->d_coeff = dBeta;
    whole_system->right->c_boundary = interBeta;
    whole_system->right->u = (double *) calloc(whole_system->right->n, sizeof(double));
    whole_system->right->c = (double *) calloc(whole_system->right->n, sizeof(double));
    whole_system->right->future_c = (double *) calloc(whole_system->right->n, sizeof(double));
    fill_transformed_grid(whole_system->right->u, whole_system->right->n, "right", grid_type, geometric_ratio);
    for (i = 0; i < whole_system->right->n; i++)
    {{
        whole_system->right->c[i] = initialBeta;
        whole_system->right->future_c[i] = whole_system->right->c[i];
    }}
    whole_system->right->c[0] = whole_system->right->c_boundary;
    whole_system->right->future_c[0] = whole_system->right->c_boundary;

    FILE *fpt = fopen("results.txt", "w");
    fprintf(fpt, "Time\\tInterface Position\\n");
    for (i = 0; i < n_time_steps + 1; i++)
    {{
        out_interface(whole_system, double(i) * time_step, fpt);
        if (i < n_time_steps)
        {{
            tmp = take_step_planar(whole_system, time_step, tol);
            if (tmp < 0)
            {{
                fclose(fpt);
                return 2;
            }}
        }}
    }}

    free(whole_system->left->u);
    free(whole_system->left->c);
    free(whole_system->left->future_c);
    free(whole_system->right->u);
    free(whole_system->right->c);
    free(whole_system->right->future_c);
    free(whole_system->left);
    free(whole_system->right);
    free(whole_system);
    fclose(fpt);
    return 0;
}}
"""


def compile_and_run_authors_cpp_with_params(params, source_dir=None, build_dir=None, compiler=None, return_runtime=False):
    """
    Compiles and runs a generated C++ driver with user-provided parameters.

    The generated driver is written to the build directory and links against the
    authors' untouched ``subroutines.cpp``, ``trimatrix.cpp``, and ``InOut.cpp``
    files. This keeps the original MAP source tree read-only while allowing the
    Python run parameters to be mirrored exactly in the C++ reference run.
    """
    source_dir = pathlib.Path(source_dir) if source_dir is not None else AUTHOR_CPP_SOURCE_DIR
    compiler = compiler or shutil.which("g++")
    if compiler is None:
        raise FileNotFoundError("Could not find g++ on PATH.")
    compiler_path = pathlib.Path(compiler)
    compiler_dir = compiler_path.parent if compiler_path.parent != pathlib.Path(".") else None

    cleanup = False
    if build_dir is None:
        temp_root = REPO_ROOT / ".pytest_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        build_dir = temp_root / f"illingworth_cpp_params_{uuid.uuid4().hex}"
        build_dir.mkdir(parents=True, exist_ok=False)
        cleanup = True
    else:
        build_dir = pathlib.Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

    cpp_params = fig3_params_to_author_cpp_params(params)
    driver_path = build_dir / "generated_illingworth_driver.cpp"
    exe = build_dir / f"illingworth_reference_{uuid.uuid4().hex}.exe"
    sources = [
        driver_path,
        source_dir / "subroutines.cpp",
        source_dir / "trimatrix.cpp",
        source_dir / "InOut.cpp",
    ]
    try:
        driver_path.write_text(_render_author_cpp_driver(cpp_params), encoding="utf-8")
        compile_cmd = [compiler, "-I", str(source_dir), *[str(s) for s in sources], "-o", str(exe)]
        env = None
        if compiler_dir is not None:
            env = dict(os.environ)
            env["PATH"] = str(compiler_dir) + os.pathsep + env.get("PATH", "")
        compile_start = time.perf_counter()
        compile_result = subprocess.run(compile_cmd, text=True, capture_output=True, env=env)
        compile_runtime_s = time.perf_counter() - compile_start
        if compile_result.returncode != 0:
            raise RuntimeError(
                "Generated authors' C++ compile failed with exit code "
                f"{compile_result.returncode}.\nSTDOUT:\n{compile_result.stdout}\nSTDERR:\n{compile_result.stderr}"
            )

        run_start = time.perf_counter()
        run_result = subprocess.run([str(exe)], cwd=build_dir, text=True, capture_output=True, env=env)
        run_runtime_s = time.perf_counter() - run_start
        results_path = build_dir / "results.txt"
        if run_result.returncode != 0:
            raise RuntimeError(
                "Generated authors' executable failed.\n"
                f"Exit code: {run_result.returncode}\nSTDOUT:\n{run_result.stdout}\nSTDERR:\n{run_result.stderr}"
            )
        if not results_path.exists():
            raise RuntimeError(
                "Generated authors' executable did not produce results.txt.\n"
                f"STDOUT:\n{run_result.stdout}\nSTDERR:\n{run_result.stderr}"
            )
        result = parse_results(results_path)
        if return_runtime:
            return result, {
                "cpp_compile_runtime_s": compile_runtime_s,
                "cpp_run_runtime_s": run_runtime_s,
                "cpp_total_runtime_s": compile_runtime_s + run_runtime_s,
            }
        return result
    finally:
        if cleanup:
            shutil.rmtree(build_dir, ignore_errors=True)


def compile_and_run_authors_cpp(source_dir=None, build_dir=None, compiler=None):
    """
    Compiles and runs the untouched MAP C++ default case.

    Build products are written outside the authors' source directory. If
    ``build_dir`` is omitted, a temporary directory is used and cleaned up with
    ``ignore_errors=True`` to avoid Windows file-lock failures after compiler
    errors.
    """
    source_dir = pathlib.Path(source_dir) if source_dir is not None else AUTHOR_CPP_SOURCE_DIR
    compiler = compiler or shutil.which("g++")
    if compiler is None:
        raise FileNotFoundError("Could not find g++ on PATH.")
    compiler_path = pathlib.Path(compiler)
    compiler_dir = compiler_path.parent if compiler_path.parent != pathlib.Path(".") else None

    cleanup = False
    if build_dir is None:
        temp_root = REPO_ROOT / ".pytest_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        build_dir = temp_root / f"illingworth_cpp_{uuid.uuid4().hex}"
        build_dir.mkdir(parents=True, exist_ok=False)
        cleanup = True
    else:
        build_dir = pathlib.Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

    exe = build_dir / f"illingworth_reference_{uuid.uuid4().hex}.exe"
    sources = [
        source_dir / "ConservativeIFF.cpp",
        source_dir / "subroutines.cpp",
        source_dir / "trimatrix.cpp",
        source_dir / "InOut.cpp",
    ]
    try:
        compile_cmd = [compiler, *[str(s) for s in sources], "-o", str(exe)]
        env = None
        if compiler_dir is not None:
            env = dict(os.environ)
            env["PATH"] = str(compiler_dir) + os.pathsep + env.get("PATH", "")
        compile_result = subprocess.run(compile_cmd, text=True, capture_output=True, env=env)
        if compile_result.returncode != 0:
            raise RuntimeError(
                "Authors' C++ compile failed with exit code "
                f"{compile_result.returncode}.\nSTDOUT:\n{compile_result.stdout}\nSTDERR:\n{compile_result.stderr}"
            )

        run_result = subprocess.run([str(exe)], cwd=build_dir, text=True, capture_output=True, env=env)
        results_path = build_dir / "results.txt"
        if not results_path.exists():
            raise RuntimeError(
                "Authors' executable did not produce results.txt.\n"
                f"Exit code: {run_result.returncode}\nSTDOUT:\n{run_result.stdout}\nSTDERR:\n{run_result.stderr}"
            )
        return parse_results(results_path)
    finally:
        if cleanup:
            shutil.rmtree(build_dir, ignore_errors=True)


def compare_default_case(compiler=None, build_dir=None):
    """Runs both implementations and returns comparison arrays and error metrics."""
    cpp_time, cpp_s = compile_and_run_authors_cpp(compiler=compiler, build_dir=build_dir)
    py_time, py_s = run_python_default()
    if len(cpp_time) != len(py_time):
        raise ValueError(f"Length mismatch: C++ has {len(cpp_time)} rows, Python has {len(py_time)} rows.")
    if not np.allclose(cpp_time, py_time, rtol=0.0, atol=1e-12):
        raise ValueError("C++ and Python output times do not match.")
    abs_diff = np.abs(cpp_s - py_s)
    rel_diff = abs_diff / np.maximum(np.abs(cpp_s), 1e-300)
    return {
        "time": cpp_time,
        "cpp_s": cpp_s,
        "python_s": py_s,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "max_abs_diff": float(np.max(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
    }


def compare_fig3_result_to_authors_cpp(result, compiler=None, build_dir=None):
    """
    Compares a Python Figure-3 run against a parameter-matched C++ reference.

    The generated C++ driver writes every fixed timestep. The Python result may
    be recorded less frequently, so comparison is performed at the Python
    recorded times by indexing the C++ history with ``time / time_step_s``.
    A pandas ``DataFrame`` with side-by-side interface histories is included in
    the returned comparison dictionary under ``"dataframe"``.
    """
    import pandas as pd

    p = result["params"]
    (cpp_time, cpp_s), runtime_info = compile_and_run_authors_cpp_with_params(
        p,
        compiler=compiler,
        build_dir=build_dir,
        return_runtime=True,
    )
    py_time = np.asarray(result["time_s"], dtype=np.float64)
    py_s = np.asarray(result["liquid_half_width_um"], dtype=np.float64)
    dt = float(p["time_step_s"])
    cpp_indices = np.rint(py_time / dt).astype(int)
    if np.any(cpp_indices < 0) or np.any(cpp_indices >= len(cpp_time)):
        raise ValueError("Python recorded times extend outside the generated C++ reference history.")
    matched_cpp_time = cpp_time[cpp_indices]
    time_atol = max(abs(dt) * 1e-8, 1e-12)
    if not np.allclose(matched_cpp_time, py_time, rtol=0.0, atol=time_atol):
        max_time_diff = float(np.max(np.abs(matched_cpp_time - py_time)))
        raise ValueError(f"C++ and Python output times do not match; max difference is {max_time_diff:.3e}.")
    matched_cpp_s = cpp_s[cpp_indices]
    abs_diff = np.abs(matched_cpp_s - py_s)
    rel_diff = abs_diff / np.maximum(np.abs(matched_cpp_s), 1e-300)
    comparison_df = pd.DataFrame(
        {
            "time_s": py_time,
            "python_interface_um": py_s,
            "authors_cpp_interface_um": matched_cpp_s,
            "interface_abs_diff_um": abs_diff,
            "interface_rel_diff": rel_diff,
            "authors_cpp_time_s": matched_cpp_time,
            "authors_cpp_step_index": cpp_indices,
        }
    )
    python_runtime_s = result.get("python_runtime_s")
    runtime_ratio = np.nan
    if python_runtime_s is not None and runtime_info["cpp_run_runtime_s"] > 0:
        runtime_ratio = float(python_runtime_s / runtime_info["cpp_run_runtime_s"])
    return {
        "time": py_time,
        "cpp_s": matched_cpp_s,
        "python_s": py_s,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "max_abs_diff": float(np.max(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "cpp_time": cpp_time,
        "cpp_interface": cpp_s,
        "dataframe": comparison_df,
        "python_runtime_s": None if python_runtime_s is None else float(python_runtime_s),
        **runtime_info,
        "python_to_cpp_run_runtime_ratio": runtime_ratio,
    }


def print_comparison_summary(comparison, print_table=False):
    """Prints a compact comparison summary suitable for scripts or notebooks."""
    print(f"Rows compared: {len(comparison['time'])}")
    print(f"Max absolute interface-position difference: {comparison['max_abs_diff']:.16e}")
    print(f"Max relative interface-position difference: {comparison['max_rel_diff']:.16e}")
    if comparison.get("python_runtime_s") is not None:
        print(f"Python solve runtime: {comparison['python_runtime_s']:.6g} s")
    if comparison.get("cpp_run_runtime_s") is not None:
        print(f"C++ run runtime: {comparison['cpp_run_runtime_s']:.6g} s")
        print(f"C++ compile runtime: {comparison['cpp_compile_runtime_s']:.6g} s")
        print(f"C++ compile+run runtime: {comparison['cpp_total_runtime_s']:.6g} s")
    if np.isfinite(comparison.get("python_to_cpp_run_runtime_ratio", np.nan)):
        print(f"Python/C++ run runtime ratio: {comparison['python_to_cpp_run_runtime_ratio']:.6g}")
    if print_table:
        print("\nTime\tC++ s\tPython s\tAbs diff")
        for t, cpp_s, py_s, diff in zip(
            comparison["time"],
            comparison["cpp_s"],
            comparison["python_s"],
            comparison["abs_diff"],
        ):
            print(f"{t:.12g}\t{cpp_s:.16g}\t{py_s:.16g}\t{diff:.3e}")


def run_from_config(config=None):
    """
    Runs the comparison using editable in-file settings.

    This is the notebook/interactive-window entry point. Edit ``SCRIPT_CONFIG``
    near the top of this file, then run this cell or call ``run_from_config()``.
    """
    config = SCRIPT_CONFIG if config is None else {**SCRIPT_CONFIG, **dict(config)}
    if config["run_mode"] == "cpp_comparison":
        comparison = compare_default_case(compiler=config["compiler"], build_dir=config["build_dir"])
        print_comparison_summary(comparison, print_table=config["print_table"])
        return comparison
    if config["run_mode"] == "fig3_present_work":
        return plot_fig3_present_work({**FIG3_NOTEBOOK_CONFIG, **config["fig3_overrides"]})
    raise ValueError("SCRIPT_CONFIG['run_mode'] must be 'cpp_comparison' or 'fig3_present_work'.")


def build_parser():
    parser = argparse.ArgumentParser(description="Compare the Python Illingworth planar model to the authors' C++ code.")
    parser.add_argument("--compiler", default=None, help="Path to g++; defaults to PATH lookup.")
    parser.add_argument("--build-dir", default=None, help="Optional directory for C++ build products.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    comparison = compare_default_case(build_dir=args.build_dir, compiler=args.compiler)
    print_comparison_summary(comparison, print_table=False)
    return comparison


# SCRIPT_CONFIG["run_mode"] = "fig3_present_work"
# SCRIPT_CONFIG["fig3_overrides"] = {
#     "time_step_s": 10.0,
#     "t_end_s": 1.0e5,
#     "show": True,
# }

# Example interactive call:
# result, ax = plot_fig3_present_work({
#     "t_end_s": 10.0,
#     "time_step_s": 0.01,
#     "show": True,
#     "record": True,
# })

if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        result, ax = plot_fig3_present_work(FIG3_NOTEBOOK_CONFIG)
        # comparison = run_from_config()
    else:
        result, ax = plot_fig3_present_work(FIG3_NOTEBOOK_CONFIG)
        # main()

 # %%
