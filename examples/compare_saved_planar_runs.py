#%%
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np


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
        
def _resolve_this_file():
    """
    Returns this script's path without trusting stale Interactive Window globals.

    In notebook/interactive execution, VS Code can leave ``__file__`` bound to
    whichever file most recently populated the shared kernel namespace. We
    therefore accept ``__file__`` only when it already points to this script
    and otherwise fall back to locating the file from the current working
    directory.
    """
    expected_name = "compare_saved_planar_runs.py"
    if "__file__" in globals():
        candidate = pathlib.Path(__file__).resolve()
        if candidate.name == expected_name:
            return candidate

    cwd = pathlib.Path.cwd().resolve()
    candidates = [cwd / expected_name, cwd / "examples" / expected_name]
    candidates.extend(parent / "examples" / expected_name for parent in cwd.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"Could not resolve {expected_name} from cwd={cwd}.")


THIS_FILE = _resolve_this_file()
SCRIPT_DIR = THIS_FILE.parent

COMPARE_SAVED_PLANAR_RUNS_CONFIG = {
    "olaye_run_path": SCRIPT_DIR / "Olaye2020" / "olaye2020_fig5_saved_run.npz",
    "illingworth_run_path": SCRIPT_DIR / "Illingworth2005" / "illingworth2005_fig3_saved_run.npz",
    "olaye_label": None,
    "illingworth_label": None,
    "plot_lee_and_oh_n3013": True,
    "lee_and_oh_n3013_path": pathlib.Path.home() / "Downloads" / "LeeAndOh_NiP_results_100sec_N3013.npz",
    "lee_and_oh_n3013_label": "Lee and Oh calculated results (N=3013)",
    "plot_figure5_rough": True,
    "plot_figure5_rough2": True,
    "figure5_rough_path": SCRIPT_DIR / "Olaye2020" / "Olaye2020_fig5_alt_PresentModelRough_curve.csv",
    "figure5_rough2_path": SCRIPT_DIR / "Olaye2020" / "Olaye2020_fig5_alt_PresentModelRough2_curve.csv",
    "x_axis": "log",
    "out": SCRIPT_DIR / "saved_planar_runs_overlay.png",
    "show": True,
    "plot_theoretical_max": True,
    "show_parameter_summary": True,
    "compareToAnalyticalSoln": True,
    "analytical_out": SCRIPT_DIR / "saved_planar_runs_analytical_early.png",
    # Set to a number of seconds, None for all saved times, or "semi_infinite"
    # to estimate the cutoff from when diffusion reaches a far boundary.
    "analytical_time_max_s": "auto",
    "semi_infinite_erfc_argument_min": 5.0,
    "analytical_max_points_per_run": None,
}


def _load_saved_run(path):
    """
    Loads a saved planar-run payload and validates the shared overlay contract.

    Required arrays are ``time_s`` and ``half_width_um`` with at least two
    finite values each.
    """
    load_path = pathlib.Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Saved run does not exist: {load_path}")

    with np.load(load_path, allow_pickle=False) as loaded:
        if "time_s" not in loaded or "half_width_um" not in loaded:
            raise ValueError(f"{load_path} must contain saved arrays: time_s, half_width_um")

        time_s = np.asarray(loaded["time_s"], dtype=np.float64)
        half_width_um = np.asarray(loaded["half_width_um"], dtype=np.float64)
        metadata = {}
        for key in ["label", "source_script", "model_family", "params_json", "model_variant"]:
            if key in loaded:
                metadata[key] = str(np.asarray(loaded[key]).reshape(-1)[0])
        for key in ["theoretical_max_um", "idealized_mass_integral", "mass_integral_initial", "mass_integral_final"]:
            if key in loaded:
                metadata[key] = float(np.asarray(loaded[key], dtype=np.float64).reshape(-1)[0])

    if time_s.shape != half_width_um.shape:
        raise ValueError(f"{load_path} has mismatched shapes for time_s and half_width_um.")

    mask = np.isfinite(time_s) & np.isfinite(half_width_um)
    time_s = time_s[mask]
    half_width_um = half_width_um[mask]
    if time_s.size < 2:
        raise ValueError(f"{load_path} must contain at least two finite saved points.")

    return {
        "path": load_path,
        "time_s": time_s,
        "half_width_um": half_width_um,
        "metadata": metadata,
    }


def _default_label(run, fallback):
    """Builds a stable plot label when optional metadata is absent."""
    return run["metadata"].get("label") or fallback


def _parse_params_json(run):
    """Returns the saved parameter dictionary when ``params_json`` is present."""
    params_json = run["metadata"].get("params_json")
    if params_json is None:
        return {}
    try:
        parsed = json.loads(params_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _require_float(params, key, run_label):
    """Returns a saved numeric parameter with a run-specific error message."""
    if key not in params:
        raise ValueError(f"{run_label} saved run is missing required analytical parameter: {key}")
    return float(params[key])


def _analytical_constants_from_saved_params(run, run_label):
    """
    Extracts binary analytical-solution constants from a saved planar run.

    The returned diffusivities use ``um^2/s`` so ``beta`` is in
    ``um/sqrt(s)`` and can be plotted directly against the saved
    ``half_width_um`` history.
    """
    params = _parse_params_json(run)
    if not params:
        raise ValueError(f"{run_label} saved run does not contain params_json for analytical comparison.")

    if {"c_liquid0_pct", "c_solid0_pct", "c_liquid_int_pct", "c_solid_int_pct"}.issubset(params):
        c_a0 = _require_float(params, "c_liquid0_pct", run_label) / 100.0
        c_b0 = _require_float(params, "c_solid0_pct", run_label) / 100.0
        c_a_eq = _require_float(params, "c_liquid_int_pct", run_label) / 100.0
        c_b_eq = _require_float(params, "c_solid_int_pct", run_label) / 100.0
    elif {"c_liquid0_atpct", "c_solid0_atpct", "c_liquid_int_atpct", "c_solid_int_atpct"}.issubset(params):
        c_a0 = _require_float(params, "c_liquid0_atpct", run_label) / 100.0
        c_b0 = _require_float(params, "c_solid0_atpct", run_label) / 100.0
        c_a_eq = _require_float(params, "c_liquid_int_atpct", run_label) / 100.0
        c_b_eq = _require_float(params, "c_solid_int_atpct", run_label) / 100.0
    else:
        raise ValueError(
            f"{run_label} saved run must contain either pct or atpct liquid/solid compositions "
            "for analytical comparison."
        )

    if {"D_liquid_um2_s", "D_solid_um2_s"}.issubset(params):
        d_a = _require_float(params, "D_liquid_um2_s", run_label)
        d_b = _require_float(params, "D_solid_um2_s", run_label)
    elif {"D_liquid_base", "D_solid_base", "D_scale"}.issubset(params):
        um2_per_m2 = 1e12
        scale = _require_float(params, "D_scale", run_label)
        d_a = _require_float(params, "D_liquid_base", run_label) * scale * um2_per_m2
        d_b = _require_float(params, "D_solid_base", run_label) * scale * um2_per_m2
    else:
        raise ValueError(
            f"{run_label} saved run must contain D_liquid_um2_s/D_solid_um2_s or "
            "D_liquid_base/D_solid_base/D_scale for analytical comparison."
        )

    s0_um = _require_float(params, "s0_um", run_label)
    if "R_um" in params:
        r_um = float(params["R_um"])
    elif "R" in params:
        r_um = float(params["R"])
    else:
        r_um = None
    beta_um_sqrt_s = solve_beta(c_a0, c_b0, c_a_eq, c_b_eq, d_a, d_b, left=-100.0, right=100.0)
    return {
        "c_a0": c_a0,
        "c_b0": c_b0,
        "c_a_eq": c_a_eq,
        "c_b_eq": c_b_eq,
        "d_a_um2_s": d_a,
        "d_b_um2_s": d_b,
        "s0_um": s0_um,
        "R_um": r_um,
        "beta_um_sqrt_s": beta_um_sqrt_s,
    }


def _semi_infinite_time_max_s(constants, run_label, erfc_argument_min):
    """
    Estimates when a finite phase stops behaving like a semi-infinite domain.

    For the planar error-function solution, ``eta = L/(2*sqrt(D*t))`` controls
    how strongly a far boundary can influence the interface region. The
    returned cutoff is the first time either initial phase thickness reaches
    ``eta == erfc_argument_min``.
    """
    erfc_argument_min = float(erfc_argument_min)
    if erfc_argument_min <= 0:
        raise ValueError("semi_infinite_erfc_argument_min must be positive.")
    if constants["R_um"] is None:
        raise ValueError(f"{run_label} saved run is missing R_um for semi-infinite cutoff estimation.")

    phase_a_width_um = constants["s0_um"]
    phase_b_width_um = constants["R_um"] - constants["s0_um"]
    if phase_a_width_um <= 0 or phase_b_width_um <= 0:
        raise ValueError(f"{run_label} saved run must have 0 < s0_um < R_um for semi-infinite cutoff estimation.")

    phase_a_time_s = (phase_a_width_um / (2.0 * erfc_argument_min)) ** 2 / constants["d_a_um2_s"]
    phase_b_time_s = (phase_b_width_um / (2.0 * erfc_argument_min)) ** 2 / constants["d_b_um2_s"]
    return min(phase_a_time_s, phase_b_time_s)


def _resolve_analytical_time_max_s(cfg, olaye_constants, illingworth_constants):
    """
    Resolves numeric, unlimited, or semi-infinite analytical plot time limits.

    ``analytical_time_max_s="semi_infinite"`` uses the most restrictive saved
    run and phase so both plotted histories remain inside the estimated
    semi-infinite window.
    """
    time_max_s = cfg.get("analytical_time_max_s")
    if time_max_s is None:
        return None, None
    if isinstance(time_max_s, str):
        if time_max_s.lower() not in {"semi_infinite", "auto"}:
            raise ValueError('analytical_time_max_s must be numeric, None, "semi_infinite", or "auto".')
        erfc_argument_min = cfg.get("semi_infinite_erfc_argument_min", 3.0)
        olaye_time_s = _semi_infinite_time_max_s(olaye_constants, "Olaye", erfc_argument_min)
        illingworth_time_s = _semi_infinite_time_max_s(illingworth_constants, "Illingworth", erfc_argument_min)
        resolved_time_s = min(olaye_time_s, illingworth_time_s)
        return resolved_time_s, {
            "mode": time_max_s.lower(),
            "erfc_argument_min": float(erfc_argument_min),
            "olaye_time_max_s": olaye_time_s,
            "illingworth_time_max_s": illingworth_time_s,
            "time_max_s": resolved_time_s,
        }
    return float(time_max_s), None


def _limit_plot_points(x, y, max_points):
    """Returns an evenly thinned plotting view while preserving endpoints."""
    if max_points is None or len(x) <= int(max_points):
        return x, y
    if int(max_points) < 2:
        raise ValueError("analytical_max_points_per_run must be at least 2 or None.")
    indices = np.linspace(0, len(x) - 1, int(max_points), dtype=np.int64)
    return x[indices], y[indices]


def _plot_analytical_early_time_comparison(olaye, illingworth, cfg):
    """
    Plots early interface change against ``sqrt(t)`` with the beta solution.

    The analytical line is ``s(t) - s0 = 2*beta*sqrt(t)``. Since beta is
    calculated from diffusivities normalized to ``um^2/s``, both axes match
    the saved run units.
    """
    olaye_constants = _analytical_constants_from_saved_params(olaye, "Olaye")
    illingworth_constants = _analytical_constants_from_saved_params(illingworth, "Illingworth")
    time_max_s, semi_infinite_cutoff = _resolve_analytical_time_max_s(cfg, olaye_constants, illingworth_constants)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)
    plotted_max_time = 0.0
    for run, constants, label, color, linestyle in [
        (olaye, olaye_constants, "Olaye saved run", "tab:blue", "solid"),
        (illingworth, illingworth_constants, "Illingworth saved run", "tab:orange", "dotted"),
    ]:
        time_s = run["time_s"]
        interface_change_um = run["half_width_um"] - constants["s0_um"]
        mask = np.isfinite(time_s) & np.isfinite(interface_change_um) & (time_s >= 0)
        if time_max_s is not None:
            mask &= time_s <= time_max_s
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"{label} has fewer than two finite points in the analytical comparison window.")
        plotted_max_time = max(plotted_max_time, float(np.max(time_s[mask])))
        sqrt_time, interface_change_plot_um = _limit_plot_points(
            np.sqrt(time_s[mask]),
            interface_change_um[mask],
            cfg.get("analytical_max_points_per_run"),
        )
        ax.plot(
            sqrt_time,
            interface_change_plot_um,
            linewidth=1.0,
            linestyle='none', #linestyle,
            marker='o',
            markersize=1,
            mfc='none',
            color=color,
            label=cfg["olaye_label"] if run is olaye and cfg["olaye_label"] else cfg["illingworth_label"] if run is illingworth and cfg["illingworth_label"] else _default_label(run, label),
            zorder=3,
        )

    if not np.isclose(olaye_constants["beta_um_sqrt_s"], illingworth_constants["beta_um_sqrt_s"], rtol=1e-10, atol=1e-12):
        print(
            "Analytical beta differs between saved runs: "
            f"Olaye={olaye_constants['beta_um_sqrt_s']:.8g}, "
            f"Illingworth={illingworth_constants['beta_um_sqrt_s']:.8g} um/sqrt(s)."
        )

    beta = illingworth_constants["beta_um_sqrt_s"]
    sqrt_time = np.linspace(0.0, math.sqrt(plotted_max_time), 250)
    ax.plot(
        sqrt_time,
        2.0 * beta * sqrt_time,
        color="0.25",
        linestyle="--",
        linewidth=1.5,
        label=f"Analytical: 2 beta sqrt(t), beta={beta:.6g} um/sqrt(s)",
        zorder=2,
    )
    ax.set_xlabel("sqrt(time) (sqrt(s))")
    ax.set_ylabel("Interface change (um)")
    ax.set_title("Early-time saved runs vs analytical moving-boundary solution")
    if semi_infinite_cutoff is not None:
        ax.text(
            0.02,
            0.98,
            "semi-infinite cutoff: "
            f"t <= {semi_infinite_cutoff['time_max_s']:.6g} s "
            f"(eta >= {semi_infinite_cutoff['erfc_argument_min']:.3g})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    analytical_out = cfg.get("analytical_out")
    if analytical_out:
        out_path = pathlib.Path(analytical_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved analytical comparison figure: {out_path}")

    if cfg.get("show", True):
        plt.show()
    else:
        plt.close(fig)

    return {
        "figure": fig,
        "axes": ax,
        "olaye_constants": olaye_constants,
        "illingworth_constants": illingworth_constants,
        "semi_infinite_cutoff": semi_infinite_cutoff,
    }


def _build_parameter_summary_text(olaye, illingworth):
    """
    Builds a compact, human-readable summary of key saved run parameters.
    """
    olaye_params = _parse_params_json(olaye)
    illingworth_params = _parse_params_json(illingworth)

    lines = ["Saved run parameters", "Olaye:"]
    for key in ["n_phase_a_nodes", "n_phase_b_nodes", "semiLog_dt", "model_variant", "t_end_s", "dt_mode"]:
        if key in olaye_params:
            lines.append(f"  {key}={olaye_params[key]}")
    for key in ["mass_integral_initial", "mass_integral_final", "idealized_mass_integral"]:
        if key in olaye["metadata"]:
            lines.append(f"  {key}={olaye['metadata'][key]:.8g}")

    lines.append("Illingworth:")
    for key in [
        "n_phase_a_nodes",
        "n_phase_b_nodes",
        "spatial_step_um",
        "grid_type",
        "dt_mode",
        "t_end_s",
        "record",
        "plot_conc",
    ]:
        if key in illingworth_params:
            lines.append(f"  {key}={illingworth_params[key]}")
    illingworth_grid_type = str(illingworth_params.get("grid_type", "constant")).lower()
    if illingworth_grid_type == "geometric":
        for key in ["geometric_ratio", "min_transformed_interval"]:
            if key in illingworth_params:
                lines.append(f"  {key}={illingworth_params[key]}")
    elif "grid_type" in illingworth_params:
        lines.append("  geometric_ratio=N/A")
        lines.append("  min_transformed_interval=N/A")
    grid_metadata = illingworth_params.get("grid_metadata")
    if illingworth_grid_type == "geometric" and isinstance(grid_metadata, dict):
        for key in [
            "phase_a_interface_interval",
            "phase_b_interface_interval",
            "phase_a_far_interval",
            "phase_b_far_interval",
            "min_actual_transformed_interval",
        ]:
            if key in grid_metadata:
                lines.append(f"  {key}={grid_metadata[key]:.8g}")
    if "timestep_label" in illingworth_params:
        lines.append(f"  timestep={illingworth_params['timestep_label']}")
    dt_mode = illingworth_params.get("dt_mode", "fixed")
    if dt_mode == "semi_log":
        for key in ["active_semiLogT0", "active_semiLog_dt", "semiLogT0", "semiLog_dt"]:
            if key in illingworth_params:
                lines.append(f"  {key}={illingworth_params[key]}")
    else:
        for key in ["active_time_step_s", "time_step_s"]:
            if key in illingworth_params:
                lines.append(f"  {key}={illingworth_params[key]}")
    for key in ["mass_integral_initial", "mass_integral_final", "idealized_mass_integral"]:
        if key in illingworth["metadata"]:
            lines.append(f"  {key}={illingworth['metadata'][key]:.8g}")

    return "\n".join(lines)


def _load_time_half_width_csv(path):
    """
    Loads a digitized overlay CSV with ``time_s`` and ``half_width_um`` columns.
    """
    load_path = pathlib.Path(path)
    data = np.genfromtxt(load_path, delimiter=",", names=True, dtype=np.float64)
    if "time_s" not in data.dtype.names or "half_width_um" not in data.dtype.names:
        raise ValueError(f"{load_path} must contain columns: time_s, half_width_um")
    time_s = np.asarray(data["time_s"], dtype=np.float64)
    half_width_um = np.asarray(data["half_width_um"], dtype=np.float64)
    mask = np.isfinite(time_s) & np.isfinite(half_width_um)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"{load_path} must contain at least two finite digitized points.")
    return time_s[mask], half_width_um[mask]


def _load_time_half_width_npz(path):
    """
    Loads an ``.npz`` overlay with ``time_s`` and ``half_width_um`` arrays.
    """
    load_path = pathlib.Path(path)
    with np.load(load_path, allow_pickle=False) as loaded:
        if "time_s" not in loaded or "half_width_um" not in loaded:
            raise ValueError(f"{load_path} must contain arrays: time_s, half_width_um")
        time_s = np.asarray(loaded["time_s"], dtype=np.float64)
        half_width_um = np.asarray(loaded["half_width_um"], dtype=np.float64)
    mask = np.isfinite(time_s) & np.isfinite(half_width_um)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"{load_path} must contain at least two finite overlay points.")
    return time_s[mask], half_width_um[mask]


def plot_saved_planar_comparison(config=None, ax=None):
    """
    Loads one saved Olaye run and one saved Illingworth run and overlays them.

    This is intended for notebook and interactive-window use. Edit
    ``COMPARE_SAVED_PLANAR_RUNS_CONFIG`` near the top of the file or pass a
    config override dict directly.
    """
    cfg = COMPARE_SAVED_PLANAR_RUNS_CONFIG if config is None else {**COMPARE_SAVED_PLANAR_RUNS_CONFIG, **dict(config)}
    if cfg["x_axis"] not in {"linear", "log"}:
        raise ValueError("x_axis must be one of ['linear', 'log'].")

    olaye = _load_saved_run(cfg["olaye_run_path"])
    illingworth = _load_saved_run(cfg["illingworth_run_path"])

    created_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)
    else:
        fig = ax.figure

    ax.plot(
        olaye["time_s"],
        olaye["half_width_um"],
        linewidth=2.0,
        color="tab:blue",
        label=cfg["olaye_label"] or _default_label(olaye, "Olaye saved run"),
        zorder=3,
    )
    ax.plot(
        illingworth["time_s"],
        illingworth["half_width_um"],
        linewidth=2.0,
        linestyle="dotted",
        color="tab:orange",
        label=cfg["illingworth_label"] or _default_label(illingworth, "Illingworth saved run"),
        zorder=3,
    )

    lee_and_oh_overlay = None
    if cfg.get("plot_lee_and_oh_n3013", False):
        lee_and_oh_path = pathlib.Path(cfg["lee_and_oh_n3013_path"])
        if lee_and_oh_path.exists():
            lee_and_oh_time_s, lee_and_oh_half_width_um = _load_time_half_width_npz(lee_and_oh_path)
            ax.plot(
                lee_and_oh_time_s,
                lee_and_oh_half_width_um,
                linewidth=1.2,
                color="black",
                label=cfg["lee_and_oh_n3013_label"],
                zorder=2,
            )
            lee_and_oh_overlay = {
                "path": lee_and_oh_path,
                "time_s": lee_and_oh_time_s,
                "half_width_um": lee_and_oh_half_width_um,
            }
        else:
            print(f"Skipping Lee and Oh overlay because file was not found: {lee_and_oh_path}")

    if cfg.get("plot_figure5_rough", False):
        rough_time_s, rough_half_width_um = _load_time_half_width_csv(cfg["figure5_rough_path"])
        ax.scatter(
            rough_time_s,
            rough_half_width_um,
            s=12,
            marker="o",
            facecolors="none",
            color="tab:gray",
            label='Olaye Fig. 5 "Present Model" rough (digitized)',
            zorder=1,
        )

    if cfg.get("plot_figure5_rough2", False):
        rough2_time_s, rough2_half_width_um = _load_time_half_width_csv(cfg["figure5_rough2_path"])
        ax.scatter(
            rough2_time_s,
            rough2_half_width_um,
            s=12,
            marker="o",
            facecolors="none",
            color="tab:pink",
            label='Olaye Fig. 5 "Present Model" rough2 (digitized)',
            zorder=1,
        )

    if cfg.get("plot_theoretical_max", True) and "theoretical_max_um" in illingworth["metadata"]:
        theoretical_max = illingworth["metadata"]["theoretical_max_um"]
        ax.axhline(
            theoretical_max,
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            label=f"Illingworth theoretical max = {theoretical_max:.1f} um",
            zorder=1,
        )

    x_max = float(max(np.max(olaye["time_s"]), np.max(illingworth["time_s"])))
    if cfg["x_axis"] == "log":
        positive_times = np.concatenate((olaye["time_s"][olaye["time_s"] > 0], illingworth["time_s"][illingworth["time_s"] > 0]))
        if positive_times.size == 0:
            raise ValueError("Log-scale comparison requires at least one positive saved-run time.")
        x_min = float(np.min(positive_times))
        ax.set_xlim(x_min, x_max)
    else:
        x_min = float(min(np.min(olaye["time_s"]), np.min(illingworth["time_s"])))
        if x_max <= x_min:
            pad_x = max(1e-6, abs(x_min) * 1e-3 + 1e-6)
            x_min -= pad_x
            x_max += pad_x
        ax.set_xlim(x_min, x_max)

    ax.set_ylim(0, 24) #ax.set_ylim(12.5, 24)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Liquid half-width (um)")
    ax.set_title("Saved Olaye and Illingworth planar runs")
    ax.set_xscale(cfg["x_axis"])
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    parameter_summary_text = _build_parameter_summary_text(olaye, illingworth)
    if cfg.get("show_parameter_summary", False) and parameter_summary_text.strip():
        ax.text(
            1.02,
            0.98,
            parameter_summary_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )

    out = cfg.get("out")
    if out:
        out_path = pathlib.Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved overlay figure: {out_path}")

    analytical_comparison = None
    if cfg.get("compareToAnalyticalSoln", False):
        analytical_comparison = _plot_analytical_early_time_comparison(olaye, illingworth, cfg)

    if created_figure:
        if cfg.get("show", True):
            plt.show()
        else:
            plt.close(fig)

    return {
        "figure": fig,
        "axes": ax,
        "olaye": olaye,
        "illingworth": illingworth,
        "lee_and_oh_n3013": lee_and_oh_overlay,
        "olaye_params": _parse_params_json(olaye),
        "illingworth_params": _parse_params_json(illingworth),
        "parameter_summary_text": parameter_summary_text,
        "analytical_comparison": analytical_comparison,
        "config": cfg,
    }


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        plot_saved_planar_comparison()
    else:
        plot_saved_planar_comparison()

# %%
