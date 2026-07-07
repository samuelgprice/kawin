#%%
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np


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
    for key in ["spatial_step_um", "time_step_s", "t_end_s", "record", "plot_conc"]:
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
        zorder=2,
    )
    ax.plot(
        illingworth["time_s"],
        illingworth["half_width_um"],
        linewidth=2.0,
        linestyle="dotted",
        color="tab:orange",
        label=cfg["illingworth_label"] or _default_label(illingworth, "Illingworth saved run"),
        zorder=2,
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

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Liquid half-width (um)")
    ax.set_title("Saved Olaye and Illingworth planar runs")
    ax.set_xscale(cfg["x_axis"])
    ax.grid(True, alpha=0.25)
    ax.legend()

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
        "config": cfg,
    }


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        plot_saved_planar_comparison()
    else:
        plot_saved_planar_comparison()

# %%
