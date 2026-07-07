#%%
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
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
    "spatial_step_um": 0.5,
    # The text mentions a 0.01 s time step for the comparison setup. That is
    # very expensive in pure Python out to 1e5 s, so the default here is a
    # runtime-friendly value. Set this to 0.01 for the literal paper timestep.
    "time_step_s": 0.01,
    "paper_time_step_s": 0.01,
    "t_end_s": 1.0e2,
    "record": 1,
    "plot_conc": True,
    "show": True,
    "out": None,
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
    model.solve(p["n_time_steps"] * p["time_step"], iterator=explicitEulerIterator, minDtFrac=1e-14, verbose=True)
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


def _jsonable(value):
    """Converts notebook config values into JSON-serializable objects."""
    if isinstance(value, pathlib.Path):
        return str(value)
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
    with the saved Olaye Figure-5 output.
    """
    params = result["params"]
    payload_params = {
        key: value
        for key, value in params.items()
        if key not in {"show", "out", "save_run", "save_run_path", "label"}
    }
    return {
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


def save_illingworth_run_result(path, payload):
    """Saves an Illingworth Figure-3 run payload to a compressed ``.npz`` file."""
    save_path = pathlib.Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **payload)
    return save_path


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

    n_mesh = int(round(p["R_um"] / p["spatial_step_um"])) + 1
    phase_a_nodes = int(round(p["s0_um"] / p["spatial_step_um"])) + 1
    phase_b_nodes = int(round((p["R_um"] - p["s0_um"]) / p["spatial_step_um"])) + 1

    profile = ProfileBuilder([(StepProfile1D(p["s0_um"], c_liquid0, c_solid0), "P")])
    mesh = CartesianFD1D(["P"], [0.0, p["R_um"]], n_mesh)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=["LIQUID", "SOLID"],
        diffusivities={"LIQUID": p["D_liquid_um2_s"], "SOLID": p["D_solid_um2_s"]},
    )
    return MovingBoundaryIllingworthFD1DModel(
        mesh,
        ["NI", "P"],
        ["LIQUID", "SOLID"],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=p["s0_um"],
        interface_compositions=(c_liquid_int, c_solid_int),
        time_step=p["time_step_s"],
        phase_a_nodes=phase_a_nodes,
        phase_b_nodes=phase_b_nodes,
        tolerance=1.0e-8,
        record=record,
    )


def run_fig3_present_work(params=None):
    """Runs and returns data for the Figure 3 present-work liquid half-width curve."""
    p = FIG3_PRESENT_WORK_PARAMS if params is None else {**FIG3_PRESENT_WORK_PARAMS, **dict(params)}
    n_steps = int(np.ceil(p["t_end_s"] / p["time_step_s"]))
    if n_steps > 250_000:
        raise ValueError(
            f"Figure 3 run would require about {n_steps} Python implicit steps. "
            "Increase FIG3_PRESENT_WORK_PARAMS['time_step_s'] for exploratory plotting, "
            "or run a shorter t_end_s."
        )

    model = build_fig3_present_work_model(p, record=p["record"])
    model.solve(p["t_end_s"], iterator=explicitEulerIterator, minDtFrac=1e-14, verbose=True)
    # debugInPlace()
    n = model.interfaceData.N + 1
    time_s = model.interfaceData._time[:n].copy()
    liquid_half_width_um = model.interfaceData._y[:n].copy()
    return {
        "time_s": time_s,
        "liquid_half_width_um": liquid_half_width_um,
        "theoretical_max_um": theoretical_fig3_max_liquid_half_width_um(p),
        "model": model,
        "params": p,
    }


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
    ax.set_ylim(12.5, 24) #ax.set_ylim(0.0, 30.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Interface position / liquid half-width (um)")
    ax.set_title(
        "Illingworth and Golosnoy 2005 Fig. 3 present-work curve\n"
        f"dt={p['time_step_s']} s, transformed step~{p['spatial_step_um']} um"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

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
            ax_twin.legend(loc="center right")

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


def print_comparison_summary(comparison, print_table=False):
    """Prints a compact comparison summary suitable for scripts or notebooks."""
    print(f"Rows compared: {len(comparison['time'])}")
    print(f"Max absolute interface-position difference: {comparison['max_abs_diff']:.16e}")
    print(f"Max relative interface-position difference: {comparison['max_rel_diff']:.16e}")
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
        main()

 # %%
