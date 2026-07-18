#%%
"""
Notebook-first replica of Illingworth and Golosnoy 2005 Figures 4(a)-6(a).

This example reproduces the planar validation panels only. The paper also
contains spherical panels, but the Python Illingworth rework implementation is
currently planar-only, so this script intentionally omits Figs. 4(b), 5(b), and
6(b).

Edit ``NOTEBOOK_CONFIG`` in a notebook or VS Code Interactive Window, then run
``run_and_plot()`` or one of ``plot_fig4a()``, ``plot_fig5a()``, and
``plot_fig6a()``. The authors' original C++ subroutines are linked into a
generated temporary driver; the checked-in MAP source files are not edited.
"""

from __future__ import annotations

import math
import os
import pathlib
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, special


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
    """
    expected_parts = ("examples", "Illingworth2005", "replicate_illingworth2005_fig4_5_6_planar.py")
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
    """Finds the authors' MAP C++ source directory in notebook runs."""
    candidates = [
        pathlib.Path(script_dir) / "illingworth_MAP_code",
        pathlib.Path(repo_root) / "examples" / "Illingworth2005" / "illingworth_MAP_code",
        pathlib.Path.cwd() / "examples" / "Illingworth2005" / "illingworth_MAP_code",
    ]
    for candidate in candidates:
        if (candidate / "subroutines.cpp").is_file():
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
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder
from kawin.solver import explicitEulerIterator


NOTEBOOK_CONFIG = {
    "figures": ("4", "5", "6"),
    "compiler": None,
    "show": True,
    "out": SCRIPT_DIR / "illingworth2005_fig4_5_6_planar.png",
    "quick": False,
    "cpp_enabled": True,
    "print_summary": True,
    "tight_layout": True,
    "plot_extracted_data": True,
    "figure_data_dir": SCRIPT_DIR / "figureDataExtraction",
    # The Fig. 5(a) digitization is positive where the signed computation
    # ``s_numeric - s_exact`` is negative; plot it with the paper's sign.
    "paper_error_signs": {"5": -1.0, "6": 1.0},
}


VALIDATION_PARAMS = {
    "c_a": 0.4,
    "c_b": 0.2,
    "c_inf": 0.3,
    "R": 1.0,
    "D_a": 1.0,
    "D_b": 1.0,
    "t_init": 1.0e-4,
    "t_abs_end": 1.0e-2,
    "temperature": 1000.0,
    "tolerance": 1.0e-8,
}


FIGURE_DATA_FILES = {
    "4": [
        ("Analytical solution extraction", "fig4a_analyticSoln.csv", "k", "-", None),
        ("Extracted: 10 points regular", "fig4a_10pts_reg.csv", "C0", "None", "s"),
        ("Extracted: 10 points irregular", "fig4a_10pts_irreg.csv", "C1", "None", "o"),
        ("Extracted: 100 points regular", "fig4a_100pts_reg.csv", "C2", "None", "v"),
    ],
    "5": [
        ("Extracted: dt = 4e-5", "fig5a_dt_4Eminus5.csv", "C3", "None", "."),
        ("Extracted: dt = 2e-5", "fig5a_dt_2Eminus5.csv", "C4", "None", "."),
        ("Extracted: dt = 1e-5", "fig5a_dt_1Eminus5.csv", "C5", "None", "."),
    ],
    "6": [
        ("Extracted: 100 points", "fig6a_100pts.csv", "C6", "None", "."),
        ("Extracted: 200 points", "fig6a_200pts.csv", "C7", "None", "."),
        ("Extracted: 400 points", "fig6a_400pts.csv", "C8", "None", "."),
    ],
}


@dataclass(frozen=True)
class PlanarValidationCase:
    """Complete input set for one planar validation curve."""

    figure: str
    label: str
    n_phase_a_nodes: int
    n_phase_b_nodes: int
    time_step: float
    transformed_u_grid: np.ndarray | None = None
    transformed_v_grid: np.ndarray | None = None
    line_style: str = "-"
    marker: str | None = None
    record_interval: int = 1


class ConstantBinaryThermodynamics:
    """Minimal constant-diffusivity thermodynamics interface."""

    def __init__(self, phases, diffusivities):
        self.phases = list(phases)
        self.diffusivities = dict(diffusivities)

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        values = np.atleast_1d(T).astype(np.float64)
        return np.squeeze(np.ones(values.shape, dtype=np.float64) * self.diffusivities[phase])


class PlanarExactProfile:
    """Initial analytical concentration profile sampled by the FD mesh."""

    def __init__(self, t_abs, params=None):
        self.t_abs = float(t_abs)
        self.params = VALIDATION_PARAMS if params is None else {**VALIDATION_PARAMS, **dict(params)}
        self.growth_j = solve_planar_zener_j(self.params)

    def __call__(self, z):
        values = planar_exact_profile(
            np.asarray(z, dtype=np.float64).reshape(-1),
            self.t_abs,
            self.params,
            self.growth_j,
        )
        return values[:, np.newaxis]


def _format_cpp_float(value):
    """Formats a Python float as a C++ double literal with round-trip precision."""
    return format(float(value), ".17g")


def _cpp_array(values):
    """Formats a numeric vector as a C++ initializer list."""
    return ", ".join(_format_cpp_float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1))


def _merge_config(config=None):
    """Returns notebook config merged with user overrides."""
    if config is None:
        return dict(NOTEBOOK_CONFIG)
    return {**NOTEBOOK_CONFIG, **dict(config)}


def _selected_figures(config):
    """Normalizes the configured figure selection."""
    figures = config.get("figures", ("4", "5", "6"))
    if isinstance(figures, str):
        figures = (figures,)
    return tuple(str(fig).lower().replace("fig", "") for fig in figures)


def planar_similarity_integral(eta):
    """Returns the planar Zener similarity integral ``I_1(eta)``."""
    return 0.5 * math.sqrt(math.pi) * special.erfc(eta)


def solve_planar_zener_j(params=None):
    """
    Solves the planar Zener growth constant for the validation composition set.
    """
    p = VALIDATION_PARAMS if params is None else {**VALIDATION_PARAMS, **dict(params)}
    ratio = (p["c_b"] - p["c_inf"]) / (p["c_b"] - p["c_a"])

    def residual(j):
        return 2.0 * j * planar_similarity_integral(j) * math.exp(j * j) - ratio

    return float(optimize.brentq(residual, 1.0e-14, 20.0, xtol=1e-14, rtol=1e-14))


def planar_exact_interface(t_abs, params=None, j=None):
    """Returns the exact planar interface position at absolute time ``t_abs``."""
    p = VALIDATION_PARAMS if params is None else {**VALIDATION_PARAMS, **dict(params)}
    growth_j = solve_planar_zener_j(p) if j is None else float(j)
    t = np.asarray(t_abs, dtype=np.float64)
    return growth_j * np.sqrt(4.0 * p["D_b"] * t)


def planar_exact_profile(z, t_abs, params=None, j=None):
    """
    Returns the analytical planar concentration profile used for initialization.
    """
    p = VALIDATION_PARAMS if params is None else {**VALIDATION_PARAMS, **dict(params)}
    growth_j = solve_planar_zener_j(p) if j is None else float(j)
    z_arr = np.asarray(z, dtype=np.float64)
    eta = z_arr / math.sqrt(4.0 * p["D_b"] * float(t_abs))
    phase_b = p["c_inf"] + (p["c_b"] - p["c_inf"]) * special.erfc(eta) / special.erfc(growth_j)
    interface = planar_exact_interface(t_abs, p, growth_j)
    return np.where(z_arr <= interface, p["c_a"], phase_b)


def _clustered_v_grid(n_nodes):
    """
    Returns a deterministic interface-clustered phase-B grid.

    The paper does not state the exact irregular planar mesh. This uses
    ``v = x**2`` so the nodes are denser near the phase boundary at ``v=0``.
    """
    x = np.linspace(0.0, 1.0, int(n_nodes), dtype=np.float64)
    return x * x


def _regular_grid(n_nodes):
    """Returns a regular transformed grid over ``[0, 1]``."""
    return np.linspace(0.0, 1.0, int(n_nodes), dtype=np.float64)


def _base_cases(quick=False):
    """Builds the configured planar validation cases."""
    if quick:
        return [
            PlanarValidationCase("4", "10 points regular", 3, 10, 1.0e-4, line_style="--", marker="s", record_interval=1),
            PlanarValidationCase("4", "10 points irregular", 3, 10, 1.0e-4, transformed_v_grid=_clustered_v_grid(10), line_style="-.", marker="o", record_interval=1),
            PlanarValidationCase("5", "dt = 4e-5", 3, 80, 4.0e-5, line_style=":", record_interval=1),
            PlanarValidationCase("6", "100 points", 3, 40, 1.0e-5, line_style=":", record_interval=1),
        ]

    return [
        PlanarValidationCase("4", "10 points regular", 3, 10, 1.0e-4, line_style="--", marker="s", record_interval=1),
        PlanarValidationCase("4", "10 points irregular", 3, 10, 1.0e-4, transformed_v_grid=_clustered_v_grid(10), line_style="-.", marker="o", record_interval=1),
        PlanarValidationCase("4", "100 points regular", 3, 100, 1.0e-7, line_style=":", marker="v", record_interval=100),
        PlanarValidationCase("5", "dt = 4e-5", 3, 10000, 4.0e-5, line_style=":", record_interval=1),
        PlanarValidationCase("5", "dt = 2e-5", 3, 10000, 2.0e-5, line_style="--", record_interval=2),
        PlanarValidationCase("5", "dt = 1e-5", 3, 10000, 1.0e-5, line_style="-", record_interval=4),
        PlanarValidationCase("6", "100 points", 3, 100, 1.0e-7, line_style=":", record_interval=100),
        PlanarValidationCase("6", "200 points", 3, 200, 1.0e-7, line_style="--", record_interval=100),
        PlanarValidationCase("6", "400 points", 3, 400, 1.0e-7, line_style="-", record_interval=100),
    ]


def _cases_for_config(config):
    """Returns the selected planar validation cases."""
    selected = set(_selected_figures(config))
    cases = [case for case in _base_cases(bool(config.get("quick", False))) if case.figure in selected or "all" in selected]
    overrides = config.get("case_overrides", {})
    if overrides:
        updated = []
        for case in cases:
            patch = overrides.get(case.label, overrides.get(case.figure, {}))
            updated.append(replace(case, **patch) if patch else case)
        cases = updated
    return cases


def _time_bounds(config):
    """Returns initial and final absolute times for the current run."""
    t_init = float(config.get("t_init", VALIDATION_PARAMS["t_init"]))
    if config.get("quick", False):
        t_abs_end = float(config.get("t_abs_end", 7.0e-4))
    else:
        t_abs_end = float(config.get("t_abs_end", VALIDATION_PARAMS["t_abs_end"]))
    if t_abs_end <= t_init:
        raise ValueError("t_abs_end must be greater than t_init.")
    return t_init, t_abs_end


def _estimate_mesh_nodes(case, t_init, t_abs_end, max_nodes=20001):
    """Chooses a physical mesh dense enough to sample the analytical profile."""
    base = max(1001, 4 * int(case.n_phase_b_nodes), int(case.n_phase_a_nodes + case.n_phase_b_nodes))
    return min(int(max_nodes), int(base))


def build_python_planar_validation_model(case, config=None):
    """
    Builds a Python rework model for one planar validation case.
    """
    cfg = _merge_config(config)
    p = {**VALIDATION_PARAMS, **dict(cfg.get("validation_params", {}))}
    t_init, _ = _time_bounds(cfg)
    s0 = float(planar_exact_interface(t_init, p))
    mesh_nodes = int(cfg.get("mesh_nodes", _estimate_mesh_nodes(case, t_init, cfg.get("t_abs_end", VALIDATION_PARAMS["t_abs_end"]))))

    profile = ProfileBuilder([(PlanarExactProfile(t_init, p), "CR")])
    mesh = CartesianFD1D(["CR"], [0.0, p["R"]], mesh_nodes)
    mesh.setResponseProfile(profile)
    therm = ConstantBinaryThermodynamics(
        phases=["ALPHA", "BETA"],
        diffusivities={"ALPHA": p["D_a"], "BETA": p["D_b"]},
    )
    return MovingBoundaryIllingworthFD1DModel(
        mesh,
        ["NI", "CR"],
        ["ALPHA", "BETA"],
        thermodynamics=therm,
        temperature=TemperatureParameters(p["temperature"]),
        interfacePosition=s0,
        interface_compositions=(p["c_a"], p["c_b"]),
        time_step=case.time_step,
        phase_a_nodes=case.n_phase_a_nodes,
        phase_b_nodes=case.n_phase_b_nodes,
        tolerance=p["tolerance"],
        max_iterations=int(cfg.get("max_iterations", 100)),
        record=case.record_interval,
        record_pq_data=False,
        preallocate_recordings=bool(cfg.get("preallocate_recordings", True)),
        transformed_u_grid=case.transformed_u_grid,
        transformed_v_grid=case.transformed_v_grid,
    )


def run_python_planar_case(case, config=None):
    """
    Runs one planar validation case with the Python rework implementation.
    """
    cfg = _merge_config(config)
    p = {**VALIDATION_PARAMS, **dict(cfg.get("validation_params", {}))}
    t_init, t_abs_end = _time_bounds(cfg)
    elapsed = t_abs_end - t_init
    model = build_python_planar_validation_model(case, cfg)
    start = time.perf_counter()
    model.solve(elapsed, iterator=explicitEulerIterator, minDtFrac=1e-15, verbose=False)
    runtime_s = time.perf_counter() - start
    n = model.interfaceData.N + 1
    elapsed_time = model.interfaceData._time[:n].copy()
    interface = model.interfaceData._y[:n].copy()
    absolute_time = t_init + elapsed_time
    exact = planar_exact_interface(absolute_time, p)
    return {
        "time_elapsed": elapsed_time,
        "time_abs": absolute_time,
        "interface": interface,
        "exact_interface": exact,
        "error": interface - exact,
        "runtime_s": runtime_s,
        "case": case,
    }


def _case_cpp_payload(case, config=None):
    """Builds scalar and array inputs for the generated C++ driver."""
    cfg = _merge_config(config)
    p = {**VALIDATION_PARAMS, **dict(cfg.get("validation_params", {}))}
    t_init, t_abs_end = _time_bounds(cfg)
    s0 = float(planar_exact_interface(t_init, p))
    elapsed_end = t_abs_end - t_init
    n_steps = int(math.ceil(elapsed_end / case.time_step))

    u_grid = _regular_grid(case.n_phase_a_nodes) if case.transformed_u_grid is None else np.asarray(case.transformed_u_grid, dtype=np.float64)
    v_grid = _regular_grid(case.n_phase_b_nodes) if case.transformed_v_grid is None else np.asarray(case.transformed_v_grid, dtype=np.float64)
    z_right = s0 + (p["R"] - s0) * v_grid
    left_c = np.ones(case.n_phase_a_nodes, dtype=np.float64) * p["c_a"]
    right_c = planar_exact_profile(z_right, t_init, p)
    left_c[-1] = p["c_a"]
    right_c[0] = p["c_b"]

    return {
        "R": p["R"],
        "s0": s0,
        "d_alpha": p["D_a"],
        "d_beta": p["D_b"],
        "interface_alpha": p["c_a"],
        "interface_beta": p["c_b"],
        "time_step": case.time_step,
        "elapsed_end": elapsed_end,
        "n_time_steps": n_steps,
        "tolerance": p["tolerance"],
        "u_grid": u_grid,
        "v_grid": v_grid,
        "left_c": left_c,
        "right_c": right_c,
    }


def _render_author_cpp_planar_driver(payload):
    """Renders a generated C++ driver for one planar validation case."""
    p = payload
    return f"""#include <stdio.h>
#include <stdlib.h>
#include "data_structures.h"
#include "subroutines.h"
#include "InOut.h"

int main(void)
{{
    const double s_0 = {_format_cpp_float(p["s0"])};
    const double R = {_format_cpp_float(p["R"])};
    const int nAlpha = {len(p["u_grid"])};
    const int nBeta = {len(p["v_grid"])};
    const double dAlpha = {_format_cpp_float(p["d_alpha"])};
    const double dBeta = {_format_cpp_float(p["d_beta"])};
    const double interAlpha = {_format_cpp_float(p["interface_alpha"])};
    const double interBeta = {_format_cpp_float(p["interface_beta"])};
    const double time_step = {_format_cpp_float(p["time_step"])};
    const double elapsed_end = {_format_cpp_float(p["elapsed_end"])};
    const int n_time_steps = {int(p["n_time_steps"])};
    const double tol = {_format_cpp_float(p["tolerance"])};
    const double left_u[nAlpha] = {{{_cpp_array(p["u_grid"])}}};
    const double right_u[nBeta] = {{{_cpp_array(p["v_grid"])}}};
    const double left_c0[nAlpha] = {{{_cpp_array(p["left_c"])}}};
    const double right_c0[nBeta] = {{{_cpp_array(p["right_c"])}}};

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
    for (i = 0; i < whole_system->left->n; i++)
    {{
        whole_system->left->u[i] = left_u[i];
        whole_system->left->c[i] = left_c0[i];
        whole_system->left->future_c[i] = left_c0[i];
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
    for (i = 0; i < whole_system->right->n; i++)
    {{
        whole_system->right->u[i] = right_u[i];
        whole_system->right->c[i] = right_c0[i];
        whole_system->right->future_c[i] = right_c0[i];
    }}
    whole_system->right->c[0] = whole_system->right->c_boundary;
    whole_system->right->future_c[0] = whole_system->right->c_boundary;

    FILE *fpt = fopen("results.txt", "w");
    fprintf(fpt, "Time\\tInterface Position\\n");
    double elapsed = 0.0;
    for (i = 0; i < n_time_steps + 1; i++)
    {{
        out_interface(whole_system, elapsed, fpt);
        if (i < n_time_steps)
        {{
            double step_dt = time_step;
            if (elapsed + step_dt > elapsed_end)
            {{
                step_dt = elapsed_end - elapsed;
            }}
            tmp = take_step_planar(whole_system, step_dt, tol);
            if (tmp < 0)
            {{
                fclose(fpt);
                return 2;
            }}
            elapsed += step_dt;
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


def parse_results(path):
    """Parses the authors' generated ``results.txt`` interface output."""
    data = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"No rows found in {path}.")
    data = np.atleast_2d(data)
    return np.asarray(data[:, 0], dtype=np.float64), np.asarray(data[:, 1], dtype=np.float64)


def compile_and_run_authors_cpp_planar_case(case, config=None, source_dir=None, build_dir=None):
    """
    Compiles and runs the authors' original planar subroutines for one case.
    """
    cfg = _merge_config(config)
    compiler = cfg.get("compiler") or shutil.which("g++")
    if compiler is None:
        raise FileNotFoundError("Could not find g++ on PATH.")
    source_dir = pathlib.Path(source_dir) if source_dir is not None else AUTHOR_CPP_SOURCE_DIR
    compiler_path = pathlib.Path(compiler)
    compiler_dir = compiler_path.parent if compiler_path.parent != pathlib.Path(".") else None

    cleanup = False
    if build_dir is None:
        temp_root = REPO_ROOT / ".pytest_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        build_dir = temp_root / f"illingworth_fig456_cpp_{uuid.uuid4().hex}"
        build_dir.mkdir(parents=True, exist_ok=False)
        cleanup = True
    else:
        build_dir = pathlib.Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

    payload = _case_cpp_payload(case, cfg)
    driver_path = build_dir / "generated_illingworth_fig456_driver.cpp"
    exe = build_dir / f"illingworth_fig456_{uuid.uuid4().hex}.exe"
    sources = [
        driver_path,
        source_dir / "subroutines.cpp",
        source_dir / "trimatrix.cpp",
        source_dir / "InOut.cpp",
    ]
    try:
        driver_path.write_text(_render_author_cpp_planar_driver(payload), encoding="utf-8")
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
        elapsed_time, interface = parse_results(results_path)
        t_init, _ = _time_bounds(cfg)
        absolute_time = t_init + elapsed_time
        exact = planar_exact_interface(absolute_time, {**VALIDATION_PARAMS, **dict(cfg.get("validation_params", {}))})
        return {
            "time_elapsed": elapsed_time,
            "time_abs": absolute_time,
            "interface": interface,
            "exact_interface": exact,
            "error": interface - exact,
            "runtime_s": run_runtime_s,
            "compile_runtime_s": compile_runtime_s,
            "case": case,
        }
    finally:
        if cleanup:
            shutil.rmtree(build_dir, ignore_errors=True)


def _compare_python_cpp_at_python_times(python_result, cpp_result):
    """Computes C++/Python interface differences at Python recorded times."""
    py_time = np.asarray(python_result["time_elapsed"], dtype=np.float64)
    cpp_time = np.asarray(cpp_result["time_elapsed"], dtype=np.float64)
    cpp_interface = np.asarray(cpp_result["interface"], dtype=np.float64)
    matched_cpp = np.interp(py_time, cpp_time, cpp_interface)
    diff = np.asarray(python_result["interface"], dtype=np.float64) - matched_cpp
    return {
        "matched_cpp_interface": matched_cpp,
        "python_minus_cpp": diff,
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


def run_planar_validation_cases(config=None):
    """
    Runs all selected planar validation cases and returns inspectable results.
    """
    cfg = _merge_config(config)
    results = {"config": cfg, "cases": []}
    for case in _cases_for_config(cfg):
        print(f"Running Fig. {case.figure}(a), {case.label} with Python rework...")
        py_result = run_python_planar_case(case, cfg)
        item = {"case": case, "python": py_result}
        if cfg.get("cpp_enabled", True):
            print(f"Running Fig. {case.figure}(a), {case.label} with authors' C++...")
            cpp_result = compile_and_run_authors_cpp_planar_case(case, cfg)
            item["cpp"] = cpp_result
            item["python_cpp_comparison"] = _compare_python_cpp_at_python_times(py_result, cpp_result)
        results["cases"].append(item)
    if cfg.get("print_summary", True):
        print_planar_validation_summary(results)
    return results


def print_planar_validation_summary(results):
    """Prints a compact notebook-friendly run summary."""
    for item in results["cases"]:
        case = item["case"]
        py = item["python"]
        msg = f"Fig. {case.figure}(a) {case.label}: Python {py['runtime_s']:.4g} s"
        if "cpp" in item:
            cpp = item["cpp"]
            comp = item["python_cpp_comparison"]
            msg += f", C++ run {cpp['runtime_s']:.4g} s, max |Python-C++| {comp['max_abs_diff']:.3e}"
        print(msg)


def _figure_axes(figures, axes=None):
    """Creates or normalizes axes for the requested figure panels."""
    figures = tuple(figures)
    if axes is not None:
        if isinstance(axes, dict):
            return axes, None
        axes_arr = np.atleast_1d(axes).reshape(-1)
        return {fig: ax for fig, ax in zip(figures, axes_arr)}, None

    fig, axes_arr = plt.subplots(len(figures), 1, figsize=(7.2, 4.1 * len(figures)), dpi=140, squeeze=False)
    return {fig_name: ax for fig_name, ax in zip(figures, axes_arr[:, 0])}, fig


def _style_index(label):
    """Returns a stable color index for repeated Python/C++ overlays."""
    labels = [
        "10 points regular",
        "10 points irregular",
        "100 points regular",
        "dt = 4e-5",
        "dt = 2e-5",
        "dt = 1e-5",
        "100 points",
        "200 points",
        "400 points",
    ]
    return labels.index(label) if label in labels else 0


def _paper_error_sign(config, figure):
    """Returns the sign needed to match the paper's plotted error convention."""
    signs = config.get("paper_error_signs", {})
    return float(signs.get(str(figure), 1.0))


def load_extracted_figure_data(path):
    """
    Loads a no-header WebPlotDigitizer CSV as plotted ``x`` and ``y`` arrays.
    """
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns.")
    mask = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    return data[mask, 0], data[mask, 1]


def plot_extracted_figure_data(config, axes_by_fig):
    """
    Overlays digitized paper data from ``figureDataExtraction`` on each panel.
    """
    if not config.get("plot_extracted_data", True):
        return {}
    data_dir = pathlib.Path(config.get("figure_data_dir", SCRIPT_DIR / "figureDataExtraction"))
    plotted = {}
    for fig_id, ax in axes_by_fig.items():
        entries = []
        for label, filename, color, linestyle, marker in FIGURE_DATA_FILES.get(fig_id, []):
            path = data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Extracted figure data file does not exist: {path}")
            x, y = load_extracted_figure_data(path)
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                mfc='none',
                linewidth=1.0 if linestyle != "None" else 0.0,
                alpha=0.75,
                label=label,
                zorder=2,
            )
            entries.append({"label": label, "path": path, "x": x, "y": y})
        plotted[fig_id] = entries
    return plotted


def plot_planar_validation_results(results, config=None, axes=None):
    """
    Plots selected planar validation results and returns ``(results, axes)``.
    """
    cfg = _merge_config(results.get("config", config))
    figures = _selected_figures(cfg)
    if "all" in figures:
        figures = ("4", "5", "6")
    axes_by_fig, created_fig = _figure_axes(figures, axes)
    p = {**VALIDATION_PARAMS, **dict(cfg.get("validation_params", {}))}
    t_init, t_abs_end = _time_bounds(cfg)
    exact_t = np.linspace(t_init, t_abs_end, 400)
    exact_s = planar_exact_interface(exact_t, p)

    for fig_id, ax in axes_by_fig.items():
        if fig_id == "4":
            ax.plot(np.sqrt(p["D_b"] * exact_t), exact_s, color="k", lw=2.0, label="Analytical solution")
            ax.set_xlabel(r"$(D_B t)^{1/2}$")
            ax.set_ylabel("Interface position, s(t)")
            ax.set_title("Fig. 4(a): planar interface position")
        else:
            ax.plot([p["D_b"] * t_init, p["D_b"] * t_abs_end], [0.0, 0.0], color="k", lw=1.0, label="Analytical solution")
            ax.set_xlabel(r"$D_B t$")
            ax.set_ylabel(r"Error, $s(t)_{numerical} - s(t)_{exact}$")
            ax.set_title(f"Fig. {fig_id}(a): planar accuracy")
        ax.grid(True, alpha=0.25)

    for item in results["cases"]:
        case = item["case"]
        ax = axes_by_fig.get(case.figure)
        if ax is None:
            continue
        color = f"C{_style_index(case.label) % 10}"
        x_python = np.sqrt(p["D_b"] * item["python"]["time_abs"]) if case.figure == "4" else p["D_b"] * item["python"]["time_abs"]
        y_python = (
            item["python"]["interface"]
            if case.figure == "4"
            else _paper_error_sign(cfg, case.figure) * item["python"]["error"]
        )
        ax.plot(
            x_python,
            y_python,
            color=color,
            linestyle=case.line_style,
            marker=case.marker,
            markevery=max(1, len(x_python) // 12),
            ms=4,
            lw=1.4,
            label=f"Python rework: {case.label}",
        )
        if "cpp" in item:
            x_cpp = np.sqrt(p["D_b"] * item["cpp"]["time_abs"]) if case.figure == "4" else p["D_b"] * item["cpp"]["time_abs"]
            y_cpp = (
                item["cpp"]["interface"]
                if case.figure == "4"
                else _paper_error_sign(cfg, case.figure) * item["cpp"]["error"]
            )
            ax.plot(
                x_cpp,
                y_cpp,
                color=color,
                linestyle="None",
                marker=".",
                ms=2,
                alpha=0.45,
                label=f"Authors' C++: {case.label}",
            )

    results["extracted_data"] = plot_extracted_figure_data(cfg, axes_by_fig)

    for ax in axes_by_fig.values():
        ax.legend(fontsize=7)

    if created_fig is not None:
        if cfg.get("tight_layout", True):
            created_fig.tight_layout()
        out = cfg.get("out")
        if out:
            pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            created_fig.savefig(out, bbox_inches="tight")
            print(f"Saved figure: {out}")
        if cfg.get("show", True):
            plt.show()
        else:
            plt.close(created_fig)
    return results, axes_by_fig


def run_and_plot(config=None):
    """
    Runs selected planar cases and plots them for notebook inspection.
    """
    cfg = _merge_config(config)
    results = run_planar_validation_cases(cfg)
    return plot_planar_validation_results(results, cfg)


def run_from_config(config=None):
    """Notebook entry point that runs and plots using editable configuration."""
    return run_and_plot(NOTEBOOK_CONFIG if config is None else config)


def plot_fig4a(config=None, ax=None):
    """Runs and plots only the planar Fig. 4(a) cases."""
    cfg = {**_merge_config(config), "figures": ("4",)}
    results = run_planar_validation_cases(cfg)
    _, axes = plot_planar_validation_results(results, cfg, axes=ax)
    return results, axes["4"]


def plot_fig5a(config=None, ax=None):
    """Runs and plots only the planar Fig. 5(a) cases."""
    cfg = {**_merge_config(config), "figures": ("5",)}
    results = run_planar_validation_cases(cfg)
    _, axes = plot_planar_validation_results(results, cfg, axes=ax)
    return results, axes["5"]


def plot_fig6a(config=None, ax=None):
    """Runs and plots only the planar Fig. 6(a) cases."""
    cfg = {**_merge_config(config), "figures": ("6",)}
    results = run_planar_validation_cases(cfg)
    _, axes = plot_planar_validation_results(results, cfg, axes=ax)
    return results, axes["6"]


if __name__ == "__main__":
    results, axes = run_and_plot(NOTEBOOK_CONFIG)

# %%
