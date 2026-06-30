from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kawin.diffusion import MovingBoundaryIllingworthFD1DModel, TemperatureParameters
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.solver import explicitEulerIterator


AUTHOR_DEFAULT_PARAMS = {
    "s0": 1.0,
    "R": 5.0,
    "n_alpha": 100,
    "d_alpha": 1.0e-7,
    "initial_alpha": 0.8,
    "interface_alpha": 0.6,
    "n_beta": 100,
    "d_beta": 1.0e-5,
    "initial_beta": 0.4,
    "interface_beta": 0.0,
    "time_step": 0.1,
    "n_time_steps": 10,
    "tolerance": 1.0e-8,
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
    model.solve(p["n_time_steps"] * p["time_step"], iterator=explicitEulerIterator, minDtFrac=1e-14)
    n = model.interfaceData.N + 1
    return model.interfaceData._time[:n].copy(), model.interfaceData._y[:n].copy()


def parse_results(path):
    """Parses the authors' ``results.txt`` interface-position output."""
    data = np.genfromtxt(path, names=True, dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"No rows found in {path}.")
    return np.asarray(data["Time"], dtype=np.float64), np.asarray(data["Interface_Position"], dtype=np.float64)


def compile_and_run_authors_cpp(source_dir=None, build_dir=None, compiler=None):
    """
    Compiles and runs the untouched MAP C++ default case.

    Build products are written outside the authors' source directory. If
    ``build_dir`` is omitted, a temporary directory is used and cleaned up with
    ``ignore_errors=True`` to avoid Windows file-lock failures after compiler
    errors.
    """
    source_dir = pathlib.Path(source_dir) if source_dir is not None else pathlib.Path(__file__).resolve().parent / "illingworth_MAP_code"
    compiler = compiler or shutil.which("g++")
    if compiler is None:
        raise FileNotFoundError("Could not find g++ on PATH.")

    cleanup = False
    if build_dir is None:
        build_dir = pathlib.Path(tempfile.mkdtemp(prefix="illingworth_cpp_"))
        cleanup = True
    else:
        build_dir = pathlib.Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

    exe = build_dir / "ConservativeIFF.exe"
    sources = [
        source_dir / "ConservativeIFF.cpp",
        source_dir / "subroutines.cpp",
        source_dir / "trimatrix.cpp",
        source_dir / "InOut.cpp",
    ]
    try:
        compile_cmd = [compiler, *[str(s) for s in sources], "-o", str(exe)]
        compile_result = subprocess.run(compile_cmd, text=True, capture_output=True)
        if compile_result.returncode != 0:
            raise RuntimeError(
                "Authors' C++ compile failed with exit code "
                f"{compile_result.returncode}.\nSTDOUT:\n{compile_result.stdout}\nSTDERR:\n{compile_result.stderr}"
            )

        run_result = subprocess.run([str(exe)], cwd=build_dir, text=True, capture_output=True)
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


def compare_default_case():
    """Runs both implementations and returns comparison arrays and error metrics."""
    cpp_time, cpp_s = compile_and_run_authors_cpp()
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


def build_parser():
    parser = argparse.ArgumentParser(description="Compare the Python Illingworth planar model to the authors' C++ code.")
    parser.add_argument("--compiler", default=None, help="Path to g++; defaults to PATH lookup.")
    parser.add_argument("--build-dir", default=None, help="Optional directory for C++ build products.")
    return parser


def main():
    args = build_parser().parse_args()
    cpp_time, cpp_s = compile_and_run_authors_cpp(build_dir=args.build_dir, compiler=args.compiler)
    py_time, py_s = run_python_default()
    if not np.allclose(cpp_time, py_time, rtol=0.0, atol=1e-12):
        raise ValueError("C++ and Python output times do not match.")
    abs_diff = np.abs(cpp_s - py_s)
    rel_diff = abs_diff / np.maximum(np.abs(cpp_s), 1e-300)
    print(f"Rows compared: {len(cpp_time)}")
    print(f"Max absolute interface-position difference: {np.max(abs_diff):.16e}")
    print(f"Max relative interface-position difference: {np.max(rel_diff):.16e}")


if __name__ == "__main__":
    main()
