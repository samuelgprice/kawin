"""Probe the pinned fork wheel without source trees or editable-install hooks."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
FORK = PROJECT.parents[1]
OUTPUT = PROJECT / ".verification"
from verify import pinned_fork_commit


def run(args, cwd, timeout=120):
    """Run a bounded packaging command, retaining diagnostics on failure."""
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                            timeout=timeout)
    with (OUTPUT / "fork-build.log").open("a", encoding="utf-8") as log:
        log.write(result.stdout + result.stderr)
    result.check_returncode()


def main():
    """Build a Git snapshot and report import compatibility as a separate gate.

    The pin comes from package metadata, not the frozen reference. An optional
    installed binary slice compares all three modes and nonuniform-grid bulk
    probes with the separately captured reference supplied by verify.py, without
    adding either checkout to the import path. A failed fork
    import is recorded separately; unexpected missing dependencies fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-target", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if bool(args.bridge_target) != bool(args.reference):
        parser.error("--bridge-target and --reference must be supplied together")
    dependency_commit = pinned_fork_commit()
    OUTPUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fork-probe-", dir=OUTPUT) as scratch:
        scratch = Path(scratch)
        archive = scratch / "source.zip"
        run(["git", "archive", "--format=zip", "-o", str(archive), dependency_commit], FORK)
        source = scratch / "source"
        with zipfile.ZipFile(archive) as z:
            z.extractall(source)
        wheels = scratch / "wheels"
        run([sys.executable, "-m", "pip", "wheel", str(source), "--no-deps",
             "--no-build-isolation", "--no-index", "--wheel-dir", str(wheels)],
            scratch)
        wheel, = wheels.glob("*.whl")
        with zipfile.ZipFile(wheel) as z:
            members = z.namelist()
        target = scratch / "installed"
        run([sys.executable, "-m", "pip", "install", str(wheel), "--no-deps",
             "--no-index", "--target", str(target)], scratch)
        # -I -S excludes PYTHONPATH, current-directory imports and .pth/editable
        # hooks. Add only the wheel and dependency directories, without site.main.
        code = """
import importlib.util, json, site, sys, traceback
from pathlib import Path
target = Path(sys.argv[1])
sys.path.insert(0, str(target))
if len(sys.argv) > 2:
    sys.path.insert(0, sys.argv[2])
sys.path.extend(site.getsitepackages())
class NoResearch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'examples', 'debugpy'}:
            raise ModuleNotFoundError('Research/debugger module excluded', name=fullname)
sys.meta_path.insert(0, NoResearch())
import kawin
assert Path(kawin.__file__).resolve() == target / 'kawin/__init__.py'
try:
    import kawin.diffusion
except ModuleNotFoundError as exc:
    traceback.print_exc(file=sys.stderr)
    report = dict(import_ok=False, missing_module=exc.name)
else:
    report = dict(import_ok=True, missing_module=None)
    if len(sys.argv) > 2:
        import numpy as np
        import moving_boundary
        assert Path(moving_boundary.__file__).is_relative_to(Path(sys.argv[2]))
        from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel as Model
        import moving_boundary.solvers.binary as binary
        from moving_boundary.numerics import tridiagonal
        bridge_target = Path(sys.argv[2]).resolve()
        for module in (binary, tridiagonal):
            assert Path(module.__file__).resolve().is_relative_to(bridge_target)
        assert Model._new_concentration_left_planar.__module__ == binary.__name__
        assert Model._new_concentration_right_planar.__module__ == binary.__name__
        spec = importlib.util.spec_from_file_location('binary_cases', sys.argv[5])
        cases = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cases)
        arrays = cases.run_cases(Model, sys.argv[3])
        arrays.update(cases.run_bulk_cases(Model))
        with np.load(sys.argv[4]) as reference:
            assert set(reference.files) == arrays.keys()
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, reference[key], err_msg=key)
        assert 'examples' not in sys.modules
        report.update(binary_parity=True, arrays_compared=len(arrays),
                      binary_modes=3,
                      kernel_accelerated=tridiagonal._solve_illingworth_tridiagonal_numba is not None,
                      history_arrays=sum(k.startswith('binary_') for k in arrays),
                      bulk_branch_arrays=sum(k.startswith('bulk__') for k in arrays),
                      solver_origin='<WHEEL_TARGET>/moving_boundary/solvers/binary.py',
                      kernel_origin='<WHEEL_TARGET>/moving_boundary/numerics/tridiagonal.py',
                      bridge_origin='<WHEEL_TARGET>/moving_boundary/__init__.py')
print(json.dumps(report))
"""
        worker_args = [str(target)]
        if args.bridge_target is not None:
            worker_args.extend([str(args.bridge_target.resolve()),
                                str(PROJECT / "examples/binary_constant.py"),
                                str(args.reference.resolve()),
                                str(PROJECT / "tools/binary_cases.py")])
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", code, *worker_args],
            cwd=scratch, capture_output=True, text=True, timeout=60)
        (OUTPUT / "fork-import.log").write_text(
            result.stdout + result.stderr, encoding="utf-8")
        result.check_returncode()
        report = json.loads(result.stdout)
        report.update(dependency_commit=dependency_commit,
                      wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                      kawin_origin="<FORK_WHEEL_TARGET>/kawin/__init__.py",
                      wheel_contains_examples=any(x.startswith("examples/") for x in members),
                      setup="pinned Git archive wheel; -I -S; no source/editable paths")
        (OUTPUT / "fork-probe.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not report["import_ok"] and report["missing_module"] not in {
                "examples", "examples.debugInPlace"}:
            raise RuntimeError(f"Unexpected fork import failure: {report}")
        print(json.dumps(report))


if __name__ == "__main__":
    main()