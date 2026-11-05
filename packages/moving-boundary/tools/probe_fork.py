"""Probe the pinned fork wheel without source trees or editable-install hooks."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
FORK = PROJECT.parents[1]
OUTPUT = PROJECT / ".verification"
BASELINE = "ae9fb2971a9fd49132f71ddc883a5bd5406644b6"


def run(args, cwd, timeout=120):
    """Run a bounded packaging command, retaining diagnostics on failure."""
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                            timeout=timeout)
    with (OUTPUT / "fork-build.log").open("a", encoding="utf-8") as log:
        log.write(result.stdout + result.stderr)
    result.check_returncode()


def main():
    """Build a Git snapshot and report import compatibility as a separate gate.

    A failed fork import is recorded, not confused with a successful source
    bridge test. Unexpected missing dependencies fail this diagnostic.
    """
    OUTPUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fork-probe-", dir=OUTPUT) as scratch:
        scratch = Path(scratch)
        archive = scratch / "source.zip"
        run(["git", "archive", "--format=zip", "-o", str(archive), BASELINE], FORK)
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
import json, site, sys, traceback
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.extend(site.getsitepackages())
import kawin
assert Path(kawin.__file__).resolve() == Path(sys.argv[1]) / 'kawin/__init__.py'
try:
    import kawin.diffusion
except ModuleNotFoundError as exc:
    traceback.print_exc(file=sys.stderr)
    print(json.dumps(dict(import_ok=False, missing_module=exc.name)))
else:
    print(json.dumps(dict(import_ok=True, missing_module=None)))
"""
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", code, str(target)],
            cwd=scratch, capture_output=True, text=True, timeout=60)
        (OUTPUT / "fork-import.log").write_text(
            result.stdout + result.stderr, encoding="utf-8")
        result.check_returncode()
        report = json.loads(result.stdout)
        report.update(baseline_commit=BASELINE,
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