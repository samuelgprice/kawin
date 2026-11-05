"""Bounded build/install/parity verification without modifying shared packages."""

from pathlib import Path
import argparse
from email.parser import Parser
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import re
import time
import zipfile
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
FORK = PROJECT.parents[1]
OUTPUT = PROJECT / ".verification"
EXPECTED = "ae9fb2971a9fd49132f71ddc883a5bd5406644b6"


def pinned_fork_commit():
    """Read the immutable dependency pin independently of the numerical baseline."""
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [r for r in project["project"]["dependencies"] if r.startswith("kawin @ ")]
    if len(requirements) != 1:
        raise ValueError("Expected one explicit kawin dependency")
    commit = requirements[0].rsplit("@", 1)[-1]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("The fork dependency must use a full immutable commit")
    return commit


def run(label, args, *, cwd, env=None, timeout=600):
    """Run a batch subprocess with a wall-time limit and retain its full log."""
    started = time.monotonic()
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True,
                            text=True, timeout=timeout)
    (OUTPUT / f"{label}.log").write_text(
        result.stdout + result.stderr, encoding="utf-8")
    print(f"{label}: exit={result.returncode}, {time.monotonic()-started:.2f}s",
          flush=True)
    if result.returncode:
        raise RuntimeError(f"{label} failed: {result.stdout}\n{result.stderr}")
    return result.stdout


def check_baseline(root, expected):
    """Reject tracked baseline changes and untracked Python shadowing.

    Package-only commits are allowed. The reference source, examples required by
    its imports, packaging inputs and fixtures remain frozen independently of HEAD.
    """
    # Compare content, not branch position: package-only commits are permitted.
    run("baseline-content", ["git", "diff", "--exit-code", expected, "--",
                            "kawin", "examples", "pyproject.toml", "MANIFEST.in",
                            "LICENSE.txt"], cwd=root, timeout=30)
    untracked = run("baseline-untracked", ["git", "ls-files", "--others",
                    "--exclude-standard", "--", "kawin"], cwd=root, timeout=30)
    if untracked.strip():
        raise RuntimeError("Untracked kawin files could shadow the frozen baseline")


def finish(report, strict=False):
    """Print both gates; strict installation acceptance returns status 2 if open."""
    print("PASS: source-backed bridge and numerical baseline")
    installed = report["installed_fork"]
    if not installed["import_ok"]:
        print(f"OPEN GATE: installed-fork import fails ({installed['missing_module']})")
        return 2 if strict else 0
    print("PASS: installed-fork import")
    if installed.get("binary_parity"):
        print("PASS: installed bridge/fork binary parity against frozen reference")
    return 0


def inspect_wheel(directory):
    """Select the sole fresh wheel and inspect identity without a fixed version."""
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one fresh wheel, found {len(wheels)}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        metadata_paths = [n for n in members if n.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise RuntimeError("Wheel must contain exactly one METADATA file")
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode())
        if metadata["Name"] != "moving-boundary":
            raise RuntimeError(f"Unexpected distribution: {metadata['Name']}")
        dist_info = metadata_paths[0].rsplit("/", 1)[0] + "/"
        assert all(n.startswith(("moving_boundary/", dist_info)) for n in members)
        assert not any("kawin/" in n or n.endswith(".npz") for n in members)
        assert any(pinned_fork_commit() in req for req in metadata.get_all("Requires-Dist", []))
    return wheel, members


def main(argv=None):
    """Verify the development bridge and optionally require installed-fork import."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-installed-fork", action="store_true",
                        help="return exit status 2 when the installed-fork gate is open")
    args = parser.parse_args(argv)
    OUTPUT.mkdir(exist_ok=True)
    (FORK / ".pytest_tmp").mkdir(exist_ok=True)
    sha = run("commit", ["git", "rev-parse", "HEAD"], cwd=FORK, timeout=30).strip()
    check_baseline(FORK, EXPECTED)
    fixture = FORK / "kawin/tests/data/illingworth_substitutional_baseline.npz"
    assert fixture.is_file(), "Required open baseline fixture is missing"
    wheel_dir = Path(tempfile.mkdtemp(prefix="wheels-", dir=OUTPUT))
    run("build", [sys.executable, "-m", "pip", "wheel", str(PROJECT),
                  "--no-deps", "--no-build-isolation", "--no-index",
                  "--wheel-dir", str(wheel_dir)], cwd=PROJECT, timeout=120)
    wheel, members = inspect_wheel(wheel_dir)
    # A fresh target on each invocation avoids stale modules without deleting files.
    target = Path(tempfile.mkdtemp(prefix="installed-", dir=OUTPUT))
    run("install", [sys.executable, "-m", "pip", "install", str(wheel),
                    "--no-deps", "--no-index", "--target", str(target)],
        cwd=PROJECT, timeout=120)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(target), str(FORK)))
    env["PYTHONNOUSERSITE"] = "1"
    capture = """
import hashlib, importlib.metadata as md, json, os, platform, sys
from pathlib import Path
import numpy as np
import kawin, moving_boundary
from kawin.tests.illingworth_baseline_cases import BASELINE_PATH, run_all_cases
from kawin.diffusion.mesh import MovingBoundaryIllingworthFD1D as binary
from kawin.diffusion.mesh import MovingBoundaryIllingworthTernaryFD1D as ternary
target, fork, out = map(Path, sys.argv[1:])
assert Path(moving_boundary.__file__).is_relative_to(target)
assert Path(kawin.__file__).resolve() == fork / 'kawin/__init__.py'
np.testing.assert_allclose(np.linalg.solve([[2.,1.],[1.,3.]], [1.,2.]), [0.2,0.6], rtol=0, atol=1e-15)
fresh = run_all_cases()
with np.load(BASELINE_PATH) as stored:
    assert set(stored.files) == set(fresh)
    for key, value in fresh.items():
        assert np.array_equal(value, stored[key], equal_nan=True), key
np.savez(out / 'fresh-baseline.npz', **fresh)
report = dict(
    python=sys.version, executable=sys.executable, platform=platform.platform(),
    conda_prefix=os.environ.get('CONDA_PREFIX'), cwd=str(Path.cwd()),
    origins=dict(kawin=kawin.__file__, moving_boundary=moving_boundary.__file__),
    versions={name: md.version(name) for name in
              ['numpy','scipy','numba','pycalphad','pytest','setuptools','wheel','pip','kawin','moving-boundary']},
    acceleration=dict(binary=binary._solve_illingworth_tridiagonal_numba is not None,
                      ternary=ternary._solve_illingworth_block_tridiagonal_numba is not None),
    arrays={key: dict(shape=list(np.asarray(value).shape),
                     sha256=hashlib.sha256(np.asarray(value).tobytes()).hexdigest())
            for key,value in fresh.items()},
    baseline_exact=True)
(out / 'capture.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
"""
    with tempfile.TemporaryDirectory(prefix="moving-boundary-outside-") as outside:
        run("capture", [sys.executable, "-c", capture, str(target), str(FORK),
                        str(OUTPUT)], cwd=outside, env=env)
        run("installed-fork", [
            sys.executable, str(PROJECT / "tools/probe_fork.py"),
            "--bridge-target", str(target),
            "--reference", str(OUTPUT / "fresh-baseline.npz")],
            cwd=outside, env=env)
        run("parity", [sys.executable, "-m", "pytest",
                       str(PROJECT / "tests"), "--import-mode=importlib",
                       "--basetemp", str(FORK / ".pytest_tmp/package_bridge"),
                       "--junitxml", str(OUTPUT / "pytest.xml"),
                       "-q"], cwd=outside, env=env)
        run("example", [sys.executable, str(PROJECT / "examples/binary_constant.py")],
            cwd=outside, env=env)
    report = json.loads((OUTPUT / "capture.json").read_text(encoding="utf-8"))
    suites = ET.parse(OUTPUT / "pytest.xml").getroot().findall(".//testsuite")
    counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if counts["tests"] == 0 or any(counts[k] for k in ("failures", "errors", "skipped")):
        raise RuntimeError(f"Incomplete bridge acceptance: {counts}")
    report.update(commit=sha, baseline_commit=EXPECTED,
                  dependency_commit=pinned_fork_commit(), wheel_members=members,
                  wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  checks={"baseline_capture": "exact", "bridge_tests": counts,
                          "outside_example": "passed"},
                  dependency_setup="frozen source reference plus separate installed bridge/fork probe")
    report["installed_fork"] = json.loads(
        (OUTPUT / "fork-probe.json").read_text(encoding="utf-8"))
    # Retain full diagnostic paths locally; durable evidence uses stable roles.
    (OUTPUT / "verification-full.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["executable"] = "<CONDA_PREFIX>/python.exe"
    report["conda_prefix"] = "<CONDA_PREFIX>"
    report["cwd"] = "<OUTSIDE_CHECKOUTS>"
    report["origins"] = {
        "kawin": "<FORK>/kawin/__init__.py",
        "moving_boundary": "<WHEEL_TARGET>/moving_boundary/__init__.py"}
    (PROJECT / "docs/verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return finish(report, args.require_installed_fork)


if __name__ == "__main__":
    sys.exit(main())
