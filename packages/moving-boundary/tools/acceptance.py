"""Quick/full installed-package acceptance with explicit per-quantity outcomes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / ".verification/acceptance"
REFERENCE = PROJECT.parents[2] / "mb-ref-06dc7b4"
MANIFEST = PROJECT / "tools/acceptance_manifest.json"


def run(args, *, cwd=None, timeout=120, env=None):
    """Run a bounded batch job and preserve diagnostic output on failure."""
    result = subprocess.run(args, cwd=cwd or OUTPUT, env=env, capture_output=True,
                            text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def validate_reference(reference):
    """Require the requested detached commit, a clean tree, and frozen input hashes."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert run(["git", "rev-parse", "HEAD"], cwd=reference, timeout=30).strip() == manifest["reference_commit"]
    result = subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=reference,
                            capture_output=True, timeout=30)
    assert result.returncode == 1, "Reference must be detached"
    assert not run(["git", "status", "--porcelain"], cwd=reference, timeout=30).strip(), "Reference must be clean"
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((reference / name).read_bytes()).hexdigest() == digest, name
    return manifest


def worker(case, role, *, reference=REFERENCE, state=None, prepare=False, timeout=60):
    """Execute a single worker outside source roots; retain outputs locally."""
    path = OUTPUT / ("reference" if role == "reference" else "package") / case
    args = [sys.executable, "-P", "-S", str(PROJECT / "tools/acceptance_worker.py"),
        "--role", role, "--root", str(reference), "--case", case, "--output", str(path),
        "--artifacts", str(OUTPUT / "inputs")]
    if state:
        args.extend(["--target", state["target"], "--host", state["host"]])
    if prepare:
        args.append("--prepare")
    env = {key:value for key,value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", PYTHONNOUSERSITE="1",
               NUMBA_CACHE_DIR=str(OUTPUT / "numba"))
    started = time.monotonic()
    try:
        output = run(args, env=env, timeout=timeout)
    except Exception as exc:
        (path.parent).mkdir(parents=True, exist_ok=True)
        (path.with_suffix(".log")).write_text(str(exc), encoding="utf-8")
        raise
    path.with_suffix(".log").write_text(output, encoding="utf-8")
    report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    report["wall_seconds"] = time.monotonic() - started
    return report


def setup_targets():
    """Build/install package and exact fork wheels privately, without shared changes."""
    from email.parser import Parser
    import tomllib
    build = Path(tempfile.mkdtemp(prefix="install-", dir=OUTPUT))
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    pin, = [r.rsplit("@", 1)[-1] for r in project["project"]["dependencies"] if r.startswith("kawin @ ")]
    archive = build / "fork.zip"
    run(["git", "archive", "-o", str(archive), pin], cwd=PROJECT.parents[1], timeout=30)
    with zipfile.ZipFile(archive) as z:
        z.extractall(build / "source")
    for name, source in (("host", build / "source"), ("target", PROJECT)):
        wheel_dir = build / (name + "-wheels")
        run([sys.executable, "-m", "pip", "wheel", str(source), "--no-deps", "--no-build-isolation",
             "--no-index", "--wheel-dir", str(wheel_dir)], timeout=120)
        wheel, = wheel_dir.glob("*.whl")
        run([sys.executable, "-m", "pip", "install", str(wheel), "--no-deps", "--no-index",
             "--target", str(build / name)], timeout=120)
    state = dict(host=str(build / "host"), target=str(build / "target"), pin=pin,
                 manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
    (OUTPUT / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def quantity(key):
    """Classify recorded values while keeping times separate from their histories."""
    if key.endswith("/time") or key.endswith("/final_time") or key.endswith("_time"):
        return "accepted_times"
    if key.startswith("exact/"):
        return "analytic_error"
    if "etaData" in key:
        return "eta"
    if any(name in key for name in ("Inventory", "inventory", "totalMoles", "Closure", "concData")):
        return "inventory"
    if "interfaceData" in key or "rightBoundaryData" in key:
        return "interface"
    return "profiles"


def compare(case, package, *, tolerance="exact", require=()):
    """Compare every quantity, rejecting altered shapes/masks/status/providers."""
    import numpy as np
    ref = OUTPUT / "reference" / case
    got = OUTPUT / "package" / case
    expected = json.loads(ref.with_suffix(".json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert expected["status"] == package["status"] == "PASS"
    assert expected.get("stop_status") == package.get("stop_status")
    for family in require:
        provider = package.get("providers", {}).get(family)
        if provider is not None:
            assert provider.startswith("moving_boundary."), (case, family, provider)
    results = {}
    with np.load(ref.with_suffix(".npz")) as reference, np.load(got.with_suffix(".npz")) as actual:
        assert set(reference.files) == set(actual.files), case
        for key in reference.files:
            wanted, value = reference[key], actual[key]
            group = quantity(key)
            ok = wanted.shape == value.shape
            if ok:
                ok = np.array_equal(np.isnan(wanted), np.isnan(value))
            if ok and tolerance == "exact":
                ok = np.array_equal(wanted, value, equal_nan=True)
            elif ok:
                settings = manifest["public_tolerances"][group]
                scale = 1.
                if group == "interface":
                    scale = expected.get("length_scale", 1.)
                elif group == "accepted_times":
                    scale = max(1., expected.get("requested_duration", 1.))
                elif group == "inventory":
                    scale = expected.get("length_scale", 1.) / expected.get("minimum_molar_volume", 1.)
                ok = np.allclose(wanted, value, rtol=settings["rtol"], atol=settings["atol"]*scale, equal_nan=True)
            results[group] = results.get(group, True) and bool(ok)
            if not ok:
                results.setdefault("failed_keys", []).append(key)
    results["stop_status"] = True
    for group in ("interface", "profiles", "eta", "inventory", "accepted_times", "analytic_error"):
        results.setdefault(group, None)
    return results


def contract_case(case, state=None, *, reference=REFERENCE, tolerance="exact", require=()):
    """Entry point for package pytest: compare an isolated contract with its fixture."""
    if state is None:
        state = json.loads((OUTPUT / "state.json").read_text(encoding="utf-8"))
    report = worker(case, "package", reference=reference, state=state)
    result = compare(case, report, tolerance=tolerance, require=require)
    assert not result.get("failed_keys"), result
    return result


def run_pytest(state, *, reference, tolerance, require, timeout=180):
    """Run package contracts and kernel/tooling tests against the installed targets."""
    state.update(reference=str(reference), tolerance=tolerance, require=list(require))
    configured = OUTPUT / "state.json"
    configured.write_text(json.dumps(state, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update(MB_ACCEPTANCE_STATE=str(configured), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([state["target"], state["host"]]),
               NUMBA_CACHE_DIR=str(OUTPUT / "numba"))
    code = """
import site, sys
from pathlib import Path
sys.path[:0] = sys.argv[1:3]
sys.path.extend(site.getsitepackages())
import kawin
assert Path(kawin.__file__).resolve() == Path(sys.argv[2]) / 'kawin/__init__.py'
import pytest
raise SystemExit(pytest.main(sys.argv[3:]))
"""
    tests = ["test_acceptance_contracts.py", "test_tridiagonal.py", "test_batch_safety.py", "test_verification.py"]
    args = [sys.executable, "-I", "-S", "-c", code, state["target"], state["host"],
            *[str(PROJECT / "tests" / name) for name in tests],
            str(PROJECT / "tests/test_bridge.py") + "::test_namespace_import_is_lazy",
            str(PROJECT / "tests/test_bridge.py") + "::test_missing_fork_error",
            "--import-mode=importlib", "--basetemp", str(OUTPUT / "pytest"), "-q",
            "--junitxml", str(OUTPUT / "pytest.xml")]
    output = run(args, env=env, timeout=timeout)
    (OUTPUT / "pytest.log").write_text(output, encoding="utf-8")
    print(output[-1000:], flush=True)
    import xml.etree.ElementTree as ET
    suites = ET.parse(OUTPUT / "pytest.xml").getroot().findall(".//testsuite")
    counts = {key:sum(int(suite.get(key,0)) for suite in suites) for key in ("tests","failures","errors","skipped")}
    assert counts["tests"] and not any(counts[k] for k in ("failures","errors","skipped")), counts
    return counts


def main(argv=None):
    """Capture fixtures or run quick/full acceptance with an explicit outcome table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "quick", "full", "setup"))
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--reuse-targets", action="store_true")
    parser.add_argument("--require-family", nargs="*", default=[])
    parser.add_argument("--tolerance", choices=("exact", "public"), default="exact")
    parser.add_argument("--tier", choices=("quick", "full"), default="quick")
    args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "inputs").mkdir(exist_ok=True)
    manifest = validate_reference(args.reference)
    full = args.action == "full" or args.tier == "full"
    cases = args.cases or list(manifest["quick"]) + ["C32", "LEGACY"]
    # Real sampling fit the approved minute pilot; sampling runs afresh in O05.
    if not args.cases and (OUTPUT / "inputs/FeCrNi.tdb").exists():
        pilot = OUTPUT / "pilot/O05-sampling.json"
        if full or (pilot.exists() and json.loads(pilot.read_text())["elapsed_seconds"] <= 60):
            cases.append("O05")
    skipped = {}
    if full and not args.cases:
        cases.extend(manifest["full_drift"])
        private = {"O01":"exb5.json","O02":"exb5.json","O03a":"femnc.json","O03b":"femnc.json", "O04":"fecrni.surrogate","O07":"fec.json"}
        for case, name in private.items():
            if (OUTPUT / "inputs" / name).exists():
                cases.append(case)
            else:
                skipped[case] = "cached input missing"
        skipped["O06"] = "TC adapter not yet batch-safe; disabled as approved"
    if args.action == "setup":
        print(json.dumps(setup_targets()))
        return 0
    started = time.monotonic()
    results, failures = {}, []
    state = None
    if args.action != "capture":
        state = json.loads((OUTPUT / "state.json").read_text(encoding="utf-8")) if args.reuse_targets else setup_targets()
    else:
        worker("prepare", "reference", reference=args.reference, prepare=True)
    if args.action == "capture" and not args.cases:
        cases = cases + manifest["contracts"]
    for case in cases:
        limit = 1200 if full else 600
        remaining = limit - (time.monotonic() - started)
        if remaining <= 0:
            failures.append("overall timeout")
            break
        try:
            report = worker(case, "reference" if args.action == "capture" else "package",
                            reference=args.reference, state=state,
                            timeout=min(180 if case.startswith("F") else 60, max(1,remaining)))
            result = {"capture":True} if args.action == "capture" else compare(case, report,
                tolerance=args.tolerance, require=args.require_family)
            results[case] = result
            if result.get("failed_keys"):
                failures.append(case)
            print(case, "FAIL" if result.get("failed_keys") else "PASS",
                  " ".join(f"{k}={'N/A' if v is None else ('PASS' if v else 'FAIL')}" for k,v in result.items() if k != "failed_keys"), flush=True)
        except Exception as exc:
            failures.append(case)
            results[case] = dict(error=str(exc))
            print(case, "FAIL", str(exc)[-2000:], flush=True)
    tests = None
    if args.action in {"quick", "full"} and not args.cases:
        try:
            remaining = (1200 if full else 600) - (time.monotonic()-started)
            tests = run_pytest(state, reference=args.reference, tolerance=args.tolerance,
                               require=args.require_family, timeout=max(1,remaining))
        except Exception as exc:
            failures.append("pytest")
            print("PYTEST FAIL", str(exc)[-3000:], flush=True)
    for case, reason in skipped.items():
        print(case, "SKIPPED", reason, flush=True)
    validate_reference(args.reference)
    summary = dict(action=args.action, cases=results, failures=failures, skipped=skipped,
                   pytest=tests, elapsed_seconds=time.monotonic()-started)
    (OUTPUT / (args.action + "-summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(results)} cases, {len(failures)} failures, {summary['elapsed_seconds']:.2f}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
