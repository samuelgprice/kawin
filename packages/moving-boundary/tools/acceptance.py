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
from functools import lru_cache

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / ".verification/acceptance"
REFERENCE = PROJECT.parents[2] / "mb-ref-06dc7b4"
MANIFEST = PROJECT / "tools/acceptance_manifest.json"


@lru_cache(maxsize=1)
def metric_tools():
    """Load sibling reporting code without adding a checkout to the import path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("acceptance_metrics", PROJECT / "tools/acceptance_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def worker_environment(case):
    """Isolate Python configuration; only O05 needs hash-stable provider compilation.

    NumPy seeding is deliberately absent: O05's variation comes from unordered
    symbolic construction/CSE, not a random sampler. All other cases retain
    independently randomized interpreter hashes and untouched NumPy RNG state.
    """
    env = {key:value for key,value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               NUMBA_CACHE_DIR=str(OUTPUT / "numba"))
    if case == "O05":
        env["PYTHONHASHSEED"] = "0"
    return env


def worker(case, role, *, reference=REFERENCE, state=None, prepare=False, timeout=60, output=None):
    """Execute an origin-checked worker; optional diagnostic paths never replace fixtures."""
    path = Path(output) if output is not None else OUTPUT / ("reference" if role == "reference" else "package") / case
    args = [sys.executable, "-P", "-S", str(PROJECT / "tools/acceptance_worker.py"),
        "--role", role, "--root", str(reference), "--case", case, "--output", str(path),
        "--artifacts", str(OUTPUT / "inputs")]
    if state:
        args.extend(["--target", state["target"], "--host", state["host"]])
    if prepare:
        args.append("--prepare")
    env = worker_environment(case)
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


def bit_identical(left, right):
    """Require equal shapes, dtypes and bytes, including signed zero and NaN payloads."""
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def check_repeatability(case, *, reference, directory, timeout=120):
    """Compare two fresh unseeded reference processes without changing any fixture.

    Every unaffected quick run and pytest contract is checked. A difference fails
    the gate; it must never trigger another seed exception automatically.
    """
    import numpy as np
    assert case != "O05", "O05 is the documented hash-sensitive generation workflow"
    started = time.monotonic()
    folder = Path(directory) / case
    reports = []
    for repeat in ("a", "b"):
        remaining = timeout - (time.monotonic()-started)
        assert remaining > 0, "Unseeded repeatability budget exceeded"
        report = worker(case, "reference", reference=reference,
                        output=folder / repeat, timeout=min(60,remaining))
        assert report["hash_seed"] is None and not report["numpy_seed_override"]
        reports.append(report)
    with np.load(folder / "a.npz") as left, np.load(folder / "b.npz") as right:
        assert set(left.files) == set(right.files), case
        failed = [key for key in left.files if not bit_identical(left[key],right[key])]
        result = dict(case=case, bit_identical=not failed, arrays=len(left.files), failed_keys=failed,
                      elapsed_seconds=time.monotonic()-started)
    assert reports[0].get("stop_status") == reports[1].get("stop_status"), case
    (folder / "comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    assert not failed, ("Unseeded reference is not repeatable",case,failed)
    return result


def setup_targets():
    """Install a package wheel beside stock kawin in the separate test environment."""
    from email.parser import Parser
    import tomllib
    build = Path(tempfile.mkdtemp(prefix="install-", dir=OUTPUT))
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "kawin==0.5.0" in project["project"]["dependencies"]
    import site
    from importlib.metadata import version
    assert version("kawin") == "0.5.0"
    host = Path(site.getsitepackages()[-1])
    stock = PROJECT / ".verification/kawin-0.5.0-py3-none-any.whl"
    assert hashlib.sha256(stock.read_bytes()).hexdigest() == "9f7a1497c5dff4314536860ac6fd0b96a68243dd681db8eb106dbe33fe25cbaf"
    with zipfile.ZipFile(stock) as archive:
        for member in archive.namelist():
            if member.startswith("kawin/") and not member.endswith("/"):
                assert (host / member).read_bytes() == archive.read(member), member
    for name, source in (("target", PROJECT),):
        wheel_dir = build / (name + "-wheels")
        run([sys.executable, "-m", "pip", "wheel", str(source), "--no-deps", "--no-build-isolation",
             "--no-index", "--wheel-dir", str(wheel_dir)], timeout=120)
        wheel, = wheel_dir.glob("*.whl")
        run([sys.executable, "-m", "pip", "install", str(wheel), "--no-deps", "--no-index",
             "--target", str(build / name)], timeout=120)
    state = dict(host=str(host), target=str(build / "target"), public_version="0.5.0",
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
    """Keep frozen acceptance decisions and record per-array/quantity deviation margins."""
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
    metrics = metric_tools()
    measured = {name:{} for name in metrics.QUANTITIES}
    with np.load(ref.with_suffix(".npz")) as reference, np.load(got.with_suffix(".npz")) as actual:
        assert set(reference.files) == set(actual.files), case
        for key in reference.files:
            wanted, value = reference[key], actual[key]
            group = quantity(key)
            settings = manifest["public_tolerances"][group]
            scale = 1.
            if group == "interface" or (group == "analytic_error" and key == "exact/interface_error"):
                scale = expected.get("length_scale", 1.)
            elif group == "accepted_times":
                scale = max(1., expected.get("requested_duration", 1.))
            elif group == "inventory":
                scale = expected.get("length_scale", 1.) / expected.get("minimum_molar_volume", 1.)
            measured[group][key] = metrics.array_metrics(wanted,value,
                rtol=settings["rtol"] if tolerance == "public" else 0.,
                atol=settings["atol"]*scale if tolerance == "public" else 0.)
            ok = wanted.shape == value.shape
            if ok:
                ok = np.array_equal(np.isnan(wanted), np.isnan(value))
            if ok and tolerance == "exact":
                ok = np.array_equal(wanted, value, equal_nan=True)
            elif ok:
                ok = np.allclose(wanted, value, rtol=settings["rtol"], atol=settings["atol"]*scale, equal_nan=True)
            results[group] = results.get(group, True) and bool(ok)
            if not ok:
                results.setdefault("failed_keys", []).append(key)
    results["stop_status"] = True
    for group in ("interface", "profiles", "eta", "inventory", "accepted_times", "analytic_error"):
        results.setdefault(group, None)
    quantities = {group:metrics.quantity_metrics(arrays) for group,arrays in measured.items()}
    quantities["stop_status"] = dict(applicable=True,bit_identical=True,
        max_absolute_deviation=None,max_relative_deviation=None,tolerance="exact structural equality")
    record = dict(case=case,tolerance_mode=tolerance,passed=not results.get("failed_keys"),
        bit_identical=all(stats["bit_identical"] for stats in quantities.values() if stats["applicable"]), quantities=quantities)
    got.with_suffix(".deviations.json").write_text(json.dumps(record,indent=2,allow_nan=False),encoding="utf-8")
    return results


def report_deviations(action, cases, *, contracts, skipped, started, tolerance, record=False, evidence=None):
    """Persist local metrics; export a compact tracked report only when explicitly requested.

    Contract comparisons are included separately from the run table. No fixture
    is read or written here; compare() already measured the existing outputs.
    Recording requires a passing, complete public quick/full gate.
    Only current-run files are eligible, preventing a failed case from silently
    reusing a previous successful measurement.
    """
    metrics=metric_tools()
    def load_current(ids):
        records={}
        for case in ids:
            path=OUTPUT / "package" / (case+".deviations.json")
            if path.exists() and path.stat().st_mtime >= started:
                records[case]=json.loads(path.read_text(encoding="utf-8"))
        return records
    runs=load_current(cases)
    contract_runs=load_current(contracts)
    summary=metrics.summarize(runs)
    contract_summary=metrics.summarize(contract_runs)
    local=dict(cases=runs,contracts=contract_runs,summary=summary,contract_summary=contract_summary,skipped=skipped)
    (OUTPUT / (action+"-deviations.json")).write_text(json.dumps(local,indent=2,allow_nan=False),encoding="utf-8")
    metrics.print_summary("Acceptance",summary)
    if contract_runs:
        metrics.print_summary("Contracts",contract_summary)
    if record:
        record_deviations(action,local,tolerance=tolerance,evidence=evidence)
    return metrics.public_summary(summary)


def write_report_json(path, document, *, compact_cases=False):
    """Write JSON preserving existing line endings; compact exports use one line per case."""
    newline = b"\r\n" if path.exists() and b"\r\n" in path.read_bytes() else b"\n"
    if compact_cases:
        # Keep the case table reviewable without repeating array-level metadata.
        cases = document["cases"]
        placeholder = "__CASE_TABLE__"
        text = json.dumps(dict(document, cases=placeholder), indent=2, allow_nan=False)
        rows = ["    " + json.dumps(case) + ": " + json.dumps(value, allow_nan=False)
                for case,value in cases.items()]
        text = text.replace(json.dumps(placeholder), "{\n" + ",\n".join(rows) + "\n  }")
    else:
        text = json.dumps(document, indent=2, allow_nan=False)
    path.write_bytes((text + "\n").encode("utf-8").replace(b"\n", newline))


def record_deviations(action, local, *, tolerance, evidence):
    """Export only a complete passing public gate, retaining the exact TC redaction.

    The latest recorded tier owns one compact case table (including contracts).
    Normal runs never call this function. Historical zero-error worst entries in
    acceptance.json are normalized without changing their recorded measurements.
    """
    assert action in {"quick","full"} and tolerance == "public"
    assert not evidence["failures"] and evidence["pytest"]
    assert not any(evidence["pytest"][key] for key in ("failures","errors","skipped"))
    assert evidence["pytest"]["tests"] > 0
    assert set(local["cases"]) == set(evidence["cases"]), "Incomplete run metrics"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(local["contracts"]) == set(manifest["contracts"]), "Incomplete contract metrics"
    assert all(record["passed"] for record in [*local["cases"].values(),*local["contracts"].values()])
    metrics = metric_tools()
    summary = metrics.public_summary(local["summary"])
    contract_summary = metrics.public_summary(local["contract_summary"])
    document = dict(schema=2, tier=action, reference_commit=manifest["reference_commit"], host_version="0.5.0",
        definitions=dict(relative_error="abs(actual-reference)/abs(reference); nonzero error at zero reference is unbounded (null plus flag)",
            tolerance_fraction="max elementwise abs(error)/(scaled_atol + rtol*abs(reference))",
            within_10x="tolerance fraction >= 0.1",bit_identical="same shape, dtype and bytes including signed zeros and NaN payloads",
            privacy="TC-derived numerical metrics are retained only in ignored local output; case coverage is labelled",
            mixed_tolerances="Distinct applied tolerances are listed once per quantity; array assignments stay local"),
        run_cases=list(local["cases"]),contract_cases=list(local["contracts"]),
        summary=summary,contract_summary=contract_summary,skipped=local["skipped"],
        cases=metrics.public_cases(dict(local["cases"],**local["contracts"])))
    write_report_json(PROJECT / "docs/acceptance-deviations.json",document,compact_cases=True)
    path = PROJECT / "docs/acceptance.json"
    acceptance = json.loads(path.read_text(encoding="utf-8"))
    for entry in acceptance.get("review_followup",{}).values():
        if isinstance(entry,dict) and isinstance(entry.get("deviations"),dict):
            worst = entry["deviations"].get("worst")
            if worst and not worst.get("unbounded") and worst.get("fraction") == 0 and worst.get("error") == 0:
                entry["deviations"]["worst"] = None
    acceptance.setdefault("review_followup",{})[action] = dict(elapsed_seconds=evidence["elapsed_seconds"],
        pytest=evidence["pytest"],deviations=summary)
    public = acceptance.setdefault("public_kawin",{})
    public[action] = dict(elapsed_seconds=evidence["elapsed_seconds"],pytest=evidence["pytest"],skipped=local["skipped"])
    public[action+"_runs"] = len(local["cases"])
    write_report_json(path,acceptance)


def contract_case(case, state=None, *, reference=REFERENCE, tolerance="exact", require=()):
    """Entry point for package pytest: compare an isolated contract with its fixture."""
    if state is None:
        state = json.loads((OUTPUT / "state.json").read_text(encoding="utf-8"))
    if state.get("repeatability_directory"):
        check_repeatability(case, reference=reference, directory=state["repeatability_directory"])
    report = worker(case, "package", reference=reference, state=state)
    result = compare(case, report, tolerance=tolerance, require=require)
    assert not result.get("failed_keys"), result
    return result


def run_pytest(state, *, reference, tolerance, require, timeout=180):
    """Run package contracts and kernel/tooling tests against the installed targets."""
    state.update(reference=str(reference), tolerance=tolerance, require=list(require))
    configured = OUTPUT / "state.json"
    configured.write_text(json.dumps(state, indent=2), encoding="utf-8")
    env = {key:value for key,value in os.environ.items() if not key.startswith("PYTHON")}
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
    tests = ["test_acceptance_contracts.py", "test_tridiagonal.py", "test_batch_safety.py", "test_verification.py", "test_compatibility.py", "test_acceptance_reproducibility.py", "test_acceptance_metrics.py", "test_acceptance_reporting.py"]
    args = [sys.executable, "-I", "-S", "-c", code, state["target"], state["host"],
            *[str(PROJECT / "tests" / name) for name in tests],
            str(PROJECT / "tests/test_bridge.py") + "::test_namespace_import_is_lazy",
            str(PROJECT / "tests/test_bridge.py") + "::test_binary_example",
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
    parser.add_argument("--require-family", nargs="*", default=["binary", "ternary", "three_phase"])
    parser.add_argument("--tolerance", choices=("exact", "public"), default="public")
    parser.add_argument("--tier", choices=("quick", "full"), default="quick")
    parser.add_argument("--record", action="store_true",
                        help="update committed reports after a passing complete public quick/full run")
    args = parser.parse_args(argv)
    if args.record and (args.action not in {"quick","full"} or args.tolerance != "public" or args.cases is not None):
        parser.error("--record requires a complete public-tolerance quick or full run")
    # The parent is invoked after required Conda activation. The cloned runtime
    # inherits those native DLL paths; prepend the clone's equivalent locations.
    if args.action != "capture":
        config = json.loads((PROJECT / ".verification/public-environment.json").read_text())
        executable = Path(config["python"])
        if Path(sys.executable).resolve() != executable.resolve():
            env = os.environ.copy()
            prefix = executable.parent
            env["PATH"] = os.pathsep.join([str(prefix), str(prefix/"Library/bin"), str(prefix/"Scripts"), env.get("PATH", "")])
            env["CONDA_PREFIX"] = str(prefix)
            return subprocess.run([str(executable), str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
                env=env, stdin=subprocess.DEVNULL, timeout=1250 if args.action=="full" else 650).returncode
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
    if not args.cases and not (OUTPUT / "inputs/FeCrNi.tdb").exists():
        skipped["O05"] = "database input missing"
    if full and not args.cases:
        cases.extend(manifest["full_drift"])
        private = {"O01":"exb5.json","O02":"exb5.json","O03a":"femnc.json","O03b":"femnc.json", "O04":"fecrni.surrogate","O07":"fec.json"}
        for case, name in private.items():
            if (OUTPUT / "inputs" / name).exists():
                cases.append(case)
            else:
                skipped[case] = "cached input missing"
        skipped["O06"] = "TC adapter not yet batch-safe; disabled as approved"
    # Private hashes never enter committed evidence. Present corrupted inputs
    # fail instead of being disguised as missing licensed/provider coverage.
    inputs_manifest = OUTPUT / "inputs/private-inputs.json"
    if inputs_manifest.exists():
        enabled_inputs = {"O01":"exb5.json", "O02":"exb5.json", "O03a":"femnc.json",
            "O03b":"femnc.json", "O04":"fecrni.surrogate", "O05":"FeCrNi.tdb", "O07":"fec.json"}
        recorded = json.loads(inputs_manifest.read_text(encoding="utf-8"))
        for case in cases:
            name = enabled_inputs.get(case)
            if name is not None and name in recorded:
                for relative, digest in recorded[name].get("sha256", {}).items():
                    location = OUTPUT / "inputs"
                    if (location / name).is_dir():
                        location = location / name
                    assert hashlib.sha256((location / relative).read_bytes()).hexdigest() == digest, case + " input changed"
    if args.action == "setup":
        print(json.dumps(setup_targets()))
        return 0
    started = time.monotonic()
    started_timestamp = time.time()
    results, failures = {}, []
    state = None
    if args.action != "capture":
        state = json.loads((OUTPUT / "state.json").read_text(encoding="utf-8")) if args.reuse_targets else setup_targets()
        state["repeatability_directory"] = tempfile.mkdtemp(prefix="unseeded-", dir=OUTPUT)
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
            if args.action in {"quick","full"} and case in set(manifest["quick"]) | {"C32","LEGACY"}:
                check_repeatability(case, reference=args.reference, directory=state["repeatability_directory"],
                    timeout=max(1,limit-(time.monotonic()-started)))
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
    if args.action in {"quick","full"}:
        summary["deviations"] = report_deviations(args.action,cases,
            contracts=manifest["contracts"] if tests else (),skipped=skipped,
            started=started_timestamp,tolerance=args.tolerance,
            record=args.record and not failures,evidence=summary)
    if state and state.get("repeatability_directory"):
        checks = [json.loads(path.read_text()) for path in Path(state["repeatability_directory"]).glob("*/comparison.json")]
        summary["unseeded_repeatability"] = {entry["case"]:entry for entry in checks}
        print(f"Unseeded reference repeatability: {len(checks)} cases checked twice; "
              f"{sum(not entry['bit_identical'] for entry in checks)} differences", flush=True)
    (OUTPUT / (args.action + "-summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(results)} cases, {len(failures)} failures, {summary['elapsed_seconds']:.2f}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
