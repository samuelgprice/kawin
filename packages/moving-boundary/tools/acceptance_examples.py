"""Example acceptance on the installed wheel, separate from the approved quick and full gates.

Each case runs an example's quick-check settings and requires:

- recorded effective settings equal to the accepted case's, except approved named differences;
- parity arrays with the fixture's exact key set and identical bytes (matching behaviour);
- the same stop status.

Then it runs the portable example and analytic tests with ``MB_EXAMPLE_TDB`` set to the local
database. ``acceptance_manifest.json`` and every existing fixture are only read. Reference
settings are new derived local files under ``.verification/acceptance/example_settings``.

Commands (through the conda wrapper, from any directory):
    python tools/acceptance_examples.py capture-settings
    python tools/acceptance_examples.py run [--tier quick|full] [--slow] [--reuse-targets]
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "tools/example_manifest.json"


def load_tool():
    """Load acceptance.py for its reference validation, wheel setup and environment rules."""
    spec = importlib.util.spec_from_file_location("acceptance", PROJECT / "tools/acceptance.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def worker(tool, role, case, state, output, *, env_case, timeout=120):
    """Run one bounded worker with the same isolation rules as acceptance.py."""
    arguments = [sys.executable, "-P", "-S", str(PROJECT / "tools/example_worker.py"), "--role", role, "--case", case,
                 "--root", str(tool.REFERENCE), "--examples", str(PROJECT / "examples"),
                 "--artifacts", str(tool.OUTPUT / "inputs"), "--output", str(output)]
    if role == "package":
        arguments.extend(["--target", state["target"], "--host", state["host"]])
    return tool.run(arguments, env=tool.worker_environment(env_case), timeout=timeout)


def capture_settings(tool, manifest):
    """Record reference settings once for each mapped case; never overwrites."""
    folder = tool.OUTPUT / "example_settings"
    for case, entry in manifest["cases"].items():
        path = folder / entry["reference"]
        target = path.with_suffix(".settings.json")
        if target.exists():
            print(entry["reference"], "already recorded", flush=True)
            continue
        worker(tool, "reference", case, None, path, env_case=entry["reference"])
        print(entry["reference"], "recorded", flush=True)


def compare_case(tool, settings, case, entry, allowed):
    """Compare one package run with its frozen fixture; returns a list of failures."""
    import numpy as np
    failures = []
    got = tool.OUTPUT / "package_examples" / case
    reference = tool.OUTPUT / "reference" / entry["reference"]
    recorded = tool.OUTPUT / "example_settings" / (entry["reference"] + ".settings.json")
    if not recorded.exists():
        failures.append("reference settings not recorded; run capture-settings")
    else:
        found = settings.differences(json.loads(recorded.read_text(encoding="utf-8")),
                                     json.loads(got.with_suffix(".settings.json").read_text(encoding="utf-8")), allowed)
        failures.extend("setting differs: " + name for name in found)
    expected = json.loads(reference.with_suffix(".json").read_text(encoding="utf-8"))
    actual = json.loads(got.with_suffix(".json").read_text(encoding="utf-8"))
    if expected.get("stop_status") != actual["stop_status"]:
        failures.append("stop status differs")
    with np.load(reference.with_suffix(".npz")) as ref, np.load(got.with_suffix(".npz")) as new:
        if set(ref.files) != set(new.files):
            failures.append(f"key set differs: missing {sorted(set(ref.files) - set(new.files))}, extra {sorted(set(new.files) - set(ref.files))}")
        for key in sorted(set(ref.files) & set(new.files)):
            if not tool.bit_identical(ref[key], new[key]):
                failures.append("array differs: " + key)
    return failures, actual["elapsed_seconds"]


def run_tests(tool, state, slow, timeout):
    """Run the portable tests against the installed wheel; slow tests only when requested."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([state["target"], state["host"]]),
               NUMBA_CACHE_DIR=str(tool.OUTPUT / "numba"), MB_EXAMPLE_TDB=str(tool.OUTPUT / "inputs/FeCrNi.tdb"))
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
    xml = tool.OUTPUT / "pytest-examples.xml"
    arguments = [sys.executable, "-I", "-S", "-c", code, state["target"], state["host"],
                 str(PROJECT / "tests/test_analytic.py"), str(PROJECT / "tests/test_examples.py"),
                 "--import-mode=importlib", "--basetemp", str(tool.OUTPUT / "pytest-examples"), "-q",
                 "-m", "slow" if slow == "only" else "not slow", "--junitxml", str(xml), "-p", "no:cacheprovider"]
    output = tool.run(arguments, cwd=tool.OUTPUT, env=env, timeout=timeout)
    print(output[-600:], flush=True)
    import xml.etree.ElementTree as ET
    root = ET.parse(xml).getroot()
    suites = root.findall(".//testsuite")
    counts = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    names = [case.get("name") for case in root.iter("testcase")]
    # A marker alone doesn't exclude anything, so confirm what actually ran.
    assert ("test_three_phase_errors_converge_with_refinement" in names) == (slow == "only"), names
    assert counts["tests"] and not any(counts[key] for key in ("failures", "errors", "skipped")), counts
    return counts


def main(argv=None):
    """Re-execute in the public environment, then capture settings or run the example cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture-settings", "run"))
    parser.add_argument("--tier", choices=("quick", "full"), default="full")
    parser.add_argument("--slow", action="store_true", help="also run tests marked slow, with their own timeout")
    parser.add_argument("--reuse-targets", action="store_true", help="reuse the last installed wheel")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    tool = load_tool()
    config = json.loads((PROJECT / ".verification/public-environment.json").read_text())
    executable = Path(config["python"])
    if Path(sys.executable).resolve() != executable.resolve():
        env = os.environ.copy()
        prefix = executable.parent
        env["PATH"] = os.pathsep.join([str(prefix), str(prefix / "Library/bin"), str(prefix / "Scripts"), env.get("PATH", "")])
        env["CONDA_PREFIX"] = str(prefix)
        return subprocess.run([str(executable), str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
                              env=env, stdin=subprocess.DEVNULL, timeout=900).returncode
    (tool.OUTPUT / "package_examples").mkdir(parents=True, exist_ok=True)
    tool.validate_reference(tool.REFERENCE)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.action == "capture-settings":
        capture_settings(tool, manifest)
        tool.validate_reference(tool.REFERENCE)
        return 0
    spec = importlib.util.spec_from_file_location("example_settings", PROJECT / "tools/example_settings.py")
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    state = json.loads((tool.OUTPUT / "state.json").read_text(encoding="utf-8")) if args.reuse_targets else tool.setup_targets()
    started = time.monotonic()
    failures = {}
    skipped = {}
    for case, entry in manifest["cases"].items():
        if args.tier == "quick" and entry["tier"] != "quick":
            continue
        source = entry.get("surrogate")
        if source and source != "lifecycle" and not (tool.OUTPUT / "inputs" / source).exists():
            skipped[case] = "local input missing: " + source
            print(case, "SKIPPED", skipped[case], flush=True)
            continue
        try:
            worker(tool, "package", case, state, tool.OUTPUT / "package_examples" / case, env_case=case)
            found, seconds = compare_case(tool, settings, case, entry, manifest["allowed_setting_differences"].get(case, {}))
        except Exception as exc:
            found, seconds = ["worker failed: " + str(exc)[-1500:]], 0.0
        print(case, entry["example"], entry["config"], "vs", entry["reference"], "FAIL" if found else "PASS", f"{seconds:.1f}s", *found, flush=True)
        if found:
            failures[case] = found
    tests = None
    if not args.skip_tests:
        tests = run_tests(tool, state, "no", timeout=420)
        if args.slow:
            tests = {"fast": tests, "slow": run_tests(tool, state, "only", timeout=300)}
    tool.validate_reference(tool.REFERENCE)
    elapsed = time.monotonic() - started
    summary = dict(tier=args.tier, cases=len(manifest["cases"]), failures=failures, skipped=skipped, tests=tests,
                   pending_fixture=sorted(manifest["pending_fixture"]), elapsed_seconds=elapsed)
    (tool.OUTPUT / "examples-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(failures)} failing cases, pending fixtures {sorted(manifest['pending_fixture'])}, {elapsed:.1f}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
