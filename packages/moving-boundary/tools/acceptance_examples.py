"""Example acceptance on the installed wheel, separate from the approved quick and full gates.

Each case runs an example's quick-check settings and requires:

- recorded effective settings equal to the accepted case's, except approved named differences;
- parity arrays with the fixture's exact key set and identical bytes (matching behaviour);
- the same stop status.

Then it runs the portable example and analytic tests with ``MB_EXAMPLE_TDB`` set to the local
database. ``acceptance_manifest.json`` and every existing fixture are only read. The adopted
reference settings under ``.verification/acceptance/example_settings`` are hash-checked before
use: non-private cases against ``settings_sha256`` in ``example_manifest.json``, the O04/O05
(TDB/TC-derived) ones against ``example_settings`` in the local ``inputs/private-inputs.json``.
A case's private surrogate input is checked against ``private-inputs.json`` the same way.

``run`` needs no research checkout: the fixtures it compares with are hash-validated by
``acceptance.validate_expected`` before and after (``validate_references``). The reference
cases' own inputs are not required: a missing ``open-surrogate.npz`` fails C24 in quick/full
acceptance, but E04 builds its own surrogate and still runs here.
``capture-settings --research PATH`` is historical (reference role, frozen research checkout,
validated by ``acceptance.validate_research_checkout``): it re-records
every mapped case into a fresh ``settings-capture-*`` scratch folder, never into
``example_settings/``, and reports whether each capture equals the adopted file byte for byte.

Commands (through the conda wrapper, from any directory):
    python tools/acceptance_examples.py capture-settings --research <frozen research checkout>
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


def worker(tool, role, case, state, output, *, env_case, timeout=120, research=None):
    """Run one bounded worker with the same isolation rules as acceptance.py.

    Package-role workers get no research root; reference-role (historical) workers need one.
    """
    arguments = [sys.executable, "-P", "-S", str(PROJECT / "tools/example_worker.py"), "--role", role, "--case", case,
                 "--examples", str(PROJECT / "examples"),
                 "--artifacts", str(tool.OUTPUT / "inputs"), "--output", str(output)]
    if role == "package":
        arguments.extend(["--target", state["target"], "--host", state["host"]])
    else:
        assert research is not None, "reference-role example workers are historical and need --research"
        arguments.extend(["--root", str(research)])
    return tool.run(arguments, env=tool.worker_environment(env_case), timeout=timeout)


def capture_settings(tool, manifest, research):
    """Historical: re-record each mapped case's reference settings into a new scratch folder.

    Never writes ``example_settings/``. Prints, per case, whether the capture equals the adopted
    file byte for byte, and returns ``(folder, {reference: verdict})``.
    """
    import tempfile
    folder = Path(tempfile.mkdtemp(prefix="settings-capture-", dir=tool.OUTPUT))
    adopted = tool.OUTPUT / "example_settings"
    verdicts = {}
    for case, entry in manifest["cases"].items():
        path = folder / entry["reference"]
        worker(tool, "reference", case, None, path, env_case=entry["reference"], research=research)
        name = path.with_suffix(".settings.json").name
        old = adopted / name
        verdicts[entry["reference"]] = ("no adopted file" if not old.exists() else
                                        "identical" if old.read_bytes() == (folder / name).read_bytes() else "DIFFERS")
        print(entry["reference"], "recorded in", folder.name, "-", verdicts[entry["reference"]], flush=True)
    return folder, verdicts


def validate_references(tool, references):
    """Hash-validate the reference artifacts the examples compare with (Git-free, no research).

    Missing original inputs of those reference cases (e.g. C24's ``open-surrogate.npz``) are not
    required, since no reference case is rerun here (plan follow-up 5); a present adopted input
    must still match its hash, and each example's own inputs are checked per case in ``main``.
    """
    return tool.validate_expected(references, required_inputs_for=())


def private_inputs(tool):
    """The local private-input hash record, or ``None`` when this machine has none."""
    return tool.load_private_inputs()


def settings_digests(manifest, private):
    """Expected sha256 per adopted settings file name: committed, plus local private entries.

    With no (or a malformed) private record, the O04/O05 settings simply have no digest, so
    ``check_settings_hash`` reports them as not hashed when their case runs.
    """
    digests = dict(manifest["settings_sha256"])
    entry = (private or {}).get("example_settings")
    recorded = entry.get("sha256") if isinstance(entry, dict) else None
    if isinstance(recorded, dict):
        digests.update(recorded)
    return digests


def check_settings_hash(path, digests):
    """Failures for one adopted settings file: unhashed or changed (an absent file is reported elsewhere)."""
    import hashlib
    if not path.exists():
        return []
    if path.name not in digests:
        return ["adopted settings not hashed: " + path.name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != digests[path.name]:
        return ["adopted settings changed: " + path.name]
    return []


def check_input_hash(tool, name, private):
    """Failures for a case's available private input against ``private-inputs.json``.

    Delegates to ``acceptance.private_input_problems``: a missing record file, a missing entry,
    an empty or malformed hash map, an unrecorded, missing or changed file all fail. An input
    this machine lacks is skipped by the caller before this check.
    """
    return tool.private_input_problems(name, private)


def compare_case(tool, settings, case, entry, allowed, digests):
    """Compare one package run with its expected fixture; returns a list of failures.

    The fixture is ``tool.expected_reference(entry["reference"])``: ``reference/<case>`` with the
    adopted ``example_settings`` file, or, for an approved case, the approval's arrays with the
    package-captured settings stored and hashed beside them (an approved mapped case without
    them fails).
    """
    import numpy as np
    failures = []
    got = tool.OUTPUT / "package_examples" / case
    selected = tool.expected_reference(entry["reference"])
    if selected["source"] == "reference":
        recorded = tool.OUTPUT / "example_settings" / (entry["reference"] + ".settings.json")
    elif selected["settings"] is None:
        recorded = None
        failures.append(f"approved reference {selected['source']} has no captured settings for {entry['reference']}")
    else:
        recorded = selected["settings"]
        digests = {recorded.name: selected["settings_sha256"]}
    problems = [] if recorded is None else (
        ["adopted reference settings missing: " + recorded.name] if not recorded.exists() else check_settings_hash(recorded, digests))
    failures.extend(problems)
    if recorded is not None and not problems:
        found = settings.differences(json.loads(recorded.read_text(encoding="utf-8")),
                                     json.loads(got.with_suffix(".settings.json").read_text(encoding="utf-8")), allowed)
        failures.extend("setting differs: " + name for name in found)
    expected = json.loads(selected["json"].read_text(encoding="utf-8"))
    actual = json.loads(got.with_suffix(".json").read_text(encoding="utf-8"))
    if expected.get("stop_status") != actual["stop_status"]:
        failures.append("stop status differs")
    with np.load(selected["npz"]) as ref, np.load(got.with_suffix(".npz")) as new:
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
    """Re-execute in the public environment, then capture settings or run the example cases.

    Installs the opt-in independence guards first (``batch_safety.guard_independence``).
    """
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parent / "batch_safety.py"))["guard_independence"]()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture-settings", "run"))
    parser.add_argument("--tier", choices=("quick", "full"), default="full")
    parser.add_argument("--slow", action="store_true", help="also run tests marked slow, with their own timeout")
    parser.add_argument("--reuse-targets", action="store_true", help="reuse the last installed wheel")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--research", type=Path, help="frozen research checkout; capture-settings (historical) only")
    args = parser.parse_args(argv)
    if (args.action == "capture-settings") != (args.research is not None):
        parser.error("--research is required by capture-settings and not accepted by run")
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
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.action == "capture-settings":
        tool.validate_research_checkout(args.research)
        _, verdicts = capture_settings(tool, manifest, args.research)
        tool.validate_research_checkout(args.research)
        return 0 if all(verdict == "identical" for verdict in verdicts.values()) else 1
    # The fixtures each example is compared with are hash-validated before and after (no research).
    references = sorted({entry["reference"] for entry in manifest["cases"].values()})
    validate_references(tool, references)
    spec = importlib.util.spec_from_file_location("example_settings", PROJECT / "tools/example_settings.py")
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    state = json.loads((tool.OUTPUT / "state.json").read_text(encoding="utf-8")) if args.reuse_targets else tool.setup_targets()
    started = time.monotonic()
    failures = {}
    skipped = {}
    private = private_inputs(tool)
    digests = settings_digests(manifest, private)
    for case, entry in manifest["cases"].items():
        if args.tier == "quick" and entry["tier"] != "quick":
            continue
        source = entry.get("surrogate")
        if source and source != "lifecycle" and not (tool.OUTPUT / "inputs" / source).exists():
            skipped[case] = "local input missing: " + source
            print(case, "SKIPPED", skipped[case], flush=True)
            continue
        names = {name for name in (source, entry.get("provider_arrays")) if name and name != "lifecycle"}
        unverified = [failure for name in sorted(names) for failure in check_input_hash(tool, name, private)]
        if unverified:
            print(case, entry["example"], entry["config"], "vs", entry["reference"], "FAIL", *unverified, flush=True)
            failures[case] = unverified
            continue
        try:
            worker(tool, "package", case, state, tool.OUTPUT / "package_examples" / case, env_case=case)
            found, seconds = compare_case(tool, settings, case, entry, manifest["allowed_setting_differences"].get(case, {}), digests)
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
    validate_references(tool, references)
    elapsed = time.monotonic() - started
    summary = dict(tier=args.tier, cases=len(manifest["cases"]), failures=failures, skipped=skipped, tests=tests,
                   pending_fixture=sorted(manifest["pending_fixture"]), elapsed_seconds=elapsed)
    (tool.OUTPUT / "examples-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(failures)} failing cases, pending fixtures {sorted(manifest['pending_fixture'])}, {elapsed:.1f}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
