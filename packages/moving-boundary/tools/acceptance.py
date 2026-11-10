"""Quick/full installed-package acceptance with explicit per-quantity outcomes.

Normal actions (``quick``, ``full``, ``setup``) need no research checkout and no Git: package
workers build from ``tools/definitions`` on the installed wheel, ``validate_expected`` checks
every hash-recorded expected artifact and adopted input before and after the run, and unseeded
repeatability runs two extra package-role processes per case.

Historical actions are explicit opt-ins on the frozen research checkout:
``historical capture --research PATH`` records reference-role fixtures (refusing to overwrite
any existing ``reference/`` file), and ``historical repeat --research PATH`` runs the reference
role twice per case and requires both runs to equal the adopted ``reference/`` bit for bit.
Both validate the checkout first (``validate_research_checkout``).
"""
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
MANIFEST = PROJECT / "tools/acceptance_manifest.json"


@lru_cache(maxsize=1)
def metric_tools():
    """Load sibling reporting code without adding a checkout to the import path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("acceptance_metrics", PROJECT / "tools/acceptance_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def thermocalc_tools():
    """Load TC tooling without importing its optional licensed engine."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("acceptance_thermocalc", PROJECT / "tools/acceptance_thermocalc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison_array(case, key, value):
    """Normalize only O06's documented metadata cache path; all other bytes remain exact."""
    if case == "O06" and key == "provider/metadata_json":
        return thermocalc_tools().normalize_metadata(value)
    return value


def run(args, *, cwd=None, timeout=120, env=None):
    """Run a bounded batch job and preserve diagnostic output on failure."""
    result = subprocess.run(args, cwd=cwd or OUTPUT, env=env, capture_output=True,
                            text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def run_tc_worker(args, *, env, timeout):
    """Bound a TC process tree so a hung licence/server cannot outlive its worker."""
    process = subprocess.Popen(args,cwd=OUTPUT,env=env,stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        stdout,stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
        else:
            process.kill()
        process.communicate(timeout=10)
        raise
    if process.returncode:
        raise RuntimeError(stdout + stderr)
    return stdout


def validate_expected(cases):
    """Normal-run validation without any research checkout or Git (plan §2).

    Checks, failing on the first problem:

    - each case's ``reference/<case>.{json,npz}`` against ``expected_artifacts`` (a case without an
      entry fails; O06/O08 frozen hashes are also checked in ``compare``);
    - each adopted input in ``adopted_inputs`` that exists must match its sha256; a missing one
      fails every case that ``requires`` it (``open-surrogate.npz`` is required by C24), and is
      never regenerated or skipped;
    - every ``tools/definitions`` file against ``docs/definitions-provenance.json``.

    ``private-inputs.json`` entries are checked by ``main`` for the enabled optional cases.
    Returns the manifest.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["expected_artifacts"]["cases"]
    for case in cases:
        assert case in expected, case + ": no expected_artifacts entry"
        for suffix in ("json", "npz"):
            path = OUTPUT / "reference" / f"{case}.{suffix}"
            assert path.exists(), f"{case}: expected artifact missing: reference/{path.name}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[case][suffix], \
                f"{case}: expected artifact changed: reference/{path.name}"
    for name, entry in manifest["adopted_inputs"]["files"].items():
        path = OUTPUT / "inputs" / name
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], "adopted input changed: inputs/" + name
        else:
            needed = sorted(set(entry["required_by"]) & set(cases))
            assert not needed, f"required input missing: inputs/{name} (needed by {', '.join(needed)}; never regenerated)"
    provenance = json.loads((PROJECT / "docs/definitions-provenance.json").read_text(encoding="utf-8"))
    for stem, entry in provenance["definitions"].items():
        data = (PROJECT / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], "definitions file changed: " + entry["file"]
    return manifest


def load_private_inputs():
    """The parsed local ``inputs/private-inputs.json``, or ``None`` when this machine has none."""
    path = OUTPUT / "inputs/private-inputs.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def private_input_problems(name, recorded):
    """Hash problems for one *available* private input (file or directory) under ``inputs/``.

    ``recorded`` is ``load_private_inputs()``. An input that is present on disk must be fully
    and correctly recorded, so this never passes silently: a missing ``private-inputs.json``, a
    missing entry, an empty or malformed ``sha256`` map, a digest that is not 64 lowercase hex
    characters, a record that does not name exactly the file (or every file of the directory),
    and a missing or changed member are each reported. Absent optional inputs are the caller's
    documented skip and are not checked here. Returns a list of messages (empty when valid).
    """
    import re
    location = OUTPUT / "inputs" / name
    assert location.exists(), "private_input_problems checks available inputs only: " + name
    if recorded is None:
        return ["private-inputs.json missing; cannot verify available input: " + name]
    entry = recorded.get(name)
    if entry is None:
        return ["input not hashed in private-inputs.json: " + name]
    digests = entry.get("sha256") if isinstance(entry, dict) else None
    if not isinstance(digests, dict) or not digests:
        return ["input hash record empty or malformed: " + name]
    malformed = sorted(str(key) for key, value in digests.items()
                       if not (isinstance(key, str) and isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)))
    if malformed:
        return [f"input hash record malformed: {name}/{key}" for key in malformed]
    if location.is_dir():
        base = location
        on_disk = {path.relative_to(location).as_posix() for path in location.rglob("*") if path.is_file()}
    else:
        base = location.parent
        on_disk = {name}
    problems = [f"input file not hashed: {name}/{key}" for key in sorted(on_disk - set(digests))]
    for key, digest in sorted(digests.items()):
        path = base / key
        if key not in on_disk or not path.is_file():
            problems.append(f"input file missing: {name}/{key}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"input changed: {name}/{key}")
    return problems


def validate_research_checkout(reference):
    """Historical actions only: the detached, clean frozen commit and its research source hashes."""
    assert reference is not None, "historical actions need --research <frozen research checkout>"
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
    """Isolate Python configuration; only O05/O08 need hash-stable provider compilation.

    NumPy seeding is deliberately absent: O05/O08 variation comes from unordered
    symbolic construction/CSE, not a random sampler. All other cases retain
    independently randomized interpreter hashes and untouched NumPy RNG state.
    """
    env = {key:value for key,value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               NUMBA_CACHE_DIR=str(OUTPUT / "numba"))
    if case in {"O05","O08"}:
        env["PYTHONHASHSEED"] = "0"
    return env


def worker(case, role, *, research=None, state=None, prepare=False, timeout=60, output=None, probe_thermocalc=False):
    """Execute an origin-checked worker; diagnostics use separate destinations.

    Package-role workers get no research root. Reference-role workers (historical only) need
    ``research``; their default ``reference/`` destination never overwrites an existing fixture,
    for any case, since every ``reference/`` pair is an adopted, hash-recorded artifact.
    """
    path = Path(output) if output is not None else OUTPUT / ("reference" if role == "reference" else "package") / case
    if role == "reference" and path.resolve() == (OUTPUT / "reference" / case).resolve():
        assert not any(path.with_suffix(suffix).exists() for suffix in (".json",".npz")), case + " fixture is frozen; refusing overwrite"
    args = [sys.executable, "-P", "-S", str(PROJECT / "tools/acceptance_worker.py"),
        "--role", role, "--case", case, "--output", str(path), "--artifacts", str(OUTPUT / "inputs")]
    if role == "reference":
        assert research is not None, "reference-role workers are historical and need --research"
        args.extend(["--root", str(research)])
    if state:
        args.extend(["--target", state["target"], "--host", state["host"]])
    if prepare:
        args.append("--prepare")
    if probe_thermocalc:
        args.append("--probe-thermocalc")
    env = worker_environment(case)
    started = time.monotonic()
    try:
        output = (run_tc_worker if case == "O06" else run)(args, env=env, timeout=timeout)
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


def npz_identity(generated, adopted):
    """Compare two ``.npz`` files array-wise, and their ZIP containers separately (F7, §3a).

    Array identity means the same key set and, per key, the same dtype, shape and bytes
    (``bit_identical``); arrays load with ``allow_pickle=False``. Container differences (member
    order, timestamps, compression, sizes) are reported under ``container`` as metadata only and
    never affect ``array_identical``. Nothing is written.
    """
    import numpy as np
    with np.load(generated, allow_pickle=False) as left, np.load(adopted, allow_pickle=False) as right:
        common = sorted(set(left.files) & set(right.files))
        arrays = {key:dict(dtype=str(right[key].dtype), shape=list(right[key].shape),
                           bit_identical=bit_identical(left[key], right[key])) for key in common}
        result = dict(generated_only=sorted(set(left.files) - set(right.files)),
                      adopted_only=sorted(set(right.files) - set(left.files)), arrays=arrays)
    def members(path):
        with zipfile.ZipFile(path) as archive:
            return [dict(name=item.filename, date_time=list(item.date_time), compress_type=item.compress_type,
                         compress_size=item.compress_size, file_size=item.file_size, crc=item.CRC)
                    for item in archive.infolist()]
    left_members, right_members = members(generated), members(adopted)
    result["container"] = dict(byte_identical=Path(generated).read_bytes() == Path(adopted).read_bytes(),
                               members_equal=left_members == right_members,
                               generated=left_members, adopted=right_members)
    result["array_identical"] = (not result["generated_only"] and not result["adopted_only"]
                                 and all(item["bit_identical"] for item in arrays.values()))
    return result


def regenerate_surrogate(state, directory):
    """Rebuild ``open-surrogate.npz`` with a package-role ``prepare`` into ``directory`` (F7).

    The worker builds ``_surrogate`` from ``tools/definitions`` (no research file is read) and
    saves to ``directory/prepare.npz``; the adopted ``inputs/open-surrogate.npz`` is only read.
    Returns ``npz_identity(generated, adopted)`` plus the generated path and worker status.
    """
    output = Path(directory) / "prepare"
    report = worker("prepare", "package", state=state, prepare=True, output=output)
    assert report["status"] == "PASS", report.get("error")
    assert report.get("case_definitions") == "tools/definitions", "prepare did not use package definitions"
    result = npz_identity(output.with_suffix(".npz"), OUTPUT / "inputs/open-surrogate.npz")
    result.update(generated=str(output.with_suffix(".npz")), worker_status=report["status"])
    return result


def _repeat_pair(case, role, *, folder, timeout, state=None, research=None):
    """Run two fresh unseeded ``role`` processes into ``folder/{a,b}`` and compare them bitwise."""
    import numpy as np
    started = time.monotonic()
    reports = []
    for repeat in ("a", "b"):
        remaining = timeout - (time.monotonic()-started)
        assert remaining > 0, "Unseeded repeatability budget exceeded"
        report = worker(case, role, research=research, state=state,
                        output=folder / repeat, timeout=min(300 if case == "O06" else 60,remaining))
        assert report["hash_seed"] is None and not report["numpy_seed_override"]
        reports.append(report)
    with np.load(folder / "a.npz") as left, np.load(folder / "b.npz") as right:
        assert set(left.files) == set(right.files), case
        failed = [key for key in left.files if not bit_identical(comparison_array(case,key,left[key]),comparison_array(case,key,right[key]))]
        arrays = len(left.files)
    assert reports[0].get("stop_status") == reports[1].get("stop_status"), case
    return dict(case=case, role=role, bit_identical=not failed, arrays=arrays, failed_keys=failed,
                elapsed_seconds=time.monotonic()-started), reports


def check_repeatability(case, *, state, directory, timeout=120):
    """Compare two fresh unseeded package-role processes without changing any fixture (plan §2).

    Every unaffected quick run and pytest contract is checked. A difference fails
    the gate; it must never trigger another seed exception automatically.
    """
    assert case not in {"O05","O08"}, "O05/O08 are documented hash-sensitive generation workflows"
    folder = Path(directory) / case
    result, _ = _repeat_pair(case, "package", folder=folder, timeout=timeout, state=state)
    (folder / "comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    assert not result["failed_keys"], ("Unseeded package role is not repeatable",case,result["failed_keys"])
    return result


def historical_repeat(case, *, research, directory, timeout=300):
    """Historical: two unseeded reference-role runs must equal each other and ``reference/<case>``.

    O05 (seeded generation) runs once under ``PYTHONHASHSEED=0``; O06/O08 are hash-frozen
    fixtures and are not repeated. Writes only under ``directory``.
    """
    import numpy as np
    if case in {"O06", "O08"}:
        return dict(case=case, skipped="hash-frozen fixture; not repeated")
    folder = Path(directory) / case
    if case == "O05":
        reports = [worker(case, "reference", research=research, output=folder / "a", timeout=timeout)]
        result = dict(case=case, role="reference", bit_identical=True, failed_keys=[])
    else:
        result, reports = _repeat_pair(case, "reference", folder=folder, timeout=timeout, research=research)
    expected = json.loads((OUTPUT / "reference" / (case + ".json")).read_text(encoding="utf-8"))
    with np.load(folder / "a.npz") as run_, np.load(OUTPUT / "reference" / (case + ".npz")) as adopted:
        keys_equal = set(run_.files) == set(adopted.files)
        differing = sorted(key for key in set(run_.files) & set(adopted.files)
                           if not bit_identical(comparison_array(case, key, run_[key]), comparison_array(case, key, adopted[key])))
        result.update(arrays=len(adopted.files))
    result.update(reference_keys_equal=keys_equal, reference_differing=differing,
                  reference_stop_status_equal=expected.get("stop_status") == reports[0].get("stop_status"),
                  reproduces_reference=keys_equal and not differing and expected.get("stop_status") == reports[0].get("stop_status"))
    (folder / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
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


MOLAR_INVENTORY_NAMES = ("molarInventoryData", "totalMolesData", "dependentMoleClosureErrorData")


def inventory_scale(key, expected):
    """Absolute-tolerance scale for one inventory-group array.

    Arrays in moles (component moles, total moles and the dependent-mole closure
    error of explicit-volume models) scale with length / minimum molar volume.
    Composition-based inventories (``inventoryData``, ``state/_initialInventory``,
    ``concData``) are composition times length in every model, so they scale with
    length only. Without a recorded minimum molar volume both reduce to length.
    """
    scale = expected.get("length_scale", 1.)
    if any(name in key for name in MOLAR_INVENTORY_NAMES):
        scale /= expected.get("minimum_molar_volume", 1.)
    return scale


def contract_scales(case, package, keys, manifest):
    """Validated per-model scales for a contract's ``model{i}/`` arrays (item 7).

    The package report's ``model_scales`` must equal the manifest's ``contract_model_scales``
    entry exactly, its keys must equal the npz's ``model{i}/`` prefixes (none missing, none
    extra), and every value must be finite and positive; anything else fails the case. The
    historical reference JSONs carry no scales and are not edited.
    """
    import math
    recorded = manifest["contract_model_scales"]["cases"][case]
    assert package.get("model_scales") == recorded, (case, "model_scales differ from contract_model_scales",
                                                     package.get("model_scales"), recorded)
    prefixes = {key.split("/", 1)[0] for key in keys if key.startswith("model")}
    assert prefixes == set(recorded), (case, "model prefixes differ from contract_model_scales",
                                       sorted(prefixes ^ set(recorded)))
    for prefix, scales in recorded.items():
        assert scales and all(isinstance(v, float) and math.isfinite(v) and v > 0 for v in scales.values()), (case, prefix, scales)
    return recorded


def compare(case, package, *, tolerance="exact", require=()):
    """Keep frozen gates; O06/O08 require bytes even in public mode.

    O06 compares every numerical array bitwise and all structural arrays after
    normalizing only metadata_json.thermocalc_config.cache_dir. Both new cases
    verify fixture hashes before comparison; existing tolerances are unchanged.
    Contracts (C25-C29) scale each ``model{i}/`` array by that model's validated
    ``contract_model_scales`` entry (``contract_scales``), exactly as run cases scale
    by their report's ``length_scale``/``minimum_molar_volume``.
    """
    import numpy as np
    ref = OUTPUT / "reference" / case
    got = OUTPUT / "package" / case
    if case in {"O06","O08"}:
        section = "thermocalc_sampling" if case == "O06" else "default_phase_sampling"
        frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))[section]["reference_sha256"]
        assert set(frozen) == {case+".json",case+".npz"}, case + " requires its frozen fixture hashes"
        for filename,digest in frozen.items():
            assert hashlib.sha256((ref.parent / filename).read_bytes()).hexdigest() == digest, case + " frozen fixture changed: " + filename
    expected = json.loads(ref.with_suffix(".json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert expected["status"] == package["status"] == "PASS"
    assert expected.get("stop_status") == package.get("stop_status")
    if case == "O08":
        assert expected["default_phase_contract"] == package["default_phase_contract"], "Default-phase labels/validity changed"
    for family in require:
        provider = package.get("providers", {}).get(family)
        if provider is not None:
            assert provider.startswith("moving_boundary."), (case, family, provider)
    results = {}
    structural = {}
    metrics = metric_tools()
    measured = {name:{} for name in metrics.QUANTITIES}
    with np.load(ref.with_suffix(".npz")) as reference, np.load(got.with_suffix(".npz")) as actual:
        assert set(reference.files) == set(actual.files), case
        models = contract_scales(case, package, reference.files, manifest) if case in manifest["contracts"] else {}
        for key in reference.files:
            wanted, value = comparison_array(case,key,reference[key]), comparison_array(case,key,actual[key])
            group = quantity(key)
            if case == "O06" and (wanted.dtype.kind not in "bifcu" or value.dtype.kind not in "bifcu"):
                ok = bit_identical(wanted,value)
                structural[key] = dict(bit_identical=ok)
                results[group] = results.get(group,True) and ok
                if not ok:
                    results.setdefault("failed_keys",[]).append(key)
                continue
            settings = manifest["public_tolerances"][group]
            scaling = models.get(key.split("/", 1)[0], expected)
            scale = 1.
            if group == "interface" or (group == "analytic_error" and key == "exact/interface_error"):
                scale = scaling.get("length_scale", 1.)
            elif group == "accepted_times":
                scale = max(1., expected.get("requested_duration", 1.))
            elif group == "inventory":
                scale = inventory_scale(key, scaling)
            measured[group][key] = metrics.array_metrics(wanted,value,
                rtol=settings["rtol"] if tolerance == "public" and case not in {"O06","O08"} else 0.,
                atol=settings["atol"]*scale if tolerance == "public" and case not in {"O06","O08"} else 0.)
            ok = wanted.shape == value.shape
            if ok:
                ok = np.array_equal(np.isnan(wanted), np.isnan(value))
            if ok and case in {"O06","O08"}:
                ok = bit_identical(wanted,value)
            elif ok and tolerance == "exact":
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
        bit_identical=all(stats["bit_identical"] for stats in quantities.values() if stats["applicable"])
            and all(item["bit_identical"] for item in structural.values()), quantities=quantities)
    if structural:
        record["structural_arrays"] = structural
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


def contract_case(case, state=None, *, tolerance="exact", require=()):
    """Entry point for package pytest: compare an isolated contract with its fixture."""
    if state is None:
        state = json.loads((OUTPUT / "state.json").read_text(encoding="utf-8"))
    validate_expected([case])
    if state.get("repeatability_directory"):
        check_repeatability(case, state=state, directory=state["repeatability_directory"])
    report = worker(case, "package", state=state)
    result = compare(case, report, tolerance=tolerance, require=require)
    assert not result.get("failed_keys"), result
    return result


def run_pytest(state, *, tolerance, require, timeout=180):
    """Run package contracts and kernel/tooling tests against the installed targets."""
    state.update(tolerance=tolerance, require=list(require))
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
    tests = ["test_acceptance_contracts.py", "test_tridiagonal.py", "test_batch_safety.py", "test_verification.py", "test_compatibility.py", "test_acceptance_reproducibility.py", "test_acceptance_metrics.py", "test_acceptance_reporting.py", "test_pycalphad_provider.py", "test_default_phase_acceptance.py", "test_thermocalc_provider.py", "test_thermocalc_acceptance.py"]
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
    """Run quick/full acceptance (no research), or an explicit historical capture/repeat."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("quick", "full", "setup", "historical"))
    parser.add_argument("step", nargs="?", choices=("capture", "repeat"), help="historical actions only")
    parser.add_argument("--research", type=Path, help="frozen research checkout; historical actions only")
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
    historical = args.action == "historical"
    if historical and (args.step is None or args.research is None):
        parser.error("historical needs a step (capture or repeat) and --research <frozen research checkout>")
    if not historical and (args.step is not None or args.research is not None):
        parser.error("normal actions take no historical step and no research checkout")
    # The parent is invoked after required Conda activation. The cloned runtime
    # inherits those native DLL paths; prepend the clone's equivalent locations.
    if not historical:
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
    manifest = validate_research_checkout(args.research) if historical else json.loads(MANIFEST.read_text(encoding="utf-8"))
    full = args.action == "full" or args.tier == "full"
    cases = args.cases or list(manifest["quick"]) + ["C32", "LEGACY"]
    # Real sampling fit the approved minute pilot; sampling runs afresh in O05.
    if not args.cases and (OUTPUT / "inputs/FeCrNi.tdb").exists():
        pilot = OUTPUT / "pilot/O05-sampling.json"
        if full or (pilot.exists() and json.loads(pilot.read_text())["elapsed_seconds"] <= 60):
            cases.append("O05")
    skipped = {}
    default_case = manifest.get("default_phase_sampling")
    if default_case and not args.cases and (full or default_case["tier"] == "quick"):
        if (OUTPUT / "inputs" / default_case["database_input"]).exists():
            cases.append(default_case["case"])
        else:
            skipped[default_case["case"]] = "database input missing"
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
    # Private hashes never enter committed evidence. Every available input of an enabled case
    # must be fully recorded and unchanged (a missing record fails); absent optional inputs keep
    # their documented skips above. Corrupted or unrecorded inputs are never disguised as
    # missing licensed/provider coverage.
    enabled_inputs = {"O01":"exb5.json", "O02":"exb5.json", "O03a":"femnc.json",
        "O03b":"femnc.json", "O04":"fecrni.surrogate", "O05":"FeCrNi.tdb", "O07":"fec.json", "O08":"FeCrNi.tdb"}
    recorded = load_private_inputs()
    for case in cases:
        name = enabled_inputs.get(case)
        if name is not None and (OUTPUT / "inputs" / name).exists():
            problems = private_input_problems(name, recorded)
            assert not problems, case + " input not verified: " + "; ".join(problems)
    if args.action == "setup":
        print(json.dumps(setup_targets()))
        return 0
    if historical and not args.cases:
        cases = cases + manifest["contracts"]
    # Everything this run may compare (or historically repeat) is hash-validated first; O06 is
    # included for a complete full run even before its availability probe.
    checked = []
    if not (historical and args.step == "capture"):
        checked = list(dict.fromkeys(cases + ([] if args.cases or historical else manifest["contracts"])
                                     + (["O06"] if full and not args.cases and not historical else [])))
        validate_expected(checked)
    started = time.monotonic()
    started_timestamp = time.time()
    results, failures = {}, []
    state = None
    if not historical:
        state = json.loads((OUTPUT / "state.json").read_text(encoding="utf-8")) if args.reuse_targets else setup_targets()
        state["repeatability_directory"] = tempfile.mkdtemp(prefix="unseeded-", dir=OUTPUT)
    elif args.step == "capture" and args.cases != ["O06"]:
        # Historical prepare writes to scratch; it never overwrites the adopted C24 input.
        worker("prepare", "reference", research=args.research, prepare=True,
               output=Path(tempfile.mkdtemp(prefix="prepare-", dir=OUTPUT)) / "prepare")
    if historical and args.step == "repeat":
        directory = Path(tempfile.mkdtemp(prefix="historical-repeat-", dir=OUTPUT))
        for case in cases:
            try:
                result = historical_repeat(case, research=args.research, directory=directory)
            except Exception as exc:
                result = dict(case=case, error=str(exc)[-2000:])
            results[case] = result
            ok = "skipped" in result or (result.get("bit_identical") and result.get("reproduces_reference"))
            if not ok:
                failures.append(case)
            print(case, "SKIPPED " + result["skipped"] if "skipped" in result else ("PASS" if ok else "FAIL"),
                  result.get("error", ""), flush=True)
        validate_research_checkout(args.research)
        validate_expected(checked)
        summary = dict(action="historical repeat", research=str(args.research), directory=str(directory),
                       cases=results, failures=failures, elapsed_seconds=time.monotonic()-started)
        (OUTPUT / "historical-repeat-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Historical repeat: {len(results)} cases, {len(failures)} failures, {summary['elapsed_seconds']:.2f}s", flush=True)
        return 1 if failures else 0
    if full and not args.cases and not historical:
        probe_dir = Path(tempfile.mkdtemp(prefix="tc-availability-",dir=OUTPUT))
        try:
            probe = worker("O06","package",state=state,
                           output=probe_dir / "O06",probe_thermocalc=True,timeout=60)
            assert probe["status"] == "PASS", "TC availability probe failed"
            availability = probe["thermocalc_availability"]
        except subprocess.TimeoutExpired:
            availability = dict(available=False,reason="TC session/licence availability probe timed out (60 s)")
        if availability["available"]:
            cases.append("O06")
        else:
            skipped["O06"] = availability["reason"]
    for case in cases:
        limit = 1200 if full else 600
        remaining = limit - (time.monotonic() - started)
        if remaining <= 0:
            failures.append("overall timeout")
            break
        try:
            report = worker(case, "reference" if historical else "package",
                            research=args.research, state=state,
                            timeout=min(300 if case == "O06" else (180 if case.startswith("F") else 60), max(1,remaining)))
            if not historical and case in set(manifest["quick"]) | {"C32","LEGACY"} | ({"O06"} if full and manifest["thermocalc_sampling"]["repeat_unseeded"] else set()):
                check_repeatability(case, state=state, directory=state["repeatability_directory"],
                    timeout=max(1,limit-(time.monotonic()-started)))
            result = {"capture":True} if historical else compare(case, report,
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
            tests = run_pytest(state, tolerance=args.tolerance,
                               require=args.require_family, timeout=max(1,remaining))
        except Exception as exc:
            failures.append("pytest")
            print("PYTEST FAIL", str(exc)[-3000:], flush=True)
    for case, reason in skipped.items():
        print(case, "SKIPPED", reason, flush=True)
    if historical:
        validate_research_checkout(args.research)
    else:
        validate_expected(checked)
    summary = dict(action=args.action if not historical else "historical capture", cases=results, failures=failures, skipped=skipped,
                   pytest=tests, elapsed_seconds=time.monotonic()-started)
    if args.action in {"quick","full"}:
        summary["deviations"] = report_deviations(args.action,cases,
            contracts=manifest["contracts"] if tests else (),skipped=skipped,
            started=started_timestamp,tolerance=args.tolerance,
            record=args.record and not failures,evidence=summary)
    if state and state.get("repeatability_directory"):
        checks = [json.loads(path.read_text()) for path in Path(state["repeatability_directory"]).glob("*/comparison.json")]
        summary["unseeded_repeatability"] = {entry["case"]:entry for entry in checks}
        print(f"Unseeded package repeatability: {len(checks)} cases checked twice; "
              f"{sum(not entry['bit_identical'] for entry in checks)} differences", flush=True)
    (OUTPUT / (("historical-capture" if historical else args.action) + "-summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(results)} cases, {len(failures)} failures, {summary['elapsed_seconds']:.2f}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
