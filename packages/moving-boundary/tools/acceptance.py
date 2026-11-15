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

Approval (plan §5) is the one action that uses Git, and only to check the development repository:
``approve --cases ... --reason "LEDGERS Rxx"`` runs from a clean package tree, enforces the
correctness gates on every case (package tests, contract assertion groups, two bit-identical
package runs, ``approval_criteria``) and the public-tolerance comparison of every unlisted case,
reports the listed cases' deviations, and promotes their candidate artifacts crash-safely
(``tools/acceptance_approval.py``). ``approve --rollback ID --reason ...`` points those cases back.
Normal runs read whichever reference is in effect (``expected_reference``) and stay Git-free.

Layout corrections (``tools/acceptance_layout.py``, rule ``r09-data-tail``) are an additional
strict proof, never a relaxation. ``quick|full --layout-correction RULE`` is a non-promoting check:
every gate of the normal run still runs (repeatability, contract assertion groups, the package
tests), normal comparisons are recorded and reported, and each case and contract is classified
against its selected reference as ``identical``, ``tail-removal`` or ``other``; any ``other``, or a
case-versus-case settings difference of a corrected mapped case, fails the run. ``approve
--layout-correction RULE --cases ...`` adopts such a correction (see ``approve``).
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


@lru_cache(maxsize=1)
def approval_tools():
    """Load the approval module (reference selection, validation, promotion); it never runs Git."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("acceptance_approval", Path(__file__).resolve().parent / "acceptance_approval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def criteria_tools():
    """Load the approval correctness criteria (``approval_criteria`` measurements)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("acceptance_criteria", Path(__file__).resolve().parent / "acceptance_criteria.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def layout_tools():
    """Load the strict layout-correction rules (``acceptance_layout``; pure, no I/O)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("acceptance_layout", Path(__file__).resolve().parent / "acceptance_layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def settings_tools():
    """Load the example-settings comparison (``example_settings.differences`` and ``BOOKKEEPING``)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("example_settings", Path(__file__).resolve().parent / "example_settings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_reference(case, manifest=None):
    """The expected artifacts ``case`` is compared with (one helper for every comparison).

    Used by ``compare``, ``contract_assertions`` and ``acceptance_examples.compare_case``. With no
    ``approved_references`` section, or the case not approved, this is ``reference/<case>`` exactly
    as before (O06/O08 also carry their frozen hashes); an approved case reads its approval's
    immutable folder and recorded hashes. See ``acceptance_approval.expected_reference``.
    """
    if manifest is None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return approval_tools().expected_reference(case, manifest, OUTPUT)


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


def validate_expected(cases, *, required_inputs_for=None):
    """Normal-run validation without any research checkout or Git (plan §2).

    Checks, failing on the first problem:

    - each case's ``reference/<case>.{json,npz}`` against ``expected_artifacts`` (a case without an
      entry fails; O06/O08 frozen hashes are also checked in ``compare``);
    - each adopted input in ``adopted_inputs`` that exists must match its sha256; a missing one
      fails every case of ``required_inputs_for`` (default: ``cases``) that ``requires`` it
      (``open-surrogate.npz`` is required by C24), and is never regenerated or skipped. Example
      acceptance passes ``()``: it compares with the reference artifacts but does not rerun the
      reference cases, so their original inputs are not needed (follow-up 5);
    - every ``tools/definitions`` file against ``docs/definitions-provenance.json``;
    - ``approved_references`` when present (``acceptance_approval.validate``): every approval
      record and its committed report hash, and the approved files of each of ``cases`` in effect.
      No Git: ancestry is checked only when ``approve`` creates an approval.

    ``reference/<case>`` is checked even for an approved case: it is never rewritten.
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
            needed = sorted(set(entry["required_by"]) & set(cases if required_inputs_for is None else required_inputs_for))
            assert not needed, f"required input missing: inputs/{name} (needed by {', '.join(needed)}; never regenerated)"
    provenance = json.loads((PROJECT / "docs/definitions-provenance.json").read_text(encoding="utf-8"))
    for stem, entry in provenance["definitions"].items():
        data = (PROJECT / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], "definitions file changed: " + entry["file"]
    approval_tools().validate(manifest, cases, project=PROJECT, output=OUTPUT)
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


def _repeat_pair(case, role, *, folder, timeout, state=None, research=None, seed=None):
    """Run two fresh ``role`` processes into ``folder/{a,b}`` and compare them bitwise.

    Unseeded by default; ``seed="0"`` is only for O05/O08, whose workers always run under
    ``PYTHONHASHSEED=0`` (``worker_environment``), and asserts that seed was used.
    """
    import numpy as np
    started = time.monotonic()
    reports = []
    for repeat in ("a", "b"):
        remaining = timeout - (time.monotonic()-started)
        assert remaining > 0, "Unseeded repeatability budget exceeded"
        cap = 300 if case == "O06" else (180 if case.startswith("F") else 60)   # same caps as a gate run
        report = worker(case, role, research=research, state=state,
                        output=folder / repeat, timeout=min(cap,remaining))
        assert report["hash_seed"] == seed and not report["numpy_seed_override"]
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

    The expected artifacts are whichever ``expected_reference`` selects (``reference/<case>``
    unless an approval is in effect); hashes it returns (O06/O08 frozen, or approved) are checked
    first. The local deviation record names that source.
    """
    import numpy as np
    got = OUTPUT / "package" / case
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = expected_reference(case, manifest)
    for suffix, digest in (selected["sha256"] or {}).items():
        path = selected[suffix]
        label = " frozen fixture changed: " if selected["source"] == "reference" else " approved reference changed: "
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, case + label + path.name
    expected = json.loads(selected["json"].read_text(encoding="utf-8"))
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
    with np.load(selected["npz"]) as reference, np.load(got.with_suffix(".npz")) as actual:
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
    record = dict(case=case,reference=selected["source"],tolerance_mode=tolerance,passed=not results.get("failed_keys"),
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


def contract_assertions(case, report, manifest=None):
    """Correctness gate of one contract (C25-C29): its assertion groups ran and passed.

    The assertion groups are the contract's test functions, executed inside the package worker
    (an assertion failure fails the worker). Required: status ``PASS``, stop status
    ``contract_assertions_passed``, the same non-empty list of assertion-group names as the
    expected reference records, and a ``contracts/assertion_groups`` count equal to its length.
    Separate from the array comparison (``compare``). Returns the group names.
    """
    import numpy as np
    expected = json.loads(expected_reference(case, manifest)["json"].read_text(encoding="utf-8"))
    assert report.get("status") == "PASS" and report.get("stop_status") == "contract_assertions_passed", \
        (case, "contract assertions did not pass", report.get("status"), report.get("stop_status"))
    groups = report.get("contracts")
    assert groups and groups == expected.get("contracts"), (case, "assertion groups differ", groups, expected.get("contracts"))
    with np.load(OUTPUT / "package" / (case + ".npz")) as data:
        assert int(data["contracts/assertion_groups"]) == len(groups), (case, "assertion-group count differs")
    return groups


def contract_case(case, state=None, *, tolerance="exact", require=()):
    """Entry point for package pytest: a contract's assertion groups, then its array comparison.

    ``contract_assertions`` always gates. The comparison with the expected reference gates too,
    except for a contract listed in ``state["approval_candidates"]``: its differences are then
    recorded (``.deviations.json``) and reported, not failed. Only ``approve`` (its listed
    contracts) and a ``--layout-correction`` run (every contract, each then gated by its strict
    layout verdict instead) set that list; normal runs always clear it.
    """
    if state is None:
        state = json.loads((OUTPUT / "state.json").read_text(encoding="utf-8"))
    validate_expected([case])
    if state.get("repeatability_directory"):
        check_repeatability(case, state=state, directory=state["repeatability_directory"])
    report = worker(case, "package", state=state)
    contract_assertions(case, report)
    result = compare(case, report, tolerance=tolerance, require=require)
    if case in (state.get("approval_candidates") or ()):
        return dict(result, comparison="reported: approval candidate")
    assert not result.get("failed_keys"), result
    return result


def run_pytest(state, *, tolerance, require, timeout=180, full_suite=False):
    """Run package contracts and kernel/tooling tests against the installed targets.

    Normal quick/full runs use the fixed list below and are portable: tests marked ``git`` (they
    run Git themselves, e.g. the historical ``test_frozen_content_guard``) are deselected, so a
    normal gate never invokes Git (plan §7, follow-up 7). ``full_suite=True`` (``approve``, which
    uses Git anyway) runs the whole ``tests/`` directory instead, ``git`` tests included, with the
    contracts configured and ``MB_EXAMPLE_TDB`` set, so every package test, including ones added
    later, gates an approval. Either way the run must have tests and no failures, errors or skips
    (``slow`` tests stay deselected).
    """
    state.update(tolerance=tolerance, require=list(require))
    configured = OUTPUT / "state.json"
    configured.write_text(json.dumps(state, indent=2), encoding="utf-8")
    env = {key:value for key,value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(MB_ACCEPTANCE_STATE=str(configured), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([state["target"], state["host"]]),
               NUMBA_CACHE_DIR=str(OUTPUT / "numba"))
    if full_suite:
        env["MB_EXAMPLE_TDB"] = str(OUTPUT / "inputs/FeCrNi.tdb")
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
    tests = ["test_acceptance_contracts.py", "test_tridiagonal.py", "test_batch_safety.py", "test_verification.py", "test_compatibility.py", "test_acceptance_reproducibility.py", "test_acceptance_metrics.py", "test_acceptance_reporting.py", "test_pycalphad_provider.py", "test_default_phase_acceptance.py", "test_thermocalc_provider.py", "test_thermocalc_acceptance.py", "test_acceptance_approval.py"]
    selection = [str(PROJECT / "tests")] if full_suite else [
        *[str(PROJECT / "tests" / name) for name in tests],
        str(PROJECT / "tests/test_bridge.py") + "::test_namespace_import_is_lazy",
        str(PROJECT / "tests/test_bridge.py") + "::test_binary_example",
        str(PROJECT / "tests/test_bridge.py") + "::test_missing_fork_error"]
    args = [sys.executable, "-I", "-S", "-c", code, state["target"], state["host"], *selection,
            "--import-mode=importlib", "--basetemp", str(OUTPUT / "pytest"), "-q",
            "-m", "not slow" if full_suite else "not slow and not git",
            "--junitxml", str(OUTPUT / "pytest.xml")]
    output = run(args, env=env, timeout=timeout)
    (OUTPUT / "pytest.log").write_text(output, encoding="utf-8")
    print(output[-1000:], flush=True)
    import xml.etree.ElementTree as ET
    suites = ET.parse(OUTPUT / "pytest.xml").getroot().findall(".//testsuite")
    counts = {key:sum(int(suite.get(key,0)) for suite in suites) for key in ("tests","failures","errors","skipped")}
    assert counts["tests"] and not any(counts[k] for k in ("failures","errors","skipped")), counts
    return counts


PRIVATE_CASE_INPUTS = {"O01":"exb5.json", "O02":"exb5.json", "O03a":"femnc.json",
    "O03b":"femnc.json", "O04":"fecrni.surrogate", "O05":"FeCrNi.tdb", "O07":"fec.json", "O08":"FeCrNi.tdb"}


def normal_cases(manifest, *, full, explicit=None):
    """The run cases of a quick or full gate, and the documented skips (no side effects).

    ``explicit`` (``--cases``) runs exactly those. Otherwise: the quick list, C32 and LEGACY;
    O05 when the TDB is present (quick only if its pilot fit a minute); O08 when its database is
    present; full adds the F cases and the cached O cases whose inputs exist. O06 is added later,
    after its availability probe.
    """
    cases = list(explicit) if explicit else list(manifest["quick"]) + ["C32", "LEGACY"]
    # Real sampling fit the approved minute pilot; sampling runs afresh in O05.
    if not explicit and (OUTPUT / "inputs/FeCrNi.tdb").exists():
        pilot = OUTPUT / "pilot/O05-sampling.json"
        if full or (pilot.exists() and json.loads(pilot.read_text())["elapsed_seconds"] <= 60):
            cases.append("O05")
    skipped = {}
    default_case = manifest.get("default_phase_sampling")
    if default_case and not explicit and (full or default_case["tier"] == "quick"):
        if (OUTPUT / "inputs" / default_case["database_input"]).exists():
            cases.append(default_case["case"])
        else:
            skipped[default_case["case"]] = "database input missing"
    if not explicit and not (OUTPUT / "inputs/FeCrNi.tdb").exists():
        skipped["O05"] = "database input missing"
    if full and not explicit:
        cases.extend(manifest["full_drift"])
        private = {"O01":"exb5.json","O02":"exb5.json","O03a":"femnc.json","O03b":"femnc.json", "O04":"fecrni.surrogate","O07":"fec.json"}
        for case, name in private.items():
            if (OUTPUT / "inputs" / name).exists():
                cases.append(case)
            else:
                skipped[case] = "cached input missing"
    return cases, skipped


def check_private_inputs(cases):
    """Every available private input of an enabled case must be fully recorded and unchanged.

    Private hashes never enter committed evidence. A missing record fails; absent optional
    inputs keep their documented skips (``normal_cases``). Corrupted or unrecorded inputs are
    never disguised as missing licensed/provider coverage.
    """
    recorded = load_private_inputs()
    for case in cases:
        name = PRIVATE_CASE_INPUTS.get(case)
        if name is not None and (OUTPUT / "inputs" / name).exists():
            problems = private_input_problems(name, recorded)
            assert not problems, case + " input not verified: " + "; ".join(problems)


# Approval (plan §5). Approve repeats every case and runs the whole tests/ suite, more than a
# normal gate, so it has its own budget.
APPROVE_BUDGET_SECONDS = {"quick": 900, "full": 1500}
PRIVATE_SETTINGS_CASES = frozenset({"O04", "O05"})   # example settings hashed only in private-inputs.json


def repository_checks():
    """Approval only (Git): the package tree is clean, and every recorded approval is an ancestor of HEAD.

    "Clean" means ``git status --porcelain --untracked-files=all`` reports nothing under the package
    directory (ignored files such as ``.verification/`` don't count). Returns the HEAD sha.
    """
    status = run(["git", "status", "--porcelain", "--untracked-files=all", "--", "."], cwd=PROJECT, timeout=60)
    assert not status.strip(), "approve needs a clean package tree; commit or set aside:\n" + status
    head = run(["git", "rev-parse", "HEAD"], cwd=PROJECT, timeout=30).strip()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for approval_id, record in ((manifest.get("approved_references") or {}).get("approvals") or {}).items():
        try:
            run(["git", "merge-base", "--is-ancestor", record["package_commit"], head], cwd=PROJECT, timeout=30)
        except RuntimeError as exc:
            raise AssertionError(f"approval {approval_id}: package_commit {record['package_commit']} is not an ancestor of HEAD {head}") from exc
    return head


def candidate_matches_repeat(case, state):
    """A listed case's candidate (``package/<case>``) must equal its first repeat run array for array."""
    import numpy as np
    folder = Path(state["repeatability_directory"]) / case
    with np.load(OUTPUT / "package" / (case + ".npz")) as candidate, np.load(folder / "a.npz") as repeat:
        assert set(candidate.files) == set(repeat.files), (case, "candidate and repeat key sets differ")
        differing = [key for key in candidate.files
                     if not bit_identical(comparison_array(case, key, candidate[key]), comparison_array(case, key, repeat[key]))]
    assert not differing, (case, "candidate differs from its repeat runs", differing)


def approval_repeat(case, *, listed, manifest, state, timeout):
    """Approval's repeatability gate: two package runs bit-identical, for every run case (plan §5).

    Every case in every tier (user decision after M7b): unseeded pairs, except O05/O08, which repeat
    as a seeded pair under their documented ``PYTHONHASHSEED=0`` rule. A listed case's candidate
    must also equal its repeats.
    """
    if case in {"O05", "O08"}:
        folder = Path(state["repeatability_directory"]) / case
        result, _ = _repeat_pair(case, "package", folder=folder, timeout=timeout, state=state, seed="0")
        (folder / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        assert not result["failed_keys"], ("Seeded package role is not repeatable", case, result["failed_keys"])
    else:
        result = check_repeatability(case, state=state, directory=state["repeatability_directory"], timeout=timeout)
    if case in listed:
        candidate_matches_repeat(case, state)
    return result


def criteria_results(case, manifest):
    """``approval_criteria`` evaluated on the candidate ``package/<case>.npz`` (unmeasurable fails)."""
    import numpy as np
    tools = criteria_tools()
    if not tools.criteria_for(case, manifest["approval_criteria"]):
        return []
    with np.load(OUTPUT / "package" / (case + ".npz"), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return tools.evaluate(case, arrays, manifest["approval_criteria"])


def capture_example_settings(cases, state, directory):
    """Package-captured settings for each listed case an example maps to (``example_worker --settings-only``).

    Built from ``tools/definitions`` on the installed wheel, never solved, written under
    ``directory``. Returns ``{case: settings path}``.
    """
    examples = json.loads((PROJECT / "tools/example_manifest.json").read_text(encoding="utf-8"))
    mapped = {}
    for example, entry in examples["cases"].items():
        if entry["reference"] in cases:
            mapped.setdefault(entry["reference"], example)
    captured = {}
    for case, example in sorted(mapped.items()):
        output = Path(directory) / case
        run([sys.executable, "-P", "-S", str(PROJECT / "tools/example_worker.py"), "--role", "package",
             "--case", example, "--examples", str(PROJECT / "examples"), "--artifacts", str(OUTPUT / "inputs"),
             "--output", str(output), "--target", state["target"], "--host", state["host"], "--settings-only"],
            env=worker_environment(case), timeout=120)
        captured[case] = output.with_suffix(".settings.json")
    return captured


def layout_verdict(case, rule, manifest=None, *, candidate=None):
    """Strict layout classification of a package output against its previously selected reference.

    The selected reference (``expected_reference``) is hash-checked first (its approved or frozen
    hashes, else ``expected_artifacts``), both reports must have status ``PASS`` and the same stop
    status, and the arrays are classified by ``acceptance_layout.classify`` with only O06's
    documented metadata normalization. ``candidate`` defaults to ``package/<case>``. Also writes
    the full verdict to ``<candidate>.layout.json`` (ignored local output). Returns the verdict.
    """
    import numpy as np
    manifest = manifest or json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = expected_reference(case, manifest)
    digests = selected["sha256"] or manifest["expected_artifacts"]["cases"][case]
    for suffix in ("json", "npz"):
        assert hashlib.sha256(selected[suffix].read_bytes()).hexdigest() == digests[suffix], \
            f"{case}: selected reference changed: {selected[suffix].name}"
    candidate = Path(candidate) if candidate is not None else OUTPUT / "package" / case
    expected = json.loads(selected["json"].read_text(encoding="utf-8"))
    actual = json.loads(candidate.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(selected["npz"], allow_pickle=False) as reference, np.load(candidate.with_suffix(".npz"), allow_pickle=False) as got:
        result = layout_tools().classify(case, {key: reference[key] for key in reference.files},
                                         {key: got[key] for key in got.files}, rule=rule, normalize=comparison_array)
    if not (expected.get("status") == actual.get("status") == "PASS"):
        result.update(verdict="other", problems=result["problems"] + ["status is not PASS on both sides"])
    if expected.get("stop_status") != actual.get("stop_status"):
        result.update(verdict="other", problems=result["problems"] + ["stop status differs"])
    result["reference"] = selected["source"]
    candidate.with_suffix(".layout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def example_mapped_cases():
    """``{case: first example}`` for every case an example maps to (``example_manifest.json``)."""
    examples = json.loads((PROJECT / "tools/example_manifest.json").read_text(encoding="utf-8"))
    mapped = {}
    for example, entry in examples["cases"].items():
        mapped.setdefault(entry["reference"], example)
    return mapped


def selected_case_settings(case, manifest):
    """The settings file currently selected for a mapped ``case`` and the digest it must have.

    Approved with captured settings: the approval's file and recorded hash. Otherwise (frozen
    reference, or an approval that carried private settings forward): the adopted
    ``example_settings/<case>.settings.json`` with its committed ``settings_sha256`` (C04, F04,
    C16, C24, ...) or, for O04/O05, its digest in the local ``private-inputs.json``. Returns
    ``(path, digest, private)``; ``digest`` is ``None`` when no hash is recorded.
    """
    selected = expected_reference(case, manifest)
    if selected["source"] != "reference" and not selected["settings_carried_forward"]:
        assert selected["settings"] is not None, f"{case}: approval {selected['source']} has no captured settings"
        return selected["settings"], selected["settings_sha256"], False
    path = OUTPUT / "example_settings" / f"{case}.settings.json"
    examples = json.loads((PROJECT / "tools/example_manifest.json").read_text(encoding="utf-8"))
    if path.name in examples["settings_sha256"]:
        return path, examples["settings_sha256"][path.name], False
    entry = (load_private_inputs() or {}).get("example_settings")
    recorded = entry.get("sha256") if isinstance(entry, dict) else None
    return path, (recorded or {}).get(path.name) if isinstance(recorded, dict) else None, True


def check_case_settings(cases, state, directory, manifest):
    """Case-versus-case settings proof for a layout correction or ``--verify-private-settings`` (never relaxed).

    For each mapped case of ``cases``: hash-validate the currently selected settings first
    (``selected_case_settings``), capture fresh package settings through the definitions-based
    ``--settings-only`` path (``capture_example_settings``) and require
    ``example_settings.differences(selected, captured, allowed=())`` to be empty. Only the named
    ``BOOKKEEPING`` attributes are skipped; the example-versus-case exceptions of
    ``example_manifest.allowed_setting_differences`` do not apply. Captures of private cases
    (O04/O05) are deleted again here, so no private capture outlives the check. That holds on
    every exit: the capture step is inside the cleanup, and the capture files are removed by
    their known names ``<directory>/<case>.settings.json``, so a capture that fails or is
    interrupted after an earlier private case was written (or while one is being written) leaves
    nothing behind, and the original exception propagates unchanged. Only those capture files
    are removed; adopted settings, public captures and other files are never touched.

    Returns ``(public, private, problems)``: ``{case: captured path}`` for public cases, the
    private cases that passed, and the failures (empty when every mapped case matches).
    """
    settings = settings_tools()
    mapped = sorted(set(cases) & set(example_mapped_cases()))
    problems, public, private = [], {}, []
    selections = {}
    for case in mapped:
        path, digest, is_private = selected_case_settings(case, manifest)
        if not path.exists():
            problems.append(f"{case}: selected settings missing: {path.name}")
        elif digest is None:
            problems.append(f"{case}: selected settings not hashed: {path.name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"{case}: selected settings changed: {path.name}")
        else:
            selections[case] = (path, is_private)
    if problems:
        return public, private, problems
    captured = {}
    try:
        captured = capture_example_settings(mapped, state, directory)
        for case in mapped:
            path, is_private = selections[case]
            found = settings.differences(json.loads(path.read_text(encoding="utf-8")),
                                         json.loads(Path(captured[case]).read_text(encoding="utf-8")), allowed=())
            problems.extend(f"{case}: setting differs: {name}" for name in found)
            if not found:
                (private.append(case) if is_private else public.__setitem__(case, captured[case]))
    finally:
        # ``captured`` is empty when the capture step itself raised, so the files are also named directly.
        for case in mapped:
            if selections[case][1]:
                for path in {Path(directory) / f"{case}.settings.json", *([Path(captured[case])] if case in captured else [])}:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError as exc:   # never mask the exception being propagated
                        print(f"WARNING: private settings capture not removed: {path.name}: {exc}", flush=True)
    return public, private, problems


def public_deviations(case, record):
    """Committed per-array deviations of a listed case against its previous expected artifact.

    TC-derived cases (``acceptance_metrics.PRIVATE_CASES``) and the private-settings cases
    (``PRIVATE_SETTINGS_CASES``: O04, and O05 with its private database) keep their numbers in
    ignored output; the committed entry is verdicts only.
    """
    metrics = metric_tools()
    if case in metrics.PRIVATE_CASES or case in PRIVATE_SETTINGS_CASES:
        return dict(private_metrics=True, passed_public_tolerance=record["passed"], bit_identical=record["bit_identical"])
    arrays = {}
    for quantity, stats in record["quantities"].items():
        for key, item in (stats.get("arrays") or {}).items():
            arrays[key] = dict(quantity=quantity, **{name: item.get(name) for name in (
                "bit_identical", "max_absolute_deviation", "max_relative_deviation", "max_tolerance_fraction")})
    return dict(reference=record.get("reference"), passed_public_tolerance=record["passed"],
                bit_identical=record["bit_identical"], arrays=dict(sorted(arrays.items())))


def public_criteria(case, results):
    """Criteria results for the committed report (values of TC-derived cases are not exported)."""
    if case in metric_tools().PRIVATE_CASES:
        return [dict(criterion=item["criterion"], passed=item["passed"], private_metrics=True) for item in results]
    return [{key: item.get(key) for key in ("criterion", "value", "limit", "passed", "error") if key in item} for item in results]


def approve(listed, *, reason, tier="quick", require=("binary", "ternary", "three_phase"), layout_correction=None,
            verify_private_settings=False):
    """Approve the listed cases' package outputs as their new expected references (plan §5).

    From a clean package tree (``repository_checks``), run the whole ``tier`` gate at public
    tolerance and stop at the first failure (nothing is written then):

    - correctness gates for every case, listed or not: two bit-identical package runs
      (``approval_repeat``; a listed candidate must equal its repeats), ``approval_criteria``
      (``criteria_results``), the whole package ``tests/`` suite on the freshly built targets
      (``run_pytest(full_suite=True)``), incl. the contract assertion groups;
    - reference comparisons: every unlisted case must pass at public tolerance; a listed case's
      differences against its previous expected artifact are reported, not failed.

    The installed targets are always built afresh from the clean tree (``setup_targets``), never
    reused from ``state.json``, so every candidate comes from the ``package_commit`` recorded with
    the approval. Then re-check the tree and HEAD, capture package settings for listed cases an example maps to,
    and promote crash-safely (``acceptance_approval.promote``) with a committed report in
    ``docs/approvals/``. O04/O05 cannot be listed in a plain approval: their example settings are
    hashed only in the private record. Returns 0 on success, 1 on a failed gate.

    ``verify_private_settings`` (``--verify-private-settings``) lets an ordinary numerical approval
    list O04/O05. Every gate above runs unchanged and no layout verdict is involved. In addition,
    before promotion, the adopted settings of each listed O04/O05 are proven equal to a fresh
    package capture (``check_case_settings``: the selected file must exist and match its digest in
    the local ``private-inputs.json``; only the bookkeeping attributes are skipped; the private
    capture is deleted again, also on failure). A missing, unhashed, changed or different setting
    fails the approval with nothing promoted; changed private settings cannot be adopted here.
    The cases then keep their adopted settings (``acceptance_approval.CARRIED_FORWARD``) and the
    record states ``private_settings = "verified-equal"``. Manifest and report hold no settings
    content, settings hash or private digest, and the per-array deviations of O04/O05 are
    committed as verdicts only (``public_deviations``); O05's criteria keep value and limit.

    ``layout_correction`` (``--layout-correction RULE``, e.g. ``r09-data-tail``) adds a strict
    structural proof on top of every gate above (``layout_verdict``, ``acceptance_layout``): each
    listed case and contract must be an exact layout correction of its previously selected
    reference (verdict ``tail-removal``; an unchanged listed case is refused), and each unlisted one
    must be bit-identical (verdict ``identical``), so the listed set equals the differing set and no
    numerical change is adopted under public tolerance. Mapped cases' settings are proven equal
    case versus case (``check_case_settings``) before promotion; public ones store the fresh
    capture as usual, and only in this mode may O04/O05 be listed, with their adopted private
    settings carried forward (``acceptance_approval.CARRIED_FORWARD``). The committed report then
    holds structural layout verdicts instead of numerical deviations.
    """
    import datetime
    tools = approval_tools()
    listed = list(dict.fromkeys(listed))
    assert listed, "approve needs at least one case"
    assert layout_correction is None or layout_correction in layout_tools().RULES, \
        "unknown layout correction: " + str(layout_correction)
    refused = sorted(set(listed) & PRIVATE_SETTINGS_CASES)
    assert not (verify_private_settings and layout_correction is not None), \
        "--verify-private-settings is for a numerical approval, not a layout correction"
    assert not verify_private_settings or refused, "--verify-private-settings needs O04 or O05 among the listed cases"
    assert not refused or layout_correction is not None or verify_private_settings, \
        "approving " + ", ".join(refused) + " is not supported: their example settings are private"
    head = repository_checks()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    full = tier == "full"
    cases, skipped = normal_cases(manifest, full=full)
    contracts = list(manifest["contracts"])
    covered = set(cases) | set(contracts) | ({"O06"} if full else set())
    unknown = sorted(set(listed) - covered)
    assert not unknown, f"not run by approve --tier {tier} on this machine: {', '.join(unknown)}"
    check_private_inputs(cases)
    checked = list(dict.fromkeys(cases + contracts + (["O06"] if full else [])))
    validate_expected(checked)
    left = tools.leftovers(manifest, project=PROJECT, output=OUTPUT, manifest_path=MANIFEST)
    if any(left.values()):
        print("Leftovers from earlier attempts (not used; clean up by hand):", json.dumps(left), flush=True)
    state = setup_targets()   # fresh build from the clean tree at `head`; never a saved target
    state["repeatability_directory"] = tempfile.mkdtemp(prefix="approve-unseeded-", dir=OUTPUT)
    state["approval_candidates"] = sorted(set(listed) & set(contracts))
    started, started_timestamp = time.monotonic(), time.time()
    limit = APPROVE_BUDGET_SECONDS[tier]
    gates = dict(repeatability={}, criteria={}, unlisted_passed=[], listed={}, layout={})
    failure = None

    def check_layout(case):
        if layout_correction is None:
            return
        verdict = layout_verdict(case, layout_correction, manifest)
        gates["layout"][case] = layout_tools().structural(verdict)
        wanted = "tail-removal" if case in listed else "identical"
        assert verdict["verdict"] == wanted, (case, "listed" if case in listed else "unlisted",
                                              f"layout verdict {verdict['verdict']}, required {wanted}", verdict["problems"])
    try:
        if full:
            probe = worker("O06", "package", state=state, output=Path(tempfile.mkdtemp(prefix="tc-availability-", dir=OUTPUT)) / "O06",
                           probe_thermocalc=True, timeout=60)
            if probe["thermocalc_availability"]["available"]:
                cases.append("O06")
            else:
                skipped["O06"] = probe["thermocalc_availability"]["reason"]
                assert "O06" not in listed, "O06 is listed but TC is unavailable: " + skipped["O06"]
        for case in cases:
            remaining = limit - (time.monotonic() - started)
            assert remaining > 0, f"approve budget of {limit} s exceeded"
            report = worker(case, "package", state=state,
                            timeout=min(300 if case == "O06" else (180 if case.startswith("F") else 60), max(1, remaining)))
            repeat = approval_repeat(case, listed=listed, manifest=manifest, state=state,
                                     timeout=max(1, limit - (time.monotonic() - started)))
            if repeat is not None:
                gates["repeatability"][case] = dict(bit_identical=repeat["bit_identical"], arrays=repeat["arrays"])
            comparison = compare(case, report, tolerance="public", require=require)
            results = criteria_results(case, manifest)
            gates["criteria"][case] = results
            bad = [item for item in results if not item["passed"]]
            assert not bad, (case, "approval_criteria failed", bad)
            if case in listed:
                gates["listed"][case] = "differs" if comparison.get("failed_keys") else "within public tolerance"
            else:
                assert not comparison.get("failed_keys"), (case, "unlisted case fails its public-tolerance comparison",
                                                           comparison["failed_keys"])
                gates["unlisted_passed"].append(case)
            check_layout(case)
            print(case, "listed" if case in listed else "unlisted", "PASS",
                  gates["listed"].get(case, ""), flush=True)
        remaining = limit - (time.monotonic() - started)
        assert remaining > 0, f"approve budget of {limit} s exceeded"
        gates["pytest"] = run_pytest(state, tolerance="public", require=require, timeout=max(1, remaining), full_suite=True)
        for case in contracts:
            deviations = OUTPUT / "package" / (case + ".deviations.json")
            assert deviations.exists() and deviations.stat().st_mtime >= started_timestamp, (case, "contract not compared in this run")
            record = json.loads(deviations.read_text(encoding="utf-8"))
            if case in listed:
                candidate_matches_repeat(case, state)
                gates["listed"][case] = "within public tolerance" if record["passed"] else "differs"
            else:
                assert record["passed"], (case, "unlisted contract fails its public-tolerance comparison")
                gates["unlisted_passed"].append(case)
            check_layout(case)
        if layout_correction is not None:
            missing = sorted(set(listed) - set(gates["layout"]))
            assert not missing, ("listed but not run", missing)
            settings_dir = tempfile.mkdtemp(prefix="approve-settings-", dir=OUTPUT)
            layout_settings, carried, settings_problems = check_case_settings(listed, state, settings_dir, manifest)
            assert not settings_problems, ("settings proof failed", settings_problems)
        elif verify_private_settings:
            settings_dir = tempfile.mkdtemp(prefix="approve-settings-", dir=OUTPUT)
            _, carried, settings_problems = check_case_settings(refused, state, settings_dir, manifest)
            assert not settings_problems, ("private settings proof failed", settings_problems)
            assert sorted(carried) == refused, ("private settings proof did not cover", sorted(set(refused) - set(carried)))
        for case in listed:
            for suffix in ("json", "npz"):
                path = OUTPUT / "package" / f"{case}.{suffix}"
                assert path.stat().st_mtime >= started_timestamp, (case, "candidate not written by this run")
        validate_expected(checked)
        assert repository_checks() == head, "HEAD changed during approve"
    except Exception as exc:
        failure = str(exc)[-3000:]
    finally:
        state["approval_candidates"] = []
        (OUTPUT / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    summary = dict(action="approve", listed=listed, reason=reason, tier=tier, package_commit=head, skipped=skipped,
                   layout_correction=layout_correction, gates=gates, failure=failure, elapsed_seconds=time.monotonic() - started)
    if failure:
        (OUTPUT / "approve-summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print("APPROVE FAIL (nothing promoted):", failure, flush=True)
        return 1
    if layout_correction is None:
        if not verify_private_settings:
            carried = []
        settings = capture_example_settings([case for case in listed if case not in carried], state,
                                            tempfile.mkdtemp(prefix="approve-settings-", dir=OUTPUT))
    else:
        settings = layout_settings
    files = {case: dict(json=OUTPUT / "package" / f"{case}.json", npz=OUTPUT / "package" / f"{case}.npz") for case in listed}
    for case, path in settings.items():
        files[case]["settings"] = path
    now = datetime.datetime.now(datetime.timezone.utc)
    created = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    deviations = {case: json.loads((OUTPUT / "package" / (case + ".deviations.json")).read_text(encoding="utf-8")) for case in listed}
    document = dict(schema=1, kind="approval", created_utc=created, package_commit=head, reason=reason, tier=tier,
                    tolerance="public", cases=listed, skipped=skipped,
                    gates=dict(pytest=gates["pytest"], repeatability=gates["repeatability"],
                               criteria={case: public_criteria(case, results) for case, results in gates["criteria"].items() if results},
                               contract_assertion_groups="passed for " + ", ".join(contracts),
                               unlisted_cases_passed_public_tolerance=gates["unlisted_passed"]),
                    listed_cases=gates["listed"],
                    settings_captured=sorted(settings))
    if layout_correction is None:
        document["deviations_vs_previous"] = {case: public_deviations(case, record) for case, record in deviations.items()}
        if verify_private_settings:
            document.update(private_settings=tools.VERIFIED_EQUAL, settings_carried_forward=sorted(carried))
    else:
        document["gates"]["unlisted_cases_bit_identical"] = sorted(
            case for case, verdict in gates["layout"].items() if verdict["verdict"] == "identical")
        document.update(layout_correction=layout_correction,
                        layout_verdicts={case: gates["layout"][case] for case in listed},
                        settings_carried_forward=sorted(carried))
    approval_id = tools.new_id(manifest, project=PROJECT, output=OUTPUT, package_commit=head, now=now)
    record = dict(created_utc=created, package_commit=head, reason=reason, tier=tier, cases=listed)
    if layout_correction is not None:
        record["layout_correction"] = layout_correction
    if verify_private_settings:
        record["private_settings"] = tools.VERIFIED_EQUAL
    tools.promote(project=PROJECT, output=OUTPUT, manifest_path=MANIFEST, approval_id=approval_id,
                  record=record, files=files, report=document, carried_forward=carried)
    validate_expected(checked)
    summary.update(approval_id=approval_id)
    (OUTPUT / "approve-summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"APPROVED {approval_id}: {', '.join(listed)} ({time.monotonic() - started:.1f}s)", flush=True)
    print("Commit tools/acceptance_manifest.json and force-add the report: git add -f "
          f"packages/moving-boundary/docs/approvals/{approval_id}.json", flush=True)
    return 0


def rollback_approval(approval_id, *, reason):
    """``approve --rollback ID``: point the approval's cases back (atomic manifest replace; no file moves).

    From a clean package tree, like ``approve``; the approval's cases are hash-validated before and
    after (the restored selection must be intact). Returns 0.
    """
    import datetime
    repository_checks()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record = ((manifest.get("approved_references") or {}).get("approvals") or {}).get(approval_id)
    assert record is not None, "no such approval: " + approval_id
    validate_expected(record["cases"])
    utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = approval_tools().rollback(project=PROJECT, output=OUTPUT, manifest_path=MANIFEST,
                                    approval_id=approval_id, reason=reason, utc=utc)
    validate_expected(record["cases"])
    restored = new["approved_references"]["history"][-1]["restored"]
    print(f"ROLLED BACK {approval_id}: " + ", ".join(f"{case} -> {source}" for case, source in restored.items()), flush=True)
    print("Commit tools/acceptance_manifest.json.", flush=True)
    return 0


def main(argv=None):
    """Run quick/full acceptance (no research), an explicit historical capture/repeat, or an approval.

    The opt-in independence guards (``batch_safety.guard_independence``: ``MB_FORBID_RESEARCH``,
    ``MB_FORBID_GIT``) are installed first; workers and test runs inherit them.
    """
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parent / "batch_safety.py"))["guard_independence"]()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("quick", "full", "setup", "historical", "approve"))
    parser.add_argument("step", nargs="?", choices=("capture", "repeat"), help="historical actions only")
    parser.add_argument("--research", type=Path, help="frozen research checkout; historical actions only")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--reuse-targets", action="store_true")
    parser.add_argument("--require-family", nargs="*", default=["binary", "ternary", "three_phase"])
    parser.add_argument("--tolerance", choices=("exact", "public"), default="public")
    parser.add_argument("--tier", choices=("quick", "full"), default="quick")
    parser.add_argument("--record", action="store_true",
                        help="update committed reports after a passing complete public quick/full run")
    parser.add_argument("--reason", help="approve only: the LEDGERS entry justifying the approval or rollback")
    parser.add_argument("--rollback", metavar="APPROVAL_ID", help="approve only: point that approval's cases back")
    parser.add_argument("--layout-correction", choices=("r09-data-tail",),
                        help="complete quick/full: non-promoting strict layout proof against the selected references; "
                             "approve --cases: adopt an exact layout correction (tools/acceptance_layout.py)")
    parser.add_argument("--verify-private-settings", action="store_true",
                        help="approve --cases only: allow O04/O05 in a numerical approval after proving their adopted "
                             "private settings equal a fresh package capture; they are carried forward, never stored")
    args = parser.parse_args(argv)
    if args.verify_private_settings and (args.action != "approve" or args.rollback is not None
                                         or args.layout_correction is not None):
        parser.error("--verify-private-settings belongs to approve --cases without --layout-correction")
    if args.record and (args.action not in {"quick","full"} or args.tolerance != "public" or args.cases is not None):
        parser.error("--record requires a complete public-tolerance quick or full run")
    if args.layout_correction is not None:
        if args.action == "approve":
            if args.rollback is not None:
                parser.error("--layout-correction belongs to approve --cases, not --rollback")
        elif args.action not in {"quick", "full"} or args.cases is not None or args.record or args.reuse_targets:
            parser.error("--layout-correction needs a complete quick/full run (no --cases, --record or --reuse-targets) or approve --cases")
    if args.action == "approve":
        if args.step is not None or args.research is not None or args.tolerance != "public":
            parser.error("approve takes no historical step, no research checkout and only public tolerance")
        if args.reuse_targets:
            parser.error("approve always builds fresh targets from HEAD; --reuse-targets is for diagnostic runs")
        if not (args.reason and args.reason.strip()):
            parser.error('approve needs --reason "LEDGERS Rxx"')
        if bool(args.cases) == (args.rollback is not None):
            parser.error("approve needs exactly one of --cases CASE ... or --rollback APPROVAL_ID")
    elif args.reason is not None or args.rollback is not None:
        parser.error("--reason and --rollback belong to approve")
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
            timeout = (APPROVE_BUDGET_SECONDS[args.tier] + 300 if args.action == "approve"
                       else 1250 if args.action == "full" else 650)
            return subprocess.run([str(executable), str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
                env=env, stdin=subprocess.DEVNULL, timeout=timeout).returncode
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "inputs").mkdir(exist_ok=True)
    if args.action == "approve":
        if args.rollback is not None:
            return rollback_approval(args.rollback, reason=args.reason)
        return approve(args.cases, reason=args.reason, tier=args.tier, require=args.require_family,
                       layout_correction=args.layout_correction, verify_private_settings=args.verify_private_settings)
    layout = args.layout_correction
    manifest = validate_research_checkout(args.research) if historical else json.loads(MANIFEST.read_text(encoding="utf-8"))
    full = args.action == "full" or args.tier == "full"
    cases, skipped = normal_cases(manifest, full=full, explicit=args.cases)
    check_private_inputs(cases)
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
    layout_results, comparison_failures = {}, []
    state = None
    if not historical:
        state = json.loads((OUTPUT / "state.json").read_text(encoding="utf-8")) if args.reuse_targets else setup_targets()
        state["repeatability_directory"] = tempfile.mkdtemp(prefix="unseeded-", dir=OUTPUT)
        # Normal runs gate every contract comparison. A layout run reports the comparisons and
        # gates every contract on its layout verdict instead (assertion groups still gate).
        state["approval_candidates"] = list(manifest["contracts"]) if layout else []
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
            if layout:
                verdict = layout_verdict(case, layout, manifest)
                layout_results[case] = verdict
                if result.get("failed_keys"):
                    comparison_failures.append(case)
                if verdict["verdict"] == "other":
                    failures.append(case)
                print(case, "LAYOUT", verdict["verdict"].upper(), "comparison", "FAIL" if result.get("failed_keys") else "PASS",
                      *verdict["problems"], flush=True)
                continue
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
    if layout and tests:
        for case in manifest["contracts"]:
            try:
                written = OUTPUT / "package" / (case + ".npz")
                assert written.exists() and written.stat().st_mtime >= started_timestamp, (case, "contract not run in this layout run")
                record = json.loads((OUTPUT / "package" / (case + ".deviations.json")).read_text(encoding="utf-8"))
                verdict = layout_verdict(case, layout, manifest)
            except Exception as exc:
                verdict = dict(case=case, verdict="other", corrected=[], problems=[str(exc)[-1000:]])
                record = dict(passed=False)
            layout_results[case] = verdict
            if not record["passed"]:
                comparison_failures.append(case)
            if verdict["verdict"] == "other":
                failures.append(case)
            print(case, "LAYOUT", verdict["verdict"].upper(), "comparison", "PASS" if record["passed"] else "FAIL",
                  *verdict["problems"], flush=True)
    layout_settings_problems = []
    if layout:
        differing = [case for case, verdict in layout_results.items() if verdict["verdict"] == "tail-removal"]
        try:
            _, _, layout_settings_problems = check_case_settings(
                differing, state, tempfile.mkdtemp(prefix="layout-settings-", dir=OUTPUT), manifest)
        except Exception as exc:
            layout_settings_problems = ["settings proof failed: " + str(exc)[-1000:]]
        if layout_settings_problems:
            failures.append("settings")
        for problem in layout_settings_problems:
            print("SETTINGS FAIL", problem, flush=True)
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
    if layout:
        by_verdict = {name: sorted(case for case, verdict in layout_results.items() if verdict["verdict"] == name)
                      for name in ("identical", "tail-removal", "other")}
        mapped = sorted(set(by_verdict["tail-removal"]) & set(example_mapped_cases()))
        summary["layout"] = dict(rule=layout, verdicts=by_verdict, comparison_failures=sorted(comparison_failures),
                                 settings_checked=mapped, settings_problems=layout_settings_problems,
                                 cases={case: layout_tools().structural(verdict) for case, verdict in layout_results.items()})
        (OUTPUT / (args.action + "-layout-summary.json")).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"Layout {layout}: identical {len(by_verdict['identical'])}; tail-removal {len(by_verdict['tail-removal'])} "
              f"{by_verdict['tail-removal']}; other {by_verdict['other']}; settings checked {mapped}, "
              f"{len(layout_settings_problems)} problems; normal comparison failures (reported, not gated here) "
              f"{sorted(comparison_failures)}", flush=True)
    (OUTPUT / (("historical-capture" if historical else args.action) + "-summary.json")).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"{len(results)} cases, {len(failures)} failures, {summary['elapsed_seconds']:.2f}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
