"""Regeneration verification of adopted inputs (plan §3a, batch M5b); verification only.

Recreates each adopted, research-generated input with package-owned code only (installed wheel
plus ``tools/definitions``; no research file is read) into a fresh ``regeneration-*`` scratch
folder, and compares it with the adopted original. Nothing is written to ``inputs/``,
``example_settings/`` or ``reference/``; the adopted files' sha256 must be unchanged afterwards.

Checks and their single outcome each (``verified``, ``failed``, ``unavailable``, ``not
comparable``; the four caches without package generators are ``deferred``):

- ``open-surrogate.npz``: package-role ``prepare`` (F7). Array identity (key set; dtype, shape and
  bytes per key, non-numeric included), ``metadata_json`` decoded and compared field by field, ZIP
  container reported separately as metadata only.
- ``pycalphad-reference.npz`` (needs ``FeCrNi.tdb``): package-role O05 sampling under
  ``PYTHONHASHSEED=0``; same comparison, plus the adopted file's numeric arrays against
  ``reference/O05.npz`` ``provider/*``.
- the 10 ``example_settings/*.settings.json``: ``example_worker.py --settings-only`` builds each
  accepted case from ``tools/definitions`` (no solve); exact byte and JSON equality (O04 needs
  ``fecrni.surrogate``, O05 needs ``FeCrNi.tdb``).

The allowed-differences list is empty; any difference fails the check and is reported, never
fixed by replacing an artifact. Backend versions (pycalphad, numpy, scipy, symengine) of this
environment are compared with ``--generation-versions`` (a JSON object of the versions of the
environment that generated the adopted files); if they differ, or none are given, a check that
matched is reported as ``not comparable`` instead of ``verified``. kawin differs by design
(research fork vs stock 0.5.0) and is only recorded.

The full local report goes to ``.verification/acceptance/regeneration-<digest>.json``.
``--summary PATH`` also writes a redacted summary for commit: outcomes, reasons, key counts and
container notes only, no hashes or values of TDB/TC-derived data.

    python tools/acceptance_regeneration.py run [--generation-versions FILE] [--summary FILE] [--reuse-targets]
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
BACKENDS = ("pycalphad", "numpy", "scipy", "symengine")
DEFERRED = {
    "exb5.json": "no package generation path (deferred): research Thermo-Calc study workflow",
    "femnc.json": "no package generation path (deferred): research Thermo-Calc study workflow",
    "fec.json": "no package generation path (deferred): research Thermo-Calc study workflow",
    "fecrni.surrogate": "no package generation path (deferred): research pycalphad study workflow",
}
SETTINGS_NEEDS = {"O04": "fecrni.surrogate", "O05": "FeCrNi.tdb"}


def load(name, path):
    """Load a sibling tool by file without adding a checkout to the import path."""
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha256(path):
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(value, prefix=""):
    """Dotted-path view of nested JSON, so metadata can be compared field by field."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out or ({prefix: {}} if prefix else {})  # an empty top-level object has no fields
    return {prefix: value}


def metadata_fields(generated, adopted):
    """Decode both files' ``metadata_json`` and list the dotted fields that differ."""
    import numpy as np
    decoded = []
    for path in (generated, adopted):
        with np.load(path, allow_pickle=False) as data:
            decoded.append(flatten(json.loads(str(data["metadata_json"][()]))) if "metadata_json" in data.files else None)
    left, right = decoded
    if left is None or right is None:
        return dict(present=(left is not None, right is not None), fields=0, differing=["metadata_json missing"])
    differing = sorted(key for key in set(left) | set(right) if left.get(key, object()) != right.get(key, object()))
    return dict(present=(True, True), fields=len(right), differing=differing)


def npz_check(tool, generated, adopted):
    """Array identity, decoded metadata and container notes for one regenerated npz."""
    identity = tool.npz_identity(generated, adopted)
    metadata = metadata_fields(generated, adopted)
    identity["metadata"] = metadata
    identity["matched"] = identity["array_identical"] and not metadata["differing"]
    return identity


def outcome(matched, comparable, reason=None):
    """One outcome per check: a match counts as verified only when the backends are comparable."""
    if not matched:
        return "failed", reason
    if not comparable["comparable"]:
        return "not comparable", comparable["reason"]
    return "verified", None


def backends(path):
    """This environment's backend versions, compared with the generation environment's."""
    from importlib.metadata import version
    here = {name: version(name) for name in BACKENDS + ("kawin",)}
    if path is None:
        return dict(here=here, generation=None, comparable=False,
                    reason="generation backend versions not supplied (--generation-versions)")
    generation = json.loads(Path(path).read_text(encoding="utf-8"))
    differing = [name for name in BACKENDS if generation.get(name) != here[name]]
    return dict(here=here, generation=generation, comparable=not differing,
                reason=("backend versions differ: " + ", ".join(differing)) if differing else None)


def check_open_surrogate(tool, state, scratch, comparable):
    """F7 regeneration of the synthetic C24 input; a missing adopted file fails (required input)."""
    adopted = tool.OUTPUT / "inputs/open-surrogate.npz"
    if not adopted.exists():
        return dict(outcome="failed", reason="required input missing: inputs/open-surrogate.npz")
    result = tool.regenerate_surrogate(state, scratch / "open-surrogate")
    detail = npz_check(tool, result["generated"], adopted)
    state_, reason = outcome(detail["matched"], comparable, "regenerated file differs from the adopted one")
    return dict(outcome=state_, reason=reason, detail=detail)


def check_pycalphad_reference(tool, examples, state, scratch, comparable, private):
    """Package O05 sampling (PYTHONHASHSEED=0) against the adopted E05 surrogate."""
    adopted = tool.OUTPUT / "inputs/pycalphad-reference.npz"
    database = tool.OUTPUT / "inputs/FeCrNi.tdb"
    if not database.exists():
        return dict(outcome="unavailable", reason="database input missing: FeCrNi.tdb")
    if not adopted.exists():
        return dict(outcome="failed", reason="adopted file missing: inputs/pycalphad-reference.npz")
    problems = (examples.check_input_hash(tool, "FeCrNi.tdb", private)
                + examples.check_input_hash(tool, "pycalphad-reference.npz", private))
    if problems:
        return dict(outcome="failed", reason="; ".join(problems))
    output = scratch / "pycalphad" / "O05"
    report = tool.worker("O05", "package", state=state, output=output, timeout=300)
    assert report["status"] == "PASS", report.get("error")
    detail = npz_check(tool, output.with_name("O05.pycalphad.npz"), adopted)
    import numpy as np
    with np.load(adopted, allow_pickle=False) as surrogate, np.load(tool.OUTPUT / "reference/O05.npz") as fixture:
        numeric = {key for key in surrogate.files if surrogate[key].dtype.kind in "bifcu"}
        provider = {key[len("provider/"):] for key in fixture.files if key.startswith("provider/")}
        matching = sorted(key for key in numeric & provider if tool.bit_identical(surrogate[key], fixture["provider/" + key]))
        detail["reference_O05_provider"] = dict(key_sets_equal=numeric == provider, arrays=len(provider),
                                                bit_identical=len(matching), differing=sorted((numeric & provider) - set(matching)))
    fixture_ok = detail["reference_O05_provider"]["key_sets_equal"] and not detail["reference_O05_provider"]["differing"]
    state_, reason = outcome(detail["matched"] and fixture_ok, comparable,
                             "regenerated file or reference/O05 provider arrays differ")
    return dict(outcome=state_, reason=reason, detail=detail)


def settings_worker(tool, state, case, output, reference):
    """Run ``example_worker.py --settings-only`` (package role) for one example case."""
    arguments = [sys.executable, "-P", "-S", str(PROJECT / "tools/example_worker.py"), "--role", "package",
                 "--case", case, "--examples", str(PROJECT / "examples"),
                 "--artifacts", str(tool.OUTPUT / "inputs"), "--output", str(output),
                 "--target", state["target"], "--host", state["host"], "--settings-only"]
    return tool.run(arguments, env=tool.worker_environment(reference), timeout=300)


def check_settings(tool, examples, state, scratch, comparable, private):
    """Regenerate every adopted settings file from tools/definitions; exact equality, no solve."""
    manifest = json.loads((PROJECT / "tools/example_manifest.json").read_text(encoding="utf-8"))
    digests = examples.settings_digests(manifest, private)
    results = {}
    for case, entry in manifest["cases"].items():
        reference = entry["reference"]
        adopted = tool.OUTPUT / "example_settings" / (reference + ".settings.json")
        need = SETTINGS_NEEDS.get(reference)
        if need and not (tool.OUTPUT / "inputs" / need).exists():
            results[reference] = dict(outcome="unavailable", reason="input missing: " + need)
            continue
        problems = ([] if adopted.exists() else ["adopted settings missing: " + adopted.name])
        problems += examples.check_settings_hash(adopted, digests)
        if need:
            problems += examples.check_input_hash(tool, need, private)
        if problems:
            results[reference] = dict(outcome="failed", reason="; ".join(problems))
            continue
        output = scratch / "settings" / reference
        settings_worker(tool, state, case, output, reference)
        generated = output.with_suffix(".settings.json")
        new, old = json.loads(generated.read_text(encoding="utf-8")), json.loads(adopted.read_text(encoding="utf-8"))
        flat_new, flat_old = flatten(new), flatten(old)
        differing = sorted(key for key in set(flat_new) | set(flat_old) if flat_new.get(key, object()) != flat_old.get(key, object()))
        byte_identical = generated.read_bytes() == adopted.read_bytes()
        state_, reason = outcome(byte_identical and not differing, comparable, "regenerated settings differ")
        results[reference] = dict(outcome=state_, reason=reason, example=case, byte_identical=byte_identical,
                                  json_equal=not differing, fields=len(flat_old), differing=differing)
    return results


def redact(report):
    """Committed summary: outcomes, reasons, counts and container notes; no hashes or values."""
    def npz(check):
        detail = check.get("detail")
        out = dict(outcome=check["outcome"], reason=check.get("reason"))
        if detail:
            out.update(keys_compared=len(detail["arrays"]), key_sets_equal=not detail["generated_only"] and not detail["adopted_only"],
                       arrays_bit_identical=sum(item["bit_identical"] for item in detail["arrays"].values()),
                       metadata_fields=detail["metadata"]["fields"], metadata_differing=len(detail["metadata"]["differing"]),
                       container=dict(byte_identical=detail["container"]["byte_identical"],
                                      members_equal=detail["container"]["members_equal"]))
            if "reference_O05_provider" in detail:
                provider = detail["reference_O05_provider"]
                out["reference_O05_provider"] = dict(key_sets_equal=provider["key_sets_equal"], arrays=provider["arrays"],
                                                     bit_identical=provider["bit_identical"])
        return out
    settings = {name: {key: value for key, value in check.items() if key in ("outcome", "reason", "byte_identical", "json_equal", "fields")}
                for name, check in report["checks"]["example_settings"].items()}
    backends = dict(report["backends"])
    if backends.get("generation"):  # local install paths stay out of committed evidence
        backends["generation"] = {key: value for key, value in backends["generation"].items() if key != "_environment"}
    return dict(schema=1, plan="AUTHORITY_SWITCH_PLAN.md section 3a (M5b)", backends=backends,
                adopted_unchanged=report["adopted_unchanged"], open_items=report["open_items"],
                checks={"open-surrogate.npz": npz(report["checks"]["open-surrogate.npz"]),
                        "pycalphad-reference.npz": npz(report["checks"]["pycalphad-reference.npz"]),
                        "example_settings": settings, "deferred": report["checks"]["deferred"]})


def adopted_files(tool):
    """Adopted files this verification reads; their hashes must not change."""
    names = [tool.OUTPUT / "inputs/open-surrogate.npz", tool.OUTPUT / "inputs/pycalphad-reference.npz"]
    names += sorted((tool.OUTPUT / "example_settings").glob("*.settings.json"))
    return {path.relative_to(tool.OUTPUT).as_posix(): sha256(path) for path in names if path.exists()}


def main(argv=None):
    """Re-execute in the public environment, then run every regeneration check into scratch."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("run",))
    parser.add_argument("--generation-versions", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--reuse-targets", action="store_true")
    args = parser.parse_args(argv)
    tool = load("acceptance", PROJECT / "tools/acceptance.py")
    config = json.loads((PROJECT / ".verification/public-environment.json").read_text())
    executable = Path(config["python"])
    if Path(sys.executable).resolve() != executable.resolve():
        env = os.environ.copy()
        prefix = executable.parent
        env["PATH"] = os.pathsep.join([str(prefix), str(prefix / "Library/bin"), str(prefix / "Scripts"), env.get("PATH", "")])
        env["CONDA_PREFIX"] = str(prefix)
        return subprocess.run([str(executable), str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
                              env=env, stdin=subprocess.DEVNULL, timeout=1200).returncode
    examples = load("acceptance_examples", PROJECT / "tools/acceptance_examples.py")
    started = time.monotonic()
    tool.validate_expected(["O05"])  # reference/O05.npz is read by the pycalphad check
    before = adopted_files(tool)
    state = json.loads((tool.OUTPUT / "state.json").read_text(encoding="utf-8")) if args.reuse_targets else tool.setup_targets()
    scratch = Path(tempfile.mkdtemp(prefix="regeneration-", dir=tool.OUTPUT))
    comparable = backends(args.generation_versions)
    private = examples.private_inputs(tool)
    checks = {"open-surrogate.npz": check_open_surrogate(tool, state, scratch, comparable),
              "pycalphad-reference.npz": check_pycalphad_reference(tool, examples, state, scratch, comparable, private),
              "example_settings": check_settings(tool, examples, state, scratch, comparable, private),
              "deferred": {name: dict(outcome="deferred", reason=reason) for name, reason in DEFERRED.items()}}
    after = adopted_files(tool)
    tool.validate_expected(["O05"])
    flat = {"open-surrogate.npz": checks["open-surrogate.npz"], "pycalphad-reference.npz": checks["pycalphad-reference.npz"],
            **{f"example_settings/{name}.settings.json": check for name, check in checks["example_settings"].items()}}
    open_items = {name: f"{check['outcome']}: {check.get('reason')}" for name, check in flat.items() if check["outcome"] != "verified"}
    report = dict(scratch=str(scratch), backends=comparable, adopted_before=before, adopted_after=after,
                  adopted_unchanged=before == after, checks=checks, open_items=open_items,
                  elapsed_seconds=time.monotonic() - started)
    body = json.dumps(report, indent=1, sort_keys=True)
    local = tool.OUTPUT / f"regeneration-{hashlib.sha256(body.encode()).hexdigest()[:12]}.json"
    local.write_text(body, encoding="utf-8")
    if args.summary:
        args.summary.write_text(json.dumps(redact(report), indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for name, check in flat.items():
        print(f"{name}: {check['outcome']}" + (f" ({check['reason']})" if check.get("reason") else ""), flush=True)
    for name, check in checks["deferred"].items():
        print(f"{name}: deferred ({check['reason']})", flush=True)
    print(f"adopted files unchanged: {report['adopted_unchanged']}; open items: {len(open_items)}; report {local.name}; "
          f"{report['elapsed_seconds']:.1f}s", flush=True)
    failed = [name for name, check in flat.items() if check["outcome"] == "failed"]
    return 1 if failed or not report["adopted_unchanged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
