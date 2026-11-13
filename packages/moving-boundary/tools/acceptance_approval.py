"""Approved references: reference selection, Git-free validation, crash-safe promotion, rollback (plan §5).

The committed manifest section ``approved_references`` records every approval and which one is in
effect per case::

    "approved_references": {
      "description": "...",
      "in_effect": {"<case>": "<approval id>", ...},
      "approvals": {"<approval id>": {created_utc, package_commit, reason, tier, cases, directory,
                                      [layout_correction,]
                                      artifacts: {case: {json, npz[, settings]}}, previous: {case: id | "reference"},
                                      report: {path, sha256}}},
      "history": [{"event": "approve" | "rollback", "approval_id", "utc", ...}, ...]
    }

With the section absent, every case reads ``reference/<case>`` exactly as before. An approval ID is
``<UTC %Y%m%dT%H%M%SZ>-<7-char package sha>-<counter>``; the counter skips every ID already used by
a record, a directory (complete or ``.partial``) or a report, so a retry never collides with what an
interrupted attempt left behind.

Promotion (``promote``) is crash-safe; the old manifest stays valid until the single commit point:

1. copy the candidate files into ``reference-approved/<id>.partial/``;
2. re-hash every file and reload it (``np.load(allow_pickle=False)``, ``json.loads``);
3. rename the folder to ``reference-approved/<id>/`` (never over an existing one);
   then write ``docs/approvals/<id>.json`` (temp file + replace; never over an existing report);
4. validate the selection the new manifest puts in effect, then write it to a temp file beside the
   manifest and ``os.replace`` it over the manifest. A
   ``PermissionError`` (OneDrive sync lock) is retried briefly; then the temp file is removed and
   ``ApprovalError`` is raised with the old manifest unchanged.

Settings of a case an example maps to are normally a package-captured file stored and hashed
with the approval. The one exception is ``CARRIED_FORWARD``: for O04/O05 (``PRIVATE_SETTINGS_CASES``,
whose settings hashes exist only in the local ``private-inputs.json``) an approval made with a
``layout_correction`` (``LAYOUT_RULES``) may record ``"settings": "private-carried-forward"``
instead of a hash. No settings file is copied and no settings hash is written; example acceptance
keeps using the adopted, privately hashed file. ``validate`` refuses the marker anywhere else.

Approved folders and reports are immutable: nothing here overwrites or deletes them, and nothing
writes under ``reference/``. ``rollback`` is one more atomic manifest replace that points the
approval's cases back at what was in effect before it; no file moves, and it is refused with the
manifest unchanged if the restored selection does not validate. Leftovers of interrupted
attempts are listed by ``leftovers`` and cleaned up only by hand.

This module never runs Git; the ancestry and clean-tree checks belong to ``acceptance.py approve``.
"""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import os
import re
import time

SECTION = "approved_references"
APPROVED = "reference-approved"
REPORTS = "docs/approvals"
SUFFIXES = ("json", "npz")
FROZEN_SECTIONS = {"O06": "thermocalc_sampling", "O08": "default_phase_sampling"}
ID_PATTERN = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{7}-[1-9][0-9]*")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
DESCRIPTION = ("Approved reference artifacts (plan section 5). in_effect maps a case to the approval whose "
               "artifacts it is compared with; cases not listed read reference/<case>. Artifacts live locally under "
               ".verification/acceptance/<directory> and are sha256-checked on every run; each approval's report "
               "(docs/approvals/<id>.json) is committed and sha256-checked. Written only by "
               "tools/acceptance.py approve; history keeps every approval and rollback.")
CARRIED_FORWARD = "private-carried-forward"
PRIVATE_SETTINGS_CASES = frozenset({"O04", "O05"})   # example settings hashed only in private-inputs.json
LAYOUT_RULES = frozenset({"r09-data-tail"})          # acceptance_layout.RULES
REPLACE_ATTEMPTS = 20
REPLACE_DELAY = 0.25
_os_replace = os.replace
_os_rename = os.rename


class ApprovalError(RuntimeError):
    """An approval or rollback was refused or could not complete; the previous state stays valid."""


def sha256(path):
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def section_of(manifest):
    """The ``approved_references`` section, or ``None`` when no approval was ever made."""
    return manifest.get(SECTION)


def expected_reference(case, manifest, output):
    """Where ``case``'s expected artifacts are, and which hashes ``compare`` must check first.

    Returns ``dict(case, source, json, npz, sha256, settings, settings_sha256,
    settings_carried_forward)``. ``source`` is ``"reference"`` or the approval ID in effect. Not
    approved: ``output/reference/<case>.*`` exactly as before; ``sha256`` is the frozen O06/O08
    pair from their manifest sections (and ``None`` for other cases, which ``validate_expected``
    checks against ``expected_artifacts``). Approved: the approval's immutable folder, its recorded
    artifact hashes, and, for a case an example maps to, the package-captured settings file and its
    hash, or ``settings_carried_forward=True`` (and no settings file) when the approval carried the
    adopted private settings forward (``CARRIED_FORWARD``).
    """
    approval_id = ((section_of(manifest) or {}).get("in_effect") or {}).get(case)
    if approval_id is None:
        base = Path(output) / "reference"
        selected = dict(case=case, source="reference", json=base / f"{case}.json", npz=base / f"{case}.npz",
                        sha256=None, settings=None, settings_sha256=None, settings_carried_forward=False)
        if case in FROZEN_SECTIONS:
            frozen = manifest[FROZEN_SECTIONS[case]]["reference_sha256"]
            assert set(frozen) == {case + ".json", case + ".npz"}, case + " requires its frozen fixture hashes"
            selected["sha256"] = {suffix: frozen[f"{case}.{suffix}"] for suffix in SUFFIXES}
        return selected
    record = manifest[SECTION]["approvals"][approval_id]
    folder = Path(output) / record["directory"]
    hashes = record["artifacts"][case]
    carried = hashes.get("settings") == CARRIED_FORWARD
    stored = "settings" in hashes and not carried
    return dict(case=case, source=approval_id, json=folder / f"{case}.json", npz=folder / f"{case}.npz",
                sha256={suffix: hashes[suffix] for suffix in SUFFIXES},
                settings=folder / f"{case}.settings.json" if stored else None,
                settings_sha256=hashes["settings"] if stored else None, settings_carried_forward=carried)


def validate(manifest, cases, *, project, output):
    """Git-free check of ``approved_references`` (part of ``validate_expected``); raises AssertionError.

    Every record: a well-formed ID, a 40-hex ``package_commit``, a non-empty ``reason``, unique cases
    that all have ``expected_artifacts`` entries, ``directory == reference-approved/<id>``, 64-hex
    artifact hashes for exactly its cases, ``previous`` naming ``reference`` or a recorded approval,
    and its committed report present with the recorded sha256. ``in_effect`` must name recorded
    approvals that list the case. For each of ``cases`` in effect, the approved files must exist
    with their recorded hashes. A ``layout_correction`` must name a known rule; the
    ``CARRIED_FORWARD`` settings marker is accepted only for O04/O05 of a layout-correction
    approval. Absent section: nothing to check.
    """
    section = section_of(manifest)
    if section is None:
        return
    assert isinstance(section, dict) and {"in_effect", "approvals", "history"} <= set(section), \
        "approved_references needs in_effect, approvals and history"
    expected = manifest["expected_artifacts"]["cases"]
    approvals = section["approvals"]
    for approval_id, record in approvals.items():
        where = "approved_references " + approval_id
        assert ID_PATTERN.fullmatch(approval_id), where + ": malformed approval ID"
        assert isinstance(record.get("package_commit"), str) and HEX40.fullmatch(record["package_commit"]), \
            where + ": package_commit must be a full commit sha"
        assert isinstance(record.get("reason"), str) and record["reason"].strip(), where + ": reason missing"
        listed = record.get("cases")
        assert isinstance(listed, list) and listed and len(set(listed)) == len(listed), where + ": cases missing"
        unknown = sorted(set(listed) - set(expected))
        assert not unknown, where + ": no expected_artifacts entry for " + ", ".join(unknown)
        assert record.get("directory") == f"{APPROVED}/{approval_id}", where + ": directory must be " + f"{APPROVED}/{approval_id}"
        layout = record.get("layout_correction")
        assert layout is None or layout in LAYOUT_RULES, where + ": unknown layout_correction " + repr(layout)
        artifacts = record.get("artifacts")
        assert isinstance(artifacts, dict) and set(artifacts) == set(listed), where + ": artifact hashes must cover exactly its cases"
        for case, hashes in artifacts.items():
            assert set(SUFFIXES) <= set(hashes) <= set(SUFFIXES) | {"settings"}, f"{where}: {case} artifact hashes malformed"
            if hashes.get("settings") == CARRIED_FORWARD:
                assert case in PRIVATE_SETTINGS_CASES and layout is not None, \
                    f"{where}: {case} settings marker {CARRIED_FORWARD} is only for O04/O05 of a layout correction"
            assert all(isinstance(value, str) and HEX64.fullmatch(value) for kind, value in hashes.items()
                       if not (kind == "settings" and value == CARRIED_FORWARD)), f"{where}: {case} artifact hashes malformed"
        previous = record.get("previous")
        assert isinstance(previous, dict) and set(previous) == set(listed), where + ": previous must cover exactly its cases"
        assert all(value == "reference" or value in approvals for value in previous.values()), where + ": previous names an unknown approval"
        report = record.get("report") or {}
        assert report.get("path") == f"{REPORTS}/{approval_id}.json", where + ": report path must be " + f"{REPORTS}/{approval_id}.json"
        path = Path(project) / report["path"]
        assert path.is_file(), where + ": approval report missing: " + report["path"]
        assert sha256(path) == report.get("sha256"), where + ": approval report changed: " + report["path"]
    for case, approval_id in section["in_effect"].items():
        assert approval_id in approvals, f"approved_references: {case} in effect from unknown approval {approval_id}"
        assert case in approvals[approval_id]["cases"], f"approved_references: {approval_id} does not approve {case}"
    for entry in section["history"]:
        assert entry.get("event") in {"approve", "rollback"} and entry.get("approval_id") in approvals, \
            "approved_references: malformed history entry"
    for case in cases:
        approval_id = section["in_effect"].get(case)
        if approval_id is None:
            continue
        record = approvals[approval_id]
        for kind, digest in record["artifacts"][case].items():
            if kind == "settings" and digest == CARRIED_FORWARD:
                continue   # no file: the adopted settings stay in example_settings/, privately hashed
            name = f"{case}.settings.json" if kind == "settings" else f"{case}.{kind}"
            path = Path(output) / record["directory"] / name
            assert path.is_file(), f"{case}: approved artifact missing: {record['directory']}/{name}"
            assert sha256(path) == digest, f"{case}: approved artifact changed: {record['directory']}/{name}"


def _taken(approval_id, manifest, project, output):
    root = Path(output) / APPROVED
    return (approval_id in ((section_of(manifest) or {}).get("approvals") or {})
            or (root / approval_id).exists() or (root / (approval_id + ".partial")).exists()
            or (Path(project) / REPORTS / (approval_id + ".json")).exists())


def new_id(manifest, *, project, output, package_commit, now=None):
    """A fresh approval ID ``<UTC>-<short sha>-<counter>`` that no record, folder or report uses."""
    now = now or datetime.now(timezone.utc)
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    counter = 1
    while _taken(f"{stamp}-{package_commit[:7]}-{counter}", manifest, project, output):
        counter += 1
    return f"{stamp}-{package_commit[:7]}-{counter}"


def leftovers(manifest, *, project, output, manifest_path):
    """What interrupted or unrecorded attempts left behind (reported; cleanup is explicit)."""
    approvals = set(((section_of(manifest) or {}).get("approvals") or {}))
    root = Path(output) / APPROVED
    folders = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    reports = Path(project) / REPORTS
    manifest_path = Path(manifest_path)
    return dict(
        partial=[name for name in folders if name.endswith(".partial")],
        unreferenced=[name for name in folders if not name.endswith(".partial") and name not in approvals],
        unreferenced_reports=sorted(path.name for path in reports.glob("*.json") if path.stem not in approvals) if reports.is_dir() else [],
        manifest_temps=sorted(path.name for path in manifest_path.parent.glob(f".{manifest_path.name}.*.tmp")))


def _retry(operation, source, target, what):
    """Run ``operation(source, target)``, retrying ``PermissionError`` (OneDrive locks), then fail loudly."""
    last = None
    for _ in range(REPLACE_ATTEMPTS):
        try:
            operation(source, target)
            return
        except PermissionError as exc:
            last = exc
            time.sleep(REPLACE_DELAY)
    raise ApprovalError(f"could not {what} after {REPLACE_ATTEMPTS} attempts ({last}); the previous state is unchanged and valid")


def manifest_text(text, section):
    """The manifest text with ``approved_references`` set to ``section`` as its last key.

    Every other byte is kept: the section is appended before the final brace, or the existing
    (last) section is replaced. The file's newline style (LF or CRLF) is preserved; the result is
    checked to parse to the old manifest plus the new section.
    """
    newline = "\r\n" if "\r\n" in text else "\n"
    body = text.replace("\r\n", "\n")
    old = json.loads(body)
    marker = '\n  "' + SECTION + '": '
    assert body.count(marker) <= 1, "approved_references appears more than once"
    if SECTION in old:
        assert list(old)[-1] == SECTION, "approved_references must be the manifest's last key"
        base = body[:body.index(marker)]
        assert base.endswith(","), "unexpected manifest layout before approved_references"
        base = base[:-1]
    else:
        stripped = body.rstrip()
        assert stripped.endswith("}"), "manifest must end with a closing brace"
        base = stripped[:-1].rstrip()
    block = json.dumps({SECTION: section}, indent=2)[2:-2]
    new = base + ",\n" + block + "\n}\n"
    parsed = json.loads(new)
    assert {key: value for key, value in parsed.items() if key != SECTION} == \
           {key: value for key, value in old.items() if key != SECTION}, "manifest rewrite changed another section"
    assert parsed[SECTION] == section and list(parsed)[-1] == SECTION
    return new.replace("\n", newline)


def write_manifest(manifest_path, section, tag, *, fault=None):
    """Step 4: the single commit point (temp file beside the manifest, then ``os.replace``)."""
    manifest_path = Path(manifest_path)
    text = manifest_text(manifest_path.read_bytes().decode("utf-8"), section)
    temp = manifest_path.with_name(f".{manifest_path.name}.{tag}.tmp")
    temp.write_bytes(text.encode("utf-8"))
    try:
        (fault or (lambda step: None))("manifest-temp-written")
        _retry(_os_replace, temp, manifest_path, "replace " + manifest_path.name)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return json.loads(text)


def _write_new_file(path, data):
    """Write ``data`` to a path that must not exist, via a temp file and replace."""
    path = Path(path)
    if path.exists():
        raise ApprovalError("refusing to overwrite " + path.as_posix())
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_bytes(data)
    try:
        _retry(_os_replace, temp, path, "write " + path.name)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _reload(path, kind):
    """Step 2: every candidate file must load (arrays without pickle, JSON documents)."""
    if kind == "npz":
        import numpy as np
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                data[key]
    else:
        json.loads(Path(path).read_text(encoding="utf-8"))


def promote(*, project, output, manifest_path, approval_id, record, files, report, fault=None, carried_forward=()):
    """Promote candidate files as approval ``approval_id`` (steps 1-4 in the module docstring).

    ``record`` carries ``created_utc, package_commit, reason, tier, cases`` (and, for a layout
    correction, ``layout_correction``); ``files`` maps each case to ``{"json": path, "npz": path[,
    "settings": path]}``; ``report`` is the gate evidence, written to ``docs/approvals/<id>.json``
    with the artifact hashes and previous selection added. ``carried_forward`` lists O04/O05 cases
    of a layout correction whose adopted private settings stay in place: they get the
    ``CARRIED_FORWARD`` marker instead of a settings file and hash. ``fault(step)`` is a test hook
    called after each step ("written", "validated", "renamed", "reported", "manifest-temp-written",
    "replaced"). Returns the new manifest.
    """
    fault = fault or (lambda step: None)
    project, output, manifest_path = Path(project), Path(output), Path(manifest_path)
    if not ID_PATTERN.fullmatch(approval_id):
        raise ApprovalError("malformed approval ID " + approval_id)
    cases = list(record["cases"])
    if set(files) != set(cases):
        raise ApprovalError("candidate files must cover exactly the listed cases")
    carried_forward = sorted(set(carried_forward))
    if carried_forward:
        if record.get("layout_correction") not in LAYOUT_RULES:
            raise ApprovalError("settings can be carried forward only by a layout-correction approval")
        bad = [case for case in carried_forward
               if case not in cases or case not in PRIVATE_SETTINGS_CASES or "settings" in files[case]]
        if bad:
            raise ApprovalError("settings can be carried forward only for listed O04/O05 without a settings file: " + ", ".join(bad))
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    if _taken(approval_id, manifest, project, output):
        raise ApprovalError("approval ID already used: " + approval_id)
    root = output / APPROVED
    partial, final = root / (approval_id + ".partial"), root / approval_id
    names = {case: {kind: (f"{case}.settings.json" if kind == "settings" else f"{case}.{kind}") for kind in sources}
             for case, sources in files.items()}
    hashes = {case: {kind: sha256(source) for kind, source in sources.items()} for case, sources in files.items()}
    copied = copy.deepcopy(hashes)   # what step 2 checks; markers are added after it
    # 1. write into <id>.partial
    partial.mkdir(parents=True)
    for case, sources in files.items():
        for kind, source in sources.items():
            (partial / names[case][kind]).write_bytes(Path(source).read_bytes())
    fault("written")
    # 2. validate hashes and reload
    for case, sources in files.items():
        for kind in sources:
            path = partial / names[case][kind]
            if sha256(path) != copied[case][kind]:
                raise ApprovalError(f"{case}: copied {path.name} does not match the candidate")
            _reload(path, "npz" if kind == "npz" else "json")
    for case in carried_forward:
        hashes[case]["settings"] = CARRIED_FORWARD
    fault("validated")
    # 3. rename to <id>, then write the committed report
    _retry(_os_rename, partial, final, "rename " + partial.name)
    fault("renamed")
    section = copy.deepcopy(section_of(manifest)) or dict(description=DESCRIPTION, in_effect={}, approvals={}, history=[])
    previous = {case: section["in_effect"].get(case, "reference") for case in cases}
    directory = f"{APPROVED}/{approval_id}"
    document = dict(report, approval_id=approval_id, directory=directory, artifacts=hashes, previous=previous)
    data = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")
    report_path = project / REPORTS / (approval_id + ".json")
    _write_new_file(report_path, data)
    fault("reported")
    # 4. the single commit point
    section["approvals"][approval_id] = dict(record, cases=cases, directory=directory, artifacts=hashes, previous=previous,
                                             report=dict(path=f"{REPORTS}/{approval_id}.json", sha256=hashlib.sha256(data).hexdigest()))
    section["in_effect"] = dict(sorted({**section["in_effect"], **{case: approval_id for case in cases}}.items()))
    section["history"].append(dict(event="approve", approval_id=approval_id, utc=record["created_utc"], cases=cases))
    _check_proposed(manifest, section, cases, project=project, output=output, what="approval")
    new = write_manifest(manifest_path, section, approval_id, fault=fault)
    fault("replaced")
    return new


def _check_proposed(manifest, section, cases, *, project, output, what):
    """Validate the selection a new section would put in effect, before anything is replaced.

    Runs ``validate`` on the manifest as it would be after the replace, for ``cases``: every
    record and report, and the files each case would then read (e.g. the earlier approval a
    rollback restores). A problem raises ``ApprovalError``; the manifest is still untouched.
    """
    proposed = dict(manifest)
    proposed[SECTION] = section
    try:
        validate(proposed, cases, project=project, output=output)
    except AssertionError as exc:
        raise ApprovalError(f"{what} refused, the manifest is unchanged: the selection it would put in effect is invalid: {exc}") from exc


def rollback(*, project, output, manifest_path, approval_id, reason, utc, fault=None):
    """Point ``approval_id``'s cases back at what was in effect before it (an atomic manifest replace).

    Refused unless the approval is in effect for every one of its cases (roll back later approvals
    of the same cases first), and refused, with the manifest unchanged, unless the restored
    selection validates (``_check_proposed``: e.g. the earlier approval's files are present with
    their recorded hashes). No file is moved or deleted. Returns the new manifest.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    section = copy.deepcopy(section_of(manifest))
    if not section or approval_id not in section["approvals"]:
        raise ApprovalError("no such approval: " + approval_id)
    if not (isinstance(reason, str) and reason.strip()):
        raise ApprovalError("a rollback needs a reason")
    record = section["approvals"][approval_id]
    superseded = {case: section["in_effect"].get(case, "reference") for case in record["cases"]
                  if section["in_effect"].get(case) != approval_id}
    if superseded:
        raise ApprovalError(f"{approval_id} is not in effect for {superseded}; roll back the later approval first")
    restored = {}
    for case in record["cases"]:
        restored[case] = record["previous"][case]
        if restored[case] == "reference":
            del section["in_effect"][case]
        else:
            section["in_effect"][case] = restored[case]
    section["in_effect"] = dict(sorted(section["in_effect"].items()))
    section["history"].append(dict(event="rollback", approval_id=approval_id, utc=utc, reason=reason, restored=restored))
    _check_proposed(manifest, section, record["cases"], project=project, output=output, what="rollback")
    return write_manifest(manifest_path, section, "rollback-" + approval_id, fault=fault)
