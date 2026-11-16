"""Hash-checked export, check and restore of the local acceptance artifacts (``.verification/``).

Exact acceptance needs files that stay out of Git: frozen references, approved references, adopted
settings, inputs, the stock kawin wheel and the C33/F33 capture. This tool moves them to another
checkout without new hash rules. Every file it selects carries a digest from an existing record.
The one exception is the O05 pilot timing file, whose digest only the bundle records.

    python tools/acceptance_transfer.py export  --dest <dir>
    python tools/acceptance_transfer.py check   <bundle>
    python tools/acceptance_transfer.py restore <bundle> [--dry-run]

The tool always acts on the checkout it lives in: export runs the source checkout's copy, and
restore runs the destination's copy, so the destination's committed records are the authority. It
uses the standard library only and never runs Git.

**Selection** (``select_entries``; paths relative to the package root):

- *required*:
  - ``reference/<case>.{json,npz}`` for every ``expected_artifacts`` case;
  - every approval record's files, including superseded approvals (no settings file for
    ``private-carried-forward``);
  - publicly hashed ``example_settings`` and ``adopted_inputs``;
  - the stock wheel;
  - each referenced ``baseline-staging/<capture_id>/`` folder.
- *private*, optional (absent is recorded as absent; present must verify fully):
  - ``inputs/private-inputs.json`` and the inputs it covers;
  - the privately hashed O04/O05 settings;
  - the O06 input.

  Private resources are discovered from the case/example mappings, the record keys and the
  unhashed files in ``example_settings/``, not from the record alone. An available resource
  without a valid record entry is refused, never dropped.
- *optional*: ``pilot/O05-sampling.json``.

Everything else is never exported and is reported only by name and count: run folders, logs,
unrecorded inputs, evidence, caches and the machine-specific ``public-environment.json`` pointer,
which each destination supplies itself.

**Captures**: the sha256 of ``capture.json`` must equal the manifest's
``capture_record.capture_json_sha256``, and this is checked before anything else. Then the
capture id must match, the captured files must match both ``capture.json`` and
``expected_artifacts``, and no unlisted file may be present.

**Paths** (``check_path``, ``confined``):

- relative POSIX paths under the allowed prefixes; no ``..``, ``.``, empty segment, backslash,
  ``:``, NUL, or trailing dot or space;
- case-insensitive duplicates are refused;
- each path must resolve inside its root;
- no existing component from the root down may be a symlink, junction/mount point or
  app-execution link. Cloud-file placeholders are allowed because they don't redirect paths.

**Export**: verify the source strictly, then copy into ``<dest>/<id>.partial/``. Write
``bundle.json`` there and validate the staged contents (``verify_contents``). Rename to ``<id>``
only after that validation passes. A destination inside a Git work tree is refused, unless it is
under this package's ignored ``.verification/``. Bundles hold private files and private digests,
so they always stay local.

**Restore**: ``check`` the bundle, recompute the destination's selection and preflight every target:

- *absent* targets are scheduled;
- *present and matching* targets are left untouched;
- *conflicting* targets refuse the whole restore before the first write.

Recorded private directories are preflighted member by member, and their planned combined
contents must equal the recorded set. Files are written to the dedicated temporary area
``.verification/acceptance/.transfer-tmp/<bundle_id>/`` (outside every hash-recorded directory),
re-hashed, and published with ``acceptance_baseline._publish``, which never overwrites.
``private-inputs.json`` is installed last. When the destination has no private record, the bundle's
validated record covers matching files left by an interrupted restore. Retrying the same bundle
completes an interrupted restore. The strict checks then run on the result.
"""
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
SCHEMA = 1
RECORD = "bundle.json"
ACCEPTANCE = ".verification/acceptance"
WHEEL = ".verification/kawin-0.5.0-py3-none-any.whl"
PRIVATE_RECORD = ACCEPTANCE + "/inputs/private-inputs.json"
PILOT = ACCEPTANCE + "/pilot/O05-sampling.json"
TMP_AREA = ACCEPTANCE + "/.transfer-tmp"
POINTER = ".verification/public-environment.json"
AREAS = ("reference", "reference-approved", "example_settings", "inputs", "pilot", "baseline-staging")
PREFIXES = tuple(f"{ACCEPTANCE}/{area}/" for area in AREAS)
HEX64 = re.compile("[0-9a-f]{64}")
SPECIAL_SOURCES = frozenset({"lifecycle"})   # example_manifest "surrogate" token that is not an input path
REDIRECTING_TAGS = frozenset({getattr(stat, "IO_REPARSE_TAG_SYMLINK", 0xA000000C),
                              getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003),
                              getattr(stat, "IO_REPARSE_TAG_APPEXECLINK", 0x8000001B)})
CLASSES = ("required", "private", "optional")


class TransferError(RuntimeError):
    """A transfer was refused; ``problems`` lists every reason (paths and verdicts, never digests)."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


@lru_cache(maxsize=None)
def _tool(name):
    """Load a sibling acceptance module by path (no checkout is added to the import path)."""
    spec = importlib.util.spec_from_file_location(f"acceptance_transfer_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(rel):
    """Comparison key for duplicate detection: Windows case aliases collide on every platform."""
    return os.path.normcase(rel).casefold()


def check_path(rel, *, temporary=False):
    """Refuse a package-relative path outside the transfer policy; returns ``rel``.

    Accepted: a relative POSIX path that is the stock wheel or lies under one of ``PREFIXES``
    (``temporary=True`` also allows ``TMP_AREA``), with no empty, ``.`` or ``..`` segment, no
    backslash, ``:`` (drives, alternate data streams), NUL or leading slash, and no segment ending
    in a dot or space (Windows would silently strip it).
    """
    problem = None
    if not isinstance(rel, str) or not rel:
        problem = "empty or non-text path"
    elif rel.startswith("/") or any(char in rel for char in "\\:\0"):
        problem = "absolute path or forbidden character"
    elif any(part in ("", ".", "..") or part != part.rstrip(". ") for part in rel.split("/")):
        problem = "empty, relative or dot/space-terminated segment"
    elif not (rel == WHEEL or rel.startswith(PREFIXES) or (temporary and rel.startswith(TMP_AREA + "/"))):
        problem = "outside the transferable locations"
    if problem:
        raise TransferError([f"path refused ({problem}): {rel!r}"])
    return rel


def is_redirecting(status):
    """True for a symlink, junction/mount point or app-execution link (``os.lstat`` result).

    Other reparse points, such as OneDrive cloud placeholders, don't redirect a path and pass.
    """
    return stat.S_ISLNK(status.st_mode) or getattr(status, "st_reparse_tag", 0) in REDIRECTING_TAGS


def confined(root, rel, *, temporary=False):
    """``root / rel`` after the path policy, the link policy and canonical containment.

    Every existing component from ``root`` down to the file (the file included) is ``lstat``-ed and
    must not redirect (``is_redirecting``); ``root``'s own ancestors are not inspected. The
    canonical path must stay under the canonical ``root``.
    """
    check_path(rel, temporary=temporary)
    root = Path(root)
    current = root
    for part in rel.split("/"):
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            break
        if is_redirecting(status):
            raise TransferError([f"path refused (link or junction at {current.relative_to(root).as_posix()}): {rel}"])
    real_root = os.path.realpath(root)
    real = os.path.realpath(root / rel)
    if os.path.normcase(os.path.commonpath([real_root, real])) != os.path.normcase(real_root):
        raise TransferError([f"path refused (resolves outside its root): {rel}"])
    return root / rel


def _entry(path, digest, cls, group):
    return dict(path=path, sha256=digest, cls=cls, group=group)


def _committed_entries(project, manifest, examples):
    """Entries whose digests come from committed records alone (plus the O06 input, private)."""
    acceptance, approval = _tool("acceptance"), _tool("acceptance_approval")
    entries, problems = [], []
    expected = manifest["expected_artifacts"]["cases"]
    for case, entry in sorted(expected.items()):
        for suffix in ("json", "npz"):
            entries.append(_entry(f"{ACCEPTANCE}/reference/{case}.{suffix}", entry.get(suffix), "required", "reference"))
    try:
        approval.validate(manifest, (), project=Path(project), output=Path(project) / ACCEPTANCE)
    except (AssertionError, KeyError, TypeError) as exc:
        problems.append(f"approval records malformed: {exc}")
    else:
        approvals = ((manifest.get(approval.SECTION) or {}).get("approvals") or {})
        for approval_id, record in sorted(approvals.items()):
            for case, hashes in sorted(record["artifacts"].items()):
                for kind, digest in sorted(hashes.items()):
                    if kind == "settings" and digest == approval.CARRIED_FORWARD:
                        continue
                    name = f"{case}.settings.json" if kind == "settings" else f"{case}.{kind}"
                    entries.append(_entry(f"{ACCEPTANCE}/{record['directory']}/{name}", digest, "required", "approved"))
    for name, digest in sorted(examples["settings_sha256"].items()):
        entries.append(_entry(f"{ACCEPTANCE}/example_settings/{name}", digest, "required", "settings"))
    for name, entry in sorted(manifest["adopted_inputs"]["files"].items()):
        entries.append(_entry(f"{ACCEPTANCE}/inputs/{name}", entry.get("sha256"), "required", "adopted-input"))
    entries.append(_entry(WHEEL, acceptance.STOCK_WHEEL_SHA256, "required", "wheel"))
    return entries, problems


def capture_digests(manifest):
    """``{capture_id: committed capture.json sha256}``; entries naming one id must agree on it."""
    groups, problems = {}, []
    for case, entry in sorted(manifest["expected_artifacts"]["cases"].items()):
        if "capture_id" in entry:
            digest = (entry.get("capture_record") or {}).get("capture_json_sha256")
            groups.setdefault(entry["capture_id"], set()).add(digest)
    digests = {}
    for capture_id, values in sorted(groups.items()):
        value = next(iter(values))
        if len(values) != 1 or not isinstance(value, str) or not HEX64.fullmatch(value):
            problems.append(f"capture {capture_id}: manifest entries lack or disagree on capture_record.capture_json_sha256")
        else:
            digests[capture_id] = value
    return digests, problems


def capture_entries(manifest, capture_id, committed, data):
    """Entries of one capture folder from its ``capture.json`` bytes, checked in the documented order.

    1. ``data`` must hash to the committed ``capture_json_sha256`` (altered provenance fails even
       when every fixture byte is unchanged).
    2. ``capture.json`` must parse and name ``capture_id``.
    3. Each captured file's digest must agree with ``expected_artifacts`` where that case has an
       entry.

    The extra-file check is the caller's (it needs a directory listing).
    """
    folder = f"{ACCEPTANCE}/baseline-staging/{capture_id}"
    if data is None:
        return [], [f"{folder}/capture.json missing"]
    if hashlib.sha256(data).hexdigest() != committed:
        return [], [f"{folder}/capture.json does not match the committed capture_record.capture_json_sha256"]
    try:
        document = json.loads(data.decode("utf-8"))
        files = document["files"]
        assert document["capture_id"] == capture_id and isinstance(files, dict) and files
    except (ValueError, KeyError, TypeError, AssertionError):
        return [], [f"{folder}/capture.json malformed or names another capture"]
    expected = manifest["expected_artifacts"]["cases"]
    entries = [_entry(f"{folder}/capture.json", committed, "required", "capture")]
    problems = []
    for case, pair in sorted(files.items()):
        for suffix in ("json", "npz"):
            digest = (pair or {}).get(suffix) if isinstance(pair, dict) else None
            if case in expected and expected[case].get(suffix) != digest:
                problems.append(f"{folder}: {case}.{suffix} in capture.json disagrees with expected_artifacts")
            entries.append(_entry(f"{folder}/{case}.{suffix}", digest, "required", "capture"))
    return entries, problems


def parse_private(data):
    """Parse ``private-inputs.json`` bytes; ``(record, problems)``. Shapes are checked per entry later."""
    try:
        record = json.loads(data.decode("utf-8"))
    except ValueError:
        return None, ["private-inputs.json malformed (not JSON)"]
    if not isinstance(record, dict):
        return None, ["private-inputs.json malformed (not an object)"]
    return record, []


def _digest_map(record, name):
    entry = record.get(name)
    digests = entry.get("sha256") if isinstance(entry, dict) else None
    if not isinstance(digests, dict) or not digests:
        return None, [f"private record entry empty or malformed: {name}"]
    bad = sorted(str(key) for key, value in digests.items()
                 if not (isinstance(key, str) and isinstance(value, str) and HEX64.fullmatch(value)))
    if bad:
        return None, [f"private record entry malformed: {name}/{key}" for key in bad]
    return digests, []


def private_input_entries(name, record):
    """Member entries of one recorded private input, following ``acceptance.private_input_problems``.

    A record naming exactly the input (``{name: digest}``) is a file; otherwise its keys are files
    of the directory ``inputs/<name>/``. Every member path must pass ``check_path``, so a key
    cannot escape its directory. Returns ``(entries, is_directory, problems)``.
    """
    digests, problems = _digest_map(record, name)
    if problems:
        return [], False, problems
    base = f"{ACCEPTANCE}/inputs/{name}"
    directory = set(digests) != {name}
    entries = []
    for key, digest in sorted(digests.items()):
        rel = f"{base}/{key}" if directory else base
        try:
            check_path(rel)
        except TransferError as exc:
            problems.extend(exc.problems)
            continue
        if directory and not rel.startswith(base + "/"):
            problems.append(f"private record key escapes its directory: {name}/{key}")
            continue
        entries.append(_entry(rel, digest, "private", f"private-input:{name}"))
    return entries, directory, problems


def private_input_candidates(manifest, examples, record):
    """Recognized private input names (case/example mappings) plus the record's own keys."""
    acceptance = _tool("acceptance")
    names = set(acceptance.PRIVATE_CASE_INPUTS.values())
    if manifest.get("default_phase_sampling"):
        names.add(manifest["default_phase_sampling"]["database_input"])
    for entry in examples["cases"].values():
        for key in ("surrogate", "provider_arrays"):
            value = entry.get(key)
            if value and value not in SPECIAL_SOURCES:
                names.add(value)
    names |= {name for name in (record or {}) if name != "example_settings"}
    names -= set(manifest["adopted_inputs"]["files"])
    if manifest.get("thermocalc_sampling"):
        names.discard(manifest["thermocalc_sampling"]["input"])
    return sorted(names)


def _settings_candidates(examples, record, views):
    public = set(examples["settings_sha256"])
    entry = (record or {}).get("example_settings")
    names = set(entry.get("sha256") or {}) if isinstance(entry, dict) and isinstance(entry.get("sha256"), dict) else set()
    for view in views:
        folder = Path(view) / ACCEPTANCE / "example_settings"
        if folder.is_dir():
            names |= {path.name for path in folder.iterdir() if path.is_file()} - public
    return sorted(names)


def _present(views, rel):
    return any(os.path.lexists(Path(view) / rel) for view in views)


def select_entries(project, views, private_bytes, read_capture):
    """Every file the records of ``project`` select, given the effective private record.

    ``views`` are roots laid out like the package (the source checkout, or the destination and the
    bundle); a private or optional resource counts as available when it exists in any of them.
    ``private_bytes`` is the effective ``private-inputs.json`` (``None`` when there is none) and
    ``read_capture(capture_id)`` returns ``capture.json`` bytes or ``None``. Returns
    ``(entries, absent, directories, problems)``: ``directories`` maps each recorded private
    directory or capture folder to its exact member set; ``absent`` lists optional resources
    present in no view.
    """
    project = Path(project)
    manifest = json.loads((project / "tools/acceptance_manifest.json").read_text(encoding="utf-8"))
    examples = json.loads((project / "tools/example_manifest.json").read_text(encoding="utf-8"))
    entries, problems = _committed_entries(project, manifest, examples)
    absent, directories = [], {}
    digests, more = capture_digests(manifest)
    problems += more
    for capture_id, committed in sorted(digests.items()):
        found, more = capture_entries(manifest, capture_id, committed, read_capture(capture_id))
        entries += found
        problems += more
        directories[f"{ACCEPTANCE}/baseline-staging/{capture_id}"] = {entry["path"] for entry in found}
    record = None
    if private_bytes is not None:
        record, more = parse_private(private_bytes)
        problems += more
        entries.append(_entry(PRIVATE_RECORD, hashlib.sha256(private_bytes).hexdigest(), "private", "private-record"))
    for name in private_input_candidates(manifest, examples, record):
        rel = f"{ACCEPTANCE}/inputs/{name}"
        try:
            check_path(rel)
        except TransferError as exc:
            problems += exc.problems
            continue
        present = _present(views, rel)
        if record is None or name not in record:
            if present:
                problems.append(("private-inputs.json missing or malformed; cannot verify available input: "
                                 if record is None else "available private input has no record entry: ") + f"inputs/{name}")
            else:
                absent.append(rel)
            continue
        found, directory, more = private_input_entries(name, record)
        problems += more
        if present:
            entries += found
            if directory:
                directories[rel] = {entry["path"] for entry in found}
        else:
            absent.append(rel)
    settings_digests = {}
    if record is not None and "example_settings" in record:
        settings_digests, more = _digest_map(record, "example_settings")
        settings_digests = settings_digests or {}
        problems += more
    for name in _settings_candidates(examples, record, views):
        rel = f"{ACCEPTANCE}/example_settings/{name}"
        try:
            check_path(rel)
        except TransferError as exc:
            problems += exc.problems
            continue
        present = _present(views, rel)
        digest = settings_digests.get(name)
        if digest is None:
            if present:
                problems.append(f"adopted settings not hashed: example_settings/{name}")
            else:
                absent.append(rel)
            continue
        if present:
            entries.append(_entry(rel, digest, "private", "private-settings"))
        else:
            absent.append(rel)
    tc = manifest.get("thermocalc_sampling")
    if tc:
        rel = f"{ACCEPTANCE}/inputs/{tc['input']}"
        if _present(views, rel):
            entries.append(_entry(rel, tc["input_sha256"], "private", "tc-input"))
        else:
            absent.append(rel)
    if _present(views, PILOT):
        entries.append(_entry(PILOT, None, "optional", "pilot"))
    else:
        absent.append(PILOT)
    for entry in entries:
        try:
            check_path(entry["path"])
        except TransferError as exc:
            problems += exc.problems
        if entry["group"] != "pilot" and not (isinstance(entry["sha256"], str) and HEX64.fullmatch(entry["sha256"])):
            problems.append(f"record digest missing or malformed for {entry['path']}")
    seen = set()
    for entry in entries:
        if _key(entry["path"]) in seen:
            problems.append(f"duplicate path (case alias or repeated record): {entry['path']}")
        seen.add(_key(entry["path"]))
    return entries, sorted(set(absent)), directories, list(dict.fromkeys(problems))


def _local_private_bytes(project):
    path = Path(project) / PRIVATE_RECORD
    if not os.path.lexists(path):
        return None
    return confined(project, PRIVATE_RECORD).read_bytes()


def _members_on_disk(root, rel):
    folder = Path(root) / rel
    if not folder.is_dir():
        return set()
    return {f"{rel}/{path.relative_to(folder).as_posix()}" for path in folder.rglob("*") if path.is_file()}


def verify_local(project):
    """Strict check of ``project``'s own artifacts (export, and after every restore).

    Every selected file must be present (optional and private resources only when available),
    inside the path policy and equal to its recorded digest. Each available private input also
    passes the unchanged ``acceptance.private_input_problems``, each settings file passes
    ``acceptance_examples.check_settings_hash``, and capture folders hold exactly their recorded
    files. Returns ``(entries, absent)`` with the pilot digest filled in; raises ``TransferError``.
    """
    project = Path(project)
    acceptance, examples_tool = _tool("acceptance"), _tool("acceptance_examples")
    problems = []
    try:
        private_bytes = _local_private_bytes(project)
    except TransferError as exc:
        raise TransferError(exc.problems) from None

    def read_capture(capture_id):
        rel = f"{ACCEPTANCE}/baseline-staging/{capture_id}/capture.json"
        return confined(project, rel).read_bytes() if os.path.lexists(project / rel) else None

    entries, absent, directories, problems = select_entries(project, [project], private_bytes, read_capture)
    for entry in entries:
        try:
            path = confined(project, entry["path"])
        except TransferError as exc:
            problems += exc.problems
            continue
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
            continue
        digest = _sha(path)
        if entry["sha256"] is None:
            entry["sha256"] = digest
        elif digest != entry["sha256"]:
            problems.append(f"changed (differs from its record): {entry['path']}")
    for folder, members in directories.items():
        if folder.startswith(f"{ACCEPTANCE}/baseline-staging/"):
            for extra in sorted(_members_on_disk(project, folder) - members):
                problems.append(f"unlisted file in capture folder: {extra}")
    if private_bytes is not None and not problems:
        record = json.loads(private_bytes.decode("utf-8"))
        for name in sorted({entry["group"].split(":", 1)[1] for entry in entries if entry["group"].startswith("private-input:")}):
            problems += acceptance.private_input_problems(name, record, inputs=project / ACCEPTANCE / "inputs")
        manifest = json.loads((project / "tools/example_manifest.json").read_text(encoding="utf-8"))
        digests = examples_tool.settings_digests(manifest, record)
        for entry in entries:
            if entry["group"] in {"settings", "private-settings"}:
                problems += examples_tool.check_settings_hash(project / entry["path"], digests)
    if problems:
        raise TransferError(problems)
    return entries, absent


def _new_id(dest, manifest_sha, now):
    base = f"{now.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{manifest_sha[:7]}"
    counter = 1
    while (Path(dest) / f"{base}-{counter}").exists() or (Path(dest) / f"{base}-{counter}.partial").exists():
        counter += 1
    return f"{base}-{counter}"


def guard_destination(project, dest):
    """Refuse a bundle destination inside a Git work tree or inside a transferred area.

    Allowed: anywhere outside Git work trees (a parent holding ``.git``, found by walking the
    filesystem, never by running Git), or under this package's ignored ``.verification/`` outside
    the transferred areas and the temporary area.
    """
    real = Path(os.path.realpath(dest))
    allowed = Path(os.path.realpath(Path(project) / ".verification"))
    acceptance = Path(os.path.realpath(Path(project) / ACCEPTANCE))
    blocked = [acceptance / area for area in AREAS] + [acceptance / ".transfer-tmp"]
    if any(real == path or path in real.parents for path in blocked):
        raise TransferError([f"bundle destination is inside a transferred area: {dest}"])
    if real == allowed or allowed in real.parents:
        return
    for folder in (real, *real.parents):
        if (folder / ".git").exists():
            raise TransferError([f"bundle destination is inside a Git work tree ({folder}); bundles hold private files: {dest}"])


def verify_contents(folder, expected=None):
    """Check a bundle folder (complete or staging) against its ``bundle.json``; returns the record.

    ``bundle.json`` must parse with the current schema and name the folder (``<id>`` or
    ``<id>.partial``). Every listed path must pass the path and link policy inside the folder,
    must not alias another, and must be present with its recorded size and digest. No unlisted
    file may be present. With ``expected`` (export), the listing must equal those entries exactly.
    """
    folder = Path(folder)
    problems = []
    try:
        record = json.loads(confined_bundle(folder, RECORD).read_text(encoding="utf-8"))
        files = record["files"]
        assert record.get("schema") == SCHEMA and isinstance(files, list)
        assert folder.name in (record["bundle_id"], record["bundle_id"] + ".partial")
    except (OSError, ValueError, KeyError, TypeError, AssertionError, TransferError):
        raise TransferError([f"{folder.name}: bundle.json missing or malformed"]) from None
    listed, keys = {}, {}
    for item in files:
        try:
            rel, digest, size, cls = item["path"], item["sha256"], item["size"], item["class"]
            assert cls in CLASSES and isinstance(size, int) and isinstance(digest, str) and HEX64.fullmatch(digest)
            path = confined(folder, rel)
        except (KeyError, TypeError, AssertionError):
            problems.append(f"bundle.json entry malformed: {item!r}"[:200])
            continue
        except TransferError as exc:
            problems += exc.problems
            continue
        if _key(rel) in keys:
            problems.append(f"duplicate path in bundle (case alias): {rel}")
            continue
        keys[_key(rel)] = rel
        listed[rel] = item
        if not path.is_file():
            problems.append(f"bundle file missing: {rel}")
        elif path.stat().st_size != size or _sha(path) != digest:
            problems.append(f"bundle file changed: {rel}")
    on_disk = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()} - {RECORD}
    problems += [f"unlisted file in bundle: {rel}" for rel in sorted(on_disk - set(listed))]
    if expected is not None:
        wanted = {entry["path"]: entry["sha256"] for entry in expected}
        if {rel: item["sha256"] for rel, item in listed.items()} != wanted:
            problems.append(f"{folder.name}: staged listing differs from the verified selection")
    if problems:
        raise TransferError(problems)
    return record


def confined_bundle(folder, rel):
    """``folder / rel`` for the bundle record itself (a plain file name, no links)."""
    path = Path(folder) / rel
    if os.path.lexists(path) and is_redirecting(os.lstat(path)):
        raise TransferError([f"path refused (link): {rel}"])
    return path


def check_bundle(bundle):
    """Public check: refuses a ``.partial`` folder, then ``verify_contents``."""
    bundle = Path(bundle)
    if bundle.name.endswith(".partial"):
        raise TransferError([f"{bundle.name}: an incomplete export (.partial) is never a bundle"])
    if not bundle.is_dir():
        raise TransferError([f"{bundle}: not a bundle folder"])
    return verify_contents(bundle)


def export_bundle(project, dest, *, now=None, fault=None):
    """Export the verified selection of ``project`` into a new ``<dest>/<bundle_id>/``.

    Order: strict source check (``verify_local``), destination guard, exclusive copies into
    ``<id>.partial/`` with each copy re-hashed, ``bundle.json`` written into the staging folder,
    ``verify_contents`` of the staged folder against the verified selection, then the rename to
    ``<id>``. ``fault(step)`` is a test hook after ``copied``, ``recorded``, ``validated`` and
    ``published``. An interruption leaves only ``<id>.partial``, which nothing reads; a retry gets
    a new id. Returns ``(bundle folder, record)``.
    """
    fault = fault or (lambda step: None)
    project, dest = Path(project), Path(dest)
    write_new = _tool("acceptance_baseline")._write_new
    entries, absent = verify_local(project)
    guard_destination(project, dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_sha = _sha(project / "tools/acceptance_manifest.json")
    bundle_id = _new_id(dest, manifest_sha, now or datetime.now(timezone.utc))
    final, partial = dest / bundle_id, dest / (bundle_id + ".partial")
    partial.mkdir()
    files = []
    for entry in entries:
        target = partial / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        write_new(target, confined(project, entry["path"]).read_bytes())
        if _sha(target) != entry["sha256"]:
            raise TransferError([f"source changed while copying: {entry['path']}"])
        files.append(dict(path=entry["path"], sha256=entry["sha256"], size=target.stat().st_size, **{"class": entry["cls"]}))
    fault("copied")
    record = dict(schema=SCHEMA, bundle_id=bundle_id, created_utc=(now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  manifest_sha256=manifest_sha, example_manifest_sha256=_sha(project / "tools/example_manifest.json"),
                  files=files, absent=absent)
    write_new(partial / RECORD, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    fault("recorded")
    verify_contents(partial, expected=entries)
    fault("validated")
    if final.exists():
        raise TransferError([f"bundle {bundle_id} appeared during the export; not replaced"])
    os.rename(partial, final)
    fault("published")
    return final, record


def _temporary(bundle_id, rel):
    """This bundle's temporary for ``rel``: a short flat name in the dedicated area (MAX_PATH)."""
    return f"{TMP_AREA}/{bundle_id}/{hashlib.sha256(rel.encode('utf-8')).hexdigest()[:24]}.tmp"


def preflight(project, bundle, record):
    """Plan a restore without writing: ``(install, untouched, unrequired, absent)``; raises on any refusal.

    The effective private record is the destination's (which must equal the bundle's when both
    exist) or, when the destination has none, the bundle's validated copy. Each selected target is
    absent (scheduled from the bundle, whose copy must carry the destination's recorded digest),
    present and matching (untouched) or conflicting. Recorded private directories and capture
    folders are planned member by member, and their planned combined contents must equal the
    recorded set: an unrecorded extra member, a conflicting member or a missing member the bundle
    cannot supply refuses the restore.
    """
    project, bundle = Path(project), Path(bundle)
    listed = {item["path"]: item for item in record["files"]}
    problems = []
    try:
        local = _local_private_bytes(project)
    except TransferError as exc:
        raise TransferError(exc.problems) from None
    if local is not None:
        if PRIVATE_RECORD in listed and hashlib.sha256(local).hexdigest() != listed[PRIVATE_RECORD]["sha256"]:
            problems.append("conflict: the destination's private-inputs.json differs from the bundle's")
        private_bytes = local
    else:
        private_bytes = (bundle / PRIVATE_RECORD).read_bytes() if PRIVATE_RECORD in listed else None

    def read_capture(capture_id):
        rel = f"{ACCEPTANCE}/baseline-staging/{capture_id}/capture.json"
        if rel in listed:
            return (bundle / rel).read_bytes()
        return confined(project, rel).read_bytes() if os.path.lexists(project / rel) else None

    entries, absent, directories, more = select_entries(project, [project, bundle], private_bytes, read_capture)
    problems += more
    install, untouched = [], []
    for entry in entries:
        rel = entry["path"]
        digest = entry["sha256"] if entry["sha256"] is not None else (listed.get(rel) or {}).get("sha256")
        try:
            target = confined(project, rel)
            confined(project, _temporary(record["bundle_id"], rel), temporary=True)
        except TransferError as exc:
            problems += exc.problems
            continue
        if os.path.lexists(target):
            if not target.is_file():
                problems.append(f"conflict: not a regular file: {rel}")
            elif digest is not None and _sha(target) != digest:
                problems.append(f"conflict: exists and differs from its record: {rel}")
            else:
                untouched.append(rel)
        elif rel not in listed:
            problems.append(f"missing from the bundle (required by this checkout's records): {rel}")
        elif listed[rel]["sha256"] != digest:
            problems.append(f"bundle copy does not match this checkout's record: {rel}")
        else:
            install.append((rel, digest))
    for folder, members in sorted(directories.items()):
        for extra in sorted(_members_on_disk(project, folder) - members):
            problems.append(f"conflict: unrecorded file in a recorded directory: {extra}")
    selected = {entry["path"] for entry in entries}
    unrequired = sorted(set(listed) - selected)
    if problems:
        raise TransferError(problems)
    install.sort(key=lambda item: item[0] == PRIVATE_RECORD)   # the private record goes last
    return install, untouched, unrequired, absent


def _install(project, bundle, bundle_id, rel, digest, fault):
    """Install one file through this bundle's temporary; never overwrites an authoritative file."""
    baseline = _tool("acceptance_baseline")
    target = confined(project, rel)
    temporary = confined(project, _temporary(bundle_id, rel), temporary=True)
    if os.path.lexists(target):
        if target.is_file() and _sha(target) == digest:
            if os.path.lexists(temporary):
                temporary.unlink()          # interrupted after publication: the copy is complete
            return "already present"
        raise TransferError([f"conflict: {rel} appeared during the restore and differs; not overwritten"])
    if os.path.lexists(temporary):
        temporary.unlink()                  # interrupted during the write or before publication
    temporary.parent.mkdir(parents=True, exist_ok=True)
    baseline._write_new(temporary, confined(bundle, rel).read_bytes())
    if _sha(temporary) != digest:
        temporary.unlink()
        raise TransferError([f"temporary copy does not match its record: {rel}"])
    fault("temp-written", rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    confined(project, rel)                  # link policy again, just before publication
    try:
        baseline._publish(temporary, target)
    except FileExistsError:
        raise TransferError([f"conflict: {rel} appeared during the restore; not overwritten"]) from None
    if _sha(target) != digest:
        raise TransferError([f"published bytes do not match the record: {rel}"])
    fault("published", rel)
    return "installed"


def restore_bundle(project, bundle, *, dry_run=False, fault=None):
    """Restore ``bundle`` into ``project`` (the checkout this tool belongs to in normal use).

    ``check_bundle``, then ``preflight`` (no write on any refusal), then each scheduled file through
    ``_install`` with ``private-inputs.json`` last, then ``verify_local`` on the result. The
    dedicated temporary area keeps temporaries outside every hash-recorded directory; only this
    bundle's empty temporary folder is removed, and other files there are reported and left alone.
    ``fault(step, path)`` is a test hook. Returns a summary dict.
    """
    fault = fault or (lambda step, rel: None)
    project, bundle = Path(project), Path(bundle)
    record = check_bundle(bundle)
    install, untouched, unrequired, absent = preflight(project, bundle, record)
    area = project / TMP_AREA
    others = sorted(path.name for path in area.iterdir() if path.name != record["bundle_id"]) if area.is_dir() else []
    summary = dict(bundle_id=record["bundle_id"], scheduled=[rel for rel, _ in install], untouched=untouched,
                   unrequired=unrequired, absent=absent, other_temporaries=others, installed=[], already_present=[])
    if dry_run:
        return summary
    for rel in untouched:
        # A matching target may have been published by an interrupted run of this bundle between
        # the hard link and the temporary's removal: discard only this bundle's temporary for it.
        temporary = confined(project, _temporary(record["bundle_id"], rel), temporary=True)
        if os.path.lexists(temporary):
            temporary.unlink()
    for rel, digest in install:
        status = _install(project, bundle, record["bundle_id"], rel, digest, fault)
        summary["installed" if status == "installed" else "already_present"].append(rel)
    own = project / TMP_AREA / record["bundle_id"]
    if own.is_dir() and not any(own.iterdir()):
        own.rmdir()
    verify_local(project)
    return summary


def excluded_summary(project, selected):
    """What an export leaves behind: counts of other ``.verification`` entries and unselected names."""
    project = Path(project)
    acceptance = project / ACCEPTANCE

    def group(prefix, path):
        """Collapse run folders such as ``install-abc123`` into ``install-*``."""
        stem = path.name.split("-")[0] + "-*" if path.is_dir() and "-" in path.name else path.name
        return f"{prefix}/{stem}"

    counts = {}
    for parent, skip in ((project / ".verification", {"acceptance", Path(WHEEL).name}), (acceptance, set(AREAS))):
        for path in sorted(parent.iterdir()) if parent.is_dir() else []:
            if path.name not in skip:
                key = group(parent.relative_to(project).as_posix(), path)
                counts[key] = counts.get(key, 0) + 1
    names = []
    for area in AREAS:
        folder = acceptance / area
        if folder.is_dir():
            names += sorted(path.relative_to(project).as_posix() for path in folder.rglob("*")
                            if path.is_file() and path.relative_to(project).as_posix() not in selected)
    return counts, names


def _counts(items):
    totals = {cls: 0 for cls in CLASSES}
    for item in items:
        totals[item] += 1
    return ", ".join(f"{cls} {count}" for cls, count in totals.items())


def main(argv=None):
    """Command line for this checkout (see the module docstring); returns the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export", help="export this checkout's verified local artifacts")
    export.add_argument("--dest", required=True, type=Path)
    check = sub.add_parser("check", help="verify a bundle against its bundle.json")
    check.add_argument("bundle", type=Path)
    restore = sub.add_parser("restore", help="restore a bundle into this checkout")
    restore.add_argument("bundle", type=Path)
    restore.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "export":
            folder, record = export_bundle(PROJECT, args.dest)
            counts, names = excluded_summary(PROJECT, {item["path"] for item in record["files"]})
            print(f"exported {folder}")
            print(f"files: {len(record['files'])} ({_counts(item['class'] for item in record['files'])})")
            print("absent optional resources: " + (", ".join(record["absent"]) or "none"))
            print(f"not exported: {POINTER} (supply it separately in each destination)")
            for key, count in sorted(counts.items()):
                print(f"not exported: {key} ({count})")
            print(f"not exported inside transferred areas: {len(names)} file(s)")
            for name in names:
                print("  " + name)
            return 0
        if args.action == "check":
            record = check_bundle(args.bundle)
            print(f"bundle {record['bundle_id']} verified: {len(record['files'])} files "
                  f"({_counts(item['class'] for item in record['files'])})")
            return 0
        summary = restore_bundle(PROJECT, args.bundle, dry_run=args.dry_run)
        print(f"bundle {summary['bundle_id']}: {len(summary['untouched'])} present and matching (untouched), "
              f"{len(summary['scheduled'])} to install")
        for rel in summary["scheduled"]:
            print("  install " + rel)
        for rel in summary["unrequired"]:
            print("  not required by this checkout's records (not installed): " + rel)
        for name in summary["other_temporaries"]:
            print("  other temporary left alone: " + f"{TMP_AREA}/{name}")
        print("absent optional resources: " + (", ".join(summary["absent"]) or "none"))
        if args.dry_run:
            print("dry run: nothing written")
            return 0
        print(f"installed {len(summary['installed'])}, already present {len(summary['already_present'])}")
        acceptance = _tool("acceptance")
        manifest = acceptance.validate_expected(sorted(json.loads(
            (PROJECT / "tools/acceptance_manifest.json").read_text(encoding="utf-8"))["expected_artifacts"]["cases"]))
        acceptance.check_private_inputs(sorted(acceptance.PRIVATE_CASE_INPUTS))
        print(f"post-install verification passed: {len(manifest['expected_artifacts']['cases'])} expected cases, "
              "approved references, adopted inputs, definitions and available private inputs")
        pointer = PROJECT / POINTER
        if pointer.is_file():
            try:
                fields = json.loads(pointer.read_text(encoding="utf-8"))
                ok = {"python", "host", "version", "stock_files_verified", "wheel_sha256"} <= set(fields) and \
                    fields["wheel_sha256"] == acceptance.STOCK_WHEEL_SHA256 and Path(fields["python"]).is_file()
            except (ValueError, TypeError, KeyError):
                ok = False
            print(f"{POINTER}: " + ("present and well formed" if ok else "present but malformed or stale (fix by hand)"))
        else:
            print(f"{POINTER}: absent; supply it separately before running acceptance (README)")
        return 0
    except TransferError as exc:
        print("refused:", file=sys.stderr)
        for problem in exc.problems:
            print("  " + problem, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
