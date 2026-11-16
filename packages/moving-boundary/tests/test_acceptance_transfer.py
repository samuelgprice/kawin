"""Local-artifact transfer (``tools/acceptance_transfer.py``) on synthetic tmp_path projects only.

Covers the selection classes and exclusions, export ordering and refusals, capture provenance
pinned to ``capture_record.capture_json_sha256``, bundle checks, path and link confinement,
restore preflight (untouched, scheduled, conflicting), private-record recovery, interruption
and retry through the dedicated temporary area, and output privacy. No real artifact is read.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import types

import pytest

PROJECT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 16, 9, 0, 0, tzinfo=timezone.utc)
ACC = ".verification/acceptance"
CID = "20261115T224052Z-a26d099-1"
A1 = "20261111T222849Z-5aedde7-1"
A2 = "20261113T174227Z-3047510-1"
SURROGATE = f"{ACC}/inputs/fecrni.surrogate"
MEMBERS = ("a.json", "b.npz", "sub/c.txt")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = _load("acceptance_transfer_test", PROJECT / "tools/acceptance_transfer.py")
WHEEL_BYTES = b"synthetic stock wheel"


@pytest.fixture(autouse=True)
def _synthetic_wheel(monkeypatch):
    """The synthetic projects carry a fake wheel; point the pinned digest at it."""
    monkeypatch.setattr(T._tool("acceptance"), "STOCK_WHEEL_SHA256", hashlib.sha256(WHEEL_BYTES).hexdigest())


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _put(root, rel, data):
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha(data)


def _json(data):
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_private(root, record):
    _put(root, f"{ACC}/inputs/private-inputs.json", _json(record))


def _private_record(root):
    return json.loads((Path(root) / f"{ACC}/inputs/private-inputs.json").read_text(encoding="utf-8"))


def _source(tmp_path):
    """A complete synthetic source checkout: records in tools/ and docs/, artifacts in .verification/."""
    src = tmp_path / "src"
    expected = {}
    for case in ("C01", "C04", "O04", "C33"):
        expected[case] = {suffix: _put(src, f"{ACC}/reference/{case}.{suffix}", f"{case}.{suffix} frozen".encode())
                          for suffix in ("json", "npz")}
    capture = {"C33": dict(expected["C33"])}
    for suffix in ("json", "npz"):
        _put(src, f"{ACC}/baseline-staging/{CID}/C33.{suffix}", f"C33.{suffix} frozen".encode())
    capture_digest = _put(src, f"{ACC}/baseline-staging/{CID}/capture.json",
                          _json({"capture_id": CID, "files": capture, "head": "a26d099"}))
    expected["C33"].update(capture_id=CID, capture_record={"capture_json_sha256": capture_digest})
    artifacts1 = {"C01": {suffix: _put(src, f"{ACC}/reference-approved/{A1}/C01.{suffix}", f"C01.{suffix} A1".encode())
                          for suffix in ("json", "npz")}}
    artifacts2 = {case: {suffix: _put(src, f"{ACC}/reference-approved/{A2}/{case}.{suffix}", f"{case}.{suffix} A2".encode())
                         for suffix in ("json", "npz")} for case in ("C01", "O04")}
    artifacts2["O04"]["settings"] = "private-carried-forward"
    approvals = {}
    for approval_id, artifacts, previous, extra in (
            (A1, artifacts1, {"C01": "reference"}, {}),
            (A2, artifacts2, {"C01": A1, "O04": "reference"}, {"private_settings": "verified-equal"})):
        report = _put(src, f"docs/approvals/{approval_id}.json", _json({"approval_id": approval_id}))
        approvals[approval_id] = dict(package_commit="0123456789abcdef0123456789abcdef01234567", reason="LEDGERS test",
                                      cases=sorted(artifacts), directory=f"reference-approved/{approval_id}",
                                      artifacts=artifacts, previous=previous,
                                      report={"path": f"docs/approvals/{approval_id}.json", "sha256": report}, **extra)
    manifest = {
        "expected_artifacts": {"cases": expected},
        "adopted_inputs": {"files": {"open-surrogate.npz": {
            "sha256": _put(src, f"{ACC}/inputs/open-surrogate.npz", b"adopted"), "required_by": ["C24"]}}},
        "thermocalc_sampling": {"input": "O06.json", "input_sha256": _put(src, f"{ACC}/inputs/O06.json", b"tc input")},
        "default_phase_sampling": {"database_input": "FeCrNi.tdb"},
        "approved_references": {"in_effect": {"C01": A2, "O04": A2}, "approvals": approvals,
                                "history": [{"event": "approve", "approval_id": A1}, {"event": "approve", "approval_id": A2}]},
    }
    _put(src, "tools/acceptance_manifest.json", _json(manifest))
    examples = {"cases": {"E04": {"surrogate": "lifecycle"}, "E06": {"surrogate": "fecrni.surrogate"},
                          "E05": {"surrogate": "pycalphad-reference.npz", "provider_arrays": "pycalphad-reference.npz"}},
                "settings_sha256": {"C04.settings.json": _put(src, f"{ACC}/example_settings/C04.settings.json", b"C04 settings")}}
    _put(src, "tools/example_manifest.json", _json(examples))
    record = {"exb5.json": {"sha256": {"exb5.json": _put(src, f"{ACC}/inputs/exb5.json", b"private exb5")}},
              "fecrni.surrogate": {"sha256": {m: _put(src, f"{SURROGATE}/{m}", f"member {m}".encode()) for m in MEMBERS}},
              "example_settings": {"sha256": {"O04.settings.json": _put(src, f"{ACC}/example_settings/O04.settings.json", b"O04 private")}}}
    _write_private(src, record)
    _put(src, T.WHEEL, WHEEL_BYTES)
    _put(src, f"{ACC}/pilot/O05-sampling.json", b'{"elapsed_seconds": 30}')
    _put(src, f"{ACC}/pilot/C04.json", b"pilot run output")
    _put(src, f"{ACC}/reference/C01.log", b"log")
    _put(src, f"{ACC}/reference/prepare.json", b"prepare")
    _put(src, f"{ACC}/inputs/pycalphad.npz", b"unrecorded")
    _put(src, f"{ACC}/install-abc123/target/file.py", b"run output")
    _put(src, ".verification/public-environment.json", b'{"python": "C:/elsewhere/python.exe"}')
    _put(src, ".verification/evidence/c33/notes.txt", b"evidence")
    return src


def _destination(tmp_path, src, name="dst"):
    """A clean checkout of the same commit: committed records only, no .verification/."""
    dst = tmp_path / name
    for rel in ("tools/acceptance_manifest.json", "tools/example_manifest.json"):
        _put(dst, rel, (src / rel).read_bytes())
    shutil.copytree(src / "docs", dst / "docs")
    return dst


def _tree(root):
    """Every file (bytes digest and mtime) and directory under ``root`` (zero-write checks)."""
    root = Path(root)
    files = {p.relative_to(root).as_posix(): (_sha(p.read_bytes()), p.stat().st_mtime_ns)
             for p in sorted(root.rglob("*")) if p.is_file()}
    folders = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir()}
    return files, folders


def _export(src, tmp_path, **kwargs):
    return T.export_bundle(src, tmp_path / "bundles", now=NOW, **kwargs)


def _rewrite_bundle_record(bundle, change):
    """Edit a bundle and keep bundle.json self-consistent (simulates a tampered but well-formed bundle)."""
    record = json.loads((bundle / T.RECORD).read_text(encoding="utf-8"))
    change(record)
    for item in record["files"]:
        path = bundle / item["path"]
        if path.is_file():
            item["sha256"], item["size"] = _sha(path.read_bytes()), path.stat().st_size
    (bundle / T.RECORD).write_bytes(_json(record))


def _drop_from_bundle(bundle, rel):
    (bundle / rel).unlink()
    _rewrite_bundle_record(bundle, lambda record: record.update(files=[i for i in record["files"] if i["path"] != rel]))


def _link_dir(link, target):
    """A junction (Windows, no privilege needed) or a symlink (POSIX) at ``link`` pointing to ``target``."""
    Path(target).mkdir(parents=True, exist_ok=True)
    Path(link).parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def _raise_at(step_name, rel=None, exc=RuntimeError):
    def fault(step, path=None):
        if step == step_name and (rel is None or path == rel):
            raise exc(f"injected at {step} {path}")
    return fault


# ---------------------------------------------------------------- selection ----

def test_selection_classes_absences_and_exclusions(tmp_path):
    src = _source(tmp_path)
    entries, absent = T.verify_local(src)
    by_path = {e["path"]: e for e in entries}
    assert {f"{ACC}/reference/{c}.{s}" for c in ("C01", "C04", "O04", "C33") for s in ("json", "npz")} <= set(by_path)
    assert f"{ACC}/reference-approved/{A1}/C01.json" in by_path           # superseded approval kept
    assert f"{ACC}/reference-approved/{A2}/O04.npz" in by_path
    assert not any(p.endswith("O04.settings.json") and "reference-approved" in p for p in by_path)   # carried forward
    assert by_path[f"{ACC}/baseline-staging/{CID}/capture.json"]["cls"] == "required"
    assert by_path[T.WHEEL]["cls"] == "required"
    assert by_path[f"{ACC}/example_settings/C04.settings.json"]["cls"] == "required"
    assert by_path[f"{ACC}/example_settings/O04.settings.json"]["cls"] == "private"
    assert {f"{SURROGATE}/{m}" for m in MEMBERS} | {f"{ACC}/inputs/exb5.json", T.PRIVATE_RECORD,
                                                     f"{ACC}/inputs/O06.json"} <= set(by_path)
    assert by_path[T.PILOT]["cls"] == "optional" and by_path[T.PILOT]["sha256"]
    assert {f"{ACC}/inputs/{n}" for n in ("femnc.json", "fec.json", "FeCrNi.tdb", "pycalphad-reference.npz")} <= set(absent)
    assert f"{ACC}/inputs/lifecycle" not in absent
    for left in (f"{ACC}/reference/C01.log", f"{ACC}/reference/prepare.json", f"{ACC}/inputs/pycalphad.npz",
                 f"{ACC}/pilot/C04.json", ".verification/public-environment.json"):
        assert left not in by_path
    counts, names = T.excluded_summary(src, set(by_path))
    assert f"{ACC}/install-*" in counts and ".verification/public-environment.json" in counts
    assert ".verification/evidence" in counts
    assert {f"{ACC}/reference/C01.log", f"{ACC}/inputs/pycalphad.npz", f"{ACC}/pilot/C04.json"} <= set(names)


# ------------------------------------------------------------------- export ----

def test_export_validates_staging_then_publishes(tmp_path):
    src = _source(tmp_path)
    steps = []
    folder, record = _export(src, tmp_path, fault=lambda step: steps.append(step))
    assert steps == ["copied", "recorded", "validated", "published"]
    assert folder.name == record["bundle_id"] and not list(folder.parent.glob("*.partial"))
    assert T.check_bundle(folder)["bundle_id"] == record["bundle_id"]
    entries, absent = T.verify_local(src)
    assert {i["path"]: i["sha256"] for i in record["files"]} == {e["path"]: e["sha256"] for e in entries}
    assert record["absent"] == absent and T.POINTER not in {i["path"] for i in record["files"]}
    again, _ = _export(src, tmp_path)
    assert again.name.endswith("-2")                                      # ids are never reused


@pytest.mark.parametrize("step", ["copied", "recorded", "validated"])
def test_export_interruption_leaves_only_partial(tmp_path, step):
    src = _source(tmp_path)
    with pytest.raises(RuntimeError, match="injected"):
        _export(src, tmp_path, fault=_raise_at(step))
    folders = [p.name for p in (tmp_path / "bundles").iterdir()]
    assert len(folders) == 1 and folders[0].endswith(".partial")
    with pytest.raises(T.TransferError, match="partial"):
        T.check_bundle(tmp_path / "bundles" / folders[0])
    retry, _ = _export(src, tmp_path)
    assert retry.name.endswith("-2") and T.check_bundle(retry)


def test_export_validation_failure_is_never_published(tmp_path):
    src = _source(tmp_path)

    def corrupt(step):
        if step == "recorded":
            staged = next((tmp_path / "bundles").glob("*.partial"))
            (staged / f"{ACC}/reference/C01.json").write_bytes(b"flipped")

    with pytest.raises(T.TransferError, match="bundle file changed"):
        _export(src, tmp_path, fault=corrupt)
    assert [p.name.endswith(".partial") for p in (tmp_path / "bundles").iterdir()] == [True]


def _remove(rel):
    return lambda src: (src / rel).unlink()


def _edit_private(change):
    def apply(src):
        record = _private_record(src)
        change(record)
        _write_private(src, record)
    return apply


def _edit_manifest(change):
    def apply(src):
        manifest = json.loads((src / "tools/acceptance_manifest.json").read_text(encoding="utf-8"))
        change(manifest)
        (src / "tools/acceptance_manifest.json").write_bytes(_json(manifest))
    return apply


EXPORT_REFUSALS = {
    "missing required": (_remove(f"{ACC}/reference/C04.npz"), "missing"),
    "changed approved": (lambda src: _put(src, f"{ACC}/reference-approved/{A1}/C01.json", b"changed"), "changed"),
    "malformed approval": (_edit_manifest(lambda m: m["approved_references"]["approvals"][A1].pop("reason")),
                           "approval records malformed"),
    "no private record": (_remove(T.PRIVATE_RECORD), "private-inputs.json missing"),
    "missing entry": (_edit_private(lambda r: r.pop("fecrni.surrogate")), "no record entry"),
    "record not an object": (lambda src: _put(src, T.PRIVATE_RECORD, b"[]"), "malformed"),
    "empty map": (_edit_private(lambda r: r.update({"exb5.json": {"sha256": {}}})), "malformed"),
    "bad hex": (_edit_private(lambda r: r.update({"exb5.json": {"sha256": {"exb5.json": "XYZ"}}})), "malformed"),
    "unrecorded member": (lambda src: _put(src, f"{SURROGATE}/extra.bin", b"extra"), "not hashed"),
    "unhashed settings": (lambda src: _put(src, f"{ACC}/example_settings/X99.settings.json", b"x"), "not hashed"),
    "key escapes directory": (_edit_private(lambda r: r["fecrni.surrogate"]["sha256"].update({"../exb5.json": "0" * 64})),
                              "path refused"),
}


@pytest.mark.parametrize("name", sorted(EXPORT_REFUSALS))
def test_export_refusals_write_nothing(tmp_path, name):
    src = _source(tmp_path)
    change, message = EXPORT_REFUSALS[name]
    change(src)
    with pytest.raises(T.TransferError) as caught:
        _export(src, tmp_path)
    assert any(message in problem for problem in caught.value.problems), caught.value.problems
    assert not (tmp_path / "bundles").exists()


def test_export_refuses_a_destination_inside_a_git_work_tree(tmp_path):
    src = _source(tmp_path)
    (tmp_path / "repo/.git").mkdir(parents=True)
    with pytest.raises(T.TransferError, match="Git work tree"):
        T.export_bundle(src, tmp_path / "repo/bundles", now=NOW)
    with pytest.raises(T.TransferError, match="transferred area"):
        T.export_bundle(src, src / ACC / "reference/bundles", now=NOW)
    folder, _ = T.export_bundle(src, src / ".verification/transfer", now=NOW)    # ignored local area: allowed
    assert T.check_bundle(folder)


# ------------------------------------------------------------------ capture ----

def test_capture_provenance_is_pinned_to_the_committed_record_hash(tmp_path):
    src = _source(tmp_path)
    path = src / f"{ACC}/baseline-staging/{CID}/capture.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["head"] = "fffffff"                                          # provenance only; fixtures untouched
    path.write_bytes(_json(document))
    with pytest.raises(T.TransferError, match="capture_json_sha256"):
        T.verify_local(src)


@pytest.mark.parametrize("change, message", [
    (lambda src: _put(src, f"{ACC}/baseline-staging/{CID}/C33.npz", b"other"), "changed"),
    (lambda src: _put(src, f"{ACC}/baseline-staging/{CID}/notes.txt", b"extra"), "unlisted file in capture folder"),
    (_edit_manifest(lambda m: m["expected_artifacts"]["cases"]["C01"].update(
        capture_id=CID, capture_record={"capture_json_sha256": "0" * 64})), "disagree"),
])
def test_capture_folder_refusals(tmp_path, change, message):
    src = _source(tmp_path)
    change(src)
    with pytest.raises(T.TransferError) as caught:
        T.verify_local(src)
    assert any(message in problem for problem in caught.value.problems), caught.value.problems


# -------------------------------------------------------------------- check ----

@pytest.mark.parametrize("change, message", [
    (lambda b: (b / f"{ACC}/reference/C01.json").write_bytes(b"x"), "bundle file changed"),
    (lambda b: (b / f"{ACC}/reference/C01.json").unlink(), "bundle file missing"),
    (lambda b: _put(b, f"{ACC}/reference/C99.json", b"x"), "unlisted file in bundle"),
    (lambda b: (b / T.RECORD).write_bytes(b"{"), "bundle.json missing or malformed"),
    (lambda b: _rewrite_bundle_record(b, lambda r: r["files"].append(
        dict(path="../escape.json", sha256="0" * 64, size=0, **{"class": "required"}))), "path refused"),
])
def test_check_refuses_a_damaged_bundle(tmp_path, change, message):
    folder, _ = _export(_source(tmp_path), tmp_path)
    change(folder)
    with pytest.raises(T.TransferError) as caught:
        T.check_bundle(folder)
    assert any(message in problem for problem in caught.value.problems), caught.value.problems


# ----------------------------------------------------------- path policy ----

@pytest.mark.parametrize("rel", [
    "", "/abs/x", f"{ACC}/reference/../inputs/x", f"{ACC}/reference/./x", f"{ACC}/reference//x",
    f"{ACC}\\reference\\x", "C:/x", f"{ACC}/reference/x.json:stream", f"{ACC}/reference/x.", f"{ACC}/reference/x ",
    "//server/share/x", f"{ACC}/other/x", "tools/acceptance.py", f"{ACC}/reference/x\0",
    f"{T.TMP_AREA}/id/x.tmp",
])
def test_path_policy_refusals(rel):
    with pytest.raises(T.TransferError, match="path refused"):
        T.check_path(rel)


def test_path_policy_accepts_transferable_locations():
    for rel in (T.WHEEL, f"{ACC}/reference/C01.json", f"{SURROGATE}/sub/c.txt", T.PILOT):
        assert T.check_path(rel) == rel
    assert T.check_path(f"{T.TMP_AREA}/id/x.tmp", temporary=True)


def test_case_aliases_are_duplicates(tmp_path):
    folder, _ = _export(_source(tmp_path), tmp_path)

    def alias(record):
        item = next(i for i in record["files"] if i["path"] == f"{ACC}/reference/C01.json")
        record["files"].append(dict(item, path=f"{ACC}/reference/c01.JSON"))

    _rewrite_bundle_record(folder, alias)
    with pytest.raises(T.TransferError) as caught:
        T.check_bundle(folder)
    assert any("duplicate" in p or "missing" in p for p in caught.value.problems)
    assert any("duplicate" in p for p in caught.value.problems)
    src = _source(tmp_path / "second")
    examples = json.loads((src / "tools/example_manifest.json").read_text(encoding="utf-8"))
    examples["settings_sha256"]["c04.SETTINGS.json"] = examples["settings_sha256"]["C04.settings.json"]
    (src / "tools/example_manifest.json").write_bytes(_json(examples))
    with pytest.raises(T.TransferError, match="duplicate"):
        T.verify_local(src)


def test_redirect_classifier():
    regular = types.SimpleNamespace(st_mode=0o100644, st_reparse_tag=0)
    cloud = types.SimpleNamespace(st_mode=0o100644, st_reparse_tag=0x9000701A)    # OneDrive placeholder
    junction = types.SimpleNamespace(st_mode=0o040755, st_reparse_tag=0xA0000003)
    symlink = types.SimpleNamespace(st_mode=0o120777, st_reparse_tag=0)
    assert not T.is_redirecting(regular) and not T.is_redirecting(cloud)
    assert T.is_redirecting(junction) and T.is_redirecting(symlink)


def test_restore_refuses_a_destination_junction_without_writing(tmp_path):
    src = _source(tmp_path)
    folder, _ = _export(src, tmp_path)
    dst = _destination(tmp_path, src)
    outside = tmp_path / "outside"
    _link_dir(dst / f"{ACC}/reference-approved", outside)
    before = _tree(dst), _tree(outside)
    with pytest.raises(T.TransferError, match="link or junction"):
        T.restore_bundle(dst, folder)
    assert (_tree(dst), _tree(outside)) == before


def test_export_refuses_a_source_junction(tmp_path):
    src = _source(tmp_path)
    moved = tmp_path / "moved-surrogate"
    shutil.move(str(src / SURROGATE), str(moved))
    _link_dir(src / SURROGATE, moved)
    with pytest.raises(T.TransferError, match="link or junction"):
        _export(src, tmp_path)
    assert not (tmp_path / "bundles").exists()


# ------------------------------------------------------------------ restore ----

def test_restore_into_a_clean_checkout_then_rerun_writes_nothing(tmp_path):
    src = _source(tmp_path)
    folder, record = _export(src, tmp_path)
    dst = _destination(tmp_path, src)
    summary = T.restore_bundle(dst, folder)
    assert set(summary["installed"]) == {i["path"] for i in record["files"]}
    assert summary["installed"][-1] == T.PRIVATE_RECORD
    assert not (dst / T.TMP_AREA / record["bundle_id"]).exists()
    assert not (dst / T.POINTER).exists()
    T.verify_local(dst)
    assert T._tool("acceptance").private_input_problems("fecrni.surrogate", _private_record(dst),
                                                        inputs=dst / ACC / "inputs") == []
    before = _tree(dst)
    again = T.restore_bundle(dst, folder)
    assert again["scheduled"] == [] and len(again["untouched"]) == len(record["files"])
    assert _tree(dst) == before


def test_dry_run_on_the_source_checkout_writes_nothing(tmp_path):
    src = _source(tmp_path)
    folder, record = _export(src, tmp_path)
    before = _tree(src)
    summary = T.restore_bundle(src, folder, dry_run=True)
    assert summary["scheduled"] == [] and len(summary["untouched"]) == len(record["files"])
    assert _tree(src) == before


@pytest.mark.parametrize("prepare, message", [
    (lambda dst: _put(dst, f"{ACC}/reference/C01.json", b"different"), "exists and differs"),
    (lambda dst: _put(dst, T.PRIVATE_RECORD, b"{}\n"), "private-inputs.json differs"),
    (lambda dst: _put(dst, f"{SURROGATE}/extra.bin", b"extra"), "unrecorded file in a recorded directory"),
    (lambda dst: _put(dst, f"{ACC}/baseline-staging/{CID}/notes.txt", b"extra"), "unrecorded file in a recorded directory"),
])
def test_restore_conflicts_refuse_before_any_write(tmp_path, prepare, message):
    src = _source(tmp_path)
    folder, _ = _export(src, tmp_path)
    dst = _destination(tmp_path, src)
    prepare(dst)
    before = _tree(dst)
    with pytest.raises(T.TransferError) as caught:
        T.restore_bundle(dst, folder)
    assert any(message in problem for problem in caught.value.problems), caught.value.problems
    assert _tree(dst) == before


def _bundle_without_exb5(tmp_path):
    src = _source(tmp_path / "noexb5")
    (src / f"{ACC}/inputs/exb5.json").unlink()
    _edit_private(lambda r: r.pop("exb5.json"))(src)
    return src, _export(src, tmp_path / "noexb5")[0]


@pytest.mark.parametrize("scenario, message", [
    ("record lacks entry", "no record entry"),
    ("record malformed", "malformed"),
    ("record differs", "private-inputs.json differs"),
    ("no record, uncovered input", "no record entry"),
    ("no record, mismatching input", "exists and differs"),
    ("tampered bundle record", "no record entry"),
])
def test_restore_private_record_refusals(tmp_path, scenario, message):
    src = _source(tmp_path)
    folder, _ = _export(src, tmp_path)
    dst = _destination(tmp_path, src)
    if scenario == "record lacks entry":
        _, folder = _bundle_without_exb5(tmp_path)
        _put(dst, T.PRIVATE_RECORD, (folder / T.PRIVATE_RECORD).read_bytes())
        _put(dst, f"{ACC}/inputs/exb5.json", b"private exb5")
    elif scenario == "record malformed":
        _put(dst, T.PRIVATE_RECORD, b"[]")
    elif scenario == "record differs":
        _put(dst, T.PRIVATE_RECORD, (folder / T.PRIVATE_RECORD).read_bytes() + b" ")   # well formed, other bytes
    elif scenario == "no record, uncovered input":
        _put(dst, f"{ACC}/inputs/femnc.json", b"private femnc")
    elif scenario == "no record, mismatching input":
        _put(dst, f"{ACC}/inputs/exb5.json", b"other bytes")
    else:
        def strip(record):
            data = json.loads((folder / T.PRIVATE_RECORD).read_text(encoding="utf-8"))
            data.pop("exb5.json")
            (folder / T.PRIVATE_RECORD).write_bytes(_json(data))
        _rewrite_bundle_record(folder, strip)
    before = _tree(dst)
    with pytest.raises(T.TransferError) as caught:
        T.restore_bundle(dst, folder)
    assert any(message in problem for problem in caught.value.problems), caught.value.problems
    assert _tree(dst) == before


def test_restore_recovers_private_files_without_a_destination_record(tmp_path):
    src = _source(tmp_path)
    folder, _ = _export(src, tmp_path)
    dst = _destination(tmp_path, src)
    _put(dst, f"{ACC}/inputs/exb5.json", b"private exb5")                # left by an interrupted restore
    stamp = (dst / f"{ACC}/inputs/exb5.json").stat().st_mtime_ns
    summary = T.restore_bundle(dst, folder)
    assert f"{ACC}/inputs/exb5.json" in summary["untouched"] and T.PRIVATE_RECORD in summary["installed"]
    assert (dst / f"{ACC}/inputs/exb5.json").stat().st_mtime_ns == stamp
    T.verify_local(dst)


def test_bundle_coverage_missing_required_and_unrequired_files(tmp_path):
    src = _source(tmp_path)
    folder, _ = _export(src, tmp_path)
    dst = _destination(tmp_path, src)
    _drop_from_bundle(folder, f"{ACC}/reference/C04.npz")
    before = _tree(dst)
    with pytest.raises(T.TransferError, match="missing from the bundle"):
        T.restore_bundle(dst, folder)
    assert _tree(dst) == before
    folder, _ = _export(src, tmp_path)
    _put(folder, f"{ACC}/reference/C99.json", b"not recorded here")
    _rewrite_bundle_record(folder, lambda r: r["files"].append(
        dict(path=f"{ACC}/reference/C99.json", sha256="0" * 64, size=0, **{"class": "required"})))
    summary = T.restore_bundle(dst, folder)
    assert summary["unrequired"] == [f"{ACC}/reference/C99.json"] and not (dst / f"{ACC}/reference/C99.json").exists()


# ----------------------------------------------------------- interruption ----

def _authoritative_files_are_complete(dst, record):
    for item in record["files"]:
        path = dst / item["path"]
        if path.exists():
            assert _sha(path.read_bytes()) == item["sha256"], item["path"]


def _surrogate_listing(dst):
    folder = dst / SURROGATE
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} if folder.is_dir() else set()


@pytest.mark.parametrize("step, exc", [("temp-written", RuntimeError), ("published", RuntimeError),
                                       ("temp-written", KeyboardInterrupt)])
def test_interrupted_restore_retries_from_the_same_bundle(tmp_path, step, exc):
    src = _source(tmp_path)
    folder, record = _export(src, tmp_path)
    dst = _destination(tmp_path, src)
    with pytest.raises(exc):
        T.restore_bundle(dst, folder, fault=_raise_at(step, f"{ACC}/reference-approved/{A2}/C01.npz", exc))
    _authoritative_files_are_complete(dst, record)
    T.restore_bundle(dst, folder)
    T.verify_local(dst)
    assert not (dst / T.TMP_AREA / record["bundle_id"]).exists()


def test_interrupted_before_the_private_record_then_retry(tmp_path):
    src = _source(tmp_path)
    folder, record = _export(src, tmp_path)
    dst = _destination(tmp_path, src)
    with pytest.raises(RuntimeError):
        T.restore_bundle(dst, folder, fault=_raise_at("published", f"{ACC}/inputs/exb5.json"))
    assert (dst / f"{ACC}/inputs/exb5.json").is_file() and not (dst / T.PRIVATE_RECORD).exists()
    T.restore_bundle(dst, folder)
    assert T._tool("acceptance").private_input_problems("exb5.json", _private_record(dst), inputs=dst / ACC / "inputs") == []
    T.verify_local(dst)


@pytest.mark.parametrize("mode", ["partial temp", "temp complete", "linked, temp kept"])
def test_surrogate_member_interruptions_keep_temporaries_outside_the_directory(tmp_path, mode):
    src = _source(tmp_path)
    folder, record = _export(src, tmp_path)
    dst = _destination(tmp_path, src)
    _put(dst, f"{T.TMP_AREA}/OTHER-BUNDLE/abc.tmp", b"another bundle's temporary")
    _put(dst, f"{T.TMP_AREA}/stray.txt", b"unrelated leftover")
    others = {rel: _tree(dst)[0][rel] for rel in (f"{T.TMP_AREA}/OTHER-BUNDLE/abc.tmp", f"{T.TMP_AREA}/stray.txt")}
    member = f"{SURROGATE}/b.npz"
    with pytest.raises(RuntimeError):
        T.restore_bundle(dst, folder, fault=_raise_at("temp-written", member))
    temporary = dst / T._temporary(record["bundle_id"], member)
    assert temporary.is_file() and _surrogate_listing(dst) <= set(MEMBERS)
    if mode == "partial temp":
        temporary.write_bytes(temporary.read_bytes()[:3])
    elif mode == "linked, temp kept":
        os.link(temporary, dst / member)                                 # interrupted between link and unlink
    assert _surrogate_listing(dst) <= set(MEMBERS)                         # no temporary inside the directory
    summary = T.restore_bundle(dst, folder)
    assert set(summary["other_temporaries"]) == {"OTHER-BUNDLE", "stray.txt"}
    assert _surrogate_listing(dst) == set(MEMBERS) and not temporary.exists()
    assert T._tool("acceptance").private_input_problems("fecrni.surrogate", _private_record(dst),
                                                        inputs=dst / ACC / "inputs") == []
    assert {rel: _tree(dst)[0][rel] for rel in others} == others
    T.verify_local(dst)


def _partially_restored_directory(tmp_path):
    src = _source(tmp_path)
    folder, record = _export(src, tmp_path)
    dst = _destination(tmp_path, src)
    first = f"{SURROGATE}/{MEMBERS[0]}"
    with pytest.raises(RuntimeError):
        T.restore_bundle(dst, folder, fault=_raise_at("published", first))
    assert _surrogate_listing(dst) == {MEMBERS[0]}
    return folder, dst, first


def test_partially_restored_private_directory_retry(tmp_path):
    folder, dst, first = _partially_restored_directory(tmp_path)
    stamp = (dst / first).stat().st_mtime_ns
    plan = T.restore_bundle(dst, folder, dry_run=True)
    assert first in plan["untouched"]
    assert {f"{SURROGATE}/{m}" for m in MEMBERS[1:]} <= set(plan["scheduled"])
    T.restore_bundle(dst, folder)
    assert (dst / first).stat().st_mtime_ns == stamp and _surrogate_listing(dst) == set(MEMBERS)
    assert T._tool("acceptance").private_input_problems("fecrni.surrogate", _private_record(dst),
                                                        inputs=dst / ACC / "inputs") == []


@pytest.mark.parametrize("damage, message", [
    ("conflicting member", "exists and differs"),
    ("extra member", "unrecorded file in a recorded directory"),
    ("bundle lacks a missing member", "missing from the bundle"),
])
def test_partially_restored_private_directory_refusals(tmp_path, damage, message):
    folder, dst, first = _partially_restored_directory(tmp_path)
    if damage == "conflicting member":
        (dst / first).write_bytes(b"conflict")
    elif damage == "extra member":
        _put(dst, f"{SURROGATE}/extra.bin", b"extra")
    else:
        _drop_from_bundle(folder, f"{SURROGATE}/{MEMBERS[2]}")
    before = _tree(dst)
    with pytest.raises(T.TransferError) as caught:
        T.restore_bundle(dst, folder)
    assert any(message in problem for problem in caught.value.problems), caught.value.problems
    assert _tree(dst) == before


# --------------------------------------------------------- output, helpers ----

def test_cli_export_prints_no_digest(tmp_path, monkeypatch, capsys):
    src = _source(tmp_path)
    monkeypatch.setattr(T, "PROJECT", src)
    assert T.main(["export", "--dest", str(tmp_path / "bundles")]) == 0
    folder = next((tmp_path / "bundles").iterdir())
    assert T.main(["check", str(folder)]) == 0
    output = capsys.readouterr()
    assert not re.search("[0-9a-f]{64}", output.out + output.err)
    assert "public-environment.json (supply it separately" in output.out
    _put(src, f"{SURROGATE}/extra.bin", b"extra")
    assert T.main(["export", "--dest", str(tmp_path / "bundles")]) == 1
    assert not re.search("[0-9a-f]{64}", capsys.readouterr().err)


SENTINEL = "SENTINEL-PRIVATE-CONTENT-7f3a"


def _sentinel_bundle(tmp_path):
    """A bundle whose private exb5.json holds a sentinel; returns (source, bundle, exb5 rel path)."""
    src = _source(tmp_path)
    digest = _put(src, f"{ACC}/inputs/exb5.json", SENTINEL.encode())
    _edit_private(lambda r: r.update({"exb5.json": {"sha256": {"exb5.json": digest}}}))(src)
    folder, _ = _export(src, tmp_path)
    return src, folder, f"{ACC}/inputs/exb5.json"


def _tamper_entry(folder, rel, change):
    """Edit one bundle.json entry in place (no digest or size recomputation)."""
    record = json.loads((folder / T.RECORD).read_text(encoding="utf-8"))
    index = next(i for i, item in enumerate(record["files"]) if item["path"] == rel)
    record["files"][index] = change(dict(record["files"][index]))
    (folder / T.RECORD).write_bytes(_json(record))


def _assert_nothing_private(text):
    assert not re.search("[0-9a-f]{64}", text), text
    assert SENTINEL not in text, text


TAMPERED_ENTRIES = {
    "invalid class": lambda item: dict(item, **{"class": "bogus"}),
    "digest as path": lambda item: dict(item, path=item["sha256"]),
    "sentinel as path": lambda item: dict(item, path=SENTINEL),
    "sentinel as size": lambda item: dict(item, size=SENTINEL),
    "sentinel as digest": lambda item: dict(item, sha256=SENTINEL),
    "missing field": lambda item: {key: value for key, value in item.items() if key != "class"},
    "not an object": lambda item: [item["path"], item["sha256"], SENTINEL],
}


@pytest.mark.parametrize("name", sorted(TAMPERED_ENTRIES))
def test_cli_check_refuses_a_malformed_entry_without_revealing_it(tmp_path, capsys, name):
    _, folder, rel = _sentinel_bundle(tmp_path)
    _tamper_entry(folder, rel, TAMPERED_ENTRIES[name])
    before = _tree(folder)
    assert T.main(["check", str(folder)]) == 1
    output = capsys.readouterr()
    assert "bundle.json entry" in output.err
    _assert_nothing_private(output.out + output.err)
    assert _tree(folder) == before


@pytest.mark.parametrize("name", sorted(TAMPERED_ENTRIES) + ["destination conflict"])
def test_cli_restore_refusals_write_nothing_and_reveal_nothing(tmp_path, monkeypatch, capsys, name):
    src, folder, rel = _sentinel_bundle(tmp_path)
    dst = _destination(tmp_path, src)
    if name == "destination conflict":
        _put(dst, rel, (SENTINEL + "-other").encode())                 # differing private file at the destination
    else:
        _tamper_entry(folder, rel, TAMPERED_ENTRIES[name])
    monkeypatch.setattr(T, "PROJECT", dst)
    before = _tree(dst)
    assert T.main(["restore", str(folder)]) == 1
    output = capsys.readouterr()
    assert "refused" in output.err
    _assert_nothing_private(output.out + output.err)
    assert _tree(dst) == before


def test_wheel_digest_is_shared_and_the_tool_never_runs_git():
    pinned = re.search(r"'([0-9a-f]{64})'", (PROJECT / "tools/setup_public_environment.py").read_text(encoding="utf-8")).group(1)
    acceptance = _load("acceptance_transfer_wheel_check", PROJECT / "tools/acceptance.py")
    assert acceptance.STOCK_WHEEL_SHA256 == pinned
    source = (PROJECT / "tools/acceptance_transfer.py").read_text(encoding="utf-8")
    assert "subprocess" not in source and '"git"' not in source


def test_private_input_problems_accepts_another_inputs_root(tmp_path):
    acceptance = T._tool("acceptance")
    _put(tmp_path, "inputs/exb5.json", b"data")
    record = {"exb5.json": {"sha256": {"exb5.json": _sha(b"data")}}}
    assert acceptance.private_input_problems("exb5.json", record, inputs=tmp_path / "inputs") == []
    assert acceptance.private_input_problems("exb5.json", {}, inputs=tmp_path / "inputs") == \
        ["input not hashed in private-inputs.json: exb5.json"]
