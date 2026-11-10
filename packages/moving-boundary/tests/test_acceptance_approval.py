"""M7b: the approval path on synthetic tmp_path projects (plan §5).

Reference selection, Git-free validation of approved references, crash-safe promotion with
interruption after every step, rollback, the contract assertion/comparison split, the approve
orchestration's gates, and the example comparison's selection. Nothing here touches the real
project: every test patches acceptance.py's PROJECT/OUTPUT/MANIFEST to a tmp_path tree.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
SHA = "0123456789abcdef0123456789abcdef01234567"
NOW = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
STEPS = ("written", "validated", "renamed", "reported", "manifest-temp-written")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pair(folder, case, value, **report):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{case}.npz", x=np.array([0.0, value]))
    (folder / f"{case}.json").write_text(json.dumps(dict(status="PASS", stop_status="completed", **report)), encoding="utf-8")
    return dict(json=folder / f"{case}.json", npz=folder / f"{case}.npz")


def _project(tmp_path, monkeypatch, cases=("C01", "O08"), newline="\n"):
    """acceptance.py on a synthetic project with reference pairs for ``cases`` and no approvals."""
    tool = _load("approval_tool", PROJECT / "tools/acceptance.py")
    tool.metric_tools()  # cached before PROJECT is patched
    approval = tool.approval_tools()
    output = tmp_path / ".verification/acceptance"
    (output / "inputs").mkdir(parents=True)
    expected = {}
    for case in cases:
        pair = _pair(output / "reference", case, 1.0)
        expected[case] = dict(json=_sha(pair["json"]), npz=_sha(pair["npz"]))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/definitions-provenance.json").write_text(json.dumps(dict(definitions={})), encoding="utf-8")
    (tmp_path / "tools").mkdir()
    manifest = dict(schema=1, quick=[c for c in cases if c.startswith("C")], contracts=[],
                    public_tolerances=dict(profiles=dict(rtol=1e-10, atol=1e-12)),
                    default_phase_sampling=dict(case="O08", reference_sha256={"O08.json": expected.get("O08", {}).get("json", ""),
                                                                               "O08.npz": expected.get("O08", {}).get("npz", "")}),
                    thermocalc_sampling=dict(repeat_unseeded=False),
                    expected_artifacts=dict(cases=expected), adopted_inputs=dict(files={}),
                    approval_criteria=dict(analytic={}, conservation={}))
    path = tmp_path / "tools/acceptance_manifest.json"
    path.write_bytes((json.dumps(manifest, indent=2) + "\n").replace("\n", newline).encode("utf-8"))
    monkeypatch.setattr(tool, "PROJECT", tmp_path)
    monkeypatch.setattr(tool, "OUTPUT", output)
    monkeypatch.setattr(tool, "MANIFEST", path)
    return tool, approval, output, path


def _promote(tool, approval, files, *, fault=None, reason="LEDGERS R99"):
    manifest = json.loads(tool.MANIFEST.read_text(encoding="utf-8"))
    approval_id = approval.new_id(manifest, project=tool.PROJECT, output=tool.OUTPUT, package_commit=SHA, now=NOW)
    record = dict(created_utc="2026-11-10T12:00:00Z", package_commit=SHA, reason=reason, tier="quick", cases=sorted(files))
    approval.promote(project=tool.PROJECT, output=tool.OUTPUT, manifest_path=tool.MANIFEST, approval_id=approval_id,
                     record=record, files=files, report=dict(schema=1, kind="approval"), fault=fault)
    return approval_id


def _snapshot(folder):
    return {path.relative_to(folder).as_posix(): (_sha(path), path.stat().st_mtime_ns)
            for path in sorted(Path(folder).rglob("*")) if path.is_file()}


def test_selection_without_approvals_is_unchanged(tmp_path, monkeypatch):
    tool, _, output, _ = _project(tmp_path, monkeypatch)
    selected = tool.expected_reference("C01")
    assert (selected["source"], selected["json"], selected["npz"], selected["sha256"], selected["settings"]) == \
           ("reference", output / "reference/C01.json", output / "reference/C01.npz", None, None)
    frozen = tool.expected_reference("O08")
    assert frozen["sha256"] == dict(json=_sha(output / "reference/O08.json"), npz=_sha(output / "reference/O08.npz"))
    tool.validate_expected(["C01", "O08"])


def test_promotion_routes_selection_and_keeps_the_manifest_layout(tmp_path, monkeypatch):
    tool, approval, output, path = _project(tmp_path, monkeypatch)
    before_text, before_reference = path.read_bytes(), _snapshot(output / "reference")
    approval_id = _promote(tool, approval, {"C01": _pair(tmp_path / "candidate", "C01", 2.0)})
    assert approval_id == "20261110T120000Z-0123456-1"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert list(manifest)[-1] == "approved_references"
    assert {k: v for k, v in manifest.items() if k != "approved_references"} == json.loads(before_text)
    assert path.read_bytes().startswith(before_text.rstrip()[:-1].rstrip()) and b"\r" not in path.read_bytes()
    section = manifest["approved_references"]
    assert section["in_effect"] == {"C01": approval_id}
    record = section["approvals"][approval_id]
    assert record["directory"] == "reference-approved/" + approval_id and record["previous"] == {"C01": "reference"}
    selected = tool.expected_reference("C01")
    assert selected["source"] == approval_id and selected["npz"] == output / "reference-approved" / approval_id / "C01.npz"
    with np.load(selected["npz"]) as data:
        assert data["x"][1] == 2.0
    report = tmp_path / "docs/approvals" / (approval_id + ".json")
    assert b"\r" not in report.read_bytes() and record["report"]["sha256"] == _sha(report)
    assert json.loads(report.read_text(encoding="utf-8"))["previous"] == {"C01": "reference"}
    tool.validate_expected(["C01", "O08"])
    assert _snapshot(output / "reference") == before_reference  # nothing writes under reference/
    assert approval.leftovers(manifest, project=tmp_path, output=output, manifest_path=path) == \
           dict(partial=[], unreferenced=[], unreferenced_reports=[], manifest_temps=[])


def test_crlf_manifest_keeps_its_newlines(tmp_path, monkeypatch):
    tool, approval, _, path = _project(tmp_path, monkeypatch, newline="\r\n")
    _promote(tool, approval, {"C01": _pair(tmp_path / "candidate", "C01", 2.0)})
    data = path.read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n") > 0
    tool.validate_expected(["C01"])


def test_compare_reads_the_reference_in_effect(tmp_path, monkeypatch):
    tool, approval, output, _ = _project(tmp_path, monkeypatch)
    candidate = _pair(tmp_path / "candidate", "C01", 2.0)
    _pair(output / "package", "C01", 2.0)
    report = json.loads((output / "package/C01.json").read_text(encoding="utf-8"))
    assert tool.compare("C01", report, tolerance="exact")["failed_keys"] == ["x"]
    approval_id = _promote(tool, approval, {"C01": candidate})
    assert "failed_keys" not in tool.compare("C01", report, tolerance="exact")
    record = json.loads((output / "package/C01.deviations.json").read_text(encoding="utf-8"))
    assert record["reference"] == approval_id and record["bit_identical"]


def test_compare_checks_approved_hashes(tmp_path, monkeypatch):
    tool, approval, output, _ = _project(tmp_path, monkeypatch)
    approval_id = _promote(tool, approval, {"C01": _pair(tmp_path / "candidate", "C01", 2.0)})
    _pair(output / "package", "C01", 2.0)
    np.savez(output / "reference-approved" / approval_id / "C01.npz", x=np.array([0.0, 3.0]))
    with pytest.raises(AssertionError, match="C01 approved reference changed: C01.npz"):
        tool.compare("C01", dict(status="PASS", stop_status="completed"))


def test_approval_ids_never_collide_with_leftovers(tmp_path, monkeypatch):
    tool, approval, output, path = _project(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    first = approval.new_id(manifest, project=tmp_path, output=output, package_commit=SHA, now=NOW)
    (output / "reference-approved" / (first + ".partial")).mkdir(parents=True)
    second = approval.new_id(manifest, project=tmp_path, output=output, package_commit=SHA, now=NOW)
    (tmp_path / "docs/approvals").mkdir(parents=True)
    (tmp_path / "docs/approvals" / (second + ".json")).write_text("{}")
    third = approval.new_id(manifest, project=tmp_path, output=output, package_commit=SHA, now=NOW)
    assert [first, second, third] == [f"20261110T120000Z-0123456-{n}" for n in (1, 2, 3)]
    assert approval.ID_PATTERN.fullmatch(third)


def _interrupt_at(step):
    def fault(name):
        if name == step:
            raise RuntimeError("injected failure at " + step)
    return fault


@pytest.mark.parametrize("step", STEPS + ("replace",))
def test_interruption_keeps_the_old_state_valid_and_a_retry_succeeds(tmp_path, monkeypatch, step):
    tool, approval, output, path = _project(tmp_path, monkeypatch)
    old = path.read_bytes()
    files = {"C01": _pair(tmp_path / "candidate", "C01", 2.0)}
    if step == "replace":
        real = approval._os_replace
        def locked(source, target):
            if Path(target).name == path.name:
                raise PermissionError("locked by sync")
            return real(source, target)
        monkeypatch.setattr(approval, "_os_replace", locked)
        monkeypatch.setattr(approval, "REPLACE_ATTEMPTS", 3)
        monkeypatch.setattr(approval, "REPLACE_DELAY", 0.0)
        with pytest.raises(approval.ApprovalError, match="could not replace acceptance_manifest.json after 3 attempts"):
            _promote(tool, approval, files)
        monkeypatch.setattr(approval, "_os_replace", real)
    else:
        with pytest.raises(RuntimeError, match="injected failure at " + step):
            _promote(tool, approval, files, fault=_interrupt_at(step))
    assert path.read_bytes() == old
    tool.validate_expected(["C01"])
    assert tool.expected_reference("C01")["source"] == "reference"
    first = "20261110T120000Z-0123456-1"
    left = approval.leftovers(json.loads(old), project=tmp_path, output=output, manifest_path=path)
    assert left["manifest_temps"] == []
    if step in ("written", "validated"):
        assert left["partial"] == [first + ".partial"] and left["unreferenced"] == []
    else:
        assert left["partial"] == [] and left["unreferenced"] == [first]
    assert left["unreferenced_reports"] == ([] if step in ("written", "validated", "renamed") else [first + ".json"])
    retry = _promote(tool, approval, files)
    assert retry == "20261110T120000Z-0123456-2"
    tool.validate_expected(["C01"])
    assert tool.expected_reference("C01")["source"] == retry


def test_failure_after_the_commit_point_leaves_the_new_state_valid(tmp_path, monkeypatch):
    tool, approval, _, _ = _project(tmp_path, monkeypatch)
    files = {"C01": _pair(tmp_path / "candidate", "C01", 2.0)}
    with pytest.raises(RuntimeError, match="injected failure at replaced"):
        _promote(tool, approval, files, fault=_interrupt_at("replaced"))
    tool.validate_expected(["C01"])
    assert tool.expected_reference("C01")["source"] == "20261110T120000Z-0123456-1"
    assert _promote(tool, approval, files) == "20261110T120000Z-0123456-2"
    tool.validate_expected(["C01"])


def test_a_transient_lock_is_retried(tmp_path, monkeypatch):
    tool, approval, _, path = _project(tmp_path, monkeypatch)
    real, calls = approval._os_replace, []
    def flaky(source, target):
        calls.append(Path(target).name)
        if Path(target).name == path.name and calls.count(path.name) <= 2:
            raise PermissionError("locked by sync")
        return real(source, target)
    monkeypatch.setattr(approval, "_os_replace", flaky)
    monkeypatch.setattr(approval, "REPLACE_DELAY", 0.0)
    approval_id = _promote(tool, approval, {"C01": _pair(tmp_path / "candidate", "C01", 2.0)})
    assert calls.count(path.name) == 3 and tool.expected_reference("C01")["source"] == approval_id


def test_rollback_points_back_without_moving_files(tmp_path, monkeypatch):
    tool, approval, output, path = _project(tmp_path, monkeypatch)
    first = _promote(tool, approval, {"C01": _pair(tmp_path / "a", "C01", 2.0)})
    second = _promote(tool, approval, {"C01": _pair(tmp_path / "b", "C01", 3.0)})
    assert first != second
    files = _snapshot(output / "reference-approved")
    with pytest.raises(approval.ApprovalError, match="not in effect"):
        approval.rollback(project=tmp_path, output=output, manifest_path=path, approval_id=first, reason="LEDGERS R99", utc="2026-11-10T13:00:00Z")
    with pytest.raises(approval.ApprovalError, match="needs a reason"):
        approval.rollback(project=tmp_path, output=output, manifest_path=path, approval_id=second, reason=" ", utc="2026-11-10T13:00:00Z")
    approval.rollback(project=tmp_path, output=output, manifest_path=path, approval_id=second, reason="LEDGERS R99", utc="2026-11-10T13:00:00Z")
    tool.validate_expected(["C01"])
    assert tool.expected_reference("C01")["source"] == first
    approval.rollback(project=tmp_path, output=output, manifest_path=path, approval_id=first, reason="LEDGERS R99", utc="2026-11-10T13:01:00Z")
    tool.validate_expected(["C01"])
    assert tool.expected_reference("C01")["source"] == "reference"
    section = json.loads(path.read_text(encoding="utf-8"))["approved_references"]
    assert [entry["event"] for entry in section["history"]] == ["approve", "approve", "rollback", "rollback"]
    assert set(section["approvals"]) == {first, second} and section["in_effect"] == {}
    assert _snapshot(output / "reference-approved") == files


def test_rollback_interrupted_during_the_replace_keeps_the_manifest(tmp_path, monkeypatch):
    tool, approval, output, path = _project(tmp_path, monkeypatch)
    approval_id = _promote(tool, approval, {"C01": _pair(tmp_path / "a", "C01", 2.0)})
    old = path.read_bytes()
    with pytest.raises(RuntimeError, match="injected"):
        approval.rollback(project=tmp_path, output=output, manifest_path=path, approval_id=approval_id, reason="LEDGERS R99", utc="2026-11-10T13:00:00Z",
                          fault=_interrupt_at("manifest-temp-written"))
    assert path.read_bytes() == old and not list(path.parent.glob(".*.tmp"))
    tool.validate_expected(["C01"])
    assert tool.expected_reference("C01")["source"] == approval_id


def test_rejected_rollback_leaves_the_manifest_byte_for_byte_unchanged(tmp_path, monkeypatch):
    """Review finding: rolling back to an approval whose files are gone must fail before the replace."""
    tool, approval, output, path = _project(tmp_path, monkeypatch)
    first = _promote(tool, approval, {"C01": _pair(tmp_path / "a", "C01", 2.0)})
    second = _promote(tool, approval, {"C01": _pair(tmp_path / "b", "C01", 3.0)})
    (output / "reference-approved" / first / "C01.npz").unlink()
    old = path.read_bytes()
    with pytest.raises(approval.ApprovalError, match=f"rollback refused, the manifest is unchanged.*approved artifact missing: reference-approved/{first}/C01.npz"):
        approval.rollback(project=tmp_path, output=output, manifest_path=path, approval_id=second,
                          reason="LEDGERS R99", utc="2026-11-10T13:00:00Z")
    assert path.read_bytes() == old and not list(path.parent.glob(".*.tmp"))
    monkeypatch.setattr(tool, "repository_checks", lambda: SHA)
    with pytest.raises(approval.ApprovalError, match="rollback refused"):
        tool.rollback_approval(second, reason="LEDGERS R99")
    assert path.read_bytes() == old
    tool.validate_expected(["C01"])  # the selection in effect (the newer approval) is still intact
    assert tool.expected_reference("C01")["source"] == second


def _tamper(tmp_path, output, section, approval_id, how):
    record = section["approvals"][approval_id]
    if how == "artifact":
        np.savez(output / record["directory"] / "C01.npz", x=np.array([9.0]))
    elif how == "missing artifact":
        (output / record["directory"] / "C01.json").unlink()
    elif how == "report":
        (tmp_path / record["report"]["path"]).write_text("{}\n")
    elif how == "missing report":
        (tmp_path / record["report"]["path"]).unlink()
    elif how == "unknown in effect":
        section["in_effect"]["C01"] = "20261110T120000Z-0123456-9"
    elif how == "directory":
        record["directory"] = "reference"
    elif how == "commit":
        record["package_commit"] = SHA[:7]
    elif how == "malformed id":
        section["approvals"]["latest"] = section["approvals"].pop(approval_id)
        section["in_effect"]["C01"] = "latest"
    elif how == "unknown case":
        record["cases"].append("C99")


@pytest.mark.parametrize("how, message", [
    ("artifact", "C01: approved artifact changed"), ("missing artifact", "C01: approved artifact missing"),
    ("report", "approval report changed"), ("missing report", "approval report missing"),
    ("unknown in effect", "in effect from unknown approval"), ("directory", "directory must be reference-approved/"),
    ("commit", "package_commit must be a full commit sha"), ("malformed id", "malformed approval ID"),
    ("unknown case", "no expected_artifacts entry for C99")])
def test_validation_fails_on_any_approval_problem(tmp_path, monkeypatch, how, message):
    tool, approval, output, path = _project(tmp_path, monkeypatch)
    approval_id = _promote(tool, approval, {"C01": _pair(tmp_path / "a", "C01", 2.0)})
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _tamper(tmp_path, output, manifest["approved_references"], approval_id, how)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match=message):
        tool.validate_expected(["C01"])


def test_contract_assertions_gate_and_the_comparison_is_reported_only_for_listed_candidates(tmp_path, monkeypatch):
    tool, _, output, _ = _project(tmp_path, monkeypatch, cases=("C25",))
    groups = ["test_a", "test_b"]
    reference = json.loads((output / "reference/C25.json").read_text(encoding="utf-8"))
    (output / "reference/C25.json").write_text(json.dumps(dict(reference, contracts=groups)), encoding="utf-8")
    monkeypatch.setattr(tool, "validate_expected", lambda cases: None)
    def worker(case, role, **kwargs):
        (output / "package").mkdir(exist_ok=True)
        np.savez(output / "package/C25.npz", **{"contracts/assertion_groups": np.array(worker.count)})
        return dict(status="PASS", stop_status="contract_assertions_passed", contracts=worker.groups)
    worker.count, worker.groups = 2, groups
    monkeypatch.setattr(tool, "worker", worker)
    monkeypatch.setattr(tool, "compare", lambda case, report, **kwargs: dict(failed_keys=["model0/x"]))
    listed = dict(target="t", host="h", approval_candidates=["C25"])
    assert tool.contract_case("C25", listed)["comparison"] == "reported: approval candidate"
    with pytest.raises(AssertionError, match="failed_keys"):
        tool.contract_case("C25", dict(target="t", host="h", approval_candidates=[]))
    worker.groups = ["test_a"]
    with pytest.raises(AssertionError, match="assertion groups differ"):
        tool.contract_case("C25", listed)
    worker.groups, worker.count = groups, 1
    with pytest.raises(AssertionError, match="assertion-group count differs"):
        tool.contract_case("C25", listed)


def _approve_project(tmp_path, monkeypatch, *, values, failing_criteria=()):
    """Mocked workers produce ``values[case]``; everything else (compare, promotion, validation) is real."""
    tool, approval, output, path = _project(tmp_path, monkeypatch, cases=("C01", "C02"))
    monkeypatch.setattr(tool, "repository_checks", lambda: SHA)
    monkeypatch.setattr(tool, "normal_cases", lambda manifest, full, explicit=None: (["C01", "C02"], {}))
    monkeypatch.setattr(tool, "setup_targets", lambda: dict(target="t", host="h"))
    monkeypatch.setattr(tool, "approval_repeat", lambda case, **kwargs: dict(bit_identical=True, arrays=1))
    monkeypatch.setattr(tool, "run_pytest", lambda state, **kwargs: dict(tests=1, failures=0, errors=0, skipped=0))
    monkeypatch.setattr(tool, "capture_example_settings", lambda cases, state, directory: {})
    monkeypatch.setattr(tool, "criteria_results", lambda case, manifest: [dict(
        case=case, criterion="conservation/inventory", value=1e-15, limit=1e-13, passed=case not in failing_criteria)])
    def worker(case, role, **kwargs):
        _pair(output / "package", case, values[case])
        return json.loads((output / "package" / f"{case}.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(tool, "worker", worker)
    return tool, output, path


def test_approve_promotes_listed_changes_and_gates_unlisted_cases(tmp_path, monkeypatch):
    tool, output, path = _approve_project(tmp_path, monkeypatch, values=dict(C01=1.0, C02=2.0))
    assert tool.approve(["C02"], reason="LEDGERS R99") == 0
    section = json.loads(path.read_text(encoding="utf-8"))["approved_references"]
    (approval_id,) = section["approvals"]
    assert section["in_effect"] == {"C02": approval_id}
    report = json.loads((tmp_path / "docs/approvals" / (approval_id + ".json")).read_text(encoding="utf-8"))
    assert report["gates"]["unlisted_cases_passed_public_tolerance"] == ["C01"] and report["listed_cases"] == {"C02": "differs"}
    assert report["deviations_vs_previous"]["C02"]["arrays"]["x"]["bit_identical"] is False
    assert report["reason"] == "LEDGERS R99" and report["package_commit"] == SHA
    assert json.loads((output / "state.json").read_text(encoding="utf-8"))["approval_candidates"] == []
    tool.validate_expected(["C01", "C02"])


@pytest.mark.parametrize("values, failing, message", [
    (dict(C01=2.0, C02=2.0), (), "unlisted case fails its public-tolerance comparison"),
    (dict(C01=1.0, C02=2.0), ("C01",), "approval_criteria failed")])
def test_approve_writes_nothing_when_a_gate_fails(tmp_path, monkeypatch, values, failing, message):
    tool, output, path = _approve_project(tmp_path, monkeypatch, values=values, failing_criteria=failing)
    before = path.read_bytes()
    assert tool.approve(["C02"], reason="LEDGERS R99") == 1
    assert path.read_bytes() == before and not (output / "reference-approved").exists() and not (tmp_path / "docs/approvals").exists()
    assert message in json.loads((output / "approve-summary.json").read_text(encoding="utf-8"))["failure"]


def test_approve_always_builds_fresh_targets(tmp_path, monkeypatch):
    """Review finding: a saved target in state.json (possibly from another commit) is never used."""
    tool, output, _ = _approve_project(tmp_path, monkeypatch, values=dict(C01=1.0, C02=2.0))
    import inspect
    assert "reuse_targets" not in inspect.signature(tool.approve).parameters  # no path to a saved target
    (output / "state.json").write_text(json.dumps(dict(target="stale-target", host="h")), encoding="utf-8")
    builds, targets = [], set()
    monkeypatch.setattr(tool, "setup_targets", lambda: builds.append(1) or dict(target="fresh-target", host="h"))
    worker = tool.worker
    monkeypatch.setattr(tool, "worker", lambda case, role, **kwargs: targets.add(kwargs["state"]["target"]) or worker(case, role, **kwargs))
    assert tool.approve(["C02"], reason="LEDGERS R99") == 0
    assert builds == [1] and targets == {"fresh-target"}


def test_approve_refuses_private_settings_cases_and_unrun_cases(tmp_path, monkeypatch):
    tool, _, _ = _approve_project(tmp_path, monkeypatch, values=dict(C01=1.0, C02=1.0))
    with pytest.raises(AssertionError, match="O05 is not supported"):
        tool.approve(["O05"], reason="LEDGERS R99")
    with pytest.raises(AssertionError, match="not run by approve --tier quick on this machine: F04"):
        tool.approve(["F04"], reason="LEDGERS R99")


def test_repository_checks_need_a_clean_tree_and_ancestry(tmp_path, monkeypatch):
    tool, approval, _, _ = _project(tmp_path, monkeypatch)
    _promote(tool, approval, {"C01": _pair(tmp_path / "a", "C01", 2.0)})
    answers = {"status": " M tools/acceptance.py\n", "rev-parse": SHA + "\n"}
    def run(args, **kwargs):
        if args[1] == "merge-base":
            raise RuntimeError("not an ancestor")
        return answers[args[1]]
    monkeypatch.setattr(tool, "run", run)
    with pytest.raises(AssertionError, match="clean package tree"):
        tool.repository_checks()
    answers["status"] = ""
    with pytest.raises(AssertionError, match="is not an ancestor of HEAD"):
        tool.repository_checks()


@pytest.mark.parametrize("argv", [
    ["approve", "--cases", "C01"], ["approve", "--reason", "LEDGERS R99"],
    ["approve", "--cases", "C01", "--rollback", "20261110T120000Z-0123456-1", "--reason", "r"],
    ["approve", "--cases", "C01", "--reason", "r", "--tolerance", "exact"],
    ["approve", "--cases", "C01", "--reason", "r", "--research", "x"], ["approve", "repeat", "--cases", "C01", "--reason", "r"],
    ["approve", "--cases", "C01", "--reason", "r", "--reuse-targets"],
    ["quick", "--reason", "r"], ["full", "--rollback", "20261110T120000Z-0123456-1"]])
def test_cli_rejects_malformed_approval_requests(tmp_path, monkeypatch, argv):
    tool, _, output, _ = _project(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as error:
        tool.main(argv)
    assert error.value.code == 2


def test_examples_compare_with_the_approved_arrays_and_settings(tmp_path, monkeypatch):
    tool, approval, output, _ = _project(tmp_path, monkeypatch, cases=("C04",))
    examples = _load("approval_examples", PROJECT / "tools/acceptance_examples.py")
    candidate = _pair(tmp_path / "candidate", "C04", 2.0)
    candidate["settings"] = tmp_path / "candidate/C04.settings.json"
    candidate["settings"].write_text('{"a": 1}', encoding="utf-8")
    _pair(output / "package_examples", "E01", 2.0, elapsed_seconds=1.0)
    (output / "package_examples/E01.settings.json").write_text('{"a": 1}', encoding="utf-8")
    seen = []
    class Settings:
        @staticmethod
        def differences(recorded, got, allowed):
            seen.append(recorded)
            return [] if recorded == got else ["a"]
    entry = dict(reference="C04")
    (output / "example_settings").mkdir()
    (output / "example_settings/C04.settings.json").write_text('{"a": 0}', encoding="utf-8")
    digests = {"C04.settings.json": _sha(output / "example_settings/C04.settings.json")}
    failures, _ = examples.compare_case(tool, Settings, "E01", entry, {}, digests)
    assert failures == ["setting differs: a", "array differs: x"]
    _promote(tool, approval, {"C04": candidate})
    failures, _ = examples.compare_case(tool, Settings, "E01", entry, {}, digests)
    assert failures == [] and seen[-1] == {"a": 1}
    (output / "reference-approved").joinpath(tool.expected_reference("C04")["source"], "C04.settings.json").write_text('{"a": 2}')
    assert examples.compare_case(tool, Settings, "E01", entry, {}, digests)[0] == ["adopted settings changed: C04.settings.json"]


def test_approval_reports_are_pinned_to_lf():
    text = (PROJECT / "docs/approvals/.gitattributes").read_text(encoding="utf-8")
    assert "*.json text eol=lf" in text.splitlines()
