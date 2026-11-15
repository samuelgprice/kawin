"""R09c-1: the strict ``r09-data-tail`` layout proof and its approval and example tooling.

The layout rule is an additional structural proof against each case's previously selected
reference, never a relaxation: only an exact removal of one trailing, bitwise-duplicate row from
a declared data time/values pair passes, and an unlisted case must stay bit-identical. Everything
here runs on synthetic tmp_path projects (acceptance.py's PROJECT/OUTPUT/MANIFEST patched), with
the workers, repeat runs and package tests mocked, as in ``test_acceptance_approval.py``. Nothing
touches the real project, its references or its approvals.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
SHA = "0123456789abcdef0123456789abcdef01234567"
NOW = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)
RULE = "r09-data-tail"
STEPS = ("written", "validated", "renamed", "reported", "manifest-temp-written")
BOOKKEEPING_ADDED_LATER = ("_pendingCandidate", "_awaitingFreshCandidate", "_solveMinDtFrac", "_solveMaxDtFrac",
                           "_outerMinTimeStep", "_outerMaxTimeStep")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


layout = _load("layout_rules_under_test", PROJECT / "tools/acceptance_layout.py")


# Arrays of a binary-like run case whose data history ends with the legacy duplicate row.


def _reference_arrays(duplicate_tail=True):
    values = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.5, 0.6] if duplicate_tail else [0.5, 0.7]]
    return {
        "data/time": np.array([0.0, 0.5, 1.0, 1.0]),
        "data/values": np.array(values),
        "interfaceData/time": np.array([0.0, 0.5, 1.0]),
        "interfaceData/values": np.array([1.0, 1.1, 1.2]),
        "profiles/compositions": np.array([[0.2, 0.8], [0.3, 0.7]]),
    }


def _variant(name, reference=None):
    """Candidate arrays: an exact tail removal, or one of the changes the rule must refuse."""
    arrays = {key: value.copy() for key, value in (reference or _reference_arrays()).items()}
    if name == "identical":
        return arrays
    if name == "shift":   # a numerical change outside public tolerance
        arrays["profiles/compositions"] = arrays["profiles/compositions"] + 1e-6
        return arrays
    if name == "tiny":   # within public tolerance, but not bit-identical
        arrays["profiles/compositions"] = arrays["profiles/compositions"] + 1e-16
        return arrays
    if name in ("tail", "tail+tiny", "tail+earlier"):
        arrays["data/time"], arrays["data/values"] = arrays["data/time"][:-1], arrays["data/values"][:-1]
        if name == "tail+tiny":
            arrays["profiles/compositions"] = arrays["profiles/compositions"] + 1e-16
        if name == "tail+earlier":
            arrays["data/values"][1, 0] += 1e-16
        return arrays
    if name == "interface-tail":
        arrays["interfaceData/time"], arrays["interfaceData/values"] = arrays["interfaceData/time"][:-1], arrays["interfaceData/values"][:-1]
        return arrays
    if name == "two-rows":
        arrays["data/time"], arrays["data/values"] = arrays["data/time"][:-2], arrays["data/values"][:-2]
        return arrays
    if name == "only-time":
        arrays["data/time"] = arrays["data/time"][:-1]
        return arrays
    if name == "dtype":
        arrays["data/time"], arrays["data/values"] = arrays["data/time"][:-1].astype(np.float32), arrays["data/values"][:-1]
        return arrays
    if name == "extra-key":
        arrays["data/time"], arrays["data/values"] = arrays["data/time"][:-1], arrays["data/values"][:-1]
        arrays["extra"] = np.zeros(1)
        return arrays
    raise KeyError(name)


# The rule itself


def test_identical_and_exact_tail_removal_are_recognized():
    reference = _reference_arrays()
    assert layout.classify("C01", reference, _variant("identical"))["verdict"] == "identical"
    result = layout.classify("C01", reference, _variant("tail"))
    assert result["verdict"] == "tail-removal" and result["problems"] == []
    assert result["corrected"] == [["data/time", "data/values"]]
    assert layout.structural(result) == dict(rule=RULE, verdict="tail-removal",
                                             corrected=[["data/time", "data/values"]], problems=[])


@pytest.mark.parametrize("name, message", [
    ("tiny", "array changed outside the declared pairs: profiles/compositions"),
    ("tail+tiny", "array changed outside the declared pairs: profiles/compositions"),
    ("tail+earlier", "kept rows differ from the reference"),
    ("interface-tail", "array changed outside the declared pairs: interfaceData/time"),
    ("two-rows", "not exactly one row removed"),
    ("only-time", "only one array of the pair changed"),
    ("dtype", "dtype changed"),
    ("extra-key", "key set differs"),
])
def test_every_other_change_is_refused(name, message):
    result = layout.classify("C01", _reference_arrays(), _variant(name))
    assert result["verdict"] == "other"
    assert any(message in problem for problem in result["problems"]), result["problems"]


def test_a_removed_row_must_duplicate_the_row_before_it():
    reference = _reference_arrays(duplicate_tail=False)
    result = layout.classify("C01", reference, _variant("tail", reference))
    assert result["verdict"] == "other"
    assert any("not a bitwise duplicate" in problem for problem in result["problems"])


def test_a_duplicate_time_with_different_values_is_not_a_duplicate_row():
    reference = _reference_arrays()
    reference["data/values"][-1, 1] = np.nextafter(reference["data/values"][-1, 1], 1.0)
    assert layout.classify("C01", reference, _variant("tail", reference))["verdict"] == "other"


def test_contract_and_legacy_pair_forms():
    t, y = np.array([0.0, 1.0, 1.0]), np.array([[1.0], [2.0], [2.0]])
    contract = {"model0/data/time": t, "model0/data/values": y, "model1/data/time": t[:1], "model1/data/values": y[:1]}
    candidate = dict(contract, **{"model0/data/time": t[:-1], "model0/data/values": y[:-1]})
    result = layout.classify("C25", contract, candidate)
    assert result["verdict"] == "tail-removal" and result["corrected"] == [["model0/data/time", "model0/data/values"]]
    legacy = {"binary_phase_uniform__data_time": t, "binary_phase_uniform__data_y": y}
    shortened = {"binary_phase_uniform__data_time": t[:-1], "binary_phase_uniform__data_y": y[:-1]}
    assert layout.classify("LEGACY", legacy, shortened)["verdict"] == "tail-removal"
    assert layout.classify("C01", legacy, shortened)["verdict"] == "other"   # LEGACY form only for LEGACY


def test_only_the_o06_metadata_normalization_applies():
    tool = _load("layout_o06_tool", PROJECT / "tools/acceptance.py")
    def metadata(cache, extra):
        return np.array(json.dumps(dict(thermocalc_config=dict(cache_dir=cache, database="X"), extra=extra)))
    reference = dict(_reference_arrays(), **{"provider/metadata_json": metadata("C:/a", 1)})
    same = dict(_variant("tail"), **{"provider/metadata_json": metadata("D:/b/c", 1)})
    other = dict(_variant("tail"), **{"provider/metadata_json": metadata("D:/b/c", 2)})
    assert layout.classify("O06", reference, same, normalize=tool.comparison_array)["verdict"] == "tail-removal"
    assert layout.classify("O06", reference, other, normalize=tool.comparison_array)["verdict"] == "other"
    assert layout.classify("C01", reference, same, normalize=tool.comparison_array)["verdict"] == "other"


def test_unknown_rule_is_refused():
    with pytest.raises(AssertionError, match="unknown layout correction"):
        layout.classify("C01", _reference_arrays(), _variant("tail"), rule="tolerance")


# Synthetic project for approve and the layout run


def _settings_record(**changes):
    record = {"class": "Model", "duration": "1.0", "solve": {"minDtFrac": 1e-14},
              "attributes": {"timeStep": {"value": "0.1"}, "tolerance": {"value": "1e-12"},
                             **{name: {"value": None} for name in BOOKKEEPING_ADDED_LATER}}}
    for key, value in changes.items():
        if key == "drop_bookkeeping":
            for name in BOOKKEEPING_ADDED_LATER:
                record["attributes"].pop(name)
        elif key == "attributes":
            record["attributes"].update(value)
        else:
            record[key] = value
    return record


def _write_pair(folder, case, arrays, stop_status="completed"):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{case}.npz", **arrays)
    (folder / f"{case}.json").write_text(json.dumps(dict(status="PASS", stop_status=stop_status, elapsed_seconds=1.0)), encoding="utf-8")
    return dict(json=folder / f"{case}.json", npz=folder / f"{case}.npz")


def _project(tmp_path, monkeypatch, *, references=None, private_digest=True, with_o04=False):
    """acceptance.py on a tmp project: C01, C02, O05 with duplicate-tail references; C01->E01, O05->E05.

    ``with_o04`` adds the other private-settings case, O04 (mapped by E04, privately hashed).
    """
    tool = _load("layout_tool", PROJECT / "tools/acceptance.py")
    tool.metric_tools()
    approval = tool.approval_tools()
    output = tmp_path / ".verification/acceptance"
    (output / "inputs").mkdir(parents=True)
    expected = {}
    for case in ("C01", "C02", "O05") + (("O04",) if with_o04 else ()):
        pair = _write_pair(output / "reference", case, (references or {}).get(case, _reference_arrays()))
        expected[case] = dict(json=_sha(pair["json"]), npz=_sha(pair["npz"]))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/definitions-provenance.json").write_text(json.dumps(dict(definitions={})), encoding="utf-8")
    (tmp_path / "tools").mkdir()
    tolerance = dict(rtol=1e-10, atol=1e-12)
    manifest = dict(schema=1, quick=["C01", "C02"], contracts=[],
                    public_tolerances={group: tolerance for group in ("profiles", "accepted_times", "interface", "inventory", "eta", "analytic_error")},
                    default_phase_sampling=None, thermocalc_sampling=dict(repeat_unseeded=False),
                    expected_artifacts=dict(cases=expected), adopted_inputs=dict(files={}),
                    approval_criteria=dict(analytic={}, conservation={}))
    path = tmp_path / "tools/acceptance_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "example_settings").mkdir()
    adopted = {"C01": _settings_record(drop_bookkeeping=True), "O05": _settings_record(drop_bookkeeping=True)}
    if with_o04:
        adopted["O04"] = _settings_record(drop_bookkeeping=True, duration="2.0")
    for case, record in adopted.items():
        (output / "example_settings" / f"{case}.settings.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    examples = dict(cases={"E01": dict(example="e", config="FAST", reference="C01", tier="quick"),
                           "E05": dict(example="e", config="FAST", reference="O05", tier="quick")},
                    allowed_setting_differences={"E05": {"timeStep": "example-versus-case exception"}},
                    settings_sha256={"C01.settings.json": _sha(output / "example_settings/C01.settings.json")},
                    pending_fixture={})
    if with_o04:
        examples["cases"]["E04"] = dict(example="e", config="FAST", reference="O04", tier="quick")
    (tmp_path / "tools/example_manifest.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    private = {}
    if private_digest:
        private["example_settings"] = dict(sha256={f"{case}.settings.json": _sha(output / "example_settings" / f"{case}.settings.json")
                                                   for case in adopted if case != "C01"})
    (output / "inputs/private-inputs.json").write_text(json.dumps(private), encoding="utf-8")
    monkeypatch.setattr(tool, "PROJECT", tmp_path)
    monkeypatch.setattr(tool, "OUTPUT", output)
    monkeypatch.setattr(tool, "MANIFEST", path)
    return tool, approval, output, path


def _approve_project(tmp_path, monkeypatch, *, variants, captured=None, failing_criteria=(), references=None, **kwargs):
    """Mocked workers write ``variants[case]``; settings captures return ``captured[case]``."""
    tool, approval, output, path = _project(tmp_path, monkeypatch, references=references, **kwargs)
    monkeypatch.setattr(tool, "repository_checks", lambda: SHA)
    run_cases = ["C01", "C02", "O05"] + (["O04"] if kwargs.get("with_o04") else [])
    monkeypatch.setattr(tool, "normal_cases", lambda manifest, full, explicit=None: (list(run_cases), {}))
    monkeypatch.setattr(tool, "check_private_inputs", lambda cases: None)
    monkeypatch.setattr(tool, "setup_targets", lambda: dict(target="t", host="h"))
    monkeypatch.setattr(tool, "approval_repeat", lambda case, **kw: dict(bit_identical=True, arrays=5))
    monkeypatch.setattr(tool, "run_pytest", lambda state, **kw: dict(tests=1, failures=0, errors=0, skipped=0))
    monkeypatch.setattr(tool, "criteria_results", lambda case, manifest: [dict(
        case=case, criterion="conservation/inventory", value=1e-15, limit=1e-13, passed=case not in failing_criteria)])
    captures = {**dict(C01=_settings_record(), O05=_settings_record(), O04=_settings_record(duration="2.0")), **(captured or {})}
    calls = []

    def capture(cases, state, directory):
        calls.append(sorted(cases))
        out = {}
        for case in sorted(set(cases) & ({"C01", "O05"} | ({"O04"} if kwargs.get("with_o04") else set()))):
            out[case] = Path(directory) / f"{case}.settings.json"
            out[case].write_text(json.dumps(captures[case], indent=1, sort_keys=True), encoding="utf-8")
        return out
    monkeypatch.setattr(tool, "capture_example_settings", capture)

    def worker(case, role, **kw):
        references_ = references or {}
        return json.loads(_write_pair(output / "package", case, _variant(variants[case], references_.get(case)))["json"].read_text(encoding="utf-8"))
    monkeypatch.setattr(tool, "worker", worker)
    return tool, approval, output, path, calls


def _nothing_promoted(tmp_path, output, path, before):
    assert path.read_bytes() == before
    assert not (output / "reference-approved").exists() and not (tmp_path / "docs/approvals").exists()


def test_layout_approval_promotes_exact_tail_removals_with_structural_evidence(tmp_path, monkeypatch):
    tool, approval, output, path, calls = _approve_project(
        tmp_path, monkeypatch, variants=dict(C01="tail", C02="identical", O05="tail"))
    private_digest = json.loads((output / "inputs/private-inputs.json").read_text())["example_settings"]["sha256"]["O05.settings.json"]
    adopted_o05 = (output / "example_settings/O05.settings.json").read_bytes()

    assert tool.approve(["C01", "O05"], reason="LEDGERS R09c", layout_correction=RULE) == 0

    manifest_text = path.read_text(encoding="utf-8")
    section = json.loads(manifest_text)["approved_references"]
    (approval_id,) = section["approvals"]
    record = section["approvals"][approval_id]
    assert record["layout_correction"] == RULE and section["in_effect"] == {"C01": approval_id, "O05": approval_id}
    assert record["artifacts"]["O05"]["settings"] == approval.CARRIED_FORWARD
    assert len(record["artifacts"]["C01"]["settings"]) == 64
    folder = output / record["directory"]
    assert sorted(p.name for p in folder.iterdir()) == ["C01.json", "C01.npz", "C01.settings.json", "O05.json", "O05.npz"]
    report_text = (tmp_path / record["report"]["path"]).read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert "deviations_vs_previous" not in report
    assert report["layout_verdicts"] == {case: dict(rule=RULE, verdict="tail-removal", corrected=[["data/time", "data/values"]],
                                                    problems=[]) for case in ("C01", "O05")}
    assert report["settings_captured"] == ["C01"] and report["settings_carried_forward"] == ["O05"]
    assert report["gates"]["unlisted_cases_bit_identical"] == ["C02"]
    assert private_digest not in manifest_text and private_digest not in report_text
    assert (output / "example_settings/O05.settings.json").read_bytes() == adopted_o05
    assert [p for p in output.rglob("O05.settings.json") if p.parent.name != "example_settings"] == []  # capture removed
    assert calls == [["C01", "O05"]]
    selected = tool.expected_reference("O05")
    assert selected["settings_carried_forward"] and selected["settings"] is None
    tool.validate_expected(["C01", "C02", "O05"])


@pytest.mark.parametrize("variants, listed, message", [
    (dict(C01="identical", C02="identical", O05="tail"), ["C01", "O05"], "layout verdict identical, required tail-removal"),
    (dict(C01="tail", C02="tiny", O05="identical"), ["C01"], "layout verdict other, required identical"),
    (dict(C01="tail+tiny", C02="identical", O05="identical"), ["C01"], "layout verdict other, required tail-removal"),
    (dict(C01="interface-tail", C02="identical", O05="identical"), ["C01"], "layout verdict other, required tail-removal"),
    (dict(C01="tail", C02="tail", O05="identical"), ["C01"], "unlisted case fails its public-tolerance comparison"),
])
def test_layout_approval_refuses_anything_but_the_exact_differing_set(tmp_path, monkeypatch, variants, listed, message):
    tool, _, output, path, _ = _approve_project(tmp_path, monkeypatch, variants=variants)
    before = path.read_bytes()
    assert tool.approve(listed, reason="LEDGERS R09c", layout_correction=RULE) == 1
    _nothing_promoted(tmp_path, output, path, before)
    assert message in json.loads((output / "approve-summary.json").read_text(encoding="utf-8"))["failure"]


def test_layout_approval_refuses_a_removed_row_that_was_not_a_duplicate(tmp_path, monkeypatch):
    references = dict(C01=_reference_arrays(duplicate_tail=False))
    tool, _, output, path, _ = _approve_project(tmp_path, monkeypatch, variants=dict(C01="tail", C02="identical", O05="identical"),
                                                references=references)
    before = path.read_bytes()
    assert tool.approve(["C01"], reason="LEDGERS R09c", layout_correction=RULE) == 1
    _nothing_promoted(tmp_path, output, path, before)
    assert "not a bitwise duplicate" in json.loads((output / "approve-summary.json").read_text(encoding="utf-8"))["failure"]


@pytest.mark.parametrize("gate", ["criteria", "repeat", "pytest"])
def test_every_existing_gate_still_blocks_a_valid_layout_correction(tmp_path, monkeypatch, gate):
    tool, _, output, path, _ = _approve_project(tmp_path, monkeypatch, variants=dict(C01="tail", C02="identical", O05="identical"),
                                                failing_criteria=("C01",) if gate == "criteria" else ())
    if gate == "repeat":
        def repeat(case, **kw):
            raise AssertionError(("Unseeded package role is not repeatable", case, ["data/values"]))
        monkeypatch.setattr(tool, "approval_repeat", repeat)
    if gate == "pytest":   # contract assertion groups and the whole tests/ suite run here
        def failing(state, **kw):
            raise AssertionError({"tests": 9, "failures": 1, "errors": 0, "skipped": 0})
        monkeypatch.setattr(tool, "run_pytest", failing)
    before = path.read_bytes()
    assert tool.approve(["C01"], reason="LEDGERS R09c", layout_correction=RULE) == 1
    _nothing_promoted(tmp_path, output, path, before)


@pytest.mark.parametrize("case, captured, message", [
    ("C01", _settings_record(attributes={"timeStep": {"value": "0.2"}}), "C01: setting differs: attributes.timeStep"),
    ("O05", _settings_record(attributes={"tolerance": {"value": "1e-10"}}), "O05: setting differs: attributes.tolerance"),
    ("C01", _settings_record(attributes={"residualTolerance": {"value": "1e-13"}}), "C01: setting differs: attributes.residualTolerance"),
    ("O05", _settings_record(solve={"minDtFrac": 1e-12}), "O05: setting differs: solve"),
    ("C01", _settings_record(attributes={"gridNodes": {"value": "101.0"}}), "C01: setting differs: attributes.gridNodes"),
])
def test_layout_approval_requires_case_versus_case_settings_equality(tmp_path, monkeypatch, case, captured, message):
    tool, _, output, path, _ = _approve_project(tmp_path, monkeypatch, variants=dict(C01="tail", C02="identical", O05="tail"),
                                                captured={case: captured})
    before = path.read_bytes()
    assert tool.approve(["C01", "O05"], reason="LEDGERS R09c", layout_correction=RULE) == 1
    _nothing_promoted(tmp_path, output, path, before)
    assert message in json.loads((output / "approve-summary.json").read_text(encoding="utf-8"))["failure"]
    assert [p for p in output.rglob("O05.settings.json") if p.parent.name != "example_settings"] == []


def test_example_versus_case_exceptions_do_not_apply_to_the_case_settings_proof(tmp_path, monkeypatch):
    """E05's allowed timeStep difference is an example-versus-case exception, not a case-versus-case one."""
    tool, _, output, path, _ = _approve_project(tmp_path, monkeypatch, variants=dict(C01="identical", C02="identical", O05="tail"),
                                                captured=dict(O05=_settings_record(attributes={"timeStep": {"value": "0.2"}})))
    assert tool.approve(["O05"], reason="LEDGERS R09c", layout_correction=RULE) == 1
    assert "O05: setting differs: attributes.timeStep" in json.loads((output / "approve-summary.json").read_text())["failure"]


@pytest.mark.parametrize("problem, message", [
    ("public changed", "C01: selected settings changed"), ("private changed", "O05: selected settings changed"),
    ("private unhashed", "O05: selected settings not hashed")])
def test_selected_settings_are_hash_validated_before_any_capture(tmp_path, monkeypatch, problem, message):
    tool, _, output, path, calls = _approve_project(tmp_path, monkeypatch, variants=dict(C01="tail", C02="identical", O05="tail"),
                                                    private_digest=problem != "private unhashed")
    if problem != "private unhashed":
        case = "C01" if problem == "public changed" else "O05"
        (output / "example_settings" / f"{case}.settings.json").write_text("{}", encoding="utf-8")
    before = path.read_bytes()
    assert tool.approve(["C01", "O05"], reason="LEDGERS R09c", layout_correction=RULE) == 1
    _nothing_promoted(tmp_path, output, path, before)
    assert message in json.loads((output / "approve-summary.json").read_text(encoding="utf-8"))["failure"]
    assert calls == []


def test_private_settings_cases_still_need_a_layout_correction_or_the_proof_option(tmp_path, monkeypatch):
    tool, _, _, _, _ = _approve_project(tmp_path, monkeypatch, variants=dict(C01="identical", C02="identical", O05="tail"))
    with pytest.raises(AssertionError, match="O05 is not supported: their example settings are private"):
        tool.approve(["O05"], reason="LEDGERS R09c")
    with pytest.raises(AssertionError, match="unknown layout correction"):
        tool.approve(["O05"], reason="LEDGERS R09c", layout_correction="within-tolerance")


# The carried-forward marker is valid only for O04/O05 of a layout correction


def _promote(tool, approval, files, *, layout_correction=RULE, carried_forward=(), fault=None):
    manifest = json.loads(tool.MANIFEST.read_text(encoding="utf-8"))
    approval_id = approval.new_id(manifest, project=tool.PROJECT, output=tool.OUTPUT, package_commit=SHA, now=NOW)
    record = dict(created_utc="2026-11-13T12:00:00Z", package_commit=SHA, reason="LEDGERS R09c", tier="full", cases=sorted(files))
    if layout_correction is not None:
        record["layout_correction"] = layout_correction
    approval.promote(project=tool.PROJECT, output=tool.OUTPUT, manifest_path=tool.MANIFEST, approval_id=approval_id,
                     record=record, files=files, report=dict(schema=1, kind="approval"), fault=fault, carried_forward=carried_forward)
    return approval_id


def _candidate(tmp_path, case):
    return _write_pair(tmp_path / "candidate", case, _variant("tail"))


@pytest.mark.parametrize("layout_correction, case, files_settings, message", [
    (None, "O05", False, "only by a layout-correction approval"),
    (RULE, "C01", False, "only for listed O04/O05"),
    (RULE, "O05", True, "only for listed O04/O05"),
])
def test_promote_refuses_an_unrestricted_carry_forward(tmp_path, monkeypatch, layout_correction, case, files_settings, message):
    tool, approval, output, path = _project(tmp_path, monkeypatch)
    files = {case: _candidate(tmp_path, case)}
    if files_settings:
        files[case]["settings"] = output / "example_settings/O05.settings.json"
    before = path.read_bytes()
    with pytest.raises(approval.ApprovalError, match=message):
        _promote(tool, approval, files, layout_correction=layout_correction, carried_forward=[case])
    assert path.read_bytes() == before and not (output / "reference-approved").exists()


@pytest.mark.parametrize("how, message", [
    ("marker on a public case", "only for O04/O05 of a layout correction"),
    ("marker without layout", "only for O04/O05 of a layout correction"),
    ("unknown rule", "unknown layout_correction"),
    ("malformed marker", "artifact hashes malformed"),
])
def test_validation_refuses_an_invalid_marker_or_rule(tmp_path, monkeypatch, how, message):
    tool, approval, _, path = _project(tmp_path, monkeypatch)
    approval_id = _promote(tool, approval, {"C01": _candidate(tmp_path, "C01"), "O05": _candidate(tmp_path, "O05")},
                           carried_forward=["O05"])
    tool.validate_expected(["C01", "O05"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    record = manifest["approved_references"]["approvals"][approval_id]
    if how == "marker on a public case":
        record["artifacts"]["C01"]["settings"] = approval.CARRIED_FORWARD
    elif how == "marker without layout":
        del record["layout_correction"]
    elif how == "unknown rule":
        record["layout_correction"] = "public-tolerance"
    else:
        record["artifacts"]["O05"]["settings"] = "private-carried"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match=message):
        tool.validate_expected(["C01", "O05"])


@pytest.mark.parametrize("step", STEPS)
def test_carry_forward_promotion_stays_crash_safe_and_can_be_rolled_back(tmp_path, monkeypatch, step):
    tool, approval, output, path = _project(tmp_path, monkeypatch)
    old = path.read_bytes()
    files = {"O05": _candidate(tmp_path, "O05")}

    def fault(name):
        if name == step:
            raise RuntimeError("injected failure at " + step)
    with pytest.raises(RuntimeError, match="injected failure at " + step):
        _promote(tool, approval, files, carried_forward=["O05"], fault=fault)
    assert path.read_bytes() == old and tool.expected_reference("O05")["source"] == "reference"
    tool.validate_expected(["O05"])
    retry = _promote(tool, approval, files, carried_forward=["O05"])
    assert tool.expected_reference("O05")["source"] == retry and tool.expected_reference("O05")["settings_carried_forward"]
    tool.validate_expected(["O05"])
    approval.rollback(project=tmp_path, output=output, manifest_path=path, approval_id=retry, reason="LEDGERS R09c",
                      utc="2026-11-13T13:00:00Z")
    assert tool.expected_reference("O05")["source"] == "reference"
    tool.validate_expected(["O05"])


# Examples


def _example(tool, output, variant, settings=None):
    _write_pair(output / "package_examples", "E05", _variant(variant))
    record = settings or json.loads((output / "example_settings/O05.settings.json").read_text())
    (output / "package_examples/E05.settings.json").write_text(json.dumps(record), encoding="utf-8")


def test_examples_check_layout_and_keep_the_carried_forward_private_settings(tmp_path, monkeypatch):
    tool, approval, output, _ = _project(tmp_path, monkeypatch)
    examples = _load("layout_examples", PROJECT / "tools/acceptance_examples.py")
    settings = tool.settings_tools()
    private = json.loads((output / "inputs/private-inputs.json").read_text())
    digests = examples.settings_digests(json.loads((tmp_path / "tools/example_manifest.json").read_text()), private)
    entry = dict(reference="O05")
    _example(tool, output, "tail")
    assert examples.compare_case(tool, settings, "E05", entry, {}, digests)[0] == ["array differs: data/time", "array differs: data/values"]
    verdicts = {}
    assert examples.compare_case(tool, settings, "E05", entry, {}, digests, layout=RULE, verdicts=verdicts)[0] == []
    assert verdicts == {"E05": "tail-removal"}
    _example(tool, output, "tail+tiny")
    failures, _ = examples.compare_case(tool, settings, "E05", entry, {}, digests, layout=RULE, verdicts=verdicts)
    assert failures == ["layout: array changed outside the declared pairs: profiles/compositions"] and verdicts["E05"] == "other"
    # After a carry-forward approval the arrays come from the approval and the settings stay the adopted, private ones.
    _promote(tool, approval, {"O05": _candidate(tmp_path, "O05")}, carried_forward=["O05"])
    _example(tool, output, "identical")
    _write_pair(output / "package_examples", "E05", _variant("tail"))
    assert examples.compare_case(tool, settings, "E05", entry, {}, digests)[0] == []
    _example(tool, output, "tail", settings=_settings_record(attributes={"timeStep": {"value": "0.3"}}))
    assert examples.compare_case(tool, settings, "E05", entry, {}, digests)[0] == ["setting differs: attributes.timeStep"]
    (output / "example_settings/O05.settings.json").write_text("{}", encoding="utf-8")
    assert examples.compare_case(tool, settings, "E05", entry, {}, digests)[0] == ["adopted settings changed: O05.settings.json"]


# The non-promoting layout run


def _layout_run(tmp_path, monkeypatch, variants):
    tool, _, output, path, calls = _approve_project(tmp_path, monkeypatch, variants=variants)
    (tmp_path / ".verification/public-environment.json").write_text(json.dumps(dict(python=sys.executable)), encoding="utf-8")
    monkeypatch.setattr(tool, "check_repeatability", lambda case, **kw: dict(case=case, bit_identical=True))
    monkeypatch.setattr(tool, "run_pytest", lambda state, **kw: dict(tests=1, failures=0, errors=0, skipped=0))
    before = path.read_bytes()
    code = tool.main(["quick", "--tolerance", "exact", "--layout-correction", RULE])
    assert path.read_bytes() == before and not (output / "reference-approved").exists()
    return code, json.loads((output / "quick-layout-summary.json").read_text(encoding="utf-8"))["layout"], calls


def test_layout_run_passes_exact_corrections_and_reports_the_normal_comparison(tmp_path, monkeypatch):
    code, summary, calls = _layout_run(tmp_path, monkeypatch, dict(C01="tail", C02="identical", O05="tail"))
    assert code == 0
    assert summary["verdicts"] == {"identical": ["C02"], "tail-removal": ["C01", "O05"], "other": []}
    assert summary["comparison_failures"] == ["C01", "O05"]   # the normal exact comparison still fails, reported
    assert summary["settings_checked"] == ["C01", "O05"] and summary["settings_problems"] == [] and calls == [["C01", "O05"]]


@pytest.mark.parametrize("variant", ["tiny", "tail+tiny", "interface-tail", "two-rows"])
def test_layout_run_fails_on_any_other_change(tmp_path, monkeypatch, variant):
    code, summary, _ = _layout_run(tmp_path, monkeypatch, dict(C01=variant, C02="identical", O05="identical"))
    assert code == 1 and summary["verdicts"]["other"] == ["C01"]


@pytest.mark.parametrize("argv", [
    ["full", "--layout-correction", RULE, "--cases", "C01"],
    ["full", "--layout-correction", RULE, "--record"],
    ["quick", "--layout-correction", RULE, "--reuse-targets"],
    ["quick", "--layout-correction", "public-tolerance"],
    ["setup", "--layout-correction", RULE],
    ["approve", "--rollback", "20261113T120000Z-0123456-1", "--reason", "r", "--layout-correction", RULE]])
def test_cli_rejects_malformed_layout_requests(tmp_path, monkeypatch, argv):
    tool, _, _, _ = _project(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as error:
        tool.main(argv)
    assert error.value.code == 2


# Numerical approval of O04/O05 with verified-equal private settings (LEDGERS R13)


NUMERICAL = dict(C01="shift", C02="identical", O05="shift", O04="tiny")


def _private_digests(output):
    return json.loads((output / "inputs/private-inputs.json").read_text())["example_settings"]["sha256"]


def _no_private_capture_left(output):
    for case in ("O04", "O05"):
        assert [p for p in output.rglob(f"{case}.settings.json") if p.parent.name != "example_settings"] == []


def test_numerical_approval_carries_verified_private_settings_forward(tmp_path, monkeypatch):
    """O04/O05 with numerical differences: every gate runs, the marker and proof field are recorded, nothing private is committed."""
    tool, approval, output, path, calls = _approve_project(tmp_path, monkeypatch, variants=NUMERICAL, with_o04=True)
    gates = []
    for name in ("approval_repeat", "criteria_results", "run_pytest"):
        original = getattr(tool, name)
        monkeypatch.setattr(tool, name, lambda *a, _name=name, _original=original, **kw: gates.append(_name) or _original(*a, **kw))
    digests = _private_digests(output)
    adopted = {case: (output / "example_settings" / f"{case}.settings.json").read_bytes() for case in ("O04", "O05")}

    assert tool.approve(["C01", "O04", "O05"], reason="LEDGERS R13", verify_private_settings=True) == 0

    assert gates.count("approval_repeat") == 4 and gates.count("criteria_results") == 4 and gates.count("run_pytest") == 1
    manifest_text = path.read_text(encoding="utf-8")
    section = json.loads(manifest_text)["approved_references"]
    (approval_id,) = section["approvals"]
    record = section["approvals"][approval_id]
    assert record["private_settings"] == approval.VERIFIED_EQUAL and "layout_correction" not in record
    assert section["in_effect"] == {case: approval_id for case in ("C01", "O04", "O05")}
    for case in ("O04", "O05"):
        assert record["artifacts"][case]["settings"] == approval.CARRIED_FORWARD
    assert len(record["artifacts"]["C01"]["settings"]) == 64
    folder = output / record["directory"]
    assert sorted(p.name for p in folder.iterdir()) == ["C01.json", "C01.npz", "C01.settings.json", "O04.json", "O04.npz",
                                                         "O05.json", "O05.npz"]
    report_text = (tmp_path / record["report"]["path"]).read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["private_settings"] == "verified-equal" and report["settings_carried_forward"] == ["O04", "O05"]
    assert report["settings_captured"] == ["C01"] and "layout_verdicts" not in report
    assert report["listed_cases"] == {"C01": "differs", "O04": "within public tolerance", "O05": "differs"}
    assert report["gates"]["unlisted_cases_passed_public_tolerance"] == ["C02"]
    # Deviations: numbers for the public case, verdicts only for O04 and O05.
    deviations = report["deviations_vs_previous"]
    assert deviations["C01"]["arrays"]["profiles/compositions"]["max_absolute_deviation"] == pytest.approx(1e-6)
    assert deviations["O05"] == dict(private_metrics=True, passed_public_tolerance=False, bit_identical=False)
    assert deviations["O04"] == dict(private_metrics=True, passed_public_tolerance=True, bit_identical=False)
    # Criteria: O04 (TC-derived) as verdicts; O05 keeps value and limit, as in the earlier reports.
    criteria = report["gates"]["criteria"]
    assert criteria["O04"] == [dict(criterion="conservation/inventory", passed=True, private_metrics=True)]
    assert criteria["O05"] == [dict(criterion="conservation/inventory", value=1e-15, limit=1e-13, passed=True)]
    # No settings content, settings hash or private digest is committed; the adopted files are untouched.
    for text in (manifest_text, report_text):
        assert not any(digest in text for digest in digests.values())
        assert "minDtFrac" not in text and "timeStep" not in text
    for case, data in adopted.items():
        assert (output / "example_settings" / f"{case}.settings.json").read_bytes() == data
    _no_private_capture_left(output)
    assert calls == [["O04", "O05"], ["C01"]]   # the private proof first, then the public capture only
    for case in ("O04", "O05"):
        selected = tool.expected_reference(case)
        assert selected["source"] == approval_id and selected["settings_carried_forward"] and selected["settings"] is None
        chosen, digest, private = tool.selected_case_settings(case, json.loads(manifest_text))
        assert chosen == output / "example_settings" / f"{case}.settings.json" and digest == digests[chosen.name] and private
    tool.validate_expected(["C01", "C02", "O04", "O05"])


def test_numerical_approval_of_private_cases_needs_the_explicit_option(tmp_path, monkeypatch):
    tool, _, output, path, calls = _approve_project(tmp_path, monkeypatch, variants=NUMERICAL, with_o04=True)
    before = path.read_bytes()
    with pytest.raises(AssertionError, match="O04, O05 is not supported: their example settings are private"):
        tool.approve(["O04", "O05"], reason="LEDGERS R13")
    with pytest.raises(AssertionError, match="needs O04 or O05 among the listed cases"):
        tool.approve(["C01"], reason="LEDGERS R13", verify_private_settings=True)
    with pytest.raises(AssertionError, match="not a layout correction"):
        tool.approve(["O05"], reason="LEDGERS R13", verify_private_settings=True, layout_correction=RULE)
    _nothing_promoted(tmp_path, output, path, before)
    assert calls == []


@pytest.mark.parametrize("problem, message", [
    ("setting differs", "O05: setting differs: attributes.tolerance"),
    ("solve differs", "O04: setting differs: solve"),
    ("file missing", "O05: selected settings missing: O05.settings.json"),
    ("not hashed", "O04: selected settings not hashed: O04.settings.json"),
    ("no private record", "O04: selected settings not hashed: O04.settings.json"),
    ("hash differs", "O04: selected settings changed: O04.settings.json"),
])
def test_private_settings_proof_failures_promote_nothing_and_leave_no_capture(tmp_path, monkeypatch, problem, message):
    captured = {}
    if problem == "setting differs":
        captured["O05"] = _settings_record(attributes={"tolerance": {"value": "1e-10"}})
    if problem == "solve differs":
        captured["O04"] = _settings_record(duration="2.0", solve={"minDtFrac": 1e-12})
    tool, _, output, path, calls = _approve_project(tmp_path, monkeypatch, variants=NUMERICAL, with_o04=True, captured=captured,
                                                    private_digest=problem != "no private record")
    if problem == "file missing":
        (output / "example_settings/O05.settings.json").unlink()
    if problem == "not hashed":
        private = json.loads((output / "inputs/private-inputs.json").read_text())
        del private["example_settings"]["sha256"]["O04.settings.json"]
        (output / "inputs/private-inputs.json").write_text(json.dumps(private), encoding="utf-8")
    if problem == "hash differs":
        (output / "example_settings/O04.settings.json").write_text(json.dumps(_settings_record(duration="3.0")), encoding="utf-8")
    before = path.read_bytes()
    assert tool.approve(["C01", "O04", "O05"], reason="LEDGERS R13", verify_private_settings=True) == 1
    _nothing_promoted(tmp_path, output, path, before)
    failure = json.loads((output / "approve-summary.json").read_text(encoding="utf-8"))["failure"]
    assert "private settings proof failed" in failure and message in failure
    _no_private_capture_left(output)
    assert calls == ([["O04", "O05"]] if problem in ("setting differs", "solve differs") else [])


@pytest.mark.parametrize("gate, message", [
    ("unlisted", "unlisted case fails its public-tolerance comparison"),
    ("criteria", "approval_criteria failed"),
    ("repeat", "not repeatable"),
    ("pytest", "failures"),
])
def test_every_existing_gate_still_blocks_a_private_settings_approval(tmp_path, monkeypatch, gate, message):
    variants = dict(NUMERICAL, C02="shift") if gate == "unlisted" else NUMERICAL
    tool, _, output, path, calls = _approve_project(tmp_path, monkeypatch, variants=variants, with_o04=True,
                                                    failing_criteria=("O05",) if gate == "criteria" else ())
    if gate == "repeat":
        def repeat(case, **kw):
            raise AssertionError(("Unseeded package role is not repeatable", case, ["data/values"]))
        monkeypatch.setattr(tool, "approval_repeat", repeat)
    if gate == "pytest":
        def failing(state, **kw):
            raise AssertionError({"tests": 9, "failures": 1, "errors": 0, "skipped": 0})
        monkeypatch.setattr(tool, "run_pytest", failing)
    before = path.read_bytes()
    assert tool.approve(["C01", "O04", "O05"], reason="LEDGERS R13", verify_private_settings=True) == 1
    _nothing_promoted(tmp_path, output, path, before)
    assert message in json.loads((output / "approve-summary.json").read_text(encoding="utf-8"))["failure"]
    assert calls == []   # the gates fail before any settings are captured


def test_private_settings_approval_rolls_back_to_the_previous_selection(tmp_path, monkeypatch):
    """After a layout approval, a numerical one, and its rollback, O05 selects the same adopted settings throughout."""
    tool, approval, output, path, _ = _approve_project(tmp_path, monkeypatch, variants=dict(C01="identical", C02="identical", O05="tail"))
    assert tool.approve(["O05"], reason="LEDGERS R09c", layout_correction=RULE) == 0
    first = tool.expected_reference("O05")["source"]
    selection = tool.selected_case_settings("O05", json.loads(path.read_text(encoding="utf-8")))
    arrays = _variant("tail")
    arrays["profiles/compositions"] = arrays["profiles/compositions"] + 1e-6
    monkeypatch.setattr(tool, "worker", lambda case, role, **kw: json.loads(_write_pair(
        output / "package", case, arrays if case == "O05" else _variant("identical"))["json"].read_text(encoding="utf-8")))
    # approval IDs are per second and commit: the counter keeps them distinct
    assert tool.approve(["O05"], reason="LEDGERS R13", verify_private_settings=True) == 0
    second = tool.expected_reference("O05")["source"]
    assert second != first and tool.selected_case_settings("O05", json.loads(path.read_text(encoding="utf-8"))) == selection
    record = json.loads(path.read_text(encoding="utf-8"))["approved_references"]["approvals"][second]
    assert record["previous"] == {"O05": first} and record["private_settings"] == "verified-equal"

    # A refused rollback (the restored approval's file is gone) leaves the manifest unchanged.
    restored = output / "reference-approved" / first / "O05.npz"
    data, before = restored.read_bytes(), path.read_bytes()
    restored.unlink()
    with pytest.raises(approval.ApprovalError, match="rollback refused, the manifest is unchanged"):
        approval.rollback(project=tmp_path, output=output, manifest_path=path, approval_id=second, reason="LEDGERS R13",
                          utc="2026-11-14T13:00:00Z")
    assert path.read_bytes() == before
    restored.write_bytes(data)

    monkeypatch.setattr(tool, "repository_checks", lambda: SHA)
    assert tool.rollback_approval(second, reason="LEDGERS R13") == 0
    assert tool.expected_reference("O05")["source"] == first
    assert tool.selected_case_settings("O05", json.loads(path.read_text(encoding="utf-8"))) == selection
    tool.validate_expected(["C01", "C02", "O05"])


def _promote_verified(tool, approval, files, *, carried_forward, private_settings="verified-equal", layout_correction=None):
    manifest = json.loads(tool.MANIFEST.read_text(encoding="utf-8"))
    approval_id = approval.new_id(manifest, project=tool.PROJECT, output=tool.OUTPUT, package_commit=SHA, now=NOW)
    record = dict(created_utc="2026-11-14T12:00:00Z", package_commit=SHA, reason="LEDGERS R13", tier="full", cases=sorted(files))
    if private_settings is not None:
        record["private_settings"] = private_settings
    if layout_correction is not None:
        record["layout_correction"] = layout_correction
    approval.promote(project=tool.PROJECT, output=tool.OUTPUT, manifest_path=tool.MANIFEST, approval_id=approval_id,
                     record=record, files=files, report=dict(schema=1, kind="approval"), carried_forward=carried_forward)
    return approval_id


@pytest.mark.parametrize("cases, kwargs, message", [
    (["O05"], dict(carried_forward=["O05"], private_settings=None), "only by a layout-correction approval or one with verified-equal"),
    (["O05"], dict(carried_forward=["O05"], private_settings="assumed-equal"), "private_settings must be verified-equal"),
    (["O05"], dict(carried_forward=["O05"], layout_correction=RULE), "private_settings must be verified-equal"),
    (["C01"], dict(carried_forward=[]), "private_settings must be verified-equal"),
    (["C01"], dict(carried_forward=["C01"]), "only for listed O04/O05"),
    (["C01", "O05"], dict(carried_forward=[], private_settings=None), "private settings must be carried forward for O05"),
])
def test_promote_refuses_an_unproven_or_misplaced_private_settings_record(tmp_path, monkeypatch, cases, kwargs, message):
    tool, approval, output, path = _project(tmp_path, monkeypatch)
    before = path.read_bytes()
    with pytest.raises(approval.ApprovalError, match=message):
        _promote_verified(tool, approval, {case: _candidate(tmp_path, case) for case in cases}, **kwargs)
    assert path.read_bytes() == before and not (output / "reference-approved").exists()


@pytest.mark.parametrize("how, message", [
    ("marker without the proof field", "only for O04/O05 of a layout correction or of an approval with verified-equal"),
    ("marker on a public case", "only for O04/O05 of a layout correction or of an approval with verified-equal"),
    ("unknown proof", "unknown private_settings"),
    ("proof beside a layout correction", "private_settings and layout_correction exclude each other"),
    ("proof without a carried case", "private_settings without a carried-forward case"),
    ("private case with a settings hash", "O05 must carry its private settings forward"),
])
def test_validation_refuses_an_invalid_private_settings_record(tmp_path, monkeypatch, how, message):
    tool, approval, _, path = _project(tmp_path, monkeypatch)
    approval_id = _promote_verified(tool, approval, {"C01": _candidate(tmp_path, "C01"), "O05": _candidate(tmp_path, "O05")},
                                    carried_forward=["O05"])
    tool.validate_expected(["C01", "O05"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    record = manifest["approved_references"]["approvals"][approval_id]
    if how == "marker without the proof field":
        del record["private_settings"]
    elif how == "marker on a public case":
        record["artifacts"]["C01"]["settings"] = approval.CARRIED_FORWARD
    elif how == "unknown proof":
        record["private_settings"] = "assumed-equal"
    elif how == "proof beside a layout correction":
        record["layout_correction"] = RULE
    elif how == "proof without a carried case":
        record["artifacts"]["O05"].pop("settings")
    else:   # an ordinary record that stores a settings hash for a private case
        del record["private_settings"]
        record["artifacts"]["O05"]["settings"] = "0" * 64
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match=message):
        tool.validate_expected(["C01", "O05"])


def test_examples_use_the_adopted_private_settings_after_a_numerical_approval(tmp_path, monkeypatch):
    tool, approval, output, _ = _project(tmp_path, monkeypatch)
    examples = _load("layout_examples_numerical", PROJECT / "tools/acceptance_examples.py")
    settings = tool.settings_tools()
    private = json.loads((output / "inputs/private-inputs.json").read_text())
    digests = examples.settings_digests(json.loads((tmp_path / "tools/example_manifest.json").read_text()), private)
    entry = dict(reference="O05")
    _promote_verified(tool, approval, {"O05": _write_pair(tmp_path / "candidate", "O05", _variant("shift"))}, carried_forward=["O05"])
    _example(tool, output, "shift")
    assert examples.compare_case(tool, settings, "E05", entry, {}, digests)[0] == []   # arrays from the approval, adopted settings
    _example(tool, output, "identical")
    assert examples.compare_case(tool, settings, "E05", entry, {}, digests)[0] == ["array differs: profiles/compositions"]
    _example(tool, output, "shift", settings=_settings_record(attributes={"timeStep": {"value": "0.3"}}))
    assert examples.compare_case(tool, settings, "E05", entry, {}, digests)[0] == ["setting differs: attributes.timeStep"]
    _example(tool, output, "shift")
    assert examples.compare_case(tool, settings, "E05", entry, {}, {})[0] != []   # no private digest: not verifiable
    (output / "example_settings/O05.settings.json").write_text("{}", encoding="utf-8")
    assert examples.compare_case(tool, settings, "E05", entry, {}, digests)[0] == ["adopted settings changed: O05.settings.json"]
    (output / "example_settings/O05.settings.json").unlink()
    assert examples.compare_case(tool, settings, "E05", entry, {}, digests)[0] == ["adopted reference settings missing: O05.settings.json"]


@pytest.mark.parametrize("argv", [
    ["quick", "--verify-private-settings"],
    ["approve", "--rollback", "20261113T120000Z-0123456-1", "--reason", "r", "--verify-private-settings"],
    ["approve", "--cases", "O05", "--reason", "r", "--layout-correction", RULE, "--verify-private-settings"]])
def test_cli_rejects_misplaced_private_settings_requests(tmp_path, monkeypatch, argv):
    tool, _, _, _ = _project(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as error:
        tool.main(argv)
    assert error.value.code == 2


# A capture that fails or is interrupted part-way leaves no private capture behind (LEDGERS R13 follow-up)


def _failing_capture(tool, monkeypatch, error, calls):
    """Capture that completes C01 and O04, then fails while O05 is being captured (a partial O05 file exists)."""
    def capture(cases, state, directory):
        calls.append(sorted(cases))
        for case in sorted(cases):
            path = Path(directory) / f"{case}.settings.json"
            if case == "O05":
                path.write_text('{"class": "Mod', encoding="utf-8")   # truncated, as an interrupted worker would leave it
                raise error
            path.write_text(json.dumps(_settings_record(duration="2.0") if case == "O04" else _settings_record(),
                                       indent=1, sort_keys=True), encoding="utf-8")
        return {}
    monkeypatch.setattr(tool, "capture_example_settings", capture)


def _captures_in(directory):
    return sorted(path.name for path in Path(directory).iterdir())


@pytest.mark.parametrize("error", [RuntimeError("worker timed out"), KeyboardInterrupt()], ids=["failure", "interruption"])
def test_check_case_settings_removes_private_captures_when_a_later_capture_fails(tmp_path, monkeypatch, error):
    """O04 is captured, O05's capture fails: both private files go, the original exception propagates."""
    tool, _, output, path, calls = _approve_project(tmp_path, monkeypatch, variants=NUMERICAL, with_o04=True)
    _failing_capture(tool, monkeypatch, error, calls)
    directory = tmp_path / "captures"
    directory.mkdir()
    (directory / "notes.txt").write_text("unrelated", encoding="utf-8")
    (directory / "O05.settings.json.keep").write_text("unrelated", encoding="utf-8")
    adopted = {name: (output / "example_settings" / name).read_bytes() for name in ("C01.settings.json", "O04.settings.json", "O05.settings.json")}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(type(error)) as raised:
        tool.check_case_settings(["C01", "O04", "O05"], dict(target="t", host="h"), directory, manifest)
    assert raised.value is error
    # The private captures are gone, complete or partial; the public capture and unrelated files are not touched.
    assert _captures_in(directory) == ["C01.settings.json", "O05.settings.json.keep", "notes.txt"]
    for name, data in adopted.items():
        assert (output / "example_settings" / name).read_bytes() == data
    _no_private_capture_left(output)
    assert calls == [["C01", "O04", "O05"]]


@pytest.mark.parametrize("mode", ["private-settings", "layout"])
def test_approval_promotes_nothing_and_leaves_no_private_capture_when_a_capture_fails(tmp_path, monkeypatch, mode):
    variants = NUMERICAL if mode == "private-settings" else dict(C01="tail", C02="identical", O05="tail", O04="tail")
    tool, _, output, path, calls = _approve_project(tmp_path, monkeypatch, variants=variants, with_o04=True)
    _failing_capture(tool, monkeypatch, RuntimeError("worker timed out"), calls)
    before = path.read_bytes()
    kwargs = dict(verify_private_settings=True) if mode == "private-settings" else dict(layout_correction=RULE)
    assert tool.approve(["C01", "O04", "O05"], reason="LEDGERS R13", **kwargs) == 1
    _nothing_promoted(tmp_path, output, path, before)
    assert "worker timed out" in json.loads((output / "approve-summary.json").read_text(encoding="utf-8"))["failure"]
    _no_private_capture_left(output)


@pytest.mark.parametrize("mode", ["private-settings", "layout"])
def test_an_interrupted_capture_promotes_nothing_and_leaves_no_private_capture(tmp_path, monkeypatch, mode):
    variants = NUMERICAL if mode == "private-settings" else dict(C01="tail", C02="identical", O05="tail", O04="tail")
    tool, _, output, path, calls = _approve_project(tmp_path, monkeypatch, variants=variants, with_o04=True)
    _failing_capture(tool, monkeypatch, KeyboardInterrupt(), calls)
    before = path.read_bytes()
    kwargs = dict(verify_private_settings=True) if mode == "private-settings" else dict(layout_correction=RULE)
    with pytest.raises(KeyboardInterrupt):
        tool.approve(["C01", "O04", "O05"], reason="LEDGERS R13", **kwargs)
    _nothing_promoted(tmp_path, output, path, before)
    _no_private_capture_left(output)
    assert len(calls) == 1
