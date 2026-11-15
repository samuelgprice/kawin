"""Package-owned baselines (C33/F33 tooling) on synthetic tmp_path projects; nothing touches the real one.

Covers registration refusals, the fixture split and scientific digest, the fixed independent
checks, crash-safe staging, the documented exclusive install and its recovery, verification,
the ``baseline`` orchestration's gates, the historical skip, and Option S approval compatibility
(package-origin files in ``reference/`` with unchanged selection and rollback).
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
NOW = datetime(2026, 11, 16, 9, 0, 0, tzinfo=timezone.utc)
BOUNDS = dict(eta_max_abs=5e-3, profile_max_abs=1e-3, k_rel_abs=1e-2,
              interface_max_abs_over_displacement=1e-2, final_inventory_drift_abs=1e-9)
ORIGIN = 0.3 + 1e-12


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load("acceptance_baseline_test", PROJECT / "tools/acceptance_baseline.py")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _snapshot(folder):
    return {p.relative_to(folder).as_posix(): _sha(p) for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def _arrays(scale=1.0):
    """Arrays a C33 run would record, within every independent bound when ``scale`` = 1."""
    t = np.linspace(0.0, 0.3, 6)
    s_exact = ORIGIN + 7.6e-3 * np.sqrt(t + 0.1)
    return {"interfaceData/time": t, "interfaceData/values": s_exact + 4e-6 * scale,
            "exact/interface_error": np.full(6, 4e-6 * scale), "exact/eta_error": np.full(6, -4.6e-4 * scale),
            "exact/profile_error_X": np.full(62, 1.2e-4 * scale), "exact/profile_error_Y": np.full(62, -1.0e-4 * scale),
            "exact/k_rel_error": np.array([7e-4 * scale]),
            "inventoryData/values": np.array([[0.174, 0.051], [0.174 + 2e-11 * scale, 0.051 - 5e-12 * scale]])}


def _report(case="C33", **extra):
    report = dict(case=case, role="package", kawin_origin="C:/somewhere/kawin/__init__.py", hash_seed=None,
                  numpy_seed_override=False, final_time=0.3, status="PASS", arrays=8, elapsed_seconds=1.23,
                  residual_budget_inputs=dict(residual_tolerance=1e-11, min_composition=1e-10, domain_length=0.6,
                                              explicit_composition_scale=None, inventory_record_interval=1))
    report.update(extra)
    return report


def _npz_bytes(tmp_path, arrays, name="x.npz"):
    path = tmp_path / name
    np.savez_compressed(path, **arrays)
    return path.read_bytes()


def _manifest(expected=None):
    entry = dict(configuration=dict(similarity_origin=ORIGIN), independent_bounds=BOUNDS)
    return dict(expected_artifacts=dict(cases=expected or {}),
                package_baselines=dict(cases={"C33": entry, "F33": entry}))


# ---------------------------------------------------------------- pure helpers

def test_registration_refuses_unregistered_research_and_existing_cases():
    BASE.check_registration(_manifest(), ["C33", "F33"])
    with pytest.raises(BASE.BaselineError, match="not a registered"):
        BASE.check_registration(_manifest(dict(C01=dict(json="a", npz="b"))), ["C01"])     # research case
    with pytest.raises(BASE.BaselineError, match="not a registered"):
        BASE.check_registration(_manifest(), ["C99"])
    with pytest.raises(BASE.BaselineError, match="already has an expected_artifacts entry"):
        BASE.check_registration(_manifest(dict(C33=dict(json="a", npz="b", captured_role="package"))), ["C33"])
    with pytest.raises(BASE.BaselineError, match="no package-captured"):
        BASE.check_registration(_manifest(), ["C33"], verify=True)
    with pytest.raises(BASE.BaselineError, match="needs --cases"):
        BASE.check_registration(_manifest(), [])
    BASE.check_registration(_manifest(dict(C33=dict(captured_role="package", scientific_sha256="d"))), ["C33"], verify=True)


def test_fixture_report_strips_run_records_and_requires_an_unseeded_pass():
    stripped = BASE.fixture_report(_report(wall_seconds=2.0))
    assert not set(BASE.STRIPPED_FIELDS) & set(stripped) and stripped["final_time"] == 0.3
    assert json.loads(BASE.fixture_json_bytes(stripped)) == stripped
    for bad in (dict(status="FAIL"), dict(hash_seed="0"), dict(numpy_seed_override=True)):
        with pytest.raises(BASE.BaselineError):
            BASE.fixture_report({**_report(), **bad})


def test_scientific_digest_ignores_run_records_but_not_content():
    arrays, stripped = _arrays(), BASE.fixture_report(_report())
    digest = BASE.scientific_sha256(stripped, arrays)
    assert BASE.scientific_sha256({**stripped, "elapsed_seconds": 9.9, "kawin_origin": "x"}, arrays) == digest
    assert BASE.scientific_sha256({**stripped, "final_time": 0.31}, arrays) != digest
    changed = dict(arrays, **{"exact/k_rel_error": np.array([7.0000001e-4])})
    assert BASE.scientific_sha256(stripped, changed) != digest
    retyped = dict(arrays, **{"exact/k_rel_error": arrays["exact/k_rel_error"].astype(np.float32)})
    assert BASE.scientific_sha256(stripped, retyped) != digest


def test_independent_checks_use_the_fixed_source_bounds():
    good = BASE.independent_checks(_arrays(), BOUNDS, ORIGIN)
    assert good["passed"] and good["values"]["final_inventory_drift_abs"] == pytest.approx(2e-11)
    assert good["values"]["interface_max_abs_over_displacement"] == pytest.approx(4e-6 / good["displacement"])
    bad = BASE.independent_checks(_arrays(scale=60.0), BOUNDS, ORIGIN)     # 60 x: drift 1.2e-9 > 1e-9
    assert not bad["passed"] and "final_inventory_drift_abs" in bad["failures"]
    with pytest.raises(BASE.BaselineError, match="missing"):
        BASE.independent_checks({k: v for k, v in _arrays().items() if k != "exact/k_rel_error"}, BOUNDS, ORIGIN)


def _files(tmp_path):
    return {case: dict(json=BASE.fixture_json_bytes(BASE.fixture_report(_report(case))),
                       npz=_npz_bytes(tmp_path, _arrays(), f"{case}-src.npz")) for case in ("C33", "F33")}


@pytest.mark.parametrize("step", ["written", "validated", "recorded"])
def test_an_interrupted_capture_leaves_only_an_inert_partial_and_a_retry_succeeds(tmp_path, step):
    root = tmp_path / "staging"
    first = BASE.new_capture_id(root, SHA, NOW)

    def fault(at):
        if at == step:
            raise KeyboardInterrupt(at)
    with pytest.raises(KeyboardInterrupt):
        BASE.write_capture(root, first, _files(tmp_path), dict(package_commit=SHA), fault=fault)
    assert BASE.leftovers(root, _manifest()) == dict(partial=[first + ".partial"], unreferenced=[])
    assert not (root / first).exists()
    second = BASE.new_capture_id(root, SHA, NOW)
    assert second != first and second.endswith("-2")
    folder, hashes = BASE.write_capture(root, second, _files(tmp_path), dict(package_commit=SHA))
    record = json.loads((folder / BASE.CAPTURE_RECORD).read_text(encoding="utf-8"))
    assert record["capture_id"] == second and record["files"] == hashes and record["package_commit"] == SHA
    assert BASE.leftovers(root, _manifest()) == dict(partial=[first + ".partial"], unreferenced=[second])
    referenced = _manifest(dict(C33=dict(capture_id=second)))
    assert BASE.leftovers(root, referenced)["unreferenced"] == []
    with pytest.raises(BASE.BaselineError, match="never reused"):
        BASE.write_capture(root, first, _files(tmp_path), {})
    with pytest.raises(BASE.BaselineError, match="never reused"):
        BASE.write_capture(root, second, _files(tmp_path), {})


def _capture(tmp_path):
    root = tmp_path / "staging"
    folder, _ = BASE.write_capture(root, BASE.new_capture_id(root, SHA, NOW), _files(tmp_path), dict(package_commit=SHA))
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "C01.json").write_text("{}", encoding="utf-8")       # a research fixture that must stay untouched
    return folder, reference


def test_install_is_exclusive_hash_checked_and_idempotent(tmp_path):
    folder, reference = _capture(tmp_path)
    research = _sha(reference / "C01.json")
    status = BASE.install_capture(folder, reference)
    assert set(status.values()) == {"installed"} and len(status) == 4
    assert BASE.install_capture(folder, reference) == {name: "already installed" for name in status}
    for name in status:
        assert (reference / name).read_bytes() == (folder / name).read_bytes()
    assert _sha(reference / "C01.json") == research


def test_an_interrupted_install_is_completed_from_the_same_capture_only(tmp_path, monkeypatch):
    folder, reference = _capture(tmp_path)
    original = BASE._write_new
    calls = []

    def flaky(path, data):
        calls.append(Path(path).name)
        if len(calls) == 2:
            raise KeyboardInterrupt("interrupted between files")
        original(path, data)
    monkeypatch.setattr(BASE, "_write_new", flaky)
    with pytest.raises(KeyboardInterrupt):
        BASE.install_capture(folder, reference)
    monkeypatch.setattr(BASE, "_write_new", original)
    assert sorted(p.name for p in reference.iterdir()) == ["C01.json", "C33.json"]
    status = BASE.install_capture(folder, reference)
    assert status["C33.json"] == "already installed" and status["C33.npz"] == "installed"


def _truncating_write(path, data):
    """Simulate an interruption in the middle of a file write: half the bytes, then stop."""
    with open(path, "xb") as handle:
        handle.write(data[: len(data) // 2])
    raise KeyboardInterrupt("interrupted mid-write")


def test_an_interruption_during_a_write_is_recovered_from_the_same_capture(tmp_path, monkeypatch):
    folder, reference = _capture(tmp_path)
    original = BASE._write_new
    monkeypatch.setattr(BASE, "_write_new", _truncating_write)
    with pytest.raises(KeyboardInterrupt):
        BASE.install_capture(folder, reference)
    temporary = reference / BASE._temporary_name("C33.json", folder.name)
    assert not (reference / "C33.json").exists() and 0 < temporary.stat().st_size < (folder / "C33.json").stat().st_size
    monkeypatch.setattr(BASE, "_write_new", original)
    status = BASE.install_capture(folder, reference)              # same capture, same record
    assert set(status.values()) == {"installed"} and len(status) == 4
    for name in status:
        assert (reference / name).read_bytes() == (folder / name).read_bytes()
    assert sorted(p.name for p in reference.iterdir()) == ["C01.json", "C33.json", "C33.npz", "F33.json", "F33.npz"]
    # the truncated bytes went to a temporary sibling, never to the final name


@pytest.mark.parametrize("when", ["before publication", "after publication"])
def test_an_interruption_around_publication_is_recovered_from_the_same_capture(tmp_path, monkeypatch, when):
    folder, reference = _capture(tmp_path)
    original = BASE._publish

    def interrupted(temporary, target):
        assert BASE._sha(temporary) == BASE._sha(folder / target.name)   # verified before publication
        if when == "after publication":
            import os
            os.link(temporary, target)                                   # published, temporary not yet removed
        raise KeyboardInterrupt(when)
    monkeypatch.setattr(BASE, "_publish", interrupted)
    with pytest.raises(KeyboardInterrupt):
        BASE.install_capture(folder, reference)
    first = sorted(p.name for p in reference.iterdir())
    assert any(name.endswith(".install-tmp") for name in first)
    assert ("C33.json" in first) == (when == "after publication")
    monkeypatch.setattr(BASE, "_publish", original)
    status = BASE.install_capture(folder, reference)
    assert status["C33.json"] == ("already installed" if when == "after publication" else "installed")
    assert sorted(p.name for p in reference.iterdir()) == ["C01.json", "C33.json", "C33.npz", "F33.json", "F33.npz"]
    for name in status:
        assert (reference / name).read_bytes() == (folder / name).read_bytes()


def test_install_leaves_matching_files_and_unrelated_leftovers_untouched(tmp_path):
    folder, reference = _capture(tmp_path)
    BASE.install_capture(folder, reference)
    other_capture_tmp = reference / ".C33.json.20261101T000000Z-fedcba9-1.install-tmp"   # another capture's leftover
    other_capture_tmp.write_bytes(b"not ours")
    (reference / "notes.txt").write_text("unrelated", encoding="utf-8")
    stats = {p.name: (p.stat().st_mtime_ns, _sha(p)) for p in reference.iterdir()}
    assert set(BASE.install_capture(folder, reference).values()) == {"already installed"}
    assert {p.name: (p.stat().st_mtime_ns, _sha(p)) for p in reference.iterdir()} == stats


def test_a_conflicting_destination_is_refused_even_with_a_stale_temporary(tmp_path):
    folder, reference = _capture(tmp_path)
    stale = reference / BASE._temporary_name("C33.json", folder.name)
    stale.write_bytes(b"half a file")
    (reference / "C33.json").write_bytes(b"someone else's fixture")
    with pytest.raises(BASE.BaselineError, match="exists and differs"):
        BASE.install_capture(folder, reference)
    assert (reference / "C33.json").read_bytes() == b"someone else's fixture"
    assert stale.read_bytes() == b"half a file"                  # nothing changed before the refusal


def test_publication_never_overwrites(tmp_path):
    temporary, target = tmp_path / "t", tmp_path / "d"
    temporary.write_bytes(b"new")
    target.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        BASE._publish(temporary, target)
    assert target.read_bytes() == b"old" and temporary.read_bytes() == b"new"


def test_install_never_overwrites_and_refuses_bad_sources(tmp_path):
    folder, reference = _capture(tmp_path)
    (reference / "C33.npz").write_bytes(b"different bytes")
    with pytest.raises(BASE.BaselineError, match="exists and differs"):
        BASE.install_capture(folder, reference)
    assert (reference / "C33.npz").read_bytes() == b"different bytes"
    (reference / "C33.npz").unlink()
    for name in ("C33.json", "F33.json", "F33.npz"):
        (reference / name).unlink(missing_ok=True)
    tampered = folder / "F33.json"
    good = tampered.read_bytes()
    tampered.write_bytes(good + b" ")
    before = _snapshot(reference)
    with pytest.raises(BASE.BaselineError, match="does not match capture.json"):
        BASE.install_capture(folder, reference)
    assert _snapshot(reference) == before                         # checked before copying anything
    tampered.write_bytes(good)
    partial = folder.with_name(folder.name + ".partial")
    folder.rename(partial)
    with pytest.raises(BASE.BaselineError, match="not a completed capture"):
        BASE.install_capture(partial, reference)


def test_verify_compares_arrays_and_retained_fields_only():
    arrays, reference_report = _arrays(), BASE.fixture_report(_report())
    digest = BASE.scientific_sha256(reference_report, arrays)
    same = BASE.verify_identity(_report(elapsed_seconds=7.7, kawin_origin="elsewhere"), dict(arrays),
                                reference_report, arrays, digest)
    assert same["passed"] and same["digest_equal"] and not same["differing_fields"]
    altered = dict(arrays, **{"exact/eta_error": arrays["exact/eta_error"] * (1 + 1e-15)})
    result = BASE.verify_identity(_report(), altered, reference_report, arrays, digest)
    assert not result["passed"] and result["differing_arrays"] == ["exact/eta_error"] and not result["digest_equal"]
    result = BASE.verify_identity(_report(final_time=0.31), arrays, reference_report, arrays)
    assert not result["passed"] and result["differing_fields"] == ["final_time"] and result["digest_equal"] is None
    missing = {k: v for k, v in arrays.items() if k != "exact/k_rel_error"}
    assert not BASE.verify_identity(_report(), missing, reference_report, arrays)["keys_equal"]


# ---------------------------------------------------------------- orchestration on a synthetic project

def _project(tmp_path, monkeypatch, *, package_entry=False):
    """acceptance.py on a synthetic project: research C01, registered C33/F33 (C33 optionally captured)."""
    tool = _load("baseline_tool", PROJECT / "tools/acceptance.py")
    tool.metric_tools()
    output = tmp_path / ".verification/acceptance"
    (output / "inputs").mkdir(parents=True)
    reference = output / "reference"
    reference.mkdir()
    np.savez(reference / "C01.npz", x=np.array([0.0, 1.0]))
    (reference / "C01.json").write_text(json.dumps(dict(status="PASS", stop_status="completed")), encoding="utf-8")
    expected = {"C01": dict(json=_sha(reference / "C01.json"), npz=_sha(reference / "C01.npz"), captured_role="reference")}
    if package_entry:
        stripped = BASE.fixture_report(_report())
        (reference / "C33.json").write_bytes(BASE.fixture_json_bytes(stripped))
        np.savez_compressed(reference / "C33.npz", **_arrays())
        expected["C33"] = dict(json=_sha(reference / "C33.json"), npz=_sha(reference / "C33.npz"), captured_role="package",
                               capture_id="20261116T090000Z-0123456-1",
                               scientific_sha256=BASE.scientific_sha256(stripped, _arrays()))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/definitions-provenance.json").write_text(json.dumps(dict(definitions={})), encoding="utf-8")
    (tmp_path / "tools").mkdir()
    entry = dict(configuration=dict(similarity_origin=ORIGIN), independent_bounds=BOUNDS)
    manifest = dict(schema=1, quick=["C01"] + (["C33"] if package_entry else []), contracts=[], full_drift=[],
                    public_tolerances=dict(profiles=dict(rtol=1e-10, atol=1e-12)),
                    default_phase_sampling=dict(case="O08", reference_sha256={}), thermocalc_sampling=dict(repeat_unseeded=False),
                    package_baselines=dict(cases={"C33": entry, "F33": entry}),
                    expected_artifacts=dict(cases=expected), adopted_inputs=dict(files={}),
                    approval_criteria=dict(analytic={}, conservation={}))
    path = tmp_path / "tools/acceptance_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(tool, "PROJECT", tmp_path)
    monkeypatch.setattr(tool, "OUTPUT", output)
    monkeypatch.setattr(tool, "MANIFEST", path)
    return tool, output, path


def _fake_runs(tool, monkeypatch, tmp_path, *, scale=1.0, differ=False):
    target = tmp_path / "target/moving_boundary"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("# synthetic", encoding="utf-8")
    calls = dict(setup=0, repeat=[])
    monkeypatch.setattr(tool, "repository_checks", lambda: SHA)

    def setup():
        calls["setup"] += 1
        return dict(target=str(tmp_path / "target"), host=str(tmp_path), manifest_sha256="m" * 64,
                    wheel_name="moving_boundary-0.1.0-py3-none-any.whl", wheel_sha256="w" * 64)
    monkeypatch.setattr(tool, "setup_targets", setup)

    def repeat(case, role, *, folder, timeout, state=None, **kwargs):
        assert role == "package" and state is not None
        calls["repeat"].append(case)
        folder.mkdir(parents=True)
        for run in ("a", "b"):
            np.savez_compressed(folder / f"{run}.npz", **_arrays(scale))
            (folder / f"{run}.json").write_text(json.dumps(_report(case), indent=2), encoding="utf-8")
        return (dict(case=case, role=role, bit_identical=not differ, arrays=8, failed_keys=["x"] if differ else []),
                [_report(case), _report(case)])
    monkeypatch.setattr(tool, "_repeat_pair", repeat)
    return calls


def test_baseline_capture_writes_only_a_new_staging_folder(tmp_path, monkeypatch, capsys):
    tool, output, _ = _project(tmp_path, monkeypatch)
    calls = _fake_runs(tool, monkeypatch, tmp_path)
    # The synthetic project's C33/F33 are entry-free (pending); the real registry activated them.
    criteria = tool.criteria_tools()
    monkeypatch.setattr(criteria, "PENDING_ANALYTIC", {"C33": "two_phase", "F33": "two_phase"})
    monkeypatch.setattr(tool, "criteria_tools", lambda: criteria)
    before = _snapshot(output / "reference")
    assert tool.baseline(["C33", "F33"], now=NOW) == 0
    assert calls["setup"] == 1 and calls["repeat"] == ["C33", "F33"]
    assert _snapshot(output / "reference") == before                 # nothing installed by the capture
    folder = output / "baseline-staging" / f"20261116T090000Z-{SHA[:7]}-1"
    record = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
    assert record["package_commit"] == SHA and record["clean_tree"] and record["fresh_targets"]
    assert set(record["cases"]) == {"C33", "F33"} and record["cases"]["C33"]["independent_checks"]["passed"]
    assert record["cases"]["C33"]["run_elapsed_seconds"] == [1.23, 1.23]
    fixture = json.loads((folder / "C33.json").read_text(encoding="utf-8"))
    assert "kawin_origin" not in fixture and "elapsed_seconds" not in fixture
    with np.load(folder / "C33.npz") as data:
        assert record["cases"]["C33"]["scientific_sha256"] == BASE.scientific_sha256(fixture, {k: data[k] for k in data.files})
    summary = json.loads((output / "baseline-summary.json").read_text(encoding="utf-8"))
    assert set(summary["preview"]["C33"]["analytic"]) == {"interface", "eta", "profile_X", "profile_Y", "k"}
    assert record["wheel"] == dict(name="moving_boundary-0.1.0-py3-none-any.whl", sha256="w" * 64)
    assert len(record["target_tree_sha256"]) == 64
    budget = record["cases"]["C33"]["residual_budget_diagnostic"]
    assert budget["gate"] is False and budget["allowance"][0] == pytest.approx(1e-11 * 0.174)
    text = (folder / "capture.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text and str(tmp_path).replace("\\", "/") not in text   # no absolute paths


def test_residual_budget_diagnostic_follows_the_planned_formula():
    inputs = dict(residual_tolerance=1e-11, min_composition=1e-10, domain_length=0.6,
                  explicit_composition_scale=None, inventory_record_interval=1)
    inventory = np.array([[0.174, 0.051], [0.174 + 2e-12, 0.051 - 1e-12], [0.174 + 1e-12, 0.051 - 3e-12]])
    result = BASE.residual_budget_diagnostic({"inventoryData/values": inventory}, inputs)
    expected = 1e-11 * np.abs(inventory[:-1]).sum(axis=0)          # step-start inventories, floor 6e-11 inactive
    np.testing.assert_allclose(result["allowance"], expected, rtol=1e-15)
    np.testing.assert_allclose(result["observed_max_drift"], [2e-12, 3e-12], rtol=0, atol=1e-16)    # round-off of 0.174 + 2e-12
    np.testing.assert_allclose(result["observed_final_drift"], [1e-12, 3e-12], rtol=0, atol=1e-16)
    assert result["steps"] == 2 and result["gate"] is False and result["units"].startswith("m ")
    assert "not a rigorous conservation bound and not a correctness gate" in result["limitations"][0]
    legacy = BASE.residual_budget_diagnostic({"inventoryData/values": inventory}, dict(inputs, explicit_composition_scale=[1.0, 1.0]))
    np.testing.assert_allclose(legacy["allowance"], [2 * 1e-11 * 0.6] * 2, rtol=1e-15)   # R * s_k = 0.6 dominates
    dilute = BASE.residual_budget_diagnostic({"inventoryData/values": inventory * 1e-12}, inputs)
    np.testing.assert_allclose(dilute["allowance"], [2 * 1e-11 * 0.6 * 1e-10] * 2, rtol=1e-15)   # the R*minComposition floor
    with pytest.raises(BASE.BaselineError, match="not recorded every step"):
        BASE.residual_budget_diagnostic({"inventoryData/values": inventory}, dict(inputs, inventory_record_interval=5))
    with pytest.raises(BASE.BaselineError, match="cannot be computed as specified"):
        BASE.residual_budget_diagnostic({"inventoryData/values": inventory}, None)


@pytest.mark.parametrize("problem", ["dirty", "differ", "bounds", "registered"])
def test_baseline_capture_writes_nothing_when_a_gate_fails(tmp_path, monkeypatch, problem):
    tool, output, path = _project(tmp_path, monkeypatch, package_entry=problem == "registered")
    calls = _fake_runs(tool, monkeypatch, tmp_path, scale=60.0 if problem == "bounds" else 1.0, differ=problem == "differ")
    if problem == "dirty":
        def dirty():
            raise AssertionError("approve needs a clean package tree")
        monkeypatch.setattr(tool, "repository_checks", dirty)
    before = _snapshot(output)
    if problem in {"dirty", "registered"}:
        expected = AssertionError if problem == "dirty" else tool.baseline_tools().BaselineError
        with pytest.raises(expected, match="clean package tree" if problem == "dirty" else "already has an expected_artifacts entry"):
            tool.baseline(["C33"], now=NOW)
        assert calls["setup"] == 0 and calls["repeat"] == []            # refused before building or running
    else:
        assert tool.baseline(["C33"], now=NOW) == 1
    staging = output / "baseline-staging"
    assert not staging.exists() or not any(staging.iterdir())
    assert {k: v for k, v in _snapshot(output).items() if not k.startswith("baseline-runs-")} == before


def test_historical_actions_skip_package_baselines(tmp_path, monkeypatch):
    tool, _, path = _project(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    skipped = {}
    assert tool.drop_package_baselines(manifest, ["C01", "C33", "F33", "LEGACY"], skipped) == ["C01", "LEGACY"]
    assert set(skipped) == {"C33", "F33"} and all("no research reference role" in v for v in skipped.values())


@pytest.mark.parametrize("argv", [["baseline"], ["baseline", "--cases"], ["quick", "--verify"],
                                  ["baseline", "--cases", "C33", "--record"],
                                  ["baseline", "historical", "--cases", "C33"]])
def test_cli_rejects_malformed_baseline_requests(argv):
    tool = _load("baseline_cli", PROJECT / "tools/acceptance.py")
    with pytest.raises(SystemExit):
        tool.main(argv)


def test_option_s_package_origin_baseline_keeps_selection_rollback_and_validation(tmp_path, monkeypatch):
    """Package-origin files live in reference/; approvals and rollback behave exactly as for research cases."""
    tool, output, path = _project(tmp_path, monkeypatch, package_entry=True)
    approval = tool.approval_tools()
    research_before = {k: v for k, v in _snapshot(output / "reference").items() if k.startswith("C01")}
    selected = tool.expected_reference("C33")
    assert selected["source"] == "reference" and selected["npz"] == output / "reference/C33.npz"
    tool.validate_expected(["C01", "C33"])
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    np.savez_compressed(candidate / "C33.npz", **_arrays(1.01))
    (candidate / "C33.json").write_bytes(BASE.fixture_json_bytes(BASE.fixture_report(_report())))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    approval_id = approval.new_id(manifest, project=tmp_path, output=output, package_commit=SHA, now=NOW)
    approval.promote(project=tmp_path, output=output, manifest_path=path, approval_id=approval_id,
                     record=dict(created_utc="2026-11-16T09:00:00Z", package_commit=SHA, reason="LEDGERS test",
                                 tier="quick", cases=["C33"]),
                     files={"C33": dict(json=candidate / "C33.json", npz=candidate / "C33.npz")},
                     report=dict(schema=1, kind="approval"))
    assert tool.expected_reference("C33")["source"] == approval_id           # approval takes precedence
    assert json.loads(path.read_text(encoding="utf-8"))["approved_references"]["approvals"][approval_id]["previous"] == {"C33": "reference"}
    tool.validate_expected(["C01", "C33"])
    original = (output / "reference/C33.npz").read_bytes()
    (output / "reference/C33.npz").write_bytes(original + b"\0")
    with pytest.raises(AssertionError, match="expected artifact changed"):
        tool.validate_expected(["C33"])                                      # still validated while approved
    (output / "reference/C33.npz").write_bytes(original)
    approval.rollback(project=tmp_path, output=output, manifest_path=path, approval_id=approval_id,
                      reason="LEDGERS test", utc="2026-11-16T10:00:00Z")
    selected = tool.expected_reference("C33")
    assert selected["source"] == "reference" and selected["npz"] == output / "reference/C33.npz"
    tool.validate_expected(["C01", "C33"])
    assert tool.expected_reference("C01")["source"] == "reference"
    assert {k: v for k, v in _snapshot(output / "reference").items() if k.startswith("C01")} == research_before
