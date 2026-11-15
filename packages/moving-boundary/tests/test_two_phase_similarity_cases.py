"""C33/F33 (package-owned two-phase ternary analytic cases): definition, metrics, criteria, registration.

Activated state (Stage 3): C33 runs in ``quick`` and F33 in ``full_drift``; both have package-captured
``expected_artifacts`` entries (capture 20261115T224052Z-a26d099-1) and ``approval_criteria`` equal to
the frozen Stage 0 limits; the criteria registry lists them as kind ``two_phase``.
"""
import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((PROJECT / "tools/acceptance_manifest.json").read_text(encoding="utf-8"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASES = _load("two_phase_similarity_cases", PROJECT / "tools/two_phase_similarity_cases.py")


@pytest.fixture(scope="module")
def exact():
    return CASES.solution(CASES.DirectInputs())


class _History:
    def __init__(self, times, values):
        self._time, self._y, self.N = np.asarray(times, float), np.asarray(values, float), len(times) - 1


def _exact_stub(exact, *, eta_offset=0.0, profile_offset=0.0):
    """A solved-model stand-in holding the exact state (optionally offset) on a C33-like grid."""
    tm = np.concatenate(([0.0], np.geomspace(1e-4, 0.3, 40)))
    s = exact.interface_position(tm + CASES.T0)
    t_end = float(tm[-1]) + CASES.T0
    x_left = np.linspace(0.0, float(s[-1]), 31)
    x_right = np.linspace(float(s[-1]), CASES.DOMAIN_LENGTH, 31)
    left = exact.profile(x_left, t_end, phase="left") + profile_offset
    right = exact.profile(x_right, t_end, phase="right")
    inventory = np.tile([0.174, 0.051], (len(tm), 1))
    return SimpleNamespace(
        interfaceData=_History(tm, s), etaData=_History(tm, np.full_like(tm, exact.eta + eta_offset)),
        inventoryData=_History(tm, inventory), currentTime=float(tm[-1]),
        getCompositions=lambda: np.vstack([left, right]),
        getPhysicalPhaseProfiles=lambda time=None: ((x_left, left), (x_right, right)),
        getTransformedState=lambda time=None: (left, right))


def test_frozen_configuration_matches_the_manifest_registration():
    entries = MANIFEST["package_baselines"]["cases"]
    assert set(entries) == {"C33", "F33"}
    for case, entry in entries.items():
        config = dict(entry["configuration"])
        assert config.pop("end_time") == CASES.END_TIMES[case]
        frozen = {k: v for k, v in CASES.FROZEN.items() if k != "end_times"}
        assert config == frozen, case
        assert entry["source"]["definition"] == "tools/two_phase_similarity_cases.py"
        assert entry["independent_bounds"] == dict(eta_max_abs=5e-3, profile_max_abs=1e-3, k_rel_abs=1e-2,
                                                   interface_max_abs_over_displacement=1e-2, final_inventory_drift_abs=1e-9)
    assert entries["C33"]["frozen_proposed_limits"] == dict(interface=8.7e-6, eta=9.3e-4, profile_X=2.5e-4, profile_Y=2.1e-4,
                                                            k=1.5e-3, composition_inventory=1.3e-9)
    assert entries["F33"]["frozen_proposed_limits"] == dict(interface=1.5e-5, eta=9.3e-4, profile_X=2.6e-4, profile_Y=2.2e-4,
                                                            k=1.8e-3, composition_inventory=1.3e-9)
    test_constants = _load("analytic_tests", PROJECT / "tests/test_analytic.py")
    np.testing.assert_array_equal(CASES.D_LEFT, test_constants.D_LEFT)
    np.testing.assert_array_equal(CASES.D_RIGHT, test_constants.D_RIGHT)
    np.testing.assert_array_equal(CASES.C_LEFT_FAR, test_constants.C_LEFT_FAR)
    np.testing.assert_array_equal(CASES.C_RIGHT_FAR, test_constants.C_RIGHT_FAR)


CAPTURE_ID = "20261115T224052Z-a26d099-1"
CAPTURE_COMMIT = "a26d0991a3c378cb0cdb70149b577e0ba877529b"


def test_cases_are_activated_in_exactly_their_tiers_with_package_provenance():
    assert "C33" in MANIFEST["quick"] and "F33" not in MANIFEST["quick"]
    assert "F33" in MANIFEST["full_drift"] and "C33" not in MANIFEST["full_drift"]
    for section in ("contracts", "cached"):
        assert not {"C33", "F33"} & set(MANIFEST[section]), section
    entries = MANIFEST["package_baselines"]["cases"]
    for case in ("C33", "F33"):
        tier = entries[case]["tier_on_activation"]
        assert case in MANIFEST[tier]
        artifact = MANIFEST["expected_artifacts"]["cases"][case]
        assert artifact["captured_role"] == "package" and artifact["capture_id"] == CAPTURE_ID
        assert artifact["captured_by"] == artifact["capture_record"]["package_commit"] == CAPTURE_COMMIT
        assert len(artifact["scientific_sha256"]) == 64 and artifact["capture_record"]["repeat"]["bit_identical"]
        assert artifact["capture_record"]["independent_checks_passed"]
        assert set(artifact["capture_record"]["wheel"]) == {"name", "sha256"}
        adopted = {**{name: spec["limit"] for name, spec in MANIFEST["approval_criteria"]["analytic"][case].items()},
                   **{name: spec["limit"] for name, spec in MANIFEST["approval_criteria"]["conservation"][case].items()}}
        assert adopted == entries[case]["frozen_proposed_limits"], case
    # The original package baselines are in effect (no approval selects other files).
    assert not {"C33", "F33"} & set((MANIFEST.get("approved_references") or {}).get("in_effect", {}))
    tool = _load("acceptance_tool", PROJECT / "tools/acceptance.py")
    quick, _ = tool.normal_cases(MANIFEST, full=False)
    full, _ = tool.normal_cases(MANIFEST, full=True)
    assert "C33" in quick and "F33" not in quick and {"C33", "F33"} <= set(full)
    criteria = _load("acceptance_criteria", PROJECT / "tools/acceptance_criteria.py")
    assert criteria.ANALYTIC_CASES["C33"] == criteria.ANALYTIC_CASES["F33"] == "two_phase"
    assert criteria.PENDING_ANALYTIC == {}
    families = (set(criteria.BINARY) | set(criteria.TERNARY) | set(criteria.THREE_PHASE_IMPLICIT_EQUAL)
                | set(criteria.THREE_PHASE_UNEQUAL) | set(criteria.THREE_PHASE_EXPLICIT_EQUAL))
    assert not {"C33", "F33"} & families          # their conservation spec comes from the two_phase kind
    assert "C33" in MANIFEST["approval_criteria"]["not_covered"]["ternary_analytic"]


def test_worker_registers_the_cases_as_package_owned_ternary(monkeypatch):
    monkeypatch.setattr(builtins, "input", builtins.input)   # the worker installs an input guard on import
    worker = _load("acceptance_worker_module", PROJECT / "tools/acceptance_worker.py")
    assert {"C33", "F33"} <= worker.PACKAGE_DEFINITIONS and worker.PACKAGE_OWNED == {"C33", "F33"}
    assert worker.case_family("C33") == worker.case_family("F33") == "ternary"
    expected = {"C01": "binary", "F04": "binary", "LEGACY": "binary", "O07": "binary", "C05": "ternary",
                "C24": "ternary", "O04": "ternary", "C09": "three_phase", "C11": "three_phase",
                "F11": "three_phase", "F12": "three_phase", "C32": "three_phase", "O02": "three_phase"}
    assert {case: worker.case_family(case) for case in expected} == expected


def test_collect_records_zero_errors_for_the_exact_state(exact):
    cases_module = _load("acceptance_cases", PROJECT / "tools/acceptance_cases.py")
    arrays = cases_module.collect(_exact_stub(exact), ("two_phase", exact, CASES.T0))
    for key in ("exact/interface_error", "exact/eta_error", "exact/profile_error_X", "exact/profile_error_Y"):
        assert np.max(np.abs(arrays[key])) < 1e-13, key
    assert arrays["exact/profile_error_X"].shape == arrays["exact/profile_error_Y"].shape == (62,)
    assert arrays["exact/k_rel_error"].shape == (1,) and abs(arrays["exact/k_rel_error"][0]) < 1e-12
    assert len(arrays["exact/interface_error"]) == len(arrays["interfaceData/time"])


def test_collect_reports_offsets_per_component_and_phase(exact):
    arrays = CASES.exact_arrays(_exact_stub(exact, eta_offset=2e-4, profile_offset=np.array([3e-5, -7e-5])), exact,
                                CASES.T0, *(lambda s: (s.interfaceData._y, s.interfaceData._time, s.etaData._y))(
                                    _exact_stub(exact, eta_offset=2e-4)))
    np.testing.assert_allclose(arrays["exact/eta_error"], 2e-4, rtol=0, atol=1e-15)
    np.testing.assert_allclose(arrays["exact/profile_error_X"][:31], 3e-5, rtol=0, atol=1e-15)
    np.testing.assert_allclose(arrays["exact/profile_error_Y"][:31], -7e-5, rtol=0, atol=1e-15)
    assert np.max(np.abs(arrays["exact/profile_error_X"][31:])) < 1e-15   # right phase untouched


@pytest.mark.parametrize("make_inputs", ["direct", "bindings"])
def test_built_model_starts_from_the_setup_estimate_of_the_exact_profile(exact, make_inputs):
    """Public setup interpolates the exact T0 profile per phase, estimates eta and pins the tie line."""
    if make_inputs == "direct":
        inputs = CASES.DirectInputs()
    else:
        inputs = _load("definitions_bindings_test", PROJECT / "tools/definitions/_bindings.py").Context()
        inputs.family = "ternary"
    model, duration, analytic = CASES.build(inputs, "C33")
    assert duration == pytest.approx(0.3) and analytic[0] == "two_phase" and analytic[2] == CASES.T0
    if make_inputs == "bindings":
        assert inputs.providers == {"ternary": "moving_boundary.solvers.illingworth.ternary"}
    model.setup()
    analytic_module = CASES.DirectInputs().imported_module("moving_boundary.analytic")
    two_phase = CASES.DirectInputs().imported_module("moving_boundary.analytic._two_phase")
    assert two_phase.interface_resolved_start_time(model, exact, n_cells=5) <= CASES.T0   # T0 rule (Stage 0: 0.0747)
    for matrix, distance in ((CASES.D_LEFT, CASES.S0), (CASES.D_RIGHT, CASES.DOMAIN_LENGTH - CASES.S0)):
        assert analytic_module.semi_infinite_validity_time(matrix, distance) > max(CASES.END_TIMES.values())
    eta0 = float(model.etaData._y[0])
    delta = eta0 - exact.eta
    assert 1e-6 < abs(delta) < 1e-4          # the setup estimate, not the analytic eta (Stage 0: -5.5e-5)
    (x_left, _), (x_right, _) = model.getPhysicalPhaseProfiles(0.0)
    left, right = model.getTransformedState(0.0)
    err_left = np.asarray(left) - exact.profile(x_left, CASES.T0, phase="left")
    err_right = np.asarray(right) - exact.profile(x_right, CASES.T0, phase="right")
    slope_left = np.subtract(*CASES.CLOSURE["left"][::-1])
    slope_right = np.subtract(*CASES.CLOSURE["right"][::-1])
    np.testing.assert_allclose(err_left[-1], slope_left * delta, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(err_right[0], slope_right * delta, rtol=1e-9, atol=1e-15)
    interior = max(np.max(np.abs(err_left[:-1])), np.max(np.abs(err_right[1:])))
    assert interior < 1e-5                   # physical-mesh interpolation (Stage 0: 2.0e-6)


def _fixture(tmp_path, exact, case="C33"):
    arrays = CASES.exact_arrays(_exact_stub(exact, profile_offset=np.array([1e-4, -5e-5])), exact, CASES.T0,
                                *(lambda s: (s.interfaceData._y, s.interfaceData._time, s.etaData._y))(_exact_stub(exact)))
    arrays["exact/interface_error"] = arrays["exact/interface_error"] + 3e-6
    arrays["exact/eta_error"] = arrays["exact/eta_error"] - 4e-4
    arrays["exact/k_rel_error"] = np.array([7e-4])
    arrays["inventoryData/values"] = np.array([[0.174, 0.051], [0.174 + 2e-11, 0.051], [0.174 + 1e-11, 0.051 - 5e-12]])
    np.savez(tmp_path / f"{case}.npz", **arrays)
    (tmp_path / f"{case}.json").write_text(json.dumps(dict(case=case, status="PASS")), encoding="utf-8")
    return arrays


def test_preview_applies_the_existing_rules_with_explicit_labels(tmp_path, exact, monkeypatch):
    """Preview serves pending cases only: a synthetic pending two-phase case C99; activated C33 is refused."""
    criteria = _load("acceptance_criteria_preview", PROJECT / "tools/acceptance_criteria.py")
    monkeypatch.setattr(criteria, "PENDING_ANALYTIC", {"C99": "two_phase"})
    _fixture(tmp_path, exact, case="C99")
    _fixture(tmp_path, exact, case="C33")
    before = sorted(p.name for p in tmp_path.iterdir())
    result = criteria.preview("C99", tmp_path / "C99.json", tmp_path / "C99.npz")
    assert sorted(p.name for p in tmp_path.iterdir()) == before          # writes nothing
    analytic, conservation = result["analytic"], result["conservation"]
    assert set(analytic) == {"interface", "eta", "profile_X", "profile_Y", "k"}
    for name, spec in analytic.items():
        assert spec["metric"] == "max_abs" and spec["limit"] == criteria.round_up(spec["current"], 2), name
    assert analytic["profile_X"]["current"] == pytest.approx(1e-4) and analytic["profile_Y"]["current"] == pytest.approx(5e-5)
    assert analytic["k"]["current"] == pytest.approx(7e-4) and analytic["eta"]["current"] == pytest.approx(4e-4)
    inventory = conservation["composition_inventory"]
    assert inventory["current"] == pytest.approx(2e-11 / 0.174)
    assert inventory["limit"] == criteria.round_up(inventory["current"], 10, 1e-13)
    assert "relative normalized drift" in inventory["quantity"] and "absolute source bound" in inventory["quantity"]
    for registered in ("C11", "C33"):          # registered (activated) cases use propose
        with pytest.raises(criteria.CriteriaError, match="not a pending"):
            criteria.preview(registered, tmp_path / "C33.json", tmp_path / "C33.npz")
    assert criteria.main(["preview", "--case", "C99", "--json", str(tmp_path / "C99.json"),
                          "--npz", str(tmp_path / "C99.npz")]) == 0
    with pytest.raises(SystemExit):
        criteria.main(["preview", "--case", "C99"])


def test_registered_kinds_keep_their_specs():
    """The proposal_for refactor leaves every registered case's criterion definitions unchanged."""
    criteria = _load("acceptance_criteria_specs", PROJECT / "tools/acceptance_criteria.py")
    stored = MANIFEST["approval_criteria"]
    for case in sorted(set(stored["analytic"]) | set(stored["conservation"])):
        analytic, conservation = criteria._specs(case, {})
        for kind, specs in (("analytic", analytic), ("conservation", conservation)):
            for name, spec in specs.items():
                recorded = stored[kind][case][name]
                assert {k: recorded[k] for k in spec} == spec, (case, name)
            assert set(specs) == set(stored[kind].get(case, {})), (case, kind)
