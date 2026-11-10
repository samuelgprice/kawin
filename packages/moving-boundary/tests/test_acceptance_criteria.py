"""M7: approval correctness criteria never pass when a measurement cannot be computed."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT / ".verification/acceptance/reference"


def _tool():
    spec = importlib.util.spec_from_file_location("criteria_tool", PROJECT / "tools/acceptance_criteria.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _criteria():
    return json.loads((PROJECT / "tools/acceptance_manifest.json").read_text(encoding="utf-8"))["approval_criteria"]


@pytest.mark.parametrize("value, factor, floor, expected", [
    (1.702e-3, 2, 0.0, 0.0035), (7.035e-3, 2, 0.0, 0.015), (2.771e-3, 2, 0.0, 0.0056),
    (4.5e-9, 10, 1e-13, 4.5e-8), (1.17e-9, 10, 1e-13, 1.2e-8), (1.1e-16, 10, 1e-13, 1e-13), (0.0, 10, 1e-13, 1e-13)])
def test_limits_round_up_to_two_significant_figures(value, factor, floor, expected):
    limit = _tool().round_up(value, factor, floor)
    assert limit == pytest.approx(expected, rel=1e-12) and limit >= factor * value


@pytest.mark.parametrize("arrays, spec, message", [
    ({}, dict(metric="max_abs", arrays=["exact/interface_error"]), "array missing"),
    ({"a": np.array([])}, dict(metric="max_abs", arrays=["a"]), "empty array"),
    ({"a": np.array([0.0, np.nan])}, dict(metric="max_abs", arrays=["a"]), "non-finite"),
    ({"a": np.array([0.0, np.inf])}, dict(metric="max_abs", arrays=["a"]), "non-finite"),
    ({"a": np.array([1.0])}, dict(metric="relative_drift", arrays=["a"]), "fewer than two history points"),
    ({"a": np.array([0.0, 1.0])}, dict(metric="relative_drift", arrays=["a"]), "zero initial magnitude"),
    ({"e": np.zeros(3), "t": np.array([0.0, 1.0, 1.0])}, dict(metric="closure_relative", arrays=["e", "t"]), "zero initial total"),
    ({"p": np.zeros(5)}, dict(metric="max_abs_outside_nodes", arrays=["p"], nodes=6, excluded_nodes=[1, 2]), "frozen mask expects 6"),
    ({"p": np.zeros(5)}, dict(metric="max_abs_outside_nodes", arrays=["p"], nodes=5, excluded_nodes=[0, 4]), "empty profile mask"),
    ({"p": np.array([0.0, np.nan, 0.0])}, dict(metric="max_abs_outside_nodes", arrays=["p"], nodes=3, excluded_nodes=[1, 1]), "non-finite"),
    ({"a": np.zeros(2)}, dict(metric="unknown", arrays=["a"]), "unknown metric")])
def test_unmeasurable_criteria_raise(arrays, spec, message):
    tool = _tool()
    with pytest.raises(tool.CriteriaError, match=message):
        tool.measure(arrays, spec)


def test_masked_profile_and_drift_values():
    tool = _tool()
    spec = dict(metric="max_abs_outside_nodes", arrays=["p"], nodes=6, excluded_nodes=[2, 3])
    assert tool.measure({"p": np.array([0.1, -0.2, 9.0, -9.0, 0.3, 0.0])}, spec) == 0.3  # excluded jump nodes ignored
    drift = dict(metric="relative_drift", arrays=["q"])
    assert tool.measure({"q": np.array([[2.0, 4.0], [2.0, 4.0 + 1e-9], [2.0 - 2e-9, 4.0]])}, drift) == pytest.approx(5e-10)
    closure = dict(metric="closure_relative", arrays=["e", "t"])
    assert tool.measure({"e": np.array([0.0, -2e-16]), "t": np.array([0.5, 0.5])}, closure) == pytest.approx(4e-16)


def test_evaluate_reports_unmeasurable_and_exceeded_criteria_as_failures():
    tool = _tool()
    criteria = dict(analytic={"C99": {"interface": dict(metric="max_abs", arrays=["i"], limit=1e-3)}},
                    conservation={"C99": {"inventory": dict(metric="relative_drift", arrays=["q"], limit=1e-12)}})
    results = tool.evaluate("C99", {"i": np.array([1e-3]), "q": np.array([1.0, np.nan])}, criteria)
    assert [(r["criterion"], r["passed"]) for r in results] == [("analytic/interface", True), ("conservation/inventory", False)]
    assert "non-finite" in results[1]["error"] and results[1]["value"] is None
    assert not tool.evaluate("C99", {"i": np.array([2e-3]), "q": np.ones(2)}, criteria)[0]["passed"]


def test_committed_criteria_follow_the_approved_rules():
    tool = _tool()
    criteria = _criteria()
    assert set(criteria["analytic"]) == {"C04", "C31", "F04", "C11", "C12", "F11", "F12"}
    assert not set(criteria["conservation"]) & {"C25", "C26", "C27", "C28", "C29", "O08"}
    for case, specs in criteria["analytic"].items():
        for name, spec in specs.items():
            assert spec["limit"] == tool.round_up(spec["current"], 2), (case, name)
        if case in {"C04", "C31", "F04"}:
            profile = specs["profile"]
            first, last = profile["excluded_nodes"]
            assert profile["metric"] == "max_abs_outside_nodes" and profile["nodes"] == 401 and 0 < first <= last < 400
            assert profile["mask"]["half_width"] == pytest.approx(specs["interface"]["limit"] + 2 * profile["mask"]["h"])
    for case, specs in criteria["conservation"].items():
        for name, spec in specs.items():
            assert spec["limit"] == tool.round_up(spec["current"], 10, 1e-13) and spec["limit"] >= 1e-13, (case, name)
    families = {family: cases for family, cases in (("binary", tool.BINARY), ("ternary", tool.TERNARY),
                ("three_phase", tool.THREE_PHASE_IMPLICIT_EQUAL + tool.THREE_PHASE_UNEQUAL + tool.THREE_PHASE_EXPLICIT_EQUAL))}
    for family, cases in families.items():
        assert set(cases) <= set(criteria["conservation"]), family
    for case in tool.THREE_PHASE_UNEQUAL:
        assert "composition_inventory" not in criteria["conservation"][case] and case in criteria["reported_only"]


@pytest.mark.skipif(not (REFERENCE / "C04.npz").exists(), reason="local frozen references absent")
def test_committed_criteria_equal_a_fresh_proposal_from_the_frozen_references():
    """The stored record is reproducible: one implementation measures and proposes."""
    tool = _tool()
    manifest = json.loads((PROJECT / "tools/acceptance_manifest.json").read_text(encoding="utf-8"))
    proposal = json.loads(json.dumps(tool.propose(manifest["expected_artifacts"]["cases"])))
    for key in ("analytic", "conservation", "reported_only"):
        assert proposal[key] == manifest["approval_criteria"][key], key
    for case in sorted(set(manifest["approval_criteria"]["analytic"]) | set(manifest["approval_criteria"]["conservation"])):
        _, arrays = tool.load_reference(case, manifest["expected_artifacts"]["cases"])
        assert all(result["passed"] for result in tool.evaluate(case, arrays, manifest["approval_criteria"])), case
