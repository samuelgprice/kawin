"""Opt-in harness tests with synthetic clocks/data; never run solver benchmarks.

Invoke explicitly with ``python -m pytest tools/performance_checks.py``. The
filename deliberately stays outside normal pytest discovery and acceptance.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import performance
import performance_results as results
import performance_worker as worker


def saved(times=(1., 1., 1.)):
    """Make a minimal saved result independent of this computer's speed."""
    return dict(schema=1, status="ok", environment=dict(machine="synthetic", versions=dict(numpy="2")),
                measurement=dict(protocol="test"),
                cases={"case": dict(status="ok", configuration=dict(nodes=21),
                                    metrics=dict(solve=results.summarize(times)),
                                    work=[dict(accepted_steps=3)]*len(times))})


@pytest.mark.parametrize("samples", [[], [0.], [-1.], [float("nan")], [float("inf")]])
def test_summary_rejects_invalid_samples(samples):
    with pytest.raises(ValueError):
        results.summarize(samples)


def test_summary_retains_samples_and_robust_spread():
    assert results.summarize([1., 2., 3.]) == dict(samples=[1., 2., 3.], median=2.,
                                                min=1., max=3., mad=1., relative_mad=.5)


def test_ratio_review_and_explicit_threshold():
    a, b = saved(), saved([1.5]*3)
    row = results.compare(a, b)["cases"][0]
    assert row["metrics"]["solve"]["ratio"] == 1.5
    assert row["metrics"]["solve"]["status"] == "review"
    assert "REVIEW" in results.format_comparison(results.compare(a, b))
    assert results.compare(a, b, slowdown=.6)["cases"][0]["metrics"]["solve"]["status"] == "observed"


def test_noise_and_absolute_guard_suppress_small_flags():
    noisy = results.compare(saved([.5, 1., 1.5]), saved([1.3]*3))
    assert noisy["cases"][0]["metrics"]["solve"]["status"] == "observed"
    tiny = results.compare(saved([.0001]*3), saved([.0005]*3))
    assert tiny["cases"][0]["metrics"]["solve"]["status"] == "observed"


@pytest.mark.parametrize("option,value", [("slowdown", -1), ("noise_multiplier", float("nan")),
                                         ("minimum_seconds", float("inf"))])
def test_threshold_validation(option, value):
    with pytest.raises(ValueError):
        results.compare(saved(), saved(), **{option: value})


def test_environment_differences_are_explicit_and_confounded():
    b = saved()
    b["environment"]["versions"]["numpy"] = "3"
    report = results.compare(saved(), b)
    assert report["environment_differences"] == ["versions.numpy"]
    assert report["cases"][0]["status"] == "environment_differs"


@pytest.mark.parametrize("group", ["configuration", "measurement"])
def test_incompatible_configuration_or_protocol_has_no_ratio(group):
    b = saved()
    (b["cases"]["case"][group] if group == "configuration" else b[group])["changed"] = True
    row = results.compare(saved(), b)["cases"][0]
    assert row["status"] == "incompatible"
    assert not row["metrics"]


def test_missing_cases_and_metrics_are_reported():
    b = saved()
    b["cases"]["new"] = deepcopy(b["cases"]["case"])
    b["cases"]["case"]["metrics"] = {}
    report = results.compare(saved(), b)
    assert report["cases"][0]["metrics"]["solve"]["status"] == "missing_metric"
    assert report["cases"][1]["status"] == "missing_baseline"
    assert results.compare(b, saved())["cases"][1]["status"] == "missing_candidate"


def test_failed_run_has_no_speed_claim():
    b = saved()
    b["cases"]["case"]["status"] = "failed"
    row = results.compare(saved(), b)["cases"][0]
    assert row["status"] == "invalid_run" and not row["metrics"]


def test_incomplete_suite_has_no_speed_claim():
    b = saved()
    b["status"] = "running"
    row = results.compare(saved(), b)["cases"][0]
    assert row["status"] == "invalid_run" and not row["metrics"]


def test_unknown_schema_is_rejected():
    b = saved()
    b["schema"] = 999
    with pytest.raises(ValueError):
        results.compare(saved(), b)


def test_work_changes_and_unequal_sample_counts():
    assert not results.compare(saved(), saved([1., 1.]))["cases"][0]["work_changed"]
    b = saved()
    b["cases"]["case"]["work"][0] = dict(accepted_steps=4)
    assert results.compare(saved(), b)["cases"][0]["work_changed"]


class FakeModel:
    """Two accepted synthetic steps with a first, observable finalization."""

    def __init__(self):
        self.currentTime = 0.
        self.constraints = SimpleNamespace(minComposition=1e-10)
        self._lastImplicitIterations = 2
        self._lastStepRetries = 1
        self.events = []

    def setup(self):
        """Record setup without advancing state."""
        self.events.append("setup")

    def solve(self, duration, **kwargs):
        """Advance through the wrapped commit and first finalization APIs."""
        self.setup()
        self.events.append("solve")
        self.postProcess(duration/2, None)
        self.postProcess(duration, None)
        self.postSolve()

    def postProcess(self, time, state):
        """Commit a synthetic accepted state."""
        self.currentTime = time
        return state, False

    def postSolve(self):
        """Record the first actual finalization."""
        self.events.append("finalize")

    def reset(self):
        """Restore the initial time."""
        self.currentTime = 0.
        self.events.append("reset")


def test_harness_times_boundaries_and_counts_commits():
    model = FakeModel()
    ticks = iter(range(100))
    case = dict(dt=.5, steps=2)
    def validate(model, case):
        assert model.currentTime == 1.
        model.events.append("validate")
        return dict(valid=True)
    metrics, work, valid, settings = worker.measure_once(case, lambda c: model, validate, clock=lambda: next(ticks))
    assert metrics == dict(setup=1, solve=5, finalize=1, reset=1)
    assert work["accepted_steps"] == 2 and work["_lastStepRetries"] == 2
    assert work["_lastImplicitIterations"] == 4
    assert work["_lastImplicitCandidateEvaluations"] is None
    assert model.events == ["setup", "solve", "finalize", "validate", "reset"]
    assert valid == dict(valid=True) and settings["constraints"]["minComposition"] == 1e-10


def test_validation_failure_propagates_and_is_not_timed_as_success():
    def invalid(*args):
        raise ValueError("Invalid numerical result")
    with pytest.raises(ValueError, match="Invalid numerical"):
        worker.measure_once(dict(dt=.5, steps=2), lambda c: FakeModel(), invalid)


def test_final_noop_does_not_duplicate_diagnostic_counts():
    class NoopModel(FakeModel):
        """Add a clock-only final remainder after two real accepted steps."""

        def solve(self, duration, **kwargs):
            """Advance time through one final no-op that consumes its flag."""
            self.setup()
            self.postProcess(duration/3, None)
            self.postProcess(2*duration/3, None)
            self._nearFinalNoop = True
            self.postProcess(duration, None)
            self.postSolve()

        def postProcess(self, time, state):
            """Consume the no-op flag as the real two-phase solvers do."""
            self._nearFinalNoop = False
            return super().postProcess(time, state)

    _, work, _, _ = worker.measure_once(dict(dt=.5, steps=2), lambda c: NoopModel(), lambda *a: {})
    assert work["accepted_steps"] == 2 and work["final_noop_commits"] == 1
    assert work["_lastImplicitIterations"] == 4


def test_worker_warmups_fresh_instances_and_failure_checkpoint(tmp_path, monkeypatch):
    models = []
    def build(case):
        model = FakeModel()
        models.append(model)
        return model
    def validate(model, case):
        if case["id"] == "bad":
            raise ValueError("invalid output")
        return dict(valid=True)
    monkeypatch.setitem(sys.modules, "performance_cases", SimpleNamespace(build_model=build, validate_model=validate))
    monkeypatch.setattr(worker, "environment", lambda *args: {})
    monkeypatch.setattr("builtins.input", input)  # restore the worker's prompt guard after this test
    path = tmp_path / "request.json"
    path.write_text(json.dumps(dict(target=str(tmp_path), host=str(tmp_path), identity={}, measurement={},
                                   warmups=1, samples=2, cases=[dict(id="good", dt=.5, steps=2),
                                                              dict(id="bad", dt=.5, steps=2)])), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["performance_worker.py", str(path)])
    assert worker.main() == 1
    report = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert len(models) == 4 and len({id(model) for model in models}) == 4
    assert len(report["cases"]["good"]["metrics"]["solve"]["samples"]) == 2
    assert report["cases"]["bad"]["status"] == "failed"
    assert "invalid output" in report["cases"]["bad"]["error"]


def test_unique_directories_and_compare_never_overwrites_baseline(tmp_path):
    a = performance.new_folder(tmp_path)
    b = performance.new_folder(tmp_path)
    assert a != b and a.is_dir() and b.is_dir()
    baseline, candidate = a / "results.json", b / "results.json"
    baseline.write_text(json.dumps(saved()), encoding="utf-8")
    candidate.write_text(json.dumps(saved([2.]*3)), encoding="utf-8")
    original = baseline.read_bytes()
    assert performance.main(["compare", str(baseline), str(candidate), "--output", str(tmp_path / "reports")]) == 0
    assert baseline.read_bytes() == original
    assert len(list((tmp_path / "reports").glob("*/comparison.json"))) == 1


def test_git_optional_in_export(tmp_path, monkeypatch):
    monkeypatch.setattr(performance, "PROJECT", tmp_path / "export/packages/moving-boundary")
    assert performance.git_identity() == dict(available=False, reason="No Git metadata (export)")


def test_effective_nonfinite_control_is_valid_json():
    model = FakeModel()
    model.constraints.unbounded = float("inf")
    assert json.loads(json.dumps(worker.effective_settings(model), allow_nan=False))["constraints"]["unbounded"] == "inf"
