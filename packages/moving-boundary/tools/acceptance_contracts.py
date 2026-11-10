"""Approved lifecycle/failure contracts, with isolated main/package snapshots."""
import numpy as np
import pytest
import inspect


def run_contract(inputs, case_id, collect):
    """Execute existing contract assertions and return all resulting accepted states."""
    tests = inputs.root / "kawin/tests"
    inputs.track_models = True
    completed = []
    if case_id in {"C25", "C26", "C27"}:
        kind = {"C25":"binary", "C26":"ternary", "C27":"three_phase"}[case_id]
        names = ["test_setup_is_idempotent", "test_explicit_setup_then_solve_matches_solve",
                 "test_continued_solve_keeps_reference_inventory_and_history"]
        m = inputs.load(tests / "test_illingworth_setup_and_diagnostics.py", names)
        for name in names:
            getattr(m, name)(kind)
            completed.append(name)
        # A reset is an explicit reinitialization, not a continuation.
        last = inputs.models[-1]
        last.reset()
        last.setup()
    else:
        if case_id == "C28":
            selections = [("test_illingworth_ternary_three_phase.py", [
                "test_three_phase_final_remainder_below_outer_minimum_is_solved_and_accepted",
                "test_three_phase_tiny_final_remainder_commits_all_histories_and_allows_continuation"])]
        else:
            selections = [("test_illingworth_binary_variable_diffusivity.py", [
                "test_failed_step_is_retried_with_smaller_dt", "test_retry_exhaustion_reraises_last_error"]),
                ("test_illingworth_ternary_three_phase.py", [
                "test_three_phase_failed_retries_do_not_mutate_accepted_boundary_or_histories",
                "test_three_phase_retry_below_outer_minimum_is_rejected_before_wrapper"])]
        for file, names in selections:
            inputs.family = "binary" if "binary" in file else "three_phase"
            m = inputs.load(tests / file, names)
            for name in names:
                with pytest.MonkeyPatch.context() as patch:
                    function = getattr(m, name)
                    function(patch) if "monkeypatch" in inspect.signature(function).parameters else function()
                completed.append(name)
    arrays = {}
    for i, model in enumerate(inputs.models):
        for key, value in collect(model).items():
            arrays[f"model{i}/{key}"] = value
    arrays["contracts/assertion_groups"] = np.array(len(completed), dtype=int)
    return arrays, completed


def model_scales(models):
    """Per-model tolerance scales for the ``model{i}/`` array prefixes (item 7).

    Mirrors what the worker records for run cases: ``length_scale`` is the physical mesh span
    (``ptp(mesh.z)``) and ``minimum_molar_volume`` is ``min(_phaseMolarVolumes)``, present only
    for models with explicit phase molar volumes (``inventory_scale`` then defaults to 1).
    Measured after the contract assertions, from the same tracked models whose states are saved.
    """
    scales = {}
    for i, model in enumerate(models):
        entry = {"length_scale": float(np.ptp(np.ravel(model.mesh.z)))}
        volumes = getattr(model, "_phaseMolarVolumes", None)
        if volumes is not None:
            entry["minimum_molar_volume"] = float(np.min(volumes))
        scales[f"model{i}"] = entry
    return scales
