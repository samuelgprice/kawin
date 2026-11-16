"""Reusable example capabilities, checked independently of example wrappers."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from moving_boundary.providers.constant import ConstantDiffusivity
from moving_boundary.run_diagnostics import evaluate_bulk_coverage, inventory_drift
from moving_boundary.sampling import ternary_simplex_points
from moving_boundary.surrogate_artifacts import (
    SurrogateArtifactCompatibilityError,
    SurrogateArtifactIntegrityError,
    load_or_build_surrogate_artifact,
    load_or_build_diagnostic_surrogate,
    write_surrogate_diagnostic_members,
)
from test_surrogate_artifacts import _surrogate


def test_constant_diffusivity_preserves_temperature_shapes_and_ignores_composition():
    provider = ConstantDiffusivity({"A": 2.0, "B": 3.0})
    assert provider.getInterdiffusivity([0.1], 800.0, phase="A") == 2.0
    assert np.ndim(provider.getInterdiffusivity([0.9], [900.0], phase="B")) == 0
    np.testing.assert_array_equal(
        provider.getInterdiffusivity([0.1, 0.2], [[800.0, 900.0]], phase="B"), [3.0, 3.0])
    provider.clearCache()
    with pytest.raises(ValueError, match="Unknown phase"):
        provider.getInterdiffusivity(0.1, 800, phase="C")


@pytest.mark.parametrize("values", [{}, {"A": 0.0}, {"A": -1.0}, {"A": np.nan}, {"A": np.inf}])
def test_constant_diffusivity_rejects_invalid_values(values):
    with pytest.raises(ValueError, match="positive"):
        ConstantDiffusivity(values)


def test_simplex_grid_preserves_order_projection_and_symmetry():
    points = ternary_simplex_points(0.1)
    assert points.shape == (66, 2)
    full = np.column_stack((points, 1.0 - points.sum(axis=1)))
    assert np.all(full >= 1e-5 - 1e-12)
    # Projection shifts edge coordinates; preserve the underlying grid order.
    assert np.all(np.diff(ternary_simplex_points(0.1, floor=0)[:, 0]) >= 0)
    np.testing.assert_allclose(full[1], [1e-5, 0.1 - 0.5e-5, 0.9 - 0.5e-5], rtol=0, atol=1e-12)
    rows = {tuple(row) for row in np.round(full, 9)}
    for order in ((1, 0, 2), (2, 1, 0), (0, 2, 1)):
        assert {tuple(row) for row in np.round(full[:, order], 9)} == rows


@pytest.mark.parametrize("spacing,floor", [(0, 0), (-1, 0), (np.inf, 0), (0.3, 0),
                                             (0.1, -0.1), (0.1, 1/3), (0.1, np.nan)])
def test_simplex_grid_validates_inputs(spacing, floor):
    with pytest.raises(ValueError):
        ternary_simplex_points(spacing, floor)


@pytest.mark.parametrize("context", ["general", "interface"])
def test_surrogate_samples_are_ordered_copies(context):
    surrogate = _surrogate()
    points, matrices = surrogate.get_diffusivity_samples("ALPHA", query_context=context)
    np.testing.assert_array_equal(points, [[0.20, 0.10], [0.30, 0.10]])
    np.testing.assert_array_equal(matrices, np.repeat(np.eye(2)[None], 2, axis=0))
    points[:] = 0
    matrices[:] = 0
    again, values = surrogate.get_diffusivity_samples("ALPHA", query_context=context)
    assert np.all(again > 0) and np.all(np.diagonal(values, axis1=1, axis2=2) > 0)
    with pytest.raises(ValueError, match="phase"):
        surrogate.get_diffusivity_samples("unknown")
    with pytest.raises(ValueError, match="query_context"):
        surrogate.get_diffusivity_samples("ALPHA", query_context="unknown")


def test_build_reload_and_explicit_rebuild(tmp_path):
    path = tmp_path / "nested" / "bundle"
    calls = []
    def builder():
        calls.append(1)
        return _surrogate(), {}
    first, action = load_or_build_surrogate_artifact(path, {"revision": 1}, builder)
    assert action == "built" and len(calls) == 1
    second, action = load_or_build_surrogate_artifact(path, {"revision": 1}, builder)
    assert action == "reloaded" and len(calls) == 1
    assert first.manifest["build_fingerprint"] == second.manifest["build_fingerprint"]
    with pytest.raises(SurrogateArtifactCompatibilityError, match="--rebuild"):
        load_or_build_surrogate_artifact(path, {"revision": 2}, builder)
    member = second.member_path("model")
    member.write_bytes(b"damage")
    with pytest.raises(SurrogateArtifactIntegrityError):
        load_or_build_surrogate_artifact(path, {"revision": 1}, builder)
    assert len(calls) == 1
    _, action = load_or_build_surrogate_artifact(path, {"revision": 2}, builder, rebuild=True)
    assert action == "built" and len(calls) == 2


def test_partial_bundle_and_missing_required_member_do_not_publish(tmp_path):
    path = tmp_path / "bundle"
    path.mkdir()
    (path / "partial").write_text("keep")
    def forbidden():
        pytest.fail("an incomplete bundle must not rebuild silently")
    with pytest.raises(SurrogateArtifactIntegrityError, match="no manifest.json"):
        load_or_build_surrogate_artifact(path, {}, forbidden)
    assert (path / "partial").read_text() == "keep"
    with pytest.raises(SurrogateArtifactIntegrityError, match="omitted required"):
        load_or_build_surrogate_artifact(tmp_path / "empty", {}, lambda: (_surrogate(), {}), required_members=("report",))
    assert not (tmp_path / "empty/manifest.json").exists()


def test_diagnostic_members_keep_schemas_and_reload_never_captures(tmp_path):
    surrogate = _surrogate()
    surrogate.metadata["failed_bulk_points"] = [{"point": [0.1, 0.2]}]
    records = [{"record_type": "kinetics", "value": 1}]
    members = write_surrogate_diagnostic_members(
        tmp_path / "staging", surrogate, report_roles=("failed_bulk_points",),
        kinetics_records=records, backend="synthetic", elements=["Z", "X", "Y"])
    assert json.loads(members["failed_bulk_points"]["path"].read_text()) == surrogate.metadata["failed_bulk_points"]
    lines = [json.loads(line) for line in members["kinetics_diagnostics"]["path"].read_text().splitlines()]
    assert lines[0] == {"record_type": "metadata", "schema_version": 1,
                        "backend": "synthetic", "elements": ["Z", "X", "Y"]}
    assert lines[1] == {**records[0], "backend": "synthetic"}
    path = tmp_path / "bundle"
    load_or_build_surrogate_artifact(path, {}, lambda: (surrogate, members), required_members=("failed_bulk_points",))
    bundle, action = load_or_build_surrogate_artifact(
        path, {}, lambda: pytest.fail("reload must not capture records"), required_members=("failed_bulk_points",))
    assert action == "reloaded" and bundle.member_path("kinetics_diagnostics").is_file()
    assert write_surrogate_diagnostic_members(tmp_path / "none", surrogate) == {}


def test_inventory_history_and_final_measurements_are_distinct():
    history = [[2.0, 4.0], [2.0, 4.4], [1.9, 4.0], [2.0, 4.0]]
    assert inventory_drift(history) == pytest.approx((0.4, 0.1))
    assert inventory_drift(history, reduction="final") == (0.0, 0.0)
    assert inventory_drift([2.0, 3.0, 2.0]) == (1.0, 0.5)
    assert inventory_drift([[0.0], [0.0]]) == (0.0, 0.0)
    assert inventory_drift([[0.0], [1.0]]) == (1.0, np.inf)


def test_diagnostic_workflow_keeps_optional_old_bundles_usable_and_captures_only_on_build(tmp_path):
    path = tmp_path / "bundle"
    path.mkdir()  # an explicitly empty directory can be built
    calls = []
    def builder(records):
        calls.append(records)
        if records is not None:
            records.append({"record_type": "kinetics", "value": 2})
        surrogate = _surrogate()
        surrogate.metadata["failed_bulk_points"] = []
        return surrogate
    bundle, action = load_or_build_diagnostic_surrogate(
        path, {}, builder, report_roles=("failed_bulk_points",))
    assert action == "built" and calls == [None]
    assert "kinetics_diagnostics" not in bundle.manifest["members"]
    bundle, action = load_or_build_diagnostic_surrogate(
        path, {}, builder, report_roles=("failed_bulk_points",), capture_kinetics=True,
        backend="synthetic", elements=["Z", "X", "Y"])
    assert action == "reloaded" and calls == [None]
    bundle, action = load_or_build_diagnostic_surrogate(
        path, {}, builder, report_roles=("failed_bulk_points",), rebuild=True,
        capture_kinetics=True, backend="synthetic", elements=["Z", "X", "Y"])
    assert action == "built" and len(calls) == 2
    records = [json.loads(line) for line in bundle.member_path("kinetics_diagnostics").read_text().splitlines()]
    assert records[1]["value"] == 2


@pytest.mark.parametrize("history", [[], [[np.nan]], [[np.inf]], np.ones((2, 2, 2))])
def test_inventory_rejects_undefined_histories(history):
    with pytest.raises(ValueError):
        inventory_drift(history)


@pytest.mark.parametrize("points", [np.array([[0., 0.], [1., 0.], [0., 1.]]),
                                    np.array([[0., 0.], [1., 0.]]), np.empty((0, 2))])
def test_bulk_coverage_handles_hulls_degenerate_sets_and_nonfinite_queries(points):
    surrogate = SimpleNamespace(get_diffusivity_samples=lambda phase: (points, None))
    queries = np.array([[0., 0.], [0.2, 0.2], [2., 2.], [np.nan, 0.]])
    report = evaluate_bulk_coverage(surrogate, {"A": queries})["A"]
    expected = [True, True, False, False] if len(points) == 3 else [bool(len(points)), False, False, False]
    np.testing.assert_array_equal(report["inside"], expected)
    assert report["outside"] == 4 - sum(expected)
    assert evaluate_bulk_coverage(surrogate, {"A": np.empty((0, 2))})["A"]["count"] == 0


def test_new_modules_import_without_plotting_or_thermodynamic_engines():
    code = """
import sys
import moving_boundary.providers.constant, moving_boundary.sampling
import moving_boundary.run_diagnostics, moving_boundary.plotting
assert not {'matplotlib.pyplot', 'plotly', 'pycalphad', 'tc_python', 'kawin'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
