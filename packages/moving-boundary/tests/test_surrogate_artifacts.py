"""Ported from research ``kawin/tests/test_surrogate_artifacts.py`` at ``7af24f4`` (LF-normalized sha256
``b64430a6cb255ca97050f248523578a5ab5b04c077ba9025c88974a2bd0595a0``) by
``tools/port_tests.py`` (batch 1); do not edit by hand. Imports resolve to the package as
``tools/definitions/_bindings.py`` does; the full transform list is in
``docs/test-port-provenance.json``. Excluded tests: test_cuniti_tc_python_bundle_smoke (deferred: see FUTURE_PORTS.md § Deferred tests).
"""
import json
from pathlib import Path

import numpy as np
import pytest

import moving_boundary.surrogate_artifacts as artifact_module
from moving_boundary import SurrogateArtifactBundle, SurrogateArtifactCompatibilityError, SurrogateArtifactIntegrityError, TernaryMovingBoundaryThermodynamicsSurrogate, canonicalize_build_spec, prepare_surrogate_artifact, surrogate_build_fingerprint


def _surrogate(scale=1.0):
    eta = np.asarray([0.0, 1.0])
    tielines = {
        "ALPHA": np.asarray([[0.20, 0.10], [0.30, 0.10]]),
        "BETA": np.asarray([[0.10, 0.20], [0.15, 0.25]]),
    }
    compositions = {
        "interface": {phase: values.copy() for phase, values in tielines.items()},
        "general": {phase: values.copy() for phase, values in tielines.items()},
    }
    matrices = {
        context: {
            phase: np.repeat((np.eye(2) * scale)[None, :, :], 2, axis=0)
            for phase in tielines
        }
        for context in ("interface", "general")
    }
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=("Z", "X", "Y"),
        phases=("ALPHA", "BETA"),
        tieline_phases=("ALPHA", "BETA"),
        temperature=1000.0,
        eta_samples=eta,
        tieline_compositions=tielines,
        diffusivity_compositions=compositions,
        diffusivities=matrices,
        validity_policy="legacy",
    )


def _jsonl(path, schema=2):
    path.write_text(
        json.dumps({"record_type": "metadata", "schema_version": schema}) + "\n"
        + json.dumps({"record_type": "kinetics", "value": 1}) + "\n",
        encoding="utf-8",
    )
    return path


def test_build_fingerprint_is_stable_for_mapping_order_and_array_layout():
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    first = {"temperature": 1000.0, "points": np.ascontiguousarray(values)}
    second = {"points": np.asfortranarray(values), "temperature": np.float64(1000.0)}

    canonical_first, fingerprint_first = surrogate_build_fingerprint(first)
    canonical_second, fingerprint_second = surrogate_build_fingerprint(second)

    assert canonical_first == canonical_second
    assert fingerprint_first == fingerprint_second


@pytest.mark.parametrize(
    "changed",
    [
        {"temperature": 1001.0},
        {"phases": ["ALPHA", "GAMMA"]},
        {"tc": {"calculation_version": 2}},
        {"database": "DB2"},
        {"points": np.asarray([[0.2, 0.1], [0.4, 0.2]])},
        {"interpolation": "simplex_linear"},
        {"probe": {"point": [0.3, 0.2]}},
        {"sources": {"adapter": "different"}},
    ],
)
def test_build_fingerprint_changes_for_numerical_inputs(changed):
    base = {
        "temperature": 1000.0,
        "phases": ["ALPHA", "BETA"],
        "tc": {"calculation_version": 1},
        "database": "DB1",
        "points": np.asarray([[0.2, 0.1], [0.3, 0.2]]),
        "interpolation": "nearest",
        "probe": {"point": [0.2, 0.2]},
        "sources": {"adapter": "same"},
    }
    modified = dict(base)
    modified.update(changed)
    assert surrogate_build_fingerprint(base)[1] != surrogate_build_fingerprint(modified)[1]


def test_bundle_roundtrip_checks_model_and_sidecar(tmp_path):
    sidecar = _jsonl(tmp_path / "kinetics.jsonl")
    spec = {"temperature": 1000.0, "points": np.asarray([[0.2, 0.1]])}
    bundle = SurrogateArtifactBundle.publish(
        tmp_path / "model.surrogate",
        _surrogate(),
        spec,
        {"kinetics_diagnostics": {"path": sidecar, "schema_version": 2}},
    )

    loaded = SurrogateArtifactBundle.load(
        bundle.path, spec, required_members=("kinetics_diagnostics",)
    )

    assert loaded.member_path("kinetics_diagnostics").is_file()
    assert loaded.surrogate.tieline_phases == ("ALPHA", "BETA")
    assert np.allclose(
        loaded.surrogate.getInterdiffusivity([0.2, 0.1], 1000.0, phase="ALPHA"),
        np.eye(2),
    )


def test_bundle_rejects_mismatch_and_reports_field(tmp_path):
    bundle = SurrogateArtifactBundle.publish(tmp_path / "model.surrogate", _surrogate(), {"temperature": 1000.0})

    with pytest.raises(SurrogateArtifactCompatibilityError, match=r"build_spec\.temperature"):
        SurrogateArtifactBundle.load(bundle.path, {"temperature": 1100.0})


def test_bundle_rejects_modified_member(tmp_path):
    bundle = SurrogateArtifactBundle.publish(tmp_path / "model.surrogate", _surrogate(), {"version": 1})
    bundle.member_path("model").write_bytes(b"not an npz")

    with pytest.raises(SurrogateArtifactIntegrityError, match="unexpected size|SHA-256"):
        SurrogateArtifactBundle.load(bundle.path, {"version": 1})


def test_bundle_rejects_missing_required_member_and_identity_mismatch(tmp_path):
    bundle = SurrogateArtifactBundle.publish(tmp_path / "model.surrogate", _surrogate(), {"version": 1})
    with pytest.raises(SurrogateArtifactIntegrityError, match="missing required members"):
        SurrogateArtifactBundle.load(
            bundle.path, {"version": 1}, required_members=("kinetics_diagnostics",)
        )

    manifest_path = bundle.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_identity"]["temperature"] = 999.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SurrogateArtifactIntegrityError, match="identity"):
        SurrogateArtifactBundle.load(bundle.path, {"version": 1})


def test_bundle_rejects_jsonl_schema_and_unsafe_path(tmp_path):
    sidecar = _jsonl(tmp_path / "kinetics.jsonl", schema=1)
    with pytest.raises(SurrogateArtifactIntegrityError, match="JSONL schema"):
        SurrogateArtifactBundle.publish(
            tmp_path / "schema.surrogate",
            _surrogate(),
            {"version": 1},
            {"kinetics": {"path": sidecar, "schema_version": 2}},
        )

    bundle = SurrogateArtifactBundle.publish(tmp_path / "path.surrogate", _surrogate(), {"version": 1})
    manifest_path = bundle.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"]["model"]["path"] = "../outside.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SurrogateArtifactIntegrityError, match="Unsafe artifact member path"):
        SurrogateArtifactBundle.load(bundle.path, {"version": 1})


def test_prepare_defaults_to_rebuild_and_reload_never_calls_builder(tmp_path):
    calls = []

    def builder():
        calls.append("build")
        return _surrogate(scale=len(calls)), {}

    path = tmp_path / "model.surrogate"
    first = prepare_surrogate_artifact(path, {"version": 1}, builder)
    second = prepare_surrogate_artifact(path, {"version": 1}, builder)
    loaded = prepare_surrogate_artifact(path, {"version": 1}, builder, reload=True)

    assert calls == ["build", "build"]
    assert first.manifest["build_fingerprint"] == second.manifest["build_fingerprint"]
    assert np.allclose(loaded.surrogate.diffusivities["general"]["ALPHA"], 2.0 * np.eye(2))


def test_prepare_reload_is_fail_closed(tmp_path):
    def unexpected_builder():
        raise AssertionError("reload must not invoke the builder")

    with pytest.raises(SurrogateArtifactIntegrityError, match="manifest does not exist"):
        prepare_surrogate_artifact(
            tmp_path / "missing.surrogate", {"version": 1}, unexpected_builder, reload=True
        )


def test_failed_manifest_publication_preserves_previous_bundle(tmp_path, monkeypatch):
    path = tmp_path / "model.surrogate"
    original = SurrogateArtifactBundle.publish(path, _surrogate(), {"version": 1})
    original_manifest = (path / "manifest.json").read_bytes()
    real_replace = artifact_module.os.replace

    def fail_manifest_replace(source, destination):
        if Path(destination).name == "manifest.json":
            raise OSError("simulated manifest failure")
        return real_replace(source, destination)

    monkeypatch.setattr(artifact_module.os, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="simulated manifest failure"):
        SurrogateArtifactBundle.publish(path, _surrogate(scale=2.0), {"version": 1})

    assert (path / "manifest.json").read_bytes() == original_manifest
    restored = SurrogateArtifactBundle.load(original.path, {"version": 1})
    assert np.allclose(restored.surrogate.diffusivities["general"]["ALPHA"], np.eye(2))


def test_canonical_build_spec_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        canonicalize_build_spec({"temperature": np.nan})
