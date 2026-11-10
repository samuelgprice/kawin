"""Item 7: contract arrays are scaled per model, from exactly validated manifest scales."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCALES = {"model0": {"length_scale": 5.0},
          "model1": {"length_scale": 1.0, "minimum_molar_volume": 1.5}}
MOLAR = "molarInventoryData/values"
INTERFACE = "interfaceData/values"


def _tool(tmp_path, monkeypatch, scales=SCALES):
    spec = importlib.util.spec_from_file_location(
        "contract_scale_tool", Path(__file__).parents[1] / "tools/acceptance.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    tolerances = {group: dict(rtol=0., atol=1e-12) for group in
                  ("interface", "profiles", "eta", "inventory", "accepted_times", "analytic_error")}
    manifest = dict(contracts=["C99"], public_tolerances=tolerances,
                    contract_model_scales=dict(cases={"C99": scales}))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(tool, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(tool, "OUTPUT", tmp_path)
    return tool


def _write(tool, deviation, *, keys=("model0", "model1"), report_scales=SCALES):
    base = {f"{prefix}/{name}": np.array([1.0, 2.0]) for prefix in keys for name in (INTERFACE, MOLAR)}
    base["contracts/assertion_groups"] = np.array(3)
    moved = {key: value + deviation.get(key, 0.) for key, value in base.items()}
    for role, arrays in (("reference", base), ("package", moved)):
        (tool.OUTPUT / role).mkdir(exist_ok=True)
        np.savez(tool.OUTPUT / role / "C99.npz", **arrays)
    report = dict(status="PASS", stop_status="contract_assertions_passed", model_scales=report_scales)
    (tool.OUTPUT / "reference/C99.json").write_text(
        json.dumps(dict(status="PASS", stop_status="contract_assertions_passed")), encoding="utf-8")
    return report


def test_public_allowance_uses_each_models_scale(tmp_path, monkeypatch):
    tool = _tool(tmp_path, monkeypatch)
    # model0: interface atol 1e-12 * 5 admits 4e-12; model1 molar atol 1e-12 / 1.5 rejects 8e-13.
    report = _write(tool, {"model0/" + INTERFACE: 4e-12, "model1/" + MOLAR: 5e-13})
    assert not tool.compare("C99", report, tolerance="public").get("failed_keys")
    report = _write(tool, {"model1/" + MOLAR: 8e-13})
    assert tool.compare("C99", report, tolerance="public")["failed_keys"] == ["model1/" + MOLAR]
    report = _write(tool, {"model0/" + INTERFACE: 4e-12})
    assert tool.compare("C99", report, tolerance="exact")["failed_keys"] == ["model0/" + INTERFACE]


@pytest.mark.parametrize("report_scales, keys, message", [
    ({"model0": {"length_scale": 5.0}, "model1": {"length_scale": 1.0}}, ("model0", "model1"), "model_scales differ"),
    (SCALES, ("model0",), "model prefixes differ"),
    (SCALES, ("model0", "model1", "model2"), "model prefixes differ"),
])
def test_scale_or_prefix_mismatch_fails(tmp_path, monkeypatch, report_scales, keys, message):
    tool = _tool(tmp_path, monkeypatch)
    report = _write(tool, {}, keys=keys, report_scales=report_scales)
    with pytest.raises(AssertionError, match=message):
        tool.compare("C99", report, tolerance="public")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_non_finite_or_non_positive_scales_fail(tmp_path, monkeypatch, bad):
    scales = {"model0": {"length_scale": bad}, "model1": SCALES["model1"]}
    tool = _tool(tmp_path, monkeypatch, scales=scales)
    report = _write(tool, {}, report_scales=scales)
    with pytest.raises(AssertionError):
        tool.compare("C99", report, tolerance="public")


def test_committed_scales_cover_every_contract():
    manifest = json.loads((Path(__file__).parents[1] / "tools/acceptance_manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["contract_model_scales"]["cases"]
    assert set(recorded) == set(manifest["contracts"])
    for case, models in recorded.items():
        assert list(models) == [f"model{i}" for i in range(len(models))], case
        assert all(set(scales) <= {"length_scale", "minimum_molar_volume"} for scales in models.values()), case
