"""M5b: regeneration outcomes, decoded metadata and the redacted summary."""
import importlib.util
import json
from pathlib import Path

import numpy as np


def _tool():
    spec = importlib.util.spec_from_file_location(
        "regeneration_tool", Path(__file__).parents[1] / "tools/acceptance_regeneration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_match_is_verified_only_with_comparable_backends():
    tool = _tool()
    comparable = dict(comparable=True, reason=None)
    assert tool.outcome(True, comparable) == ("verified", None)
    assert tool.outcome(True, dict(comparable=False, reason="backend versions differ: numpy")) == (
        "not comparable", "backend versions differ: numpy")
    assert tool.outcome(False, comparable, "differs") == ("failed", "differs")
    assert tool.backends(None)["comparable"] is False


def test_backend_difference_is_named(tmp_path):
    tool = _tool()
    here = tool.backends(None)["here"]
    path = tmp_path / "versions.json"
    path.write_text(json.dumps(dict(here, numpy="0.0", kawin="0.4.0")), encoding="utf-8")
    result = tool.backends(path)
    assert not result["comparable"] and result["reason"] == "backend versions differ: numpy"
    path.write_text(json.dumps(dict(here, kawin="0.4.0")), encoding="utf-8")
    assert tool.backends(path)["comparable"]  # kawin differs by design and is only recorded


def test_metadata_compared_field_by_field(tmp_path):
    tool = _tool()
    def save(name, metadata):
        np.savez(tmp_path / name, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)), x=np.zeros(1))
        return tmp_path / name
    a = save("a.npz", {"source": {"tdb": "x", "phases": ["A"]}, "n": 2})
    b = save("b.npz", {"source": {"tdb": "y", "phases": ["A"]}, "n": 2})
    empty = save("empty.npz", {})
    assert tool.metadata_fields(a, a) == dict(present=(True, True), fields=3, differing=[])
    assert tool.metadata_fields(b, a)["differing"] == ["source.tdb"]
    assert tool.metadata_fields(empty, empty)["fields"] == 0


def test_redacted_summary_has_no_hashes_or_paths():
    tool = _tool()
    detail = dict(arrays={"k": dict(dtype="<f8", shape=[1], bit_identical=True)}, generated_only=[], adopted_only=[],
                  container=dict(byte_identical=True, members_equal=True, generated=[], adopted=[]),
                  metadata=dict(present=(True, True), fields=1, differing=[]), matched=True, array_identical=True)
    check = dict(outcome="verified", reason=None, detail=detail)
    report = dict(backends=dict(here={}, generation={"_environment": "C:/env", "numpy": "1"}, comparable=True, reason=None),
                  adopted_unchanged=True, open_items={}, adopted_before={"a": "f" * 64},
                  checks={"open-surrogate.npz": check, "pycalphad-reference.npz": check,
                          "example_settings": {"C04": dict(outcome="verified", reason=None, byte_identical=True,
                                                           json_equal=True, fields=3, differing=[])},
                          "deferred": {}})
    summary = json.dumps(tool.redact(report))
    assert "f" * 64 not in summary and "C:/env" not in summary and '"dtype"' not in summary
    assert json.loads(summary)["checks"]["open-surrogate.npz"]["arrays_bit_identical"] == 1
