"""M5: adopted example settings and private surrogate inputs are hash-checked before use."""
import hashlib
import importlib.util
import json
from pathlib import Path


def _examples():
    spec = importlib.util.spec_from_file_location(
        "example_hash_tool", Path(__file__).parents[1] / "tools/acceptance_examples.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def test_committed_manifest_hashes_every_non_private_reference():
    manifest = json.loads((Path(__file__).parents[1] / "tools/example_manifest.json").read_text(encoding="utf-8"))
    references = {entry["reference"] + ".settings.json" for entry in manifest["cases"].values()}
    assert set(manifest["settings_sha256"]) == references - {"O04.settings.json", "O05.settings.json"}
    assert all(len(value) == 64 for value in manifest["settings_sha256"].values())


def test_settings_hash_unhashed_changed_and_private(tmp_path):
    tool = _examples()
    path = tmp_path / "C04.settings.json"
    path.write_bytes(b'{"a": 1}')
    private_path = tmp_path / "O05.settings.json"
    private_path.write_bytes(b'{"b": 2}')
    manifest = dict(settings_sha256={"C04.settings.json": _sha(b'{"a": 1}')})
    digests = tool.settings_digests(manifest, {"example_settings": {"sha256": {"O05.settings.json": _sha(b'{"b": 2}')}}})
    assert tool.check_settings_hash(path, digests) == [] and tool.check_settings_hash(private_path, digests) == []
    assert tool.check_settings_hash(private_path, tool.settings_digests(manifest, {})) == [
        "adopted settings not hashed: O05.settings.json"]
    path.write_bytes(b'{"a": 2}')
    assert tool.check_settings_hash(path, digests) == ["adopted settings changed: C04.settings.json"]


def _acceptance(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "example_hash_acceptance", Path(__file__).parents[1] / "tools/acceptance.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr(tool, "OUTPUT", tmp_path)
    inputs = tmp_path / "inputs"
    (inputs / "bundle").mkdir(parents=True)
    (inputs / "surrogate.npz").write_bytes(b"npz")
    (inputs / "other.npz").write_bytes(b"other")
    (inputs / "bundle/member.json").write_bytes(b"{}")
    return tool, inputs


def test_private_input_hash_for_file_and_directory(tmp_path, monkeypatch):
    tool, inputs = _acceptance(tmp_path, monkeypatch)
    examples = _examples()
    private = {"surrogate.npz": {"sha256": {"surrogate.npz": _sha(b"npz")}},
               "bundle": {"sha256": {"member.json": _sha(b"{}")}}}
    assert examples.check_input_hash(tool, "surrogate.npz", private) == []
    assert examples.check_input_hash(tool, "bundle", private) == []
    assert examples.check_input_hash(tool, "other.npz", private) == ["input not hashed in private-inputs.json: other.npz"]
    (inputs / "bundle/member.json").write_bytes(b"[]")
    assert examples.check_input_hash(tool, "bundle", private) == ["input changed: bundle/member.json"]


def test_private_input_records_that_cannot_verify_fail(tmp_path, monkeypatch):
    """Missing record file, empty map, malformed map or digest, and incomplete coverage all fail."""
    tool, inputs = _acceptance(tmp_path, monkeypatch)
    examples = _examples()
    assert examples.check_input_hash(tool, "surrogate.npz", None) == [
        "private-inputs.json missing; cannot verify available input: surrogate.npz"]
    for entry in ({"sha256": {}}, {"status": "PRESENT"}, {"sha256": []}, "PRESENT"):
        assert examples.check_input_hash(tool, "surrogate.npz", {"surrogate.npz": entry}) == [
            "input hash record empty or malformed: surrogate.npz"], entry
    assert examples.check_input_hash(tool, "surrogate.npz", {"surrogate.npz": {"sha256": {"surrogate.npz": "abc"}}}) == [
        "input hash record malformed: surrogate.npz/surrogate.npz"]
    assert examples.check_input_hash(tool, "surrogate.npz", {"surrogate.npz": {"sha256": {"x.npz": _sha(b"npz")}}}) == [
        "input file not hashed: surrogate.npz/surrogate.npz", "input file missing: surrogate.npz/x.npz"]
    (inputs / "bundle/extra.json").write_bytes(b"{}")
    assert examples.check_input_hash(tool, "bundle", {"bundle": {"sha256": {"member.json": _sha(b"{}")}}}) == [
        "input file not hashed: bundle/extra.json"]
    assert examples.settings_digests(dict(settings_sha256={}), None) == {}
    assert examples.settings_digests(dict(settings_sha256={}), {"example_settings": {"sha256": "bad"}}) == {}
