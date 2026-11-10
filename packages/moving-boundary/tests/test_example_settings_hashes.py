"""M5: adopted example settings and private surrogate inputs are hash-checked before use."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


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


def test_private_input_hash_for_file_and_directory(tmp_path):
    tool = _examples()
    fake = SimpleNamespace(OUTPUT=tmp_path)
    inputs = tmp_path / "inputs"
    (inputs / "bundle").mkdir(parents=True)
    (inputs / "surrogate.npz").write_bytes(b"npz")
    (inputs / "bundle/member.json").write_bytes(b"{}")
    private = {"surrogate.npz": {"sha256": {"surrogate.npz": _sha(b"npz")}},
               "bundle": {"sha256": {"member.json": _sha(b"{}")}}}
    assert tool.check_input_hash(fake, "surrogate.npz", private) == []
    assert tool.check_input_hash(fake, "bundle", private) == []
    assert tool.check_input_hash(fake, "other.npz", private) == ["input not hashed in private-inputs.json: other.npz"]
    (inputs / "bundle/member.json").write_bytes(b"[]")
    assert tool.check_input_hash(fake, "bundle", private) == ["input changed: bundle/member.json"]
