"""Package-owned case definitions: hash-checked, research-free, and keyed like today's loader."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
CHECKOUT = PROJECT.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bindings():
    return _load("definitions_bindings_tests", PROJECT / "tools/definitions/_bindings.py")


def test_tables_and_lookup_key_match_the_historical_loader_and_materializer():
    bindings = _bindings()
    historical = _load("acceptance_inputs_tables", PROJECT / "tools/acceptance_inputs.py")
    materializer = _load("materialize_definitions_keys", PROJECT / "tools/materialize_definitions.py")
    assert bindings.MODELS == historical.MODELS and bindings.MODULES == historical.MODULES
    args = ("examples/x.py", ["b", "a"], {4001: 401, 3.6e6: 3600.}, ["run"], ["surrogate"], "continue")
    assert bindings.lookup_key(*args) == materializer.lookup_key(*args)


def test_every_definition_file_matches_its_provenance_and_the_manifest_source_hashes():
    document = json.loads((PROJECT / "docs/definitions-provenance.json").read_text(encoding="utf-8"))
    manifest = json.loads((PROJECT / "tools/acceptance_manifest.json").read_text(encoding="utf-8"))
    assert document["research_commit"] == manifest["reference_commit"]
    files = {path.name for path in (PROJECT / "tools/definitions").glob("*.py")} - {"_bindings.py"}
    assert files == {Path(entry["file"]).name for entry in document["definitions"].values()}
    keys = [entry["key"] for entry in document["definitions"].values()]
    assert len(keys) == len(set(keys))
    for stem, entry in document["definitions"].items():
        assert hashlib.sha256((PROJECT / entry["file"]).read_bytes()).hexdigest() == entry["sha256"], stem
        assert entry["source_sha256"] == manifest["source_sha256"][entry["source"]], stem
        assert entry["ast_equal"] is True, stem


def test_source_keys_cannot_be_opened_and_unregistered_loads_fail():
    bindings = _bindings()
    key = bindings.SourceKey() / "kawin/tests" / "illingworth_baseline_cases.py"
    assert key.as_posix() == "kawin/tests/illingworth_baseline_cases.py"
    assert bindings.SourceKey().joinpath("examples", "a", "b").with_suffix(".py").as_posix() == "examples/a/b.py"
    with pytest.raises(TypeError, match="never files"):
        open(key)
    context = bindings.Context()
    with pytest.raises(LookupError, match="No package-owned definition"):
        context.load(key, ["not_materialized"])
    with pytest.raises(TypeError, match="SourceKey"):
        context.load(CHECKOUT / "kawin/tests/illingworth_baseline_cases.py", ["_binary_model"])


def test_package_context_builds_a_case_without_opening_research_files():
    bindings = _bindings()
    opened = []
    active = [True]

    def audit(event, args):
        if active[0] and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.append(Path(os.fsdecode(args[0])).resolve())

    sys.addaudithook(audit)
    try:
        context = bindings.Context()
        context.family = "binary"
        source = context.load(context.root / "kawin/tests/illingworth_baseline_cases.py", ["_binary_model"])
        model = source._binary_model("phase_uniform", False)
    finally:
        active[0] = False  # audit hooks cannot be removed
    assert type(model).__module__.startswith("moving_boundary."), type(model).__module__
    assert context.providers["binary"].startswith("moving_boundary.")
    research = [CHECKOUT / "kawin", CHECKOUT / "examples"]
    leaked = [path for path in opened if any(path.is_relative_to(root) for root in research) or "mb-ref-" in str(path)]
    assert not leaked, leaked
    assert source.__file__ == "kawin/tests/illingworth_baseline_cases.py"
