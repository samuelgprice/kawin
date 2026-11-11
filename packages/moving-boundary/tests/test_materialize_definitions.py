"""The definitions materializer reproduces today's CaseInputs AST exactly, from verbatim text.

Uses a synthetic research tree, so no research checkout is needed: the unchanged
``acceptance_inputs.CaseInputs.load`` compiles the reference AST and the materializer must
emit commented source whose AST is identical.
"""
import ast
import importlib.util
from pathlib import Path

import pytest

DEMO = '''"""Research module docstring; never selected."""
from __future__ import annotations
import numpy as np, os
from examples.helper import helper_value, unused_name
from math import sqrt, floor

NODES = 4001  # comment kept
UNUSED = 1


def MovingBoundaryIllingworthTernaryFD1DModel(*args, **kwargs):
    return dict(args=args, **kwargs)


# A comment above build.
def build(options):
    from examples.helper import helper_value as hv
    model = MovingBoundaryIllingworthTernaryFD1DModel(
        NODES, 2001,  # inline comment
        terminal_thin_phase_policy="prompt",
        record=hv > 0,
        **options,
    )
    raise RuntimeError("cut away")


def simple():
    return MovingBoundaryIllingworthTernaryFD1DModel(np.sqrt(4.0) + sqrt(helper_value), terminal_thin_phase_policy="x")


def empty():
    return MovingBoundaryIllingworthTernaryFD1DModel()
'''


def _tool():
    spec = importlib.util.spec_from_file_location(
        "materialize_definitions_tests", Path(__file__).parents[1] / "tools/materialize_definitions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def research(tmp_path):
    root = tmp_path / "research"
    (root / "examples").mkdir(parents=True)
    (root / "examples/demo.py").write_text(DEMO, encoding="utf-8")
    (root / "examples/helper.py").write_text("helper_value = 9.0\nunused_name = 0\n", encoding="utf-8")
    return root


def test_emitted_source_has_the_compiled_ast_and_keeps_comments(research):
    tool = _tool()
    entries = {"demo": ("examples/demo.py", ["build", "simple", "empty"], {4001: 401, 2001: 201},
                        ["build"], [], "continue", ["X01"])}
    calls, _ = tool.record_bindinputs(research, entries)
    files, provenance = tool.materialize(research, calls, "test-commit", {})
    assert set(files) == {"demo", "_nested_helper_helper_value_continue"}
    text = files["demo"].decode("utf-8")
    for kept in ("# comment kept", "# A comment above build.", "# inline comment", "return model",
                 "_input_options(options)", "NODES = 401", "401"):
        assert kept in text
    for dropped in ("cut away", "unused_name", "floor", "UNUSED", "Research module docstring", '"prompt"'):
        assert dropped not in text
    assert text.count('terminal_thin_phase_policy="continue"') == 3
    kinds = {item["kind"] for item in provenance["demo"]["transforms"]}
    assert kinds == {"constant", "constructor_cutoff", "import", "policy_removed", "policy_injected", "options_wrapped"}
    assert provenance["demo"]["ast_equal"] and provenance["_nested_helper_helper_value_continue"]["nested_by"] == ["demo"]

    namespace = dict(_input_symbol=lambda module, name: 9.0, _input_module=None,
                     _input_options=lambda o: {k: v for k, v in o.items() if k != "terminal_thin_phase_policy"})
    exec(compile(text, "demo", "exec"), namespace)
    assert namespace["build"]({"dt": 1, "terminal_thin_phase_policy": "prompt"}) == dict(
        args=(401, 201), record=True, dt=1, terminal_thin_phase_policy="continue")
    assert namespace["simple"]() == dict(args=(5.0,), terminal_thin_phase_policy="continue")
    assert namespace["empty"]() == dict(args=(), terminal_thin_phase_policy="continue")


def test_ast_mismatch_aborts(research):
    tool = _tool()
    entries = {"demo": ("examples/demo.py", ["simple"], {}, [], [], "raise", ["X01"])}
    calls, _ = tool.record_bindinputs(research, entries)
    call = next(c for c in calls.values() if c["stem"] == "demo")
    call["ast"].body.append(ast.Pass())
    with pytest.raises(AssertionError, match="AST mismatch"):
        tool.materialize(research, calls, "test-commit", {})


MIGRATION = '''def build(R, options):
    params = {"tolerance": 1.0e-8}
    return dict(tolerance=1e-10, record=True, **options)
'''


def test_api_migration_edits_keep_the_ast_proof(research, monkeypatch):
    """``api_migration`` edits are emitted as text and applied to the recorded AST, so equivalence still holds."""
    tool = _tool()
    (research / "examples/mig.py").write_text(MIGRATION, encoding="utf-8")
    edits = [
        dict(function="build", op="dict_value", key="tolerance", old="1.0e-8", new="2.0e-9"),
        dict(function="build", op="keyword", call="dict", arg="tolerance", old="1e-10", new="1e-10 / R"),
        dict(function="build", op="insert_keyword", call="dict", after="tolerance", arg="bulk_picard_rtol", new="1e-10"),
        dict(function="build", op="kwargs_default", call="dict", arg="extra", new="1"),
    ]
    monkeypatch.setattr(tool.API, "EDITS", {"examples/mig.py": edits})
    entries = {"mig": ("examples/mig.py", ["build"], {}, [], [], "raise", ["X02"])}
    calls, _ = tool.record_bindinputs(research, entries)
    files, provenance = tool.materialize(research, calls, "test-commit", {})
    text = files["mig"].decode("utf-8")
    assert 'dict(tolerance=1e-10 / R, bulk_picard_rtol=1e-10, record=True, **{"extra": 1, **options})' in text
    assert "api_migration" in text and provenance["mig"]["ast_equal"]
    assert provenance["mig"]["migrated_ast_sha256"] != provenance["mig"]["ast_sha256"]
    assert [item["op"] for item in provenance["mig"]["transforms"] if item["kind"] == "api_migration"] == [
        edit["op"] for edit in edits]
    namespace = {}
    exec(compile(text, "mig", "exec"), namespace)
    assert namespace["build"](5.0, {}) == dict(tolerance=1e-10 / 5.0, bulk_picard_rtol=1e-10, record=True, extra=1)

    monkeypatch.setattr(tool.API, "EDITS", {"examples/mig.py": [dict(edits[0], old="1.0e-9")]})
    with pytest.raises(tool.API.MigrationError, match="matched 0 places"):
        tool.materialize(research, calls, "test-commit", {})
