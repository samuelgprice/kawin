"""Verification must fail promptly on prompts and never rely on solver defaults."""
import importlib.util
from pathlib import Path
import builtins

import pytest


def _tool():
    spec = importlib.util.spec_from_file_location(
        "batch_safety_tests", Path(__file__).parents[1] / "tools/batch_safety.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_is_forbidden_before_loading_case_inputs(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.setattr(builtins, "input", builtins.input)
    tool.forbid_input()
    source = tmp_path / "case.py"
    source.write_text("input('must not wait')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"input\(\) is forbidden"):
        tool.load_case_module(source, "forbidden_case")


@pytest.mark.parametrize("policy", ["raise", "continue"])
def test_explicit_policy_replaces_defaults_and_options_without_mutating_source(tmp_path, policy):
    source = tmp_path / "case.py"
    original = """
def MovingBoundaryIllingworthTernaryFD1DModel(**kwargs):
    return kwargs
options = dict(terminal_thin_phase_policy='prompt', dt=1)
def build():
    return MovingBoundaryIllingworthTernaryFD1DModel(**options)
"""
    source.write_text(original, encoding="utf-8")
    module = _tool().load_case_module(source, "explicit_case_" + policy, policy=policy)
    assert module.build() == dict(terminal_thin_phase_policy=policy, dt=1)
    assert module.options["terminal_thin_phase_policy"] == "prompt"
    assert source.read_text(encoding="utf-8") == original
