"""Batch-only guards and explicit policy injection for frozen case inputs.

These are verification utilities, not changes to either solver's public defaults.
The loader transforms only case constructor keywords in an in-memory AST; source
files and production kawin module globals are never patched.
"""

import ast
import builtins
from pathlib import Path
import sys
import types


def forbid_input():
    """Make any attempted interactive input fail immediately, including imports."""
    builtins.input = _forbidden_input


def _forbidden_input(*args, **kwargs):
    """Raise a distinct error rather than hanging or disguising a missing resource."""
    raise RuntimeError("Interactive input() is forbidden in acceptance/verification workers")


def load_case_module(path, name, *, policy="raise", replacements=None):
    """Load frozen case inputs with an explicit ternary terminal policy.

    ``replacements`` maps import module names for the isolated installed worker;
    it changes this input module's AST only, never sys.modules entries for kawin.
    All ternary constructor calls get the requested policy, including calls that
    originally used a different policy. Source bytes and arithmetic are untouched.
    """
    if policy not in {"raise", "continue"}:
        raise ValueError("Batch policy must be raise or continue")
    path = Path(path).resolve()
    replacements = replacements or {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    class ExplicitPolicy(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            node.module = replacements.get(node.module, node.module)
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if name in {"MovingBoundaryIllingworthTernaryFD1DModel",
                        "MovingBoundaryIllingworthTernaryThreePhaseFD1DModel"}:
                node.keywords = [kw for kw in node.keywords if kw.arg != "terminal_thin_phase_policy"]
                # Explicit **options dictionaries can carry the original policy.
                # A copied dictionary expression avoids mutating input-module globals.
                for keyword in node.keywords:
                    if keyword.arg is None:
                        keyword.value = ast.Call(
                            func=ast.Name(id="_batch_options", ctx=ast.Load()),
                            args=[keyword.value], keywords=[])
                node.keywords.append(ast.keyword(arg="terminal_thin_phase_policy",
                                                 value=ast.Constant(policy)))
            return node

    def options_without_policy(options):
        return {key: value for key, value in options.items() if key != "terminal_thin_phase_policy"}

    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__dict__["_batch_options"] = options_without_policy
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(ExplicitPolicy().visit(tree)), str(path), "exec"), module.__dict__)
    return module
