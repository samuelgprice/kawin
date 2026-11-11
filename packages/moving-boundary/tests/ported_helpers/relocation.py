"""Monkeypatching across the package's module split (hand-written; used by ported research tests).

Research solver modules were split during extraction. For example, research
``kawin.diffusion.MovingBoundaryIllingworthTernaryFDM`` became
``moving_boundary.solvers.illingworth.ternary`` plus helpers in ``...illingworth.shared``, which
``ternary`` re-imports with ``from .shared import ...``. A research test that patched one global of
the research module changed what every function of that module read. In the package, each function
reads the global of the module that defines it, so patching ``ternary`` alone no longer reaches a
function now defined in ``shared``.

``setattr_owners`` patches the name on the given module and on every package module along its
re-import chain (``owner_chain`` follows the top-level ``from ... import`` statements in the source),
with the same value object, so the patch reaches the code it reached in research. ``tools/port_tests.py``
uses ``owner_chain`` to decide where a ported test needs this, and records each use.
"""
import ast
import importlib
import importlib.util
import inspect


def _module_imports(tree):
    """Top-level ``from ... import`` statements, including those inside top-level ``try``/``if`` blocks."""
    pending, found = list(tree.body), []
    while pending:
        node = pending.pop(0)
        if isinstance(node, ast.ImportFrom):
            found.append(node)
        elif isinstance(node, (ast.Try, ast.If)):
            pending.extend(node.body + node.orelse + getattr(node, "finalbody", [])
                           + [stmt for handler in getattr(node, "handlers", []) for stmt in handler.body])
    return found


def owner_chain(module, name):
    """``[(module, name), ...]``: the module, then each package module it re-imports ``name`` from."""
    chain = [(module, name)]
    while True:
        module, name = chain[-1]
        source = None
        for node in _module_imports(ast.parse(inspect.getsource(module))):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    source = (node, alias.name)
        if source is None:
            return chain
        node, original = source
        target = (importlib.util.resolve_name("." * node.level + (node.module or ""), module.__package__)
                  if node.level else node.module)
        if not target.startswith("moving_boundary"):
            return chain
        chain.append((importlib.import_module(target), original))


def setattr_owners(patcher, module, name, value, raising=True):
    """``patcher.setattr`` the same ``value`` on ``module`` and along its re-import chain for ``name``."""
    for owner, attribute in owner_chain(module, name):
        patcher.setattr(owner, attribute, value, raising=raising)
