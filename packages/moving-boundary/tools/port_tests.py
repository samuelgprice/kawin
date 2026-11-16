"""Port research Illingworth tests into the package test suite (plan §6, item 16; historical tool).

    python tools/port_tests.py --research <frozen research checkout> --batch N [--check]

Reads research ``kawin/tests/<file>.py`` by path (research code is never imported) and writes
``tests/<file>.py``, plus every ``kawin.tests.<helper>`` module those files import, transitively,
as ``tests/ported_helpers/<helper>.py``. The text stays verbatim except for these transforms,
each recorded in ``docs/test-port-provenance.json``:

- *imports*: a research import is resolved name by name exactly as
  ``tools/definitions/_bindings.Context.symbol`` resolves it in the package role: solver classes
  from ``moving_boundary.solvers``; research-only modules (``_bindings.MODULES``, plus the
  port-only ``PORT_MODULES``) to their package modules, never back to research;
  ``kawin.diffusion``/``kawin.diffusion.mesh`` names to the
  first of ``moving_boundary``, ``moving_boundary.mesh`` and
  ``moving_boundary.solvers.illingworth.shared`` that has them, else stock kawin 0.5.0. A name
  with no counterpart is an error;
- *helper imports*: ``kawin.tests.<helper>`` becomes ``ported_helpers.<helper>`` (``tests/conftest.py``
  puts ``tests/`` on ``sys.path`` so this works under ``--import-mode=importlib`` too); a helper
  that is itself a research test module (``HELPER_SUBSETS``) is reduced to the named definitions
  and what they need (``extract_subset``) under a non-test file name;
- *relocated example helpers*: imports from the research similarity example scripts
  (``EXAMPLE_MODULES``) bind ``moving_boundary.analytic``, where those helpers now live; every
  name must exist there. Any other research example import is an error;
- *patches across the module split*: ``<patcher>.setattr(<research module alias>, NAME, value)``
  becomes ``setattr_owners(<patcher>, <alias>, NAME, value)`` (``tests/ported_helpers/relocation.py``)
  when the package module re-imports NAME from another package module (research kept both in one
  module), or when NAME is not a literal; the patch then reaches the same code as in research;
- *strings*: a string constant naming a research module (e.g. a ``skipif`` that imports
  ``kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D``) is renamed to the package module;
- *deferred tests*: tests needing unported functionality (``DEFERRED``, each listed in
  FUTURE_PORTS.md § Deferred tests) are removed with their decorators, and so are the
  module-level imports only they used;
- *superseded tests*: tests whose premise a deliberate package change removed (``SUPERSEDED``, each
  with its reason and the package test that replaces it) are removed in the same way and recorded
  as ``superseded test``. Unlike a deferred test, a superseded one is not waiting for a port;
- *API migrations*: call sites listed for the source in ``tools/api_migrations.py`` are migrated
  to a changed package API by small recorded text edits (kind ``api_migration``, e.g. LEDGERS R08's
  dimensionless binary tolerance);
- *docstring*: a provenance paragraph (source, commit, sha256, exclusions) is prepended.

Research test files deferred whole (``DEFERRED_FILES``) are recorded under ``deferred_files`` with
their counts and reason; nothing is written for them.

``--check`` regenerates in memory and requires the committed files and provenance record to be
identical (newlines normalized); nothing is written. Any research module path left in the output
is an error. Validation that the ported tests pass is the package test suite itself.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import re
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
TESTS = PROJECT / "tests"
HELPERS = "ported_helpers"
RECORD = PROJECT / "docs/test-port-provenance.json"

BATCHES = {
    1: ["test_illingworth_setup_and_diagnostics.py", "test_illingworth_ternary_three_phase.py",
        "test_illingworth_interstitials_three_phase.py", "test_surrogate_artifacts.py"],
    2: ["test_illingworth_binary_variable_diffusivity.py", "test_illingworth_interstitials.py",
        "test_transformed_grids.py"],
    3: ["test_illingworth_ternary_three_phase_similarity.py", "test_illingworth_interstitials_ternary.py",
        "test_illingworth_ternary_physical_scale.py", "test_similarity_comparison_basis.py",
        "test_illingworth_ternary_similarity.py"],
    4: ["test_illingworth_ternary.py"],
    5: ["test_surrogate_diagnostics.py"],
}
# Tests needing unported functionality: removed here, listed in FUTURE_PORTS.md § Deferred tests.
DEFERRED = {
    "test_surrogate_artifacts.py": {
        "test_cuniti_tc_python_bundle_smoke": "opt-in licence smoke test (KAWIN_RUN_TC_PYTHON_SMOKE=1, "
            "TC-Python with TCHEA5/MOBHEA4): in every normal run it skips, and approve's whole-suite gate "
            "allows no skips; the adapter it uses is already ported (moving_boundary.providers.thermocalc), "
            "so it can be ported as an opt-in check outside the gate",
    },
    "test_illingworth_ternary.py": {
        "test_ternary_illingworth_quasi_binary_cu_zn_dummy_matches_binary_brass_case":
            "builds its binary reference with build_fig3_present_work_model/build_fig6_illingworth_case_params from "
            "the research example examples/Illingworth2005/compare_illingworth2005_planar.py, which is not ported "
            "(FUTURE_PORTS entry 6, decision per script)",
    },
    "test_surrogate_diagnostics.py": {
        "test_three_phase_example_helper_routes_interfaces_truth_and_merged":
            "tests plot_surrogate_diagnostics_for_run of the research script "
            "examples/ternaryExamples/IllingworthTernaryThreePhaseStable.py, which is not ported "
            "(FUTURE_PORTS entry 6, decision per script)",
    },
}
# Tests superseded by a deliberate package change: removed here like deferred tests, but recorded
# with the reason and the package test that replaces them (``replaced_by``, a pytest node id under
# ``tests/``). Not listed in FUTURE_PORTS.md: nothing remains to be ported.
SUPERSEDED = {
    "test_illingworth_ternary.py": {
        "test_ternary_interface_success_diagnostics_count_candidate_and_jacobian_work": dict(
            reason="LEDGERS R13: it pins the scaled residual of the earlier default normalization (divided by the "
                   "domain length); its builder takes no keyword arguments and serves many other tests, which stay "
                   "on the new default",
            replaced_by="tests/test_illingworth_ternary_residual_normalization.py::"
                        "test_superseded_research_diagnostics_test_holds_with_the_unit_scale"),
    },
    "test_illingworth_ternary_three_phase_similarity.py": {
        "test_three_phase_unit_composition_scale_matches_default_bitwise": dict(
            reason="LEDGERS R13: an explicit scale of 1.0 no longer equals the default; it selects the earlier "
                   "default expression",
            replaced_by="tests/test_illingworth_ternary_residual_normalization.py::"
                        "test_unit_scale_reproduces_the_pre_r13_default_run"),
    },
    "test_surrogate_diagnostics.py": {
        "test_plotly_optional_dependency_is_declared": dict(
            reason="FUTURE_PORTS entry 1: it reads the research pyproject.toml and environment.yml; the package "
                   "declares the same diagnostics extra in its own pyproject.toml and pins the tested Plotly in "
                   "examples/requirements-tested.txt",
            replaced_by="tests/test_surrogate_diagnostics_imports.py::"
                        "test_plotly_optional_dependency_is_declared_by_the_package"),
        "test_plotly_import_error_keeps_evaluation_available": dict(
            reason="FUTURE_PORTS entry 1: the missing-Plotly message names the package extra "
                   "moving-boundary[diagnostics]; the research test matches the research text kawin[diagnostics]",
            replaced_by="tests/test_surrogate_diagnostics_imports.py::"
                        "test_plotly_import_error_keeps_evaluation_available_and_names_the_package_extra"),
    },
}
# Research-only modules relocated for the port that acceptance definitions never use, so they are
# not in ``_bindings.MODULES`` (which must equal ``acceptance_inputs.MODULES``). Resolved the same way.
PORT_MODULES = {
    "kawin.diffusion.SurrogateDiagnostics": "moving_boundary.surrogate_diagnostics",
}
# Research example modules whose helpers were relocated into the package (plan §6, FUTURE_PORTS.md
# "Deferred tests": the similarity helpers exist in moving_boundary.analytic). Each name must exist.
EXAMPLE_MODULES = {
    "examples.ternaryExamples.ternaryTwoPhaseSimilarity": "moving_boundary.analytic",
    "examples.ternaryExamples.ternaryThreePhaseSimilarity": "moving_boundary.analytic",
}
# Research test files deferred whole (FUTURE_PORTS.md § Deferred tests): recorded, never written.
DEFERRED_FILES = {
    3: {"test_illingworth_ternary_convergence.py": "needs IllingworthTernaryConvergence (FUTURE_PORTS entry 4)",
        "test_three_phase_examples.py": "tests the research script IllingworthTernaryThreePhaseStable (FUTURE_PORTS entry 6)"},
}
# Helpers that are themselves research test modules: only these names (and the top-level code they
# need) are extracted, into ported_helpers/<target>.py, so no research test is copied or collected.
HELPER_SUBSETS = {
    "test_diffusion": ("diffusion_helpers", ["ConstantBinaryThermodynamics", "VariableBinaryThermodynamics",
                                             "_build_illingworth_default_model"]),
    "test_illingworth_ternary": ("ternary_helpers", ["_IdentityTernaryThermodynamics", "_LinearInterfaceEquilibrium"]),
}
RESEARCH_PREFIXES = ("kawin.diffusion.MovingBoundary", "kawin.diffusion.mesh.MovingBoundary",
                     "kawin.diffusion.mesh.TransformedGrids", "kawin.diffusion._illingworth",
                     "kawin.diffusion.SurrogateArtifacts", "kawin.diffusion.SurrogateDiagnostics",
                     "kawin.tests", "examples.")


class PortError(RuntimeError):
    """A research test cannot be ported mechanically; nothing is written."""


def _bindings():
    spec = importlib.util.spec_from_file_location("port_bindings", PROJECT / "tools/definitions/_bindings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BINDINGS = _bindings()
MODULES = {**BINDINGS.MODULES, **PORT_MODULES}


def _relocation():
    spec = importlib.util.spec_from_file_location("port_relocation", TESTS / HELPERS / "relocation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RELOCATION = _relocation()


def _api_migrations():
    spec = importlib.util.spec_from_file_location("port_api_migrations", PROJECT / "tools/api_migrations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


API = _api_migrations()


def resolve(module, name):
    """The package (or stock kawin) module a research ``from module import name`` binds, as ``_bindings``."""
    if name in BINDINGS.MODELS and module.startswith("kawin.diffusion"):
        return "moving_boundary.solvers"
    choices = []
    if module == "kawin.thermo" and name == "MulticomponentThermodynamics":
        choices.append("moving_boundary.compatibility.thermo")
    if module in MODULES:
        choices.append(MODULES[module])
    if module in {"kawin.diffusion", "kawin.diffusion.mesh"}:
        choices.extend(["moving_boundary", "moving_boundary.mesh", "moving_boundary.solvers.illingworth.shared"])
    for choice in choices:
        if hasattr(importlib.import_module(choice), name):
            return choice
    if module in MODULES:
        raise PortError(f"{module}.{name} has no package counterpart in {MODULES[module]}")
    if hasattr(importlib.import_module(module), name):
        return module
    raise PortError(f"{module}.{name} exists neither in the package nor in stock kawin")


def rename_module(text):
    """A research module path (or a dotted name below one) renamed to its package module, else ``None``."""
    for research, package in sorted(MODULES.items(), key=lambda item: -len(item[0])):
        if text == research or text.startswith(research + "."):
            return package + text[len(research):]
    for helper, (target, _) in HELPER_SUBSETS.items():
        research = "kawin.tests." + helper
        if text == research or text.startswith(research + "."):
            return f"{HELPERS}.{target}" + text[len(research):]
    if text == "kawin.tests" or text.startswith("kawin.tests."):
        return HELPERS + text[len("kawin.tests"):]
    return None


def extract_subset(source, names, relative):
    """The verbatim top-level segments of ``source`` that define ``names``, with everything they need.

    Follows references to other top-level definitions and imports transitively; keeps only the
    import aliases used; each segment is marked with its research line range. A module with a
    star import, or a requested name that is not defined at top level, raises ``PortError``.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    if any(isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names) for node in tree.body):
        star = True
    else:
        star = False
    defined = {}
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined[node.name] = index
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        defined[name.id] = index
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    defined[alias.asname or alias.name.split(".")[0]] = index
    missing = [name for name in names if name not in defined]
    if missing:
        raise PortError(f"{relative}: names not defined at top level: {missing}")
    wanted, queue = set(), list(names)
    while queue:
        name = queue.pop()
        if name in wanted:
            continue
        wanted.add(name)
        node = tree.body[defined[name]]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name) and inner.id in defined and inner.id not in wanted:
                queue.append(inner.id)
    builtins = set(dir(__builtins__)) if not isinstance(__builtins__, dict) else set(__builtins__)
    if star:
        free = {inner.id for name in wanted for inner in ast.walk(tree.body[defined[name]])
                if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load)}
        unknown = sorted(free - set(defined) - builtins)
        locals_ = {inner.id for name in wanted for inner in ast.walk(tree.body[defined[name]])
                   if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store)}
        locals_ |= {arg.arg for name in wanted for inner in ast.walk(tree.body[defined[name]])
                    if isinstance(inner, (ast.FunctionDef, ast.Lambda)) for arg in ast.walk(inner.args) if isinstance(arg, ast.arg)}
        unknown = [name for name in unknown if name not in locals_]
        if unknown:
            raise PortError(f"{relative}: names may come from a star import: {unknown}")
    segments = []
    for index in sorted({defined[name] for name in wanted}):
        node = tree.body[index]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep = [alias for alias in node.names if (alias.asname or alias.name.split(".")[0]) in wanted]
            if isinstance(node, ast.ImportFrom):
                text = f"from {node.module} import " + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep)
            else:
                text = "import " + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep)
            segments.append(f"# --- {relative}:{node.lineno}-{node.end_lineno} (aliases used)\n{text}\n")
        else:
            first, last = _span(node)
            segments.append(f"# --- {relative}:{first}-{last}\n" + "".join(lines[first - 1:last]))
    return "\n\n".join(segment.rstrip("\n") + "\n" for segment in segments)


def _offsets(source):
    """Map (line, utf-8 column) from ``ast`` to character offsets in ``source``."""
    starts, total = [], 0
    lines = source.splitlines(keepends=True)
    for line in lines:
        starts.append(total)
        total += len(line)
    def offset(line, column):
        text = lines[line - 1]
        return starts[line - 1] + len(text.encode("utf-8")[:column].decode("utf-8"))
    return offset


def _span(node):
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return first, node.end_lineno


def _names_used(tree, skip):
    """Names read anywhere in ``tree`` outside import statements and the line spans in ``skip``."""
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not any(a <= node.lineno <= b for a, b in skip):
            used.add(node.id)
    return used


def top_level_definitions(source):
    """Names a module defines itself at top level (functions, classes, assignments; not imports)."""
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                names.update(inner.id for inner in ast.walk(target) if isinstance(inner, ast.Name))
    return names


def port_text(source, name, *, deferred=None, superseded=None, header, research_defines=None):
    """Port one research module's text; returns ``(text, transforms, helpers, counts)``.

    ``deferred`` maps top-level test names to reasons. ``superseded`` maps top-level test names to
    ``dict(reason=..., replaced_by=...)``; both kinds are removed, and a name may not be in both.
    ``counts`` gains a ``superseded`` entry only when there are any, so records of files without
    superseded tests are unchanged. ``header`` is the provenance paragraph.
    ``research_defines(module)`` returns the top-level names a research module defines (read by
    path); it decides whether a patch on that module's alias follows the package re-import chain.
    """
    deferred = dict(deferred or {})
    superseded = dict(superseded or {})
    both = sorted(set(deferred) & set(superseded))
    if both:
        raise PortError(f"{name}: tests listed as both deferred and superseded: {both}")
    for test, entry in superseded.items():
        if set(entry) != {"reason", "replaced_by"} or not all(isinstance(v, str) and v for v in entry.values()):
            raise PortError(f"{name}: superseded test {test} needs a non-empty reason and replaced_by")
    tree = ast.parse(source)
    offset = _offsets(source)
    lines = source.splitlines(keepends=True)
    tests = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")]
    missing = sorted(set(deferred) - set(tests))
    if missing:
        raise PortError(f"{name}: deferred tests not found: {missing}")
    missing = sorted(set(superseded) - set(tests))
    if missing:
        raise PortError(f"{name}: superseded tests not found: {missing}")
    removed = [_span(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and (node.name in deferred or node.name in superseded)]
    used_before = _names_used(tree, [])
    used_after = _names_used(tree, removed)
    edits, transforms, helpers = [], [], set()

    def inside_removed(node):
        return any(a <= node.lineno <= b for a, b in removed)

    def segment(node):
        return source[offset(node.lineno, node.col_offset):offset(node.end_lineno, node.end_col_offset)]

    # ``import <research module> as alias``: patches on these aliases may need setattr_owners.
    module_aliases = {alias.asname: (alias.name, rename_module(alias.name)) for node in ast.walk(tree)
                      if isinstance(node, ast.Import) for alias in node.names if alias.asname and rename_module(alias.name)}
    needs_relocation_import, runtime_modules = False, {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setattr"
                and len(node.args) >= 3 and isinstance(node.args[0], ast.Name) and node.args[0].id in module_aliases
                and not inside_removed(node)):
            research_module, package = module_aliases[node.args[0].id]
            if research_defines is None:
                raise PortError(f"{name}:{node.lineno}: a patch on a research module alias needs research_defines")
            key = node.args[1]
            extra = []
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                chain = RELOCATION.owner_chain(importlib.import_module(package), key.value)
                defined = key.value in research_defines(research_module)
                if not (defined and len(chain) > 1):
                    continue   # research patched its own binding only, or nothing moved: unchanged
                owners = [f"{module.__name__}.{attribute}" for module, attribute in chain]
            else:
                constant = "_RESEARCH_DEFINES_" + re.sub(r"\W", "_", research_module.rsplit(".", 1)[-1])
                runtime_modules[constant] = research_module
                extra = [f"research_defines={constant}"]
                owners = f"resolved at run time against the names {research_module} defines ({constant})"
            parts = [segment(node.func.value)] + [segment(arg) for arg in node.args]
            parts += [f"{keyword.arg}={segment(keyword.value)}" for keyword in node.keywords] + extra
            edits.append((offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset),
                          "setattr_owners(" + ", ".join(parts) + ")"))
            transforms.append(dict(kind="monkeypatch through the package re-import chain", line=node.lineno,
                                   call=segment(node), owners=owners))
            needs_relocation_import = True
    if needs_relocation_import:
        last = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))][-1]
        position = offset(last.end_lineno, last.end_col_offset)
        inserted = "\nfrom ported_helpers.relocation import setattr_owners"
        for constant, research_module in sorted(runtime_modules.items()):
            names = ", ".join(repr(item) for item in sorted(research_defines(research_module)))
            inserted += (f"\n\n# Top-level names research {research_module.replace('.', '/')}.py defines (tools/port_tests.py), for patches\n"
                         f"# whose attribute name is known only at run time.\n{constant} = frozenset({{{names}}})")
            transforms.append(dict(kind="research definitions constant", name=constant, research=research_module))
        edits.append((position, position, inserted))
        transforms.append(dict(kind="import", research="(none)", package="ported_helpers.relocation.setattr_owners"))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not inside_removed(node):
            top = node in tree.body
            indent = lines[node.lineno - 1][: node.col_offset]
            groups, changed = {}, False
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    bound = alias.asname or alias.name
                    if top and bound in used_before and bound not in used_after:
                        transforms.append(dict(kind="dropped import (only used by deferred tests)", name=f"{module}.{alias.name}"))
                        changed = True
                        continue
                    target = module
                    if module.startswith("kawin.tests"):
                        target = rename_module(module)
                        helpers.add(module.split(".", 2)[2])
                    elif module.startswith(("kawin.diffusion", "kawin.thermo")) or module in MODULES:
                        target = resolve(module, alias.name)
                    elif module in EXAMPLE_MODULES:
                        target = EXAMPLE_MODULES[module]
                        if not hasattr(importlib.import_module(target), alias.name):
                            raise PortError(f"{name}:{node.lineno}: {module}.{alias.name} has no counterpart in {target}")
                    elif module.startswith("examples"):
                        raise PortError(f"{name}:{node.lineno}: research example import {module} in a ported test")
                    if target != module:
                        changed = True
                        transforms.append(dict(kind="import", research=f"{module}.{alias.name}", package=f"{target}.{alias.name}"))
                    groups.setdefault(target, []).append(alias.name + (f" as {alias.asname}" if alias.asname else ""))
                text = "\n".join(f"{indent if i else ''}from {target} import {', '.join(names)}" for i, (target, names) in enumerate(groups.items()))
            else:
                parts = []
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    if top and bound in used_before and bound not in used_after:
                        transforms.append(dict(kind="dropped import (only used by deferred tests)", name=alias.name))
                        changed = True
                        continue
                    renamed = rename_module(alias.name)
                    if alias.name.startswith("examples"):
                        raise PortError(f"{name}:{node.lineno}: research example import {alias.name}")
                    if renamed:
                        if not alias.asname:
                            raise PortError(f"{name}:{node.lineno}: 'import {alias.name}' without an alias cannot be renamed")
                        changed = True
                        transforms.append(dict(kind="import", research=alias.name, package=renamed))
                    parts.append((renamed or alias.name) + (f" as {alias.asname}" if alias.asname else ""))
                text = f"import {', '.join(parts)}" if parts else ""
            if changed:
                start, end = offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)
                if not text:   # the whole statement goes: remove its line(s)
                    start, end = offset(node.lineno, 0), offset(node.end_lineno, len(lines[node.end_lineno - 1].encode("utf-8")))
                edits.append((start, end, text))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and not inside_removed(node):
            renamed = rename_module(node.value)
            if renamed and node.value.startswith(RESEARCH_PREFIXES):
                start, end = offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)
                segment = source[start:end]
                edits.append((start, end, segment.replace(node.value, renamed, 1)))
                transforms.append(dict(kind="string", research=node.value, package=renamed))
    for edit in API.edits_for(name):
        target = API.locate(tree, edit, name)
        if target is None or inside_removed(target[1]):
            continue
        start, end, replacement = API.text_edit(edit, target, lambda n: offset(n.lineno, n.col_offset),
                                                lambda n: offset(n.end_lineno, n.end_col_offset), source)
        edits.append((start, end, replacement))
        transforms.append(API.record(edit, target[1].lineno))
    for first, last in removed:
        start = offset(first, 0)
        end = offset(last, 0) + len(lines[last - 1])
        while end < len(source) and source[end] == "\n":   # its trailing blank lines; the preceding ones stay
            end += 1
        edits.append((start, end, ""))
    for test, reason in deferred.items():
        transforms.append(dict(kind="deferred test", name=test, reason=reason))
    for test, entry in superseded.items():
        transforms.append(dict(kind="superseded test", name=test, reason=entry["reason"], replaced_by=entry["replaced_by"]))
    doc = tree.body[0] if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant) and isinstance(tree.body[0].value.value, str) else None
    if doc is not None:
        start, end = offset(doc.lineno, doc.col_offset), offset(doc.end_lineno, doc.end_col_offset)
        segment = source[start:end]
        if segment.startswith('"""') and segment.endswith('"""'):
            edits.append((start, end, '"""' + header + "\n\n" + segment[3:]))
        else:
            edits.append((start, start, '"""' + header + '\n"""\n'))
    else:
        edits.append((0, 0, '"""' + header + '\n"""\n'))
    edits.sort(key=lambda edit: (edit[0], edit[1]))
    for (a1, b1, _), (a2, b2, _) in zip(edits, edits[1:]):
        if a2 < b1:
            raise PortError(f"{name}: overlapping edits at {a1}-{b1} and {a2}-{b2}")
    text = source
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    if removed:
        text = text.rstrip("\n") + "\n"   # a removed final test leaves no trailing blank lines
    ast.parse(text)
    leftovers = sorted({match for match in re.findall(r"[A-Za-z_.]*(?:kawin\.diffusion\.(?:mesh\.)?(?:MovingBoundary|TransformedGrids|_illingworth|SurrogateArtifacts|SurrogateDiagnostics)\w*|kawin\.tests\.\w+|examples\.\w+(?:\.\w+)*)", text)})
    if leftovers:
        raise PortError(f"{name}: research module paths left after porting: {leftovers}")
    counts = dict(total=len(tests), deferred=len(deferred), ported=len(tests) - len(deferred) - len(superseded))
    if superseded:
        counts["superseded"] = len(superseded)
    return text, transforms, sorted(helpers), counts


def _sha(text):
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def research_commit(research):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=research, check=True, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL).stdout.strip()


def port_batch(research, batch):
    """Every output (``{relative path: (text, record)}``) for one batch, including its helpers."""
    research = Path(research)
    commit = research_commit(research)
    outputs, queue, seen = {}, [(name, False) for name in BATCHES[batch]], set()
    definitions = {}

    def research_defines(module):
        """Top-level names the research module defines (read by path; never imported)."""
        if module not in definitions:
            path = research / (module.replace(".", "/") + ".py")
            definitions[module] = top_level_definitions(path.read_bytes().decode("utf-8"))
        return definitions[module]

    while queue:
        name, helper = queue.pop(0)
        if (name, helper) in seen:
            continue
        seen.add((name, helper))
        relative = f"kawin/tests/{name}" if not helper else f"kawin/tests/{name}.py"
        full = (research / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")
        source, selected = full, None
        if helper and name in HELPER_SUBSETS:
            selected = HELPER_SUBSETS[name][1]
            source = extract_subset(full, selected, relative)
        deferred = DEFERRED.get(name, {})
        superseded = SUPERSEDED.get(name, {})
        excluded = "; ".join([f"{test} (deferred: see FUTURE_PORTS.md § Deferred tests)" for test in deferred]
                             + [f"{test} (superseded by {entry['replaced_by']})" for test, entry in superseded.items()]) or "none"
        scope = f" Selected names (with what they need): {', '.join(selected)}." if selected else ""
        header = (f"Ported from research ``{relative}`` at ``{commit[:7]}`` (LF-normalized sha256\n"
                  f"``{_sha(full)}``) by\n"
                  f"``tools/port_tests.py``{'' if helper else f' (batch {batch})'}; do not edit by hand. Imports resolve to the package as\n"
                  f"``tools/definitions/_bindings.py`` does; the full transform list is in\n"
                  f"``docs/test-port-provenance.json``. Excluded tests: {excluded}.{scope}")
        text, transforms, helpers, counts = port_text(source, relative, deferred=deferred, superseded=superseded,
                                                      header=header, research_defines=research_defines)
        if selected:
            target = f"tests/{HELPERS}/{HELPER_SUBSETS[name][0]}.py"
        else:
            target = f"tests/{HELPERS}/{name}.py" if helper else f"tests/{name}"
        # Helpers are shared across batches, so they record no batch.
        entry = dict(batch=None if helper else batch, source=relative, research_commit=commit, source_sha256=_sha(full),
                     output_sha256=_sha(text), counts=None if helper else counts, transforms=transforms)
        if selected:
            entry["selected_names"] = selected
        outputs[target] = (text, entry)
        queue.extend((module, True) for module in helpers)
    return outputs


def deferred_files(research, batch):
    """Record entries for this batch's research test files deferred whole (no output is written)."""
    research = Path(research)
    commit = research_commit(research)
    entries = {}
    for name, reason in DEFERRED_FILES.get(batch, {}).items():
        relative = f"kawin/tests/{name}"
        source = (research / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")
        total = sum(1 for node in ast.parse(source).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"))
        entries[relative] = dict(batch=batch, source=relative, research_commit=commit, source_sha256=_sha(source),
                                 counts=dict(total=total, deferred=total, ported=0), reason=reason)
    return entries


def main(argv=None):
    """Write (or ``--check``) one batch of ported tests and update the provenance record."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--research", type=Path, required=True, help="frozen research checkout (read by path only)")
    parser.add_argument("--batch", type=int, choices=sorted(BATCHES), required=True)
    parser.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    args = parser.parse_args(argv)
    outputs = port_batch(args.research, args.batch)
    whole = deferred_files(args.research, args.batch)
    record = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else dict(
        description="Ported research tests (plan section 6, item 16): source, research commit, sha256 of the source and "
                    "of the output (newlines normalized), per-file counts and every transform. Written by tools/port_tests.py.",
        files={})
    record.setdefault("deferred_files", {})
    if args.check:
        problems = [f"deferred file record differs: {name}" for name, entry in whole.items()
                    if record["deferred_files"].get(name) != entry]
        for target, (text, entry) in outputs.items():
            path = PROJECT / target
            if not path.exists() or path.read_text(encoding="utf-8").replace("\r\n", "\n") != text:
                problems.append("differs or missing: " + target)
            if record["files"].get(target) != entry:
                problems.append("provenance record differs: " + target)
        print("\n".join(problems) or f"batch {args.batch}: {len(outputs)} files identical to a fresh port")
        return 1 if problems else 0
    (TESTS / HELPERS).mkdir(exist_ok=True)
    init = TESTS / HELPERS / "__init__.py"
    if not init.exists():
        init.write_text('"""Research test helpers ported by tools/port_tests.py (see docs/test-port-provenance.json)."""\n',
                        encoding="utf-8", newline="\n")
    for target, (text, entry) in outputs.items():
        (PROJECT / target).write_text(text, encoding="utf-8", newline="\n")
        record["files"][target] = entry
        counts = entry["counts"]
        print(target, (f"ported {counts['ported']} / deferred {counts['deferred']} / "
                       f"superseded {counts.get('superseded', 0)} / total {counts['total']}") if counts else "(helper)")
    for name, entry in whole.items():
        record["deferred_files"][name] = entry
        print(name, f"ported 0 / deferred {entry['counts']['deferred']} / total {entry['counts']['total']} (deferred whole)")
    record["files"] = dict(sorted(record["files"].items()))
    record["deferred_files"] = dict(sorted(record["deferred_files"].items()))
    RECORD.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
