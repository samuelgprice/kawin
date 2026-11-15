"""Recorded API-migration edits that the generators apply to research text (historical tooling).

``tools/materialize_definitions.py`` and ``tools/port_tests.py`` emit research source verbatim apart
from recorded transforms. When a package API changes meaning, the call sites they emit must be
migrated so the generated cases keep their intent. Hand edits would break the byte-for-byte
``--check`` against the frozen oracle, so the migration is a recorded transform of kind
``api_migration``: a small text edit at a named call site, listed in provenance, and applied
identically to the recorded research AST so the materializer's equivalence proof still holds
(emitted AST == recorded AST with the same edits applied).

``EDITS`` maps a research path to its edits. Each edit names the enclosing ``function`` (any
``def`` of that name in the emitted segment) and one operation:

- ``keyword``: in a call to ``call``, replace the value of keyword ``arg`` (source text ``old``)
  with ``new``;
- ``insert_keyword``: in a call to ``call``, insert ``arg=new`` directly after keyword ``after``;
- ``append_keyword``: in a call to ``call`` that has positional arguments and no keyword arguments,
  append ``arg=new`` after the last positional argument (calls that already pass any keyword are
  not candidates, so a keyword-free call can be told apart from its keyword-passing neighbours);
- ``dict_value``: in a dict literal, replace the value of string key ``key`` (``old``) with ``new``;
- ``kwargs_default``: in a call to ``call``, turn ``**X`` into ``**{"arg": new, **X}``, a default
  that an entry of ``X`` still overrides.

One operation names a module-level assignment instead of a function:

- ``module_dict_value``: in the statement ``assignment = {...}`` at module level (a single plain
  name target and a dict literal; never a nested or function-local one), replace the value of
  string key ``key`` (``old``) with ``new``. Other dictionaries and constants are never touched.

An edit whose function or assignment is emitted must match exactly once, and its ``old`` text must
match the research text, otherwise ``MigrationError`` is raised; nothing is skipped silently. Two
module-level assignments to the same name in the research source are ambiguous and also raise
(``check_source``). The materializer also fails when a listed ``module_dict_value`` edit for a
materialized source is never applied (``unapplied``).

An edit may carry its own ``reason`` (the ledger entry that required it); without one the reason is
``REASON``. Each provenance record states the reason of its edit.
"""
import ast

REASON = "LEDGERS R08 binary tolerance API"
R13_REASON = "LEDGERS R13 ternary residual normalization: the test asserts the earlier default scale"


class MigrationError(ValueError):
    """A recorded migration edit did not match the research source exactly once."""


# LEDGERS R08: the binary ``tolerance`` became dimensionless (relative to the domain length R) and
# ``bulk_picard_rtol=None`` no longer follows it. Absolute tolerances become ``old / R``; where the
# implicit mode ran, the Picard tolerance it used to imply is passed explicitly.
EDITS = {
    "kawin/tests/illingworth_baseline_cases.py": [  # C01, LEGACY, C25 (R = 5)
        dict(function="_binary_model", op="keyword", call="MovingBoundaryIllingworthFD1DModel",
             arg="tolerance", old="1.0e-8", new="2.0e-9"),
        dict(function="_binary_model", op="insert_keyword", call="MovingBoundaryIllingworthFD1DModel",
             after="tolerance", arg="bulk_picard_rtol", new="1.0e-8"),
    ],
    "kawin/tests/test_diffusion.py": [  # C02, C03, C30, C29, variable-D tests (R = 5)
        dict(function="_build_illingworth_default_model", op="dict_value", key="tolerance", old="1.0e-8", new="2.0e-9"),
        dict(function="_build_illingworth_default_model", op="kwargs_default", call="MovingBoundaryIllingworthFD1DModel",
             arg="bulk_picard_rtol", new="1.0e-8"),
    ],
    "kawin/tests/test_illingworth_binary_variable_diffusivity.py": [  # tight tests (R = 5)
        dict(function="test_variable_d_conserves_transformed_inventory", op="keyword", call="_model",
             arg="tolerance", old="1e-13", new="2e-14"),
        dict(function="test_implicit_step_satisfies_discrete_equations_at_converged_profile", op="keyword",
             call="_model", arg="tolerance", old="1e-13", new="2e-14"),
    ],
    "kawin/tests/test_illingworth_interstitials.py": [  # C13, C14, C15, interstitial tests (R = 5 and 40)
        dict(function="_binary_model", op="keyword", call="dict", arg="tolerance", old="1e-10", new="1e-10 / R"),
        dict(function="_binary_model", op="insert_keyword", call="dict", after="tolerance",
             arg="bulk_picard_rtol", new="1e-10"),
    ],
    "examples/Interstitials/fe_c_exb1a.py": [  # O07 (SI lengths, lagged)
        dict(function="run_kawin", op="keyword", call="MovingBoundaryIllingworthFD1DModel",
             arg="tolerance", old="1.0e-14", new="1.0e-14 / R"),
    ],
    # C04, C31, F04 (R = 1). User decision 2026-11-11: run at the new default 1e-12 (superseding
    # "keep 1e-8"), so the binary_planar_growth examples, which use the defaults, reproduce them.
    "examples/Illingworth2005/replicate_illingworth2005_fig4_5_6_planar.py": [
        dict(assignment="VALIDATION_PARAMS", op="module_dict_value", key="tolerance", old="1.0e-8", new="1.0e-12"),
    ],
    # LEDGERS R13: without an explicit scale the ternary models now normalize the interface residual by
    # each component's inventory. These two research tests assert values of the earlier default (every
    # row divided by the length term), so the model they call "default" requests it explicitly with
    # ``interface_residual_composition_scale=1.0``, which keeps the earlier expression bit for bit.
    # Both edits are scoped to the test function; no builder or materialized definition is touched.
    "kawin/tests/test_illingworth_ternary_three_phase.py": [
        dict(function="test_three_phase_molar_volume_state_is_immutable_and_equal_volume_kinematics_are_exact",
             op="insert_keyword", call="_make_stationary_model", after="phase_molar_volumes",
             arg="interface_residual_composition_scale", new="1.0", reason=R13_REASON),
    ],
    "kawin/tests/test_illingworth_ternary_three_phase_similarity.py": [
        dict(function="test_three_phase_interface_residual_composition_scale_rows", op="append_keyword",
             call="_build_model", arg="interface_residual_composition_scale", new="1.0", reason=R13_REASON),
    ],
}


def reasons(transforms):
    """The distinct reasons of the ``api_migration`` records in ``transforms``, in order of first use."""
    found = []
    for item in transforms:
        if item.get("kind") == "api_migration" and item["reason"] not in found:
            found.append(item["reason"])
    return found


def edits_for(path):
    """The recorded edits for one research path (an empty list when it has none)."""
    return list(EDITS.get(str(path).replace("\\", "/"), ()))


def _functions(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]


def _calls(function, name):
    return [node for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]


def _same(node, text):
    return ast.dump(node) == ast.dump(ast.parse(text, mode="eval").body)


def _targets(function, edit):
    """``(container, node)`` pairs the edit applies to inside one function (verified ``old`` text)."""
    op = edit["op"]
    found = []
    if op in ("keyword", "insert_keyword"):
        arg = edit["arg"] if op == "keyword" else edit["after"]
        for call in _calls(function, edit["call"]):
            for keyword in call.keywords:
                if keyword.arg == arg and (op == "insert_keyword" or _same(keyword.value, edit["old"])):
                    found.append((call, keyword))
    elif op == "dict_value":
        for node in ast.walk(function):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == edit["key"] and _same(value, edit["old"]):
                        found.append((node, value))
    elif op == "append_keyword":
        for call in _calls(function, edit["call"]):
            if call.args and not call.keywords and not isinstance(call.args[-1], ast.Starred):
                found.append((call, call.args[-1]))
    elif op == "kwargs_default":
        for call in _calls(function, edit["call"]):
            found.extend((call, keyword) for keyword in call.keywords if keyword.arg is None)
    else:
        raise MigrationError(f"unknown api_migration op {op!r}")
    return found


def _module_assignments(tree, name):
    """Module-level ``name = ...`` statements in ``tree``: a module's body, or ``tree`` itself when it is one.

    Only single plain-name targets count; nested, function-local, tuple or annotated assignments never do.
    """
    statements = tree.body if isinstance(tree, ast.Module) else [tree]
    return [node for node in statements if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name]


def _locate_module_dict_value(tree, edit, where):
    """``(dict, value)`` for a ``module_dict_value`` edit, ``None`` when the assignment is absent."""
    assignments = _module_assignments(tree, edit["assignment"])
    if not assignments:
        return None
    if len(assignments) > 1:
        raise MigrationError(f"{where}: api_migration {edit}: {len(assignments)} module-level assignments to "
                             f"{edit['assignment']} are ambiguous, expected exactly 1")
    value = assignments[0].value
    if not isinstance(value, ast.Dict):
        raise MigrationError(f"{where}: api_migration {edit}: {edit['assignment']} is not assigned a dict literal")
    found = [(value, item) for key, item in zip(value.keys, value.values)
             if isinstance(key, ast.Constant) and key.value == edit["key"] and _same(item, edit["old"])]
    if len(found) != 1:
        raise MigrationError(f"{where}: api_migration {edit} matched {len(found)} places, expected exactly 1")
    return found[0]


def locate(tree, edit, where):
    """The single ``(container, node)`` an edit applies to in ``tree``, or ``None`` when its function is absent.

    Raises ``MigrationError`` when the function is present but the edit matches zero or several places.
    ``module_dict_value`` edits look only at module-level assignments (see the module docstring).
    """
    if edit["op"] == "module_dict_value":
        return _locate_module_dict_value(tree, edit, where)
    functions = _functions(tree, edit["function"])
    if not functions:
        return None
    found = [target for function in functions for target in _targets(function, edit)]
    if len(found) != 1:
        raise MigrationError(f"{where}: api_migration {edit} matched {len(found)} places, expected exactly 1")
    return found[0]


def text_edit(edit, target, start, end, source_text):
    """``(start, end, replacement)`` in character offsets for one located edit.

    ``start(node)`` and ``end(node)`` map an AST node to character offsets in ``source_text``.
    """
    container, node = target
    op = edit["op"]
    if op in ("keyword", "dict_value", "module_dict_value"):
        old = source_text[start(node.value if op == "keyword" else node):end(node.value if op == "keyword" else node)]
        if ast.dump(ast.parse(old, mode="eval").body) != ast.dump(ast.parse(edit["old"], mode="eval").body):
            raise MigrationError(f"api_migration {edit}: source text {old!r} differs from the recorded old value")
        value = node.value if op == "keyword" else node
        return start(value), end(value), edit["new"]
    if op == "insert_keyword":
        position = end(node.value)
        return position, position, f", {edit['arg']}={edit['new']}"
    if op == "append_keyword":
        position = end(node)
        return position, position, f", {edit['arg']}={edit['new']}"
    value = node.value  # kwargs_default
    return start(value), end(value), f'{{"{edit["arg"]}": {edit["new"]}, **{source_text[start(value):end(value)]}}}'


def record(edit, line):
    """The provenance record of one applied edit (its own ``reason``, else ``REASON``)."""
    data = dict(kind="api_migration", reason=edit.get("reason", REASON), line=line)
    if "function" in edit:
        data["function"] = edit["function"]
    else:
        data["assignment"] = edit["assignment"]
    data["op"] = edit["op"]
    for name in ("call", "arg", "after", "key", "old", "new"):
        if name in edit:
            data[name] = edit[name]
    return data


def check_source(tree, path):
    """Locate every ``module_dict_value`` edit for ``path`` in the full research module ``tree``.

    A generator emits only selected statements (the last binding of a name), so an ambiguous
    research module could otherwise look unambiguous; this raises ``MigrationError`` instead.
    """
    for edit in edits_for(path):
        if edit["op"] == "module_dict_value":
            locate(tree, edit, path)


def unapplied(path, transforms):
    """The listed ``module_dict_value`` edits for ``path`` that no record in ``transforms`` shows as applied.

    Function-scoped edits are excluded: a generator emits only selected functions (the materializer
    and the test port select different ones from the same research file), so an unselected
    function's edit legitimately applies elsewhere or nowhere.
    """
    def signature(data):
        return sorted((key, value) for key, value in data.items() if key != "line")
    applied = [signature(item) for item in transforms if item.get("kind") == "api_migration"]
    return [edit for edit in edits_for(path)
            if edit["op"] == "module_dict_value" and signature(record(edit, None)) not in applied]


def apply_to_ast(tree, path):
    """Apply ``path``'s recorded edits to an AST in place (for the equivalence proof); returns ``tree``."""
    for edit in edits_for(path):
        target = locate(tree, edit, path)
        if target is None:
            continue
        container, node = target
        new = ast.parse(edit["new"], mode="eval").body
        op = edit["op"]
        if op == "keyword":
            node.value = new
        elif op in ("dict_value", "module_dict_value"):
            container.values[container.values.index(node)] = new
        elif op == "insert_keyword":
            index = container.keywords.index(node)
            container.keywords.insert(index + 1, ast.keyword(arg=edit["arg"], value=new))
        elif op == "append_keyword":
            container.keywords.append(ast.keyword(arg=edit["arg"], value=new))
        else:  # kwargs_default: parse the same template the text edit writes, then keep the original X
            merged = ast.parse(f'{{"{edit["arg"]}": {edit["new"]}, **_}}', mode="eval").body
            merged.values[1] = node.value
            node.value = merged
    return tree
