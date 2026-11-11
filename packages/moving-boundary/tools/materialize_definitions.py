"""Materialize package-owned case definitions from the frozen research checkout (historical only).

Normal acceptance never runs this tool. It needs ``--research`` (the frozen oracle checkout,
validated exactly as ``acceptance.validate_research_checkout`` does) and the kawin Conda environment,
because it executes today's unchanged ``acceptance_inputs.CaseInputs.load`` in the reference role.

For every registered ``inputs.load(...)`` call (``ENTRIES``), and for every nested load those
definitions trigger through ``_input_symbol`` on a research module, it:

1. records the AST that ``CaseInputs.load``/``BindInputs`` compiles today, by intercepting
   ``compile`` in the loaded ``acceptance_inputs`` namespace (the file itself is not edited);
2. emits the same selected definitions as **verbatim source segments** (comments kept) into
   ``tools/definitions/<stem>.py`` and applies today's transforms as small, recorded text edits:
   numeric constant substitution, constructor cut-off (``return model`` after the first
   ``model = ...``), ``terminal_thin_phase_policy`` keyword injection with ``**options`` wrapped in
   ``_input_options``, import rewriting into ``_input_symbol``/``_input_module`` calls, and the
   ``api_migration`` edits recorded in ``tools/api_migrations.py`` (call sites migrated to a changed
   package API, e.g. LEDGERS R08);
3. proves equivalence: the emitted module's AST, with only its leading docstring removed, must
   equal the recorded AST with the same ``api_migration`` edits applied to it (``ast.dump``, no
   other normalization). Any mismatch aborts before writing.

The emitted modules keep ``_input_symbol(module, name)`` calls rather than static imports, so
the package-role resolver in ``tools/definitions/_bindings.py`` reproduces today's run-time
resolution (including its fallbacks) without an AST or a research root. Research-module symbols
resolve to the materialized nested definitions, never to research files.

Provenance (source path, research commit, source sha256, transforms, AST digest and the run-time
lookup key) is written to ``docs/definitions-provenance.json``; existing entries for other stems
are preserved. ``--check`` regenerates in memory and compares byte for byte without writing.

Usage (kawin environment, from anywhere):
    python tools/materialize_definitions.py --research <mb-ref-...> [--entries STEM ...] [--check | --dry-run]
"""
from pathlib import Path
import argparse
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
DEFINITIONS = PROJECT / "tools/definitions"
PROVENANCE = PROJECT / "docs/definitions-provenance.json"
POLICY_KEYWORD = "terminal_thin_phase_policy"

TESTS = "kawin/tests/"
EXAMPLES = "examples/"
# One entry per distinct top-level ``inputs.load`` call in the acceptance tooling:
# stem -> (research path, names, constants, constructors, override names, policy, cases).
ENTRIES = {
    "baseline_binary": (TESTS + "illingworth_baseline_cases.py", ["_binary_model"], {}, [], [], "raise", ["C01"]),
    "baseline_legacy": (TESTS + "illingworth_baseline_cases.py", ["_binary_model", "_collect"], {}, [], [], "raise", ["LEGACY"]),
    "binary_variable_diffusivity": (TESTS + "test_illingworth_binary_variable_diffusivity.py",
        ["_model", "StrongVariableBinaryThermodynamics", "CountingVariableThermodynamics"], {}, [], [], "raise", ["C02", "C03", "C30"]),
    "illingworth2005_planar": (EXAMPLES + "Illingworth2005/replicate_illingworth2005_fig4_5_6_planar.py",
        ["PlanarValidationCase", "build_python_planar_validation_model", "_clustered_v_grid",
         "planar_exact_interface", "planar_exact_profile", "VALIDATION_PARAMS"], {}, [], [], "raise", ["C04", "C31", "F04"]),
    "baseline_ternary_three_phase": (TESTS + "illingworth_baseline_cases.py",
        ["_ternary_model", "_three_phase_model", "_CoupledTernaryThermodynamics", "_SmoothBulkTernaryThermodynamics"],
        {}, [], [], "raise", ["C05", "C06", "C07", "C08", "C09", "C10"]),
    "three_phase_similarity": (TESTS + "test_illingworth_ternary_three_phase_similarity.py", ["_solution", "_build_model"],
        {6001: 401}, [], [], "raise", ["C11", "C12", "F11", "F12"]),
    "interstitials_binary": (TESTS + "test_illingworth_interstitials.py",
        ["_binary_model", "_VariableBinaryThermodynamics", "StepProfile1D"], {}, [], [], "raise", ["C13", "C14", "C15"]),
    "interstitials_ternary": (TESTS + "test_illingworth_interstitials_ternary.py",
        ["_model", "_closure", "_diffusivities", "_SpyTernaryDiffusivity"], {}, [], [], "raise", ["C16", "C17", "C18", "C19"]),
    "interstitials_three_phase": (TESTS + "test_illingworth_interstitials_three_phase.py", ["_model"], {}, [], [], "raise", ["C20", "C21"]),
    "three_phase_cases": (TESTS + "test_illingworth_ternary_three_phase.py",
        ["_make_nonstationary_unequal_volume_model", "_make_eta_varying_three_phase_model"], {}, [], [], "raise", ["C22", "C23"]),
    "three_phase_terminal": (TESTS + "test_illingworth_ternary_three_phase.py", ["_make_nonstationary_unequal_volume_model"],
        {}, [], [], "continue", ["C32"]),
    "surrogate_artifacts": (TESTS + "test_surrogate_artifacts.py", ["_surrogate"], {}, [], [], "raise", ["C24", "prepare"]),
    "contracts_setup_and_diagnostics": (TESTS + "test_illingworth_setup_and_diagnostics.py",
        ["test_setup_is_idempotent", "test_explicit_setup_then_solve_matches_solve",
         "test_continued_solve_keeps_reference_inventory_and_history"], {}, [], [], "raise", ["C25", "C26", "C27"]),
    "contracts_three_phase_final_remainder": (TESTS + "test_illingworth_ternary_three_phase.py",
        ["test_three_phase_final_remainder_below_outer_minimum_is_solved_and_accepted",
         "test_three_phase_tiny_final_remainder_commits_all_histories_and_allows_continuation"], {}, [], [], "raise", ["C28"]),
    "contracts_binary_retry": (TESTS + "test_illingworth_binary_variable_diffusivity.py",
        ["test_failed_step_is_retried_with_smaller_dt", "test_retry_exhaustion_reraises_last_error"], {}, [], [], "raise", ["C29"]),
    "contracts_three_phase_retry": (TESTS + "test_illingworth_ternary_three_phase.py",
        ["test_three_phase_failed_retries_do_not_mutate_accepted_boundary_or_histories",
         "test_three_phase_retry_below_outer_minimum_is_rejected_before_wrapper"], {}, [], [], "raise", ["C29"]),
    "exb5_run_kawin2": (EXAMPLES + "ternaryExamples/exb5_gamma_alpha_gamma.py", ["run_kawin2"],
        {4001: 401, 8001: 801, 3.6e6: 3600.}, ["run_kawin2"], [], "continue", ["O01"]),
    "exb5_run_kawin3": (EXAMPLES + "ternaryExamples/exb5_gamma_alpha_gamma.py", ["run_kawin3"],
        {4001: 401, 8001: 801, 3.6e6: 3600.}, ["run_kawin3"], [], "continue", ["O02"]),
    "fe_c_exb1a": (EXAMPLES + "Interstitials/fe_c_exb1a.py", ["run_kawin"], {4001: 401}, ["run_kawin"], [], "raise", ["O07"]),
    "ferrite_growth_le": (EXAMPLES + "Interstitials/ternary_ferrite_growth_le.py",
        ["run_kawin", "tie_line_closure", "TabulatedTernaryDiffusivity"], {2001: 401}, ["run_kawin"], [], "continue", ["O03a", "O03b"]),
    "fe_mn_c_ferrite_case": (EXAMPLES + "Interstitials/fe_mn_c_ferrite.py", ["CASE"], {}, [], [], "continue", ["O03a", "O03b"]),
    "three_phase_symmetric_closure": (EXAMPLES + "Interstitials/three_phase_symmetric_check.py", ["_AusteniteFerriteClosure"],
        {}, [], [], "continue", ["O03b"]),
    "fecrni_analytic_comparison": (EXAMPLES + "ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_AnalyticComparison.py",
        ["build_comparison_model", "comparison_geometry", "interface_matrices_at_eta"], {}, [],
        ["surrogate", "PHASE_A", "PHASE_B", "PHYSICAL_NODES"], "raise", ["O04", "O05", "O06"]),
    "ternary_two_phase_similarity_matrix": (EXAMPLES + "ternaryExamples/ternaryTwoPhaseSimilarity.py",
        ["FixedMatrixTernaryDiffusivity"], {}, [], [], "raise", ["O04", "O05", "O06"]),
}


def lookup_key(path, names, constants, constructors, overrides, policy):
    """Canonical run-time key of one load call; ``_bindings.Context.load`` computes the same.

    Unlike ``CaseInputs``' in-process cache key it includes the policy, because the policy is
    baked into the emitted keyword. Numbers are keyed by ``repr`` so 4001 and 4001.0 stay distinct.
    """
    return json.dumps([str(path), sorted(names), sorted([repr(k), repr(v)] for k, v in constants.items()),
                       list(constructors), sorted(overrides), policy])


def module(name, path):
    """Load a sibling tool by file without adding the tools folder to ``sys.path``."""
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


API = module("api_migrations", Path(__file__).resolve().parent / "api_migrations.py")


def record_bindinputs(root, entries):
    """Run today's ``CaseInputs.load`` for each entry; return every compiled call with its AST.

    Each top-level entry uses a fresh ``CaseInputs`` (reference role, the entry's policy), so
    nested research-symbol loads are recorded under the policy that applies in that case's worker.
    ``compile`` is intercepted only in the ``acceptance_inputs`` module namespace.
    """
    inputs_module = module("acceptance_inputs", PROJECT / "tools/acceptance_inputs.py")
    MODELS.update(inputs_module.MODELS)
    MODULES.update(inputs_module.MODULES)
    recorded = []
    stack = []

    def capture(tree, filename, mode, *args, **kwargs):
        # compile() runs before the module body executes, so a top-level call is recorded
        # before the nested loads its body triggers; depth tells them apart.
        call = dict(stack[-1])
        call.update(ast=tree, depth=len(stack))
        recorded.append(call)
        return builtins.compile(tree, filename, mode, *args, **kwargs)

    inputs_module.compile = capture

    class Recording(inputs_module.CaseInputs):
        """CaseInputs whose loads (including nested ones) are attributed to their arguments."""
        def load(self, path, names, *, constants=None, constructors=(), overrides=None):
            relative = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
            stack.append(dict(path=relative, names=list(names), constants=dict(constants or {}),
                              constructors=list(constructors), overrides=sorted(overrides or {}), policy=self.policy))
            try:
                return super().load(path, names, constants=constants, constructors=constructors, overrides=overrides)
            finally:
                stack.pop()

    calls = {}
    for stem, (path, names, constants, constructors, overrides, policy, cases) in entries.items():
        start = len(recorded)
        inputs = Recording(root, "reference", policy)
        inputs.load(Path(root) / path, names, constants=constants, constructors=constructors,
                    overrides={name: object() for name in overrides})
        top = [call for call in recorded[start:] if call["depth"] == 1]
        assert len(top) == 1 and top[0]["path"] == path and top[0]["names"] == list(names), stem
        for call in recorded[start:]:
            nested = call["depth"] > 1
            key = lookup_key(call["path"], call["names"], call["constants"], call["constructors"], call["overrides"], call["policy"])
            entry = calls.setdefault(key, dict(call, stem=None if nested else stem, cases=set(), nested_by=set()))
            if ast.dump(entry["ast"]) != ast.dump(call["ast"]):
                raise AssertionError(f"one load key compiled to two different ASTs: {key}")
            if nested:
                entry["nested_by"].add(stem)
            else:
                entry["stem"] = stem
            entry["cases"].update(cases)
    top_level = {lookup_key(*spec[:6]): stem for stem, spec in entries.items()}
    for key, call in calls.items():
        call["stem"] = top_level.get(key, call["stem"])
        if call["stem"] is None:
            tail = Path(call["path"]).stem
            call["stem"] = "_nested_" + tail + "_" + "_".join(sorted(n.strip("_") for n in call["names"])) + \
                ("" if call["policy"] == "raise" else "_" + call["policy"])
    return calls, inputs_module


class Source:
    """Original research text with UTF-8 column offsets converted to character positions."""
    def __init__(self, text):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def pos(self, lineno, col):
        """Absolute character index of an AST (line, UTF-8 byte column) position."""
        line = self.lines[lineno - 1]
        return self.starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8"))

    def start(self, node):
        return self.pos(node.lineno, node.col_offset)

    def end(self, node):
        return self.pos(node.end_lineno, node.end_col_offset)

    def line_indent(self, node):
        """Indentation of a statement that must begin its own line."""
        line = self.lines[node.lineno - 1]
        prefix = line[:self.start(node) - self.starts[node.lineno - 1]]
        if prefix.strip():
            raise NotImplementedError(f"statement does not start its line: {node.lineno}")
        return prefix


def select(tree, names, overrides):
    """Reproduce ``CaseInputs.load``'s definition closure (selection order and requested names)."""
    bindings, future = {}, []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            bindings[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                future.append(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bindings[alias.asname or alias.name.split(".")[0]] = node
            else:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for item in ast.walk(target):
                        if isinstance(item, ast.Name):
                            bindings[item.id] = node
    wanted, selected, requested = list(names), set(), set()
    while wanted:
        name = wanted.pop()
        requested.add(name)
        if name in overrides:
            continue
        node = bindings.get(name)
        if node is None or id(node) in selected:
            continue
        selected.add(id(node))
        wanted.extend(item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load))
    body = future + [node for node in tree.body if id(node) in selected and node not in future]
    return body, selected, requested


def import_replacement(node, top_selected, requested, modules):
    """Replacement statements for one import, or ``None`` when ``BindInputs`` keeps it unchanged."""
    if isinstance(node, ast.Import):
        if len(node.names) == 1 and not (node.names[0].name in modules and node.names[0].asname):
            return None
        return [f"{a.asname} = _input_module({a.name!r})" if a.name in modules and a.asname
                else ast.unparse(ast.Import(names=[a])) for a in node.names]
    aliases = [a for a in node.names if not top_selected or (a.asname or a.name) in requested]
    if node.module and node.module.startswith(("kawin", "examples")):
        return [f"{a.asname or a.name} = _input_symbol({node.module!r}, {a.name!r})" for a in aliases if a.name != "*"]
    if len(aliases) == len(node.names):
        return None
    return [ast.unparse(ast.ImportFrom(module=node.module, names=aliases, level=node.level))]


def emit_segment(source, node, call, selected, requested, models):
    """Return (text, transforms) for one selected top-level node with today's edits applied."""
    constants, constructors, policy = call["constants"], set(call["constructors"]), call["policy"]
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    while first > 1 and source.lines[first - 2].lstrip().startswith("#"):
        first -= 1
    if node.col_offset:
        raise NotImplementedError("top-level node not at column 0")
    begin, finish = source.starts[first - 1], source.end(node)
    tail = source.lines[node.end_lineno - 1][finish - source.starts[node.end_lineno - 1]:]
    if tail.strip() and not tail.strip().startswith("#"):
        raise NotImplementedError(f"code after a top-level statement on line {node.end_lineno}")
    if tail.strip():
        finish += len(tail.rstrip("\r\n"))  # keep a trailing end-of-line comment
    replacements, insertions, transforms = [], {}, []

    def replace(start, end, text, record):
        replacements.append((start, end, text))
        transforms.append(record)

    def insert(position, text):
        insertions.setdefault(position, []).append(text)

    cut = []
    for item in ast.walk(node):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in constructors:
            for statement in item.body:
                if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "model" for t in statement.targets):
                    start, end = source.end(statement), source.end(item)
                    cut.append((start, end))
                    replace(start, end, "\n" + source.line_indent(statement) + "return model",
                            dict(kind="constructor_cutoff", function=item.name, after_line=statement.end_lineno,
                                 removed_lines=[statement.end_lineno + 1, item.end_lineno]))
                    break
            else:
                raise ValueError("No model constructor assignment in " + item.name)

    def inside_cut(position):
        return any(start <= position < end for start, end in cut)

    for item in ast.walk(node):
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            # BindInputs filters aliases only for selected top-level imports (never __future__).
            lines = import_replacement(item, id(item) in selected, requested, set(MODULES))
            if lines is not None and not inside_cut(source.start(item)):
                indent = "" if item is node else source.line_indent(item)
                replace(source.start(item), source.end(item), ("\n" + indent).join(lines),
                        dict(kind="import", line=item.lineno, to=lines))
        elif isinstance(item, ast.Constant) and isinstance(item.value, (int, float)) and item.value in constants:
            if inside_cut(source.start(item)):
                continue
            text = source.text[source.start(item):source.end(item)]
            assert ast.literal_eval(text) == item.value, (text, item.value)
            replace(source.start(item), source.end(item), repr(constants[item.value]),
                    dict(kind="constant", line=item.lineno, **{"from": text, "to": repr(constants[item.value])}))
        elif isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and models.get(item.func.id, "binary") != "binary":
            if inside_cut(source.start(item)):
                continue
            elements = sorted(item.args + item.keywords, key=lambda e: (e.lineno, e.col_offset))
            removed = [e for e in elements if isinstance(e, ast.keyword) and e.arg == POLICY_KEYWORD]
            kept = [e for e in elements if not (isinstance(e, ast.keyword) and e.arg == POLICY_KEYWORD)]
            added = f'{POLICY_KEYWORD}="{policy}"'
            for e in removed:
                index = elements.index(e)
                if index + 1 < len(elements):
                    replace(source.start(e), source.start(elements[index + 1]), "",
                            dict(kind="policy_removed", line=e.lineno, call=item.func.id,
                                 text=source.text[source.start(e):source.end(e)]))
                elif index > 0:
                    replace(source.end(elements[index - 1]), source.end(e), "",
                            dict(kind="policy_removed", line=e.lineno, call=item.func.id,
                                 text=source.text[source.start(e):source.end(e)]))
                else:
                    replace(source.start(e), source.end(e), "",
                            dict(kind="policy_removed", line=e.lineno, call=item.func.id,
                                 text=source.text[source.start(e):source.end(e)]))
            for e in kept:
                if isinstance(e, ast.keyword) and e.arg is None:
                    insert(source.start(e.value), "_input_options(")
                    insert(source.end(e.value), ")")
                    transforms.append(dict(kind="options_wrapped", line=e.lineno, call=item.func.id))
            if kept:
                insert(source.end(kept[-1]), ", " + added)
            elif removed:
                insert(source.start(removed[0]), added)
            else:
                opening = source.text.index("(", source.end(item.func))
                insert(opening + 1, added)
            transforms.append(dict(kind="policy_injected", line=item.lineno, call=item.func.id, policy=policy))
    for edit in API.edits_for(call["path"]):
        target = API.locate(node, edit, f"{call['path']}:{node.lineno}")
        if target is None:
            continue
        start, end, text = API.text_edit(edit, target, source.start, source.end, source.text)
        if inside_cut(start):
            raise API.MigrationError(f"api_migration {edit} falls inside a constructor cut-off")
        line = target[1].value.lineno if hasattr(target[1], "arg") else target[1].lineno
        if start == end:
            insert(start, text)
            transforms.append(API.record(edit, line))
        else:
            replace(start, end, text, API.record(edit, line))
    # Edits starting inside a cut-off were skipped above, so the remaining replacements must be
    # disjoint; an insertion may sit at a replacement's start (it goes first) or end, never inside.
    replacements.sort()
    for (s1, e1, _), (s2, e2, _) in zip(replacements, replacements[1:]):
        if e1 > s2:
            raise ValueError(f"overlapping edits at {s1}-{e1} and {s2}-{e2}")
    events = sorted([(p, 0, texts) for p, texts in insertions.items()] +
                    [(s, 1, (e, t)) for s, e, t in replacements], key=lambda event: event[:2])
    out, position = [], begin
    for point, order, payload in events:
        if not begin <= point <= finish or point < position:
            raise ValueError(f"edit at {point} falls outside the segment or inside a replaced span")
        out.append(source.text[position:point])
        position = point
        if order == 0:
            out.extend(payload)
        else:
            position, text = payload
            out.append(text)
    out.append(source.text[position:finish])
    return "".join(out), transforms


# Filled from the unchanged acceptance_inputs module at run time (single source of truth).
MODELS = {}
MODULES = {}


def materialize(root, calls, research_commit, source_hashes):
    """Emit each recorded call, prove AST equality, and return (files, provenance entries)."""
    files, provenance = {}, {}
    for key, call in sorted(calls.items(), key=lambda item: item[1]["stem"]):
        path = Path(root) / call["path"]
        source = Source(path.read_text(encoding="utf-8"))
        tree = ast.parse(source.text, filename=str(path))
        API.check_source(tree, call["path"])
        body, selected, requested = select(tree, call["names"], set(call["overrides"]))
        segments, transforms = [], []
        for node in body:
            text, edits = emit_segment(source, node, call, selected, requested, MODELS)
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            segments.append(f"# --- {call['path']}:{first}-{node.end_lineno}\n{text}\n")
            transforms.extend(edits)
        digest = source_hashes.get(call["path"]) or hashlib.sha256(path.read_bytes()).hexdigest()
        recorded = ast.dump(call["ast"])
        expected = ast.dump(API.apply_to_ast(copy.deepcopy(call["ast"]), call["path"]))
        migrated = expected != recorded
        constants = ", ".join(f"{k!r} -> {v!r}" for k, v in call["constants"].items()) or "none"
        header = (f'"""Package-owned case definitions materialized from research ``{call["path"]}``.\n\n'
                  f"Source: research commit ``{research_commit}``, sha256 ``{digest}``.\n"
                  f"Selected names: {', '.join(call['names'])}. Cases: {', '.join(sorted(call['cases']))}.\n"
                  f"Terminal thin-phase policy injected: ``{call['policy']}``. Constants: {constants}. "
                  f"Constructor cut-off: {', '.join(call['constructors']) or 'none'}.\n"
                  f"Module globals supplied by the loader: {', '.join(call['overrides']) or 'none'}.\n\n"
                  "Generated by ``tools/materialize_definitions.py``; do not edit by hand. The body is the\n"
                  "verbatim research text with the transforms listed in ``docs/definitions-provenance.json``,\n"
                  "and its AST equals the one ``acceptance_inputs.CaseInputs.load`` compiles today"
                  + (" with the same recorded\n``api_migration`` edits applied (" + API.REASON + "). Executed only\n" if migrated else ". Executed only\n") +
                  "by ``tools/definitions/_bindings.py``, which supplies ``_input_symbol``, ``_input_module`` and\n"
                  "``_input_options`` without reading any research file.\n"
                  '"""\n')
        text = header + "\n".join(segments)
        emitted = ast.parse(text)
        assert isinstance(emitted.body[0], ast.Expr) and isinstance(emitted.body[0].value, ast.Constant)
        emitted.body = emitted.body[1:]
        actual = ast.dump(emitted)
        if expected != actual:
            raise AssertionError(f"AST mismatch for {call['stem']} ({call['path']} {call['names']})")
        flags = sorted({item.id for item in ast.walk(call["ast"]) if isinstance(item, ast.Name) and item.id in {"__file__", "__name__"}})
        data = text.encode("utf-8")
        files[call["stem"]] = data
        provenance[call["stem"]] = dict(
            file=f"tools/definitions/{call['stem']}.py", sha256=hashlib.sha256(data).hexdigest(),
            source=call["path"], source_sha256=digest, names=call["names"],
            constants=[[repr(k), repr(v)] for k, v in call["constants"].items()], constructors=call["constructors"],
            overrides=call["overrides"], policy=call["policy"], cases=sorted(call["cases"]),
            nested_by=sorted(call["nested_by"]), key=key, ast_equal=True,
            ast_sha256=hashlib.sha256(recorded.encode("utf-8")).hexdigest(), module_dunder_names=flags,
            transforms=transforms)
        if migrated:
            provenance[call["stem"]]["migrated_ast_sha256"] = hashlib.sha256(expected.encode("utf-8")).hexdigest()
    # Every listed module-level api_migration edit for a materialized source must have been applied somewhere.
    for source_path in sorted({call["path"] for call in calls.values()}):
        applied = [item for entry in provenance.values() if entry["source"] == source_path for item in entry["transforms"]]
        missing = API.unapplied(source_path, applied)
        if missing:
            raise API.MigrationError(f"{source_path}: api_migration edits never applied: {missing}")
    return files, provenance


def main(argv=None):
    """Validate the research checkout, record, emit, prove, then write or check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--entries", nargs="*", help="top-level stems to materialize (default: all)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    mode.add_argument("--dry-run", action="store_true", help="prove AST equality only; write nothing")
    mode.add_argument("--emit-to", type=Path, help="write the emitted files to this folder for review; no provenance")
    args = parser.parse_args(argv)
    tool = module("acceptance", PROJECT / "tools/acceptance.py")
    manifest = tool.validate_research_checkout(args.research)
    unknown = set(args.entries or ()) - set(ENTRIES)
    if unknown:
        parser.error("unknown entries: " + ", ".join(sorted(unknown)))
    sys.path.insert(0, str(args.research.resolve()))
    # Always record the whole table, so stems, headers (cases, nested_by) and hashes don't depend
    # on which entries a run writes; --entries only restricts the files written or checked.
    calls, _ = record_bindinputs(args.research, ENTRIES)
    files, provenance = materialize(args.research, calls, manifest["reference_commit"], manifest["source_sha256"])
    if args.entries:
        wanted = set(args.entries)
        keep = {stem for stem, entry in provenance.items() if stem in wanted or wanted & set(entry["nested_by"])}
        files = {stem: data for stem, data in files.items() if stem in keep}
        provenance = {stem: entry for stem, entry in provenance.items() if stem in keep}
    for stem, entry in provenance.items():
        print(f"{stem}: AST equal; {len(entry['transforms'])} transforms; cases {','.join(entry['cases'])}"
              + (f"; nested by {','.join(entry['nested_by'])}" if entry["nested_by"] else "")
              + (f"; uses {entry['module_dunder_names']}" if entry["module_dunder_names"] else ""))
    if args.dry_run:
        return 0
    if args.emit_to:
        args.emit_to.mkdir(parents=True, exist_ok=True)
        for stem, data in files.items():
            (args.emit_to / f"{stem}.py").write_bytes(data)
        (args.emit_to / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
        return 0
    existing = json.loads(PROVENANCE.read_text(encoding="utf-8")) if PROVENANCE.exists() else None
    if args.check:
        failed = [stem for stem, data in files.items()
                  if not (DEFINITIONS / f"{stem}.py").exists() or (DEFINITIONS / f"{stem}.py").read_bytes() != data
                  or existing is None or existing["definitions"].get(stem) != provenance[stem]]
        print("check:", "OK" if not failed else "MISMATCH " + ", ".join(failed))
        return 1 if failed else 0
    DEFINITIONS.mkdir(exist_ok=True)
    for stem, data in files.items():
        (DEFINITIONS / f"{stem}.py").write_bytes(data)
    document = existing or dict(schema=1)
    document.update(research_commit=manifest["reference_commit"], materializer="tools/materialize_definitions.py",
        equivalence=("ast.dump of each emitted module without its docstring equals the AST that the unchanged "
                     "acceptance_inputs.CaseInputs.load compiles at research_commit, with the recorded api_migration "
                     "edits (tools/api_migrations.py) applied to it; no other normalization. ast_sha256 is the "
                     "recorded AST, migrated_ast_sha256 (when present) the AST after those edits"))
    document.setdefault("definitions", {}).update(provenance)
    document["definitions"] = dict(sorted(document["definitions"].items()))
    keys = [entry["key"] for entry in document["definitions"].values()]
    assert len(keys) == len(set(keys)), "two definition files registered for one load key"
    PROVENANCE.write_bytes((json.dumps(document, indent=1) + "\n").encode("utf-8"))
    print(f"wrote {len(files)} definition files and {PROVENANCE.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
