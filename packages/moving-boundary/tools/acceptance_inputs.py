"""Load only approved case definitions, binding constructors without global patches."""
import ast
import importlib
import importlib.util
from pathlib import Path
import sys
import types

MODELS = {
    "MovingBoundaryIllingworthFD1DModel": "binary",
    "MovingBoundaryIllingworthTernaryFD1DModel": "ternary",
    "MovingBoundaryIllingworthTernaryThreePhaseFD1DModel": "three_phase",
}
MODULES = {
    "kawin.diffusion.MovingBoundaryIllingworthFDM": "moving_boundary.solvers.binary",
    "kawin.diffusion.MovingBoundaryIllingworthTernaryFDM": "moving_boundary.solvers.ternary",
    "kawin.diffusion.MovingBoundaryIllingworthTernaryThreePhaseFDM": "moving_boundary.solvers.three_phase",
    "kawin.diffusion.mesh.MovingBoundaryIllingworthFD1D": "moving_boundary.numerics.binary",
    "kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D": "moving_boundary.numerics.ternary",
    "kawin.diffusion.mesh.TransformedGrids": "moving_boundary.mesh.TransformedGrids",
    "kawin.diffusion.MovingBoundaryEquilibrium": "moving_boundary.equilibrium",
    "kawin.diffusion._illingworth_interstitials": "moving_boundary._illingworth_interstitials",
    "kawin.diffusion.MovingBoundarySurrogates": "moving_boundary.surrogates",
    "kawin.diffusion.SurrogateArtifacts": "moving_boundary.surrogate_artifacts",
}


class CaseInputs:
    """Read immutable input definitions; isolate reference/package constructor bindings.

    This dependency closure selects definitions and constants, never executes an
    interactive example's module body, plotting cells, cache fallback or main().
    Unrelocated families are explicitly recorded as host-backed intermediates;
    callers must require relocated origins for the family under acceptance.
    """
    def __init__(self, root, role, policy="raise"):
        self.root, self.role, self.policy = Path(root), role, policy
        self.modules = {}
        self.providers = {}
        self.track_models = False
        self.models = []
        self.family = None

    def model(self, name):
        """Resolve an extracted class or report the still-unrelocated host family."""
        family = MODELS[name]
        legacy = {
            "binary": "MovingBoundaryIllingworthFDM",
            "ternary": "MovingBoundaryIllingworthTernaryFDM",
            "three_phase": "MovingBoundaryIllingworthTernaryThreePhaseFDM",
        }[family]
        if self.role == "package":
            try:
                cls = getattr(importlib.import_module("moving_boundary.solvers"), name)
            except AttributeError:
                cls = getattr(importlib.import_module("kawin.diffusion." + legacy), name)
        else:
            cls = getattr(importlib.import_module("kawin.diffusion." + legacy), name)
        self.providers[family] = cls.__module__
        if self.track_models:
            owner = self
            class Tracked(cls):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    owner.models.append(self)
            return Tracked
        return cls

    def symbol(self, module, name):
        """Bind only local input-module imports, never production module globals."""
        if name in MODELS:
            return self.model(name)
        if name == "CartesianFD1D" and self.role == "package":
            def mesh(*args, **kwargs):
                target_name = next(key for key,value in MODELS.items() if value == self.family)
                try:
                    cls = getattr(importlib.import_module("moving_boundary.solvers"), target_name)
                    extracted = cls.__module__.startswith("moving_boundary.")
                except AttributeError:
                    extracted = False
                module_name = "moving_boundary.mesh" if extracted else "kawin.diffusion.mesh"
                return getattr(importlib.import_module(module_name), "CartesianFD1D")(*args, **kwargs)
            # The intermediate subclass still validates the fork's Cartesian type.
            try:
                importlib.import_module("moving_boundary.mesh")
            except ModuleNotFoundError:
                return getattr(importlib.import_module("kawin.diffusion.mesh"), name)
            return mesh
        if module.startswith(("kawin.tests", "examples")):
            path = self.root.joinpath(*module.split(".")).with_suffix(".py")
            return getattr(self.load(path, [name]), name)
        if self.role == "package":
            choices = []
            if module == "kawin.thermo" and name == "MulticomponentThermodynamics":
                choices.append("moving_boundary.compatibility.thermo")
            if module in MODULES:
                choices.append(MODULES[module])
            if module in {"kawin.diffusion", "kawin.diffusion.mesh"}:
                choices.extend(["moving_boundary", "moving_boundary.mesh", "moving_boundary.shared"])
            for choice in choices:
                try:
                    value = importlib.import_module(choice)
                except ModuleNotFoundError as exc:
                    if not (exc.name == choice or choice.startswith(exc.name + ".")):
                        raise
                    continue
                if hasattr(value, name):
                    return getattr(value, name)
        return getattr(importlib.import_module(module), name)

    def imported_module(self, module):
        """Resolve a qualified import locally; never register an alias under kawin."""
        if self.role == "package" and module in MODULES:
            try:
                return importlib.import_module(MODULES[module])
            except ModuleNotFoundError as exc:
                if not MODULES[module].startswith(exc.name):
                    raise
        return importlib.import_module(module)

    def load(self, path, names, *, constants=None, constructors=(), overrides=None):
        """Execute the transitive AST definition closure for the requested names."""
        path = Path(path)
        constants = constants or {}
        overrides = overrides or {}
        key = (str(path), tuple(sorted(names)), tuple(sorted(constants.items())), tuple(constructors), tuple(overrides))
        if key in self.modules:
            return self.modules[key]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bindings = {}
        future = []
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
            wanted.extend(item.id for item in ast.walk(node)
                          if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load))
        body = future + [node for node in tree.body if id(node) in selected and node not in future]
        owner = self

        class BindInputs(ast.NodeTransformer):
            def visit_Import(self, node):
                result = []
                for alias in node.names:
                    if alias.name in MODULES and alias.asname:
                        result.append(ast.Assign(targets=[ast.Name(id=alias.asname, ctx=ast.Store())],
                            value=ast.Call(func=ast.Name(id="_input_module", ctx=ast.Load()),
                            args=[ast.Constant(alias.name)], keywords=[])))
                    else:
                        result.append(ast.Import(names=[alias]))
                return result
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                if node.name in constructors:
                    for i, statement in enumerate(node.body):
                        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "model" for t in statement.targets):
                            node.body = node.body[:i+1] + [ast.Return(ast.Name(id="model", ctx=ast.Load()))]
                            break
                    else:
                        raise ValueError("No model constructor assignment in " + node.name)
                return node
            def visit_Constant(self, node):
                if isinstance(node.value, (int, float)) and node.value in constants:
                    return ast.copy_location(ast.Constant(constants[node.value]), node)
                return node

            def visit_ImportFrom(self, node):
                # A selected import statement can contain unrelated aliases.
                # Retain only names used by the definition closure; nested imports
                # are part of the selected function body and remain intact.
                aliases = [alias for alias in node.names if id(node) not in selected
                           or (alias.asname or alias.name) in requested]
                if node.module and node.module.startswith(("kawin", "examples")):
                    return [ast.Assign(targets=[ast.Name(id=alias.asname or alias.name, ctx=ast.Store())],
                        value=ast.Call(func=ast.Name(id="_input_symbol", ctx=ast.Load()),
                        args=[ast.Constant(node.module), ast.Constant(alias.name)], keywords=[]))
                        for alias in aliases if alias.name != "*"]
                node.names = aliases
                return node

            def visit_Call(self, node):
                self.generic_visit(node)
                name = node.func.id if isinstance(node.func, ast.Name) else ""
                if name in MODELS and MODELS[name] != "binary":
                    node.keywords = [kw for kw in node.keywords if kw.arg != "terminal_thin_phase_policy"]
                    for kw in node.keywords:
                        if kw.arg is None:
                            kw.value = ast.Call(func=ast.Name(id="_input_options", ctx=ast.Load()), args=[kw.value], keywords=[])
                    node.keywords.append(ast.keyword(arg="terminal_thin_phase_policy", value=ast.Constant(owner.policy)))
                return node

        module = types.ModuleType("acceptance_inputs_" + str(len(self.modules)))
        module.__file__ = str(path)
        module.__dict__.update(_input_symbol=self.symbol, _input_module=self.imported_module,
            _input_options=lambda options: {k: v for k, v in options.items() if k != "terminal_thin_phase_policy"})
        module.__dict__.update(overrides)
        self.modules[key] = module
        sys.modules[module.__name__] = module
        result = BindInputs().visit(ast.Module(body=body, type_ignores=[]))
        exec(compile(ast.fix_missing_locations(result), str(path), "exec"), module.__dict__)
        for name in names:
            if not hasattr(module, name):
                raise AttributeError(f"Input definition {name} not found in {path}")
        return module
