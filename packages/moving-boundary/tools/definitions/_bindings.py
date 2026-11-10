"""Package-role run-time bindings for the materialized case definitions.

No AST, no research root and no research file reads. ``Context`` is the package-role
counterpart of the historical ``acceptance_inputs.CaseInputs``: the same ``load``/``symbol``/
``model``/``imported_module`` interface and the same resolution rules for the package role, but
``load`` executes a hash-checked file from ``tools/definitions/`` chosen by the lookup key that
``tools/materialize_definitions.py`` recorded in ``docs/definitions-provenance.json``.

Conventions that keep the materialized definitions behaving exactly as today's AST loads:

- ``root`` is a ``SourceKey``: research-relative paths are lookup keys only and cannot be opened.
- The terminal thin-phase policy is baked into each definitions file (one file per policy). The
  lookup key includes the current ``policy``; the in-process module cache uses ``CaseInputs``'
  key, which does not, so module sharing within one worker is unchanged.
- Module globals named in ``overrides`` are set before the body runs, as ``CaseInputs`` does.
- Each module's ``__file__`` is its research-relative source key (a string, never opened), so a
  definition that derives paths from ``__file__`` gets the same path tail without a checkout.
- Model classes record their provider module, and are subclassed to record every constructed
  model when ``track_models`` is set (the contracts' accepted states).
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

PROJECT = Path(__file__).resolve().parents[2]
PROVENANCE = PROJECT / "docs/definitions-provenance.json"
POLICY_KEYWORD = "terminal_thin_phase_policy"

# Must equal acceptance_inputs.MODELS/MODULES (asserted by tests/test_definitions_bindings.py);
# copied because the package role must not import the historical loader.
MODELS = {
    "MovingBoundaryIllingworthFD1DModel": "binary",
    "MovingBoundaryIllingworthTernaryFD1DModel": "ternary",
    "MovingBoundaryIllingworthTernaryThreePhaseFD1DModel": "three_phase",
}
MODULES = {
    "kawin.diffusion.MovingBoundaryIllingworthFDM": "moving_boundary.solvers.illingworth.binary",
    "kawin.diffusion.MovingBoundaryIllingworthTernaryFDM": "moving_boundary.solvers.illingworth.ternary",
    "kawin.diffusion.MovingBoundaryIllingworthTernaryThreePhaseFDM": "moving_boundary.solvers.illingworth.three_phase",
    "kawin.diffusion.mesh.MovingBoundaryIllingworthFD1D": "moving_boundary.solvers.illingworth.numerics.binary",
    "kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D": "moving_boundary.solvers.illingworth.numerics.ternary",
    "kawin.diffusion.mesh.TransformedGrids": "moving_boundary.mesh.TransformedGrids",
    "kawin.diffusion.MovingBoundaryEquilibrium": "moving_boundary.equilibrium",
    "kawin.diffusion._illingworth_interstitials": "moving_boundary.solvers.illingworth._interstitials",
    "kawin.diffusion.MovingBoundarySurrogates": "moving_boundary.surrogates",
    "kawin.diffusion.SurrogateArtifacts": "moving_boundary.surrogate_artifacts",
}


def lookup_key(path, names, constants, constructors, overrides, policy):
    """Same canonical key as ``materialize_definitions.lookup_key`` (asserted by the tests)."""
    return json.dumps([str(path), sorted(names), sorted([repr(k), repr(v)] for k, v in constants.items()),
                       list(constructors), sorted(overrides), policy])


class SourceKey:
    """A research-relative path used only to look up a definition; opening it is an error."""

    def __init__(self, path=""):
        self.path = str(path).replace("\\", "/").strip("/")

    def __truediv__(self, other):
        return SourceKey(f"{self.path}/{other}" if self.path else other)

    def joinpath(self, *parts):
        result = self
        for part in parts:
            result = result / part
        return result

    def with_suffix(self, suffix):
        head, _, name = self.path.rpartition("/")
        stem = name.rsplit(".", 1)[0] if "." in name else name
        return SourceKey((head + "/" if head else "") + stem + suffix)

    def as_posix(self):
        return self.path

    def __str__(self):
        return self.path

    def __repr__(self):
        return f"SourceKey({self.path!r})"

    def __fspath__(self):
        raise TypeError("research source keys are lookup keys, never files: " + self.path)


def _options_without_policy(options):
    """Copy ``**options`` without its policy entry; the materialized call adds the policy last."""
    return {key: value for key, value in options.items() if key != POLICY_KEYWORD}


class Context:
    """Package-role case inputs backed by the materialized definitions (see module docstring)."""

    def __init__(self, policy="raise", *, provenance=PROVENANCE):
        self.role, self.policy = "package", policy
        self.root = SourceKey()
        self.modules = {}
        self.providers = {}
        self.track_models = False
        self.models = []
        self.family = None
        document = json.loads(Path(provenance).read_text(encoding="utf-8"))
        self.registry = {entry["key"]: entry for entry in document["definitions"].values()}

    def model(self, name):
        """Resolve a relocated solver class, or report the still-unrelocated host family."""
        family = MODELS[name]
        legacy = {"binary": "MovingBoundaryIllingworthFDM", "ternary": "MovingBoundaryIllingworthTernaryFDM",
                  "three_phase": "MovingBoundaryIllingworthTernaryThreePhaseFDM"}[family]
        try:
            cls = getattr(importlib.import_module("moving_boundary.solvers"), name)
        except AttributeError:
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
        """Bind one name a definition imports; research modules map to materialized definitions."""
        if name in MODELS:
            return self.model(name)
        if name == "CartesianFD1D":
            def mesh(*args, **kwargs):
                target_name = next(key for key, value in MODELS.items() if value == self.family)
                try:
                    cls = getattr(importlib.import_module("moving_boundary.solvers"), target_name)
                    extracted = cls.__module__.startswith("moving_boundary.")
                except AttributeError:
                    extracted = False
                module_name = "moving_boundary.mesh" if extracted else "kawin.diffusion.mesh"
                return getattr(importlib.import_module(module_name), "CartesianFD1D")(*args, **kwargs)
            try:
                importlib.import_module("moving_boundary.mesh")
            except ModuleNotFoundError:
                return getattr(importlib.import_module("kawin.diffusion.mesh"), name)
            return mesh
        if module.startswith(("kawin.tests", "examples")):
            return getattr(self.load(self.root.joinpath(*module.split(".")).with_suffix(".py"), [name]), name)
        choices = []
        if module == "kawin.thermo" and name == "MulticomponentThermodynamics":
            choices.append("moving_boundary.compatibility.thermo")
        if module in MODULES:
            choices.append(MODULES[module])
        if module in {"kawin.diffusion", "kawin.diffusion.mesh"}:
            choices.extend(["moving_boundary", "moving_boundary.mesh", "moving_boundary.solvers.illingworth.shared"])
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
        if module in MODULES:
            try:
                return importlib.import_module(MODULES[module])
            except ModuleNotFoundError as exc:
                if not MODULES[module].startswith(exc.name):
                    raise
        return importlib.import_module(module)

    def load(self, path, names, *, constants=None, constructors=(), overrides=None):
        """Execute the hash-checked definitions file registered for this exact load call."""
        constants = constants or {}
        overrides = overrides or {}
        source = path.as_posix() if isinstance(path, SourceKey) else None
        if source is None:
            raise TypeError("package-role loads take research-relative SourceKey paths, got " + repr(path))
        cache = (source, tuple(sorted(names)), tuple(sorted(constants.items())), tuple(constructors), tuple(overrides))
        if cache in self.modules:
            return self.modules[cache]
        key = lookup_key(source, names, constants, constructors, overrides, self.policy)
        entry = self.registry.get(key)
        if entry is None:
            raise LookupError("No package-owned definition for " + key + "; materialize it first")
        file = PROJECT / entry["file"]
        data = file.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise AssertionError("Definitions file changed since it was materialized: " + entry["file"])
        module = types.ModuleType("acceptance_inputs_" + str(len(self.modules)))
        module.__file__ = entry["source"]
        module.__dict__.update(_input_symbol=self.symbol, _input_module=self.imported_module,
                               _input_options=_options_without_policy)
        module.__dict__.update(overrides)
        self.modules[cache] = module
        sys.modules[module.__name__] = module
        exec(compile(data, str(file), "exec"), module.__dict__)
        for name in names:
            if not hasattr(module, name):
                raise AttributeError(f"Definition {name} not found in {entry['file']}")
        return module
