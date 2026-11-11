"""Batch-only guards and explicit policy injection for frozen case inputs.

These are verification utilities, not changes to either solver's public defaults.
``guard_independence`` is the opt-in M9 guard against research access and Git calls.
The loader transforms only case constructor keywords in an in-memory AST; source
files and production kawin module globals are never patched.
"""

import ast
import builtins
import os
from pathlib import Path
import re
import sys
import threading
import types


def forbid_input():
    """Make any attempted interactive input fail immediately, including imports."""
    builtins.input = _forbidden_input


def _forbidden_input(*args, **kwargs):
    """Raise a distinct error rather than hanging or disguising a missing resource."""
    raise RuntimeError("Interactive input() is forbidden in acceptance/verification workers")


# Research-only module names (plan §7). Stock kawin 0.5.0 has none of the kawin.diffusion ones.
FORBIDDEN_TOP_LEVEL = frozenset({"examples", "acceptance_inputs"})
FORBIDDEN_PACKAGES = ("kawin.tests",)
FORBIDDEN_NAME_PREFIXES = ("kawin.diffusion.MovingBoundary", "kawin.diffusion._illingworth_")
_GIT_PROGRAMS = frozenset({"git", "git.exe", "git.cmd", "git.bat"})
_SHELLS = frozenset({"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",
                     "bash", "bash.exe", "sh", "sh.exe"})
_GIT_IN_COMMAND = re.compile(r"""(?:^|[\s"'\\/])git(?:\.exe|\.cmd|\.bat)?(?=$|[\s"'])""", re.IGNORECASE)


class IndependenceViolation(RuntimeError):
    """A research access or Git call under an active independence guard.

    Deliberately not an ``ImportError`` or ``OSError``, so ``except ImportError`` /
    ``except OSError`` fallbacks cannot turn a violation into a silent alternative path.
    """


def forbidden_module(name):
    """Whether ``name`` is a research-only module the package role must never import."""
    return (name.split(".")[0] in FORBIDDEN_TOP_LEVEL
            or any(name == package or name.startswith(package + ".") for package in FORBIDDEN_PACKAGES)
            or name.startswith(FORBIDDEN_NAME_PREFIXES))


def git_invocation(program, args):
    """Whether a process launch (audit-event arguments) runs Git, directly or through a shell.

    ``program`` is the explicit executable (may be ``None``); ``args`` is the argument list or a
    command string. A string is searched for a ``git`` command token; a list is checked by its
    program name and, when that program is a shell, by every later token.
    """
    def name(value):
        try:
            return os.path.basename(os.fsdecode(value).strip("\"'")).lower()
        except TypeError:
            return ""
    if program is not None and name(program) in _GIT_PROGRAMS:
        return True
    if isinstance(args, (str, bytes)):
        return bool(_GIT_IN_COMMAND.search(os.fsdecode(args)))
    try:
        tokens = [os.fsdecode(token) for token in args or ()]
    except TypeError:
        return False
    if not tokens:
        return False
    if name(tokens[0]) in _GIT_PROGRAMS:
        return True
    return name(tokens[0]) in _SHELLS and any(_GIT_IN_COMMAND.search(token) for token in tokens[1:])


def guard_independence(environ=None):
    """Install the independence guards (plan §7, M9) that the environment asks for; opt-in.

    - ``MB_FORBID_RESEARCH=1``: importing a research-only module (``examples.*``,
      ``kawin.tests.*``, ``acceptance_inputs``, ``kawin.diffusion.MovingBoundary*``,
      ``kawin.diffusion._illingworth_*``), and opening, listing or scanning any path under the
      ``os.pathsep``-separated ``MB_FORBID_ROOTS`` or with a component starting ``mb-ref-``
      (checked before and after resolving links), raises ``IndependenceViolation``. So does a
      forbidden root on ``sys.path`` (checked at every import) or a forbidden module already
      imported when the guard is installed.
    - ``MB_FORBID_GIT=1``: launching Git through ``subprocess``, ``os.system``, ``os.exec*``,
      ``os.spawn*``, ``os.posix_spawn`` or ``os.startfile`` raises ``IndependenceViolation``. The
      M9 driver also takes Git off ``PATH`` for child processes that are not Python.
    - ``MB_GUARD_LOG``: each violation is also appended to this file, so one swallowed by a broad
      ``except`` still fails the run that checks the log afterwards.

    Audit hooks cannot be removed, so this is process-wide and permanent; a second call in the
    same process does nothing. Child processes inherit the variables, and every tool entry point
    (acceptance and example workers, the test ``conftest``, the acceptance ``main``s) calls this.
    Returns the set of guards active in this process.
    """
    environ = os.environ if environ is None else environ
    research = environ.get("MB_FORBID_RESEARCH") == "1"
    git = environ.get("MB_FORBID_GIT") == "1"
    installed = getattr(sys, "_mb_independence_guards", None)
    if installed is not None or not (research or git):
        return installed or frozenset()
    roots = [os.path.normcase(os.path.abspath(root)).rstrip("\\/")
             for root in environ.get("MB_FORBID_ROOTS", "").split(os.pathsep) if root.strip()]
    log = environ.get("MB_GUARD_LOG")
    busy = threading.local()

    def violation(message):
        if log:
            previous = getattr(busy, "active", False)
            busy.active = True   # the log write itself is not audited
            try:
                with open(log, "a", encoding="utf-8") as stream:
                    stream.write(f"{os.getpid()} {message}\n")
            finally:
                busy.active = previous
        raise IndependenceViolation(message)

    def forbidden_path(value):
        if isinstance(value, int):
            return False
        try:
            text = os.fsdecode(value)
        except TypeError:
            return False
        candidates = {os.path.normcase(os.path.abspath(text))}
        try:
            candidates.add(os.path.normcase(os.path.realpath(text)))
        except (OSError, ValueError):
            pass
        for path in candidates:
            if any(path == root or path.startswith(root + os.sep) for root in roots):
                return True
            if any(part.startswith("mb-ref-") for part in re.split(r"[\\/]", path)):
                return True
        return False

    checked_paths = set()

    def check_sys_path():
        for entry in list(sys.path):
            if entry not in checked_paths:
                if entry and forbidden_path(entry):
                    violation("research path on sys.path: " + entry)
                checked_paths.add(entry)

    class NoResearchImports:
        """First meta-path finder: refuse research modules before any loader reads source."""

        def find_spec(self, fullname, path=None, target=None):
            if forbidden_module(fullname):
                violation("research import: " + fullname)
            check_sys_path()
            return None

    def audit(event, args):
        if getattr(busy, "active", False):
            return
        busy.active = True   # path resolution below must not re-enter the hook
        try:
            check(event, args)
        finally:
            busy.active = False

    def check(event, args):
        if research:
            if event == "import" and forbidden_module(args[0]):
                violation("research import: " + args[0])
            elif event in {"open", "os.listdir", "os.scandir"} and args and forbidden_path(args[0]):
                violation(f"research path {event}: {os.fsdecode(args[0])}")
        if git:
            if event == "subprocess.Popen":
                launch = (args[0], args[1])
            elif event == "os.system":
                launch = (None, args[0])
            elif event in {"os.exec", "os.posix_spawn"}:
                launch = (args[0], args[1])
            elif event == "os.spawn":
                launch = (args[1], args[2])
            elif event == "os.startfile":
                launch = (args[0], None)
            else:
                return
            if git_invocation(*launch):
                violation(f"Git invocation blocked (MB_FORBID_GIT=1): {launch[1] if launch[1] is not None else launch[0]!r}")

    active = frozenset(name for name, on in (("research", research), ("git", git)) if on)
    sys._mb_independence_guards = active
    if research:
        for name in list(sys.modules):
            if forbidden_module(name):
                violation("research module imported before the guard: " + name)
        check_sys_path()
        sys.meta_path.insert(0, NoResearchImports())
    sys.addaudithook(audit)
    return active


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
