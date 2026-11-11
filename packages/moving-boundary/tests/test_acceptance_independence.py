"""M9 (plan §7): the opt-in independence guard (``batch_safety.guard_independence``).

``MB_FORBID_RESEARCH=1`` turns research imports and reads under ``MB_FORBID_ROOTS`` (or any
``mb-ref-*`` path) into ``IndependenceViolation``; ``MB_FORBID_GIT=1`` does the same for Git
launches. Audit hooks cannot be removed, so each scenario runs in its own process.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _tool():
    spec = importlib.util.spec_from_file_location(
        "independence_guard_tests", Path(__file__).parents[1] / "tools/batch_safety.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GUARD_SCRIPT = r"""
import os, runpy, subprocess, sys
guards = runpy.run_path(sys.argv[1])["guard_independence"]()
action, _, value = sys.argv[2].partition(":")
try:
    if action == "import":
        __import__(value)
    elif action == "open":
        open(value, "rb").close()
    elif action == "listdir":
        os.listdir(value)
    elif action == "swallow":
        try:
            open(value, "rb").close()
        except Exception:
            pass
    elif action == "syspath":
        sys.path.insert(0, value)
        __import__("json.tool")
    elif action == "git":
        subprocess.run(["git", "--version"], capture_output=True, timeout=30)
    elif action == "git-shell":
        subprocess.run("git --version", shell=True, capture_output=True, timeout=30)
    elif action == "git-path":
        subprocess.run([value, "--version"], capture_output=True, timeout=30)
    elif action == "python":
        subprocess.run([sys.executable, "-c", "pass"], check=True, timeout=60)
except Exception as exc:
    print("RAISED", type(exc).__name__, exc)
else:
    print("OK", sorted(guards))
"""


def _guarded(tmp_path, action, *, research=True, git=False, roots=()):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "MB_FORBID", "MB_GUARD"))}
    log = tmp_path / "guard.log"
    env.update(MB_FORBID_ROOTS=os.pathsep.join(str(root) for root in roots), MB_GUARD_LOG=str(log))
    if research:
        env["MB_FORBID_RESEARCH"] = "1"
    if git:
        env["MB_FORBID_GIT"] = "1"
    result = subprocess.run([sys.executable, "-I", "-c", _GUARD_SCRIPT,
                             str(Path(__file__).parents[1] / "tools/batch_safety.py"), action],
                            env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip(), (log.read_text(encoding="utf-8") if log.exists() else "")


@pytest.fixture
def research_tree(tmp_path):
    root = tmp_path / "research"
    (root / "kawin").mkdir(parents=True)
    (root / "kawin/secret.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "mb-ref-0000000").mkdir()
    (tmp_path / "mb-ref-0000000/file.txt").write_text("x", encoding="utf-8")
    (tmp_path / "allowed.txt").write_text("x", encoding="utf-8")
    return root


@pytest.mark.parametrize("module", ["examples.ThermoCalc.tc_python_adapter", "kawin.tests.illingworth_baseline_cases",
                                    "kawin.tests", "acceptance_inputs",
                                    "kawin.diffusion.MovingBoundaryIllingworthFDM",
                                    "kawin.diffusion._illingworth_shared"])
def test_guard_blocks_research_imports(tmp_path, module):
    out, log = _guarded(tmp_path, "import:" + module)
    assert out.startswith("RAISED IndependenceViolation research import: " + module), out
    assert "research import: " + module in log


@pytest.mark.parametrize("relative, blocked", [("research/kawin/secret.py", True), ("mb-ref-0000000/file.txt", True),
                                               ("allowed.txt", False)])
def test_guard_blocks_reads_under_research_roots(tmp_path, research_tree, relative, blocked):
    out, log = _guarded(tmp_path, "open:" + str(tmp_path / relative), roots=[research_tree])
    assert out.startswith("RAISED IndependenceViolation research path open") == blocked, out
    assert out.startswith("OK") != blocked and bool(log) == blocked


def test_guard_blocks_listing_and_logs_swallowed_violations(tmp_path, research_tree):
    out, _ = _guarded(tmp_path, "listdir:" + str(research_tree / "kawin"), roots=[research_tree])
    assert out.startswith("RAISED IndependenceViolation research path os.listdir"), out
    out, log = _guarded(tmp_path, "swallow:" + str(research_tree / "kawin/secret.py"), roots=[research_tree])
    assert out.startswith("OK") and "research path open" in log and "secret.py" in log  # the log keeps it


def test_guard_rejects_a_research_root_on_sys_path(tmp_path, research_tree):
    out, _ = _guarded(tmp_path, "syspath:" + str(research_tree / "kawin"), roots=[research_tree])
    assert out.startswith("RAISED IndependenceViolation research path on sys.path"), out


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are Windows-only")
def test_guard_follows_links_into_research_roots(tmp_path, research_tree):
    link = tmp_path / "link"
    subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(link), str(research_tree)], check=True,
                   capture_output=True, stdin=subprocess.DEVNULL, timeout=30)
    out, _ = _guarded(tmp_path, "open:" + str(link / "kawin/secret.py"), roots=[research_tree])
    assert out.startswith("RAISED IndependenceViolation research path open"), out


@pytest.mark.parametrize("action", ["git", "git-shell", "git-path:C:/Program Files/Git/cmd/git.exe"])
def test_guard_blocks_git(tmp_path, action):
    out, log = _guarded(tmp_path, action, research=False, git=True)
    assert out.startswith("RAISED IndependenceViolation Git invocation blocked"), out
    assert "Git invocation blocked" in log


def test_guard_allows_other_processes_and_is_off_by_default(tmp_path, research_tree):
    out, log = _guarded(tmp_path, "python", git=True, roots=[research_tree])
    assert out == "OK ['git', 'research']" and not log
    out, log = _guarded(tmp_path, "open:" + str(research_tree / "kawin/secret.py"), research=False, roots=[research_tree])
    assert out == "OK []" and not log


@pytest.mark.parametrize("program, args, expected", [
    (None, ["git", "status"], True), (None, ["C:\Program Files\Git\cmd\git.EXE", "rev-parse"], True),
    ("C:/Git/bin/git.exe", ["anything"], True), (None, "git rev-parse HEAD", True),
    (None, 'C:\Windows\system32\cmd.exe /c "git status"', True), (None, ["cmd", "/d", "/c", "git", "status"], True),
    (None, ["python", "-m", "pytest", "-m", "not git"], False), (None, ["pip", "wheel", "."], False),
    (None, "python -c pass", False), (None, ["digit.exe"], False), (None, [], False)])
def test_git_invocation_detection(program, args, expected):
    assert _tool().git_invocation(program, args) is expected


def test_forbidden_module_names():
    tool = _tool()
    assert all(tool.forbidden_module(name) for name in (
        "examples", "examples.helper", "kawin.tests", "kawin.tests.x", "acceptance_inputs",
        "kawin.diffusion.MovingBoundaryIllingworthTernaryFDM", "kawin.diffusion._illingworth_shared"))
    assert not any(tool.forbidden_module(name) for name in (
        "kawin", "kawin.diffusion", "kawin.diffusion.mesh", "kawin.testsuite", "moving_boundary.examples",
        "mb_example_binary"))
