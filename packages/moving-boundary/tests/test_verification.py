"""Regression coverage for baseline guards without changing the real Git history."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile

import pytest


@pytest.mark.git
@pytest.mark.parametrize("change", ["package_commit", "tracked_solver", "untracked_solver"])
def test_frozen_content_guard(tmp_path, monkeypatch, request, change):
    """Allow package commits but reject tracked changes and source shadowing.

    Historical tooling that runs Git itself, so it is marked ``git`` and kept out of the
    portable normal gate (plan §7, follow-up 7); ``approve`` and plain suite runs include it.
    The disposable repository lives under a short system temp root, not ``tmp_path``: its Git
    object paths would exceed Windows' 260-character MAX_PATH below a deep ``--basetemp``
    (follow-up 6).
    """
    spec = importlib.util.spec_from_file_location(
        "bridge_verify", Path(__file__).parents[1] / "tools/verify.py")
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(verify, "OUTPUT", logs)
    short = tempfile.TemporaryDirectory(prefix="mbgit-", ignore_cleanup_errors=True)
    request.addfinalizer(short.cleanup)
    repo = Path(short.name) / "r"
    repo.mkdir()

    def git(*args):
        result = subprocess.run(
            ["git", "-c", "user.name=Verification", "-c",
             "user.email=verification@example.invalid", *args],
            cwd=repo, capture_output=True, text=True, check=True, timeout=30)
        return result.stdout.strip()

    git("init")
    (repo / "kawin").mkdir()
    source = repo / "kawin/model.py"
    source.write_text("baseline = True\n", encoding="utf-8")
    git("add", "kawin")
    git("commit", "-m", "Disposable test baseline")
    baseline = git("rev-parse", "HEAD")
    if change == "package_commit":
        (repo / "packages").mkdir()
        (repo / "packages/checkpoint.txt").write_text("new\n", encoding="utf-8")
        git("add", "packages")
        git("commit", "-m", "Disposable package-only commit")
        assert git("rev-parse", "HEAD") != baseline
        verify.check_baseline(repo, baseline)
    else:
        if change == "tracked_solver":
            source.write_text("baseline = False\n", encoding="utf-8")
        else:
            (repo / "kawin/shadow.py").write_text("shadow = True\n", encoding="utf-8")
        with pytest.raises(RuntimeError):
            verify.check_baseline(repo, baseline)
@pytest.mark.parametrize("installed,strict,expected", [
    (False, False, 0), (False, True, 2), (True, False, 0), (True, True, 0)])
def test_gate_summary(installed, strict, expected, capsys):
    """An open gate is always visible and fails strict installation acceptance."""
    spec = importlib.util.spec_from_file_location(
        "gate_verify", Path(__file__).parents[1] / "tools/verify.py")
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    status = verify.finish({"installed_fork": {
        "import_ok": installed, "missing_module": None if installed else "examples"
    }}, strict)
    assert status == expected
    output = capsys.readouterr().out
    assert "PASS: source-backed bridge" in output
    assert ("OPEN GATE: installed-fork import fails (examples)" in output) == (not installed)


def test_wheel_discovery_uses_metadata(tmp_path):
    """A changed version works; multiple artifacts are rejected as ambiguous."""
    import zipfile
    spec = importlib.util.spec_from_file_location(
        "wheel_verify", Path(__file__).parents[1] / "tools/verify.py")
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    wheel = tmp_path / "moving_boundary-9.8.7-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("moving_boundary/__init__.py", "")
        archive.writestr("moving_boundary-9.8.7.dist-info/METADATA",
                         "Name: moving-boundary\nVersion: 9.8.7\n"
                         f"Requires-Dist: {verify.pinned_host_requirement()}\n")
    assert verify.inspect_wheel(tmp_path)[0] == wheel
    (tmp_path / "stale.whl").touch()
    with pytest.raises(RuntimeError, match="exactly one"):
        verify.inspect_wheel(tmp_path)

def test_dependency_pin_is_separate_from_reference():
    """The runtime requirement follows metadata without moving the frozen oracle."""
    import moving_boundary
    spec = importlib.util.spec_from_file_location(
        "pin_verify", Path(__file__).parents[1] / "tools/verify.py")
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    assert verify.pinned_host_requirement() == "kawin==" + moving_boundary.PUBLIC_KAWIN_VERSION
    assert moving_boundary.REFERENCE_COMMIT != verify.EXPECTED