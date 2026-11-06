"""Bounded build/install/parity verification without modifying shared packages."""

from pathlib import Path
import argparse
from email.parser import Parser
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import re
import time
import zipfile
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
FORK = PROJECT.parents[1]
OUTPUT = PROJECT / ".verification"
EXPECTED = "ae9fb2971a9fd49132f71ddc883a5bd5406644b6"


def pinned_host_requirement():
    """Read the exact public runtime requirement separately from the reference."""
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    requirement, = [r for r in project["project"]["dependencies"] if r.startswith("kawin==")]
    return requirement


def run(label, args, *, cwd, env=None, timeout=600):
    """Run a batch subprocess with a wall-time limit and retain its full log."""
    started = time.monotonic()
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True,
                            text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    (OUTPUT / f"{label}.log").write_text(
        result.stdout + result.stderr, encoding="utf-8")
    print(f"{label}: exit={result.returncode}, {time.monotonic()-started:.2f}s",
          flush=True)
    if result.returncode:
        raise RuntimeError(f"{label} failed: {result.stdout}\n{result.stderr}")
    return result.stdout


def check_baseline(root, expected):
    """Reject tracked baseline changes and untracked Python shadowing.

    Package-only commits are allowed. The reference source, examples required by
    its imports, packaging inputs and fixtures remain frozen independently of HEAD.
    """
    # Compare content, not branch position: package-only commits are permitted.
    run("baseline-content", ["git", "diff", "--exit-code", expected, "--",
                            "kawin", "examples", "pyproject.toml", "MANIFEST.in",
                            "LICENSE.txt"], cwd=root, timeout=30)
    untracked = run("baseline-untracked", ["git", "ls-files", "--others",
                    "--exclude-standard", "--", "kawin"], cwd=root, timeout=30)
    if untracked.strip():
        raise RuntimeError("Untracked kawin files could shadow the frozen baseline")


def finish(report, strict=False):
    """Print both gates; strict installation acceptance returns status 2 if open."""
    print("PASS: source-backed bridge and numerical baseline")
    installed = report["installed_fork"]
    if not installed["import_ok"]:
        print(f"OPEN GATE: installed-fork import fails ({installed['missing_module']})")
        return 2 if strict else 0
    print("PASS: installed-fork import")
    if installed.get("binary_parity"):
        print("PASS: installed bridge/fork binary parity against frozen reference")
    return 0


def inspect_wheel(directory):
    """Select the sole fresh wheel and inspect identity without a fixed version."""
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one fresh wheel, found {len(wheels)}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        metadata_paths = [n for n in members if n.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise RuntimeError("Wheel must contain exactly one METADATA file")
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode())
        if metadata["Name"] != "moving-boundary":
            raise RuntimeError(f"Unexpected distribution: {metadata['Name']}")
        dist_info = metadata_paths[0].rsplit("/", 1)[0] + "/"
        assert all(n.startswith(("moving_boundary/", dist_info)) for n in members)
        assert not any("kawin/" in n or n.endswith(".npz") for n in members)
        assert any(req.replace(" ", "") == pinned_host_requirement() for req in metadata.get_all("Requires-Dist", []))
    return wheel, members


def main(argv=None):
    """Run the current quick public-kawin gate through the activated runtime."""
    return subprocess.run([sys.executable, str(PROJECT / 'tools/acceptance.py'),
        'quick', *(sys.argv[1:] if argv is None else argv)],
        timeout=650, stdin=subprocess.DEVNULL).returncode


if __name__ == '__main__':
    raise SystemExit(main())
